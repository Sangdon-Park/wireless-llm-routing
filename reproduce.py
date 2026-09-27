"""Verify data, fit the mixture, summarize results, or rerun wireless replay."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import t
from qbr.data import DATA, Study, ALIASES
from qbr.policy import BudgetController, NativeController
from qbr.quality import check_integrals, fit_mixture
from qbr import replay

METHODS = ('carrot','routellm','service','prompt','calibrated')


def verify():
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest.items():
        actual = hashlib.sha256((DATA/name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'Data checksum mismatch: {name}')
    study = Study()
    if len(study.rows) != 225 or len(study.expected) != 1260:
        raise ValueError('Incomplete study data')
    coefficients = {}
    for service in ('dialogue','summary','code'):
        z = study.weights
        alpha = fit_mixture(z[service+'_oof_forest'],z[service+'_oof_uniform'],z[service+'_oof_quality'])
        np.testing.assert_allclose(alpha,study.config['alpha'][service],rtol=0,atol=1e-12)
        coefficients[service] = alpha
    check_integrals()
    for seed in study.config['seeds']:
        study.requests('primary',seed)
    expected = {(c,s,m,'paper') for c in study.config['conditions'] for s in study.config['seeds'] for m in METHODS}
    expected |= {('primary',s,m,'pf') for s in study.config['seeds'] for m in ('service','prompt','calibrated')}
    actual = [(r['condition'],r['seed'],r['method'],r['scheduler']) for r in study.expected]
    assert set(actual) == expected and len(actual) == len(expected)
    print(json.dumps({'data_verified':True,'task_pairs':225,'paper_trajectories':1200,
        'pf_trajectories':60,'alpha':coefficients},indent=2))


def summarize():
    study = Study()
    output = {}
    for scope in ('primary','equal12'):
        rows = [r for r in study.expected if r['scheduler']=='paper' and (scope=='equal12' or r['condition']=='primary')]
        means = {m:{k:float(np.mean([r['metrics'][k] for r in rows if r['method']==m]))
                    for k in ('objective','delay','delay_p95','shortfall','money')} for m in METHODS}
        comparisons = {}
        for baseline in METHODS[:-1]:
            differences = []
            for seed in study.config['seeds']:
                ours = [r['metrics']['objective'] for r in rows if r['seed']==seed and r['method']=='calibrated']
                theirs = [r['metrics']['objective'] for r in rows if r['seed']==seed and r['method']==baseline]
                differences.append(float(np.mean(ours)-np.mean(theirs)))
            mean = float(np.mean(differences))
            sem = float(np.std(differences,ddof=1)/np.sqrt(len(differences)))
            half = float(t.ppf(.975,len(differences)-1))*sem
            comparisons[baseline] = {'cost_reduction_pct':100*(1-means['calibrated']['objective']/means[baseline]['objective']),
                'paired_difference_s':mean,'descriptive_ci95_s':[mean-half,mean+half]}
            if baseline in ('service','prompt'):
                comparisons[baseline]['prespecified_one_sided_9875_upper_s'] = mean+float(t.ppf(.9875,19))*sem
        output[scope] = {'means':means,'calibrated_vs':comparisons}
    return output


_STUDY = None

def initialize():
    global _STUDY
    _STUDY = Study()


def episode(job):
    condition,seed,method,scheduler = job
    study = _STUDY or Study()
    scenario,trace = study.requests(condition,seed)
    if method in ALIASES:
        risks = study.risks(method,trace)
        factory = lambda *a,**k: BudgetController(*a,risks=risks,**k)
    else:
        routes = study.native_routes(method,trace)
        factory = lambda *a,**k: NativeController(*a,routes=routes,**k)
    wrapper = replay.wrapper(study.tasks,scenario,seed,trace,study.prediction_seconds(method,seed))
    previous = replay.radio_plan
    if scheduler == 'pf':
        replay.radio_plan = replay.PF(scenario.users)
    try:
        result = replay.simulate(wrapper,study.rows,study.profiles,study.forecast,controller_factory=factory)
    finally:
        replay.radio_plan = previous
    metrics = replay.account(wrapper,result,study.rows)
    reference = study.reference(condition,seed,method,scheduler)
    errors = []
    for key in ('objective','delay','delay_p95','shortfall','quality','money','discarded_bits'):
        np.testing.assert_allclose(metrics[key],reference['metrics'][key],rtol=0,atol=1e-8,err_msg=key)
        errors.append(abs(metrics[key]-reference['metrics'][key]))
    assert metrics['terminal_statuses'] == reference['metrics']['terminal_statuses']
    choice_hash = hashlib.sha256(np.asarray(result['choices'],dtype='<i8').tobytes()).hexdigest()
    assert choice_hash == reference['choices_sha256'], 'Route choices differ from frozen experiment'
    return {'condition':condition,'seed':seed,'method':method,'scheduler':scheduler,
        'metrics':metrics,'max_metric_error':max(errors),'choices_match':True,'elapsed_s':result['elapsed_s']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('verify','summary','figures','replay'))
    parser.add_argument('--conditions',nargs='+',default=['primary'],help='Condition names, or all')
    parser.add_argument('--seeds',nargs='+',type=int,help='Defaults to all 20 published seeds')
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    parser.add_argument('--scheduler',choices=('paper','pf'),default='paper')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    args = parser.parse_args()
    if args.command=='verify':
        verify()
        return
    if args.command=='summary':
        result = summarize()
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(result,indent=2))
        return
    if args.command=='figures':
        from plot_results import plot
        plot(args.output)
        return
    study = Study()
    conditions = list(study.config['conditions']) if args.conditions==['all'] else args.conditions
    seeds = args.seeds or study.config['seeds']
    if not set(conditions)<=set(study.config['conditions']) or not set(seeds)<=set(study.config['seeds']):
        parser.error('Use the published conditions and seeds listed in data/config.json')
    if args.scheduler=='pf' and (conditions!=['primary'] or not set(args.methods)<=set(ALIASES)):
        parser.error('Published PF sensitivity uses primary and service/prompt/calibrated only')
    if args.workers<1:
        parser.error('--workers must be positive')
    jobs = [(c,s,m,args.scheduler) for s in seeds for c in conditions for m in args.methods]
    args.output.mkdir(parents=True,exist_ok=True)
    destination = args.output/('replay_'+args.scheduler+'.jsonl')
    started = time.perf_counter()
    # Refuse to overwrite a previous rerun silently.
    with destination.open('x',encoding='utf-8') as f:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
            for n,result in enumerate(pool.map(episode,jobs),1):
                f.write(json.dumps(result,allow_nan=False)+'\n')
                f.flush()
                print(json.dumps({'completed':n,'total':len(jobs),**{k:result[k] for k in ('condition','seed','method','max_metric_error')}}),flush=True)
    print(json.dumps({'passed':True,'trajectories':len(jobs),'seconds':time.perf_counter()-started,'output':str(destination)}))


if __name__=='__main__':
    main()
