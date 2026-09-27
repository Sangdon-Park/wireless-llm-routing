"""Summarize every supplementary control and task-bootstrap replicate."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from qbr.data import DATA,Study


def summarize(records):
    study=Study()
    plan=json.loads((DATA/'strengthening_plan.json').read_text())
    new=[r for r in records if r['replicate']==-1]
    boot=[r for r in records if r['replicate']>=0]
    required={(c,s,m,-1) for c in study.config['conditions'] for s in study.config['seeds'] for m in plan['controls']['methods']}
    required|={(c,s,m,b) for b in range(plan['bootstrap']['replicates']) for c in study.config['conditions']
               for s in plan['bootstrap']['traffic_seeds'] for m in plan['bootstrap']['methods']}
    keys=[(r['condition'],r['seed'],r['method'],r['replicate']) for r in records]
    assert len(keys)==len(set(keys))==len(required) and set(keys)==required
    assert all(r['max_airtime_sum']<=1+1e-10 for r in records)
    old=[r for r in study.expected if r['scheduler']=='paper']
    full=old+new
    methods=['carrot','routellm','service','prompt','calibrated','mean','radio_mean','radio_calibrated']
    output={'complete_control_runs':len(new),'complete_bootstrap_runs':len(boot),
            'bootstrap_replicates':plan['bootstrap']['replicates'],'traffic_seeds_per_bootstrap':plan['bootstrap']['traffic_seeds']}
    for scope in ('primary','equal12'):
        rows=[r for r in full if scope=='equal12' or r['condition']=='primary']
        means={m:{k:float(np.mean([r['metrics'][k] for r in rows if r['method']==m]))
                  for k in ('objective','delay','delay_p95','shortfall','money')} for m in methods}
        paired={}
        for ours,other in [('calibrated','mean'),('radio_calibrated','radio_mean'),
                           ('calibrated','radio_mean'),('radio_calibrated','calibrated'),('calibrated','prompt')]:
            differences=[]
            for seed in study.config['seeds']:
                samples=[[r['metrics']['objective'] for r in rows if r['method']==m and r['seed']==seed] for m in (ours,other)]
                differences.append(np.mean(samples[0])-np.mean(samples[1]))
            differences=np.array(differences)
            half=t.ppf(.975,19)*differences.std(ddof=1)/np.sqrt(20)
            paired[ours+'_minus_'+other]={'difference_s':float(differences.mean()),
                'descriptive_paired_t95_s':[float(differences.mean()-half),float(differences.mean()+half)],
                'reduction_pct':float(100*(1-means[ours]['objective']/means[other]['objective']))}
        sensitivity={}
        for other in ('mean','prompt'):
            differences=[];reductions=[]
            for b in range(plan['bootstrap']['replicates']):
                selected=[r for r in boot if r['replicate']==b and (scope=='equal12' or r['condition']=='primary')]
                ours=np.mean([r['metrics']['objective'] for r in selected if r['method']=='calibrated'])
                baseline=np.mean([r['metrics']['objective'] for r in selected if r['method']==other])
                differences.append(float(ours-baseline));reductions.append(float(100*(1-ours/baseline)))
            point={m:np.mean([r['metrics']['objective'] for r in rows if r['method']==m and r['seed'] in plan['bootstrap']['traffic_seeds']]) for m in ('calibrated',other)}
            sensitivity['calibrated_minus_'+other]={
                'original_bank_same_two_seed_difference_s':float(point['calibrated']-point[other]),
                'replicate_difference_s':differences,'replicate_reduction_pct':reductions,
                'percentile95_difference_s':np.quantile(differences,[.025,.975]).tolist(),
                'percentile95_reduction_pct':np.quantile(reductions,[.025,.975]).tolist(),
                'lower_cost_replicates':sum(d<0 for d in differences),
                'mean_difference_s':float(np.mean(differences))}
        output[scope]={'means':means,'matched_comparisons':paired,'task_composition':sensitivity}
    output['by_condition']={c:{m:float(np.mean([r['metrics']['objective'] for r in full if r['condition']==c and r['method']==m])) for m in methods} for c in study.config['conditions']}
    output['new_controller_mean_ms']={m:float(np.mean([r['controller_mean_ms'] for r in new if r['method']==m])) for m in plan['controls']['methods']}
    return output


def load_records(path=None):
    path=Path(path) if path else DATA/'strengthening_results.json.gz'
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/strengthening_summary.json'))
    parser.add_argument('--rerun-dir',type=Path,help='Compare complete controls/bootstrap JSONL reruns with the published archive')
    args=parser.parse_args()
    records=load_records(args.records)
    if args.rerun_dir:
        key=lambda r:(r['condition'],r['seed'],r['method'],r['replicate'])
        expected={key(r):r for r in records}
        repeated=[json.loads(line) for p in sorted(args.rerun_dir.glob('*.jsonl')) for line in p.read_text().splitlines()]
        assert len(repeated)==len(expected) and {key(r) for r in repeated}==set(expected)
        for row in repeated:
            reference=expected[key(row)]
            assert row['choices_sha256']==reference['choices_sha256']
            for metric in ('objective','delay','delay_p95','shortfall','quality','money','discarded_bits'):
                np.testing.assert_allclose(row['metrics'][metric],reference['metrics'][metric],rtol=0,atol=1e-8)
            assert row['metrics']['terminal_statuses']==reference['metrics']['terminal_statuses']
        print(json.dumps({'complete_rerun_matches':len(repeated)}))
        records=repeated
    result=summarize(records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    compact={k:{'matched':result[k]['matched_comparisons'],'task_composition':{
        name:{a:b for a,b in values.items() if not a.startswith('replicate_')} for name,values in result[k]['task_composition'].items()}}
        for k in ('primary','equal12')}
    print(json.dumps(compact,indent=2))

if __name__=='__main__':main()
