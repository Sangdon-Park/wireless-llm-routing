"""Fixed utility and shortfall controls with the same CARROT head and queues.

This is a full-RPC utility-rule transfer, not a native SFS/TTFT reproduction.
Achieved cost always uses the original request target, including UtilityQueue.
"""
from pathlib import Path
import gzip
import hashlib
import json
import time
import numpy as np
from scipy.stats import t
from .data import Study, DATA
from .policy import BudgetController, scores, best, required_actions
from . import replay

SUPPLEMENT = Path(__file__).resolve().parents[1] / 'data'
METHODS = ('utility_queue', 'shortfall_queue')
METRICS = ('objective', 'delay', 'delay_p95', 'shortfall', 'quality', 'money')
PAIRS = (('carrot_queue', 'shortfall_queue'),
         ('shortfall_queue', 'utility_queue'), ('carrot_queue', 'utility_queue'))


class ComparisonController(BudgetController):
    def __init__(self,*a,qualities,method,**kw):
        super().__init__(*a,**kw);self.qualities=qualities;self.method=method
    def choose(self,now,j,service,target,user,observed,wire,throughput):
        began=time.perf_counter();waits,_,_=self.compute_state(now)
        allowed=required_actions(self.forecast.decision,service,self.mode,self.fixed)
        score_target=1. if self.method=='utility_queue' else target
        risk=np.full((2,2),np.nan)
        for route,budget in zip(*np.where(allowed)):
            risk[route,budget]=max(score_target-float(self.qualities[j,route]),0.)
        coefficient=.5 if self.method=='cq_neutral' else 0.
        value=scores(self.forecast.decision,service,score_target,waits,risk=risk,required=allowed,coefficient=coefficient)
        route,budget=best(value,allowed)
        assert self.choices[j] is None
        self.choices[j]=(int(route),int(budget));self.decision_times.append(time.perf_counter()-began)
        self.actions.append(dict(job=j,time=float(now),route=int(route),budget=int(budget),service=int(service),
            target=float(target),score_target=float(score_target),user=int(user),waits=waits.tolist(),
            values=value.min(1).tolist(),predicted_quality=self.qualities[j].tolist()))
        return int(route),[0,0],value.min(1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_records():
    with gzip.open(SUPPLEMENT/'utility_results.json.gz', 'rt', encoding='utf-8') as f:
        rows = json.load(f)
    with gzip.open(DATA/'information_results.json.gz', 'rt', encoding='utf-8') as f:
        rows += [r for r in json.load(f) if r['method']=='carrot_queue']
    plan = read(SUPPLEMENT/'utility_plan.json')
    expected = {(m,c,s) for m in ('carrot_queue',)+METHODS
                for c in plan['conditions'] for s in plan['seeds']}
    keys = [(r['method'],r['condition'],r['seed']) for r in rows]
    assert len(keys)==720 and len(set(keys))==720 and set(keys)==expected
    return plan, rows


def summary():
    plan, rows = load_records()
    index = {(r['method'],r['condition'],r['seed']):r for r in rows}
    scopes = {'primary':['primary'], 'equal12':plan['conditions'],
              'equal11_excluding_load_0.8':[c for c in plan['conditions'] if c!='load_0.8']}
    scopes.update({c:[c] for c in plan['conditions']})
    out = {}
    for scope, conditions in scopes.items():
        def sample(method, metric='objective'):
            return np.array([np.mean([index[method,c,s]['metrics'][metric]
                            for c in conditions]) for s in plan['seeds']])
        means = {m:{k:float(sample(m,k).mean()) for k in METRICS}
                 for m in ('carrot_queue',)+METHODS}
        comparisons = {}
        for a,b in PAIRS:
            delta = sample(a)-sample(b)
            mean = float(delta.mean())
            half = float(t.ppf(.975,19)*delta.std(ddof=1)/np.sqrt(20))
            comparisons[a+'_minus_'+b] = dict(difference_s=mean,
                paired_t95=[mean-half,mean+half], seed_differences=delta.tolist(),
                reduction_pct=100*(1-means[a]['objective']/means[b]['objective']))
        out[scope] = dict(means=means, comparisons=comparisons)
    gate = {s:all(out[s]['comparisons']['carrot_queue_minus_'+m]['paired_t95'][1]<0
                  for m in METHODS) for s in list(scopes)[:3]}
    return dict(scopes=out, descriptive_gate=gate, independent_confirmation=False,
                new_method_validated=False, native_sfs_reproduced=False)


def verify():
    """Verify portable files and recompute the complete archived summary."""
    manifest = read(SUPPLEMENT/'utility_manifest.json')
    root = Path(__file__).resolve().parents[1]
    for kind in ('sha256', 'source_sha256_lf'):
        for name, expected in manifest[kind].items():
            path = root/name
            if not path.exists():
                path = DATA.parent/name  # permits a source-only overlay checkout
            content = path.read_bytes()
            if kind=='source_sha256_lf':
                content = content.replace(b'\r\n', b'\n')
            assert hashlib.sha256(content).hexdigest()==expected, name
    expected = read(SUPPLEMENT/'utility_summary.json')
    actual = summary()
    def compare(a,b):
        if isinstance(a,dict):
            assert set(a)==set(b)
            for key in a: compare(a[key],b[key])
        elif isinstance(a,list):
            assert len(a)==len(b)
            for x,y in zip(a,b): compare(x,y)
        elif type(a) is float:
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
        else:
            assert a==b
    compare(actual,expected)
    return dict(passed=True, new_paths=480, archived_CQ_paths=240,
                conditions=12, seeds=20, all_scopes_and_components_match=True)


def initialize():
    global BANK, HEADS, TIMES, REFERENCES
    verify()
    BANK = Study()
    HEADS = dict(np.load(DATA/'information_predictions.npz',allow_pickle=False))
    TIMES = dict(np.load(DATA/'information_times.npz',allow_pickle=False))
    np.testing.assert_array_equal(HEADS['carrot'],BANK.external['evaluation_quality'])
    assert len(BANK.rows)==225
    _, rows = load_records()
    REFERENCES = {(r['method'],r['condition'],r['seed']):r for r in rows}


def episode(condition, seed, method):
    """Rerun one published control path and check its choices and all metrics."""
    if method not in METHODS:
        raise ValueError('Choose utility_queue or shortfall_queue')
    if 'BANK' not in globals():
        initialize()
    reference = REFERENCES[method,condition,seed]
    sc,tr = BANK.requests(condition,seed)
    assert sc.beta==12 and sc.gamma==1000 and sc.n==1200 and sc.warmup==200
    seconds = BANK.prediction_seconds('carrot',seed)+TIMES[str(seed)+'_carrot']
    quality = HEADS['carrot'][tr.task]
    factory = lambda *a,**kw:ComparisonController(*a,qualities=quality,method=method,**kw)
    wrapped = replay.wrapper(BANK.tasks,sc,seed,tr,seconds)
    result = replay.simulate(wrapped,BANK.rows,BANK.profiles,BANK.forecast,controller_factory=factory)
    metrics = replay.account(wrapped,result,BANK.rows)
    choice_hash = hashlib.sha256(np.asarray(result['choices'],dtype='<i8').tobytes()).hexdigest()
    assert choice_hash==reference['choices_sha256'], 'Routing choices differ'
    errors = {}
    for key in METRICS+('discarded_bits',):
        errors[key] = abs(metrics[key]-reference['metrics'][key])
        np.testing.assert_allclose(metrics[key],reference['metrics'][key],rtol=0,atol=1e-8,err_msg=key)
    assert metrics['terminal_statuses']==reference['metrics']['terminal_statuses']
    assert result['max_airtime_sum']<=1+1e-10
    return dict(condition=condition,seed=seed,method=method,metrics=metrics,
                choices_sha256=choice_hash,choices_match=True,max_metric_error=max(errors.values()),
                head_mean_ms=float(np.mean(seconds)*1000),max_airtime_sum=result['max_airtime_sum'])
