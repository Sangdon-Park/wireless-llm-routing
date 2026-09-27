"""Bao decision-rule transfers with the archived queue/radio-debt estimates."""
import gzip
import hashlib
import json
from math import fsum
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.stats import t as student_t
from .data import Study, DATA
from . import replay
from .policy import BudgetController
from .strengthening import radio_correction
from .bao_rule import bao_rule_transfer, EstimatedLatencies, FusionParameters

METHODS=('bao_queue','bao_radio')
DISPLAY_NAMES={'bao_queue':'Bao-Queue','bao_radio':'Bao-Debt'}


class BaoController(BudgetController):
    def __init__(self, *args, probabilities, method, alpha, theta, **kwargs):
        super().__init__(*args, **kwargs)
        assert method in METHODS
        self.probabilities = np.asarray(probabilities, dtype=float)
        assert self.probabilities.shape == (len(self.choices),)
        assert np.isfinite(self.probabilities).all() and ((0 <= self.probabilities) & (self.probabilities <= 1)).all()
        self.method = method
        self.parameters = FusionParameters(theta=theta, alpha_per_second=alpha)

    def radio_state(self, now, state):
        self.radio_observed_at = float(now)
        self.radio_backlog = np.bincount(state['users'],
            weights=np.maximum(state['visible'] - state['delivered'], 0.), minlength=self.sc.users)

    def choose(self, now, j, service, target, user, observed, wire, throughput):
        began = perf_counter()
        assert self.choices[j] is None
        waits, active, queued = self.compute_state(now)
        budget = int(service == 0)
        profile = self.forecast.decision
        assert profile.enabled[service, :, budget].all()
        serving = waits[:, None] + profile.mu[service]
        correction = np.zeros((2, 2))
        if self.method == 'bao_radio':
            assert self.radio_observed_at == now
            correction = radio_correction(self.radio_backlog, observed,
                  self.forecast.mean_bits[service] * self.sc.payload_scale, serving, user)
        times = (serving + correction)[:, budget]
        choice = bao_rule_transfer(float(self.probabilities[j]),
                 EstimatedLatencies(float(times[0]), float(times[1])), self.parameters)
        route = choice.cloud
        self.choices[j] = (route, budget)
        elapsed = perf_counter() - began
        self.decision_times.append(elapsed)
        self.actions.append(dict(job=j, time=float(now), route=route, budget=budget,
            service=service, target=float(target), user=user, waits=waits.tolist(),
            latency_seconds=times.tolist(), radio_excess_s=correction[:, budget].tolist(),
            semantic_probability=float(self.probabilities[j]), fused_score=choice.fused_score,
            threshold=self.parameters.theta, alpha_per_second=self.parameters.alpha_per_second,
            observed_capacity=np.asarray(observed).tolist() if self.method == 'bao_radio' else None,
            backlog_bits=self.radio_backlog.tolist() if self.method == 'bao_radio' else None,
            active_predictions=active, queued_predictions=queued, wall_s=elapsed,
            label='Bao-rule transfer', controller_cpu_excluded=True))
        # Values are descriptive latency estimates. The executor uses the
        # selected route only and does not minimize these returned values.
        return route, [0, 0], times

def plan():
    return json.loads((DATA/'bao_plan.json').read_text(encoding='utf-8'))


def records():
    with gzip.open(DATA/'bao_results.json.gz','rt',encoding='utf-8') as stream:return json.load(stream)


def assert_same(actual,expected,path='record'):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),path
        for key in expected:assert_same(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),path
        for j,(a,b) in enumerate(zip(actual,expected)):assert_same(a,b,path+str(j))
    elif isinstance(expected,(int,float)):
        assert np.isfinite(actual) and abs(actual-expected)<=1e-8,(path,actual,expected)
    else: assert actual==expected,(path,actual,expected)


def verify():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((DATA/'bao_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    for name,digest in manifest['source_sha256_lf'].items():
        from .publication import verify_source
        verify_source(root, name, digest)
    p=plan();study=Study();rows=records()
    assert p['conditions']==study.config['conditions'] and p['seeds']==study.config['seeds']
    expected={(c,s,m) for c in p['conditions'] for s in p['seeds'] for m in METHODS}
    assert len(rows)==len(expected)==480
    assert {(r['condition'],r['seed'],r['method']) for r in rows}==expected
    for row in rows:
        fixed=p['selected'][row['method']]
        assert row['alpha']==fixed['alpha'] and row['theta']==fixed['theta']
        assert row['max_airtime_sum']<=1+1e-10
    assert_same(summary(),json.loads((DATA/'bao_summary.json').read_text(encoding='utf-8')))
    return dict(verified=True,trajectories=480,task_pairs=225,display_names=DISPLAY_NAMES,
                calibration_search_reproducible=False,provider_calls=0)


def initialize():
    global STUDY, PLAN
    STUDY=Study();PLAN=plan()


def episode(job):
    condition,seed,method=job
    selected=PLAN['selected'][method]
    sc,tr=STUDY.requests(condition,seed)
    probabilities=STUDY.external['evaluation_routellm'][tr.task]
    seconds=STUDY.prediction_seconds('routellm',seed)
    factory=lambda *a,**k:BaoController(*a,probabilities=probabilities,method=method,
                         alpha=selected['alpha'],theta=selected['theta'],**k)
    wrapper=replay.wrapper(STUDY.rows,sc,seed,tr,seconds)
    result=replay.simulate(wrapper,STUDY.rows,STUDY.profiles,STUDY.forecast,controller_factory=factory)
    assert result['provider_calls']==0 and result['controller_and_radio_cpu_excluded']
    return dict(condition=condition,seed=seed,method=method,alpha=selected['alpha'],theta=selected['theta'],
       metrics=replay.account(wrapper,result,STUDY.rows),
       choices_sha256=hashlib.sha256(np.asarray(result['choices'],dtype='<i8').tobytes()).hexdigest(),
       cloud_fraction=float(np.mean(np.asarray(result['choices'])[sc.warmup:,0])),
       head_mean_ms=float(np.mean(seconds))*1000,max_airtime_sum=result['max_airtime_sum'])


def summary():
    from .information_controls import all_records
    study,controls=all_records()
    rows=records()+controls
    table={(r['condition'],r['seed'],r['method']):r for r in rows}
    methods=sorted({r['method'] for r in rows});seeds=study.config['seeds']
    assert len(table)==len(rows)==12*20*len(methods)
    scopes={c:[c] for c in study.config['conditions']}
    scopes['equal12']=list(study.config['conditions'])
    scopes['equal11_non10k']=[c for c in study.config['conditions'] if c!='bandwidth_10000']
    out={}
    for scope,conditions in scopes.items():
        means={m:{k:fsum(table[c,s,m]['metrics'][k] for c in conditions for s in seeds)/(len(conditions)*20)
               for k in ('objective','delay','delay_p95','shortfall','quality','money')} for m in methods}
        comparisons={}
        for method in METHODS:
            for reference in methods:
                if method==reference:continue
                delta=[fsum(table[c,s,method]['metrics']['objective']-table[c,s,reference]['metrics']['objective']
                           for c in conditions)/len(conditions) for s in seeds]
                mean=fsum(delta)/20
                half=float(student_t.ppf(.975,19))*float(np.std(delta,ddof=1))/np.sqrt(20)
                comparisons[method+'_minus_'+reference]=dict(mean_delta=mean,
                    conditional_seed_t95=[mean-half,mean+half],fixed_task_bank=True,task_generalization_interval=False)
        out[scope]=dict(means=means,comparisons=comparisons)
    return out
