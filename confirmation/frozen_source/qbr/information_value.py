"""Value-of-information and robustness study (v1.4).

One element of the serving score changes at a time; all methods share CARROT's
frozen paired-quality predictions, their charged runtime, the causal wait
estimator, the shared downlink scheduler, and the original achieved cost
L + beta*shortfall + 1000*fees. Retrospective on the same 225 task pairs; no
new responses, provider calls or fitting. Every run is published, including
the failed load-derived-coefficient gate and all adverse robustness results.
"""
from pathlib import Path
from types import SimpleNamespace
import gzip
import hashlib
import json
import time
import numpy as np
from scipy.stats import t
from .data import Study, DATA
from .radio import Scenario, make_requests, MIX, SERVICES
from .policy import best, required_actions
from .strengthening import RadioController, task_mapping, resampled_trace
from .information_controls import release_correction
from . import replay

BETAS = (4., 8., 12., 16., 24., 36.)
LOADS = (.35, .5, .65, .8, .9)
KAPPAS = (.125, .25, 1., 2.)
BOOT_SEEDS = (23924000, 23924001)
BOOT_MASTER = 2026092301
METRICS = ('objective', 'delay', 'delay_p95', 'shortfall', 'quality', 'money')
RESULTS = DATA/'information_value_results.json.gz'
BOOTSTRAP = DATA/'information_value_bootstrap.json.gz'
EXTENSION = DATA/'information_value_extension.json.gz'
PAYLOADS = (1., 2., 4., 8., 12.)


def spec(method):
    """Method name -> controller specification; '@bX' sets the rule's quality weight."""
    s = dict(rule='shortfall', wait=True, kappa=.5, beta=12., release=False, fixed=None,
             quality='carrot', charge='carrot')
    base, _, suffix = method.partition('@b')
    if suffix: s['beta'] = float(suffix)
    if base == 'cq': pass
    elif base == 'cq_release': s['release'] = True
    elif base == 'fpi': s['kappa'] = 'fpi'
    elif base == 'fpi_release': s.update(kappa='fpi', release=True)
    elif base == 'shortfall': s['kappa'] = 0.
    elif base == 'utility': s.update(kappa=0., rule='utility')
    elif base == 'intrinsic': s.update(kappa=0., wait=False)
    elif base.startswith('k'): s['kappa'] = float(base[1:])
    elif base == 'always_edge': s.update(fixed=0, charge='none')
    elif base == 'always_cloud': s.update(fixed=1, charge='none')
    elif base == 'oracle_cq': s['quality'] = 'oracle'
    elif base == 'mean_cq': s.update(quality='mean', charge='calibrated')
    elif base == 'mean_release': s.update(quality='mean', charge='calibrated', release=True)
    else: raise ValueError(method)
    return s


class InformationController(RadioController):
    """Eq. (2) of the manuscript with per-route kappa, optional radio term (5)."""
    def __init__(self, *a, qualities, spec, kappa, **kw):
        super().__init__(*a, **kw)
        self.q = qualities; self.s = spec; self.k = np.asarray(kappa, float)[:, None]
        self.radio_observed_at = None; self.radio_backlog = np.zeros(self.sc.users)

    def choose(self, now, j, service, target, user, observed, wire, throughput):
        began = time.perf_counter(); s = self.s; p = self.forecast.decision
        allowed = required_actions(p, service, self.mode, self.fixed)
        if s['fixed'] is not None:
            route = s['fixed']; budget = int(np.flatnonzero(allowed[route])[0]); values = np.zeros(2)
        else:
            waits = self.compute_state(now)[0] if s['wait'] else np.zeros(2)
            score_target = 1. if s['rule'] == 'utility' else float(target)
            risk = np.maximum(score_target-self.q[j], 0.)[:, None]
            # Same addition order as qbr.policy.scores for exact archived parity.
            v = waits[:, None]+p.mu[service]+1000.*p.money[service]
            v = v+self.k*(waits[:, None]*p.mu[service]+.5*p.second[service])
            v = v+s['beta']*risk
            if s['release']:
                assert self.radio_observed_at == now
                v = v+release_correction(self.radio_backlog, observed,
                    self.forecast.mean_bits[service]*self.sc.payload_scale, waits, p.mu[service], user)
            v = np.where(allowed, v, np.inf)
            route, budget = best(v, allowed); values = v.min(1)
        self.choices[j] = (int(route), int(budget))
        self.decision_times.append(time.perf_counter()-began)
        self.actions.append(dict(job=j, route=int(route), budget=int(budget), service=int(service),
                                 target=float(target), user=int(user)))
        return int(route), [0, 0], values


def fpi_kappa(study, rho):
    """Eq. (4): kappa_m = rho / ((1-rho) * traffic-weighted mean RPC duration of route m)."""
    return rho/((1-rho)*(MIX @ study.arrival_mu))


def airtime_load(study=None, conditions=('bandwidth_10000', 'bandwidth_30000', 'primary',
                                         'bandwidth_300000', 'bandwidth_1000000')):
    """Nominal airtime load rho_air = lambda*E[bits]*mean_u 1/E_t[C_u] (no routing replay)."""
    from .radio import Channel
    study = study or Study(); mb = study.forecast.mean_bits
    bits = MIX @ np.array([np.nanmean(mb[0, :, 1]), np.nanmean(mb[1, :, 0]), np.nanmean(mb[2, :, 0])])
    out = {}
    for c in conditions:
        sc = Scenario(**study.config['conditions'][c])
        lam = sc.rho*(sc.edge_slots/(MIX @ study.arrival_mu[:, 0])+sc.cloud_slots/(MIX @ study.arrival_mu[:, 1]))
        inverse = []
        for seed in range(23924000, 23924005):
            ch = Channel(sc, seed); x = np.array([ch.sample(k) for k in range(0, 20000, 7)])
            inverse.append(np.mean(1/x.mean(0)))
        out[c] = dict(lambda_per_s=float(lam), mean_bits=float(bits), rho_air=float(lam*bits*np.mean(inverse)))
    return out


def initialize():
    global STUDY, HEADS, TIMES, ORACLE, IDS
    STUDY = Study()
    HEADS = dict(np.load(DATA/'information_predictions.npz', allow_pickle=False))
    TIMES = dict(np.load(DATA/'information_times.npz', allow_pickle=False))
    ORACLE = np.array([[r['budgets']['strict512' if r['service'] == 'dialogue' else 'full'][m]['quality']
                        for m in ('edge', 'cloud')] for r in STUDY.rows])
    IDS = {s: [i for i, t in enumerate(STUDY.tasks) if t['service'] == n] for s, n in enumerate(SERVICES)}


def mean_quality(trace):
    """Per-request mean of the calibrated quantile-forest mixture (qbr.strengthening.mean_risks)."""
    out = np.full((len(trace.task), 2), np.nan)
    for k, service in enumerate(SERVICES):
        take = np.flatnonzero(trace.service == k); budget = int(k == 0)
        weights = STUDY.weights[service+'_weights'][[STUDY.positions[int(i)] for i in trace.task[take]]]
        prior = np.array([STUDY.forecast.decision.quality[k, r, budget].mean() for r in (0, 1)])
        alpha = STUDY.config['alpha'][service]
        out[take] = alpha*(weights @ STUDY.weights[service+'_scores'])+(1-alpha)*prior
    return out


def episode(exp, condition, rho, edge_slots, seed, method, replicate=-1, payload=None, details=False):
    """Replay one path. rho/edge_slots/payload=None keep the published condition's values."""
    if 'STUDY' not in globals(): initialize()
    s = spec(method)
    d = dict(STUDY.config['conditions'][condition])
    if rho is not None: d['rho'] = rho
    if edge_slots is not None: d['edge_slots'] = edge_slots
    if payload is not None: d['payload_scale'] = payload
    sc = Scenario(**d)
    tr = make_requests(SimpleNamespace(mu=STUDY.arrival_mu, ids={(k, sc.split): v for k, v in IDS.items()}), sc, seed)
    for key in ('task', 'service', 'quality'):
        np.testing.assert_array_equal(getattr(tr, key), STUDY.times[f'{seed}_{key}'])
    if replicate >= 0:
        tr = resampled_trace(tr, task_mapping(STUDY, replicate, BOOT_MASTER))
    if s['quality'] == 'mean': quality = mean_quality(tr)
    else: quality = (HEADS['carrot'] if s['quality'] == 'carrot' else ORACLE)[tr.task]
    if s['charge'] == 'carrot': seconds = STUDY.prediction_seconds('carrot', seed)+TIMES[str(seed)+'_carrot']
    elif s['charge'] == 'calibrated': seconds = STUDY.prediction_seconds('calibrated', seed)
    else: seconds = np.zeros(len(tr.task))
    kappa = fpi_kappa(STUDY, sc.rho) if s['kappa'] == 'fpi' else np.full(2, float(s['kappa']))
    factory = lambda *a, **k: InformationController(*a, qualities=quality, spec=s, kappa=kappa, **k)
    wrapped = replay.wrapper(STUDY.tasks, sc, seed, tr, seconds)
    result = replay.simulate(wrapped, STUDY.rows, STUDY.profiles, STUDY.forecast, controller_factory=factory)
    metrics = replay.account(wrapped, result, STUDY.rows)
    choices = np.asarray(result['choices'], dtype='<i8')
    row = dict(exp=exp, condition=condition, rho=sc.rho, edge_slots=sc.edge_slots, seed=seed,
                method=method, replicate=replicate, metrics=metrics,
                choices_sha256=hashlib.sha256(choices.tobytes()).hexdigest(),
                cloud_fraction=float(np.mean(choices[sc.warmup:, 0])),
                kappa=kappa.tolist(), max_airtime_sum=result['max_airtime_sum'])
    if payload is not None: row['payload_scale'] = sc.payload_scale
    if details:
        w = sc.warmup; arrivals = np.array([q['arrival'] for q in wrapped['result']['requests']])
        row['_details'] = dict(route=choices[w:, 0], target=tr.quality[w:], predicted=quality[w:],
            shortfall=np.array(result['shortfall']), fee=np.array(result['cost_usd']),
            delay=np.array(result['completion'])[w:]-arrivals[w:])
    return row


DIAGNOSTIC = DATA/'information_value_radio_diagnostic.json'


def plan_diagnostic(study=None):
    """v1.6 radio decision diagnostic: (predictor, condition, payload, seed)."""
    study = study or Study(); seeds = study.config['seeds']
    settings = [('primary', x) for x in (1., 4., 8., 12.)]+[('bandwidth_10000', 1.)]
    return [(p, c, x, s) for p in ('carrot', 'mean') for c, x in settings for s in seeds]


def radio_pair(predictor, condition, payload, seed):
    """Paired Queue/Radio replay; descriptive flip decomposition, not a causal per-decision effect."""
    queue, radio = ('cq', 'cq_release') if predictor == 'carrot' else ('mean_cq', 'mean_release')
    a = episode('diag', condition, None, None, seed, queue, -1, payload, details=True)
    b = episode('diag', condition, None, None, seed, radio, -1, payload, details=True)
    qa, qb = a.pop('_details'), b.pop('_details')
    flip = qa['route'] != qb['route']; n = len(flip); r = np.arange(n)
    qf = lambda d: 12*d['shortfall']+1000*d['fee']
    pred = lambda d: 12*np.maximum(d['target']-d['predicted'][r, d['route']], 0.)
    return dict(predictor=predictor, condition=condition, payload_scale=float(payload), seed=seed,
                queue_choices_sha256=a['choices_sha256'], radio_choices_sha256=b['choices_sha256'],
                queue_objective=a['metrics']['objective'], radio_objective=b['metrics']['objective'],
                flip_rate=float(flip.mean()),
                flips_to_cloud=float(np.mean(qb['route'][flip] == 1)) if flip.any() else 0.,
                quality_fee_change=float((qf(qb)-qf(qa))[flip].sum()/n),
                predicted_quality_change=float((pred(qb)-pred(qa))[flip].sum()/n),
                realized_quality_change=float((12*(qb['shortfall']-qa['shortfall']))[flip].sum()/n),
                delay_change=float(qb['delay'].mean()-qa['delay'].mean()))


def diagnostic_summary():
    """Per predictor and setting: flip rate, direction, and the paired cost decomposition."""
    rows = json.loads(DIAGNOSTIC.read_text(encoding='utf-8'))
    out = {}
    for c, x in [('primary', 1.), ('primary', 4.), ('primary', 8.), ('primary', 12.), ('bandwidth_10000', 1.)]:
        for p in ('carrot', 'mean'):
            R = [r for r in rows if (r['predictor'], r['condition'], r['payload_scale']) == (p, c, x)]
            g = lambda k: np.array([r[k] for r in R])
            out[f'{p}|{c}|{x:g}'] = dict(pairs=len(R), flip_rate=float(g('flip_rate').mean()),
                flips_to_cloud=float(g('flips_to_cloud').mean()),
                total=_ci(g('radio_objective')-g('queue_objective')), quality_fee=_ci(g('quality_fee_change')),
                predicted_quality=_ci(g('predicted_quality_change')),
                realized_quality=_ci(g('realized_quality_change')), delay=_ci(g('delay_change')))
    return out


def verify_diagnostic():
    """Counts, decomposition identity, and hash parity with the v1.4/v1.5 records."""
    rows = json.loads(DIAGNOSTIC.read_text(encoding='utf-8'))
    assert len(rows) == len(plan_diagnostic()) == 200
    ext = load_extension(); main, _ = load_records()
    with gzip.open(DATA/'information_results.json.gz', 'rt', encoding='utf-8') as f: info = json.load(f)
    with gzip.open(DATA/'strengthening_results.json.gz', 'rt', encoding='utf-8') as f: strong = json.load(f)
    ref = {(r['method'], 'primary', r['payload_scale'], r['seed']): r['choices_sha256'] for r in ext if r['exp'] == 'payload'}
    ref.update({(r['method'], r['condition'], 1., r['seed']): r['choices_sha256'] for r in main
                if r['exp'] == 'main' and r['condition'] == 'bandwidth_10000' and r['method'] in ('cq', 'cq_release')})
    ref.update({('mean_release', r['condition'], 1., r['seed']): r['choices_sha256'] for r in info
                if r['condition'] == 'bandwidth_10000' and r['method'] == 'mean_release'})
    ref.update({('mean_cq', r['condition'], 1., r['seed']): r['choices_sha256'] for r in strong
                if r['condition'] == 'bandwidth_10000' and r['replicate'] == -1 and r['method'] == 'mean'})
    for r in rows:
        q, d = ('cq', 'cq_release') if r['predictor'] == 'carrot' else ('mean_cq', 'mean_release')
        key = (r['condition'], r['payload_scale'], r['seed'])
        assert r['queue_choices_sha256'] == ref[(q,)+key] and r['radio_choices_sha256'] == ref[(d,)+key]
        assert abs(r['radio_objective']-r['queue_objective']-r['quality_fee_change']-r['delay_change']) < 1e-9
    expected = json.loads((DATA/'information_value_radio_diagnostic_summary.json').read_text(encoding='utf-8'))
    actual = json.loads(json.dumps(diagnostic_summary()))
    for k in expected:
        for m in expected[k]:
            np.testing.assert_allclose(actual[k][m], expected[k][m], rtol=0, atol=1e-12)
    return dict(passed=True, pairs=200, hash_parity=True)


def plan_extension(study=None):
    """v1.5 jobs: (exp, condition, rho, edge_slots, seed, method, replicate, payload)."""
    study = study or Study(); conds = list(study.config['conditions']); seeds = study.config['seeds']
    jobs = [('payload', 'primary', None, None, s, m, -1, x) for x in PAYLOADS
            for m in ('cq', 'cq_release', 'mean_cq', 'mean_release', 'intrinsic') for s in seeds]
    jobs += [('serial12', c, None, 1, s, m, -1, None) for m in
             ('intrinsic', 'utility', 'shortfall', 'cq', 'fpi', 'cq_release', 'oracle_cq', 'always_cloud',
              'mean_cq', 'mean_release') for c in conds for s in seeds]
    return jobs


def plan(study=None):
    """All published jobs: (exp, condition, rho, edge_slots, seed, method, replicate)."""
    study = study or Study()
    conds = list(study.config['conditions']); seeds = study.config['seeds']; jobs = []
    for m in ('cq', 'cq_release', 'fpi', 'fpi_release', 'always_edge', 'always_cloud', 'oracle_cq'):
        jobs += [('main', c, None, None, s, m, -1) for c in conds for s in seeds]
    for b in BETAS:
        for m in ('cq', 'fpi', 'shortfall', 'utility'):
            jobs += [('beta', c, None, None, s, f'{m}@b{b:g}', -1) for c in conds for s in seeds]
    for k in KAPPAS:
        jobs += [('kappa', c, None, None, s, f'k{k:g}', -1) for c in conds for s in seeds]
    for r in LOADS:
        for m in ('intrinsic', 'shortfall', 'utility', 'cq', 'fpi', 'k0.25', 'k1', 'k2'):
            jobs += [('load', 'primary', r, None, s, m, -1) for s in seeds]
    for r in (.5, .65, .8):
        for m in ('intrinsic', 'shortfall', 'utility', 'cq', 'fpi'):
            jobs += [('serial', 'primary', r, 1, s, m, -1) for s in seeds]
    boot = [('boot', c, None, None, s, m, k) for k in range(200) for c in conds
            for s in BOOT_SEEDS for m in ('cq', 'fpi', 'shortfall', 'utility')]
    return jobs, boot


def load_records():
    with gzip.open(RESULTS, 'rt', encoding='utf-8') as f: rows = json.load(f)
    with gzip.open(BOOTSTRAP, 'rt', encoding='utf-8') as f: boot = json.load(f)
    return rows, boot


def load_extension():
    with gzip.open(EXTENSION, 'rt', encoding='utf-8') as f: return json.load(f)


def extension_summary():
    """v1.5 payload sweep and 12-condition serial-edge evaluation."""
    study = Study(); rows = load_extension(); seeds = study.config['seeds']; conds = list(study.config['conditions'])
    with gzip.open(DATA/'information_results.json.gz', 'rt', encoding='utf-8') as f: info = json.load(f)
    with gzip.open(DATA/'strengthening_results.json.gz', 'rt', encoding='utf-8') as f: strong = json.load(f)
    ref = {(r['method'], r['seed']): r['choices_sha256'] for r in info if r['condition'] == 'primary'}
    ref.update({('mean', r['seed']): r['choices_sha256'] for r in strong
                if r['condition'] == 'primary' and r['replicate'] == -1 and r['method'] == 'mean'})
    names = {'cq': 'carrot_queue', 'cq_release': 'carrot_release', 'mean_cq': 'mean',
             'mean_release': 'mean_release', 'intrinsic': 'carrot_intrinsic'}
    P = {(r['method'], r['payload_scale'], r['seed']): r for r in rows if r['exp'] == 'payload'}
    out = dict(parity_payload1=all(P[m, 1., s]['choices_sha256'] == ref[names[m], s] for m in names for s in seeds))
    air = json.loads((DATA/'airtime_load.json').read_text(encoding='utf-8'))['conditions']['primary']['rho_air']
    pay = {}
    for x in PAYLOADS:
        o = lambda m: np.array([P[m, x, s]['metrics']['objective'] for s in seeds])
        pay[f'{x:g}'] = dict(rho_air=air*x, means={m: float(o(m).mean()) for m in names},
            carrot_radio=_ci(o('cq_release')-o('cq')), mean_radio=_ci(o('mean_release')-o('mean_cq')),
            queue=_ci(o('cq')-o('intrinsic')))
    out['payload'] = pay
    R = {(r['method'], r['condition'], r['seed']): r for r in rows if r['exp'] == 'serial12'}
    S = lambda m, k='objective': np.array([np.mean([R[m, c, s]['metrics'][k] for c in conds]) for s in seeds])
    ms = sorted({k[0] for k in R})
    means = {m: {k: float(S(m, k).mean()) for k in METRICS} for m in ms}
    for m in ms: means[m]['cloud_fraction'] = float(np.mean([R[m, c, s]['cloud_fraction'] for c in conds for s in seeds]))
    comps = {f'{a} - {b}': _ci(S(a)-S(b)) for a, b in (('cq', 'utility'), ('fpi', 'utility'), ('fpi', 'cq'),
             ('shortfall', 'utility'), ('shortfall', 'intrinsic'), ('cq', 'shortfall'), ('cq_release', 'cq'),
             ('mean_release', 'mean_cq'), ('oracle_cq', 'cq'))}
    out['serial12'] = dict(means=means, comparisons=comps)
    return out


def _ci(d):
    d = np.asarray(d); m = float(d.mean()); h = float(t.ppf(.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
    return [m, m-h, m+h]


def summary():
    """Recompute every number reported in the manuscript from the published records."""
    study = Study(); rows, boot = load_records()
    seeds = study.config['seeds']; conds = list(study.config['conditions'])
    with gzip.open(DATA/'information_results.json.gz', 'rt', encoding='utf-8') as f: arch = json.load(f)
    with gzip.open(DATA/'utility_results.json.gz', 'rt', encoding='utf-8') as f: arch += json.load(f)
    A = {(r['method'], r['condition'], r['seed']): r for r in arch}
    E = {}
    for r in rows:
        if r['exp'] in ('main', 'beta', 'kappa'): E[r['method'], r['condition'], r['seed']] = r
    for c in conds:
        for s in seeds: E['carrot_intrinsic', c, s] = A['carrot_intrinsic', c, s]
    parity = dict(
        cq=all(E['cq', c, s]['choices_sha256'] == A['carrot_queue', c, s]['choices_sha256'] for c in conds for s in seeds),
        cq_release=all(E['cq_release', c, s]['choices_sha256'] == A['carrot_release', c, s]['choices_sha256'] for c in conds for s in seeds),
        utility=all(E['utility@b12', c, s]['choices_sha256'] == A['utility_queue', c, s]['choices_sha256'] for c in conds for s in seeds),
        shortfall=all(E['shortfall@b12', c, s]['choices_sha256'] == A['shortfall_queue', c, s]['choices_sha256'] for c in conds for s in seeds))
    obj = lambda r, b: r['metrics']['delay']+b*r['metrics']['shortfall']+1000*r['metrics']['money']

    def sample(m, cs, key='objective', b=None):
        f = (lambda r: r['metrics'][key]) if b is None else (lambda r: obj(r, b))
        return np.array([np.mean([f(E[m, c, s]) for c in cs]) for s in seeds])
    out = dict(parity=parity, scopes={})
    methods = ('cq', 'fpi', 'cq_release', 'fpi_release', 'shortfall@b12', 'utility@b12', 'always_edge',
               'always_cloud', 'oracle_cq', 'carrot_intrinsic', 'k0.125', 'k0.25', 'k1', 'k2')
    pairs = [('fpi', 'cq'), ('fpi', 'utility@b12'), ('cq', 'utility@b12'), ('cq', 'shortfall@b12'),
             ('fpi', 'shortfall@b12'), ('cq_release', 'cq'), ('fpi_release', 'fpi'), ('oracle_cq', 'cq'),
             ('cq', 'carrot_intrinsic')]
    for scope in ['equal12']+conds:
        cs = conds if scope == 'equal12' else [scope]
        means = {m: {k: float(sample(m, cs, k).mean()) for k in METRICS} for m in methods}
        for m in methods:
            means[m]['cloud_fraction'] = float(np.mean([E[m, c, s].get('cloud_fraction', np.nan) for c in cs for s in seeds]))
        comps = {f'{a} - {b}': _ci(sample(a, cs)-sample(b, cs)) for a, b in pairs}
        for a, b in (('cq', 'utility@b12'), ('fpi', 'cq')):
            for k in ('quality', 'delay', 'delay_p95'):
                comps[f'{a} - {b} [{k}]'] = _ci(sample(a, cs, k)-sample(b, cs, k))
        out['scopes'][scope] = dict(means=means, comparisons=comps)
    out['beta'] = {f'{b:g}': {f'{a} - utility': _ci(sample(f'{a}@b{b:g}', conds, b=b)-sample(f'utility@b{b:g}', conds, b=b))
                              for a in ('cq', 'fpi', 'shortfall')} for b in BETAS}
    hind = {}
    for base in ('utility', 'shortfall'):
        best_b = min(BETAS, key=lambda b: sample(f'{base}@b{b:g}', conds, b=12.).mean())
        hind[base] = dict(rule_beta=best_b, objective12=float(sample(f'{base}@b{best_b:g}', conds, b=12.).mean()),
                          cq_minus_best=_ci(sample('cq', conds)-sample(f'{base}@b{best_b:g}', conds, b=12.)),
                          fpi_minus_best=_ci(sample('fpi', conds)-sample(f'{base}@b{best_b:g}', conds, b=12.)))
    out['hindsight'] = hind
    for exp in ('load', 'serial'):
        L = {(r['method'], r['rho'], r['seed']): r for r in rows if r['exp'] == exp}
        res = {}
        for rho in sorted({k[1] for k in L}):
            ms = sorted({k[0] for k in L if k[1] == rho})
            S = lambda m, k='objective': np.array([L[m, rho, s]['metrics'][k] for s in seeds])
            res[f'{rho:g}'] = dict(
                means={m: {k: float(S(m, k).mean()) for k in METRICS} for m in ms},
                comparisons={f'{a} - {b}': _ci(S(a)-S(b)) for a in ms for b in ms if a != b and
                             (a, b) in {('fpi', 'cq'), ('cq', 'utility'), ('fpi', 'utility'), ('cq', 'shortfall'),
                                        ('cq', 'intrinsic'), ('fpi', 'intrinsic'), ('utility', 'intrinsic'),
                                        ('shortfall', 'intrinsic'), ('shortfall', 'utility')}})
        out[exp] = res
    B = {(r['method'], r['condition'], r['seed'], r['replicate']): r['metrics']['objective'] for r in boot}
    reps = sorted({r['replicate'] for r in boot})
    bo = dict(replicates=len(reps))
    for a, b in (('cq', 'utility'), ('cq', 'shortfall'), ('fpi', 'utility'), ('fpi', 'cq'), ('shortfall', 'utility')):
        d = np.array([np.mean([B[a, c, s, k]-B[b, c, s, k] for c in conds for s in BOOT_SEEDS]) for k in reps])
        bo[f'{a} - {b}'] = dict(mean=float(d.mean()), p2_5=float(np.percentile(d, 2.5)),
                                p97_5=float(np.percentile(d, 97.5)), fraction_negative=float(np.mean(d < 0)))
    out['bootstrap'] = bo
    return out


def fpi_gate(result):
    """Declared before outcomes: equal-12 upper CI <= 0 and not worse than CQ at any load."""
    upper = result['scopes']['equal12']['comparisons']['fpi - cq'][2]
    loads = {r: v['means']['fpi']['objective'] <= v['means']['cq']['objective'] for r, v in result['load'].items()}
    return dict(equal12_upper=upper, not_worse_by_load=loads, passed=bool(upper <= 0 and all(loads.values())))


def verify():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((DATA/'information_value_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    for name, expected in manifest['source_sha256_lf'].items():
        content = (root/name).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(content).hexdigest() == expected, name
    rows, boot = load_records(); jobs, bjobs = plan()
    key = lambda r: (r['exp'], r['condition'], r['seed'], r['method'], r['replicate'], r['rho'], r['edge_slots'])
    assert len(rows) == len(jobs) == 9500 and len({key(r) for r in rows}) == 9500
    assert len(boot) == len(bjobs) == 19200 and len({key(r) for r in boot}) == 19200
    for r in rows+boot:
        v = r['metrics']; assert abs(v['objective']-v['delay']-12*v['shortfall']-1000*v['money']) < 1e-9
        assert 0 <= r['max_airtime_sum'] <= 1+1e-10
    expected = json.loads((DATA/'information_value_summary.json').read_text(encoding='utf-8'))
    actual = json.loads(json.dumps(summary()))

    def compare(a, b):
        if isinstance(a, dict):
            assert set(a) == set(b); [compare(a[k], b[k]) for k in a]
        elif isinstance(a, list):
            assert len(a) == len(b); [compare(x, y) for x, y in zip(a, b)]
        elif isinstance(a, float):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
        else:
            assert a == b
    compare(actual, expected)
    assert all(actual['parity'].values())
    gate = fpi_gate(actual); assert gate['passed'] is False
    ext = load_extension(); ejobs = plan_extension()
    ekey = lambda r: (r['exp'], r['condition'], r['seed'], r['method'], r['edge_slots'], r.get('payload_scale'))
    assert len(ext) == len(ejobs) == 2900 and len({ekey(r) for r in ext}) == 2900
    for r in ext:
        v = r['metrics']; assert abs(v['objective']-v['delay']-12*v['shortfall']-1000*v['money']) < 1e-9
    eexpected = json.loads((DATA/'information_value_extension_summary.json').read_text(encoding='utf-8'))
    eactual = json.loads(json.dumps(extension_summary())); compare(eactual, eexpected)
    assert eactual['parity_payload1']
    diagnostic = verify_diagnostic()
    return dict(passed=True, paths=9500, bootstrap_paths=19200, extension_paths=2900,
                radio_diagnostic_pairs=diagnostic['pairs'],
                archived_parity=actual['parity'], payload1_parity=True,
                fpi_gate_passed=gate['passed'], provider_calls=0, new_model_responses=0)
