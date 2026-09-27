"""Frozen fresh450 replay and confirmatory analysis on top of the unchanged public controllers.

Uses the qbr.information_value controllers with their data globals replaced by the fresh bank:
CARROT and forest predictions from predictions/ (computed before unsealing), per-task mean
prediction runtime, calibration serving state unchanged. Decisions follow commitment.json and
executor_commitment.json. Usage:
  python confirm.py run --bank private/bank.json.gz --out results        (real, once)
  python confirm.py run --bank private_synthetic/bank.json.gz --out synthetic --replicates 3
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from paths import REPOSITORY_ROOT
from types import SimpleNamespace
import argparse
import gzip
import json
import sys
import numpy as np
from scipy.stats import t as student

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT))
from qbr import information_value as iv  # noqa: E402
from qbr.radio import SERVICES, make_requests  # noqa: E402
from fresh_bank import FreshTraffic  # noqa: E402

MASTER = 2026092501
KNOWN_EXPOSED = ('ifbench_multiturn_ifbench_2473', 'ifbench_multiturn_ifbench_3018', 'xsum_38693675')
BOOT_SEEDS = (23924000, 23924001)


class FreshStudy:
    def __init__(self, bank_path, variant='mid', exclude=()):
        f = FreshTraffic(); bank = json.load(gzip.open(bank_path, 'rt', encoding='utf-8'))
        self.config, self.forecast, self.arrival_mu = f.config, f.forecast, f.arrival_mu
        self.tasks = f.tasks; index = {t['id']: i for i, t in enumerate(self.tasks)}
        self.rows, self.profiles = [], {}
        quality = np.full((450, 2), np.nan); ready = []
        for task in bank['tasks']:
            k = index[task['id']]
            if not task['replay_ready'] or task['id'] in exclude:
                continue
            mode = 'strict512' if task['service'] == 'dialogue' else 'full'; budget = {}
            for r, name in enumerate(('edge', 'cloud')):
                x = task['routes'][name]
                q = {'mid': x['quality'], 'low': x['quality_low'], 'high': x['quality_high']}[variant]
                quality[k, r] = q
                budget[name] = dict(quality=q, service_s=x['service_s'], payload_bytes=x['payload_bytes'],
                    all_attempt_payload_bytes=x['all_attempt_payload_bytes'], terminal_status=x['terminal_status'],
                    response_sha256=x['response_sha256'], input_tokens=x['input_tokens'],
                    cached_input_tokens=x['cached_input_tokens'], output_tokens=x['output_tokens'],
                    thought_tokens=x['thought_tokens'])
                self.profiles[task['id'], r] = ([{'time_s': a, 'utf8_bytes': b} for a, b in x['chunks']],
                                                x['payload_bytes'], x['service_s'], x['resets'])
            self.rows.append(dict(id=task['id'], service=task['service'], budgets={mode: budget}))
            ready.append(k)
        self.ready = sorted(ready); self.oracle = quality
        self.ids = {s: [i for i in self.ready if self.tasks[i]['service'] == n] for s, n in enumerate(SERVICES)}
        carrot = np.load(HERE/'predictions/carrot.npz'); forest = np.load(HERE/'predictions/forest.npz')
        self.carrot = carrot['quality']; self.forest_mean = forest['mean_quality']
        self.task_seconds = {}
        traces = np.load(HERE/'predictions/traces.npz')
        for name, arr in (('carrot', carrot), ('calibrated', forest)):
            total = np.zeros(450); count = np.zeros(450)
            for seed in self.config['seeds']:
                np.add.at(total, traces[f'{seed}_task'], arr[f'{seed}_seconds']); np.add.at(count, traces[f'{seed}_task'], 1)
            self.task_seconds[name] = np.where(count > 0, total/np.maximum(count, 1), np.nan)
            fill = np.nanmean(self.task_seconds[name])   # tasks never drawn in timing traces: service-free mean
            self.task_seconds[name] = np.where(np.isfinite(self.task_seconds[name]), self.task_seconds[name], fill)
        self.times = {}
        for seed in self.config['seeds']:
            tr = make_requests(SimpleNamespace(mu=self.arrival_mu, ids={(k, 'calibration'): v for k, v in self.ids.items()}),
                               iv.Scenario(**self.config['conditions']['primary']), seed)
            for key in ('task', 'service', 'quality'): self.times[f'{seed}_{key}'] = getattr(tr, key)

    def prediction_seconds(self, method, seed):
        return self.task_seconds[method][self.times[f'{seed}_task']]


def install(bank_path, variant='mid', exclude=()):
    s = FreshStudy(bank_path, variant, tuple(exclude))
    iv.STUDY = s; iv.HEADS = {'carrot': s.carrot}; iv.ORACLE = s.oracle; iv.IDS = s.ids
    iv.TIMES = {f'{seed}_carrot': np.zeros(1400) for seed in s.config['seeds']}
    iv.mean_quality = lambda tr: s.forest_mean[tr.task]

    def mapping(study, replicate, master):
        rng = np.random.default_rng(np.random.SeedSequence([MASTER, int(replicate)])); m = np.arange(450)
        for k in range(3):
            ids = np.array(s.ids[k]); m[ids] = rng.choice(ids, size=len(ids), replace=True)
        return m
    iv.task_mapping = mapping


def init(bank_path, variant, exclude):
    install(bank_path, variant, exclude)


def job(args):
    exp, cond, rho, slots, seed, method, rep, payload = args
    r = iv.episode(exp, cond, rho, slots, seed, method, rep, payload)
    return dict(exp=exp, condition=cond, rho=r['rho'], edge_slots=r['edge_slots'], payload=payload, seed=seed,
                method=method, replicate=rep, objective=r['metrics']['objective'], metrics=r['metrics'],
                cloud_fraction=r['cloud_fraction'])


def plans(conds, seeds, replicates):
    boot = []
    for rep in range(replicates):
        for s in BOOT_SEEDS:
            boot += [('H1', 'primary', rho, None, s, m, rep, None) for rho in (.35, .9) for m in ('shortfall', 'intrinsic')]
            boot += [('H2', c, None, None, s, m, rep, None) for c in conds for m in ('oracle_cq', 'cq', 'utility')]
            boot += [('H3', 'primary', None, None, s, m, rep, x) for x in (1., 12.) for m in ('mean_cq', 'mean_release')]
            boot += [('H4', c, None, 1, s, m, rep, None) for c in conds for m in ('fpi', 'cq')]
    full = []
    for s in seeds:
        full += [('table', c, None, None, s, m, -1, None) for c in conds for m in
                 ('intrinsic', 'utility', 'shortfall', 'cq', 'fpi', 'cq_release', 'oracle_cq', 'mean_cq', 'mean_release',
                  'always_edge', 'always_cloud')]
        full += [('serial', c, None, 1, s, m, -1, None) for c in conds for m in
                 ('utility', 'shortfall', 'cq', 'fpi', 'cq_release', 'oracle_cq', 'mean_cq', 'mean_release', 'always_cloud')]
        full += [('load', 'primary', rho, None, s, m, -1, None) for rho in (.35, .5, .65, .8, .9)
                 for m in ('intrinsic', 'shortfall', 'utility', 'cq', 'fpi')]
        full += [('payload', 'primary', None, None, s, m, -1, x) for x in (1., 2., 4., 8., 12.)
                 for m in ('cq', 'cq_release', 'mean_cq', 'mean_release')]
        full += [('beta', c, None, None, s, f'{m}@b{b:g}', -1, None) for c in conds
                 for b in (4., 8., 12., 16., 24., 36.) for m in ('cq', 'shortfall', 'utility', 'fpi')]
    return boot, full


def sensitivity_plan(conds, seeds):
    out = []
    for s in seeds:
        out += [('H1', 'primary', rho, None, s, m, -1, None) for rho in (.35, .9) for m in ('shortfall', 'intrinsic')]
        out += [('H2', c, None, None, s, m, -1, None) for c in conds for m in ('oracle_cq', 'cq', 'utility')]
        out += [('H3', 'primary', None, None, s, m, -1, x) for x in (1., 12.) for m in ('mean_cq', 'mean_release')]
        out += [('H4', c, None, 1, s, m, -1, None) for c in conds for m in ('fpi', 'cq')]
    return out


def execute(bank, variant, exclude, jobs, path, workers):
    path = Path(path)
    if path.exists():
        return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    rows = []
    with ProcessPoolExecutor(workers, initializer=init, initargs=(bank, variant, tuple(exclude))) as pool:
        for r in pool.map(job, jobs, chunksize=4):
            rows.append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n', encoding='utf-8')
    return rows


def summarize(rows, conds, seeds, replicate_axis):
    O = {}
    for r in rows:
        O[(r['exp'], r['condition'], round(r['rho'], 4), r['edge_slots'], r['payload'], r['method'], r['seed'], r['replicate'])] = r['objective']
    rho_c = {c: round(iv.Scenario(**iv.STUDY.config['conditions'][c]).rho, 4) for c in conds}
    units = sorted({(r['replicate'] if replicate_axis else r['seed']) for r in rows if r['exp'] in ('H1', 'H2', 'H3', 'H4')})

    def S(exp, cond, rho, slots, payload, method):
        vals = []
        for u in units:
            ss = BOOT_SEEDS if replicate_axis else [u]; rep = u if replicate_axis else -1
            vals.append(np.mean([O[(exp, cond, rho, slots, payload, method, s, rep)] for s in ss]))
        return np.array(vals)

    def over(exp, slots, method):
        return np.mean([S(exp, c, rho_c[c], slots, None, method) for c in conds], axis=0)
    h1_9 = S('H1', 'primary', .9, 4, None, 'shortfall')-S('H1', 'primary', .9, 4, None, 'intrinsic')
    h1_35 = S('H1', 'primary', .35, 4, None, 'shortfall')-S('H1', 'primary', .35, 4, None, 'intrinsic')
    oracle = over('H2', 4, 'oracle_cq')-over('H2', 4, 'cq'); refine = over('H2', 4, 'cq')-over('H2', 4, 'utility')
    q1 = S('H3', 'primary', .65, 4, 1., 'mean_cq'); r1 = S('H3', 'primary', .65, 4, 1., 'mean_release')
    q12 = S('H3', 'primary', .65, 4, 12., 'mean_cq'); r12 = S('H3', 'primary', .65, 4, 12., 'mean_release')
    h4 = over('H4', 1, 'fpi')-over('H4', 1, 'cq')
    if replicate_axis:
        iv_ = lambda d: [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    else:
        def iv_(d):
            m = float(np.mean(d)); h = float(student.ppf(.975, len(d)-1)*np.std(d, ddof=1)/np.sqrt(len(d))); return [m, m-h, m+h]
    rel1 = (r1-q1)/q1
    out = dict(units=len(units),
        H1=dict(diff_rho09=iv_(h1_9), diff_rho035=iv_(h1_35), growth=iv_(h1_9-h1_35)),
        H2=dict(oracle_minus_cq=iv_(oracle), cq_minus_utility=iv_(refine)),
        H3=dict(scale12=iv_(r12-q12), scale1_relative=iv_(rel1)),
        H4=dict(fpi_minus_cq_serial=iv_(h4)))
    out['decisions'] = dict(
        H1=bool(out['H1']['diff_rho09'][2] < 0 and out['H1']['growth'][2] < 0),
        H2=bool(out['H2']['oracle_minus_cq'][1] < 0 and out['H2']['oracle_minus_cq'][2] < -abs(out['H2']['cq_minus_utility'][0])),
        H3=bool(out['H3']['scale12'][2] < 0 and -0.007 <= out['H3']['scale1_relative'][1] and out['H3']['scale1_relative'][2] <= 0.007),
        H4=bool(out['H4']['fpi_minus_cq_serial'][2] < 0))
    out['confirmed'] = int(sum(out['decisions'].values()))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=('run',))
    ap.add_argument('--bank', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--replicates', type=int, default=2000); ap.add_argument('--seeds', type=int, default=20); ap.add_argument('--workers', type=int, default=20)
    a = ap.parse_args(); out = HERE/a.out; out.mkdir(exist_ok=True)
    install(a.bank)
    conds = list(iv.STUDY.config['conditions']); seeds = iv.STUDY.config['seeds'][:a.seeds]
    boot, full = plans(conds, seeds, a.replicates)
    result = dict(bank=a.bank, ready_tasks=len(iv.STUDY.ready), per_service=[len(v) for v in iv.STUDY.ids.values()])
    rows = execute(a.bank, 'mid', (), full, out/'full.jsonl', a.workers)
    result['primary_t'] = summarize(execute(a.bank, 'mid', (), sensitivity_plan(conds, seeds), out/'h_mid.jsonl', a.workers),
                                    conds, seeds, False)
    for variant in ('low', 'high'):
        rows_v = execute(a.bank, variant, (), sensitivity_plan(conds, seeds), out/f'h_{variant}.jsonl', a.workers)
        result[f'quality_{variant}_t'] = summarize(rows_v, conds, seeds, False)
    rows_e = execute(a.bank, 'mid', KNOWN_EXPOSED, sensitivity_plan(conds, seeds), out/'h_exclude_exposed.jsonl', a.workers)
    result['exclude_known_exposed_t'] = summarize(rows_e, conds, seeds, False)
    rows_b = execute(a.bank, 'mid', (), boot, out/'bootstrap.jsonl', a.workers)
    result['primary_bootstrap'] = summarize(rows_b, conds, seeds, True)
    (out/'result.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(dict(decisions=result['primary_bootstrap']['decisions'], confirmed=result['primary_bootstrap']['confirmed'])))


if __name__ == '__main__':
    main()
