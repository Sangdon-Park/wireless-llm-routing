"""Post-confirmation Fresh450 static-split experiment (exploratory).

This archived runner requires the private assembly environment, including
confirm.py and private/bank_declared.json.gz. It uses confirm.install and the
public qbr controllers. Queue-aware reference rows come from results/full.jsonl.
The experiment was added after confirmation and is not a confirmatory hypothesis.
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import confirm  # noqa: E402

BANK = str(HERE/'private/bank_declared.json.gz')
OUT = HERE/'exploratory_v18'/'static_fresh.jsonl'
GRID = [round(0.05*i, 2) for i in range(1, 19)]


def init():
    confirm.install(BANK)


def run(job):
    exp, cond, rho, slots, seed, method = job
    r = confirm.iv.episode(exp, cond, rho, slots, seed, method)
    return dict(exp=exp, condition=cond, rho=r['rho'], edge_slots=r['edge_slots'], seed=seed, method=method,
                objective=r['metrics']['objective'], metrics=r['metrics'], cloud_fraction=r['cloud_fraction'])


if __name__ == '__main__':
    confirm.install(BANK)
    conds = list(confirm.iv.STUDY.config['conditions']); seeds = confirm.iv.STUDY.config['seeds']
    static = ['static_prop']+[f'static_p{p:g}' for p in GRID]
    jobs = []
    for slots in (None, 1):
        jobs += [('static12', c, None, slots, x, m) for m in static for c in conds for x in seeds]
        jobs += [('static_load', 'primary', r, slots, x, m) for r in (.35, .9) for m in static for x in seeds]
    jobs += [('queue_load', 'primary', r, 1, x, m) for r in (.35, .9) for m in ('intrinsic', 'shortfall', 'cq', 'fpi', 'utility') for x in seeds]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open('x', encoding='utf-8') as f, ProcessPoolExecutor(22, initializer=init) as pool:
        for i, fu in enumerate(as_completed([pool.submit(run, j) for j in jobs]), 1):
            f.write(json.dumps(fu.result())+'\n')
            if i % 1000 == 0: print(i, flush=True)
    print('complete', len(jobs), flush=True)
