"""Summarize the exploratory Fresh450 static-split comparison.

This analysis was added after confirmation. static_split.jsonl.gz holds the static-split paths (run_static.py, protocol in
data/information_value_review_plan.json); queue-aware references are the confirmation's own paths in
../results/full.jsonl.gz. Averages cover 12 conditions and 20 traffic seeds, with paired 95% t intervals.
"""
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent


def ci(d):
    d = np.asarray(d); m = float(d.mean()); h = float(t.ppf(.975, len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
    return [m, m-h, m+h]


def main():
    static = [json.loads(x) for x in gzip.open(HERE/'static_split.jsonl.gz', 'rt', encoding='utf-8')]
    full = [json.loads(x) for x in gzip.open(HERE.parent/'results/full.jsonl.gz', 'rt', encoding='utf-8')]
    conds = sorted({r['condition'] for r in static if r['exp'] == 'static12'}); seeds = sorted({r['seed'] for r in static})
    out = {}
    for slots, exp in ((1, 'serial'), (4, 'table')):
        S = {(r['method'], r['condition'], r['seed']): r['objective'] for r in static if r['exp'] == 'static12' and r['edge_slots'] == slots}
        Q = {(r['method'], r['condition'], r['seed']): r['objective'] for r in full if r['exp'] == exp and r['edge_slots'] == slots}
        avg = lambda D, m: np.array([np.mean([D[m, c, s] for c in conds]) for s in seeds])
        grid = {m: avg(S, m) for m in sorted({k[0] for k in S})}
        best = min(grid, key=lambda m: grid[m].mean())
        res = dict(static_prop=float(grid['static_prop'].mean()), static_best=best, static_best_cost=float(grid[best].mean()))
        for m in ('utility', 'shortfall', 'cq', 'fpi'):
            res[m] = float(avg(Q, m).mean()); res[f'{m} - static_best'] = ci(avg(Q, m)-grid[best])
        out[f'equal12_slots{slots}'] = res
    text = json.dumps(out, indent=1)+'\n'
    (HERE/'summary.json').write_text(text, encoding='utf-8', newline='\n')
    print(text, end='')


if __name__ == '__main__':
    main()
