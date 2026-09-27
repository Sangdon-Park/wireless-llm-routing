"""Draw the four value-of-information panels from archived runs (v1.8).

(a) one serial edge slot: cost versus load, including the capacity-aware static split chosen in hindsight;
(b) one serial edge slot: rule refinements against Utility-Queue at matched quality weights;
(c) radio correction versus nominal airtime load (four edge slots; one-slot Mean payload curve dotted);
(d) four edge slots: cost versus correlation of predicted and realized route-quality differences.
Plotting uses archived results; intervals condition on the recorded task bank.
"""
from pathlib import Path
import gzip, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE/'data'
OUT = HERE/'outputs'/'figures'
AIR = {c: v['rho_air'] for c, v in json.loads((DATA/'airtime_load.json').read_text())['conditions'].items()}
C = {'blue': '#2166AC', 'orange': '#C47717', 'green': '#21836B', 'purple': '#8A5EA4', 'grey': '#6B7680', 'red': '#B2182B'}
GRID = [round(0.05*i, 2) for i in range(1, 19)]


def ci(a):
    a = np.asarray(a); return a.mean(), t.ppf(.975, len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))


def read_gzip(p):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        return json.load(f)


def discrimination():
    """Mix-weighted per-service correlation of predicted and realized route-quality differences."""
    from qbr import information_value as iv
    from qbr.radio import MIX, SERVICES
    iv.initialize()
    svc = np.array([t_['service'] for t_ in iv.STUDY.tasks]); truth = iv.ORACLE[:, 1]-iv.ORACLE[:, 0]
    class Trace: pass
    tr = Trace(); tr.task = np.arange(len(svc)); tr.service = np.array([SERVICES.index(s) for s in svc])
    heads = dict(np.load(DATA/'information_predictions.npz'))
    preds = {'CARROT': heads['carrot'], 'Mean': iv.mean_quality(tr), 'RouteLLM-score': heads['routellm_score']}
    for sg in (0., .05, .1, .2, .3, .5): preds[f'noisy{sg:g}'] = iv.noisy_oracle(sg, len(svc))
    out = {}
    for k, p in preds.items():
        d = p[:, 1]-p[:, 0]
        out[k] = float(sum(MIX[i]*(np.corrcoef(d[svc == s], truth[svc == s])[0, 1] if np.std(d[svc == s]) > 0 else 0.)
                           for i, s in enumerate(SERVICES)))
    return out


def main():
    main_rows = read_gzip(DATA/'information_value_results.json.gz')
    ext = read_gzip(DATA/'information_value_extension.json.gz')
    rev = read_gzip(DATA/'information_value_review.json.gz')
    arch = [r for r in json.loads((DATA/'expected_results.json').read_text()) if r['scheduler'] == 'paper']
    arch += [r for r in read_gzip(DATA/'strengthening_results.json.gz')]
    arch += read_gzip(DATA/'information_results.json.gz')
    seeds = sorted({r['seed'] for r in main_rows if r['exp'] == 'main'})
    conds = sorted({r['condition'] for r in main_rows if r['exp'] == 'main'})
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.2, 'axes.labelsize': 7.3, 'axes.titlesize': 8,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65})
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 3.7))
    data = {}
    obj = lambda r: r['metrics']['objective']

    # (a) one serial edge slot, cost versus load.
    L = {}
    for r in main_rows:
        if r['exp'] == 'serial': L[(r['method'], round(r['rho'], 2), r['seed'])] = obj(r)
    for r in rev:
        if r['exp'] == 'serial_load' or (r['exp'] == 'static_load' and r['edge_slots'] == 1):
            L[(r['method'], round(r['rho'], 2), r['seed'])] = obj(r)
    rhos = [.35, .5, .65, .8, .9]
    best = {rho: min(['static_prop']+[f'static_p{p:g}' for p in GRID],
                     key=lambda m: np.mean([L[m, rho, s] for s in seeds])) for rho in rhos}
    series = [('intrinsic', 'Intrinsic (no queue)', C['grey'], 'v'), ('static', 'Static split, best in hindsight', C['red'], 'x'),
              ('utility', 'Utility–Queue', C['purple'], '^'), ('shortfall', 'Shortfall–Queue', C['green'], 's'),
              ('cq', 'CARROT–Queue', C['orange'], 'D'), ('fpi', 'FPI–Queue', C['blue'], 'o')]
    for m, lab, col, mk in series:
        key = (lambda rho: best[rho]) if m == 'static' else (lambda rho: m)
        st = [ci([L[key(rho), rho, s] for s in seeds]) for rho in rhos]
        y = np.array([a for a, _ in st]); h = np.array([b for _, b in st])
        ax[0, 0].plot(rhos, y, color=col, marker=mk, ms=3.2, lw=1.1, label=lab)
        ax[0, 0].fill_between(rhos, y-h, y+h, color=col, alpha=.12, lw=0)
        data.setdefault('a', {})[m] = dict(mean=y.tolist(), half95=h.tolist(), static_choice=[best[r] for r in rhos] if m == 'static' else None)
    ax[0, 0].set(xlabel='Normalized arrival load $\\rho$', ylabel='Mean weighted cost (s)', xticks=rhos, ylim=(5.5, 8.4))
    ax[0, 0].legend(frameon=False, fontsize=5.5, loc='upper left', ncol=1, handlelength=1.5)
    ax[0, 0].set_title('(a) Queue state, one edge server', loc='left', fontweight='bold')

    # (b) one serial edge slot, matched quality weight, paired differences to Utility-Queue.
    B = {(r['method'], r['condition'], r['seed']): r['metrics'] for r in rev if r['exp'] == 'serial_beta'}
    betas = sorted({float(k[0].split('@b')[1]) for k in B})
    cost = lambda v, b: v['delay']+b*v['shortfall']+1000*v['money']
    for m, lab, col, mk, dx in [('shortfall', 'Shortfall', C['green'], 's', -.04), ('cq', 'CARROT–Queue', C['orange'], 'D', 0.),
                                ('fpi', 'FPI', C['blue'], 'o', .04)]:
        st = [ci([np.mean([cost(B[f'{m}@b{b:g}', c, s], b)-cost(B[f'utility@b{b:g}', c, s], b) for c in conds]) for s in seeds])
              for b in betas]
        y = np.array([a for a, _ in st]); h = np.array([q for _, q in st]); x = np.array(betas)*np.exp(dx)
        ax[0, 1].errorbar(x, y, yerr=h, color=col, marker=mk, ms=3.2, lw=1.0, capsize=2, label=lab)
        data.setdefault('b', {})[m] = dict(beta=betas, mean=y.tolist(), half95=h.tolist())
    ax[0, 1].axhline(0, color='#4B5259', ls='--', lw=.7)
    ax[0, 1].set_xscale('log'); ax[0, 1].set_xticks(betas, labels=[f'{b:g}' for b in betas]); ax[0, 1].minorticks_off()
    ax[0, 1].set(xlabel='Quality weight $\\beta$ (s), router and cost', ylabel='Cost minus Utility–Queue (s)')
    ax[0, 1].legend(frameon=False, fontsize=5.8, loc='upper left')
    ax[0, 1].set_title('(b) Rule refinements, one edge server', loc='left', fontweight='bold')

    # (c) radio correction versus nominal airtime load.
    A = {(r['method'], r['condition'], r['seed']): obj(r) for r in arch if r.get('replicate', -1) == -1}
    N = {(r['method'], r['condition'], r['seed']): obj(r) for r in main_rows if r['exp'] == 'main'}
    X = {(r['method'], r['payload_scale'], r['seed']): obj(r) for r in ext if r['exp'] == 'payload'}
    X1 = {(r['method'], r['payload_scale'], r['seed']): obj(r) for r in rev if r['exp'] == 'serial_payload'}
    specs = sorted(AIR, key=AIR.get); xb = np.array([AIR[c] for c in specs])
    scales = sorted({k[1] for k in X}); xp = AIR['primary']*np.array(scales)
    rel = lambda add, base: [100*(add[s]-base[s])/np.mean(list(base.values())) for s in seeds]
    for name, lab, col, mk, bw, pay in [
            ('mean', 'Mean predictor', C['blue'], 'o', (A, 'mean_release', 'mean'), ('mean_release', 'mean_cq')),
            ('carrot', 'CARROT', C['orange'], 'D', (N, 'cq_release', 'cq'), ('cq_release', 'cq'))]:
        src, add, base = bw
        st = [ci(rel({s: src[add, c, s] for s in seeds}, {s: src[base, c, s] for s in seeds})) for c in specs]
        y = np.array([a for a, _ in st]); h = np.array([q for _, q in st])
        ax[1, 0].plot(xb, y, color=col, marker=mk, ms=3, lw=1., ls='--', mfc='white', label=f'{lab}, bandwidth')
        data.setdefault('c', {})[name+'_bandwidth'] = dict(airtime=xb.tolist(), mean_pct=y.tolist(), half95=h.tolist())
        st = [ci(rel({s: X[pay[0], x, s] for s in seeds}, {s: X[pay[1], x, s] for s in seeds})) for x in scales]
        y = np.array([a for a, _ in st]); h = np.array([q for _, q in st])
        ax[1, 0].plot(xp, y, color=col, marker=mk, ms=3.2, lw=1.2, label=f'{lab}, payload')
        ax[1, 0].fill_between(xp, y-h, y+h, color=col, alpha=.12, lw=0)
        data['c'][name+'_payload'] = dict(airtime=xp.tolist(), scale=scales, mean_pct=y.tolist(), half95=h.tolist())
    st = [ci(rel({s: X1['mean_release', x, s] for s in seeds}, {s: X1['mean_cq', x, s] for s in seeds})) for x in scales]
    y = np.array([a for a, _ in st]); h = np.array([q for _, q in st])
    ax[1, 0].plot(xp, y, color=C['grey'], marker='o', ms=2.8, lw=1., ls=':', label='Mean, payload, one edge server')
    data['c']['mean_payload_one_slot'] = dict(airtime=xp.tolist(), mean_pct=y.tolist(), half95=h.tolist())
    ax[1, 0].axhline(0, color='#4B5259', ls='--', lw=.7); ax[1, 0].set_xscale('log')
    ticks = [.012, .04, .12, .4, 1.2]
    ax[1, 0].set_xticks(ticks, labels=[f'{v:g}' for v in ticks]); ax[1, 0].minorticks_off()
    ax[1, 0].set(xlabel=r'Nominal airtime load $\rho_{\mathrm{air}}$', ylabel='Radio minus Queue cost (%)')
    ax[1, 0].legend(frameon=False, fontsize=5.4, loc='lower left', ncol=1, handlelength=2.2)
    ax[1, 0].set_title('(c) Radio correction', loc='left', fontweight='bold')

    # (d) predictor discrimination versus cost (four edge slots, 12 conditions, CARROT-Queue controller).
    disc = discrimination()
    Z = {(r['method'], r['condition'], r['seed']): obj(r) for r in rev if r['exp'] == 'noisy'}
    xs, ys, hs = [], [], []
    for sg in (0., .05, .1, .2, .3, .5):
        m = f'noisy{sg:g}'; a, h = ci([np.mean([Z[m, c, s] for c in conds]) for s in seeds])
        xs.append(disc[m]); ys.append(a); hs.append(h)
    ax[1, 1].errorbar(xs, ys, yerr=hs, color=C['grey'], marker='o', ms=3, lw=1., capsize=2, label='Noisy oracle ($\\sigma$ = 0–0.5)')
    data['d'] = dict(noisy=dict(discrimination=xs, cost=ys, half95=hs))
    point = {'CARROT': ([N['cq', c, s] for c in conds for s in seeds], C['orange'], 'D'),
             'Mean': ([A['mean', c, s] for c in conds for s in seeds], C['blue'], 'o'),
             'RouteLLM-score': ([A['routellm_score_queue', c, s] for c in conds for s in seeds], C['purple'], '^')}
    for k, (vals, col, mk) in point.items():
        v = np.array(vals).reshape(len(conds), len(seeds)).mean(0); a, h = ci(v)
        ax[1, 1].errorbar([disc[k]], [a], yerr=[h], color=col, marker=mk, ms=4, capsize=2, ls='none', label=k)
        data['d'][k] = dict(discrimination=disc[k], cost=a, half95=h)
    ax[1, 1].set(xlabel='Correlation of predicted and realized route difference', ylabel='Mean weighted cost (s)', xlim=(0, 1.05))
    ax[1, 1].legend(frameon=False, fontsize=5.6, loc='upper right')
    ax[1, 1].set_title('(d) Quality information', loc='left', fontweight='bold')

    for p in ax.flat:
        p.set_axisbelow(True); p.grid(axis='y', alpha=.18, lw=.5); p.tick_params(width=.6, length=3)
    fig.subplots_adjust(left=.075, right=.99, bottom=.115, top=.925, wspace=.3, hspace=.72)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT/'results_information.pdf', bbox_inches='tight'); fig.savefig(OUT/'results_information.png', dpi=220, bbox_inches='tight')
    (OUT/'information_value_figure_data.json').write_text(json.dumps(data, indent=1))
    print(json.dumps({k: data['d'][k] for k in ('CARROT', 'Mean', 'RouteLLM-score')}, indent=1))


if __name__ == '__main__':
    main()
