"""Verify the public confirmation files and recompute decisions from published results.

No private data is needed: prompts, responses, and per-task grades are not published.
Public-export checksums verify file integrity, not the date of the archived records.
  python confirmation/verify_confirmation.py
"""
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import numpy as np
from scipy.stats import t as student

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BOOT_SEEDS = (23924000, 23924001)
PRIVATE = {'predictions/prompts.json'}  # task inputs are not redistributed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain():
    manifest = json.loads((HERE/'publication_manifest.json').read_text(encoding='utf-8'))
    for section in ('records', 'executor_files'):
        for name, item in manifest[section].items():
            assert sha(ROOT/name) == item['published_sha256'], name
    for name, expected in manifest['controller_snapshot'].items():
        assert sha(ROOT/name) == expected, name
    for name, expected in manifest['unchanged_numerical_files'].items():
        assert sha(ROOT/name) == expected, name
    for name, item in manifest['source_files_lf'].items():
        content = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(content).hexdigest() == item['published_sha256'], name
    plan = json.loads((HERE/'commitment.json').read_text(encoding='utf-8'))
    for name, expected in plan['frozen_methods']['source_sha256'].items():
        assert sha(HERE/'frozen_source'/name) == expected, name
    ec = json.loads((HERE/'executor_commitment.json').read_text(encoding='utf-8'))
    receipt = json.loads((HERE/'exposure_receipt.json').read_text(encoding='utf-8'))
    assert sha(HERE/'commitment.json') == ec['analysis_commitment_sha256'] == receipt['analysis_commitment_sha256']
    assert sha(HERE/'executor_commitment.json') == receipt['executor_commitment_sha256']
    checked = []
    for name, expected in ec['executor_sha256'].items():
        if name in PRIVATE:
            continue
        path = HERE/('executor/'+name if name.endswith('.py') else name)
        assert sha(path) == expected, name
        checked.append(name)
    return dict(verification_scope='Public export integrity and recomputation of published decisions; archived timing is not independently verified.',
                analysis_commitment=sha(HERE/'commitment.json'), executor_commitment=sha(HERE/'executor_commitment.json'),
                exposure_receipt=sha(HERE/'exposure_receipt.json'), executor_files_verified=len(checked),
                archived_controller_files_verified=len(plan['frozen_methods']['source_sha256']),
                unchanged_numerical_files_verified=len(manifest['unchanged_numerical_files']),
                private_files_not_published=sorted(PRIVATE))


def rows_from(folder):
    out = []
    for name in ('h_mid', 'h_low', 'h_high', 'h_exclude_exposed'):
        with gzip.open(folder/f'{name}.jsonl.gz', 'rt', encoding='utf-8') as f:
            out.append((name, [json.loads(x) for x in f]))
    with gzip.open(folder/'bootstrap_objectives.csv.gz', 'rt', encoding='utf-8') as f:
        boot = []
        for r in csv.DictReader(f):
            boot.append(dict(exp=r['exp'], condition=r['condition'], rho=float(r['rho']), edge_slots=int(r['edge_slots']),
                             payload=float(r['payload']) if r['payload'] else None, method=r['method'],
                             seed=int(r['seed']), replicate=int(r['replicate']), objective=float(r['objective'])))
    return out, boot


def summarize(rows, conds, rho_c, replicate_axis):
    """Identical decision logic to executor/confirm.py summarize()."""
    O = {(r['exp'], r['condition'], round(r['rho'], 4), r['edge_slots'], r['payload'], r['method'], r['seed'], r['replicate']): r['objective'] for r in rows}
    units = sorted({(r['replicate'] if replicate_axis else r['seed']) for r in rows})

    def S(exp, cond, rho, slots, payload, method):
        vals = []
        for u in units:
            ss = BOOT_SEEDS if replicate_axis else [u]; rep = u if replicate_axis else -1
            vals.append(np.mean([O[(exp, cond, rho, slots, payload, method, s, rep)] for s in ss]))
        return np.array(vals)

    over = lambda exp, slots, method: np.mean([S(exp, c, rho_c[c], slots, None, method) for c in conds], axis=0)
    h1_9 = S('H1', 'primary', .9, 4, None, 'shortfall')-S('H1', 'primary', .9, 4, None, 'intrinsic')
    h1_35 = S('H1', 'primary', .35, 4, None, 'shortfall')-S('H1', 'primary', .35, 4, None, 'intrinsic')
    oracle = over('H2', 4, 'oracle_cq')-over('H2', 4, 'cq'); refine = over('H2', 4, 'cq')-over('H2', 4, 'utility')
    q1 = S('H3', 'primary', .65, 4, 1., 'mean_cq'); r1 = S('H3', 'primary', .65, 4, 1., 'mean_release')
    q12 = S('H3', 'primary', .65, 4, 12., 'mean_cq'); r12 = S('H3', 'primary', .65, 4, 12., 'mean_release')
    h4 = over('H4', 1, 'fpi')-over('H4', 1, 'cq')
    if replicate_axis:
        iv = lambda d: [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    else:
        def iv(d):
            m = float(np.mean(d)); h = float(student.ppf(.975, len(d)-1)*np.std(d, ddof=1)/np.sqrt(len(d))); return [m, m-h, m+h]
    out = dict(units=len(units), H1=dict(diff_rho09=iv(h1_9), diff_rho035=iv(h1_35), growth=iv(h1_9-h1_35)),
               H2=dict(oracle_minus_cq=iv(oracle), cq_minus_utility=iv(refine)),
               H3=dict(scale12=iv(r12-q12), scale1_relative=iv((r1-q1)/q1)), H4=dict(fpi_minus_cq_serial=iv(h4)))
    out['decisions'] = dict(
        H1=bool(out['H1']['diff_rho09'][2] < 0 and out['H1']['growth'][2] < 0),
        H2=bool(out['H2']['oracle_minus_cq'][1] < 0 and out['H2']['oracle_minus_cq'][2] < -abs(out['H2']['cq_minus_utility'][0])),
        H3=bool(out['H3']['scale12'][2] < 0 and -0.007 <= out['H3']['scale1_relative'][1] and out['H3']['scale1_relative'][2] <= 0.007),
        H4=bool(out['H4']['fpi_minus_cq_serial'][2] < 0))
    out['confirmed'] = int(sum(out['decisions'].values()))
    return out


def compare(a, b):
    if isinstance(a, dict):
        for k in a: compare(a[k], b[k])
    elif isinstance(a, list):
        for x, y in zip(a, b): compare(x, y)
    elif isinstance(a, float):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
    else:
        assert a == b


def main():
    config = json.loads((ROOT/'data/config.json').read_text(encoding='utf-8'))
    conds = list(config['conditions']); rho_c = {c: round(float(config['conditions'][c]['rho']), 4) for c in conds}
    report = dict(chain=chain())
    names = dict(h_mid='primary_t', h_low='quality_low_t', h_high='quality_high_t', h_exclude_exposed='exclude_known_exposed_t')
    for folder in ('results', 'results_frozen448'):
        published = json.loads((HERE/folder/'result.json').read_text(encoding='utf-8'))
        seeded, boot = rows_from(HERE/folder)
        for name, rows in seeded:
            compare(summarize(rows, conds, rho_c, False), published[names[name]])
        again = summarize(boot, conds, rho_c, True); compare(again, published['primary_bootstrap'])
        report[folder] = dict(decisions=again['decisions'], confirmed=again['confirmed'], replicates=again['units'])
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
