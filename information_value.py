"""Verify, summarize, rerun, or extend the value-of-information study (v1.4-v1.8)."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from qbr import information_value as study


def rerun(args):
    rows, boot = study.load_records()
    rho = None if args.rho is None else float(args.rho)
    slots = None if args.edge_slots is None else int(args.edge_slots)
    result = study.episode(args.exp, args.condition, rho, slots, args.seed, args.method, args.replicate, args.payload)
    review = ('static12', 'static_load', 'noisy', 'serial_beta', 'serial_payload', 'serial_load')
    pool = (study.load_extension() if args.exp in ('payload', 'serial12') else study.load_review() if args.exp in review
            else boot if args.replicate >= 0 else rows)
    match = [r for r in pool
             if (r['exp'], r['condition'], r['seed'], r['method'], r['replicate']) ==
             (args.exp, args.condition, args.seed, args.method, args.replicate)
             and abs(r['rho']-result['rho']) < 1e-12 and r['edge_slots'] == result['edge_slots']
             and r.get('payload_scale') == result.get('payload_scale')]
    if match:
        ref = match[0]
        assert ref['choices_sha256'] == result['choices_sha256'], 'Routing choices differ'
        for k in study.METRICS:
            np.testing.assert_allclose(result['metrics'][k], ref['metrics'][k], rtol=0, atol=1e-8, err_msg=k)
        result['matches_published_record'] = True
    else:
        result['matches_published_record'] = None
    return result


def replay_all(args):
    jobs, boot = study.plan()
    jobs = {'main': jobs, 'boot': boot, 'extension': study.plan_extension(), 'review': study.plan_review()}[args.phase]
    if args.limit: jobs = jobs[:args.limit]
    out = args.output; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f, ProcessPoolExecutor(args.workers, initializer=study.initialize) as pool:
        for fu in as_completed([pool.submit(study.episode, *j) for j in jobs]):
            f.write(json.dumps(fu.result(), allow_nan=False)+'\n')
    return dict(paths=len(jobs), output=str(out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_mutually_exclusive_group(required=True)
    a.add_argument('--verify', action='store_true', help='Check hashes and recompute every reported summary')
    a.add_argument('--summary', action='store_true')
    a.add_argument('--gate', action='store_true', help='Report the declared load-derived-coefficient gate')
    a.add_argument('--extension-summary', action='store_true', help='Payload sweep and serial-edge results (v1.5)')
    a.add_argument('--diagnostic-summary', action='store_true', help='Paired radio decision diagnostic (v1.6)')
    a.add_argument('--review-summary', action='store_true', help='Static split, noisy oracle, one-edge-server sweeps (v1.8)')
    a.add_argument('--airtime', action='store_true', help='Recompute data/airtime_load.json values')
    a.add_argument('--rerun', action='store_true', help='Rerun one path and compare with the published record')
    a.add_argument('--replay', choices=('main', 'boot', 'extension', 'review'), dest='phase', help='Rerun a whole phase to a new JSONL file')
    p.add_argument('--exp', default='main', choices=('main', 'beta', 'kappa', 'load', 'serial', 'boot', 'payload', 'serial12',
                                                'static12', 'static_load', 'noisy', 'serial_beta', 'serial_payload', 'serial_load'))
    p.add_argument('--payload', type=float)
    p.add_argument('--condition', default='primary')
    p.add_argument('--rho'); p.add_argument('--edge-slots')
    p.add_argument('--seed', type=int, default=23924000)
    p.add_argument('--method', default='cq', help="e.g. cq, fpi, utility@b8, oracle_cq, k1, static_prop, static_p0.75, noisy0.3")
    p.add_argument('--replicate', type=int, default=-1)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--limit', type=int)
    p.add_argument('--output', type=Path, help='New file; existing files are not overwritten')
    args = p.parse_args()
    if args.verify: result = study.verify()
    elif args.summary: result = study.summary()
    elif args.gate: result = study.fpi_gate(study.summary())
    elif args.extension_summary: result = study.extension_summary()
    elif args.diagnostic_summary: result = study.diagnostic_summary()
    elif args.review_summary: result = study.review_summary()
    elif args.airtime: result = study.airtime_load()
    elif args.rerun: result = rerun(args)
    else:
        if not args.output: p.error('--replay needs --output')
        result = replay_all(args); args.output = None
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f: f.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
