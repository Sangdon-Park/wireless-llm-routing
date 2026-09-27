"""Verify or reproduce the 480 Bao decision-rule transfer trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from qbr import bao


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','summary','replay'))
    parser.add_argument('--workers',type=int,choices=range(1,9),default=4)
    parser.add_argument('--smoke',action='store_true',help='Replay only both primary arms on the first archived seed')
    parser.add_argument('--output',type=Path,default=Path('outputs/bao'))
    args=parser.parse_args()
    report=bao.verify()
    if args.action=='verify':
        print(json.dumps(report,indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action=='summary':
        with (args.output/'summary.json').open('x',encoding='utf-8') as stream:
            json.dump(bao.summary(),stream,indent=2,allow_nan=False)
        print(args.output/'summary.json');return
    plan=bao.plan()
    jobs=[(c,s,m) for c in plan['conditions'] for s in plan['seeds'] for m in bao.METHODS]
    assert len(jobs)==480
    if args.smoke: jobs=[('primary',plan['seeds'][0],m) for m in bao.METHODS]
    expected={(r['condition'],r['seed'],r['method']):r for r in bao.records()}
    # Exclusive creation preserves existing outputs, including partial replays.
    with (args.output/'records.jsonl').open('x',encoding='utf-8') as stream:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=bao.initialize) as pool:
            for number,row in enumerate(pool.map(bao.episode,jobs),1):
                reference=expected[row['condition'],row['seed'],row['method']]
                bao.assert_same(row,reference)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if number%20==0:print(json.dumps(dict(verified=number,total=len(jobs))),flush=True)
    print(json.dumps(dict(verified=len(jobs),route_hashes_equal=True,metric_tolerance=1e-8)))


if __name__=='__main__':main()
