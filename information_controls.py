"""Verify, summarize, or rerun the information-matched controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from qbr import information_controls as controls


def assert_same(actual, expected, path='metrics'):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),path
        for key in expected:assert_same(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,(int,float)):
        assert abs(actual-expected)<1e-8,(path,actual,expected)
    else:
        assert actual==expected,(path,actual,expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','summary','replay'))
    parser.add_argument('--conditions',nargs='+',default=['primary'])
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--methods',nargs='+',choices=controls.METHODS,default=list(controls.METHODS))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path('outputs/information'))
    args=parser.parse_args();report=controls.verify()
    if args.action=='verify':print(json.dumps(report,indent=2));return
    args.output.mkdir(parents=True,exist_ok=True)
    if args.action=='summary':
        path=args.output/'summary.json';path.write_text(json.dumps(controls.summary(),indent=2)+'\n')
        print(path);return
    study=controls.Study()
    conditions=study.config['conditions'] if args.conditions==['all'] else args.conditions
    seeds=study.config['seeds'] if args.seeds is None else args.seeds
    assert all(c in study.config['conditions'] for c in conditions)
    assert all(s in study.config['seeds'] for s in seeds)
    jobs=[(c,s,m) for c in conditions for s in seeds for m in args.methods]
    expected={(r['condition'],r['seed'],r['method']):r for r in controls.load_records()}
    with (args.output/'records.jsonl').open('x',encoding='utf-8') as f:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=controls.initialize) as pool:
            for i,row in enumerate(pool.map(controls.episode,jobs),1):
                ref=expected[(row['condition'],row['seed'],row['method'])]
                assert row['choices_sha256']==ref['choices_sha256'],(row['condition'],row['seed'],row['method'])
                assert_same(row['metrics'],ref['metrics'])
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                if i%120==0:print(json.dumps({'verified':i,'total':len(jobs)}),flush=True)
    print(json.dumps({'verified':len(jobs),'total':len(jobs),'route_hashes_equal':True}),flush=True)


if __name__=='__main__':main()
