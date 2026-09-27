"""Summarize or rerun the fixed utility/shortfall queue comparison."""
import argparse
import json
from pathlib import Path
from qbr import utility_controls as controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify',action='store_true')
    action.add_argument('--summary',action='store_true')
    action.add_argument('--rerun',action='store_true')
    parser.add_argument('--condition',default='primary')
    parser.add_argument('--seed',type=int,default=23924000)
    parser.add_argument('--method',choices=controls.METHODS,default='utility_queue')
    parser.add_argument('--output',type=Path,help='Optional new JSON file; existing files are not overwritten')
    args = parser.parse_args()
    if args.verify:
        result = controls.verify()
    elif args.summary:
        controls.verify()
        result = controls.summary()
    else:
        controls.initialize()
        plan,_ = controls.load_records()
        if args.condition not in plan['conditions'] or args.seed not in plan['seeds']:
            parser.error('Use one published condition and seed from data/utility_plan.json')
        result = controls.episode(args.condition,args.seed,args.method)
    text = json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:
            f.write(text)
    print(text,end='')


if __name__=='__main__':
    main()
