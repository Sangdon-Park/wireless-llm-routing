"""Run the fixed supplementary controls and paired task-bank bootstrap."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from qbr.data import DATA, Study
from qbr.policy import BudgetController
from qbr.strengthening import RadioController, mean_risks, task_mapping, resampled_trace
from qbr import replay

STUDY = None
PLAN = None


def initialize():
    global STUDY,PLAN
    STUDY = Study()
    PLAN = json.loads((DATA/'strengthening_plan.json').read_text())


def episode(job):
    condition,seed,method,replicate = job
    study = STUDY
    scenario,trace = study.requests(condition,seed)
    if replicate>=0:
        mapping = task_mapping(study,replicate,PLAN['bootstrap']['master_seed'])
        trace = resampled_trace(trace,mapping)
    risks = mean_risks(study,trace) if method in ('mean','radio_mean') else study.risks('prompt' if method=='prompt' else 'calibrated',trace)
    controller = RadioController if method.startswith('radio_') else BudgetController
    factory = lambda *a,**k:controller(*a,risks=risks,**k)
    timing = 'prompt' if method=='prompt' else 'calibrated'
    wrapped = replay.wrapper(study.tasks,scenario,seed,trace,study.prediction_seconds(timing,seed))
    result = replay.simulate(wrapped,study.rows,study.profiles,study.forecast,controller_factory=factory)
    metrics = replay.account(wrapped,result,study.rows)
    choice_hash = hashlib.sha256(np.asarray(result['choices'],dtype='<i8').tobytes()).hexdigest()
    if replicate<0 and method in ('prompt','calibrated'):
        reference = study.reference(condition,seed,method)
        assert choice_hash == reference['choices_sha256']
        for key in ('objective','delay','delay_p95','shortfall','quality','money','discarded_bits'):
            np.testing.assert_allclose(metrics[key],reference['metrics'][key],rtol=0,atol=1e-8)
    return dict(condition=condition,seed=seed,method=method,replicate=replicate,
        metrics=metrics,choices_sha256=choice_hash,
        cloud_fraction=float(np.mean(np.array(result['choices'])[scenario.warmup:,0])),
        controller_mean_ms=float(np.mean(result['controller_wall_s']))*1000,
        max_airtime_sum=result['max_airtime_sum'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('controls','bootstrap','check'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,default=Path('outputs/strengthening'))
    parser.add_argument('--start',type=int,default=0,help='First bootstrap replicate, for disjoint resumable batches')
    parser.add_argument('--stop',type=int,help='Exclusive bootstrap bound; defaults to the plan count')
    args=parser.parse_args()
    initialize()
    conditions=list(STUDY.config['conditions'])
    if args.command=='check':
        jobs=[('primary',23924000,m,-1) for m in ('prompt','calibrated','mean','radio_mean','radio_calibrated')]
    elif args.command=='controls':
        jobs=[(c,s,m,-1) for c in conditions for s in STUDY.config['seeds'] for m in PLAN['controls']['methods']]
    else:
        stop=PLAN['bootstrap']['replicates'] if args.stop is None else args.stop
        assert 0<=args.start<stop<=PLAN['bootstrap']['replicates']
        jobs=[(c,s,m,b) for b in range(args.start,stop) for c in conditions
              for s in PLAN['bootstrap']['traffic_seeds'] for m in PLAN['bootstrap']['methods']]
    assert args.workers>0
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/(args.command+f'_{args.start:03d}.jsonl')
    started=time.perf_counter()
    with path.open('x',encoding='utf-8') as f:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
            for n,row in enumerate(pool.map(episode,jobs,chunksize=1),1):
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                if n%30==0 or n==len(jobs):
                    print(json.dumps({'complete':n,'total':len(jobs),'seconds':round(time.perf_counter()-started,1),
                                      'replicate':row['replicate']}),flush=True)
    print(json.dumps({'passed_accounting':True,'runs':len(jobs),'output':str(path)}))

if __name__=='__main__':main()
