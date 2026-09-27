"""Quantile-forest Mean predictor on the 450 fresh prompts and per-request timing (validation env)."""
import json, sys, time
from pathlib import Path
from paths import private_workspace
import numpy as np
sys.path.insert(0, str(private_workspace() / 'research-workspace/code/wcl_redesign'))
import calibrated_followup_study as cf
from fresh_bank import FreshTraffic, HERE
from qbr.radio import SERVICES

if __name__ == '__main__':
    f = FreshTraffic(); prompts = [t['prompt'] for t in f.task_rows]
    state = cf.load(); models = state[0]
    # Mean quality exactly as qbr.information_value.mean_quality: alpha*forest mean + (1-alpha)*prior mean.
    mean = np.full((450, 2), np.nan); weights = {}
    for k, service in enumerate(SERVICES):
        idx = f.ids[k]; w = models[service].weights([prompts[i] for i in idx]); weights[service] = w
        budget = int(k == 0)
        prior = np.array([f.forecast.decision.quality[k, r, budget].mean() for r in (0, 1)])
        alpha = f.config['alpha'][service]
        mean[idx] = alpha*(w @ models[service].scores)+(1-alpha)*prior
    assert np.isfinite(mean).all()
    traces = dict(np.load(HERE/'predictions/traces.npz')); out = dict(mean_quality=mean)
    for s in SERVICES: out[s+'_weights'] = weights[s]; out[s+'_indices'] = np.array(f.ids[SERVICES.index(s)])
    for seed in f.config['seeds']:
        task, service, q = traces[f'{seed}_task'], traces[f'{seed}_service'], traces[f'{seed}_quality']
        for j in range(40): cf.risk('calibrated_quality', SERVICES[int(service[j])], prompts[int(task[j])], float(q[j]), *state)
        seconds = []
        for k, s, qq in zip(task, service, q):
            begin = time.perf_counter_ns(); cf.risk('calibrated_quality', SERVICES[int(s)], prompts[int(k)], float(qq), *state)
            seconds.append((time.perf_counter_ns()-begin)/1e9)
        out[f'{seed}_seconds'] = np.array(seconds)
        print(json.dumps(dict(seed=seed, mean_ms=1000*float(np.mean(seconds)))), flush=True)
    np.savez_compressed(HERE/'predictions/forest.npz', **out)
    json.dump(dict(outcomes_read=False, alpha=f.config['alpha']), open(HERE/'predictions/forest_receipt.json', 'x'), indent=1)
