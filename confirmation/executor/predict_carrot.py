"""CARROT-KNN-SBERT on the 450 fresh prompts and per-request GPU timing (routerdc runtime)."""
import json, sys, time
from pathlib import Path
from paths import private_workspace
import numpy as np
sys.path.insert(0, str(private_workspace() / 'research-workspace/code/wcl_redesign'))
import external_native_gpu_v1 as g

HERE = Path(__file__).resolve().parent
MODEL = str(private_workspace() / 'wcl_redesign_2026-09-20/external_native_comparison_v1/carrot_model.npz')

if __name__ == '__main__':
    prompts = json.load(open(HERE/'predictions/prompts.json', encoding='utf-8'))
    traces = dict(np.load(HERE/'predictions/traces.npz'))
    packed = dict(np.load(MODEL)); m = g.Encoders()
    for _ in range(20): m.encode('An inference warmup prompt.', 'carrot')
    quality = np.zeros((450, 2)); cost = np.zeros((450, 2))
    for k, p in enumerate(prompts):
        q, c = g.knn_predict(packed, m.encode(p, 'carrot')); quality[k] = q; cost[k] = c
    out = dict(quality=quality, cost=cost)
    seeds = sorted({int(k.split('_')[0]) for k in traces})
    for seed in seeds:
        seconds = []
        for k in traces[f'{seed}_task']:
            m.sync(); begin = time.perf_counter_ns()
            q, c = g.knn_predict(packed, m.encode(prompts[int(k)], 'carrot'))
            m.sync(); seconds.append((time.perf_counter_ns()-begin)/1e9)
            assert np.array_equal(q, quality[int(k)])
        out[f'{seed}_seconds'] = np.array(seconds)
        print(json.dumps(dict(seed=seed, mean_ms=1000*float(np.mean(seconds)))), flush=True)
    np.savez_compressed(HERE/'predictions/carrot.npz', **out)
    json.dump(dict(device=m.device, model_sha256=__import__('hashlib').sha256(open(MODEL, 'rb').read()).hexdigest(),
                   outcomes_read=False), open(HERE/'predictions/carrot_receipt.json', 'x'), indent=1)
