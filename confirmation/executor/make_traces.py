"""Save the 20 fixed traffic task sequences (task/service/target are independent of rho, slots, payload)."""
import json
import numpy as np
from fresh_bank import FreshTraffic, HERE

if __name__ == '__main__':
    f = FreshTraffic(); out = {}
    for seed in f.config['seeds']:
        tr = f.trace(f.scenario('primary'), seed)
        for rho, slots, payload in ((.9, None, None), (.35, 1, 12.)):
            other = f.trace(f.scenario('primary', rho, slots, payload), seed)
            for k in ('task', 'service', 'quality', 'user'):
                np.testing.assert_array_equal(getattr(tr, k), getattr(other, k))
        for k in ('task', 'service', 'quality'):
            out[f'{seed}_{k}'] = getattr(tr, k)
    np.savez_compressed(HERE/'predictions/traces.npz', **out)
    json.dump([t['prompt'] for t in f.task_rows], open(HERE/'predictions/prompts.json', 'x', encoding='utf-8'), ensure_ascii=False)
    print('traces', len(f.config['seeds']))
