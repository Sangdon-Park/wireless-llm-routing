"""Synthetic bank for end-to-end testing: fresh task ids with OLD public responses (no sealed access)."""
import gzip, json
import numpy as np
from fresh_bank import FreshTraffic, HERE

if __name__ == '__main__':
    f = FreshTraffic(); old = f.old; rng = np.random.default_rng(7)
    by_service = {s: [r for r in old.rows if r['service'] == s] for s in ('dialogue', 'summary', 'code')}
    bank = []
    for k, t in enumerate(f.task_rows):
        src = by_service[t['service']][rng.integers(len(by_service[t['service']]))]
        mode = 'strict512' if t['service'] == 'dialogue' else 'full'; routes = {}
        for r, name in enumerate(('edge', 'cloud')):
            x = src['budgets'][mode][name]; chunks, payload, service_s, resets = old.profiles[src['id'], r]
            q = x['quality']; lo, hi = (q, q) if k % 50 else (0., 1.)   # every 50th task: unresolved bounds
            routes[name] = dict(quality=(lo+hi)/2, quality_low=lo, quality_high=hi, quality_status='synthetic',
                resolved=k % 50 != 0, service_s=service_s, payload_bytes=payload,
                all_attempt_payload_bytes=x['all_attempt_payload_bytes'], terminal_status=x['terminal_status'],
                response_sha256=x['response_sha256'], input_tokens=x['input_tokens'],
                cached_input_tokens=x['cached_input_tokens'], output_tokens=x['output_tokens'],
                thought_tokens=x['thought_tokens'], usage_complete=True,
                chunks=[[c['time_s'], c['utf8_bytes']] for c in chunks], resets=list(resets))
        bank.append(dict(id=t['id'], service=t['service'], routes=routes, replay_ready=k not in (5, 205, 405), unresolved=[]))
    (HERE/'private_synthetic').mkdir(exist_ok=True)
    with gzip.open(HERE/'private_synthetic/bank.json.gz', 'wt', encoding='utf-8') as fh:
        json.dump(dict(synthetic=True, tasks=bank), fh)
    print('synthetic tasks', len(bank))
