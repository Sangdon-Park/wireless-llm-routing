"""Validate the code grader against reservation canonical solutions and failing test cases."""
import collections, json, sys
from paths import private_workspace
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(private_workspace() / 'research-workspace/wcl_goal_20260923/sealed_grader_preflight_20260924'))
import adapter
from fresh_bank import tasks, HERE

def run(case):
    t, kind = case
    if kind == 'canonical': text = '```python\n' + t['canonical_solution'] + '\n```'
    else: text = '```python\ndef ' + t['entry_point'] + '(*args, **kwargs):\n    return None\n```'
    r = adapter.grade(dict(text=text, entry_point=t['entry_point'], function_prompt=t['function_prompt'],
                           tests_source=t['tests_source']))['result']
    return dict(id=t['id'], kind=kind, status=r.get('status'), cause=r.get('cause'))

if __name__ == '__main__':
    code = [t for t in tasks() if t['service'] == 'code']
    with ThreadPoolExecutor(8) as pool:
        out = list(pool.map(run, [(t, k) for t in code for k in ('canonical', 'wrong')]))
    (HERE/'validation').mkdir(exist_ok=True)
    json.dump(out, open(HERE/'validation/code_grader.json', 'x'), indent=1)
    print(json.dumps({k: dict(collections.Counter(o['status'] for o in out if o['kind'] == k)) for k in ('canonical', 'wrong')}))
