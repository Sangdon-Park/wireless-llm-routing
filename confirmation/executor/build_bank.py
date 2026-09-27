"""Frozen fresh450 grading and replay-bank assembly (runs once, after the exposure receipt).

Every sealed read goes through read_sealed(), which appends to the access log. All 450 tasks and
900 response slots are kept; failures stay as unresolved cells with bounds [0, 1] (never zero).

Grade contract (GRADE_CONTRACT below, hashed into the executor commitment):
- dialogue: native multi-turn IFBench strict fraction of instructions (dialogue_adapter.grade_dialogue,
  pinned IFBench runtime), answer established unless the capture is transport_failed;
- summary: two fixed judges, .50/.35/.15 over 0..4, averaged; missing/invalid judges give bounds;
  known-empty answer scores 0 (summary_adapter.parse_judge / aggregate_summary);
- code: restricted WASI grader (sealed_grader_preflight_20260924/adapter.grade): pass=1, test_failure=0,
  invalid_program=0, every other status unresolved; no normalizer second evaluation;
- replay quality = midpoint of bounds (primary), bounds kept for sensitivity; unresolved usage keeps known
  token counts (known-fee lower bound).
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from paths import private_workspace
import gzip
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
G = private_workspace() / 'research-workspace/wcl_goal_20260923'
ACQ = G/'fresh_inventory/acquisition_v2'
RUN = ACQ/'run'
SEALED = RUN/'sealed'
ASSEMBLY = G/'confirmation_assembly_20260924'
S_GRADER = G/'sealed_grader_preflight_20260924'
PRIVATE = HERE/'private'
LOG = HERE/'exposure_access_log.jsonl'
SEMANTIC_PY = str(private_workspace() / 'semantic-runtime-v1/Scripts/python.exe')
sys.path[:0] = [str(ASSEMBLY), str(S_GRADER), str(HERE)]
import assembly  # noqa: E402
import summary_adapter  # noqa: E402
from fresh_bank import tasks  # noqa: E402

GRADE_CONTRACT = dict(schema='fresh450_grade_contract_v1', dialogue='IFBench strict fraction via dialogue_adapter',
    summary='two judges .50/.35/.15 over 0..4 averaged; empty answer 0; missing judge -> bounds',
    code='restricted WASI S adapter: pass 1, test_failure 0, invalid_program 0, others unresolved; no normalizer',
    established_answer='terminal_status != transport_failed', replay_quality='midpoint of bounds (primary)',
    usage='known token totals when unresolved')
GRADE_CONTRACT_SHA = hashlib.sha256(assembly.canonical(GRADE_CONTRACT)).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_sealed(relative, purpose):
    path = SEALED/relative
    data = path.read_bytes()
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(at=stamp(), path=str(Path(relative)).replace('\\', '/'),
                                sha256=hashlib.sha256(data).hexdigest(), purpose=purpose))+'\n')
    return data


def meta():
    design_sha = hashlib.sha256((RUN/'design.json').read_bytes()).hexdigest()
    epochs = {json.loads(p.read_text(encoding='utf-8-sig'))['epoch_sha256'] for p in (RUN/'epochs').glob('*.json')}
    epochs.add(json.loads((RUN/'epoch_baseline.json').read_text(encoding='utf-8-sig'))['epoch_sha256'])
    protocol = json.loads((ACQ/'acquisition_protocol.json').read_text(encoding='utf-8-sig'))
    config = json.loads((G/'fresh_inventory/reservation_v1/configuration.json').read_text(encoding='utf-8'))
    return design_sha, epochs, protocol['allowed_returned_models'], config['models']


def bundle(phase, route, identity, purpose):
    rel = Path(phase)/route/(identity+'.json')
    raw = read_sealed(rel, purpose)
    receipt = read_sealed(Path('record_receipts')/rel, purpose)
    intent = read_sealed(Path('intents')/rel, purpose)
    artifacts = {name: read_sealed(Path(name.replace('\\', '/')), purpose)
                 for name in json.loads(receipt.decode('utf-8-sig'))['attempt_artifacts']}
    return raw, receipt, intent, artifacts


def response_cell(task, route, design_sha, epochs, allowed, models):
    public = assembly.public_projection(task)
    try:
        raw_b, rec_b, int_b, art = bundle('responses', route, task['id'], 'fresh450_confirmation_response')
        epoch = json.loads(raw_b.decode('utf-8-sig'))['epoch_sha256']
        assembly.require(epoch in epochs, 'unknown_epoch')
        raw, evidence = assembly.verify_bundle(raw_b, rec_b, int_b, art,
            contract=assembly.expected_call(public, route, models[route]), design_sha256=design_sha,
            epoch_sha256=epoch, allowed_models=allowed[route])
        return dict(ok=True, raw=raw, evidence=evidence)
    except Exception as exc:  # retained as an unresolved cell, never dropped
        return dict(ok=False, error=f'{type(exc).__name__}:{exc}')


def judge_rating(task, answer, evidence, judge_route, design_sha, epochs, allowed, models):
    try:
        identity = task['id']+'_'+answer['route']
        raw_b, rec_b, int_b, art = bundle('judge_captures', judge_route, identity, 'fresh450_confirmation_judge')
        receipt = json.loads(rec_b.decode('utf-8-sig'))
        assembly.require(receipt['intent_sha256'] == assembly.sha(int_b), 'judge_intent_hash')
        assembly.require(all(assembly.sha(b) == receipt['attempt_artifacts'][n] for n, b in art.items()), 'judge_artifact_hash')
        epoch = json.loads(raw_b.decode('utf-8-sig'))['epoch_sha256']
        assembly.require(epoch in epochs, 'unknown_epoch')
        expected = summary_adapter.expected_contract(answer, evidence, article=task['article'],
            reference=task['reference'], judge_route=judge_route, judge_model=models[judge_route])
        return summary_adapter.parse_judge(raw_b, expected=expected, expected_record_sha256=receipt['record_sha256'],
            design_sha256=design_sha, epoch_sha256=epoch, allowed_models=allowed[judge_route])
    except Exception:
        return None


def established(raw):
    return raw['capture']['terminal_status'] != 'transport_failed'


def code_grade(task, raw):
    import adapter
    if not established(raw):
        return dict(resolved=False, score=None, bounds=[0., 1.], status='unresolved_generation')
    r = adapter.grade(dict(text=raw['capture']['text'], entry_point=task['entry_point'],
                           function_prompt=task['function_prompt'], tests_source=task['tests_source']))['result']
    status = r.get('status')
    if status in ('pass', 'test_failure', 'invalid_program'):
        score = 1. if status == 'pass' else 0.
        return dict(resolved=True, score=score, bounds=[score, score], status=status)
    return dict(resolved=False, score=None, bounds=[0., 1.], status=str(status), cause=r.get('cause'))


def dialogue_grades(items):
    """Run the pinned IFBench runtime in the semantic env; items are (task, text, established)."""
    PRIVATE.mkdir(exist_ok=True)
    src = PRIVATE/'dialogue_inputs.json'; dst = PRIVATE/'dialogue_grades.json'
    src.write_text(json.dumps([dict(i=k, task=t, text=x, established=e) for k, (t, x, e) in enumerate(items)],
                              ensure_ascii=False), encoding='utf-8')
    subprocess.run([SEMANTIC_PY, str(HERE/'grade_dialogue.py'), str(src), str(dst)], check=True)
    out = json.loads(dst.read_text(encoding='utf-8'))
    src.unlink()
    return out


def as_grade(g, task, route, raw, evidence, service):
    if g['resolved']:
        lo = hi = float(g['score'])
    else:
        lo, hi = map(float, g['bounds'])
    return dict(kind='hidden_quality', task_id=task['id'], route=route, service=service,
                response_sha256=raw['response_sha256'], raw_sha256=evidence['raw_sha256'],
                grade_contract_sha256=GRADE_CONTRACT_SHA, resolved=bool(g['resolved']),
                score=float(g['score']) if g['resolved'] else None, bounds=[lo, hi], status=g['status'])


def main():
    assert not (PRIVATE/'bank.json.gz').exists(), 'bank already built'
    receipt = json.loads((HERE/'exposure_receipt.json').read_text(encoding='utf-8'))
    assert receipt['authorized'] is True
    design_sha, epochs, allowed, models = meta()
    rows = tasks()
    cells = {}
    with ThreadPoolExecutor(8) as pool:
        futures = {(t['id'], r): pool.submit(response_cell, t, r, design_sha, epochs, allowed, models)
                   for t in rows for r in ('edge', 'cloud')}
        cells = {k: f.result() for k, f in futures.items()}
    grades = {}
    dialogue = [(t, r) for t in rows if t['service'] == 'dialogue' for r in ('edge', 'cloud') if cells[t['id'], r]['ok']]
    dg = dialogue_grades([(t, cells[t['id'], r]['raw']['capture']['text'], established(cells[t['id'], r]['raw']))
                          for t, r in dialogue])
    for (t, r), g in zip(dialogue, dg):
        s = g['score']
        grades[t['id'], r] = (dict(resolved=True, score=s, bounds=[s, s], status='scored') if g['status'] == 'scored'
                              else dict(resolved=False, score=None, bounds=[0., 1.], status=g['status']))
    code = [(t, r) for t in rows if t['service'] == 'code' for r in ('edge', 'cloud') if cells[t['id'], r]['ok']]
    with ThreadPoolExecutor(8) as pool:
        for (t, r), g in zip(code, pool.map(lambda p: code_grade(p[0], cells[p[0]['id'], p[1]]['raw']), code)):
            grades[t['id'], r] = g
    for t in rows:
        if t['service'] != 'summary': continue
        for r in ('edge', 'cloud'):
            c = cells[t['id'], r]
            if not c['ok']: continue
            judges = {j: judge_rating(t, c['raw'], c['evidence'], j, design_sha, epochs, allowed, models) for j in ('edge', 'cloud')}
            a = summary_adapter.aggregate_summary(c['raw'], c['evidence'], judges, grade_contract_sha256=GRADE_CONTRACT_SHA)
            grades[t['id'], r] = dict(resolved=a['resolved'], score=a['score'], bounds=a['bounds'], status=a['status'],
                                      valid_judges=a['valid_judge_scores'])
    bank = []
    for t in rows:
        public = assembly.public_projection(t); route_cells = {}; summary = {}
        for r in ('edge', 'cloud'):
            c = cells[t['id'], r]
            if not c['ok']:
                route_cells[r] = dict(replay_ready=False, error=c['error']); summary[r] = dict(error=c['error']); continue
            g = as_grade(grades[t['id'], r], t, r, c['raw'], c['evidence'], t['service'])
            cell = assembly.replay_response(c['raw'], c['evidence'], assembly.quality_sidecar(
                g, identity=t['id'], route=r, service=t['service'], response_sha256=c['raw']['response_sha256'],
                raw_sha256=c['evidence']['raw_sha256'], grade_contract_sha256=GRADE_CONTRACT_SHA),
                service=t['service'], traffic_protocol='pinned_raw_streaming')
            route_cells[r] = cell
            resp = cell['response']; lo, hi = resp['quality_bounds']
            summary[r] = dict(quality=(lo+hi)/2, quality_low=lo, quality_high=hi, quality_status=resp['quality_status'],
                resolved=grades[t['id'], r]['resolved'], service_s=resp['service_s'], payload_bytes=resp['payload_bytes'],
                all_attempt_payload_bytes=resp['all_attempt_payload_bytes'], terminal_status=resp['terminal_status'],
                response_sha256=resp['response_sha256'], input_tokens=resp['input_tokens'],
                cached_input_tokens=resp['cached_input_tokens'], output_tokens=resp['output_tokens'],
                thought_tokens=resp['thought_tokens'], usage_complete=resp['usage_complete'],
                chunks=[[c['time_s'], c['utf8_bytes']] for c in cell['profile']['chunks']],
                resets=cell['profile']['reset_times_s'])
        paired = assembly.paired_task(public, route_cells)
        bank.append(dict(id=t['id'], service=t['service'], routes=summary, replay_ready=paired['replay_ready'],
                         unresolved=paired['unresolved']))
    PRIVATE.mkdir(exist_ok=True)
    with gzip.open(PRIVATE/'bank.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(dict(grade_contract=GRADE_CONTRACT, grade_contract_sha256=GRADE_CONTRACT_SHA, tasks=bank), f)
    print(json.dumps(dict(tasks=len(bank), cells_ok=sum(c['ok'] for c in cells.values()),
                          resolved=sum(bool(g['resolved']) for g in grades.values()))))


if __name__ == '__main__':
    main()
