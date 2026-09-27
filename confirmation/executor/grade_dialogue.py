"""Pinned IFBench multi-turn grading (semantic runtime, which has the IFBench dependencies)."""
import json
import sys
from paths import private_workspace

RT = str(private_workspace() / 'wcl_redesign_2026-09-20/ifbench_runtime_preflight_v1')
sys.path[:0] = [RT+'/runtime', RT+'/dependencies',
                str(private_workspace() / 'research-workspace/wcl_goal_20260923/confirmation_assembly_20260924')]
import nltk  # noqa: E402
nltk.data.path[:] = [RT+'/nltk_data']
import evaluation_lib as ev  # noqa: E402
import dialogue_adapter as da  # noqa: E402

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    items = json.load(open(src, encoding='utf-8'))
    out = []
    for item in items:
        r = da.grade_dialogue(item['task'], item['text'], ev, established_answer=item['established'])
        out.append(dict(i=item['i'], status=r['status'], score=r.get('score')))
    json.dump(out, open(dst, 'x', encoding='utf-8'))
