"""Fresh450 task roster, traffic traces, and a Study-like object for the frozen qbr replay.

Reads only the outcome-free reservation (task inputs) and the frozen calibration serving state.
Replay rows (responses and grades) are attached by build_bank.py after authorization.
"""
from pathlib import Path
from paths import REPOSITORY_ROOT, private_workspace
from types import SimpleNamespace
import hashlib
import json
import sys
import numpy as np

REPO = REPOSITORY_ROOT
sys.path.insert(0, str(REPO))
from qbr.data import Study  # noqa: E402
from qbr.radio import Scenario, make_requests, SERVICES  # noqa: E402

HERE = Path(__file__).resolve().parent
G = private_workspace() / 'research-workspace/wcl_goal_20260923'
CONFIGURATION = G/'fresh_inventory/reservation_v1/configuration.json'
CONFIGURATION_SHA = '907671cd20993dfa6272ad490a00556f692848aacf58ef7a7bd4962f7dcec1b6'
BOOT_SEEDS = (23924000, 23924001)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tasks():
    """All 450 reserved tasks in reservation order (the fixed task index)."""
    assert sha(CONFIGURATION) == CONFIGURATION_SHA
    rows = json.loads(CONFIGURATION.read_text(encoding='utf-8'))['tasks']
    assert len(rows) == 450 and [sum(r['service'] == s for r in rows) for s in SERVICES] == [150, 150, 150]
    return rows


def public_prompt(task):
    """Generation-visible prompt (serialized native chat for dialogue), as sent to both models."""
    return task['prompt']


class FreshTraffic:
    """Old calibration arrival normalisation and replay conditions; fresh task identities."""
    def __init__(self):
        self.old = Study()
        self.config = self.old.config
        self.arrival_mu = self.old.arrival_mu
        self.forecast = self.old.forecast
        self.task_rows = tasks()
        self.tasks = [{'id': t['id'], 'service': t['service']} for t in self.task_rows]
        self.ids = {s: [i for i, t in enumerate(self.tasks) if t['service'] == n] for s, n in enumerate(SERVICES)}

    def scenario(self, condition, rho=None, edge_slots=None, payload=None):
        d = dict(self.config['conditions'][condition])
        if rho is not None: d['rho'] = rho
        if edge_slots is not None: d['edge_slots'] = edge_slots
        if payload is not None: d['payload_scale'] = payload
        return Scenario(**d)

    def trace(self, sc, seed):
        return make_requests(SimpleNamespace(mu=self.arrival_mu, ids={(k, sc.split): v for k, v in self.ids.items()}),
                             sc, seed)


def task_mapping(replicate, master_seed=2026092501):
    """Whole-task resampling within service strata (150 per service), fixed seed sequence."""
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, int(replicate)]))
    mapping = np.arange(450)
    rows = tasks()
    for service in SERVICES:
        ids = np.array([i for i, t in enumerate(rows) if t['service'] == service])
        mapping[ids] = rng.choice(ids, size=len(ids), replace=True)
    return mapping
