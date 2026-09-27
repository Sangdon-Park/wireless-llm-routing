"""Portable numeric inputs for the measured-response replay."""
from pathlib import Path
from types import SimpleNamespace
import gzip
import json
import numpy as np
from .policy import ActionProfile, BudgetForecast
from .radio import SERVICES, Scenario, make_requests

DATA = Path(__file__).resolve().parents[1] / 'data'
ALIASES = {'service':'baseline', 'prompt':'raw_quality', 'calibrated':'calibrated_quality'}


class Study:
    def __init__(self, directory=DATA):
        directory = Path(directory)
        self.config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
        self.expected = json.loads((directory/'expected_results.json').read_text(encoding='utf-8'))
        with gzip.open(directory/'responses.json.gz', 'rt', encoding='utf-8') as f:
            self.rows = json.load(f)
        self.tasks = [{'id':r['id'], 'service':r['service']} for r in self.rows]
        self.profiles = {}
        for row in self.rows:
            mode = 'strict512' if row['service']=='dialogue' else 'full'
            for route, name in enumerate(('edge','cloud')):
                r = row['budgets'][mode][name]
                chunks = [{'time_s':t, 'utf8_bytes':n} for t,n in r['chunks']]
                self.profiles[row['id'],route] = (chunks,r['payload_bytes'],r['service_s'],r['resets'])
        with np.load(directory/'serving_state.npz', allow_pickle=False) as z:
            enabled = z['enabled']
            quality, durations = {}, {}
            for key in zip(*np.where(enabled)):
                suffix = '_'.join(map(str,key))
                quality[key] = z['quality_'+suffix]
                durations[key] = z['duration_'+suffix]
            profile = ActionProfile(z['mu'],z['second'],z['money'],enabled,quality)
            self.forecast = BudgetForecast(profile,durations,z['mean_bits'],())
            self.arrival_mu = z['arrival_mu']
        self.weights = self.load_npz(directory/'quality_weights.npz')
        self.times = self.load_npz(directory/'prediction_times.npz')
        self.external = self.load_npz(directory/'external_predictions.npz')
        self.positions = {}
        for service in SERVICES:
            for position,index in enumerate(self.weights[service+'_indices']):
                self.positions[int(index)] = position

    @staticmethod
    def load_npz(path):
        with np.load(path, allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    def requests(self, condition, seed):
        scenario = Scenario(**self.config['conditions'][condition])
        ids = {(s,scenario.split):[i for i,t in enumerate(self.tasks) if t['service']==name]
               for s,name in enumerate(SERVICES)}
        trace = make_requests(SimpleNamespace(mu=self.arrival_mu,ids=ids), scenario, seed)
        for key in ('task','service','quality'):
            np.testing.assert_array_equal(getattr(trace,key), self.times[f'{seed}_{key}'])
        return scenario, trace

    def prediction_seconds(self, method, seed):
        key = ALIASES[method] if method in ALIASES else 'external_'+method
        return self.times[f'{seed}_{key}_seconds']

    def risks(self, method, trace):
        """Evaluate expected shortfall at each sampled target using the frozen distributions."""
        values = np.full((len(trace.task),2,2), np.nan)
        for j,(index,s,q) in enumerate(zip(trace.task,trace.service,trace.quality)):
            service = SERVICES[int(s)]
            budget = int(s==0)
            mask = np.zeros((2,2),bool)
            mask[:,budget] = True
            prior = self.forecast.decision.risks(int(s),float(q),mask)
            if method == 'service':
                values[j] = prior
                continue
            weights = self.weights[service+'_weights'][self.positions[int(index)]]
            scores = self.weights[service+'_scores']
            risk = weights @ np.maximum(float(q)-scores,0.)
            alpha = 1. if method=='prompt' else self.config['alpha'][service]
            if alpha < 1.:
                risk = alpha*risk + (1-alpha)*prior[:,budget]
            values[j,:,budget] = risk
        return values

    def native_routes(self, method, trace):
        value = self.config['operating_points'][method]
        if method == 'carrot':
            score = (1-value)*self.external['evaluation_quality'] - value*self.external['evaluation_cost']
            choices = score.argmax(1)
        elif method == 'routellm':
            choices = (self.external['evaluation_routellm'] >= value).astype(int)
        else:
            raise ValueError(method)
        return choices[trace.task]

    def reference(self, condition, seed, method, scheduler='paper'):
        return next(r for r in self.expected if
                    (r['condition'],r['seed'],r['method'],r['scheduler']) == (condition,seed,method,scheduler))
