"""Matched mean/distribution controls and causal radio-workload controls."""
from time import perf_counter
import numpy as np
from .policy import BudgetController, scores, best, required_actions
from .radio import SERVICES, Requests


def mean_risks(study, trace):
    """Shortfall of the mean of exactly the calibrated quality distribution."""
    result = np.full((len(trace.task),2,2),np.nan)
    for s,service in enumerate(SERVICES):
        take = np.flatnonzero(trace.service==s)
        budget = int(s==0)
        indices = [study.positions[int(i)] for i in trace.task[take]]
        weights = study.weights[service+'_weights'][indices]
        prediction = weights @ study.weights[service+'_scores']
        prior = np.array([study.forecast.decision.quality[s,r,budget].mean() for r in (0,1)])
        alpha = study.config['alpha'][service]
        expectation = alpha*prediction+(1-alpha)*prior
        result[take,:,budget] = np.maximum(trace.quality[take,None]-expectation,0.)
    return result


def resampled_trace(trace, mapping):
    """Resample complete paired tasks; retain the traffic and target draws."""
    return Requests(trace.arrival.copy(),trace.service.copy(),mapping[trace.task],
                    trace.user.copy(),trace.quality.copy())


def task_mapping(study, replicate, master_seed=2026092301):
    rng = np.random.default_rng(np.random.SeedSequence([master_seed,int(replicate)]))
    mapping = np.arange(len(study.tasks))
    for service in SERVICES:
        ids = np.array([i for i,t in enumerate(study.tasks) if t['service']==service])
        assert len(ids)==75
        mapping[ids] = rng.choice(ids,size=len(ids),replace=True)
    return mapping


def radio_correction(backlog, capacity, payload, serving, user):
    """Observed-band drain surrogate minus overlap with predicted RPC service.

    Sum_u B_u/C_u is the airtime needed to drain all currently visible bytes
    with frozen observed capacities. Add this request's calibration mean payload.
    A fluid overlap approximation is max(serving, drain), so only its excess
    above serving is added to the unchanged serving-cost score. No coefficient
    is fitted and no future arrivals, chunks, capacities, or outcomes are used.
    """
    backlog,capacity,payload,serving = map(np.asarray,(backlog,capacity,payload,serving))
    assert backlog.shape==capacity.shape and np.all(backlog>=0) and np.all(capacity>0)
    assert payload.shape==serving.shape==(2,2)
    drain = float(np.sum(backlog/capacity))+payload/float(capacity[user])
    return np.maximum(drain-serving,0.)


class RadioController(BudgetController):
    def radio_state(self,now,state):
        # This hook contains only prefix observations released by the executor.
        self.radio_observed_at = float(now)
        self.radio_backlog = np.bincount(state['users'],
            weights=np.maximum(state['visible']-state['delivered'],0.),minlength=self.sc.users)

    def choose(self,now,j,service,target,user,observed,wire,throughput):
        started = perf_counter()
        assert self.radio_observed_at==now
        waits,active,queued = self.compute_state(now)
        profile = self.forecast.decision
        allowed = required_actions(profile,service,self.mode,self.fixed)
        values = scores(profile,service,float(target),waits,risk=self.risks[j],required=allowed)
        serving = waits[:,None]+profile.mu[service]
        correction = radio_correction(self.radio_backlog,observed,
            self.forecast.mean_bits[service]*self.sc.payload_scale,serving,user)
        values = np.where(allowed,values+correction,np.inf)
        route,budget = best(values,allowed)
        self.choices[j] = (int(route),int(budget))
        self.decision_times.append(perf_counter()-started)
        route_values = values.min(1)
        self.actions.append(dict(job=j,time=float(now),route=int(route),budget=int(budget),
            service=service,target=float(target),user=user,waits=waits.tolist(),
            values=route_values.tolist(),radio_excess_s=correction[:,budget].tolist(),
            observed_capacity=np.asarray(observed).tolist(),backlog_bits=self.radio_backlog.tolist()))
        return int(route),[0,0],route_values
