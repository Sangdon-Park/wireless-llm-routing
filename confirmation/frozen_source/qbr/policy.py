"""Queue-observed serving cost and the paper's fixed-budget routing rule."""
from dataclasses import dataclass
from collections import deque
from time import perf_counter
import heapq
import numpy as np

@dataclass(frozen=True)
class ActionProfile:
    # Shape(service, route, budget), budget0full /1strict512.
    mu: np.ndarray
    second: np.ndarray
    money: np.ndarray
    enabled: np.ndarray
    quality: dict

    def __post_init__(self):
        shape=self.mu.shape
        assert len(shape)==3 and shape[1:]==(2,2)
        assert self.second.shape==self.money.shape==self.enabled.shape==shape
        assert self.enabled.dtype==bool and np.all(self.enabled[:,:,0])
        assert np.all(np.isfinite(self.mu[self.enabled])) and np.all(self.mu[self.enabled]>=0)
        assert np.all(np.isfinite(self.second[self.enabled])) and np.all(self.second[self.enabled]>=0)
        assert np.all(np.isfinite(self.money[self.enabled])) and np.all(self.money[self.enabled]>=0)
        assert np.all(self.second[self.enabled]+1e-10>=self.mu[self.enabled]**2)
        for key in zip(*np.where(self.enabled)):
            q=np.asarray(self.quality[key],dtype=float)
            assert q.ndim==1 and len(q)>0 and np.all(np.isfinite(q)) and np.all((q>=0)&(q<=1))
            assert np.all(q[1:]>=q[:-1])

    def risks(self,service,target,required=None):
        assert 0<=target<=1
        required=self.enabled[service] if required is None else np.asarray(required,dtype=bool)
        assert required.shape==(2,2) and np.all(~required|self.enabled[service])
        values=np.full((2,2),np.nan)
        for route,budget in zip(*np.where(required)):
            q=np.asarray(self.quality[service,route,budget],dtype=float)
            # Direct finite expectation; prefix acceleration, if later used,
            # requires independent equality and runtime checks before adoption.
            values[route,budget]=float(np.maximum(target-q,0).mean())
        return values


def scores(profile,service,target,waits,*,risk=None,required=None,beta=12.,gamma=1000.,coefficient=.5):
    waits=np.asarray(waits,dtype=float)
    assert waits.shape==(2,) and np.all(np.isfinite(waits)) and np.all(waits>=0)
    assert beta>=0 and gamma>=0 and coefficient>=0
    mu=profile.mu[service];second=profile.second[service];money=profile.money[service]
    required=profile.enabled[service] if required is None else np.asarray(required,dtype=bool)
    assert required.shape==(2,2) and np.all(~required|profile.enabled[service])
    risk=profile.risks(service,target,required) if risk is None else np.asarray(risk,dtype=float)
    assert risk.shape==(2,2)
    rr=risk[required]
    assert np.all(np.isfinite(rr)) and np.all((rr>=0)&(rr<=target+1e-12))
    # Same addition order and terms as the strong two-route baseline.
    result=waits[:,None]+mu+gamma*money
    result+=coefficient*(waits[:,None]*mu+.5*second)
    result+=beta*risk
    return np.where(required,result,np.inf)


def best(values,allowed):
    choices=[(r,b) for r,b in zip(*np.where(allowed))]
    assert choices and all(np.isfinite(values[r,b]) for r,b in choices)
    # Preserve the old cloud-on-route-tie behavior; prefer full on budget ties.
    return min(choices,key=lambda a:(float(values[a]),-int(a[0]),int(a[1])))


def required_actions(profile,service,mode='joint',fixed=(0,0)):
    """Minimal quality-head inputs: fixed controls do not evaluate unused budgets."""
    if mode in ('joint','sequential'):
        return profile.enabled[service].copy()
    elif mode in ('full','fixed'):
        assert mode=='full' or fixed in ((0,0),(0,1),(1,0),(1,1))
        budgets=(0,0) if mode=='full' else fixed
        allowed=np.zeros((2,2),dtype=bool)
        for route,budget in enumerate(budgets):
            # Non-dialogue has no strict action. This is declared availability,
            # not missing-response imputation or task-specific fallback.
            allowed[route,budget if profile.enabled[service,route,budget] else 0]=True
    else:raise ValueError(mode)
    return allowed


def select(profile,service,target,waits,mode='joint',fixed=(0,0),*,risk=None):
    required=required_actions(profile,service,mode,fixed)
    risk=profile.risks(service,target,required) if risk is None else risk
    values=scores(profile,service,target,waits,risk=risk,required=required)
    allowed=required.copy()
    if mode=='sequential':
        idle=scores(profile,service,target,np.zeros(2),risk=risk,required=required)
        allowed=np.zeros((2,2),dtype=bool)
        for route in (0,1):
            candidates=[b for b in (0,1) if profile.enabled[service,route,b]]
            budget=min(candidates,key=lambda b:(float(idle[route,b]),b))
            allowed[route,budget]=True
    route,budget=best(values,allowed)
    logged=[[float(v) if np.isfinite(v) else None for v in row] for row in values]
    return dict(route=int(route),budget=int(budget),score=float(values[route,budget]),
                values=logged,allowed=allowed.tolist(),mode=mode)

@dataclass(frozen=True)
class BudgetForecast:
    decision: ActionProfile
    durations: dict
    mean_bits: np.ndarray
    training_ids: tuple

    def residual(self,service,route,budget,age):
        assert age>=0 and self.decision.enabled[service,route,budget]
        sample=self.durations[service,route,budget]
        tail=sample[sample>age]
        # Exactly the established empirical-survival fallback, now by known mode.
        return float(tail.mean()-age) if len(tail) else float(self.decision.mu[service,route,budget])

@dataclass
class KnownJob:
    service:int
    route:int
    budget:int
    user:int
    admitted:float
    start:float|None=None
    ended:bool=False
    generated:float=0.


class BudgetController:
    def __init__(self,forecast,scenario,choices,mode,fixed,risks=None):
        # No physical calibration rows, realized response, future slots or channel.
        self.forecast=forecast;self.sc=scenario;self.choices=choices
        self.mode=mode;self.fixed=fixed;self.risks=risks
        assert scenario.beta==12 and scenario.gamma==1000
        self.jobs={};self.waiting=[deque(),deque()];self.actions=[];self.decision_times=[]
        self.events=[];self.now=0.

    def channel(self,now,observed):
        self.now=float(now)

    def admit(self,j,service,route,user,ignored_slot,now):
        chosen_route,budget=self.choices[j]
        assert route==chosen_route and j not in self.jobs
        self.jobs[j]=KnownJob(service,route,budget,user,float(now));self.waiting[route].append(j)
        self.events.append(dict(event='admit',job=j,observed_at=self.now,at=float(now),
                                service=service,route=route,budget=budget,user=user))

    def start(self,j,at):
        if j not in self.jobs:return
        job=self.jobs[j];assert job.admitted<=at<=self.now+1e-9
        job.start=float(at)
        if j in self.waiting[job.route]:self.waiting[job.route].remove(j)
        self.events.append(dict(event='start',job=j,observed_at=self.now,at=float(at)))

    def release(self,j,bits):
        if j in self.jobs:self.jobs[j].generated+=float(bits)

    def end(self,j):
        if j not in self.jobs:return
        job=self.jobs[j];job.ended=True
        if j in self.waiting[job.route]:self.waiting[job.route].remove(j)
        self.events.append(dict(event='end',job=j,observed_at=self.now))

    def complete(self,j):
        if j in self.jobs:
            job=self.jobs.pop(j);assert job.ended and j not in self.waiting[job.route]
        # Completion can occur inside the current radio slot; retain callback
        # ordering rather than falsely timestamping it at the slot's start.
        self.events.append(dict(event='complete',job=j,callback_tick=self.now))

    def compute_state(self,now):
        waits=[];active_records=[];queued_records=[]
        for route,slots in enumerate((self.sc.edge_slots,self.sc.cloud_slots)):
            active=[(j,job) for j,job in self.jobs.items()
                    if job.route==route and job.start is not None and not job.ended]
            assert len(active)<=slots
            availability=[]
            for j,job in active:
                age=max(0.,now-job.start)
                remaining=self.forecast.residual(job.service,route,job.budget,age)
                availability.append(remaining)
                active_records.append(dict(job=j,route=route,budget=job.budget,service=job.service,
                    observed_start=job.start,age=age,estimated_remaining=remaining))
            availability += [0.]*(slots-len(active));heapq.heapify(availability)
            for j in self.waiting[route]:
                job=self.jobs[j];assert job.start is None and not job.ended
                mean=float(self.forecast.decision.mu[job.service,route,job.budget])
                begin=heapq.heappop(availability);end=begin+mean;heapq.heappush(availability,end)
                queued_records.append(dict(job=j,route=route,budget=job.budget,service=job.service,
                    estimated_service=mean,estimated_start=begin,estimated_end=end))
            waits.append(availability[0])
        return np.asarray(waits),active_records,queued_records

    def choose(self,now,j,service,target,user,observed,wire,throughput):
        began=perf_counter();waits,active,queued=self.compute_state(now)
        risk=None if self.risks is None else self.risks[j]
        chosen=select(self.forecast.decision,service,float(target),waits,self.mode,self.fixed,risk=risk)
        assert self.choices[j] is None
        self.choices[j]=(chosen['route'],chosen['budget'])
        route_values=np.array([min(v for b,v in enumerate(chosen['values'][r]) if chosen['allowed'][r][b]) for r in (0,1)])
        elapsed=perf_counter()-began;self.decision_times.append(elapsed)
        self.actions.append(dict(job=j,time=float(now),route=chosen['route'],budget=chosen['budget'],
            service=service,target=float(target),user=user,waits=waits.tolist(),
            values=route_values.tolist(),all_action_values=chosen['values'],allowed=chosen['allowed'],
            mode=self.mode,fixed_budgets=list(self.fixed),wall_s=elapsed,
            active_predictions=active,queued_predictions=queued,state_event_count=len(self.events)))
        return chosen['route'],[0,0],route_values


def front_ready(arrivals,seconds):
    arrivals=np.asarray(arrivals,dtype=float);seconds=np.asarray(seconds,dtype=float)
    assert arrivals.shape==seconds.shape and np.all(np.isfinite(seconds)) and np.all(seconds>=0)
    ready=np.empty(len(arrivals));free=0.
    for j,(at,service) in enumerate(zip(arrivals,seconds)):
        free=max(float(at),free)+float(service);ready[j]=free
    return ready

class NativeController(BudgetController):
    def __init__(self,*a,routes,**k):
        super().__init__(*a,**k);self.native_routes=routes
    def choose(self,now,j,service,target,user,observed,wire,throughput):
        route=int(self.native_routes[j]);budget=1 if service==0 else 0
        self.choices[j]=(route,budget);self.decision_times.append(0.)
        self.actions.append(dict(job=j,time=float(now),route=route,budget=budget,service=service,target=float(target),user=user))
        return route,[0,0],np.array([0.,0.])
