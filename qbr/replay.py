"""Streaming RPC/downlink replay with charged prediction time and retry accounting."""
import heapq, math, time
import numpy as np
from .radio import Frame, shares, Channel, Scenario, DT, HALF, EPS
from .policy import BudgetController, front_ready
from . import radio

ROUTES = ('edge', 'cloud')
SERVICES = ('dialogue', 'summary', 'code')

def provider_fee(response,route):
    if route==0:return response['service_s']/3600
    return (.75*(response['input_tokens']-response['cached_input_tokens'])+
            .075*response['cached_input_tokens']+
            3.75*(response['output_tokens']+response['thought_tokens']))/1e6


def radio_plan(visible,accepted,terminal,first_tick,eos_tick,active,observed_capacity):
    """Allocate airtime using released output, observed completion events, and observed rates."""
    count=len(observed_capacity);tq=[[] for _ in range(count)];aq=[[] for _ in range(count)]
    tb=np.zeros(count);ab=np.zeros(count)
    for u in range(count):
        for j in active[u]:
            if visible[j]-accepted[j]>EPS:
                (tq if terminal[j] else aq)[u].append(j)
        tq[u].sort(key=lambda j:(visible[j]-accepted[j],eos_tick[j],j))
        aq[u].sort(key=lambda j:(first_tick[j],j))
        tb[u]=sum(visible[j]-accepted[j] for j in tq[u])
        ab[u]=sum(visible[j]-accepted[j] for j in aq[u])
    order=np.lexsort((np.arange(count),-observed_capacity))
    f=shares(tb,observed_capacity,order=order)
    g=shares(ab,observed_capacity,max(0.,1-f.sum()),order)
    left=max(0.,1-(f+g).sum());eligible=tb>EPS if np.any(tb>EPS) else ab>EPS
    for u in order:
        if eligible[u]:f[u]+=left;break
    assert (f+g).sum()<=1+1e-10
    return [(u,f[u]+g[u],Frame([(j,accepted[j],visible[j]) for j in tq[u]+aq[u]])) for u in range(count) if f[u]+g[u]>0]


def simulate(wrapper,rows,profiles,forecast,*,controller_factory=BudgetController):
    """Replay recorded responses, exposing only observed events to the routing policy."""
    old=wrapper['result'];sc=Scenario(**wrapper['scenario']);count=len(old['requests']);assert sc.dt==DT
    choices=[None]*count;controller=controller_factory(forecast,sc,choices,'fixed',(1,1))
    table={r['id']:r for r in rows};arrivals=np.array([q['arrival'] for q in old['requests']])
    ready=np.array(old['router_accounting']['ready']);head=np.array(old['router_accounting']['seconds'])
    service=np.array([a['service'] for a in old['controller_actions']],int)
    target=np.array([a['target'] for a in old['controller_actions']])
    users=np.array([a['user'] for a in old['controller_actions']],int)
    channel=Channel(sc,wrapper['seed']);free=[np.zeros(sc.edge_slots),np.zeros(sc.cloud_slots)]
    events=[];schedule=[];lookups=[];next_request=0
    visible=np.zeros(count);delivered=np.zeros(count);terminal=np.zeros(count,bool)
    completion=np.full(count,np.nan);last=np.full(count,np.nan)
    first_tick=np.full(count,np.iinfo(np.int64).max,np.int64)
    # Policy may read eos_tick only for publicly terminal requests.
    eos_tick=np.full(count,-1,int);active=[set() for _ in range(sc.users)]
    shortfall=np.zeros(count);charge=np.zeros(count);selected_quality=np.zeros(count)
    generated=np.zeros(count);discarded=np.zeros(count);resets=[]
    ledger=dict(unique=0.,duplicate=0.,padding=0.,wire=0.,reserved_airtime_s=0.,payload_unit='bits')
    horizon=math.ceil(old['simulation_seconds']/DT)+20000;max_fraction=0.;started=time.perf_counter()
    def terminal_finish(j,at):
        if terminal[j] and visible[j]-delivered[j]<=EPS and not np.isfinite(completion[j]):
            completion[j]=at;controller.complete(j)
    for tick in range(horizon):
        now=tick*DT;obs=channel.sample(max(0,tick*2-round(sc.csi_delay_s/HALF)))
        controller.channel(now,obs)
        while events and events[0][0]<=tick:
            _,kind,j,value=heapq.heappop(events)
            if kind==0:
                _,reset,bits=value
                if reset:
                    skipped=max(0.,visible[j]-delivered[j]);discarded[j]+=skipped
                    resets.append(dict(job=j,observed_at=now,discarded_bits=skipped,
                        generated_before_reset=visible[j],logical_delivered_before_reset=delivered[j]))
                    delivered[j]=visible[j];first_tick[j]=np.iinfo(np.int64).max
                else:
                    controller.release(j,bits);visible[j]+=bits;generated[j]+=bits
                    first_tick[j]=min(first_tick[j],tick);active[users[j]].add(j)
            elif kind==1:
                controller.end(j);terminal[j]=True;eos_tick[j]=tick;terminal_finish(j,now)
            elif kind==2:controller.start(j,value)
            else:raise AssertionError(kind)
        while next_request<count and ready[next_request]<=now+1e-10:
            j=next_request;s=int(service[j]);u=int(users[j])
            # Pass copies of observed radio state to controllers with a radio_state hook.
            if hasattr(controller,'radio_state'):
                controller.radio_state(now,dict(visible=visible[:j].copy(),delivered=delivered[:j].copy(),terminal=terminal[:j].copy(),
                    users=users[:j].copy(),first_tick=first_tick[:j].copy(),eos_tick=eos_tick[:j].copy()))
            route,ignored,values=controller.choose(now,j,s,float(target[j]),u,obs,None,None)
            assert choices[j] is not None and choices[j][0]==route
            budget=choices[j][1];expected_budget=1 if s==0 else 0;assert budget==expected_budget
            # Read the selected response only after the controller chooses a route.
            identity=old['requests'][j]['task_id'];row=table[identity]
            response=row['budgets']['strict512' if budget else 'full'][ROUTES[route]]
            profile=profiles[identity,route]
            chunks,byte_count,duration=profile[:3]
            assert duration==response['service_s'] and byte_count==response['payload_bytes']
            slot=int(np.argmin(free[route]));begin=max(now,float(free[route][slot]));end=begin+duration
            free[route][slot]=end
            schedule.append(dict(route=route,slot=slot+(sc.edge_slots if route else 0),admission=now,start=begin,end=end,service=s))
            lookups.append(dict(job=j,route=route,budget=budget,response_sha256=response['response_sha256']))
            controller.admit(j,s,route,u,slot,now)
            if begin<=now:controller.start(j,begin)
            else:heapq.heappush(events,(max(tick+1,math.ceil(begin/DT-1e-10)),2,j,begin))
            stream=[(c['time_s'],False,8.*c['utf8_bytes']*sc.payload_scale) for c in chunks if c['utf8_bytes']]
            if len(profile)==4:stream += [(at,True,0.) for at in profile[3]]
            stream.sort(key=lambda item:(item[0],not item[1]))
            for sequence,(at,reset,bits) in enumerate(stream):
                heapq.heappush(events,(max(tick+1,math.ceil((begin+at)/DT-1e-10)),0,j,(sequence,reset,bits)))
            heapq.heappush(events,(max(tick+1,math.ceil(end/DT-1e-10)),1,j,0.))
            selected_quality[j]=response['quality'];shortfall[j]=max(0.,target[j]-selected_quality[j])
            charge[j]=provider_fee(response,route)+head[j]/3600;next_request+=1
        streams=radio_plan(visible,delivered,terminal,first_tick,eos_tick,active,obs)
        fraction=sum(f for u,f,frame in streams);max_fraction=max(max_fraction,fraction);ledger['reserved_airtime_s']+=fraction*DT
        before=np.isfinite(completion)
        for half in (0,1):
            actual=channel.sample(tick*2+half)
            for u,f,frame in streams:
                rate=f*actual[u]
                if rate<=0:continue
                sent,unique,duplicate,padding=frame.consume(rate*HALF,rate,now+half*HALF,
                    delivered,completion,terminal,visible,last)
                for name,value in zip(('wire','unique','duplicate','padding'),(sent,unique,duplicate,padding)):ledger[name]+=value
        finished=terminal & ~np.isfinite(completion) & (visible-delivered<1e-6) & np.isfinite(last)
        completion[finished]=np.maximum(last[finished],eos_tick[finished]*DT)
        for j in np.flatnonzero(~before & np.isfinite(completion)):controller.complete(int(j))
        for u in range(sc.users):active[u]={j for j in active[u] if not terminal[j] or visible[j]-delivered[j]>EPS}
        if next_request==count and np.isfinite(completion).all():break
    else:raise AssertionError('Live streaming trace did not drain')
    take=slice(sc.warmup,None);delay=completion[take]-arrivals[take]
    objective=delay+sc.beta*shortfall[take]+sc.gamma*charge[take]
    assert np.max(abs(generated-delivered))<1e-5
    assert abs(ledger['unique']-float(np.sum(delivered-discarded)))<1e-3
    assert abs(ledger['wire']-ledger['unique']-ledger['duplicate']-ledger['padding'])<1e-3
    assert np.min(completion-eos_tick*DT)>-1e-7
    return dict(completion=completion.tolist(),objective_per_request=objective.tolist(),
        objective=float(objective.mean()),delay=float(delay.mean()),delay_p95=float(np.quantile(delay,.95)),
        shortfall=shortfall[take].tolist(),cost_usd=charge[take].tolist(),selected_quality=selected_quality[take].tolist(),
        choices=choices,physical_cpu_schedule=schedule,controller_actions=controller.actions,
        observed_controller_events=controller.events,selected_response_manifest=lookups,
        generated_bits=generated.tolist(),delivered_bits=(delivered-discarded).tolist(),ledger=ledger,
        discarded_bits=discarded.tolist(),retry_resets=resets,
        max_airtime_sum=max_fraction,elapsed_s=time.perf_counter()-started,
        controller_wall_s=controller.decision_times,controller_and_radio_cpu_excluded=True,
        trace_ticks=tick+1,provider_calls=0,new_policy_enabled=controller_factory is not BudgetController)

def wrapper(task_rows,sc,seed,tr,seconds):
    ready=front_ready(tr.arrival,seconds)
    return dict(seed=int(seed),scenario=sc.dictionary(),result=dict(
        requests=[dict(task_id=task_rows[int(k)]['id'],arrival=float(at)) for k,at in zip(tr.task,tr.arrival)],
        controller_actions=[dict(service=int(s),target=float(q),user=int(u)) for s,q,u in zip(tr.service,tr.quality,tr.user)],
        router_accounting=dict(seconds=list(map(float,seconds)),ready=ready.tolist()),
        # Reserve time to drain the queues after the last request is ready.
        simulation_seconds=float(ready[-1]+1000.)))


def account(w,result,rows):
    table={r['id']:r for r in rows};warm=w['scenario']['warmup'];head_s=w['result']['router_accounting']['seconds']
    sf=[];fees=[];quality=[];statuses={};slots={};generated=[]
    for j,(request,action,cpu,chosen,manifest) in enumerate(zip(w['result']['requests'],
            result['controller_actions'],result['physical_cpu_schedule'],result['choices'],result['selected_response_manifest'])):
        route,budget=chosen;row=table[request['task_id']]
        r=row['budgets']['strict512' if budget else 'full'][('edge','cloud')[route]]
        assert budget==(1 if row['service']=='dialogue' else 0)
        assert manifest['response_sha256']==r['response_sha256']
        assert cpu['route']==route and cpu['start']>=cpu['admission']
        assert cpu['start']>=slots.get(cpu['slot'],0.)-1e-10
        slots[cpu['slot']]=cpu['end']
        assert abs(cpu['end']-cpu['start']-r['service_s'])<1e-8
        assert result['completion'][j]>=cpu['end']-1e-8
        generated.append(r['all_attempt_payload_bytes']*8*w['scenario']['payload_scale'])
        sf.append(max(0.,action['target']-r['quality']));quality.append(r['quality'])
        fee=r['service_s']/3600 if route==0 else (.75*(r['input_tokens']-r['cached_input_tokens'])+
            .075*r['cached_input_tokens']+3.75*(r['output_tokens']+r['thought_tokens']))/1e6
        fees.append(fee+head_s[j]/3600)
        if j>=warm:statuses[r['terminal_status']]=statuses.get(r['terminal_status'],0)+1
    delay=np.array(result['completion'][warm:])-np.array([r['arrival'] for r in w['result']['requests'][warm:]])
    objective=delay+12*np.array(sf[warm:])+1000*np.array(fees[warm:])
    for a,b in ((objective,result['objective_per_request']),(sf[warm:],result['shortfall']),
        (fees[warm:],result['cost_usd']),(quality[warm:],result['selected_quality']),
        (generated,result['generated_bits']),
        (generated,np.array(result['delivered_bits'])+np.array(result['discarded_bits']))):
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-5)
    assert abs(float(objective.mean())-result['objective'])<1e-10
    return dict(objective=float(objective.mean()),delay=float(delay.mean()),delay_p95=float(np.quantile(delay,.95)),
        shortfall=float(np.mean(sf[warm:])),quality=float(np.mean(quality[warm:])),money=float(np.mean(fees[warm:])),
        terminal_statuses=statuses,discarded_bits=float(np.sum(result['discarded_bits'])))

class PF:
    def __init__(self,users):
        self.ema=np.ones(users);self.previous=None;self.user=None;self.alpha=np.exp(-radio.DT/5);self.calls=0;self.max_fraction=0.
    def __call__(self,visible,accepted,terminal,first_tick,eos_tick,active,capacity):
        if self.previous is None:
            self.previous=np.zeros(len(accepted));self.user=np.full(len(accepted),-1,int)
        for u,jobs in enumerate(active):
            for j in jobs:
                assert self.user[j] in (-1,u);self.user[j]=u
        delta=accepted-self.previous;assert np.min(delta)>-1e-7
        assert np.max(np.abs(delta[self.user<0]),initial=0)<1e-7
        sent=np.bincount(self.user[self.user>=0],weights=delta[self.user>=0],minlength=len(capacity))
        if self.calls:self.ema=self.alpha*self.ema+(1-self.alpha)*sent/radio.DT
        self.previous=accepted.copy();self.calls+=1
        queues=[];backlog=np.zeros(len(capacity))
        for u,jobs in enumerate(active):
            done=[j for j in jobs if terminal[j] and visible[j]-accepted[j]>radio.EPS]
            live=[j for j in jobs if not terminal[j] and visible[j]-accepted[j]>radio.EPS]
            done.sort(key=lambda j:(visible[j]-accepted[j],eos_tick[j],j));live.sort(key=lambda j:(first_tick[j],j))
            queues.append(done+live);backlog[u]=sum(visible[j]-accepted[j] for j in queues[-1])
        fractions=radio.allocate(backlog,capacity,radio.DT,rule='pf',average=self.ema)
        assert np.all(fractions>=0) and fractions.sum()<=1+1e-10
        self.max_fraction=max(self.max_fraction,float(fractions.sum()))
        return [(u,float(f),radio.Frame([(j,accepted[j],visible[j]) for j in queues[u]])) for u,f in enumerate(fractions) if f>0]
