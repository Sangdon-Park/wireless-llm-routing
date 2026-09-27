"""Information-matched routing controls; native published routers stay separate."""
import gzip
import hashlib
import json
from time import perf_counter
import numpy as np
from scipy.stats import t
from sklearn.model_selection import KFold
from .data import Study, DATA
from . import replay
from .policy import BudgetController, scores, best, required_actions
from .strengthening import RadioController, mean_risks

METHODS = ('carrot_queue','carrot_radio','routellm_score_queue','routellm_score_radio',
    'mean_release','calibrated_release','carrot_release','routellm_score_release',
    'mean_static','mean_intrinsic','carrot_static','carrot_intrinsic')

def predict(x, y, queries, k):
    # Stable input order resolves identical preference-score distances.
    order = np.argsort(np.abs(np.asarray(queries)[:, None]-x[None, :]), axis=1, kind='stable')[:, :k]
    return y[order].mean(axis=1)


def release_correction(backlog,capacity,payload,waits,duration,user):
    debt=float(np.sum(np.asarray(backlog)/np.asarray(capacity)))
    payload=np.asarray(payload);waits=np.asarray(waits)[:,None];duration=np.asarray(duration)
    radio=np.maximum(np.maximum(debt,waits)+payload/float(capacity[user])-waits-duration,0.)
    return np.where(payload>0,radio,0.)


class ReleaseController(RadioController):
    def choose(self,now,j,service,target,user,observed,wire,throughput):
        started=perf_counter()
        assert self.radio_observed_at==now
        waits,active,queued=self.compute_state(now)
        profile=self.forecast.decision
        allowed=required_actions(profile,service,self.mode,self.fixed)
        values=scores(profile,service,float(target),waits,risk=self.risks[j],required=allowed)
        correction=release_correction(self.radio_backlog,observed,
            self.forecast.mean_bits[service]*self.sc.payload_scale,waits,profile.mu[service],user)
        values=np.where(allowed,values+correction,np.inf)
        route,budget=best(values,allowed)
        self.choices[j]=(int(route),int(budget))
        self.decision_times.append(perf_counter()-started)
        route_values=values.min(1)
        self.actions.append(dict(job=j,time=float(now),route=int(route),budget=int(budget),
            service=service,target=float(target),user=user,waits=waits.tolist(),
            values=route_values.tolist(),radio_excess_s=correction[:,budget].tolist(),
            observed_capacity=np.asarray(observed).tolist(),backlog_bits=self.radio_backlog.tolist()))
        return int(route),[0,0],route_values


class AblationController(BudgetController):
    def __init__(self, *args, intrinsic=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.intrinsic = intrinsic

    def choose(self, now, j, service, target, user, observed, wire, throughput):
        started = perf_counter()
        allowed = required_actions(self.forecast.decision, service, self.mode, self.fixed)
        # Static retains the empty-queue score, including the fixed second moment.
        # Intrinsic also removes the entire workload potential penalty.
        values = scores(self.forecast.decision, service, float(target), np.zeros(2),
                        risk=self.risks[j], required=allowed,
                        coefficient=0. if self.intrinsic else .5)
        route, budget = best(values, allowed)
        self.choices[j] = (int(route), int(budget))
        self.decision_times.append(perf_counter()-started)
        self.actions.append(dict(job=j, time=float(now), route=int(route), budget=int(budget),
                                 service=service, target=float(target), user=user))
        return int(route), [0, 0], values.min(1)


def initialize():
    global STUDY, HEADS, TIMES
    STUDY=Study()
    HEADS=dict(np.load(DATA/'information_predictions.npz',allow_pickle=False))
    TIMES=dict(np.load(DATA/'information_times.npz',allow_pickle=False))


def episode(job):
    condition,seed,method=job
    assert method in METHODS
    sc,tr=STUDY.requests(condition,seed)
    if method.startswith('mean_'):
        risks=mean_risks(STUDY,tr);seconds=STUDY.prediction_seconds('calibrated',seed)
    elif method=='calibrated_release':
        risks=STUDY.risks('calibrated',tr);seconds=STUDY.prediction_seconds('calibrated',seed)
    else:
        head='carrot' if method.startswith('carrot') else 'routellm_score'
        risks=np.full((len(tr.task),2,2),np.nan)
        for s in range(3):
            take=np.flatnonzero(tr.service==s)
            risks[take,:,int(s==0)]=np.maximum(tr.quality[take,None]-HEADS[head][tr.task[take]],0.)
        seconds=STUDY.prediction_seconds('carrot' if head=='carrot' else 'routellm',seed)+TIMES[str(seed)+'_'+head]
    if method.endswith('_release'):cls=ReleaseController
    elif method.endswith('_radio'):cls=RadioController
    elif method.endswith(('_static','_intrinsic')):cls=AblationController
    else:cls=BudgetController
    extra={'intrinsic':method.endswith('_intrinsic')} if cls is AblationController else {}
    factory=lambda *a,**k:cls(*a,risks=risks,**extra,**k)
    wrapped=replay.wrapper(STUDY.tasks,sc,seed,tr,seconds)
    result=replay.simulate(wrapped,STUDY.rows,STUDY.profiles,STUDY.forecast,controller_factory=factory)
    return dict(condition=condition,seed=seed,method=method,metrics=replay.account(wrapped,result,STUDY.rows),
        choices_sha256=hashlib.sha256(np.asarray(result['choices'],dtype='<i8').tobytes()).hexdigest(),
        cloud_fraction=float(np.mean(np.array(result['choices'])[sc.warmup:,0])),
        head_mean_ms=1000*float(np.mean(seconds)),max_airtime_sum=result['max_airtime_sum'])


def load_records():
    with gzip.open(DATA/'information_results.json.gz','rt',encoding='utf-8') as f:return json.load(f)


def all_records():
    study=Study();rows=load_records()
    with gzip.open(DATA/'strengthening_results.json.gz','rt',encoding='utf-8') as f:
        rows += [r for r in json.load(f) if r['replicate']==-1]
    rows += [r for r in study.expected if r['scheduler']=='paper']
    return study,rows


def summary():
    study,rows=all_records();out={}
    pairs=[('mean','mean_intrinsic'),('carrot_queue','carrot_intrinsic'),('mean','mean_static'),
        ('carrot_queue','carrot_static'),('carrot_intrinsic','carrot'),('carrot_queue','carrot'),
        ('mean_release','mean'),('carrot_release','carrot_queue'),('mean_release','carrot_release'),
        ('mean_release','radio_mean'),('calibrated_release','mean_release')]
    for scope in ['equal12']+list(study.config['conditions']):
        take=[r for r in rows if scope=='equal12' or r['condition']==scope]
        means={m:{k:float(np.mean([r['metrics'][k] for r in take if r['method']==m]))
          for k in ('objective','delay','delay_p95','shortfall','money')} for m in sorted({r['method'] for r in rows})}
        comparisons={}
        for a,b in pairs:
            ds=[np.mean([r['metrics']['objective'] for r in take if r['method']==a and r['seed']==s])-
                np.mean([r['metrics']['objective'] for r in take if r['method']==b and r['seed']==s]) for s in study.config['seeds']]
            d=float(np.mean(ds));h=float(t.ppf(.975,19)*np.std(ds,ddof=1)/np.sqrt(20))
            comparisons[a+'_minus_'+b]=dict(difference_s=d,paired_t95=[d-h,d+h],
                reduction_pct=100*(1-means[a]['objective']/means[b]['objective']))
        out[scope]=dict(means=means,comparisons=comparisons)
    return out


def verify():
    manifest=json.loads((DATA/'information_manifest.json').read_text())
    for name,h in manifest['data_hashes'].items():
        assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==h,name
    study=Study();rows=load_records()
    keys={(c,s,m) for c in study.config['conditions'] for s in study.config['seeds'] for m in METHODS}
    assert len(rows)==len(keys)==2880
    assert {(r['condition'],r['seed'],r['method']) for r in rows}==keys
    for r in rows:
        v=r['metrics'];assert abs(v['objective']-v['delay']-12*v['shortfall']-1000*v['money'])<1e-9
        assert 0<=r['max_airtime_sum']<=1+1e-10
    z=np.load(DATA/'information_fit.npz',allow_pickle=False)
    heads=np.load(DATA/'information_predictions.npz',allow_pickle=False)
    cv_losses=[];selected=[]
    for service in range(3):
        ids=np.flatnonzero(z['train_service']==service)
        test=np.flatnonzero(z['evaluation_service']==service)
        x,y=z['train_score'][ids],z['train_quality'][ids]
        folds=list(KFold(5,shuffle=True,random_state=23926000).split(ids));losses={}
        for k in (1,2,4,8,16,32,64,128):
            if any(k>len(a) for a,b in folds):continue
            oof=np.full_like(y,np.nan)
            for train,valid in folds:oof[valid]=predict(x[train],y[train],x[valid],k)
            losses[k]=float(np.mean((oof-y)**2))
        best_k=min(losses,key=lambda k:(losses[k],-k));selected.append(best_k)
        np.testing.assert_allclose(predict(x,y,z['evaluation_score'][test],best_k),heads['routellm_score'][test],rtol=0,atol=1e-14)
        cv_losses.append(losses)
    assert selected==[128,32,128]
    np.testing.assert_array_equal(heads['carrot'],study.external['evaluation_quality'])
    return dict(records=2880,adapter_fit_verified=True,selected_neighbors=selected,
        new_model_responses=0,release_candidate_passed=False,original_radio_candidate_passed=False)
