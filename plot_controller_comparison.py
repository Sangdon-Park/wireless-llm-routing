"""Draw the archived four-panel controller comparison from public results.

Traffic-seed intervals condition on the recorded task bank. Task resamples
reuse the recorded responses. Plotting requires only the archived results.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t, gaussian_kde


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values):
    values=np.asarray(values)
    half=t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
    return float(values.mean()),float(half)


def draw(data,destination):
    data=Path(data);destination=Path(destination)
    source_paths=[data/name for name in (
        'config.json','expected_results.json','strengthening_results.json.gz',
        'information_results.json.gz','bao_results.json.gz','bao_summary.json',
        'utility_results.json.gz','utility_summary.json')]
    config=read(data/'config.json')
    seeds=sorted(config['seeds']);conditions=list(config['conditions'])
    assert len(seeds)==20 and len(conditions)==12
    strengthening=read_gzip(data/'strengthening_results.json.gz')
    rows=[r for r in read(data/'expected_results.json') if r['scheduler']=='paper']
    rows += [r for r in strengthening if r['replicate']==-1]
    rows += read_gzip(data/'information_results.json.gz')
    rows += read_gzip(data/'bao_results.json.gz')
    index={(r['method'],r['condition'],r['seed']):r['metrics']['objective'] for r in rows}
    assert len(index)==len(rows)
    audited=read(data/'bao_summary.json')
    for method in {r['method'] for r in rows}:
        for condition in conditions:
            values=[index[method,condition,s] for s in seeds]
            assert abs(np.mean(values)-audited[condition]['means'][method]['objective'])<1e-10
    def sample(method,condition=None):
        return np.array([np.mean([index[method,c,s] for c in
                        (conditions if condition is None else [condition])]) for s in seeds])
    result=read(data/'utility_summary.json')
    utility=read_gzip(data/'utility_results.json.gz')
    utility_index={(r['method'],r['condition'],r['seed']):r['metrics']['objective'] for r in utility}
    expected={(m,c,s) for m in ('utility_queue','shortfall_queue') for c in conditions for s in seeds}
    assert len(utility)==480 and set(utility_index)==expected
    def utility_sample(method):
        mapping=index if method=='carrot_queue' else utility_index
        return np.array([np.mean([mapping[method,c,s] for c in conditions]) for s in seeds])
    for left,right in (('shortfall_queue','utility_queue'),
                       ('carrot_queue','shortfall_queue'),('carrot_queue','utility_queue')):
        actual=utility_sample(left)-utility_sample(right)
        archived=result['scopes']['equal12']['comparisons'][left+'_minus_'+right]
        np.testing.assert_array_equal(actual,archived['seed_differences'])
    assert result['independent_confirmation'] is False
    assert result['new_method_validated'] is False
    assert result['native_sfs_reproduced'] is False
    specs=[(10,'bandwidth_10000'),(30,'bandwidth_30000'),(100,'primary'),(300,'bandwidth_300000'),(1000,'bandwidth_1000000')]
    x=np.array([p[0] for p in specs])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'axes.labelsize':7.3,'axes.titlesize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
    fig,ax=plt.subplots(2,2,figsize=(7.2,4.65))
    colors={'mean':'#2166AC','carrot':'#C47717','calibrated':'#21836B','routellm':'#8A5EA4'}
    plotted={'sources':{p.name:sha(p) for p in source_paths},
             'all_archived_means_match':True,'a':{},'b':{},'c':{},'d':{},
             'fixed_task_bank':True,'independent_confirmation':False,
             'new_method_validated':False,'native_sfs_reproduced':False}
    # Use a shared color scale and a fixed controller order.
    methods=[('carrot_queue','Queue / CARROT'),
             ('mean','Queue / Mean'),
             ('calibrated','Queue / Calibrated'),
             ('routellm_score_queue','Queue / RouteLLM-score'),
             ('carrot','CARROT native'),
             ('routellm','RouteLLM native'),
             ('bao_queue','Bao-Queue transfer'),
             ('bao_radio','Bao-Debt transfer')]
    matrix=[]; bandwidth_records=[]
    for method,label in methods:
        samples=[sample(method,c) for _,c in specs]; stats=[ci(a) for a in samples]
        y=np.array([a[0] for a in stats]); h=np.array([a[1] for a in stats])
        matrix.append(y)
        plotted['a'][method]={'samples':np.array(samples).tolist(),'means':y.tolist(),'half95':h.tolist()}
        for (bandwidth,condition),mu,half in zip(specs,y,h):
            bandwidth_records.append({'method':method,'label':label,'condition':condition,
                'bandwidth_khz':bandwidth,'n_seeds':len(seeds),'mean_cost_s':float(mu),
                'ci95_lower_s':float(mu-half),'ci95_upper_s':float(mu+half)})
    matrix=np.asarray(matrix)
    lower=np.floor(matrix.min()); upper=np.ceil(matrix.max())
    # Use the same raw-cost scale across all methods and bandwidths.
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap=LinearSegmentedColormap.from_list('cost',plt.get_cmap('Blues')(np.linspace(.03,.47,256)))
    norm=Normalize(lower,upper)
    panel=ax[0,0]
    panel.imshow(matrix,cmap=cmap,norm=norm,aspect='auto',extent=(0,5,8,0),interpolation='nearest')
    for row,(_,label) in enumerate(methods):
        panel.text(-.13,row+.5,label,ha='right',va='center',fontsize=6.7,color='#252B32')
        for col,value in enumerate(matrix[row]):
            panel.text(col+.5,row+.5,f'{value:.3f}',ha='center',va='center',fontsize=7.1,color='#18212A')
    for row in range(9):
        panel.plot([0,5],[row,row],color='white',lw=.8)
    for col in range(6):
        panel.plot([col,col],[0,8],color='white',lw=.8)
    panel.plot([-2.9,5],[4,4],color='#72808D',lw=.55)
    for col,bandwidth in enumerate(x):
        panel.text(col+.5,-.52,str(bandwidth),ha='center',va='center',fontsize=7.1)
    panel.text(2.5,-1.38,'Shared bandwidth (kHz)',ha='center',va='center',fontsize=7.1)
    panel.text(-.13,-.52,'Routing method',ha='right',va='center',fontsize=6.7,color='#56616C')
    # Draw the color key with the matrix's normalization.
    panel.imshow(np.linspace(lower,upper,256)[None,:],cmap=cmap,norm=norm,
                 aspect='auto',extent=(0,5,8.88,8.5),interpolation='nearest')
    panel.text(-.13,8.7,'Mean cost (s)',ha='right',va='center',fontsize=6.7)
    for value in np.linspace(lower,upper,3):
        xpos=5*(value-lower)/(upper-lower)
        panel.text(xpos,9.38,f'{value:g}',ha='center',va='center',fontsize=6.7)
    panel.set(xlim=(-3.02,5.07),ylim=(9.92,-1.9))
    panel.set_axis_off()
    panel.set_title('(a) Cost by method and bandwidth',loc='left',fontweight='bold')
    plotted['a_display']={'format':'numeric_matrix','decimal_places':3,
        'color_scale':'common_raw_cost_s','color_min':float(lower),'color_max':float(upper),
        'row_order':[m for m,_ in methods],'bandwidths_khz':x.tolist(),
        'uncertainty_file':'bandwidth_comparison.csv','confidence':'two-sided Student t, df=19'}

    # Set axis limits from all seed differences and confidence limits.
    pairs=[('shortfall_queue_minus_utility_queue','Shortfall - Utility',colors['mean'],'o'),
           ('carrot_queue_minus_shortfall_queue','Queue - Shortfall',colors['calibrated'],'s'),
           ('carrot_queue_minus_utility_queue','Queue - Utility',colors['carrot'],'D')]
    limits=[0.]
    for j,(key,label,color,marker) in enumerate(pairs):
        item=result['scopes']['equal12']['comparisons'][key]
        a=np.asarray(item['seed_differences'],dtype=float)
        assert a.shape==(20,) and np.isfinite(a).all()
        mu,h=ci(a); y=2-j
        np.testing.assert_allclose(mu,item['difference_s'],rtol=0,atol=1e-12)
        np.testing.assert_allclose([mu-h,mu+h],item['paired_t95'],rtol=0,atol=1e-12)
        # Deterministic vertical offsets distinguish close points; x remains exact.
        ax[0,1].scatter(a,y+np.linspace(-.13,.13,20),s=13,color=color,alpha=.55,
                         marker=marker,linewidths=0,zorder=2)
        ax[0,1].errorbar(mu,y,xerr=h,fmt=marker,color=color,capsize=3,ms=5,lw=1.5,zorder=4)
        plotted['b'][key]={'seeds':seeds,'samples':a.tolist(),'mean':mu,'half95':h,
                           'label':label,'scope':'equal12'}
        limits += a.tolist()+[mu-h,mu+h]
    minimum,maximum=min(limits),max(limits);pad=max(.015,.08*(maximum-minimum))
    ax[0,1].axvline(0,color='#4B5259',ls='--',lw=.8)
    ax[0,1].set(yticks=[2,1,0],yticklabels=[p[1] for p in pairs],ylim=(-.45,2.45),
                  xlabel='12-condition cost difference (s)',xlim=(minimum-pad,maximum+pad))
    ax[0,1].tick_params(axis='y',labelsize=6.4)
    ax[0,1].set_title('(b) Our controller: clipping and workload',loc='left',fontweight='bold')
    ax[0,1].text(.02,.02,'Fixed CARROT predictions; common queue estimator',
                  transform=ax[0,1].transAxes,fontsize=5.7,va='bottom')

    radio_pairs=[('mean_release','mean','Mean (ours)','mean','o'),
                 ('calibrated_release','calibrated','Calibrated (ours)','calibrated','D'),
                 ('carrot_release','carrot_queue','CARROT predictor','carrot','s'),
                 ('routellm_score_release','routellm_score_queue','RouteLLM-score adapter','routellm','^')]
    for added,base,label,color,marker in radio_pairs:
        samples=[sample(added,c)-sample(base,c) for _,c in specs]; stats=[ci(a) for a in samples]
        y=np.array([a[0] for a in stats]);h=np.array([a[1] for a in stats])
        ax[1,0].plot(x,y,color=colors[color],marker=marker,lw=1.15,ms=3.3,label=label)
        ax[1,0].fill_between(x,y-h,y+h,color=colors[color],alpha=.10,lw=0)
        plotted['c'][added+'_minus_'+base]={'samples':np.array(samples).tolist(),'means':y.tolist(),'half95':h.tolist()}
    ax[1,0].axhline(0,color='#4B5259',ls='--',lw=.7)
    ax[1,0].set(xlabel='Shared bandwidth (kHz)',ylabel='Radio minus Queue cost (s)',ylim=(-.37,.14))
    ax[1,0].set_title('(c) Our controller: radio correction',loc='left',fontweight='bold')
    ax[1,0].legend(frameon=False,ncol=2,fontsize=5.8,loc='lower right',columnspacing=.7,handlelength=1.4)

    boot=[r for r in strengthening if r['replicate']>=0]
    bindex={(r['replicate'],r['method'],r['condition'],r['seed']):r['metrics']['objective'] for r in boot}
    assert len(bindex)==len(boot)==14400
    bootseeds=sorted({r['seed'] for r in boot}); assert len(bootseeds)==2
    differences={m:np.array([np.mean([bindex[b,'calibrated',c,s]-bindex[b,m,c,s] for c in conditions for s in bootseeds]) for b in range(200)]) for m in ('mean','prompt')}
    grid=np.linspace(min(a.min() for a in differences.values())-.025,max(a.max() for a in differences.values())+.025,500)
    for m,color,label in [('mean',colors['mean'],'Calibrated minus Mean'),('prompt',colors['calibrated'],'Calibrated minus Prompt')]:
        values=differences[m]; y=gaussian_kde(values)(grid);lo,hi=np.quantile(values,[.025,.975])
        ax[1,1].plot(grid,y,color=color,lw=1.2,label=label)
        ax[1,1].fill_between(grid,0,y,color=color,alpha=.06,lw=0)
        ax[1,1].fill_between(grid,0,y,where=(grid>=lo)&(grid<=hi),color=color,alpha=.18,lw=0)
        plotted['d'][m]={'replicate_differences':values.tolist(),'percentile95':[float(lo),float(hi)]}
    ax[1,1].axvline(0,color='#4B5259',ls='--',lw=.7)
    ax[1,1].set(xlabel='12-condition cost difference (s)',ylabel='Estimated density',ylim=(0,None))
    ax[1,1].set_title('(d) Our quality model: task sensitivity',loc='left',fontweight='bold')
    ax[1,1].legend(frameon=False,fontsize=6,loc='upper right')
    assert all(abs(result['scopes'][c]['means']['carrot_queue']['objective']-
                   np.mean(sample('carrot_queue',c)))<1e-12 for c in conditions)
    for panel in (ax[1,0],):
        panel.set_xscale('log');panel.set_xticks(x,labels=[str(v) for v in x])
    for panel in (ax[0,1],ax[1,0],ax[1,1]):
        panel.set_axisbelow(True);panel.tick_params(width=.6,length=3)
        panel.grid(axis='x' if panel is ax[0,1] else 'y',alpha=.18,lw=.5)
    fig.subplots_adjust(left=.077,right=.99,bottom=.10,top=.95,wspace=.38,hspace=.49)
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'bandwidth_comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(bandwidth_records[0]))
        writer.writeheader();writer.writerows(bandwidth_records)
    fig.savefig(destination/'results_comparison.pdf',bbox_inches='tight')
    fig.savefig(destination/'results_comparison.png',dpi=230,bbox_inches='tight')
    plt.close(fig)
    (destination/'figure_data.json').write_text(json.dumps(plotted,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'archive_rows':len(rows),'all_means_verified':True,
        'paired_queue_contrasts':3,'paired_queue_points':60,'radio_predictors':4,
        'bootstrap_replicates':200,'provider_calls':0,'new_simulations':0,
        'output':str(destination)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--output',type=Path,default=Path('outputs/figures'))
    args=parser.parse_args()
    draw(args.data_dir,args.output)


if __name__=='__main__':
    main()
