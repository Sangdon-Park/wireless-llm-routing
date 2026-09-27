"""Matched wireless controls and paired task-composition sensitivity."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t,gaussian_kde
from qbr.data import Study,DATA
from strengthening_summary import load_records,summarize


def plot(destination):
    destination=Path(destination);destination.mkdir(exist_ok=True,parents=True)
    records=load_records();summary=summarize(records);study=Study()
    main=[r for r in study.expected if r['scheduler']=='paper']+[r for r in records if r['replicate']==-1]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'axes.titlesize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65})
    fig,grid_axes=plt.subplots(2,2,figsize=(7.2,4.55))
    axes=grid_axes[1]
    values=[(10,'bandwidth_10000'),(30,'bandwidth_30000'),(100,'primary'),(300,'bandwidth_300000'),(1000,'bandwidth_1000000')]
    original=[('carrot','CARROT','#df8615','s','--'),('routellm','RouteLLM-BERT','#8662b0','^','-.'),
              ('service','Service','#657383','D',':'),('prompt','Prompt','#2f95bf','v','--'),
              ('calibrated','Calibrated','#00877e','o','-')]
    for method,label,color,marker,line in original:
        samples=[np.array([r['metrics']['objective'] for r in main if r['method']==method and r['condition']==condition]) for _,condition in values]
        y=np.array([a.mean() for a in samples]);half=np.array([t.ppf(.975,19)*a.std(ddof=1)/np.sqrt(20) for a in samples])
        x=np.array([v[0] for v in values])
        grid_axes[0,0].plot(x,y,color=color,label=label,marker=marker,ls=line,lw=1.25,ms=3.4)
        grid_axes[0,0].fill_between(x,y-half,y+half,color=color,alpha=.09,lw=0)
    grid_axes[0,0].set_xscale('log');grid_axes[0,0].set_xticks(x,labels=[str(v) for v in x])
    grid_axes[0,0].set(xlabel='Shared bandwidth (kHz)',ylabel='Weighted response cost (s)')
    grid_axes[0,0].set_title('(a) External and internal routers',fontweight='bold')
    with np.load(DATA/'primary_latencies.npz',allow_pickle=False) as z:
        for method,label,color,marker,line in original:
            latency=np.sort(z[method]);assert len(latency)==24000
            grid_axes[0,1].plot(latency,np.arange(1,len(latency)+1)/len(latency),color=color,ls=line,lw=1.15)
    grid_axes[0,1].set(xlabel='Response latency (s)',ylabel='Empirical CDF',ylim=(0,1.02))
    grid_axes[0,1].set_title('(b) Primary latency distribution',fontweight='bold')
    labels={'mean':'Mean','calibrated':'Calibrated','radio_mean':'Radio-Mean','radio_calibrated':'Radio-Calibrated'}
    colors={'mean':'#cf7b22','calibrated':'#00877e','radio_mean':'#7a52a1','radio_calibrated':'#26628c'}
    for method,marker,line in [('mean','s','--'),('calibrated','o','-'),('radio_mean','^','--'),('radio_calibrated','D','-')]:
        y=[];half=[]
        for x,condition in values:
            sample=np.array([r['metrics']['objective'] for r in main if r['method']==method and r['condition']==condition])
            assert len(sample)==20
            y.append(sample.mean());half.append(t.ppf(.975,19)*sample.std(ddof=1)/np.sqrt(20))
        x=np.array([v[0] for v in values]);y=np.array(y);half=np.array(half)
        axes[0].plot(x,y,color=colors[method],label=labels[method],marker=marker,ls=line,lw=1.25,ms=3.4)
        axes[0].fill_between(x,y-half,y+half,color=colors[method],alpha=.09,lw=0)
    axes[0].set_xscale('log');axes[0].set_xticks(x,labels=[str(v) for v in x])
    axes[0].set(xlabel='Shared bandwidth (kHz)',ylabel='Weighted response cost (s)')
    axes[0].set_title('(c) Matched quality and radio controls',fontweight='bold')
    axes[0].legend(frameon=False,ncol=2,fontsize=6.4,loc='upper right',columnspacing=.7,handlelength=1.5)
    differences={m:np.array(summary['equal12']['task_composition']['calibrated_minus_'+m]['replicate_difference_s']) for m in ('mean','prompt')}
    lower=min(x.min() for x in differences.values());upper=max(x.max() for x in differences.values());span=upper-lower
    grid=np.linspace(lower-.08*span,upper+.08*span,400)
    for method,color,label in [('mean','#cf7b22','Calibrated − Mean'),('prompt','#00877e','Calibrated − Prompt')]:
        density=gaussian_kde(differences[method])(grid)
        axes[1].plot(grid,density,color=color,lw=1.25,label=label)
        axes[1].fill_between(grid,0,density,color=color,alpha=.17,lw=0)
        interval=np.quantile(differences[method],[.025,.975])
        inside=(grid>=interval[0])&(grid<=interval[1])
        axes[1].fill_between(grid,0,density,where=inside,color=color,alpha=.18,lw=0)
    axes[1].axvline(0,color='#333333',lw=.85,ls='--')
    axes[1].set(xlabel='12-condition cost difference (s)',ylabel='Estimated density',ylim=(0,None))
    axes[1].set_title('(d) 200 paired task-bank resamples',fontweight='bold')
    axes[1].legend(frameon=False,fontsize=6.6,loc='upper right')
    for ax in grid_axes.flat:
        ax.grid(axis='y',alpha=.22,lw=.5);ax.set_axisbelow(True);ax.tick_params(width=.6,length=3)
    handles,legend_labels=grid_axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,ncol=5,frameon=False,loc='upper center',fontsize=7,
               columnspacing=1.5,handlelength=2.1)
    fig.subplots_adjust(left=.075,right=.992,bottom=.11,top=.88,wspace=.32,hspace=.48)
    fig.savefig(destination/'results_comparison.pdf',bbox_inches='tight')
    fig.savefig(destination/'results_comparison.png',dpi=220,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('outputs'))
    plot(parser.parse_args().output)
