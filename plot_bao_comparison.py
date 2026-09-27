"""All plotted quantities are archived trajectories or paired resamples."""
from pathlib import Path
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t, gaussian_kde
from qbr import information_controls as controls
from qbr import bao
from strengthening_summary import load_records, summarize


def draw(rows,summary,bootstrap,destination):
    destination=Path(destination);destination.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.labelsize':7.5,
        'axes.titlesize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65})
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.65))
    specs=[(10,'bandwidth_10000'),(30,'bandwidth_30000'),(100,'primary'),(300,'bandwidth_300000'),(1000,'bandwidth_1000000')]
    x=np.array([v[0] for v in specs])
    colors={'mean':'#2869A5','carrot':'#BC6D15','routellm':'#8B62A8','calibrated':'#247C69'}
    styles=[('carrot','CARROT native','#BC6D15','s','--'),('routellm','RouteLLM native','#8B62A8','^','--'),
        ('carrot_queue','CARROT--Queue','#BC6D15','s','-'),('routellm_score_queue','RouteLLM-score--Queue','#8B62A8','^','-'),
        ('mean','Mean--Queue','#2869A5','o','-'),('calibrated','Calibrated--Queue','#247C69','D',':'),
        ('bao_queue','Bao--Queue','#333333','v','-.'),('bao_radio','Bao--Debt','#AA3B48','P','-.')]
    for method,label,color,marker,line in styles:
        samples=[np.array([r['metrics']['objective'] for r in rows if r['method']==method and r['condition']==condition]) for _,condition in specs]
        assert all(len(s)==20 for s in samples)
        y=np.array([a.mean() for a in samples]);half=np.array([t.ppf(.975,19)*a.std(ddof=1)/np.sqrt(20) for a in samples])
        axes[0,0].plot(x,y,color=color,label=label,marker=marker,ls=line,lw=1.1,ms=3.1,mfc='white' if line=='--' else color)
        axes[0,0].fill_between(x,y-half,y+half,color=color,alpha=.055,lw=0)
    axes[0,0].set(xlabel='Shared bandwidth (kHz)',ylabel='Weighted response cost (s)')
    axes[0,0].set_title('(a) External rules and matched controls',loc='left',fontweight='bold')
    axes[0,0].legend(frameon=False,fontsize=5.9,ncol=2,loc='upper right',columnspacing=.5,handlelength=1.4)
    axes[0,0].set_ylim(4.9,9.45)
    for ax in (axes[0,0],axes[1,0]):
        ax.set_xscale('log');ax.set_xticks(x,labels=[str(v) for v in x])

    # Equal-condition averages are formed within seed before the interval.
    for head,color,offset in [('mean',colors['mean'],.12),('carrot',colors['carrot'],-.12)]:
        for j,mode in enumerate(('intrinsic','static','queue')):
            method='mean' if head=='mean' and mode=='queue' else head+'_'+mode
            samples=np.array([np.mean([r['metrics']['objective'] for r in rows if r['method']==method and r['seed']==seed]) for seed in sorted({r['seed'] for r in rows})])
            assert len(samples)==20
            half=t.ppf(.975,19)*samples.std(ddof=1)/np.sqrt(20)
            yy=2-j+offset
            axes[0,1].scatter(samples,np.full(20,yy),s=7,color=color,alpha=.17,zorder=2,linewidths=0)
            axes[0,1].errorbar(samples.mean(),yy,xerr=half,fmt='o' if head=='mean' else 's',ms=4,
                color=color,capsize=2.5,lw=1.2,label=('Mean' if head=='mean' else 'CARROT') if j==0 else None,zorder=4)
    axes[0,1].set(yticks=[0,1,2],yticklabels=['Queue','Static','Intrinsic'],ylim=(-.45,2.55),
        xlabel='12-condition weighted response cost (s)',xlim=(5.15,9.35))
    axes[0,1].set_title('(b) Workload-score ablation',loc='left',fontweight='bold')
    axes[0,1].legend(frameon=False,fontsize=6.5,ncol=2,loc='lower right')
    axes[0,1].grid(axis='x',alpha=.2,lw=.5)

    for key,label,color,marker in [('mean_release_minus_mean','Mean',colors['mean'],'o'),
                                 ('carrot_release_minus_carrot_queue','CARROT',colors['carrot'],'s')]:
        stats=[summary[c]['comparisons'][key] for _,c in specs]
        y=np.array([a['difference_s'] for a in stats]);lo=np.array([a['paired_t95'][0] for a in stats]);hi=np.array([a['paired_t95'][1] for a in stats])
        axes[1,0].plot(x,y,color=color,marker=marker,lw=1.3,ms=3.6,label=label)
        axes[1,0].fill_between(x,lo,hi,color=color,alpha=.16,lw=0)
    axes[1,0].axhline(0,color='#4B5259',ls='--',lw=.7)
    axes[1,0].set(xlabel='Shared bandwidth (kHz)',ylabel='Radio minus Queue cost (s)',ylim=(-.33,.12))
    axes[1,0].set_title('(c) Incremental effect of radio state',loc='left',fontweight='bold')
    axes[1,0].legend(frameon=False,ncol=2,fontsize=6.7,loc='lower right')

    differences={m:np.array(bootstrap['equal12']['task_composition']['calibrated_minus_'+m]['replicate_difference_s']) for m in ('mean','prompt')}
    grid=np.linspace(min(a.min() for a in differences.values())-.025,max(a.max() for a in differences.values())+.025,500)
    for method,color,label in [('mean',colors['mean'],'Calibrated minus Mean'),('prompt',colors['calibrated'],'Calibrated minus Prompt')]:
        y=gaussian_kde(differences[method])(grid);lo,hi=np.quantile(differences[method],[.025,.975])
        axes[1,1].plot(grid,y,color=color,lw=1.2,label=label)
        axes[1,1].fill_between(grid,0,y,color=color,alpha=.07,lw=0)
        axes[1,1].fill_between(grid,0,y,where=(grid>=lo)&(grid<=hi),color=color,alpha=.19,lw=0)
    axes[1,1].axvline(0,color='#4B5259',ls='--',lw=.7)
    axes[1,1].set(xlabel='12-condition cost difference (s)',ylabel='Estimated density',ylim=(0,None))
    axes[1,1].set_title('(d) Task-composition sensitivity',loc='left',fontweight='bold')
    axes[1,1].legend(frameon=False,fontsize=6,loc='upper right')
    for ax in axes.flat:
        ax.set_axisbelow(True);ax.tick_params(width=.6,length=3)
        if ax is not axes[0,1]:ax.grid(axis='y',alpha=.20,lw=.5)
    fig.subplots_adjust(left=.076,right=.99,bottom=.10,top=.95,wspace=.38,hspace=.47)
    fig.savefig(destination/'results_comparison.pdf',bbox_inches='tight')
    fig.savefig(destination/'results_comparison.png',dpi=240,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/bao-figure'))
    args=parser.parse_args()
    controls.verify();bao.verify();study,rows=controls.all_records()
    rows+=bao.records()
    draw(rows,controls.summary(),summarize(load_records()),args.output)
