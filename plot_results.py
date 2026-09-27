"""Regenerate the original baseline panels from all published observations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
from qbr.data import DATA, Study

METHODS = ('carrot','routellm','service','prompt','calibrated')
LABELS = ('CARROT','RouteLLM-BERT','Service (ours)','Prompt (ours)','Calibrated (ours)')
COLORS = ('#df8615','#8662b0','#657383','#2f95bf','#00877e')
MARKERS = ('s','^','D','v','o')
LINES = ('--','-.',':','--','-')


def plot(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    study = Study()
    records = [r for r in study.expected if r['scheduler']=='paper']
    conditions = study.config['conditions']
    primary = conditions['primary']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65})
    fig,axes = plt.subplots(2,2,figsize=(7.2,4.4))
    def matching(field):
        return sorted((values[field],name) for name,values in conditions.items()
            if all(values[k]==v for k,v in primary.items() if k!=field))
    def sweep(ax,field):
        selected = matching(field)
        for m,label,color,marker,line in zip(METHODS,LABELS,COLORS,MARKERS,LINES):
            x,means,intervals = [],[],[]
            for value,name in selected:
                samples = np.array([r['metrics']['objective'] for r in records if r['condition']==name and r['method']==m])
                assert len(samples)==20
                x.append(value);means.append(samples.mean())
                intervals.append(t.ppf(.975,19)*samples.std(ddof=1)/np.sqrt(20))
            x = np.asarray(x)/(1000 if field=='bandwidth_hz' else 1)
            means,intervals = np.asarray(means),np.asarray(intervals)
            ax.plot(x,means,color=color,marker=marker,ls=line,lw=1.3,ms=3.5,label=label)
            ax.fill_between(x,means-intervals,means+intervals,color=color,alpha=.09,lw=0)
        ax.set_ylabel('Weighted response cost (s)')
        ax.set_xticks(x)
    sweep(axes[0,0],'bandwidth_hz')
    axes[0,0].set_xscale('log')
    axes[0,0].set_xticks([10,30,100,300,1000],labels=['10','30','100','300','1000'])
    axes[0,0].set_xlabel('Shared bandwidth (kHz)')
    axes[0,0].set_title('(a) Bandwidth sweep',fontweight='bold')
    with np.load(DATA/'primary_latencies.npz',allow_pickle=False) as z:
        for m,color,line in zip(METHODS,COLORS,LINES):
            x = np.sort(z[m]);assert len(x)==24000
            axes[0,1].plot(x,np.arange(1,len(x)+1)/len(x),color=color,ls=line,lw=1.2)
    axes[0,1].set(xlabel='Response latency (s)',ylabel='Empirical CDF',ylim=(0,1.02))
    axes[0,1].set_title('(b) Primary latency distribution',fontweight='bold')
    parts = []
    for m in METHODS:
        rows = [r['metrics'] for r in records if r['condition']=='primary' and r['method']==m]
        parts.append([np.mean([r['delay'] for r in rows]),12*np.mean([r['shortfall'] for r in rows]),1000*np.mean([r['money'] for r in rows])])
    parts = np.array(parts);bottom=np.zeros(5)
    for k,(label,color) in enumerate(zip(('Latency','Quality shortfall','Monetary charge'),('#29617f','#dfad59','#a4c6bc'))):
        axes[1,0].bar(np.arange(5),parts[:,k],bottom=bottom,color=color,width=.76,label=label,edgecolor='white',linewidth=.5)
        bottom += parts[:,k]
    errors = [t.ppf(.975,19)*np.std([r['metrics']['objective'] for r in records if r['condition']=='primary' and r['method']==m],ddof=1)/np.sqrt(20) for m in METHODS]
    axes[1,0].errorbar(np.arange(5),bottom,yerr=errors,fmt='none',ecolor='#303030',capsize=2,lw=.7)
    axes[1,0].set_xticks(range(5),['CARROT','RouteLLM','Service','Prompt','Calibrated'],fontsize=7)
    axes[1,0].set_ylabel('Weighted cost components (s)')
    axes[1,0].set_ylim(0,7.4)
    axes[1,0].legend(frameon=False,fontsize=6.5,loc='upper center',ncol=3,
                     handlelength=1,columnspacing=.8)
    axes[1,0].set_title('(c) Primary cost breakdown',fontweight='bold')
    sweep(axes[1,1],'rho')
    axes[1,1].set_xlabel('Normalized arrival load')
    axes[1,1].set_title('(d) Load sensitivity',fontweight='bold')
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.22,lw=.5)
        ax.set_axisbelow(True)
        ax.tick_params(width=.6,length=3)
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,frameon=False,loc='upper center',fontsize=7,
        columnspacing=1.3,handlelength=2.3)
    fig.subplots_adjust(left=.075,right=.99,bottom=.11,top=.88,wspace=.32,hspace=.48)
    fig.savefig(output/'results.pdf',bbox_inches='tight')
    fig.savefig(output/'results.png',dpi=200,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'figure':str(output/'results.pdf'),'methods':5,'conditions':12,'seeds':20}))
