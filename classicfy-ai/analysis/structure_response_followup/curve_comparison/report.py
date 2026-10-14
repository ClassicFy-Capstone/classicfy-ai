"""Render transparent comparisons and an individual tension-response example."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1]
REPO=WORK/'capstone-2026-2-classicfy'


def rows(name):return list(csv.DictReader((OUT/name).open()))


def main():
    summary={(r['role'],r['method']):r for r in rows('summary.csv')+rows('coupled_summary.csv')}
    controls={(r['role'],r['method'],r['control']):r for r in rows('control_summary.csv')+rows('coupled_control_summary.csv')}
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    phrase=[('F_summary','Fixed summary'),('P_summary','Phrase summary'),('P_curve_centered','Phrase curves'),('P_curve_centered:shuffle','Curves shuffled')]
    tension=[('H_summary','Tension summary'),('H_curve_centered','Peak-aligned curves'),('H_paired_curve','Paired curves'),('H_context_only','Score curves only')]
    panels=[('E-PC','Curve agreement with amplitude',phrase),('E-PN','Curve shape, amplitude removed',phrase),
        ('E-HL','Tension-response lag agreement',tension),('E-T','General change magnitude',
            [('P_summary','Phrase summary'),('P_curve_centered','Phrase curves'),('H_summary','Tension summary'),('H_paired_curve','Paired curves')])]
    for ax,(axis,title,methods) in zip(axes.ravel(),panels):
        x=np.arange(len(methods));width=.35
        for offset,(role,color) in enumerate((('test','#2563eb'),('validation','#0f766e'))):
            values=[]
            for method,label in methods:
                if method.endswith(':shuffle'):r=controls[role,method.split(':')[0],'point_order_permuted']
                else:r=summary[role,method]
                values.append(float(r[axis]))
            bars=ax.bar(x+(offset-.5)*width,values,width,color=color,label=role)
            ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
        ax.axhline(50,color='#64748b',lw=1,ls='--');ax.set_ylim(0,100)
        ax.set_xticks(x,[label for method,label in methods],rotation=12,ha='right')
        ax.set_title(title+' ('+axis+')',fontsize=12);ax.set_ylabel('Relative score (random expectation 50)')
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(loc='upper left')
    fig.suptitle('Summary vs individual ordered curves: same 329 recordings\n241 gallery; 45 test / 43 validation queries; opposite-view evaluation',fontsize=14)
    fig.savefig(OUT/'01_comparison.png',dpi=160);plt.close(fig)
    # One fixed query, its middle strain-peak window, and top-ranked candidates.
    c=np.load(OUT/'curves.npz');d=np.load(OUT/'comparison.npz');keys=c['keys'].tolist();index={k:i for i,k in enumerate(keys)}
    qi=int(d['query_indices'][0]);view=0;kind='H2'
    source=c[f'{qi}/{view}/{kind}']-c[f'{qi}/{view}/mean'][None,:,None]
    qwin=len(source)//2
    qspan=c[f'{qi}/{view}/{kind}_spans'][qwin]
    qt=np.round(c[f'{qi}/tension'][2,qspan[0]:qspan[1]],12)
    ranks=rows('rankings.csv')+rows('coupled_rankings.csv')
    pair_scales=json.loads((OUT/'coupled_scales.json').read_text())['A_to_B/H2']
    ps=np.array(pair_scales['play_sd']);ts=float(pair_scales['tone_sd'][0])
    examples=[dict(key=keys[qi],curve=source[qwin],tension=qt,span=qspan.tolist(),title='Query')]
    for method,label in (('H_summary','Summary top result'),('H_paired_curve','Paired-curve top result')):
        record=next(r for r in ranks if r['direction']=='A_to_B' and r['query']==keys[qi] and r['method']==method and r['rank']=='1')
        ci=index[record['candidate']]
        cc=c[f'{ci}/{view}/{kind}']-c[f'{ci}/{view}/mean'][None,:,None]
        spans=c[f'{ci}/{view}/{kind}_spans']
        ct=np.array([np.round(c[f'{ci}/tension'][2,a:b],12) for a,b in spans])
        costs=.5*np.mean(((cc-source[qwin])/ps[None,:,None])**2,axis=(1,2))+.5*np.mean(((ct-qt)/ts)**2,axis=1)
        win=int(np.argmin(costs))
        examples.append(dict(key=keys[ci],curve=cc[win],tension=ct[win],span=spans[win].tolist(),title=label))
    fig,axes=plt.subplots(3,3,figsize=(13,8),sharex=True,sharey='row',layout='constrained')
    for j,example in enumerate(examples):
        values=[example['tension'],example['curve'][0],example['curve'][2]]
        for row,value in enumerate(values):
            ax=axes[row,j];ax.plot(np.arange(-8,9),value,lw=2,color=('#7c3aed','#2563eb','#0f766e')[row]);ax.axvline(0,color='#64748b',ls='--',lw=1)
            if row:ax.axhline(0,color='#94a3b8',lw=.7)
            ax.spines[['top','right']].set_visible(False)
            if j==0:ax.set_ylabel(('Score tensile strain','Centered tempo residual','Centered dynamics residual')[row])
        axes[0,j].set_title(example['title']+'\n'+Path(example['key']).name,fontsize=10)
        axes[2,j].set_xlabel('Original beats relative to score peak')
    fig.suptitle('Individual windows remain separate; no across-window mean curve\nIllustration only: first test query, middle strain window, locally nearest window in each top-ranked recording',fontsize=12)
    fig.savefig(OUT/'02_individual_curves.png',dpi=160);plt.close(fig)
    (OUT/'example.json').write_text(json.dumps([{k:v for k,v in e.items() if k not in ('curve','tension')} for e in examples],indent=2)+'\n')
    print('Rendered comparison and individual-curve figures')


if __name__=='__main__':main()
