"""Report every prespecified endpoint and control without selecting a winner."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/structure_response_comparison')
    args=parser.parse_args();out=args.out
    cohort=json.loads((out/'cohort.json').read_text())
    summary=pd.read_csv(out/'summary.csv');family=pd.read_csv(out/'per_family.csv');controls=pd.read_csv(out/'controls.csv')
    methods=['B','B+T','B+F','B+P','B+H','B+P+H','M-CT','Q','structure_only']
    axes_names=['E-L','E-T','E-R','E-S']
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    for row,role in enumerate(['test','validation']):
        for column,axis in enumerate(axes_names):
            ax=axes[row,column];values=summary[(summary.role==role)&(summary.axis==axis)].set_index('method').loc[methods,'score']*100
            ax.bar(np.arange(len(methods)),values,color=['#8b949e','#bbc1c8','#789ec0','#4286c0','#e9a051','#9671b6','#aaa18f','#cecece','#cecece'])
            ax.axhline(50,color='#555555',ls='--',lw=.8);ax.set_ylim(35,85)
            ax.set_xticks(np.arange(len(methods)),methods,rotation=65,ha='right',fontsize=8)
            ax.set_title(role+' / '+axis);ax.set_ylabel('Held-out S@5 (0-100)')
    fig.suptitle('Same cohort, disjoint-beat consistency; random expectation 50\nExploratory: '
        +str(cohort['query_families']['test'])+' test and '+str(cohort['query_families']['validation'])
        +' validation families; no listening ground truth',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(out/('01_all_endpoints.'+ext),dpi=160)
    plt.close(fig)
    contrasts=[('B+P','B+F'),('B+H','B+F'),('B+P+H','B+T')];contrast_rows=[]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,role in zip(axes,['test','validation']):
        table=family[(family.role==role)&(family.axis=='E-T')].pivot(index='family_id',columns='method',values='score')
        x=np.arange(len(table));width=.24
        for j,(a,b) in enumerate(contrasts):
            delta=(table[a]-table[b])*100
            ax.bar(x+(j-1)*width,delta,width,label=a+' vs '+b)
            contrast_rows.extend(dict(role=role,family_id=f,added=a,reference=b,delta_points=float(d)) for f,d in delta.items())
        ax.set_xticks(x,table.index,rotation=35,ha='right',fontsize=8);ax.axhline(0,color='#555555',lw=.8)
        ax.set_title(role+' / E-T per family');ax.set_ylabel('Difference in score points');ax.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(out/('02_per_family_contrasts.'+ext),dpi=160)
    plt.close(fig)
    pd.DataFrame(contrast_rows).to_csv(out/'contrasts.csv',index=False)
    control_summary=[]
    for (role,control,method,axis),group in controls.groupby(['role','control','method','axis']):
        values=group.groupby('seed').score.mean()
        control_summary.append(dict(role=role,control=control,method=method,axis=axis,mean=float(values.mean()),p05=float(values.quantile(.05)),p95=float(values.quantile(.95)),seeds=len(values)))
    pd.DataFrame(control_summary).to_csv(out/'control_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,role in zip(axes,['test','validation']):
        ms=['B+P','B+H','B+P+H'];values=[]
        for method in ms:
            values.append(controls[(controls.role==role)&(controls.control=='structure_shift')&(controls.method==method)&(controls.axis=='E-T')].groupby('seed').score.mean().to_numpy()*100)
        ax.boxplot(values,tick_labels=ms)
        for i,method in enumerate(ms,1):
            observed=float(summary[(summary.role==role)&(summary.method==method)&(summary.axis=='E-T')].score.iloc[0])*100
            ax.scatter(i,observed,color='#b44339',marker='D',zorder=3)
        ax.set_title(role+' / structure alignment shift');ax.set_ylabel('Held-out E-T S@5');ax.set_ylim(50,75)
    fig.suptitle('Boxes: 20 fixed circular-shift seeds; red diamonds: actual alignment\nSeed spread is not a population confidence interval',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(out/('03_structure_shift.'+ext),dpi=160)
    plt.close(fig)
    stats=dict(interpretation='No robust additional benefit over fixed-segment responses established: validation and count-balanced controls do not sustain test gains; H closely matches structure-shift control; phrase-conditioned E-S does not beat same-work expectation.',
        primary='E-T',test_families=cohort['query_families']['test'],validation_families=cohort['query_families']['validation'],confidence_intervals=None,
        main_scores=summary.to_dict(orient='records'),controls=control_summary)
    (out/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    p=out/'provenance.json';prov=json.loads(p.read_text())
    for code in (Path(__file__),Path(__file__).with_name('verify_structure_response_comparison.py')):
        prov['code'][str(code.relative_to(ROOT))]=hashlib.sha256(code.read_bytes()).hexdigest()
    prov['results']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='provenance.json'}
    p.write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
