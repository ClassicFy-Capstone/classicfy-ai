"""Plot provisional phrase context and original residuals; no selection by effect."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from validate_score_beat_bridge import rows,sha

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/phrase_response')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/phrase_response_run02')
    args=parser.parse_args();coverage=rows(args.out/'coverage.csv')
    z=np.load(args.artifacts/'sequences.npz',allow_pickle=False)
    pilots=['Bach/Fugue/bwv_848/midi_score.mid','Beethoven/Piano_Sonatas/31-2/midi_score.mid','Mozart/Piano_Sonatas/8-1/midi_score.mid']
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    for ax,piece in zip(axes,pilots):
        i=next(i for i,r in enumerate(coverage) if r['piece']==piece)
        key=coverage[i]['key'];x=z[f'{i}/values'];m=z[f'{i}/mask'];labels=z[f'{i}/phase'];owners=z[f'{i}/owner'];cuts=z[f'{i}/breaks']
        limit=min(200,len(m));index=np.arange(limit)
        for channel,color in [(0,'#356eb5'),(2,'#b44b35')]:
            y=np.where(m[:limit],x[channel,:limit],np.nan)
            y[cuts[:limit]]=np.nan
            ax.plot(index,y,color=color,lw=1.1,label=['Tempo','Dynamics'][channel//2])
        for j in range(limit):
            if m[j] and labels[j]>=0:ax.axvspan(j,j+1,color=['#6fa5db','#bbbbbb','#e5a173'][labels[j]],alpha=.12,lw=0)
            if j and (owners[j]!=owners[j-1] or cuts[j]):ax.axvline(j,color='#555555',lw=.5,ls='--')
        ax.set_title(key,fontsize=9);ax.set_xlabel('Original beat interval index');ax.set_ylabel('Normalized residual');ax.legend(loc='upper right')
    fig.suptitle('Provisional score phrase phases: start / middle / end\nDashed: phrase or structural cut; first 200 intervals',fontsize=12)
    for suffix in ('png','pdf'):fig.savefig(args.out/('01_phrase_response.'+suffix),dpi=160)
    plt.close(fig)
    sensitivity=rows(args.out/'boundary_sensitivity.csv')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    conditions=['shift_-0.5','shift_0.5','exclude_crossing']
    data=[[float(r['response_rms']) for r in sensitivity if r['condition']==c and r['response_rms']!='nan'] for c in conditions]
    ax.boxplot(data,tick_labels=['-0.5 quarter','+0.5 quarter','Exclude crossing'],showfliers=False)
    ax.set(ylabel='RMS change of supported response coordinates',title='Boundary sensitivity across 570 recordings')
    for suffix in ('png','pdf'):fig.savefig(args.out/('02_boundary_sensitivity.'+suffix),dpi=160)
    plt.close(fig)
    provenance=json.loads((args.out/'provenance.json').read_text())
    for p in [Path(__file__),Path(__file__).with_name('verify_phrase_response.py')]:provenance['code'][str(p.relative_to(ROOT))]=sha(p)
    provenance['results'].update({p.name:sha(p) for p in args.out.iterdir() if p.suffix in ('.png','.pdf') or p.name=='verification.json'})
    (args.out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
