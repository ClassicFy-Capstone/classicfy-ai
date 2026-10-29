"""Show symbolic score context and observed residuals at identical quarters."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from embedding.score_context_data import sha256,dump_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/score_context_validation')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/score_context_validation_run02')
    args=parser.parse_args()
    with np.load(args.artifacts/'joined/dataset.npz',allow_pickle=False) as z:
        manifest=json.loads(str(z['manifest']));selected=[];composers=set()
        for i,row in enumerate(manifest['records']):
            composer=row['key'].split('/')[0]
            if composer not in composers:selected.append((i,row));composers.add(composer)
            if len(selected)==3:break
        fig,axes=plt.subplots(len(selected),2,figsize=(13,8),layout='constrained')
        for (i,row),(left,right) in zip(selected,np.atleast_2d(axes)):
            ref=row['score_ref'];q=z[f'scores/{ref}/q'];raw=z[f'scores/{ref}/raw'];n=min(32,len(raw))
            left.imshow(raw[:n,12:24].T,origin='lower',aspect='auto',interpolation='nearest',extent=[q[0],q[n],-.5,11.5],cmap='Blues')
            left.set(yticks=np.arange(12),yticklabels=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'],xlabel='Score quarter',ylabel='Pitch class',title='S: written overlap density')
            performance=z[f'records/{i}/performance'];mask=z[f'records/{i}/observation_mask']
            for c,color in [(0,'#347cb0'),(2,'#be5c37')]:
                values=np.where(mask[:n,c],performance[:n,c],np.nan)
                right.step(q[:n+1],np.r_[values,values[-1]],where='post',color=color,label=manifest['performance_channels'][c])
            right.set(xlabel='Same score quarter',ylabel='Original normalized residual',title=row['key'],xlim=(q[0],q[n]));right.legend(fontsize=8)
        fig.suptitle('Available ASAP source: deterministic score S aligned to observed R\nFirst 32 quarters of first recording per first three composers; no learned C displayed',fontsize=12)
        for ext in ('png','pdf'):fig.savefig(args.out/('01_score_and_performance.'+ext),dpi=160)
        plt.close(fig)
    path=args.out/'provenance.json';provenance=json.loads(path.read_text())
    provenance['code'][str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
    provenance['results'].update({p.name:sha256(p) for p in args.out.iterdir() if p.suffix in ('.png','.pdf')})
    dump_json(path,provenance)


if __name__=='__main__':main()
