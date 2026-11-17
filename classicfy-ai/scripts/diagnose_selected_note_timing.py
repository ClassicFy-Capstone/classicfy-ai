"""Read-only train/validation diagnostics of frozen note AE timing information."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from train_selected_note_ae import SelectedNoteCorpus, batches, dump, load_model, file_hash

CHANNELS=[1,7,9]
NAMES=['onset','hold_minimum_delay','gap_minimum_delay']


def summary(x,center=0.):
    x=np.asarray(x,dtype=float)
    energy=(x-center)**2
    order=np.argsort(np.abs(x-center))
    return dict(count=len(x),mean=float(x.mean()),std=float(x.std()),
        zero_fraction=float(np.mean(np.abs(x)<1e-8)),
        quantiles=dict(zip(['min','p01','p05','p25','p50','p75','p95','p99','max'],
                          np.quantile(x,[0,.01,.05,.25,.5,.75,.95,.99,1]).tolist())),
        top_1_percent_squared_energy_share=float(energy[order[-max(1,int(.01*len(x))):]].sum()/max(energy.sum(),1e-20)),
        top_5_percent_squared_energy_share=float(energy[order[-max(1,int(.05*len(x))):]].sum()/max(energy.sum(),1e-20)))


def targets(corpus,items):
    values=np.full((len(items),9),np.nan)
    for i,w in enumerate(items):
        r=corpus.arrays['R'][w['start']:w['stop']]
        m=corpus.arrays['M'][w['start']:w['stop']]
        onset=r[m[:,1],1]
        if len(onset)>=8:
            values[i,:3]=[onset.mean(),onset.std(),np.quantile(onset,.9)-np.quantile(onset,.1)]
        for j,ch in enumerate([7,9]):
            x=r[m[:,ch],ch];positive=x[x>1e-8]
            if len(x)>=8:values[i,3+3*j:5+3*j]=[x.mean(),len(positive)/len(x)]
            if len(positive)>=3:values[i,5+3*j]=positive.mean()
    return values


def ridge_probe(train_x,train_y,train_w,val_x,val_y,val_w):
    valid=np.isfinite(train_y)&(train_w>0)
    check=np.isfinite(val_y)&(val_w>0)
    x,y,w=train_x[valid].astype(float),train_y[valid],train_w[valid]
    vx,vy,vw=val_x[check].astype(float),val_y[check],val_w[check]
    mean=np.average(x,axis=0,weights=w)
    scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale=np.maximum(scale,1e-6)
    ym=np.average(y,weights=w)
    x=(x-mean)/scale;vx=(vx-mean)/scale
    coefficients=np.linalg.solve(x.T@(w[:,None]*x)+10*np.eye(x.shape[1]),x.T@(w*(y-ym)))
    pred=vx@coefficients+ym
    denom=np.sum(vw*(vy-np.average(vy,weights=vw))**2)
    return dict(train_windows=int(valid.sum()),validation_windows=int(check.sum()),
        validation_r2=float(1-np.sum(vw*(pred-vy)**2)/denom) if denom>1e-15 else None)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    corpus=SelectedNoteCorpus(args.run/'corpus')
    model,state=load_model(args.run/'evaluated_best.pt');scaler=state['scaler']
    train_rows=np.zeros(corpus.manifest['total_notes'],bool)
    val_rows=np.zeros_like(train_rows)
    for r in corpus.manifest['clips']:
        if r['split']=='train':train_rows[r['start']:r['stop']]=True
        if r['split']=='validation':val_rows[r['start']:r['stop']]=True
    distributions={}
    for rows,label in [(train_rows,'train'),(val_rows,'validation')]:
        distributions[label]={}
        for ch,name in zip(CHANNELS,NAMES):
            x=corpus.arrays['R'][rows&corpus.arrays['M'][:,ch],ch].astype(float)
            d=summary(x,scaler['R_mean'][ch]);d['train_scale']=scaler['R_scale'][ch]
            d['scaled_p90_minus_p10']=float(np.subtract(*np.quantile(x,[.9,.1]))/scaler['R_scale'][ch])
            if ch!=1:d['positive_delay_summary']=summary(x[x>1e-8])
            distributions[label][name]=d
    dump(args.out/'distributions.json',distributions)
    encoded={};validation_prediction=[];validation_truth=[]
    began=time.time()
    for split in ['train','validation']:
        ordered=[w for group in batches(corpus.items(split),64) for w in group]
        zs,cs=[],[]
        for index,group in enumerate(batches(ordered,64),1):
            b=corpus.collate(group,scaler);context=model.context(b)
            z=model.ae.encode(context,b['R'],b['M'],b['P'],b['lengths'])['z']
            zs.append(z.numpy());cs.append((context.sum(1)/b['lengths'][:,None]).numpy())
            if split=='validation':
                pred=model.ae.decode(context,z,b['P'],b['lengths'],b['beat_id'])['R_hat'].numpy()
                truth=b['R'].numpy();mask=b['M'].numpy()
                for ch in CHANNELS:
                    validation_prediction.append((ch,pred[:,:,ch][mask[:,:,ch]]))
                    validation_truth.append((ch,truth[:,:,ch][mask[:,:,ch]]))
            if index%50==0:print(f'{split} {index} batches, {time.time()-began:.1f}s',flush=True)
        encoded[split]=dict(z=np.concatenate(zs),c=np.concatenate(cs),y=targets(corpus,ordered),
            weights=np.array([w['clip_weight'] for w in ordered]))
    target_names=['onset_mean','onset_std','onset_p90_p10',
        'hold_delay_mean','hold_positive_fraction','hold_positive_mean',
        'gap_delay_mean','gap_positive_fraction','gap_positive_mean']
    probes={}
    tr,va=encoded['train'],encoded['validation']
    for k,label in enumerate(target_names):
        probes[label]={}
        for feature in ['c','z','c_plus_z']:
            tx=tr[feature] if feature!='c_plus_z' else np.c_[tr['c'],tr['z']]
            vx=va[feature] if feature!='c_plus_z' else np.c_[va['c'],va['z']]
            probes[label][feature]=ridge_probe(tx,tr['y'][:,k],tr['weights'],vx,va['y'][:,k],va['weights'])
    predictions={}
    for ch,name in zip(CHANNELS,NAMES):
        y=np.concatenate([x for c,x in validation_truth if c==ch]).astype(float)
        pred=np.concatenate([x for c,x in validation_prediction if c==ch]).astype(float)
        raw=y*scaler['R_scale'][ch]+scaler['R_mean'][ch]
        predictions[name]=dict(count=len(y),r2=float(1-np.sum((pred-y)**2)/np.sum((y-y.mean())**2)),
            prediction_std_over_target_std=float(pred.std()/y.std()),
            correlation=float(np.corrcoef(y,pred)[0,1]),
            raw_mae_quarters=float(np.mean(np.abs(pred-y))*scaler['R_scale'][ch]))
        if ch!=1:
            positive=raw>1e-7
            for label,keep in [('zero',~positive),('positive',positive)]:
                predictions[name][label]=dict(count=int(keep.sum()),
                    mae_quarters=float(np.mean(np.abs(pred[keep]-y[keep]))*scaler['R_scale'][ch]),
                    target_mean_quarters=float(raw[keep].mean()),
                    prediction_mean_quarters=float((pred[keep]*scaler['R_scale'][ch]+scaler['R_mean'][ch]).mean()))
    result=dict(checkpoint_sha256=file_hash(args.run/'evaluated_best.pt'),
        scope='Frozen train-only AE, train-fit probes, validation diagnostics only. No weight updates or test inference.',
        probe='Weighted ridge alpha10, train-only feature scaling, clip-weighted validation R2; six statistical targets plus onset statistics. Not a causal or perceptual-style test.',
        distributions=distributions,decoder=predictions,frozen_latent_probes=probes,seconds=time.time()-began)
    dump(args.out/'diagnosis.json',result)
    np.savez_compressed(args.out/'frozen_probe_features.npz',**{f'{s}_{k}':v for s,r in encoded.items() for k,v in r.items()})
    print(json.dumps(dict(decoder=predictions,probes=probes),indent=2),flush=True)


if __name__=='__main__':main()
