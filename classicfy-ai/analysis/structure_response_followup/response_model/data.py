"""Train-only scaling of frozen individual response curves and score context."""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1];REPO=WORK/'capstone-2026-2-classicfy';PREVIOUS=WORK/'diagnostics/curve_comparison'
sys.path.insert(0,str(REPO/'classicfy-ai/src'))
from embedding.retrieval_probe import weights
KINDS=('F','P','H0','H1','H2')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def lag_targets(x,tension,valid,metric):
    target=np.zeros(54);mask=np.zeros(54,bool)
    if metric<0:return target,mask
    for channel,c in enumerate((0,2)):
        dt=np.diff(tension[metric]);dx=np.diff(x[c]);pair=valid[metric,:-1]&valid[metric,1:]
        for offset,lag in enumerate(range(-4,5)):
            if lag>=0:a,b,good=dt[:16-lag],dx[lag:],pair[:16-lag]
            else:a,b,good=dt[-lag:],dx[:16+lag],pair[-lag:]
            coordinate=metric*18+channel*9+offset
            if good.sum()>=8 and min(a[good].std(),b[good].std())>1e-10:
                target[coordinate]=np.corrcoef(a[good],b[good])[0,1];mask[coordinate]=True
    return target,mask


def response_targets(x,tension,valid,metric):
    dx=np.diff(x[[0,2]],axis=1);rms=np.sqrt((dx*dx).mean(axis=1))
    supported=rms>1e-6
    shape=np.divide(dx,rms[:,None],out=np.zeros_like(dx),where=supported[:,None])
    direction=np.where(dx>1e-6,2,np.where(dx<-1e-6,0,1))
    peaks=x[[0,2]].argmax(axis=1)
    peak_valid=x[[0,2]].std(axis=1)>1e-6
    lag,lag_mask=lag_targets(x,tension,valid,metric)
    return dict(shape=shape,shape_mask=np.repeat(supported[:,None],16,axis=1),direction=direction,
                peaks=peaks,peak_mask=peak_valid,lag=lag,lag_mask=lag_mask)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    source=PREVIOUS/'curves.npz';base=PREVIOUS/'comparison.npz'
    provenance=json.loads((PREVIOUS/'preparation_verification.json').read_text())
    assert sha(source)==provenance['curves_sha256']
    curves=np.load(source);reference=np.load(base)
    keys=curves['keys'].tolist();ids=reference['eligible']
    with (REPO/'classicfy-ai/analysis/phrase_response/coverage.csv').open() as f:lookup={r['key']:r for r in csv.DictReader(f)}
    metadata=[lookup[key] for key in keys]
    with (REPO/'classicfy-ai/analysis/phrase_response/phrase_hypotheses.csv').open() as f:
        primary={r['piece']:r['part_id'] for r in csv.DictReader(f)}
    lengths={i:curves[f'{i}/tension'].shape[1] for i in ids};key_index={keys[i]:i for i in ids}
    written={i:np.full((lengths[i],2),np.nan) for i in ids}
    map_path=WORK/'datasets/score_structure_full_run01/beat_map.csv.gz'
    assert sha(map_path)==provenance['map_sha256']
    with gzip.open(map_path,'rt') as f:
        for row in csv.DictReader(f):
            if row['key'] in key_index:
                i=key_index[row['key']]
                if row['part_id']==primary[metadata[i]['piece']] and row['written_start_quarters'] and row['written_stop_quarters']:
                    written[i][int(row['index'])]=(float(row['written_start_quarters']),float(row['written_stop_quarters']))
    arrays={};manifests=[]
    for view in (0,1):
        xrows,crows,vrows,durations,owners,types=[],[],[],[],[],[]
        targets={k:[] for k in ('shape','shape_mask','direction','peaks','peak_mask','lag','lag_mask')}
        spans=[]
        for i in ids:
            tension=np.round(curves[f'{i}/tension'],12)
            for ki,kind in enumerate(KINDS):
                values=curves[f'{i}/{view}/{kind}']-curves[f'{i}/{view}/mean'][None,:,None]
                locations=curves[f'{i}/{view}/{kind}_spans']
                for j,(start,stop) in enumerate(locations):
                    positions=written[i][start:stop].mean(axis=1) if kind=='P' else np.arange(start,stop)
                    assert np.isfinite(positions).all() and (np.diff(positions)>0).all()
                    target=np.linspace(positions[0],positions[-1],17)
                    context=np.array([np.interp(target,positions,row[start:stop]) for row in tension])
                    valid=np.isfinite(context)
                    context=np.nan_to_num(context,nan=0.)
                    labels=response_targets(values[j],context,valid,ki-2 if ki>=2 else -1)
                    xrows.append(values[j]);crows.append(context);vrows.append(valid);durations.append(np.log1p(stop-start))
                    owners.append(i);types.append(ki);spans.append((start,stop))
                    for key,value in labels.items():targets[key].append(value)
        prefix=str(view)+'/'
        arrays[prefix+'x']=np.asarray(xrows,np.float32);arrays[prefix+'context']=np.asarray(crows,np.float32)
        arrays[prefix+'context_valid']=np.asarray(vrows,bool);arrays[prefix+'duration']=np.array(durations,np.float32)
        arrays[prefix+'owner']=np.array(owners,int);arrays[prefix+'type']=np.array(types,int);arrays[prefix+'spans']=np.array(spans,int)
        for name,values in targets.items():arrays[prefix+name]=np.asarray(values)
        counts={role:sum(metadata[i]['evaluation_role']==role for i in owners) for role in ('train','validation','test')}
        manifests.append(dict(view=view,curves=len(owners),role_curve_counts=counts,
            lag_supported_coordinates=int(arrays[prefix+'lag_mask'].sum())))
    arrays['keys']=curves['keys'];arrays['eligible']=ids;arrays['gallery_indices']=reference['gallery_indices'];arrays['query_indices']=reference['query_indices']
    np.savez_compressed(OUT/'examples.npz',**arrays)
    manifest=dict(source_curves_sha256=sha(source),source_comparison_sha256=sha(base),map_sha256=sha(map_path),
        examples_sha256=sha(OUT/'examples.npz'),views=manifests,context='Three tension values+three validity channels+position+log score-interval span+five type channels',
        targets='17-point curve differences/positions; H lag units are original beats. These are observed normalized-residual properties, not raw crescendo/ritardando labels.')
    (OUT/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifests),flush=True)


class ViewData:
    def __init__(self,view):
        z=np.load(OUT/'examples.npz');prefix=str(view)+'/'
        self.arrays={k[len(prefix):]:z[k] for k in z.files if k.startswith(prefix)}
        self.keys=z['keys'].tolist();self.ids=z['eligible'];self.gids=z['gallery_indices'];self.qids=z['query_indices']
        with (REPO/'classicfy-ai/analysis/phrase_response/coverage.csv').open() as f:
            lookup={r['key']:r for r in csv.DictReader(f)}
        self.metadata=[lookup[k] for k in self.keys]
        self.roles={role:np.flatnonzero(np.array([self.metadata[i]['evaluation_role'] for i in self.arrays['owner']])==role)
            for role in ('train','validation','test')}
        self.role_weights={role:self.hierarchical_weights(indices) for role,indices in self.roles.items()}
        train=self.roles['train'];w=self.role_weights['train']
        x=self.arrays['x'][train].astype(float)
        xmean=np.einsum('n,nc->c',w,x.mean(axis=2));xsecond=np.einsum('n,nc->c',w,(x*x).mean(axis=2))
        xsd=np.sqrt(np.maximum(xsecond-xmean*xmean,1e-12))
        c=self.arrays['context'][train].astype(float);valid=self.arrays['context_valid'][train]
        count=np.einsum('n,nc->c',w,valid.mean(axis=2))
        cmean=np.einsum('n,nc->c',w,(c*valid).mean(axis=2))/count
        csecond=np.einsum('n,nc->c',w,(c*c*valid).mean(axis=2))/count
        csd=np.sqrt(np.maximum(csecond-cmean*cmean,1e-12))
        dur=self.arrays['duration'][train].astype(float);dm=float(w@dur);ds=float(np.sqrt(w@((dur-dm)**2)))
        lag=self.arrays['lag'][train];lv=self.arrays['lag_mask'][train];lc=w@lv
        lm=np.divide(w@(lag*lv),lc,out=np.zeros(54),where=lc>0)
        second=np.divide(w@(lag*lag*lv),lc,out=np.zeros(54),where=lc>0)
        ls=np.sqrt(np.maximum(second-lm*lm,1e-12))
        self.scales=dict(xmean=xmean.tolist(),xsd=xsd.tolist(),cmean=cmean.tolist(),csd=csd.tolist(),duration_mean=dm,duration_sd=ds,
            lag_mean=lm.tolist(),lag_sd=ls.tolist(),training_owner_indices=sorted(set(self.arrays['owner'][train].tolist())))
        self.x=torch.tensor((self.arrays['x']-xmean[None,:,None])/xsd[None,:,None],dtype=torch.float32)
        scaled=(self.arrays['context']-cmean[None,:,None])/csd[None,:,None]
        scaled=np.where(self.arrays['context_valid'],scaled,0.)
        n=len(self.x);position=np.broadcast_to(np.linspace(0,1,17),(n,1,17))
        duration=np.broadcast_to(((self.arrays['duration']-dm)/ds)[:,None,None],(n,1,17))
        kind=np.broadcast_to(np.eye(5)[self.arrays['type']][:,:,None],(n,5,17))
        self.context=torch.tensor(np.concatenate((scaled,self.arrays['context_valid'],position,duration,kind),axis=1),dtype=torch.float32)
        self.targets={k:torch.tensor(self.arrays[k],dtype=torch.long if k in ('direction','peaks') else torch.bool if k.endswith('mask') else torch.float32)
            for k in ('shape','shape_mask','direction','peaks','peak_mask','lag','lag_mask')}
        self.targets['lag']=torch.tensor(np.where(self.arrays['lag_mask'],(self.arrays['lag']-lm)/ls,0.),dtype=torch.float32)
        assert torch.isfinite(self.x).all() and torch.isfinite(self.context).all()

    def hierarchical_weights(self,indices):
        owners=sorted(set(self.arrays['owner'][indices]));rw=weights([self.metadata[i] for i in owners]);lookup=dict(zip(owners,rw))
        counts={}
        for j in indices:
            key=(self.arrays['owner'][j],self.arrays['type'][j]);counts[key]=counts.get(key,0)+1
        result=np.array([lookup[self.arrays['owner'][j]]/(5*counts[self.arrays['owner'][j],self.arrays['type'][j]]) for j in indices])
        assert abs(result.sum()-1)<1e-10
        return result


if __name__=='__main__':prepare()
