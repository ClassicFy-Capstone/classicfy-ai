"""Interpretable structure responses and disjoint-beat numeric judges."""

import numpy as np

from features.structure_response import response,split_view,fixed_labels
from .retrieval_probe import summary,pearson


def view_descriptors(x,mask,pairs,phase,owner,tension,tension_bin,key_index,view,shift_seed=None):
    valid,pair,blocks=split_view(mask,pairs,view)
    phase=np.asarray(phase).copy();owner=np.asarray(owner).copy();bins=np.asarray(tension_bin).copy()
    if shift_seed is not None:
        rng=np.random.default_rng(shift_seed)
        for start in blocks:
            offset=int(rng.integers(1,64));span=slice(start,start+64)
            phase[span]=np.roll(phase[span],offset);owner[span]=np.roll(owner[span],offset)
            bins[:,span]=np.roll(bins[:,span],offset,axis=1)
    fixed,fixed_owner=fixed_labels(len(mask))
    p,_=response(x,valid,pair,phase,owner)
    f,_=response(x,valid,pair,fixed,fixed_owner)
    hs=[]
    for c in range(3):
        mp=pair.copy();mp[1:]&=np.isfinite(tension[c,:-1])&np.isfinite(tension[c,1:])
        if c==2:mp[1:]&=key_index[:-1]==key_index[1:]
        h,_=response(x,valid&np.isfinite(tension[c]),mp,bins[c])
        hs.append(h)
    observed=np.diff(x)[:,pair[1:]]
    time=np.r_[observed.mean(axis=1),np.sqrt((observed*observed).mean(axis=1)),np.mean(observed[0]*observed[2])] if observed.shape[1]>=4 else np.full(15,np.nan)
    structure=[]
    for label in range(3):structure.append(np.mean(phase[valid]==label) if valid.any() else np.nan)
    for c in range(3):
        values=tension[c,valid&np.isfinite(tension[c])]
        structure.extend([np.mean(values),np.std(values)] if len(values) else [np.nan,np.nan])
    return dict(B=summary(x,valid),T=time,F=f,P=p,H=np.concatenate(hs),S=np.asarray(structure),
                Q=np.array([np.log1p(len(mask)),float(mask.mean())]))


def heldout_judges(x,mask,pairs,phase,owner,view):
    valid,pair,blocks=split_view(mask,pairs,view)
    level=[];temporal=[]
    for c in range(7):
        v=x[c,valid]
        level.extend([np.median(abs(v)) if c==1 else np.median(v),np.ptp(np.percentile(v,[25,75]))] if len(v)>=100 else [np.nan,np.nan])
        rms=[]
        for start in blocks:
            for offset in (0,32):
                a=start+offset;good=pair[a+1:a+32]
                if good.sum()>=25:rms.append(np.sqrt(np.mean(np.diff(x[c,a:a+32])[good]**2)))
        temporal.extend(np.log1p(np.percentile(rms,[25,50,75])) if len(rms)>=3 else [np.nan]*3)
    rs=[]
    for start in blocks:
        good=pair[start+1:start+64]
        r=pearson(np.diff(x[0,start:start+64])[good],np.diff(x[2,start:start+64])[good],51)
        if np.isfinite(r):rs.append(r)
    relation=np.r_[np.percentile(rs,[25,50,75]),np.std(rs)] if len(rs)>=2 else np.full(4,np.nan)
    # Separate summary coordinates: positive-change fractions and joint sign.
    conditioned=[]
    for label in range(3):
        good=pair.copy();good[0]=False
        good[1:]&=(phase[:-1]==label)&(phase[1:]==label)&(owner[:-1]==owner[1:])
        d=np.diff(x)[:,good[1:]]
        if d.shape[1]>=4:conditioned.extend([np.mean(d[0]>1e-6),np.mean(d[2]>1e-6),np.mean(d[0]*d[2]>1e-12)])
        else:conditioned.extend([np.nan]*3)
    return {'E-L':np.asarray(level),'E-T':np.asarray(temporal),'E-R':relation,'E-S':np.asarray(conditioned)}
