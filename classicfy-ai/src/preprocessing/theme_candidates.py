"""Label-free, multi-scale symbolic thumbnail candidates, not a form classifier.

All times are performance seconds. The detector never reads annotations, titles,
composer identity, or confidence calibration data. Scores are heuristic evidence,
not probabilities that a passage is a musicological theme.
"""
from dataclasses import asdict, dataclass

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class ThemeConfig:
    max_frames: int = 600
    frame_seconds: float = .75
    min_seconds: float = 10.
    max_seconds: float = 120.
    top_k: int = 5
    max_start_fraction: float = .35


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def features(notes, config=ThemeConfig()):
    """Onset chroma plus upper/lower onset envelopes; no synthesized audio."""
    a = np.asarray(notes, dtype=float)
    if a.ndim != 2 or a.shape[1] != 4 or len(a) < 16:
        raise ValueError('too_few_valid_notes')
    a = a[np.isfinite(a).all(1) & (a[:,2]>a[:,1]) & (a[:,0]>=0) & (a[:,0]<128)]
    if len(a) < 16:
        raise ValueError('too_few_valid_notes')
    a = a[np.argsort(a[:,1],kind='stable')]
    first, last = float(a[:,1].min()), float(a[:,2].max())
    duration = last-first
    if duration < 3:
        raise ValueError('performance_shorter_than_three_seconds')
    step = max(config.frame_seconds, duration/config.max_frames)
    n = max(4,int(np.ceil(duration/step)))
    bins = np.minimum(n-1,((a[:,1]-first)/step).astype(int))
    pc = a[:,0].astype(int)%12
    onset = np.zeros((n,12),np.float32)
    np.add.at(onset,(bins,pc),1.)
    # Approximate melodic/bass envelopes by grouping near-simultaneous onsets.
    groups = np.round((a[:,1]-first)/.04).astype(np.int64)
    _, idx = np.unique(groups,return_index=True)
    upper, lower = np.zeros_like(onset),np.zeros_like(onset)
    group_bins=np.minimum(n-1,((a[idx,1]-first)/step).astype(int))
    np.add.at(upper,(group_bins,np.maximum.reduceat(a[:,0],idx).astype(int)%12),1.)
    np.add.at(lower,(group_bins,np.minimum.reduceat(a[:,0],idx).astype(int)%12),1.)
    channels=[gaussian_filter1d(x,.8,axis=0) for x in (onset,upper,lower)]
    # Downweight pitch classes persistent throughout the movement (ostinato).
    channels=[unit(np.maximum(x-.35*x.mean(0),0)) for x in channels]
    x=np.concatenate([channels[0]*.55,channels[1]*.75,channels[2]*.35],axis=1)
    x=unit(x).astype(np.float32)
    density=gaussian_filter1d(onset.sum(1),1.)
    return a,x,first,last,step,density


@njit(cache=True)
def diagonal_prefix(s):
    """Prefix sums on each diagonal, so variable window comparisons are O(1)."""
    n=len(s)
    result=np.zeros((n+1,n+1),np.float32)
    for i in range(n):
        for j in range(n):
            result[i+1,j+1]=s[i,j]+result[i,j]
    return result


def propose(notes, config=ThemeConfig()):
    a,x,first,last,step,density=features(notes,config)
    n=len(x)
    # Novelty is a change in harmonic/melodic texture, not a cadence label.
    smooth=gaussian_filter1d(x,2.,axis=0)
    radius=max(1,round(3/step))
    novelty=np.linalg.norm(smooth[np.minimum(np.arange(n)+radius,n-1)]-
                           smooth[np.maximum(np.arange(n)-radius,0)],axis=1)
    # Local thinning/space is a weak phrase-boundary cue.
    thinning=np.maximum(0,gaussian_filter1d(density,5)-density)
    if thinning.max()>0: thinning/=thinning.max()
    novelty=novelty/(np.quantile(novelty,.95)+1e-8)
    boundary=np.clip(.8*novelty+.2*thinning,0,1)
    peaks,_=find_peaks(boundary,distance=max(2,round(4/step)),prominence=.04)
    edges=np.unique(np.r_[0,peaks,n-1]).astype(int)
    starts=edges[edges<=min(n*config.max_start_fraction,180/step)]
    if len(starts)>20:
        starts=np.unique(np.r_[0,sorted(starts[1:],key=lambda i:-boundary[i])[:19]])
    minimum=min(config.min_seconds,(last-first)*.45)
    maximum=min(config.max_seconds,(last-first)*.6)
    windows=[]
    for start in starts:
        stops=list(edges[(edges-start)*step>=minimum])
        # A fixed duration candidate is still available if novelty is flat.
        stops+= [min(n-1,start+round(v/step)) for v in (15,24,32,45,60,90,120)]
        for end in sorted(set(stops)):
            length=(end-start)*step
            if minimum<=length<=maximum and end>start+2:
                windows.append((int(start),int(end)))
    if not windows:
        windows=[(0,max(2,min(n-1,round(30/step))))]
    repeat=np.zeros(len(windows));repeat_at=np.full(len(windows),-1)
    window_array=np.asarray(windows,dtype=int)
    starts_array=window_array[:,0];ends_array=window_array[:,1]
    widths=ends_array-starts_array
    js=np.arange(n)[None,:]
    targets=np.minimum(n,js+widths[:,None])
    valid=(js<n-widths[:,None]) & (np.abs(js-starts_array[:,None])>=np.maximum(widths,round(8/step))[:,None])
    # A single transposition is held fixed over each whole diagonal window.
    # Maximizing cellwise would spuriously splice unrelated harmonic paths.
    for shift in range(12):
        y=np.concatenate([np.roll(x[:,k:k+12],shift,axis=1) for k in (0,12,24)],axis=1)
        s=x@y.T
        prefix=diagonal_prefix(s)
        values=(prefix[ends_array[:,None],targets]-prefix[starts_array[:,None],js])/widths[:,None]
        values=np.where(valid,values,-1.)
        best=np.argmax(values,axis=1)
        scores=values[np.arange(len(windows)),best]
        improve=scores>repeat
        repeat[improve]=scores[improve];repeat_at[improve]=best[improve]
    candidates=[]
    global_variation=float(np.mean(np.std(x,axis=0)))
    for k,(start,end) in enumerate(windows):
        seconds=(end-start)*step
        early=np.exp(-start*step/45.)
        duration_prior=np.exp(-.5*(np.log(max(seconds,1)/32.)/.7)**2)
        end_strength=float(boundary[min(end,n-1)])
        start_strength=1. if start==0 else float(boundary[start])
        # Require changing melodic content to avoid a repeated static chord.
        variation=float(np.mean(np.std(x[start:end],axis=0)))
        specificity=float(np.clip(variation/max(global_variation,.04),0,1))
        evidence=float(np.clip((repeat[k]-.30)/.65,0,1))*specificity
        score=.42*evidence+.18*end_strength+.08*start_strength+.17*early+.15*duration_prior
        confidence=float(np.clip(.55*evidence+.25*end_strength+.20*specificity,0,1))
        start_time=first+start*step
        end_time=min(last,first+end*step)
        # Snap only to nearby actual note onsets, not arbitrary silence frames.
        onset=a[:,1]
        if start:
            near=onset[np.abs(onset-start_time)<=min(step,1.5)]
            if len(near):start_time=float(near[np.argmin(np.abs(near-start_time))])
        near=onset[np.abs(onset-end_time)<=min(step,1.5)]
        if len(near):end_time=float(near[np.argmin(np.abs(near-end_time))])
        if end_time<=start_time:continue
        descriptor=unit(x[start:end].mean(0)[None,:])[0]
        candidates.append(dict(start_seconds=float(start_time),end_seconds=float(end_time),
                               ranking_score=float(score),heuristic_confidence=confidence,
                               recurrence=float(repeat[k]),repeat_start_seconds=float(first+repeat_at[k]*step) if repeat_at[k]>=0 else None,
                               boundary_strength=end_strength,melodic_specificity=specificity,
                               fingerprint=descriptor.tolist(),first_note_seconds=first,last_note_seconds=last,
                               relative_start=float((start_time-first)/(last-first)),
                               frame_resolution_seconds=step,
                               relative_duration=float((end_time-start_time)/(last-first))))
    # Diversity suppression retains alternative phrases, not five nearly equal windows.
    selected=[]
    for c in sorted(candidates,key=lambda c:-c['ranking_score']):
        if all(interval_iou(c['start_seconds'],c['end_seconds'],p['start_seconds'],p['end_seconds'])<.75 for p in selected):
            selected.append(c)
            if len(selected)==config.top_k:break
    if not selected:raise ValueError('no_nonempty_candidate')
    return selected


def interval_iou(a,b,c,d):
    return max(0.,min(b,d)-max(a,c))/max(max(b,d)-min(a,c),1e-12)


def consensus(records):
    """Unlabelled within-composition agreement; never accesses reference labels.

    It is intentionally limited to candidate selection, not claimed as exact
    note alignment. Tempo/repeat differences can still select different phrases.
    """
    valid=[r for r in records if r.get('candidates')]
    output={}
    for r in valid:
        ranked=[]
        for rank,c in enumerate(r['candidates']):
            votes=[]
            for other in valid:
                if other['perf_id']==r['perf_id']:continue
                best=0.
                for oc in other['candidates'][:3]:
                    time=np.exp(-abs(c['relative_start']-oc['relative_start'])/.06-
                                abs(c['relative_duration']-oc['relative_duration'])/.10)
                    similarity=max(0.,float(np.dot(c['fingerprint'],oc['fingerprint'])))
                    best=max(best,time*similarity)
                votes.append(best)
            agreement=float(np.median(votes)) if votes else 0.
            score=c['ranking_score']+.10*agreement
            ranked.append((score,rank,agreement))
        _,rank,agreement=max(ranked)
        output[r['perf_id']]=dict(candidate_index=rank,composition_agreement=agreement,
                                  available_other_performances=len(valid)-1)
    return output
