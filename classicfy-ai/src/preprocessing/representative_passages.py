"""Work-level representative passages; all scores are unvalidated proxies.

Repetition: non-overlapping subsequence-DTW paths and coverage (not the exact
optimal FMP path-family algorithm). Centrality: ordered content descriptors.
Neither score is a guarantee of thematic importance or perceptual identity.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit
from .theme_candidates import ThemeConfig, features, interval_iou, unit


@dataclass(frozen=True)
class PassageConfig:
    max_frames: int = 300
    lengths: tuple = (15., 25., 35.)
    stride_seconds: float = 6.
    top_k: int = 4
    minimum_representation_ratio: float = .8
    minimum_alignment_similarity: float = .65


@njit(cache=True)
def subsequence_match(similarity):
    """Query to reference; steps (1,1),(1,2),(2,1), free reference ends."""
    n,m=similarity.shape
    d=np.full((n+1,m+1),np.inf);d[0,:]=0.
    parent=np.zeros((n+1,m+1),np.uint8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=1.-similarity[i-1,j-1]
            best=d[i-1,j-1]+cost;direction=1
            if j>=2 and d[i-1,j-2]+cost+.03<best:
                best=d[i-1,j-2]+cost+.03;direction=2
            if i>=2 and d[i-2,j-1]+2*cost+.03<best:
                best=d[i-2,j-1]+2*cost+.03;direction=3
            d[i,j]=best;parent[i,j]=direction
    starts=np.zeros(m,np.int64)
    for end in range(1,m+1):
        i=n;j=end
        while i>0 and j>0:
            p=parent[i,j]
            if p==1:i-=1;j-=1
            elif p==2:i-=1;j-=2
            else:i-=2;j-=1
        starts[end-1]=max(j,0)
    return np.maximum(0.,1.-d[n,1:]/n),starts


def content(notes,config):
    return features(notes,ThemeConfig(max_frames=config.max_frames,frame_seconds=.75))


def snap_endpoints(a,start,end,first,last):
    """High-resolution onset/gap refinement, no cadence claim."""
    onsets=np.unique(a[:,1]);gaps=np.r_[0.,np.diff(onsets)]
    result=[]
    for t in (start,end):
        ix=np.flatnonzero(np.abs(onsets-t)<=1.)
        if len(ix):
            score=np.minimum(gaps[ix],.5)-.2*np.abs(onsets[ix]-t)
            t=float(onsets[ix[np.argmax(score)]])
        result.append(float(np.clip(t,first,last)))
    return result


def propose_passages(notes,config=PassageConfig()):
    a,x,first,last,step,density=content(notes,config)
    n=len(x);windows=[];descriptors=[];information=[]
    for seconds in config.lengths:
        width=max(3,min(n,int(round(min(seconds,(last-first)*.8)/step))))
        for start in range(0,n-width+1,max(1,round(config.stride_seconds/step))):
            end=start+width
            if (start,end) in windows:continue
            windows.append((start,end))
            chunks=np.array_split(x[start:end],4)
            descriptors.append(np.concatenate([c.mean(0) if len(c) else np.zeros(x.shape[1]) for c in chunks]))
            information.append(float(np.mean(np.std(x[start:end],axis=0))))
    if not windows:raise ValueError('no_valid_window')
    z=unit(np.asarray(descriptors));s=z@z.T
    # Exclude overlapping windows: nearby copies must not inflate centrality.
    centrality=[]
    for i,(a0,b0) in enumerate(windows):
        valid=[j for j,(c,d) in enumerate(windows) if b0<=c or d<=a0]
        centrality.append(float(np.mean(s[i,valid])) if valid else 0.)
    recurrence=np.zeros(len(windows));coverage=np.zeros(len(windows))
    for shift in range(12):
        shifted=np.concatenate([np.roll(x[:,k:k+12],shift,axis=1) for k in (0,12,24)],axis=1)
        sim=x@shifted.T
        for i,(start,end) in enumerate(windows):
            values,starts=subsequence_match(sim[start:end])
            chosen=[];union=np.zeros(n,bool)
            for j in np.argsort(-values):
                lo=int(starts[j]);hi=int(j+1)
                if values[j]<.72:break
                if hi>start and lo<end:continue
                if any(hi>c and lo<d for c,d in chosen):continue
                chosen.append((lo,hi));union[lo:hi]=True
                if len(chosen)==4:break
            if chosen:
                quality=max(float(values[hi-1]) for lo,hi in chosen)
                cov=float(union.mean());fitness=2*quality*cov/max(quality+cov,1e-8)
                if fitness>recurrence[i]:recurrence[i]=fitness;coverage[i]=cov
    candidates=[]
    for i,(start,end) in enumerate(windows):
        variability=min(1.,information[i]/.07)
        # Explicit engineering defaults, requiring ablation/listening validation.
        score=(.5*centrality[i]+.5*recurrence[i])*variability
        ta,tb=snap_endpoints(a,first+start*step,min(last,first+end*step),first,last)
        if tb-ta<2:continue
        candidates.append(dict(start_seconds=ta,end_seconds=tb,representation_score=float(score),
              content_centrality=centrality[i],repetition_fitness=float(recurrence[i]),
              repetition_coverage=float(coverage[i]),content_variation=information[i],
              content_descriptor=z[i].tolist(),frame_resolution_seconds=step,
              boundary_type='onset_gap_refinement_not_cadence',identity_verified=False))
    candidates.sort(key=lambda c:-c['representation_score'])
    selected=[];best=candidates[0]['representation_score'] if candidates else 0.
    for c in candidates:
        if c['representation_score']<best*config.minimum_representation_ratio:continue
        if any(interval_iou(c['start_seconds'],c['end_seconds'],p['start_seconds'],p['end_seconds'])>.5
               or np.dot(c['content_descriptor'],p['content_descriptor'])>.995 for p in selected):continue
        selected.append(c)
        if len(selected)==config.top_k:break
    if not selected:raise ValueError('no_nonempty_passage')
    return selected


@njit(cache=True)
def global_path(sim):
    n,m=sim.shape;d=np.full((n+1,m+1),np.inf);d[0,0]=0.
    parent=np.zeros((n+1,m+1),np.uint8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            v=d[i-1,j-1];p=1
            if d[i-1,j]+.12<v:v=d[i-1,j]+.12;p=2
            if d[i,j-1]+.12<v:v=d[i,j-1]+.12;p=3
            d[i,j]=v+1.-sim[i-1,j-1];parent[i,j]=p
    i=n;j=m;path=[]
    while i>0 and j>0:
        path.append((i-1,j-1));p=parent[i,j]
        if p==1:i-=1;j-=1
        elif p==2:i-=1
        else:j-=1
    return path[::-1]


def map_passages(reference,target,candidates,config=PassageConfig()):
    _,rx,rf,rl,rs,_=content(reference,config)
    a,tx,tf,tl,ts,_=content(target,config)
    sim=rx@tx.T;path=global_path(sim.astype(np.float32))
    groups={}
    for i,j in path:groups.setdefault(i,[]).append(j)
    iq=np.array(sorted(groups));jq=np.array([np.median(groups[i]) for i in iq])
    output=[]
    for index,c in enumerate(candidates):
        qa=(c['start_seconds']-rf)/rs;qb=(c['end_seconds']-rf)/rs
        ja,jb=np.interp([qa,qb],iq,jq)
        region=[(i,j) for i,j in path if qa<=i<=qb]
        similarity=float(np.mean([sim[i,j] for i,j in region])) if region else 0.
        slope=(jb-ja)/max(qb-qa,1e-8)*ts/rs
        # Repeats/omissions are not safely resolved by forced global warping:
        # reject implausibly stretched or collapsed local regions instead.
        accepted=similarity>=config.minimum_alignment_similarity and .4<=slope<=2.5
        start,end=snap_endpoints(a,tf+ja*ts,min(tl,tf+jb*ts),tf,tl)
        accepted=bool(accepted and end-start>=2.)
        output.append(dict(candidate_index=index,start_seconds=start,end_seconds=end,
               alignment_similarity=similarity,local_time_ratio=float(slope),alignment_accepted=accepted,
               mapping_method='content_DTW_approximate',mapping_is_ground_truth=False))
    return output


def style_proxy(notes,start,end,reference_duration,whole_notes):
    a=np.asarray(notes);v=a[(a[:,1]>=start)&(a[:,1]<end)]
    if len(v)<8:return None
    whole=np.asarray(whole_notes)
    velocity=(v[:,3]-np.median(whole[:,3]))/max(np.std(whole[:,3]),5.)
    # Fixed scales, not candidate-wise standardization. Relative durations and
    # spread are descriptive proxies, not matched-note articulation ground truth.
    return np.array([np.log((end-start)/reference_duration)/.2,
                     np.quantile(velocity,.8)-np.quantile(velocity,.2),
                     np.std(np.log(np.maximum(v[:,2]-v[:,1],.01)))])


def choose_passage(candidates,vectors):
    """Quality-filtered median pair distance, leave-one-performance-out stability."""
    differences=[];supports=[]
    for index in range(len(candidates)):
        valid=[np.asarray(v[index]) for v in vectors if v[index] is not None]
        supports.append(len(valid))
        distances=[np.mean(np.abs(a-b)) for i,a in enumerate(valid) for b in valid[i+1:]]
        differences.append(float(np.median(distances)) if len(valid)>=3 else None)
    eligible=[i for i,v in enumerate(differences) if v is not None]
    selected=max(eligible,key=lambda i:differences[i]) if eligible else 0
    votes=[];leave_out={i:[] for i in eligible}
    if eligible:
        for omitted in range(len(vectors)):
            local=[]
            for index in eligible:
                a=[np.asarray(v[index]) for j,v in enumerate(vectors) if j!=omitted and v[index] is not None]
                ds=[np.mean(np.abs(x-y)) for j,x in enumerate(a) for y in a[j+1:]]
                value=float(np.median(ds)) if len(a)>=3 else -1.
                local.append(value)
                if value>=0:leave_out[index].append(value)
            if max(local)>=0:votes.append(eligible[int(np.argmax(local))])
    stable_scores=[None]*len(candidates)
    for index in eligible:
        stable_scores[index]=min(differences[index],float(np.quantile(leave_out[index],.2))) if leave_out[index] else differences[index]
    if eligible:selected=max(eligible,key=lambda i:stable_scores[i])
    stability=float(np.mean(np.array(votes)==selected)) if votes else None
    return dict(selected_candidate=selected,style_difference=differences,stable_style_score=stable_scores,supports=supports,
                selection_stability=stability,style_selection_available=bool(eligible),
                selection_method='robust_style_proxy' if eligible else 'representativeness_fallback',
                identity_verified=False,style_embedding_trained=False)
