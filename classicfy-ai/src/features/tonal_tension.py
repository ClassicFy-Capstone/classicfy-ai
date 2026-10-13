"""Spelling-sensitive Spiral Array tension with explicit key uncertainty.

Mathematical conventions follow Partitura's documented Spiral Array model.
Clouds use strictly positive written-duration overlap in a half-open interval.
No performed timing, velocity or retrieval result enters the calculation.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.distance import pdist

FIFTHS = dict(F=-1,C=0,G=1,D=2,A=3,E=4,B=5)
WEIGHTS = np.array([516.,315.,168.])/999.
HEIGHT = math.sqrt(2/15)*(math.pi/2)**2
SCALE = 1/(12*HEIGHT)  # C to B-sharp, twelve fifth steps
MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])


def fifth_point(index):
    t=index*math.pi/2
    return np.array([math.sin(t),math.cos(t),index*HEIGHT])


def pitch_point(step, alter):
    if step not in FIFTHS or int(alter)!=alter:
        raise ValueError('Unsupported pitch spelling or fractional alteration')
    return fifth_point(FIFTHS[step]+7*int(alter))


def chord_center(tonic, mode):
    if mode not in ('major','minor'):
        raise ValueError('Expected explicit major/minor hypothesis')
    third=4 if mode=='major' else -3
    return WEIGHTS @ np.array([fifth_point(tonic),fifth_point(tonic+1),fifth_point(tonic+third)])


def key_center(fifths, mode):
    if mode=='major':
        tonic=fifths
        chords=[chord_center(tonic,'major'),chord_center(tonic+1,'major'),chord_center(tonic-1,'major')]
    elif mode=='minor':
        tonic=fifths+3
        chords=[chord_center(tonic,'minor'),
            .75*chord_center(tonic+1,'major')+.25*chord_center(tonic+1,'minor'),
            .75*chord_center(tonic-1,'minor')+.25*chord_center(tonic-1,'major')]
    else:
        raise ValueError('Mode must be explicit; never default unknown mode to major')
    return WEIGHTS @ np.array(chords)


@dataclass(frozen=True)
class KeyContext:
    start: float
    stop: float
    fifths: int | None
    mode: str | None
    source: str
    status: str
    major_correlation: float | None
    minor_correlation: float | None
    margin: float | None


class TonalContext:
    """Precompute score-only key hypotheses and clouds for written intervals."""

    def __init__(self, score, mode_margin=.1):
        if not 0<=mode_margin<=2:
            raise ValueError('Invalid mode correlation margin')
        self.notes=[n for n in score.notes if n.pitch is not None and n.duration_quarters>0 and not n.is_cue]
        self.onsets=np.array([float(n.onset_quarters) for n in self.notes])
        self.offsets=np.array([float(n.offset_quarters) for n in self.notes])
        self.supported=np.array([n.pitch.alter.denominator==1 for n in self.notes])
        self.points=np.array([pitch_point(n.pitch.step,n.pitch.alter) if good else np.full(3,np.nan)
            for n,good in zip(self.notes,self.supported)])
        self.contexts=[];self.cache={}
        begin=min(float(m.onset_quarters) for m in score.measures)
        end=max(float(m.onset_quarters+m.duration_quarters) for m in score.measures)
        positions=sorted({begin,end}|{float(k.onset_quarters) for k in score.key_signatures if begin<float(k.onset_quarters)<end})
        for a,b in zip(positions,positions[1:]):
            current={}
            for key in score.key_signatures:
                if float(key.onset_quarters)<=a:current[key.part_id,key.staff]=key
            signatures={(k.fifths,k.mode) for k in current.values() if not k.alterations}
            if not current or any(k.alterations or k.fifths is None for k in current.values()) or len(signatures)!=1:
                self.contexts.append(KeyContext(a,b,None,None,'musicxml','missing_or_conflicting_signature',None,None,None));continue
            fifths,mode=next(iter(signatures))
            if mode in ('major','minor'):
                self.contexts.append(KeyContext(a,b,fifths,mode,'explicit_musicxml','explicit_mode',None,None,None));continue
            if mode is not None:
                self.contexts.append(KeyContext(a,b,fifths,None,'musicxml','unsupported_mode',None,None,None));continue
            hist=np.zeros(12)
            weights=np.maximum(0,np.minimum(b,self.offsets)-np.maximum(a,self.onsets))
            for n,w in zip(self.notes,weights):
                if n.pitch.midi_pitch.denominator==1:hist[int(n.pitch.midi_pitch)%12]+=w
            root=(7*fifths)%12
            if hist.sum()<=0 or hist.std()<=1e-12:
                self.contexts.append(KeyContext(a,b,fifths,None,'score_profile','insufficient_pitch_histogram',None,None,None));continue
            major=float(np.corrcoef(hist,np.roll(MAJOR_PROFILE,root))[0,1])
            minor=float(np.corrcoef(hist,np.roll(MINOR_PROFILE,(root+9)%12))[0,1])
            margin=abs(major-minor)
            guessed='major' if major>minor else 'minor'
            self.contexts.append(KeyContext(a,b,fifths,guessed if margin>=mode_margin else None,
                'signature_restricted_duration_profile','estimated_mode' if margin>=mode_margin else 'ambiguous_mode',major,minor,margin))

    def cloud(self, start, stop):
        if not math.isfinite(start) or not math.isfinite(stop) or stop<=start:
            return dict(center=None,diameter=math.nan,strain=math.nan,strain_major=math.nan,strain_minor=math.nan,
                status='invalid_interval',key_index=-1,notes=0,weight=0.)
        identity=(start,stop)
        if identity in self.cache:return self.cache[identity]
        weights=np.maximum(0,np.minimum(stop,self.offsets)-np.maximum(start,self.onsets))
        active=weights>0
        context=next((i for i,k in enumerate(self.contexts) if k.start<=start and stop<=k.stop+1e-10),-1)
        result=dict(center=None,diameter=math.nan,strain=math.nan,strain_major=math.nan,strain_minor=math.nan,
                    status='silent',key_index=context,notes=int(active.sum()),weight=float(weights.sum()))
        if active.any() and self.supported[active].all():
            points=self.points[active];weight=weights[active]
            center=(points*weight[:,None]).sum(axis=0)/weight.sum()
            result.update(center=center,diameter=(float(pdist(points).max()) if len(points)>1 else 0.)*SCALE,status='score_cloud')
            if context>=0:
                key=self.contexts[context]
                if key.fifths is not None:
                    for mode in ('major','minor'):
                        result['strain_'+mode]=float(np.linalg.norm(center-key_center(key.fifths,mode))*SCALE)
                    if key.mode is not None:result['strain']=result['strain_'+key.mode]
        elif active.any():result['status']='unsupported_microtone'
        self.cache[identity]=result
        return result

    def project(self, written, cuts):
        if len(written)!=len(cuts):raise ValueError('Written interval / cut length differs')
        values=np.full((3,len(written)),np.nan);support=[];previous=None
        for i,((a,b),cut) in enumerate(zip(written,cuts)):
            row=self.cloud(float(a),float(b));center=row['center']
            if center is not None:
                values[0,i]=row['diameter'];values[2,i]=row['strain']
                if previous is not None and not cut:values[1,i]=np.linalg.norm(center-previous)*SCALE
            previous=center
            support.append({k:v for k,v in row.items() if k!='center'})
        return values,support


def tension_labels(values, valid):
    """Score-only tertiles; ties stay together, unsupported bins remain absent."""
    values=np.asarray(values,float);valid=np.asarray(valid,bool)&np.isfinite(values)
    labels=np.full(len(values),-1,int)
    if valid.any():
        lo,hi=np.quantile(values[valid],[1/3,2/3])
        labels[valid]=np.where(values[valid]<=lo,0,np.where(values[valid]<=hi,1,2))
        return labels,(float(lo),float(hi))
    return labels,(math.nan,math.nan)
