"""Deterministic, versioned score features on the performance's quarter grid."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from preprocessing.score_midi import read_score_midi

SCHEMA_VERSION = 'score-context-v1'
PITCH_CLASSES = ('C','Cs','D','Ds','E','F','Fs','G','Gs','A','As','B')
SCORE_CHANNELS = (tuple('onset_'+p for p in PITCH_CLASSES)
    + tuple('overlap_'+p for p in PITCH_CLASSES)
    + ('onset_count','pitch_mean','pitch_min','pitch_max','onset_duration_mean','silent'))
LOG_COLUMNS = tuple(range(25)) + (28,)


def score_context_schema():
    return dict(version=SCHEMA_VERSION, channels=list(SCORE_CHANNELS), dimension=len(SCORE_CHANNELS),
        coordinates='quarter-note score positions; half-open intervals',
        onset='pitch-class counts; onset_count is their sum',
        overlap='sum of positive written-duration overlap / interval quarter width; polyphony can exceed 1',
        register='overlap-duration-weighted mean and min/max MIDI pitch of overlapping notes',
        duration='mean full duration of notes beginning in the interval; zero if no onsets',
        silence='1 if no positive pitched overlap, else 0; silent bins are observed score data',
        transform='log1p columns 0:25 and 28; pitch columns 25:28 divided by 127; silence unchanged',
        unsupported='outside encoded score extent is masked; unclosed notes reject extraction',
        midi_policy='FIFO note pairing; remove exact pitch/onset/offset duplicates; zero-length notes reported and omitted; drums omitted; no inferred XML tie/grace/voice labels')


def validate_quarter_edges(q):
    source = np.asarray(q)
    if not (np.issubdtype(source.dtype,np.integer) or np.issubdtype(source.dtype,np.floating)):
        raise ValueError('Quarter edges must be numeric real coordinates')
    q = np.asarray(source, dtype=np.float64)
    if q.ndim != 1 or len(q) < 2 or not np.isfinite(q).all() or q[0] < 0 or np.any(np.diff(q) <= 0):
        raise ValueError('Expected finite increasing nonnegative quarter edges')
    return q


@dataclass(frozen=True)
class ScoreContextFeatures:
    q: np.ndarray
    raw: np.ndarray
    valid: np.ndarray
    source_sha256: str
    diagnostics: dict

    @property
    def cache_key(self):
        payload = SCHEMA_VERSION.encode() + self.source_sha256.encode() + np.asarray(self.q,dtype='<f8').tobytes()
        return hashlib.sha256(payload).hexdigest()


def extract_score_context(notes, q, *, score_stop, source_sha256='', diagnostics=None):
    """Use score notes only; retain every requested interval and its coverage mask."""
    q = validate_quarter_edges(q)
    if not np.isfinite(score_stop) or score_stop <= 0:
        raise ValueError('Expected a positive encoded score extent')
    diagnostics = dict(diagnostics or {})
    if diagnostics.get('unclosed_notes',0):
        raise ValueError('Unclosed score notes have unknown durations')
    triples = []
    for note in notes:
        p,a,b = note.pitch,note.onset_beats,note.offset_beats
        if (isinstance(p,(bool,np.bool_)) or int(p)!=p or not 0<=p<=127
                or not np.isfinite([a,b]).all() or a<0 or b<=a or b>score_stop+1e-10):
            raise ValueError('Invalid symbolic score note')
        triples.append((int(p),float(a),float(b)))
    # The public note-based API follows the MIDI loader's exact duplicate policy.
    triples = sorted(set(triples), key=lambda n:(n[1],n[0],n[2]))
    pitches = np.array([n[0] for n in triples],int)
    starts = np.array([n[1] for n in triples],float)
    stops = np.array([n[2] for n in triples],float)
    valid = q[1:] <= score_stop+1e-10
    raw = np.full((len(q)-1,len(SCORE_CHANNELS)),np.nan)
    for i,(a,b) in enumerate(zip(q,q[1:])):
        if not valid[i]:
            continue
        row = np.zeros(len(SCORE_CHANNELS))
        onset = (starts>=a)&(starts<b)
        lengths = np.maximum(0,np.minimum(b,stops)-np.maximum(a,starts))
        active = lengths>0
        row[:12] = np.bincount(pitches[onset]%12,minlength=12)
        row[12:24] = np.bincount(pitches%12,weights=lengths,minlength=12)/(b-a)
        row[24] = onset.sum()
        if active.any():
            row[25:28] = [np.average(pitches[active],weights=lengths[active]),pitches[active].min(),pitches[active].max()]
        if onset.any():
            row[28] = (stops[onset]-starts[onset]).mean()
        row[29] = not active.any()
        raw[i] = row
    return ScoreContextFeatures(q.copy(),raw,valid,source_sha256,diagnostics)


def load_score_context(path, q):
    path = Path(path)
    notes,stop,diagnostics = read_score_midi(path)
    return extract_score_context(notes,q,score_stop=stop,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),diagnostics=diagnostics)


def encode_score_features(raw, valid):
    """Fixed transforms, then zero only unsupported rows; silence stays explicit."""
    raw = np.asarray(raw,float);valid = np.asarray(valid)
    if valid.dtype!=np.bool_ or raw.shape!=(len(valid),len(SCORE_CHANNELS)):
        raise ValueError('Expected [T,30] score features and a boolean [T] mask')
    if not np.isfinite(raw[valid]).all() or np.any(raw[valid]<0):
        raise ValueError('Valid raw score features must be finite and nonnegative')
    result = np.zeros_like(raw)
    result[valid] = raw[valid]
    result[:,LOG_COLUMNS] = np.log1p(result[:,LOG_COLUMNS])
    result[:,25:28] /= 127.
    return result


@dataclass(frozen=True)
class ScoreContextScale:
    mean: tuple[float,...]
    scale: tuple[float,...]
    training_score_hashes: tuple[str,...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if (self.schema_version!=SCHEMA_VERSION or len(self.mean)!=len(SCORE_CHANNELS)
                or len(self.scale)!=len(SCORE_CHANNELS) or not np.isfinite(self.mean).all()
                or not np.isfinite(self.scale).all() or min(self.scale)<=0
                or self.mean[29]!=0 or self.scale[29]!=1 or not self.training_score_hashes):
            raise ValueError('Invalid score-context scale/schema')

    def transform(self, features):
        values = encode_score_features(features.raw,features.valid)
        values[features.valid] = (values[features.valid]-np.asarray(self.mean))/np.asarray(self.scale)
        return values

    def to_dict(self):
        return dict(schema_version=self.schema_version,mean=list(self.mean),scale=list(self.scale),
            training_score_hashes=list(self.training_score_hashes),
            weighting='equal score weight, equal distinct quarter interval weight within score; silence unscaled')

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(value['mean']),tuple(value['scale']),tuple(value['training_score_hashes']),value['schema_version'])


def fit_score_context_scale(training_features):
    """Caller selects training records; repeated performances/grids add no weight."""
    groups = {}
    for features in training_features:
        if not features.source_sha256:
            raise ValueError('Training score identity is required')
        rows = groups.setdefault(features.source_sha256,{})
        values = encode_score_features(features.raw,features.valid)
        for i in np.flatnonzero(features.valid):
            coordinate = tuple(float(v) for v in features.q[i:i+2])
            if coordinate in rows and not np.array_equal(rows[coordinate],values[i]):
                raise ValueError('Same score/interval has conflicting symbolic features')
            rows[coordinate] = values[i]
    groups = {k:v for k,v in groups.items() if v}
    if not groups:
        raise ValueError('No supported training score intervals')
    arrays = [np.array([v[q] for q in sorted(v)]) for _,v in sorted(groups.items())]
    mean = np.mean([a.mean(axis=0) for a in arrays],axis=0)
    variance = np.mean([np.mean((a-mean)**2,axis=0) for a in arrays],axis=0)
    scale = np.sqrt(variance);scale[scale<=1e-8] = 1.
    mean[29] = 0.;scale[29] = 1.
    return ScoreContextScale(tuple(mean),tuple(scale),tuple(sorted(groups)))
