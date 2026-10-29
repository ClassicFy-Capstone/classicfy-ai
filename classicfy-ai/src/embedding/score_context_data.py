"""Portable symbolic-score/performance joins and recording-preserving batches.

Export and loading use the existing preprocessing dependencies, without numba.
PyTorch is imported by the collator.
Neither quarter coordinates nor missing performance positions are compacted.
"""

from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from features.score_context import (SCHEMA_VERSION,SCORE_CHANNELS,ScoreContextScale,
    encode_score_features,fit_score_context_scale,load_score_context,score_context_schema,
    validate_quarter_edges)

NATIVE_CHANNELS = ('tempo','rubato','dynamics','articulation','pedal_depth','pedal_down_ratio','pedal_changes')
DATASET_VERSION = 'conditional-score-input-v1'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def dump_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def _read_map(path, root):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        records = list(csv.DictReader(stream))
    by_key = {};work_splits = defaultdict(set);score_splits = defaultdict(set)
    for row in records:
        if any(not row.get(k) for k in ('key','work','score_path','score_sha256')):
            raise ValueError('Score map needs key, work, score_path, score_sha256')
        key = row['key']
        if key in by_key:
            raise ValueError('Duplicate recording in score map: '+key)
        digest = row['score_sha256']
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Expected lowercase full score SHA256')
        split = row.get('split') or 'unassigned'
        if split not in ('train','validation','test','unassigned'):
            raise ValueError('Unknown split: '+split)
        score_path = Path(row['score_path'])
        if not score_path.is_absolute():
            score_path = root/score_path
        if not score_path.is_file() or sha256(score_path)!=digest:
            raise ValueError('Missing or hash-mismatched score: '+str(score_path))
        by_key[key] = dict(row,split=split,resolved_score_path=score_path)
        work_splits[row['work']].add(split);score_splits[digest].add(split)
    if not by_key:
        raise ValueError('Empty score map')
    if any(len(v)>1 for v in [*work_splits.values(),*score_splits.values()]):
        raise ValueError('A canonical work or identical score crosses split assignments')
    return by_key


def _performance_records(cache):
    """Two explicit existing formats: ATEPP channel cache or portable quarter NPZ."""
    manifest = json.loads(str(cache['manifest']))
    if isinstance(manifest,list):
        seen = set()
        for i,record in enumerate(manifest):
            key = record['key']
            if key in seen:
                raise ValueError('Duplicate quarter recording: '+key)
            seen.add(key)
            yield record,cache[f'{i}/q'],cache[f'{i}/values'].T,cache[f'{i}/valid'].T,None
    elif isinstance(manifest,dict) and isinstance(manifest.get('records'),list):
        provenance = manifest.get('provenance',{})
        if provenance.get('dataset')!='ATEPP-1.2':
            raise ValueError('Channel cache must identify ATEPP-1.2')
        if provenance.get('aligner')!='dual_dtw':
            raise ValueError('Expected the quality-accepted DualDTW source cache')
        source_rows = provenance.get('source_files',[])
        sources = {r['key']:r for r in source_rows}
        if len(sources)!=len(source_rows):
            raise ValueError('Duplicate recording in source provenance')
        channels = defaultdict(dict)
        for i,row in enumerate(manifest['records']):
            if row['channel'] not in NATIVE_CHANNELS or row['channel'] in channels[row['key']]:
                raise ValueError('Unknown or duplicate performance channel')
            channels[row['key']][row['channel']] = (i,row)
        for key,records in sorted(channels.items()):
            if set(records)!=set(NATIVE_CHANNELS) or key not in sources:
                raise ValueError('Incomplete channels or source provenance: '+key)
            first_i,first = records[NATIVE_CHANNELS[0]]
            q = cache[f'{first_i}/score_beats'];values=[];masks=[]
            for channel in NATIVE_CHANNELS:
                i,row = records[channel]
                if row['piece']!=first['piece'] or not np.array_equal(cache[f'{i}/score_beats'],q):
                    raise ValueError('Channel score grids/identity disagree: '+key)
                values.append(cache[f'{i}/standardized_values']);masks.append(cache[f'{i}/standardized_mask'])
            yield dict(key=key,channels=list(NATIVE_CHANNELS),group=first['piece']),q,np.array(values).T,np.array(masks).T,sources[key]['score_sha256']
    else:
        raise ValueError('Unsupported source performance-cache format')


def export_score_context(performance_cache, score_map, out, *, dataset, score_root=None,
                         fit_scale=False, scale_path=None):
    """Join an explicitly identified source score to every original R interval.

    Existing performance values/normalization are preserved; only masked entries
    are cleaned to zero. Fitting the optional score scaler requires declared
    train splits. Existing performance references may still be corpus-fitted.
    """
    performance_cache,score_map,out = map(Path,(performance_cache,score_map,out))
    root = Path(score_root) if score_root is not None else score_map.resolve().parent
    if dataset not in ('ATEPP','ASAP') or (fit_scale and scale_path is not None):
        raise ValueError('Choose a source dataset and either fit or load score scale')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty')
    mapping = _read_map(score_map,root)
    features={};record_rows=[];arrays={};channels=None;seen=set();score_rows=[]
    with np.load(performance_cache,allow_pickle=False) as cache:
        for record,q,values,mask,source_digest in _performance_records(cache):
            key = record['key']
            if key not in mapping:
                raise ValueError('Recording absent from score map: '+key)
            seen.add(key);mapped=mapping[key]
            if source_digest is not None and mapped['score_sha256']!=source_digest:
                raise ValueError('Score does not match original alignment provenance: '+key)
            if source_digest is not None and dataset!='ATEPP':
                raise ValueError('ATEPP source cache cannot be relabeled')
            q = validate_quarter_edges(q)
            if not np.allclose(np.diff(q),1.,rtol=0,atol=1e-8):
                raise ValueError('The first adapter requires one-quarter source intervals')
            row_channels = record['channels']
            if not row_channels or len(set(row_channels))!=len(row_channels):
                raise ValueError('Expected named unique performance channels')
            if channels is None:channels=list(row_channels)
            if row_channels!=channels or values.shape!=(len(q)-1,len(channels)) or mask.shape!=values.shape or mask.dtype!=np.bool_:
                raise ValueError('Performance channel/shape/mask mismatch: '+key)
            if not np.isfinite(values[mask]).all():
                raise ValueError('Nonfinite valid performance observation: '+key)
            identity=hashlib.sha256(SCHEMA_VERSION.encode()+mapped['score_sha256'].encode()+np.asarray(q,dtype='<f8').tobytes()).hexdigest()
            if identity not in features:
                features[identity]=load_score_context(mapped['resolved_score_path'],q)
            feature=features[identity]
            if feature.source_sha256!=mapped['score_sha256']:
                raise ValueError('Source changed while exporting')
            i=len(record_rows)
            cleaned=np.where(mask,values,0).astype(np.float32)
            if not np.isfinite(cleaned).all():raise ValueError('Performance overflows float32')
            arrays[f'records/{i}/performance']=cleaned
            arrays[f'records/{i}/observation_mask']=mask.copy()
            valid_rows=mask.any(axis=1)&feature.valid
            record_rows.append(dict(key=key,work=mapped['work'],split=mapped['split'],score_ref=identity,
                source_score_sha256=feature.source_sha256,original_intervals=len(q)-1,
                score_intervals=int(feature.valid.sum()),usable_intervals=int(valid_rows.sum()),
                observed_feature_values=int(mask.sum()),score_supported_observations=int(mask[feature.valid].sum()),
                source_valid_performance_sha256=hashlib.sha256(np.asarray(values[mask],dtype='<f8').tobytes()+mask.tobytes()).hexdigest(),
                exported_valid_performance_sha256=hashlib.sha256(np.asarray(cleaned[mask],dtype='<f4').tobytes()+mask.tobytes()).hexdigest()))
    if seen!=set(mapping):
        raise ValueError('Score map contains recordings absent from performance cache')
    scale = None
    if fit_scale:
        train_refs={r['score_ref'] for r in record_rows if r['split']=='train'}
        scale=fit_score_context_scale([features[ref] for ref in sorted(train_refs)])
    elif scale_path is not None:
        scale=ScoreContextScale.from_dict(json.loads(Path(scale_path).read_text()))
        forbidden={r['source_score_sha256'] for r in record_rows if r['split'] in ('validation','test')}
        if forbidden&set(scale.training_score_hashes):
            raise ValueError('Loaded score scale was fitted on an evaluation score')
    for ref,feature in sorted(features.items()):
        values=scale.transform(feature) if scale else encode_score_features(feature.raw,feature.valid)
        values=values.astype(np.float32)
        if not np.isfinite(values).all():raise ValueError('Score features overflow float32')
        arrays[f'scores/{ref}/q']=feature.q
        arrays[f'scores/{ref}/raw']=feature.raw
        arrays[f'scores/{ref}/score']=values
        arrays[f'scores/{ref}/valid']=feature.valid
        score_rows.append(dict(score_ref=ref,score_sha256=feature.source_sha256,
            intervals=len(feature.valid),valid_intervals=int(feature.valid.sum()),diagnostics=feature.diagnostics))
    manifest=dict(version=DATASET_VERSION,dataset=dataset,dataset_label_source='explicit caller declaration; source hashes retained',
        score_schema=score_context_schema(),performance_channels=channels,records=record_rows,scores=score_rows,
        score_scale=scale.to_dict() if scale else None,
        performance_policy='Original valid standardized values retained to float32; no reference/scale refit; masked entries cleaned to 0. Source normalization may use full catalog.',
        source_inputs=dict(performance_cache_sha256=sha256(performance_cache),score_map_sha256=sha256(score_map)),
        segmentation='Not fitted/chosen by exporter. Caller supplies maximum segment length; unsupported score cuts the sequence, missing R does not compact time.')
    arrays['manifest']=np.array(json.dumps(manifest,ensure_ascii=False,allow_nan=False))
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'dataset.npz',**arrays)
    dump_json(out/'manifest.json',manifest)
    if scale:dump_json(out/'score_scale.json',scale.to_dict())
    dump_json(out/'integrity.json',dict(dataset_sha256=sha256(out/'dataset.npz'),manifest_sha256=sha256(out/'manifest.json'),
        score_scale_sha256=sha256(out/'score_scale.json') if scale else None))
    return manifest


@dataclass(frozen=True)
class ScoreContextSegment:
    key: str
    q: np.ndarray
    score: np.ndarray
    performance: np.ndarray
    observation_mask: np.ndarray


def read_score_context_segments(directory, *, max_segment_length):
    """Return recording-owned segments; all-empty recordings remain empty lists."""
    directory=Path(directory)
    if isinstance(max_segment_length,bool) or not isinstance(max_segment_length,int) or max_segment_length<1:
        raise ValueError('Specify a positive maximum segment length')
    integrity=json.loads((directory/'integrity.json').read_text())
    if sha256(directory/'dataset.npz')!=integrity['dataset_sha256'] or sha256(directory/'manifest.json')!=integrity['manifest_sha256']:
        raise ValueError('Export integrity mismatch')
    if integrity['score_scale_sha256'] is not None and sha256(directory/'score_scale.json')!=integrity['score_scale_sha256']:
        raise ValueError('Score scale integrity mismatch')
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['version']!=DATASET_VERSION or manifest['score_schema']!=score_context_schema():
        raise ValueError('Unsupported dataset/schema')
    result=[]
    with np.load(directory/'dataset.npz',allow_pickle=False) as z:
        if json.loads(str(z['manifest']))!=manifest:raise ValueError('Embedded manifest differs')
        for i,row in enumerate(manifest['records']):
            ref=row['score_ref'];q=z[f'scores/{ref}/q'];score=z[f'scores/{ref}/score'];valid=z[f'scores/{ref}/valid']
            perf=z[f'records/{i}/performance'];mask=z[f'records/{i}/observation_mask'];segments=[]
            edges=np.diff(np.r_[False,valid,False].astype(int))
            for start,stop in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                for a in range(int(start),int(stop),max_segment_length):
                    b=min(a+max_segment_length,int(stop))
                    if mask[a:b].any():
                        segments.append(ScoreContextSegment(row['key'],q[a:b+1].copy(),score[a:b].copy(),perf[a:b].copy(),mask[a:b].copy()))
            result.append(segments)
    return manifest,result


def collate_score_context(recordings):
    """Pad [recording][segment] without flattening away recording ownership."""
    import torch
    if not recordings or any(not segments for segments in recordings):
        raise ValueError('Each batch recording needs at least one observed segment')
    first=recordings[0][0]
    if first.score.ndim!=2 or first.performance.ndim!=2 or min(first.score.shape[1],first.performance.shape[1])<1:
        raise ValueError('Score/performance segment features must be nonempty matrices')
    ds=first.score.shape[1];dr=first.performance.shape[1]
    b=len(recordings);k=max(map(len,recordings));t=max(len(s.score) for ss in recordings for s in ss)
    score=np.zeros((b,k,t,ds),np.float32);performance=np.zeros((b,k,t,dr),np.float32)
    observed=np.zeros((b,k,t,dr),bool);time=np.zeros((b,k,t),bool);segment=np.zeros((b,k),bool)
    q_start=np.full((b,k,t),np.nan);q_stop=q_start.copy();keys=[]
    for i,segments in enumerate(recordings):
        if len({s.key for s in segments})!=1:raise ValueError('A recording contains segments belonging to different recordings')
        keys.append(segments[0].key)
        for j,s in enumerate(segments):
            n=len(s.score);q=validate_quarter_edges(s.q)
            if (n<1 or s.score.shape!=(n,ds) or s.performance.shape!=(n,dr) or s.observation_mask.shape!=(n,dr)
                    or s.observation_mask.dtype!=np.bool_ or len(q)!=n+1 or not s.observation_mask.any()
                    or not np.isfinite(s.score).all() or not np.isfinite(s.performance[s.observation_mask]).all()):
                raise ValueError('Invalid segment shape, coordinates, observations or features')
            score[i,j,:n]=s.score;performance[i,j,:n]=np.where(s.observation_mask,s.performance,0)
            observed[i,j,:n]=s.observation_mask;time[i,j,:n]=True;segment[i,j]=True
            q_start[i,j,:n]=q[:-1];q_stop[i,j,:n]=q[1:]
    if not np.isfinite(score).all() or not np.isfinite(performance).all():raise ValueError('Batch overflows float32')
    return dict(score=torch.from_numpy(score),performance=torch.from_numpy(performance),
        observation_mask=torch.from_numpy(observed),time_mask=torch.from_numpy(time),segment_mask=torch.from_numpy(segment),
        keys=keys,q_start=q_start,q_stop=q_stop)
