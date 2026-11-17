"""Recover absolute XML beat cells without changing any common-AE input.

A beat_id is recording-local. Cross-recording G references must instead match
the actual [XML quarter start, XML quarter end) used to calculate that R value.
"""
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from .selected_note_data import file_hash,fingerprint
from preprocessing.selected_note_clip import mapped_quarters


G_PROFILE_VERSION='selected-clip-relative-g-v2-absolute-beat-cells'
QUARTER_TICKS=100_000_000


def build_tempo_coordinates(corpus,selected_csv,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/'tempo_cells.npy';ready=directory/'tempo_coordinates.json'
    contract=dict(corpus_id=corpus.manifest['id'],selected_csv_sha256=file_hash(selected_csv),
                  version=G_PROFILE_VERSION,quarter_ticks=QUARTER_TICKS)
    if ready.exists():
        old=json.loads(ready.read_text(encoding='utf-8'))
        if old['contract']!=contract or file_hash(path)!=old['coordinate_sha256']:
            raise ValueError('Absolute tempo-coordinate cache mismatch')
        return np.load(path,mmap_mode='r'),old
    with open(selected_csv,encoding='utf-8-sig') as stream:
        selected={r['perf_id']:r for r in csv.DictReader(stream)}
    clips=corpus.manifest['clips']
    if set(selected)!={c['perf_id'] for c in clips}:raise ValueError('Selected clip identities differ')
    preparation_roots={str(Path(c['cache_path']).parent.parent) for c in clips}
    if len(preparation_roots)!=1:raise ValueError('Expected one verified preparation source')
    preparation=json.loads((Path(next(iter(preparation_roots)))/'clip_preparation_summary.json').read_text(encoding='utf-8'))
    preparation_signature=preparation['signature']
    cells=np.lib.format.open_memmap(path,mode='w+',dtype=np.int64,shape=(corpus.manifest['total_notes'],2))
    cells[:]=-1
    bridges={};alignment_hashes={};sources=[]
    for ci,c in enumerate(clips):
        row=selected[c['perf_id']];bridge_path=Path(c['source_score_cache'])
        key=str(bridge_path)
        if key not in bridges:
            bridge_hash=file_hash(bridge_path)
            bridge_report=json.loads(bridge_path.with_suffix('.json').read_text(encoding='utf-8'))
            if bridge_hash!=bridge_report['cache_sha256']:raise ValueError('Prepared bridge cache changed')
            with bridge_path.open('rb') as stream:xml,score,bridge=pickle.load(stream)
            bridges[key]=dict(sha=bridge_hash,xml=bridge['source_xml_sha256'],midi=bridge['source_midi_sha256'],
                traversal=score.traversal_id,midi_q=np.asarray(bridge['midi_anchor_q']),xml_q=np.asarray(bridge['xml_anchor_q']))
            del xml,score,bridge
        b=bridges[key]
        if (b['xml']!=c['source_xml_sha256'] or b['midi']!=c['score_midi_sha256'] or b['traversal']!=c['traversal_id']):
            raise ValueError('Bridge identity differs from the prepared clip')
        alignment=c['alignment_path']
        if alignment not in alignment_hashes:alignment_hashes[alignment]=file_hash(alignment)
        expected=hashlib.sha256((preparation_signature+row['source_sha256']+b['sha']+
            alignment_hashes[alignment]+row['start_seconds']+row['end_seconds']).encode()).hexdigest()
        if expected!=c['signature']:raise ValueError('Alignment/source/boundary differs from preparation: '+c['perf_id'])
        with np.load(alignment,allow_pickle=False) as f:q=f['score_beats'].copy()
        converted=mapped_quarters(q,b['midi_q'],b['xml_q'])
        section=slice(c['start'],c['stop']);beat=np.asarray(corpus.arrays['beat_id'][section])
        known=np.asarray(corpus.arrays['M'][section,0]);valid=beat>=0
        if np.any(known&~valid) or np.any(beat[valid]>=len(converted)-1):raise ValueError('Invalid prepared tempo index')
        local=np.full((len(beat),2),-1,np.int64)
        local[valid,0]=np.rint(converted[beat[valid]]*QUARTER_TICKS).astype(np.int64)
        local[valid,1]=np.rint(converted[beat[valid]+1]*QUARTER_TICKS).astype(np.int64)
        note_q=np.asarray(corpus.arrays['note_q'][section])
        if (np.any(local[known,1]<=local[known,0]) or
                np.any(note_q[known]<local[known,0]/QUARTER_TICKS-1e-7) or
                np.any(note_q[known]>local[known,1]/QUARTER_TICKS+1e-7)):
            raise ValueError('Observed R tempo does not bind to its XML beat interval: '+c['perf_id'])
        cells[section]=local
        sources.append(dict(perf_id=c['perf_id'],alignment_sha256=alignment_hashes[alignment],bridge_sha256=b['sha'],
            coordinate_grid_sha256=hashlib.sha256(np.asarray(converted,np.float64).tobytes()).hexdigest(),
            observed_tempo_notes=int(known.sum()),local_grid_start=float(converted[0]),local_grid_end=float(converted[-1])))
        if (ci+1)%500==0:print('verified absolute tempo coordinates',ci+1,'/',len(clips),flush=True)
    cells.flush()
    result=dict(contract=contract,coordinate_sha256=file_hash(path),sources=sources,
                source_signatures_verified_against_AE_preparation=True,all_clips=len(clips))
    result['id']=fingerprint(result)
    temporary=ready.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2),encoding='utf-8');temporary.replace(ready)
    return np.load(path,mmap_mode='r'),result
