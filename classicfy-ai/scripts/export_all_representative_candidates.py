"""Export every saved v2 candidate for every performance, without selecting one.

Uses frozen work candidates/mappings; no inference, re-ranking or annotation use.
Uncertain mappings with valid intervals are retained under midi_review/.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import warnings

import mido
import numpy as np
import pandas as pd
import pretty_midi

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
from extract_atepp_themes import clip_midi, read_sequence


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def validate_clip(path, original, start, end, expected_count):
    """Independent MIDI note decode, plus closed-note/pedal and duration checks."""
    counts=Counter();pedal={};time=0.;onsets=0
    for msg in mido.MidiFile(path):
        time+=msg.time
        if msg.type=='note_on' and msg.velocity:
            counts[msg.channel,msg.note]+=1;onsets+=1
        elif msg.type in ('note_on','note_off'):
            counts[msg.channel,msg.note]-=1
            if counts[msg.channel,msg.note]<0:raise ValueError('orphan_note_off')
        elif msg.type=='control_change' and msg.control==64:pedal[msg.channel]=msg.value
    if any(counts.values()) or any(v>=64 for v in pedal.values()):raise ValueError('hanging_note_or_pedal')
    if abs(time-(end-start))>.002 or onsets!=expected_count:raise ValueError('duration_or_count_mismatch')
    expected=sorted((i.program,i.is_drum,n.pitch,n.velocity,round(max(n.start,start)-start,3),round(min(n.end,end)-start,3))
        for i in original.instruments for n in i.notes if n.start<end-1e-9 and n.end>start+1e-9
        and round((min(n.end,end)-start)*1000)>round((max(n.start,start)-start)*1000))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');clipped=pretty_midi.PrettyMIDI(str(path))
    actual=sorted((i.program,i.is_drum,n.pitch,n.velocity,round(n.start,3),round(n.end,3))
                  for i in clipped.instruments for n in i.notes)
    same=len(expected)==len(actual)
    if same and expected:
        e=np.asarray(expected,dtype=float);a=np.asarray(actual,dtype=float)
        same=bool(np.array_equal(e[:,:4],a[:,:4]) and np.allclose(e[:,4:],a[:,4:],atol=.0011,rtol=0))
    if not same:raise ValueError(f'independent_note_mismatch:{len(expected)}:{len(actual)}')
    return len(actual)


def export_performance(source, work, out):
    """One row per work candidate, including missing/rejected/unexportable maps."""
    out=Path(out);pid=source['perf_id'];wid=source['composition_id']
    maps={m['candidate_index']:m for m in work.get('mappings',{}).get(pid,[])}
    rows=[];sequence=None;original=None
    digest=sha(source['local_midi']) if Path(source['local_midi']).is_file() else ''
    source_ok=digest==source['sha256'] and bool(digest)
    load_error=''
    for index,c in enumerate(work['candidates']):
        mapping=maps.get(index)
        row=dict(perf_id=pid,composition_id=wid,reference_perf_id=work['reference_perf_id'],
            candidate_index=index,candidate_number=index+1,artist=source.get('artist',''),
            composer=source.get('composer',''),track=source.get('track',''),source_quality=source.get('quality',''),
            source_sha256=digest,source_hash_matches_audit=source_ok,
            reference_start_seconds=c['start_seconds'],reference_end_seconds=c['end_seconds'],
            representation_score=c['representation_score'],confidence_level='unvalidated',
            requires_review=True,identity_verified=False,cadence_status='not_inferred',
            automatic_selection_applied=False,clip_relative_path='',export_validation='not_run')
        if mapping is None:
            row.update(status='mapping_missing',error='no_saved_mapping_for_candidate');rows.append(row);continue
        row.update({k:v for k,v in mapping.items() if k!='candidate_index'})
        start=float(mapping['start_seconds']);end=float(mapping['end_seconds'])
        row['duration_seconds']=end-start
        row['review_reasons']='perceptual_representativeness_not_validated'
        if not mapping['alignment_accepted']:row['review_reasons']+=';mapping_uncertain'
        if end-start>60:row['review_reasons']+=';longer_than_60s'
        if source.get('quality','') not in ('','high quality'):row['review_reasons']+=';source_quality_flag'
        if not source_ok:
            row.update(status='source_mismatch',error='source_hash_differs_from_frozen_audit');rows.append(row);continue
        if not np.isfinite([start,end]).all() or start<0 or end<=start:
            row.update(status='invalid_interval',error='nonpositive_or_invalid_mapped_interval');rows.append(row);continue
        folder='midi' if mapping['alignment_accepted'] else 'midi_review'
        relative=Path(folder)/wid/pid/f'candidate_{index+1:02d}.mid';path=out/relative
        try:
            if load_error:raise ValueError(load_error)
            if sequence is None:
                try:
                    sequence=read_sequence(source['local_midi'])
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore');original=pretty_midi.PrettyMIDI(source['local_midi'])
                except Exception as exc:
                    load_error=f'source_decode:{exc}';raise
            n=clip_midi(source['local_midi'],path,start,end,sequence=sequence)
            checked=validate_clip(path,original,start,end,n)
            row.update(status='exported' if mapping['alignment_accepted'] else 'exported_review',
                clip_relative_path=relative.as_posix(),note_count=n,independently_checked_notes=checked,
                export_validation='passed',clip_sha256=sha(path))
        except Exception as exc:
            row.update(status='export_failed',error=f'{type(exc).__name__}: {exc}',export_validation='failed')
            if path.exists():
                quarantine=out/'quarantine'/wid/pid/path.name
                quarantine.parent.mkdir(parents=True,exist_ok=True);path.replace(quarantine)
        rows.append(row)
    return rows


def export_work(task):
    sources,input_root,destination,signature=task
    out=Path(destination);wid=sources[0]['composition_id'];input_path=Path(input_root)/'works'/f'{wid}.json'
    work=json.loads(input_path.read_text(encoding='utf-8'));input_hash=sha(input_path)
    checkpoint=out/'works'/f'{wid}.json'
    if checkpoint.exists():
        prior=json.loads(checkpoint.read_text(encoding='utf-8'))
        if prior.get('signature')==signature and prior.get('input_work_sha256')==input_hash:
            files_ok=all(r['status'] not in ('export_failed','source_mismatch') and
                (not r['clip_relative_path'] or ((out/r['clip_relative_path']).is_file() and
                 sha(out/r['clip_relative_path'])==r['clip_sha256'])) for r in prior['segments'])
            sources_ok=all(sha(r['local_midi'])==r['sha256'] for r in sources)
            if files_ok and sources_ok:return prior
    rows=[]
    for source in sources:rows.extend(export_performance(source,work,out))
    result=dict(work=wid,reference_perf_id=work['reference_perf_id'],candidates=work['candidates'],
                mappings=work['mappings'],segments=rows,signature=signature,input_work_sha256=input_hash,
                automatic_selection_applied=False)
    dump(checkpoint,result)
    return result


def main():
    workspace=HERE.parents[2];parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=workspace/'outputs/atepp_representative_v2_full')
    parser.add_argument('--out',type=Path,default=workspace/'outputs/atepp_representative_v2_all_candidates')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--work-ids',help='Optional comma-separated work IDs for smoke checks')
    args=parser.parse_args();input_root=args.input.resolve();out=args.out.resolve()
    if out==input_root:parser.error('Output must differ from frozen selected-only input')
    out.mkdir(parents=True,exist_ok=True)
    audit=pd.read_csv(input_root/'input_midi_audit.csv',dtype={'perf_id':str,'composition_id':str}).fillna('')
    if args.work_ids:audit=audit[audit.composition_id.isin(args.work_ids.split(','))]
    if not len(audit):parser.error('No matching performances')
    signature=hashlib.sha256(Path(__file__).read_bytes()+(HERE/'extract_atepp_themes.py').read_bytes()).hexdigest()
    groups=list(audit.groupby('composition_id',sort=True));results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(export_work,(g.to_dict('records'),str(input_root),str(out),signature)):wid for wid,g in groups}
        for future in as_completed(futures):
            # Fail loudly on a whole-work failure; completed work checkpoints remain resumable.
            results.append(future.result())
            if len(results)%25==0 or len(results)==len(groups):print(f'works {len(results)}/{len(groups)}',flush=True)
    rows=[row for result in results for row in result['segments']]
    data=pd.DataFrame(rows).sort_values(['composition_id','perf_id','candidate_index'])
    expected={(source.perf_id,index) for result in results
              for source in audit[audit.composition_id==result['work']].itertuples() for index in range(len(result['candidates']))}
    actual=set(zip(data.perf_id,data.candidate_index))
    if actual!=expected or len(data)!=len(expected):raise ValueError('candidate_coverage_accounting_error')
    data.to_csv(out/'segments.csv',index=False,encoding='utf-8-sig')
    candidate_rows=[dict(composition_id=r['work'],reference_perf_id=r['reference_perf_id'],candidate_index=i,candidate_number=i+1,
       duration_seconds=c['end_seconds']-c['start_seconds'],**{k:v for k,v in c.items() if k!='content_descriptor'})
       for r in results for i,c in enumerate(r['candidates'])]
    pd.DataFrame(candidate_rows).sort_values(['composition_id','candidate_index']).to_csv(out/'candidates.csv',index=False,encoding='utf-8-sig')
    p=data.groupby(['perf_id','composition_id']).agg(candidate_count=('candidate_index','size'),
       exported_count=('status',lambda s:s.isin(['exported','exported_review']).sum()),
       accepted_exported_count=('status',lambda s:(s=='exported').sum()),
       review_exported_count=('status',lambda s:(s=='exported_review').sum())).reset_index()
    p.to_csv(out/'performances.csv',index=False,encoding='utf-8-sig')
    data[(data.status!='exported')|data.review_reasons.str.contains('longer_than_60s|source_quality_flag',na=False)].to_csv(
        out/'review_queue.csv',index=False,encoding='utf-8-sig')
    shutil.copy2(input_root/'config.json',out/'candidate_generation_config.json')
    if not args.work_ids:shutil.copy2(input_root/'input_midi_audit.csv',out/'input_midi_audit.csv')
    else:audit.to_csv(out/'input_midi_audit.csv',index=False,encoding='utf-8-sig')
    summary=dict(works=len(groups),performances=len(audit),work_candidates=len(candidate_rows),
       performance_candidate_rows=len(data),statuses=data.status.value_counts().to_dict(),
       midi_files=int(data.status.isin(['exported','exported_review']).sum()),automatic_selection_applied=False,
       candidates_and_mappings_reused=True,annotations_used=False,youtube_accessed=False,identity_verified=False,
       signature=signature,input_root=str(input_root),input_summary_sha256=sha(input_root/'summary.json'),
       audit_sha256=sha(out/'input_midi_audit.csv'))
    dump(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
