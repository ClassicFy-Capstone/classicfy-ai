"""Annotation-free, work-level preview selection. No full-corpus rerun is implicit in tests."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pretty_midi

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
from preprocessing.representative_passages import PassageConfig,propose_passages,map_passages,style_proxy,choose_passage


def work_task(task):
    rows,destination,export,signature=task
    from threadpoolctl import threadpool_limits
    limiter=threadpool_limits(1)
    out=Path(destination);work=rows[0]['composition_id'];folder=out/'works';folder.mkdir(exist_ok=True)
    fingerprints=[]
    for row in rows:
        source=Path(row['local_midi'])
        row['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else ''
        fingerprints.append([row['perf_id'],row['source_sha256'],row['quality']])
    checkpoint=folder/f'{work}.json'
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text(encoding='utf-8'))
        if previous.get('signature')==signature and previous.get('fingerprints')==fingerprints and previous.get('export_requested')==export:
            valid=True
            for item in previous['segments']:
                if item['status']=='export_failed':valid=False;break
                if item['status']=='exported':
                    path=Path(item['clip_path'])
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item.get('clip_sha256'):valid=False;break
            if valid:return previous
    notes={};failures=[]
    for row in rows:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');m=pretty_midi.PrettyMIDI(row['local_midi'])
            a=np.asarray([[n.pitch,n.start,n.end,n.velocity] for i in m.instruments if not i.is_drum for n in i.notes],float)
            if len(a)<16:raise ValueError('too_few_valid_notes')
            notes[row['perf_id']]=a
        except Exception as exc:failures.append(dict(perf_id=row['perf_id'],composition_id=work,status='read_failed',error=str(exc)))
    if not notes:return dict(work=work,candidates=[],selection=None,segments=failures)
    clean=[r for r in rows if r['perf_id'] in notes and r['quality'] in ('','high quality')]
    pool=clean or [r for r in rows if r['perf_id'] in notes]
    median=float(np.median([np.max(notes[r['perf_id']][:,2]) for r in pool]))
    reference=min(pool,key=lambda r:(abs(np.max(notes[r['perf_id']][:,2])-median),r['perf_id']))
    rid=reference['perf_id'];ref=notes[rid];config=PassageConfig()
    try:candidates=propose_passages(ref,config)
    except Exception as exc:
        return dict(work=work,candidates=[],selection=None,segments=failures+[
            dict(perf_id=r['perf_id'],composition_id=work,status='candidate_failed',error=str(exc)) for r in rows if r['perf_id'] in notes])
    mapped={};vectors=[]
    for row in rows:
        pid=row['perf_id']
        if pid not in notes:continue
        try:
            mapping=map_passages(ref,notes[pid],candidates,config) if pid!=rid else [
                dict(candidate_index=i,start_seconds=c['start_seconds'],end_seconds=c['end_seconds'],
                     alignment_similarity=1.,local_time_ratio=1.,alignment_accepted=True,
                     mapping_method='reference_exact',mapping_is_ground_truth=False) for i,c in enumerate(candidates)]
            mapped[pid]=mapping
            if row['quality'] in ('','high quality'):
                vectors.append([style_proxy(notes[pid],m['start_seconds'],m['end_seconds'],
                       candidates[i]['end_seconds']-candidates[i]['start_seconds'],notes[pid])
                       if m['alignment_accepted'] else None for i,m in enumerate(mapping)])
        except Exception as exc:failures.append(dict(perf_id=pid,composition_id=work,status='mapping_failed',error=str(exc)))
    selection=choose_passage(candidates,vectors);chosen=selection['selected_candidate'];segments=failures
    for row in rows:
        pid=row['perf_id']
        if pid not in mapped:continue
        mapping=mapped[pid][chosen]
        record=dict(perf_id=pid,composition_id=work,reference_perf_id=rid,candidate_index=chosen,
                    source_midi=row['local_midi'],quality=row['quality'],**{k:v for k,v in mapping.items() if k!='candidate_index'},
                    source_sha256=row['source_sha256'],
                    representation_score=candidates[chosen]['representation_score'],
                    confidence_level='unvalidated',requires_review=True,identity_verified=False,
                    cadence_status='not_inferred',style_embedding_trained=False,
                    style_difference_proxy=selection['style_difference'][chosen],
                    stable_style_score=selection['stable_style_score'][chosen],
                    style_support=selection['supports'][chosen],selection_stability=selection['selection_stability'],
                    selection_method=selection['selection_method'])
        if not mapping['alignment_accepted']:
            record.update(status='mapping_rejected',review_reasons='local_alignment_or_repeat_uncertain')
        else:
            record.update(status='mapped',review_reasons='representativeness_and_perceptual_style_not_calibrated')
            if selection['selection_stability'] is not None and selection['selection_stability']<.6:
                record['review_reasons']+=';unstable_style_selection'
            if not selection['style_selection_available']:record['review_reasons']+=';insufficient_style_comparisons'
            if row['quality'] not in ('','high quality'):record['review_reasons']+=';source_quality_flag'
            if export:
                try:
                    from extract_atepp_themes import clip_midi
                    from run_atepp_theme_candidates import validate_export
                    path=out/'midi'/work/f'{pid}.mid'
                    count=clip_midi(row['local_midi'],path,mapping['start_seconds'],mapping['end_seconds'])
                    validate_export(path,row['local_midi'],mapping['start_seconds'],mapping['end_seconds'],count)
                    record.update(status='exported',clip_path=str(path),note_count=count,export_validation='passed',
                                  clip_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                except Exception as exc:
                    record.update(status='export_failed',error=str(exc))
                    if path.exists():
                        quarantine=out/'quarantine';quarantine.mkdir(exist_ok=True);path.replace(quarantine/f'{work}_{pid}.mid')
        segments.append(record)
    result=dict(work=work,reference_perf_id=rid,reference_selection='quality_then_median_duration',
                candidates=candidates,mappings=mapped,selection=selection,segments=segments,
                signature=signature,fingerprints=fingerprints,export_requested=export)
    temporary=checkpoint.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');temporary.replace(checkpoint)
    return result


def main():
    workspace=HERE.parents[2];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=workspace/'outputs/atepp_representative_v2')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--limit-works',type=int)
    p.add_argument('--work-ids',help='Optional comma-separated composition IDs for a bounded validation run')
    p.add_argument('--no-export',action='store_true')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    audit=pd.read_csv(HERE.parent/'analysis/atepp_research/midi_audit.csv',dtype={'perf_id':str,'composition_id':str}).fillna('')
    groups=list(audit.groupby('composition_id',sort=True))
    if args.work_ids:groups=[(w,g) for w,g in groups if w in args.work_ids.split(',')]
    if args.limit_works:groups=groups[:args.limit_works]
    if not groups:p.error('No matching composition IDs')
    sources=[HERE.parent/'src/preprocessing/representative_passages.py',Path(__file__),
             HERE/'extract_atepp_themes.py',HERE/'run_atepp_theme_candidates.py']
    signature=hashlib.sha256(b''.join(path.read_bytes() for path in sources)).hexdigest()
    (out/'config.json').write_text(json.dumps(dict(engine='representative_v2',config=asdict(PassageConfig()),
          source_sha256=signature,annotations_required=False,identity_guaranteed=False,
          exact_fmp_algorithm=False,style_proxy_not_learned_embedding=True),indent=2),encoding='utf-8')
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(work_task,(g.to_dict('records'),str(out),not args.no_export,signature)):w for w,g in groups}
        for future in as_completed(tasks):
            try:results.append(future.result())
            except Exception as exc:
                w=tasks[future];g=audit[audit.composition_id==w]
                results.append(dict(work=w,candidates=[],selection=None,segments=[dict(perf_id=r.perf_id,composition_id=w,status='work_failed',error=str(exc)) for r in g.itertuples()]))
            print(f'works {len(results)}/{len(groups)}',flush=True)
    segments=[s for r in results for s in r['segments']]
    data=pd.DataFrame(segments);expected=sum(len(g) for _,g in groups)
    if len(data)!=expected or data.perf_id.nunique()!=expected:raise ValueError('coverage_accounting_error')
    data.sort_values('perf_id').to_csv(out/'segments.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([dict(composition_id=r['work'],candidate_index=i,**{k:v for k,v in c.items() if k!='content_descriptor'})
        for r in results for i,c in enumerate(r['candidates'])]).to_csv(out/'candidates.csv',index=False,encoding='utf-8-sig')
    data[data.status.isin(['read_failed','mapping_failed','mapping_rejected','candidate_failed','work_failed','export_failed'])].to_csv(out/'review_queue.csv',index=False,encoding='utf-8-sig')
    summary=dict(works=len(groups),performances=expected,statuses=data.status.value_counts().to_dict(),
                 candidate_count=sum(len(r['candidates']) for r in results),identity_verified=False,
                 perceptual_improvement_evaluated=False,annotations_used=False,youtube_accessed=False,
                 source_signature=signature,audit_sha256=hashlib.sha256((HERE.parent/'analysis/atepp_research/midi_audit.csv').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
