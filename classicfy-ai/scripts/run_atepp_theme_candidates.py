"""Corpus-wide label-free MIDI candidate detection, held-out evaluation, export.

Inference/consensus precede reading any reference intervals. Annotation-backed
selection is a separate final overlay and is never used as predicted accuracy.
"""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import mido
import pretty_midi
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
from preprocessing.theme_candidates import ThemeConfig, propose, consensus, interval_iou

_THREAD_LIMITER=None


def limit_worker_threads():
    global _THREAD_LIMITER
    if _THREAD_LIMITER is None:
        _THREAD_LIMITER=threadpool_limits(1)


def dump(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def infer_one(task):
    row,out,signature=task
    limit_worker_threads()
    pid=row['perf_id'];path=Path(row['local_midi']);cache=Path(out)/'predictions'/f'{pid}.json'
    digest=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ''
    if cache.exists():
        saved=json.loads(cache.read_text(encoding='utf-8'))
        if saved.get('detector_signature')==signature and saved.get('source_sha256')==digest:return saved
    result=dict(perf_id=pid,composition_id=row['composition_id'],source_midi=str(path),source_sha256=digest,
                detector_signature=signature,source_quality=row['quality'],source_alignment_status=row['status'])
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            midi=pretty_midi.PrettyMIDI(str(path))
        notes=[[n.pitch,n.start,n.end,n.velocity] for i in midi.instruments if not i.is_drum for n in i.notes]
        result.update(candidates=propose(notes),status='inferred',source_notes=len(notes),
                      midi_warnings=[str(w.message) for w in caught])
    except Exception as exc:
        result.update(candidates=[],status='inference_failed',error=f'{type(exc).__name__}: {exc}')
    dump(cache,result)
    return result


def work_split(primary,audit):
    works=sorted(primary.composition_id.unique(),key=lambda s:hashlib.sha256(('theme-eval-v1/'+s).encode()).hexdigest())
    # The previous conversation and implementation smoke test inspected 05943.
    # Keep its entire work in development, never call it held-out test evidence.
    inspected=set(audit.loc[audit.perf_id=='05943','composition_id'])
    dev=[w for w in works if w in inspected]
    for w in works:
        if len(dev)>=max(1,round(.3*len(works))):break
        if w not in dev:dev.append(w)
    return {w:'development' if w in dev else 'test' for w in works}


def weighted_selection(predictions,audit,length_weight,opening_weight):
    result={}
    for _,group in audit.groupby('composition_id'):
        records=[]
        for pid in group.perf_id:
            record=predictions[pid];candidates=[]
            for c in record['candidates']:
                duration=c['end_seconds']-c['start_seconds']
                length_prior=float(np.exp(-.5*(np.log(max(duration,1)/30.)/.65)**2))
                opening_prior=float(np.exp(-(c['start_seconds']-c['first_note_seconds'])/25.))
                candidates.append(dict(c,ranking_score=c['ranking_score']+length_weight*length_prior+opening_weight*opening_prior))
            records.append(dict(record,candidates=candidates))
        result.update(consensus(records))
    return result


def tune_ranking(predictions,audit,primary,out):
    """Small explicit grid on development works only; held-out labels untouched."""
    splits=work_split(primary,audit)
    gold=primary[primary.composition_id.map(splits)=='development']
    dev_audit=audit[audit.composition_id.isin(gold.composition_id)]
    scores=[]
    for length_weight in (0.,.15,.30,.60):
        for opening_weight in (0.,.10,.20):
            selection=weighted_selection(predictions,dev_audit,length_weight,opening_weight)
            rows=[]
            for g in gold.to_dict('records'):
                if g['perf_id'] not in selection:
                    rows.append((g['composition_id'],0.));continue
                c=predictions[g['perf_id']]['candidates'][selection[g['perf_id']]['candidate_index']]
                rows.append((g['composition_id'],interval_iou(c['start_seconds'],c['end_seconds'],g['start_seconds'],g['end_seconds'])))
            frame=pd.DataFrame(rows,columns=['work','iou'])
            metric=float(frame.groupby('work').iou.mean().mean()) if len(frame) else 0.
            scores.append(dict(length_weight=length_weight,opening_weight=opening_weight,development_work_macro_iou=metric))
    best=max(scores,key=lambda x:(x['development_work_macro_iou'],-x['length_weight'],-x['opening_weight']))
    dump(out/'ranking_calibration.json',dict(development_works=sorted(gold.composition_id.unique()),grid=scores,
         selected=best,test_labels_used=False,objective='development work-macro mean IoU',
         reference_is_automatically_transferred=True))
    print(f'Development-only ranking calibration: {best}',flush=True)
    return weighted_selection(predictions,audit,best['length_weight'],best['opening_weight'])


def evaluate(predictions,selection,primary,audit,out):
    splits=work_split(primary,audit)
    dump(out/'evaluation_split.json',dict(composition_split=splits,split_unit='entire composition_id',
         reference='existing automatically transferred BPSD v2 labels, not independent human timestamps',
         inspected_performance_forced_to_development='05943',ranking_tuned_on_test=False))
    records=[]
    for gold in primary.to_dict('records'):
        pid=gold['perf_id'];prediction=predictions.get(pid)
        if not prediction or not prediction.get('candidates'):
            records.append(dict(perf_id=pid,composition_id=gold['composition_id'],split=splits[gold['composition_id']],
                                method='automatic',iou=0.,success_iou_05=False,prediction_failed=True))
            continue
        selected=prediction['candidates'][selection[pid]['candidate_index']]
        for method,c in [('automatic',selected),('first_30_seconds',dict(start_seconds=selected['first_note_seconds'],
                               end_seconds=min(selected['last_note_seconds'],selected['first_note_seconds']+30.)))]:
            iou=interval_iou(c['start_seconds'],c['end_seconds'],gold['start_seconds'],gold['end_seconds'])
            records.append(dict(perf_id=pid,composition_id=gold['composition_id'],split=splits[gold['composition_id']],
                         method=method,iou=iou,success_iou_05=bool(iou>=.5),
                         start_error_seconds=abs(c['start_seconds']-gold['start_seconds']),
                         end_error_seconds=abs(c['end_seconds']-gold['end_seconds']),
                         heuristic_confidence=selected['heuristic_confidence'],
                         top5_oracle_iou=max(interval_iou(p['start_seconds'],p['end_seconds'],gold['start_seconds'],gold['end_seconds']) for p in prediction['candidates']),
                         source_sha256=prediction['source_sha256'],prediction_failed=False))
    raw=pd.DataFrame(records)
    # Exact duplicate source MIDI does not get repeated evaluation weight.
    data=raw.drop_duplicates(['source_sha256','method','composition_id']) if 'source_sha256' in raw else raw
    data.to_csv(out/'evaluation_performance.csv',index=False,encoding='utf-8-sig')
    macro=[]
    for (split,method,work),g in data.groupby(['split','method','composition_id']):
        macro.append(dict(split=split,method=method,composition_id=work,performances=len(g),
                          mean_iou=float(g.iou.mean()),success_iou_05=float(g.success_iou_05.mean()),
                          start_mae_seconds=float(g.start_error_seconds.mean()),end_mae_seconds=float(g.end_error_seconds.mean()),
                          start_within_3s=float((g.start_error_seconds<=3).mean()),
                          end_within_3s=float((g.end_error_seconds<=3).mean()),
                          top5_oracle_iou=float(g.top5_oracle_iou.mean())))
    work=pd.DataFrame(macro);work.to_csv(out/'evaluation_work.csv',index=False,encoding='utf-8-sig')
    reports=[]
    for (split,method),g in work.groupby(['split','method']):
        rng=np.random.default_rng(20261105)
        boot=np.mean(rng.choice(g.success_iou_05.to_numpy(),size=(2000,len(g)),replace=True),axis=1)
        reports.append(dict(split=split,method=method,works=len(g),performances=int(g.performances.sum()),
                       mean_iou=float(g.mean_iou.mean()),iou_05_accuracy=float(g.success_iou_05.mean()),
                       accuracy_work_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
                       start_mae_seconds=float(g.start_mae_seconds.mean()),end_mae_seconds=float(g.end_mae_seconds.mean()),
                       start_within_3s=float(g.start_within_3s.mean()),end_within_3s=float(g.end_within_3s.mean()),
                       top5_oracle_iou=float(g.top5_oracle_iou.mean())))
    # Fit only a confidence cutoff, on development works. The ranking itself is fixed.
    dev=data[(data.split=='development')&(data.method=='automatic')]
    threshold=1.01
    calibration=[]
    for cut in np.arange(.4,1.,.05):
        subset=dev[dev.heuristic_confidence>=cut]
        bywork=subset.groupby('composition_id').success_iou_05.mean()
        precision=float(bywork.mean()) if len(bywork) else 0.
        calibration.append(dict(cutoff=float(cut),works=len(bywork),performances=len(subset),work_macro_precision=precision))
        if len(bywork)>=3 and len(subset)>=15 and precision>=.8 and threshold>1:
            threshold=float(cut)
    summary=dict(metrics=reports,primary_reference_performances=len(primary),primary_reference_works=primary.composition_id.nunique(),
                 evaluated_unique_performances=int(data[data.method=='automatic'].perf_id.nunique()),
                 definition='IoU >= 0.5 against FIRST annotated exposition FirstGroup or variation Theme',
                 reference_is_automatically_transferred=True,test_split_is_composition_disjoint=True,
                 scope_limitation='Beethoven piano sonata first movements only; transfer coverage is a selected subset; no all-composer accuracy claim',
                 medium_confidence_cutoff=threshold,calibration=calibration,
                 confidence_is_probability=False,test_used_to_choose_threshold=False)
    dump(out/'evaluation.json',summary)
    return threshold,summary


def validate_export(path,source,a,b,expected_count):
    from collections import Counter
    counts=Counter();pedal={};t=0.;onsets=0
    for msg in mido.MidiFile(path):
        t+=msg.time
        if msg.type=='note_on' and msg.velocity:
            counts[(msg.channel,msg.note)]+=1;onsets+=1
        elif msg.type in ('note_on','note_off'):
            counts[(msg.channel,msg.note)]-=1
            if counts[(msg.channel,msg.note)]<0:raise ValueError('orphan_note_off')
        elif msg.type=='control_change' and msg.control==64:pedal[msg.channel]=msg.value
    if any(counts.values()) or any(v>=64 for v in pedal.values()):raise ValueError('hanging_note_or_pedal')
    if abs(t-(b-a))>.002 or onsets!=expected_count:raise ValueError('duration_or_count_mismatch')
    original=pretty_midi.PrettyMIDI(source)
    expected=sorted((i.program,i.is_drum,n.pitch,n.velocity,round(max(n.start,a)-a,3),round(min(n.end,b)-a,3))
        for i in original.instruments for n in i.notes if n.start<b-1e-9 and n.end>a+1e-9
        and round((min(n.end,b)-a)*1000)>round((max(n.start,a)-a)*1000))
    clipped=pretty_midi.PrettyMIDI(str(path))
    actual=sorted((i.program,i.is_drum,n.pitch,n.velocity,round(n.start,3),round(n.end,3)) for i in clipped.instruments for n in i.notes)
    same=len(expected)==len(actual)
    if same and len(expected):
        e=np.asarray(expected,dtype=float);v=np.asarray(actual,dtype=float)
        same=bool(np.array_equal(e[:,:4],v[:,:4]) and np.allclose(e[:,4:],v[:,4:],atol=.0011,rtol=0))
    if not same:
        raise ValueError(f'independent_note_mismatch:{len(expected)}:{len(actual)}')
    return len(actual)


def export_one(task):
    row,out=task
    limit_worker_threads()
    from extract_atepp_themes import clip_midi
    root=Path(out);cache=root/'export_checks'/f"{row['perf_id']}.json"
    path=root/'midi'/row['confidence_level']/row['composition_id']/f"{row['perf_id']}.mid"
    result=dict(row)
    if cache.exists():
        prior=json.loads(cache.read_text(encoding='utf-8'))
        if (prior.get('status')=='exported' and prior.get('source_sha256')==row['source_sha256']
            and prior.get('start_seconds')==row['start_seconds'] and prior.get('end_seconds')==row['end_seconds']
            and Path(prior['clip_path']).is_file()):return prior
    try:
        n=clip_midi(row['source_midi'],path,row['start_seconds'],row['end_seconds'])
        checked=validate_export(path,row['source_midi'],row['start_seconds'],row['end_seconds'],n)
        result.update(status='exported',clip_path=str(path),note_count=n,independently_checked_notes=checked,
                      clip_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),export_validation='passed')
    except Exception as exc:
        result.update(status='export_failed',clip_path='',error=f'{type(exc).__name__}: {exc}',export_validation='failed')
        if path.exists():
            quarantine=root/'quarantine'/path.name;quarantine.parent.mkdir(exist_ok=True);path.replace(quarantine)
    dump(cache,result)
    return result


def parallel(items,worker,workers,label):
    results=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,item):item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%100==0 or len(results)==len(items):
                print(f'{label}: {len(results)}/{len(items)} ({time.monotonic()-start:.0f}s)',flush=True)
    return results


def legacy_main():
    workspace=HERE.parents[2]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=workspace/'outputs/atepp_all_theme_candidates')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--stage',choices=['infer','all'],default='all')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    for folder in ('predictions','export_checks'):(out/folder).mkdir(exist_ok=True)
    audit=pd.read_csv(HERE.parent/'analysis/atepp_research/midi_audit.csv',dtype={'perf_id':str,'composition_id':str}).fillna('')
    if args.limit:audit=audit.head(args.limit)
    config=asdict(ThemeConfig())
    signature=hashlib.sha256((HERE.parent/'src/preprocessing/theme_candidates.py').read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()
    dump(out/'detector_config.json',dict(config=config,detector_signature=signature,labels_used_in_inference=False,
         output_is_candidate_not_verified_theme=True,ranking_weights_fixed_before_test=True))
    predictions=parallel([(r,str(out),signature) for r in audit.to_dict('records')],infer_one,args.workers,'infer')
    pred={r['perf_id']:r for r in predictions}
    selection={}
    for _,group in audit.groupby('composition_id'):
        selection.update(consensus([pred[pid] for pid in group.perf_id]))
    dump(out/'label_free_selection.json',selection)
    flattened=[]
    for r in predictions:
        for i,c in enumerate(r['candidates']):
            flattened.append(dict(perf_id=r['perf_id'],composition_id=r['composition_id'],candidate_rank=i+1,
                                  selected_by_consensus=selection[r['perf_id']]['candidate_index']==i,
                                  **{k:v for k,v in c.items() if k!='fingerprint'}))
    pd.DataFrame(flattened).sort_values(['perf_id','candidate_rank']).to_csv(out/'candidates.csv',index=False,encoding='utf-8-sig')
    if args.stage=='infer':return
    # First annotation read occurs ONLY AFTER all automatic candidates/selection are saved.
    primary=pd.read_csv(workspace/'outputs/atepp_theme_extraction/primary_segments.csv',dtype={'perf_id':str,'composition_id':str})
    primary=primary[primary.perf_id.isin(audit.perf_id)]
    if len(primary):
        selection=tune_ranking(pred,audit,primary,out)
        dump(out/'selection_after_development_calibration.json',selection)
        for c in flattened:c['selected_by_consensus']=selection[c['perf_id']]['candidate_index']==c['candidate_rank']-1
        pd.DataFrame(flattened).sort_values(['perf_id','candidate_rank']).to_csv(out/'candidates.csv',index=False,encoding='utf-8-sig')
        cutoff,evaluation=evaluate(pred,selection,primary,audit,out)
    else:cutoff,evaluation=1.01,dict(metrics=[],medium_confidence_cutoff=1.01)
    annotations={r['perf_id']:r for r in primary.to_dict('records')}
    rows=[];failed=[]
    for source in audit.to_dict('records'):
        pid=source['perf_id'];r=pred[pid]
        base=dict(perf_id=pid,composition_id=source['composition_id'],artist=source['artist'],composer=source['composer'],track=source['track'],
                  source_midi=r['source_midi'],source_sha256=r['source_sha256'],source_quality=source['quality'],
                  source_alignment_status=source['status'],source_hash_matches_audit=r['source_sha256']==source['sha256'],
                  human_verified=False,cadence_status='not_inferred',youtube_accessed=False)
        if not r['candidates']:
            failed.append(dict(**base,status='inference_failed',error=r.get('error','no_candidates')));continue
        chosen=r['candidates'][selection[pid]['candidate_index']]
        reasons=[]
        if chosen['heuristic_confidence']<cutoff:reasons.append('below_development_precision_cutoff' if cutoff<=1 else 'no_reliable_cutoff_on_development_works')
        if chosen['recurrence']<.65:reasons.append('weak_recurrence')
        if chosen['boundary_strength']<.35:reasons.append('weak_boundary')
        if selection[pid]['available_other_performances']==0:reasons.append('single_performance_no_consensus')
        elif selection[pid]['composition_agreement']<.45:reasons.append('low_between_performance_agreement')
        if chosen['frame_resolution_seconds']>2:reasons.append('coarse_temporal_resolution')
        if source['quality'] in ('low quality','background noise','corrupted','applause'):reasons.append('source_quality_flag')
        if r.get('midi_warnings'):reasons.append('source_midi_warning')
        backed=annotations.get(pid)
        if backed:
            start,end=float(backed['start_seconds']),float(backed['end_seconds'])
            method='annotation_transfer';level='annotation_backed'
            reasons=[x for x in reasons if x in ('source_quality_flag','source_midi_warning')]
        else:
            start,end=chosen['start_seconds'],chosen['end_seconds'];method='label_free_midi'
            level='low' if reasons else 'medium'
        rows.append(dict(**base,start_seconds=start,end_seconds=end,duration_seconds=end-start,
                         method=method,confidence_level=level,low_confidence=bool(reasons),review_reasons=';'.join(reasons),
                         heuristic_confidence=chosen['heuristic_confidence'],confidence_is_probability=False,
                         automatic_start_seconds=chosen['start_seconds'],automatic_end_seconds=chosen['end_seconds'],
                         automatic_candidate_rank=selection[pid]['candidate_index']+1,
                         recurrence=chosen['recurrence'],boundary_strength=chosen['boundary_strength'],
                         composition_agreement=selection[pid]['composition_agreement'],
                         annotation_label=backed['label'] if backed else '',
                         annotation_reference_is_automatic_transfer=bool(backed),requires_human_review=True))
    exported=parallel([(r,str(out)) for r in rows],export_one,args.workers,'export+verify')
    all_rows=sorted(exported+failed,key=lambda r:r['perf_id'])
    data=pd.DataFrame(all_rows)
    data.to_csv(out/'segments.csv',index=False,encoding='utf-8-sig')
    data[(data.status!='exported')|data.low_confidence.fillna(True)].to_csv(out/'low_confidence_and_failures.csv',index=False,encoding='utf-8-sig')
    data[data.status!='exported'].to_csv(out/'failures.csv',index=False,encoding='utf-8-sig')
    assert len(data)==len(audit) and data.perf_id.nunique()==len(audit)
    success=data[data.status=='exported']
    summary=dict(source_performances=len(audit),source_compositions=audit.composition_id.nunique(),
                 inferred_performances=len(rows),exported_performances=len(success),exported_compositions=success.composition_id.nunique(),
                 failures=int((data.status!='exported').sum()),candidate_intervals=len(flattened),
                 methods=success.method.value_counts().to_dict(),confidence_levels=success.confidence_level.value_counts().to_dict(),
                 low_confidence_exported=int(success.low_confidence.sum()),all_exported_independently_note_verified=True,
                 independently_checked_notes=int(success.independently_checked_notes.sum()),
                 source_files_changed_from_audit=int((~data.source_hash_matches_audit).sum()),
                 evaluation_file='evaluation.json',confidence_cutoff_from_development=cutoff,
                 youtube_accessed=False,cadence_extracted=False,human_musical_boundary_verified=False,
                 detector_signature=signature,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    # Preserve the old experiment explicitly; default uses work-level v2 and
    # a new output directory so old caches/metrics are never reinterpreted.
    if '--legacy' in sys.argv:
        sys.argv.remove('--legacy');legacy_main()
    else:
        from run_representative_passages import main as representative_main
        representative_main()


if __name__=='__main__':main()
