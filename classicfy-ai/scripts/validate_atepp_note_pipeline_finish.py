"""Bounded-memory final checks used by validate_atepp_note_pipeline.py."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from embedding.note_performance_data import (NotePerformancePair, NotePerformanceSegment,
    collate_note_performance, load_note_performance)
from embedding.note_score_context_data import NoteScoreSegment, load_note_score_context, save_note_score_scale, load_note_score_scale
from embedding.note_score_encoder import NoteScoreEncoder
from embedding.style_profile_data import save_style_artifact, load_style_artifact
from embedding.style_triplets import build_triplets, calibrate_distances
from features.note_performance_normalization import (NotePerformanceScaler, fit_note_performance_scaler,
    save_note_performance_scaler, load_note_performance_scaler)
from features.note_score_context import fit_note_score_scale
from features.style_profile import (G_NAMES, StyleConfig, StyleRecording, FrozenProfileScaler, artifact, fit_profile_scaler,
    fit_reference, make_windows, reference_delta, scale_profile, stable_id, summarize_window)
from preprocessing.musicxml_score import read_musicxml
from validate_atepp_note_pipeline import dump, read_log


def save_checked(value, path):
    if path.exists():
        if load_style_artifact(path)['id'] != value['id']:
            raise ValueError('Existing validation artifact differs: ' + str(path))
    else:
        save_style_artifact(value, path)


@lru_cache(maxsize=4)
def cached_score(path):
    return load_note_score_context(path)[0]


@lru_cache(maxsize=4)
def cached_reference(path):
    return json.loads(Path(path).read_text())


def finish(args, metadata):
    torch.set_num_threads(1)
    out=args.out
    results=read_log(out/'alignments.jsonl')
    if len({r['perf_id'] for r in results}) != len(metadata):
        raise ValueError('Complete alignment screening before finish; missing metadata rows')
    accepted=sorted([r for r in results if r['status']=='extracted'],key=lambda r:r['perf_id'])
    if not accepted:
        dump(out/'final_summary.json',{'statuses':dict(Counter(r['status'] for r in results)), 'pre_AE_records':0,
                                      'failure':'no accepted automatic XML alignments','AE_or_loss_executed':False})
        return
    scores={r['score_path']:r for r in read_log(out/'scores.jsonl') if r['status']=='prepared'}
    rows={r['perf_id']:r for r in metadata}
    cfg=StyleConfig()
    groupings=defaultdict(list)
    # First pass: immutable per-performance windows and source index; no corpus-sized R array.
    entries=[]
    count,mean,m2=np.zeros(11,dtype=np.int64),np.zeros(11),np.zeros(11)
    training=[];train_score_paths=set();global_coverage=np.zeros(11,dtype=np.int64)
    for index,result in enumerate(accepted,1):
        r=load_note_performance(result['performance_cache'])
        fingerprint=r.fingerprint
        path=out/'windows'/f'{r.record_key}.json'
        windows=None
        if path.exists():
            try:
                cached=json.loads(path.read_text())
                if all(w['R_fingerprint']==fingerprint for w in cached):
                    windows=cached
            except json.JSONDecodeError:
                pass
        if windows is None:
            x=read_musicxml(result['xml_path'])
            windows=make_windows(StyleRecording(r,x,r.provenance['performance']['midi_sha256']))
            dump(path,windows)
        if hashlib.sha256(Path(result['xml_path']).read_bytes()).hexdigest()!=r.score_source_sha256:
            raise ValueError('XML source changed')
        performance_id=f'{r.dataset}:{r.record_key}'
        recording_id=r.provenance['performance']['midi_sha256']
        entry={'performance_id':performance_id,'record_key':r.record_key,'performance_cache':result['performance_cache'],
               'xml_path':result['xml_path'],'R_fingerprint':fingerprint,'windows_file':str(path),
               'windows_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'work_key':r.work_key,
               'split':r.split,'recording_id':recording_id,'score_path':rows[r.record_key]['score_path']}
        entries.append(entry)
        group=stable_id([r.work_key,r.score_source_sha256,r.traversal_id]);entry['cohort_key']=group
        groupings[group].append(entry)
        global_coverage += r.mask.sum(axis=0)
        if r.split=='train':
            local=fit_note_performance_scaler([r],fit_manifest_id='ATEPP-validation-single-record')
            n=np.asarray(local.valid_counts);mu=np.asarray(local.mean)
            variance=np.array([sd*sd if state=='fitted' else 0. for sd,state in zip(local.scale,local.channel_status)])
            total=count+n;delta=mu-mean;valid=n>0
            m2[valid] += variance[valid]*n[valid]+delta[valid]**2*count[valid]*n[valid]/total[valid]
            mean[valid] += delta[valid]*n[valid]/total[valid]
            count=total;training.extend(local.training_records)
            train_score_paths.add(scores[entry['score_path']]['score_cache'])
        if index%100==0: print('windows/scaler',index,'/',len(accepted),flush=True)
    window_index=artifact('window-index',cfg,sources=entries,format='per-record full window JSON with SHA256',
                          split_protocol_sha256=hashlib.sha256((out/'splits.json').read_bytes()).hexdigest())
    save_checked(window_index,out/'window_index.json')
    sd=np.sqrt(np.divide(m2,count,out=np.zeros(11),where=count>0))
    status=tuple('unobserved' if n==0 else 'constant' if s==0 else 'fitted' for n,s in zip(count,sd))
    sd[sd==0]=1
    scaler=NotePerformanceScaler(tuple(map(float,mean)),tuple(map(float,sd)),tuple(map(int,count)),tuple(bool(n) for n in count),
        status,window_index['id'],tuple(sorted(training,key=lambda t:(t['dataset'],t['record_key']))))
    if not (out/'R_scaler.json').exists(): save_note_performance_scaler(scaler,out/'R_scaler.json')
    assert load_note_performance_scaler(out/'R_scaler.json').fingerprint==scaler.fingerprint
    score_scale=fit_note_score_scale(load_note_score_context(path)[0] for path in sorted(train_score_paths))
    if not (out/'S_scale.json').exists(): save_note_score_scale(score_scale,out/'S_scale.json')
    # Reference fixed once for each train cohort; the index fixes the full snapshot.
    refs={}
    for group,items in sorted(groupings.items()):
        recordings=[]
        xml=read_musicxml(items[0]['xml_path']) if items[0]['split']=='train' else None
        for entry in items:
            if entry['split']=='train':
                recordings.append(StyleRecording(load_note_performance(entry['performance_cache']),
                    xml,entry['recording_id'],entry['performance_id']))
        ref=fit_reference(recordings,cfg)
        path=out/'references'/f'{group}.json';save_checked(ref,path)
        refs[group]={'path':str(path),'id':ref['id']}
    ref_index=artifact('reference-index',cfg,parents={'windows':window_index['id']},fit_split='train',cohorts=refs)
    save_checked(ref_index,out/'reference_index.json')
    mini_profiles=[];profile_files=[];batch_outcomes=[];notes=full=tails=0
    scaled_sum,scaled_sq,scaled_count=np.zeros(11),np.zeros(11),np.zeros(11,dtype=np.int64)
    torch.manual_seed(cfg.seed);encoder=NoteScoreEncoder().eval();encoded=set()
    for index,entry in enumerate(entries,1):
        r=load_note_performance(entry['performance_cache'])
        before=r.fingerprint
        ref=cached_reference(refs[entry['cohort_key']]['path'])
        s=cached_score(scores[entry['score_path']]['score_cache'])
        windows=json.loads(Path(entry['windows_file']).read_text())
        pair=NotePerformancePair(s,r)
        scaled=scaler.transform(r)
        np.testing.assert_allclose(scaler.inverse_transform(scaled,r.mask)[r.mask],r.values[r.mask],atol=2e-6,rtol=2e-6)
        assert np.isfinite(scaled).all() and (scaled[~r.mask]==0).all()
        if r.split=='train':
            for channel in range(11):
                v=scaler.transform_beats(r)[r.beats.valid] if channel==0 else scaled[r.mask[:,channel],channel]
                scaled_sum[channel]+=v.astype(float).sum();scaled_sq[channel]+=(v.astype(float)**2).sum();scaled_count[channel]+=len(v)
        positions=np.array([m['score_onset_q'] for m in r.metadata])
        deltas=reference_delta(r,ref);record_profiles=[];first=True
        for w in windows:
            notes+=w['N'];tails+=int(w['tail']);full+=int(not w['tail'])
            p=summarize_window(r,w,ref,cfg,deltas=deltas)
            p['cohort_reference_id']=p['reference_id'];p['reference_id']=ref_index['id']
            p['profile_id']=stable_id({k:v for k,v in p.items() if k!='profile_id'})
            record_profiles.append(p)
            # Compact sampling view points to the full immutable window/profile files.
            keep=('work_id','score_version_id','repeat_route_id','performance_id','recording_id','split',
                  'same_window_key','window_id','q_start','q_end','tail','N','R_fingerprint','dataset','record_key')
            mw={k:w[k] for k in keep};mw['windows_file']=entry['windows_file'];mw['performance_cache']=entry['performance_cache']
            mini_profiles.append({**p,'window':mw})
            # Every window's exact row ownership is checked; one variable-length
            # collated batch per recording checks padding/model-input contracts.
            actual=np.flatnonzero((positions>=w['q_start']) &
                                  (positions<w['q_end']))
            assert actual.tolist()==w['row_indices']
            if not first or w['tail'] or not w['N']: continue
            first=False
            a,b=np.searchsorted(s.onset_q,[w['q_start'],w['q_end']],side='left')
            item=NotePerformanceSegment(pair,NoteScoreSegment(s,int(a),int(b),int(a),int(b)))
            batch=collate_note_performance([item],scaler,score_scale=score_scale)
            assert batch['R_scaled'].shape==(1,w['N'],11)
            assert not batch['R_loss_mask'][:,:,0].any()
            assert torch.isfinite(batch['R_scaled']).all()
            assert not batch['R_mask'][~batch['note_mask']].any()
            if s.cache_key not in encoded:
                with torch.inference_mode(): context=encoder(batch)
                assert context.shape[:2]==batch['note_mask'].shape and torch.isfinite(context).all()
                encoded.add(s.cache_key)
            batch_outcomes.append({'performance_id':entry['performance_id'],'window_id':w['window_id'],
                                   'shape':list(batch['R_scaled'].shape),'valid':int(batch['R_mask'].sum())})
        path=out/'profiles'/f'{r.record_key}.json';dump(path,record_profiles)
        profile_files.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        assert before==load_note_performance(entry['performance_cache']).fingerprint
        if index%100==0: print('profiles/batches',index,'/',len(entries),flush=True)
    profile_index=artifact('profile-index',cfg,parents={'windows':window_index['id'],'reference':ref_index['id']},files=profile_files)
    save_checked(profile_index,out/'profile_index.json')
    train=[p for p in mini_profiles if p['window']['split']=='train']
    gs=fit_profile_scaler(train,cfg,profiles_id=profile_index['id']);save_checked(gs,out/'g_scaler.json')
    compact_scaler=FrozenProfileScaler.from_artifact(gs)
    for entry in entries:
        path=out/'profiles'/f"{entry['record_key']}.json"
        # Keep scaled data separate from the raw g profiles.
        p=json.loads(path.read_text())
        dump(out/'profiles_scaled'/path.name,[scale_profile(item,compact_scaler) for item in p])
    return complete_metric_report(args, metadata, locals())


def complete_metric_report(args, metadata, state):
    out=args.out
    train=state['train']
    gs=state['gs']
    cfg=state['cfg']
    profile_index=state['profile_index']
    mini_profiles=state['mini_profiles']
    window_index=state['window_index']
    ref_index=state['ref_index']
    entries=state['entries']
    results=state['results']
    accepted=state['accepted']
    scaler=state['scaler']
    score_scale=state['score_scale']
    global_coverage=state['global_coverage']
    batch_outcomes=state['batch_outcomes']
    encoded=state['encoded']
    notes=state['notes']
    full=state['full']
    tails=state['tails']
    scaled_sum=state['scaled_sum']
    scaled_sq=state['scaled_sq']
    scaled_count=state['scaled_count']
    import logging
    logger=logging.getLogger('embedding.style_triplets');logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(out/'metric_progress.log'))
    print('calibrating',len(train),'train profiles',flush=True)
    if (out/'calibration.json').exists():
        calibration=load_style_artifact(out/'calibration.json','distance-calibration',cfg)
        if calibration['parents']!={'profiles':profile_index['id'],'scaler':gs['id']}:
            raise ValueError('Cached calibration dependency mismatch')
    else:
        calibration=calibrate_distances(train,gs,cfg,profiles_id=profile_index['id'])
    save_checked(calibration,out/'calibration.json')
    triplets=build_triplets(train,gs,calibration,cfg,profiles_id=profile_index['id'])
    save_checked(triplets,out/'triplets.json')
    # Exercise the actual pre-AE DataLoader wrapper on a bounded cross-work corpus.
    # The AE model and loss modules are never imported or run here.
    from embedding.note_ae_data import NoteAECorpus
    from embedding.style_profile_data import make_window_artifact
    from torch.utils.data import DataLoader
    selected=[]
    for split in ('train','validation','test'):
        selected.extend([e for e in entries if e['split']==split][:8])
    for t in triplets['triplets'][:3]:
        for role in ('anchor','positive','negative'):
            pid=t[role]['performance_id']
            selected.extend(e for e in entries if e['performance_id']==pid)
    selected=list({e['performance_id']:e for e in selected}.values())
    subset=make_window_artifact(selected,Path('.'),cfg)
    save_checked(subset,out/'input_probe_windows.json')
    corpus=NoteAECorpus(subset,r_scaler=scaler,score_scale=score_scale)
    input_probe=[]
    for split in ('train','validation','test'):
        dataset=corpus.dataset(split)
        loader=DataLoader(dataset,batch_size=3,collate_fn=corpus.collate)
        for bi,batch in enumerate(loader):
            assert batch['R'].shape[-1]==11 and batch['M'].shape==batch['R'].shape
            assert torch.isfinite(batch['R']).all() and not batch['M'][~batch['P']].any()
            assert torch.equal(batch['lengths'],batch['P'].sum(1))
            input_probe.append({'split':split,'shape':list(batch['R'].shape),'padding':int((~batch['P']).sum())})
            if bi>=1: break
    dump(out/'input_probe.json',{'records':len(selected),'batches':input_probe,'AE_or_loss_executed':False})
    raw=read_log(out/'raw.jsonl')
    summary={'metadata_performances':len(metadata),'screening':dict(Counter(r['status'] for r in results)),
        'raw_scan':json.loads((out/'scan_summary.json').read_text()),
        'raw_notes':sum(r.get('notes',0) for r in raw),'raw_unknown_offsets':sum(r.get('unknown_offsets',0) for r in raw),
        'records_with_CC64':sum(r.get('sustain_events',0)>0 for r in raw),'records_with_CC67':sum(r.get('soft_events',0)>0 for r in raw),'R_records':len(entries),
        'source_works':len({r['composition_id'] for r in accepted}),'window_rows':notes,'full_windows':full,'tails':tails,
        'R_valid_counts':global_coverage.tolist(),'R_scaler_train_counts':scaler.valid_counts,
        'batch_records':len(batch_outcomes),'input_wrapper_probe':input_probe,'score_context_sources':len(encoded),'g_profiles':len(mini_profiles),
        'g_feature_counts':{name:sum(p['g_mask'][j] for p in mini_profiles) for j,name in enumerate(G_NAMES)},
        'g_core_eligible':sum(p['sampling_eligible'] for p in mini_profiles),'g_active_features':gs['trained_feature_mask'],
        'calibration_status':calibration['status'],'calibration_signatures':{k:{n:v for n,v in d.items() if n!='pairs'} for k,d in calibration['signatures'].items()},
        'triplets':len(triplets['triplets']),'triplet_outcomes':triplets['reason_counts'],
        'triplets_per_work':dict(Counter(t['anchor']['work_id'] for t in triplets['triplets'])),
        'split_protocol':json.loads((out/'splits.json').read_text())['policy'],
        'scaled_train_mean':np.divide(scaled_sum,scaled_count,out=np.zeros(11),where=scaled_count>0).tolist() if state.get('moments_observed',True) else None,
        'scaled_train_second_moment':np.divide(scaled_sq,scaled_count,out=np.zeros(11),where=scaled_count>0).tolist() if state.get('moments_observed',True) else None,
        'R_source_checksums_preserved':True,'AE_or_loss_executed':False,
        'alignment_is_ground_truth':False,'alignment_policy':'existing DualDTW quality gates, exact XML-derived score MIDI bridge; no guessed bundled MIDI correspondence',
        'scope':'all metadata raw files screened; all eligible/accepted automatic alignments processed; unsupported/missing/quality/resource cases explicitly excluded'}
    dump(out/'batch_checks.json',batch_outcomes);dump(out/'final_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('R_records','batch_records','full_windows','tails','g_core_eligible','triplets')},ensure_ascii=False))


def finalize_cached(args, metadata):
    """Reuse completed, hashed R/window/profile checks; resume only metric outputs."""
    out=args.out
    window_index=load_style_artifact(out/'window_index.json','window-index')
    profile_index=load_style_artifact(out/'profile_index.json','profile-index')
    ref_index=load_style_artifact(out/'reference_index.json','reference-index')
    cfg=StyleConfig(**profile_index['config'])
    gs=load_style_artifact(out/'g_scaler.json','profile-scaler',cfg)
    if gs['parents']['profiles']!=profile_index['id'] or profile_index['parents']!={'windows':window_index['id'],'reference':ref_index['id']}:
        raise ValueError('Cached finalization dependency mismatch')
    entries=window_index['sources'];by_id={e['performance_id']:e for e in entries}
    results=read_log(out/'alignments.jsonl');accepted=[r for r in results if r['status']=='extracted']
    if len(accepted)!=len(entries):raise ValueError('Accepted inputs changed after profile extraction')
    scaler=load_note_performance_scaler(out/'R_scaler.json')
    if scaler.fit_manifest_id!=window_index['id']:raise ValueError('R scaler/window source mismatch')
    score_scale=load_note_score_scale(out/'S_scale.json')
    mini_profiles=[];notes=full=tails=0;encoded=set()
    keep=('work_id','score_version_id','repeat_route_id','performance_id','recording_id','split',
          'same_window_key','window_id','q_start','q_end','tail','N','R_fingerprint','dataset','record_key')
    for file in profile_index['files']:
        path=Path(file['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=file['sha256']:raise ValueError('Profile file changed')
        for p in json.loads(path.read_text()):
            w=p['window'];e=by_id[w['performance_id']]
            notes+=w['N'];full+=int(not w['tail']);tails+=int(w['tail'])
            if not w['tail'] and w['N']:encoded.add(w['score_ref'])
            mw={k:w[k] for k in keep};mw.update(windows_file=e['windows_file'],performance_cache=e['performance_cache'])
            mini_profiles.append({**p,'window':mw})
    train=[p for p in mini_profiles if p['window']['split']=='train']
    from features.note_performance import R_CHANNELS
    global_coverage=np.array([sum(r['coverage']['channels'][name]['valid'] for r in accepted) for name in R_CHANNELS])
    batch_outcomes=read_log(out/'input_batches.jsonl')
    if len(batch_outcomes)!=len(entries) or any(r['status']!='passed' for r in batch_outcomes):
        raise ValueError('Finish all pre-AE input batches before cached metric finalization')
    scaled_sum=scaled_sq=scaled_count=np.zeros(11)
    moments_observed=False
    return complete_metric_report(args,metadata,locals())
