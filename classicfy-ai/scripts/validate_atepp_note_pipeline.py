"""Resumable real ATEPP pre-AE validation; never run an AE or a loss.

scan inventories every metadata MIDI and XML. align builds exact XML-derived
score bridges and applies the existing quality-gated automatic note matcher.
finish exercises R/cache/scalers/batches and the frozen g/triplet pipeline.
All derived files and validation-only splits live under --out.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.environ.get('CLASSICFY_VALIDATION_SRC', str(Path(__file__).resolve().parents[1] / 'src')))
BAD_QUALITY = {'corrupted', 'low quality', 'background noise', 'applause'}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_log(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    key = {'raw.jsonl':'perf_id','scores.jsonl':'score_path','alignments.jsonl':'perf_id'}.get(path.name)
    return list({row[key]:row for row in rows}.values()) if key else rows


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def raw_worker(task):
    from preprocessing.performance_midi import load_performance_midi
    root, row = task
    result = {'perf_id': row['perf_id'], 'composition_id': row['composition_id'], 'quality': row['quality']}
    try:
        r = load_performance_midi(Path(root) / row['midi_path'])
        result.update(status='parsed', notes=len(r.notes), sha256=r.provenance['midi_sha256'],
            channels=r.selected_channels, duration=r.duration_sec, unknown_offsets=sum(n.key_offset_sec is None for n in r.notes),
            zero_durations=sum(n.key_offset_sec == n.onset_sec for n in r.notes),
            sustain_events=sum(c.number == 64 for c in r.controls), soft_events=sum(c.number == 67 for c in r.controls),
            diagnostics=dict(Counter(d.split(':')[0] for d in r.diagnostics)))
    except Exception as exc:
        result.update(status='rejected', error=f'{type(exc).__name__}: {exc}')
    return result


def score_worker(task):
    from preprocessing.musicxml_score import read_musicxml
    from preprocessing.score_midi import read_score_midi
    from preprocessing.xml_score_midi_bridge import render_xml_score_bridge
    from features.note_score_context import extract_note_score_context
    from embedding.note_score_context_data import save_note_score_context
    root, out, relative = task
    path = Path(root) / relative
    result = {'score_path': relative, 'key': hashlib.sha256(relative.encode()).hexdigest()[:20]}
    dest = Path(out) / 'scores' / result['key']
    try:
        x = read_musicxml(path)
        result.update(parsed_xml=True, notes=len(x.notes), visits=len(x.visits), source_sha256=x.source_sha256)
        s = extract_note_score_context(x)
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / 'S').exists():
            save_note_score_context(s, dest / 'S')
        canonical = dest / 'canonical.mid'
        if canonical.exists():
            canonical.unlink()  # This run's own derived file, never the bundled source.
        bridge = render_xml_score_bridge(x, canonical)
        dump(dest / 'bridge.json', bridge)
        result.update(status='prepared', score_midi=str(canonical), bridge=str(dest / 'bridge.json'),
                      score_cache=str(dest / 'S'), aliases=len(bridge['score_id_to_occurrence']),
                      bridge_excluded=bridge['excluded'])
        bundled = Path(str(path) + '.midi')
        if bundled.is_file():
            notes, _, diagnostics = read_score_midi(bundled)
            known = {(n.pitch, float(n.onset)) for n in x.notes}
            result['bundled_midi_audit'] = {'notes': len(notes), 'exact_pitch_onset_hits': sum((n.pitch, n.onset_beats) in known for n in notes),
                                          'diagnostics': diagnostics, 'used_for_alignment': False}
    except Exception as exc:
        result.update(status='rejected', error=f'{type(exc).__name__}: {exc}')
    return result


def map_work_splits(rows, raw, out):
    # Separate diagnostic protocol: never rewrite an existing experiment split.
    # Preserve known ASAP work assignments where an explicit overlap match exists.
    ai = Path(__file__).resolve().parents[1]
    frozen = ai.parents[1] / 'datasets/temporal_embedding_cluster_run01/dataset.json'
    overlap = ai / 'analysis/dataset_compatibility/asap_atepp_overlap/work_crosswalk.csv'
    inherited = defaultdict(set)
    if frozen.is_file() and overlap.is_file():
        old = json.loads(frozen.read_text())['split']
        with overlap.open(encoding='utf-8-sig') as f:
            for record in csv.DictReader(f):
                if record['status'] != 'same_unit':
                    continue
                known = set()
                for path in record['asap_score_paths'].split(';'):
                    midi_key = str(Path(path.strip()).with_name('midi_score.mid'))
                    if midi_key in old:
                        known.add(old[midi_key])
                for work in record['atepp_composition_ids'].split(';'):
                    inherited[work.strip()].update(known)
    # Group identical XML and identical recording bytes before assigning works.
    parent = {r['composition_id']: r['composition_id'] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)] = min(a,b)
    by_xml, by_hash = {}, {}
    score_hashes = {r['score_path']:r.get('source_sha256',r['score_path']) for r in read_log(out/'scores.jsonl')}
    raw_by_id = {r['perf_id']: r for r in raw}
    for r in rows:
        work=r['composition_id']
        for mapping, key in ((by_xml, score_hashes.get(r['score_path'],r['score_path'])), (by_hash, raw_by_id.get(r['perf_id'],{}).get('sha256'))):
            if key:
                if key in mapping: union(work,mapping[key])
                mapping[key]=work
    sets=defaultdict(list)
    for work in parent: sets[find(work)].append(work)
    assignments, conflicts = {}, {}
    for group, works in sets.items():
        known=set().union(*(inherited[w] for w in works))
        if len(known)>1:
            conflicts[group]=sorted(known)
            split='unassigned'
        elif known: split=next(iter(known))
        else:
            rank=int(hashlib.sha256(('2026:ATEPP:'+group).encode()).hexdigest()[:8],16)%100
            split='train' if rank<70 else 'validation' if rank<85 else 'test'
        for work in works: assignments[work]={'split':split,'work_key':'ATEPP/'+group}
    result={'policy':'isolated validation protocol; known frozen ASAP assignments inherited; other work groups stable SHA256 70/15/15; conflicts unassigned',
            'seed':2026,'assignments':assignments,'conflicts':conflicts,'existing_experiment_splits_modified':False}
    dump(out/'splits.json',result)
    return result


def scan(args, rows):
    args.out.mkdir(parents=True, exist_ok=True)
    for name, tasks, worker, field in [
        ('raw', [(str(args.root),row) for row in rows], raw_worker, 'perf_id'),
        ('scores', [(str(args.root),str(args.out),p) for p in sorted({r['score_path'] for r in rows if r['score_path']})], score_worker,'score_path')]:
        path=args.out/(name+'.jsonl'); old=read_log(path);done={r[field] for r in old if not (name=='scores' and args.retry_rejected and r['status']=='rejected')}
        tasks=[t for t in tasks if (t[1][field] if name=='raw' else t[2]) not in done]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(worker,task) for task in tasks]
            for i,future in enumerate(as_completed(futures),1):
                row=future.result();append(path,row)
                if i%100==0 or i==len(tasks): print(name,len(done)+i,'/',len(done)+len(tasks),flush=True)
    raw, scores=read_log(args.out/'raw.jsonl'),read_log(args.out/'scores.jsonl')
    map_work_splits(rows,raw,args.out)
    dump(args.out/'scan_summary.json',{'raw':dict(Counter(r['status'] for r in raw)),
        'scores':dict(Counter(r['status'] for r in scores)), 'raw_errors':dict(Counter(r.get('error') for r in raw if r['status']!='parsed')),
        'score_errors':dict(Counter(r.get('error') for r in scores if r['status']!='prepared'))})


def align_worker(task):
    os.environ['CLASSICFY_TORCH_THREADS']='1'
    from preprocessing.performance_midi import load_performance_midi
    from preprocessing.midi_loader import MidiData, NoteEvent
    from preprocessing.score_midi import load_score_midi
    from preprocessing.research_alignment import align_research
    from preprocessing.note_performance_adapters import prepare_atepp_note_performance
    from embedding.note_score_context_data import load_note_score_context
    from embedding.note_performance_data import save_note_performance, load_note_performance
    from features.note_performance import inverse_r_physical, summarize_note_performance
    import numpy as np
    root, out, row, source, split, max_cells=task
    result={'perf_id':row['perf_id'],'composition_id':row['composition_id'],'split':split['split']}
    started=time.monotonic()
    try:
        performance=load_performance_midi(Path(root)/row['midi_path'])
        # The published matcher uses raw key releases, not pedal-extended offsets.
        usable=[n for n in performance.notes if n.key_offset_sec is not None and n.key_offset_sec>n.onset_sec]
        midi=MidiData([NoteEvent(n.pitch,n.onset_sec,n.key_offset_sec,n.velocity,0,n.program) for n in usable],[],performance.duration_sec)
        score=load_score_midi(source['score_midi'])
        if len(score)*len(usable)>max_cells:
            result.update(status='resource_limit',error=f'DualDTW note product exceeds {max_cells} cells; no approximate fallback')
            return result
        alignment,q,seconds,metrics=align_research(score,midi,'dual_dtw')
        result['alignment_metrics']=metrics
        if not metrics['accepted']:
            result.update(status='alignment_rejected',error=metrics['reject_reasons']);return result
        dest=Path(out)/'performances'/row['perf_id'];dest.mkdir(parents=True,exist_ok=True)
        cache=dest/'alignment.npz'
        np.savez_compressed(cache,score_ids=np.array([s.note_id for s,p in alignment.matches]),
            performance_ids=np.array([p.note_id for s,p in alignment.matches]),
            score_onsets=np.array([s.onset_beats for s,p in alignment.matches]),score_offsets=np.array([s.offset_beats for s,p in alignment.matches]),
            performance_onsets=np.array([p.onset for s,p in alignment.matches]),performance_offsets=np.array([p.offset for s,p in alignment.matches]),
            pitch=np.array([s.pitch for s,p in alignment.matches]),score_beats=q,performance_beats=seconds)
        bridge=json.loads(Path(source['bridge']).read_text());bridge['alignment_accepted']=True
        bridge['alignment_metrics']=metrics;dump(dest/'bridge.json',bridge)
        features,_=load_note_score_context(source['score_cache'])
        s,r=prepare_atepp_note_performance(Path(root)/row['score_path'],source['score_midi'],Path(root)/row['midi_path'],cache,
            bridge,record_key=row['perf_id'],work_key=split['work_key'],split=split['split'],score_features=features)
        restored,mask=inverse_r_physical(r.values,r.mask,r.local_tau)
        np.testing.assert_allclose(restored[mask],r.physical[mask],rtol=1e-10,atol=1e-10)
        path=dest/'R'
        if not path.exists(): save_note_performance(r,path)
        assert load_note_performance(path).fingerprint==r.fingerprint
        result.update(status='extracted',performance_cache=str(path),xml_path=str(Path(root)/row['score_path']),
            work_key=split['work_key'],fingerprint=r.fingerprint,coverage=summarize_note_performance(r),
            alignment_audit=r.provenance['alignment'])
    except Exception as exc:
        result.update(status='error',error=f'{type(exc).__name__}: {exc}')
    finally:
        result['seconds']=time.monotonic()-started
    return result


def align(args, rows):
    raw={r['perf_id']:r for r in read_log(args.out/'raw.jsonl')}
    scores={r['score_path']:r for r in read_log(args.out/'scores.jsonl')}
    splits=json.loads((args.out/'splits.json').read_text())['assignments']
    path=args.out/'alignments.jsonl';old=read_log(path)
    rows_by_id={r['perf_id']:r for r in rows}
    done={r['perf_id'] for r in old if not (args.retry_resources and r['status']=='resource_limit') and not (r['status']=='score_unsupported'
          and scores.get(rows_by_id[r['perf_id']]['score_path'],{}).get('status')=='prepared')}
    tasks=[]
    for r in rows:
        if r['perf_id'] in done: continue
        reason=None
        if raw[r['perf_id']]['status']!='parsed': reason='raw_midi_rejected'
        elif not r['score_path']: reason='no_XML'
        elif scores[r['score_path']]['status']!='prepared': reason='score_unsupported'
        elif r['quality'] in BAD_QUALITY: reason='quality_label_excluded'
        elif splits[r['composition_id']]['split']=='unassigned': reason='frozen_split_conflict'
        if reason:
            append(path,{'perf_id':r['perf_id'],'composition_id':r['composition_id'],'status':reason});continue
        tasks.append((str(args.root),str(args.out),r,scores[r['score_path']],splits[r['composition_id']],args.max_note_cells))
    dump(args.out/'alignment_run_settings.json',{'workers':args.workers,'max_note_cells':args.max_note_cells,'retry_resources':args.retry_resources,'aligner':'DualDTW','quality_gates':'unchanged existing research_alignment.py'})
    print('alignment eligible:',len(tasks),flush=True)
    if args.limit: tasks=tasks[:args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(align_worker,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result();append(path,result)
            if i%10==0 or i==len(tasks): print('alignment',i,'/',len(tasks),result['status'],flush=True)
    results=read_log(path)
    dump(args.out/'alignment_summary.json',{'statuses':dict(Counter(r['status'] for r in results)),
        'errors':dict(Counter(r.get('error') for r in results if r.get('error'))),'metadata_rows':len(rows)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['scan','align','finish','finalize'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--retry-resources',action='store_true');p.add_argument('--max-note-cells',type=int,default=120_000_000);p.add_argument('--retry-rejected',action='store_true');p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int)
    args=p.parse_args();args.root=args.root.resolve();args.out=args.out.resolve()
    rows=list(csv.DictReader((args.root/'ATEPP-metadata-1.2.csv').open(encoding='utf-8-sig')))
    if args.stage=='scan': scan(args,rows)
    elif args.stage=='align': align(args,rows)
    else:
        from validate_atepp_note_pipeline_finish import finish,finalize_cached
        (finish if args.stage=='finish' else finalize_cached)(args,rows)


if __name__=='__main__': main()
