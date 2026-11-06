"""Exclude supplied performance IDs from the current, already-filtered handoff."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rs,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rs for k in r))
    temp=p.with_suffix(p.suffix+'.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
    temp.replace(p)
def dump(p,obj):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)


def run(root,id_file):
    root=root.resolve();id_file=id_file.resolve();archive=root.with_suffix('.zip')
    tokens=[s.strip() for s in id_file.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if not all(re.fullmatch(r'\d{1,5}',s) for s in tokens):raise ValueError('IDs must be numeric lines')
    ids={s.zfill(5) for s in tokens};rows=read(root/'segments.csv');audit=read(root/'input_midi_audit.csv')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    prior=json.loads((root/'verification.json').read_text(encoding='utf-8'))
    if prior['errors']:raise ValueError('Baseline verification failed')
    with zipfile.ZipFile(archive) as z:
        for name in ['segments.csv','summary.json','input_midi_audit.csv','candidates.csv']:
            if z.read(root.name+'/'+name)!=(root/name).read_bytes():raise ValueError('Folder differs from latest ZIP')
    excluded=[r for r in rows if r['perf_id'] in ids];kept=[r for r in rows if r['perf_id'] not in ids]
    matched={r['perf_id'] for r in excluded}
    if not matched:raise ValueError('No matching performances')
    outside=(root.parent/'atepp_representative_v2_id_excluded').resolve()
    if outside.exists():raise ValueError('Exclusion backup already exists; inspect before retry')
    if outside.is_relative_to(root) or not outside.is_relative_to(root.parent):raise ValueError('Unsafe exclusion folder')
    plans=[]
    for r in excluded:
        if r['status'] not in ('exported','exported_review'):continue
        rel=Path(r['clip_relative_path']);src=(root/rel).resolve();dst=(outside/rel).resolve()
        if not any(src.is_relative_to(root/folder) for folder in ['midi','midi_review']) or not dst.is_relative_to(outside):raise ValueError('Unsafe candidate path')
        if not src.is_file() or sha(src)!=r['clip_sha256']:raise ValueError('Candidate hash mismatch')
        plans.append((src,dst))
    outside.mkdir();backup=outside/'metadata_before_filter';backup.mkdir()
    for p in root.iterdir():
        if p.is_file():shutil.copy2(p,backup/p.name)
    shutil.copy2(id_file,outside/'excluded_performance_ids.txt')
    for src,dst in plans:dst.parent.mkdir(parents=True,exist_ok=True);src.replace(dst)
    write(outside/'excluded_candidate_records.csv',excluded)
    write(root/'segments.csv',kept)
    audit=[r for r in audit if r['perf_id'] not in ids];write(root/'input_midi_audit.csv',audit)
    active_works={r['composition_id'] for r in audit}
    removed_works=[]
    for p in (root/'works').glob('*.json'):
        w=json.loads(p.read_text(encoding='utf-8'));wid=w['work']
        if not any(pid in ids for pid in w['mappings']) and wid in active_works:continue
        shutil.copy2(p,backup/f'work_{wid}.json')
        if wid not in active_works:
            dest=outside/'removed_works'/p.name;dest.parent.mkdir(exist_ok=True);p.replace(dest);removed_works.append(wid);continue
        w['mappings']={pid:ms for pid,ms in w['mappings'].items() if pid not in ids}
        w['segments']=[r for r in w['segments'] if r['perf_id'] not in ids]
        w['performance_id_filter_applied']=True;dump(p,w)
    candidates=[r for r in read(root/'candidates.csv') if r['composition_id'] in active_works];write(root/'candidates.csv',candidates)
    if (root/'excluded_candidates.csv').exists():
        old=read(root/'excluded_candidates.csv');write(root/'excluded_candidates.csv',[r for r in old if r['perf_id'] not in ids],list(old[0]))
    groups={}
    for r in kept:groups.setdefault((r['perf_id'],r['composition_id']),[]).append(r)
    perf=[]
    for (pid,wid),rs in sorted(groups.items()):
        c=Counter(r['status'] for r in rs)
        perf.append(dict(perf_id=pid,composition_id=wid,candidate_count=len(rs),exported_count=c['exported']+c['exported_review'],accepted_exported_count=c['exported'],review_exported_count=c['exported_review'],excluded_duration_count=c['excluded_duration']))
    write(root/'performances.csv',perf)
    write(root/'review_queue.csv',[r for r in kept if r['status']!='exported' or 'source_quality_flag' in r.get('review_reasons','')])
    active=[r for r in kept if r['status'] in ('exported','exported_review')]
    physical={p.relative_to(root).as_posix() for folder in ['midi','midi_review'] for p in (root/folder).rglob('*.mid')}
    if physical!={r['clip_relative_path'] for r in active}:raise ValueError('File/manifest mismatch')
    if any(Path(p).parent.name in ids for p in physical) or any(r['perf_id'] in ids for r in kept):raise ValueError('Excluded ID remains')
    normal=[r for r in active if r['status']=='exported']
    if not all(10<=float(r['duration_seconds'])<=60 for r in normal):raise ValueError('Previous duration filter lost')
    detail=dict(ids_supplied=len(ids),matched_performances=len(matched),unmatched_ids=sorted(ids-matched),removed_candidate_rows=len(excluded),removed_midi_files=len(plans),removed_normal_midi=sum(r['status']=='exported' for r in excluded),removed_review_midi=sum(r['status']=='exported_review' for r in excluded),remaining_normal_midi=len(normal),remaining_review_midi=sum(r['status']=='exported_review' for r in active),removed_work_count=len(removed_works),ids=sorted(ids),id_file_sha256=sha(id_file))
    summary.update(works=len(active_works),performances=len(audit),work_candidates=len(candidates),performance_candidate_rows=len(kept),statuses=dict(Counter(r['status'] for r in kept)),midi_files=len(active),audit_sha256=sha(root/'input_midi_audit.csv'),performance_id_filter=detail)
    summary['duration_filter']['remaining_normal_midi']=len(normal)
    dump(root/'summary.json',summary);dump(outside/'filter_manifest.json',detail)
    check=dict(prior,works=len(active_works),performances=len(audit),work_candidates=len(candidates),accounted_performance_candidates=len(kept),expected_performance_candidates=len(kept),statuses=summary['statuses'],midi_files=len(active),independent_content_checked_notes=sum(int(float(r['independently_checked_notes'])) for r in active),source_hashes_rechecked=len(audit),duration_filter=summary['duration_filter'],performance_id_filter=detail,verification_stage='ID_filter_after_verified_duration_filter',base_verification_sha256=sha(backup/'verification.json'),errors=0,error_details=[])
    dump(root/'verification.json',check)
    print(json.dumps({k:v for k,v in detail.items() if k!='ids'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('ids',type=Path);a=p.parse_args();run(a.root,a.ids)
