"""Remove out-of-range normal MIDI candidates; retain exclusions outside handoff."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def apply_filter(root):
    root=root.resolve();outside=(root.parent/'atepp_representative_v2_duration_excluded').resolve()
    if not root.is_dir() or not outside.is_relative_to(root.parent) or outside.is_relative_to(root):raise ValueError('unsafe_root')
    with (root/'segments.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    excluded=[r for r in rows if r['status']=='exported' and (float(r['duration_seconds'])<10 or float(r['duration_seconds'])>60)]
    if not excluded:raise ValueError('No new exclusions; inspect existing filter metadata')
    prior=json.loads((root/'verification.json').read_text(encoding='utf-8'))
    if prior['errors']:raise ValueError('Prior export is not verified')
    plans=[]
    for row in excluded:
        rel=Path(row['clip_relative_path']);src=(root/rel).resolve();dst=(outside/rel).resolve()
        if not src.is_relative_to(root/'midi') or not dst.is_relative_to(outside) or not src.is_file() or dst.exists():raise ValueError('unsafe_or_missing_clip')
        if sha(src)!=row['clip_sha256']:raise ValueError('clip_changed_before_filter')
        plans.append((row,src,dst))
    outside.mkdir(parents=True,exist_ok=True)
    backup=outside/'metadata_before_filter';backup.mkdir(exist_ok=True)
    for name in ['segments.csv','performances.csv','review_queue.csv','summary.json','verification.json','README.md','run_manifest.json']:
        shutil.copy2(root/name,backup/name)
    saved=[];changes={}
    for row,src,dst in plans:
        dst.parent.mkdir(parents=True,exist_ok=True);src.replace(dst)
        reason='under_10_seconds' if float(row['duration_seconds'])<10 else 'over_60_seconds'
        saved.append(dict(row,exclusion_reason=reason))
        row.update(status='excluded_duration',previous_clip_relative_path=row['clip_relative_path'],
                   clip_relative_path='',exclusion_reason=reason)
        changes[row['perf_id'],int(row['candidate_index'])]=row
    write_csv(root/'excluded_candidates.csv',saved);write_csv(outside/'excluded_candidates.csv',saved)
    write_csv(root/'segments.csv',rows)
    affected={r['composition_id'] for r in excluded}
    for wid in affected:
        path=root/'works'/f'{wid}.json';work=json.loads(path.read_text(encoding='utf-8'))
        shutil.copy2(path,backup/f'work_{wid}.json')
        for record in work['segments']:
            key=(record['perf_id'],int(record['candidate_index']))
            if key in changes:
                r=changes[key];record.update(status='excluded_duration',clip_relative_path='',
                    previous_clip_relative_path=r['previous_clip_relative_path'],exclusion_reason=r['exclusion_reason'])
        work['duration_filter_applied_to_normal_midi']=True;dump(path,work)
    groups={}
    for r in rows:groups.setdefault((r['perf_id'],r['composition_id']),[]).append(r)
    perf=[]
    for (pid,wid),rs in sorted(groups.items()):
        c=Counter(r['status'] for r in rs)
        perf.append(dict(perf_id=pid,composition_id=wid,candidate_count=len(rs),exported_count=c['exported']+c['exported_review'],
                         accepted_exported_count=c['exported'],review_exported_count=c['exported_review'],excluded_duration_count=c['excluded_duration']))
    write_csv(root/'performances.csv',perf)
    review=[r for r in rows if r['status']!='exported' or 'source_quality_flag' in r.get('review_reasons','')]
    write_csv(root/'review_queue.csv',review)
    active=[r for r in rows if r['status'] in ('exported','exported_review')]
    physical={p.relative_to(root).as_posix() for folder in ('midi','midi_review') for p in (root/folder).rglob('*.mid')}
    if physical!={r['clip_relative_path'] for r in active}:raise ValueError('files_vs_manifest_mismatch')
    normal=[r for r in active if r['status']=='exported']
    if not all(10<=float(r['duration_seconds'])<=60 for r in normal):raise ValueError('filter_failed')
    counts=dict(Counter(r['status'] for r in rows))
    details=dict(applied_to='midi_only',minimum_seconds_inclusive=10,maximum_seconds_inclusive=60,
        excluded=len(excluded),under_10=sum(r['exclusion_reason']=='under_10_seconds' for r in saved),
        over_60=sum(r['exclusion_reason']=='over_60_seconds' for r in saved),remaining_normal_midi=len(normal))
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    summary.update(statuses=counts,midi_files=len(active),duration_filter=details);dump(root/'summary.json',summary)
    check=dict(prior,statuses=counts,midi_files=len(active),
        independent_content_checked_notes=sum(int(float(r['independently_checked_notes'])) for r in active),
        duration_filter=details,base_verification_sha256=sha(backup/'verification.json'),
        verification_stage='duration_filter_after_verified_export',
        retained_content_checks_from_prior_verified_export=True,errors=0,error_details=[])
    dump(root/'verification.json',check);dump(outside/'filter_manifest.json',details)
    print(json.dumps(details,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    apply_filter(parser.parse_args().root)
