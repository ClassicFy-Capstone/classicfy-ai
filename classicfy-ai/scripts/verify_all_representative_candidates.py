"""Verify all-candidate delivery coverage, paths, content-check records and hashes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_all_representative_candidates import dump, sha


def verify(root, full_corpus=True):
    root=root.resolve();summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    source_root=Path(summary['input_root'])
    data=pd.read_csv(root/'segments.csv',dtype={'perf_id':str,'composition_id':str,'reference_perf_id':str}).fillna('')
    audit=pd.read_csv(root/'input_midi_audit.csv',dtype={'perf_id':str,'composition_id':str}).fillna('')
    candidate_csv=pd.read_csv(root/'candidates.csv',dtype={'composition_id':str})
    perf_csv=pd.read_csv(root/'performances.csv',dtype={'perf_id':str,'composition_id':str})
    errors=[];expected=set();source_changed=0;notes=0
    def fail(reason,**values):errors.append(dict(reason=reason,**values))
    # Every file is still hashed; independent I/O overlaps on Windows.
    requests={Path(p) for p in audit.local_midi}
    requests.update(root/p for p in data.loc[data.status.isin(['exported','exported_review']),'clip_relative_path'])
    digests={}
    def file_digest(path):return sha(path) if path.is_file() else None
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(file_digest,path):path for path in requests}
        for index,future in enumerate(as_completed(jobs),1):
            digests[jobs[future]]=future.result()
            if index%5000==0 or index==len(jobs):print(f'file hashes {index}/{len(jobs)}',flush=True)
    if full_corpus:
        official=pd.read_csv(Path(__file__).resolve().parents[3]/'ATEPP_dataset/ATEPP-metadata-1.2.csv',dtype={'perf_id':str})
        if set(audit.perf_id)!=set(official.perf_id) or len(audit)!=len(official):fail('official_performance_coverage_mismatch')
    for source_index,source in enumerate(audit.to_dict('records'),1):
        if digests[Path(source['local_midi'])]!=source['sha256']:
            source_changed+=1;fail('source_changed_from_audit',perf_id=source['perf_id'])
        if source_index%2000==0:print(f'source hashes {source_index}/{len(audit)}',flush=True)
    for work_index,(wid,group) in enumerate(audit.groupby('composition_id'),1):
        work=json.loads((root/'works'/f'{wid}.json').read_text(encoding='utf-8'))
        frozen_path=source_root/'works'/f'{wid}.json'
        frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
        if sha(frozen_path)!=work['input_work_sha256']:fail('frozen_work_changed',composition_id=wid)
        if work['candidates']!=frozen['candidates'] or work['mappings']!=frozen['mappings']:fail('candidates_or_maps_changed',composition_id=wid)
        if work['automatic_selection_applied'] or 'selection' in work:fail('unexpected_final_selection',composition_id=wid)
        if work['signature']!=summary['signature']:fail('mixed_export_versions',composition_id=wid)
        sub=data[data.composition_id==wid]
        for pid in group.perf_id:
            for index in range(len(work['candidates'])):expected.add((wid,pid,index))
        for row in sub.to_dict('records'):
            pid=row['perf_id'];index=int(row['candidate_index'])
            mappings={m['candidate_index']:m for m in work['mappings'].get(pid,[])}
            mapping=mappings.get(index)
            if not mapping:
                if row['status']!='mapping_missing':fail('unreported_missing_mapping',perf_id=pid,candidate_index=index)
                continue
            if not np.allclose([row['start_seconds'],row['end_seconds']],
                [mapping['start_seconds'],mapping['end_seconds']],rtol=0,atol=1e-8):fail('mapped_time_mismatch',perf_id=pid,candidate_index=index)
            if bool(row['automatic_selection_applied']):fail('unexpected_selection_flag',perf_id=pid)
            if row['status'] in ('exported','exported_review'):
                expected_folder='midi' if mapping['alignment_accepted'] else 'midi_review'
                expected_path=f'{expected_folder}/{wid}/{pid}/candidate_{index+1:02d}.mid'
                path=root/row['clip_relative_path']
                if row['clip_relative_path']!=expected_path:fail('nonportable_or_wrong_path',perf_id=pid,candidate_index=index)
                if digests.get(path)!=row['clip_sha256']:fail('clip_hash_mismatch',perf_id=pid,candidate_index=index)
                if row['export_validation']!='passed':fail('content_check_not_passed',perf_id=pid,candidate_index=index)
                if (row['status']=='exported')!=bool(mapping['alignment_accepted']):fail('uncertain_mapping_not_separated',perf_id=pid,candidate_index=index)
                notes+=int(row['independently_checked_notes'])
            elif row['clip_relative_path']:fail('nonexported_row_has_clip',perf_id=pid,candidate_index=index)
            if row['status']=='invalid_interval' and mapping['end_seconds']>mapping['start_seconds']>=0:
                fail('valid_interval_marked_invalid',perf_id=pid,candidate_index=index)
        if work_index%250==0:print(f'verified works {work_index}/{audit.composition_id.nunique()}',flush=True)
    actual=list(zip(data.composition_id,data.perf_id,data.candidate_index.astype(int)))
    if set(actual)!=expected or len(actual)!=len(expected):fail('performance_candidate_coverage_mismatch')
    if set(perf_csv.perf_id)!=set(audit.perf_id) or len(perf_csv)!=len(audit):fail('performance_summary_coverage_mismatch')
    expected_work_candidates={(w,i) for w,p,i in expected}
    if set(zip(candidate_csv.composition_id,candidate_csv.candidate_index.astype(int)))!=expected_work_candidates:
        fail('work_candidate_table_mismatch')
    paths=[p.relative_to(root).as_posix() for folder in ('midi','midi_review') for p in (root/folder).rglob('*.mid')]
    recorded=data[data.status.isin(['exported','exported_review'])].clip_relative_path.tolist()
    if set(paths)!=set(recorded) or len(paths)!=len(recorded):fail('physical_files_vs_manifest_mismatch')
    result=dict(works=audit.composition_id.nunique(),performances=len(audit),work_candidates=len(expected_work_candidates),
        accounted_performance_candidates=len(data),expected_performance_candidates=len(expected),
        statuses=data.status.value_counts().to_dict(),midi_files=len(paths),
        independent_content_checked_notes=notes,source_hashes_rechecked=len(audit),source_changed=source_changed,
        automatic_selection_applied=False,errors=len(errors),error_details=errors)
    dump(root/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args();verify(args.root,full_corpus=not args.partial)
