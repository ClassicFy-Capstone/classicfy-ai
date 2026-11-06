"""Final coverage/provenance checks; per-file MIDI content checks run at export."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def verify(root):
    workspace=Path(__file__).resolve().parents[3];root=root.resolve()
    data=pd.read_csv(root/'segments.csv',dtype={'perf_id':str,'composition_id':str}).fillna('')
    audit=pd.read_csv(workspace/'capstone-2026-2-classicfy/classicfy-ai/analysis/atepp_research/midi_audit.csv',dtype={'perf_id':str,'composition_id':str}).fillna('').set_index('perf_id')
    official=pd.read_csv(workspace/'ATEPP_dataset/ATEPP-metadata-1.2.csv',dtype={'perf_id':str})
    signature=json.loads((root/'config.json').read_text())['source_sha256']
    errors=[];cache={};checked_notes=0;source_changes=0
    if len(data)!=len(official) or data.perf_id.nunique()!=len(official) or set(data.perf_id)!=set(official.perf_id):
        errors.append(dict(reason='official_input_coverage_mismatch'))
    for row in data.to_dict('records'):
        source=Path(audit.loc[row['perf_id'],'local_midi'])
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!=audit.loc[row['perf_id'],'sha256']:
            source_changes+=1;errors.append(dict(perf_id=row['perf_id'],reason='source_changed_from_audit'))
        if row['status']!='exported':continue
        path=Path(row['clip_path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['clip_sha256']:
            errors.append(dict(perf_id=row['perf_id'],reason='clip_hash_mismatch'))
        if row['export_validation']!='passed':errors.append(dict(perf_id=row['perf_id'],reason='content_check_not_passed'))
        checked_notes+=int(row['note_count'])
        work=row['composition_id']
        if work not in cache:cache[work]=json.loads((root/'works'/f'{work}.json').read_text(encoding='utf-8'))
        record=cache[work]
        if record['signature']!=signature:errors.append(dict(perf_id=row['perf_id'],reason='mixed_code_versions'))
        chosen=record['selection']['selected_candidate'];mapping=record['mappings'][row['perf_id']][chosen]
        if int(row['candidate_index'])!=chosen or not mapping['alignment_accepted']:
            errors.append(dict(perf_id=row['perf_id'],reason='not_common_selected_candidate'))
        if not np.allclose([row['start_seconds'],row['end_seconds']],
                           [mapping['start_seconds'],mapping['end_seconds']],rtol=0,atol=1e-8):
            errors.append(dict(perf_id=row['perf_id'],reason='mapped_time_mismatch'))
    physical=list((root/'midi').rglob('*.mid'))
    successful=data[data.status=='exported']
    if {p.stem for p in physical}!=set(successful.perf_id) or len(physical)!=len(successful):
        errors.append(dict(reason='physical_files_vs_manifest_mismatch'))
    result=dict(official_performances=len(official),accounted_performances=len(data),works=data.composition_id.nunique(),
                statuses=data.status.value_counts().to_dict(),exported_midi_files=len(physical),
                content_checked_notes=checked_notes,source_hashes_rechecked=len(data),source_changed_from_audit=source_changes,
                errors=len(errors),error_details=errors,identity_or_preference_accuracy_evaluated=False)
    (root/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if errors:raise SystemExit(1)


if __name__=='__main__':verify(Path(sys.argv[1]))
