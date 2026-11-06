"""Final corpus coverage/hash verification, after per-file independent MIDI checks."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main(root):
    root=root.resolve();workspace=Path(__file__).resolve().parents[3]
    data=pd.read_csv(root/'segments.csv',dtype={'perf_id':str,'composition_id':str})
    official=pd.read_csv(workspace/'ATEPP_dataset/ATEPP-metadata-1.2.csv',dtype={'perf_id':str})
    candidates=pd.read_csv(root/'candidates.csv',dtype={'perf_id':str})
    assert len(data)==len(official) and data.perf_id.nunique()==len(official)
    assert set(data.perf_id)==set(official.perf_id)
    assert set(candidates.perf_id)==set(official.perf_id)
    assert candidates.groupby('perf_id').selected_by_consensus.sum().eq(1).all()
    errors=[];checked_notes=0
    for row in data.to_dict('records'):
        source=Path(row['source_midi'])
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['source_sha256']:
            errors.append(dict(perf_id=row['perf_id'],reason='source_hash_changed_after_inference'))
        if row['status']!='exported':continue
        path=Path(row['clip_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(dict(perf_id=row['perf_id'],reason='missing_or_external_clip'));continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['clip_sha256']:
            errors.append(dict(perf_id=row['perf_id'],reason='clip_hash_mismatch'))
        if not (np.isfinite([row['start_seconds'],row['end_seconds']]).all()
                and 0<=row['start_seconds']<row['end_seconds']):
            errors.append(dict(perf_id=row['perf_id'],reason='invalid_interval'))
        if row['export_validation']!='passed' or row['note_count']!=row['independently_checked_notes']:
            errors.append(dict(perf_id=row['perf_id'],reason='independent_validation_not_passed'))
        checked_notes+=int(row['independently_checked_notes'])
        relative=root/row['clip_relative_path']
        if relative.resolve()!=path:errors.append(dict(perf_id=row['perf_id'],reason='portable_path_mismatch'))
    current=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    replay=root/'evaluation_before_export_optimization.json'
    unchanged=None
    if replay.exists():
        old=json.loads(replay.read_text(encoding='utf-8'))
        unchanged=old['metrics']==current['metrics'] and old['medium_confidence_cutoff']==current['medium_confidence_cutoff']
        if not unchanged:errors.append(dict(reason='model_metrics_changed_during_export_optimization'))
    result=dict(metadata_performances=len(official),covered_performances=len(data),
                source_compositions=int(data.composition_id.nunique()),
                exported_clips=int(data.status.eq('exported').sum()),failed_performances=int(data.status.ne('exported').sum()),
                independently_checked_notes=checked_notes,candidate_intervals=len(candidates),
                all_candidates_have_exactly_one_automatic_selection=True,
                model_metrics_unchanged_during_io_optimization=unchanged,
                source_hashes_rechecked_after_export=True,clip_hashes_rechecked_after_domain_flag_moves=True,
                human_musical_boundaries_verified=False,errors=len(errors),error_details=errors)
    (root/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if errors:raise SystemExit(1)


if __name__=='__main__':main(Path(sys.argv[1]))
