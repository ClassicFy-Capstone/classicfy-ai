import importlib.util
from pathlib import Path

import numpy as np
import pytest

AI=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('selected_era_listening',AI/'scripts/prepare_selected_era_listening.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_pinned_ranker_keeps_same_work_far_measured_from_anchor():
    helper,_=module.original_helpers()
    rows=[dict(performance_id=p,work_id=w,recording_id=r) for p,w,r in
          [('A','workA','a'),('B','workB','b'),('C','workB','c'),('D','workB','d')]]
    angles=np.array([0.,.2,-.3,.4]);vectors=np.column_stack([np.cos(angles),np.sin(angles)])
    b,c,pool,_,_=helper['select_near_far'](rows,vectors,0)
    assert b==1 and c==3 and set(pool)=={2,3}
    assert (1-vectors[b]@vectors[2])>(1-vectors[b]@vectors[3])


def test_nearest_single_recording_work_is_not_replaced_with_second_nearest_work():
    helper,_=module.original_helpers()
    rows=[dict(performance_id=p,work_id=w,recording_id=p) for p,w in
          [('A','a'),('B','b'),('C','c'),('D','c')]]
    angles=np.array([0.,.1,.3,.6]);vectors=np.column_stack([np.cos(angles),np.sin(angles)])
    with pytest.raises(ValueError,match='no distinct'):
        helper['select_near_far'](rows,vectors,0)


def test_mapping_identity_and_exact_excerpt_are_required():
    clip=dict(source_sha256='source',composition_id='work',start_seconds=12.25,end_seconds=42.5)
    row={**clip,'perf_id':'00001','youtube_match_status':'matched_auto',
         'youtube_start_seconds':'71.1234','youtube_end_seconds':'101.9876'}
    assert module.eligible_mapping(row,clip)
    assert not module.eligible_mapping({**row,'youtube_match_status':'review_required'},clip)
    with pytest.raises(ValueError):module.eligible_mapping({**row,'source_sha256':'another'},clip)
    with pytest.raises(ValueError):module.eligible_mapping({**row,'start_seconds':13},clip)
