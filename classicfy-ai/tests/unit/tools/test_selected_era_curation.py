"""Reject misleading demonstrations and preserve conservative structure matching."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('curation',Path(__file__).resolve().parents[3]/'scripts/selected_era_curation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('gn,gf,qn,qf',[(float('nan'),8,.5,.5),(8,1,.5,.5),(1,1.01,.5,.5),(1,8,.1,.5)])
def test_large_embedding_gap_does_not_override_invalid_reversed_ambiguous_or_weak_G(gn,gf,qn,qf):
    assert not module.pair_is_clear(.05,1.2,gn,gf,qn,qf,2)


def test_clear_G_and_embedding_order_is_accepted():
    assert module.pair_is_clear(.1,.9,.2,8,.6,.6,2)
    assert not module.pair_is_clear(.9,.1,.2,8,.6,.6,2)


def test_titles_resolve_movement_and_keep_ambiguous_combined_work_unclassified():
    assert module.structure_tag('Prelude and Fugue No. 1: II. Fugue')=='fugue'
    assert module.structure_tag('Prelude and Fugue No. 1: I. Prelude')=='prelude'
    assert module.structure_tag('Chromatic Fantasia and Fugue in D minor') is None
    assert module.structure_tag('Piano Sonata: I. Allegro')=='sonata_fast_1'
    assert module.structure_tag('Piano Sonata: II. Allegretto')=='sonata_fast_2'
    assert module.structure_tag('Études Op. 10: No. 1 Prélude')=='etude'


def test_score_texture_is_based_on_score_quarters_not_performance_seconds():
    clips=[dict(composition_id='a',start=0,stop=4,notes=4)]
    assert module.texture_profiles(clips,np.array([0.,0.,1.,2.]))['a']==pytest.approx([1.,4/3])
    assert module.texture_ratios([1.,2.],[2.,1.])==[2.,2.]


def test_same_artist_candidates_cannot_produce_case():
    angles=np.array([0.,.1,1.5]);vectors=np.column_stack([np.cos(angles),np.sin(angles)])
    metadata=[dict(perf_id=p,work=w,era='test',source=p,split='train') for p,w in [('a','A'),('b','B'),('c','B')]]
    rows={p:dict(track='Invention No. 1',artist='same',youtube_url=p) for p in ['a','b','c']}
    raw=np.zeros((3,23));raw[2]=5
    scalers={'test':dict(scaler=dict(trained=[True]*23,center=[0]*23,scale=[1]*23),calibration=dict(median=2))}
    cases,_,_=module.curate(vectors,['a','b','c'],['test']*3,metadata,rows,dict.fromkeys(rows,True),set(),
        [raw,np.ones_like(raw,dtype=bool),np.ones_like(raw)],scalers,lambda x:x,dict(A=[1,1],B=[1,1]))
    assert cases==[]
