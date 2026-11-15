import numpy as np
import pytest
import torch
from embedding.selected_style_coordinates import G_PROFILE_VERSION, QUARTER_TICKS

from embedding.era_note_style import (EraStyleHead, aggregate_clip_latents,
    capped_style_weight, fit_g_scaler, g_objective, pair_tables, selected_profiles)


def test_head_starts_as_identity_and_cannot_exceed_residual_bound():
    head=EraStyleHead(np.ones(4))
    z=torch.randn(9,4)
    torch.testing.assert_close(head(z)['z_style'],z)
    with torch.no_grad():head.net[-1].bias.fill_(100)
    assert torch.all((head(z)['z_style']-z).abs()<=.500001)


def test_checkpoint_loader_preserves_weights_and_frozen_score_mode(tmp_path):
    from dataclasses import asdict
    from embedding.era_note_style import load_era_checkpoint
    from embedding.note_autoencoder import NotePerformanceAutoencoder
    model=NotePerformanceAutoencoder();head=EraStyleHead(np.ones(64))
    payload=dict(format='era-note-ae-head-v2',supervision_version=G_PROFILE_VERSION,era='baroque',model_config=asdict(model.config),
                 score_config=asdict(model.score_encoder.config),state_dict=model.state_dict(),
                 head_state_dict=head.state_dict(),head_scale=[1.]*64,scaler=dict(id='test'))
    path=tmp_path/'model.pt';torch.save(payload,path)
    restored,new_head,state=load_era_checkpoint(path)
    for key,value in model.state_dict().items():torch.testing.assert_close(value,restored.state_dict()[key])
    z=torch.randn(3,64)
    torch.testing.assert_close(head(z)['e'],new_head(z)['e'])
    restored.train()
    assert not restored.score_encoder.training
    assert not any(p.requires_grad for p in restored.score_encoder.parameters())


def test_clip_pooling_keeps_tail_and_has_exact_gradient_weights():
    z=torch.tensor([[1.,2.],[3.,4.],[100.,200.],[5.,6.]],requires_grad=True)
    owner=torch.tensor([0,0,0,1]);weights=torch.tensor([3.,1.,0.,2.])
    result=aggregate_clip_latents(z,owner,weights,2)
    torch.testing.assert_close(result,torch.tensor([[1.5,2.5],[5.,6.]]))
    result.sum().backward()
    torch.testing.assert_close(z.grad[:,0],torch.tensor([.75,.25,0.,1.]))
    with pytest.raises(ValueError):aggregate_clip_latents(z,owner,weights*0,2)


def test_style_gradient_cap_and_zero_confidence():
    z=torch.randn(5,4,requires_grad=True)
    rec=z.square().mean();style=100*z.square().mean()
    weight,diagnostic=capped_style_weight(rec,style,z,.3,.25)
    assert weight==pytest.approx(.0025,rel=1e-5)
    assert diagnostic['applied_gradient_ratio']<=.250001
    d=torch.tensor([[float('nan'),.1,2.],[.1,float('nan'),1.],[2.,1.,float('nan')]])
    e=torch.nn.functional.normalize(z[:3],dim=-1)
    loss=g_objective(e,d,torch.zeros_like(d),dict(median=1.,iqr=1.))['loss']
    assert float(loss.detach())==0
    assert torch.equal(torch.autograd.grad(loss,z)[0],torch.zeros_like(z))


def test_inactive_lambda_ceilings_produce_identical_optimizer_trajectory():
    trajectories=[]
    for ceiling in (.1,.3):
        z=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
        optimizer=torch.optim.AdamW([z],lr=.01)
        trajectory=[]
        for epoch in range(1,7):
            optimizer.zero_grad();rec=z.square().mean();style=100*z.square().mean()
            requested=ceiling*min(1.,epoch/3)
            weight,_=capped_style_weight(rec,style,z,requested)
            assert weight<requested
            (rec+weight*style).backward();optimizer.step();trajectory.append(z.detach().clone())
        trajectories.append(torch.stack(trajectory))
    torch.testing.assert_close(*trajectories,rtol=0,atol=0)


@pytest.mark.parametrize('mode,rec_coefficient,g_coefficient',[
    ('joint',1.,.1),('reconstruction_only',1.,None),('g_only',None,.1)])
def test_ablation_removes_only_requested_direct_loss_gradient(mode,rec_coefficient,g_coefficient):
    from embedding.era_note_style import compose_era_objective
    values=[torch.tensor(2.,requires_grad=True) for _ in range(4)]
    loss=compose_era_objective(*values,.1,mode)
    gradients=torch.autograd.grad(loss,values,allow_unused=True)
    for actual,expected in zip(gradients,(rec_coefficient,g_coefficient,.05,.01)):
        if expected is None:assert actual is None
        else:assert float(actual)==pytest.approx(expected)


@pytest.mark.parametrize('alpha',[0.,.01,.1,.3,1.])
def test_weak_reconstruction_weight_changes_only_direct_rec_gradient(alpha):
    from embedding.era_note_style import compose_era_objective
    values=[torch.tensor(2.,requires_grad=True) for _ in range(4)]
    gradients=torch.autograd.grad(compose_era_objective(*values,.07,'joint',alpha),values)
    for actual,expected in zip(gradients,(alpha,.07,.05,.01)):
        assert float(actual)==pytest.approx(expected)


def test_g_only_keeps_decoder_weights_frozen_but_trains_encoder_and_head():
    from embedding.era_note_style import compose_era_objective
    from embedding.note_autoencoder import ConditionalNoteAutoencoder,NoteAEConfig
    from embedding.note_ae_loss import reconstruction_loss
    torch.manual_seed(3)
    model=ConditionalNoteAutoencoder(NoteAEConfig(context_dim=4,condition_dim=4,input_dim=8,
        hidden_size=4,layers=1,dropout=0.,latent_dim=4))
    head=EraStyleHead(np.ones(4))
    context=torch.randn(3,8,4);target=torch.randn(3,8,11)
    mask=torch.ones_like(target,dtype=torch.bool);padding=torch.ones((3,8),dtype=torch.bool)
    lengths=torch.full((3,),8,dtype=torch.long);beats=torch.arange(8).expand(3,-1)
    z=model.encode(context,target,mask,padding,lengths)['z']
    prediction=model.decode(context,z,padding,lengths,beats)['R_hat']
    reconstruction=reconstruction_loss(prediction,target,mask,padding,beats)
    rec=reconstruction['loss'];styled=head(z)
    d=torch.tensor([[float('nan'),.1,4.],[.1,float('nan'),4.],[4.,4.,float('nan')]])
    g=g_objective(styled['e'],d,torch.ones_like(d),dict(median=1.,iqr=1.))['loss']
    weight,_=capped_style_weight(rec,g,z,.1)
    loss=compose_era_objective(rec,g,(z-z.detach()).square().mean(),styled['head_penalty'],weight,'g_only')
    del rec,reconstruction,prediction
    optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=.001,weight_decay=.1)
    decoder_before={k:v.clone() for k,v in model.decoder.state_dict().items()}
    loss.backward()
    assert all(p.grad is None for p in model.decoder.parameters())
    assert model.encoder.weight_ih_l0.grad.abs().sum()>0
    assert head.net[-1].weight.grad.abs().sum()>0
    optimizer.step()
    for key,value in model.decoder.state_dict().items():torch.testing.assert_close(value,decoder_before[key],rtol=0,atol=0)


def test_near_far_geometry_beats_reversed_geometry():
    d=torch.tensor([[float('nan'),.1,4.],[.1,float('nan'),4.],[4.,4.,float('nan')]])
    q=torch.ones_like(d)
    near=torch.tensor([[1.,0.],[.99,.1],[0.,1.]])
    near=torch.nn.functional.normalize(near,dim=-1)
    far=near[[0,2,1]]
    cal=dict(median=1.,iqr=2.)
    assert g_objective(near,d,q,cal)['loss']<g_objective(far,d,q,cal)['loss']


def test_scaling_is_train_only_and_pairs_are_era_split_work_isolated():
    rng=np.random.default_rng(1);raw=rng.normal(size=(50,23))
    known=np.ones_like(raw,bool);quality=np.ones_like(raw)*.8
    fit=fit_g_scaler(raw,known,list(range(30)))
    altered=raw.copy();altered[30:]=100000
    assert fit==fit_g_scaler(altered,known,list(range(30)))
    md=[dict(era='romantic',split='train',work=str(i//2),source=str(i)) for i in range(50)]
    d,q=pair_tables(raw,known,quality,md,fit,list(range(30)))
    assert np.isnan(np.diag(d)).all() and np.isnan(d[0,1])
    assert np.isfinite(d[0,2]) and q[0,2]==pytest.approx(.8)
    md[1]['era']='baroque_classical'
    with pytest.raises(ValueError):pair_tables(raw,known,quality,md,fit,[0,1])
    md[1]['era']='romantic';md[1]['split']='test'
    with pytest.raises(ValueError):pair_tables(raw,known,quality,md,fit,[0,1])


def test_reference_requires_distinct_sources_and_never_borrows_other_split():
    class Corpus:pass
    corpus=Corpus();clips=[];values=[]
    for i in range(6):
        clips.append(dict(split='train' if i<3 else 'validation',composition_id='work',
                          source_xml_sha256='xml',traversal_id='path',source_sha256=str(i),start=8*i,stop=8*i+8))
        values.append(np.full((8,11),float(i)))
    corpus.manifest=dict(clips=clips)
    corpus.arrays=dict(R=np.concatenate(values),M=np.ones((48,11),bool),original_score_rows=np.tile(np.arange(8),6),
                       note_to_onset=np.tile(np.arange(8),6),beat_id=np.tile(np.arange(8),6),note_q=np.tile(np.arange(8),6))
    cells=np.tile(np.column_stack((np.arange(8),np.arange(1,9))), (6,1))*QUARTER_TICKS
    raw,known,quality=selected_profiles(corpus,cells)
    assert known[:,:9].all()
    # Split-specific median is 1 for train and 4 for validation.
    assert raw[0,0]==pytest.approx(-1) and raw[3,0]==pytest.approx(-1)
    clips[2]['source_sha256']='1'
    _,known,_=selected_profiles(corpus,cells)
    assert not known[:3].any() and known[3:,:9].all()


def test_tempo_references_absolute_cells_not_recording_local_ids():
    from types import SimpleNamespace
    clips=[dict(split='train',composition_id='work',source_xml_sha256='xml',traversal_id='path',
                source_sha256=str(i),start=i*8,stop=(i+1)*8) for i in range(3)]
    arrays=dict(R=np.repeat(np.arange(3,dtype=float),8)[:,None]*np.ones((24,11)),
                M=np.ones((24,11),bool),original_score_rows=np.tile(np.arange(8),3),
                note_to_onset=np.tile(np.arange(8),3),beat_id=np.tile(np.arange(8),3),note_q=np.tile(np.arange(8),3))
    corpus=SimpleNamespace(manifest=dict(clips=clips),arrays=arrays)
    cells=np.tile(np.column_stack((np.arange(8),np.arange(1,9))), (3,1))*QUARTER_TICKS
    baseline=selected_profiles(corpus,cells)
    arrays['beat_id'][8:16]+=100
    arrays['beat_id'][16:]+=200
    shifted=selected_profiles(corpus,cells)
    for left,right in zip(baseline,shifted):np.testing.assert_array_equal(left,right)
    assert baseline[1][:,:3].all()
    np.testing.assert_allclose(baseline[0][:,0],[-1,0,1])
    # Identical local IDs must not pool observations from disjoint score cells.
    arrays['beat_id']=np.tile(np.arange(8),3)
    cells[8:16]+=100*QUARTER_TICKS;cells[16:]+=200*QUARTER_TICKS
    _,known,_=selected_profiles(corpus,cells)
    assert not known[:,:3].any()
    assert known[:,3:].all()


def test_tempo_uses_observed_attack_when_first_note_of_beat_is_masked():
    from types import SimpleNamespace
    clips=[dict(split='train',composition_id='work',source_xml_sha256='xml',traversal_id='path',
                source_sha256=str(i),start=i*16,stop=(i+1)*16) for i in range(3)]
    beats=np.tile(np.repeat(np.arange(8),2),3)
    mask=np.ones((48,11),bool);mask[::2,0]=False
    arrays=dict(R=np.repeat(np.arange(3,dtype=float),16)[:,None]*np.ones((48,11)),M=mask,
                original_score_rows=np.tile(np.arange(16),3),note_to_onset=beats,beat_id=beats,note_q=beats)
    corpus=SimpleNamespace(manifest=dict(clips=clips),arrays=arrays)
    cells=np.column_stack((beats,beats+1))*QUARTER_TICKS
    raw,known,_=selected_profiles(corpus,cells)
    assert known[:,:3].all()
    np.testing.assert_allclose(raw[:,0],[-1,0,1])


@pytest.mark.parametrize('failure',['total','group','channel','spread','rank','drift'])
def test_checkpoint_guards_reject_reconstruction_damage_or_collapse(failure):
    import copy
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from train_era_note_ae import guards
    base=dict(reconstruction=dict(loss=1.,group_losses=[1.]*5,note_weighted_channel_mse=[1.]*11),
              embedding=dict(e_dimension_std=.1,effective_rank=10.,z_anchor_mse=0.))
    assert guards(base,base)['passed']
    bad=copy.deepcopy(base)
    if failure=='total':bad['reconstruction']['loss']=1.051
    elif failure=='group':bad['reconstruction']['group_losses'][4]=1.101
    elif failure=='channel':bad['reconstruction']['note_weighted_channel_mse'][7]=1.201
    elif failure=='spread':bad['embedding']['e_dimension_std']=.049
    elif failure=='rank':bad['embedding']['effective_rank']=4.9
    else:bad['embedding']['z_anchor_mse']=.101
    assert not guards(bad,base)['passed']


def test_paired_work_bootstrap_does_not_count_recordings_as_independent_works():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from analyze_era_loss_ablation import paired_work_interval
    works=np.array(['large']*100+['small','third'])
    right=np.zeros(102);left=np.array([1.]*100+[0.,0.])
    result=paired_work_interval(left,right,works,kind='accuracy',replicates=1000)
    assert result['work_count']==3
    assert result['work_macro_delta']==pytest.approx(100/3)
    equal=paired_work_interval(left,left,works,kind='accuracy',replicates=1000)
    assert equal['ci95']==[0.,0.] and equal['direction']=='uncertain'
    distance=paired_work_interval(np.ones(3),np.full(3,2.),np.array(['a','b','c']),kind='distance_error',replicates=1000)
    assert distance['work_macro_delta']==pytest.approx(-50.)
    assert distance['ci95']==[-50.,-50.] and distance['direction']=='improvement_supported'
