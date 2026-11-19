"""Verify era artifacts, complete coverage, immutable common AE, and reload parity."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from embedding.era_note_style import ERA_COMPOSERS, load_era_checkpoint, aggregate_clip_latents, selected_profiles
from embedding.selected_note_data import SelectedNoteCorpus, file_hash
from embedding.selected_style_coordinates import G_PROFILE_VERSION, QUARTER_TICKS
from train_selected_note_ae import dump


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--common-run',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    report=json.loads((args.run/'final_report.json').read_text(encoding='utf-8'))
    metadata=json.loads((args.run/'metadata.json').read_text(encoding='utf-8'))
    protocol=json.loads((args.run/'protocol.json').read_text(encoding='utf-8'))
    profiles=json.loads((args.run/'profile_contract.json').read_text(encoding='utf-8'))
    coordinates=json.loads((args.run/'coordinates/tempo_coordinates.json').read_text(encoding='utf-8'))
    assert protocol['supervision_version']==profiles['version']==coordinates['contract']['version']==G_PROFILE_VERSION
    assert profiles['coordinates_id']==coordinates['id']
    assert file_hash(args.run/'coordinates/tempo_cells.npy')==coordinates['coordinate_sha256']
    assert coordinates['all_clips']==5082 and coordinates['source_signatures_verified_against_AE_preparation']
    with np.load(args.run/'profiles.npz') as f:
        assert f['raw'].shape==f['known'].shape==f['quality'].shape==(5082,23)
        assert np.isfinite(f['raw']).all() and np.isfinite(f['quality']).all()
        assert (f['quality'][f['known']]>0).all() and (f['quality']<=1).all()
        assert (f['quality'][~f['known']]==0).all()
    corpus=SelectedNoteCorpus(args.common_run/'corpus')
    cells=np.load(args.run/'coordinates/tempo_cells.npy',mmap_mode='r')
    recomputed=selected_profiles(corpus,cells)
    with np.load(args.run/'profiles.npz') as f:
        for name,value in zip(('raw','known','quality'),recomputed):
            np.testing.assert_array_equal(value,f[name],err_msg='G profile regeneration mismatch: '+name)
    del recomputed
    tempo_error=0.;tempo_notes=0
    for clip in corpus.manifest['clips']:
        section=slice(clip['start'],clip['stop'])
        observed=np.asarray(corpus.arrays['M'][section,0])
        if not observed.any():continue
        beat=np.asarray(corpus.arrays['beat_id'][section])[observed]
        with np.load(clip['alignment_path'],allow_pickle=False) as f:seconds=f['performance_beats']
        width=np.diff(cells[section][observed],axis=1)[:,0]/QUARTER_TICKS
        expected=np.log2(60*width/(seconds[beat+1]-seconds[beat]))
        stored=np.asarray(corpus.arrays['R'][section,0])[observed]
        assert np.isfinite(expected).all()
        error=float(np.max(np.abs(stored-expected)));tempo_error=max(tempo_error,error)
        assert error<1e-5, 'R tempo is not bound to recovered interval: '+clip['perf_id']
        tempo_notes+=int(observed.sum())
    for name,expected in corpus.manifest['files'].items():
        assert file_hash(corpus.path/(name+'.npy'))==expected, 'Packed input changed: '+name
    for name,expected in protocol['pretraining_hashes'].items():
        assert file_hash(args.common_run/name)==expected, 'Common AE was changed'
    for field in ('composition_id','source_sha256','source_xml_sha256','score_midi_sha256'):
        seen={}
        for clip in corpus.manifest['clips']:
            assert seen.setdefault(clip[field],clip['split'])==clip['split'], 'Split leakage'
    with np.load(report['embeddings']) as f:arrays={k:f[k] for k in f.files}
    assert len(arrays['perf_ids'])==len(set(arrays['perf_ids']))==5082
    assert arrays['perf_ids'].tolist()==[m['perf_id'] for m in metadata]
    assert arrays['era'].tolist()==[m['era'] for m in metadata]
    for name in ('z_ae','z_style','e'):
        assert arrays[name].shape==(5082,64) and np.isfinite(arrays[name]).all()
    np.testing.assert_allclose(np.linalg.norm(arrays['e'],axis=1),1.,atol=1e-6)
    with np.load(args.common_run/'all_5082_embeddings.npz') as f:
        common_z,common_e,common_ids=f['z'],f['e'],f['perf_ids']
    assert np.array_equal(common_ids,arrays['perf_ids'])
    common=torch.load(args.common_run/'final_all_5082.pt',map_location='cpu',weights_only=False)
    checks={}
    for era in ERA_COMPOSERS:
        info=report['eras'][era];path=Path(info['checkpoint'])
        assert file_hash(path)==info['sha256']
        model,head,state=load_era_checkpoint(path)
        assert state['era']==era and state['base_sha256']==protocol['pretraining_hashes']['final_all_5082.pt']
        budget=float(state['config']['gradient_budget'])
        if protocol.get('configs'):
            assert state['config']==protocol['configs'][era]
        for value in list(model.state_dict().values())+list(head.state_dict().values()):
            if value.is_floating_point():assert torch.isfinite(value).all()
        assert all(torch.equal(value,common['state_dict'][key]) for key,value in model.state_dict().items()
                   if key.startswith('score_encoder.')), 'Frozen score encoder changed'
        ids=[i for i,m in enumerate(metadata) if m['era']==era]
        assert info['clips']==len(ids)
        assert info['guard']['passed'], 'Production guard failed'
        for row in state['history']:
            assert row['clips_seen']==len(ids), 'Epoch omitted clips'
            assert row['style']['applied_gradient_ratio']<=budget+1e-6
        samples=np.random.default_rng(2026).choice(ids,min(3,len(ids)),replace=False)
        error=0.
        with torch.no_grad():
            for ci in samples:
                windows=[w for w in corpus.items() if w['clip_index']==ci]
                batch=corpus.collate(windows,state['scaler'])
                z=model.encode_batch(batch)['z']
                pooled=aggregate_clip_latents(z,torch.zeros(len(windows),dtype=torch.long),
                    batch['M'].any(-1).sum(-1).to(torch.float32),1)
                prediction=head(pooled)
                for key,value in [('z_ae',pooled),('z_style',prediction['z_style']),('e',prediction['e'])]:
                    error=max(error,float(np.max(np.abs(value.numpy()[0]-arrays[key][ci]))))
                    np.testing.assert_allclose(value.numpy()[0],arrays[key][ci],atol=2e-5,rtol=2e-5)
        encoder_changed=any(not torch.equal(v,common['state_dict'][k]) for k,v in model.state_dict().items()
                            if k.startswith('ae.encoder.') or k.startswith('ae.latent.'))
        decoder_changed=any(not torch.equal(v,common['state_dict'][k]) for k,v in model.state_dict().items()
                            if k.startswith('ae.decoder.'))
        if state['epoch']>0:
            assert encoder_changed, 'Trained era encoder did not fine-tune'
            if state['config'].get('objective_mode')=='g_only':
                assert not decoder_changed, 'G-only decoder changed'
                for name in ('decoder','onset_head','velocity_head','articulation_head','pedal_head','tempo_head'):
                    assert all(torch.equal(v,common['state_dict'][k]) for k,v in model.state_dict().items()
                               if k.startswith('ae.'+name+'.')), 'G-only decode-only weights changed: '+name
                assert state['config'].get('reconstruction_weight',0)==0
            else:assert decoder_changed, 'Reconstruction-trained decoder did not fine-tune'
        else:
            assert not encoder_changed and not decoder_changed, 'Baseline fallback changed weights'
        raw=arrays['z_ae'][ids]
        if info.get('embeddings'):
            assert file_hash(info['embeddings'])==info['embeddings_sha256']
            with np.load(info['embeddings']) as f:
                for name in ('perf_ids','composition_ids','era','z_ae','z_style','e'):
                    np.testing.assert_array_equal(f[name],arrays[name][ids])
        raw_cos=np.sum(raw*common_z[ids],axis=1)/np.maximum(np.linalg.norm(raw,axis=1)*np.linalg.norm(common_z[ids],axis=1),1e-12)
        style_cos=np.sum(arrays['e'][ids]*common_e[ids],axis=1)
        relative_shift=np.linalg.norm(arrays['z_style'][ids]-raw,axis=1)/np.maximum(np.linalg.norm(raw,axis=1),1e-12)
        def quantiles(values):
            return dict(zip(('min','p01','median','p99','max'),np.quantile(values,[0,.01,.5,.99,1]).tolist()))
        checks[era]=dict(clips=len(ids),works=len({metadata[i]['work'] for i in ids}),epoch=state['epoch'],
                         reload_max_error=error,encoder_changed=encoder_changed,decoder_changed=decoder_changed,
                         baseline_fallback=state['epoch']==0,
                         score_encoder_unchanged=True,raw_z_cosine_to_common=quantiles(raw_cos),
                         style_e_cosine_to_common=quantiles(style_cos),head_relative_shift=quantiles(relative_shift))
    result=dict(status='passed',all_5082_finite_unit_embeddings=True,split_identity_checks=True,
                G_supervision_version=G_PROFILE_VERSION,absolute_tempo_coordinates_verified=True,
                all_G_profiles_regenerated_identically=True,
                R_tempo_interval_roundtrip=dict(observed_notes=tempo_notes,max_log2_qpm_error=tempo_error),
                common_checkpoints_unchanged=True,packed_input_hashes_verified=True,
                eras=checks,embeddings_sha256=file_hash(report['embeddings']))
    if report.get('embeddings_sha256'):
        assert result['embeddings_sha256']==report['embeddings_sha256']
    dump(args.run/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
