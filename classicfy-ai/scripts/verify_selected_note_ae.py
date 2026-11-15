"""Independently verify the finished selected-clip AE artifacts without retesting.

Checks the all-data training lineage, held-out model lineage, original selected
IDs, saved embeddings, and a deterministic sample of reloaded encoder outputs.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_selected_note_ae import SelectedNoteCorpus, dump, file_hash, load_model, batches


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--selected-csv',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=args.run
    report=read(root/'final_report.json')
    selection=read(root/'selection.json')
    test=read(root/'test_evaluation.json')
    corpus=SelectedNoteCorpus(root/'corpus')
    with args.selected_csv.open(encoding='utf-8-sig',newline='') as stream:
        original=list(csv.DictReader(stream))
    original_ids={r['perf_id'] for r in original}
    assert len(original)==len(original_ids)==5082
    records=corpus.manifest['clips']
    assert {r['perf_id'] for r in records}==original_ids
    for field in ('composition_id','source_sha256','source_xml_sha256','score_midi_sha256'):
        ownership={}
        for row in records:
            previous=ownership.setdefault(row[field],row['split'])
            assert previous==row['split'],f'Split leakage in {field}'

    final=root/'final_all_5082.pt'
    evaluated=root/'evaluated_best.pt'
    assert file_hash(final)==report['final_sha256']
    assert file_hash(evaluated)==test['checkpoint_sha256']
    assert report['status']=='complete' and not report['production_has_independent_test']
    model,state=load_model(final)
    assert state['format']=='selected-musicxml-note-ae-v1'
    assert state['training_scope']=='all_5082_refit'
    assert not state['independent_test_available'] and not state['G_loss_enabled']
    assert state['corpus_id']==corpus.manifest['id']
    assert state['trial']==selection['trial']
    assert state['epoch']==selection['best_epoch']==report['selected_epoch']
    assert set(state['scaler']['fit_ids'])==original_ids
    assert len(state['history'])==state['epoch']
    assert [r['epoch'] for r in state['history']]==list(range(1,state['epoch']+1))
    assert all(r['clips_seen']==5082 and r['validation'] is None for r in state['history'])
    assert all(torch.isfinite(value).all() for value in state['state_dict'].values())
    held_out=torch.load(evaluated,map_location='cpu',weights_only=False)
    train_ids={r['perf_id'] for r in records if r['split']=='train'}
    assert set(held_out['scaler']['fit_ids'])==train_ids
    assert held_out['training_scope']=='train_split'
    assert held_out['epoch']==selection['best_epoch']
    assert held_out['trial']==state['trial']
    assert test['normal']['clips']==corpus.manifest['split_counts']['test']

    with np.load(root/'all_5082_embeddings.npz',allow_pickle=False) as saved:
        z,e=saved['z'].copy(),saved['e'].copy()
        assert saved['perf_ids'].tolist()==[r['perf_id'] for r in records]
        assert saved['composition_ids'].tolist()==[r['composition_id'] for r in records]
    assert z.shape==e.shape==(5082,model.config.latent_dim)
    assert np.isfinite(z).all() and np.isfinite(e).all()
    np.testing.assert_allclose(np.linalg.norm(e,axis=1),1,atol=2e-6,rtol=2e-6)
    assert float(z.std(0).mean())>1e-8 and float(e.std(0).mean())>1e-8
    sample=sorted({0,len(records)//2,len(records)-1,
                   *np.random.default_rng(2026).choice(len(records),3,replace=False).tolist()})
    maximum_error=0.
    for index in sample:
        items=[w for w in corpus.items() if w['clip_index']==index]
        total=np.zeros(model.config.latent_dim,dtype=np.float64)
        weight_sum=0
        for group in batches(items,16):
            batch=corpus.collate(group,state['scaler'])
            vectors=model.encode_batch(batch)['z'].numpy()
            for bi,vector in enumerate(vectors):
                weight=int(batch['M'][bi].any(-1).sum())
                total+=vector*weight
                weight_sum+=weight
        recreated=(total/weight_sum).astype(np.float32)
        np.testing.assert_allclose(recreated,z[index],atol=2e-5,rtol=2e-5)
        maximum_error=max(maximum_error,float(np.max(np.abs(recreated-z[index]))))
    result=dict(status='passed',clips=5082,epochs=state['epoch'],latent_dimension=z.shape[1],
        original_selected_ids_match=True,all_clips_seen_every_epoch=True,
        no_evaluation_split_identity_overlap=True,finite_parameters_and_embeddings=True,
        unit_normalized_e=True,reloaded_embedding_sample_ids=[records[i]['perf_id'] for i in sample],
        reloaded_embedding_max_abs_error=maximum_error,
        mean_z_dimension_std=float(z.std(0).mean()),mean_e_dimension_std=float(e.std(0).mean()),
        final_checkpoint_sha256=file_hash(final),evaluated_checkpoint_sha256=file_hash(evaluated),
        embeddings_sha256=file_hash(root/'all_5082_embeddings.npz'),
        evaluated_test_loss=test['normal']['loss'],
        evaluated_test_reduction_against_mean=1-test['normal']['loss']/test['train_channel_mean']['loss'],
        production_has_independent_test=False,
        test_policy='Existing frozen metrics checked; no repeated test inference or retuning')
    dump(root/'final_artifact_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
