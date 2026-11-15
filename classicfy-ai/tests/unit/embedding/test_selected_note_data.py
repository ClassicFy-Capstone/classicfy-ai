import json

import numpy as np

from embedding.selected_note_data import connected_splits, file_hash, pack_corpus, SelectedNoteCorpus


def test_identity_connected_components_never_cross_splits():
    rows=[dict(perf_id=str(i),composition_id=str(i),source_sha256=f'p{i}',
               source_xml_sha256=f'x{i}',score_midi_sha256=f's{i}') for i in range(30)]
    rows[1]['source_xml_sha256']=rows[0]['source_xml_sha256']
    rows[2]['source_sha256']=rows[1]['source_sha256']
    rows[3]['score_midi_sha256']=rows[2]['score_midi_sha256']
    split,count=connected_splits(rows)
    assert len(set(split[:4]))==1
    assert count==27 and set(split)=={'train','validation','test'}


def test_packing_retains_single_note_tail_and_never_splits_chords(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    for i,n in enumerate([131,4]):
        groups=np.arange(n)//2
        mask=np.ones((n,11),bool)
        values=np.full((n,11),1+10*i,np.float32)
        path=prepared/f'{i}.npz'
        np.savez(path,numeric=np.zeros((n,17),np.float32),categorical=np.ones((n,14),int),
            flags=np.zeros((n,17)),field_known=np.zeros((n,48),bool),R=values,M=mask,
            beat_id=groups,note_to_onset=groups,onset_q=np.arange(groups[-1]+1,dtype=float),
            original_score_rows=np.arange(n))
        row=dict(status='passed',perf_id=str(i),composition_id=str(i),source_sha256=f'p{i}',
            source_xml_sha256=f'x{i}',score_midi_sha256=f's{i}',traversal_id=f't{i}',
            notes=n,cache_path=str(path),cache_sha256=file_hash(path))
        (prepared/f'{i}.json').write_text(json.dumps(row))
    manifest=pack_corpus(prepared,tmp_path/'packed',max_notes=64,expected_clips=2)
    first=[w for w in manifest['windows'] if w['clip_index']==0]
    assert [w['N'] for w in first]==[64,64,3]
    assert sum(w['N'] for w in first)==131
    assert abs(sum(w['clip_weight'] for w in first)-1)<1e-8
    corpus=SelectedNoteCorpus(tmp_path/'packed');scale=corpus.fit_scalers(all_data=True)
    train_scale=corpus.fit_scalers()
    train_id=next(r['perf_id'] for r in manifest['clips'] if r['split']=='train')
    assert train_scale['fit_ids']==[train_id]
    np.testing.assert_allclose(train_scale['R_mean'],1+10*int(train_id))
    batch=corpus.collate([first[0],first[-1]],scale)
    assert batch['lengths'].tolist()==[64,3]
    assert not batch['P'][1,3:].any() and not batch['M'][1,3:].any()
    assert batch['note_to_onset'][1,:3].tolist()==[0,0,1]
