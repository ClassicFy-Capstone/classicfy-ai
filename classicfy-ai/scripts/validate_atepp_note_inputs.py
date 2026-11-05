"""Validate every eligible full-window input through the actual pre-AE collator.

No AE model, loss, optimizer or training loop is run. Records are loaded one at
a time in bounded workers; shared train scalers remain fixed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
import json
from pathlib import Path
import sys
import os

sys.path.insert(0,os.environ.get('CLASSICFY_VALIDATION_SRC',str(Path(__file__).resolve().parents[1]/'src')))


@lru_cache(maxsize=4)
def resources(out,score_cache):
    import torch
    torch.set_num_threads(1)
    from embedding.note_score_context_data import load_note_score_context,load_note_score_scale
    from features.note_performance_normalization import load_note_performance_scaler
    return (load_note_score_context(score_cache)[0],load_note_score_scale(Path(out)/'S_scale.json'),
            load_note_performance_scaler(Path(out)/'R_scaler.json'))


def check_record(task):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from embedding.note_ae_data import NoteAEWindows,collate_ae_windows
    from embedding.note_performance_data import NotePerformancePair,NotePerformanceSegment,load_note_performance
    from embedding.note_score_context_data import NoteScoreSegment
    entry,score_cache,out=task
    result={'performance_id':entry['performance_id']}
    try:
        s,score_scale,scaler=resources(out,score_cache)
        r=load_note_performance(entry['performance_cache'])
        if r.fingerprint!=entry['R_fingerprint']:raise ValueError('R cache changed')
        pair=NotePerformancePair(s,r)
        windows=json.loads(Path(entry['windows_file']).read_text())
        items=[];affected=0
        for w in windows:
            rows=w['row_indices']
            if w['tail'] or not rows or not r.mask[rows].any():continue
            a,b=np.searchsorted(s.onset_q,[w['q_start'],w['q_end']],side='left')
            items.append({'window':w,'segment':NotePerformanceSegment(pair,NoteScoreSegment(s,int(a),int(b),int(a),int(b)))})
            left=-np.inf if a==0 else s.onset_q[a]
            right=np.inf if b==len(s.onset_q) else s.onset_q[b]
            old=r.beats.valid & (r.beats.q[:-1]>=left) & (r.beats.q[:-1]<right)
            actual=r.beats.valid & (r.beats.q[:-1]>=w['q_start']) & (r.beats.q[:-1]<w['q_end'])
            affected+=int(np.any(old!=actual))
        collate=partial(collate_ae_windows,r_scaler=scaler,score_scale=score_scale)
        normalized=scaler.transform(r)
        batches=padding=owned_count=0
        ownership=np.zeros(len(r.beats.valid),np.int64)
        for batch in DataLoader(NoteAEWindows(items),batch_size=2,collate_fn=collate):
            assert torch.isfinite(batch['R']).all()
            assert batch['R_state']=='train-channel-scaled-once'
            assert not batch['M'][~batch['P']].any()
            assert torch.equal(batch['lengths'],batch['P'].sum(1))
            assert not batch['R_loss_mask'][:,:,0].any()
            padding+=int((~batch['P']).sum())
            for bi,w in enumerate(batch['window_metadata']):
                rows=w['row_indices'];n=len(rows)
                np.testing.assert_array_equal(batch['R'][bi,:n].numpy(),normalized[rows])
                np.testing.assert_array_equal(batch['M'][bi,:n].numpy(),r.mask[rows])
                expected=(r.beats.valid & (r.beats.q[:-1]>=w['q_start']) & (r.beats.q[:-1]<w['q_end'])
                          & scaler.trained_channel_mask[0])
                np.testing.assert_array_equal(batch['beat_loss_mask'][bi,:len(expected)].numpy(),expected)
                ownership+=expected
                owned_count+=int(expected.sum())
            batches+=1
        assert not (ownership>1).any()
        assert load_note_performance(entry['performance_cache']).fingerprint==entry['R_fingerprint']
        result.update(status='passed',windows=len(items),batches=batches,padding_positions=padding,
                      owned_beats=owned_count,affected_windows_under_old_onset_bounds=affected)
    except Exception as exc:result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();out=args.out.resolve()
    index=json.loads((out/'window_index.json').read_text())
    scores={}
    for line in (out/'scores.jsonl').open():
        s=json.loads(line)
        if s['status']=='prepared':scores[s['source_sha256']]=s['score_cache']
    # The source row keys were already checked by the window index creation.
    tasks=[]
    for entry in index['sources']:
        first=json.loads(Path(entry['windows_file']).read_text())[0]
        tasks.append((entry,scores[first['score_version_id']],str(out)))
    log=out/'input_batches.jsonl';done={}
    if log.exists():
        done={r['performance_id']:r for r in (json.loads(line) for line in log.open())}
    tasks=[t for t in tasks if done.get(t[0]['performance_id'],{}).get('status')!='passed']
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(check_record,t) for t in tasks]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result();done[result['performance_id']]=result
            with log.open('a') as stream:stream.write(json.dumps(result)+'\n')
            if i%100==0 or i==len(tasks):print('pre-AE input records',i,'/',len(tasks),result['status'],flush=True)
    values=list(done.values())
    summary={'records':len(values),'passed':sum(v['status']=='passed' for v in values),'failed':[v for v in values if v['status']!='passed'],
             'full_windows':sum(v.get('windows',0) for v in values),'batches':sum(v.get('batches',0) for v in values),
             'padding_positions':sum(v.get('padding_positions',0) for v in values),'owned_beats':sum(v.get('owned_beats',0) for v in values),
             'affected_windows_under_old_onset_bounds':sum(v.get('affected_windows_under_old_onset_bounds',0) for v in values),
             'AE_or_loss_executed':False}
    (out/'input_batch_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
    if summary['failed']:raise SystemExit(1)


if __name__=='__main__':main()
