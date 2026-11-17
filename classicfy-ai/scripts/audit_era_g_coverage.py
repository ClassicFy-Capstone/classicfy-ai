"""Replay deterministic batches to audit which clips actually entered G loss.

No neural inference, model selection, or weight updates. All reconstruction
clips remain included, even when peer reference support cannot define G.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_era_note_ae import ERA_COMPOSERS, tables_for, clip_batches, dump


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    metadata=read(args.run/'metadata.json');protocol=read(args.run/'protocol.json')
    final=read(args.run/'final_report.json')
    g_only=final.get('objective_mode')=='g_only'
    with np.load(args.run/'profiles.npz') as f:raw,known,quality=[f[k] for k in ('raw','known','quality')]
    report={}
    for era in ERA_COMPOSERS:
        ids,scaler,table,calibration=tables_for(metadata,raw,known,quality,era,True)
        assert read(args.run/era/'g_production_scaler.json')==dict(scaler=scaler,calibration=calibration)
        state=torch.load(final['eras'][era]['checkpoint'],map_location='cpu',weights_only=False)
        distance,confidence=table['train'];local={ci:i for i,ci in enumerate(ids['train'])}
        seen=set();counts={ci:0 for ci in ids['train']};total_pairs=0
        for row in state['history']:
            epoch=row['epoch'];pair_count=0
            for group in clip_batches(ids['train'],protocol['batch_clips'],state['config']['seed']+epoch):
                ix=np.array([local[i] for i in group])
                valid=np.isfinite(distance[np.ix_(ix,ix)])
                assert np.all(confidence[np.ix_(ix,ix)][valid]>0)
                pair_count+=int(valid.sum())
                active=valid.any(0)|valid.any(1)
                for ci in np.asarray(group)[active]:seen.add(int(ci));counts[int(ci)]+=1
            assert pair_count==row['style']['pairs'], 'Replayed G exposure differs from training log'
            total_pairs+=pair_count
        report[era]=dict(training_clips=len(ids['train']),reconstruction_clips=0 if g_only else len(ids['train']),final_epoch=state['epoch'],
            clips_entering_G_objective=len(seen),anchor_only_clips=len(ids['train'])-len(seen) if g_only else 0,
            reconstruction_only_clips=0 if g_only else len(ids['train'])-len(seen),
            directed_pair_exposures=total_pairs,
            G_epoch_counts_by_perf_id={metadata[i]['perf_id']:counts[i] for i in ids['train']})
    result=dict(scope='Actual deterministic batch replay through each final checkpoint epoch',
        objective_mode='g_only' if g_only else 'joint',
        training_clips=sum(r['training_clips'] for r in report.values()),
        reconstruction_clips=sum(r['reconstruction_clips'] for r in report.values()),
        clips_entering_G_objective=sum(r['clips_entering_G_objective'] for r in report.values()),eras=report,
        explanation=('G absence means insufficient comparable/reference-supported profiles or no eligible batch partner; '
                     +('all clips still enter encoder and anchor, no direct reconstruction.' if g_only else 'all clips still enter reconstruction.')))
    dump(args.run/'production_G_coverage.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='eras'},indent=2))
    for era,row in report.items():print(era,{k:v for k,v in row.items() if k!='G_epoch_counts_by_perf_id'})


if __name__=='__main__':main()
