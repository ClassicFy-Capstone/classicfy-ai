"""Read-only real-batch CPU probe; backward is measured, no optimizer is run."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from train_era_note_ae import (Runtime,SelectedNoteCorpus,restore,forward,clip_batches,tables_for,
                              g_objective,capped_style_weight,dump,file_hash)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--common-run',type=Path,required=True)
    p.add_argument('--era',required=True)
    args=p.parse_args();torch.set_num_interop_threads(1)
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    metadata=read(args.run/'metadata.json')
    with np.load(args.run/'profiles.npz') as f:raw,known,quality=[f[k] for k in ('raw','known','quality')]
    corpus=SelectedNoteCorpus(args.common_run/'corpus')
    runtime=Runtime(corpus,args.common_run/'final_all_5082.pt',args.run/'production_cache')
    checkpoint=args.run/args.era/'production/last.pt';before=file_hash(checkpoint)
    model,head,scale,state=restore(checkpoint,runtime)
    ids,_,tables,calibration=tables_for(metadata,raw,known,quality,args.era,True)
    local={ci:i for i,ci in enumerate(ids['train'])};d,q=tables['train']
    schedule=clip_batches(ids['train'],16,state['config']['seed']+state['epoch']+1)
    examples=[schedule[i] for i in (0,len(schedule)//2,len(schedule)-2)]
    # Disable stochastic dropout to make numerical parity across thread counts meaningful.
    # Autograd and all encoder/decoder/head backward passes remain enabled.
    model.eval();head.eval()
    results=[];reference_losses=None
    for threads in (1,2,4,1):
        torch.set_num_threads(threads);durations=[];losses=[]
        for j,group in enumerate(examples):
            model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
            torch.manual_seed(20261113+j);gc.collect()
            began=time.perf_counter()
            result=forward(model,head,runtime,group,scale)
            ix=np.array([local[i] for i in group])
            g=g_objective(result['e'],torch.from_numpy(d[np.ix_(ix,ix)]),torch.from_numpy(q[np.ix_(ix,ix)]),calibration)
            weight,_=capped_style_weight(result['rec_loss'],g['loss'],result['z'],.1,.25)
            objective=result['rec_loss']+weight*g['loss']+.05*result['anchor']+.01*result['head_penalty']
            objective.backward()
            durations.append(time.perf_counter()-began);losses.append(float(objective.detach()))
            del result,g,objective
        if reference_losses is None:reference_losses=losses
        error=float(np.max(np.abs(np.array(losses)-reference_losses)))
        assert error<1e-5, 'Thread change altered objective beyond tolerance'
        row=dict(threads=threads,seconds=durations,mean_seconds=float(np.mean(durations)),objective_max_delta=error)
        results.append(row);print(json.dumps(row),flush=True)
    assert file_hash(checkpoint)==before
    means={t:float(np.mean([x for row in results if row['threads']==t for x in row['seconds']])) for t in (1,2,4)}
    report=dict(scope='No optimizer/weight updates; actual logical16-clip batches including G gradient cap and backward; dropout disabled for parity',
                checkpoint_sha256=before,results=results,mean_seconds_by_threads=means,selected_threads=min(means,key=means.get))
    dump(args.run/'actual_batch_thread_probe.json',report)


if __name__=='__main__':main()
