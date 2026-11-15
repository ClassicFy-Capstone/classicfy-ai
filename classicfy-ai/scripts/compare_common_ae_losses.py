"""Select a robust reconstruction candidate on validation, then report its test.

Selection rule must be written to experiment_protocol.json before training.
Never overwrite baseline or promote a candidate failing the nonpedal guard.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import train_selected_common_ae as runner


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def best_validation(path):
    state=read(path/'status.json')
    if state['stage']!='complete':raise ValueError(f'Run not complete: {path}')
    if state['best_epoch']==0:
        return read(path/'protocol.json')['initial_validation']
    history=[json.loads(line) for line in (path/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    return next(row['validation'] for row in history if row['epoch']==state['best_epoch'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    args=p.parse_args();out=args.experiment
    if (out/'comparison.json').exists():raise ValueError('Comparison already saved')
    protocol=read(out/'experiment_protocol.json');base=Path(protocol['baseline'])
    manifest=read(base/'dataset.json')
    initial=torch.load(base/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
    baseline=best_validation(base);bc=np.array(baseline['channel_mse'])
    comparisons=[]
    for item in protocol['candidates']:
        path=out/item['name'];cm=read(path/'dataset.json')
        if cm!=manifest:raise ValueError('Candidate input manifest/split differs from baseline')
        candidate=torch.load(path/'initial.pt',map_location='cpu',weights_only=True)['state_dict']
        if candidate.keys()!=initial.keys() or any(not torch.equal(v,candidate[k]) for k,v in initial.items()):
            raise ValueError('Candidate initialization differs from baseline')
        with np.load(base/'input_sequences.npz',allow_pickle=False) as left,np.load(path/'input_sequences.npz',allow_pickle=False) as right:
            if set(left.files)!=set(right.files):raise ValueError('Candidate cache keys differ')
            for k in left.files:
                if not np.array_equal(left[k],right[k]):raise ValueError(f'Candidate cache differs: {k}')
        metrics=best_validation(path);ch=np.array(metrics['channel_mse'])
        comparisons.append({'name':item['name'],'checkpoint':str(path/'best.pt'),
            'best_epoch':read(path/'status.json')['best_epoch'], 'validation':metrics,
            'validation_channel_improvement_percent':(100*(1-ch/bc)).tolist(),
            'pedal_mean_mse':float(ch[4:].mean()),
            'passes_nonpedal_guard':bool(np.all(ch[:4]<=bc[:4]*1.05)),
            'improves_pedal':bool(ch[4:].mean()<bc[4:].mean())})
    eligible=[r for r in comparisons if r['passes_nonpedal_guard'] and r['improves_pedal']]
    selected=min(eligible,key=lambda r:r['pedal_mean_mse']) if eligible else None
    result={'baseline_validation':baseline,'candidates':comparisons,
            'selected':selected['name'] if selected else 'baseline',
            'checks':'Identical manifests, cached input arrays and initial weights verified; selection uses validation only.',
            'test':None}
    runner.dump(out/'selection.json',result)
    if selected:
        torch.set_num_threads(4)
        seq=[]
        with np.load(base/'input_sequences.npz',allow_pickle=False) as z:
            for r in manifest['sequences']:
                if r['split']=='test':seq.append({**r,'values':z[r['key']+'/values'],'valid':z[r['key']+'/valid']})
        dataset=runner.AllObservedWindows(seq)
        model,checkpoint=runner.load_model(Path(selected['checkpoint']))
        test=runner.evaluate(model,dataset,**checkpoint['reconstruction_objective'])
        old=read(base/'test.json')
        result['test']={'candidate':test,'baseline':old,
            'channel_improvement_percent':(100*(1-np.array(test['channel_mse'])/np.array(old['channel_mse']))).tolist()}
        runner.dump(out/selected['name']/'test_after_selection.json',result['test'])
        shutil.copyfile(selected['checkpoint'],out/'selected_model.pt')
        result['selected_checkpoint']={'path':str(out/'selected_model.pt'),
            'source':selected['checkpoint'],'sha256':runner.digest(out/'selected_model.pt')}
    runner.dump(out/'comparison.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
