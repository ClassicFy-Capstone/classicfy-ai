"""Validation-only weak reconstruction sweep against protected G-only.

Never read test diagnostics, select production weights or overwrite run03 models.
The G cap still uses UNWEIGHTED reconstruction gradients for all coefficients.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_era_note_ae import (ERA_COMPOSERS,SelectedNoteCorpus,Runtime,tables_for,
    fit,restore,guards,dump,file_hash,G_PROFILE_VERSION)
from analyze_era_loss_ablation import diagnose,paired_work_interval


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--common-run',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    out=args.run/'weak_reconstruction_validation';out.mkdir(exist_ok=True)
    metadata=read(args.run/'metadata.json');contract=read(args.run/'profile_contract.json')
    if contract['version']!=G_PROFILE_VERSION:raise ValueError('Invalid G version')
    selections={e:read(args.run/e/'selection.json') for e in ERA_COMPOSERS}
    sources={e:{c:args.run/'ablation'/e/c for c in ('g_only','joint','common_frozen')} for e in ERA_COMPOSERS}
    protocol=dict(version='weak-rec-validation-v1',coefficients=[0.,.01,.1,.3,1.],
        train_new=[.01,.1,.3],seed=2026,batch_clips=16,supervision=contract,
        initialization_sha256=file_hash(args.common_run/'evaluated_best.pt'),
        horizons={e:s['best_epoch'] for e,s in selections.items()},
        existing_sources={e:{c:file_hash(d/'validation_evaluation.json') for c,d in cs.items()} for e,cs in sources.items()},
        loss='alpha*Rec + effective_lambda*G + .05*z_anchor + .01*head_shift',
        cap='Same .25 budget against UNWEIGHTED Rec gradient; do not rescale cap by alpha; effective lambda remains adaptive',
        controls='Same initialization, seed, batch order, nominal G weight, optimizer, fixed horizon and preservation rules',
        checkpoint='Last epoch at fixed horizon; no best-epoch search for candidates',
        validation_only=True,previous_test_seen=True,training_seeds=1,
        useful_weak_rec='Common guards pass; >=.1% work-macro Rec improvement vs G-only; adjusted CI upper<0; pair MSE <=1.01*G-only; ranking>=G-only-1pp',
        selection='Among useful weak candidates minimize validation G score; otherwise alpha0. Also report minimum-G-score guard-passing candidate separately.',
        uncertainty='Exploratory paired work bootstrap5000; nine new comparisons per outcome; intervals do not correct for adaptive validation selection',
        scope='Existing test motivated this search; no new independent confirmation; no production replacement')
    if (out/'protocol.json').exists() and read(out/'protocol.json')!=protocol:raise ValueError('Sweep protocol changed')
    dump(out/'protocol.json',protocol)
    with np.load(args.run/'profiles.npz') as f:raw,known,quality=[f[k] for k in ('raw','known','quality')]
    corpus=SelectedNoteCorpus(args.common_run/'corpus')
    runtime=Runtime(corpus,args.common_run/'evaluated_best.pt',args.run/'evaluation_cache')
    summaries={}
    for era in ERA_COMPOSERS:
        ids,scaler,tables,calibration=tables_for(metadata,raw,known,quality,era)
        scale=torch.tensor(np.maximum(runtime.teacher_clip[ids['train']].std(0),.05))
        conditions={};arrays={};paths={}
        for coefficient,condition in [(0.,'g_only'),(1.,'joint')]:
            directory=sources[era][condition]
            conditions[str(coefficient)]=read(directory/'validation_evaluation.json')['metrics']
            with np.load(directory/'validation_diagnostics.npz') as f:arrays[str(coefficient)]={k:f[k] for k in f.files}
            paths[str(coefficient)]=str(directory/'last.pt') if condition=='g_only' else selections[era]['path']
        common=read(sources[era]['common_frozen']/'validation_evaluation.json')['metrics']
        for coefficient in protocol['train_new']:
            trial=out/era/f'alpha_{coefficient}'
            config={**selections[era]['config'],'objective_mode':'joint','reconstruction_weight':coefficient}
            fit(runtime,era,ids['train'],ids['validation'],tables,calibration,scale,config,trial,16,protocol['horizons'][era])
            checkpoint=trial/'last.pt'
            signature=dict(checkpoint_sha256=file_hash(checkpoint),supervision=contract,split='validation')
            if (trial/'validation.json').exists():
                saved=read(trial/'validation.json')
                if saved['signature']!=signature:raise ValueError('Validation checkpoint changed')
                metrics=saved['metrics']
                with np.load(trial/'diagnostics.npz') as f:data={k:f[k] for k in f.files}
            else:
                model,head,loaded_scale,state=restore(checkpoint,runtime)
                torch.testing.assert_close(loaded_scale,scale)
                metrics,data=diagnose(model,head,runtime,ids['validation'],scale,tables['validation'],calibration,16)
                np.savez_compressed(trial/'diagnostics.npz',**data)
                dump(trial/'validation.json',dict(signature=signature,metrics=metrics))
            conditions[str(coefficient)]=metrics;arrays[str(coefficient)]=data;paths[str(coefficient)]=str(checkpoint)
        work=np.array([metadata[i]['work'] for i in ids['validation']]);comparisons={};useful=[]
        for coefficient in protocol['train_new']:
            key=str(coefficient);data=arrays[key];base=arrays['0.0'];metrics=conditions[key]
            assert np.array_equal(data['ids'],base['ids']) and np.array_equal(data['triplets'],base['triplets'])
            comparison={name:paired_work_interval(data[field],base[field],work,kind=kind)
                for name,field,kind in [('ranking','ranking_per_anchor','accuracy'),
                ('distance_error','distance_error_per_anchor','distance_error'),
                ('reconstruction','reconstruction_per_clip','reconstruction')]}
            guard=guards(metrics,common)
            rec=comparison['reconstruction'];base_metrics=conditions['0.0']
            checks=dict(common_preserved=guard['passed'],rec_improvement_at_least_point1_percent=rec['work_macro_delta']<=-.1,
                rec_improvement_supported=rec['familywise95_approx_for9_contrasts'][1]<0,
                distance_within1percent=metrics['geometry']['pair_mse']<=1.01*base_metrics['geometry']['pair_mse'],
                ranking_within1pp=metrics['geometry']['ranking_accuracy']>=base_metrics['geometry']['ranking_accuracy']-.01)
            comparisons[key]=dict(**comparison,guard=guard,usefulness_checks=checks,useful=all(checks.values()))
            if all(checks.values()):useful.append(key)
        eligible=[k for k,m in conditions.items() if guards(m,common)['passed']]
        best=min(eligible,key=lambda k:conditions[k]['geometry']['score'])
        chosen=min(useful,key=lambda k:conditions[k]['geometry']['score']) if useful else '0.0'
        summaries[era]=dict(epochs=protocol['horizons'][era],validation_conditions=conditions,comparisons_to_protected_G= comparisons,
            recommended_alpha=float(chosen),lowest_G_score_alpha=float(best),useful_weak_alphas=list(map(float,useful)),
            checkpoint_paths=paths,production_changed=False)
        dump(out/era/'summary.json',summaries[era]);dump(out/'progress.json',summaries)
        print('VALIDATION SWEEP COMPLETE',era,'recommended alpha',chosen,'best G score alpha',best,flush=True)
    dump(out/'report.json',dict(status='complete',protocol=protocol,eras=summaries))


if __name__=='__main__':main()
