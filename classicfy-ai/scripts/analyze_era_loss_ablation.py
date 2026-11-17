"""Controlled post-hoc attribution of reconstruction and G fine-tuning terms.

Fixed common start, clips, seeds, horizon, learning rates, heads and preservation
regularizers. No checkpoint/hyperparameter is selected from these test results.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_era_note_ae import (ERA_COMPOSERS,SelectedNoteCorpus,Runtime,make_model,restore,forward,
    fit,tables_for,clip_batches,Metrics,geometry_metrics,embedding_diagnostics,guards,dump,file_hash,G_PROFILE_VERSION)


def evaluation_triplets(distance,confidence,calibration):
    rng=np.random.default_rng(20261113);rows=[]
    for a in range(len(distance)):
        candidates=np.flatnonzero(np.isfinite(distance[a]))
        if len(candidates)<4:continue
        ds=distance[a,candidates]
        near=candidates[ds<=np.quantile(ds,.2)];far=candidates[ds>=np.quantile(ds,.8)]
        for _ in range(4):
            p,n=int(rng.choice(near)),int(rng.choice(far))
            if distance[a,n]-distance[a,p]<.1*calibration['iqr']:continue
            rows.append((a,p,n))
    return np.asarray(rows,dtype=int).reshape(-1,3)


def paired_work_interval(left,right,works,*,kind,replicates=5000,seed=20261113):
    """Paired anchor-work bootstrap, conditional on a fixed candidate pool.

    Accuracy uses per-anchor means then equal work means, not independent
    triplet resampling. Nine predeclared primary comparisons get an additional
    Bonferroni-adjusted interval; this does not cover training-seed uncertainty.
    """
    left,right=np.asarray(left,float),np.asarray(right,float)
    works=np.asarray(works)
    if left.shape!=right.shape or left.shape!=works.shape:raise ValueError('Paired samples must align')
    mask=np.isfinite(left)&np.isfinite(right)
    keys=np.unique(works[mask]);a=[];b=[]
    for key in keys:
        use=mask&(works==key);a.append(left[use].mean());b.append(right[use].mean())
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<2:raise ValueError('At least two works required for uncertainty')
    rng=np.random.default_rng(seed);indices=rng.integers(len(a),size=(replicates,len(a)))
    la,lb=a[indices].mean(1),b[indices].mean(1)
    if kind=='accuracy':samples=100*(la-lb);point=100*(a.mean()-b.mean())
    elif kind in ('reconstruction','distance_error'):
        samples=100*(la/np.maximum(lb,1e-12)-1);point=100*(a.mean()/max(b.mean(),1e-12)-1)
    else:raise ValueError('Unknown paired comparison')
    ci=np.quantile(samples,[.025,.975]).tolist()
    family_ci=np.quantile(samples,[.05/(2*9),1-.05/(2*9)]).tolist()
    lower_better=kind!='accuracy'
    supported=(family_ci[1]<0 if lower_better else family_ci[0]>0)
    worse=(family_ci[0]>0 if lower_better else family_ci[1]<0)
    return dict(work_count=len(keys),paired_clips=int(mask.sum()),work_macro_delta=float(point),
        unit='percentage_points' if kind=='accuracy' else 'relative_percent',
        ci95=ci,familywise95_approx_for9_contrasts=family_ci,
        direction='improvement_supported' if supported else 'degradation_supported' if worse else 'uncertain',
        bootstrap_replicates=replicates,per_work_left=a.tolist(),per_work_right=b.tolist(),works=keys.tolist())


@torch.no_grad()
def diagnose(model,head,runtime,ids,scale,table,calibration,size):
    model.eval();head.eval();metrics=Metrics();zs=[];es=[]
    positions={ci:i for i,ci in enumerate(ids)};per_clip=np.zeros(len(ids))
    for group in clip_batches(ids,size):
        r=forward(model,head,runtime,group,scale)
        metrics.add(r['rec'],r['pred'],r['batch'])
        for ci,loss,weight in zip(r['batch']['clip_indices'].tolist(),r['rec']['window_losses'].numpy(),r['batch']['clip_weight'].numpy()):
            per_clip[positions[ci]]+=float(loss*weight)
        zs.append(r['clip_z'].numpy());es.append(r['e'].numpy())
    z,e=np.concatenate(zs),np.concatenate(es)
    triplets=evaluation_triplets(*table,calibration)
    a,p,n=triplets.T
    distance=1-e@e.T
    target_distance,confidence=table
    valid=np.isfinite(target_distance)
    target=np.where(valid,target_distance/(target_distance+calibration['median']),0.)
    pair_error=np.where(valid,confidence*(distance-target)**2,0.)
    pair_count=valid.sum(1)
    per_anchor_distance=np.divide(pair_error.sum(1),pair_count,out=np.full(len(ids),np.nan),where=pair_count>0)
    correct=(distance[a,p]<distance[a,n]).astype(float)
    count=np.bincount(a,minlength=len(ids));wins=np.bincount(a,weights=correct,minlength=len(ids))
    accuracy=np.divide(wins,count,out=np.full(len(ids),np.nan),where=count>0)
    result=dict(reconstruction=metrics.result(),geometry=geometry_metrics(e,*table,calibration),
                embedding=embedding_diagnostics(z,e,runtime.teacher_clip[ids],scale.numpy()))
    assert abs(correct.mean()-result['geometry']['ranking_accuracy'])<1e-10
    assert abs(per_clip.mean()-result['reconstruction']['loss'])<1e-6
    return result,dict(ids=np.asarray(ids),z_ae=z,e=e,reconstruction_per_clip=per_clip,
                       distance_error_per_anchor=per_anchor_distance,
                       ranking_per_anchor=accuracy,triplets=triplets,triplet_correct=correct)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--common-run',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    out=args.run/'ablation';out.mkdir(exist_ok=True)
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    metadata=read(args.run/'metadata.json')
    profile_contract=read(args.run/'profile_contract.json')
    if profile_contract['version']!=G_PROFILE_VERSION:raise ValueError('Invalidated G profile version')
    with np.load(args.run/'profiles.npz') as f:
        raw,known,quality=[f[k] for k in ('raw','known','quality')]
    selections={era:read(args.run/era/'selection.json') for era in ERA_COMPOSERS}
    protocol=dict(version='era-finetuning-loss-ablation-v2',supervision=profile_contract,seed=2026,batch_clips=16,
        common_checkpoint_sha256=file_hash(args.common_run/'evaluated_best.pt'),
        conditions=['common_frozen','reconstruction_only','g_only','joint'],
        comparison='Incremental fine-tuning terms; all models already have reconstruction pretraining',
        equal_conditions='same common initialization, train clips, seed, head, LR, epoch horizon, anchor/head penalties and G-gradient cap',
        G_only='direct reconstruction term removed; reconstruction gradient still defines the common G cap',
        selection='control last epoch at the already-fixed joint horizon; do not select best control epoch or retune production',
        endpoints='fixed G near/far ranking, pair distance MSE, Spearman, reconstruction, raw z drift, collapse checks',
        uncertainty='paired anchor-work bootstrap for ranking, distance error and reconstruction; candidate pool fixed; 5000 draws; 95% and approximate Bonferroni familywise95% for9 contrasts separately per outcome',
        post_hoc=True,test_previously_inspected=True,training_seed_replications=1,
        eras={era:dict(epochs=s['best_epoch'],config=s['config'],joint_sha256=file_hash(s['path'])) for era,s in selections.items()})
    protocol_path=out/'protocol.json'
    if protocol_path.exists() and read(protocol_path)!=protocol:raise ValueError('Ablation protocol changed')
    dump(protocol_path,protocol)
    corpus=SelectedNoteCorpus(args.common_run/'corpus')
    runtime=Runtime(corpus,args.common_run/'evaluated_best.pt',args.run/'evaluation_cache')
    summary={}
    for era,selection in selections.items():
        ids,scaler,tables,calibration=tables_for(metadata,raw,known,quality,era)
        # Do not silently refit different supervision for the controls.
        assert read(args.run/era/'g_train_scaler.json')==dict(scaler=scaler,calibration=calibration)
        scale=torch.tensor(np.maximum(runtime.teacher_clip[ids['train']].std(0),.05))
        paths={'common_frozen':runtime.checkpoint,'joint':Path(selection['path'])}
        for mode in ('reconstruction_only','g_only'):
            config={**selection['config'],'objective_mode':mode}
            fit(runtime,era,ids['train'],ids['validation'],tables,calibration,scale,config,out/era/mode,
                protocol['batch_clips'],selection['best_epoch'])
            paths[mode]=out/era/mode/('last.pt' if selection['best_epoch'] else 'best.pt')
        results={};arrays={}
        for condition in protocol['conditions']:
            if condition=='common_frozen':model,head=make_model(runtime,scale)
            else:
                model,head,loaded_scale,state=restore(paths[condition],runtime)
                torch.testing.assert_close(loaded_scale,scale)
            results[condition]={};arrays[condition]={}
            for split in ('validation','test'):
                destination=out/era/condition/f'{split}_evaluation.json'
                tensor_path=out/era/condition/f'{split}_diagnostics.npz'
                signature=dict(checkpoint_sha256=file_hash(paths[condition]),split=split,
                               corpus_id=corpus.manifest['id'],scope='post-hoc fixed-horizon diagnostic; no selection')
                if destination.exists():
                    saved=read(destination)
                    if saved['signature']!=signature:raise ValueError('Diagnostic checkpoint changed')
                    result=saved['metrics']
                    with np.load(tensor_path) as f:data={k:f[k] for k in f.files}
                else:
                    result,data=diagnose(model,head,runtime,ids[split],scale,tables[split],calibration,protocol['batch_clips'])
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(tensor_path,**data)
                    dump(destination,dict(signature=signature,metrics=result))
                results[condition][split]=result;arrays[condition][split]=data
        comparisons={}
        for split in ('validation','test'):
            comparisons[split]={}
            work=np.array([metadata[i]['work'] for i in ids[split]])
            for other in ('common_frozen','reconstruction_only','g_only'):
                joint,control=arrays['joint'][split],arrays[other][split]
                assert np.array_equal(joint['triplets'],control['triplets'])
                metric_a,metric_b=results['joint'][split],results[other][split]
                comparisons[split]['joint_minus_'+other]=dict(
                    micro_ranking_delta_pp=100*(metric_a['geometry']['ranking_accuracy']-metric_b['geometry']['ranking_accuracy']),
                    G_score_change_percent=100*(metric_a['geometry']['score']/metric_b['geometry']['score']-1),
                    ranking_work_bootstrap=paired_work_interval(joint['ranking_per_anchor'],control['ranking_per_anchor'],work,kind='accuracy'),
                    distance_error_work_bootstrap=paired_work_interval(joint['distance_error_per_anchor'],control['distance_error_per_anchor'],work,kind='distance_error'),
                    reconstruction_work_bootstrap=paired_work_interval(joint['reconstruction_per_clip'],control['reconstruction_per_clip'],work,kind='reconstruction'))
        summary[era]=dict(epochs=selection['best_epoch'],results=results,comparisons=comparisons,
                         common_preservation_guards={condition:{split:guards(results[condition][split],results['common_frozen'][split])
                         for split in ('validation','test')} for condition in protocol['conditions'] if condition!='common_frozen'})
        dump(out/era/'summary.json',summary[era]);dump(out/'progress.json',summary)
        print('ABLATION COMPLETE',era,flush=True)
    dump(out/'report.json',dict(status='complete',protocol=protocol,eras=summary,
        limitation='Post-hoc, single training seed, proxy G labels. Anchor-work intervals hold the candidate pool fixed; not new external confirmation or a perceptual listening test.'))


if __name__=='__main__':main()
