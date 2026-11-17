"""Copy a common note AE into three independently fine-tuned era AE + heads.

Validation selection starts from train-only pretraining. Production starts from
the existing all-5082 checkpoint, using the selected schedule and all era clips.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from embedding.selected_note_data import SelectedNoteCorpus, file_hash, fingerprint
from embedding.selected_style_coordinates import build_tempo_coordinates, G_PROFILE_VERSION
from embedding.era_note_style import (ERA_COMPOSERS, G_NAMES, era_metadata, selected_profiles,
    fit_g_scaler, pair_tables, distance_calibration, EraStyleHead, g_objective, capped_style_weight,
    aggregate_clip_latents, compose_era_objective)
from embedding.note_ae_loss import reconstruction_loss
from train_selected_note_ae import load_model, batches, dump, save_checkpoint, seeded, Metrics


class Runtime:
    def __init__(self, corpus, checkpoint, directory, batch_size=64):
        self.corpus, self.checkpoint, self.directory = corpus, Path(checkpoint), Path(directory)
        self.model, self.payload = load_model(checkpoint)
        self.scaler = self.payload['scaler']
        self.base_hash = file_hash(checkpoint)
        self.windows = corpus.items()
        self.by_clip = [[] for _ in corpus.manifest['clips']]
        for wi, window in enumerate(self.windows):
            window['wi'] = wi
            self.by_clip[window['clip_index']].append(window)
        self.directory.mkdir(parents=True, exist_ok=True)
        contract = dict(corpus=corpus.manifest['id'], checkpoint=self.base_hash, scaler=self.scaler['id'])
        ready = self.directory/'ready.json'
        if ready.exists():
            if json.loads(ready.read_text(encoding='utf-8')) != contract:
                raise ValueError('Frozen context cache contract mismatch')
        else:
            context = np.lib.format.open_memmap(self.directory/'context.npy', mode='w+', dtype=np.float32,
                                               shape=(corpus.manifest['total_notes'],128))
            z = np.zeros((len(self.windows),64), np.float32)
            weights = np.zeros(len(self.windows), np.float32)
            self.model.eval()
            with torch.no_grad():
                for step, group in enumerate(batches(self.windows,batch_size),1):
                    batch = corpus.collate(group,self.scaler)
                    c = self.model.context(batch)
                    encoded = self.model.ae.encode(c,batch['R'],batch['M'],batch['P'],batch['lengths'])['z']
                    for j,w in enumerate(group):
                        context[w['start']:w['stop']] = c[j,:w['N']].numpy()
                        z[w['wi']] = encoded[j].numpy()
                        weights[w['wi']] = batch['M'][j].any(-1).sum().item()
                    if step % 50 == 0: print('cache',self.directory.name,step,flush=True)
            context.flush(); np.save(self.directory/'z.npy',z); np.save(self.directory/'weights.npy',weights)
            dump(ready,contract)
        self.context = np.load(self.directory/'context.npy',mmap_mode='r')
        self.teacher = np.load(self.directory/'z.npy')
        self.weights = np.load(self.directory/'weights.npy')
        self.teacher_clip = np.stack([np.average(self.teacher[[w['wi'] for w in ws]], axis=0,
                                      weights=self.weights[[w['wi'] for w in ws]]) for ws in self.by_clip])

    def batch(self, ids):
        group = [w for ci in ids for w in self.by_clip[ci]]
        group.sort(key=lambda w:w['N'])
        batch = self.corpus.collate(group,self.scaler)
        c = np.zeros((len(group),int(batch['lengths'].max()),128),np.float32)
        for j,w in enumerate(group): c[j,:w['N']] = self.context[w['start']:w['stop']]
        local = {ci:i for i,ci in enumerate(ids)}
        owner = torch.tensor([local[w['clip_index']] for w in group])
        weights = torch.tensor([self.weights[w['wi']] for w in group])
        totals = torch.zeros(len(ids)).index_add(0,owner,weights)
        if (totals<=0).any(): raise ValueError('Clip without observed notes')
        return batch,torch.from_numpy(c),owner,weights/totals[owner],torch.from_numpy(self.teacher[[w['wi'] for w in group]])


def make_model(runtime, scale):
    # Score context remains fixed; the performance encoder AND decoder fine-tune.
    model,_ = load_model(runtime.checkpoint)
    model.score_encoder.requires_grad_(False)
    model.score_encoder.eval()
    head = EraStyleHead(scale)
    return model,head


def forward(model,head,runtime,ids,scale):
    batch,c,owner,weights,teacher = runtime.batch(ids)
    z = model.ae.encode(c,batch['R'],batch['M'],batch['P'],batch['lengths'])['z']
    pred = model.ae.decode(c,z,batch['P'],batch['lengths'],batch['beat_id'])['R_hat']
    rec = reconstruction_loss(pred,batch['R'],batch['M'],batch['P'],batch['beat_id'])
    rec_loss = (rec['window_losses']*batch['clip_weight']).sum()/len(ids)
    clip_z = aggregate_clip_latents(z,owner,weights,len(ids))
    styled = head(clip_z)
    anchor = (((z-teacher)/scale).square().mean(-1)*batch['clip_weight']).sum()/len(ids)
    return dict(batch=batch,pred=pred,rec=rec,rec_loss=rec_loss,z=z,clip_z=clip_z,
                anchor=anchor,**styled)


def clip_batches(ids, size, seed=None):
    ids = list(map(int,ids))
    if seed is not None: np.random.default_rng(seed).shuffle(ids)
    return [ids[i:i+size] for i in range(0,len(ids),size)]


def embedding_diagnostics(z,e,teacher,scale):
    singular = np.linalg.svd(e-e.mean(0),compute_uv=False)
    p = singular**2/max(float((singular**2).sum()),1e-12)
    return dict(e_dimension_std=float(e.std(0).mean()),
                effective_rank=float(np.exp(-(p*np.log(np.maximum(p,1e-15))).sum())),
                z_anchor_mse=float(np.mean(((z-teacher)/scale)**2)),
                z_mean_cosine_to_common=float(np.mean(np.sum(z*teacher,axis=1)/
                    np.maximum(np.linalg.norm(z,axis=1)*np.linalg.norm(teacher,axis=1),1e-12))))


def geometry_metrics(e,d,q,calibration):
    valid = np.isfinite(d)
    if valid.sum()<2: return dict(pairs=int(valid.sum()),score=None)
    actual = 1-e@e.T
    target = d[valid]/(d[valid]+calibration['median'])
    pair_loss = float(np.mean(q[valid]*(actual[valid]-target)**2))
    rng = np.random.default_rng(20261113)
    hits,hinges,weights = [],[],[]
    for a in range(len(e)):
        candidates = np.flatnonzero(valid[a])
        if len(candidates)<4: continue
        ds = d[a,candidates]
        near = candidates[ds<=np.quantile(ds,.2)]
        far = candidates[ds>=np.quantile(ds,.8)]
        for _ in range(4):
            p,n = int(rng.choice(near)),int(rng.choice(far))
            gap = float(d[a,n]-d[a,p])
            if gap<.1*calibration['iqr']: continue
            margin = np.clip(.2*gap/calibration['iqr'],.05,.4)
            weight = min(float(q[a,p]),float(q[a,n]))
            hits.append(float(actual[a,p]<actual[a,n]));weights.append(weight)
            hinges.append(weight*max(0.,float(actual[a,p]-actual[a,n])+margin))
    rho = float(spearmanr(d[valid],actual[valid]).statistic)
    return dict(pairs=int(valid.sum()),pair_mse=pair_loss,spearman=rho if np.isfinite(rho) else None,
                triplets=len(hits),ranking_accuracy=float(np.mean(hits)) if hits else None,
                confidence_weighted_accuracy=float(np.average(hits,weights=weights)) if hits else None,
                rank_hinge=float(np.mean(hinges)) if hinges else None,
                score=.5*pair_loss+.5*float(np.mean(hinges)) if hinges else pair_loss)


@torch.no_grad()
def evaluate(model,head,runtime,ids,scale,table,calibration,size):
    model.eval();head.eval();metrics=Metrics();zs=[];es=[]
    for group in clip_batches(ids,size):
        result=forward(model,head,runtime,group,scale)
        metrics.add(result['rec'],result['pred'],result['batch'])
        zs.append(result['clip_z'].numpy());es.append(result['e'].numpy())
    z,e=np.concatenate(zs),np.concatenate(es)
    return dict(reconstruction=metrics.result(),geometry=geometry_metrics(e,*table,calibration),
                embedding=embedding_diagnostics(z,e,runtime.teacher_clip[ids],scale.numpy())),z,e


def guards(result,baseline):
    rec,base=result['reconstruction'],baseline['reconstruction']
    total=rec['loss']/max(base['loss'],1e-12)
    groups=np.array(rec['group_losses'])/np.maximum(base['group_losses'],1e-8)
    channel=np.array(rec['note_weighted_channel_mse'])/np.maximum(base['note_weighted_channel_mse'],1e-6)
    spread=result['embedding']['e_dimension_std']/max(baseline['embedding']['e_dimension_std'],1e-12)
    rank=result['embedding']['effective_rank']/max(baseline['embedding']['effective_rank'],1e-12)
    checks=dict(total_reconstruction=bool(total<=1.05),every_group=bool(np.all(groups<=1.10)),
                every_channel_mse=bool(np.all(channel<=1.20)),embedding_spread=bool(spread>=.5),
                effective_rank=bool(rank>=.5),latent_drift=bool(result['embedding']['z_anchor_mse']<=.1))
    return dict(passed=all(checks.values()),checks=checks,rec_ratio=total,
                group_ratios=groups.tolist(),channel_mse_ratios=channel.tolist(),spread_ratio=spread,rank_ratio=rank)


def save_model(path,model,head,runtime,era,scale,config,epoch,history,**extra):
    save_checkpoint(path,dict(format='era-note-ae-head-v2',supervision_version=G_PROFILE_VERSION,era=era,state_dict=model.state_dict(),
        head_state_dict=head.state_dict(),head_scale=scale.tolist(),model_config=asdict(model.config),
        score_config=asdict(model.score_encoder.config),scaler=runtime.scaler,
        base_checkpoint=str(runtime.checkpoint),base_sha256=runtime.base_hash,
        corpus_id=runtime.corpus.manifest['id'],config=config,epoch=epoch,history=history,
        score_encoder_frozen=True,**extra))


def fit(runtime,era,train_ids,val_ids,tables,calibration,scale,config,out,size,epochs,production=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    mode=config.get('objective_mode','joint')
    if mode not in ('joint','reconstruction_only','g_only'):raise ValueError('Invalid objective mode')
    seeded(config['seed']);model,head=make_model(runtime,scale)
    optimizer=torch.optim.AdamW([dict(params=model.ae.parameters(),lr=config['ae_lr']),
                                  dict(params=head.parameters(),lr=config['head_lr'])],weight_decay=1e-4)
    check_ids=train_ids if production else val_ids
    check_table=tables['train'] if production else tables['validation']
    baseline_path=out/'baseline.json'
    if baseline_path.exists(): baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    else:
        baseline,_,_=evaluate(model,head,runtime,check_ids,scale,check_table,calibration,size)
        dump(baseline_path,baseline)
    start,history,best,best_epoch=0,[],baseline['geometry']['score'],0
    last=out/'last.pt'
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['config']!=config or state['base_sha256']!=runtime.base_hash or state['corpus_id']!=runtime.corpus.manifest['id']:
            raise ValueError('Resume contract mismatch')
        model.load_state_dict(state['state_dict']);head.load_state_dict(state['head_state_dict'])
        optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng'])
        start,history,best,best_epoch=state['epoch'],state['history'],state['best_score'],state['best_epoch']
    if not (out/'best.pt').exists():
        save_model(out/'best.pt',model,head,runtime,era,scale,config,0,[],baseline=baseline)
    local={ci:i for i,ci in enumerate(train_ids)}
    d,q=tables['train']
    for epoch in range(start+1,epochs+1):
        began=time.time();model.train();model.score_encoder.eval();head.train()
        metrics=Metrics();seen=set();sums=Counter();steps=clip_batches(train_ids,size,config['seed']+epoch)
        requested=0. if mode=='reconstruction_only' else config['lambda_g']*min(1.,epoch/3)
        for step,ids in enumerate(steps,1):
            optimizer.zero_grad(set_to_none=True)
            result=forward(model,head,runtime,ids,scale)
            ix=np.array([local[i] for i in ids])
            g=g_objective(result['e'],torch.from_numpy(d[np.ix_(ix,ix)]),torch.from_numpy(q[np.ix_(ix,ix)]),calibration)
            weight,grad=capped_style_weight(result['rec_loss'],g['loss'],result['z'],requested,config['gradient_budget'])
            objective=compose_era_objective(result['rec_loss'],g['loss'],result['anchor'],result['head_penalty'],weight,mode,
                                          config.get('reconstruction_weight',1.))
            if not torch.isfinite(objective): raise ValueError('Nonfinite objective')
            metrics.add(result['rec'],result['pred'],result['batch'])
            if mode=='g_only':
                # The cap has already read the reconstruction gradient. Its unused
                # decoder graph need not survive the G/anchor backward pass.
                del result['rec'],result['pred'],result['rec_loss']
            objective.backward();torch.nn.utils.clip_grad_norm_(list(model.ae.parameters())+list(head.parameters()),1.,error_if_nonfinite=True)
            optimizer.step();seen.update(ids)
            for key,value in dict(g_loss=float(g['loss'].detach()),anchor=float(result['anchor'].detach()),
                                  pairs=g['pairs'],triplets=g['triplets'],
                                  ceiling_active_steps=int(requested>0 and weight>=requested),**grad).items(): sums[key]+=value
            if step%20==0 or step==len(steps):
                status=dict(stage='production' if production else 'selection',era=era,epoch=epoch,step=step,
                            steps=len(steps),clips_seen=len(seen),seconds=time.time()-began,
                            reconstruction_so_far=metrics.result()['loss'])
                dump(out/'status.json',status)
                print(out.name,era,epoch,f'{step}/{len(steps)}',f"rec={status['reconstruction_so_far']:.5f}",flush=True)
        if seen!=set(train_ids):raise AssertionError('Epoch omitted clips')
        result,_,_=evaluate(model,head,runtime,check_ids,scale,check_table,calibration,size)
        guard=guards(result,baseline)
        row=dict(epoch=epoch,train=metrics.result(),validation=None if production else result,
                 production_fit=result if production else None,guard=guard,clips_seen=len(seen),
                 style={k:v/len(steps) if k not in ('pairs','triplets','ceiling_active_steps') else v for k,v in sums.items()},seconds=time.time()-began)
        history.append(row)
        score=result['geometry']['score']
        if guard['passed'] and score is not None and score<best-1e-5:
            best,best_epoch=score,epoch
            save_model(out/'best.pt',model,head,runtime,era,scale,config,epoch,history,baseline=baseline,evaluation=result)
        save_model(last,model,head,runtime,era,scale,config,epoch,history,optimizer=optimizer.state_dict(),
                   torch_rng=torch.get_rng_state(),best_score=best,best_epoch=best_epoch,baseline=baseline)
        dump(out/'history.json',history)
        print('EPOCH',era,config['lambda_g'],epoch,'G',round(score,6),'guard',guard['passed'],'best',best_epoch,
              'seconds',round(time.time()-began,1),flush=True)
    summary=dict(era=era,config=config,best_epoch=best_epoch,best_score=best,baseline=baseline,
                 path=str(out/'best.pt'),history=history,production=production)
    dump(out/'result.json',summary)
    return summary


def restore(path,runtime):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('supervision_version')!=G_PROFILE_VERSION:raise ValueError('Invalidated G supervision version')
    if state['base_sha256']!=runtime.base_hash:raise ValueError('Wrong base')
    scale=torch.tensor(state['head_scale'])
    model,head=make_model(runtime,scale)
    model.load_state_dict(state['state_dict']);head.load_state_dict(state['head_state_dict'])
    return model,head,scale,state


def prepare(args,corpus):
    args.out.mkdir(parents=True,exist_ok=True)
    metadata=era_metadata(corpus,args.selected_csv)
    tempo_cells,coordinates=build_tempo_coordinates(corpus,args.selected_csv,args.out/'coordinates')
    path=args.out/'profiles.npz'
    contract=dict(corpus_id=corpus.manifest['id'],metadata=fingerprint(metadata),version=G_PROFILE_VERSION,
                  coordinates_id=coordinates['id'])
    if path.exists():
        if json.loads((args.out/'profile_contract.json').read_text(encoding='utf-8'))!=contract:
            raise ValueError('Profile contract changed')
        with np.load(path) as data: raw,known,quality=[data[k] for k in ('raw','known','quality')]
    else:
        raw,known,quality=selected_profiles(corpus,tempo_cells)
        np.savez_compressed(path,raw=raw,known=known,quality=quality)
        dump(args.out/'profile_contract.json',contract)
    dump(args.out/'metadata.json',metadata)
    audit={era:{split:dict(clips=len(ids:=[i for i,m in enumerate(metadata) if m['era']==era and m['split']==split]),
             works=len({metadata[i]['work'] for i in ids}),all_core_g=int(known[ids,:9].all(1).sum()),
             feature_counts=known[ids].sum(0).tolist()) for split in ('train','validation','test')} for era in ERA_COMPOSERS}
    dump(args.out/'input_audit.json',dict(eras=audit,composers=ERA_COMPOSERS,features=list(G_NAMES),
         all_clips=len(metadata),g_missing_policy='keep all reconstruction clips; no fabricated G labels',
         reference='same split/work/XML traversal; distinct source MIDI; at least3; target may be included',
         tempo='actual XML quarter interval references; quarter-weighted tempo summaries; contiguous absolute interval roughness',
         coordinates_id=coordinates['id'],supervision_version=G_PROFILE_VERSION))
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    return metadata,raw,known,quality


def tables_for(metadata,raw,known,quality,era,production=False):
    ids={split:[i for i,m in enumerate(metadata) if m['era']==era and (production or m['split']==split)]
         for split in (('train',) if production else ('train','validation','test'))}
    # Duplicate source MIDI never increases fitting support.
    seen=set();fit_ids=[]
    for i in ids['train']:
        if metadata[i]['source'] not in seen: fit_ids.append(i);seen.add(metadata[i]['source'])
    scaler=fit_g_scaler(raw,known,fit_ids)
    table={}
    for split,indices in ids.items():
        if production:
            # All-data reference cohorts remain work-isolated; permit pairing all works.
            md=[{**m,'split':'production'} for m in metadata]
        else:md=metadata
        table[split]=pair_tables(raw,known,quality,md,scaler,indices)
    calibration=distance_calibration(table['train'][0])
    return ids,scaler,table,calibration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-run',type=Path,required=True)
    parser.add_argument('--selected-csv',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','select','production','all'],default='all')
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--batch-clips',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    corpus=SelectedNoteCorpus(args.common_run/'corpus')
    metadata,raw,known,quality=prepare(args,corpus)
    protocol=dict(version='era-note-ae-head-v2',supervision_version=G_PROFILE_VERSION,corpus_id=corpus.manifest['id'],epochs=args.epochs,
        batch_clips=args.batch_clips,lambda_candidates=[.1,.3],ae_lr=.00005,head_lr=.001,
        gradient_budget=.25,anchor_weight=.05,head_penalty=.01,warmup_epochs=3,
        redundant_ceiling='A larger ceiling is algebraically identical if the smaller ceiling is inactive in every step of the complete trajectory; reuse with explicit equivalence record',
        score_encoder='frozen; performance encoder and decoder independently fine-tune per era',
        g='0.5 confidence-weighted bounded distance MSE + 0.5 confidence-weighted adaptive triplet',
        guard='rec<=1.05 baseline; each group<=1.10; each channel MSE<=1.20; z drift<=.1; spread/rank>=.5',
        selection='minimum validation G score among guard-passing epochs including common baseline',
        test='once after all choices frozen; cohort targets use held-out recordings, no test fitting',
        production='existing all5082 checkpoint copied per era; selected max epoch/weight; all era clips',
        pretraining_hashes={name:file_hash(args.common_run/name) for name in ('evaluated_best.pt','final_all_5082.pt')})
    protocol_path=args.out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Experiment protocol changed; use new output')
    dump(protocol_path,protocol)
    if args.stage=='prepare':return
    selections={}
    if args.stage in ('all','select'):
        runtime=Runtime(corpus,args.common_run/'evaluated_best.pt',args.out/'evaluation_cache')
        for era in ERA_COMPOSERS:
            ids,scaler,table,calibration=tables_for(metadata,raw,known,quality,era)
            scale=torch.tensor(np.maximum(runtime.teacher_clip[ids['train']].std(0),.05))
            dump(args.out/era/'g_train_scaler.json',dict(scaler=scaler,calibration=calibration))
            results=[]
            for weight in protocol['lambda_candidates']:
                config=dict(lambda_g=weight,ae_lr=protocol['ae_lr'],head_lr=protocol['head_lr'],gradient_budget=.25,seed=2026)
                trial_out=args.out/era/f'lambda_{weight}'
                if results and len(results[0]['history'])==args.epochs and all(
                        h['style'].get('ceiling_active_steps',1)==0 for h in results[0]['history']):
                    # Induction: identical initialization/RNG and every update uses
                    # the same cap, hence increasing an inactive ceiling changes
                    # neither objective, gradients, optimizer state nor next step.
                    source=results[0];trial_out.mkdir(parents=True,exist_ok=True)
                    proof=dict(source=source['path'],source_lambda=source['config']['lambda_g'],
                               equivalent_lambda=weight,all_epochs=args.epochs,all_steps_smaller_ceiling_inactive=True,
                               reason='Exact cap-equivalent trajectory; no separately trained checkpoint')
                    dump(trial_out/'equivalence.json',proof)
                    dump(trial_out/'baseline.json',source['baseline']);dump(trial_out/'history.json',source['history'])
                    result={**source,'config':config,'equivalence':proof}
                    dump(trial_out/'result.json',result);results.append(result)
                    print('CAP-EQUIVALENT',era,weight,'reuses',source['config']['lambda_g'],flush=True)
                    continue
                pruning_path=trial_out/'pruned.json'
                pruning=json.loads(pruning_path.read_text(encoding='utf-8')) if pruning_path.exists() else None
                budget=pruning['completed_epochs'] if pruning else args.epochs
                if (not isinstance(budget,int) or not 1<=budget<=args.epochs
                        or (pruning and not (trial_out/'last.pt').exists())):
                    raise ValueError('Pruning must refer to a completed, saved epoch')
                result=fit(runtime,era,ids['train'],ids['validation'],table,calibration,scale,config,
                           trial_out,args.batch_clips,budget)
                if pruning:
                    result['pruning']=pruning
                    dump(trial_out/'result.json',result)
                results.append(result)
            selected=min(results,key=lambda r:r['best_score'])
            frozen=args.out/era/'selection.json'
            selection={k:v for k,v in selected.items() if k!='history'}
            if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))!=selection:
                raise ValueError('Selection changed after freezing')
            dump(frozen,selection);selections[era]=selection
        # Evaluate only once every era's hyperparameters have been frozen.
        for era,selection in selections.items():
            test_path=args.out/era/'test.json'
            if test_path.exists():continue
            ids,scaler,table,calibration=tables_for(metadata,raw,known,quality,era)
            model,head,scale,state=restore(selection['path'],runtime)
            test,_,_=evaluate(model,head,runtime,ids['test'],scale,table['test'],calibration,args.batch_clips)
            common,identity=make_model(runtime,scale)
            baseline,_,_=evaluate(common,identity,runtime,ids['test'],scale,table['test'],calibration,args.batch_clips)
            dump(test_path,dict(selected=test,common=baseline,guard=guards(test,baseline),
                 checkpoint_sha256=file_hash(selection['path']),independent_of_supervised_finetuning=True,
                 limitation='G proxy evaluation; common AE test previously inspected; no perceptual-style claim'))
        del runtime
    if args.stage in ('all','production'):
        runtime=Runtime(corpus,args.common_run/'final_all_5082.pt',args.out/'production_cache')
        reports={};export={k:[] for k in ('indices','z_ae','z_style','e','era')}
        for era in ERA_COMPOSERS:
            selection=json.loads((args.out/era/'selection.json').read_text(encoding='utf-8'))
            ids,scaler,table,calibration=tables_for(metadata,raw,known,quality,era,True)
            scale=torch.tensor(np.maximum(runtime.teacher_clip[ids['train']].std(0),.05))
            dump(args.out/era/'g_production_scaler.json',dict(scaler=scaler,calibration=calibration))
            result=fit(runtime,era,ids['train'],[],table,calibration,scale,selection['config'],args.out/era/'production',
                       args.batch_clips,selection['best_epoch'],True)
            final=args.out/era/'final.pt';shutil.copyfile(result['path'],final)
            model,head,scale,state=restore(final,runtime)
            evaluation,z,e=evaluate(model,head,runtime,ids['train'],scale,table['train'],calibration,args.batch_clips)
            with torch.no_grad():zs=head(torch.from_numpy(z))['z_style'].numpy()
            export['indices'].extend(ids['train']);export['z_ae'].extend(z);export['z_style'].extend(zs)
            export['e'].extend(e);export['era'].extend([era]*len(ids['train']))
            reports[era]=dict(clips=len(ids['train']),selected_lambda=selection['config']['lambda_g'],
                selection_epoch=selection['best_epoch'],production_epoch=state['epoch'],checkpoint=str(final),
                sha256=file_hash(final),production_training_diagnostics=evaluation,baseline=result['baseline'],
                guard=guards(evaluation,result['baseline']),independent_test=False)
            dump(args.out/'production_progress.json',reports)
        order=np.argsort(export['indices'])
        if sorted(export['indices'])!=list(range(len(metadata))):raise AssertionError('Missing/duplicate export')
        np.savez_compressed(args.out/'all_5082_era_embeddings.npz',
             **{k:np.asarray(v)[order] for k,v in export.items() if k!='indices'},
             perf_ids=np.array([m['perf_id'] for m in metadata]),composition_ids=np.array([m['work'] for m in metadata]))
        dump(args.out/'final_report.json',dict(status='complete',eras=reports,clips=len(metadata),
            embeddings=str(args.out/'all_5082_era_embeddings.npz'),
            comparison_policy='cosine distances within the same era only; era spaces not cross-calibrated',
            limitation='G is a measured profile proxy, not listening ground truth; production fits all data'))


if __name__=='__main__':main()
