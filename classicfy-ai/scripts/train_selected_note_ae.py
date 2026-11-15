"""Validation-selected note AE experiments, followed by a full-5082 refit.

The test set is opened only after selection. The production refit deliberately
uses all clips and is NOT reported as having an independent test evaluation.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from embedding.selected_note_data import SelectedNoteCorpus, pack_corpus, fingerprint, file_hash
from embedding.note_autoencoder import NotePerformanceAutoencoder, NoteAEConfig
from embedding.note_score_encoder import NoteScoreEncoderConfig
from embedding.note_ae_loss import reconstruction_loss
from features.note_performance import note_performance_schema
from features.note_score_context import note_score_schema


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(path)


def save_checkpoint(path,payload):
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp);tmp.replace(path)


def seeded(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def make_model(trial):
    hidden=trial['hidden_size']
    config=NoteAEConfig(context_dim=128,condition_dim=64,input_dim=128,
        hidden_size=hidden,layers=trial['layers'],dropout=.1,latent_dim=64)
    return NotePerformanceAutoencoder(config,NoteScoreEncoderConfig(hidden_size=64,context_dim=128))


def batches(items,size,*,seed=None):
    order=list(range(len(items)))
    if seed is None:
        order.sort(key=lambda i:items[i]['N'])
        return [[items[i] for i in order[j:j+size]] for j in range(0,len(order),size)]
    rng=np.random.default_rng(seed);rng.shuffle(order)
    result=[]
    for left in range(0,len(order),size*20):
        bucket=sorted(order[left:left+size*20],key=lambda i:items[i]['N'])
        result.extend([items[i] for i in bucket[j:j+size]] for j in range(0,len(bucket),size))
    rng.shuffle(result)
    assert sum(len(batch) for batch in result)==len(items)
    return result


class Metrics:
    def __init__(self):
        self.loss=self.weight=0.
        self.groups=np.zeros(5);self.support=np.zeros(5)
        self.sse=np.zeros(11);self.count=np.zeros(11);self.sum=np.zeros(11);self.sq=np.zeros(11)
        self.clips={}
    def add(self,loss,prediction,batch):
        weights=batch['clip_weight'].detach().numpy()
        losses=loss['window_losses'].detach().numpy()
        valid=loss['valid_windows'].detach().numpy()
        self.loss+=float((losses*weights).sum());self.weight+=float((weights*valid).sum())
        self.groups+=(loss['group_losses'].detach().numpy()*weights[:,None]).sum(0)
        self.support+=(loss['group_mask'].detach().numpy()*weights[:,None]).sum(0)
        for ci,v,w in zip(batch['clip_indices'].tolist(),losses,weights):
            self.clips[ci]=self.clips.get(ci,0.)+float(v*w)
        m=batch['M'].numpy();y=batch['R'].numpy();p=prediction.detach().numpy()
        self.sse+=np.where(m,(p-y)**2,0).sum((0,1))
        self.count+=m.sum((0,1));self.sum+=np.where(m,y,0).sum((0,1));self.sq+=np.where(m,y*y,0).sum((0,1))
    def result(self):
        denominator=np.maximum(self.count,1);variance=self.sq-self.sum*self.sum/denominator
        return dict(loss=self.loss/max(self.weight,1e-12),
            group_losses=(self.groups/np.maximum(self.support,1e-12)).tolist(),
            note_weighted_channel_mse=(self.sse/denominator).tolist(),
            note_weighted_channel_r2=[float(1-a/b) if b>1e-8 else None for a,b in zip(self.sse,variance)],
            observed_channel_counts=self.count.astype(int).tolist(),clips=len(self.clips))


@torch.no_grad()
def evaluate(model,corpus,scaler,items,batch_size,*,mode='normal',z_pool=None):
    model.eval();metrics=Metrics();position=0
    ordered=batches(items,batch_size)
    if mode=='shuffle_z':
        if z_pool is None:
            z_pool=torch.cat([model.encode_batch(corpus.collate(group,scaler))['z'] for group in ordered])
        permutation=np.random.default_rng(2026).permutation(len(z_pool))
        z_pool=z_pool[permutation]
    for group in ordered:
        batch=corpus.collate(group,scaler)
        if mode=='normal':prediction=model(batch)['R_hat']
        elif mode=='train_channel_mean':prediction=torch.zeros_like(batch['R'])
        else:
            context=model.context(batch)
            if mode=='zero_z':z=torch.zeros((len(group),model.config.latent_dim))
            elif mode=='shuffle_z':z=z_pool[position:position+len(group)]
            else:raise ValueError(mode)
            prediction=model.ae.decode(context,z,batch['P'],batch['lengths'],batch['beat_id'])['R_hat']
        loss=reconstruction_loss(prediction,batch['R'],batch['M'],batch['P'],batch['beat_id'],delta=1.)
        metrics.add(loss,prediction,batch);position+=len(group)
    return metrics.result()


def checkpoint_payload(model,trial,scaler,corpus,epoch,history,**extra):
    return dict(format='selected-musicxml-note-ae-v1',state_dict=model.state_dict(),
        model_config=asdict(model.config),score_config=asdict(model.score_encoder.config),
        scaler=scaler,corpus_id=corpus.manifest['id'],trial=trial,epoch=epoch,history=history,
        R_schema=note_performance_schema(),S_schema=note_score_schema(),
        preparation='original MusicXML S + raw performance MIDI note R11; selected clip boundary masks',
        G_loss_enabled=False,**extra)


def load_model(path):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=NotePerformanceAutoencoder(NoteAEConfig(**payload['model_config']),NoteScoreEncoderConfig(**payload['score_config']))
    model.load_state_dict(payload['state_dict'])
    model.r_scaler_fingerprint=payload['scaler']['id'];model.score_scale_fingerprint=payload['scaler']['id']
    model.eval()
    return model,payload


def write_curves(output, comparisons, production):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for name,result in comparisons.items():
        history=result['history']
        axes[0].plot([r['epoch'] for r in history],
                     [r['validation']['loss'] for r in history],label=name)
    axes[0].set(title='Validation: equal-clip, equal-group Huber',xlabel='Epoch',ylabel='Loss')
    axes[0].legend()
    history=production['history']
    axes[1].plot([r['epoch'] for r in history],[r['train']['loss'] for r in history])
    axes[1].set(title='Production refit: all 5,082 clips (training only)',xlabel='Epoch',ylabel='Loss')
    for axis in axes:axis.grid(alpha=.2)
    figure.savefig(Path(output)/'loss_curves.png',dpi=160)
    plt.close(figure)


def fit(trial,corpus,scaler,out,*,max_epochs,batch_size,all_data=False,lr_schedule=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    seeded(trial['seed']);model=make_model(trial)
    model.r_scaler_fingerprint=scaler['id'];model.score_scale_fingerprint=scaler['id']
    optimizer=torch.optim.AdamW(model.parameters(),lr=trial['learning_rate'],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=3,threshold=1e-4,min_lr=1e-5)
    start,history,best,best_epoch,stale=0,[],float('inf'),0,0
    last=out/'last.pt'
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['corpus_id']!=corpus.manifest['id'] or state['scaler']['id']!=scaler['id'] or state['trial']!=trial:
            raise ValueError('Resume data/model contract mismatch')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);torch.set_rng_state(state['torch_rng'])
        start,history,best,best_epoch,stale=(state[k] for k in ('epoch','history','best_loss','best_epoch','stale'))
    items=corpus.items(None if all_data else 'train')
    validation=[] if all_data else corpus.items('validation')
    clip_ids={w['clip_index'] for w in items}
    normalizer=len(items)/len(clip_ids)
    parameters=sum(p.numel() for p in model.parameters())
    for epoch in range(start+1,max_epochs+1):
        if not all_data and stale>=7:break
        if lr_schedule is not None:
            for group in optimizer.param_groups:group['lr']=lr_schedule[epoch-1]
        learning_rate=optimizer.param_groups[0]['lr']
        model.train();metrics=Metrics();seen=set();began=time.time()
        ordered=batches(items,batch_size,seed=trial['seed']+epoch)
        for step,group in enumerate(ordered,1):
            batch=corpus.collate(group,scaler);seen.update(batch['clip_indices'].tolist())
            optimizer.zero_grad(set_to_none=True)
            prediction=model(batch)['R_hat']
            result=reconstruction_loss(prediction,batch['R'],batch['M'],batch['P'],batch['beat_id'],delta=1.)
            if not result['valid_windows'].any():
                metrics.add(result,prediction,batch)
                continue
            group_weights=torch.tensor([1.,1.,1.,1.,trial['pedal_weight']])
            present=result['group_mask']*group_weights[None,:]
            windows=(result['group_losses']*group_weights[None,:]).sum(-1)/present.sum(-1).clamp_min(1)
            loss=(windows*batch['clip_weight']).mean()*normalizer
            if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            metrics.add(result,prediction,batch)
            if step%50==0 or step==len(ordered):
                dump(out/'status.json',dict(stage='full_refit' if all_data else 'training',epoch=epoch,
                    step=step,steps=len(ordered),clips_seen=len(seen),seconds=time.time()-began,
                    train_loss_so_far=metrics.loss/max(metrics.weight,1e-12),learning_rate=learning_rate))
                print(f"{out.name} epoch {epoch} {step}/{len(ordered)} loss={metrics.loss/max(metrics.weight,1e-12):.5f}",flush=True)
        if seen!=clip_ids:raise AssertionError('An epoch omitted a selected clip')
        train=metrics.result()
        val=None if all_data else evaluate(model,corpus,scaler,validation,batch_size)
        row=dict(epoch=epoch,learning_rate=learning_rate,train=train,validation=val,
                 seconds=time.time()-began,clips_seen=len(seen))
        history.append(row)
        selected=train['loss'] if all_data else val['loss']
        improved=selected<best-1e-4
        if improved:best,best_epoch,stale=selected,epoch,0
        else:stale+=1
        if not all_data:scheduler.step(selected)
        payload=checkpoint_payload(model,trial,scaler,corpus,epoch,history,
            training_scope='all_5082_refit' if all_data else 'train_split',
            independent_test_available=not all_data,best_loss=best,best_epoch=best_epoch,stale=stale,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state())
        save_checkpoint(last,payload)
        if improved and not all_data:
            compact={k:v for k,v in payload.items() if k not in ('optimizer','scheduler','torch_rng')}
            save_checkpoint(out/'best.pt',compact)
        dump(out/'history.json',history)
        dump(out/'status.json',dict(stage='epoch_complete',epoch=epoch,best_epoch=best_epoch,best_loss=best,
            max_epochs=max_epochs,parameters=parameters,stale=stale))
        print(f"{out.name} EPOCH {epoch} train={train['loss']:.6f} val={selected:.6f} best={best_epoch}",flush=True)
    summary=dict(trial=trial,best_loss=best,best_epoch=best_epoch,completed_epochs=len(history),
                 parameters=parameters,path=str(out/'best.pt' if not all_data else last),history=history)
    dump(out/'result.json',summary)
    return summary


@torch.no_grad()
def export_embeddings(model,corpus,scaler,path,batch_size):
    model.eval();count=len(corpus.manifest['clips'])
    sums=np.zeros((count,model.config.latent_dim));weights=np.zeros(count)
    for group in batches(corpus.items(),batch_size):
        batch=corpus.collate(group,scaler);z=model.encode_batch(batch)['z'].numpy()
        for bi,(window,vector) in enumerate(zip(group,z)):
            weight=int(batch['M'][bi].any(-1).sum());ci=window['clip_index']
            sums[ci]+=vector*weight;weights[ci]+=weight
    if np.any(weights<=0):raise AssertionError('Missing clip embedding')
    z=(sums/weights[:,None]).astype(np.float32)
    e=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    if not np.isfinite(z).all():raise ValueError('Nonfinite final embedding')
    np.savez_compressed(path,z=z,e=e,perf_ids=np.array([r['perf_id'] for r in corpus.manifest['clips']]),
                        composition_ids=np.array([r['composition_id'] for r in corpus.manifest['clips']]))
    return dict(clips=count,latent_dim=z.shape[1],mean_dimension_std=float(z.std(0).mean()),
                aggregation='observed-note-count weighted mean of unnormalized window z; unknown-only windows weight zero; final L2 normalization for e')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--max-notes',type=int,default=128)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    manifest=pack_corpus(args.prepared/'clips',args.out/'corpus',max_notes=args.max_notes)
    corpus=SelectedNoteCorpus(args.out/'corpus');scaler=corpus.fit_scalers()
    for name,expected in manifest['files'].items():
        if file_hash(args.out/'corpus'/(name+'.npy'))!=expected:
            raise ValueError('Packed tensor hash mismatch: '+name)
    dump(args.out/'train_scalers.json',scaler)
    clips=manifest['clips']
    mapping=np.array([r['xml_mapping_fraction'] for r in clips])
    audit=dict(clips=len(clips),works=len({r['composition_id'] for r in clips}),
        source_performances=len({r['source_sha256'] for r in clips}),
        total_score_note_rows=manifest['total_notes'],windows=len(manifest['windows']),
        max_window_notes=max(w['N'] for w in manifest['windows']),
        split_counts=manifest['split_counts'],
        mapping_fraction_quantiles=dict(zip(('min','p05','median','p95','max'),np.quantile(mapping,[0,.05,.5,.95,1]).tolist())),
        original_alignment_matches=sum(r['original_clip_alignment_matches'] for r in clips),
        xml_mapped_alignment_matches=sum(r['xml_mapped_clip_matches'] for r in clips),
        observed_channel_counts=np.array([r['observed_by_channel'] for r in clips]).sum(0).tolist(),
        boundary_masked_channel_counts=np.array([r['boundary_masked_by_channel'] for r in clips]).sum(0).tolist(),
        no_dropped_clips=True,no_dropped_score_rows=True,
        limitation='All selected clips are included; unmatched notes and unavailable channels are masked, not fabricated.')
    dump(args.out/'input_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    trials=[dict(name=name,hidden_size=hidden,layers=layers,pedal_weight=pedal,seed=2026,learning_rate=.001)
        for name,hidden,layers,pedal in [('compact',32,1,1.),('standard',64,1,1.),
                                        ('deep',64,2,1.),('pedal15',64,1,1.5)]]
    protocol=dict(corpus_id=manifest['id'],trials=trials,initial_epochs=8,finalist_epochs=30,finalists=2,
        patience=7,selection='minimum validation equal-five-group Huber delta1, equal clip weight',
        test_policy='evaluate selected checkpoint once; no settings changed from test results',
        production='refit chosen configuration from scratch on all 5082 for validation-selected epoch count',
        batch_size=args.batch_size,threads=args.threads,max_notes=args.max_notes,
        chunk_policy='contiguous complete onset and beat groups; retain every tail and note row',
        split_counts=manifest['split_counts'],G_loss=False)
    protocol_path=args.out/'experiment_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Frozen experiment protocol differs')
    dump(protocol_path,protocol)
    screening=args.out/'screening_results.json'
    if screening.exists():
        results=json.loads(screening.read_text(encoding='utf-8'))
    else:
        results=[]
        for trial in trials:
            results.append(fit(trial,corpus,scaler,args.out/trial['name'],max_epochs=8,batch_size=args.batch_size))
        dump(screening,results)
    ranked=sorted(results,key=lambda r:r['best_loss'])
    finalists=[]
    for result in ranked[:2]:
        trial=result['trial']
        finalists.append(fit(trial,corpus,scaler,args.out/trial['name'],max_epochs=30,batch_size=args.batch_size))
    selected=min(finalists,key=lambda r:r['best_loss'])
    selection_path=args.out/'selection.json'
    if selection_path.exists():
        old=json.loads(selection_path.read_text(encoding='utf-8'))
        if old['trial']!=selected['trial'] or old['best_epoch']!=selected['best_epoch']:
            raise ValueError('Selection changed after being frozen')
    dump(selection_path,selected)
    evaluated=args.out/'evaluated_best.pt';shutil.copyfile(selected['path'],evaluated)
    model,payload=load_model(evaluated)
    diagnostics_path=args.out/'validation_latent_diagnostics.json'
    if not diagnostics_path.exists():
        diagnostics={mode:evaluate(model,corpus,scaler,corpus.items('validation'),args.batch_size,mode=mode)
            for mode in ('normal','train_channel_mean','zero_z','shuffle_z')}
        diagnostics.update(checkpoint_sha256=file_hash(evaluated),
            purpose='Check whether the decoder uses z; no test-set selection or perceptual-style claim')
        dump(diagnostics_path,diagnostics)
    elif json.loads(diagnostics_path.read_text(encoding='utf-8'))['checkpoint_sha256']!=file_hash(evaluated):
        raise ValueError('Checkpoint changed after the validation diagnostics')
    test_path=args.out/'test_evaluation.json'
    if not test_path.exists():
        test={mode:evaluate(model,corpus,scaler,corpus.items('test'),args.batch_size,mode=mode)
              for mode in ('normal','train_channel_mean','zero_z','shuffle_z')}
        test.update(checkpoint_sha256=file_hash(evaluated),selected_from_validation_only=True)
        dump(test_path,test)
    elif json.loads(test_path.read_text(encoding='utf-8'))['checkpoint_sha256']!=file_hash(evaluated):
        raise ValueError('Checkpoint changed after the frozen test evaluation')
    production_scaler=corpus.fit_scalers(all_data=True)
    dump(args.out/'production_scalers.json',production_scaler)
    schedule=[r['learning_rate'] for r in selected['history'][:selected['best_epoch']]]
    production=fit(selected['trial'],corpus,production_scaler,args.out/'all_5082_refit',
        max_epochs=selected['best_epoch'],batch_size=args.batch_size,all_data=True,lr_schedule=schedule)
    final_payload=torch.load(production['path'],map_location='cpu',weights_only=False)
    compact={k:v for k,v in final_payload.items() if k not in ('optimizer','scheduler','torch_rng')}
    compact['selected_hyperparameters_from']=str(selection_path)
    final=args.out/'final_all_5082.pt';save_checkpoint(final,compact)
    model,reloaded=load_model(final)
    example=corpus.collate(corpus.items()[:2],production_scaler)
    with torch.no_grad():
        if not torch.isfinite(model(example)['R_hat']).all():raise ValueError('Reloaded checkpoint is nonfinite')
    exported=export_embeddings(model,corpus,production_scaler,args.out/'all_5082_embeddings.npz',args.batch_size)
    final_comparison={r['trial']['name']:r for r in results}
    final_comparison.update({r['trial']['name']:r for r in finalists})
    write_curves(args.out,final_comparison,production)
    report=dict(status='complete',selected_trial=selected['trial'],selected_epoch=selected['best_epoch'],
        final_checkpoint=str(final),final_sha256=file_hash(final),evaluated_checkpoint=str(evaluated),
        production_training_clips=len(production_scaler['fit_ids']),evaluation_split_counts=manifest['split_counts'],
        embeddings=exported,comparison=[{k:r[k] for k in ('trial','best_loss','best_epoch','completed_epochs')} for r in final_comparison.values()],
        scope='best within the predeclared tested settings; no global-optimum or perceptual-style claim',
        production_has_independent_test=False)
    dump(args.out/'final_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
