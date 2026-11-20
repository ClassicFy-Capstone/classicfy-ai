"""All-selected-clip era refits with G and preservation terms, no direct Rec.

Reuses only verified v2 input profiles and immutable common-AE context. The
train-only evaluation heads are evidence, never production initialization.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from train_era_note_ae import (ERA_COMPOSERS,G_PROFILE_VERSION,SelectedNoteCorpus,Runtime,
    tables_for,fit,restore,evaluate,guards,dump,file_hash)


def copy_verified(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if file_hash(source)!=file_hash(target):raise ValueError('Prepared artifact mismatch: '+str(target))
    else:shutil.copyfile(source,target)
    if file_hash(source)!=file_hash(target):raise ValueError('Copy verification failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--common-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--budget-selection',type=Path,
                   help='Root containing validate_era_g_budget.py results for all eras')
    p.add_argument('--era',choices=ERA_COMPOSERS,
                   help='Produce one era independently; use --assemble-only to combine')
    p.add_argument('--assemble-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.out.resolve().is_relative_to(args.source_run.resolve()):raise ValueError('Separate production output required')
    if args.era and args.assemble_only:
        p.error('--era cannot be combined with assembly')
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    source=args.source_run;out=args.out;out.mkdir(parents=True,exist_ok=True)
    assert read(source/'verification.json')['status']=='passed'
    assert read(source/'weak_reconstruction_validation/verification.json')['status']=='passed'
    contract=read(source/'profile_contract.json')
    if contract['version']!=G_PROFILE_VERSION:raise ValueError('Invalidated G supervision')
    if not args.era:
        for name in ['profiles.npz','profile_contract.json','metadata.json','input_audit.json',
                     'coordinates/tempo_cells.npy','coordinates/tempo_coordinates.json']:
            copy_verified(source/name,out/name)
    corpus=SelectedNoteCorpus(args.common_run/'corpus');metadata=read(source/'metadata.json')
    assert len(metadata)==5082 and contract['corpus_id']==corpus.manifest['id']
    decisions=read(source/'weak_reconstruction_validation/report.json')
    ablation=read(source/'ablation/protocol.json')
    if decisions['status']!='complete':raise ValueError('Unfinished validation search')
    assert all(d['recommended_alpha']==0 for d in decisions['eras'].values())
    cache=source/'production_cache'
    reuse=read(source/'common_cache_reuse.json')['caches']['production_cache']
    for name,expected in reuse['hashes'].items():
        assert file_hash(cache/name)==expected,'Changed common cache: '+name
    configurations={e:{**ablation['eras'][e]['config'],'objective_mode':'g_only','reconstruction_weight':0.}
                    for e in ERA_COMPOSERS}
    budgets={}
    epochs={e:ablation['eras'][e]['epochs'] for e in ERA_COMPOSERS}
    hashes={n:file_hash(args.common_run/n) for n in ('evaluated_best.pt','final_all_5082.pt')}
    if args.budget_selection:
        for era,config in configurations.items():
            if args.era and era!=args.era:
                continue
            path=args.budget_selection/era/'selection.json'
            decision=read(path)
            trial=decision['protocol']
            budget=decision['selected_budget']
            if (decision['status']!='complete' or budget not in (.5,1.)
                or trial['era']!=era or trial['horizon']!=epochs[era] or trial['batch_clips']!=16
                or trial['base_config']!={k:v for k,v in config.items() if k!='reconstruction_weight'}
                or trial['initialization_sha256']!=hashes['evaluated_best.pt']
                or trial['profiles_sha256']!=file_hash(source/'profiles.npz')
                or trial['metadata_sha256']!=file_hash(source/'metadata.json')
                or not decision['trials'][str(budget)]['guard']['passed']):
                raise ValueError('Invalid relaxed budget selection: '+era)
            config['gradient_budget']=budget
            budgets[era]=file_hash(path)
    protocol=dict(version='protected-g-all5082-v1',supervision_version=G_PROFILE_VERSION,
        objective_mode='g_only',corpus_id=corpus.manifest['id'],batch_clips=16,
        epochs_by_era=epochs,configs=configurations,
        loss='effective_lambda*G + .05*z_anchor + .01*head_shift; no direct reconstruction',
        G_cap='25% of unweighted reconstruction gradient at window z; warmup3epochs, nominal lambda .1',
        source_run=str(source),validation_decision_sha256=file_hash(source/'weak_reconstruction_validation/report.json'),
        initialization='final_all_5082.pt copied independently per era; identity initialized bounded residual head',
        pretraining_hashes=hashes,
        cache_contract=read(cache/'ready.json'),
        coverage='All clips enter encoder/anchor and monitoring; only valid comparable clips enter G; no fake G labels',
        selection='Minimum full-fit G score among common-preservation guard-passing epochs; common baseline fallback allowed',
        evaluation='All-data fit diagnostics, no independent test; existing train-only test evidence remains separate')
    if budgets:
        protocol.update(version='relaxed-g-all5082-v1',budget_selection_sha256=budgets,
                        G_cap='Per-era configured fraction of unweighted reconstruction gradient at window z; warmup3epochs, nominal lambda .1')
    if not args.era:
        if (out/'protocol.json').exists() and read(out/'protocol.json')!=protocol:raise ValueError('Production protocol changed')
        dump(out/'protocol.json',protocol)
    with np.load(source/'profiles.npz') as f:raw,known,quality=[f[k] for k in ('raw','known','quality')]
    runtime=None if args.assemble_only else Runtime(corpus,args.common_run/'final_all_5082.pt',cache)
    reports={};export={k:[] for k in ['indices','z_ae','z_style','e','era']}
    for era in ERA_COMPOSERS:
        if args.era and era!=args.era:
            continue
        report_path=out/era/'production_report.json'
        if args.assemble_only:
            reports[era]=read(report_path)
            info=reports[era]
            if (info['config']!=configurations[era] or file_hash(info['checkpoint'])!=info['sha256']
                or file_hash(info['embeddings'])!=info['embeddings_sha256']):
                raise ValueError('Per-era result does not match the assembly protocol: '+era)
            indices=[i for i,m in enumerate(metadata) if m['era']==era]
            with np.load(info['embeddings'],allow_pickle=False) as data:
                if data['perf_ids'].tolist()!=[metadata[i]['perf_id'] for i in indices]:
                    raise ValueError('Per-era embeddings out of order: '+era)
                export['indices'].extend(indices)
                for field in ('z_ae','z_style','e','era'):
                    export[field].extend(data[field])
            continue
        ids,scaler,tables,calibration=tables_for(metadata,raw,known,quality,era,True)
        scale=torch.tensor(np.maximum(runtime.teacher_clip[ids['train']].std(0),.05))
        dump(out/era/'g_production_scaler.json',dict(scaler=scaler,calibration=calibration))
        result=fit(runtime,era,ids['train'],[],tables,calibration,scale,configurations[era],out/era/'production',16,
                   protocol['epochs_by_era'][era],True)
        final=out/era/'final.pt';shutil.copyfile(result['path'],final)
        model,head,scale,state=restore(final,runtime)
        metrics,z,e=evaluate(model,head,runtime,ids['train'],scale,tables['train'],calibration,16)
        guard=guards(metrics,result['baseline'])
        if not guard['passed']:raise ValueError('Final model failed preservation: '+era)
        with torch.no_grad():zs=head(torch.from_numpy(z))['z_style'].numpy()
        era_path=out/era/'embeddings.npz'
        np.savez_compressed(era_path,z_ae=z,z_style=zs,e=e,
            perf_ids=np.array([metadata[i]['perf_id'] for i in ids['train']]),
            composition_ids=np.array([metadata[i]['work'] for i in ids['train']]),era=np.array([era]*len(e)))
        export['indices'].extend(ids['train']);export['z_ae'].extend(z);export['z_style'].extend(zs)
        export['e'].extend(e);export['era'].extend([era]*len(e))
        reports[era]=dict(clips=len(e),works=len({metadata[i]['work'] for i in ids['train']}),
            config=configurations[era],
            objective_mode='g_only',max_epochs=protocol['epochs_by_era'][era],production_epoch=state['epoch'],
            checkpoint=str(final),sha256=file_hash(final),embeddings=str(era_path),embeddings_sha256=file_hash(era_path),
            production_training_diagnostics=metrics,baseline=result['baseline'],guard=guard,independent_test=False)
        dump(report_path,reports[era])
        if not args.era:
            dump(out/'production_progress.json',reports)
        print('FINAL ERA SAVED',era,'epoch',state['epoch'],'clips',len(e),flush=True)
    if args.era:
        return
    order=np.argsort(export['indices'])
    if sorted(export['indices'])!=list(range(5082)):raise ValueError('Missing or duplicate embedding')
    combined=out/'all_5082_era_embeddings.npz'
    np.savez_compressed(combined,**{k:np.asarray(v)[order] for k,v in export.items() if k!='indices'},
        perf_ids=np.array([m['perf_id'] for m in metadata]),composition_ids=np.array([m['work'] for m in metadata]))
    dump(out/'final_report.json',dict(status='complete',objective_mode='g_only',clips=5082,eras=reports,
        embeddings=str(combined),embeddings_sha256=file_hash(combined),independent_test=False,
        source_evaluation=str(args.budget_selection if budgets else source/'ablation/report.json'),
        comparison_policy='Use cosine distance of e only within the same era; spaces are not cross-calibrated',
        limitation='All-data fit diagnostics; reference-ineligible clips receive anchor only, not direct G or direct reconstruction'))
    print('ALL5082 PROTECTED G EMBEDDINGS SAVED',flush=True)


if __name__=='__main__':main()
