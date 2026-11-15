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
    args=p.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.out.resolve()==args.source_run.resolve():raise ValueError('Separate production output required')
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    source=args.source_run;out=args.out;out.mkdir(parents=True,exist_ok=True)
    assert read(source/'verification.json')['status']=='passed'
    assert read(source/'weak_reconstruction_validation/verification.json')['status']=='passed'
    contract=read(source/'profile_contract.json')
    if contract['version']!=G_PROFILE_VERSION:raise ValueError('Invalidated G supervision')
    for name in ['profiles.npz','profile_contract.json','metadata.json','input_audit.json',
                 'coordinates/tempo_cells.npy','coordinates/tempo_coordinates.json']:
        copy_verified(source/name,out/name)
    corpus=SelectedNoteCorpus(args.common_run/'corpus');metadata=read(out/'metadata.json')
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
    protocol=dict(version='protected-g-all5082-v1',supervision_version=G_PROFILE_VERSION,
        objective_mode='g_only',corpus_id=corpus.manifest['id'],batch_clips=16,
        epochs_by_era={e:ablation['eras'][e]['epochs'] for e in ERA_COMPOSERS},configs=configurations,
        loss='effective_lambda*G + .05*z_anchor + .01*head_shift; no direct reconstruction',
        G_cap='25% of unweighted reconstruction gradient at window z; warmup3epochs, nominal lambda .1',
        source_run=str(source),validation_decision_sha256=file_hash(source/'weak_reconstruction_validation/report.json'),
        initialization='final_all_5082.pt copied independently per era; identity initialized bounded residual head',
        pretraining_hashes={n:file_hash(args.common_run/n) for n in ['evaluated_best.pt','final_all_5082.pt']},
        cache_contract=read(cache/'ready.json'),
        coverage='All clips enter encoder/anchor and monitoring; only valid comparable clips enter G; no fake G labels',
        selection='Minimum full-fit G score among common-preservation guard-passing epochs; common baseline fallback allowed',
        evaluation='All-data fit diagnostics, no independent test; existing train-only test evidence remains separate')
    if (out/'protocol.json').exists() and read(out/'protocol.json')!=protocol:raise ValueError('Production protocol changed')
    dump(out/'protocol.json',protocol)
    with np.load(out/'profiles.npz') as f:raw,known,quality=[f[k] for k in ('raw','known','quality')]
    runtime=Runtime(corpus,args.common_run/'final_all_5082.pt',cache)
    reports={};export={k:[] for k in ['indices','z_ae','z_style','e','era']}
    for era in ERA_COMPOSERS:
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
            objective_mode='g_only',max_epochs=protocol['epochs_by_era'][era],production_epoch=state['epoch'],
            checkpoint=str(final),sha256=file_hash(final),embeddings=str(era_path),embeddings_sha256=file_hash(era_path),
            production_training_diagnostics=metrics,baseline=result['baseline'],guard=guard,independent_test=False)
        dump(out/'production_progress.json',reports)
        print('FINAL ERA SAVED',era,'epoch',state['epoch'],'clips',len(e),flush=True)
    order=np.argsort(export['indices'])
    if sorted(export['indices'])!=list(range(5082)):raise ValueError('Missing or duplicate embedding')
    combined=out/'all_5082_era_embeddings.npz'
    np.savez_compressed(combined,**{k:np.asarray(v)[order] for k,v in export.items() if k!='indices'},
        perf_ids=np.array([m['perf_id'] for m in metadata]),composition_ids=np.array([m['work'] for m in metadata]))
    dump(out/'final_report.json',dict(status='complete',objective_mode='g_only',clips=5082,eras=reports,
        embeddings=str(combined),embeddings_sha256=file_hash(combined),independent_test=False,
        source_evaluation=str(source/'ablation/report.json'),
        comparison_policy='Use cosine distance of e only within the same era; spaces are not cross-calibrated',
        limitation='All-data fit diagnostics; reference-ineligible clips receive anchor only, not direct G or direct reconstruction'))
    print('ALL5082 PROTECTED G EMBEDDINGS SAVED',flush=True)


if __name__=='__main__':main()
