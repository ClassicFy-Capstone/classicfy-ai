"""Reconstruct verified sequences and freeze retrieval inputs before ranking."""

import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'classicfy-ai/src'))
from embedding.retrieval_probe import CHANNELS, hand_descriptors, summary, partition, judges
from audit_asap_metadata import read_csv, write_csv, sha256


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def reconstruct(data, legacy):
    archive = subprocess.check_output(['git','archive',legacy['learned_source_commit'],
        'classicfy-ai/src','classicfy-ai/scripts'], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='classicfy-retrieval-source-') as directory:
        snapshot = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(snapshot, filter='data')
        names = subprocess.check_output(['git','ls-tree','-r','--name-only','20a4168',
            'classicfy-ai/src/features','classicfy-ai/src/preprocessing'],cwd=ROOT,text=True).splitlines()
        original = [n for n in names if n.endswith('.py')]
        for n in original:
            if (snapshot/n).read_bytes() != subprocess.check_output(['git','show',f'20a4168:{n}'],cwd=ROOT):
                raise ValueError('Original collector code differs')
        files = [snapshot/n for n in original if '/preprocessing/' in n or
            ('/features/' in n and Path(n).name not in ('normalization.py','__init__.py'))]
        sys.path[:0] = [str(snapshot/'classicfy-ai/scripts'), str(snapshot/'classicfy-ai/src')]
        spec=importlib.util.spec_from_file_location('retrieval_original_collector',snapshot/'classicfy-ai/scripts/validate_embedding.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        reference=json.loads((data/'temporal_embedding_cluster_run01/dataset.json').read_text())
        groups,audits,scales,provenance=module.collect_groups(SimpleNamespace(asap_root=(data/'ASAP').resolve(),
            nasap_root=(data/'nASAP').resolve(),cache=data/'feature_normalization_cluster_raw.npz',**reference['cohort_settings']),
            provenance_source_files=files)
        if scales!=reference['scales'] or provenance!=legacy['source_provenance']:
            raise ValueError('Original normalization provenance differs')
        lengths={(r['piece'],r['grid']):r['total_intervals'] for r in audits if r['accepted']}
        sequences={}
        for group in groups:
            index=np.asarray(group['shared_indices']);length=lengths[group['piece'],group['grid']]
            for key,values in zip(group['keys'],group['values'],strict=True):
                x=np.zeros((7,length),float);m=np.zeros(length,bool)
                x[:,index]=values;m[index]=True;sequences[key]=(x,m)
        yield sequences, snapshot


def encode_views(snapshot, data, entries, windows, rows):
    import torch
    import embedding
    embedding.__path__.insert(0,str(snapshot/'classicfy-ai/src/embedding'))
    from embedding.extended_experiments import load
    torch.set_num_threads(2)
    results={}
    for entry in entries:
        path=data/entry['checkpoint']
        if sha256(path)!=entry['checkpoint_sha256']:
            raise ValueError('Saved checkpoint differs')
        model,_=load(path)
        for view in ('A','B'):
            parts=[];owners=[]
            for i,r in enumerate(rows):
                for x,m in windows[view][r['key']]:
                    parts.append((np.where(m[None,:],x,0).astype(np.float32),np.broadcast_to(m,(7,64)).copy()))
                    owners.append(i)
            vectors=np.full((len(rows),128),np.nan)
            if parts:
                x=torch.from_numpy(np.array([p[0] for p in parts]));m=torch.from_numpy(np.array([p[1] for p in parts]))
                latent=[]
                with torch.no_grad():
                    for start in range(0,len(parts),64):
                        latent.append(model.encode(x[start:start+64],m[start:start+64]).numpy())
                latent=np.concatenate(latent);owner=np.asarray(owners)
                for i in np.unique(owner):
                    z=latent[owner==i];vectors[i]=np.r_[z.mean(axis=0),z.std(axis=0)]
            name=f"{view}_{entry['arm']}_{entry['seed']}"
            results[name]=vectors.astype(float)
            print('Encoded',name,'windows',len(parts),'finite performances',int(np.isfinite(vectors).all(axis=1).sum()),flush=True)
    return results


def support(rows, arrays, view=None):
    count=Counter(); ids=[]
    for i,r in enumerate(rows):
        matrices=[v for n,v in arrays.items() if n.startswith(view+'_')] if view else list(arrays.values())
        good=(r['evaluation_role'] in ('train','validation','test') and r['work_type']!='unclassified'
              and all(np.isfinite(v[i]).all() for v in matrices))
        if good:
            ids.append(i)
            if r['evaluation_role']=='train':count[r['piece'],r['grid']]+=1
    return [i for i in ids if rows[i]['evaluation_role']!='train' or count[rows[i]['piece'],rows[i]['grid']]>=3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent/'datasets')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/metadata_retrieval_results')
    parser.add_argument('--cache',type=Path,default=ROOT.parent/'datasets/metadata_retrieval_run01')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.cache.mkdir(parents=True,exist_ok=True)
    design=ROOT/'classicfy-ai/analysis/metadata_retrieval_design'
    protocol=json.loads((design/'protocol.json').read_text())
    provenance=json.loads((design/'design_provenance.json').read_text())
    for name,digest in provenance['sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Design input changed: '+name)
    legacy=json.loads((ROOT/'classicfy-ai/analysis/metadata_information_design/protocol.json').read_text())
    if sha256(args.data/'feature_normalization_cluster_raw.npz')!=json.loads((args.data/'metadata_temporal_exports_run01/manifest.json').read_text())['cache_sha256']:
        raise ValueError('Raw source changed')
    rows=read_csv(ROOT/protocol['source_files']['role_manifest'])
    with np.load(ROOT/protocol['source_files']['existing_inputs'],allow_pickle=False) as z:
        if z['keys'].tolist()!=[r['key'] for r in rows]:raise ValueError('Input order differs')
        whole={name:z[name].copy() for name in z.files if name!='keys'}
    threshold_raw=subprocess.check_output(['git','show',f"{legacy['hand_source_commit']}:classicfy-ai/analysis/temporal_shape/protocol.json"],cwd=ROOT)
    thresholds={r['channel']:r['value'] for r in json.loads(threshold_raw)['thresholds']}
    signature={'protocol':sha256(design/'protocol.json'),'sources':provenance['sha256'],
        'code':sha256(__file__),'core':sha256(ROOT/'classicfy-ai/src/embedding/retrieval_probe.py'),
        'threshold':hashlib.sha256(threshold_raw).hexdigest()}
    metadata_path=args.cache/'prepared.json';archive_path=args.cache/'prepared.npz'
    if metadata_path.exists() and archive_path.exists() and json.loads(metadata_path.read_text())['signature']==signature:
        meta=json.loads(metadata_path.read_text());arrays=dict(np.load(archive_path,allow_pickle=False));print('Reusing verified preparation cache',flush=True)
    else:
        arrays={f'whole_{name}':v for name,v in whole.items()};partitions=[];cohorts=[];windows={'A':{},'B':{}}
        single_keep=legacy['retained_descriptor_coordinates']['single']
        check={'summary_max_error':0.,'single_max_error':0.,'relation_max_error':0.,'finite_masks_equal':True}
        entries=legacy['selected_model_exports']
        for sequences,snapshot in reconstruct(args.data,legacy):
            if set(sequences)!={r['key'] for r in rows}:raise ValueError('Sequence identities differ')
            collected={name:[] for name in ('A_B','A_T','A_R','B_B','B_T','B_R','A_E-L','A_E-T','A_E-R','B_E-L','B_E-T','B_E-R')}
            for i,r in enumerate(rows):
                x,m=sequences[r['key']]
                t,rel=hand_descriptors(x,m,[thresholds[c] for c in CHANNELS],single_keep)
                for name,new in [('summary',summary(x,m)),('single',t),('relation',rel)]:
                    old=whole[{'summary':'B','single':'T','relation':'R'}[name]][i]
                    if not np.array_equal(np.isfinite(new),np.isfinite(old)):raise ValueError('Full descriptor finite mask differs: '+r['key']+' '+name)
                    error=float(np.max(abs(new[np.isfinite(new)]-old[np.isfinite(old)])))
                    check[name+'_max_error']=max(check[name+'_max_error'],error)
                    if error>1e-10:raise ValueError('Full descriptor differs: '+name)
                for parity,view in enumerate(('A','B')):
                    mask,blocks=partition(x,m,parity);windows[view][r['key']]=[(x[:,a:a+64],m[a:a+64]) for a in blocks]
                    partitions.append(dict(key=r['key'],view=view,total_beats=len(m),full_blocks=len(m)//64,
                        valid_blocks=len(blocks),valid_beats=int(mask.sum()),starts='|'.join(map(str,blocks))))
                    b=summary(x,mask);t,rel=hand_descriptors(x,mask,[thresholds[c] for c in CHANNELS],single_keep)
                    judge=judges(x,mask,blocks)
                    if len(blocks)<2:
                        b[:]=np.nan;t[:]=np.nan;rel[:]=np.nan
                    for name,value in [('B',b),('T',t),('R',rel),*judge.items()]:collected[f'{view}_{name}'].append(value)
                if (i+1)%100==0:print('Measured original-index views',i+1,'/',len(rows),flush=True)
            arrays.update({name:np.asarray(v) for name,v in collected.items()})
            arrays.update(encode_views(snapshot,args.data,entries,windows,rows))
        for experiment in ('whole','split'):
            selected={n:v for n,v in arrays.items() if n.startswith('whole_')} if experiment=='whole' else {n:v for n,v in arrays.items() if n.startswith(('A_','B_'))}
            ids=support(rows,selected)
            for i,r in enumerate(rows):
                bad=[n for n,v in selected.items() if not np.isfinite(v[i]).all()]
                included=i in ids
                cohorts.append({k:r[k] for k in ('key','piece','grid','family_id','composer','work_type','evaluation_role')}|
                    dict(experiment=experiment,included=included,reasons='|'.join(bad) if bad else
                    ('unknown_type' if r['work_type']=='unclassified' else 'excluded_split' if r['evaluation_role']=='excluded' else 'fewer_than_three_gallery_recordings' if not included else '')))
        meta={'signature':signature,'partitions':partitions,'cohort':cohorts,'source_recalculation':check}
        np.savez_compressed(archive_path,**arrays);dump(metadata_path,meta)
    write_csv(args.out/'cohort.csv',meta['cohort']);write_csv(args.out/'partition_manifest.csv',meta['partitions'])
    availability=[]
    for experiment in ('whole','split'):
        chosen=[r for r in meta['cohort'] if r['experiment']==experiment and r['included']]
        gallery=[r for r in chosen if r['evaluation_role']=='train']
        for role in ('train','validation','test'):
            subset=[r for r in chosen if r['evaluation_role']==role]
            availability.append(dict(experiment=experiment,role=role,recordings=len(subset),works=len({r['piece'] for r in subset}),
                families=len({r['family_id'] for r in subset}),gallery_recordings=len(gallery),gallery_families=len({r['family_id'] for r in gallery})))
    write_csv(args.out/'availability.csv',availability)
    dump(args.out/'input_hashes.json',signature)
    dump(args.out/'preflight.json',{'status':'cohort_and_partitions_frozen_before_ranking','availability':availability,
        'source_recalculation':meta['source_recalculation'],'prepared_archive_sha256':sha256(archive_path),
        'prepared_metadata_sha256':sha256(metadata_path),'parameters_changed':False,'scales':'Will be frozen by runner before any ranking'})
    for r in availability:print(r,flush=True)


if __name__=='__main__':main()
