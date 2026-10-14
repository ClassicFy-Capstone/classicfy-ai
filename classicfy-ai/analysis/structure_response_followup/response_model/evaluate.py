"""Frozen retrieval evaluation of learned individual response embeddings."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from data import OUT,PREVIOUS,ViewData,sha,KINDS
from model import ResponseEncoder,masks
from train import validate,MODES
sys.path.insert(0,str(PREVIOUS))
from compare import channel_scale,set_distances,calibrated,evaluate,macro,weights

AXES=('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL')


def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load(view,mode,initial=False):
    source=OUT/f'view{view}_{mode}'/('initial.pt' if initial else 'best.pt')
    checkpoint=torch.load(source,weights_only=True)
    assert checkpoint['data_sha256']==sha(OUT/'examples.npz')
    model=ResponseEncoder(checkpoint['mode']=='context_only');model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint


def encode(model,data,view,permutation_seed=None):
    values=data.x.clone()
    if permutation_seed is not None:
        for i in data.ids:
            for kind in range(5):
                rows=np.flatnonzero((data.arrays['owner']==i)&(data.arrays['type']==kind))
                rng=np.random.default_rng(np.random.SeedSequence([permutation_seed,view,kind,int(i)]))
                permutation=rng.permutation(17)
                values[rows]=values[rows][:,:,permutation]
    vectors=[]
    with torch.no_grad():
        for start in range(0,len(values),256):vectors.append(model.encode(values[start:start+256],data.context[start:start+256]).numpy())
    vectors=np.concatenate(vectors)
    assert np.isfinite(vectors).all()
    sets={kind:{i:vectors[(data.arrays['owner']==i)&(data.arrays['type']==k),:,None] for i in data.ids} for k,kind in enumerate(KINDS)}
    return vectors,sets


def retrieval(sets,data,base,direction,fixed=None):
    gallery=[data.metadata[i] for i in data.gids];gw=weights(gallery);families=np.array([r['family_id'] for r in gallery])
    blocks={};scales={}
    for kind in KINDS:
        if fixed is None:
            scale=channel_scale(sets[kind],data.gids,gw)
            d,median=calibrated(set_distances(sets[kind],data.ids,data.gids,scale),data.gids,gw,families)
            scales[kind]=dict(sd=scale[0].tolist(),active=scale[1].tolist(),median=median)
        else:
            record=fixed[kind];scale=(np.array(record['sd']),np.array(record['active']))
            d=set_distances(sets[kind],data.ids,data.gids,scale)/record['median']
        blocks[kind]=d
    h=np.sqrt(np.mean([blocks[k]**2 for k in ('H0','H1','H2')],axis=0))
    if fixed is None:
        h,hm=calibrated(h,data.gids,gw,families);scales['H']=dict(median=hm)
    else:h/=fixed['H']['median']
    p=blocks['P'];b=base[f'{direction}/distance/B_summary']
    result=dict(F=blocks['F'],P=p,H=h,**{'P+H':np.sqrt((p*p+h*h)/2),'B+P+H':np.sqrt((b*b+p*p+h*h)/3)})
    return result,scales


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    reference=np.load(PREVIOUS/'comparison.npz');manifest=json.loads((OUT/'data_manifest.json').read_text())
    assert sha(PREVIOUS/'comparison.npz')==manifest['source_comparison_sha256']
    assert sha(OUT/'examples.npz')==manifest['examples_sha256']
    records,rankings,controls,head_metrics,model_manifest=[],[],[],[],[]
    stored={};all_scales={}
    for view,direction in enumerate(('A_to_B','B_to_A')):
        data=ViewData(view);assert np.array_equal(data.gids,reference['gallery_indices']) and np.array_equal(data.qids,reference['query_indices'])
        evaluations={axis:reference[f'{direction}/evaluation/{axis}'] for axis in AXES}
        for mode in (*MODES,'untrained'):
            model,checkpoint=load(view,'response_aux' if mode=='untrained' else mode,mode=='untrained')
            assert checkpoint['scales']==data.scales
            assert checkpoint['train_owner_indices']==data.gids.tolist()
            assert not set(checkpoint['train_owner_indices'])&set(data.qids.tolist())
            vectors,sets=encode(model,data,view)
            directory=OUT/f'view{view}_{mode}';directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/'embeddings.npz',vectors=vectors,owners=data.arrays['owner'],types=data.arrays['type'])
            distances,scales=retrieval(sets,data,reference,direction)
            all_scales[direction+'/'+mode]=scales
            model_manifest.append(dict(view=view,mode=mode,epoch=checkpoint['epoch'],parameters=checkpoint['parameters'],
                minimum_train_embedding_sd=float(min(min(scales[k]['sd']) for k in KINDS)),
                train_owner_indices=checkpoint['train_owner_indices'],validation_owner_indices=checkpoint['validation_owner_indices']))
            for kind,d in distances.items():
                method=mode+'_'+kind
                r,rr=evaluate(method,d,evaluations,data.metadata,data.gids,data.qids)
                records.extend(dict(direction=direction,**x) for x in r);rankings.extend(dict(direction=direction,**x) for x in rr)
                stored[direction+'/'+method]=d
            # Test reconstruction/response heads are inspected only now, after checkpoint selection.
            test=data.roles['test'];w=data.role_weights['test'];g=torch.Generator().manual_seed(20261114+view)
            result=validate(model,data,test,w,masks(len(test),g),masks(len(test),g),checkpoint['mode'])
            head_metrics.append(dict(view=view,mode=mode,**result))
            for seed in range(20261014,20261034):
                allowed=reference[f'{direction}_balanced_{seed}'] if f'{direction}_balanced_{seed}' in reference.files else None
                if allowed is None:
                    from collections import defaultdict
                    groups=defaultdict(list)
                    for j,i in enumerate(data.gids):
                        row=data.metadata[i];groups[row['piece'],row['grid']].append(j)
                    rng=np.random.default_rng(seed)
                    allowed=np.array(sorted(j for group in groups.values() if len(group)>=3 for j in rng.choice(group,3,replace=False)))
                for kind,d in distances.items():
                    method=mode+'_'+kind;r,_=evaluate(method,d,evaluations,data.metadata,data.gids,data.qids,seed,allowed)
                    for role in ('test','validation'):
                        rs=[x for x in r if x['role']==role]
                        controls.append(dict(direction=direction,control='three_candidates',seed=seed,mode=mode,kind=kind,role=role,
                            **{axis:macro(rs,axis) for axis in AXES}))
            if mode!='context_only':
                for seed in range(20261014,20261019):
                    _,shuffled=encode(model,data,view,seed)
                    dd,_=retrieval(shuffled,data,reference,direction,scales)
                    for kind,d in dd.items():
                        r,_=evaluate(mode+'_'+kind,d,evaluations,data.metadata,data.gids,data.qids)
                        for role in ('test','validation'):
                            rs=[x for x in r if x['role']==role]
                            controls.append(dict(direction=direction,control='performance_order_permuted',seed=seed,mode=mode,kind=kind,role=role,
                                **{axis:macro(rs,axis) for axis in AXES}))
            print('Evaluated',direction,mode,'epoch',checkpoint['epoch'],flush=True)
    summaries=[]
    for role in ('test','validation'):
        for method in sorted(set(r['method'] for r in records)):
            rs=[r for r in records if r['role']==role and r['method']==method]
            summaries.append(dict(role=role,method=method,**{a:macro(rs,a) for a in AXES},**{a+'_same_work':macro(rs,a+'_same_work') for a in AXES}))
    cs=[]
    groups=sorted(set((r['role'],r['mode'],r['kind'],r['control']) for r in controls))
    for role,mode,kind,control in groups:
        rs=[r for r in controls if (r['role'],r['mode'],r['kind'],r['control'])==(role,mode,kind,control)]
        cs.append(dict(role=role,mode=mode,kind=kind,control=control,**{a:float(np.mean([r[a] for r in rs])) for a in AXES}))
    write('scores.csv',records);write('rankings.csv',rankings);write('summary.csv',summaries)
    write('controls.csv',controls);write('control_summary.csv',cs);write('test_head_metrics.csv',head_metrics)
    np.savez_compressed(OUT/'distances.npz',**stored)
    (OUT/'retrieval_scales.json').write_text(json.dumps(all_scales,indent=2)+'\n')
    (OUT/'model_manifest.json').write_text(json.dumps(model_manifest,indent=2)+'\n')
    for r in summaries:
        if r['method'].startswith(('response_aux_','reconstruction_only_','context_only_')):
            print(r['role'],r['method'],{a:round(r[a],2) for a in AXES},flush=True)


if __name__=='__main__':main()
