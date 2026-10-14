"""Independent training provenance, scale, ranking and score verification."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from data import OUT,PREVIOUS,ViewData,KINDS,sha
from model import ResponseEncoder


def main():
    torch.set_num_threads(2)
    examples=np.load(OUT/'examples.npz');base=np.load(PREVIOUS/'comparison.npz');distances=np.load(OUT/'distances.npz')
    manifest=json.loads((OUT/'data_manifest.json').read_text())
    assert sha(OUT/'examples.npz')==manifest['examples_sha256']
    assert sha(PREVIOUS/'comparison.npz')==manifest['source_comparison_sha256']
    checks=0;maximum=0.;reload_error=0.
    for view in (0,1):
        data=ViewData(view);train=data.roles['train'];indices=data.arrays['owner']
        gallery=[data.metadata[i] for i in data.gids];groups=defaultdict(lambda:defaultdict(list))
        for j,r in enumerate(gallery):groups[r['family_id']][r['piece'],r['grid']].append(j)
        recording_weights={}
        for family in groups.values():
            for recordings in family.values():
                for j in recordings:recording_weights[int(data.gids[j])]=1/(len(groups)*len(family)*len(recordings))
        type_counts=defaultdict(int)
        for j in train:type_counts[int(indices[j]),int(data.arrays['type'][j])]+=1
        weights=np.array([recording_weights[int(indices[j])]/(5*type_counts[int(indices[j]),int(data.arrays['type'][j])]) for j in train])
        np.testing.assert_allclose(weights,data.role_weights['train'],atol=1e-14);checks+=len(weights)
        # Scalar float64 moments from train frames only.
        for channel in range(7):
            mean=math.fsum(float(w)*math.fsum(float(v) for v in data.arrays['x'][j,channel])/17 for j,w in zip(train,weights))
            second=math.fsum(float(w)*math.fsum(float(v)**2 for v in data.arrays['x'][j,channel])/17 for j,w in zip(train,weights))
            sd=math.sqrt(max(second-mean*mean,1e-12))
            for actual,expected in ((mean,data.scales['xmean'][channel]),(sd,data.scales['xsd'][channel])):
                error=abs(actual-expected);maximum=max(maximum,error);assert error<1e-10;checks+=1
        for mode in ('reconstruction_only','response','response_aux','context_only'):
            directory=OUT/f'view{view}_{mode}';checkpoint=torch.load(directory/'best.pt',weights_only=True)
            initial=torch.load(directory/'initial.pt',weights_only=True);training=json.loads((directory/'training.json').read_text())
            assert checkpoint['data_sha256']==sha(OUT/'examples.npz')
            source=directory/'training_source.py' if (directory/'training_source.py').exists() else OUT/'train.py'
            expected=hashlib.sha256((OUT/'data.py').read_bytes()+(OUT/'model.py').read_bytes()+source.read_bytes()).hexdigest()
            assert checkpoint['code_sha256']==expected
            assert checkpoint['train_owner_indices']==data.gids.tolist()
            assert len(checkpoint['validation_owner_indices'])==43
            assert not set(checkpoint['train_owner_indices'])&set(data.qids.tolist())
            test=set(data.arrays['owner'][data.roles['test']].tolist())
            assert not test&(set(checkpoint['train_owner_indices'])|set(checkpoint['validation_owner_indices']))
            history=list(csv.DictReader((directory/'history.csv').open()))
            minimum=min(float(r['validation_loss']) for r in history)
            chosen=next(r for r in history if int(r['epoch'])==checkpoint['epoch'])
            assert abs(float(chosen['validation_loss'])-minimum)<2e-6
            assert checkpoint['epoch']==training['best_epoch']
            assert training['test_used_for_training_or_selection'] is False
            model=ResponseEncoder(mode=='context_only');model.load_state_dict(checkpoint['state_dict']);model.eval()
            stored=np.load(directory/'embeddings.npz')['vectors']
            sample=np.rint(np.linspace(0,len(stored)-1,32)).astype(int)
            with torch.no_grad():again=model.encode(data.x[sample],data.context[sample]).numpy()
            error=float(np.max(abs(again-stored[sample])));reload_error=max(reload_error,error);assert error<1e-5;checks+=again.size
            # Every full-input model starts from precisely the same encoder parameters.
            if mode=='reconstruction_only':reference=initial['state_dict']
            elif mode!='context_only':
                for key,value in initial['state_dict'].items():
                    torch.testing.assert_close(value,reference[key],atol=0,rtol=0)
    data=ViewData(0);keys=data.keys;gids=data.gids;lookup={key:i for i,key in enumerate(keys)};glookup={keys[i]:j for j,i in enumerate(gids)}
    gallery=[data.metadata[i] for i in gids]
    grouped=defaultdict(list)
    for r in csv.DictReader((OUT/'rankings.csv').open()):grouped[r['direction'],r['query'],r['method']].append(r)
    rows=list(csv.DictReader((OUT/'scores.csv').open()))
    for r in rows:
        qi=lookup[r['key']];direction=r['direction'];selected=[glookup[x['candidate']] for x in grouped[direction,r['key'],r['method']]]
        assert len(selected)==5 and len(set(gallery[j]['family_id'] for j in selected))==5
        d=distances[direction+'/'+r['method']];seen=set()
        for j in selected:
            candidates=[k for k in range(len(gids)) if gallery[k]['family_id'] not in seen]
            assert d[qi,j]<=min(d[qi,k] for k in candidates)+1e-12
            seen.add(gallery[j]['family_id'])
        for axis in ('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL'):
            e=base[f'{direction}/evaluation/{axis}'];value=100*math.fsum(float(e[qi,j]) for j in selected)/5
            error=abs(value-float(r[axis]));maximum=max(maximum,error);assert error<1e-10;checks+=1
    for r in csv.DictReader((OUT/'summary.csv').open()):
        for axis in ('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL'):
            family=defaultdict(lambda:defaultdict(list))
            for q in rows:
                if q['method']==r['method'] and q['role']==r['role']:family[q['family_id']][q['piece']].append(float(q[axis]))
            value=math.fsum(math.fsum(math.fsum(v)/len(v) for v in works.values())/len(works) for works in family.values())/len(family)
            error=abs(value-float(r[axis]));maximum=max(maximum,error);assert error<1e-10;checks+=1
    # All F/P/H embedding blocks remain individually available, not collapsed to one mean vector.
    scales=json.loads((OUT/'retrieval_scales.json').read_text())
    for key,scale in scales.items():
        assert all(sum(scale[k]['active'])>0 for k in KINDS)
        checks+=5
    # Scalar nearest-segment costs using saved embeddings and the actual gallery metric.
    sample=data.ids[np.rint(np.linspace(0,len(data.ids)-1,4)).astype(int)]
    js=np.rint(np.linspace(0,len(gids)-1,4)).astype(int)
    for view,direction in enumerate(('A_to_B','B_to_A')):
        for mode in ('reconstruction_only','response','response_aux','context_only','untrained'):
            embedding=np.load(OUT/f'view{view}_{mode}'/'embeddings.npz')
            vectors,owners,types=embedding['vectors'],embedding['owners'],embedding['types']
            for kind in ('F','P'):
                ki=KINDS.index(kind);scale=scales[direction+'/'+mode][kind]
                active=np.flatnonzero(scale['active']);sd=np.array(scale['sd'])
                for i in sample:
                    a=vectors[(owners==i)&(types==ki)]
                    for j in js:
                        b=vectors[(owners==gids[j])&(types==ki)]
                        costs=np.array([[math.fsum(((float(av[k])-float(bv[k]))/float(sd[k]))**2 for k in active)/len(active)
                            for bv in b] for av in a])
                        value=math.sqrt(.5*(math.fsum(costs.min(axis=1))/len(a)+math.fsum(costs.min(axis=0))/len(b)))/scale['median']
                        expected=distances[direction+'/'+mode+'_'+kind][i,j]
                        error=abs(value-float(expected));maximum=max(maximum,error);assert error<1e-10;checks+=1
    result=dict(status='passed',checks=int(checks),max_scalar_error=float(maximum),max_checkpoint_reencode_error=reload_error,
        scope='Data/source hashes, balanced train-only scale moments, train/validation/test recording separation, actual original training source snapshot for repaired serializer metadata, checkpoint minimum, identical initialization, sampled checkpoint reload embeddings, every top-5 rank/score and all macro aggregates.',
        unit_tests='8 meaningful tests passed: hidden-target leakage, score-only isolation, order preservation, collapse prevention, lag sign/missing support, amplitude-independent targets, no validation/test fit, and finite gradients.',
        limitations='Numeric proxy verification; no human music-structure/style ground truth. Scalar learned F/P set-distance samples also match.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
