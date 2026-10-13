"""Independent distance/ranking/score verification of the frozen comparison.

Production retrieval/response functions are never imported. Prior independently
verified phrase/tension CSVs support the P/H inputs; raw arrays support B/T/F,
evaluation arithmetic and selected structure shifts.
"""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(rows):
    family=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(rows):family[r['family_id']][r['piece'],r['grid']].append(i)
    w=np.zeros(len(rows))
    for works in family.values():
        for indices in works.values():
            for i in indices:w[i]=1/len(family)/len(works)/len(indices)
    return w


def matrix(x,y,scale=None,width=None):
    active=np.array(scale['active']) if scale else np.ones(x.shape[1],bool)
    sd=np.array(scale['scale']) if scale else np.ones(x.shape[1])
    groups=[np.flatnonzero(active)]
    if width:
        groups=[]
        for channels in ([0],[1],[2],[3],[4,5,6]):
            group=[c*width+j for c in channels for j in range(width) if active[c*width+j]]
            if group:groups.append(group)
    result=np.zeros((len(x),len(y)))
    for group in groups:
        part=np.zeros_like(result)
        for c in group:part+=((x[:,c,None]-y[None,:,c])/sd[c])**2
        result+=part/len(group)/len(groups)
    return np.sqrt(result)


def ranking(d,families,key,seed,k=5):
    hashed=int.from_bytes(hashlib.sha256(f'{seed}/{key}'.encode()).digest()[:8],'little')
    p=np.random.default_rng(hashed).random(len(d))
    indices=sorted((j for j in range(len(d)) if math.isfinite(d[j])),key=lambda j:(d[j],j))
    ordered=[];at=0
    while at<len(indices):
        stop=at+1
        while stop<len(indices) and d[indices[stop]]-d[indices[at]]<=1e-12:stop+=1
        ordered+=sorted(indices[at:stop],key=lambda j:p[j]);at=stop
    selected=[];seen=set()
    for j in ordered:
        if families[j] not in seen:
            selected.append(j);seen.add(families[j])
        if len(selected)==k:return selected
    raise AssertionError('Unsupported K')


def scalar_response(x,valid,pair,label,owner=None):
    expected=[]
    for phase in range(3):
        ids=[i for i in range(len(valid)) if valid[i] and label[i]==phase]
        ps=[i for i in ids if i and pair[i] and valid[i-1] and label[i-1]==phase and (owner is None or owner[i-1]==owner[i])]
        expected.extend([math.fsum(float(x[c,i]) for i in ids)/len(ids) if len(ids)>=8 else math.nan for c in range(7)])
        expected.extend([math.fsum(float(x[c,i]-x[c,i-1]) for i in ps)/len(ps) if len(ps)>=4 else math.nan for c in range(7)])
        expected.extend([math.sqrt(math.fsum(float(x[c,i]-x[c,i-1])**2 for i in ps)/len(ps)) if len(ps)>=4 else math.nan for c in range(7)])
        expected.append(math.fsum(float((x[0,i]-x[0,i-1])*(x[2,i]-x[2,i-1])) for i in ps)/len(ps) if len(ps)>=4 else math.nan)
    return expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/structure_response_comparison')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/structure_response_comparison_run02')
    parser.add_argument('--source',type=Path,default=ROOT.parent/'datasets/phrase_response_run02')
    parser.add_argument('--phrases',type=Path,default=ROOT/'classicfy-ai/analysis/phrase_response')
    parser.add_argument('--tonal-report',type=Path,default=ROOT/'classicfy-ai/analysis/tonal_tension')
    args=parser.parse_args();prov=json.loads((args.out/'provenance.json').read_text())
    for name,expected in prov['inputs'].items():assert digest(name)==expected,name
    for name,expected in prov['code'].items():assert digest(ROOT/name)==expected,name
    for name,expected in prov['results'].items():assert digest(args.out/name)==expected,name
    assert digest(args.artifacts/'comparison.npz')==prov['artifact_sha256']
    z=np.load(args.artifacts/'comparison.npz',allow_pickle=False);source=np.load(args.source/'sequences.npz',allow_pickle=False)
    keys=z['keys'].tolist();gids=z['gallery_indices'];qids=z['query_indices']
    original=read(args.phrases/'coverage.csv');by_key={r['key']:(i,r) for i,r in enumerate(original)}
    metadata=[by_key[k][1] for k in keys];gallery=[metadata[i] for i in gids];families=[r['family_id'] for r in gallery]
    w=weights(gallery);scales=json.loads((args.out/'scales.json').read_text());protocol=json.loads((args.out/'protocol.json').read_text())
    phrase={(r['key'],r['view']):r for r in read(args.phrases/'responses.csv')}
    with gzip.open(args.tonal_report/'responses.csv.gz','rt',encoding='utf-8-sig') as stream:
        tonal={(r['key'],r['metric'],r['view']):r for r in csv.DictReader(stream)}
    checks=0;maximum=0.
    def equal(a,b):
        nonlocal checks,maximum
        a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape,(a.shape,b.shape)
        assert np.array_equal(np.isfinite(a),np.isfinite(b))
        good=np.isfinite(a);checks+=a.size
        error=float(np.max(abs(a[good]-b[good]))) if good.any() else 0.;maximum=max(maximum,error)
        if error>1e-9:raise AssertionError(error)
    rank_rows=read(args.out/'rankings.csv');score_rows=read(args.out/'scores.csv');control_rows=read(args.out/'controls.csv')
    work_groups=defaultdict(list)
    for j,r in enumerate(gallery):work_groups[r['piece'],r['grid']].append(j)
    assert all(len(group)>=3 for group in work_groups.values())
    assert not ({metadata[i]['family_id'] for i in qids}&set(families))
    matrix_checks=0
    for view,direction in enumerate(('A_to_B','B_to_A')):
        rank_cache={};control_rank_cache={}
        for i,key in enumerate(keys):
            original_index,r=by_key[key];x=source[f'{original_index}/values'];mask=source[f'{original_index}/mask'];base_pairs=source[f'{original_index}/pairs']
            n=len(mask);valid=np.array([bool(mask[j]) and j//64%2==view and j<n//64*64 for j in range(n)])
            pair=np.array([bool(base_pairs[j]) and j>0 and valid[j] and valid[j-1] and j%64!=0 for j in range(n)])
            xx=x[:,valid];level=xx.mean(axis=1);level[1]=np.median(abs(xx[1]))
            b=np.column_stack([level,np.ptp(np.percentile(xx,[5,95],axis=1),axis=0)]).ravel()
            equal(b,z[direction+'_input_B'][i])
            delta=x[:,1:]-x[:,:-1];observed=delta[:,pair[1:]]
            time=np.r_[observed.mean(axis=1),np.sqrt((observed**2).mean(axis=1)),np.mean(observed[0]*observed[2])]
            equal(time,z[direction+'_input_T'][i])
            index=np.arange(n);fixed=np.minimum(2,(index%16)*3//16)
            equal(scalar_response(x,valid,pair,fixed,index//16),z[direction+'_input_F'][i])
            equal([float(phrase[key,'AB'[view]][f'p{j}']) for j in range(66)],z[direction+'_input_P'][i])
            equal([float(tonal[key,name,'AB'[view]][f'h{j}']) for name in ('cloud_diameter','cloud_momentum','tensile_strain') for j in range(66)],z[direction+'_input_H'][i])
            # Verify held-out E-L/T/R/S arithmetic from the other original blocks.
            v=1-view;chosen=np.array([bool(mask[j]) and j//64%2==v and j<n//64*64 for j in range(n)])
            selected_pairs=np.array([bool(base_pairs[j]) and j>0 and chosen[j] and chosen[j-1] and j%64!=0 for j in range(n)])
            blocks=[j for j in range(0,n-63,64) if j//64%2==v]
            el=[];et=[];ers=[]
            for c in range(7):
                values=x[c,chosen];el.extend([float(np.median(abs(values))) if c==1 else float(np.median(values)),float(np.ptp(np.percentile(values,[25,75])))])
                rms=[]
                for start in blocks:
                    for offset in (0,32):
                        a=start+offset;ps=[j for j in range(a+1,a+32) if selected_pairs[j]]
                        if len(ps)>=25:rms.append(math.sqrt(math.fsum(float(x[c,j]-x[c,j-1])**2 for j in ps)/len(ps)))
                et.extend(np.log1p(np.percentile(rms,[25,50,75])))
            for a in blocks:
                ps=[j for j in range(a+1,a+64) if selected_pairs[j]]
                if len(ps)>=51:
                    dx=np.array([x[0,j]-x[0,j-1] for j in ps]);dy=np.array([x[2,j]-x[2,j-1] for j in ps])
                    dx-=dx.mean();dy-=dy.mean()
                    if min(np.std(dx),np.std(dy))>1e-12:ers.append(float(np.dot(dx,dy)/np.sqrt(np.dot(dx,dx)*np.dot(dy,dy))))
            er=np.r_[np.percentile(ers,[25,50,75]),np.std(ers)]
            phase=source[f'{original_index}/phase'];owner=source[f'{original_index}/owner'];es=[]
            for p in range(3):
                ps=[j for j in range(1,n) if selected_pairs[j] and phase[j-1]==phase[j]==p and owner[j-1]==owner[j]]
                d0=np.array([x[0,j]-x[0,j-1] for j in ps]);d2=np.array([x[2,j]-x[2,j-1] for j in ps])
                es.extend([np.mean(d0>1e-6),np.mean(d2>1e-6),np.mean(d0*d2>1e-12)])
            for name,a in [('E-L',el),('E-T',et),('E-R',er),('E-S',es)]:equal(a,z[direction+'_judge_'+name][i])
            # Two fixed seeds validate actual circular-shift arithmetic, not just saved distances.
            for seed in (20261014,20261033):
                shifted_phase=phase.copy();shifted_owner=owner.copy()
                hashed=int.from_bytes(hashlib.sha256(f'{seed}/{key}/{view}'.encode()).digest()[:8],'little');rng=np.random.default_rng(hashed)
                for a in range(0,n-63,64):
                    if a//64%2==view:
                        offset=int(rng.integers(1,64));shifted_phase[a:a+64]=np.roll(shifted_phase[a:a+64],offset);shifted_owner[a:a+64]=np.roll(shifted_owner[a:a+64],offset)
                equal(scalar_response(x,valid,pair,shifted_phase,shifted_owner),z[f'{direction}_shift_{seed}_P'][i])
        blocks={}
        for name in ('B','T','F','P','H','S','Q'):
            x=z[direction+'_input_'+name];scale=scales[direction]['scales'][name]
            if scale is not None:
                mean=np.sum(x[gids]*w[:,None],axis=0);sd=np.sqrt(np.sum((x[gids]-mean)**2*w[:,None],axis=0))
                equal(mean,scale['mean']);equal(sd,scale['scale']);assert (sd>1e-12).tolist()==scale['active']
            blocks[name]=matrix(x,x[gids],scale,2 if name=='B' else None);matrix_checks+=1
            medians=[]
            for a in range(len(gids)):
                for b in range(a+1,len(gids)):
                    if families[a]!=families[b]:medians.append((float(blocks[name][gids[a],b]),float(w[a]*w[b])))
            medians.sort(key=lambda r:r[0]);half=math.fsum(p for d,p in medians)/2;running=0.
            for d,p in medians:
                running+=p
                if running>=half:median=d;break
            equal(median,scales[direction]['block_medians'][name])
        for method,components in protocol['arms'].items():
            if method=='M-CT':expected=np.array([[.5*(r['composer']!=g['composer'])+.5*(r['work_type']!=g['work_type']) for g in gallery] for r in metadata])
            else:expected=np.sqrt(sum((blocks[c]/scales[direction]['block_medians'][c])**2 for c in components)/len(components))
            equal(expected,z[direction+'_distance_'+method])
        eval_scores={}
        for axis in ('E-L','E-T','E-R','E-S'):
            x=z[direction+'_judge_'+axis];ed=matrix(x,x[gids],scales[direction]['evaluation_scales'][axis],2 if axis=='E-L' else 3 if axis=='E-T' else None)
            scores=[]
            for ds in ed:
                order=np.argsort(ds,kind='stable');sorted_distances=ds[order];cumulative=np.r_[0.,np.cumsum(w[order])]
                left=np.searchsorted(sorted_distances,ds-1e-12,side='left');right=np.searchsorted(sorted_distances,ds+1e-12,side='right')
                scores.append(1-cumulative[right]+.5*(cumulative[right]-cumulative[left]))
            eval_scores[axis]=np.array(scores);equal(scores,z[direction+'_evaluation_'+axis])
        # Refit all shifted blocks from training only and recompute their distances.
        for seed in protocol['controls']['structure_shift_seeds']:
            shifted_blocks={}
            for name in ('P','H'):
                x=z[f'{direction}_shift_{seed}_{name}']
                mean=np.sum(x[gids]*w[:,None],axis=0);sd=np.sqrt(np.sum((x[gids]-mean)**2*w[:,None],axis=0))
                scale=dict(active=(sd>1e-12).tolist(),scale=sd.tolist())
                shifted_distance=matrix(x,x[gids],scale)
                weighted=[]
                for a in range(len(gids)):
                    for b in range(a+1,len(gids)):
                        if families[a]!=families[b]:weighted.append((float(shifted_distance[gids[a],b]),float(w[a]*w[b])))
                weighted.sort(key=lambda r:r[0]);half=math.fsum(p for d,p in weighted)/2;running=0.
                for d,p in weighted:
                    running+=p
                    if running>=half:median=d;break
                shifted_blocks[name]=shifted_distance/median
            for method in ('B+P','B+H','B+P+H'):
                pieces=[blocks['B']/scales[direction]['block_medians']['B']]+[shifted_blocks[name] for name in method.split('+')[1:]]
                expected=np.sqrt(sum(part**2 for part in pieces)/len(pieces))
                equal(expected,z[f'{direction}_shift_{seed}_distance_{method}'])
        qi_by_key={metadata[i]['key']:i for i in qids}
        for row in [r for r in rank_rows if r['direction']==direction]:
            qi=qi_by_key[row['query']];ds=z[direction+'_distance_'+row['method']][qi]
            selected=ranking(ds,families,row['query'],20261014)
            j=selected[int(row['rank'])-1];assert gallery[j]['key']==row['candidate'];equal(ds[j],float(row['distance']))
        for row in [r for r in score_rows if r['direction']==direction]:
            qi=qi_by_key[row['key']];ds=z[direction+'_distance_'+row['method']][qi];scores=eval_scores[row['axis']][qi]
            selected=[]
            seeds=range(20261014,20261114) if row['method']=='M-CT' else [20261014]
            for seed in seeds:
                identity=(row['method'],qi,seed)
                if identity not in rank_cache:rank_cache[identity]=ranking(ds,families,row['key'],seed)
                selected.append(rank_cache[identity])
            value=np.mean([np.mean(scores[s]) for s in selected]);equal(value,float(row['score']))
            expected=np.mean([np.mean([np.mean(scores[work_groups[gallery[j]['piece'],gallery[j]['grid']]]) for j in choice]) for choice in selected])
            equal(expected,float(row['same_work_expected']))
        # Every seed's shifted and count-balanced ranks are reselected and macro-scored.
        for row in [r for r in control_rows if r['direction']==direction]:
            method=row['method'];seed=int(row['seed']);scores=eval_scores[row['axis']];d=z[direction+'_distance_'+method]
            if row['control']=='structure_shift':d=z[f'{direction}_shift_{seed}_distance_{method}']
            selected_by_family=defaultdict(lambda:defaultdict(list))
            for qi in qids:
                if metadata[qi]['evaluation_role']!=row['role']:continue
                ds=d[qi].copy()
                if row['control']=='balanced_gallery_3':
                    chosen=z[f'{direction}_balanced_{seed}'];assert len(chosen)==3*len(work_groups)
                    assert all(sum(j in chosen for j in group)==3 for group in work_groups.values())
                    inactive=np.ones(len(ds),bool);inactive[chosen]=False;ds[inactive]=np.inf
                identity=(row['control'],method,seed,qi)
                if identity not in control_rank_cache:control_rank_cache[identity]=ranking(ds,families,metadata[qi]['key'],seed)
                top=control_rank_cache[identity]
                selected_by_family[metadata[qi]['family_id']][metadata[qi]['piece']].append(float(np.mean(scores[qi,top])))
            value=np.mean([np.mean([np.mean(v) for v in works.values()]) for works in selected_by_family.values()])
            equal(value,float(row['score']))
    # Independently aggregate both direction rows by work then family.
    for row in read(args.out/'summary.csv'):
        selected=[r for r in score_rows if all(r[k]==row[k] for k in ('role','method','axis'))]
        groups=defaultdict(lambda:defaultdict(list))
        for r in selected:groups[r['family_id']][r['piece']].append(float(r['score']))
        equal(np.mean([np.mean([np.mean(v) for v in works.values()]) for works in groups.values()]),float(row['score']))
    result=dict(status='passed',numeric_checks=checks,maximum_absolute_error=maximum,
        common_performances=len(keys),gallery=len(gids),queries=len(qids),matrix_blocks=matrix_checks,
        scope='Original-index B/T/F and held-out judges; prior verified P/H CSV inputs; two deterministic phase shifts; independent train scales/main and all shifted distances/percentiles; all main and control ranks/scores/macro aggregation; no production retrieval functions',
        limitation='Other shifted-response arithmetic and auxiliary score-only S/Q descriptors rely on unit tests and saved inputs; all their distances and ranks are checked',code_sha256=digest(Path(__file__)))
    (args.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
