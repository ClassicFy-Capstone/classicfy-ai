"""Compare frozen summaries against sets of individual ordered curves."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
REPO = WORK/'capstone-2026-2-classicfy'
sys.path.insert(0, str(REPO/'classicfy-ai/src'))
from embedding.retrieval_probe import weights, fit_scale, distance, weighted_quantile, ranked, priorities, percentile_scores


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def macro(rows, field):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r['family_id']][r['piece']].append(r[field])
    return float(np.mean([np.mean([np.mean(v) for v in works.values()]) for works in groups.values()]))


def channel_scale(curves, gids, gw):
    means = np.array([curves[i].mean(axis=(0,2)) for i in gids])
    seconds = np.array([(curves[i]**2).mean(axis=(0,2)) for i in gids])
    variance = gw @ seconds - (gw @ means)**2
    active = variance > 1e-12
    assert active.any()
    return np.sqrt(np.maximum(variance,1e-12)), active


def set_distances(curves, ids, gids, scale):
    sd, active = scale
    vectors = {i: (curves[i][:,active,:]/sd[None,active,None]).reshape(len(curves[i]),-1) for i in ids}
    starts = np.r_[0,np.cumsum([len(vectors[i]) for i in gids])]
    pool = np.concatenate([vectors[i] for i in gids])
    result = np.full((335,len(gids)),np.inf)
    for i in ids:
        costs = cdist(vectors[i],pool,'sqeuclidean')/vectors[i].shape[1]
        for j in range(len(gids)):
            local = costs[:,starts[j]:starts[j+1]]
            result[i,j] = np.sqrt(.5*(local.min(axis=1).mean()+local.min(axis=0).mean()))
    return result


def calibrated(d, gids, gw, families):
    a,b = np.triu_indices(len(gids),1)
    good = families[a]!=families[b]; a,b=a[good],b[good]
    median = float(weighted_quantile(d[gids[a],b],gw[a]*gw[b],.5))
    assert median > 1e-12
    return d/median, median


def control_permutation(curves, seed, view, kind):
    result = {}
    kind_id = {'F':0,'P':1,'H0':2,'H1':3,'H2':4}[kind]
    for i,x in curves.items():
        rng = np.random.default_rng(np.random.SeedSequence([seed,view,kind_id,i]))
        result[i] = x[:,:,rng.permutation(x.shape[2])]
    return result


def evaluate(method,d,evaluations,metadata,gids,qids,seed=20261014,allowed=None):
    gallery = [metadata[i] for i in gids]
    family = np.array([r['family_id'] for r in gallery])
    work_groups = defaultdict(list)
    for j,r in enumerate(gallery):
        if allowed is None or j in allowed:
            work_groups[r['piece'],r['grid']].append(j)
    records, ranking = [], []
    for qi in qids:
        values = d[qi].copy()
        if allowed is not None:
            values[np.setdiff1d(np.arange(len(gids)),allowed)] = np.inf
        selected = ranked(values,family,priorities(seed,metadata[qi]['key'],len(gids)),5)
        assert selected is not None
        row=metadata[qi]
        record = dict(key=row['key'],piece=row['piece'],family_id=row['family_id'],role=row['evaluation_role'],method=method)
        for axis,e in evaluations.items():
            record[axis]=100*float(e[qi,selected].mean())
            record[axis+'_same_work']=100*float(np.mean([e[qi,work_groups[gallery[j]['piece'],gallery[j]['grid']]].mean() for j in selected]))
        records.append(record)
        ranking.extend(dict(query=row['key'],method=method,rank=rank,candidate=gallery[j]['key'],distance=float(values[j])) for rank,j in enumerate(selected,1))
    return records, ranking


def main():
    z=np.load(WORK/'datasets/standalone_structure_run01/standalone.npz',allow_pickle=False)
    curves_z=np.load(OUT/'curves.npz',allow_pickle=False)
    prep=json.loads((OUT/'preparation_verification.json').read_text())
    assert sha(OUT/'curves.npz')==prep['curves_sha256']
    assert sha(WORK/'datasets/standalone_structure_run01/standalone.npz')==prep['source_sha256']
    assert np.array_equal(z['keys'],curves_z['keys'])
    with (REPO/'classicfy-ai/analysis/phrase_response/coverage.csv').open() as f:
        lookup={r['key']:r for r in csv.DictReader(f)}
    metadata=[lookup[k] for k in z['keys']]
    cohort=json.loads((OUT/'cohort.json').read_text())
    ids=np.array(cohort['joint']); gids=np.array([i for i in ids if metadata[i]['evaluation_role']=='train'])
    qids=np.array([i for i in ids if metadata[i]['evaluation_role'] in ('test','validation')])
    gallery=[metadata[i] for i in gids]; gw=weights(gallery)
    families=np.array([r['family_id'] for r in gallery])
    assert not {metadata[i]['family_id'] for i in qids}&set(families)
    results,rankings,controls,scale_records=[],[],[],{}
    arrays={'keys':z['keys'],'eligible':ids,'gallery_indices':gids,'query_indices':qids}
    for view,direction in enumerate(('A_to_B','B_to_A')):
        inputs={name:z[f'{direction}_input_{name}'] for name in ('B','T','F','P','H')}
        distances={}
        for name,x in inputs.items():
            d=distance(x,x[gids],None if name=='B' else fit_scale(x[gids],gw),group_width=2 if name=='B' else None)
            distances[name+'_summary'],_=calibrated(d,gids,gw,families)
        distances['P+H_summary']=np.sqrt((distances['P_summary']**2+distances['H_summary']**2)/2)
        for name in ('P','H'):
            x=inputs[name].reshape(335,-1,22)[:,:,7:].reshape(335,-1)
            d=distance(x,x[gids],fit_scale(x[gids],gw))
            distances[name+'_change_only'],_=calibrated(d,gids,gw,families)
        distances['P+H_change_only']=np.sqrt((distances['P_change_only']**2+distances['H_change_only']**2)/2)
        curve_data={}; fitted={}; scale_records[direction]={}
        for centered in (False,True):
            variant='centered' if centered else 'raw'
            blocks={}
            for kind in ('F','P','H0','H1','H2'):
                data={i:curves_z[f'{i}/{view}/{kind}'].copy() for i in ids}
                if centered:
                    data={i:x-curves_z[f'{i}/{view}/mean'][None,:,None] for i,x in data.items()}
                scale=channel_scale(data,gids,gw)
                d=set_distances(data,ids,gids,scale)
                d,median=calibrated(d,gids,gw,families)
                blocks[kind]=d
                curve_data[kind,variant]=data; fitted[kind,variant]=scale
                scale_records[direction][kind+'_'+variant]=dict(sd=scale[0].tolist(),active=scale[1].tolist(),median=median)
            for kind in ('F','P'):
                distances[kind+'_curve_'+variant]=blocks[kind]
            h=np.sqrt(np.mean([blocks['H'+str(k)]**2 for k in range(3)],axis=0))
            # Calibrate H as a block before assigning it equal weight with P.
            h,median=calibrated(h,gids,gw,families)
            scale_records[direction]['H_'+variant]=dict(median=median)
            distances['H_curve_'+variant]=h
            distances['P+H_curve_'+variant]=np.sqrt((blocks['P']**2+h**2)/2)
        evaluations={}
        for axis in ('E-L','E-T','E-R','E-S'):
            a=z[f'{direction}_judge_{axis}']
            d=distance(a,a[gids],fit_scale(a[gids],gw),group_width=2 if axis=='E-L' else 3 if axis=='E-T' else None)
            evaluations[axis]=percentile_scores(d,gw)
        # These judges come only from the opposite view.
        judge={i:curves_z[f'{i}/{1-view}/P_judge'] for i in ids}
        d=set_distances(judge,ids,gids,channel_scale(judge,gids,gw))
        finite_d=d[ids]
        evaluations['E-PC']=np.zeros_like(d)
        evaluations['E-PC'][ids]=percentile_scores(finite_d,gw)
        normalized_judge={}
        for i in ids:
            x=judge[i]; rms=np.sqrt(np.mean(x*x,axis=2))
            good=(rms>1e-6).all(axis=1)
            assert good.any(), (i,'no amplitude-normalized phrase curve')
            normalized_judge[i]=x[good]/rms[good,:,None]
        d=set_distances(normalized_judge,ids,gids,channel_scale(normalized_judge,gids,gw))
        evaluations['E-PN']=np.zeros_like(d)
        evaluations['E-PN'][ids]=percentile_scores(d[ids],gw)
        a=np.array([curves_z[f'{i}/{1-view}/H_lag'] for i in range(335)])
        assert np.isfinite(a[ids]).all()
        d=distance(a[ids],a[gids],fit_scale(a[gids],gw))
        evaluations['E-HL']=np.zeros((335,len(gids)))
        evaluations['E-HL'][ids]=percentile_scores(d,gw)
        for axis,e in evaluations.items():
            arrays[f'{direction}/evaluation/{axis}']=e
        for method,d in distances.items():
            r,rr=evaluate(method,d,evaluations,metadata,gids,qids)
            results.extend(dict(direction=direction,**x) for x in r)
            rankings.extend(dict(direction=direction,**x) for x in rr)
            arrays[f'{direction}/distance/{method}']=d
        # Existing candidate-count control, restricted to groups with >=3 eligible recordings.
        groups=defaultdict(list)
        for j,r in enumerate(gallery):groups[r['piece'],r['grid']].append(j)
        eligible_groups=[v for v in groups.values() if len(v)>=3]
        assert len(set(families[j] for group in eligible_groups for j in group))>=5
        for seed in range(20261014,20261034):
            rng=np.random.default_rng(seed)
            allowed=np.array(sorted(j for group in eligible_groups for j in rng.choice(group,3,replace=False)))
            for method,d in distances.items():
                r,_=evaluate(method,d,evaluations,metadata,gids,qids,seed,allowed)
                for role in ('test','validation'):
                    rs=[x for x in r if x['role']==role]
                    controls.append(dict(direction=direction,control='three_candidates',seed=seed,method=method,role=role,
                        **{axis:macro(rs,axis) for axis in evaluations}))
        for seed in range(20261014,20261034):
            blocks={}
            for kind in ('F','P','H0','H1','H2'):
                permuted=control_permutation(curve_data[kind,'centered'],seed,view,kind)
                d=set_distances(permuted,ids,gids,fitted[kind,'centered'])
                blocks[kind],_=calibrated(d,gids,gw,families)
            h=np.sqrt(np.mean([blocks['H'+str(k)]**2 for k in range(3)],axis=0))
            h,_=calibrated(h,gids,gw,families)
            methods=dict(F_curve_centered=blocks['F'],P_curve_centered=blocks['P'],H_curve_centered=h,
                         **{'P+H_curve_centered':np.sqrt((blocks['P']**2+h**2)/2)})
            for method,d in methods.items():
                r,_=evaluate(method,d,evaluations,metadata,gids,qids)
                for role in ('test','validation'):
                    rs=[x for x in r if x['role']==role]
                    controls.append(dict(direction=direction,control='point_order_permuted',seed=seed,method=method,role=role,
                        **{axis:macro(rs,axis) for axis in evaluations}))
            if (seed-20261014+1)%5==0:print(direction,'order controls',seed-20261014+1,'/20',flush=True)
        print('Finished',direction,flush=True)
    axes=('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL')
    summary=[]
    for role in ('test','validation'):
        for method in sorted(set(r['method'] for r in results)):
            rs=[r for r in results if r['role']==role and r['method']==method]
            summary.append(dict(role=role,method=method,**{a:macro(rs,a) for a in axes},
                **{a+'_same_work':macro(rs,a+'_same_work') for a in axes}))
    control_summary=[]
    for control in ('three_candidates','point_order_permuted'):
        for role in ('test','validation'):
            for method in sorted(set(r['method'] for r in controls if r['control']==control)):
                rs=[r for r in controls if r['control']==control and r['role']==role and r['method']==method]
                control_summary.append(dict(control=control,role=role,method=method,**{a:float(np.mean([r[a] for r in rs])) for a in axes}))
    write('scores.csv',results);write('rankings.csv',rankings);write('summary.csv',summary)
    write('controls.csv',controls);write('control_summary.csv',control_summary)
    (OUT/'scales.json').write_text(json.dumps(scale_records,indent=2)+'\n')
    np.savez_compressed(OUT/'comparison.npz',**arrays)
    (OUT/'provenance.json').write_text(json.dumps(dict(source=prep['source_sha256'],curves=prep['curves_sha256'],
        code={p.name:sha(p) for p in (OUT/'prepare.py',Path(__file__))},
        results={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ('provenance.json','mismatch.npz','bridge_debug.npz')}),indent=2)+'\n')
    for r in summary:print(r['role'],r['method'],{a:round(r[a],2) for a in axes},flush=True)


if __name__=='__main__':main()
