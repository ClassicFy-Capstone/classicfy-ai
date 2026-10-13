"""Frozen cross-work Top-5 comparison on disjoint original-beat views."""

import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from embedding.structure_retrieval import view_descriptors,heldout_judges
from embedding.retrieval_probe import (weights,fit_scale,distance,weighted_quantile,
    ranked,priorities,percentile_scores,overlap)
from calculate_phrase_response import dump,provenance
from validate_score_beat_bridge import rows,sha,write

ARMS={'B':('B',),'B+T':('B','T'),'B+F':('B','F'),'B+P':('B','P'),
      'B+H':('B','H'),'B+P+H':('B','P','H'),'structure_only':('S',),'Q':('Q',)}
SEEDS=tuple(range(20261014,20261034))


def derived_seed(seed,key,view):
    return int.from_bytes(hashlib.sha256(f'{seed}/{key}/{view}'.encode()).digest()[:8],'little')


def macro(records,field='score'):
    groups=defaultdict(lambda:defaultdict(list))
    for row in records:groups[row['family_id']][row['piece']].append(float(row[field]))
    family={f:float(np.mean([np.mean(v) for v in works.values()])) for f,works in groups.items()}
    return float(np.mean(list(family.values()))) if family else np.nan,family


def distances(descriptors,metadata,gids):
    w=weights([metadata[i] for i in gids]);scales={};blocks={};calibration={}
    families=np.array([metadata[i]['family_id'] for i in gids]);a,b=np.triu_indices(len(gids),1)
    good=families[a]!=families[b];a,b=a[good],b[good];pair_weight=w[a]*w[b]
    for name,x in descriptors.items():
        if name=='B':scales[name]=None;blocks[name]=distance(x,x[gids],group_width=2)
        else:scales[name]=fit_scale(x[gids],w);blocks[name]=distance(x,x[gids],scales[name])
        median=float(weighted_quantile(blocks[name][gids[a],b],pair_weight,.5))
        if median<=1e-12:raise ValueError('No supported training distance: '+name)
        calibration[name]=median
    arms={name:np.sqrt(np.mean([(blocks[c]/calibration[c])**2 for c in components],axis=0)) for name,components in ARMS.items()}
    arms['M-CT']=np.array([[.5*(r['composer']!=metadata[g]['composer'])+.5*(r['work_type']!=metadata[g]['work_type']) for g in gids] for r in metadata])
    return arms,dict(scales=scales,block_medians=calibration)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT.parent/'datasets/phrase_response_run02')
    parser.add_argument('--tension',type=Path,default=ROOT.parent/'datasets/tonal_tension_run01')
    parser.add_argument('--phrases',type=Path,default=ROOT/'classicfy-ai/analysis/phrase_response')
    parser.add_argument('--tonal-report',type=Path,default=ROOT/'classicfy-ai/analysis/tonal_tension')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/structure_response_comparison')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/structure_response_comparison_run02')
    args=parser.parse_args()
    for folder in (args.out,args.artifacts):
        if folder.exists() and any(folder.iterdir()):raise ValueError('Choose empty output: '+str(folder))
        folder.mkdir(parents=True,exist_ok=True)
    source_report=args.phrases;tonal_report=args.tonal_report
    source_prov=json.loads((source_report/'provenance.json').read_text());tonal_prov=json.loads((tonal_report/'provenance.json').read_text())
    if sha(args.source/'sequences.npz')!=source_prov['artifact_sha256']:raise ValueError('Phrase arrays changed')
    if sha(args.tension/'tension.npz')!=tonal_prov['artifacts']['tension.npz']:raise ValueError('Tension arrays changed')
    protocol=dict(date='2026-10-14',phrase_commit='ceee302',tension_commit='ee20a2e',
        prior_design_sha256=sha(source_report/'protocol.json'),
        K=5,arms={**ARMS,'M-CT':['composer','work_type']},
        dimensions=dict(B=14,T=15,F=66,P=66,H=198,S=9,Q=2),
        time_baseline='15D mean/RMS seven-channel delta + Tempo-Dynamics joint moment; this is a new interpretable time baseline, not the preceding 57D/119D lag experiment',
        split='Existing clean family evaluation_role; train gallery, test primary, validation separately; both A->B and B->A',
        views='Full alternating 64-original-beat blocks; no pair across blocks, structural jumps, missing, bR, context bin and phrase owner',
        common_cohort='All primary descriptors and E-L/T/R/S finite in both views, all 20 shifted P/H descriptors finite in both views; known type; gallery >=3 per piece/grid; no outcome-based changes',
        normalization='Existing catalog common residual/scale uses whole catalog; disjoint beats are a consistency check, not fully independent new-reference inference',
        ranking='One recording per work family; same full gallery for all conditions; ties <=1e-12 with common deterministic priorities',
        training='Family-balanced train-only coordinate SD excluding constant coordinates; B keeps existing five feature-group weighting; each block calibrated by train other-family weighted median; equal squared block distances',
        judges=dict(primary='E-T: 32-beat delta RMS log1p q25/q50/q75; >=25 pairs, >=3 blocks',
            level='E-L: 7-channel median/IQR, >=100 valid beats',
            relation='E-R: Tempo-Dynamics delta r q25/q50/q75/SD; >=51 pairs in each of >=2 blocks',
            structure='E-S: per phrase phase positive Tempo delta, positive Dynamics delta, same-direction nonzero product fractions; >=4 pairs; differs from search coordinates'),
        score='Family->piece/grid->recording uniform gallery-distance percentile; tie credit 0.5; macro query work then family; general random expectation 0.5',
        controls=dict(metadata_tie_seeds=list(range(20261014,20261114)),same_work_recording_seeds=list(range(20261014,20261114)),
            structure_shift_seeds=SEEDS,shift='Circularly shift P phase/owner and H labels together within each search block, leaving feature values and original pair masks fixed; recompute responses, train scales and ranks',
            balanced_gallery='20 seeds, exactly three random train recordings per piece/grid; fixed full-gallery scales and evaluation percentiles',
            Q='Train-scaled log1p(original length) and original combined coverage baseline'),
        uncertainty='Exploratory reused evaluation families; per-family results and seed ranges, no confidence interval when fewer than 5 test families',
        evidence='A changed list is not success. Compare B+P to B+F, B+H to B+F and B+P+H to B+T on E-T, retain E-L/E-R/E-S and all controls. No claim of listening/preferences or expert phrase/key accuracy.')
    dump(args.out/'protocol.json',protocol)
    coverage=rows(source_report/'coverage.csv');z=np.load(args.source/'sequences.npz',allow_pickle=False);t=np.load(args.tension/'tension.npz',allow_pickle=False)
    if not np.array_equal(z['keys'],t['keys']):raise ValueError('Performance order differs')
    D={view:defaultdict(list) for view in (0,1)};J={view:defaultdict(list) for view in (0,1)}
    for i,r in enumerate(coverage):
        source=[z[f'{i}/{name}'] for name in ('values','mask','pairs','phase','owner')]
        tonal=[t[f'{i}/{name}'] for name in ('tension','labels','key_index')]
        for view in (0,1):
            for name,value in view_descriptors(*source,*tonal,view).items():D[view][name].append(value)
            for name,value in heldout_judges(*source,view).items():J[view][name].append(value)
        if (i+1)%100==0:print('Prepared',i+1,'/570',flush=True)
    D={v:{n:np.array(a) for n,a in d.items()} for v,d in D.items()};J={v:{n:np.array(a) for n,a in d.items()} for v,d in J.items()}
    primary=np.logical_and.reduce([np.isfinite(a).all(axis=1) for d in [*D.values(),*J.values()] for a in d.values()])
    role=np.array([r['evaluation_role'] in ('train','validation','test') and r['work_type']!='unclassified' for r in coverage])
    preliminary=np.flatnonzero(primary&role)
    shifts={};shift_supported=np.ones(len(coverage),bool)
    for seed in SEEDS:
        for view in (0,1):
            p=np.full((len(coverage),66),np.nan);h=np.full((len(coverage),198),np.nan)
            for i in preliminary:
                source=[z[f'{i}/{name}'] for name in ('values','mask','pairs','phase','owner')]
                tonal=[t[f'{i}/{name}'] for name in ('tension','labels','key_index')]
                desc=view_descriptors(*source,*tonal,view,shift_seed=derived_seed(seed,coverage[i]['key'],view))
                p[i]=desc['P'];h[i]=desc['H']
            shifts[seed,view]=(p,h)
            shift_supported&=np.isfinite(p).all(axis=1)&np.isfinite(h).all(axis=1)
        print('Prepared structure shift',seed,flush=True)
    keep=primary&role&shift_supported
    counts=Counter((r['piece'],r['grid']) for i,r in enumerate(coverage) if keep[i] and r['evaluation_role']=='train')
    keep&=np.array([r['evaluation_role']!='train' or counts[r['piece'],r['grid']]>=3 for r in coverage])
    eligibility=[]
    for i,r in enumerate(coverage):
        eligibility.append(dict(key=r['key'],piece=r['piece'],family_id=r['family_id'],role=r['evaluation_role'],
            primary_supported=bool(primary[i]),shift_supported=bool(shift_supported[i]),role_supported=bool(role[i]),included=bool(keep[i])))
    write(args.out/'eligibility.csv',eligibility)
    ids=np.flatnonzero(keep);metadata=[coverage[i] for i in ids]
    gids=np.array([i for i,r in enumerate(metadata) if r['evaluation_role']=='train'])
    qids=np.array([i for i,r in enumerate(metadata) if r['evaluation_role'] in ('test','validation')])
    gallery=[metadata[i] for i in gids];families=np.array([r['family_id'] for r in gallery]);gw=weights(gallery)
    if len(set(families))<5 or not any(metadata[i]['evaluation_role']=='test' for i in qids):raise ValueError('Frozen protocol unsupported; see eligibility.csv')
    cohort=dict(original=570,primary_supported=int(primary.sum()),primary_and_role_supported=int((primary&role).sum()),
        included=len(ids),gallery_recordings=len(gids),gallery_families=len(set(families)),gallery_works=len({r['piece'] for r in gallery}),
        queries={role:sum(metadata[i]['evaluation_role']==role for i in qids) for role in ('test','validation')},
        query_families={role:len({metadata[i]['family_id'] for i in qids if metadata[i]['evaluation_role']==role}) for role in ('test','validation')},
        keys=[r['key'] for r in metadata])
    dump(args.out/'cohort.json',cohort);print('Frozen cohort',json.dumps({k:v for k,v in cohort.items() if k!='keys'}),flush=True)
    arrays={'keys':np.array(cohort['keys']),'gallery_indices':gids,'query_indices':qids}
    score_rows=[];rank_rows=[];control_rows=[];scale_records={}
    work_groups=defaultdict(list)
    for j,r in enumerate(gallery):work_groups[r['piece'],r['grid']].append(j)
    for view in (0,1):
        direction='A_to_B' if view==0 else 'B_to_A'
        desc={n:a[ids] for n,a in D[view].items()};judge={n:a[ids] for n,a in J[1-view].items()}
        dd,scales=distances(desc,metadata,gids);scale_records[direction]=scales
        for n,a in desc.items():arrays[direction+'_input_'+n]=a
        evaluation={};eval_scales={}
        for n,a in judge.items():
            scale=fit_scale(a[gids],gw);eval_scales[n]=scale
            ed=distance(a,a[gids],scale,group_width=2 if n=='E-L' else 3 if n=='E-T' else None)
            evaluation[n]=percentile_scores(ed,gw);arrays[direction+'_judge_'+n]=a
            arrays[direction+'_evaluation_'+n]=evaluation[n]
        scales['evaluation_scales']=eval_scales
        for name,a in dd.items():arrays[direction+'_distance_'+name]=a
        baseline_ranks={}
        metadata_ranks={}
        for qi in qids:
            priority=priorities(20261014,metadata[qi]['key'],len(gallery))
            baseline_ranks[qi]=ranked(dd['B'][qi],families,priority,5)
            metadata_ranks[qi]=[ranked(dd['M-CT'][qi],families,priorities(seed,metadata[qi]['key'],len(gallery)),5) for seed in range(20261014,20261114)]
        for name in dd:
            for qi in qids:
                selected=ranked(dd[name][qi],families,priorities(20261014,metadata[qi]['key'],len(gallery)),5)
                if selected is None:raise AssertionError('Insufficient frozen gallery')
                choices=metadata_ranks[qi] if name=='M-CT' else [selected]
                overlaps=[overlap(c,meta,families) for c in choices for meta in metadata_ranks[qi]] if name!='M-CT' else [overlap(choices[0],c,families) for c in choices[1:]]
                novelty=1-float(np.mean([o['recording_overlap'] for o in overlaps]));family_overlap=float(np.mean([o['family_overlap'] for o in overlaps]))
                for rank,j in enumerate(selected,1):rank_rows.append(dict(direction=direction,query=metadata[qi]['key'],query_family=metadata[qi]['family_id'],role=metadata[qi]['evaluation_role'],method=name,rank=rank,candidate=gallery[j]['key'],candidate_family=gallery[j]['family_id'],distance=float(dd[name][qi,j])))
                for axis,scores in evaluation.items():
                    selected_scores=[float(np.mean(scores[qi,c])) for c in choices]
                    same_work=[];within=[];sw_seeds=[]
                    for choice in choices:
                        same_work.append(float(np.mean([np.mean(scores[qi,work_groups[gallery[j]['piece'],gallery[j]['grid']]]) for j in choice])))
                        within.append(float(np.mean([np.mean([float(scores[qi,j]>scores[qi,k]+1e-12)+.5*float(abs(scores[qi,j]-scores[qi,k])<=1e-12) for k in work_groups[gallery[j]['piece'],gallery[j]['grid']] if k!=j]) for j in choice])))
                    for seed in range(20261014,20261114):
                        rng=np.random.default_rng(derived_seed(seed,metadata[qi]['key'],view))
                        sw=[rng.choice(work_groups[gallery[j]['piece'],gallery[j]['grid']]) for j in selected]
                        sw_seeds.append(float(np.mean(scores[qi,sw])))
                    score_rows.append(dict(direction=direction,key=metadata[qi]['key'],piece=metadata[qi]['piece'],family_id=metadata[qi]['family_id'],role=metadata[qi]['evaluation_role'],method=name,axis=axis,
                        score=float(np.mean(selected_scores)),same_work_expected=float(np.mean(same_work)),within_work_selection=float(np.mean(within)),
                        seed_p05=float(np.quantile(selected_scores,.05)),seed_p95=float(np.quantile(selected_scores,.95)),same_work_seed_p05=float(np.quantile(sw_seeds,.05)),same_work_seed_p95=float(np.quantile(sw_seeds,.95)),
                        novelty_vs_metadata=novelty,family_overlap_vs_metadata=family_overlap))
        for seed in SEEDS:
            shifted=desc.copy();sp,sh=shifts[seed,view];shifted['P']=sp[ids];shifted['H']=sh[ids]
            sd,ss=distances(shifted,metadata,gids)
            for name in ('B+P','B+H','B+P+H'):
                for axis,scores in evaluation.items():
                    rs=[]
                    for qi in qids:
                        selected=ranked(sd[name][qi],families,priorities(seed,metadata[qi]['key'],len(gallery)),5)
                        rs.append(dict(**metadata[qi],score=float(np.mean(scores[qi,selected]))))
                    for role in ('test','validation'):
                        value,by_family=macro([r for r in rs if r['evaluation_role']==role])
                        control_rows.append(dict(direction=direction,control='structure_shift',seed=seed,method=name,axis=axis,role=role,score=value,families=len(by_family)))
                arrays[f'{direction}_shift_{seed}_P']=shifted['P'];arrays[f'{direction}_shift_{seed}_H']=shifted['H']
                arrays[f'{direction}_shift_{seed}_distance_{name}']=sd[name]
            rng=np.random.default_rng(seed)
            balanced=np.array(sorted(j for group in sorted(work_groups) for j in rng.choice(work_groups[group],3,replace=False)))
            for name in dd:
                for axis,scores in evaluation.items():
                    rs=[]
                    for qi in qids:
                        ds=np.full(len(gallery),np.inf);ds[balanced]=dd[name][qi,balanced]
                        selected=ranked(ds,families,priorities(seed,metadata[qi]['key'],len(gallery)),5)
                        rs.append(dict(**metadata[qi],score=float(np.mean(scores[qi,selected]))))
                    for role in ('test','validation'):
                        value,by_family=macro([r for r in rs if r['evaluation_role']==role])
                        control_rows.append(dict(direction=direction,control='balanced_gallery_3',seed=seed,method=name,axis=axis,role=role,score=value,families=len(by_family)))
            arrays[f'{direction}_balanced_{seed}']=balanced
        print('Ranked',direction,flush=True)
    write(args.out/'scores.csv',score_rows);write(args.out/'rankings.csv',rank_rows);write(args.out/'controls.csv',control_rows)
    summary=[];family_rows=[]
    for role in ('test','validation'):
        for method in [*ARMS,'M-CT']:
            for axis in ('E-L','E-T','E-R','E-S'):
                selected=[r for r in score_rows if r['role']==role and r['method']==method and r['axis']==axis]
                value,by_family=macro(selected);sw,_=macro(selected,'same_work_expected');within,_=macro(selected,'within_work_selection')
                summary.append(dict(role=role,method=method,axis=axis,score=value,same_work_expected=sw,within_work_selection=within,query_families=len(by_family)))
                family_rows.extend(dict(role=role,method=method,axis=axis,family_id=f,score=v) for f,v in by_family.items())
    write(args.out/'summary.csv',summary);write(args.out/'per_family.csv',family_rows)
    dump(args.out/'scales.json',scale_records)
    np.savez_compressed(args.artifacts/'comparison.npz',**arrays)
    inputs=[args.source/'sequences.npz',args.tension/'tension.npz',source_report/'provenance.json',tonal_report/'provenance.json',source_report/'coverage.csv']
    provenance(args.out,inputs,[Path(__file__),ROOT/'classicfy-ai/src/embedding/structure_retrieval.py',ROOT/'classicfy-ai/src/embedding/retrieval_probe.py',ROOT/'classicfy-ai/src/features/structure_response.py'],dict(artifact_sha256=sha(args.artifacts/'comparison.npz')))
    print('Finished comparison',flush=True)


if __name__=='__main__':main()
