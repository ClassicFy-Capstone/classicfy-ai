"""Execute fixed Top K metadata/style retrieval and disjoint-view controls."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from embedding.retrieval_probe import (EPS,weights,weighted_quantile,fit_scale,distance,
    priorities,overlap,percentile_scores)
from audit_asap_metadata import read_csv,write_csv,sha256

META=('M-C','M-T','M-CT','Q','M-CTQ')
STYLE=('B','B+T','B+T+R',*[f'{m}_{s}' for m in ('cnn64','bilstm64') for s in (20261006,20261007,20261008)])
METHODS=(*META,*STYLE)


def dump(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def tiers(d):
    output=np.full(len(d),len(d),int);order=np.flatnonzero(np.isfinite(d))
    order=order[np.argsort(d[order],kind='stable')];i=0;group=0
    while i<len(order):
        end=i+1
        while end<len(order) and d[order[end]]-d[order[i]]<=EPS:end+=1
        output[order[i:end]]=group;i=end;group+=1
    return output


def select(tier,families,priority,k,cap=True):
    eligible=np.flatnonzero(tier<len(tier));order=eligible[np.lexsort((priority[eligible],tier[eligible]))]
    if cap:
        _,first=np.unique(families[order],return_index=True);order=order[np.sort(first)]
    return order[:k] if len(order)>=k else None


def make_distances(arrays,rows,gids,prefix):
    gallery=[rows[i] for i in gids];w=weights(gallery);scales={};blocks={}
    for name in ('Q','T','R'):
        x=arrays['whole_Q' if name=='Q' else f'{prefix}_{name}']
        scales[name]=fit_scale(x[gids],w);blocks[name]=distance(x,x[gids],scales[name])
    x=arrays[f'{prefix}_B'];blocks['B']=distance(x,x[gids],group_width=2)
    train_families=np.array([r['family_id'] for r in gallery]);left,right=np.triu_indices(len(gids),1)
    valid=train_families[left]!=train_families[right];left,right=left[valid],right[valid]
    pw=w[left]*w[right];calibration={}
    for name in ('Q','T','R','B'):
        calibration[name]=float(weighted_quantile(blocks[name][gids[left],right],pw,.5))
        if calibration[name]<=EPS:raise ValueError('Zero calibration '+name)
    all_composer=np.array([r['composer'] for r in rows]);all_type=np.array([r['work_type'] for r in rows])
    d={'M-C':(all_composer[:,None]!=all_composer[gids][None,:]).astype(float),
       'M-T':(all_type[:,None]!=all_type[gids][None,:]).astype(float)}
    d['M-CT']=(d['M-C']+d['M-T'])/2
    calibration['M-CT']=float(weighted_quantile(d['M-CT'][gids[left],right],pw,.5))
    if calibration['M-CT']<=EPS:raise ValueError('Zero metadata calibration')
    d['Q']=blocks['Q'];d['M-CTQ']=np.sqrt(((d['M-CT']/calibration['M-CT'])**2+(d['Q']/calibration['Q'])**2)/2)
    d['B']=blocks['B']
    for name,components in [('B+T',('B','T')),('B+T+R',('B','T','R'))]:
        d[name]=np.sqrt(np.mean([(blocks[b]/calibration[b])**2 for b in components],axis=0))
    for name in STYLE[3:]:
        x=arrays[f'{prefix}_{name}']; norms=np.linalg.norm(x,axis=1)
        with np.errstate(invalid='ignore',divide='ignore'):
            d[name]=np.maximum(1-(x@x[gids].T)/(norms[:,None]*norms[gids][None,:]),0)
    return d,{'coordinate_scales':scales,'block_medians':calibration}


def grouped_indices(rows):
    groups=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(rows):groups[r['family_id']][r['piece'],r['grid']].append(i)
    return groups


def uniform_list(groups,rng,k):
    families=sorted(groups);selected=rng.choice(len(families),k,replace=False);result=[]
    for f in selected:
        works=list(groups[families[f]].values());work=works[int(rng.integers(len(works)))];result.append(rng.choice(work))
    return np.asarray(result,int)


def conditional_sources(rows,q,w,seeds):
    cuts=[float(weighted_quantile(q[:,0],w,p)) for p in (.25,.5,.75)]
    length=np.searchsorted(cuts,q[:,0],side='right');coverage=np.searchsorted([.8,.95],q[:,1],side='right')
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['composer'],r['work_type'],int(length[i]),int(coverage[i])].append(i)
    source=[]
    for seed in seeds:
        indices=np.arange(len(rows));rng=np.random.default_rng(seed)
        for ids in groups.values():indices[ids]=rng.permutation(ids)
        source.append(indices)
    exchange=sum(len(v) for v in groups.values() if len(v)>1)/len(rows)
    same_piece=sum(len(v) for v in groups.values() if len(v)>1 and len({rows[i]['piece'] for i in v})==1)/len(rows)
    return np.asarray(source),dict(length_cuts=cuts,exchangeable_fraction=exchange,same_piece_only_fraction=same_piece,
        status='ok' if exchange>=.8 else 'insufficient_exchange',strata=len(groups))


def metric_record(rows,q,experiment,view,method,k,cap,**values):
    return dict(experiment=experiment,view=view,role=rows[q]['evaluation_role'],query=rows[q]['key'],
        piece=rows[q]['piece'],family_id=rows[q]['family_id'],method=method,k=k,cap=cap,**values)


def aggregate(records,fields):
    # Seed/view repetitions first, then works, then query families.
    groups=defaultdict(list)
    for r in records:groups[tuple(r[f] for f in fields)].append(r)
    result=[]
    for key,values in groups.items():
        queries=defaultdict(list)
        for r in values:
            if np.isfinite(r['value']):queries[r['query']].append(r)
        by_work=defaultdict(list);query_rows=[]
        for query,rs in queries.items():
            value=float(np.mean([r['value'] for r in rs]));by_work[rs[0]['family_id'],rs[0]['piece']].append(value)
            query_rows.append(dict(query=query,piece=rs[0]['piece'],family_id=rs[0]['family_id'],value=value))
        by_family=defaultdict(list)
        for (family,piece),vs in by_work.items():by_family[family].append(np.mean(vs))
        scores=[np.mean(v) for v in by_family.values()]
        result.append(dict(zip(fields,key))|dict(value=float(np.mean(scores)) if scores else np.nan,
            queries=len(query_rows),works=len(by_work),families=len(by_family)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/metadata_retrieval_results')
    parser.add_argument('--cache',type=Path,default=ROOT.parent/'datasets/metadata_retrieval_run01')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    design=ROOT/'classicfy-ai/analysis/metadata_retrieval_design';protocol=json.loads((design/'protocol.json').read_text())
    preflight=json.loads((args.out/'preflight.json').read_text());meta=json.loads((args.cache/'prepared.json').read_text())
    if sha256(args.cache/'prepared.npz')!=preflight['prepared_archive_sha256']:raise ValueError('Prepared inputs changed')
    if sha256(design/'protocol.json')!=meta['signature']['protocol']:raise ValueError('Fixed protocol changed')
    if sha256(ROOT/'classicfy-ai/src/embedding/retrieval_probe.py')!=meta['signature']['core']:raise ValueError('Preparation core changed')
    arrays=dict(np.load(args.cache/'prepared.npz',allow_pickle=False))
    rows=read_csv(ROOT/protocol['source_files']['role_manifest']);index={r['key']:i for i,r in enumerate(rows)}
    contexts=[];scales={}
    for experiment,prefixes in [('whole',('whole',)),('split',('A','B'))]:
        chosen=[r for r in meta['cohort'] if r['experiment']==experiment and r['included']]
        gids=np.array([index[r['key']] for r in chosen if r['evaluation_role']=='train'])
        qids=np.array([index[r['key']] for r in chosen if r['evaluation_role'] in ('validation','test')])
        for prefix in prefixes:
            d,scale=make_distances(arrays,rows,gids,prefix)
            judgment={};jscales={}
            if experiment=='split':
                other='B' if prefix=='A' else 'A'
                for judge,width in [('E-L',2),('E-T',3),('E-R',None)]:
                    x=arrays[f'{other}_{judge}'];s=fit_scale(x[gids],weights([rows[i] for i in gids]));jscales[judge]=s
                    judgment[judge]=distance(x[qids],x[gids],s,width)
            scales[f'{experiment}_{prefix}']=dict(search=scale,judge=jscales)
            contexts.append((experiment,prefix,gids,qids,d,judgment))
    dump(args.out/'scales.json',scales)
    preflight['scales_sha256']=sha256(args.out/'scales.json');preflight['status']='cohort_partitions_and_scales_frozen_before_ranking'
    dump(args.out/'preflight.json',preflight)
    outputs={name:[] for name in ('overlap','metadata_exposure','style_scores','controls','novel_scores','neighbors','balanced_gallery')}
    archive={};control_support=[];availability=[]
    tie_seeds=protocol['retrieval']['tie_seeds'];control_seeds=protocol['experiment_B']['controls']['seeds']
    ks=[protocol['retrieval']['primary_k'],*protocol['retrieval']['secondary_k']];maxk=max(ks)
    for experiment,view,gids,qids,d,judgment in contexts:
        print('Retrieving',experiment,view,'queries',len(qids),'gallery',len(gids),flush=True)
        gallery=[rows[i] for i in gids];families=np.array([r['family_id'] for r in gallery]);groups=grouped_indices(gallery)
        gallery_w=weights(gallery);n=len(gids);nq=len(qids);tag=f'{experiment}_{view}'
        sources,support=conditional_sources(gallery,arrays['whole_Q'][gids],gallery_w,control_seeds)
        control_support.append(dict(experiment=experiment,view=view,**support));archive[tag+'_conditional_sources']=sources
        archive[tag+'_gallery_ids']=gids;archive[tag+'_query_ids']=qids
        scores={name:percentile_scores(value,gallery_w) for name,value in judgment.items()}
        for name,value in judgment.items():archive[f'{tag}_judge_distance_{name}']=value;archive[f'{tag}_judge_score_{name}']=scores[name]
        local_tiers={m:np.array([tiers(d[m][q]) for q in qids]) for m in METHODS}
        selections={cap:{m:np.full((len(tie_seeds),nq,maxk),-1,int) for m in METHODS} for cap in (True,False)}
        conditional={m:np.full((len(tie_seeds),nq,maxk),-1,int) for m in STYLE}
        uniforms=np.full((len(tie_seeds),nq,maxk),-1,int)
        piece_indices=defaultdict(list)
        for i,r in enumerate(gallery):piece_indices[r['piece'],r['grid']].append(i)
        samework={m:np.full((len(tie_seeds),nq,maxk),-1,int) for m in METHODS}
        for qi,q in enumerate(qids):
            forbidden=families==rows[q]['family_id']
            for m in METHODS:
                if forbidden.any():local_tiers[m][qi,forbidden]=n
            remaining={f:works for f,works in groups.items() if f!=rows[q]['family_id']}
            availability.append(dict(experiment=experiment,view=view,query=rows[q]['key'],families=len(remaining),recordings=int((~forbidden).sum()),k10_supported=len(remaining)>=maxk))
            for si,(seed,control_seed) in enumerate(zip(tie_seeds,control_seeds,strict=True)):
                priority=priorities(seed,rows[q]['key'],len(rows))[gids]
                for cap in (True,False):
                    for m in METHODS:
                        chosen=select(local_tiers[m][qi],families,priority,maxk,cap)
                        if chosen is not None:selections[cap][m][si,qi]=chosen
                rng=np.random.default_rng(int(control_seed)+int(q)*10000)
                if len(remaining)>=maxk:uniforms[si,qi]=uniform_list(remaining,rng,maxk)
                for m in METHODS:
                    chosen=selections[True][m][si,qi]
                    if chosen[0]>=0:
                        samework[m][si,qi]=[rng.choice(piece_indices[gallery[c]['piece'],gallery[c]['grid']]) for c in chosen]
                for m in STYLE:
                    remapped=local_tiers[m][qi,sources[si]].copy();remapped[forbidden]=n
                    chosen=select(remapped,families,priority,maxk,True)
                    if chosen is not None:conditional[m][si,qi]=chosen
        for cap in (True,False):
            for m in METHODS:archive[f'{tag}_rank_{int(cap)}_{m}']=selections[cap][m]
        for m in METHODS:
            archive[f'{tag}_distance_{m}']=d[m][qids];archive[f'{tag}_same_work_{m}']=samework[m]
        for m in STYLE:archive[f'{tag}_conditional_rank_{m}']=conditional[m]
        archive[tag+'_uniform_rank']=uniforms
        for qi,q in enumerate(qids):
            for k in ks:
                for cap in (True,False):
                    selected=selections[cap]
                    for m in STYLE:
                        a=selected[m][:,qi,:k]
                        if (a<0).any():continue
                        for reference in META:
                            b=selected[reference][:,qi,:k];values=[overlap(x,y,families) for x,y in zip(a,b,strict=True)]
                            for metric in values[0]:
                                v=np.array([r[metric] for r in values])
                                outputs['overlap'].append(metric_record(rows,q,experiment,view,m,k,cap,reference=reference,metric=metric,
                                    value=float(v.mean()),seed_p05=float(np.percentile(v,5)),seed_p95=float(np.percentile(v,95))))
                    for m in META:
                        a=selected[m][:,qi,:k]
                        if (a<0).any():continue
                        v=[overlap(a[i],a[(i+1)%len(a)],families) for i in range(len(a))]
                        for metric in v[0]:
                            z=np.array([r[metric] for r in v])
                            outputs['overlap'].append(metric_record(rows,q,experiment,view,m,k,cap,reference='metadata_internal_ties',metric=metric,
                                value=float(z.mean()),seed_p05=float(np.percentile(z,5)),seed_p95=float(np.percentile(z,95))))
                    for m in METHODS:
                        a=selected[m][:,qi,:k]
                        if (a<0).any():continue
                        composer=np.array([gallery[i]['composer']==rows[q]['composer'] for i in a.ravel()])
                        kind=np.array([gallery[i]['work_type']==rows[q]['work_type'] for i in a.ravel()])
                        for metric,value in [('composer_match',composer.mean()),('type_match',kind.mean()),('joint_match',(composer&kind).mean()),
                            ('log_beat_gap',np.mean(abs(arrays['whole_Q'][gids[a],0]-arrays['whole_Q'][q,0]))),
                            ('coverage_gap',np.mean(abs(arrays['whole_Q'][gids[a],1]-arrays['whole_Q'][q,1]))),
                            ('returned_work_recordings',np.mean([len(piece_indices[gallery[i]['piece'],gallery[i]['grid']]) for i in a.ravel()]))]:
                            outputs['metadata_exposure'].append(metric_record(rows,q,experiment,view,m,k,cap,metric=metric,value=float(value)))
                        if experiment=='split':
                            for judge,s in scores.items():
                                values=s[qi,a];outputs['style_scores'].append(metric_record(rows,q,experiment,view,m,k,cap,judge=judge,value=float(values.mean())))
                                if cap:
                                    sw=s[qi,samework[m][:,qi,:k]]
                                    outputs['controls'].append(metric_record(rows,q,experiment,view,m,k,cap,judge=judge,control='same_work',value=float(sw.mean()),seed_p05=float(np.percentile(sw.mean(axis=1),5)),seed_p95=float(np.percentile(sw.mean(axis=1),95))))
                                    wins=[]
                                    for i in a.ravel():
                                        other=[j for j in piece_indices[gallery[i]['piece'],gallery[i]['grid']] if j!=i]
                                        diff=judgment[judge][qi,other]-judgment[judge][qi,i]
                                        wins.append(float(np.mean((diff>EPS)+.5*(abs(diff)<=EPS))))
                                    outputs['controls'].append(metric_record(rows,q,experiment,view,m,k,cap,judge=judge,control='within_work_win',value=float(np.mean(wins)),seed_p05=np.nan,seed_p95=np.nan))
                                    if m in STYLE:
                                        shuffled=s[qi,conditional[m][:,qi,:k]]
                                        outputs['controls'].append(metric_record(rows,q,experiment,view,m,k,cap,judge=judge,control='conditional_shuffle',value=float(shuffled.mean()) if support['status']=='ok' else np.nan,
                                            seed_p05=float(np.percentile(shuffled.mean(axis=1),5)),seed_p95=float(np.percentile(shuffled.mean(axis=1),95))))
                                        novel=[];counts=[]
                                        meta_selected=selected['M-CT'][:,qi,:k]
                                        for aa,bb in zip(a,meta_selected,strict=True):
                                            nn=[i for i in aa if i not in bb];counts.append(len(nn))
                                            if len(nn)>=3:novel.append(np.mean(s[qi,nn]))
                                        outputs['novel_scores'].append(metric_record(rows,q,experiment,view,m,k,cap,judge=judge,value=float(np.mean(novel)) if novel else np.nan,
                                            supported_replicates=len(novel),mean_novel_count=float(np.mean(counts))))
                if experiment=='split':
                    for judge,s in scores.items():
                        values=s[qi,uniforms[:,qi,:k]]
                        outputs['controls'].append(metric_record(rows,q,experiment,view,'uniform',k,True,judge=judge,control='uniform',value=float(values.mean()),seed_p05=float(np.percentile(values.mean(axis=1),5)),seed_p95=float(np.percentile(values.mean(axis=1),95))))
            # Human-readable first-seed lists; all seeds archived as exact indices.
            for m in METHODS:
                for rank,i in enumerate(selections[True][m][0,qi],1):
                    if i<0:continue
                    c=gids[i];outputs['neighbors'].append(metric_record(rows,q,experiment,view,m,maxk,True,tie_seed=tie_seeds[0],rank=rank,
                        candidate=rows[c]['key'],candidate_piece=rows[c]['piece'],candidate_family=rows[c]['family_id'],distance=float(d[m][q,i])))
        # Equal-size candidate corpus sensitivity; repeat all tie seeds.
        for seed in protocol['robustness']['gallery_seeds']:
            rng=np.random.default_rng(seed);subset=[]
            for family,works in sorted(groups.items()):
                available=[i for ids in works.values() for i in ids]
                if len(available)<3:continue
                picked=[]
                while len(picked)<3:
                    ws=[v for v in works.values() if any(i not in picked for i in v)]
                    ids=ws[int(rng.integers(len(ws)))];ids=[i for i in ids if i not in picked];picked.append(int(rng.choice(ids)))
                subset.extend(picked)
            subset=np.asarray(subset);archive[f'{tag}_balanced_ids_{seed}']=subset
            for m in METHODS:
                balanced=np.full((len(tie_seeds),nq,maxk),-1,int)
                for qi,q in enumerate(qids):
                    dd=d[m][q,subset].copy();dd[families[subset]==rows[q]['family_id']]=np.inf;tt=tiers(dd)
                    for si,tie_seed in enumerate(tie_seeds):
                        pp=priorities(tie_seed,rows[q]['key'],len(rows))[gids[subset]]
                        selected=select(tt,families[subset],pp,maxk)
                        if selected is not None:balanced[si,qi]=subset[selected]
                    if experiment=='split':
                        # Scores retain the original gallery reference to compare identical scale.
                        for k in ks:
                            if (balanced[:,qi,:k]<0).any():continue
                            for judge,s in scores.items():outputs['balanced_gallery'].append(metric_record(rows,q,experiment,view,m,k,True,
                                gallery_seed=seed,judge=judge,value=float(s[qi,balanced[:,qi,:k]].mean())))
                archive[f'{tag}_balanced_rank_{seed}_{m}']=balanced
        print('Completed',tag,flush=True)
    np.savez_compressed(args.cache/'retrieval_arrays.npz',**archive)
    for name,records in outputs.items():
        if records:write_csv((args.cache if name=='neighbors' else args.out)/(name+'.csv'),records)
    write_csv(args.out/'query_availability.csv',availability);write_csv(args.out/'control_support.csv',control_support)
    # Save the lexical first test query per experiment for review, not selected by success.
    example=[]
    for experiment in ('whole','split'):
        test=[r for r in outputs['neighbors'] if r['experiment']==experiment and r['role']=='test']
        if test:
            query=sorted(test,key=lambda r:(r['piece'],r['query']))[0]['query'];example.extend(r for r in test if r['query']==query)
    if example:write_csv(args.out/'neighbors.csv',example)
    metrics=[]
    for name in ('overlap','metadata_exposure','style_scores','controls','novel_scores','balanced_gallery'):
        if not outputs[name]:continue
        fields=['experiment','role','method','k','cap']+[f for f in ('reference','metric','judge','control') if f in outputs[name][0]]
        metrics.extend(dict(table=name,**r) for r in aggregate(outputs[name],fields))
    # Heterogeneous metrics use a fixed schema with empty fields.
    all_fields=['table','experiment','role','method','k','cap','reference','metric','judge','control','value','queries','works','families']
    write_csv(args.out/'summary.csv',[{f:r.get(f,'') for f in all_fields} for r in metrics])
    per_work=[];bootstrap=[]
    for role in ('validation','test'):
        for contrast in protocol['primary_contrasts']:
            a,b,judge=contrast['candidate'],contrast['reference'],contrast['judge']
            relevant=[r for r in outputs['style_scores'] if r['role']==role and r['cap'] and r['k']==5 and r['judge']==judge and r['method'] in (a,b)]
            groups=defaultdict(lambda:defaultdict(list))
            for r in relevant:groups[r['family_id'],r['piece']][r['method']].append(r['value'])
            family=defaultdict(list)
            for (f,p),vs in groups.items():
                aa,bb=np.mean(vs[a]),np.mean(vs[b]);family[f].append(aa-bb)
                per_work.append(dict(role=role,candidate=a,reference=b,judge=judge,piece=p,family_id=f,candidate_score=float(aa),reference_score=float(bb),delta=float(aa-bb)))
            values=np.array([np.mean(v) for v in family.values()]);status='ok' if len(values)>=5 else 'fewer_than_five_families'
            lo=hi=np.nan
            if status=='ok':
                rng=np.random.default_rng(protocol['uncertainty']['seed']);draws=values[rng.integers(len(values),size=(2000,len(values)))].mean(axis=1)
                tail=(100-protocol['uncertainty']['primary_interval_percent'])/2;lo,hi=np.percentile(draws,[tail,100-tail])
            bootstrap.append(dict(role=role,candidate=a,reference=b,judge=judge,k=5,delta=float(values.mean()) if len(values) else np.nan,
                improved_works=sum(r['delta']>0 for r in per_work if r['role']==role and r['candidate']==a),families=len(values),ci_low=float(lo),ci_high=float(hi),interval_status=status))
    if per_work:write_csv(args.out/'per_work_metrics.csv',per_work)
    if bootstrap:write_csv(args.out/'bootstrap.csv',bootstrap)
    csv_files=sorted(args.out.glob('*.csv'))
    dump(args.out/'execution.json',dict(status='completed_fixed_retrieval_design',runtime_seconds=time.monotonic()-started,
        protocol_sha256=sha256(design/'protocol.json'),preflight_sha256=sha256(args.out/'preflight.json'),
        prepared_sha256=sha256(args.cache/'prepared.npz'),arrays_sha256=sha256(args.cache/'retrieval_arrays.npz'),
        runner_sha256=sha256(__file__),core_sha256=sha256(ROOT/'classicfy-ai/src/embedding/retrieval_probe.py'),
        csv_sha256={p.name:sha256(p) for p in csv_files},full_first_seed_neighbors_sha256=sha256(args.cache/'neighbors.csv'),
        replicates='100 independent paired tie/control replicates; not a 100x100 Cartesian product',
        archive='All method/model seeds, tie/control indices, distances and balanced galleries are archived outside Git',
        unsupported_intervals='Disjoint-view test and validation each contain only four query families; no bootstrap interval reported'))
    print('Finished in',round(time.monotonic()-started,2),'seconds',flush=True)


if __name__=='__main__':main()
