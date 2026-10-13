"""Independently check original-source descriptors, ranking and score exports."""

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from embedding.retrieval_probe import ranked,priorities,CHANNELS,LAGS
from audit_asap_metadata import read_csv,sha256
from prepare_metadata_retrieval_experiment import reconstruct


def independent_weights(rows):
    families=sorted({r['family_id'] for r in rows});w=[]
    for r in rows:
        works={(s['piece'],s['grid']) for s in rows if s['family_id']==r['family_id']}
        n=sum(s['piece']==r['piece'] and s['grid']==r['grid'] for s in rows)
        w.append(1/len(families)/len(works)/n)
    return np.array(w)


def independent_scores(d,w):
    result=[]
    for values in d:
        order=np.argsort(values);sorted_values=values[order];cum=np.r_[0,np.cumsum(w[order])]
        lo=np.searchsorted(sorted_values,values-1e-12,side='left');hi=np.searchsorted(sorted_values,values+1e-12,side='right')
        result.append(1-cum[hi]+.5*(cum[hi]-cum[lo]))
    return np.array(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/metadata_retrieval_results')
    parser.add_argument('--cache',type=Path,default=ROOT.parent/'datasets/metadata_retrieval_run01')
    args=parser.parse_args();execution=json.loads((args.out/'execution.json').read_text())
    def table_path(name):
        direct=args.out/name
        return direct if direct.exists() else args.cache/'tables'/name
    for name,digest in execution['csv_sha256'].items():
        assert sha256(table_path(name))==digest,name
    assert sha256(args.cache/'retrieval_arrays.npz')==execution['arrays_sha256']
    z=np.load(args.cache/'retrieval_arrays.npz');prepared=np.load(args.cache/'prepared.npz')
    legacy=json.loads((ROOT/'classicfy-ai/analysis/metadata_information_design/protocol.json').read_text())
    rows=read_csv(ROOT/'classicfy-ai/analysis/metadata_information_design/target_preflight.csv')
    protocol=json.loads((ROOT/'classicfy-ai/analysis/metadata_retrieval_design/protocol.json').read_text())
    checks=dict(independent_view_descriptor_rows=0,independent_judge_rows=0,ranking_checks=0,
        neighbor_rows_validated=0,score_rows=0,control_rows=0,maximum_numeric_error=0.)
    def close(a,b):
        a,b=np.asarray(a),np.asarray(b)
        assert np.array_equal(np.isfinite(a),np.isfinite(b))
        if np.isfinite(a).any():
            error=float(np.max(abs(a[np.isfinite(a)]-b[np.isfinite(b)])))
            checks['maximum_numeric_error']=max(checks['maximum_numeric_error'],error)
            assert error<1e-10,error
    # Use the original #14 implementations rather than the new descriptor code.
    partitions=pd.read_csv(args.out/'partition_manifest.csv').set_index(['key','view'])
    thresholds=json.loads(subprocess.check_output(['git','show',f"{legacy['hand_source_commit']}:classicfy-ai/analysis/temporal_shape/protocol.json"],cwd=ROOT))
    cuts={r['channel']:r['value'] for r in thresholds['thresholds']}
    for sequences,snapshot in reconstruct(ROOT.parent/'datasets',legacy):
        for name in ('temporal_shape','temporal_duration','temporal_coupling'):
            (snapshot/f'classicfy-ai/src/features/{name}.py').write_bytes(subprocess.check_output(['git','show',f"{legacy['hand_source_commit']}:classicfy-ai/src/features/{name}.py"],cwd=ROOT))
        import features
        features.__path__.insert(0,str(snapshot/'classicfy-ai/src/features'))
        importlib.invalidate_caches()
        from features.models import BeatSequence
        from features.temporal_shape import adjacent_changes,summarize_temporal_shape
        from features.temporal_duration import summarize_temporal_duration,lagged_correlation
        from features.temporal_coupling import cross_channel_correlation
        pairs=sorted([(a,b) for a in range(7) for b in range(a+1,7) if (a,b)!=(0,1) and not a>=4],key=lambda p:(CHANNELS[p[0]],CHANNELS[p[1]]))
        for i,row in enumerate(rows):
            x,m=sequences[row['key']]
            for view in ('A','B'):
                cell=partitions.loc[row['key'],view];starts=[] if pd.isna(cell.starts) else [int(a) for a in str(cell.starts).split('|')]
                valid=np.zeros(len(m),bool)
                for start in starts:valid[start:start+64]=m[start:start+64]
                full=[];deltas=[];level=[];temporal=[];channels=[]
                for c,name in enumerate(CHANNELS):
                    seq=BeatSequence(x[c],valid);channels.append(seq);delta=adjacent_changes(seq);deltas.append(delta)
                    shape=summarize_temporal_shape(seq,large_threshold=cuts[name]);duration=summarize_temporal_duration(seq)
                    v=x[c,valid];width=np.ptp(np.percentile(v,[5,95])) if len(v) else np.nan
                    full += [shape['large_ratio'],shape['rms_delta']/width if width>1e-12 else np.nan,shape['rise_ratio'],shape['reversal_ratio']]
                    full += [np.log1p(duration[f'level_{s}_run_mean']) if duration[f'level_{s}_run_count']>=20 else np.nan for s in ('positive','negative')]
                    for source in (seq,delta):
                        for lag in (1,8):full.append(lagged_correlation(source,lag)['correlation'])
                    level.extend([np.median(abs(v)) if c==1 else np.median(v),np.ptp(np.percentile(v,[25,75]))] if len(v)>=100 else [np.nan]*2)
                    rms=[]
                    for start in starts:
                        for a in (start,start+32):
                            dd=[x[c,t+1]-x[c,t] for t in range(a,a+31) if valid[t] and valid[t+1]]
                            if len(dd)>=25:rms.append(np.sqrt(np.mean(np.square(dd))))
                    temporal.extend(np.log1p(np.percentile(rms,[25,50,75])) if len(rms)>=3 else [np.nan]*3)
                relations=[cross_channel_correlation(deltas[a],deltas[b],lag)['correlation'] for a,b in pairs for lag in LAGS]
                expected_t=np.array(full)[legacy['retained_descriptor_coordinates']['single']];expected_r=np.asarray(relations)
                if len(starts)<2:expected_t[:]=np.nan;expected_r[:]=np.nan
                close(expected_t,prepared[f'{view}_T'][i]);close(expected_r,prepared[f'{view}_R'][i]);checks['independent_view_descriptor_rows']+=1
                rs=[]
                for a in starts:
                    pairs_valid=[t for t in range(a,a+63) if valid[t] and valid[t+1]]
                    u=np.array([x[0,t+1]-x[0,t] for t in pairs_valid]);v=np.array([x[2,t+1]-x[2,t] for t in pairs_valid])
                    if len(u)>=51 and min(np.std(u),np.std(v))>1e-12:rs.append(np.corrcoef(u,v)[0,1])
                jr=np.r_[np.percentile(rs,[25,50,75]),np.std(rs)] if len(rs)>=2 else [np.nan]*4
                for name,value in [('E-L',level),('E-T',temporal),('E-R',jr)]:close(value,prepared[f'{view}_{name}'][i]);checks['independent_judge_rows']+=1
        print('Original #14 and explicit block judges independently verified',flush=True)
    score_maps={};query_lookup={};piece_maps={};win_maps={}
    for tag in ('whole_whole','split_A','split_B'):
        gids=z[tag+'_gallery_ids'];qids=z[tag+'_query_ids'];gallery=[rows[i] for i in gids];families=np.array([r['family_id'] for r in gallery])
        piece_maps[tag]={(r['piece'],r['grid']):np.array([j for j,s in enumerate(gallery) if s['piece']==r['piece'] and s['grid']==r['grid']]) for r in gallery}
        w=independent_weights(gallery);query_lookup[tag]={rows[q]['key']:i for i,q in enumerate(qids)}
        for judge in ('E-L','E-T','E-R'):
            if f'{tag}_judge_distance_{judge}' in z.files:
                s=independent_scores(z[f'{tag}_judge_distance_{judge}'],w);close(s,z[f'{tag}_judge_score_{judge}']);score_maps[tag,judge]=s
        rank_keys=[name for name in z.files if name.startswith(tag+'_rank_')]
        for key in rank_keys:
            cap=key.startswith(tag+'_rank_1_');method=key.split('_rank_',1)[1][2:];ranks=z[key];d=z[f'{tag}_distance_{method}']
            assert np.all((ranks>=0)&(ranks<len(gids)))
            for qi,q in enumerate(qids):
                assert not np.any(families[ranks[:,qi]]==rows[q]['family_id'])
                if cap:assert all(len(set(families[indices]))==10 for indices in ranks[:,qi])
                for si in list(range(0,100,10))+[99]:
                    pp=priorities(protocol['retrieval']['tie_seeds'][si],rows[q]['key'],len(rows))[gids]
                    dd=d[qi].copy();dd[families==rows[q]['family_id']]=np.inf
                    expected=ranked(dd,families,pp,10,cap);np.testing.assert_equal(expected,ranks[si,qi]);checks['ranking_checks']+=1
            checks['neighbor_rows_validated']+=ranks.size
        # Independent ranking of conditional permutations, preserving candidate keys.
        source=z[tag+'_conditional_sources']
        for method in ('B','B+T','B+T+R'):
            ranks=z[f'{tag}_conditional_rank_{method}'];d=z[f'{tag}_distance_{method}']
            for qi,q in enumerate(qids):
                for si in (0,10,50,99):
                    pp=priorities(protocol['retrieval']['tie_seeds'][si],rows[q]['key'],len(rows))[gids]
                    dd=d[qi,source[si]].copy();dd[families==rows[q]['family_id']]=np.inf
                    np.testing.assert_equal(ranked(dd,families,pp,10),ranks[si,qi]);checks['ranking_checks']+=1
        print('Independent ranking checked',tag,flush=True)
    for row in read_csv(table_path('style_scores.csv')):
        tag=f"{row['experiment']}_{row['view']}";qi=query_lookup[tag][row['query']];k=int(row['k']);cap=int(row['cap']=='True')
        ranks=z[f"{tag}_rank_{cap}_{row['method']}"][:,qi,:k]
        close(np.mean(score_maps[tag,row['judge']][qi,ranks]),float(row['value']));checks['score_rows']+=1
    support={(r['experiment'],r['view']):r['status'] for r in read_csv(args.out/'control_support.csv')}
    for row in read_csv(table_path('controls.csv')):
        tag=f"{row['experiment']}_{row['view']}";qi=query_lookup[tag][row['query']];k=int(row['k']);s=score_maps[tag,row['judge']]
        if row['control']=='uniform':ranks=z[tag+'_uniform_rank'][:,qi,:k];expected=np.mean(s[qi,ranks])
        elif row['control']=='same_work':ranks=z[f"{tag}_same_work_{row['method']}"][:,qi,:k];expected=np.mean(s[qi,ranks])
        elif row['control']=='conditional_shuffle':
            ranks=z[f"{tag}_conditional_rank_{row['method']}"][:,qi,:k]
            expected=np.mean(s[qi,ranks]) if support[row['experiment'],row['view']]=='ok' else np.nan
        else:
            ranks=z[f"{tag}_rank_1_{row['method']}"][:,qi,:k];gids=z[tag+'_gallery_ids'];d=z[f"{tag}_judge_distance_{row['judge']}"][qi]
            cache_key=(tag,qi,row['judge'])
            if cache_key not in win_maps:
                wins=np.zeros(len(gids))
                for c,g in enumerate(gids):
                    r=rows[g];ids=piece_maps[tag][r['piece'],r['grid']];others=ids[ids!=c]
                    dif=d[others]-d[c];wins[c]=np.mean((dif>1e-12)+.5*(abs(dif)<=1e-12))
                win_maps[cache_key]=wins
            expected=np.mean(win_maps[cache_key][ranks])
        close(expected,float(row['value']));checks['control_rows']+=1
    # Explicit set/rank definitions on a fixed sample of overlap rows.
    overlap_rows=read_csv(table_path('overlap.csv'))
    checks['independent_overlap_rows']=0
    for row in overlap_rows[::37]:
        tag=f"{row['experiment']}_{row['view']}";qi=query_lookup[tag][row['query']];k=int(row['k']);cap=int(row['cap']=='True')
        a=z[f"{tag}_rank_{cap}_{row['method']}"][:,qi,:k]
        b=np.roll(a,-1,axis=0) if row['reference']=='metadata_internal_ties' else z[f"{tag}_rank_{cap}_{row['reference']}"][:,qi,:k]
        families=np.array([rows[g]['family_id'] for g in z[tag+'_gallery_ids']]);values=[]
        for aa,bb in zip(a,b):
            shared=set(aa)&set(bb)
            if row['metric']=='recording_overlap':value=len(shared)/k
            elif row['metric']=='family_overlap':value=len(set(families[aa])&set(families[bb]))/k
            else:
                value=sum(min(1/(aa.tolist().index(c)+1),1/(bb.tolist().index(c)+1)) for c in shared)/sum(1/i for i in range(1,k+1))
            values.append(value)
        close(np.mean(values),float(row['value']));close(np.percentile(values,[5,95]),[float(row['seed_p05']),float(row['seed_p95'])]);checks['independent_overlap_rows']+=1
    checks['novel_score_rows']=0
    for row in read_csv(args.out/'novel_scores.csv'):
        tag=f"{row['experiment']}_{row['view']}";qi=query_lookup[tag][row['query']];k=int(row['k'])
        a=z[f"{tag}_rank_1_{row['method']}"][:,qi,:k];b=z[f'{tag}_rank_1_M-CT'][:,qi,:k];values=[];counts=[]
        for aa,bb in zip(a,b):
            selected=[c for c in aa if c not in set(bb)];counts.append(len(selected))
            if len(selected)>=3:values.append(np.mean(score_maps[tag,row['judge']][qi,selected]))
        close(np.mean(values) if values else np.nan,float(row['value']))
        assert len(values)==int(row['supported_replicates']);close(np.mean(counts),float(row['mean_novel_count']));checks['novel_score_rows']+=1
    checks['balanced_ranking_checks']=0
    for tag in ('whole_whole','split_A','split_B'):
        gids=z[tag+'_gallery_ids'];qids=z[tag+'_query_ids'];families=np.array([rows[g]['family_id'] for g in gids])
        for seed in protocol['robustness']['gallery_seeds']:
            subset=z[f'{tag}_balanced_ids_{seed}'];unique,counts=np.unique(families[subset],return_counts=True);assert np.all(counts==3)
            for method in ('M-CT','B','B+T','B+T+R'):
                ranks=z[f'{tag}_balanced_rank_{seed}_{method}'];d=z[f'{tag}_distance_{method}']
                for qi,q in enumerate(qids):
                    pp=priorities(protocol['retrieval']['tie_seeds'][0],rows[q]['key'],len(rows))[gids[subset]]
                    dd=d[qi,subset].copy();dd[families[subset]==rows[q]['family_id']]=np.inf
                    np.testing.assert_equal(subset[ranked(dd,families[subset],pp,10)],ranks[0,qi]);checks['balanced_ranking_checks']+=1
    checks['balanced_score_rows']=0
    for row in read_csv(table_path('balanced_gallery.csv')):
        tag=f"{row['experiment']}_{row['view']}";qi=query_lookup[tag][row['query']];k=int(row['k'])
        ranks=z[f"{tag}_balanced_rank_{row['gallery_seed']}_{row['method']}"][:,qi,:k]
        close(np.mean(score_maps[tag,row['judge']][qi,ranks]),float(row['value']));checks['balanced_score_rows']+=1
    # Macro summaries are recomputed by query -> work -> family, without runner helpers.
    checks['macro_summary_rows']=0
    summary=pd.read_csv(args.out/'summary.csv').fillna('')
    for table,part in summary.groupby('table'):
        source=pd.read_csv(table_path(f'{table}.csv'));keys=['experiment','role','method','k','cap']+[f for f in ('reference','metric','judge','control') if f in source.columns]
        query=source.groupby(keys+['query','piece','family_id'],dropna=False).value.mean().reset_index()
        work=query.groupby(keys+['piece','family_id'],dropna=False).value.mean().reset_index()
        family=work.groupby(keys+['family_id'],dropna=False).value.mean().reset_index()
        macro=family.groupby(keys,dropna=False).value.mean()
        for _,row in part.iterrows():
            key=tuple(row[f] for f in keys);value=macro.loc[key]
            close(value,float(row['value']) if row['value']!='' else np.nan);checks['macro_summary_rows']+=1
    checks.update(status='passed',execution_sha256=sha256(args.out/'execution.json'),verifier_sha256=sha256(__file__),
        ranking_scope='Every query/method/cap, 11 tie seeds; primary conditional ranks at 4 control seeds; all archived main neighbor indices validated',
        overlap_scope='Every 37th overlap row, all novel/balanced score exports and all hierarchical summary means; balanced primary/meta ranks at first tie seed')
    (args.out/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks,flush=True)


if __name__=='__main__':main()
