"""Compare every eligible catalogue G distance with final era embeddings.

This is full-fit descriptive analysis, not held-out or human listening accuracy.
G comparisons retain the training contract's directed anchor signature.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from embedding.era_note_style import pair_tables
from features.style_profile import G_BLOCKS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def correlation(x, y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return None
    return float(spearmanr(x,y).statistic)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    metadata=read(args.run/'metadata.json')
    lookup={v['perf_id']:i for i,v in enumerate(metadata)}
    md=[{**v,'split':'production'} for v in metadata]
    with np.load(args.run/'profiles.npz') as source:
        raw,known,quality=[source[k] for k in ('raw','known','quality')]
    final=read(args.run/'final_report.json')
    fonts={v.name for v in font_manager.fontManager.ttflist}
    plt.rcParams['font.family']=next(n for n in ('Malgun Gothic','Noto Sans CJK KR','NanumGothic') if n in fonts)
    plt.rcParams['axes.unicode_minus']=False
    names={'baroque':'바로크','classical_transition':'고전 + 전환','romantic_impressionist':'낭만 + 인상주의'}
    report={'scope':'All 5082 embeddings; eligible same-era cross-work G comparisons only; full-fit diagnostics, no independent test',
            'distance_contract':'Directed anchor signature; reverse direction can be absent or use different dimensions. Do not treat directed pairs as independent samples.',
            'G_definition':'Mean over present blocks of mean squared standardized reference-relative feature differences; fixed production scaler, clipping +/-5.',
            'target_definition':'Dg/(Dg+saved era calibration median); monotonically maps G to [0,1), not a claim of linear proportionality to raw Dg.',
            'embedding_definition':'1-dot(e_a,e_b), final 64D L2-normalized era head output',
            'no_missing_imputation':True,'eras':{}}
    clip_rows=[];outlier_rows=[]
    scatter,axs=plt.subplots(1,3,figsize=(16,5.1))
    trend,taxs=plt.subplots(1,3,figsize=(16,4.9),sharey=True)
    for col,(era,values) in enumerate(final['eras'].items()):
        saved=read(args.run/era/'g_production_scaler.json');sc=saved['scaler']
        with np.load(args.run/era/'embeddings.npz') as source:
            perf_ids=source['perf_ids'];e=source['e']
        ids=np.array([lookup[str(p)] for p in perf_ids])
        assert len(ids)==values['clips'] and np.isfinite(e).all()
        np.testing.assert_allclose(np.linalg.norm(e,axis=1),1,atol=1e-6)
        d,q=pair_tables(raw,known,quality,md,sc,ids)
        dz=np.clip(1-e@e.T,0,2)
        valid=np.isfinite(d);r,c=np.where(valid)
        g=d[valid];emb=dz[valid];conf=q[valid]
        target=g/(g+saved['calibration']['median'])
        work=np.array([md[i]['work'] for i in ids]);sources=np.array([md[i]['source'] for i in ids])
        possible=(work[:,None]!=work[None,:])&(sources[:,None]!=sources[None,:])
        assert np.all(~valid|possible)
        expected=values['production_training_diagnostics']['geometry']
        rho=correlation(g,emb)
        mse=float(np.mean(conf*(emb-target)**2))
        assert len(g)==expected['pairs']
        assert abs(rho-expected['spearman'])<2e-6,(rho,expected['spearman'])
        assert abs(mse-expected['pair_mse'])<2e-6
        anchors=[];bins=[]
        mask=known[ids]&np.asarray(sc['trained'],bool)
        for a,pid in enumerate(perf_ids):
            candidates=np.flatnonzero(valid[a]);n=len(candidates)
            row={'era':era,'perf_id':str(pid),'work':work[a],
                 'known_G_dimensions':int(mask[a].sum()),'valid_candidates':n,
                 'incoming_comparisons':int(valid[:,a].sum())}
            if n>=20:
                gd,ed=d[a,candidates],dz[a,candidates]
                near_g=gd<=np.quantile(gd,.2);far_g=gd>=np.quantile(gd,.8)
                near_e=ed<=np.quantile(ed,.2);far_e=ed>=np.quantile(ed,.8)
                sorted_far=np.sort(ed[far_g]);near=ed[near_g]
                less=np.searchsorted(sorted_far,near,side='left')
                right=np.searchsorted(sorted_far,near,side='right')
                concordance=np.mean((len(sorted_far)-right+.5*(right-less))/len(sorted_far))
                gtop=np.argsort(gd,kind='stable')[:5];etop=np.argsort(ed,kind='stable')[:5]
                row.update(anchor_spearman=correlation(gd,ed),
                    nearG_also_nearE=float(near_e[near_g].mean()),
                    nearG_but_farE=float(far_e[near_g].mean()),
                    near20_chance=float(near_e.mean()),
                    nearG_E_mean=float(ed[near_g].mean()),farG_E_mean=float(ed[far_g].mean()),
                    near_far_order_fraction=float(concordance),
                    top5_overlap_fraction=float(len(set(gtop)&set(etop))/5),
                    top5_chance=5/n)
                anchors.append(row)
            clip_rows.append(row)
        for lo in np.arange(0,1,.1):
            left,right=np.quantile(g,[lo,min(1.,lo+.1)])
            inside=(g>=left)&((g<=right) if lo>.85 else (g<right))
            y=emb[inside]
            bins.append({'G_q_low':float(lo),'G_q_high':float(min(1.,lo+.1)),
                         'pairs':int(inside.sum()),'G_mean':float(g[inside].mean()),
                         'e_q25':float(np.quantile(y,.25)),'e_median':float(np.median(y)),
                         'e_q75':float(np.quantile(y,.75))})
        feature={}
        x=np.clip((raw[ids]-sc['center'])/sc['scale'],-5,5)
        for block,indices in G_BLOCKS.items():
            bd=np.full(d.shape,np.nan,np.float32)
            for a in range(len(ids)):
                dims=[j for j in indices if mask[a,j]]
                candidate=np.flatnonzero(valid[a])
                if dims and len(candidate):bd[a,candidate]=((x[candidate][:,dims]-x[a,dims])**2).mean(1)
            ok=np.isfinite(bd)
            feature[block]={'pairs':int(ok.sum()),'spearman':correlation(bd[ok],dz[ok])}
        last=np.quantile(conf,.75);high=conf>=last
        row_summary={
            'name':names[era],'all_clips':len(ids),'clips_with_outgoing_G':int(valid.any(1).sum()),
            'clips_with_any_G_pair':int((valid.any(0)|valid.any(1)).sum()),
            'directed_valid_pairs':len(g),'eligible_cross_work_directed_pairs':int(possible.sum()),
            'pair_coverage_fraction':float(valid.sum()/possible.sum()),
            'distinct_unordered_valid_pairs':int(np.triu(valid|valid.T,1).sum()),
            'spearman':rho,'pearson_raw_G':float(pearsonr(g,emb).statistic),
            'pearson_calibrated_G':float(pearsonr(target,emb).statistic),
            'weighted_target_MSE':mse,
            'confidence_top_quartile':{'threshold':float(last),'pairs':int(high.sum()),'spearman':correlation(g[high],emb[high])},
            'anchors_with_at_least20_candidates':len(anchors),
            'anchor_balanced':{key:float(np.mean([a[key] for a in anchors])) for key in
                ('nearG_also_nearE','nearG_but_farE','near20_chance','nearG_E_mean','farG_E_mean',
                 'near_far_order_fraction','top5_overlap_fraction','top5_chance')},
            'anchor_spearman_quantiles':np.quantile([a['anchor_spearman'] for a in anchors],[.25,.5,.75]).tolist(),
            'feature_distance_correlations':feature,'G_deciles':bins,
            'baseline_spearman':values['baseline']['geometry']['spearman']}
        report['eras'][era]=row_summary
        delta=emb-target
        take=np.argpartition(np.abs(delta),-10)[-10:]
        for k in take[np.argsort(-np.abs(delta[take]))]:
            outlier_rows.append({'era':era,'anchor':str(perf_ids[r[k]]),'candidate':str(perf_ids[c[k]]),
                'G_distance':float(g[k]),'G_target':float(target[k]),'embedding_distance':float(emb[k]),
                'confidence':float(conf[k]),'absolute_target_error':float(abs(delta[k]))})
        ax=axs[col]
        hb=ax.hexbin(target,emb,gridsize=50,mincnt=1,norm=LogNorm(),cmap='viridis',extent=(0,1,0,2))
        ax.plot([0,1],[0,1],ls='--',color='#E27355',lw=1.5,label='목표와 정확히 일치')
        ax.set(title=f'{names[era]} | 순위 상관 ρ={rho:.3f}',xlabel='G 거리의 학습 목표값: Dg / (Dg + 중앙값)',ylim=(0,2),xlim=(0,1))
        if col==0:ax.set_ylabel('임베딩 코사인 거리')
        ax.legend(fontsize=8,loc='upper left')
        scatter.colorbar(hb,ax=ax,label='비교 수 (로그 색상)')
        ax=taxs[col];xx=np.arange(1,11)
        ax.fill_between(xx,[b['e_q25'] for b in bins],[b['e_q75'] for b in bins],color='#537DA4',alpha=.22,label='중앙 50% 범위')
        ax.plot(xx,[b['e_median'] for b in bins],'o-',color='#2C638A',label='임베딩 거리 중앙값')
        ax.set(title=names[era],xlabel='G 거리 구간 (1=가장 가까운 10%, 10=가장 먼 10%)',xticks=xx)
        if col==0:ax.set_ylabel('임베딩 코사인 거리');ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        print(era,json.dumps({k:v for k,v in row_summary.items() if k not in ('G_deciles','feature_distance_correlations')},ensure_ascii=True),flush=True)
    scatter.suptitle('전체 비교 가능 쌍: G 목표 거리와 최종 임베딩 거리',fontsize=17)
    scatter.text(.5,.012,'같은 시대 · 다른 작품 · G 유효 비교만 포함 | 모든 유효 방향쌍 사용 | 전체 학습 자료의 진단',ha='center',fontsize=10)
    scatter.tight_layout(rect=(0,.05,1,.94));scatter.savefig(args.out/'distance_density.png',dpi=160);plt.close(scatter)
    trend.suptitle('G가 가까워질수록 임베딩도 가까워지는가?',fontsize=17)
    trend.text(.5,.015,'선이 우상향하면 양의 관계 | 음영이 크게 겹치면 개별 연주 쌍의 가까움·멀어짐은 자주 어긋남',ha='center',fontsize=11)
    trend.tight_layout(rect=(0,.06,1,.94));trend.savefig(args.out/'G_decile_trend.png',dpi=160);plt.close(trend)
    fig,axes=plt.subplots(1,2,figsize=(12,4.9))
    rows=list(report['eras'].values());xx=np.arange(3);labels=[r['name'] for r in rows]
    metrics=[('nearG_also_nearE','near20_chance','G 최근접 20%가 임베딩 최근접 20%에도 포함'),
             ('top5_overlap_fraction','top5_chance','G 최근접 5개와 임베딩 최근접 5개의 겹침')]
    for ax,(key,chance,title) in zip(axes,metrics):
        yy=[100*r['anchor_balanced'][key] for r in rows]
        baseline=[100*r['anchor_balanced'][chance] for r in rows]
        bars=ax.bar(xx-.16,yy,.32,color='#326D91',label='실제 일치 비율')
        ax.bar(xx+.16,baseline,.32,color='#BDC5CE',label='무작위 기대 비율')
        for b,y in zip(bars,yy):ax.text(b.get_x()+b.get_width()/2,y+.6,f'{y:.1f}%',ha='center')
        ax.set(title=title,xticks=xx,xticklabels=labels,ylabel='겹치는 비율 (%)',ylim=(0,max(yy+baseline)*1.3))
        ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('가까운 후보가 실제로 얼마나 같은가?',fontsize=17)
    fig.text(.5,.012,'각 기준 연주에서 G 유효 후보가 20개 이상일 때 계산 후 기준 연주 균등 평균 | 동일 후보 집합 사용',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(args.out/'neighbor_overlap.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4))
    matrix=np.array([[r['feature_distance_correlations'][b]['spearman'] for b in G_BLOCKS] for r in rows])
    im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
    ax.set_xticks(np.arange(5),['Tempo','Timing','Dynamics','Articulation','Pedaling']);ax.set_yticks(np.arange(3),labels)
    for i in range(3):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center')
    ax.set_title('각 표현 영역 거리와 임베딩 거리의 순위 상관')
    fig.colorbar(im,ax=ax,label='Spearman ρ')
    fig.text(.5,.02,'영역 간 상관·결측 구성이 다름: 고유 기여도나 정보 보존율을 뜻하지 않음',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,1));fig.savefig(args.out/'feature_correlations.png',dpi=160);plt.close(fig)
    for name,records in [('per_clip.csv',clip_rows),('largest_disagreements.csv',outlier_rows)]:
        fields=list(dict.fromkeys(k for record in records for k in record))
        with (args.out/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)
    report['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [args.run/'profiles.npz',args.run/'metadata.json',args.run/'final_report.json',Path(__file__)]}
    for era in final['eras']:
        for name in ('embeddings.npz','g_production_scaler.json'):
            p=args.run/era/name;report['source_hashes'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
