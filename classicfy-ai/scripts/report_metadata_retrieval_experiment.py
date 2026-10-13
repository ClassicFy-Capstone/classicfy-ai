"""Render fixed retrieval outcomes, without selecting models or examples."""

import argparse
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from embedding.retrieval_probe import weights,CHANNELS
from audit_asap_metadata import read_csv,sha256
from validate_tempo import plt,setup_style

NAMES={'M-C':'작곡가','M-T':'작품 유형','M-CT':'작곡가+유형','Q':'길이·coverage',
    'M-CTQ':'메타+길이·coverage','B':'평균·범위','B+T':'+시간 통계','B+T+R':'+관계 정보','cnn64':'CNN','bilstm64':'BiLSTM'}
JUDGES={'E-L':'대표 성향·표현 폭','E-T':'국소 변화 크기·분포','E-R':'빠르기–강약 공동 변화'}
DISPLAY=(*('M-C','M-T','M-CT','Q','M-CTQ'),*('B','B+T','B+T+R'),*('cnn64','bilstm64'))


def values(frame,method,**filters):
    selected=frame.copy()
    for field,value in filters.items():selected=selected[selected[field]==value]
    selected=selected[selected.method.str.startswith(method+'_')] if method in ('cnn64','bilstm64') else selected[selected.method==method]
    return selected.value.to_numpy(float)*100


def save(fig,out,name):
    for ext in ('png','svg','pdf'):
        path=out/f'{name}.{ext}';fig.savefig(path,dpi=170)
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/metadata_retrieval_results')
    parser.add_argument('--cache',type=Path,default=ROOT.parent/'datasets/metadata_retrieval_run01')
    args=parser.parse_args();out=args.out;setup_style();summary=pd.read_csv(out/'summary.csv')
    base=summary[(summary.role=='test')&(summary.k==5)&summary.cap]
    methods=('B','B+T','B+T+R','cnn64','bilstm64');meta=('M-C','M-T','M-CT','Q','M-CTQ')
    fig,axes=plt.subplots(1,2,figsize=(14,5.8))
    for ax,metric,title in zip(axes,('recording_overlap','family_overlap'),('공통 연주 / 5개','공통 작품군 / 5개')):
        matrix=np.array([[values(base,m,table='overlap',experiment='whole',reference=c,metric=metric).mean() for c in meta] for m in methods])
        im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
        for i,j in np.ndindex(matrix.shape):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',fontsize=12,color='white' if matrix[i,j]>55 else '#223')
        ax.set_xticks(range(5),[NAMES[m].replace('·','\n') for m in meta],rotation=18,ha='right')
        ax.set_yticks(range(5),[NAMES[m] for m in methods]);ax.set_title(title,fontsize=16,loc='left')
    fig.colorbar(im,ax=axes.tolist(),fraction=.025,pad=.04,label='겹침 (%)')
    fig.suptitle('연주 검색은 여러 메타데이터 기준과 다른 후보를 가져온다',fontsize=20,fontweight='bold')
    fig.text(.5,.015,'전곡 / test 67연주·6작품군 / 후보 351연주·32작품군 / Top 5, 작품군당 1연주 / 동점 100회 평균 / CNN·BiLSTM 모든 3 seed 평균',ha='center',fontsize=9)
    fig.subplots_adjust(left=.10,right=.88,bottom=.24,top=.85,wspace=.4);save(fig,out,'01_metadata_overlap')
    fig,axes=plt.subplots(1,3,figsize=(17,7.5))
    for ax,judge in zip(axes,JUDGES):
        for i,method in enumerate(DISPLAY):
            v=values(base,method,table='style_scores',judge=judge);value=v.mean()
            ax.barh(i,value,color='#8c999f' if i<5 else '#2878c8' if i<8 else '#da6a35',height=.65)
            ax.plot([v.min(),v.max()],[i,i],color='#44352d',lw=2)
            ax.text(max(v)+1,i,f'{value:.1f}',va='center',fontsize=10)
        ax.axvline(50,color='#555',ls='--',lw=1);ax.set_xlim(0,90)
        ax.set_yticks(range(len(DISPLAY)),[NAMES[m] for m in DISPLAY]);ax.invert_yaxis();ax.set_title(JUDGES[judge],fontsize=15,loc='left')
        ax.set_xlabel('스타일 일치 점수 / 100  ·  높을수록 가까움')
    fig.suptitle('검색에 쓰지 않은 구간에서도 전반적인 성향·폭이 더 잘 맞는가?',fontsize=20,fontweight='bold')
    fig.text(.5,.02,'test 48연주·4작품군 / A→B, B→A 평균 / 무작위 기대값=50, 정확도가 아님 / 선: 모델 3 seed 최소~최대 / 평가 작품군 부족으로 신뢰구간 없음',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.93));save(fig,out,'02_heldout_style')
    archive=np.load(args.cache/'retrieval_arrays.npz');rows=read_csv(ROOT/'classicfy-ai/analysis/metadata_information_design/target_preflight.csv')
    fig,axes=plt.subplots(1,3,figsize=(15,5.8));effects=[]
    for ax,(method,judge) in zip(axes,(('B','E-L'),('B+T','E-T'),('B+T+R','E-R'))):
        for i,control in enumerate(('actual','same_work','conditional_shuffle','uniform')):
            repetitions=[]
            for view in ('A','B'):
                tag=f'split_{view}';qids=archive[tag+'_query_ids'];local=np.array([j for j,q in enumerate(qids) if rows[q]['evaluation_role']=='test'])
                w=weights([rows[qids[j]] for j in local]);score=archive[f'{tag}_judge_score_{judge}']
                name=f'{tag}_rank_1_{method}' if control=='actual' else f'{tag}_same_work_{method}' if control=='same_work' else f'{tag}_conditional_rank_{method}' if control=='conditional_shuffle' else tag+'_uniform_rank'
                selected=archive[name][:,:,:5];v=score[np.arange(len(qids))[None,:,None],selected].mean(axis=2)[:,local]@w
                repetitions.append(v)
            v=np.mean(repetitions,axis=0)*100;lo,hi=np.percentile(v,[5,95]);value=v.mean()
            ax.scatter([i],[value],s=65,color='#2878c8' if i==0 else '#85949b');ax.plot([i,i],[lo,hi],color='#33434d',lw=2)
            ax.text(i,value+2,f'{value:.1f}',ha='center',fontsize=11)
            effects.append(dict(method=method,judge=judge,control=control,value=float(value),replicate_p05=float(lo),replicate_p95=float(hi)))
        ax.axhline(50,color='#777',ls='--',lw=1);ax.set_ylim(42,82);ax.set_xticks(range(4),['실제 검색','같은 작품\n임의 연주','조건부\n벡터 섞기','무작위'])
        ax.set_ylabel('별도 구간 스타일 점수 / 100');ax.set_title(NAMES[method]+' / '+JUDGES[judge],loc='left',fontsize=14)
    fig.suptitle('작품만 고른 효과와 특정 연주를 고른 효과를 구분',fontsize=20,fontweight='bold')
    fig.text(.5,.02,'같은 작품 대조: 반환 작품 유지, 연주만 무작위 / 조건부 섞기: 작곡가·유형·길이·coverage 구간 유지 / 선: 100회 대조의 5~95% 범위, 신뢰구간 아님',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.92));save(fig,out,'03_conditional_controls')
    pd.DataFrame(effects).to_csv(out/'control_figure_values.csv',index=False)
    works=pd.read_csv(out/'per_work_metrics.csv');fig,axes=plt.subplots(1,3,figsize=(17,6.5))
    for ax,(method,judge) in zip(axes,(('B','E-L'),('B+T','E-T'),('B+T+R','E-R'))):
        local=works[(works.role=='test')&(works.candidate==method)].sort_values('piece');v=local.delta.to_numpy()*100
        labels=[p.replace('/midi_score.mid','').replace('Piano_Sonatas/','Sonata ').replace('Etudes_op_25/','Etude25 ').replace('Ballades/','Ballade ') for p in local.piece]
        ax.barh(range(len(v)),v,color=['#2878c8' if x>0 else '#dc6436' for x in v]);ax.axvline(0,color='#666',lw=1)
        ax.set_yticks(range(len(v)),labels,fontsize=9);ax.invert_yaxis();ax.set_xlabel('기준 대비 점수 차이 (%p)')
        ax.set_title(f'{method} vs {local.reference.iloc[0]}\n{JUDGES[judge]}',loc='left',fontsize=14)
        for i,x in enumerate(v):ax.text(x,i,f'{x:+.1f}',ha='left' if x>0 else 'right',va='center',fontsize=10)
        ax.margins(x=.25)
    fig.suptitle('작품별로 보면 평균·범위의 효과와 보강 효과가 구분된다',fontsize=20,fontweight='bold')
    fig.text(.5,.02,'평가 4작품군 / 신뢰구간을 표시하지 않음 / 시간 추가의 작은 test 개선은 validation·균형 후보 대조에서 유지되지 않음',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.91));save(fig,out,'04_per_work_gain')
    neighbors=pd.read_csv(out/'neighbors.csv');whole=neighbors[neighbors.experiment=='whole'];fig,axes=plt.subplots(2,3,figsize=(17,10))
    for ax,method in zip(axes.flat,('M-C','M-T','M-CT','B','B+T','B+T+R')):
        local=whole[(whole.method==method)&(whole['rank']<=5)].sort_values('rank');ax.axis('off');ax.set_title(NAMES[method],loc='left',fontsize=17,fontweight='bold')
        for i,(_,r) in enumerate(local.iterrows()):
            piece=r.candidate_piece.replace('/midi_score.mid','');record=Path(r.candidate).name
            label=f'{i+1}. '+textwrap.fill(piece,width=40)+'\n    '+record
            ax.text(.0,.91-i*.18,label,transform=ax.transAxes,va='top',fontsize=10.5)
    query=whole.iloc[0]['query'];fig.suptitle('같은 기준 연주에서 가져온 실제 Top 5',fontsize=22,fontweight='bold')
    fig.text(.5,.925,query+' / eligible test 이름순 첫 연주 / 메타데이터 동점 첫 seed 예시',ha='center',fontsize=10)
    fig.text(.5,.018,'전곡 검색 예시 / 모든 동점 seed의 수치 결과는 별도 집계 / 이름순으로 예시를 고정했고 성능으로 고르지 않음',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.89));save(fig,out,'05_retrieval_example')
    # Separate example: A-side retrieval, B-side measurement profiles only.
    local=neighbors[(neighbors.experiment=='split')&(neighbors['view']=='A')]
    prepared=np.load(args.cache/'prepared.npz');lookup={r['key']:i for i,r in enumerate(rows)}
    scales=json.loads((out/'scales.json').read_text())['split_A']['judge']['E-L']
    mean=np.array(scales['mean']);scale=np.array(scales['scale']);active=np.array(scales['active'])
    owners=[('기준 연주',local.iloc[0]['query'],'#222')]
    for method,color in [('M-CT','#839c67'),('B','#2878c8')]:
        owners.append((NAMES[method]+' 첫 후보',local[(local.method==method)&(local['rank']==1)].iloc[0]['candidate'],color))
    fig,axes=plt.subplots(1,2,figsize=(15,5.5));profile_rows=[]
    for label,key,color in owners:
        raw=prepared['B_E-L'][lookup[key]]
        profile=np.divide(raw-mean,scale,out=np.full(14,np.nan),where=active)
        for j,ax in enumerate(axes):
            ax.plot(range(7),profile[j::2],marker='o',color=color,label=label,lw=1.7)
            for c in range(7):profile_rows.append(dict(label=label,key=key,channel=CHANNELS[c],statistic='median' if j==0 else 'IQR',standardized_value=profile[c*2+j],active=bool(active[c*2+j])))
    for ax,title in zip(axes,('평가 구간의 대표 성향 · 중앙값','평가 구간의 표현 폭 · IQR')):
        ax.set_xticks(range(7),['Tempo','Rubato','Dynamics','Articulation','Pedal\ndepth','Pedal\non','Pedal\nchanges'])
        ax.set_ylabel('train 기준 표준화 측정치');ax.set_title(title,loc='left',fontsize=15);ax.axhline(0,color='#888',ls='--',lw=.7);ax.legend(fontsize=10,frameon=False)
    fig.suptitle('검색에 쓰지 않은 구간에서 실제 스타일 측정치 비교',fontsize=20,fontweight='bold')
    fig.text(.5,.90,owners[0][1]+' / 이름순 첫 eligible query / A 검색·B 평가 / 첫 후보 하나의 설명 사례',ha='center',fontsize=9)
    fig.text(.5,.02,'Rubato 성향은 절댓값 중앙값 / 작품별 정규화 잔차이며 원래 빠르기·강약 단위가 아님 / 전체 점수는 Top 5와 모든 query를 사용',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.84));save(fig,out,'06_heldout_example_profiles')
    pd.DataFrame(profile_rows).to_csv(out/'example_profiles.csv',index=False)
    manifest=dict(execution_sha256=sha256(out/'execution.json'),reporter_sha256=sha256(__file__),
        figure_sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.suffix in ('.png','.svg','.pdf')},
        controls_sha256=sha256(out/'control_figure_values.csv'),profiles_sha256=sha256(out/'example_profiles.csv'),example_selection='Lexical first eligible test query, first fixed tie seed, no outcome selection')
    (out/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
