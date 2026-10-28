"""Render fixed-design readout results; do not fit or select models."""

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np

from audit_asap_metadata import ROOT, read_csv, write_csv, sha256
from validate_tempo import plt, setup_style

TARGETS=('change_concentration_cv','coupling_local_strength','coupling_local_variation')
LABELS={'change_concentration_cv':'강약 변화 편중', 'coupling_local_strength':'구간별 빠르기–강약 관계 강도',
        'coupling_local_variation':'구간별 관계 변동 (보조)'}
CONDITIONS=('A0','A1','B','C','D','E_CNN','E_BiLSTM')
NAMES=('메타데이터','메타+길이/coverage','+평균·범위','+개별 시간','+관계','기존 요약+CNN','기존 요약+BiLSTM')


def save(fig,out,name):
    for extension in ('png','svg','pdf'):
        path=out/f'{name}.{extension}';fig.savefig(path,dpi=180)
        if extension=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


def draw_errors(metrics,out):
    fig,axes=plt.subplots(2,2,figsize=(16,9))
    for i,target in enumerate(TARGETS[:2]):
        local=[r for r in metrics if r['target']==target and r['role']=='test']
        for j,field in enumerate(('global_mse','within_mse')):
            ax=axes[i,j];reference=float(next(r[field] for r in local if r['condition']=='A1'))
            for k,condition in enumerate(CONDITIONS):
                values=np.array([float(r[field])/reference for r in local if r['condition']==condition])
                value=values.mean()
                ax.barh(k,value,color='#8a9187' if k<2 else '#2878c8' if k<5 else '#dc6436',height=.63)
                ax.plot([values.min(),values.max()],[k,k],color='#49372c',lw=2)
                ax.text(values.max()+.03,k,f'{value:.3f}',va='center',fontsize=10)
            ax.axvline(1,color='#50594e',lw=1,ls='--')
            ax.set_yticks(np.arange(7),NAMES);ax.invert_yaxis()
            ax.set_xlim(0,2.4 if i==0 else 1.28)
            ax.set_xlabel('오차 / 메타+길이·coverage 기준 오차  ·  작을수록 좋음',fontsize=11)
            ax.set_title(LABELS[target]+' / '+('전체 오차' if j==0 else '같은 작품 안의 연주 차이 오차'),loc='left',fontsize=15)
    fig.suptitle('보강 표현을 추가하면 측정치를 더 잘 읽을 수 있는가?',fontsize=21,fontweight='bold')
    fig.text(.5,.018,'평가 작품: 강약 6작품·67연주 / 관계 5작품·56연주  ·  신경망: 모든 3 seed 평균·최소~최대  ·  기준=1, 성공률이 아님',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.95));save(fig,out,'01_prediction_errors')


def draw_primary(bootstrap,out):
    fig,axes=plt.subplots(1,2,figsize=(14,5.8))
    primary=((TARGETS[0],'C','B'),(TARGETS[1],'D','C'))
    for ax,(target,condition,reference) in zip(axes,primary):
        for i,metric in enumerate(('global_mse','within_mse')):
            r=next(r for r in bootstrap if r['target']==target and r['role']=='test' and r['condition']==condition and r['metric']==metric and r['encoder_seed']=='0')
            value=float(r['reduction_percent']);lo,hi=float(r['reduction_ci_low']),float(r['reduction_ci_high'])
            ax.barh(i,value,color='#dc6436' if value<0 else '#2878c8',height=.55)
            ax.plot([lo,hi],[i,i],color='#322c27',lw=2);ax.scatter([value],[i],color='#322c27',s=25)
            ax.text(.98,.85-i*.41,f'{value:+.1f}%\n개선 작품 {r["improved_works"]}/{r["works"]}',transform=ax.transAxes,ha='right',va='top',fontsize=12)
        ax.axvline(0,color='#444',lw=1)
        ax.set_yticks([0,1],['전체 오차','작품 내 차이 오차']);ax.invert_yaxis()
        ax.set_xlim(-48,20)
        ax.set_xlabel('기준 대비 오차 감소율 (%)  ·  음수는 오차 증가')
        ax.set_title(('시간 추가: C vs B' if condition=='C' else '관계 추가: D vs C')+'\n'+LABELS[target],loc='left',fontsize=16)
    fig.suptitle('사전에 정한 두 주 비교에서는 개선이 확인되지 않았다',fontsize=21,fontweight='bold')
    fig.text(.5,.025,'선: 같은 평가 작품군을 짝지어 재표집한 95% 구간 (2,000회)  ·  평가 작품군은 6개/5개  ·  구간이 0을 포함하면 방향이 불확실',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.89));save(fig,out,'02_primary_comparisons')


def control_effects(metrics,controls):
    result=[]
    for r in controls:
        reference='B' if r['condition']=='C' else 'C'
        base=next(m for m in metrics if m['target']==r['target'] and m['role']==r['role'] and m['condition']==reference)
        for metric in ('global_mse','within_mse'):
            a,b=float(base[metric]),float(r[metric])
            result.append({f:r[f] for f in ('condition','kind','control_seed','target','role')}|dict(reference=reference,metric=metric,reduction_percent=100*(1-b/a)))
    return result


def draw_controls(bootstrap,effects,out):
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    for i,(target,condition) in enumerate(((TARGETS[0],'C'),(TARGETS[1],'D'))):
        for j,metric in enumerate(('global_mse','within_mse')):
            ax=axes[i,j]
            r=next(r for r in bootstrap if r['target']==target and r['role']=='test' and r['condition']==condition and r['metric']==metric and r['encoder_seed']=='0')
            for x,kind in enumerate(('shuffle','noise'),1):
                values=[float(r['reduction_percent']) for r in effects if r['target']==target and r['condition']==condition and r['role']=='test' and r['metric']==metric and r['kind']==kind]
                ax.boxplot(values,positions=[x],widths=.38,showfliers=False,patch_artist=True,boxprops=dict(facecolor='#bac6d0'))
                ax.scatter(x+np.linspace(-.09,.09,len(values)),values,color='#5c6e7f',s=14,alpha=.7)
            actual=float(r['reduction_percent'])
            ax.scatter([0],[actual],color='#dc6436',s=70,zorder=3)
            ax.text(.07,actual,f'{actual:+.1f}%',va='center',fontsize=11)
            ax.axhline(0,color='#777',ls='--',lw=1)
            ax.set_xticks([0,1,2],['실제 보강','연주 대응 섞기','같은 차원 잡음'])
            ax.set_ylabel('기준 대비 오차 감소율 (%)')
            ax.set_title(LABELS[target]+' / '+('전체' if j==0 else '작품 내 차이'),loc='left',fontsize=14)
            ax.set_xlim(-.35,2.35)
    fig.suptitle('주 보강의 결과를 섞기·잡음 대조와 함께 확인',fontsize=21,fontweight='bold')
    fig.text(.5,.02,'각 대조는 20 seed  ·  추가 블록만 변경  ·  대조에서도 같은 train-only CV  ·  0 위는 개선, 아래는 악화',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.93));save(fig,out,'03_control_comparison')


def draw_work_results(works,out):
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,(target,condition,reference) in zip(axes,((TARGETS[0],'C','B'),(TARGETS[1],'D','C'))):
        before={r['piece']:r for r in works if r['target']==target and r['role']=='test' and r['condition']==reference}
        after={r['piece']:r for r in works if r['target']==target and r['role']=='test' and r['condition']==condition}
        pieces=sorted(before)
        for offset,metric,color,label in ((-.16,'global_mse','#2878c8','전체'),(.16,'within_mse','#dc6436','작품 내 차이')):
            values=[100*(1-float(after[p][metric])/float(before[p][metric])) for p in pieces]
            ax.barh(np.arange(len(pieces))+offset,values,height=.28,color=color,label=label)
        ax.axvline(0,color='#666',lw=1)
        ax.set_yticks(np.arange(len(pieces)),[p.replace('/midi_score.mid','').replace('Piano_Sonatas/','Sonata ').replace('Transcendental_Etudes/','Etude ') for p in pieces],fontsize=10)
        ax.invert_yaxis();ax.set_xlabel('기준 대비 오차 감소율 (%)');ax.legend(frameon=False)
        ax.set_title(LABELS[target]+' / '+condition+' vs '+reference,loc='left',fontsize=15)
    fig.suptitle('평균 결과에 가려지지 않도록 평가 작품별 변화도 확인',fontsize=21,fontweight='bold')
    fig.tight_layout(rect=(0,.03,1,.92));save(fig,out,'04_per_work_results')


def example_source(design,protocol,data):
    rows=read_csv(design/'target_preflight.csv')
    eligible=[r for r in rows if r['evaluation_role']=='test' and all(r[f'{t}_eligible']=='True' for t in TARGETS)]
    piece=sorted({r['piece'] for r in eligible})[0]
    local=sorted([r for r in eligible if r['piece']==piece],key=lambda r:r['key']);selected=local[len(local)//2]
    raw=subprocess.check_output(['git','archive',protocol['learned_source_commit'],'classicfy-ai/src','classicfy-ai/scripts'],cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='classicfy-information-example-') as directory:
        snapshot=Path(directory)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:archive.extractall(snapshot,filter='data')
        names=subprocess.check_output(['git','ls-tree','-r','--name-only','20a4168','classicfy-ai/src/features','classicfy-ai/src/preprocessing'],cwd=ROOT,text=True).splitlines()
        original=[n for n in names if n.endswith('.py')]
        source=[snapshot/n for n in original if '/features/' in n and Path(n).name not in ('normalization.py','__init__.py')]+[snapshot/n for n in original if '/preprocessing/' in n]
        sys.path[:0]=[str(snapshot/'classicfy-ai/scripts'),str(snapshot/'classicfy-ai/src')]
        # Import this source by a distinct module name: other report helpers
        # already loaded the working branch's validate_embedding module.
        import importlib.util
        spec=importlib.util.spec_from_file_location('verified_example_collector',snapshot/'classicfy-ai/scripts/validate_embedding.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        reference=json.loads((data/'temporal_embedding_cluster_run01/dataset.json').read_text())
        groups,audits,scales,provenance=module.collect_groups(SimpleNamespace(asap_root=(data/'ASAP').resolve(),nasap_root=(data/'nASAP').resolve(),cache=data/'feature_normalization_cluster_raw.npz',**reference['cohort_settings']),provenance_source_files=source)
        if provenance!=protocol['source_provenance'] or scales!=reference['scales']:raise ValueError('Example reference differs')
        group=next(g for g in groups if selected['key'] in g['keys']);owner=group['keys'].index(selected['key'])
        length=next(r['total_intervals'] for r in audits if r['piece']==group['piece'] and r['grid']==group['grid'])
        index=np.array(group['shared_indices'],dtype=int);t,d=np.full(length,np.nan),np.full(length,np.nan);mask=np.zeros(length,dtype=bool)
        t[index],d[index],mask[index]=group['values'][owner,0],group['values'][owner,2],True
    return selected,t,d,mask


def draw_example(design,protocol,data,out):
    selected,t,d,mask=example_source(design,protocol,data)
    rms,r,entries=[],[],[]
    for size in (32,64):
        for start in range(0,len(d)-size+1,size):
            a,b=t[start:start+size],d[start:start+size]
            valid=mask[start:start+size];paired=valid[:-1]&valid[1:]
            value=np.nan;status='insufficient_pairs'
            if paired.sum()>=(25 if size==32 else 51):
                x,y=np.diff(a)[paired],np.diff(b)[paired]
                if size==32:value=float(np.sqrt(np.mean(y*y)));status='ok'
                elif min(x.std(),y.std())>1e-12:value=float(np.corrcoef(x,y)[0,1]);status='ok'
                else:status='constant'
            (rms if size==32 else r).append(value)
            entries.append(dict(key=selected['key'],block_size=size,start=start,valid_pairs=int(paired.sum()),value=value,status=status))
    rms,r=np.array(rms),np.array(r);u=rms[np.isfinite(rms)];v=r[np.isfinite(r)]
    values=[u.std()/u.mean(),np.mean(abs(v)),v.std()]
    np.testing.assert_allclose(values,[float(selected[name]) for name in TARGETS],atol=1e-12,rtol=0)
    fig,axes=plt.subplots(3,1,figsize=(15,9))
    axes[0].plot(d,color='#2878c8',lw=1,label='강약 잔차');axes[0].plot(t,color='#dc6436',lw=1,alpha=.7,label='빠르기 잔차')
    axes[0].legend(ncols=2,frameon=False);axes[0].set_ylabel('정규화된 값');axes[0].set_title('원래 beat 위치의 feature · 결측은 연결하지 않음',loc='left',fontsize=15)
    axes[1].bar(np.arange(len(rms)),rms,color='#2878c8');axes[1].set_ylabel('변화량 RMS');axes[1].set_xlabel('32-beat 구간 번호')
    axes[1].scatter(np.flatnonzero(~np.isfinite(rms)),np.zeros((~np.isfinite(rms)).sum()),marker='x',color='#888',s=35,label='계산 불가 (0으로 대체한 값 아님)')
    axes[1].set_xlim(-.6,len(rms)-.4);axes[1].legend(frameon=False,fontsize=9)
    axes[1].set_title(f'구간별 강약 변화 크기 → 편중 CV = {values[0]:.3f}  (예측값은 log1p 변환)',loc='left',fontsize=15)
    axes[2].bar(np.arange(len(r)),r,color='#dc6436');axes[2].axhline(0,color='#777',lw=1);axes[2].set_ylim(-1,1);axes[2].set_ylabel('변화량 상관');axes[2].set_xlabel('64-beat 구간 번호')
    axes[2].scatter(np.flatnonzero(~np.isfinite(r)),np.zeros((~np.isfinite(r)).sum()),marker='x',color='#888',s=35,label='계산 불가 (0으로 대체한 값 아님)')
    axes[2].set_xlim(-.6,len(r)-.4);axes[2].legend(frameon=False,fontsize=9)
    axes[2].set_title(f'구간별 빠르기–강약 관계 → 평균 |r| = {values[1]:.3f}, 관계 변동 SD = {values[2]:.3f}',loc='left',fontsize=15)
    fig.suptitle('맞혀볼 측정치는 실제 feature에서 이렇게 계산한다',fontsize=21,fontweight='bold')
    fig.text(.5,.925,selected['key']+'  ·  평가 작품 이름순 첫 작품의 중앙 연주 (성능으로 선정하지 않음)',ha='center',fontsize=10)
    fig.text(.5,.015,'32-beat: 최소 25쌍 / 64-beat: 최소 51쌍  ·  짧은 마지막 구간 제외  ·  잔차는 원본 crescendo·accelerando와 같지 않음',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.90));save(fig,out,'00_target_example')
    write_csv(out/'example_blocks.csv',entries)
    (out/'example_selection.json').write_text(json.dumps(dict(selected_key=selected['key'],selection='lexical first eligible test piece, median eligible recording',targets=dict(zip(TARGETS,values))),ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,default=ROOT/'classicfy-ai/analysis/metadata_information_design')
    parser.add_argument('--data',type=Path,default=ROOT.parent/'datasets')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/metadata_information_results')
    args=parser.parse_args();setup_style()
    protocol=json.loads((args.design/'protocol.json').read_text())
    execution=json.loads((args.out/'execution.json').read_text())
    for name,expected in execution['csv_sha256'].items():
        if sha256(args.out/name)!=expected:raise ValueError('Verified result changed')
    metrics=read_csv(args.out/'metrics.csv');bootstrap=read_csv(args.out/'bootstrap.csv')
    controls=read_csv(args.out/'controls.csv');works=read_csv(args.out/'per_work_metrics.csv')
    effects=control_effects(metrics,controls);write_csv(args.out/'control_effects.csv',effects)
    draw_errors(metrics,args.out);draw_primary(bootstrap,args.out);draw_controls(bootstrap,effects,args.out)
    draw_work_results(works,args.out);draw_example(args.design,protocol,args.data,args.out)
    (args.out/'figure_manifest.json').write_text(json.dumps(dict(code_sha256=sha256(__file__),execution_sha256=sha256(args.out/'execution.json'),
        file_sha256={p.name:sha256(p) for p in args.out.iterdir() if p.suffix in ('.png','.svg','.pdf')},
        auxiliary_sha256={name:sha256(args.out/name) for name in ('control_effects.csv','example_blocks.csv','example_selection.json')}),indent=2)+'\n')


if __name__=='__main__':main()
