"""Standalone figures for measured dataset differences and time-unit conversion."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from _support import setup_fonts, clean_svg_outputs


def read(name):
    return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))


def main():
    setup_fonts()
    plt.rcParams.update({'axes.unicode_minus':False,
                         'svg.fonttype':'path','pdf.fonttype':42,'font.size':11})
    gaps={r['channel']:r for r in read('feature_gap_summary.csv')if r['field']=='raw_mean'}
    domain={r['stage']:r for r in read('domain_separability.csv')if r['features']=='all_channels'}
    fig,(left,right)=plt.subplots(1,2,figsize=(12.7,5.1),gridspec_kw={'width_ratios':[1.15,1]})
    fig.subplots_adjust(left=.05,right=.96,top=.7,bottom=.26,wspace=.32)
    fig.text(.05,.93,'시간축을 맞춰도 남는 ATEPP·ASAP 특징 차이',fontsize=21,weight='bold',color='#20334D')
    fig.text(.05,.85,'공통 134작품의 자료를 사용 · 분포 비교는 양쪽에 연주가 2개 이상 있는 작품만 포함',fontsize=10.5,color='#5E6D80')
    left.axis('off')
    left.set_title('4분음표 단위 원시 특징의 대표값',loc='left',pad=20,fontsize=13,weight='bold')
    names=[('dynamics','Dynamics (velocity/127)'),('articulation','Articulation (log₂ 비율)'),
           ('pedal_depth','페달 깊이 (0–1)'),('pedal_down_ratio','페달 on 비율 (0–1)'),
           ('pedal_changes','페달 전환 수 / 4분음표')]
    cells=[[label,f"{float(gaps[c]['asap_work_median']):.3f}",f"{float(gaps[c]['atepp_work_median']):.3f}"]for c,label in names]
    table=left.table(cellText=cells,colLabels=['특징','ASAP','ATEPP'],colWidths=[.62,.19,.19],
                     bbox=[0,-.02,1,1],cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(10.5)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor('#EAF0F7'if r==0 else('#F7F9FC'if r%2 else'white'))
        if r==0:cell.get_text().set_weight('bold')
        if c==0:cell.get_text().set_ha('left')
    right.set_title('특징으로 데이터 출처를 구분한 AUC',loc='left',pad=20,fontsize=13,weight='bold')
    values=[float(domain[s]['heldout_work_macro_auc'])for s in ['raw','standardized']]
    lows=[v-float(domain[s]['work_bootstrap_lower'])for v,s in zip(values,['raw','standardized'])]
    highs=[float(domain[s]['work_bootstrap_upper'])-v for v,s in zip(values,['raw','standardized'])]
    right.bar([0,1],values,width=.54,color=['#768BA6','#2A78C8'],zorder=3)
    right.errorbar([0,1],values,yerr=[lows,highs],fmt='none',ecolor='#233852',capsize=5,zorder=4)
    for i,v in enumerate(values):right.text(i,.08,f'{v:.3f}',ha='center',fontsize=16,color='white',weight='bold',zorder=5)
    right.axhline(.5,color='#8E9AAD',ls='--',lw=1,zorder=2)
    right.text(1.45,.51,'무작위 0.5',ha='right',fontsize=9,color='#6E7B8E')
    right.set(ylim=(0,1.03),xlim=(-.6,1.6),xticks=[0,1],xticklabels=['공통 패턴 제거 전','제거 + 표준화 후'],yticks=[0,.25,.5,.75,1])
    right.tick_params(length=0,labelsize=10)
    right.grid(axis='y',color='#E7EDF4',zorder=0)
    for spine in right.spines.values():spine.set_visible(False)
    fig.text(.05,.15,'표: 작품별 연주 중앙값을 구한 뒤 작품 간 중앙값. 특징별 80–91작품; 동일 녹음의 전사 오차를 측정한 수치는 아님.',fontsize=9.4,color='#5E6D80')
    fig.text(.05,.095,'AUC: 79작품 · ASAP 355연주 / ATEPP 839연주 · 작품을 분리한 5-fold 평가. 오차선은 작품 bootstrap 95% 구간.',fontsize=9.4,color='#5E6D80')
    fig.text(.05,.04,'연주자·판본·정렬·기록 방식이 함께 다름. AUC는 데이터셋 차이를 뜻하며 오토인코더의 추천 성능이나 전사 오류율이 아님.',fontsize=9.4,color='#5E6D80')
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'feature_gap_summary.{ext}',dpi=220)
    plt.close(fig)
    s=json.loads((OUT/'all_asap_conversion_summary.json').read_text())
    fig,ax=plt.subplots(figsize=(11.2,3.6))
    fig.subplots_adjust(left=.18,right=.96,top=.68,bottom=.28)
    fig.text(.05,.9,'ASAP의 시간축을 4분음표 단위로 변환',fontsize=20,weight='bold',color='#20334D')
    fig.text(.05,.79,f"정렬된 1,036개 중 동일 MIDI 복제 1개 제외 → {s['converted']:,}개 변환 성공",fontsize=11,color='#5E6D80')
    total=s['native_intervals']
    parts=[s['native_shorter_intervals'],s['native_one_quarter_intervals'],s['native_longer_intervals']]
    start=0
    for value,color,label in zip(parts,['#E5AF56','#2A78C8','#8B72BB'],['4분음표보다 짧음','4분음표 1개','4분음표보다 김']):
        width=100*value/total
        ax.barh(1,width,left=start,color=color,height=.38,label=label)
        ax.text(start+width/2,1,f'{width:.1f}%',ha='center',va='center',color='white',weight='bold')
        start+=width
    ax.barh(0,100,height=.38,color='#2A78C8')
    ax.text(50,0,'모든 구간을 4분음표 1개로 통일',ha='center',va='center',color='white',weight='bold')
    ax.set(yticks=[1,0],yticklabels=['기존 annotation beat','새 quarter-note grid'],xlim=(0,100),xticks=[])
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False,fontsize=9)
    inferred=100*s['inferred_quarter_boundaries']/s['quarter_boundaries']
    fig.text(.05,.055,f"새 경계의 {inferred:.1f}%는 annotation 사이의 시각 보간 · 반복·특수 구간은 mask로 제외 · 특징값은 MIDI 이벤트에서 재계산",fontsize=9.3,color='#5E6D80')
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'time_axis_conversion.{ext}',dpi=220)
    plt.close(fig)
    clean_svg_outputs(OUT)


if __name__=='__main__':main()
