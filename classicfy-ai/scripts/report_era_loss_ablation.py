"""Plot loss attribution with work-cluster uncertainty, without selecting models."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from features.note_performance import R_CHANNELS
from embedding.note_ae_loss import RECONSTRUCTION_GROUPS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();out=args.run/'ablation'
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    report=read(out/'report.json')
    if report['status']!='complete':raise ValueError('Ablation has not finished')
    conditions=report['protocol']['conditions']
    labels=['Common\n(frozen)','Rec + preserve','G + preserve','Rec + G\n+ preserve']
    contrasts=['joint_minus_common_frozen','joint_minus_reconstruction_only','joint_minus_g_only']
    contrast_labels=['Joint - common','Joint - Rec','Joint - G']
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    compact={}
    for row,(era,result) in enumerate(report['eras'].items()):
        points=[];rows={}
        for condition in conditions:
            metrics=result['results'][condition]['test']
            points.append(100*metrics['geometry']['ranking_accuracy'])
            rows[condition]=dict(ranking_accuracy_percent=points[-1],G_score=metrics['geometry']['score'],
                G_distance_pair_mse=metrics['geometry']['pair_mse'],distance_spearman=metrics['geometry']['spearman'],
                reconstruction_loss=metrics['reconstruction']['loss'],z_anchor_mse=metrics['embedding']['z_anchor_mse'],
                effective_rank=metrics['embedding']['effective_rank'],
                reconstruction_group_losses=dict(zip(RECONSTRUCTION_GROUPS,metrics['reconstruction']['group_losses'])),
                standardized_note_channel_mse=dict(zip(R_CHANNELS,metrics['reconstruction']['note_weighted_channel_mse'])),
                preservation_guard=True if condition=='common_frozen' else result['common_preservation_guards'][condition]['test']['passed'])
        axes[row,0].scatter(range(4),points,s=65,c=['gray','#0072B2','#D55E00','#7B3294'])
        for x,value in enumerate(points):axes[row,0].annotate(f'{value:.2f}',(x,value),xytext=(0,8),textcoords='offset points',ha='center')
        axes[row,0].set(xticks=range(4),xticklabels=labels,ylim=(min(points)-5,max(points)+5),
                        ylabel='Fixed-triplet ranking accuracy (%)',title=era)
        axes[row,0].grid(axis='y',alpha=.2)
        deltas={}
        for y,contrast in enumerate(contrasts):
            comparison=result['comparisons']['test'][contrast]
            stat=comparison['ranking_work_bootstrap'];point=stat['work_macro_delta']
            low,high=stat['familywise95_approx_for9_contrasts'];lo,hi=stat['ci95']
            color='#0072B2' if stat['direction']=='improvement_supported' else '#D55E00' if stat['direction']=='degradation_supported' else '#555555'
            axes[row,1].hlines(y,low,high,color='#999999',linewidth=2)
            axes[row,1].hlines(y,lo,hi,color=color,linewidth=5)
            axes[row,1].scatter([point],[y],color=color,zorder=3,s=45)
            deltas[contrast]=dict(micro_delta_pp=comparison['micro_ranking_delta_pp'],work_macro_delta_pp=point,
                work_macro_ci95=stat['ci95'],work_macro_familywise95_approx=stat['familywise95_approx_for9_contrasts'],
                uncertainty_direction=stat['direction'],anchor_works=stat['work_count'],
                G_score_change_percent=comparison['G_score_change_percent'],
                distance_error=comparison['distance_error_work_bootstrap'],
                reconstruction=comparison['reconstruction_work_bootstrap'])
        axes[row,1].axvline(0,color='black',linestyle='--',alpha=.5)
        axes[row,1].set(yticks=range(3),yticklabels=contrast_labels,ylim=(-.5,2.5),
                        xlabel='Equal-work ranking change (percentage points)',title='Paired work bootstrap; candidate pool fixed')
        axes[row,1].grid(axis='x',alpha=.2)
        compact[era]=dict(epochs=result['epochs'],test_conditions=rows,test_contrasts=deltas)
    axes[0,1].legend(handles=[Line2D([0],[0],color='#555555',lw=5,label='95% interval'),
                              Line2D([0],[0],color='#999999',lw=2,label='Familywise 95%, 9 contrasts')],fontsize=8)
    fig.suptitle('Fine-tuning loss ablation | post-hoc, one training seed, fixed horizon',fontsize=14)
    fig.savefig(out/'loss_contribution.png',dpi=150);plt.close(fig)
    summary=dict(scope='Fine-tuning contribution after common reconstruction pretraining; safeguards retained in every trained condition',
        distinction='Left/point accuracy is triplet-micro; confidence intervals use equal anchor-work means, so point estimates can differ',
        limitations=report['limitation'],practical_significance_threshold=None,eras=compact)
    (out/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    final=read(args.run/'final_report.json')
    final['loss_contribution_analysis']=dict(report=str(out/'report.json'),comparison=str(out/'comparison.json'),
                                           plot=str(out/'loss_contribution.png'),post_hoc=True,eras=compact)
    (args.run/'final_report.json').write_text(json.dumps(final,indent=2,ensure_ascii=False),encoding='utf-8')
    for era,r in compact.items():
        print(era)
        for name,metrics in r['test_conditions'].items():print(name,metrics)
        for name,delta in r['test_contrasts'].items():print(name,delta['micro_delta_pp'],delta['work_macro_familywise95_approx'],delta['uncertainty_direction'])


if __name__=='__main__':main()
