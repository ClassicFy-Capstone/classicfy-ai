"""Generate compact, reproducible validation curves and comparison tables."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    protocol=read(args.run/'protocol.json')
    final=read(args.run/'final_report.json')
    eras=list(final['eras'])
    fig,axes=plt.subplots(len(eras),2,figsize=(12,10),layout='constrained')
    rows=[]
    for row,era in enumerate(eras):
        selection=read(args.run/era/'selection.json')
        for weight in protocol['lambda_candidates']:
            trial=args.run/era/f'lambda_{weight}'
            label=f'lambda ceiling {weight}'
            if (trial/'equivalence.json').exists():label+=' (cap-equivalent reuse)'
            base=read(trial/'baseline.json');history=read(trial/'history.json')
            x=[0]+[h['epoch'] for h in history]
            rec=[1.]+[h['validation']['reconstruction']['loss']/base['reconstruction']['loss'] for h in history]
            g=[1.]+[h['validation']['geometry']['score']/base['geometry']['score'] for h in history]
            for ax,values in zip(axes[row],(rec,g)):
                ax.plot(x,values,marker='.',label=label)
                if selection['config']['lambda_g']==weight:
                    ax.scatter([selection['best_epoch']],[values[selection['best_epoch']]],marker='*',s=120,zorder=5)
                ax.grid(alpha=.2);ax.set_xlabel('Epoch');ax.legend(fontsize=8)
            axes[row,0].axhline(1.05,color='red',linestyle='--',alpha=.5)
            axes[row,1].axhline(1.,color='gray',linestyle='--',alpha=.5)
        axes[row,0].set_title(era+' | reconstruction / common')
        axes[row,1].set_title(era+' | G objective / common')
        test=read(args.run/era/'test.json')
        prod=final['eras'][era]
        baseline,selected=test['common'],test['selected']
        rows.append(dict(era=era,clips=prod['clips'],selection_epoch=selection['best_epoch'],
            lambda_ceiling=selection['config']['lambda_g'],production_epoch=prod['production_epoch'],
            test_reconstruction_before=baseline['reconstruction']['loss'],
            test_reconstruction_after=selected['reconstruction']['loss'],
            test_rec_change_percent=100*(selected['reconstruction']['loss']/baseline['reconstruction']['loss']-1),
            test_g_before=baseline['geometry']['score'],test_g_after=selected['geometry']['score'],
            test_g_change_percent=100*(selected['geometry']['score']/baseline['geometry']['score']-1),
            test_ranking_before=baseline['geometry']['ranking_accuracy'],
            test_ranking_after=selected['geometry']['ranking_accuracy'],
            test_spearman_before=baseline['geometry']['spearman'],test_spearman_after=selected['geometry']['spearman'],
            test_guards=test['guard'],production_guards=prod['guard']))
    fig.suptitle('Validation / common baseline | stars: selected checkpoints',fontsize=14)
    fig.savefig(args.run/'validation_curves.png',dpi=150);plt.close(fig)
    (args.run/'comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    final['heldout_comparison']=dict(
        scope='Separate train-only-pretrained evaluation checkpoints; NOT independent tests of the all-data production weights',
        uncertainty_assessment='Point estimates in this section; paired work intervals, when available, are in loss_contribution_analysis',rows=rows)
    final['geometry_point_estimates']={r['era']:
        ('improved_on_both_G_objective_and_ranking' if r['test_g_after']<r['test_g_before']
         and r['test_ranking_after']>=r['test_ranking_before'] else 'improvement_not_reproduced_on_test') for r in rows}
    (args.run/'final_report.json').write_text(json.dumps(final,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
