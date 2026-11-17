"""Plot the validation-only reconstruction coefficient tradeoff."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();out=args.run/'weak_reconstruction_validation'
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    if report['status']!='complete':raise ValueError('Validation sweep unfinished')
    coefficients=report['protocol']['coefficients'];fig,axes=plt.subplots(3,3,figsize=(15,11),layout='constrained')
    compact={}
    for row,(era,result) in enumerate(report['eras'].items()):
        base=result['validation_conditions']['0.0'];values=[[],[],[]];rows={}
        for alpha in coefficients:
            m=result['validation_conditions'][str(alpha)]
            pair_change=100*(m['geometry']['pair_mse']/base['geometry']['pair_mse']-1)
            rec_change=100*(m['reconstruction']['loss']/base['reconstruction']['loss']-1)
            rank_change=100*(m['geometry']['ranking_accuracy']-base['geometry']['ranking_accuracy'])
            for series,value in zip(values,(pair_change,rec_change,rank_change)):series.append(value)
            rows[str(alpha)]=dict(pair_MSE_change_percent=pair_change,Rec_change_percent=rec_change,
                ranking_change_pp=rank_change,ranking_percent=100*m['geometry']['ranking_accuracy'],
                G_score=m['geometry']['score'],pair_MSE=m['geometry']['pair_mse'],
                reconstruction_loss=m['reconstruction']['loss'],z_drift=m['embedding']['z_anchor_mse'])
        for col,(series,label) in enumerate(zip(values,('Pair MSE change (%)','Reconstruction change (%)','Ranking change (pp)'))):
            ax=axes[row,col];ax.plot(range(5),series,marker='o');ax.axhline(0,color='gray',ls='--')
            selected=coefficients.index(result['recommended_alpha']);ax.scatter([selected],[series[selected]],marker='*',s=150,zorder=5)
            ax.set(xticks=range(5),xticklabels=[str(a) for a in coefficients],xlabel='Reconstruction coefficient',ylabel=label,title=era)
            ax.grid(alpha=.2)
        compact[era]=dict(epochs=result['epochs'],conditions=rows,recommended_alpha=result['recommended_alpha'],
            lowest_G_score_alpha=result['lowest_G_score_alpha'],useful_weak_alphas=result['useful_weak_alphas'],
            comparisons_to_protected_G=result['comparisons_to_protected_G'])
    fig.suptitle('Validation-only weak reconstruction sweep | protected G baseline | stars: usefulness rule',fontsize=14)
    fig.savefig(out/'validation_tradeoff.png',dpi=150);plt.close(fig)
    summary=dict(scope='Validation search; no new test evaluation or production replacement',
        points='Pair-weighted distance MSE; clip-weighted Rec; triplet-weighted ranking. Work-bootstrap contrasts are separate estimands.',
        limitations='One seed; prior test motivated search; validation intervals are exploratory and do not correct adaptive selection.',
        protocol=report['protocol'],eras=compact)
    (out/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print({era:dict(recommended_alpha=x['recommended_alpha'],lowest_G_score_alpha=x['lowest_G_score_alpha']) for era,x in compact.items()})


if __name__=='__main__':main()
