"""Supplementary evaluation, never changes candidates, selection, or cutoffs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
from preprocessing.theme_candidates import interval_iou


def main(root):
    evaluation=json.loads((root/'evaluation.json').read_text(encoding='utf-8'))
    primary=pd.read_csv(root/'evaluation_performance.csv',dtype={'perf_id':str,'composition_id':str})
    automatic=primary[(primary.method=='automatic')&(primary.split=='test')]
    threshold=evaluation['medium_confidence_cutoff']
    kept=automatic[automatic.heuristic_confidence>=threshold]
    evaluation['selective_test_metrics']=dict(cutoff_fitted_on_development=threshold,
        evaluated_test_performances=len(automatic),kept_test_performances=len(kept),
        coverage=float(len(kept)/max(len(automatic),1)),
        work_macro_iou_05_precision=float(kept.groupby('composition_id').success_iou_05.mean().mean()) if len(kept) else None,
        caveat='Heuristic cutoff only; additional quality/domain/consensus flags can further reduce export confidence')
    bywork=pd.read_csv(root/'evaluation_work.csv',dtype={'composition_id':str})
    test=bywork[bywork.split=='test'].pivot(index='composition_id',columns='method',values='success_iou_05').dropna()
    if len(test):
        delta=(test.automatic-test.first_30_seconds).to_numpy()
        rng=np.random.default_rng(20261105)
        bootstrap=np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1)
        evaluation['paired_test_accuracy_difference_vs_first30']=dict(works=len(delta),difference=float(delta.mean()),
                       work_bootstrap_95ci=np.quantile(bootstrap,[.025,.975]).tolist())
    # Additional question: does the selected candidate match ANY explicitly
    # annotated theme occurrence? Keep it separate from first-theme accuracy.
    gold=pd.read_csv(HERE.parents[2]/'outputs/atepp_theme_extraction/segments.csv',dtype={'perf_id':str,'composition_id':str})
    selection=json.loads((root/'selection_after_development_calibration.json').read_text(encoding='utf-8'))
    splits=json.loads((root/'evaluation_split.json').read_text(encoding='utf-8'))['composition_split']
    rows=[]
    for pid,g in gold.groupby('perf_id'):
        path=root/'predictions'/f'{pid}.json'
        if not path.exists() or pid not in selection:continue
        prediction=json.loads(path.read_text(encoding='utf-8'))
        candidates=prediction['candidates'];chosen=candidates[selection[pid]['candidate_index']]
        references=list(zip(g.start_seconds,g.end_seconds))
        def score(c):return max(interval_iou(c['start_seconds'],c['end_seconds'],a,b) for a,b in references)
        work=str(g.composition_id.iloc[0])
        rows.append(dict(perf_id=pid,composition_id=work,source_sha256=prediction['source_sha256'],
                         split=splits.get(work,'supplementary_work_without_primary_reference'),
                         selected_iou=score(chosen),success_iou_05=score(chosen)>=.5,
                         top5_oracle_iou=max(map(score,candidates))))
    supplementary=pd.DataFrame(rows).drop_duplicates(['source_sha256','composition_id'])
    supplementary.to_csv(root/'evaluation_any_theme.csv',index=False,encoding='utf-8-sig')
    evaluation['any_annotated_theme_supplementary']=[]
    evaluation['supplementary_reference_caveat']='Available transferred intervals are incomplete; this is a match rate to available references, not accuracy against all true themes. An unmatched candidate may fall in a missing reference interval.'
    for split,g in supplementary.groupby('split'):
        macro=g.groupby('composition_id')[['selected_iou','success_iou_05','top5_oracle_iou']].mean()
        evaluation['any_annotated_theme_supplementary'].append(dict(split=split,works=len(macro),performances=len(g),
              mean_iou=float(macro.selected_iou.mean()),available_reference_match_rate_iou_05=float(macro.success_iou_05.mean()),
              top5_oracle_iou=float(macro.top5_oracle_iou.mean())))
    (root/'evaluation.json').write_text(json.dumps(evaluation,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(evaluation,indent=2,ensure_ascii=False))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
