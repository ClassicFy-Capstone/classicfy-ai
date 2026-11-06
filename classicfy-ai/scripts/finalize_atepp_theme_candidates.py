"""Conservative domain flags and portable manifests for the completed corpus run."""
import json
from pathlib import Path
import re
import sys

import pandas as pd


def main(root):
    root=root.resolve()
    data=pd.read_csv(root/'segments.csv',dtype={'perf_id':str,'composition_id':str}).fillna('')
    moved=0
    for index,row in data.iterrows():
        if row.status!='exported':continue
        scope=('Beethoven' in row.composer and re.search('sonata',row.track,re.I)
               and re.search(r':\s*(?:I\.|1\.)\s',row.track))
        outside=row.method=='label_free_midi' and not bool(scope)
        data.loc[index,'outside_evaluated_repertoire']=outside
        if outside:
            reasons=[r for r in str(row.review_reasons).split(';') if r]
            if 'outside_evaluated_repertoire' not in reasons:reasons.append('outside_evaluated_repertoire')
            data.loc[index,'review_reasons']=';'.join(reasons)
            data.loc[index,'low_confidence']=True
            data.loc[index,'confidence_level']='low'
            source=Path(row.clip_path).resolve()
            target=(root/'midi'/'low'/row.composition_id/f'{row.perf_id}.mid').resolve()
            if not source.is_relative_to(root) or not target.is_relative_to(root):
                raise ValueError('Output path escaped extraction directory')
            if source!=target:
                target.parent.mkdir(parents=True,exist_ok=True)
                source.replace(target);moved+=1
                data.loc[index,'clip_path']=str(target)
            # Export checks certify bytes/time, not review annotations. Rewrite
            # them only if the path/level changed; canonical review flags live
            # in segments.csv. Avoid thousands of unnecessary disk rewrites.
            if source!=target or row.confidence_level!='low':
                check=root/'export_checks'/f'{row.perf_id}.json'
                cached=json.loads(check.read_text(encoding='utf-8'))
                cached.update(clip_path=str(target),confidence_level='low',low_confidence=True,
                              outside_evaluated_repertoire=True,review_reasons=';'.join(reasons))
                check.write_text(json.dumps(cached,ensure_ascii=False,indent=2),encoding='utf-8')
        data.loc[index,'clip_relative_path']=str(Path(data.loc[index,'clip_path']).relative_to(root)).replace('\\','/')
    data.to_csv(root/'segments.csv',index=False,encoding='utf-8-sig')
    data.to_csv(root/'coverage.csv',index=False,encoding='utf-8-sig')
    success=data[data.status=='exported']
    success.to_csv(root/'successful_segments.csv',index=False,encoding='utf-8-sig')
    low=data[(data.status!='exported')|data.low_confidence.eq(True)]
    low.to_csv(root/'low_confidence_and_failures.csv',index=False,encoding='utf-8-sig')
    work_rows=[]
    for work,g in data.groupby('composition_id'):
        available=g[g.status=='exported']
        if 'source_quality' in available and available.source_quality.isin(['','high quality']).any():
            available=available[available.source_quality.isin(['','high quality'])]
        order=[c for c in ['composition_agreement','heuristic_confidence'] if c in available]
        if len(available) and order:available=available.sort_values(order,ascending=False)
        representative=available.iloc[0] if len(available) else None
        work_rows.append(dict(composition_id=work,composer=g.composer.iloc[0],track=g.track.iloc[0],
             performances=len(g),exported=int((g.status=='exported').sum()),
             low_confidence=int(g.low_confidence.eq(True).sum()),failures=int((g.status!='exported').sum()),
             annotation_backed=int((g.method=='annotation_transfer').sum()),
             median_agreement=float(pd.to_numeric(g.composition_agreement,errors='coerce').median()) if 'composition_agreement' in g else None,
             representative_perf_id=representative.perf_id if representative is not None else '',
             representative_clip=representative.clip_relative_path if representative is not None else ''))
    pd.DataFrame(work_rows).sort_values(['low_confidence','performances'],ascending=False).to_csv(root/'work_review_queue.csv',index=False,encoding='utf-8-sig')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    summary.update(confidence_levels=success.confidence_level.value_counts().to_dict(),
                   low_confidence_exported=int(success.low_confidence.eq(True).sum()),
                   confidence_domain_policy='All automatic candidates outside Beethoven piano sonata first movements are low confidence; no cross-composer validation available',
                   out_of_domain_candidates=int(success.outside_evaluated_repertoire.eq(True).sum()),
                   portable_manifest='successful_segments.csv:clip_relative_path')
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    config_path=root/'detector_config.json'
    if config_path.exists():
        config=json.loads(config_path.read_text(encoding='utf-8'))
        config.update(labels_used_in_candidate_generation=False,labels_used_for_development_ranking=True,
                      test_labels_used_for_ranking=False,
                      inference_definition='label-free candidate generation; subsequent ranking calibrated on development works only')
        config_path.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(moved_to_low_confidence=moved,summary=summary),ensure_ascii=False,indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
