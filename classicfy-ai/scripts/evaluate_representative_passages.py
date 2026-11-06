"""Post-hoc overlap with transferred theme annotations; never feeds selection.

Known references are incomplete. Theme containment is distinct from recovery of
an entire theme group. Rejected/failed v2 outputs score zero in unconditional
results and are also reported with accepted-only metrics and explicit coverage.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pretty_midi

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
from preprocessing.theme_candidates import interval_iou


def overlap(interval,references):
    if interval is None:return dict(iou=0.,inside_fraction=0.,reference_recall=0.,iou_hit=False,inside_80=False)
    a,b=interval
    if not np.isfinite([a,b]).all() or b<=a:raise ValueError('invalid_evaluation_interval')
    refs=sorted(references)
    merged=[]
    for c,d in refs:
        if merged and c<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],d)
        else:merged.append([c,d])
    intersection=sum(max(0.,min(b,d)-max(a,c)) for c,d in merged)
    scores=[interval_iou(a,b,c,d) for c,d in refs]
    index=int(np.argmax(scores));c,d=refs[index]
    iou=float(scores[index]);fraction=float(intersection/(b-a))
    return dict(iou=iou,inside_fraction=fraction,reference_recall=float(max(0.,min(b,d)-max(a,c))/(d-c)),
                iou_hit=iou>=.5,inside_80=fraction>=.8)


def summarize(frame):
    output=[]
    measures=['iou','inside_fraction','reference_recall','iou_hit','inside_80']
    for (task,method,split),g in frame.groupby(['reference_task','method','historical_split']):
        # One recording hash gets at most one weight within each work/method.
        g=g.drop_duplicates(['composition_id','source_sha256'])
        work=g.groupby('composition_id')[measures+['available']].mean()
        rng=np.random.default_rng(20261106)
        boot=rng.choice(work.iou_hit.to_numpy(),size=(2000,len(work)),replace=True).mean(1)
        accepted=g[g.available]
        accepted_work=accepted.groupby('composition_id')[measures].mean()
        output.append(dict(reference_task=task,method=method,historical_split=split,
              performances=len(g),works=len(work),available_performances=int(g.available.sum()),
              accepted_only_works=len(accepted_work),
              performance_coverage=float(g.available.mean()),work_macro_coverage=float(work.available.mean()),
              unconditional_work_macro={k:float(work[k].mean()) for k in measures},
              accepted_only_work_macro={k:float(accepted_work[k].mean()) for k in measures} if len(accepted_work) else None,
              iou_hit_work_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist()))
    return output


def evaluate(root):
    workspace=HERE.parents[2]
    if not (root/'summary.json').exists():raise ValueError('Run must finish before evaluation')
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    data=pd.read_csv(root/'segments.csv',dtype={'perf_id':str,'composition_id':str}).fillna('')
    gold_root=workspace/'outputs/atepp_theme_extraction'
    gold=pd.read_csv(gold_root/'segments.csv',dtype={'perf_id':str,'composition_id':str})
    primary=pd.read_csv(gold_root/'primary_segments.csv',dtype={'perf_id':str,'composition_id':str})
    audit=pd.read_csv(HERE.parent/'analysis/atepp_research/midi_audit.csv',dtype={'perf_id':str,'composition_id':str}).fillna('').set_index('perf_id')
    source_coverage=[]
    annotation_dir=workspace/'annotations/bpsd/Beethoven_Piano_Sonata_Dataset_v2/2_Annotations/ann_score_structureFine'
    for work,g in audit.groupby('composition_id'):
        row=g.iloc[0]
        if 'Beethoven' not in row.composer or not re.search(r':\s*(?:I\.|1\.)\s',row.track):continue
        match=re.search(r'Op\.\s*(\d+)\s*,?\s*(?:No\.\s*(\d+))?',row.track)
        if not match:continue
        name=f'Beethoven_Op{int(match.group(1)):03d}'+(f'No{match.group(2)}' if match.group(2) else '')+'-01'
        path=annotation_dir/f'{name}.csv'
        if not path.exists():continue
        labels=pd.read_csv(path,sep=';').structure.str.split(':').str[-1]
        has_theme=bool(labels.isin(['FirstGroup','SecondGroup','Theme']).any())
        available=gold[gold.composition_id==work]
        available_primary=primary[primary.composition_id==work]
        source_coverage.append(dict(composition_id=work,bpsd_id=name,performances=len(g),
            source_has_theme_labels=has_theme,available_theme_reference_performances=available.perf_id.nunique(),
            available_first_theme_reference_performances=available_primary.perf_id.nunique(),
            reference_status='available_partial_transfer' if len(available) else
                ('no_reliable_transferred_timestamps' if has_theme else 'source_has_no_supported_theme_label')))
    pd.DataFrame(source_coverage).to_csv(root/'theme_reference_coverage.csv',index=False,encoding='utf-8-sig')
    split_path=workspace/'outputs/atepp_all_theme_candidates/evaluation_split.json'
    splits=json.loads(split_path.read_text(encoding='utf-8'))['composition_split'] if split_path.exists() else {}
    selected=data.set_index('perf_id');primary=primary.set_index('perf_id')
    rows=[];work_cache={};excluded=[]
    for pid,reference in gold.groupby('perf_id'):
        if pid not in selected.index:
            excluded.append(dict(perf_id=pid,reason='not_in_v2_run'));continue
        row=selected.loc[pid];work=str(row.composition_id)
        source=audit.loc[pid]
        successful=row.status=='exported'
        interval=(float(row.start_seconds),float(row.end_seconds)) if successful else None
        if work not in work_cache:
            path=root/'works'/f'{work}.json'
            work_cache[work]=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        mapping=work_cache[work].get('mappings',{}).get(pid,[])
        alternatives=[(m['start_seconds'],m['end_seconds']) for m in mapping if m['alignment_accepted']]
        representation_first=(mapping[0]['start_seconds'],mapping[0]['end_seconds']) if mapping and mapping[0]['alignment_accepted'] else None
        try:
            original=pretty_midi.PrettyMIDI(source.local_midi)
            notes=[n for i in original.instruments if not i.is_drum for n in i.notes]
            first=min(n.start for n in notes);last=max(n.end for n in notes)
            baseline=(first,min(first+30.,last))
            length=interval[1]-interval[0] if interval else 25.
            matched_first=(first,min(first+length,last))
        except Exception:
            baseline=matched_first=None;first=last=0.;length=25.
        references={'any_available_theme':list(zip(reference.start_seconds,reference.end_seconds))}
        if pid in primary.index:
            p=primary.loc[pid];references['first_exposition_theme']=[(float(p.start_seconds),float(p.end_seconds))]
        historical=splits.get(work,'supplementary_work')
        for task,refs in references.items():
            base=dict(perf_id=pid,composition_id=work,source_sha256=source.sha256,
                      reference_task=task,v2_status=row.status,historical_split=historical,
                      composer=source.composer,track=source.track,selection_method=row.selection_method)
            methods=[('v2_selected',interval),('representation_only_first_candidate',representation_first),
                     ('first_30_seconds',baseline),('duration_matched_first',matched_first)]
            for method,prediction in methods:
                metric=overlap(prediction,refs)
                if task=='first_exposition_theme' and prediction is not None:
                    metric.update(start_error_seconds=abs(prediction[0]-refs[0][0]),end_error_seconds=abs(prediction[1]-refs[0][1]))
                rows.append(dict(**base,method=method,available=prediction is not None,
                    prediction_start_seconds=prediction[0] if prediction else None,
                    prediction_end_seconds=prediction[1] if prediction else None,**metric))
            # Two distinct oracles: the best IoU and best containment may be
            # different candidates. Never call these actual selected accuracy.
            scores=[overlap(p,refs) for p in alternatives]
            for criterion,name in [('iou','candidate_oracle_iou'),('inside_fraction','candidate_oracle_containment')]:
                best=max(scores,key=lambda m:m[criterion]) if scores else overlap(None,refs)
                rows.append(dict(**base,method=name,available=bool(scores),**best))
    if not rows:raise ValueError('No reference overlap; evaluation unavailable')
    frame=pd.DataFrame(rows)
    # Keep all-work descriptive results plus the old split as historical
    # diagnostics. These are not claimed as a newly sealed unseen test set.
    aggregate=summarize(frame)
    all_frame=frame.copy();all_frame.historical_split='all_reference_works'
    aggregate+=summarize(all_frame)
    frame.to_csv(root/'theme_evaluation_performance.csv',index=False,encoding='utf-8-sig')
    unique=frame.drop_duplicates(['reference_task','method','composition_id','source_sha256'])
    measures=['iou','inside_fraction','reference_recall','iou_hit','inside_80','available']
    per_work=unique.groupby(['reference_task','method','composition_id'])[measures].mean()
    per_work['performances']=unique.groupby(['reference_task','method','composition_id']).size()
    per_work.reset_index().to_csv(root/'theme_evaluation_work.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(excluded,columns=['perf_id','reason']).to_csv(root/'theme_evaluation_excluded.csv',index=False,encoding='utf-8-sig')
    result=dict(v2_corpus_summary=summary,reference_source='BPSD v2 labels previously transferred to ATEPP using automatic alignment',
       reference_is_independent_human_timestamps=False,reference_coverage_incomplete=True,
       reference_sha256={name:hashlib.sha256((gold_root/name).read_bytes()).hexdigest() for name in ['segments.csv','primary_segments.csv']},
       annotated_source_works=len(source_coverage),source_works_with_theme_labels=sum(r['source_has_theme_labels'] for r in source_coverage),
       source_reference_performances=gold.perf_id.nunique(),source_primary_performances=len(primary),
       evaluated_reference_performances=frame.perf_id.nunique(),excluded_reference_performances=len(excluded),
       annotations_used_for_v2_selection=False,parameters_changed_after_evaluation=False,
       evaluation_is_new_sealed_holdout=False,
       caveat='Theme overlap is not a test of perceptual musical identity. Any-theme results are match rates to available incomplete intervals. Rejections count as zero in unconditional metrics; accepted-only metrics must be read with coverage. Historical splits were already inspected in previous development.',
       definitions=dict(iou='best overlap with one annotated theme interval',inside_fraction='fraction of preview duration within the union of available theme intervals',
                        reference_recall='coverage of the reference theme with best IoU',inside_80='at least 80 percent of preview lies in available theme regions',
                        candidate_oracle='upper bound across accepted mapped candidates, never actual selected performance'),
       metrics=aggregate)
    (root/'theme_evaluation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# ATEPP v2 전수 추출 및 기존 주제 주석 일치도','',
       '주석은 v2 구간 선택 이후 평가에만 사용했다. 대표 구간을 고르는 목표와 최초 주제군 전체를 복원하는 목표는 다르다. 아래 수치는 이용 가능한 주석과의 일치율이며, 음악적 아이덴티티·청취 효용의 정확도가 아니다.','',
       '## 전수 결과','',f"- 작품/악장 {summary['works']:,}개, 연주 {summary['performances']:,}개.",
       f"- 상태: {summary['statuses']}",f"- 대표 후보 {summary['candidate_count']:,}개.",'',
       '## 전체 참조 작품에서의 일치도','',
       '작품 안에서 연주를 평균한 뒤 작품별 동일 가중치로 평균했다. 거부·실패는 아래 전체 분모 결과에서 0으로 계산했다. 실제 추출본만의 결과는 JSON의 accepted_only_work_macro에 별도로 있다.','',
       '| 참조 | 방법 | 작품/연주 | 추출·대응 가능 비율 | IoU≥0.5 | 미리듣기 80% 이상이 주제 안 | 평균 IoU |',
       '|---|---|---:|---:|---:|---:|---:|']
    for m in aggregate:
        if m['historical_split']!='all_reference_works':continue
        scores=m['unconditional_work_macro']
        lines.append(f"| {m['reference_task']} | {m['method']} | {m['works']}/{m['performances']} | {m['performance_coverage']:.1%} | {scores['iou_hit']:.1%} | {scores['inside_80']:.1%} | {scores['iou']:.3f} |")
    lines+=['','- first_exposition_theme: 최초 제시부 제1주제군 또는 변주곡 Theme.','- any_available_theme: 이전 전이에서 확보한 제1·제2주제군과 반복·재현부 주제군. 참조가 누락된 위치의 실제 주제를 놓친 것으로 단정할 수 없다.','- candidate_oracle: 상위 후보 중 주석과 가장 잘 맞는 후보를 사후 선택한 상한이며 실제 자동 선택 결과가 아니다.','- duration_matched_first: 실제 v2 길이를 사용하며, v2 거부·실패로 길이를 사용할 수 없을 때는 25초를 사용한다.','',
       '## 평가 범위와 한계','',
       f"- 원본 형식 주석이 연결되는 {len(source_coverage)}작품 중 {sum(r['source_has_theme_labels'] for r in source_coverage)}작품에 지원하는 주제군 라벨이 있다. 상세 제외 사유는 theme_reference_coverage.csv에 있다.",
       f"- 실제 시간 참조가 있는 연주 {gold.perf_id.nunique()}개 중 이번 실행에 포함된 {frame.perf_id.nunique()}개를 평가했다.",
       '- 참조 시간은 기존 자동 악보·연주 정렬로 옮긴 값이며 독립 수작업 타임스탬프가 아니다. 신뢰할 수 있는 전이가 없는 연주는 평가 불가로 분리하고 임의 시각을 만들지 않았다.',
       '- 이전 development/test 분할도 JSON에 유지했지만 이미 관찰한 자료이므로 새로 봉인한 미관측 시험 성능이라고 주장하지 않는다.',
       '- 짧은 미리듣기가 긴 주제부 안에 완전히 포함되어도 IoU는 낮을 수 있다. 따라서 전체 주제군 복원과 주제 안에 위치하는지를 구분한다.',
       '- v2 특징은 학습된 스타일 임베딩이 아닌 대리 특징이다. 모든 결과는 청취 검수가 필요하며 종지 검출 정확도를 이 평가로 판단하지 않는다.','',
       '## 파일','',
       '- segments.csv: 전체 처리 상태와 선택 시각.','- candidates.csv, works/: 공통 후보·연주별 대응·선택 근거.','- review_queue.csv: 대응 거부 및 처리 실패.','- theme_evaluation_performance.csv: 연주별 평가와 기준선.','- theme_evaluation.json: 전체 분모·성공본 조건부 지표, 과거 분할별 지표 및 bootstrap 구간.','- theme_reference_coverage.csv: 원본 주석과 평가 가능한 시간 참조의 범위.']
    lines+=['','- representation_only_first_candidate: 스타일 비교를 생략한 대표성 1순위. 진단용이며 v2 출력물을 교체하지 않았다.',
            '- theme_evaluation_work.csv: 작품별 일치도와 대응 가능 비율.']
    (root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    evaluate(parser.parse_args().root.resolve())
