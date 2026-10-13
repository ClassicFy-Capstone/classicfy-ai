"""Connect the frozen 55-grid/570-performance cohort to score structure."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from preprocessing import ASAPLoader,load_musicxml,load_match,load_midi_score_timeline
from preprocessing.score_measure_bridge import build_score_measure_bridge
from prepare_metadata_retrieval_experiment import reconstruct
from validate_score_beat_bridge import rows,sha,write,note_rows

CHANNELS=('tempo','rubato','dynamics','articulation','pedal_depth','pedal_down_ratio','pedal_changes')


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class CompressedTable:
    def __init__(self,path):
        self.stream=gzip.open(path,'wt',encoding='utf-8',newline='')
        self.writer=None

    def extend(self,records):
        for record in records:
            if self.writer is None:
                self.writer=csv.DictWriter(self.stream,fieldnames=list(record))
                self.writer.writeheader()
            self.writer.writerow(record)

    def close(self):
        self.stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent/'datasets')
    parser.add_argument('--inventory',type=Path,default=ROOT/'classicfy-ai/analysis/score_structure_inputs')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/score_structure_full')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/score_structure_full_run01')
    args=parser.parse_args()
    for folder in (args.out,args.artifacts):
        if folder.exists() and any(folder.iterdir()):raise ValueError('Choose empty output directories: '+str(folder))
        folder.mkdir(parents=True,exist_ok=True)
    fixed=rows(args.inventory/'performances.csv')
    if len(fixed)!=570 or len({(r['piece'],r['grid']) for r in fixed})!=55:
        raise ValueError('Expected frozen 55-grid/570-performance cohort')
    inputs={}
    def freeze(path):
        path=path.resolve();digest=sha(path);inputs[str(path)]=digest;return digest
    for name in ('works.csv','nasap_works.csv','performances.csv'):freeze(args.inventory/name)
    legacy_path=ROOT/'classicfy-ai/analysis/metadata_information_design/protocol.json'
    legacy=json.loads(legacy_path.read_text());freeze(legacy_path)
    freeze(args.data/'temporal_embedding_cluster_run01/dataset.json')
    freeze(args.data/'feature_normalization_cluster_raw.npz')
    sequences={}
    for restored,snapshot in reconstruct(args.data,legacy):sequences.update(restored)
    if set(sequences)!={r['key'] for r in fixed}:raise ValueError('Existing normalized feature cohort differs')
    annotations=json.loads((args.data/'ASAP/asap_annotations.json').read_text())
    for folder,name in [('ASAP','metadata.csv'),('ASAP','asap_annotations.json'),('nASAP','metadata.csv')]:freeze(args.data/folder/name)
    xml_inventory={('ASAP' if filename=='works.csv' else 'nASAP',r['xml_score']):r['xml_sha256']
                   for filename in ('works.csv','nasap_works.csv') for r in rows(args.inventory/filename)}
    scores={};midis={};xml_audit=[]
    for piece in sorted({r['piece'] for r in fixed}):
        work=str(Path(piece).parent);loaded={}
        for folder in ('ASAP','nASAP'):
            path=args.data/folder/work/'xml_score.musicxml';digest=freeze(path)
            if digest!=xml_inventory[folder,work+'/xml_score.musicxml']:raise ValueError('Frozen XML changed: '+str(path))
            loaded[folder]=load_musicxml(path)
            xml_audit.append(dict(piece=piece,source=folder,sha256=digest,parts=len(loaded[folder].part_names),
                notes=len(loaded[folder].notes),measures=len(loaded[folder].measures),
                diagnostics=json.dumps(dict(Counter(d.code for d in loaded[folder].diagnostics)),sort_keys=True)))
        if tuple(replace(n,xml_id=None,notation_xml=()) for n in loaded['nASAP'].notes)!=tuple(
            replace(n,xml_id=None,notation_xml=()) for n in loaded['ASAP'].notes):
            raise ValueError('Written ASAP/nASAP structured note fields differ beyond IDs/notation markup: '+piece)
        notation_differences=sum(a.notation_xml!=b.notation_xml for a,b in zip(loaded['ASAP'].notes,loaded['nASAP'].notes))
        for audit in xml_audit[-2:]:audit['notation_markup_differences']=notation_differences
        if loaded['nASAP'].measures!=loaded['ASAP'].measures:raise ValueError('Written measures differ: '+piece)
        scores[piece]=loaded['nASAP'];path=args.data/'ASAP'/piece;freeze(path)
        midis[piece]=load_midi_score_timeline(path)
    loader=ASAPLoader(args.data/'ASAP',args.data/'nASAP')
    tables={name:CompressedTable(args.artifacts/(name+'.csv.gz')) for name in
            ('xml_notes','measure_visits','match_coordinates','note_occurrences','beat_map')}
    for piece,score in scores.items():tables['xml_notes'].extend(dict(piece=piece,**r) for r in note_rows(score))
    coverage=[];arrays={};geometries={};feature_digests={};note_statuses=Counter();visit_statuses=Counter()
    transitions=Counter();examples={};failures=[]
    try:
        for index,row in enumerate(fixed):
            key=row['key'];sample=loader.get_sample(key);score=scores[row['piece']];midi=midis[row['piece']]
            if not sample.aligned or sample.robust_note_alignment is not True:raise ValueError('Unaligned frozen sample')
            if freeze(sample.note_alignment_path)!=row['note_alignment_sha256']:raise ValueError('Frozen match changed: '+key)
            freeze(sample.performance_path)
            grid=hashlib.sha256(np.asarray(sample.score_beats).tobytes()+json.dumps(sample.score_beat_types).encode()).hexdigest()[:12]
            if grid!=row['grid']:raise ValueError('Original feature grid changed: '+key)
            values,feature_mask=sequences[key];length=len(sample.score_beats)-1
            if values.shape!=(7,length) or feature_mask.shape!=(length,):raise ValueError('Original feature length differs')
            if not np.all(np.diff(sample.performance_beats)>0):raise ValueError('Expected frozen positive performance intervals')
            feature_digests[key]=hashlib.sha256(values.tobytes()+feature_mask.tobytes()).hexdigest()
            alignment=load_match(sample.note_alignment_path)
            source_notes=sorted((s.note_id,s.pitch,s.onset_beats,s.offset_beats,s.attributes)
                                for s in [n for n,p in alignment.matches]+list(alignment.deletions))
            sources_by_id={s.note_id:s for s in [n for n,p in alignment.matches]+list(alignment.deletions)}
            raw_map=annotations[key]['downbeats_score_map']
            geometry=hashlib.sha256(json.dumps([row['piece'],grid,source_notes,sample.score_downbeats,raw_map],sort_keys=True).encode()).hexdigest()
            bridge_valid=np.zeros(length,bool);break_before=np.zeros(length,bool)
            try:
                if geometry not in geometries:
                    geometries[geometry]=build_score_measure_bridge(score,alignment,midi,sample.score_beats,
                        sample.performance_beats,sample.score_beat_types,sample.score_downbeats,raw_map)
                result=geometries[geometry]
                bridge_valid=np.logical_and.reduce([np.array([b.valid for b in part.bridge.intervals]) for pid,part in result.parts])
                break_before=np.logical_or.reduce([np.array(part.interval_break_before) for pid,part in result.parts])
                tables['match_coordinates'].extend(dict(key=key,**check) for check in result.match_coordinate_checks)
                statuses=Counter();local_visits=Counter();local_transitions=Counter();part_coverage={}
                for pid,part in result.parts:
                    part_coverage[pid]=sum(b.valid for b in part.bridge.intervals)
                    tables['measure_visits'].extend(dict(key=key,part_id=pid,**asdict(v)) for v in part.visits)
                    by_note={n.note_id:n for n in part.note_occurrences}
                    local_visits.update(v.status for v in part.visits);local_transitions.update(v.transition for v in part.visits)
                    for link in part.bridge.note_links:
                        item=asdict(link);source=sources_by_id.get(link.match_id)
                        item['raw_match_onset_beats']=source.onset_beats if source else None
                        item['raw_match_duration_beats']=source.duration_beats if source else None
                        item.update(asdict(by_note[link.note_id]));item['part_id']=pid;item['key']=key
                        tables['note_occurrences'].extend([item]);statuses[link.status]+=1
                    for interval,vi,cut,written in zip(part.bridge.intervals,part.interval_visits,
                                                      part.interval_break_before,part.interval_written_quarters):
                        item=asdict(interval);i=interval.index
                        item.update(key=key,part_id=pid,visit_index=vi,break_before=cut,
                            performance_start_seconds=sample.performance_beats[i],
                            performance_stop_seconds=sample.performance_beats[i+1],
                            written_start_quarters=written[0] if written else None,
                            written_stop_quarters=written[1] if written else None)
                        tables['beat_map'].extend([item])
                note_statuses.update(statuses);visit_statuses.update(local_visits);transitions.update(local_transitions)
                status='complete' if bridge_valid.all() else 'partial'
                error=''
                # One deterministic first performance per work, chosen before feature inspection.
                examples.setdefault(row['piece'],dict(key=key,geometry=geometry,index=index))
                details=dict(schedule_method=result.schedule_method,parts=len(result.parts),
                    midi_onset_support=result.pitch_onset_support,midi_onset_total=result.pitch_onset_total,
                    inferred_boundary_indices=json.dumps(result.inferred_downbeat_indices),
                    part_valid_intervals=json.dumps(part_coverage,sort_keys=True),
                    note_statuses=json.dumps(dict(statuses),sort_keys=True),
                    visit_statuses=json.dumps(dict(local_visits),sort_keys=True),
                    transitions=json.dumps(dict(local_transitions),sort_keys=True))
            except ValueError as exc:
                status='failed';error=str(exc);failures.append(dict(key=key,error=error))
                details=dict(schedule_method='',parts=0,midi_onset_support=0,midi_onset_total=0,
                    inferred_boundary_indices='[]',part_valid_intervals='{}',note_statuses='{}',visit_statuses='{}',transitions='{}')
            combined=feature_mask&bridge_valid
            pairs=np.zeros(length,bool);pairs[1:]=combined[:-1]&combined[1:]&~break_before[1:]
            irregular=np.array([t=='bR' for t in sample.score_beat_types])
            for i in range(1,length):
                if irregular[i-1:i+2].any():pairs[i]=False
            for name,array in [('values',values),('feature_mask',feature_mask),('bridge_mask',bridge_valid),
                ('combined_mask',combined),('adjacent_pair_mask',pairs),('break_before',break_before),
                ('score_beats',np.asarray(sample.score_beats)),('performance_beats',np.asarray(sample.performance_beats))]:arrays[f'{index}/{name}']=array
            coverage.append(dict(key=key,piece=row['piece'],grid=grid,status=status,error=error,
                original_intervals=length,valid_intervals=int(bridge_valid.sum()),valid_ratio=float(bridge_valid.mean()),
                existing_feature_intervals=int(feature_mask.sum()),combined_intervals=int(combined.sum()),
                adjacent_pairs=int(pairs.sum()),breaks=int(break_before.sum()),**details))
            if (index+1)%50==0:print(f'Connected {index+1}/{len(fixed)}; failed={len(failures)}; geometries={len(geometries)}',flush=True)
    finally:
        for table in tables.values():table.close()
    arrays['keys']=np.array([r['key'] for r in fixed]);arrays['channels']=np.array(CHANNELS)
    np.savez_compressed(args.artifacts/'feature_join.npz',**arrays)
    write(args.out/'coverage.csv',coverage);write(args.out/'xml_audit.csv',xml_audit)
    works=[]
    for piece in sorted(scores):
        rs=[r for r in coverage if r['piece']==piece];total=sum(r['original_intervals'] for r in rs)
        works.append(dict(piece=piece,grid=rs[0]['grid'],performances=len(rs),complete=sum(r['status']=='complete' for r in rs),
            partial=sum(r['status']=='partial' for r in rs),failed=sum(r['status']=='failed' for r in rs),
            original_intervals=total,valid_intervals=sum(r['valid_intervals'] for r in rs),
            valid_ratio=sum(r['valid_intervals'] for r in rs)/total,
            existing_feature_intervals=sum(r['existing_feature_intervals'] for r in rs),
            combined_intervals=sum(r['combined_intervals'] for r in rs)))
    write(args.out/'works.csv',works)
    summary=dict(performances=len(fixed),work_grids=len(works),xml_inputs=len(xml_audit),
        statuses=dict(Counter(r['status'] for r in coverage)),original_intervals=sum(r['original_intervals'] for r in coverage),
        valid_intervals=sum(r['valid_intervals'] for r in coverage),
        existing_feature_intervals=sum(r['existing_feature_intervals'] for r in coverage),
        combined_intervals=sum(r['combined_intervals'] for r in coverage),
        geometries=len(geometries),note_statuses=dict(note_statuses),visit_statuses=dict(visit_statuses),
        transitions=dict(transitions),failures=failures)
    dump(args.out/'summary.json',summary)
    protocol=dict(scope='Frozen 55-grid/570-performance cohort; score structure connection only',
        inputs=inputs,channels=CHANNELS,artifact_directory=str(args.artifacts.resolve()),
        feature_policy='Reconstruct original seven-channel cohort with frozen collector/scales; preserve float64 values and shared original-index mask; no feature refit',
        feature_digests=feature_digests,source_collector_commit=legacy['learned_source_commit'],
        grid_policy='B-1 original intervals; combined mask intersects all parts and original shared feature mask',
        coordinate_policy='Downbeat map tested as zero-based written indices, combined ranges split only with exact duration brackets; global >=90% matched nongrace MIDI pitch/onset support',
        repeat_policy='Observed score-side occurrence route; one missing downbeat allowed only with unique bracket/pickup and MIDI support; source suffix kept separate from chronological visit number',
        partial_policy='Local match coordinate conflicts, unsupported transitions, missing MIDI anchors and part-specific uncertainties remain masked; no arbitrary time stretching',
        pair_policy='Block missing/invalid intervals, repeat/ending/coordinate cuts and bR neighbors',
        note_status_policy='verified means XML/match pitch, local onset translation and score MIDI onset/duration agreement; raw match duration remains in original beat units, not compared as quarters',
        limitations=['No phrase or tension extraction','No listening or retrieval evaluation','Coordinate coverage does not mean every note/duration is verified'],
        selected_examples=examples)
    dump(args.out/'protocol.json',protocol)
    from report_score_structure_full import report
    report(args.out,args.artifacts,coverage,works,examples,geometries,arrays)
    code=[Path(__file__),ROOT/'classicfy-ai/src/preprocessing/score_measure_bridge.py',
          ROOT/'classicfy-ai/scripts/report_score_structure_full.py',ROOT/'classicfy-ai/scripts/verify_score_structure_full.py',
          ROOT/'classicfy-ai/scripts/validate_score_beat_bridge.py',ROOT/'classicfy-ai/scripts/verify_score_beat_bridge.py',
          ROOT/'classicfy-ai/scripts/prepare_metadata_retrieval_experiment.py']
    code+=sorted((ROOT/'classicfy-ai/src/preprocessing').glob('*.py'))
    dump(args.out/'provenance.json',dict(code={str(p.relative_to(ROOT)):sha(p) for p in code},
        artifacts={p.name:sha(p) for p in args.artifacts.iterdir()},
        results={p.name:sha(p) for p in args.out.iterdir() if p.is_file()}))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
