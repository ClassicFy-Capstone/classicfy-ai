"""Beethoven 31-2 pilot: observed repeat visits and existing Dynamics cache."""

import argparse
from collections import Counter,defaultdict
from dataclasses import asdict,replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from preprocessing import ASAPLoader,load_musicxml,load_match,load_midi,load_midi_score_timeline
from preprocessing.score_occurrence_bridge import build_score_occurrence_bridge
from features import extract_dynamics
from validate_score_beat_bridge import rows,sha,write,note_rows,plot_overlay

WORK='Beethoven/Piano_Sonatas/31-2'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asap-root',type=Path,default=ROOT.parent/'datasets/ASAP')
    parser.add_argument('--nasap-root',type=Path,default=ROOT.parent/'datasets/nASAP')
    parser.add_argument('--inventory',type=Path,default=ROOT/'classicfy-ai/analysis/score_structure_inputs')
    parser.add_argument('--cache',type=Path,default=ROOT.parent/'datasets/feature_normalization_standardized.npz')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/score_occurrence_bridge')
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Choose an empty --out directory')
    args.out.mkdir(parents=True,exist_ok=True)
    inputs={}
    for prefix,folder in [('ASAP',args.asap_root),('nASAP',args.nasap_root)]:
        filename='works.csv' if prefix=='ASAP' else 'nasap_works.csv'
        path=folder/WORK/'xml_score.musicxml'
        saved=next(r for r in rows(args.inventory/filename) if r['xml_score']==WORK+'/xml_score.musicxml')
        if sha(path)!=saved['xml_sha256']:raise ValueError('XML differs from frozen inventory')
        inputs[prefix+'/'+WORK+'/xml_score.musicxml']=sha(path)
    score=load_musicxml(args.nasap_root/WORK/'xml_score.musicxml')
    original=load_musicxml(args.asap_root/WORK/'xml_score.musicxml')
    if tuple(replace(n,xml_id=None) for n in score.notes)!=original.notes:
        raise ValueError('ASAP/nASAP written notes differ beyond IDs')
    fixed=sorted([r for r in rows(args.inventory/'performances.csv') if r['piece']==WORK+'/midi_score.mid'],key=lambda r:r['key'])
    if len(fixed)!=6:raise ValueError('Expected the fixed six performances')
    keys={r['key'] for r in fixed}
    normalized={}
    with np.load(args.cache,allow_pickle=False) as cache:
        manifest=json.loads(str(cache['manifest']))
        for i,record in enumerate(manifest['records']):
            if record['channel']=='dynamics' and record['key'] in keys:
                if record['key'] in normalized:raise ValueError('Duplicate normalized record')
                normalized[record['key']]=dict(metadata=record,**{name:cache[f'{i}/{name}'].copy() for name in
                    ('score_beats','raw_values','raw_mask','common_values','common_mask','common_support',
                     'relative_values','relative_mask','standardized_values','standardized_mask')})
    if set(normalized)!=keys:raise ValueError('Normalized cache does not cover the fixed cohort')
    inputs['cache/feature_normalization_standardized.npz']=sha(args.cache)
    for name in ('performances.csv','works.csv','nasap_works.csv'):
        inputs['inventory/'+name]=sha(args.inventory/name)
    inputs['ASAP/'+WORK+'/midi_score.mid']=sha(args.asap_root/WORK/'midi_score.mid')
    for prefix,folder,files in [('ASAP',args.asap_root,('metadata.csv','asap_annotations.json')),
                              ('nASAP',args.nasap_root,('metadata.csv',))]:
        for name in files:inputs[prefix+'/'+name]=sha(folder/name)
    loader=ASAPLoader(args.asap_root,args.nasap_root)
    midi=load_midi_score_timeline(args.asap_root/WORK/'midi_score.mid')
    visits,notes,beats,features,coverage=[],[],[],[],[]
    plot_features=defaultdict(list)
    by_written={n.note_id:n for n in score.notes}
    for row in fixed:
        sample=loader.get_sample(row['key'])
        if not sample.aligned or sample.robust_note_alignment is not True:raise ValueError('Unaligned pilot input')
        if sha(sample.note_alignment_path)!=row['note_alignment_sha256']:raise ValueError('Match differs from inventory')
        for prefix,path,folder in [('ASAP',sample.performance_path,args.asap_root),('nASAP',sample.note_alignment_path,args.nasap_root)]:
            inputs[prefix+'/'+path.relative_to(folder.resolve()).as_posix()]=sha(path)
        saved=normalized[row['key']]
        np.testing.assert_array_equal(saved['score_beats'],sample.score_beats)
        grid=hashlib.sha256(np.asarray(sample.score_beats).tobytes()+json.dumps(sample.score_beat_types).encode()).hexdigest()[:12]
        if grid!=row['grid']:raise ValueError('Frozen feature grid changed')
        if saved['metadata']['score_types']!=sample.score_beat_types:raise ValueError('Feature beat types changed')
        raw=extract_dynamics(load_midi(sample.performance_path),sample.performance_beats)
        np.testing.assert_allclose(saved['raw_values'],raw.sequence.values,rtol=0,atol=1e-12,equal_nan=True)
        np.testing.assert_array_equal(saved['raw_mask'],raw.sequence.mask)
        scale=saved['metadata']['scale']['value']
        mask=saved['standardized_mask']
        np.testing.assert_allclose(saved['standardized_values'][mask],
            (saved['raw_values'][mask]-saved['common_values'][mask])/scale,rtol=0,atol=1e-12)
        result=build_score_occurrence_bridge(score,load_match(sample.note_alignment_path),midi,
            sample.score_beats,sample.performance_beats,sample.score_beat_types)
        pair_mask=result.adjacent_pair_mask(mask)
        visits.extend(dict(key=row['key'],**asdict(v)) for v in result.visits)
        by_note={n.note_id:n for n in result.note_occurrences}
        for link in result.bridge.note_links:
            note=by_note[link.note_id]
            notes.append(dict(key=row['key'],**asdict(link),written_note_id=note.written_note_id,
                written_xml_id=note.written_xml_id,visit_index=note.visit_index,visit_number=note.visit_number,
                source_occurrence_label=note.source_occurrence_label,written_onset_quarters=note.written_onset_quarters))
        for interval,visit_index,boundary,written in zip(result.bridge.intervals,result.interval_visits,
                result.interval_break_before,result.interval_written_quarters):
            visit=result.visits[visit_index] if visit_index is not None else None
            i=interval.index
            beats.append(dict(key=row['key'],**asdict(interval),visit_index=visit_index,
                written_measure_index=visit.measure_index if visit else None,
                visit_number=visit.visit_number if visit else None,
                written_start_quarters=written[0] if written else None,
                written_stop_quarters=written[1] if written else None,break_before=boundary))
            features.append(dict(key=row['key'],beat_index=i,visit_index=visit_index,
                written_measure_index=visit.measure_index if visit else None,
                visit_number=visit.visit_number if visit else None,break_before=boundary,
                bridge_valid=interval.valid,dynamics_raw=saved['raw_values'][i],
                dynamics_raw_mask=bool(saved['raw_mask'][i]),
                dynamics_common=saved['common_values'][i],common_mask=bool(saved['common_mask'][i]),
                dynamics_standardized=saved['standardized_values'][i],dynamics_mask=bool(mask[i]),scale=scale,
                combined_mask=interval.valid and bool(mask[i]),
                adjacent_pair_valid=pair_mask[i]))
            plot_features[row['key']].append(dict(beat_index=i,standardized=saved['standardized_values'][i],
                standardized_mask=str(bool(mask[i]))))
        coverage.append(dict(key=row['key'],grid=row['grid'],original_intervals=len(result.bridge.intervals),
            valid_intervals=sum(i.valid for i in result.bridge.intervals),visits=len(result.visits),
            rest_bracketed=sum(v.status=='rest_bracketed' for v in result.visits),
            repeat_jumps=sum(v.transition=='repeat_jump' for v in result.visits),
            ending_skips=sum(v.transition=='ending_skip' for v in result.visits),
            valid_dynamics=int(mask.sum()),note_statuses=json.dumps(dict(Counter(l.status for l in result.bridge.note_links)),sort_keys=True)))
        print(row['key'],coverage[-1],flush=True)
    raw_matrix=np.array([normalized[r['key']]['raw_values'] for r in fixed])
    raw_valid=np.array([normalized[r['key']]['raw_mask'] for r in fixed])
    common=np.array([np.median(raw_matrix[raw_valid[:,i],i]) if raw_valid[:,i].any() else np.nan
                     for i in range(raw_matrix.shape[1])])
    for saved in normalized.values():
        valid=saved['common_mask']
        np.testing.assert_allclose(saved['common_values'][valid],common[valid],rtol=0,atol=1e-12)
    for name,data in [('xml_notes.csv',note_rows(score)),('measure_visits.csv',visits),('note_occurrences.csv',notes),
                      ('beat_map.csv',beats),('feature_join.csv',features),('coverage.csv',coverage)]:
        write(args.out/name,data)
    breaks=[i for i,flag in enumerate(result.interval_break_before) if flag]
    view_notes=[]
    for link in result.bridge.note_links:
        origin=by_written[by_note[link.note_id].written_note_id]
        view_notes.append(replace(origin,note_id=link.note_id,xml_id=link.xml_id,
            onset_quarters=Fraction(str(link.score_onset_quarters))))
    display=[replace(i,measure_number=f'{result.visits[j].measure_number}({result.visits[j].visit_number})')
             if j is not None else i for i,j in zip(result.bridge.intervals,result.interval_visits)]
    detail=plot_overlay(replace(score,notes=tuple(view_notes)),display,plot_features,[],args.out,
        title_prefix='Beethoven 31-2',detail_indices=(max(0,breaks[0]-8),min(len(display),breaks[0]+16)),
        transition_indices=breaks)
    protocol=dict(work=WORK,performances=6,inputs=inputs,scope='Observed single-part repeat and first/second ending pilot',
        source_selection='Smallest match-note count of the four fixed repeat-score candidates; all six fixed performances',
        unfolding='Score-side IDs and within-measure positions, validated against score MIDI and repeat/ending signs',
        rest_policy='Only rest-only measures bracketed by exact neighboring positions/durations are inferred',
        break_policy='Repeat jumps and ending skips block adjacent differences; masked source values remain masked',
        original_break_indices=breaks,figure_detail_indices=detail,
        cache_policy='Read existing normalized Dynamics only; raw/mask re-extracted, common and fixed-scale formula checked; no refit or cache writes',
        cache_provenance=manifest['provenance'],ppq=midi.ppq,
        time_policy='Score seconds snapped to nearest MIDI tick (<=half tick); original annotation seconds and feature arrays preserved',
        limitations=['One repeated score only','No phrase/tension inference','Unknown transitions are masked; nested/general navigation not established'])
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.out/'summary.json').write_text(json.dumps(dict(work=WORK,coverage=coverage),indent=2)+'\n')
    code=[Path(__file__),ROOT/'classicfy-ai/src/preprocessing/score_occurrence_bridge.py',
          ROOT/'classicfy-ai/src/preprocessing/score_beat_bridge.py',
          ROOT/'classicfy-ai/scripts/validate_score_beat_bridge.py',ROOT/'classicfy-ai/scripts/verify_score_beat_bridge.py',
          ROOT/'classicfy-ai/scripts/verify_score_occurrence_bridge.py',ROOT/'classicfy-ai/src/preprocessing/musicxml_loader.py',
          ROOT/'classicfy-ai/src/preprocessing/note_alignment.py',ROOT/'classicfy-ai/src/preprocessing/asap_loader.py',
          ROOT/'classicfy-ai/src/preprocessing/midi_loader.py',ROOT/'classicfy-ai/src/features/dynamics.py',
          ROOT/'classicfy-ai/src/features/beat_grid.py']
    (args.out/'provenance.json').write_text(json.dumps(dict(code={p.relative_to(ROOT).as_posix():sha(p) for p in code},
        results={p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}),indent=2)+'\n')


if __name__=='__main__':
    main()
