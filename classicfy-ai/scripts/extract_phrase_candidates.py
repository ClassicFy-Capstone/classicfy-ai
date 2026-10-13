"""Audit all frozen scores and prepare three score-only phrase review pilots."""

import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
from dataclasses import asdict
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from preprocessing import load_musicxml,load_match,load_midi_score_timeline,ASAPLoader
from preprocessing.phrase_candidates import extract_phrase_candidates
from preprocessing.score_measure_bridge import build_score_measure_bridge
from validate_score_beat_bridge import rows,sha,write

PILOTS=('Bach/Fugue/bwv_848/midi_score.mid','Beethoven/Piano_Sonatas/31-2/midi_score.mid',
        'Mozart/Piano_Sonatas/8-1/midi_score.mid')


def serial(value):
    if isinstance(value,Fraction):return str(value)
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def dump(path,value):
    path.write_text(json.dumps(serial(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def choose_cases(piece,score,result):
    """Fixed cue-type coverage; no performance/features consulted."""
    selected=[];seen=set()
    tests=[('first_internal_voice_rest',lambda c:any(e.kind=='voice_rest' for e in c.cues)),
           ('first_paired_slur_end',lambda c:any(e.kind=='slur_end' for e in c.cues)),
           ('first_global_silence',lambda c:any(e.kind=='global_notated_silence' for e in c.cues)),
           ('first_harmonic_closure',lambda c:any(e.kind=='harmonic_closure_hint' for e in c.cues)),
           ('first_combined_evidence',lambda c:c.review_priority=='high_review')]
    by_note={n.note_id:n for n in score.notes}
    for reason,test in tests:
        candidate=next((c for c in result.candidates if test(c)),None)
        if candidate is None or candidate.candidate_id in seen:continue
        seen.add(candidate.candidate_id)
        ids=tuple(dict.fromkeys(n for cue in candidate.cues for n in cue.note_ids))
        anchors=[by_note[n] for n in ids]
        if not anchors:
            anchors=[min(score.notes,key=lambda n:abs(n.onset_quarters-candidate.position_quarters))]
        selected.append(dict(piece=piece,case_id='',candidate_id=candidate.candidate_id,selection_reason=reason,
            position_quarters=str(candidate.position_quarters),scope=candidate.scope,
            cue_kinds=','.join(sorted({e.kind for e in candidate.cues})),
            source_note_ids=json.dumps([n.note_id for n in anchors]),
            source_xml_ids=json.dumps([n.xml_id for n in anchors]),
            measures=json.dumps(sorted({n.measure_index for n in anchors})),status='pending_visual_review'))
    for slur in result.slurs:
        if slur.status.startswith('paired'):continue
        ids=[n for n in (slur.start_note_id,slur.stop_note_id) if n is not None]
        selected.append(dict(piece=piece,case_id='',candidate_id='',selection_reason='first_unusable_slur',
            position_quarters=str(slur.stop_onset_quarters or slur.start_quarters),scope=slur.part_id+':part',
            cue_kinds='slur_diagnostic:'+slur.status,source_note_ids=json.dumps(ids),
            source_xml_ids=json.dumps([by_note[n].xml_id for n in ids]),
            measures=json.dumps(sorted({by_note[n].measure_index for n in ids})),status='pending_visual_review'))
        break
    return selected


def map_boundary(candidate,score,part,ppq=480):
    """Map a pending written boundary to distinct observed measure visits.

    Left endpoint ownership comes from the source symbol. No boundary is made
    active for an embedding. All uncertain/provisional mappings are labelled.
    """
    by_note={n.note_id:n for n in score.notes}
    anchor_ids=[n for e in candidate.cues for n in e.note_ids if n in by_note]
    # Slur ends belong to the ending note, not the starting note's measure.
    slur_ends=[e.note_ids[-1] for e in candidate.cues if e.kind=='slur_end']
    anchor=by_note[slur_ends[0]] if slur_ends else by_note[anchor_ids[0]] if anchor_ids else None
    if anchor:
        indices={anchor.measure_index}
    else:
        indices={m.measure_index for m in score.measures if m.part_id==part[0] and
                 m.onset_quarters<=candidate.position_quarters<m.onset_quarters+m.duration_quarters}
    pid,result=part;intervals=result.bridge.intervals
    edges=[b.midi_start_quarters for b in intervals]+[intervals[-1].midi_stop_quarters]
    output=[]
    for visit in result.visits:
        if visit.measure_index not in indices:continue
        written_q=float(candidate.position_quarters)
        q=visit.score_start_quarters+written_q-visit.written_start_quarters
        within=visit.score_start_quarters-1e-10<=q<=visit.score_stop_quarters+1e-10
        nearby=min(range(len(edges)),key=lambda i:abs(edges[i]-q))
        error=abs(edges[nearby]-q)
        on_edge=error<=1/ppq+1e-4
        interval=bisect_right(edges,q)-1
        if on_edge:interval=nearby if nearby<len(intervals) else None
        elif not 0<=interval<len(intervals):interval=None
        mapping_status=('outside_observed_measure' if not within else 'unverified_measure' if
            visit.status not in ('match_midi_anchored','rest_bracketed') else 'outside_original_grid' if
            q<edges[0]-1e-10 or q>edges[-1]+1e-10 else 'interval_edge' if on_edge else 'inside_interval')
        output.append(dict(candidate_id=candidate.candidate_id,part_id=pid,visit_index=visit.index,
            written_measure_index=visit.measure_index,visit_number=visit.visit_number,
            source_occurrence_label=visit.source_occurrence_label,written_boundary_quarters=str(candidate.position_quarters),
            observed_score_quarters=q,original_edge_index=nearby if on_edge else None,
            original_interval_index=interval,edge_error_quarters=error,mapping_status=mapping_status,
            nearby_bridge_valid=interval is not None and intervals[interval].valid,
            structural_cut_nearby=any(v.break_before and abs(v.score_start_quarters-q)<=1/ppq+1e-4 for v in result.visits),
            eligible_for_embedding=False,boundary_status=candidate.status))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent/'datasets')
    parser.add_argument('--inventory',type=Path,default=ROOT/'classicfy-ai/analysis/score_structure_inputs')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/phrase_candidates')
    parser.add_argument('--artifacts',type=Path,help='Default: portable all_scores/ inside --out')
    parser.add_argument('--render',action='store_true',help='Requires optional score-render packages')
    args=parser.parse_args()
    if args.artifacts is None:args.artifacts=args.out/'all_scores'
    for directory in (args.out,args.artifacts):
        if directory.exists() and any(directory.iterdir()):raise ValueError('Choose empty output directory: '+str(directory))
        directory.mkdir(parents=True,exist_ok=True)
    inventory=rows(args.inventory/'works.csv');fixed=rows(args.inventory/'performances.csv')
    nasap_inventory={r['xml_score']:r for r in rows(args.inventory/'nasap_works.csv')}
    if len(inventory)!=55 or len(fixed)!=570:raise ValueError('Expected frozen full cohort')
    inputs={};summaries=[];pilot_scores={};pilot_results={};cases=[]
    for name in ('works.csv','nasap_works.csv','performances.csv'):inputs['inventory/'+name]=sha(args.inventory/name)
    with gzip.open(args.artifacts/'all_scores.jsonl.gz','wt',encoding='utf-8') as stream:
        for row in inventory:
            path=args.data/'nASAP'/row['xml_score'];digest=sha(path)
            if digest!=nasap_inventory[row['xml_score']]['xml_sha256']:raise ValueError('Frozen XML changed')
            inputs['nASAP/'+row['xml_score']]=digest
            score=load_musicxml(path);result=extract_phrase_candidates(score)
            encoded=dict(piece=row['piece'],**serial(asdict(result)))
            stream.write(json.dumps(encoded,ensure_ascii=False)+'\n')
            cue_counts=Counter(e.kind for c in result.candidates for e in c.cues)
            summary=dict(piece=row['piece'],grid=row['grid'],candidates=len(result.candidates),
                priorities=json.dumps(dict(Counter(c.review_priority for c in result.candidates)),sort_keys=True),
                slur_statuses=json.dumps(dict(Counter(s.status for s in result.slurs)),sort_keys=True),
                cue_counts=json.dumps(dict(cue_counts),sort_keys=True),diagnostics=len(result.diagnostics),
                confirmed_phrases=0)
            summaries.append(summary)
            if row['piece'] in PILOTS:
                pilot_scores[row['piece']]=score;pilot_results[row['piece']]=result
                cases.extend(choose_cases(row['piece'],score,result))
                dump(args.out/(Path(row['piece']).parent.name+'_'+row['piece'].split('/')[0]+'_candidates.json'),encoded)
            print(row['piece'],len(result.candidates),'candidates',flush=True)
    for i,c in enumerate(cases,1):c['case_id']=f'case_{i:02d}'
    write(args.out/'works.csv',summaries);write(args.out/'review_cases.csv',cases)
    mapped=[];slur_groups=[];visits=[]
    loader=ASAPLoader(args.data/'ASAP',args.data/'nASAP')
    annotations=json.loads((args.data/'ASAP/asap_annotations.json').read_text())
    for piece in PILOTS:
        score=pilot_scores[piece];result=pilot_results[piece]
        for j,slur in enumerate(result.slurs):
            slur_groups.append(dict(piece=piece,group_id=f'{piece}:slur:{j}',**serial(asdict(slur)),
                structure_type='slur_group',phrase_status='unconfirmed',eligible_for_embedding=False))
        reference_midi=load_midi_score_timeline(args.data/'ASAP'/piece)
        inputs['ASAP/'+piece]=sha(args.data/'ASAP'/piece)
        for row in [r for r in fixed if r['piece']==piece]:
            sample=loader.get_sample(row['key'])
            if sha(sample.note_alignment_path)!=row['note_alignment_sha256']:raise ValueError('Frozen match changed')
            inputs['nASAP/'+row['note_alignment']]=sha(sample.note_alignment_path)
            bridge=build_score_measure_bridge(score,load_match(sample.note_alignment_path),reference_midi,
                sample.score_beats,sample.performance_beats,sample.score_beat_types,
                sample.score_downbeats,annotations[row['key']]['downbeats_score_map'])
            for pid,part in bridge.parts:
                visits.extend(dict(key=row['key'],piece=piece,part_id=pid,**asdict(v)) for v in part.visits)
            for candidate in result.candidates:
                for part in bridge.parts:
                    if candidate.part_id is None or candidate.part_id==part[0]:
                        mapped.extend(dict(key=row['key'],piece=piece,scope=candidate.scope,**r)
                                      for r in map_boundary(candidate,score,part,reference_midi.ppq))
    write(args.out/'boundary_occurrences.csv',mapped);write(args.out/'slur_groups.csv',slur_groups)
    write(args.out/'pilot_measure_visits.csv',visits)
    for name in ('ASAP/metadata.csv','ASAP/asap_annotations.json','nASAP/metadata.csv'):
        inputs[name]=sha(args.data/name)
    dump(args.out/'summary.json',dict(work_grids=55,candidates=sum(s['candidates'] for s in summaries),
        pilot_works=PILOTS,pilot_performances=len({r['key'] for r in mapped}),review_cases=len(cases),
        boundary_occurrences=len(mapped),confirmed_phrases=0,active_embedding_boundaries=0,
        mapping_statuses=dict(Counter(r['mapping_status'] for r in mapped))))
    dump(args.out/'protocol.json',dict(version=1,inputs=inputs,pilot_works=PILOTS,
        selection='Three fixed works; first candidate per cue type and first unusable slur; no features inspected',
        policy=next(iter(pilot_results.values())).policy,
        artifact_location=({'relative_to_report':args.artifacts.relative_to(args.out).as_posix()}
                           if args.artifacts.is_relative_to(args.out) else {'requires_artifacts_argument':True}),
        renderer_requested=args.render,
        review_policy='Symbol/voice/score-context review; retain candidate != confirm musical phrase',
        missing_mode_policy='Keep both named major/minor hypotheses; no metadata or performance-derived key',
        mapping_policy='Written endpoints mapped separately per observed visit; keep original indices and pending status',
        limitations=['No complete phrase partition or hierarchy','Harmonic hints only cover complete root-position V-I sonorities; not a full cadence classifier',
                      'No phrase-based feature or embedding experiment yet']))
    if args.render:
        from render_phrase_review import render_review
        render_review(args.out,pilot_scores,cases)
    code=[Path(__file__),ROOT/'classicfy-ai/src/preprocessing/phrase_candidates.py',
          ROOT/'classicfy-ai/scripts/render_phrase_review.py',ROOT/'classicfy-ai/scripts/verify_phrase_candidates.py',
          ROOT/'classicfy-ai/src/preprocessing/musicxml_loader.py',ROOT/'classicfy-ai/src/preprocessing/score_measure_bridge.py',
          ROOT/'classicfy-ai/src/preprocessing/score_beat_bridge.py',ROOT/'classicfy-ai/src/preprocessing/score_occurrence_bridge.py']
    dump(args.out/'provenance.json',dict(code={p.relative_to(ROOT).as_posix():sha(p) for p in code},
        artifacts={p.name:sha(p) for p in args.artifacts.iterdir()},
        results={p.relative_to(args.out).as_posix():sha(p) for p in args.out.rglob('*') if p.is_file()}))


if __name__=='__main__':main()
