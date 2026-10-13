"""Connect written measures to original beat intervals using score evidence.

Downbeat map numbers are tested as zero-based written indices, independently
of printed measure numbers. Repeat routes come from full match occurrence IDs.
Every candidate schedule is checked against score MIDI pitch/onset anchors;
performance timing and feature values never select a mapping.
"""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite
from statistics import median
import re

from .score_beat_bridge import ScoreBeatBridge, ScoreBeatInterval, ScoreNoteLink, _tie_spans
from .score_occurrence_bridge import MeasureVisit, NoteOccurrence, ScoreOccurrenceBridge, _transition


@dataclass(frozen=True)
class MeasureBridgeResult:
    parts: tuple[tuple[str, ScoreOccurrenceBridge], ...]
    schedule_method: str
    pitch_onset_support: int
    pitch_onset_total: int
    inferred_downbeat_indices: tuple[int, ...]
    match_coordinate_checks: tuple[dict, ...]


def build_score_measure_bridge(score, alignment, midi, score_beats, performance_beats,
                               score_beat_types, score_downbeats, downbeat_map):
    if not (len(score_beats) == len(performance_beats) == len(score_beat_types) >= 2):
        raise ValueError('Expected equal aligned beat arrays')
    for values in (score_beats, performance_beats, score_downbeats):
        if any(not isfinite(t) or t < 0 for t in values) or any(a > b for a,b in zip(values,values[1:])):
            raise ValueError('Invalid beat seconds')
    if any(t not in ('b','db','bR') for t in score_beat_types):
        raise ValueError('Unknown score beat type')
    edges=tuple(round(float(midi.quarters_at_seconds(t))*midi.ppq)/midi.ppq for t in score_beats)
    downbeats=tuple(round(float(midi.quarters_at_seconds(t))*midi.ppq)/midi.ppq for t in score_downbeats)
    if any(a>=b for a,b in zip(downbeats,downbeats[1:])):
        raise ValueError('Score downbeats must be strictly increasing')
    xml={n.xml_id:n for n in score.notes if n.xml_id is not None}
    if len(xml)!=sum(n.xml_id is not None for n in score.notes):
        raise ValueError('Duplicate XML IDs')
    references={};groups=defaultdict(list)
    for source in [s for s,p in alignment.matches]+list(alignment.deletions):
        if source.note_id in references:
            raise ValueError('Duplicate full match ID')
        if source.note_id in xml:
            base,label=source.note_id,'exact'
        else:
            base,separator,label=source.note_id.rpartition('-')
            if not separator or not label.isdecimal() or int(label)<=0 or base not in xml:
                raise ValueError('Unknown match occurrence ID: '+source.note_id)
        note=xml[base]
        if note.pitch is None or source.pitch!=note.pitch.midi_pitch:
            raise ValueError('XML/match pitch mismatch: '+source.note_id)
        if not all(isfinite(q) for q in (source.onset_beats,source.offset_beats)):
            raise ValueError('Non-finite match score coordinate')
        references[source.note_id]=(note,label,source)
        groups[note.part_id,note.measure_index,label].append(source)
    midi_pitches=defaultdict(list)
    for note in midi.notes:
        if note.offset_quarters>note.onset_quarters:
            midi_pitches[note.pitch].append(note)
    for values in midi_pitches.values():values.sort(key=lambda n:n.onset_quarters)
    pitch_positions={p:[float(n.onset_quarters) for n in ns] for p,ns in midi_pitches.items()}
    tolerance=1/midi.ppq+1e-4

    def midi_candidates(pitch,q):
        positions=pitch_positions.get(pitch,[]);i=bisect_left(positions,q-tolerance);found=[]
        while i<len(positions) and positions[i]<=q+tolerance:
            found.append(midi_pitches[pitch][i]);i+=1
        return found

    all_parts=[];methods=set();total_support=total_anchors=0;inferred=set();coordinate_checks=[]
    for part_id,_ in score.part_names:
        measures=[m for m in score.measures if m.part_id==part_id]
        notes_by_measure=defaultdict(list)
        for note in score.notes:
            if note.part_id==part_id:notes_by_measure[note.measure_index].append(note)
        local={ (i,label):values for (pid,i,label),values in groups.items() if pid==part_id }
        repeated=any(mark.part_id==part_id and mark.kind in ('repeat','ending') for mark in score.barline_marks)
        if any(mark.part_id==part_id and mark.kind in ('segno','coda') for mark in score.barline_marks):
            raise ValueError('General score navigation is unsupported')
        if not repeated:
            if not isinstance(downbeat_map,list) or len(downbeat_map)!=len(downbeats):
                raise ValueError('Non-repeat score needs an equal-length downbeat map')
            starts={};indices=[]
            for k,(value,q) in enumerate(zip(downbeat_map,downbeats)):
                if isinstance(value,(int,float)) and not isinstance(value,bool) and isfinite(value) and value>=0 and int(value)==value:
                    members=[int(value)]
                elif isinstance(value,str) and re.fullmatch(r'\d+-\d+',value):
                    lo,hi=map(int,value.split('-'));members=list(range(lo,hi+1))
                    if hi<=lo:raise ValueError('Invalid combined downbeat range')
                    if hi>=len(measures):raise ValueError('Combined range outside written score')
                    duration=sum(float(measures[i].duration_quarters) for i in members)
                    if k+1>=len(downbeats) or abs(downbeats[k+1]-q-duration)>tolerance:
                        raise ValueError('Combined range lacks an exact duration bracket')
                else:raise ValueError('Unknown downbeat index')
                position=q
                for i in members:
                    starts[i]=position;indices.append(i)
                    if i!=members[0]:inferred.add(i)
                    if i<len(measures):position+=float(measures[i].duration_quarters)
            if len(set(indices))!=len(indices) or any(i>=len(measures) for i in indices):
                raise ValueError('Downbeat indices are duplicate or outside the written score')
            if any(a>=b for a,b in zip(indices,indices[1:])):
                raise ValueError('Non-repeat downbeat indices must be increasing')
            starts.setdefault(0,0.)
            # Missing written boundaries are candidates, never silently certified.
            for i in range(1,len(measures)):
                starts.setdefault(i,starts[i-1]+float(measures[i-1].duration_quarters))
            route=[]
            for i in range(len(measures)):
                labels=[label for index,label in local if index==i]
                if len(labels)>1:
                    raise ValueError('Unexpected repeated IDs in a non-repeat score')
                route.append((i,labels[0] if labels else '1'))
            positions=[starts[i] for i,label in route]
            inferred.update(i for i in starts if i not in indices)
            method='downbeat_index_midi_verified'
        else:
            observed=sorted(local,key=lambda k:min(s.onset_beats for s in local[k]))
            route=[];previous=-1
            for i,label in observed:
                missing=list(range(previous+1,i))
                if missing and all(notes_by_measure[j] and all(n.is_rest for n in notes_by_measure[j]) for j in missing):
                    route.extend((j,label) for j in missing)
                route.append((i,label));previous=i
            if len(route)==len(downbeats):
                positions=list(downbeats);method='observed_route_downbeats_midi_verified'
            elif len(route)==len(downbeats)+1:
                schedules=[]
                for k,(i,label) in enumerate(route):
                    if k==0:
                        duration=float(measures[i].duration_quarters)
                        if duration>=float(measures[i].nominal_duration_quarters or duration):continue
                        q=downbeats[0]-duration
                        if q<0:continue
                    elif k<len(downbeats):
                        previous_duration=float(measures[route[k-1][0]].duration_quarters)
                        current_duration=float(measures[i].duration_quarters)
                        if abs(downbeats[k]-downbeats[k-1]-previous_duration-current_duration)>tolerance:continue
                        q=downbeats[k-1]+previous_duration
                    else:continue
                    candidate=list(downbeats);candidate.insert(k,q)
                    support=sum(bool(midi_candidates(n.pitch.midi_pitch,start+float(n.measure_offset_quarters)))
                        for (idx,lab),start in zip(route,candidate) for n in notes_by_measure[idx]
                        if n.pitch is not None and not n.is_grace and (idx,lab) in local)
                    schedules.append((support,k,candidate))
                schedules.sort(key=lambda v:v[0],reverse=True)
                if not schedules or (len(schedules)>1 and schedules[0][0]==schedules[1][0]):
                    raise ValueError('Missing repeat downbeat has no unique MIDI-verified candidate')
                support,k,positions=schedules[0];inferred.add(k)
                method='observed_route_one_boundary_inferred_midi_verified'
            else:
                raise ValueError(f'Observed repeat route/downbeat count differs: {len(route)}/{len(downbeats)}')
        if any(a>=b for a,b in zip(positions,positions[1:])):
            raise ValueError('Proposed written measure schedule is not increasing')
        # Validate the schedule using XML positions and score MIDI, independently
        # of the match beat unit. No arbitrary within-measure time stretching.
        support=anchors=0
        for (i,label),start in zip(route,positions):
            for n in notes_by_measure[i]:
                if n.pitch is not None and not n.is_grace and any(
                    references[s.note_id][0].xml_id==n.xml_id for s in local.get((i,label),[])):
                    anchors+=1;support+=bool(midi_candidates(n.pitch.midi_pitch,start+float(n.measure_offset_quarters)))
        if len(score.part_names)==1 and (anchors<2 or support/anchors<.9):
            raise ValueError(f'Written measure schedule fails MIDI anchor support: {support}/{anchors}')
        total_support+=support;total_anchors+=anchors;methods.add(method)
        visits=[];occurrences=[];links=[];seen=Counter()
        spans,tie_heads,invalid_ties=_tie_spans(score)
        expanded=[dict(measure_index=i,start=start,stop=positions[j+1] if j+1<len(positions)
                       else start+float(measures[i].duration_quarters))
                  for j,((i,label),start) in enumerate(zip(route,positions))]
        for j,((i,label),start) in enumerate(zip(route,positions)):
            measure=measures[i];sources=local.get((i,label),[])
            source_by_xml={references[s.note_id][0].xml_id:s for s in sources}
            anchors_local=[(references[s.note_id][0],s) for s in sources if not references[s.note_id][0].is_grace]
            raw_unit_status='no_match_anchors'
            if anchors_local:
                options=[]
                for unit in (1.,2.,.5,4.,.25):
                    shifts=[s.onset_beats-unit*float(n.measure_offset_quarters) for n,s in anchors_local]
                    options.append((max(shifts)-min(shifts),unit,median(shifts)))
                spread,unit,raw_start=min(options,key=lambda v:(v[0],abs(v[1]-1)))
                raw_unit_status='local_match_unit_verified' if spread<=1e-4 else 'local_match_coordinate_conflict'
                if spread<=1e-4 and len({n.measure_offset_quarters for n,s in anchors_local})==1:
                    raw_unit_status='local_match_origin_only'
            else:unit,raw_start,spread=1.,0.,None
            count=sum(bool(midi_candidates(n.pitch.midi_pitch,start+float(n.measure_offset_quarters)))
                      for n,s in anchors_local)
            transition='start' if j==0 else _transition(score,expanded[j-1],expanded[j])
            rest=bool(notes_by_measure[i]) and all(n.is_rest for n in notes_by_measure[i])
            known_boundary=(not repeated and i in indices) or repeated
            status=('coordinate_conflict' if raw_unit_status=='local_match_coordinate_conflict' else
                    'transition_unverified' if transition.startswith('unresolved') else
                    'match_midi_anchored' if count else 'rest_bracketed' if rest and known_boundary else 'midi_unverified')
            seen[i]+=1
            coordinate_gap=(j>0 and abs(positions[j]-positions[j-1]-
                            float(measures[route[j-1][0]].duration_quarters))>tolerance)
            visit=MeasureVisit(j,i,measure.number,seen[i],label,start,expanded[j]['stop'],
                               float(measure.onset_quarters),count,status,transition,
                               transition not in ('start','consecutive') or coordinate_gap)
            visits.append(visit)
            coordinate_checks.append(dict(part_id=part_id,visit_index=j,measure_index=i,
                match_units_per_xml_quarter=unit if raw_unit_status=='local_match_unit_verified' else None,
                raw_match_measure_start=raw_start if anchors_local else None,
                raw_translation_spread=spread,status=raw_unit_status,
                distinct_xml_onsets=len({n.measure_offset_quarters for n,s in anchors_local}),
                coordinate_gap_before=coordinate_gap,written_duration_quarters=float(measure.duration_quarters),
                observed_midi_duration_quarters=expanded[j]['stop']-start))
            for note in notes_by_measure[i]:
                identifier=f'{note.note_id}@visit{j}'
                source=source_by_xml.get(note.xml_id)
                q=start+float(note.measure_offset_quarters)
                candidates=midi_candidates(note.pitch.midi_pitch,q) if note.pitch is not None else []
                duration=float(spans.get(note.note_id,note.duration_quarters))
                durations=[n for n in candidates if not n.duration_ambiguous and
                           abs(float(n.offset_quarters-n.onset_quarters)-duration)<=tolerance]
                chosen=durations or candidates
                head=tie_heads.get(note.note_id)
                link_status=('rest_position_only' if note.is_rest else 'grace_unverified' if note.is_grace else
                    'ambiguous_tie' if note.note_id in invalid_ties else 'tie_continuation_pending' if head else
                    'missing_match' if source is None else 'coordinate_conflict' if raw_unit_status=='local_match_coordinate_conflict' else
                    'verified' if len(durations)==1 else
                    'verified_onset_duration_ambiguous' if len(chosen)==1 and chosen[0].duration_ambiguous else
                    'verified_onset_duration_differs' if len(chosen)==1 else
                    'missing_midi' if not chosen else 'ambiguous_midi')
                beat=bisect_right(edges,q)-1
                beat_status='inside' if 0<=beat<len(edges)-1 else 'outside_beat_range'
                if beat_status!='inside':beat=None
                elif edges[beat+1]<=edges[beat] or performance_beats[beat+1]<=performance_beats[beat]:beat_status='zero_width'
                picked=chosen[0] if len(chosen)==1 else None
                links.append(ScoreNoteLink(identifier,note.xml_id,source.note_id if source else None,
                    f'{head}@visit{j}' if head else None,link_status,method,q,
                    start+(source.onset_beats-raw_start)/unit if source and
                    raw_unit_status in ('local_match_unit_verified','local_match_origin_only') else None,
                    None,
                    float(picked.onset_quarters) if picked else None,
                    float(picked.offset_quarters-picked.onset_quarters) if picked and not picked.duration_ambiguous else None,
                    beat,beat_status))
                occurrences.append(NoteOccurrence(identifier,note.note_id,note.xml_id,j,seen[i],label,float(note.onset_quarters)))
        from dataclasses import replace
        # A tie may start in a previous written measure/visit. Resolve by the
        # immediately preceding observed occurrence; never jump across repeats.
        for k,link in enumerate(links):
            if link.status=='tie_continuation_pending':
                head_written=tie_heads[occurrences[k].written_note_id]
                heads=[l for l,o in zip(links[:k],occurrences[:k]) if o.written_note_id==head_written and
                       o.visit_index<=occurrences[k].visit_index and not
                       any(v.break_before for v in visits[o.visit_index+1:occurrences[k].visit_index+1])]
                head=heads[-1] if heads else None
                links[k]=replace(link,tie_head_id=head.note_id if head else None,status=
                    'tie_continuation_verified' if head and head.status.startswith('verified') else 'tie_head_unverified')
        intervals=[];interval_visits=[];breaks=[];bounds=[]
        visit_starts=[v.score_start_quarters for v in visits]
        onset_counts=Counter(l.beat_index for l in links if l.status.startswith('verified'))
        for k,(left,right) in enumerate(zip(edges,edges[1:])):
            vi=bisect_right(visit_starts,left)-1
            visit=visits[vi] if 0<=vi<len(visits) and left<visits[vi].score_stop_quarters else None
            crossing=any(v.break_before and left<v.score_start_quarters<right for v in visits)
            zero=right<=left or performance_beats[k+1]<=performance_beats[k]
            covered=[v for v in visits if v.score_start_quarters<right and v.score_stop_quarters>left]
            valid=bool(visit and not zero and not crossing and right<=visits[-1].score_stop_quarters+tolerance and
                       all(v.status in ('match_midi_anchored','rest_bracketed') for v in covered))
            status=('zero_width' if zero else 'crosses_repeat_boundary' if crossing else
                    'coordinate_verified' if valid else 'measure_unverified')
            written=(visit.written_start_quarters+left-visit.score_start_quarters,
                     visit.written_start_quarters+right-visit.score_start_quarters) if visit else None
            intervals.append(ScoreBeatInterval(k,score_beats[k],score_beats[k+1],left,right,
                written[0] if written else None,written[1] if written else None,
                performance_beats[k],performance_beats[k+1],score_beat_types[k],status,valid,
                visit.measure_index if visit else None,visit.measure_number if visit else None,onset_counts[k]))
            interval_visits.append(visit.index if visit else None)
            # If a discontinuity occurred anywhere between the two starts,
            # block the pair even when the boundary-straddling beat is invalid.
            breaks.append(bool(k and any(v.break_before and edges[k-1]<v.score_start_quarters<=left for v in visits)))
            bounds.append(written)
        all_parts.append((part_id,ScoreOccurrenceBridge(ScoreBeatBridge(0.,support,tuple(links),tuple(intervals)),
            tuple(visits),tuple(occurrences),tuple(interval_visits),tuple(breaks),tuple(bounds))))
    if total_anchors<2 or total_support/total_anchors<.9:
        raise ValueError(f'Written measure schedule fails combined MIDI support: {total_support}/{total_anchors}')
    return MeasureBridgeResult(tuple(all_parts),'+'.join(sorted(methods)),total_support,total_anchors,
                               tuple(sorted(inferred)),tuple(coordinate_checks))
