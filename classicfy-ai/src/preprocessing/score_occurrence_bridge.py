"""Recover observed measure visits from score-side match IDs and MIDI evidence.

Repeat signs validate transitions; they do not prescribe performed unfolding.
The existing beat bridge checks a temporary occurrence timeline. Consumers use
written_note_id/measure_index for musical structure and visit_index for order.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from math import isfinite
from statistics import median

from .musicxml_loader import ScoreStructure
from .score_beat_bridge import MidiScoreTimeline, ScoreBeatBridge, build_score_beat_bridge


@dataclass(frozen=True)
class MeasureVisit:
    index: int
    measure_index: int
    measure_number: str
    visit_number: int
    source_occurrence_label: str
    score_start_quarters: float
    score_stop_quarters: float
    written_start_quarters: float
    anchor_count: int
    status: str
    transition: str
    break_before: bool


@dataclass(frozen=True)
class NoteOccurrence:
    note_id: str
    written_note_id: str
    written_xml_id: str | None
    visit_index: int
    visit_number: int
    source_occurrence_label: str
    written_onset_quarters: float


@dataclass(frozen=True)
class ScoreOccurrenceBridge:
    # The bridge's xml_* coordinates refer to the temporary unfolded adapter.
    # Use interval_written_quarters for the original written XML coordinates.
    bridge: ScoreBeatBridge
    visits: tuple[MeasureVisit, ...]
    note_occurrences: tuple[NoteOccurrence, ...]
    # One entry per original interval; None denotes an unsupported gap.
    interval_visits: tuple[int | None, ...]
    interval_break_before: tuple[bool, ...]
    interval_written_quarters: tuple[tuple[float,float] | None, ...]

    def adjacent_pair_mask(self,feature_mask):
        """Mask i-1→i pairs on original indices, including structural cuts."""
        if len(feature_mask)!=len(self.bridge.intervals) or any(v not in (True,False) for v in feature_mask):
            raise ValueError('Expected one boolean feature mask per original interval')
        return tuple(i>0 and not self.interval_break_before[i] and interval.valid and
            self.bridge.intervals[i-1].valid and bool(feature_mask[i]) and bool(feature_mask[i-1])
            for i,interval in enumerate(self.bridge.intervals))


class _TickTimeline(MidiScoreTimeline):
    def quarters_at_seconds(self, seconds):
        # ASAP seconds are rounded; symbolic score coordinates live on ticks.
        # Snap by at most half a tick, without modifying the stored annotation.
        value=super().quarters_at_seconds(seconds)
        return Fraction(round(value*self.ppq),self.ppq)


def _ending_ranges(score):
    opened, ranges={}, []
    for mark in score.barline_marks:
        if mark.kind!='ending':
            continue
        attributes=dict(mark.attributes)
        label=attributes.get('number','')
        if attributes.get('type')=='start':
            opened[label]=mark.measure_index
        elif attributes.get('type') in ('stop','discontinue') and label in opened:
            ranges.append((opened.pop(label),mark.measure_index,label))
    return ranges


def _transition(score,previous,current):
    before,after=previous['measure_index'],current['measure_index']
    contiguous=abs(previous['stop']-current['start'])<=1e-4
    if not contiguous:
        return 'unresolved_gap'
    if after==before+1:
        return 'consecutive'
    forwards=[m.measure_index for m in score.barline_marks if m.kind=='repeat' and
              dict(m.attributes).get('direction')=='forward' and m.measure_index<=before]
    backward=any(m.measure_index==before and m.kind=='repeat' and
                 dict(m.attributes).get('direction')=='backward' for m in score.barline_marks)
    if after<=before and backward and forwards and after==max(forwards):
        return 'repeat_jump'
    endings=_ending_ranges(score)
    if after>before+1 and any(lo==before+1 and hi==after-1 and label=='1'
        for lo,hi,label in endings) and any(lo==after and label=='2' for lo,hi,label in endings):
        return 'ending_skip'
    return 'unresolved_jump'


def build_score_occurrence_bridge(score: ScoreStructure,alignment,midi: MidiScoreTimeline,
                                  score_beats,performance_beats,score_beat_types) -> ScoreOccurrenceBridge:
    """Bridge observed repeat/first-second-ending visits of a single part.

    Within each written measure and source occurrence label, non-grace pitch
    anchors must agree on a translation (1e-4 quarter). Source labels remain
    raw; visit numbers are assigned chronologically per written measure.
    Rest-only gaps are filled only when both neighbors and the exact summed
    written durations agree. Unknown transitions retain data with valid=False.
    """
    if len(score.part_names)!=1:
        raise ValueError('Occurrence bridge currently supports one score part')
    notes_by_measure=defaultdict(list)
    xml={}
    for note in score.notes:
        notes_by_measure[note.measure_index].append(note)
        if note.xml_id is not None:
            xml[note.xml_id]=note
    references,groups={},defaultdict(list)
    for source in [s for s,p in alignment.matches]+list(alignment.deletions):
        if source.note_id in references:
            raise ValueError('Duplicate full match occurrence ID')
        if source.note_id in xml:
            identifier,label=source.note_id,'exact'
        else:
            identifier,separator,label=source.note_id.rpartition('-')
            if not separator or not label.isdecimal() or int(label)<=0 or identifier not in xml:
                raise ValueError('Unresolved XML occurrence ID: '+source.note_id)
        note=xml[identifier]
        if note.pitch is None or note.pitch.midi_pitch!=source.pitch:
            raise ValueError('Occurrence anchor pitch does not match XML: '+source.note_id)
        if not all(isfinite(q) for q in (source.onset_beats,source.offset_beats)):
            raise ValueError('Non-finite occurrence position')
        references[source.note_id]=(identifier,label,source)
        if not note.is_grace:
            groups[note.measure_index,label].append((source.onset_beats-float(note.measure_offset_quarters),source))
    candidates=[]
    for (index,label),anchors in groups.items():
        starts=[q for q,s in anchors]
        start=median(starts)
        if max(starts)-min(starts)>1e-4:
            raise ValueError('Nonuniform occurrence positions within one written measure')
        measure=score.measures[index]
        support=sum(any(n.pitch==s.pitch and n.offset_quarters>n.onset_quarters and
            abs(float(n.onset_quarters)-s.onset_beats)<=1/midi.ppq+1e-4 for n in midi.notes) for q,s in anchors)
        candidates.append(dict(measure_index=index,label=label,start=start,
            stop=start+float(measure.duration_quarters),anchor_count=support,
            status='match_midi_anchored' if support else 'midi_unverified'))
    candidates.sort(key=lambda v:(v['start'],v['measure_index'],v['label']))
    if not candidates:
        raise ValueError('No occurrence anchors')
    # Do not silently overlap visits or select one of conflicting groups.
    for previous,current in zip(candidates,candidates[1:]):
        if current['start']<previous['stop']-1e-4:
            raise ValueError('Overlapping measure visits are unsupported')
    expanded=[]
    for current in candidates:
        if expanded:
            previous=expanded[-1]
            missing=list(range(previous['measure_index']+1,current['measure_index']))
            rest_only=bool(missing) and all(notes_by_measure[i] and
                all(n.is_rest for n in notes_by_measure[i]) for i in missing)
            duration=sum(float(score.measures[i].duration_quarters) for i in missing)
            if rest_only and abs(current['start']-previous['stop']-duration)<=1e-4:
                position=previous['stop']
                for index in missing:
                    stop=position+float(score.measures[index].duration_quarters)
                    expanded.append(dict(measure_index=index,label=current['label'],start=position,
                        stop=stop,anchor_count=0,status='rest_bracketed'))
                    position=stop
        expanded.append(current)
    visits=[]
    seen=Counter()
    cloned_notes,cloned_measures,occurrences=[],[],[]
    for index,item in enumerate(expanded):
        measure=score.measures[item['measure_index']]
        seen[measure.measure_index]+=1
        transition='start' if index==0 else _transition(score,expanded[index-1],item)
        status=item['status'] if not transition.startswith('unresolved') else 'transition_unverified'
        visit=MeasureVisit(index,measure.measure_index,measure.number,seen[measure.measure_index],
            item['label'],item['start'],item['stop'],float(measure.onset_quarters),item['anchor_count'],
            status,transition,transition not in ('start','consecutive'))
        visits.append(visit)
        start=Fraction(str(item['start']))
        cloned_measures.append(replace(measure,measure_index=index,onset_quarters=start))
        for note in notes_by_measure[measure.measure_index]:
            identifier=f'{note.note_id}@visit{index}'
            # Exact match IDs avoid stripping/merging repeated note occurrences.
            source_id=note.xml_id if item['label']=='exact' else (
                f'{note.xml_id}-{item["label"]}' if note.xml_id is not None else None)
            cloned_notes.append(replace(note,note_id=identifier,xml_id=source_id,measure_index=index,
                onset_quarters=start+note.measure_offset_quarters,
                slurs=tuple(replace(mark,note_id=identifier) for mark in note.slurs)))
            occurrences.append(NoteOccurrence(identifier,note.note_id,note.xml_id,index,
                visit.visit_number,item['label'],float(note.onset_quarters)))
    # This is only a temporary timing adapter. Musical key/meter/slur context
    # must be read from the written score using the returned occurrence map.
    timing_score=replace(score,notes=tuple(cloned_notes),measures=tuple(cloned_measures),
                         barline_marks=(),key_signatures=(),time_signatures=())
    tick_midi=_TickTimeline(midi.ppq,midi.notes,midi.tempo_segments,midi.end_tick)
    bridge=build_score_beat_bridge(timing_score,alignment,tick_midi,score_beats,
                                  performance_beats,score_beat_types)
    interval_visits=[]
    interval_breaks=[]
    written_bounds=[]
    intervals=[]
    for interval in bridge.intervals:
        visit=visits[interval.measure_index] if interval.measure_index is not None else None
        valid=interval.valid and visit is not None and visit.status in ('match_midi_anchored','rest_bracketed')
        # A beat that straddles a non-consecutive written boundary is excluded.
        crossing=any(v.break_before and interval.midi_start_quarters<v.score_start_quarters<interval.midi_stop_quarters
                     for v in visits)
        valid=valid and not crossing
        changed=(bool(visit and visit.break_before and
                      abs(interval.midi_start_quarters-visit.score_start_quarters)<=1e-4) or
                 (bool(interval_visits) and (visit.index if visit else None)!=interval_visits[-1] and
                  bool(visit and visit.break_before)))
        status=(interval.status if valid or interval.status=='zero_width' else
                'crosses_repeat_boundary' if crossing else 'occurrence_unverified')
        intervals.append(replace(interval,status=status,valid=valid))
        interval_visits.append(visit.index if visit else None)
        interval_breaks.append(changed)
        written_bounds.append((visit.written_start_quarters+interval.midi_start_quarters-visit.score_start_quarters,
            visit.written_start_quarters+interval.midi_stop_quarters-visit.score_start_quarters) if visit else None)
    return ScoreOccurrenceBridge(replace(bridge,intervals=tuple(intervals)),tuple(visits),
        tuple(occurrences),tuple(interval_visits),tuple(interval_breaks),tuple(written_bounds))
