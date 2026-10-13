"""Score-only phrase boundary hypotheses with explicit scope and uncertainty.

No performance values, phrase ground truth or probability of correctness are
used. Slurs, rests and simple harmonic closures remain separate evidence.
"""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import heapq
import json
import xml.etree.ElementTree as ET

from .score_beat_bridge import _tie_spans


@dataclass(frozen=True)
class SlurSpan:
    part_id: str
    number: str
    start_note_id: str | None
    stop_note_id: str | None
    start_quarters: Fraction | None
    stop_onset_quarters: Fraction | None
    stop_release_quarters: Fraction | None
    start_scope: str | None
    stop_scope: str | None
    status: str


@dataclass(frozen=True)
class BoundaryCue:
    kind: str
    scope: str
    part_id: str | None
    position_quarters: Fraction
    uncertainty_start_quarters: Fraction
    uncertainty_stop_quarters: Fraction
    note_ids: tuple[str, ...]
    details: dict


@dataclass(frozen=True)
class BoundaryCandidate:
    candidate_id: str
    scope: str
    part_id: str | None
    position_quarters: Fraction
    uncertainty_start_quarters: Fraction
    uncertainty_stop_quarters: Fraction
    cues: tuple[BoundaryCue, ...]
    coincident_cue_kinds: tuple[str, ...]
    review_priority: str
    status: str = 'pending_score_review'


@dataclass(frozen=True)
class PhraseCandidateResult:
    slurs: tuple[SlurSpan, ...]
    candidates: tuple[BoundaryCandidate, ...]
    diagnostics: tuple[dict, ...]
    policy: dict


def voice_scope(note):
    return f'{note.part_id}:staff={note.staff}:voice={note.voice}'


def _merged(intervals):
    result=[]
    for start,stop in sorted(intervals):
        if stop<=start:continue
        if result and start<=result[-1][1]:result[-1]=(result[-1][0],max(stop,result[-1][1]))
        else:result.append((start,stop))
    return result


def pair_written_slurs(score):
    """Match part/number in XML document order, including staff/voice crossings.

    Stop precedes a new start on a shared endpoint. Conflicting starts are not
    resolved by choosing the nearest note. Unpaired/zero-span marks are retained.
    """
    active={};result=[]
    def span(start,stop,number,status):
        return SlurSpan((start or stop).part_id,number,start.note_id if start else None,
            stop.note_id if stop else None,start.onset_quarters if start else None,
            stop.onset_quarters if stop else None,stop.offset_quarters if stop else None,
            voice_scope(start) if start else None,voice_scope(stop) if stop else None,status)
    for note in score.notes:
        for mark in sorted(note.slurs,key=lambda m:dict(stop=0,continue_=1,start=2).get(m.kind,1)):
            key=note.part_id,mark.number
            if mark.kind=='start':
                if key in active:
                    previous=active.pop(key)
                    if previous is not None:result.append(span(previous,None,mark.number,'conflicting_start'))
                    result.append(span(note,None,mark.number,'conflicting_start'));active[key]=None
                else:active[key]=note
            elif mark.kind=='stop':
                if key not in active:result.append(span(None,note,mark.number,'orphan_stop'))
                else:
                    start=active.pop(key)
                    if start is None:result.append(span(None,note,mark.number,'ambiguous_stop'))
                    else:
                        status=('nonpositive_onset_span' if note.onset_quarters<=start.onset_quarters else
                                'paired_cross_scope' if voice_scope(start)!=voice_scope(note) else 'paired')
                        result.append(span(start,note,mark.number,status))
            elif key not in active:result.append(span(None,note,mark.number,'orphan_continue'))
    for (part,number),note in active.items():
        if note is not None:result.append(span(note,None,number,'orphan_start'))
    return tuple(result)


def extract_phrase_candidates(score, *, minimum_rest_quarters=Fraction(1,2)):
    minimum_rest_quarters=Fraction(minimum_rest_quarters)
    if minimum_rest_quarters<=0:raise ValueError('Minimum rest length must be positive')
    notes={n.note_id:n for n in score.notes};spans,heads,bad_ties=_tie_spans(score)
    slurs=pair_written_slurs(score);cues=[];diagnostics=[]
    def cue(kind,scope,pid,q,ids=(),lo=None,hi=None,**details):
        cues.append(BoundaryCue(kind,scope,pid,q,q if lo is None else lo,q if hi is None else hi,
                                tuple(ids),details))
    for slur in slurs:
        if slur.status not in ('paired','paired_cross_scope'):
            diagnostics.append(dict(kind='slur_pairing',status=slur.status,start_note_id=slur.start_note_id,
                                    stop_note_id=slur.stop_note_id));continue
        stop=notes[slur.stop_note_id]
        if stop.is_grace or stop.is_cue or stop.duration_quarters<=0:
            diagnostics.append(dict(kind='slur_endpoint_unusable',note_id=stop.note_id));continue
        scope=voice_scope(stop) if slur.status=='paired' else slur.part_id+':part'
        cue('slur_end',scope,slur.part_id,stop.offset_quarters,(slur.start_note_id,slur.stop_note_id),
            source_stop_onset=str(stop.onset_quarters),slur_span_quarters=str(slur.stop_onset_quarters-slur.start_quarters),
            cross_scope=slur.status=='paired_cross_scope',not_phrase_ground_truth=True)
    by_scope=defaultdict(list);pitched=[]
    for note in score.notes:
        by_scope[voice_scope(note)].append(note)
        if note.pitch is not None and not note.is_grace and not note.is_cue:
            pitched.append(note)
    for scope,local in sorted(by_scope.items()):
        held=_merged((n.onset_quarters,n.onset_quarters+spans.get(n.note_id,n.duration_quarters))
                     for n in local if n.pitch is not None and not n.is_grace and not n.is_cue)
        for rest in local:
            if not rest.is_rest or rest.duration_quarters<minimum_rest_quarters:continue
            overlaps=[(a,b) for a,b in held if a<rest.offset_quarters and b>rest.onset_quarters]
            if overlaps:
                diagnostics.append(dict(kind='rest_overlaps_pitched_scope',note_id=rest.note_id));continue
            previous=any(b<=rest.onset_quarters for a,b in held)
            following=any(a>=rest.offset_quarters for a,b in held)
            kind='voice_rest' if previous and following else 'voice_entry_rest' if following else 'voice_exit_rest'
            cue(kind,scope,rest.part_id,rest.onset_quarters,(rest.note_id,),hi=rest.offset_quarters,
                length_quarters=str(rest.duration_quarters),global_silence=False)
    held_global=_merged((n.onset_quarters,n.onset_quarters+spans.get(n.note_id,n.duration_quarters)) for n in pitched)
    for before,after in zip(held_global,held_global[1:]):
        lo,hi=before[1],after[0]
        if hi-lo>=minimum_rest_quarters:
            source=[n.note_id for n in score.notes if n.is_rest and n.onset_quarters<=lo and n.offset_quarters>=hi]
            cue('global_notated_silence','piece',None,lo,source,hi=hi,length_quarters=str(hi-lo),
                acoustic_silence_unknown=True,multiple_parts=len(score.part_names)>1)
    for note in score.notes:
        for encoded in note.notation_xml:
            node=ET.fromstring(encoded)
            kinds={element.tag.rsplit('}',1)[-1] for element in node.iter()}
            for symbol in ('fermata','breath-mark','caesura'):
                if symbol in kinds:
                    cue('notation_'+symbol.replace('-','_'),voice_scope(note),note.part_id,
                        note.offset_quarters,(note.note_id,),not_phrase_ground_truth=True)
    for mark in score.barline_marks:
        if mark.kind in ('repeat','ending'):
            cue('navigation_boundary',mark.part_id+':part',mark.part_id,mark.onset_quarters,
                symbol=mark.kind,attributes=dict(mark.attributes),not_phrase_ground_truth=True)
        elif mark.kind=='bar-style' and mark.text in ('light-light','light-heavy','heavy-light','heavy-heavy'):
            cue('section_barline',mark.part_id+':part',mark.part_id,mark.onset_quarters,
                symbol=mark.text,not_phrase_ground_truth=True)
    cues.extend(_harmonic_closure_hints(score,pitched,spans))
    grouped=defaultdict(list)
    for item in cues:grouped[item.scope,item.position_quarters].append(item)
    candidates=[]
    for (scope,q),items in sorted(grouped.items(),key=lambda pair:(pair[0][1],pair[0][0])):
        pid=items[0].part_id
        coincident=[c for c in cues if c.position_quarters==q and c not in items and
                    (c.scope=='piece' or (pid is not None and c.part_id==pid and c.scope==pid+':part'))]
        kinds={c.kind for c in items+coincident}
        families=set()
        for kind in kinds:
            if kind=='slur_end':families.add('slur')
            elif kind in ('voice_rest','global_notated_silence'):families.add('rest')
            elif kind=='harmonic_closure_hint':families.add('harmonic')
            elif kind.startswith('notation_'):families.add('pause_notation')
            elif kind=='section_barline':families.add('section')
        priority=('high_review' if len(families)>=2 else 'medium_review' if
                  kinds & {'global_notated_silence','harmonic_closure_hint','notation_fermata','section_barline'} else 'low_review')
        identity=hashlib.sha256(json.dumps([scope,str(q)]).encode()).hexdigest()[:12]
        candidates.append(BoundaryCandidate('boundary_'+identity,scope,pid,q,
            min(c.uncertainty_start_quarters for c in items),max(c.uncertainty_stop_quarters for c in items),
            tuple(items),tuple(sorted(c.kind for c in coincident)),priority))
    return PhraseCandidateResult(slurs,tuple(candidates),tuple(diagnostics),dict(
        minimum_rest_quarters=str(minimum_rest_quarters),coordinate='Written MusicXML quarter Fractions',
        slur_matching='part/number in XML document order; no nearest-note pairing',
        confidence='Review priority only, not calibrated probability',
        phrase_status='All candidates pending review; no complete phrase partition or hierarchy inferred',
        inputs='Score notes/notations/key signatures only; no MIDI performance, features or retrieval results'))


def _harmonic_closure_hints(score,pitched,tie_spans):
    """Very conservative root-position major V to major/minor I hypotheses.

    Check vertical notated sonorities at consecutive nongrace onsets, complete
    triad pitch classes and bass 5→1. A downbeat or sustained arrival is required.
    Missing mode yields two named hypotheses. This is not a cadence classifier:
    modulation, half cadences, reductions and phrase context are not resolved.
    """
    result=[]
    for pid,_ in score.part_names:
        local=[n for n in pitched if n.part_id==pid]
        keys=sorted([k for k in score.key_signatures if k.part_id==pid and k.staff in (None,1)],key=lambda k:k.onset_quarters)
        positions=sorted({n.onset_quarters for n in local})
        starts=defaultdict(list)
        for n in local:starts[n.onset_quarters].append(n)
        live={};ends=[]
        previous=None
        for q in positions:
            while ends and ends[0][0]<=q:
                stop,identifier=heapq.heappop(ends);live.pop(identifier,None)
            for n in starts[q]:
                live[n.note_id]=n
                heapq.heappush(ends,(n.onset_quarters+tie_spans.get(n.note_id,n.duration_quarters),n.note_id))
            active=list(live.values())
            if not active:previous=None;continue
            # Microtones and nontraditional key signatures are left unsupported.
            if any(n.pitch.midi_pitch.denominator!=1 for n in active):previous=None;continue
            pcs={int(n.pitch.midi_pitch)%12 for n in active};bass=min(active,key=lambda n:n.pitch.midi_pitch)
            applicable=[k for k in keys if k.onset_quarters<=q]
            key=applicable[-1] if applicable else None
            arrivals=[n for n in active if n.onset_quarters==q]
            metric=any(n.measure_offset_quarters==0 for n in arrivals)
            sustained=any(n.duration_quarters>=1 for n in arrivals)
            if previous and key and key.fifths is not None and not key.alterations and (metric or sustained):
                major_root=(7*key.fifths)%12
                modes=[key.mode] if key.mode in ('major','minor') else ['major','minor'] if key.mode is None else []
                for mode in modes:
                    tonic=(major_root+(9 if mode=='minor' else 0))%12;dominant=(tonic+7)%12
                    previous_q,previous_pcs,previous_bass,previous_key=previous
                    if previous_key!=key:continue
                    if (int(previous_bass.pitch.midi_pitch)%12!=dominant or int(bass.pitch.midi_pitch)%12!=tonic or
                        not {dominant,(dominant+4)%12,(dominant+7)%12}<=previous_pcs or
                        not {tonic,(tonic+(3 if mode=='minor' else 4))%12,(tonic+7)%12}<=pcs):continue
                    release=min(n.offset_quarters for n in arrivals)
                    latest=max(n.offset_quarters for n in arrivals)
                    result.append(BoundaryCue('harmonic_closure_hint',pid+':part',pid,release,q,latest,
                        tuple(n.note_id for n in arrivals),dict(arrival_onset_quarters=str(q),
                        previous_onset_quarters=str(previous_q),previous_pitch_classes=sorted(previous_pcs),
                        arrival_pitch_classes=sorted(pcs),bass_from=int(previous_bass.pitch.midi_pitch),
                        bass_to=int(bass.pitch.midi_pitch),tonic_pitch_class=tonic,mode=mode,
                        mode_source='explicit_musicxml' if key.mode else 'unconfirmed_key_signature_hypothesis',
                        release_choice_uncertain=release!=latest,
                        boundary_placement_uncertain=True,
                        tonic_prolongation_possible=True,not_phrase_ground_truth=True)))
            previous=(q,pcs,bass,key)
    return result


def phrase_position(start,stop,position):
    """Relative position of an explicitly supplied phrase, never infer a phrase."""
    start,stop,position=map(Fraction,(start,stop,position))
    if stop<=start:raise ValueError('Phrase length must be positive')
    if not start<=position<stop:return None
    relative=(position-start)/(stop-start)
    phase='start' if relative<Fraction(1,3) else 'middle' if relative<Fraction(2,3) else 'end'
    return relative,phase
