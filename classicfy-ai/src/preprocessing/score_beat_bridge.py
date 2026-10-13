"""Connect a written, non-repeating score to original ASAP beat intervals.

XML quarters, match quarters, MIDI quarters and performed seconds remain
distinct. Match score notes (including deletions) supply ID evidence; performed
timing never defines the symbolic mapping. Repeats require a later bridge.
"""

from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from fractions import Fraction
from math import isfinite
from numbers import Integral
from pathlib import Path
from statistics import median

import mido

from .musicxml_loader import ScoreStructure
from .note_alignment import NoteAlignment


@dataclass(frozen=True)
class MidiScoreNote:
    pitch: int
    onset_quarters: Fraction
    offset_quarters: Fraction
    track: int
    channel: int
    duration_ambiguous: bool = False


@dataclass(frozen=True)
class MidiScoreTimeline:
    ppq: int
    notes: tuple[MidiScoreNote, ...]
    # Each segment is (absolute tick, absolute seconds, microseconds/quarter).
    tempo_segments: tuple[tuple[int, Fraction, int], ...]
    end_tick: int

    def quarters_at_seconds(self, seconds: float) -> Fraction:
        if not isfinite(seconds) or seconds < 0:
            raise ValueError('Score beat seconds must be finite and non-negative')
        time = Fraction(str(seconds))
        segment = self.tempo_segments[bisect_right(
            [s[1] for s in self.tempo_segments], time) - 1]
        tick, start, tempo = segment
        position = Fraction(tick, self.ppq) + (time - start) * 1_000_000 / tempo
        if position > Fraction(self.end_tick, self.ppq) + Fraction(1, self.ppq):
            raise ValueError('Score beat is outside the MIDI timeline')
        return position


def load_midi_score_timeline(path: str | Path) -> MidiScoreTimeline:
    """Read PPQ notes and tempo changes without the optional alignment stack."""
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError('Only synchronous PPQ score MIDI is supported')
    notes = []
    end_tick = 0
    for track_index, track in enumerate(midi.tracks):
        tick, active = 0, defaultdict(deque)
        for message in track:
            tick += message.time
            if message.type not in ('note_on', 'note_off') or message.channel == 9:
                continue
            key = message.channel, message.note
            if message.type == 'note_on' and message.velocity > 0:
                overlapping = bool(active[key])
                if overlapping:
                    active[key] = deque((start, True) for start, ambiguous in active[key])
                active[key].append((tick, overlapping))
            elif active[key]:
                onset, ambiguous = active[key].popleft()
                # Zero-length grace events occur in the score export. Preserve
                # them, while the bridge excludes grace from timing anchors.
                notes.append(MidiScoreNote(message.note,
                    Fraction(onset, midi.ticks_per_beat), Fraction(tick, midi.ticks_per_beat),
                    track_index, message.channel, ambiguous))
        if any(active.values()):
            raise ValueError('Unclosed score MIDI note')
        end_tick = max(end_tick, tick)
    tick, seconds, tempo = 0, Fraction(0), 500_000
    segments = [(0, seconds, tempo)]
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        seconds += Fraction(message.time * tempo, midi.ticks_per_beat * 1_000_000)
        if message.type == 'set_tempo':
            if message.tempo <= 0:
                raise ValueError('Non-positive MIDI tempo')
            tempo = message.tempo
            if segments[-1][0] == tick:
                segments[-1] = tick, seconds, tempo
            else:
                segments.append((tick, seconds, tempo))
    if not notes:
        raise ValueError('Score MIDI contains no pitched notes')
    return MidiScoreTimeline(midi.ticks_per_beat,
        tuple(sorted(notes, key=lambda n: (n.onset_quarters, n.pitch, n.track, n.channel))),
        tuple(segments), end_tick)


@dataclass(frozen=True)
class ScoreNoteLink:
    note_id: str
    xml_id: str | None
    match_id: str | None
    tie_head_id: str | None
    status: str
    method: str
    score_onset_quarters: float
    match_onset_quarters: float | None
    match_duration_quarters: float | None
    midi_onset_quarters: float | None
    midi_duration_quarters: float | None
    beat_index: int | None
    beat_status: str


@dataclass(frozen=True)
class ScoreBeatInterval:
    index: int
    score_start_seconds: float
    score_stop_seconds: float
    midi_start_quarters: float
    midi_stop_quarters: float
    xml_start_quarters: float
    xml_stop_quarters: float
    performance_start_seconds: float
    performance_stop_seconds: float
    score_beat_type: str
    status: str
    valid: bool
    measure_index: int | None
    measure_number: str | None
    verified_onsets: int


@dataclass(frozen=True)
class ScoreBeatBridge:
    xml_to_score_offset_quarters: float
    offset_support: int
    note_links: tuple[ScoreNoteLink, ...]
    intervals: tuple[ScoreBeatInterval, ...]

    def select_intervals(self, shared_indices) -> tuple[ScoreBeatInterval, ...]:
        """Keep original indices when joining an existing compact feature array."""
        indices = tuple(shared_indices)
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices):
            raise ValueError('Shared indices must be integers')
        if any(i < 0 or i >= len(self.intervals) for i in indices) or any(
                a >= b for a, b in zip(indices, indices[1:])):
            raise ValueError('Shared indices must be increasing original beat indices')
        return tuple(self.intervals[i] for i in indices)


def _tie_spans(score):
    """Resolve exact, same-voice/staff/pitch tie chains; diagnose ambiguity."""
    starts = defaultdict(list)
    for note in score.notes:
        if note.pitch is not None and not note.is_grace:
            starts[(note.part_id, note.voice, note.staff, note.pitch,
                    note.onset_quarters)].append(note)
    spans, heads, invalid = {}, {}, set()
    for note in score.notes:
        kinds = {dict(t).get('type') for t in note.ties}
        if note.pitch is None or note.is_grace or 'stop' in kinds:
            continue
        members = [note]
        current = note
        while 'start' in {dict(t).get('type') for t in current.ties}:
            candidates = [n for n in starts[(current.part_id, current.voice,
                current.staff, current.pitch, current.offset_quarters)]
                if 'stop' in {dict(t).get('type') for t in n.ties}]
            if len(candidates) != 1:
                invalid.update(n.note_id for n in members)
                break
            current = candidates[0]
            members.append(current)
        if note.note_id not in invalid:
            spans[note.note_id] = current.offset_quarters - note.onset_quarters
            for member in members[1:]:
                if member.note_id in heads:
                    invalid.add(member.note_id)
                heads[member.note_id] = note.note_id
    return spans, heads, invalid


def build_score_beat_bridge(score: ScoreStructure, alignment: NoteAlignment,
                           midi: MidiScoreTimeline, score_beats, performance_beats,
                           score_beat_types) -> ScoreBeatBridge:
    """Verify a single-part written timeline and map B-1 original intervals.

    A constant XML/match offset is inferred from at least two distinct verified
    onset positions. Nonuniform offsets, repeated IDs and repeat marks fail
    explicitly. MIDI onsets must agree within one tick. Durations allow one
    tick of MIDI note-off shortening and 1e-4 of match decimal rounding.
    """
    if len(score.part_names) != 1 or any(mark.kind in ('repeat', 'ending', 'segno', 'coda')
                                       for mark in score.barline_marks):
        raise ValueError('Repeats and multiple-part timelines require a later bridge')
    if not (len(score_beats) == len(performance_beats) == len(score_beat_types) >= 2):
        raise ValueError('Aligned beat arrays must have the same length >= 2')
    for values in (score_beats, performance_beats):
        if any(not isfinite(t) or t < 0 for t in values) or any(
                a > b for a, b in zip(values, values[1:])):
            raise ValueError('Beat seconds must be finite, non-negative and non-decreasing')
    if any(t not in ('b', 'db', 'bR') for t in score_beat_types):
        raise ValueError('Unknown score beat type')
    xml = {n.xml_id: n for n in score.notes if n.xml_id is not None}
    references = defaultdict(list)
    for note in [s for s, p in alignment.matches] + list(alignment.deletions):
        if not all(isfinite(v) for v in (note.onset_beats, note.offset_beats)):
            raise ValueError('Non-finite match score position')
        if note.note_id in xml:
            base, method = note.note_id, 'exact_xml_id'
        else:
            base, separator, occurrence = note.note_id.rpartition('-')
            if not separator or occurrence != '1':
                raise ValueError('Only exact IDs or verified single-occurrence -1 IDs are supported')
            method = 'single_occurrence_suffix_1'
        references[base].append((note, method))
    if any(len(v) != 1 for v in references.values()):
        raise ValueError('Repeated/duplicate XML references require a later bridge')
    offsets = [r[0][0].onset_beats - float(xml[identifier].onset_quarters)
        for identifier, r in references.items() if identifier in xml and
        xml[identifier].pitch is not None and not xml[identifier].is_grace and
        xml[identifier].pitch.midi_pitch == r[0][0].pitch]
    anchors = {xml[i].onset_quarters for i in references if i in xml and
               xml[i].pitch is not None and not xml[i].is_grace and
               xml[i].pitch.midi_pitch == references[i][0][0].pitch}
    if len(anchors) < 2:
        raise ValueError('At least two distinct XML/match onset anchors are required')
    offset = median(offsets)
    if any(abs(d - offset) > 1e-4 for d in offsets):
        raise ValueError('Nonuniform XML/match onset offset is unsupported')
    edges = tuple(float(midi.quarters_at_seconds(t)) for t in score_beats)
    tolerance = 1 / midi.ppq + 1e-4
    spans, tie_heads, invalid_ties = _tie_spans(score)
    # An unknown reference must remain observable, rather than disappearing.
    if any(i not in xml for i in references):
        raise ValueError('Match score ID is absent from the XML')
    links_by_id = {}
    for note in score.notes:
        ref = references.get(note.xml_id)
        source, method = ref[0] if ref else (None, '')
        candidates = []
        position = float(note.onset_quarters) + offset
        status = 'missing_match'
        if note.is_rest:
            status = 'rest_position_only'
        elif note.is_grace:
            status = 'grace_unverified'
        elif note.note_id in invalid_ties:
            status = 'ambiguous_tie'
        elif note.note_id in tie_heads:
            status = 'tie_continuation_pending'
        elif 'stop' in {dict(t).get('type') for t in note.ties}:
            status = 'orphan_tie_stop'
        elif source is not None:
            if note.pitch.midi_pitch != source.pitch:
                status = 'pitch_mismatch'
            elif abs(position - source.onset_beats) > 1e-4:
                status = 'onset_mismatch'
            elif abs(float(spans.get(note.note_id, note.duration_quarters)) -
                     source.duration_beats) > 1e-4:
                status = 'duration_mismatch'
            else:
                onset_candidates = [n for n in midi.notes if n.pitch == source.pitch and
                    n.offset_quarters > n.onset_quarters and
                    abs(float(n.onset_quarters) - source.onset_beats) <= tolerance]
                duration_candidates = [n for n in onset_candidates if not n.duration_ambiguous and
                    abs(float(n.offset_quarters - n.onset_quarters) - source.duration_beats) <= tolerance]
                candidates = duration_candidates or onset_candidates
                status = ('verified' if len(duration_candidates) == 1 else
                          'verified_onset_duration_ambiguous' if len(candidates) == 1 and
                          candidates[0].duration_ambiguous else
                          'verified_onset_duration_differs' if len(candidates) == 1 else
                          'missing_midi' if not candidates else 'ambiguous_midi')
        index = bisect_right(edges, position) - 1
        beat_status = 'inside'
        if index < 0 or index >= len(edges) - 1:
            index, beat_status = None, 'outside_beat_range'
        elif edges[index + 1] <= edges[index] or performance_beats[index + 1] <= performance_beats[index]:
            beat_status = 'zero_width'
        midi_note = candidates[0] if len(candidates) == 1 else None
        links_by_id[note.note_id] = ScoreNoteLink(note.note_id, note.xml_id,
            source.note_id if source else None, tie_heads.get(note.note_id), status, method,
            position, source.onset_beats if source else None,
            source.duration_beats if source else None,
            float(midi_note.onset_quarters) if midi_note else None,
            float(midi_note.offset_quarters - midi_note.onset_quarters) if
            midi_note and not midi_note.duration_ambiguous else None,
            index, beat_status)
    for identifier, head in tie_heads.items():
        link = links_by_id[identifier]
        if link.status == 'tie_continuation_pending':
            links_by_id[identifier] = replace(link, status=(
                'tie_continuation_verified' if links_by_id[head].status in (
                    'verified', 'verified_onset_duration_differs', 'verified_onset_duration_ambiguous')
                else 'tie_head_unverified'))
    verified = [l for l in links_by_id.values() if l.status in (
        'verified', 'verified_onset_duration_differs', 'verified_onset_duration_ambiguous')]
    if len({l.midi_onset_quarters for l in verified}) < 2:
        raise ValueError('At least two distinct verified score MIDI anchors are required')
    lower = min(l.score_onset_quarters for l in verified)
    upper = max(l.score_onset_quarters + l.match_duration_quarters for l in verified)
    intervals = []
    for i, (left, right) in enumerate(zip(edges, edges[1:])):
        xml_left, xml_right = left - offset, right - offset
        measure = next((m for m in score.measures
            if float(m.onset_quarters) <= xml_left < float(m.onset_quarters + m.duration_quarters)), None)
        valid = (right > left and performance_beats[i + 1] > performance_beats[i] and
            left >= lower - 1e-4 and right <= upper + 1e-4 and
            measure is not None and xml_right <= float(score.measures[-1].onset_quarters +
                                                      score.measures[-1].duration_quarters) + 1e-4)
        count = sum(l.beat_index == i for l in verified)
        status = ('zero_width' if right <= left or performance_beats[i + 1] <= performance_beats[i]
                  else 'outside_verified_span' if not valid else 'verified_irregular' if
                  'bR' in score_beat_types[i:i + 2] else 'coordinate_verified')
        intervals.append(ScoreBeatInterval(i, score_beats[i], score_beats[i + 1],
            left, right, xml_left, xml_right, performance_beats[i], performance_beats[i + 1],
            score_beat_types[i], status, valid, measure.measure_index if measure else None,
            measure.number if measure else None, count))
    return ScoreBeatBridge(offset, len(offsets), tuple(links_by_id[n.note_id] for n in score.notes), tuple(intervals))
