"""Versioned, performance-independent MusicXML note features."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re

import numpy as np

from preprocessing.musicxml_score import MusicXMLScore, PARSER_VERSION
from preprocessing.score_note_mapping import TieMapping, build_tie_mapping


SCHEMA_VERSION = 'note-score-context-v2'
PAD, NONE, UNK = 0, 1, 2
SLOTS = ('tempo_base', 'tempo_relative', 'tempo_gradual', 'tempo_local',
         'dynamic_base', 'dynamic_gradual', 'dynamic_local')
NUMERIC_FIELDS = ('midi_pitch', 'duration_q', 'following_rest_q', 'nominal_measure_q',
                  'actual_measure_q', 'measure_phase', 'beat_strength', 'grace_order',
                  'grace_group_size', 'quarter_bpm') + tuple(s + '_distance_q' for s in SLOTS)
FLAG_FIELDS = ('staccato', 'tenuto', 'accent', 'strong-accent', 'fermata', 'trill-mark',
               'arpeggiate', 'slur_start', 'slur_continue', 'slur_end', 'beam_start',
               'beam_continue', 'beam_end', 'tie_start', 'tie_continue', 'tie_end', 'is_grace')
_BASE_TEMPO = ('larghissimo', 'grave', 'largo', 'larghetto', 'lento', 'adagio',
               'adagietto', 'andante', 'andantino', 'moderato', 'allegretto', 'allegro',
               'vivace', 'presto', 'prestissimo', 'metronome')
_DYNAMICS = ('pppppp', 'ppppp', 'pppp', 'ppp', 'pp', 'p', 'mp', 'mf', 'f', 'ff', 'fff', 'ffff', 'fffff', 'ffffff')
_CATEGORY_VALUES = {
    'pitch_class': tuple(str(i) for i in range(12)),
    'meter_numerator': tuple(str(i) for i in range(1, 33)),
    'meter_denominator': ('1', '2', '4', '8', '16', '32', '64'),
    'beat_grouping': ('1', '1+1', '1+1+1', '2+2', '3+3', '3+3+3', '3+3+3+3', '2+2+3', '2+3+2', '3+2+2'),
    'key_fifths': tuple(str(i) for i in range(-7, 8)),
    'mode': ('major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'ionian', 'locrian', 'none'),
    'grace_anchor': ('before', 'after'),
    'tempo_base': _BASE_TEMPO,
    'tempo_relative': ('piu_mosso', 'meno_mosso', 'doppio_movimento', 'half_time'),
    'tempo_gradual': ('accelerando', 'ritardando', 'rallentando', 'ritenuto'),
    'tempo_local': ('rubato', 'a_tempo', 'tempo_primo'),
    'dynamic_base': _DYNAMICS,
    'dynamic_gradual': ('crescendo', 'diminuendo'),
    'dynamic_local': ('sf', 'sfz', 'sffz', 'sfp', 'sfpp', 'fp', 'fz', 'rf', 'rfz', 'pf'),
}
CATEGORICAL_FIELDS = tuple(_CATEGORY_VALUES)
VOCABULARIES = {name: ('PAD', 'NONE', 'UNK') + values for name, values in _CATEGORY_VALUES.items()}
_LOG_FIELDS = frozenset(NUMERIC_FIELDS) - {'midi_pitch', 'measure_phase', 'beat_strength'}


def note_score_schema():
    return {'version': SCHEMA_VERSION, 'parser_version': PARSER_VERSION, 'numeric_fields': list(NUMERIC_FIELDS),
            'categorical_fields': list(CATEGORICAL_FIELDS), 'flag_fields': list(FLAG_FIELDS),
            'vocabularies': {k: list(v) for k, v in VOCABULARIES.items()},
            'known_order': list(NUMERIC_FIELDS + CATEGORICAL_FIELDS + FLAG_FIELDS),
            'units': 'quarter=1; MIDI pitch; quarter BPM only when explicitly printed',
            'transforms': {'log1p': sorted(_LOG_FIELDS), 'midi_pitch': '/127', 'beat_strength': '/4'},
            'pooling': 'masked_mean_and_log1p_note_count', 'scope_policy': 'explicit_voice_staff_part_system',
            'repeat_policy': 'restore_entry_snapshot_on_backward_jump',
            'voice_policy': 'default_only_for_unambiguous_single_staff_single_voice_parts',
            'grace_policy': 'written_onset_group_with_local_order'}


def _readonly(value, dtype):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class NoteScoreFeatures:
    numeric: np.ndarray
    categorical: np.ndarray
    flags: np.ndarray
    field_known: np.ndarray
    note_to_onset: np.ndarray
    onset_q: np.ndarray
    metadata: tuple[dict, ...]
    ties: TieMapping
    source_sha256: str
    traversal_id: str
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        n = len(self.metadata)
        shapes = {'numeric': (n, len(NUMERIC_FIELDS)), 'categorical': (n, len(CATEGORICAL_FIELDS)),
                  'flags': (n, len(FLAG_FIELDS)),
                  'field_known': (n, len(NUMERIC_FIELDS + CATEGORICAL_FIELDS + FLAG_FIELDS)),
                  'note_to_onset': (n,)}
        for name, shape in shapes.items():
            dtype = bool if name == 'field_known' else (np.int64 if name in {'categorical', 'note_to_onset'} else np.float32)
            array = _readonly(getattr(self, name), dtype)
            if array.shape != shape:
                raise ValueError(f'Invalid {name} shape; expected {shape}')
            object.__setattr__(self, name, array)
        q = _readonly(self.onset_q, np.float64)
        if q.ndim != 1 or not np.isfinite(q).all() or np.any(np.diff(q) <= 0):
            raise ValueError('Expected finite increasing onset positions')
        if n and (not len(q) or self.note_to_onset.min() < 0 or self.note_to_onset.max() >= len(q)):
            raise ValueError('Invalid note-to-onset mapping')
        if n and not np.array_equal(np.unique(self.note_to_onset), np.arange(len(q))):
            raise ValueError('Every onset group needs a note')
        if not np.isfinite(self.numeric).all() or not np.isfinite(self.flags).all():
            raise ValueError('Feature payload must be finite')
        for j, name in enumerate(CATEGORICAL_FIELDS):
            if np.any((self.categorical[:, j] < NONE) | (self.categorical[:, j] >= len(VOCABULARIES[name]))):
                raise ValueError(f'Invalid category in {name}')
        if self.ties.occurrence_ids != tuple(row['occurrence_id'] for row in self.metadata):
            raise ValueError('Tie mapping and score note identities differ')
        object.__setattr__(self, 'onset_q', q)

    @property
    def cache_key(self):
        payload = json.dumps(note_score_schema(), sort_keys=True) + self.source_sha256 + self.traversal_id
        return hashlib.sha256(payload.encode()).hexdigest()


def _grouping(meter):
    if meter is None:
        return None
    if len(meter.groups) > 1:
        return meter.groups
    n = meter.numerator
    if meter.denominator >= 8 and n >= 6 and n % 3 == 0:
        return (3,) * (n // 3)
    if n == 4:
        return (2, 2)
    if n <= 3:
        return (1,) * n
    return None


def _phase_strength(note, score):
    meter = note.meter
    if meter is None:
        return None, None
    actual = score.measure_durations[note.measure]
    local = note.local_onset
    if actual != meter.duration:
        if note.measure == 0 and score.implicit_measures[0] and actual < meter.duration:
            local += meter.duration - actual
        else:
            return None, None
    phase = float(local / meter.duration)
    grouping = _grouping(meter)
    if grouping is None:
        return phase, None
    unit = Fraction(4, meter.denominator)
    if local == 0:
        return phase, 4
    groups, q = [], Fraction(0)
    for width in grouping[:-1]:
        q += width * unit
        groups.append(q)
    if local in groups and (len(set(grouping)) > 1 or any(g > 1 for g in grouping)):
        return phase, 3
    if local % unit == 0:
        return phase, 2
    return phase, 1 if local % (unit / 2) == 0 else 0


@dataclass(frozen=True)
class _Mark:
    token: str | None
    start: Fraction
    known: bool = True
    value: float | None = None
    source: str = ''


def _words(text):
    text = text.casefold().replace('ù', 'u').replace('ü', 'u')
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = ' '.join(text.split())
    aliases = (
        ('tempo primo', 'tempo_local', 'tempo_primo'), ('tempo i', 'tempo_local', 'tempo_primo'),
        ('a tempo', 'tempo_local', 'a_tempo'), ('piu mosso', 'tempo_relative', 'piu_mosso'),
        ('meno mosso', 'tempo_relative', 'meno_mosso'),
        ('doppio movimento', 'tempo_relative', 'doppio_movimento'),
        ('ritardando', 'tempo_gradual', 'ritardando'), ('rit', 'tempo_gradual', 'ritardando'),
        ('rallentando', 'tempo_gradual', 'rallentando'), ('rall', 'tempo_gradual', 'rallentando'),
        ('ritenuto', 'tempo_gradual', 'ritenuto'), ('riten', 'tempo_gradual', 'ritenuto'),
        ('accelerando', 'tempo_gradual', 'accelerando'), ('accel', 'tempo_gradual', 'accelerando'),
        ('crescendo', 'dynamic_gradual', 'crescendo'), ('cresc', 'dynamic_gradual', 'crescendo'),
        ('decrescendo', 'dynamic_gradual', 'diminuendo'), ('decresc', 'dynamic_gradual', 'diminuendo'),
        ('diminuendo', 'dynamic_gradual', 'diminuendo'), ('dim', 'dynamic_gradual', 'diminuendo'),
        ('rubato', 'tempo_local', 'rubato'),
    )
    result = []
    for phrase, slot, token in aliases:
        if re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text):
            result.append((slot, token))
    for word in text.split():
        if word in _BASE_TEMPO:
            result.append(('tempo_base', word))
        elif word in _DYNAMICS:
            result.append(('dynamic_base', word))
        elif word in _CATEGORY_VALUES['dynamic_local']:
            result.append(('dynamic_local', word))
    return list(dict.fromkeys(result))


class _Directions:
    def __init__(self, score):
        self.score = score
        self.state, self.snapshots, self.wedges, self.history = {}, {}, {}, {}
        self.diagnostics = []

    def enter(self, visit):
        if visit.reset:
            if visit.measure not in self.snapshots:
                raise ValueError('Traversal jumps backward to an unseen measure')
            state, wedges, history, old_q = deepcopy(self.snapshots[visit.measure])
            shift = visit.onset - old_q
            self.state = {key: _Mark(mark.token, mark.start + shift, mark.known, mark.value, mark.source)
                          for key, mark in state.items()}
            self.wedges = {key: _Mark(mark.token, mark.start + shift, mark.known, mark.value, mark.source)
                           for key, mark in wedges.items()}
            self.history = {key: [_Mark(m.token, m.start + shift, m.known, m.value, m.source) for m in stack]
                            for key, stack in history.items()}
        self.snapshots.setdefault(visit.measure, deepcopy((self.state, self.wedges, self.history, visit.onset)))

    def apply(self, event, q):
        scope = ('*' if event.system else event.part, event.staff, event.voice)
        updates = []
        if event.kind == 'dynamic':
            slot = 'dynamic_base' if event.token in _DYNAMICS else 'dynamic_local'
            updates = [(slot, event.token)]
        elif event.kind == 'metronome':
            updates = [('tempo_base', 'metronome' if event.value is not None else 'UNK')]
        elif event.kind == 'words':
            updates = _words(event.token)
            if not updates:
                updates = [('tempo_local', 'UNK')]
                self.diagnostics.append(f'Unrecognized direction: {event.source_id}: {event.token}')
        elif event.kind == 'wedge':
            key = scope + (event.number,)
            if event.token in {'crescendo', 'diminuendo'}:
                self.wedges[key] = _Mark(event.token, q, source=event.source_id)
            elif event.token == 'stop':
                if self.wedges.pop(key, None) is None:
                    self.diagnostics.append(f'Unmatched wedge stop: {event.source_id}')
                    self.state[scope + ('dynamic_gradual',)] = _Mark('UNK', q, False)
            elif event.token != 'continue':
                self.state[scope + ('dynamic_gradual',)] = _Mark('UNK', q, False)
            return
        for slot, token in updates:
            key = scope + (slot,)
            previous = self.state.get(key)
            if slot == 'tempo_local' and token in {'a_tempo', 'tempo_primo'}:
                base_key = scope + ('tempo_base',)
                stack = self.history.get(base_key, [])
                if stack:
                    base = stack[0] if token == 'tempo_primo' else stack[-1]
                    self.state[base_key] = _Mark(base.token, q, base.known, base.value, base.source)
                for change in ('tempo_relative', 'tempo_gradual'):
                    self.state[scope + (change,)] = _Mark(None, q)
            known = token in VOCABULARIES[slot] and token != 'UNK'
            mark = _Mark(token, q, known, event.value if slot == 'tempo_base' else None, event.source_id)
            if slot == 'tempo_base':
                for change in ('tempo_relative', 'tempo_gradual', 'tempo_local'):
                    change_key = scope + (change,)
                    current = self.state.get(change_key)
                    # Independent directions at the same position must survive
                    # whichever order MusicXML serializes their elements in.
                    if current is None or current.start != q:
                        self.state[change_key] = _Mark(None, q)
            if previous is not None and previous.start == q and previous.token is not None:
                complementary = (slot == 'tempo_base' and previous.known and mark.known
                                 and 'metronome' in {previous.token, token}
                                 and previous.token != token)
                if complementary:
                    mark = _Mark(token if token != 'metronome' else previous.token, q, True,
                                 event.value if token == 'metronome' else previous.value, event.source_id)
                elif previous.token != token or (previous.value is not None and mark.value is not None and previous.value != mark.value):
                    mark = _Mark('UNK', q, False, source=event.source_id)
                    self.diagnostics.append(f'Conflicting direction at {event.source_id}')
                elif mark.value is None and previous.value is not None:
                    mark = previous
            if slot == 'tempo_base':
                stack = self.history.setdefault(key, [])
                if stack and stack[-1].start == q:
                    stack[-1] = mark
                else:
                    stack.append(mark)
            self.state[key] = mark

    def get(self, note, slot):
        candidates = []
        for key, mark in self.state.items():
            part, staff, voice, field = key
            if field != slot or part not in {'*', note.part}:
                continue
            if staff is not None and staff != note.staff or voice is not None and voice != note.voice:
                continue
            if slot == 'dynamic_local' and mark.start != note.onset:
                continue
            specificity = (part != '*') + 2 * (staff is not None) + 4 * (voice is not None)
            candidates.append((specificity, mark.start, mark))
        if slot == 'dynamic_gradual':
            wedge_candidates = []
            for key, mark in self.wedges.items():
                part, staff, voice, _ = key
                if part in {'*', note.part} and (staff is None or staff == note.staff) and (voice is None or voice == note.voice):
                    specificity = (part != '*') + 2 * (staff is not None) + 4 * (voice is not None)
                    wedge_candidates.append((specificity, mark.start, mark))
            if wedge_candidates:
                specificity = max(c[0] for c in wedge_candidates)
                active = [c for c in wedge_candidates if c[0] == specificity]
                if len({c[2].token for c in active}) > 1:
                    return _Mark('UNK', max(c[1] for c in active), False)
                candidates.extend(wedge_candidates)
        if not candidates:
            return _Mark(None, note.onset, slot not in {'tempo_base', 'dynamic_base'})
        best_scope = max(c[0] for c in candidates)
        scoped = [c for c in candidates if c[0] == best_scope]
        latest = max(c[1] for c in scoped)
        marks = [c[2] for c in scoped if c[1] == latest]
        if len({(m.token, m.value) for m in marks}) > 1:
            return _Mark('UNK', latest, False)
        return marks[0]


def _connections(score):
    values = {n.occurrence_id: {kind: [False, False, False, True] for kind in ('slur', 'beam')}
              for n in score.notes}
    for name, start_kind, end_kind in (('slur', 'start', 'stop'), ('beam', 'begin', 'end')):
        scopes = defaultdict(list)
        for note in (*score.notes, *score.rests):
            key = (note.part,) if name == 'slur' else (note.part, note.voice)
            scopes[key].append(note)
        for notes in scopes.values():
            notes.sort(key=lambda note: (note.onset, note.visit, note.document_order))
            groups = defaultdict(list)
            for note in notes:
                groups[note.onset, note.visit, note.chord_group].append(note)
            active, spans, previous_visit, previous_note = {}, [], -1, None
            for (_, visit, _), simultaneous in groups.items():
                if visit != previous_visit and score.visits[visit].reset:
                    spans.extend((first, previous_note, False) for first, _ in active.values())
                    active.clear()
                previous_visit, previous_note = visit, simultaneous[-1]
                marks = defaultdict(dict)
                for note in simultaneous:
                    for number, kind in getattr(note, name + 's'):
                        marks[number].setdefault(kind, note)
                for number, kinds in marks.items():
                    if end_kind in kinds:
                        end = kinds[end_kind]
                        first, valid = active.pop(number, (end, False))
                        spans.append((first, end, valid))
                    if start_kind in kinds:
                        first = kinds[start_kind]
                        valid = number not in active
                        if not valid:
                            old, _ = active.pop(number)
                            spans.append((old, first, False))
                        active[number] = (first, valid)
                    if 'continue' in kinds and number not in active:
                        note = kinds['continue']
                        spans.append((note, note, False))
                    unsupported = set(kinds) - {start_kind, end_kind, 'continue', 'forward hook', 'backward hook'}
                    for kind in unsupported:
                        note = kinds[kind]
                        spans.append((note, note, False))
            spans.extend((first, notes[-1], False) for first, _ in active.values())
            pitched = [note for note in notes if note.pitch is not None]
            positions = [(note.onset, note.visit, note.chord_group) for note in pitched]
            for first, last, valid in spans:
                left, right = (first.onset, first.visit, first.chord_group), (last.onset, last.visit, last.chord_group)
                if right < left:
                    raise ValueError('Connection ends before its beginning')
                for note in pitched[bisect_left(positions, left):bisect_right(positions, right)]:
                    if name == 'slur' and note.voice not in {first.voice, last.voice}:
                        continue
                    position = (note.onset, note.visit, note.chord_group)
                    states = values[note.occurrence_id][name]
                    states[0] |= position == left
                    states[1] |= left < position < right
                    states[2] |= position == right
                    states[3] &= valid and first.voice_known and last.voice_known
    return values


def extract_note_score_context(score: MusicXMLScore) -> NoteScoreFeatures:
    """Extract raw features and known masks; no performance input is accepted."""
    n = len(score.notes)
    numeric = np.zeros((n, len(NUMERIC_FIELDS)), np.float32)
    categories = np.full((n, len(CATEGORICAL_FIELDS)), NONE, np.int64)
    flags = np.zeros((n, len(FLAG_FIELDS)), np.float32)
    known = np.zeros((n, len(NUMERIC_FIELDS + CATEGORICAL_FIELDS + FLAG_FIELDS)), bool)
    ni, ci, fi = ({name: i for i, name in enumerate(fields)} for fields in (NUMERIC_FIELDS, CATEGORICAL_FIELDS, FLAG_FIELDS))
    no, co = len(NUMERIC_FIELDS), len(NUMERIC_FIELDS) + len(CATEGORICAL_FIELDS)
    def number(row, name, value):
        if value is not None:
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f'Nonfinite {name}')
            numeric[row, ni[name]], known[row, ni[name]] = value, True
    def category(row, name, token, valid=True):
        vocab = VOCABULARIES[name]
        token = 'NONE' if token is None else str(token)
        good = valid and token in vocab and token not in {'UNK', 'PAD'}
        categories[row, ci[name]] = vocab.index(token) if good else UNK
        known[row, no + ci[name]] = good
    def flag(row, name, value, valid=True):
        flags[row, fi[name]], known[row, co + fi[name]] = float(value), valid
    ties = build_tie_mapping(score)
    connections = _connections(score)
    # Explicit rest chains and grace order remain local to their original voice.
    rests = defaultdict(dict)
    voices = defaultdict(list)
    for rest in score.rests:
        rests[rest.part, rest.voice].setdefault(rest.onset, []).append(rest)
    for row, note in enumerate(score.notes):
        voices[note.part, note.voice].append((row, note))
    voice_positions = {key: sorted(note.onset for _, note in sequence) for key, sequence in voices.items()}
    onset_counts = {key: Counter(positions) for key, positions in voice_positions.items()}
    grace_info, grace_anchors = {}, {}
    anchor_chords = defaultdict(list)
    for note in score.notes:
        if not note.grace:
            anchor_chords[note.visit, note.chord_group].append(note.occurrence_id)
    for sequence in voices.values():
        sequence.sort(key=lambda item: (item[1].visit, item[1].document_order))
        groups = defaultdict(list)
        main_positions = [position for position, (_, note) in enumerate(sequence) if not note.grace]
        for position, (row, note) in enumerate(sequence):
            if note.grace:
                groups[note.visit, note.local_onset, note.grace_after].append(row)
                index = bisect_left(main_positions, position)
                if note.grace_after:
                    index -= 1
                if 0 <= index < len(main_positions) and note.voice_known:
                    anchor = sequence[main_positions[index]][1]
                    if (note.grace_after and anchor.offset <= note.onset
                            or not note.grace_after and anchor.onset >= note.onset):
                        grace_anchors[row] = tuple(anchor_chords[anchor.visit, anchor.chord_group])
        for rows in groups.values():
            for order, row in enumerate(rows, 1):
                grace_info[row] = (order, len(rows))
    row_by_visit = defaultdict(list)
    for row, note in enumerate(score.notes):
        row_by_visit[note.visit].append((row, note))
    events = defaultdict(list)
    for direction in score.directions:
        events[direction.measure].append(direction)
    engine, metadata = _Directions(score), [None] * n
    for vi, visit in enumerate(score.visits):
        engine.enter(visit)
        timeline = [(visit.onset + e.position, 0, e) for e in events[visit.measure]]
        if any(e.position < 0 or e.position > visit.duration for e in events[visit.measure]):
            raise ValueError('Cross-measure direction offset is unsupported')
        timeline += [(note.onset, 1, (row, note)) for row, note in row_by_visit[vi]]
        timeline.sort(key=lambda entry: entry[:2])
        for q, kind, item in timeline:
            if kind == 0:
                engine.apply(item, q)
                continue
            row, note = item
            number(row, 'midi_pitch', note.pitch)
            number(row, 'duration_q', note.duration)
            number(row, 'actual_measure_q', visit.duration)
            if note.meter:
                number(row, 'nominal_measure_q', note.meter.duration)
                category(row, 'meter_numerator', note.meter.numerator)
                category(row, 'meter_denominator', note.meter.denominator)
                grouping = _grouping(note.meter)
                category(row, 'beat_grouping', '+'.join(map(str, grouping)) if grouping else 'UNK', grouping is not None)
            else:
                for field in ('meter_numerator', 'meter_denominator', 'beat_grouping'):
                    category(row, field, 'UNK', False)
            phase, strength = _phase_strength(note, score)
            number(row, 'measure_phase', phase)
            number(row, 'beat_strength', strength)
            category(row, 'pitch_class', note.pitch % 12)
            category(row, 'key_fifths', note.fifths, note.fifths is not None)
            category(row, 'mode', note.mode, note.mode is not None)
            at, rest_length = note.offset, Fraction(0)
            voice_rests = rests[note.part, note.voice]
            visited = set()
            while at in voice_rests and at not in visited:
                visited.add(at)
                choices = voice_rests[at]
                if len(choices) != 1 or not choices[0].voice_known:
                    break
                rest_length += choices[0].duration
                at = choices[0].offset
            else:
                positions = voice_positions[note.part, note.voice]
                future = bisect_right(positions, note.offset) < len(positions)
                at_offset = onset_counts[note.part, note.voice][note.offset] > int(note.onset == note.offset)
                if note.voice_known and (rest_length > 0 or future or at_offset):
                    number(row, 'following_rest_q', rest_length)
            for name in FLAG_FIELDS[:7]:
                flag(row, name, name in note.flags)
            for name in ('slur', 'beam'):
                start, continuation, end, valid = connections[note.occurrence_id][name]
                for state, value in (('start', start), ('continue', continuation), ('end', end)):
                    flag(row, f'{name}_{state}', value, valid)
            flag(row, 'tie_start', 'start' in note.ties)
            flag(row, 'tie_continue', 'start' in note.ties and 'stop' in note.ties)
            flag(row, 'tie_end', 'stop' in note.ties)
            flag(row, 'is_grace', note.grace)
            order, size = grace_info.get(row, (0, 0))
            if not note.grace or note.voice_known:
                number(row, 'grace_order', order)
                number(row, 'grace_group_size', size)
            category(row, 'grace_anchor', ('after' if note.grace_after else 'before') if note.grace else None,
                     not note.grace or row in grace_anchors)
            for slot in SLOTS:
                mark = engine.get(note, slot)
                category(row, slot, mark.token, mark.known)
                if mark.known and mark.token is not None:
                    number(row, slot + '_distance_q', note.onset - mark.start)
                if slot == 'tempo_base' and mark.known and mark.value is not None and mark.value > 0:
                    number(row, 'quarter_bpm', mark.value)
            metadata[row] = {
                'xml_note_id': note.xml_id, 'source_id': note.source_id, 'occurrence_id': note.occurrence_id,
                'source_locator': note.source_locator,
                'part': note.part, 'voice': note.voice, 'staff': note.staff, 'measure_index': note.measure,
                'measure_label': note.measure_label, 'occurrence': note.occurrence, 'visit': note.visit,
                'written_onset_q': str(note.written_onset), 'unfolded_onset_q': str(note.onset),
                'duration_q': str(note.duration), 'tie_chain_id': ties.chain_ids[row],
                'written_pitch': note.written_pitch, 'document_order': note.document_order,
                'grace_anchor_occurrence_ids': grace_anchors.get(row, ()),
                'attack_owner_occurrence_id': ties.occurrence_ids[int(ties.owner_indices[row])],
                'tie_chain_duration_q': float(ties.chain_duration_q[row]),
                'tie_chain_known': bool(ties.chain_known[row]), 'is_attack': bool(ties.is_attack[row]),
                'slurs': list(note.slurs), 'beams': list(note.beams), 'notations': list(note.flags),
            }
    onset_values = sorted({note.onset for note in score.notes})
    lookup = {q: i for i, q in enumerate(onset_values)}
    return NoteScoreFeatures(numeric, categories, flags, known,
                             [lookup[note.onset] for note in score.notes], list(map(float, onset_values)),
                             tuple(metadata), ties, score.source_sha256, score.traversal_id,
                             score.diagnostics + tuple(engine.diagnostics))


def encode_note_numeric(features: NoteScoreFeatures) -> np.ndarray:
    result = features.numeric.astype(np.float64)
    known = features.field_known[:, :len(NUMERIC_FIELDS)]
    result[~known] = 0
    for i, name in enumerate(NUMERIC_FIELDS):
        if name in _LOG_FIELDS:
            if np.any(result[:, i] < 0):
                raise ValueError(f'Negative length/value in {name}')
            result[:, i] = np.log1p(result[:, i])
        elif name == 'midi_pitch':
            result[:, i] /= 127
        elif name == 'beat_strength':
            result[:, i] /= 4
    return result.astype(np.float32)


@dataclass(frozen=True)
class NoteScoreScale:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    training_sources: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION or len(self.mean) != len(NUMERIC_FIELDS) or len(self.scale) != len(NUMERIC_FIELDS):
            raise ValueError('Invalid note score scale schema')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(np.asarray(self.scale) <= 0):
            raise ValueError('Scale must be finite and positive')

    def transform(self, features):
        result = (encode_note_numeric(features) - np.asarray(self.mean)) / np.asarray(self.scale)
        return np.where(features.field_known[:, :len(NUMERIC_FIELDS)], result, 0).astype(np.float32)


def fit_note_score_scale(training_features) -> NoteScoreScale:
    """Fit on known values of explicitly supplied training scores, once per traversal."""
    unique = {features.cache_key: features for features in training_features}
    if not unique:
        raise ValueError('Need at least one training score')
    sums, squares, counts = (np.zeros(len(NUMERIC_FIELDS), np.float64) for _ in range(3))
    for features in unique.values():
        values = encode_note_numeric(features).astype(np.float64)
        valid = features.field_known[:, :len(NUMERIC_FIELDS)]
        sums += np.where(valid, values, 0).sum(axis=0)
        squares += np.where(valid, values * values, 0).sum(axis=0)
        counts += valid.sum(axis=0)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variance = np.divide(squares, counts, out=np.zeros_like(sums), where=counts > 0) - mean ** 2
    scale = np.sqrt(np.maximum(variance, 0))
    scale[scale <= 1e-12] = 1
    return NoteScoreScale(tuple(mean), tuple(scale), tuple(sorted({f.source_sha256 for f in unique.values()})))
