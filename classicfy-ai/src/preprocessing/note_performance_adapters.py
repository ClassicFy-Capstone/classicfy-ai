"""Dataset bridges into the common score/raw-MIDI/beat-map R extractor."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np

from features.note_performance import extract_note_performance
from features.note_score_context import extract_note_score_context
from features.performance_time import BeatTimeMap
from .musicxml_score import read_musicxml
from .note_alignment import NoteAlignment, PerformedNote, ScoreNote, load_match
from .performance_midi import load_performance_midi, score_seconds_to_quarters
from .score_note_mapping import link_note_alignment


@dataclass(frozen=True)
class AlignmentPolicy:
    midi_tolerance_sec: float = .003
    score_tolerance_q: float = .0001

    def __post_init__(self):
        if (not np.isfinite(self.midi_tolerance_sec) or self.midi_tolerance_sec < 0
                or not np.isfinite(self.score_tolerance_q) or self.score_tolerance_q < 0):
            raise ValueError('Alignment tolerances must be finite nonnegative seconds/quarters')


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _score(xml_path, score_features=None, traversal=None):
    xml = read_musicxml(xml_path, traversal=traversal)
    features = extract_note_score_context(xml) if score_features is None else score_features
    if (features.source_sha256 != xml.source_sha256 or features.traversal_id != xml.traversal_id
            or tuple(meta['occurrence_id'] for meta in features.metadata) != tuple(n.occurrence_id for n in xml.notes)):
        raise ValueError('S cache does not belong to this XML and traversal')
    return xml, features


def _nasap_aliases(xml, alignment):
    """Explicit nASAP convention; coordinates/pitches are verified separately."""
    direct = {n.occurrence_id for n in xml.notes}
    counts = Counter(n.source_id for n in xml.notes)
    direct.update(key for key, count in counts.items() if count == 1)
    convention = {f'{n.source_id}-{n.occurrence}': n.occurrence_id for n in xml.notes}
    return {s.note_id: convention[s.note_id] for s, _ in alignment.matches
            if s.note_id not in direct and s.note_id in convention}


def bind_alignment_to_raw_midi(xml, features, performance, alignment, *, aliases=None,
                               policy=None, verify_velocity=True):
    policy = policy or AlignmentPolicy()
    if aliases is not None and not isinstance(aliases, dict):
        raise ValueError('Alignment aliases must map score IDs to occurrence IDs')
    by_key = {n.occurrence_id: n for n in xml.notes}
    source_groups = defaultdict(list)
    for note in xml.notes:
        source_groups[note.source_id].append(note)
    for key, notes in source_groups.items():
        if len(notes) == 1:
            by_key.setdefault(key, notes[0])
    for source, target in (aliases or {}).items():
        if target not in features.ties.occurrence_ids:
            raise ValueError(f'Unknown score bridge occurrence: {target}')
        by_key[source] = by_key[target]
    by_pitch = defaultdict(list)
    for note in performance.notes:
        by_pitch[note.pitch].append(note)
    for notes in by_pitch.values():
        notes.sort(key=lambda n: n.onset_sec)
    times = {pitch: np.array([n.onset_sec for n in notes]) for pitch, notes in by_pitch.items()}
    resolved, missed, ambiguous, verified = [], [], [], 0
    for snote, pnote in alignment.matches:
        score_note = by_key.get(snote.note_id)
        if score_note is not None:
            if snote.pitch != score_note.pitch or abs(snote.onset_beats - float(score_note.onset)) > policy.score_tolerance_q:
                raise ValueError(f'XML/alignment coordinate or pitch mismatch at {snote.note_id}')
            verified += 1
        onsets = times.get(pnote.pitch, np.array([]))
        left, right = np.searchsorted(onsets, [pnote.onset - policy.midi_tolerance_sec, pnote.onset + policy.midi_tolerance_sec], side='left')
        right = int(np.searchsorted(onsets, pnote.onset + policy.midi_tolerance_sec, side='right'))
        candidates = [n for n in by_pitch.get(pnote.pitch, ())[int(left):right]
                      if (not verify_velocity or n.velocity == pnote.velocity)
                      and (n.key_offset_sec is None or abs(n.key_offset_sec - pnote.offset) <= policy.midi_tolerance_sec)]
        if not candidates:
            missed.append(pnote.note_id)
        if len(candidates) > 1:
            ambiguous.append(pnote.note_id)
        for note in candidates:
            offset = pnote.offset if note.key_offset_sec is None else note.key_offset_sec
            resolved.append((snote, PerformedNote(note.note_id, note.pitch, note.onset_sec, offset, note.velocity)))
    if alignment.matches and verified == 0:
        raise ValueError('No XML/alignment identity and quarter-coordinate bridge could be verified')
    linked = link_note_alignment(xml, features.ties, NoteAlignment(tuple(resolved), alignment.deletions, alignment.insertions),
                                 score_id_to_occurrence=aliases)
    return linked, {'midi_tolerance_sec': policy.midi_tolerance_sec, 'score_tolerance_q': policy.score_tolerance_q,
                    'verified_score_matches': verified, 'unresolved_midi_ids': sorted(set(missed)),
                    'ambiguous_midi_ids': sorted(set(ambiguous)), 'alignment_velocity_verified': verify_velocity,
                    'raw_notes_include_unaligned_onsets_for_gap': True}


def prepare_asap_note_performance(sample, xml_path, *, work_key, split, score_features=None,
                                   traversal=None, selected_channels=None, pedal_observations=None,
                                   aliases=None, policy=None, grace_reference_keys=()):
    if not sample.aligned or sample.note_alignment_path is None:
        raise ValueError('ASAP R requires beat-aligned data and note-level alignment')
    xml, score = _score(xml_path, score_features, traversal)
    alignment = load_match(sample.note_alignment_path)
    aliases = _nasap_aliases(xml, alignment) if aliases is None else aliases
    performance = load_performance_midi(sample.performance_path, selected_channels=selected_channels,
                                        pedal_observations=pedal_observations)
    links, audit = bind_alignment_to_raw_midi(xml, score, performance, alignment, aliases=aliases, policy=policy)
    q = score_seconds_to_quarters(sample.score_path, sample.score_beats)
    score_types, performance_types = sample.score_beat_types, sample.performance_beat_types
    if len(score_types) != len(q) or len(performance_types) != len(q):
        raise ValueError('ASAP beat status and anchor counts differ')
    valid = np.array([all(t != 'bR' for t in (score_types[i], score_types[i + 1], performance_types[i], performance_types[i + 1]))
                      for i in range(len(q) - 1)], bool)
    beats = BeatTimeMap(q, sample.performance_beats, valid,
                        {'granularity': 'beat', 'source': 'ASAP annotated beats',
                         'score_seconds_to_quarters': 'score MIDI tempo events and PPQ',
                         'score_midi_sha256': _hash(sample.score_path), 'bR_intervals_excluded': int((~valid).sum())})
    r = extract_note_performance(score, performance, links, beats, record_key=sample.performance_key,
                                 work_key=work_key, split=split, dataset='ASAP', grace_reference_keys=grace_reference_keys,
                                 provenance={'adapter': 'asap-note-r-v1', 'alignment': audit,
                                             'match_sha256': _hash(sample.note_alignment_path),
                                             'robust_note_alignment': sample.robust_note_alignment,
                                             'alias_policy': 'exact IDs or verified nASAP source-occurrence suffix',
                                             'xml_path': str(xml_path)})
    return score, r


def prepare_atepp_note_performance(xml_path, score_midi_path, performance_midi_path, alignment_cache,
                                    xml_bridge, *, record_key, work_key, split, score_features=None,
                                    traversal=None, selected_channels=None, pedal_observations=None, policy=None,
                                    grace_reference_keys=()):
    """Require a fingerprinted XML↔alignment-score bridge; never use residual caches."""
    xml, score = _score(xml_path, score_features, traversal)
    if (xml_bridge.get('xml_sha256') != xml.source_sha256
            or xml_bridge.get('score_midi_sha256') != _hash(score_midi_path)
            or xml_bridge.get('score_coordinate_unit') != 'quarter'
            or xml_bridge.get('alignment_accepted') is not True):
        raise ValueError('ATEPP requires a verified XML/score-MIDI quarter bridge and accepted alignment provenance')
    aliases = xml_bridge.get('score_id_to_occurrence')
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError('ATEPP XML bridge must explicitly map score IDs to occurrences')
    performance = load_performance_midi(performance_midi_path, selected_channels=selected_channels,
                                        pedal_observations=pedal_observations)
    with np.load(alignment_cache, allow_pickle=False) as cache:
        fields = ('score_ids', 'performance_ids', 'score_onsets', 'score_offsets', 'performance_onsets', 'performance_offsets', 'pitch')
        if any(name not in cache for name in (*fields, 'score_beats', 'performance_beats')):
            raise ValueError('ATEPP requires the raw note alignment cache and beat anchors')
        arrays = {name: cache[name].copy() for name in fields}
        length = len(arrays['score_ids'])
        if any(array.shape != (length,) for array in arrays.values()):
            raise ValueError('Invalid ATEPP note alignment columns')
        for name in fields[2:]:
            if not np.isfinite(arrays[name]).all():
                raise ValueError('Nonfinite ATEPP note alignment')
        matches = tuple((ScoreNote(str(sid), int(pitch), float(q), float(end), ()),
                         PerformedNote(str(pid), int(pitch), float(onset), float(offset), 64))
                        for sid, pid, q, end, onset, offset, pitch in zip(*(arrays[name] for name in fields)))
        q, seconds = cache['score_beats'].copy(), cache['performance_beats'].copy()
        if q.ndim != 1 or len(q) < 2 or not np.allclose(np.diff(q), 1., rtol=0, atol=1e-10):
            raise ValueError('Existing ATEPP adapter requires one-quarter beat anchors')
        valid = cache['beat_valid'].copy() if 'beat_valid' in cache else None
    links, audit = bind_alignment_to_raw_midi(xml, score, performance, NoteAlignment(matches, (), ()),
                                             aliases=aliases, policy=policy, verify_velocity=False)
    beats = BeatTimeMap(q, seconds, valid, {'granularity': 'beat', 'source': 'accepted ATEPP quarter-grid alignment',
                                          'estimated': True, 'alignment_sha256': _hash(alignment_cache)})
    r = extract_note_performance(score, performance, links, beats, record_key=record_key, work_key=work_key,
                                 split=split, dataset='ATEPP', grace_reference_keys=grace_reference_keys,
                                 provenance={'adapter': 'atepp-note-r-v1', 'alignment': audit,
                                             'alignment_sha256': _hash(alignment_cache), 'xml_bridge': xml_bridge})
    return score, r
