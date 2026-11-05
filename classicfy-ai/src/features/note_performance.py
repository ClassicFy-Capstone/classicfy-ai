"""Eleven note-wise performance targets from one score and one raw performance."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json

import numpy as np

from preprocessing.performance_midi import PerformanceMidi
from preprocessing.score_note_mapping import NoteAlignmentLinks
from .note_score_context import FLAG_FIELDS, NoteScoreFeatures
from .performance_time import BeatTimeMap


R_SCHEMA_VERSION = 'note-performance-v1'
R_CHANNELS = ('tempo_log2_qpm', 'onset_deviation_q', 'velocity_unit', 'articulation_log2_ratio',
              'sustain_onset_unit', 'sustain_offset_unit', 'sustain_min_hold_unit',
              'sustain_min_hold_delay_q', 'sustain_min_gap_unit', 'sustain_min_gap_delay_q', 'soft_onset_unit')
PHYSICAL_UNITS = ('QPM', 'seconds', 'MIDI_velocity', 'duration_ratio', 'CC64', 'CC64',
                  'CC64', 'seconds', 'CC64', 'seconds', 'CC67')
R_REASONS = ('valid', 'unmatched', 'ambiguous', 'tie_continuation', 'missing_performed_event',
             'outside_time_map', 'invalid_beat', 'grace_reference_unknown', 'unknown_key_offset',
             'nonpositive_key_duration', 'unknown_tie_duration', 'outside_expected_duration_map',
             'unknown_pedal', 'unknown_initial_pedal', 'incomplete_pedal', 'outside_pedal_observation',
             'missing_next_onset', 'nonfinite_transform')
R_REASONS += ('grace_no_written_duration', 'unavailable_time_map')


def note_performance_schema():
    return {'version': R_SCHEMA_VERSION, 'channels': list(R_CHANNELS), 'dimension': 11,
            'physical_units': list(PHYSICAL_UNITS), 'reasons': list(R_REASONS),
            'transform': ['log2', 'divide_by_local_tau', 'divide_by_127', 'log2',
                          'divide_by_127', 'divide_by_127', 'divide_by_127', 'divide_by_local_tau',
                          'divide_by_127', 'divide_by_local_tau', 'divide_by_127'],
            'hold': '[onset,key_offset]', 'gap': '[key_offset,next_strictly_later_onset)',
            'minimum_tie_break': 'earliest_time', 'tie_continuation': 'all_row_channels_unobserved',
            'tempo_statistics': 'one_valid_beat_per_record', 'missing_payload': 'finite_zero_with_mask',
            'centering': 'train_global_only; no piece/performance centering'}


def _array(value, dtype, shape, name):
    source = np.asarray(value)
    if dtype is bool and source.dtype != bool:
        raise ValueError(f'{name} must be boolean')
    if dtype is np.int64 and source.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain integer indices')
    result = np.array(source, dtype=dtype, copy=True)
    if result.shape != shape:
        raise ValueError(f'{name} shape must be {shape}')
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class NotePerformanceFeatures:
    record_key: str
    work_key: str
    split: str
    dataset: str
    score_ref: str
    score_source_sha256: str
    traversal_id: str
    row_keys: tuple[str, ...]
    values: np.ndarray
    mask: np.ndarray
    physical: np.ndarray
    physical_mask: np.ndarray
    reasons: np.ndarray
    beat_id: np.ndarray
    local_tau: np.ndarray
    expected_duration_sec: np.ndarray
    beats: BeatTimeMap
    metadata: tuple[dict, ...]
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.record_key, self.work_key, self.dataset, self.score_ref,
                                                       self.score_source_sha256, self.traversal_id)):
            raise ValueError('R identity and source fields must be explicit nonempty strings')
        if self.split not in {'train', 'validation', 'test', 'unassigned'}:
            raise ValueError('Unknown dataset split')
        n = len(self.row_keys)
        if len(set(self.row_keys)) != n or len(self.metadata) != n:
            raise ValueError('R rows require unique score occurrence keys and matching metadata')
        for name, dtype, shape in (
            ('values', float, (n, 11)), ('mask', bool, (n, 11)), ('physical', float, (n, 11)),
            ('physical_mask', bool, (n, 11)), ('reasons', np.int64, (n, 11)),
            ('beat_id', np.int64, (n,)), ('local_tau', float, (n,)), ('expected_duration_sec', float, (n,)),
        ):
            object.__setattr__(self, name, _array(getattr(self, name), dtype, shape, name))
        if not np.isfinite(self.values).all() or not np.isfinite(self.physical).all():
            raise ValueError('R and physical payloads must be finite')
        if np.any(self.values[~self.mask] != 0) or np.any(self.physical[~self.physical_mask] != 0):
            raise ValueError('Unobserved R/physical payloads must be zero')
        if np.any(self.mask & ~self.physical_mask):
            raise ValueError('Transformed observations need physical observations')
        if np.any((self.reasons < 0) | (self.reasons >= len(R_REASONS))) or np.any((self.reasons == 0) != self.mask):
            raise ValueError('R validity and reason codes disagree')
        if np.any((self.beat_id < -1) | (self.beat_id >= len(self.beats.valid))):
            raise ValueError('Invalid row-to-beat indices')
        if not np.isfinite(self.local_tau).all() or np.any(self.local_tau < 0):
            raise ValueError('Local quarter durations must be finite and nonnegative')
        if not np.isfinite(self.expected_duration_sec).all() or np.any(self.expected_duration_sec < 0):
            raise ValueError('Expected key durations must be finite and nonnegative')
        if any(meta.get('score_note_key') != key for key, meta in zip(self.row_keys, self.metadata)):
            raise ValueError('R metadata and row identities disagree')
        for row, meta in enumerate(self.metadata):
            if not meta.get('is_attack') and self.mask[row].any():
                raise ValueError('Tie continuations cannot own row reconstruction targets')
        if np.any(self.mask[:, (1, 7, 9)].any(axis=1) & (self.local_tau <= 0)):
            raise ValueError('Quarter timing targets require a valid local tempo')
        tempo_rows = np.flatnonzero(self.mask[:, 0])
        if len(tempo_rows):
            indices = self.beat_id[tempo_rows]
            if np.any(indices < 0) or not self.beats.valid[indices].all():
                raise ValueError('Tempo row observations require valid independent beats')
            if not np.array_equal(self.values[tempo_rows, 0], self.beats.log2_qpm[indices]):
                raise ValueError('Copied note tempo must equal the source beat tempo')

    @property
    def note_mask(self):
        result = np.ones(len(self.row_keys), bool)
        result.setflags(write=False)
        return result

    @property
    def fingerprint(self):
        digest = hashlib.sha256(json.dumps({'schema': note_performance_schema(), 'record': self.record_key,
                                            'work': self.work_key, 'split': self.split, 'dataset': self.dataset,
                                            'score_ref': self.score_ref, 'rows': self.row_keys,
                                            'provenance': self.provenance}, sort_keys=True, allow_nan=False).encode())
        for array in (self.values, self.mask, self.physical, self.physical_mask, self.reasons, self.beat_id,
                      self.local_tau, self.expected_duration_sec, self.beats.q, self.beats.seconds, self.beats.valid):
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()


class _PedalSignal:
    def __init__(self, performance, port, channel, number):
        self.observation = performance.observation(port, channel, number)
        self.end = performance.duration_sec if self.observation.end_sec is None else min(performance.duration_sec, self.observation.end_sec)
        stream = sorted((c for c in performance.controls if (c.port, c.channel, c.number) == (port, channel, number)
                         and c.time_sec >= self.observation.start_sec),
                        key=lambda c: (c.time_sec, c.order))
        at = {}
        for event in stream:
            at[event.time_sec] = event.value
        self.times = np.array(list(at), float)
        self.values = np.array(list(at.values()), float)

    def state(self, time):
        observation = self.observation
        if time < observation.start_sec or time > self.end:
            return None, 'outside_pedal_observation'
        if observation.status == 'unknown':
            return None, 'unknown_pedal'
        if not observation.complete:
            return None, 'incomplete_pedal'
        if observation.status == 'confirmed_absent':
            return 0., 'valid'
        index = np.searchsorted(self.times, time, side='right') - 1
        if index >= 0:
            return float(self.values[index]), 'valid'
        if observation.initial_value is not None:
            return float(observation.initial_value), 'valid'
        return None, 'unknown_initial_pedal'

    def minimum(self, start, stop, *, closed):
        value, reason = self.state(start)
        if value is None:
            return None, None, reason
        if stop < start or stop > self.end:
            return None, None, 'outside_pedal_observation'
        left = np.searchsorted(self.times, start, side='right')
        right = np.searchsorted(self.times, stop, side='right' if closed else 'left')
        times = np.concatenate(([start], self.times[left:right]))
        values = np.concatenate(([value], self.values[left:right]))
        index = int(np.argmin(values))
        return float(values[index]), float(times[index] - start), 'valid'


def extract_note_performance(score: NoteScoreFeatures, performance: PerformanceMidi, links: NoteAlignmentLinks,
                             beats: BeatTimeMap, *, record_key, work_key, split, dataset,
                             grace_reference_keys=(), provenance=None) -> NotePerformanceFeatures:
    """Calculate R before cropping. No C, peer performance, or residual cache is used."""
    keys = tuple(meta['occurrence_id'] for meta in score.metadata)
    if keys != links.occurrence_ids or keys != score.ties.occurrence_ids:
        raise ValueError('S/R alignment must use identical ordered score occurrence keys')
    n = len(keys)
    values, physical = np.zeros((n, 11)), np.zeros((n, 11))
    mask, physical_mask = np.zeros((n, 11), bool), np.zeros((n, 11), bool)
    reason = np.full((n, 11), R_REASONS.index('unmatched'), np.int64)
    beat_id, tau, expected = np.full(n, -1, np.int64), np.zeros(n), np.zeros(n)
    by_id = {note.note_id: note for note in performance.notes}
    onsets = np.array(sorted(note.onset_sec for note in performance.notes))
    signals = {(port, channel, number): _PedalSignal(performance, port, channel, number)
               for port, channel in performance.selected_channels for number in (64, 67)}
    grace_keys = set(grace_reference_keys)
    if grace_keys - set(keys):
        raise ValueError('Unknown grace timing reference keys')
    metadata = []
    event_owners = Counter(ids[0] for attack, ids in zip(links.attack_mask, links.performed_ids) if attack and len(ids) == 1)

    def invalid(row, channels, name):
        reason[row, channels] = R_REASONS.index(name)

    def observed(row, channel, original, transformed=None):
        if not np.isfinite(original):
            invalid(row, channel, 'nonfinite_transform')
            return
        physical[row, channel], physical_mask[row, channel] = original, True
        if transformed is not None and np.isfinite(transformed):
            values[row, channel], mask[row, channel], reason[row, channel] = transformed, True, 0
        elif transformed is not None:
            invalid(row, channel, 'nonfinite_transform')

    for row, meta in enumerate(score.metadata):
        q = float(Fraction(str(meta['unfolded_onset_q'])))
        index = beats.beat_at(q)
        beat_id[row] = index
        if index >= 0 and beats.valid[index]:
            tau[row] = beats.tau[index]
        row_meta = {'score_note_key': keys[row], 'score_onset_q': q, 'source_id': meta['source_id'],
                    'occurrence': meta['occurrence'], 'voice': meta['voice'], 'staff': meta['staff'],
                    'score_duration_q': float(Fraction(str(meta['duration_q']))),
                    'written_onset_q': meta['written_onset_q'], 'score_onset_group_id': int(score.note_to_onset[row]),
                    'beat_id': index, 'tie_chain_duration_q': float(score.ties.chain_duration_q[row]),
                    'tie_chain_known': bool(score.ties.chain_known[row]),
                    'attack_owner_key': score.ties.occurrence_ids[int(score.ties.owner_indices[row])],
                    'tie_chain_id': meta['tie_chain_id'], 'is_attack': bool(score.ties.is_attack[row]),
                    'alignment_status': links.status[row], 'performed_ids': links.performed_ids[row],
                    'is_grace': bool(score.flags[row, FLAG_FIELDS.index('is_grace')])}
        metadata.append(row_meta)
        if not score.ties.is_attack[row]:
            invalid(row, slice(None), 'tie_continuation')
            continue
        if not links.attack_mask[row]:
            invalid(row, slice(None), 'ambiguous' if links.status[row] == 'ambiguous' else 'unmatched')
            continue
        if len(links.performed_ids[row]) != 1 or links.performed_ids[row][0] not in by_id:
            invalid(row, slice(None), 'missing_performed_event')
            continue
        if event_owners[links.performed_ids[row][0]] != 1:
            invalid(row, slice(None), 'ambiguous')
            row_meta['alignment_status'] = 'ambiguous_event_ownership'
            continue
        note = by_id[links.performed_ids[row][0]]
        row_meta.update(performed_note_id=note.note_id, onset_sec=note.onset_sec,
                        key_offset_sec=note.key_offset_sec, port=note.port, channel=note.channel,
                        key_offset_source='raw_MIDI_note_off' if note.key_offset_sec is not None else 'unknown',
                        track=note.track, source_order=note.order)
        timing_reason = ('unavailable_time_map' if not len(beats.valid) else
                         'outside_time_map' if index < 0 else 'invalid_beat')
        for channel in (0, 1, 7, 9):
            invalid(row, channel, timing_reason)
        if tau[row] > 0:
            observed(row, 0, 60. / tau[row], beats.log2_qpm[index])
        reference = beats.time_at(q)
        row_meta['reference_onset_sec'] = reference
        if row_meta['is_grace'] and keys[row] not in grace_keys:
            invalid(row, 1, 'grace_reference_unknown')
        elif reference is not None:
            observed(row, 1, note.onset_sec - reference,
                     (note.onset_sec - reference) / tau[row] if tau[row] > 0 else None)
        observed(row, 2, note.velocity, note.velocity / 127.)
        invalid(row, 3, 'unknown_key_offset')
        if note.key_offset_sec is not None:
            duration = note.key_offset_sec - note.onset_sec
            row_meta['actual_key_duration_sec'] = duration
            if duration <= 0 or row_meta['is_grace']:
                invalid(row, 3, 'nonpositive_key_duration' if duration <= 0 else 'grace_no_written_duration')
            elif not score.ties.chain_known[row]:
                invalid(row, 3, 'unknown_tie_duration')
            else:
                estimate = beats.expected_duration(q, score.ties.chain_duration_q[row])
                if estimate is None:
                    invalid(row, 3, 'outside_expected_duration_map')
                else:
                    expected[row] = estimate
                    observed(row, 3, duration / estimate, np.log2(duration) - np.log2(estimate))
        sustain, soft = signals[note.port, note.channel, 64], signals[note.port, note.channel, 67]
        for channel, signal, time in ((4, sustain, note.onset_sec), (10, soft, note.onset_sec)):
            cc, why = signal.state(time)
            if cc is None:
                invalid(row, channel, why)
            else:
                observed(row, channel, cc, cc / 127.)
        if note.key_offset_sec is None:
            invalid(row, slice(5, 10), 'unknown_key_offset')
            continue
        offset = note.key_offset_sec
        cc, why = sustain.state(offset)
        if cc is None:
            invalid(row, 5, why)
        else:
            observed(row, 5, cc, cc / 127.)
        minimum, delay, why = sustain.minimum(note.onset_sec, offset, closed=True)
        if minimum is None:
            invalid(row, [6, 7], why)
        else:
            observed(row, 6, minimum, minimum / 127.)
            observed(row, 7, delay, delay / tau[row] if tau[row] > 0 else None)
        next_index = int(np.searchsorted(onsets, offset, side='right'))
        if next_index == len(onsets):
            invalid(row, [8, 9], 'missing_next_onset')
            continue
        next_onset = float(onsets[next_index])
        row_meta['next_onset_sec'] = next_onset
        minimum, delay, why = sustain.minimum(offset, next_onset, closed=False)
        if minimum is None:
            invalid(row, [8, 9], why)
        else:
            observed(row, 8, minimum, minimum / 127.)
            observed(row, 9, delay, delay / tau[row] if tau[row] > 0 else None)
    obs_meta = {f'{port}:{channel}:{number}': performance.observation(port, channel, number).__dict__
                for port, channel in performance.selected_channels for number in (64, 67)}
    return NotePerformanceFeatures(record_key, work_key, split, dataset, score.cache_key,
                                    score.source_sha256, score.traversal_id, keys, values, mask, physical,
                                    physical_mask, reason, beat_id, tau, expected, beats, tuple(metadata),
                                    {**(provenance or {}), 'performance': performance.provenance,
                                     'pedal_observations': obs_meta, 'midi_diagnostics': performance.diagnostics,
                                     'map': beats.provenance})


def inverse_r_physical(values, mask, local_tau):
    """Invert deterministic transforms; ratios/delays do not reconstruct a MIDI stream."""
    values, mask, tau = np.asarray(values, float), np.asarray(mask), np.asarray(local_tau, float)
    if values.ndim != 2 or values.shape[1] != 11 or mask.shape != values.shape or mask.dtype != bool or tau.shape != values.shape[:1]:
        raise ValueError('Expected R[N,11], boolean mask, and local_tau[N]')
    if not np.isfinite(values[mask]).all():
        raise ValueError('Observed R must be finite')
    output, valid = np.zeros_like(values), mask.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        for channel in range(11):
            selected = mask[:, channel].copy()
            if channel in (0, 3):
                output[selected, channel] = np.exp2(values[selected, channel])
            elif channel in (1, 7, 9):
                selected &= np.isfinite(tau) & (tau > 0)
                valid[:, channel] = selected
                output[selected, channel] = values[selected, channel] * tau[selected]
            else:
                output[selected, channel] = values[selected, channel] * 127
    if not np.isfinite(output[valid]).all():
        raise ValueError('Inverse physical transform exceeds floating point range')
    return output, valid


def summarize_note_performance(features):
    result = {'record_key': features.record_key, 'rows': len(features.row_keys),
              'beats': len(features.beats.valid), 'valid_beats': int(features.beats.valid.sum()),
              'alignment': dict(Counter(meta['alignment_status'] for meta in features.metadata)), 'channels': {}}
    for index, name in enumerate(R_CHANNELS):
        valid = features.values[features.mask[:, index], index]
        result['channels'][name] = {'valid': len(valid), 'total': len(features.row_keys),
                                    'quantiles': np.quantile(valid, [0, .05, .5, .95, 1]).tolist() if len(valid) else None,
                                    'reasons': dict(Counter(R_REASONS[int(r)] for r in features.reasons[:, index]))}
    return result
