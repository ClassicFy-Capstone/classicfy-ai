"""Fixed 23D, reference-relative eight-measure profiles; raw R is never changed."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from preprocessing.musicxml_score import MusicXMLScore
from preprocessing.score_note_mapping import build_tie_mapping
from .note_performance import NotePerformanceFeatures, note_performance_schema


G_NAMES = ('tempo_bias', 'tempo_spread', 'tempo_roughness', 'onset_bias', 'onset_spread',
           'velocity_bias', 'velocity_spread', 'articulation_bias', 'articulation_spread',
           'sustain_onset_bias', 'sustain_onset_spread', 'sustain_offset_bias', 'sustain_offset_spread',
           'sustain_min_hold_bias', 'sustain_min_hold_spread', 'hold_delay_bias', 'hold_delay_spread',
           'sustain_min_gap_bias', 'sustain_min_gap_spread', 'gap_delay_bias', 'gap_delay_spread',
           'soft_onset_bias', 'soft_onset_spread')
G_BLOCKS = {'tempo': [0], 'timing': [1, 2, 3, 4], 'dynamics': [5, 6],
            'articulation': [7, 8], 'pedaling': list(range(9, 23))}


@dataclass(frozen=True)
class StyleConfig:
    window_measures: int = 8
    stride_measures: int = 8
    min_reference_per_cell: int = 3
    min_coverage: float = .80
    min_valid_beats: int = 8
    min_valid_onset_groups: int = 8
    min_roughness_edges: int = 7
    min_scaler_samples: int = 20
    min_calibration_pairs: int = 100
    max_calibration_pairs: int = 20000
    min_candidates: int = 10
    near_quantile: float = .20
    far_quantile: float = .80
    gap_fraction: float = .10
    epsilon: float = 1e-8
    seed: int = 2026

    def __post_init__(self):
        if (self.window_measures, self.stride_measures) != (8, 8):
            raise ValueError('This schema requires eight-measure windows and stride eight')
        for name in ('min_reference_per_cell', 'min_valid_beats', 'min_valid_onset_groups',
                     'min_roughness_edges', 'min_scaler_samples', 'min_calibration_pairs',
                     'max_calibration_pairs', 'min_candidates'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'Invalid positive integer config: {name}')
        if (not 0 < self.min_coverage <= 1 or not 0 < self.near_quantile < self.far_quantile < 1
                or not math.isfinite(self.gap_fraction) or self.gap_fraction <= 0
                or not math.isfinite(self.epsilon) or self.epsilon <= 0
                or self.min_calibration_pairs > self.max_calibration_pairs
                or isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError('Invalid style thresholds or seed')


def stable_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def style_schema():
    return {'version': 'eight-bar-style-v1', 'names': list(G_NAMES), 'blocks': {key: list(value) for key, value in G_BLOCKS.items()},
            'R': note_performance_schema(), 'quantile': 'merge-equal-values; weighted-midpoint-linear',
            'fit_quantile': 'numpy linear', 'reference': 'fixed train cohort median; target may be included',
            'distance': 'anchor-conditional; mean of block mean squared differences'}


def artifact(kind, config, *, parents=None, **payload):
    result = {'kind': kind, 'schema': style_schema(), 'config': asdict(config),
              'config_id': stable_id(asdict(config)), 'parents': parents or {}, **payload}
    result['id'] = stable_id(result)
    return result


def check_artifact(value, kind=None, config=None):
    if value.get('schema') != style_schema() or (kind is not None and value.get('kind') != kind):
        raise ValueError('Style artifact schema/kind mismatch')
    if value.get('id') != stable_id({k: v for k, v in value.items() if k != 'id'}):
        raise ValueError('Style artifact integrity mismatch')
    parsed = StyleConfig(**value['config'])
    if value['config_id'] != stable_id(asdict(parsed)) or (config is not None and parsed != config):
        raise ValueError('Style artifact config mismatch')
    return parsed


def weighted_quantile(values, weights, probabilities):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    p = np.asarray(probabilities, float)
    if values.ndim != 1 or weights.shape != values.shape or np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError('Invalid weighted quantile arguments')
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError('Weighted quantile needs observed, positive-weight samples')
    unique, inverse = np.unique(values[valid], return_inverse=True)
    mass = np.bincount(inverse, weights=weights[valid])
    centers = (np.cumsum(mass) - mass / 2) / mass.sum()
    return np.interp(p, centers, unique)


def beat_keys(r):
    return [f'{float(a).hex()}:{float(b).hex()}' for a, b in zip(r.beats.q[:-1], r.beats.q[1:])]


def cohort_key(r):
    return stable_id([r.work_key, r.score_source_sha256, r.traversal_id])


@dataclass(frozen=True)
class StyleRecording:
    performance: NotePerformanceFeatures
    xml: MusicXMLScore
    recording_id: str
    performance_id: str = ''

    def __post_init__(self):
        r, xml = self.performance, self.xml
        if not isinstance(self.recording_id, str) or not self.recording_id:
            raise ValueError('Explicit recording identity or raw MIDI hash required')
        if (r.score_source_sha256 != xml.source_sha256 or r.traversal_id != xml.traversal_id
                or r.row_keys != tuple(n.occurrence_id for n in xml.notes)
                or any(m['score_onset_q'] != float(n.onset) for m, n in zip(r.metadata, xml.notes))):
            raise ValueError('XML and R source, route, rows and quarter coordinates must match')
        ties = build_tie_mapping(xml)
        onset_groups = {q: i for i, q in enumerate(sorted({n.onset for n in xml.notes}))}
        if any(bool(m['is_attack']) != bool(ties.is_attack[i])
               or bool(m['is_grace']) != n.grace
               or m['score_onset_group_id'] != onset_groups[n.onset]
               for i, (m, n) in enumerate(zip(r.metadata, xml.notes))):
            raise ValueError('R tie, grace and onset groups must agree with the score')
        if not self.performance_id:
            object.__setattr__(self, 'performance_id', f'{r.dataset}:{r.record_key}')
        if not isinstance(self.performance_id, str):
            raise ValueError('Performance identity must be a string')


def validate_recordings(recordings):
    by_work, by_score, by_recording, by_performance = {}, {}, {}, {}
    for rec in recordings:
        r = rec.performance
        if r.split == 'unassigned':
            raise ValueError('Assign work-level splits before making style windows')
        for mapping, key in ((by_work, r.work_key), (by_score, r.score_source_sha256),
                             (by_recording, rec.recording_id)):
            if key in mapping and mapping[key] != r.split:
                raise ValueError('Work, identical score or recording crosses splits')
            mapping[key] = r.split
        identity = (rec.recording_id, r.fingerprint)
        if rec.performance_id in by_performance and by_performance[rec.performance_id] != identity:
            raise ValueError('Conflicting duplicate performance ID')
        by_performance[rec.performance_id] = identity


def make_windows(rec):
    r, xml = rec.performance, rec.xml
    output = []
    positions = np.array([m['score_onset_q'] for m in r.metadata])
    eligible = np.array([n.grace is False and bool(r.metadata[i]['is_attack']) for i, n in enumerate(xml.notes)], dtype=bool)
    groups = np.array([m['score_onset_group_id'] for m in r.metadata], dtype=int)
    bkeys = beat_keys(r)
    for start in range(0, len(xml.visits), 8):
        visits = xml.visits[start:start + 8]
        left, right = float(visits[0].onset), float(visits[-1].onset + visits[-1].duration)
        rows = np.flatnonzero((positions >= left) & (positions < right))
        attacks = rows[eligible[rows]]
        sizes = Counter(groups[attacks])
        weights = [1. / sizes[groups[i]] if eligible[i] else 0. for i in rows]
        denom = sum(weights)
        coverage = sum(w for i, w in zip(rows, weights) if r.metadata[i].get('performed_note_id')) / denom if denom else 0.
        bis = np.flatnonzero((r.beats.q[:-1] < right) & (r.beats.q[1:] > left))
        occurrences = [f'{v.measure}@{v.occurrence}' for v in visits]
        same_key = stable_id([r.work_key, r.score_source_sha256, r.traversal_id, occurrences])
        window = {'work_id': r.work_key, 'score_version_id': r.score_source_sha256,
                  'repeat_route_id': r.traversal_id, 'performance_id': rec.performance_id,
                  'recording_id': rec.recording_id, 'split': r.split, 'cohort_key': cohort_key(r),
                  'same_window_key': same_key, 'ordered_measure_occurrence_keys': occurrences,
                  'measure_visits': [{'key': k, 'q_start': str(v.onset), 'duration_q': str(v.duration),
                                     'pickup': bool(xml.implicit_measures[v.measure])}
                                    for k, v in zip(occurrences, visits)],
                  'q_start': left, 'q_end': right, 'actual_measure_count': len(visits), 'tail': len(visits) != 8,
                  'score_note_keys': [r.row_keys[i] for i in rows], 'row_indices': rows.tolist(),
                  'note_weights': weights, 'onset_group_keys': [int(groups[i]) for i in rows],
                  'beat_keys': [bkeys[i] for i in bis], 'beat_indices': bis.tolist(),
                  'N': len(rows), 'valid_alignment_coverage': coverage,
                  'R_fingerprint': r.fingerprint, 'score_ref': r.score_ref,
                  'dataset': r.dataset, 'record_key': r.record_key}
        window['window_id'] = stable_id([rec.performance_id, same_key, r.fingerprint])
        output.append(window)
    return output


def fit_reference(recordings, config=StyleConfig()):
    recordings = sorted(recordings, key=lambda rec: rec.performance_id)
    validate_recordings(recordings)
    if any(rec.performance.split != 'train' for rec in recordings):
        raise ValueError('Reference fit requires explicit train recordings only')
    tables, seen, duplicates = {}, set(), []
    for rec in recordings:
        r, key = rec.performance, cohort_key(rec.performance)
        # The same recording contributes at most once per verified score/route cohort.
        identity = key, rec.recording_id
        if identity in seen:
            duplicates.append(rec.performance_id)
            continue
        seen.add(identity)
        table = tables.setdefault(key, {'work_id': r.work_key, 'score_version_id': r.score_source_sha256,
                                       'repeat_route_id': r.traversal_id, 'cohort': [], 'notes': {}, 'beats': {}})
        table['cohort'].append({'performance_id': rec.performance_id, 'recording_id': rec.recording_id,
                                'R_fingerprint': r.fingerprint, 'split': 'train'})
        for row, event in enumerate(r.row_keys):
            cells = table['notes'].setdefault(event, [[] for _ in range(10)])
            for k in range(1, 11):
                if r.mask[row, k]:
                    cells[k - 1].append((float(r.values[row, k]), rec.performance_id))
        for row, event in enumerate(beat_keys(r)):
            cells = table['beats'].setdefault(event, [[]])
            if r.beats.valid[row]:
                cells[0].append((float(r.beats.log2_qpm[row]), rec.performance_id))
    for table in tables.values():
        for domain in ('notes', 'beats'):
            for event, cells in table[domain].items():
                counts = [len(cell) for cell in cells]
                known = [n >= config.min_reference_per_cell for n in counts]
                table[domain][event] = {'values': [float(np.median([v for v, _ in cell])) if ok else 0.
                                                  for cell, ok in zip(cells, known)],
                                      'mask': known, 'counts': counts,
                                      'performance_ids': [[p for _, p in cell] for cell in cells]}
    return artifact('reference', config, fit_split='train', tables=tables, excluded_duplicates=duplicates,
                    training_recordings=[{'performance_id': rec.performance_id, 'R_fingerprint': rec.performance.fingerprint}
                                         for rec in recordings])


@dataclass(frozen=True)
class NotePerformanceDelta:
    """One verified, read-only R/reference snapshot shared by its windows."""
    arrays: tuple
    reference_id: str
    config_id: str
    R_fingerprint: str
    performance: NotePerformanceFeatures

    def __post_init__(self):
        for value in self.arrays:
            value.setflags(write=False)

    def __iter__(self):
        return iter(self.arrays)

    def __getitem__(self, index):
        return self.arrays[index]

    def __len__(self):
        return len(self.arrays)


def reference_delta(r, reference):
    check_artifact(reference, 'reference')
    table = reference['tables'].get(cohort_key(r), {})
    delta, mask, counts = np.zeros_like(r.values), np.zeros_like(r.mask), np.zeros(r.mask.shape, int)
    for row, key in enumerate(r.row_keys):
        cell = table.get('notes', {}).get(key)
        if cell:
            observed = r.mask[row, 1:] & np.asarray(cell['mask'])
            mask[row, 1:] = observed
            delta[row, 1:] = np.where(observed, r.values[row, 1:] - cell['values'], 0)
            counts[row, 1:] = cell['counts']
    bd, bm, bc = np.zeros(len(r.beats.valid)), np.zeros(len(r.beats.valid), bool), np.zeros(len(r.beats.valid), int)
    for row, key in enumerate(beat_keys(r)):
        cell = table.get('beats', {}).get(key)
        if cell:
            bc[row] = cell['counts'][0]
            bm[row] = bool(r.beats.valid[row] and cell['mask'][0])
            if bm[row]:
                bd[row] = r.beats.log2_qpm[row] - cell['values'][0]
    return NotePerformanceDelta((delta, mask, counts, bd, bm, bc), reference['id'], reference['config_id'], r.fingerprint, r)


def summarize_window(r, window, reference, config=StyleConfig(), *, deltas=None):
    if deltas is None:
        deltas = reference_delta(r, reference)
    if isinstance(deltas, NotePerformanceDelta):
        if (deltas.performance is not r or deltas.reference_id != reference['id'] or deltas.config_id != stable_id(asdict(config))
                or window['R_fingerprint'] != deltas.R_fingerprint):
            raise ValueError('Window/R/reference/config delta snapshot mismatch')
    else:
        check_artifact(reference, 'reference', config)
        if window['R_fingerprint'] != r.fingerprint:
            raise ValueError('Window and R fingerprint mismatch')
    delta, mask, counts, bd, bm, bc = deltas
    raw, known = np.zeros(23), np.zeros(23, bool)
    coverage, valid_counts, reference_counts, channel_reasons = [], [], [], []
    rows, bis = np.array(window['row_indices'], int), np.array(window['beat_indices'], int)
    for k in range(11):
        if k == 0:
            weights = np.maximum(0, np.minimum(r.beats.q[bis + 1], window['q_end'])
                                 - np.maximum(r.beats.q[bis], window['q_start']))
            valid, values = bm[bis], bd[bis]
            denominator = window['q_end'] - window['q_start']
            n_valid = int(valid.sum())
            minimum = config.min_valid_beats
            refs = bc[bis]
        else:
            weights = np.array(window['note_weights'])
            valid, values = mask[rows, k] & (weights > 0), delta[rows, k]
            denominator = weights.sum()
            groups = np.array(window['onset_group_keys'])
            n_valid = len(set(groups[valid]))
            minimum = config.min_valid_onset_groups
            refs = counts[rows, k]
        cov = float(weights[valid].sum() / denominator) if denominator > 0 else 0.
        coverage.append(cov)
        valid_counts.append(n_valid)
        eligible_refs = refs[weights > 0]
        reference_counts.append({'min': int(eligible_refs.min()) if len(eligible_refs) else 0,
                                 'max': int(eligible_refs.max()) if len(eligible_refs) else 0,
                                 'sufficient_fraction': float(weights[refs >= config.min_reference_per_cell].sum() / denominator)
                                 if denominator > 0 else 0.})
        why = []
        if window['tail']:
            why.append('tail')
        if denominator <= 0:
            why.append('no_population')
        if cov < config.min_coverage and reference_counts[-1]['sufficient_fraction'] < config.min_coverage:
            why.append('reference_insufficient')
        if cov < config.min_coverage:
            why.append('coverage_insufficient')
        if n_valid < minimum:
            why.append('observations_insufficient')
        channel_reasons.append(why)
        if not why:
            median, low, high = weighted_quantile(values[valid], weights[valid], [.5, .1, .9])
            indices = [0, 1] if k == 0 else [2 * k + 1, 2 * k + 2]
            raw[indices], known[indices] = [median, high - low], True
    starts = r.beats.q[:-1]
    edges = ((starts[:-1] >= window['q_start']) & (starts[1:] < window['q_end']))
    valid_edges = edges & bm[:-1] & bm[1:]
    gaps = np.diff(starts)
    edge_total, edge_valid = float(gaps[edges].sum()), float(gaps[valid_edges].sum())
    edge_coverage = edge_valid / edge_total if edge_total > 0 else 0.
    if (known[0] and valid_edges.sum() >= config.min_roughness_edges
            and edge_coverage >= config.min_coverage and edge_valid > 0):
        raw[2] = np.abs(np.diff(bd))[valid_edges].sum() / edge_valid
        known[2] = True
    reasons = sorted({reason for channel in channel_reasons[:4] for reason in channel})
    if not known[:9].all():
        reasons.append('core_missing')
    result = {'window': window, 'g_raw': raw.tolist(), 'g_mask': known.tolist(),
              'per_channel_coverage': coverage, 'valid_counts': valid_counts, 'reference_counts': reference_counts,
              'channel_reasons': channel_reasons, 'roughness_edges': int(valid_edges.sum()),
              'roughness_coverage': edge_coverage, 'sampling_eligible': bool(known[:9].all() and not window['tail']),
              'ineligible_reason_codes': sorted(set(reasons)), 'reference_id': reference['id']}
    result['profile_id'] = stable_id(result)
    return result


def unique_profiles(profiles, *, train_only=False):
    output, seen, performance_seen = [], set(), set()
    works, recordings = {}, {}
    for p in sorted(profiles, key=lambda p: p['window']['window_id']):
        w = p['window']
        if train_only and w['split'] != 'train':
            raise ValueError('Profile fit requires train only')
        for mapping, key in ((works, w['work_id']), (recordings, w['recording_id'])):
            if key in mapping and mapping[key] != w['split']:
                raise ValueError('Profile work or recording crosses splits')
            mapping[key] = w['split']
        if w['tail']:
            continue
        identity = w['recording_id'], w['same_window_key']
        performance_key = w['performance_id'], w['same_window_key']
        if identity in seen or performance_key in performance_seen:
            continue
        seen.add(identity)
        performance_seen.add(performance_key)
        raw, mask = np.asarray(p['g_raw']), np.asarray(p['g_mask'])
        if (raw.shape != (23,) or mask.shape != (23,) or mask.dtype != bool
                or not np.isfinite(raw).all() or np.any(raw[~mask] != 0)):
            raise ValueError('Expected finite masked 23D profiles')
        output.append(p)
    return output


def fit_profile_scaler(profiles, config=StyleConfig(), *, profiles_id):
    profiles = unique_profiles(profiles, train_only=True)
    center, scale, active, counts, status = [], [], [], [], []
    for j in range(23):
        values = [p['g_raw'][j] for p in profiles if p['g_mask'][j]]
        n = len(values)
        c = float(np.median(values)) if n else 0.
        sd = float(np.diff(np.quantile(values, [.25, .75], method='linear'))[0]) if n else 0.
        state = 'iqr'
        if sd <= config.epsilon:
            sd = float(np.std(values, ddof=0)) if n else 0.
            state = 'std_fallback'
        enabled = n >= config.min_scaler_samples and sd > config.epsilon
        if not enabled:
            state = 'insufficient_samples' if n < config.min_scaler_samples else 'constant'
        center.append(c)
        scale.append(sd if enabled else 1.)
        active.append(enabled)
        counts.append(n)
        status.append(state)
    failed = [name for name, indices in list(G_BLOCKS.items())[:4] if not any(active[i] for i in indices)]
    return artifact('profile-scaler', config, parents={'profiles': profiles_id}, fit_split='train',
                    center=center, scale=scale, trained_feature_mask=active, valid_counts=counts, status=status,
                    training_profile_ids=[p['profile_id'] for p in profiles],
                    metric_ready=not failed, calibration_failure_blocks=failed)


@dataclass(frozen=True)
class FrozenProfileScaler:
    """Compact immutable view of an integrity-checked global g scaler."""
    center: np.ndarray
    scale: np.ndarray
    trained: np.ndarray
    artifact_id: str

    def __post_init__(self):
        for field, dtype in (('center', float), ('scale', float), ('trained', bool)):
            value = np.array(getattr(self, field), dtype=dtype, copy=True)
            if value.shape != (23,):
                raise ValueError('Expected 23 scaler features')
            value.setflags(write=False)
            object.__setattr__(self, field, value)
        if (not self.artifact_id or not np.isfinite(self.center).all()
                or not np.isfinite(self.scale).all() or np.any(self.scale <= 0)):
            raise ValueError('Invalid g scaler snapshot')

    @classmethod
    def from_artifact(cls, scaler):
        check_artifact(scaler, 'profile-scaler')
        return cls(scaler['center'], scaler['scale'], scaler['trained_feature_mask'], scaler['id'])


def scale_profile(profile, scaler):
    if not isinstance(scaler, FrozenProfileScaler):
        scaler = FrozenProfileScaler.from_artifact(scaler)
    valid = np.asarray(profile['g_mask']) & scaler.trained
    values = np.zeros(23)
    values[valid] = (np.asarray(profile['g_raw'])[valid] - scaler.center[valid]) / scaler.scale[valid]
    if not np.isfinite(values).all():
        raise ValueError('Profile scaling overflow')
    return {**profile, 'g_scaled': values.tolist(), 'g_scaled_mask': valid.tolist(), 'scaler_id': scaler.artifact_id}


def scale_profiles(profiles, scaler):
    """Validate the full scaler once for bulk transform, preserving all outputs."""
    if not isinstance(scaler, FrozenProfileScaler):
        scaler = FrozenProfileScaler.from_artifact(scaler)
    return [scale_profile(p, scaler) for p in profiles]
