"""Whole-recording 23D style statistics, separate from eight-bar g artifacts."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib

import numpy as np

from .style_profile import (
    G_BLOCKS, G_NAMES, StyleConfig, cohort_key, stable_id, validate_recordings, weighted_quantile,
)


@dataclass(frozen=True)
class PerformanceStyleConfig:
    min_reference_per_cell: int = 3
    min_coverage: float = .8
    min_valid_beats: int = 8
    min_valid_onset_groups: int = 8
    min_roughness_edges: int = 7
    min_scaler_samples: int = 20
    epsilon: float = 1e-8

    def __post_init__(self):
        StyleConfig(**asdict(self))


def performance_style_schema():
    return {'version': 'whole-performance-style-v1', 'names': list(G_NAMES),
            'blocks': G_BLOCKS, 'scope': 'entire score traversal, including tail',
            'reference': 'same split/work/score/traversal; independent recordings; target included',
            'statistics': 'reference residual weighted median and Q90-Q10; adjacent tempo variation per quarter',
            'weights': 'tempo: score-quarter interval; notes: 1/eligible attacks at score onset',
            'quantile': 'merge equal values; weighted midpoint linear interpolation',
            'scaler': 'train unique recordings only; median/IQR; population std fallback',
            'missing': 'finite zero with explicit mask; missing weight never redistributed within chord'}


def array_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _cell_median(values, masks, minimum):
    counts = masks.sum(0)
    ordered = np.sort(np.where(masks, values, np.inf), axis=0)
    columns = np.arange(values.shape[1])
    low = np.maximum(0, (counts - 1) // 2)
    high = np.maximum(0, counts // 2)
    known = counts >= minimum
    median = np.where(known, (ordered[low, columns] + ordered[high, columns]) / 2, 0.)
    return median, known, counts


def fit_performance_reference(records, config=PerformanceStyleConfig()):
    """Fit one explicitly identified split/cohort without relabeling heldout data."""
    if not records:
        raise ValueError('Reference needs recordings')
    records = sorted(records, key=lambda r: r.performance_id)
    validate_recordings(records)
    first = records[0].performance
    if len({r.performance.split for r in records}) != 1:
        raise ValueError('Reference cannot mix splits')
    if len({cohort_key(r.performance) for r in records}) != 1:
        raise ValueError('Reference cannot mix work/score/traversal cohorts')
    unique, duplicates, seen = [], [], set()
    for rec in records:
        r = rec.performance
        if r.row_keys != first.row_keys:
            raise ValueError('Cohort score rows must match')
        if rec.recording_id in seen:
            duplicates.append(rec.performance_id)
        else:
            unique.append(rec)
            seen.add(rec.recording_id)
    shape = first.values.shape
    values, masks, counts = np.zeros(shape), np.zeros(shape, bool), np.zeros(shape, np.int64)
    for k in range(1, 11):
        values[:, k], masks[:, k], counts[:, k] = _cell_median(
            np.stack([r.performance.values[:, k] for r in unique]),
            np.stack([r.performance.mask[:, k] for r in unique]), config.min_reference_per_cell)
    # Alignment-derived tables can start/end at different beats. Match exact
    # score-quarter intervals, never array offsets or interpolated missing beats.
    cells = {}
    for rec in unique:
        beats = rec.performance.beats
        for i, interval in enumerate(zip(beats.q[:-1], beats.q[1:])):
            cell = cells.setdefault(interval, [])
            if beats.valid[i]:
                cell.append(beats.log2_qpm[i])
    intervals = sorted(cells)
    bc = np.array([len(cells[k]) for k in intervals], dtype=np.int64)
    bm = bc >= config.min_reference_per_cell
    bv = np.array([float(np.median(cells[k])) if ok else 0. for k, ok in zip(intervals, bm)])
    arrays = {'note_values': values, 'note_mask': masks, 'note_counts': counts,
              'beat_values': bv, 'beat_mask': bm, 'beat_counts': bc,
              'beat_intervals': np.array(intervals, dtype=float).reshape(-1, 2)}
    metadata = {'schema': performance_style_schema()['version'], 'config': asdict(config),
                'cohort_id': cohort_key(first), 'split': first.split, 'work_key': first.work_key,
                'score_sha256': first.score_source_sha256, 'traversal_id': first.traversal_id,
                'row_keys_sha256': stable_id(first.row_keys), 'reference_role':
                'training' if first.split == 'train' else 'evaluation-labels-only; requires peer performances',
                'members': [{'performance_id': r.performance_id, 'recording_id': r.recording_id,
                             'R_fingerprint': r.performance.fingerprint} for r in unique],
                'excluded_duplicate_ids': duplicates, 'arrays_sha256': array_digest(arrays)}
    metadata['id'] = stable_id(metadata)
    return metadata, arrays


def _interval_length(intervals):
    total, end = 0., -np.inf
    for left, right in sorted(intervals):
        total += max(0., right - max(left, end))
        end = max(end, right)
    return total


def summarize_performance(rec, reference, arrays, config=PerformanceStyleConfig(), *, ae_intervals=()):
    r, xml = rec.performance, rec.xml
    if (reference['config'] != asdict(config) or reference['cohort_id'] != cohort_key(r)
            or reference['split'] != r.split or reference['row_keys_sha256'] != stable_id(r.row_keys)):
        raise ValueError('Performance/reference/config correspondence mismatch')
    left, right = float(xml.visits[0].onset), float(xml.visits[-1].onset + xml.visits[-1].duration)
    if right <= left:
        raise ValueError('Full score needs positive extent')
    q = np.array([m['score_onset_q'] for m in r.metadata])
    eligible = np.array([m['is_attack'] and not m['is_grace'] for m in r.metadata]) & (q >= left) & (q < right)
    groups = np.array([m['score_onset_group_id'] for m in r.metadata])
    sizes = Counter(groups[eligible])
    nw = np.array([1. / sizes[g] if ok else 0. for g, ok in zip(groups, eligible)])
    bw = np.maximum(0., np.minimum(r.beats.q[1:], right) - np.maximum(r.beats.q[:-1], left))
    delta = r.values - arrays['note_values']
    observed = r.mask & arrays['note_mask']
    beat_lookup = {tuple(interval): i for i, interval in enumerate(arrays['beat_intervals'])}
    ref_values, ref_mask, ref_counts = np.zeros(len(bw)), np.zeros(len(bw), bool), np.zeros(len(bw), np.int64)
    for i, interval in enumerate(zip(r.beats.q[:-1], r.beats.q[1:])):
        at = beat_lookup.get(interval)
        if at is not None:
            ref_values[i], ref_mask[i], ref_counts[i] = (arrays[key][at] for key in ('beat_values', 'beat_mask', 'beat_counts'))
    bd = r.beats.log2_qpm - ref_values
    bm = r.beats.valid & ref_mask & (bw > 0)
    raw, known = np.zeros(23), np.zeros(23, bool)
    diagnostics = []
    for k in range(11):
        if k == 0:
            weights, values, valid, refs = bw, bd, bm, ref_counts
            positions, denominator = r.beats.q[:-1], right - left
            count, minimum = int(valid.sum()), config.min_valid_beats
        else:
            weights, values = nw, delta[:, k]
            valid, refs = observed[:, k] & eligible, arrays['note_counts'][:, k]
            positions, denominator = q, nw.sum()
            count, minimum = len(set(groups[valid])), config.min_valid_onset_groups
        coverage = float(weights[valid].sum() / denominator) if denominator else 0.
        ref_coverage = float(weights[refs >= config.min_reference_per_cell].sum() / denominator) if denominator else 0.
        reasons = []
        if not denominator:
            reasons.append('no_population')
        if ref_coverage < config.min_coverage:
            reasons.append('reference_insufficient')
        if coverage < config.min_coverage:
            reasons.append('coverage_insufficient')
        if count < minimum:
            reasons.append('observations_insufficient')
        thirds = []
        for a, b in zip(np.linspace(left, right, 4)[:-1], np.linspace(left, right, 4)[1:]):
            section = (positions >= a) & (positions < b)
            sw = (np.maximum(0., np.minimum(r.beats.q[1:], b) - np.maximum(r.beats.q[:-1], a))
                  if k == 0 else weights * section)
            den = b - a if k == 0 else sw.sum()
            thirds.append(float(sw[valid].sum() / den) if den else None)
        active_refs = refs[weights > 0]
        diagnostics.append({'coverage': coverage, 'reference_coverage': ref_coverage, 'valid_count': count,
                            'population_weight': float(denominator), 'thirds_coverage': thirds,
                            'min_reference_count': int(active_refs.min()) if len(active_refs) else 0,
                            'max_reference_count': int(active_refs.max()) if len(active_refs) else 0,
                            'reasons': reasons})
        if not reasons:
            low, median, high = weighted_quantile(values[valid], weights[valid], [.1, .5, .9])
            ids = [0, 1] if k == 0 else [2 * k + 1, 2 * k + 2]
            raw[ids], known[ids] = [median, high - low], True
    # Original adjacent beat pairs only. No connection across missing beats.
    starts = r.beats.q[:-1]
    domain = (starts[:-1] >= left) & (starts[1:] < right)
    edges = domain & bm[:-1] & bm[1:]
    gaps = np.diff(starts)
    edge_total, edge_valid = float(gaps[domain].sum()), float(gaps[edges].sum())
    edge_coverage = edge_valid / edge_total if edge_total else 0.
    if known[0] and edges.sum() >= config.min_roughness_edges and edge_coverage >= config.min_coverage and edge_valid:
        raw[2], known[2] = np.abs(np.diff(bd))[edges].sum() / edge_valid, True
    roughness_reasons = []
    if not known[0]:
        roughness_reasons.append('tempo_level_missing')
    if edges.sum() < config.min_roughness_edges:
        roughness_reasons.append('observations_insufficient')
    if edge_coverage < config.min_coverage:
        roughness_reasons.append('coverage_insufficient')
    reasons = sorted({why for d in diagnostics for why in d['reasons']} | set(roughness_reasons))
    clipped = [(max(left, a), min(right, b)) for a, b in ae_intervals if min(right, b) > max(left, a)]
    ae_weight = _interval_length(clipped)
    result = {'performance_id': rec.performance_id, 'recording_id': rec.recording_id, 'record_key': r.record_key,
              'work_key': r.work_key, 'split': r.split, 'cohort_id': reference['cohort_id'], 'reference_id': reference['id'],
              'R_fingerprint': r.fingerprint, 'g_raw': raw.tolist(), 'g_mask': known.tolist(),
              'core_valid': bool(known[:9].all()), 'all_features_valid': bool(known.all()),
              'channels': diagnostics, 'roughness': {'valid_edges': int(edges.sum()), 'coverage': edge_coverage,
                                                   'valid_quarters': edge_valid, 'reasons': roughness_reasons},
              'scope': {'q_start': left, 'q_end': right, 'quarter_length': right - left,
                        'measure_visits': len(xml.visits), 'tail_measures': len(xml.visits) % 8,
                        'ae_full_window_coverage': ae_weight / (right - left),
                        'ae_uncovered_quarters': right - left - ae_weight,
                        'beat_count': len(bw), 'eligible_onset_groups': len(sizes)},
              'reason_codes': reasons}
    result['profile_id'] = stable_id(result)
    return result


def fit_performance_scaler(profiles, config=PerformanceStyleConfig(), *, work_balanced=False):
    if any(p['split'] != 'train' for p in profiles):
        raise ValueError('Scaler fit accepts train profiles only')
    unique, seen = [], set()
    for p in sorted(profiles, key=lambda p: p['performance_id']):
        if p['recording_id'] not in seen:
            unique.append(p)
            seen.add(p['recording_id'])
    center, scales, active, counts, status = [], [], [], [], []
    for j in range(23):
        ps = [p for p in unique if p['g_mask'][j]]
        v = np.array([p['g_raw'][j] for p in ps])
        work_counts = Counter(p['work_key'] for p in ps)
        w = np.array([1. / work_counts[p['work_key']] for p in ps]) if work_balanced else np.ones(len(ps))
        qs = (weighted_quantile(v, w, [.25, .5, .75]) if work_balanced else np.quantile(v, [.25, .5, .75])) if len(v) else np.zeros(3)
        c, scale, state = float(qs[1]), float(qs[2] - qs[0]), 'iqr'
        if scale <= config.epsilon:
            scale = float(np.sqrt(np.average((v - np.average(v, weights=w)) ** 2, weights=w))) if len(v) else 0.
            state = 'std_fallback'
        enabled = len(v) >= config.min_scaler_samples and scale > config.epsilon
        if not enabled:
            state = 'insufficient_samples' if len(v) < config.min_scaler_samples else 'constant'
        center.append(c)
        scales.append(scale if enabled else 1.)
        active.append(enabled)
        counts.append(len(v))
        status.append(state)
    result = {'schema': performance_style_schema()['version'], 'config': asdict(config), 'fit_split': 'train',
              'weighting': 'equal work, equal recordings within work' if work_balanced else 'equal unique recordings',
              'quantile': 'weighted-midpoint-linear' if work_balanced else 'numpy-linear',
              'center': center, 'scale': scales, 'trained_feature_mask': active, 'valid_counts': counts, 'status': status,
              'training_profile_ids': [p['profile_id'] for p in unique]}
    result['id'] = stable_id(result)
    return result


def scale_performance_profile(profile, scaler):
    mask = np.asarray(profile['g_mask']) & np.asarray(scaler['trained_feature_mask'])
    values = np.zeros(23)
    values[mask] = ((np.asarray(profile['g_raw']) - scaler['center']) / scaler['scale'])[mask]
    return {**profile, 'g_scaled': values.tolist(), 'g_scaled_mask': mask.tolist(), 'scaler_id': scaler['id']}
