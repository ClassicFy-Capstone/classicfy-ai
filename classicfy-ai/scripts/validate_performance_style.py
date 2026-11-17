"""Independently recompute sampled full-record statistics and the complete train scaler."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from embedding.note_performance_data import load_note_performance


def quantiles(values, weights):
    # Independent dictionary aggregation of equal-value probability masses.
    mass = defaultdict(float)
    for value, weight in zip(values, weights):
        if weight > 0:
            mass[float(value)] += float(weight)
    xs = sorted(mass)
    ws = np.array([mass[x] for x in xs])
    positions = (np.cumsum(ws) - ws / 2) / ws.sum()
    return np.interp([.1, .5, .9], positions, xs)


def validate(directory):
    directory = Path(directory)
    profiles = [json.loads(line) for line in (directory / 'profiles_scaled.jsonl').read_text().splitlines()]
    protocol = json.loads((directory / 'protocol.json').read_text())
    scaler = json.loads((directory / 'scaler_train.json').read_text())
    sources = {s['performance_id']: s for s in json.loads(Path(protocol['source_index']).read_text())['sources']}
    if len(profiles) != len(sources) or len({p['performance_id'] for p in profiles}) != len(profiles):
        raise ValueError('Profile count/identity mismatch')
    raw = np.array([p['g_raw'] for p in profiles])
    mask = np.array([p['g_mask'] for p in profiles])
    scaled = np.array([p['g_scaled'] for p in profiles])
    smask = np.array([p['g_scaled_mask'] for p in profiles])
    assert raw.shape == (len(sources), 23) and mask.dtype == bool and smask.dtype == bool
    assert np.isfinite(raw).all() and np.isfinite(scaled).all()
    assert (raw[~mask] == 0).all() and (scaled[~smask] == 0).all()
    np.testing.assert_array_equal(smask, mask & scaler['trained_feature_mask'])
    expected = (raw - scaler['center']) / scaler['scale']
    np.testing.assert_allclose(scaled[smask], expected[smask], atol=1e-12, rtol=1e-12)
    with np.load(directory / 'profiles.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['performance_id'], [p['performance_id'] for p in profiles])
        for key in ('g_raw', 'g_mask', 'g_scaled', 'g_scaled_mask'):
            np.testing.assert_array_equal(arrays[key], [p[key] for p in profiles])
    unique, seen = [], set()
    for p in profiles:
        if p['split'] == 'train' and p['recording_id'] not in seen:
            unique.append(p)
            seen.add(p['recording_id'])
    assert set(scaler['training_profile_ids']) == {p['profile_id'] for p in unique}
    for j in range(23):
        values = np.array([p['g_raw'][j] for p in unique if p['g_mask'][j]])
        center = float(np.median(values)) if len(values) else 0.
        spread = float(np.percentile(values, 75) - np.percentile(values, 25)) if len(values) else 0.
        if spread <= protocol['config']['epsilon']:
            spread = float(np.std(values)) if len(values) else 0.
        active = len(values) >= protocol['config']['min_scaler_samples'] and spread > protocol['config']['epsilon']
        assert active == scaler['trained_feature_mask'][j]
        assert len(values) == scaler['valid_counts'][j]
        np.testing.assert_allclose([center, spread if active else 1.], [scaler['center'][j], scaler['scale'][j]], atol=1e-12)
    selected = []
    for split in ('train', 'validation', 'test'):
        rows = [p for p in profiles if p['split'] == split]
        valid = [p for p in rows if p['core_valid']]
        missing = [p for p in rows if not any(p['g_mask'])]
        for p in ([valid[0], valid[-1]] if valid else rows[:1]) + missing[:1]:
            if p['performance_id'] not in {x['performance_id'] for x in selected}:
                selected.append(p)
    samples = []
    for p in selected:
        r = load_note_performance(sources[p['performance_id']]['performance_cache'])
        with np.load(directory / 'references' / f'{p["cohort_id"]}.npz', allow_pickle=False) as ref:
            low, high = p['scope']['q_start'], p['scope']['q_end']
            onset_counts = defaultdict(int)
            for row in r.metadata:
                if row['is_attack'] and not row['is_grace'] and low <= row['score_onset_q'] < high:
                    onset_counts[row['score_onset_group_id']] += 1
            weights = np.array([1. / onset_counts[m['score_onset_group_id']]
                                if m['is_attack'] and not m['is_grace'] and low <= m['score_onset_q'] < high else 0.
                                for m in r.metadata])
            bw = np.array([max(0., min(b, high) - max(a, low)) for a, b in zip(r.beats.q[:-1], r.beats.q[1:])])
            beat_map = {tuple(q): (v, m) for q, v, m in zip(ref['beat_intervals'], ref['beat_values'], ref['beat_mask'])}
            rv = [beat_map.get(tuple(q), (0., False)) for q in zip(r.beats.q[:-1], r.beats.q[1:])]
            delta = r.beats.log2_qpm - np.array([v for v, _ in rv])
            bm = r.beats.valid & np.array([m for _, m in rv], dtype=bool) & (bw > 0)
            for channel in range(11):
                ids = [0, 1] if channel == 0 else [2 * channel + 1, 2 * channel + 2]
                valid = bm if channel == 0 else r.mask[:, channel] & ref['note_mask'][:, channel] & (weights > 0)
                ws = bw if channel == 0 else weights
                vs = delta if channel == 0 else r.values[:, channel] - ref['note_values'][:, channel]
                denominator = high - low if channel == 0 else weights.sum()
                coverage = ws[valid].sum() / denominator if denominator else 0.
                np.testing.assert_allclose(coverage, p['channels'][channel]['coverage'], atol=1e-12)
                if p['g_mask'][ids[0]]:
                    a, b, c = quantiles(vs[valid], ws[valid])
                    np.testing.assert_allclose([b, c - a], np.array(p['g_raw'])[ids], atol=1e-12)
            total, variation, count = 0., 0., 0
            for b in range(len(bm) - 1):
                if low <= r.beats.q[b] and r.beats.q[b + 1] < high and bm[b] and bm[b + 1]:
                    count += 1
                    total += r.beats.q[b + 1] - r.beats.q[b]
                    variation += abs(delta[b + 1] - delta[b])
            assert count == p['roughness']['valid_edges']
            if p['g_mask'][2]:
                np.testing.assert_allclose(variation / total, p['g_raw'][2], atol=1e-12)
        samples.append({'performance_id': p['performance_id'], 'split': p['split'], 'valid_dimensions': sum(p['g_mask'])})
    return {'passed': True, 'profiles': len(profiles), 'all_scaled_values_and_npz_verified': True,
            'all_23_train_scaler_dimensions_independently_recomputed': True,
            'sample_statistics_independently_recomputed': samples,
            'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    out = args.out or args.directory / 'independent_verification.json'
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
