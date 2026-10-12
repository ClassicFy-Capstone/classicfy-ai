"""Explore whether joint feature timing adds explanations beyond marginal summaries.

Reuse verified ASAP/nASAP residuals. No encoder changes or preference labels.
Cross-work case exploration is train-only; other splits are descriptive only.
"""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from analyze_temporal_shape import ROOT, REFERENCE_COMMIT, verified_source_files, restore_channels
from analyze_temporal_duration import save_figure, write_csv
from validate_embedding import CHANNELS, collect_groups, summarize, weighted_distances
from validate_tempo import plt, setup_style
from features.models import BeatSequence
from features.temporal_shape import adjacent_changes, summarize_temporal_shape
from features.temporal_duration import lagged_correlation, summarize_temporal_duration
from features.temporal_coupling import (cross_channel_correlation, joint_directions,
                                       phase_shift_control)

PAIRS = tuple(combinations(CHANNELS, 2))
LAGS = (-4, -2, -1, 0, 1, 2, 4)
REPRESENTATIVES = ('tempo', 'rubato', 'dynamics', 'articulation', 'pedal_depth')
TRIPLES = tuple(t for t in combinations(REPRESENTATIVES, 3) if not {'tempo', 'rubato'} <= set(t))
SINGLE_STATS = ('large_ratio', 'roughness', 'rise_ratio', 'reversal_ratio',
                'positive_run', 'negative_run', 'value_lag1', 'value_lag8', 'delta_lag1', 'delta_lag8')


def derived_pair(a, b):
    return {a, b} == {'tempo', 'rubato'} or (a.startswith('pedal_') and b.startswith('pedal_'))


def stable_contrast(full_a, full_b, halves_a, halves_b, *, full_cut=.3, half_cut=.15):
    gaps = np.array([full_a - full_b, *(a - b for a, b in zip(halves_a, halves_b, strict=True))])
    return bool(np.isfinite(gaps).all() and abs(gaps[0]) >= full_cut
                and np.all(np.abs(gaps[1:]) >= half_cut)
                and np.all(np.sign(gaps[1:]) == np.sign(gaps[0])))


def fit_descriptor_scales(values):
    """Train-only IQR/SD per coordinate; undefineds stay undefined, no imputation."""
    scales = np.full(values.shape[1:], np.nan)
    for index in np.ndindex(scales.shape):
        x = values[(slice(None),) + index]
        x = x[np.isfinite(x)]
        if len(x) < 10:
            continue
        s = float(np.ptp(np.percentile(x, [25, 75])) / 1.3489795003921634)
        if s <= 1e-12:
            s = float(x.std())
        if s > 1e-12:
            scales[index] = s
    return scales


def single_temporal_distances(values, scales, left, right, *, min_fraction=.8):
    """Equal musical-block weights, same supported coordinates on both sides."""
    diffs = (values[left] - values[right]) / scales
    valid = np.isfinite(diffs)
    active = np.isfinite(scales)
    counts = valid.sum(axis=2)
    supported = counts >= np.ceil(min_fraction * active.sum(axis=1))
    supported &= active.sum(axis=1)[None, :] > 0
    sums = np.sum(np.where(valid, diffs, 0) ** 2, axis=2)
    channel_mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    blocks = np.column_stack([channel_mean[:, c] for c in range(4)] +
                             [np.mean(channel_mean[:, 4:7], axis=1)])
    distances = np.sqrt(np.mean(blocks, axis=1))
    distances[~supported.all(axis=1)] = np.nan
    return distances, counts.sum(axis=1)


def verify_correlation(left, right, lag, result, checks):
    """Independent explicit index traversal plus NumPy paired covariance."""
    n = len(left)
    pairs = []
    for t in range(max(0, n - abs(lag))):
        lo, hi = t, t + abs(lag)
        if all(left.mask[k] and right.mask[k] and np.isfinite(left.values[k])
               and np.isfinite(right.values[k]) for k in range(lo, hi + 1)):
            a, b = (lo, hi) if lag >= 0 else (hi, lo)
            pairs.append((left.values[a], right.values[b]))
    if len(pairs) != result['valid_pairs']:
        raise ValueError('Independent support mismatch')
    checks['support_checks'] += 1
    if result['status'] == 'ok':
        x, y = np.array(pairs).T
        r = float(np.corrcoef(x, y)[0, 1])
        error = abs(r - result['correlation'])
        checks['max_absolute_error'] = max(checks['max_absolute_error'], error)
        checks['pearson_checks'] += 1
        if error > 1e-12:
            raise ValueError('Independent Pearson mismatch')


def measure(items, thresholds, args):
    pair_rows, triple_rows, single_rows = [], [], []
    checks = dict(support_checks=0, pearson_checks=0, max_absolute_error=0.)
    for index, item in enumerate(items):
        identity = {k: item[k] for k in ('key', 'piece', 'grid', 'split')}
        deltas = {c: adjacent_changes(item['channels'][c]) for c in CHANNELS}
        mid = len(item['channels']['tempo']) // 2
        half_deltas = [{c: adjacent_changes(BeatSequence(item['channels'][c].values[a:b],
                       item['channels'][c].mask[a:b])) for c in CHANNELS}
                       for a, b in ((0, mid), (mid, len(item['channels']['tempo'])))]
        for c in CHANNELS:
            sequence = item['channels'][c]
            valid = sequence.mask & np.isfinite(sequence.values)
            width = float(np.ptp(np.percentile(sequence.values[valid], [5, 95])))
            s = summarize_temporal_shape(sequence, large_threshold=thresholds[c])
            d = summarize_temporal_duration(sequence)
            row = identity | dict(channel=c, large_ratio=s['large_ratio'],
                roughness=s['rms_delta'] / width if width > 1e-12 else np.nan,
                rise_ratio=s['rise_ratio'], reversal_ratio=s['reversal_ratio'])
            for state in ('positive', 'negative'):
                row[f'{state}_run'] = (float(np.log1p(d[f'level_{state}_run_mean']))
                    if d[f'level_{state}_run_count'] >= 20 else np.nan)
            for mode, data in (('value', sequence), ('delta', deltas[c])):
                for lag in (1, 8):
                    row[f'{mode}_lag{lag}'] = lagged_correlation(data, lag)['correlation']
            single_rows.append(row)
        for a, b in PAIRS:
            row = identity | dict(channel_a=a, channel_b=b, related_definition=derived_pair(a, b))
            for lag in LAGS:
                result = cross_channel_correlation(deltas[a], deltas[b], lag)
                for field in ('correlation', 'valid_pairs', 'status'):
                    row[f'lag{lag}_{field}'] = result[field]
                # Independently verify every full zero-lag and every lag on every tenth recording.
                if lag == 0 or index % 10 == 0:
                    verify_correlation(deltas[a], deltas[b], lag, result, checks)
            for hi, half in enumerate(half_deltas):
                result = cross_channel_correlation(half[a], half[b])
                row[f'half{hi}_correlation'] = result['correlation']
                row[f'half{hi}_valid_pairs'] = result['valid_pairs']
                row[f'half{hi}_status'] = result['status']
            row.update(joint_directions([deltas[a], deltas[b]]))
            seed = int.from_bytes(hashlib.sha256(f'{args.seed}/{item["key"]}/{a}/{b}'.encode()).digest()[:8], 'little')
            control = phase_shift_control(deltas[a], deltas[b], samples=args.shift_samples, seed=seed)
            row.update({f'phase_{k}': v for k, v in control.items()})
            pair_rows.append(row)
        for triple in TRIPLES:
            triple_rows.append(identity | dict(channels='|'.join(triple),
                **joint_directions([deltas[c] for c in triple])))
        if (index + 1) % 100 == 0:
            print(f'Measured {index + 1}/{len(items)} recordings', flush=True)
    return pair_rows, triple_rows, single_rows, checks


def explore_pairs(items, pair_rows, single_rows):
    selected_items = [i for i in items if i['split'] == 'train' and i['channels']['tempo'].mask.mean() >= .9]
    n = len(selected_items)
    keys = [i['key'] for i in selected_items]
    lookup = {(r['key'], r['channel_a'], r['channel_b']): r for r in pair_rows}
    singles = {(r['key'], r['channel']): r for r in single_rows}
    profiles = np.array([summarize(np.array([i['channels'][c].values[i['channels'][c].mask]
                                          for c in CHANNELS])) for i in selected_items])
    # Scale fitting uses all train recordings, not the matched subset or other splits.
    train_keys = [i['key'] for i in items if i['split'] == 'train']
    all_single = np.array([[[singles[k, c][s] for s in SINGLE_STATS] for c in CHANNELS] for k in train_keys])
    scales = fit_descriptor_scales(all_single)
    values = np.array([[[singles[k, c][s] for s in SINGLE_STATS] for c in CHANNELS] for k in keys])
    left, right = np.triu_indices(n, 1)
    lengths = np.array([len(i['channels']['tempo']) for i in selected_items])
    works = np.array([i['piece'] for i in selected_items])
    eligible = (works[left] != works[right]) & (np.maximum(lengths[left], lengths[right]) /
                                               np.minimum(lengths[left], lengths[right]) <= 2)
    left, right = left[eligible], right[eligible]
    baseline = weighted_distances(profiles, profiles, 'all')[left, right]
    bcut = float(np.percentile(baseline, 5))
    bclose = baseline <= bcut
    temporal, support = single_temporal_distances(values, scales, left, right)
    tcut = float(np.percentile(temporal[np.isfinite(temporal)], 25))
    tclose = np.isfinite(temporal) & (temporal <= tcut)
    matched = bclose & tclose
    summaries, close_rows, candidates, consistent = [], [], [], []
    for a, b in PAIRS:
        if derived_pair(a, b):
            continue
        records = [lookup[k, a, b] for k in keys]
        r = np.array([v['lag0_correlation'] for v in records])
        h = np.array([[v[f'half{i}_correlation'] for i in (0, 1)] for v in records])
        gap = r[left] - r[right]
        halfgap = h[left] - h[right]
        finite = np.isfinite(gap) & np.isfinite(halfgap).all(axis=1)
        stable = finite & (np.abs(gap) >= .3) & (np.abs(halfgap) >= .15).all(axis=1)
        stable &= (np.sign(halfgap) == np.sign(gap[:, None])).all(axis=1)
        # A phase control with too little retained data is not enough to select a case.
        meaningful = np.array([v['phase_status'] == 'ok' and v['phase_control_coverage'] >= .8
                               and v['phase_abs_excess_over_shift_p95'] >= .1 for v in records])
        signal = meaningful[left] | meaningful[right]
        for label, subset in (('all', np.ones(len(left), bool)), ('baseline_close', bclose),
                              ('baseline_and_single_temporal_close', matched)):
            denom = subset & finite
            summaries.append(dict(channel_a=a, channel_b=b, subset=label,
                selected_pairs=int(subset.sum()), supported_pairs=int(denom.sum()),
                stable_gap_count=int((denom & stable).sum()),
                stable_gap_fraction=float(stable[denom].mean()) if denom.any() else np.nan,
                phase_supported_gap_count=int((denom & stable & signal).sum()),
                median_absolute_gap=float(np.median(np.abs(gap[denom]))) if denom.any() else np.nan))
        for p in np.flatnonzero(bclose):
            i, j = int(left[p]), int(right[p])
            row = dict(key_a=keys[i], key_b=keys[j], piece_a=works[i], piece_b=works[j],
                channel_a=a, channel_b=b, baseline_distance=float(baseline[p]),
                single_temporal_distance=float(temporal[p]), single_supported_coordinates=int(support[p]),
                single_temporal_close=bool(tclose[p]), correlation_a=float(r[i]), correlation_b=float(r[j]),
                signed_gap=float(gap[p]), first_half_signed_gap=float(halfgap[p, 0]),
                second_half_signed_gap=float(halfgap[p, 1]), stable_gap=bool(stable[p]),
                phase_supported=bool(signal[p]))
            close_rows.append(row)
            if matched[p] and stable[p] and signal[p]:
                candidates.append(row)
            if matched[p] and finite[p] and abs(gap[p]) <= .05 and np.all(np.abs(halfgap[p]) <= .1):
                consistent.append(row)
    choose = lambda rows: min(rows, key=lambda r: (r['baseline_distance'], r['key_a'], r['key_b'],
                                                  r['channel_a'], r['channel_b'])) if rows else None
    selected = dict(divergent=choose(candidates), consistent=choose(consistent))
    profile_rows = [dict(key=i['key'], piece=i['piece'], split=i['split'],
                        **{f'{c}_{s}': float(profiles[j, 2 * ci + k]) for ci, c in enumerate(CHANNELS)
                           for k, s in enumerate(('median_abs' if c == 'rubato' else 'mean', 'p95_p5'))})
                    for j, i in enumerate(selected_items)]
    scale_rows = [dict(channel=c, metric=s, scale=float(scales[ci, si]))
                  for ci, c in enumerate(CHANNELS) for si, s in enumerate(SINGLE_STATS)]
    info = dict(train_performances=n, train_works=len(set(works)), eligible_pairs=len(left),
        baseline_close_cut=bcut, baseline_close_pairs=int(bclose.sum()),
        single_temporal_close_cut=tcut, single_temporal_supported_pairs=int(np.isfinite(temporal).sum()),
        both_close_pairs=int(matched.sum()), divergent_case_candidates=len(candidates),
        consistent_case_candidates=len(consistent))
    return summaries, close_rows, selected, profile_rows, scale_rows, info


def overview(pair_rows, out):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 11))
    y = np.arange(len(PAIRS))
    groups = [[r for r in pair_rows if (r['channel_a'], r['channel_b']) == pair] for pair in PAIRS]
    axes[0].boxplot([[r['lag0_correlation'] for r in rows if np.isfinite(r['lag0_correlation'])]
                    for rows in groups], positions=y, orientation='horizontal', showfliers=False)
    axes[0].axvline(0, color='#777777', linewidth=.8)
    axes[0].set_xlim(-1.05, 1.05)
    axes[0].set_yticks(y, [f'{a} / {b}' for a, b in PAIRS], fontsize=10)
    axes[0].set_xlabel('같은 beat의 변화량 Pearson r')
    axes[0].set_title('연주별 동시 변화 관계', loc='left', fontsize=17)
    ratios, counts = [], []
    for rows in groups:
        eligible = [r for r in rows if r['phase_status'] == 'ok' and r['phase_control_coverage'] >= .8]
        counts.append(len(eligible))
        ratios.append(100 * np.mean([r['phase_abs_excess_over_shift_p95'] >= .1 for r in eligible]) if eligible else 0)
    axes[1].barh(y, ratios, color=['#aaaaaa' if derived_pair(a, b) else '#2878c8' for a, b in PAIRS])
    for yy, ratio, count in zip(y, ratios, counts):
        axes[1].text(ratio + 1, yy, f'{ratio:.1f}% (n={count})', va='center', fontsize=9)
    axes[1].set_xlim(0, 125)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel('|r| ≥ phase control p95 + 0.1인 연주 비율 (%)')
    axes[1].set_title('개별 변화 분포를 유지한 시점 어긋남 대조', loc='left', fontsize=17)
    fig.suptitle('여러 특징의 동시 변화: 기술적 탐색', fontsize=21, fontweight='bold', y=.98)
    fig.text(.5, .93, '전체 55작품·570연주 | 회색: 정의가 연결된 Tempo–Rubato / 페달 내부 관계 | 청취·추천 검증 없음', ha='center', fontsize=11)
    fig.subplots_adjust(left=.20, right=.98, top=.86, bottom=.10, wspace=.15)
    save_figure(fig, out, 'coupling_overview')


def plot_case(case, selected, items, pair_lookup, single_lookup, out):
    if selected is None:
        return [], []
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    curve_rows, metric_rows = [], []
    a, b = selected['channel_a'], selected['channel_b']
    for col, side in enumerate(('a', 'b')):
        key = selected[f'key_{side}']
        item = items[key]
        x, y = [adjacent_changes(item['channels'][c]) for c in (a, b)]
        valid = x.mask & y.mask
        axes[0, col].scatter(x.values[valid], y.values[valid], s=10, alpha=.4, color=('#2878c8', '#dc6436')[col])
        axes[0, col].axhline(0, color='#aaaaaa', linewidth=.8)
        axes[0, col].axvline(0, color='#aaaaaa', linewidth=.8)
        axes[0, col].set_xlabel(f'{a} Δ')
        axes[0, col].set_ylabel(f'{b} Δ')
        axes[0, col].set_title(f'{Path(key).stem}\n{item["piece"].replace("/midi_score.mid", "")}', loc='left', fontsize=12)
        row = pair_lookup[key, a, b]
        offsets = np.arange(len(LAGS))
        axes[1, col].plot(offsets, [row[f'lag{lag}_correlation'] for lag in LAGS], marker='o', color=('#2878c8', '#dc6436')[col])
        axes[1, col].set_xticks(offsets, LAGS)
        axes[1, col].set_ylim(-1.05, 1.05)
        axes[1, col].axhline(0, color='#aaaaaa', linewidth=.8)
        axes[1, col].set_xlabel(f'lag (beat) · 양수: {a}가 앞섬')
        axes[1, col].set_ylabel('변화량의 교차 상관')
        axes[1, col].set_title(f"전곡 r={row['lag0_correlation']:.3f} / 전반={row['half0_correlation']:.3f} / 후반={row['half1_correlation']:.3f}", loc='left', fontsize=12)
        metric_rows.append(dict(case=case, side=side, **row))
        for c in CHANNELS:
            metric_rows.append(dict(case=case, side=side, key=key, channel_a=c, channel_b='single',
                                   **{s: single_lookup[key, c][s] for s in SINGLE_STATS}))
        curve_rows.extend(dict(case=case, side=side, key=key, original_index=int(t),
                               channel_a=a, channel_b=b, delta_a=float(x.values[t]),
                               delta_b=float(y.values[t])) for t in np.flatnonzero(valid))
    for dimension in ('x', 'y'):
        bounds = [getattr(ax, f'get_{dimension}lim')() for ax in axes[0]]
        lo, hi = min(v[0] for v in bounds), max(v[1] for v in bounds)
        for ax in axes[0]:
            getattr(ax, f'set_{dimension}lim')(lo, hi)
    title = '개별 요약은 가까워도 동시 변화 관계는 다른 쌍' if case == 'divergent' else '개별 요약과 동시 변화 관계가 함께 가까운 쌍'
    fig.suptitle(title, fontsize=20, fontweight='bold', y=.98)
    fig.text(.5, .92, f"14D 거리 {selected['baseline_distance']:.3f} | 개별 시간 통계 거리 {selected['single_temporal_distance']:.3f} | train 탐색 사례", ha='center', fontsize=12)
    fig.text(.5, .035, '정규화 잔차의 변화 · 서로 다른 작품의 beat 위치를 맞춘 비교가 아님 · 전반/후반은 같은 녹음의 부분 점검', ha='center', fontsize=10)
    fig.subplots_adjust(left=.09, right=.98, bottom=.12, top=.82, hspace=.50, wspace=.35)
    save_figure(fig, out, f'{case}_case')
    return metric_rows, curve_rows


def write_union(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(path, [{k: row.get(k, '') for k in fields} for row in rows])


def relation_summary(pair_rows):
    rows = []
    for a, b in PAIRS:
        selected = [r for r in pair_rows if (r['channel_a'], r['channel_b']) == (a, b)]
        finite = [r for r in selected if r['lag0_status'] == 'ok']
        values = np.array([r['lag0_correlation'] for r in finite])
        works = np.array([r['piece'] for r in finite])
        variance = np.sum((values - values.mean()) ** 2)
        between = sum(np.count_nonzero(works == w) * (values[works == w].mean() - values.mean()) ** 2
                      for w in set(works))
        control = [r for r in selected if r['phase_status'] == 'ok' and r['phase_control_coverage'] >= .8]
        rows.append(dict(channel_a=a, channel_b=b, related_definition=derived_pair(a, b),
            defined_performances=len(finite), median_correlation=float(np.median(values)),
            correlation_p5=float(np.percentile(values, 5)), correlation_p95=float(np.percentile(values, 95)),
            piece_eta2=float(between / variance) if variance > 0 else np.nan,
            phase_supported_performances=len(control),
            phase_excess_count=sum(r['phase_abs_excess_over_shift_p95'] >= .1 for r in control)))
    return rows


def case_sensitivity(selected, items, pair_lookup):
    """Post-selection diagnostics only; original arrays and case choice unchanged."""
    rows = []
    for case, selection in selected.items():
        if selection is None:
            continue
        a, b = selection['channel_a'], selection['channel_b']
        for side in ('a', 'b'):
            key = selection[f'key_{side}']
            item = items[key]
            delta = [adjacent_changes(item['channels'][c]) for c in (a, b)]
            valid = delta[0].mask & delta[1].mask
            x, y = [s.values[valid] for s in delta]
            r = float(np.corrcoef(x, y)[0, 1])
            keep = (abs(x) <= np.percentile(abs(x), 99)) & (abs(y) <= np.percentile(abs(y), 99))
            trimmed = float(np.corrcoef(x[keep], y[keep])[0, 1])
            # Average ranks for ties, implemented independently of the descriptor.
            def ranks(v):
                order = np.argsort(v, kind='stable')
                ranked = np.empty(len(v), dtype=float)
                for start, stop in contiguous_equal(v[order]):
                    ranked[order[start:stop]] = (start + stop - 1) / 2 + 1
                return ranked
            rank_r = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
            loo = np.array([np.corrcoef(np.delete(x, k), np.delete(y, k))[0, 1] for k in range(len(x))])
            peers = [v['lag0_correlation'] for (k, ca, cb), v in pair_lookup.items()
                     if (ca, cb) == (a, b) and v['piece'] == item['piece'] and v['lag0_status'] == 'ok']
            rows.append(dict(case=case, side=side, key=key, channel_a=a, channel_b=b,
                valid_pairs=len(x), full_pearson=r, trimmed_pairs=int(keep.sum()),
                trim_abs_delta_quantile=.99, trimmed_pearson=trimmed, rank_correlation=rank_r,
                max_leave_one_step_change=float(np.max(abs(loo - r))),
                peer_count=len(peers), peer_mean_correlation=float(np.mean(peers)),
                peer_centered_correlation=r - float(np.mean(peers))))
    return rows


def contiguous_equal(sorted_values):
    edges = np.r_[0, np.flatnonzero(np.diff(sorted_values) != 0) + 1, len(sorted_values)]
    return zip(edges[:-1], edges[1:], strict=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asap-root', type=Path, default=ROOT.parent / 'datasets/ASAP')
    parser.add_argument('--nasap-root', type=Path, default=ROOT.parent / 'datasets/nASAP')
    parser.add_argument('--cache', type=Path, default=ROOT.parent / 'datasets/feature_normalization_cluster_raw.npz')
    parser.add_argument('--reference', type=Path, default=ROOT.parent / 'datasets/temporal_embedding_cluster_run01/dataset.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'classicfy-ai/analysis/temporal_coupling')
    parser.add_argument('--shift-samples', type=int, default=199)
    parser.add_argument('--seed', type=int, default=20261012)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    cache_hash = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if cache_hash != reference['cache_sha256']:
        raise ValueError('Raw cache differs from the reference')
    files = verified_source_files()
    print('Verified original source and cache; collecting unchanged normalized cohorts', flush=True)
    groups, audits, scales, provenance = collect_groups(SimpleNamespace(
        asap_root=args.asap_root.resolve(), nasap_root=args.nasap_root.resolve(), cache=args.cache,
        **reference['cohort_settings']), provenance_source_files=files)
    if provenance != reference['source_provenance'] or scales != reference['scales']:
        raise ValueError('Source provenance or normalization scales differ from the reference')
    items = restore_channels(groups, audits, reference['split'])
    import csv
    threshold_path = ROOT / 'classicfy-ai/analysis/temporal_shape/thresholds.csv'
    thresholds = {r['channel']: float(r['value']) for r in csv.DictReader(threshold_path.open(encoding='utf-8-sig'))}
    from features.temporal_shape import fit_change_threshold
    for c in CHANNELS:
        fitted = fit_change_threshold(i['channels'][c] for i in items if i['split'] == 'train')
        if not np.isclose(fitted.value, thresholds[c], rtol=0, atol=1e-12):
            raise ValueError(f'Old train-only threshold changed: {c}')
    pair_rows, triple_rows, single_rows, checks = measure(items, thresholds, args)
    summary, close_rows, selected, profiles, descriptor_scales, info = explore_pairs(items, pair_rows, single_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (('pair_metrics.csv', pair_rows), ('triple_directions.csv', triple_rows),
                       ('relation_summary.csv', relation_summary(pair_rows)),
                       ('single_temporal_metrics.csv', single_rows), ('additional_information.csv', summary),
                       ('close_pairs.csv', close_rows), ('profiles.csv', profiles),
                       ('descriptor_scales.csv', descriptor_scales), ('cohort_audit.csv', audits)):
        write_csv(args.out / name, rows)
    (args.out / 'selected.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n')
    overview(pair_rows, args.out)
    pair_lookup = {(r['key'], r['channel_a'], r['channel_b']): r for r in pair_rows}
    single_lookup = {(r['key'], r['channel']): r for r in single_rows}
    case_metrics, case_values = [], []
    for case, row in selected.items():
        metrics, curves = plot_case(case, row, {i['key']: i for i in items}, pair_lookup, single_lookup, args.out)
        case_metrics.extend(metrics)
        case_values.extend(curves)
    if case_metrics:
        write_union(args.out / 'case_metrics.csv', case_metrics)
        write_csv(args.out / 'case_values.csv', case_values)
        write_csv(args.out / 'case_sensitivity.csv', case_sensitivity(selected, {i['key']: i for i in items}, pair_lookup))
    qualifying = [r for r in close_rows if r['single_temporal_close'] and r['stable_gap']]
    info['unique_stable_pairs'] = len({(r['key_a'], r['key_b']) for r in qualifying})
    info['unique_phase_supported_stable_pairs'] = len({(r['key_a'], r['key_b']) for r in qualifying if r['phase_supported']})
    protocol = dict(**info, performances=len(items), works=len({i['piece'] for i in items}),
        source_provenance=provenance, cache_sha256=cache_hash,
        reference_manifest_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        source_reference=REFERENCE_COMMIT, split=reference['split'], channels=CHANNELS,
        lags=LAGS, min_pairs=32, direction_tolerance=1e-6, std_tolerance=1e-12,
        phase=dict(samples=args.shift_samples, seed=args.seed, min_segment=64, min_shift=8,
                   minimum_control_coverage_for_selection=.8, min_abs_excess=.1),
        single_stats=SINGLE_STATS, single_scale='train-only IQR/1.349 -> SD; constant unsupported; no imputation',
        single_distance='>=80% supported active coordinates per channel; equal five musical block weights; run means log1p, >=20 runs',
        pair_selection='train coverage>=.9, different works, length ratio<=2; 14D bottom5%; single-temporal bottom25% of all supported eligible pairs',
        divergence='full signed r difference>=.3 and same sign >=.15 each half; at least one side phase supported; choose smallest 14D distance across 17 exploratory relations',
        consistency='full r difference<=.05 and each half<=.1; choose smallest 14D distance across 17 exploratory relations',
        related_definition='Tempo/Rubato and pedal internal pairs reported but excluded from cross-work case search',
        triple_channels=TRIPLES, pair_status_counts=dict(Counter(r['lag0_status'] for r in pair_rows)),
        limits=['Exploratory added numeric explanations; no listening or recommendation validation.',
                'Per-work common and scale include candidates; score, beat definitions, note density and alignment effects remain.',
                'Pairs are dependent; thresholds are exploratory, not perceptual; no significance claims or test-set case selection.',
                'Circular delta shifts preserve marginals and circular autocorrelation, not all finite-boundary ordering; periodic relations may survive.',
                'Triple TV from independent marginals can be caused solely by pair dependence; no irreducible higher-order claim.',
                'Cross-correlation does not establish causality; tempo and rubato are mathematically related.',
                'Single-temporal matching controls only selected descriptors, not entire individual trajectories.'],
        environment=dict(python=sys.version, numpy=np.__version__),
        code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (Path(__file__), ROOT / 'classicfy-ai/src/features/temporal_coupling.py')},
        threshold_file_sha256=hashlib.sha256(threshold_path.read_bytes()).hexdigest())
    (args.out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    checks.update(csv_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.glob('*.csv')})
    (args.out / 'verification.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)
    print(f'Saved {len(pair_rows)} channel-pair rows and {len(triple_rows)} triple rows to {args.out}', flush=True)


if __name__ == '__main__':
    main()
