"""Compare enhanced representations on the frozen metadata pilot triplets.

Reuse #14 verified descriptors and re-encoded #13 checkpoints. No reselection
of candidates, no new encoder training, no listening ground-truth claim.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_asap_metadata import ROOT, read_csv, write_csv, unique_rows, sha256, save_figure
from compare_metadata_summary_embedding import distance_matrices, preference
from validate_embedding import CHANNELS
from validate_tempo import plt, setup_style

SINGLE_STATS = ('large_ratio', 'roughness', 'rise_ratio', 'reversal_ratio',
                'positive_run', 'negative_run', 'value_lag1', 'value_lag8', 'delta_lag1', 'delta_lag8')
LAGS = (-4, -2, -1, 0, 1, 2, 4)
STATES = ('fall', 'flat', 'rise')
HAND_REF = 'feat/#14-temporal-feature-analysis'


def git_bytes(commit, name):
    return subprocess.check_output(['git', 'show', f'{commit}:classicfy-ai/{name}'], cwd=ROOT)


def git_csv(commit, name, verification):
    raw = git_bytes(commit, f'analysis/temporal_coupling/{name}')
    if hashlib.sha256(raw).hexdigest() != verification['csv_sha256'][name]:
        raise ValueError(f'#14 verified CSV changed: {name}')
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))


def fit_scales(values, train):
    """Train-only per-coordinate IQR/1.349, SD fallback; no missing imputation."""
    scales = np.full(values.shape[1], np.nan)
    for j in range(values.shape[1]):
        x = values[train, j]
        x = x[np.isfinite(x)]
        if len(x) < 10:
            continue
        scale = np.ptp(np.percentile(x, (25, 75))) / 1.3489795003921634
        if scale <= 1e-12:
            scale = x.std()
        if scale > 1e-12:
            scales[j] = scale
    return scales


def masked_triplet_distances(values, scales, groups, weights, a, b, c, *, required_group_fraction=1.):
    """Use identical A/B/C-supported coordinates and groups for both distances.

    At least 80% of active dimensions in each used group; specified minimum
    fraction of groups. Undefined distances remain NaN, never replaced by zero.
    Zeroing below is only a masked numerator followed by support normalization.
    """
    x = values / scales
    common = np.isfinite(x[a]) & np.isfinite(x[b]) & np.isfinite(x[c])
    db, dc = (x[a] - x[b]) ** 2, (x[a] - x[c]) ** 2
    block_b, block_c, supported, counts = [], [], [], []
    for group in groups:
        count = common[:, group].sum(axis=1)
        active = np.isfinite(scales[group]).sum()
        valid = (count >= np.ceil(.8 * active)) & (active > 0)
        den = np.maximum(count, 1)
        block_b.append(np.where(common[:, group], db[:, group], 0).sum(axis=1) / den)
        block_c.append(np.where(common[:, group], dc[:, group], 0).sum(axis=1) / den)
        supported.append(valid)
        counts.append(count)
    supported = np.array(supported).T
    weight = supported * np.array(weights)
    denominator = weight.sum(axis=1)
    valid = (supported.sum(axis=1) >= np.ceil(required_group_fraction * len(groups))) & (denominator > 0)
    distance_b = np.sqrt(np.sum(weight * np.array(block_b).T, axis=1) / np.maximum(denominator, 1e-30))
    distance_c = np.sqrt(np.sum(weight * np.array(block_c).T, axis=1) / np.maximum(denominator, 1e-30))
    distance_b[~valid] = np.nan
    distance_c[~valid] = np.nan
    support = np.sum(np.array(counts).T * supported, axis=1)
    return distance_b, distance_c, support, supported.sum(axis=1)


def learned_distances(vectors, metric):
    if not np.isfinite(vectors).all() or vectors.ndim != 2:
        raise ValueError('Invalid learned vectors')
    if metric == 'cosine':
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError('Zero learned vector')
        unit = vectors / norms[:, None]
        return np.clip(1 - unit @ unit.T, 0, 2)
    if metric == 'euclidean':
        return np.sqrt(np.mean((vectors[:, None] - vectors[None, :]) ** 2, axis=2))
    raise ValueError('Unknown metric')


def summarize_records(records):
    """Denominators are explicit; no credit for unsupported comparisons."""
    summaries, works = [], []
    for subset in ('all', 'representative_b_closer', 'representative_c_closer'):
        available = [r for r in records if subset == 'all' or
                     r['representative_preference'] == (1 if subset == 'representative_b_closer' else 0)]
        valid = [r for r in available if np.isfinite(r['preference'])]
        by_work = defaultdict(list)
        for r in valid:
            by_work[r['work_id']].append(r)
        work_scores = [np.mean([r['preference'] for r in rows]) for rows in by_work.values()]
        summaries.append(dict(subset=subset, eligible=len(available), supported=len(valid),
            unsupported=len(available) - len(valid), works=len(by_work),
            b_closer=sum(r['preference'] > .5 for r in valid),
            c_closer=sum(r['preference'] < .5 for r in valid),
            ties=sum(r['preference'] == .5 for r in valid),
            query_preference=float(np.mean([r['preference'] for r in valid])) if valid else np.nan,
            work_preference=float(np.mean(work_scores)) if work_scores else np.nan))
        if subset == 'all':
            for work, group in sorted(by_work.items()):
                works.append(dict(work_id=work, work_type=group[0]['work_type'], supported=len(group),
                                  query_preference=float(np.mean([r['preference'] for r in group]))))
    return summaries, works


def family_balanced(records):
    groups = defaultdict(list)
    for r in records:
        groups[r['key']].append(r)
    result = []
    for key, rows in groups.items():
        # Require ALL frozen family pairs, so exclusions do not silently change
        # family composition or favor one type. No unsupported renormalization.
        score = float(np.mean([r['preference'] for r in rows]))
        result.append({k: rows[0][k] for k in ('key', 'work_id', 'work_type', 'representative_preference')} |
            dict(preference=score, pairs=len(rows), supported_pairs=sum(np.isfinite(r['preference']) for r in rows)))
    return result


def model_label(model):
    return dict(representative='대표값', representative_range='대표값+범위', range_only='범위',
        single_temporal='개별 시간 통계', coupling='feature 간 지연 관계', joint_direction='공동 변화 방향',
        summary_single='요약+개별 시간', summary_single_coupling='요약+개별 시간+관계').get(model, model)


def plot_results(summary, out):
    setup_style()
    chosen = ('representative_range', 'single_temporal', 'coupling', 'joint_direction', 'summary_single_coupling',
              'cnn64_best_extended', 'bilstm64_best_extended', 'transformer64_best_extended',
              'bilstm128_best_extended', 'bilstm256_best_extended')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, subset, title in zip(axes, ('all', 'representative_b_closer'),
        ('고정한 전체 135개 후보 쌍', '대표값에서 B가 더 가까웠던 57개')):
        for i, model in enumerate(chosen):
            rows = [r for r in summary if r['model'] == model and r['subset'] == subset
                    and r['design'] == 'global' and r['metric'] in ('descriptor', 'cosine')]
            values = np.array([r['query_preference'] * 100 for r in rows])
            supported = [r['supported'] for r in rows]
            if not len(values) or not np.isfinite(values).any():
                continue
            y = float(np.nanmean(values))
            ax.barh(i, y, color='#2878c8' if len(rows) == 1 else '#dc6436', height=.62)
            ax.plot([np.nanmin(values), np.nanmax(values)], [i, i], color='#492b1c', lw=2)
            count = f'n={min(supported)}' if min(supported) == max(supported) else f'n={min(supported)}–{max(supported)}'
            ax.text(float(np.nanmax(values)) + 2, i, f'{y:.1f}% ({count})', va='center', fontsize=10)
        ax.set_yticks(np.arange(len(chosen)), [model_label(x).replace('_best_extended', ' 확대학습') for x in chosen])
        ax.invert_yaxis()
        ax.set_xlim(0, 118)
        ax.set_xlabel('다른 유형 B가 더 가까운 비율 (%)')
        ax.set_title(title, loc='left', fontsize=16)
    fig.suptitle('시간 정보를 추가한 표현에서도 다른 유형의 연주가 가까운가?', fontsize=20, fontweight='bold')
    fig.text(.5, .025, '쇼팽 발라드·에튀드 / 같은 A·B·C 유지 / 학습 모델은 cosine, 3 seed 평균·최소~최대 / 지지 부족은 분모에서 제외', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .94))
    save_figure(fig, out, 'enhanced_overview')


def stability_results(records):
    """Seed consensus and metric disagreement on the same frozen queries."""
    by_model = defaultdict(list)
    for r in records:
        if r['seed'] != 0:
            by_model[r['model']].append(r)
    result = []
    for model, rows in sorted(by_model.items()):
        for subset in ('all', 'representative_b_closer'):
            local = [r for r in rows if subset == 'all' or r['representative_preference'] == 1]
            keys = sorted({r['key'] for r in local})
            lookup = {(r['key'], r['seed'], r['metric']): r['preference'] for r in local}
            seeds = sorted({r['seed'] for r in local})
            disagreements = sum(lookup[k, s, 'cosine'] != lookup[k, s, 'euclidean'] for k in keys for s in seeds)
            for metric in ('cosine', 'euclidean'):
                outcomes = np.array([[lookup[k, s, metric] for s in seeds] for k in keys])
                result.append(dict(model=model, subset=subset, metric=metric, queries=len(keys), seeds=len(seeds),
                    all_seeds_b_closer=int(np.all(outcomes == 1, axis=1).sum()),
                    all_seeds_c_closer=int(np.all(outcomes == 0, axis=1).sum()),
                    all_seeds_same=int(np.all(outcomes == outcomes[:, :1], axis=1).sum()),
                    cosine_euclidean_disagreements=disagreements, metric_comparisons=len(keys) * len(seeds)))
    return result


def plot_fixed_examples(examples, records, profiles, out):
    setup_style()
    chosen = ('representative_range', 'single_temporal', 'coupling', 'joint_direction',
              'cnn64_best_extended', 'bilstm64_best_extended', 'transformer64_best_extended',
              'bilstm128_best_extended', 'bilstm256_best_extended')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (case, example) in zip(axes, examples.items()):
        key = example['key']
        for i, model in enumerate(chosen):
            local = [r for r in records if r['key'] == key and r['model'] == model
                     and r['metric'] in ('descriptor', 'cosine')]
            scores = [(r['distance_c'] - r['distance_b']) / (r['distance_c'] + r['distance_b'])
                      for r in local if np.isfinite(r['distance_b']) and r['distance_c'] + r['distance_b'] > 0]
            if scores:
                value = float(np.mean(scores))
                ax.barh(i, value, color='#2878c8' if value > 0 else '#dc6436', height=.6)
                ax.plot([min(scores), max(scores)], [i, i], color='#442d1d', lw=2)
        ax.axvline(0, color='#555', lw=1)
        ax.set_yticks(np.arange(len(chosen)), [model_label(x).replace('_best_extended', ' 확대학습') for x in chosen])
        ax.invert_yaxis()
        ax.set_xlim(-.6, .6)
        ax.set_xlabel('← 같은 유형 C 우세     (dC−dB)/(dC+dB)     다른 유형 B 우세 →')
        ax.set_title('기존 14D에서 B가 가까웠던 사례' if case == 'cross_type_closer' else '기존 14D에서 C가 가까웠던 사례', loc='left', fontsize=15)
        lookup = {r['key']: r for r in profiles}
        for j, field in enumerate(('key', 'key_b', 'key_c')):
            row = lookup[example[field]]
            ax.text(0, -.20 - j * .04, f'{"ABC"[j]}: {row["title"]} / {Path(row["key"]).stem}',
                    transform=ax.transAxes, fontsize=9)
    fig.suptitle('이전에 고른 두 사례를 보강 표현으로 다시 비교', fontsize=20, fontweight='bold')
    fig.text(.5, .018, '사례는 기존 보고서 그대로 고정 / 학습 모델은 3 seed 평균·범위 / 막대는 모델 안의 상대 우세이며 모델 간 원거리 비교가 아님', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .94))
    save_figure(fig, out, 'fixed_examples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_summary_embedding')
    parser.add_argument('--audit', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_style_validation')
    parser.add_argument('--learned', type=Path, default=ROOT.parent / 'datasets/metadata_temporal_exports_run01')
    parser.add_argument('--reference', type=Path, default=ROOT.parent / 'datasets/temporal_embedding_cluster_run01/dataset.json')
    parser.add_argument('--hand-ref', default=HAND_REF)
    parser.add_argument('--out', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_temporal_embedding')
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    verification = json.loads((args.baseline / 'verification.json').read_text())
    for name in ('pilot_profiles.csv', 'query_comparisons.csv', 'family_comparisons.csv'):
        if sha256(args.baseline / name) != verification['csv_sha256'][name]:
            raise ValueError(f'Frozen baseline changed: {name}')
    profiles = read_csv(args.baseline / 'pilot_profiles.csv')
    queries = read_csv(args.baseline / 'query_comparisons.csv')
    family_rows = read_csv(args.baseline / 'family_comparisons.csv')
    all_rows = read_csv(args.audit / 'performances.csv')
    if sha256(args.audit / 'performances.csv') != json.loads((args.audit / 'verification.json').read_text())['csv_sha256']['performances.csv']:
        raise ValueError('Audited identities changed')
    if any(int(r['selection_tie_pairs']) != 1 for r in (*queries, *family_rows)):
        raise ValueError('This frozen pilot must have unique selections; rerun explicit tie-aware evaluation otherwise')
    key_index = {r['key']: i for i, r in enumerate(all_rows)}
    if len(key_index) != len(all_rows):
        raise ValueError('Duplicate performance keys')
    train = np.array([r['split'] == 'train' for r in all_rows])
    for row in all_rows:
        if reference['split'][row['piece']] != row['split']:
            raise ValueError('Audit split differs from learned model split')
    vectors = np.array([[float(r[f'coordinate_{j}']) for j in range(14)] for r in profiles])
    baseline_distances = distance_matrices(vectors)
    pilot_index = {r['key']: i for i, r in enumerate(profiles)}
    hand_commit = subprocess.check_output(['git', 'rev-parse', args.hand_ref], cwd=ROOT, text=True).strip()
    hand_protocol = json.loads(git_bytes(hand_commit, 'analysis/temporal_coupling/protocol.json'))
    hand_verification = json.loads(git_bytes(hand_commit, 'analysis/temporal_coupling/verification.json'))
    if hand_protocol['source_provenance'] != reference['source_provenance'] or hand_protocol['reference_manifest_sha256'] != sha256(args.reference):
        raise ValueError('#14 source/reference differs from current pilot')
    single_rows = git_csv(hand_commit, 'single_temporal_metrics.csv', hand_verification)
    pair_rows = git_csv(hand_commit, 'pair_metrics.csv', hand_verification)
    hand_profiles = unique_rows(git_csv(hand_commit, 'profiles.csv', hand_verification), 'key')
    fields = [f'{c}_{s}' for c in CHANNELS for s in ('median_abs' if c == 'rubato' else 'mean', 'p95_p5')]
    for r, v in zip(profiles, vectors):
        if not np.allclose([float(hand_profiles[r['key']][f]) for f in fields], v, atol=1e-12, rtol=0):
            raise ValueError('#14 representative/range profile differs')
    single = np.full((len(all_rows), len(CHANNELS) * len(SINGLE_STATS)), np.nan)
    seen = set()
    for r in single_rows:
        identity = (r['key'], r['channel'])
        if identity in seen:
            raise ValueError('Duplicate single temporal row')
        seen.add(identity)
        idx = key_index[r['key']]
        if any(r[f] != all_rows[idx][f] for f in ('piece', 'grid', 'split')):
            raise ValueError('Single temporal identity mismatch')
        channel = CHANNELS.index(r['channel'])
        single[idx, channel * 10:(channel + 1) * 10] = [float(r[s]) for s in SINGLE_STATS]
    if len(seen) != len(all_rows) * 7:
        raise ValueError('Incomplete single temporal cohort')
    independent = [r for r in pair_rows if r['related_definition'] == 'False']
    pairs = sorted({(r['channel_a'], r['channel_b']) for r in independent})
    coupling = np.full((len(all_rows), len(pairs) * 7), np.nan)
    directions = np.full((len(all_rows), len(pairs) * 9), np.nan)
    seen = set()
    for r in independent:
        idx, pair = key_index[r['key']], pairs.index((r['channel_a'], r['channel_b']))
        identity = (r['key'], pair)
        if identity in seen or any(r[f] != all_rows[idx][f] for f in ('piece', 'grid', 'split')):
            raise ValueError('Pair temporal identity mismatch')
        seen.add(identity)
        coupling[idx, pair * 7:(pair + 1) * 7] = [float(r[f'lag{lag}_correlation']) if r[f'lag{lag}_status'] == 'ok' else np.nan for lag in LAGS]
        directions[idx, pair * 9:(pair + 1) * 9] = [float(r[f'{a}_{b}_probability']) if r['status'] == 'ok' else np.nan for a in STATES for b in STATES]
    if len(seen) != len(all_rows) * len(pairs) or len(pairs) != 17:
        raise ValueError('Incomplete independent pair cohort')
    descriptors = dict(single_temporal=(single, [list(range(c * 10, (c + 1) * 10)) for c in range(7)],
        [1 / 5] * 4 + [1 / 15] * 3, 1.),
        coupling=(coupling, [list(range(p * 7, (p + 1) * 7)) for p in range(17)], [1 / 17] * 17, .8),
        joint_direction=(directions, [list(range(p * 9, (p + 1) * 9)) for p in range(17)], [1 / 17] * 17, .8))
    distances, model_meta, scale_rows = {}, {}, []
    combined_rows = queries + family_rows
    a = np.array([key_index[r['key']] for r in combined_rows])
    b = np.array([key_index[r['key_b']] for r in combined_rows])
    c = np.array([key_index[r['key_c']] for r in combined_rows])
    pa, pb, pc = ([pilot_index[r[field]] for r in combined_rows] for field in ('key', 'key_b', 'key_c'))
    for name, matrix in baseline_distances.items():
        distances[name] = (matrix[pa, pb], matrix[pa, pc], np.full(len(a), 14), np.ones(len(a)))
        model_meta[name] = dict(metric='descriptor', seed=0, dimensions=7 if name != 'representative_range' else 14)
    calibrations = {}
    train_indices = np.flatnonzero(train)
    left, right = np.triu_indices(len(train_indices), 1)
    tl, tr = train_indices[left], train_indices[right]
    cross_work = np.array([all_rows[x]['work_id'] != all_rows[y]['work_id'] for x, y in zip(tl, tr)])
    tl, tr = tl[cross_work], tr[cross_work]
    for name, (values, groups, weights, fraction) in descriptors.items():
        scales = fit_scales(values, train)
        distances[name] = masked_triplet_distances(values, scales, groups, weights, a, b, c, required_group_fraction=fraction)
        calibration = masked_triplet_distances(values, scales, groups, weights, tl, tr, tr, required_group_fraction=fraction)[0]
        supported = calibration[np.isfinite(calibration)]
        median = float(np.median(supported))
        if not np.isfinite(median) or median <= 0:
            raise ValueError('Invalid train distance calibration')
        calibrations[name] = dict(median=median, supported_pairs=len(supported), eligible_pairs=len(tl))
        model_meta[name] = dict(metric='descriptor', seed=0, dimensions=values.shape[1])
        for j, scale in enumerate(scales):
            scale_rows.append(dict(model=name, coordinate=j, scale=float(scale), active=bool(np.isfinite(scale))))
    # Summary calibration uses all 395 train records, not only selected near pairs.
    feature_rows = unique_rows(read_csv(ROOT / 'classicfy-ai/analysis/_feature_search_roles/embeddings.csv'), 'key')
    full_vectors = np.array([[float(feature_rows[r['key']][f'coordinate_{j}']) for j in range(14)] for r in all_rows])
    full = distance_matrices(full_vectors)['representative_range']
    calibrations['representative_range'] = dict(median=float(np.median(full[tl, tr])), supported_pairs=len(tl), eligible_pairs=len(tl))
    for name, components in (('summary_single', ('representative_range', 'single_temporal')),
                             ('summary_single_coupling', ('representative_range', 'single_temporal', 'coupling'))):
        db = np.sqrt(np.mean([(distances[k][0] / calibrations[k]['median']) ** 2 for k in components], axis=0))
        dc = np.sqrt(np.mean([(distances[k][1] / calibrations[k]['median']) ** 2 for k in components], axis=0))
        distances[name] = (db, dc, sum(distances[k][2] for k in components), np.full(len(a), len(components)))
        model_meta[name] = dict(metric='descriptor', seed=0, dimensions=sum(model_meta[k]['dimensions'] for k in components))
    manifest = json.loads((args.learned / 'manifest.json').read_text())
    if manifest['source_provenance'] != reference['source_provenance'] or manifest['reference_manifest_sha256'] != sha256(args.reference):
        raise ValueError('Learned re-encoding provenance differs')
    manifest_identities = unique_rows(manifest['identities'], 'key')
    if set(manifest_identities) != set(key_index):
        raise ValueError('Learned inference cohort differs')
    for r in all_rows:
        if any(r[f] != manifest_identities[r['key']][f] for f in ('piece', 'grid')):
            raise ValueError('Learned inference identity differs')
    for entry in manifest['exports']:
        path = args.learned / entry['file']
        if sha256(path) != entry['export_sha256']:
            raise ValueError('Re-encoded export changed')
        with np.load(path, allow_pickle=False) as z:
            keys = list(z['keys'])
            if len(keys) != len(set(keys)) or set(keys) != set(key_index):
                raise ValueError('Learned vector keys differ')
            lookup = {key: i for i, key in enumerate(keys)}
            v = z['vectors'][[lookup[r['key']] for r in profiles]].astype(np.float64)
        for metric in ('cosine', 'euclidean'):
            matrix = learned_distances(v, metric)
            label = f"{entry['arm']}_{entry['stage']}"
            identifier = f"{label}_{entry['seed']}_{metric}"
            distances[identifier] = (matrix[pa, pb], matrix[pa, pc], np.full(len(a), v.shape[1]), np.ones(len(a)))
            model_meta[identifier] = dict(model=label, metric=metric, seed=entry['seed'], dimensions=v.shape[1])
    comparisons, summary, work_results, balanced_results = [], [], [], []
    global_representative = {r['key']: float(r['representative_preference']) for r in queries}
    for identifier, (db, dc, support, groups) in distances.items():
        meta = dict(model=identifier, **model_meta[identifier]) if 'model' not in model_meta[identifier] else model_meta[identifier]
        records = []
        for i, r in enumerate(combined_rows):
            valid = np.isfinite(db[i]) and np.isfinite(dc[i])
            records.append({k: r[k] for k in ('key', 'key_b', 'key_c', 'work_id', 'work_type')} | meta |
                dict(design='global' if i < len(queries) else 'family_pair',
                     representative_preference=global_representative[r['key']],
                     distance_b=float(db[i]), distance_c=float(dc[i]),
                     preference=preference(db[i], dc[i]) if valid else np.nan,
                     status='ok' if valid else 'insufficient_common_support',
                     common_coordinates=int(support[i]), common_groups=int(groups[i])))
        comparisons.extend(records)
        for design, group in (('global', records[:len(queries)]),
                              ('family_balanced', family_balanced(records[len(queries):]))):
            summaries, works = summarize_records(group)
            summary.extend([meta | dict(design=design) | s for s in summaries])
            work_results.extend([meta | dict(design=design) | w for w in works])
            if design == 'family_balanced':
                balanced_results.extend([meta | r for r in group])
    global_comparisons = [r for r in comparisons if r['design'] == 'global']
    # Paired sensitivity comparison on one common support set; do not compare
    # the 119-record descriptor denominator against an unfiltered 135 baseline.
    hand_global = [r for r in global_comparisons if r['model'] in descriptors]
    common_keys = {r['key'] for r in queries if all(
        np.isfinite(x['preference']) for x in hand_global if x['key'] == r['key'])}
    for identifier in distances:
        meta = dict(model=identifier, **model_meta[identifier]) if 'model' not in model_meta[identifier] else model_meta[identifier]
        local = [r for r in global_comparisons if all(r[k] == meta[k] for k in ('model', 'seed', 'metric')) and r['key'] in common_keys]
        summaries, _ = summarize_records(local)
        summary.extend([meta | dict(design='common_hand_support') | s for s in summaries])
    args.out.mkdir(parents=True, exist_ok=True)
    examples = json.loads((args.baseline / 'selected.json').read_text())
    example_records = [dict(case=case, **r) for case, example in examples.items() for r in global_comparisons if r['key'] == example['key']]
    exports = {'comparisons.csv': global_comparisons, 'summary.csv': summary, 'work_results.csv': work_results,
               'balanced_queries.csv': balanced_results, 'descriptor_scales.csv': scale_rows,
               'stability.csv': stability_results(global_comparisons), 'example_comparisons.csv': example_records}
    for name, rows in exports.items():
        write_csv(args.out / name, rows)
    family_export = args.learned / 'metadata_family_comparisons.csv'
    write_csv(family_export, [r for r in comparisons if r['design'] == 'family_pair'])
    plot_results(summary, args.out)
    plot_fixed_examples(examples, global_comparisons, profiles, args.out)
    protocol = dict(queries=len(queries), family_pairs=len(family_rows), cohort=len(all_rows),
        train_scale_performances=int(train.sum()), representative_b_closer=sum(float(r['representative_preference']) == 1 for r in queries),
        common_hand_support_queries=len(common_keys),
        hand_source_commit=hand_commit, learned_source_commit=manifest['source_commit'],
        candidate_selection='Exactly the previous representative-selected A/B/C; no reselection in any enhanced representation',
        single_stats=SINGLE_STATS, pair_lags=LAGS, independent_pairs=pairs,
        scaling='Train-only IQR/1.349, SD fallback; >=10 finite samples; constant coordinates inactive; no imputation',
        support='Same A/B/C intersection for both distances; >=80% active dimensions per group; single: all channels; relations: >=80% pairs',
        weights='Single: equal five music features with three pedal channels sharing one fifth; relations: equal 17 non-derived pairs',
        hybrids='New descriptor baselines: equal component mean squared distances, each divided by median supported cross-work train distance',
        train_distance_calibrations=calibrations,
        learned='Original window-latent mean+SD pooling; cosine primary as existing retrieval; RMS Euclidean sensitivity without coordinate fitting',
        dimensions='Learned 64/128/256 per window => 128/256/512 per performance',
        aggregation='Unsupported global rows excluded with counts; family-balanced query requires all frozen family pairs supported; common_hand_support uses identical supported queries across descriptors and learned vectors',
        seeds='Every existing 20261006/07/08 run; no seed selected using results; ranges are not confidence intervals',
        source_provenance=reference['source_provenance'],
        limits=['Train-only pilot: eight Chopin works, ballade/etude; not held-out generalization.',
                'B is other work type and C is same type; either can be closer in style. No ground-truth listening labels.',
                'More B preferences do not by themselves mean a better embedding. Frozen selection uses representative coordinates.',
                'Handcrafted metrics were verified in #14; their new concatenation, weighting and hybrids are exploratory baselines.',
                'Per-work reference includes candidates; performer and musical context are not controlled.',
                'Mean+SD pooling loses global window order; more epochs/dimensions do not guarantee style retention.',
                'Euclidean/cosine operate in different learned geometries; do not compare their raw distances across models.'],
        input_sha256={name: sha256(args.baseline / name) for name in ('pilot_profiles.csv', 'query_comparisons.csv', 'family_comparisons.csv')} |
            dict(learned_manifest=sha256(args.learned / 'manifest.json'), reference=sha256(args.reference)),
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), Path(__file__).with_name('extract_metadata_temporal_embeddings.py'))})
    (args.out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    # Keep checkpoint/export checksums reviewable without putting arrays in Git.
    (args.out / 'model_sources.json').write_text(json.dumps({k: v for k, v in manifest.items() if k != 'identities'}, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'verification.json').write_text(json.dumps(dict(
        frozen_global_triplets=len(queries), frozen_family_triplets=len(family_rows),
        reencoded_exports_checked=len(manifest['exports']),
        family_detail_sha256=sha256(family_export),
        maximum_archive_reencoding_error=max(e['maximum_archive_reencoding_error'] or 0 for e in manifest['exports']),
        verified_hand_csv_sha256={k: hand_verification['csv_sha256'][k] for k in ('profiles.csv', 'single_temporal_metrics.csv', 'pair_metrics.csv')},
        csv_sha256={name: sha256(args.out / name) for name in exports}), indent=2) + '\n')
    for s in summary:
        if s['design'] == 'global' and s['subset'] in ('all', 'representative_b_closer') and s['metric'] != 'euclidean':
            print(s['model'], s['seed'], s['subset'], s['b_closer'], '/', s['supported'], 'eligible', s['eligible'])


if __name__ == '__main__':
    main()
