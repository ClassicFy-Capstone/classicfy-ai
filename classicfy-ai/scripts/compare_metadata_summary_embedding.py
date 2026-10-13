"""Compare work-type affinity with the existing representative+range embedding.

Select candidates using representative coordinates only, freeze them, evaluate
14D and unused range coordinates. This is a descriptor comparison, not labels.
"""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_asap_metadata import ROOT, LABELS, read_csv, write_csv, unique_rows, sha256, save_figure
from validate_embedding import CHANNELS, BLOCKS, weighted_distances
from validate_tempo import plt, setup_style

MODES = ('representative', 'representative_range', 'range_only')
MODE_LABELS = dict(representative='대표값 7D (후보 선정에 사용)',
                   representative_range='대표값+범위 14D', range_only='범위 7D (선정에 미사용)')
TOL = 1e-12


def distance_matrices(vectors):
    if vectors.ndim != 2 or vectors.shape[1] != 14 or not np.isfinite(vectors).all():
        raise ValueError('Expected finite N x 14 profiles')
    if np.any(vectors[:, 1::2] < 0):
        raise ValueError('p95-p5 ranges must be nonnegative')
    width = sum(np.mean((vectors[:, None, ix[1::2]] - vectors[None, :, ix[1::2]]) ** 2, axis=2)
                for ix in BLOCKS.values())
    return dict(representative=weighted_distances(vectors, vectors, 'mean_only'),
                representative_range=weighted_distances(vectors, vectors, 'all'),
                range_only=np.sqrt(width / len(BLOCKS)))


def preference(distance_b, distance_c):
    """B closer=1, C closer=0, equal within absolute tolerance=.5."""
    if not np.isfinite([distance_b, distance_c]).all() or min(distance_b, distance_c) < 0:
        raise ValueError('Comparison distances must be finite and nonnegative')
    return .5 if abs(distance_b - distance_c) <= TOL else float(distance_b < distance_c)


def choose_pairs(query, bs, cs, rows, selection_distances):
    """Joint nearest B/C by minimum sum of representative distances.

    Enforce distinct A/B/C families, same composer, opposite/same type. Return
    every minimum pair so later preferences can average selection ties instead
    of benefiting from a filename tie-break. Lexical first is display-only.
    """
    q = rows[query]
    pairs = [(b, c) for b, c in product(bs, cs)
             if len({q['family_id'], rows[b]['family_id'], rows[c]['family_id']}) == 3
             and rows[b]['composer'] == rows[c]['composer'] == q['composer']
             and rows[b]['work_type'] != q['work_type']
             and rows[c]['work_type'] == q['work_type']]
    if not pairs:
        return []
    costs = np.array([selection_distances[query, b] + selection_distances[query, c] for b, c in pairs])
    if not np.isfinite(costs).all():
        raise ValueError('Nonfinite candidate selection distances')
    minimum = costs.min()
    selected = [pair for pair, cost in zip(pairs, costs) if abs(cost - minimum) <= TOL]
    return sorted(selected, key=lambda pair: (rows[pair[0]]['key'], rows[pair[1]]['key']))


def compare(query, pairs, distances):
    if not pairs:
        raise ValueError('At least one selected pair is required')
    result = {}
    for mode in MODES:
        matrix = distances[mode]
        result[f'{mode}_preference'] = float(np.mean([preference(matrix[query, b], matrix[query, c]) for b, c in pairs]))
        result[f'{mode}_tie_fraction'] = float(np.mean([abs(matrix[query, b] - matrix[query, c]) <= TOL for b, c in pairs]))
        result[f'{mode}_mean_margin'] = float(np.mean([matrix[query, c] - matrix[query, b] for b, c in pairs]))
    return result


def evaluate(rows, distances):
    queries, family_comparisons = [], []
    for a, q in enumerate(rows):
        bs = [i for i, r in enumerate(rows) if r['work_type'] != q['work_type'] and r['family_id'] != q['family_id']]
        cs = [i for i, r in enumerate(rows) if r['work_type'] == q['work_type'] and r['family_id'] != q['family_id']]
        chosen = choose_pairs(a, bs, cs, rows, distances['representative'])
        if not chosen:
            raise ValueError(f'Audited pilot lost candidate support: {q["key"]}')
        b, c = chosen[0]
        row = {field: q[field] for field in ('key', 'composer', 'work_id', 'family_id', 'work_type')}
        row.update(query_index=a, b_index=b, c_index=c, key_b=rows[b]['key'], key_c=rows[c]['key'],
            type_b=rows[b]['work_type'], type_c=rows[c]['work_type'], family_b=rows[b]['family_id'],
            family_c=rows[c]['family_id'], b_candidates=len(bs), c_candidates=len(cs),
            b_candidate_families=len({rows[i]['family_id'] for i in bs}),
            c_candidate_families=len({rows[i]['family_id'] for i in cs}),
            random_order_cross_type_reference=len(bs) / (len(bs) + len(cs)),
            selection_tie_pairs=len(chosen), **compare(a, chosen, distances))
        for mode in MODES:
            row[f'{mode}_distance_b'] = float(distances[mode][a, b])
            row[f'{mode}_distance_c'] = float(distances[mode][a, c])
        queries.append(row)
        # Equal family-pair weighting prevents more available candidate works
        # from winning solely through one extreme minimum across a larger pool.
        by_b, by_c = defaultdict(list), defaultdict(list)
        for b in bs:
            by_b[rows[b]['family_id']].append(b)
        for c in cs:
            by_c[rows[c]['family_id']].append(c)
        for fb, fc in product(sorted(by_b), sorted(by_c)):
            selected = choose_pairs(a, by_b[fb], by_c[fc], rows, distances['representative'])
            if not selected:
                continue
            b, c = selected[0]
            family_comparisons.append(dict(key=q['key'], work_id=q['work_id'], work_type=q['work_type'],
                family_id=q['family_id'], family_b=fb, family_c=fc, key_b=rows[b]['key'], key_c=rows[c]['key'],
                selection_tie_pairs=len(selected), **compare(a, selected, distances)))
    return queries, family_comparisons


def aggregate(queries, family_rows):
    balanced = []
    for q in queries:
        local = [r for r in family_rows if r['key'] == q['key']]
        if not local:
            raise ValueError('No family-balanced comparisons')
        balanced.append({field: q[field] for field in ('key', 'work_id', 'work_type', 'family_id')} |
            {f'{mode}_{field}': float(np.mean([r[f'{mode}_{field}'] for r in local]))
             for mode in MODES for field in ('preference', 'tie_fraction', 'mean_margin')})
    summaries, works = [], []
    for design, records in (('global_nearest_representative', queries), ('family_pair_balanced', balanced)):
        for mode in MODES:
            field = f'{mode}_preference'
            groups = defaultdict(list)
            for r in records:
                groups[r['work_id']].append(r)
            for work, group in sorted(groups.items()):
                works.append(dict(design=design, mode=mode, work_id=work, work_type=group[0]['work_type'],
                    queries=len(group), cross_type_preference=float(np.mean([r[field] for r in group]))))
            for label in ('all', *sorted({r['work_type'] for r in records})):
                subset = [r for r in records if label == 'all' or r['work_type'] == label]
                wsubset = [r for r in works if r['design'] == design and r['mode'] == mode
                           and (label == 'all' or r['work_type'] == label)]
                summaries.append(dict(design=design, mode=mode, query_type=label, queries=len(subset),
                    works=len(wsubset), query_weighted_preference=float(np.mean([r[field] for r in subset])),
                    work_weighted_preference=float(np.mean([r['cross_type_preference'] for r in wsubset])),
                    tie_fraction=float(np.mean([r[f'{mode}_tie_fraction'] for r in subset])),
                    mean_margin=float(np.mean([r[f'{mode}_mean_margin'] for r in subset]))))
    return summaries, works, balanced


def contributions(queries, vectors):
    result = []
    for q in queries:
        a, b, c = q['query_index'], q['b_index'], q['c_index']
        for name, ix in BLOCKS.items():
            for mode, coords in (('representative_range', ix), ('representative', ix[::2]), ('range_only', ix[1::2])):
                db = float(np.mean((vectors[a, coords] - vectors[b, coords]) ** 2))
                dc = float(np.mean((vectors[a, coords] - vectors[c, coords]) ** 2))
                result.append(dict(key=q['key'], mode=mode, feature=name, block_distance_squared_b=db,
                                   block_distance_squared_c=dc, advantage_for_b=dc - db))
    return result


def distance_balance(queries):
    """Numerical contribution of width vs representative to each frozen pair."""
    rows = []
    for q in queries:
        for side in ('b', 'c'):
            representative = q[f'representative_distance_{side}'] ** 2
            width = q[f'range_only_distance_{side}'] ** 2
            total = representative + width
            rows.append(dict(key=q['key'], side=side,
                representative_squared_component=representative / 2,
                range_squared_component=width / 2,
                full_squared_distance=q[f'representative_range_distance_{side}'] ** 2,
                range_share=width / total if total > 0 else np.nan))
    return rows


def weighted_projection(vectors):
    """Three points in weighted 14D have a distance-preserving 2D projection."""
    weights = np.zeros(14)
    for ix in BLOCKS.values():
        weights[ix] = 1 / (len(BLOCKS) * len(ix))
    points = vectors * np.sqrt(weights)
    centered = points - points.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, singular


def plot_overview(summary, out):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, design, title in zip(axes, ('global_nearest_representative', 'family_pair_balanced'),
                                ('유형별 대표값 최근접 B/C', '작품군 쌍에 동일 비중')):
        for offset, field, color, label in ((-.18, 'query_weighted_preference', '#2878c8', '연주 동일 비중'),
                                          (.18, 'work_weighted_preference', '#dc6436', '작품 동일 비중')):
            values = [100 * next(r[field] for r in summary if r['design'] == design
                      and r['mode'] == mode and r['query_type'] == 'all') for mode in MODES]
            ax.bar(np.arange(3) + offset, values, width=.32, color=color, label=label)
            for x, y in zip(np.arange(3) + offset, values):
                ax.text(x, y + 1.5, f'{y:.1f}%', ha='center', fontsize=10)
        ax.set_xticks(np.arange(3), ('대표값\n선정 좌표', '대표값+범위\n14D', '범위만\n미사용 좌표'))
        ax.set_ylim(0, 105)
        ax.set_ylabel('다른 유형 B를 더 가깝게 평가한 비율 (%)')
        ax.set_title(title, loc='left', fontsize=17)
        ax.legend(frameon=False, fontsize=10, loc='upper left')
    fig.suptitle('작품 유형보다 특징을 따라 가까이 연결되는가?', fontsize=21, fontweight='bold')
    fig.text(.5, .025, '쇼팽 train 135연주 / 8작품 · A/B/C는 다른 작품군 · 동점 0.5 · 수치적 배치 성향이며 청취/추천 정확도가 아님', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .92))
    save_figure(fig, out, 'comparison_overview')


def select_examples(queries):
    selected = {}
    for case, predicate in (('cross_type_closer', lambda r: r['representative_range_preference'] == 1),
                            ('same_type_closer', lambda r: r['representative_range_preference'] == 0)):
        candidates = [r for r in queries if predicate(r)]
        if not candidates:
            selected[case] = None
            continue
        median = float(np.median([r['representative_range_mean_margin'] for r in candidates]))
        selected[case] = min(candidates, key=lambda r: (abs(r['representative_range_mean_margin'] - median), r['key']))
    return selected


def plot_example(case, q, rows, vectors, out):
    if q is None:
        return
    setup_style()
    ids = [q['query_index'], q['b_index'], q['c_index']]
    xy, singular = weighted_projection(vectors[ids])
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    colors = ('#222222', '#148766', '#87919d')
    labels = ('A 기준', 'B 다른 유형', 'C 같은 유형')
    for j in (1, 2):
        axes[0].plot(xy[[0, j], 0], xy[[0, j], 1], color=colors[j], linewidth=2, linestyle='-' if j == 1 else '--')
    for j in range(3):
        axes[0].scatter(*xy[j], s=180, color=colors[j], edgecolor='white', zorder=3)
        axes[0].annotate(f'{labels[j]}\n{LABELS[rows[ids[j]]["work_type"]]}', xy[j], xytext=(0, 15),
                         textcoords='offset points', ha='center', fontsize=11)
    axes[0].set_aspect('equal', adjustable='datalim')
    axes[0].margins(.35)
    axes[0].set_xlabel('PC1 · 가중 14D 좌표')
    axes[0].set_ylabel('PC2 · 가중 14D 좌표')
    axes[0].set_title('세 점의 14D 거리를 보존한 배치', loc='left', fontsize=16)
    for offset, side, color in ((-.17, 'b', colors[1]), (.17, 'c', colors[2])):
        values = [q[f'{m}_distance_{side}'] for m in MODES]
        axes[1].barh(np.arange(3) + offset, values, height=.30, color=color, label=f'{side.upper()} 후보')
        for y, x in zip(np.arange(3) + offset, values):
            axes[1].text(x + .025, y, f'{x:.3f}', va='center', fontsize=11)
    axes[1].set_yticks(np.arange(3), ('대표값 7D\n후보 선정', '대표값+범위\n14D', '범위 7D\n선정에 미사용'))
    axes[1].invert_yaxis()
    maximum = max(q[f'{m}_distance_{side}'] for m in MODES for side in ('b', 'c'))
    axes[1].set_xlim(0, maximum * 1.30)
    axes[1].set_xlabel('해당 표현의 5개 feature 동일 가중 거리')
    axes[1].set_title('작은 거리가 더 가까움', loc='left', fontsize=16)
    axes[1].legend(frameon=False)
    title = '다른 유형 B가 가까운 사례' if case == 'cross_type_closer' else '같은 유형 C가 가까운 사례'
    fig.suptitle(title, fontsize=21, fontweight='bold', y=.98)
    for j, i in enumerate(ids):
        fig.text(.05, .14 - j * .028, f'{labels[j]}: {rows[i]["title"]} / {Path(rows[i]["key"]).stem}', fontsize=10, color=colors[j])
    fig.text(.05, .032, '각 결과 집단의 거리 여유 중앙값에 가까운 설명용 사례 · 대표값으로 선정 후 14D 확인 · 청취 정답 아님', fontsize=10)
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.25, wspace=.50)
    save_figure(fig, out, case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_style_validation')
    parser.add_argument('--features', type=Path, default=ROOT / 'classicfy-ai/analysis/_feature_search_roles')
    parser.add_argument('--baseline-export', type=Path, default=ROOT / 'classicfy-ai/analysis/_embedding_validation/embeddings.csv')
    parser.add_argument('--reference', type=Path, default=ROOT.parent / 'datasets/temporal_embedding_cluster_run01/dataset.json')
    parser.add_argument('--cache', type=Path, default=ROOT.parent / 'datasets/feature_normalization_cluster_raw.npz')
    parser.add_argument('--out', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_summary_embedding')
    args = parser.parse_args()
    audit = json.loads((args.audit / 'protocol.json').read_text())
    audit_verification = json.loads((args.audit / 'verification.json').read_text())
    for name in ('performances.csv', 'pilot_options.csv'):
        if sha256(args.audit / name) != audit_verification['csv_sha256'][name]:
            raise ValueError(f'Metadata audit output changed: {name}')
    feature_path = args.features / 'embeddings.csv'
    if sha256(feature_path) != audit['input_sha256']['feature_embeddings.csv']:
        raise ValueError('14D feature export changed since metadata audit')
    reference = json.loads(args.reference.read_text())
    feature_stats = json.loads((args.features / 'stats.json').read_text())
    if sha256(args.cache) != feature_stats['raw_cache_sha256'] or feature_stats['source_provenance'] != reference['source_provenance']:
        raise ValueError('14D feature source does not match the reference/cache')
    for name, revision in (('ASAP', 'asap_revision'), ('nASAP', 'nasap_revision')):
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent / 'datasets' / name, text=True).strip()
        if actual != reference['source_provenance'][revision]:
            raise ValueError(f'{name} revision changed')
    all_rows = read_csv(args.audit / 'performances.csv')
    features = unique_rows(read_csv(feature_path), 'key')
    baseline = unique_rows(read_csv(args.baseline_export), 'key')
    if set(features) != {r['key'] for r in all_rows} or set(features) != set(baseline):
        raise ValueError('Metadata and both 14D exports have different keys')
    profile_fields = [f'{c}_{s}' for c in CHANNELS for s in ('median_abs' if c == 'rubato' else 'mean', 'p95_p5')]
    for row in all_rows:
        key = row['key']
        if any(features[key][f] != row[f] or baseline[key][f] != row[f] for f in ('piece', 'grid')):
            raise ValueError('14D identity differs from metadata audit')
        a = [float(features[key][f'coordinate_{i}']) for i in range(14)]
        b = [float(baseline[key][f]) for f in profile_fields]
        if not np.allclose(a, b, atol=1e-12, rtol=0):
            raise ValueError('Existing 14D exports disagree')
    pilot = audit['proposed_pilot']
    types = (pilot['type_a'], pilot['type_b'])
    rows = sorted([r for r in all_rows if r['split'] == 'train' and r['composer'] == pilot['composer']
                   and r['work_type'] in types], key=lambda r: r['key'])
    if len(rows) != pilot['total_supported_queries']:
        raise ValueError('Pilot query count differs from the audited selection')
    vectors = np.array([[float(features[r['key']][f'coordinate_{i}']) for i in range(14)] for r in rows])
    distances = distance_matrices(vectors)
    # Independent coordinate weighting, without the production block helper.
    weights = np.r_[np.repeat(.1, 8), np.repeat(1 / 30, 6)]
    independent_full = np.sqrt(np.sum((vectors[:, None] - vectors[None, :]) ** 2 * weights, axis=2))
    full_error = float(np.max(abs(independent_full - distances['representative_range'])))
    decomposition_error = float(np.max(abs(distances['representative_range'] ** 2 -
        (distances['representative'] ** 2 + distances['range_only'] ** 2) / 2)))
    if max(full_error, decomposition_error) > 1e-12:
        raise ValueError('Independent distance/decomposition verification failed')
    queries, family_rows = evaluate(rows, distances)
    summary, works, balanced = aggregate(queries, family_rows)
    selected = select_examples(queries)
    profiles = [r | {f'coordinate_{j}': float(value) for j, value in enumerate(v)} for r, v in zip(rows, vectors)]
    args.out.mkdir(parents=True, exist_ok=True)
    exports = {'pilot_profiles.csv': profiles, 'query_comparisons.csv': queries,
               'family_comparisons.csv': family_rows, 'balanced_queries.csv': balanced,
               'summary.csv': summary, 'work_results.csv': works,
               'feature_contributions.csv': contributions(queries, vectors),
               'distance_balance.csv': distance_balance(queries)}
    for name, data in exports.items():
        write_csv(args.out / name, data)
    (args.out / 'selected.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n')
    plot_overview(summary, args.out)
    for case, q in selected.items():
        plot_example(case, q, rows, vectors, args.out)
    info = dict(composer=pilot['composer'], types=types, queries=len(rows), works=len({r['work_id'] for r in rows}),
        family_pair_rows=len(family_rows), selection_tied_queries=sum(r['selection_tie_pairs'] > 1 for r in queries),
        full_cross_type_closer=sum(r['representative_range_preference'] == 1 for r in queries),
        full_same_type_closer=sum(r['representative_range_preference'] == 0 for r in queries),
        full_tied_queries=sum(r['representative_range_preference'] == .5 for r in queries),
        mean_to_full_reversals=sum(r['representative_preference'] + r['representative_range_preference'] == 1
                                 and r['representative_preference'] != .5 for r in queries),
        representative_cross_type_queries=sum(r['representative_preference'] == 1 for r in queries),
        full_retains_representative_cross_type=sum(r['representative_range_preference'] == 1 and r['representative_preference'] == 1 for r in queries),
        full_agrees_with_range_only=sum(r['representative_range_preference'] == r['range_only_preference'] for r in queries),
        distance_balance={side: dict(mean_range_share=float(np.nanmean([r['range_share'] for r in exports['distance_balance.csv'] if r['side'] == side])),
            median_range_share=float(np.nanmedian([r['range_share'] for r in exports['distance_balance.csv'] if r['side'] == side])))
            for side in ('b', 'c')},
        summaries=summary)
    protocol = dict(**info, representation='existing 14D representatives and p95-p5; Rubato median abs; other means; equal five musical blocks',
        primary_selection='minimum joint sum of 7D representative distances; B other type, C same type; same composer; three distinct families',
        family_balanced='select nearest representative recordings within every eligible B-family/C-family pair, then equal pair average per query',
        ties='absolute tolerance 1e-12; average over every tied selection pair; comparison tie=.5; lexical first display only',
        scopes='train-only descriptive pilot; per-work reference includes candidates; not independent new-recording generalization',
        examples='one per full-distance outcome group, nearest to that group margin median; selected after aggregation for explanation',
        normalization='unchanged source residuals and scales; no new fitting, clipping, coordinate L2 or CNN use',
        limits=['Representative-only selection shares half of 14D coordinates; 14D result is partly selection-aligned, not an independent style test.',
                'Range-only coordinates were not used in selection but share the same recordings and may correlate with representatives.',
                'No human style labels or preference truth; metadata C preference is a reference affinity, not recommendation ground truth.',
                'Only one composer and two type labels; eight works, repeated candidate works and records are dependent.',
                'Family-balanced comparison controls family-pool size; differing recordings per family still affect nearest selection.',
                'Performer identity is not verified or controlled; metadata names are derived labels.'],
        input_sha256={'metadata_protocol': sha256(args.audit / 'protocol.json'), 'metadata_performances': sha256(args.audit / 'performances.csv'),
            'features': sha256(feature_path), 'feature_stats': sha256(args.features / 'stats.json'),
            'baseline_export': sha256(args.baseline_export), 'reference': sha256(args.reference), 'raw_cache': sha256(args.cache)},
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), Path(__file__).with_name('validate_embedding.py'))},
        source_provenance=reference['source_provenance'])
    (args.out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'summary.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    verification = dict(existing_export_rows_checked=len(all_rows), full_distance_cells_checked=independent_full.size,
        maximum_full_distance_error=full_error, maximum_decomposition_error=decomposition_error,
        csv_sha256={name: sha256(args.out / name) for name in exports})
    (args.out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps({k: v for k, v in info.items() if k != 'summaries'}, ensure_ascii=False, indent=2))
    for r in summary:
        if r['query_type'] == 'all':
            print(r['design'], r['mode'], r['query_weighted_preference'], r['work_weighted_preference'])


if __name__ == '__main__':
    main()
