"""Add metadata/summary/CNN/BiLSTM distances for the fixed BWV 848 example.

Reuse the original three recordings and the verified whole-performance exports.
No candidate selection, fitting of an encoder, or change to existing figures.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from audit_asap_metadata import ROOT, read_csv, write_csv, unique_rows, sha256, save_figure
from compare_metadata_summary_embedding import distance_matrices
from compare_metadata_temporal_embeddings import learned_distances
from validate_embedding import BLOCKS
from validate_tempo import plt, setup_style

NAMES = ('MiyashitaM01M', 'Lou01M', 'SunY01M')
KEYS = tuple(f'Bach/Fugue/bwv_848/{name}.mid' for name in NAMES)
ALIASES = ('M', 'L', 'S')
PAIRS = tuple(combinations(range(3), 2))
META_FIELDS = ('composer', 'title', 'work_type')
SEEDS = (20261006, 20261007, 20261008)
MODELS = ('metadata', 'representative_range', 'cnn64', 'bilstm64')
DISPLAY_COORDS = (0, 3, 4, 6, 8)
COLORS = ('#92968f', '#2878c8', '#e26636')


def metadata_vectors(rows, vocabulary_rows, fields=META_FIELDS):
    """Equal-weight categorical groups, distance=sqrt(mean mismatched fields).

    Vocabulary comes from the audited cohort. Recording/file identifiers never
    enter the semantic metadata vector. Identical labels have zero distance.
    """
    if not fields or any(not r.get(f) for r in (*rows, *vocabulary_rows) for f in fields):
        raise ValueError('Missing semantic metadata')
    blocks = []
    vocabulary = {}
    for field in fields:
        labels = sorted({r[field] for r in vocabulary_rows})
        if any(r[field] not in labels for r in rows):
            raise ValueError('Metadata category absent from vocabulary')
        lookup = {label: i for i, label in enumerate(labels)}
        blocks.append(np.eye(len(labels))[[lookup[r[field]] for r in rows]])
        vocabulary[field] = labels
    return np.concatenate(blocks, axis=1) / np.sqrt(2 * len(fields)), vocabulary


def midrank_profile(vector, cohort):
    equal = np.isclose(cohort, vector, atol=1e-12, rtol=0)
    less = (cohort < vector) & ~equal
    return 100 * (less.sum(axis=0) + .5 * equal.sum(axis=0)) / len(cohort)


def assert_existing_profiles(profile_rows, vectors, cohort):
    """Check full vectors and the old display-only percentile coordinates."""
    rows = [r for r in profile_rows if r['panel'] == 'same_work']
    if {r['key'] for r in rows} != set(KEYS):
        raise ValueError('Existing three-recording selection changed')
    indexed = {(r['key'], int(r['coordinate'])): r for r in rows}
    if len(indexed) != len(rows) or len(rows) != 42:
        raise ValueError('Duplicate/incomplete original profiles')
    error = 0.
    for i, key in enumerate(KEYS):
        percentile = midrank_profile(vectors[i], cohort)
        for j in range(14):
            row = indexed[key, j]
            if int(row['cohort_performances']) != len(cohort):
                raise ValueError('Original reference cohort size differs')
            delta = abs(float(row['embedding_value']) - vectors[i, j])
            error = max(error, delta)
            if delta > 1e-12 or abs(float(row['within_piece_percentile']) - percentile[j]) > 1e-10:
                raise ValueError('Original profile differs from current full export')
            if (row['profile_display'] == 'True') != (j in DISPLAY_COORDS):
                raise ValueError('Original display coordinate changed')
    return error


def aggregate_pairs(records):
    summaries = []
    for model in MODELS:
        for i, j in PAIRS:
            local = [r for r in records if r['model'] == model and r['index_a'] == i and r['index_b'] == j]
            if not local:
                raise ValueError('Missing model/pair distance')
            values = [r['distance'] for r in local]
            summaries.append(dict(model=model, index_a=i, index_b=j, key_a=KEYS[i], key_b=KEYS[j],
                pair=f'{ALIASES[i]}–{ALIASES[j]}', runs=len(local),
                distance_mean=float(np.mean(values)), distance_min=float(min(values)), distance_max=float(max(values))))
    return summaries


def draw_distances(axes, records, summaries):
    titles = ('메타데이터', '평균·범위 14D', 'CNN · 전곡 128D', 'BiLSTM · 전곡 128D')
    units = ('범주 불일치 기반 거리', '5개 음악 요소 가중 거리', '코사인 거리', '코사인 거리')
    for ax, model, title, unit in zip(axes, MODELS, titles, units):
        local = [r for r in summaries if r['model'] == model]
        values = [r['distance_mean'] for r in local]
        ax.barh(np.arange(3), values, color='#a3a79f' if model == 'metadata' else '#2878c8', height=.54)
        for position, row in enumerate(local):
            if model in ('cnn64', 'bilstm64'):
                ax.plot([row['distance_min'], row['distance_max']], [position, position], color='#42342a', lw=2)
                points = [r['distance'] for r in records if r['model'] == model
                          and r['index_a'] == row['index_a'] and r['index_b'] == row['index_b']]
                ax.scatter(points, [position] * len(points), color='#42342a', s=18, zorder=3)
            else:
                ax.scatter([row['distance_mean']], [position], color='#60665d', s=24, zorder=3)
            ax.text(row['distance_max'] + (.015 if model != 'representative_range' else .02), position,
                    f"{row['distance_mean']:.3f}", va='center', fontsize=12)
        ax.set_yticks(np.arange(3), [r['pair'] for r in local])
        ax.invert_yaxis()
        ax.set_xlim((-.02 if model == 'metadata' else 0), 1.05 if model != 'representative_range' else 1.5)
        ax.grid(axis='x', alpha=.35)
        ax.set_axisbelow(True)
        ax.set_xlabel(unit, fontsize=11)
        ax.set_title(title, loc='left', fontsize=16, fontweight='bold', pad=16)
        if model == 'metadata':
            ax.text(.55, .5, '세 연주 모두\n같은 메타데이터\n→ 모든 쌍의 거리 0', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='#5a6055')


def render(records, summaries, vectors, cohort, out):
    setup_style()
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.8))
    draw_distances(axes, records, summaries)
    fig.suptitle('같은 메타데이터, 다른 연주 임베딩 거리', fontsize=24, fontweight='bold', y=.98)
    fig.text(.5, .865, 'Bach · Fugue · BWV 848    |    M: MiyashitaM01M    L: Lou01M    S: SunY01M', ha='center', fontsize=13)
    fig.text(.5, .065, '작곡가·작품명·작품 유형은 세 연주 모두 동일  |  학습 모델: 확대 예산 최선 3 seed의 평균(막대)·범위(선)·각 값(점)', ha='center', fontsize=10)
    fig.text(.5, .025, '표현별 거리 정의가 다르므로 막대 길이를 모델 간 정보량·성능의 비교로 사용하지 않는다.', ha='center', fontsize=10, color='#64675f')
    fig.subplots_adjust(left=.055, right=.98, top=.73, bottom=.23, wspace=.38)
    save_figure(fig, out, 'same_work_metadata_distances')

    fig = plt.figure(figsize=(17, 10.5))
    grid = fig.add_gridspec(2, 4, height_ratios=(1.1, 1), left=.075, right=.97,
                            top=.86, bottom=.15, hspace=.65, wspace=.42)
    ax = fig.add_subplot(grid[0, :3])
    for i, (alias, name, color, marker) in enumerate(zip(ALIASES, NAMES, COLORS, ('o', 's', '^'))):
        profile = midrank_profile(vectors[i], cohort)[list(DISPLAY_COORDS)]
        ax.plot(np.arange(5), profile, marker=marker, ms=8, color=color, lw=2.4, label=f'{alias}: {name}')
    ax.axhline(50, color='#999e94', ls=':', lw=1)
    ax.set(xticks=np.arange(5), xticklabels=('빠르기', '루바토 폭', '강약 성향', '음 유지 성향', '페달 깊이'),
           ylim=(0, 100), yticks=(0, 25, 50, 75, 100), ylabel='작품 내 상대 순위 (백분위)')
    ax.set_title('기존에 고른 세 연주의 특징 프로필', loc='left', fontsize=18, pad=15)
    ax.grid(axis='y', alpha=.4)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.19), ncols=3, frameon=False, fontsize=11)
    meta_ax = fig.add_subplot(grid[0, 3]); meta_ax.axis('off')
    meta_ax.text(0, .95, '세 연주에 동일한 정보', va='top', fontsize=17, fontweight='bold')
    meta_ax.text(0, .71, '작곡가  Bach\n\n작품명  Fugue BWV 848\n\n작품 유형  Fugue', va='top', fontsize=14)
    meta_ax.text(0, .18, '메타데이터로는 구별되지 않음\n연주 특징은 서로 다름', va='top', fontsize=12, color='#555e50')
    draw_distances([fig.add_subplot(grid[1, i]) for i in range(4)], records, summaries)
    fig.suptitle('메타데이터가 같아도 연주 표현은 다르다', fontsize=25, fontweight='bold', y=.975)
    fig.text(.5, .918, f'Bach · Fugue · BWV 848  |  기존 9연주 reference 중 동일한 3연주  |  전곡 임베딩 비교', ha='center', fontsize=13)
    fig.text(.5, .075, '위: 기존 그림과 같은 표시용 백분위  /  아래: 실제 임베딩 거리  /  학습 모델은 3 seed 평균·최소~최대', ha='center', fontsize=11)
    fig.text(.5, .04, '백분위는 거리 계산에 사용하지 않는다. 표현별 거리 눈금은 별도이며 모델 간 막대 크기로 우열을 판단하지 않는다.', ha='center', fontsize=10, color='#64675f')
    save_figure(fig, out, 'same_work_metadata_overview')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_style_validation')
    parser.add_argument('--features', type=Path, default=ROOT / 'classicfy-ai/analysis/_feature_search_roles')
    parser.add_argument('--original', type=Path, default=ROOT / 'classicfy-ai/analysis/_interpretation_embedding')
    parser.add_argument('--learned', type=Path, default=ROOT.parent / 'datasets/metadata_temporal_exports_run01')
    parser.add_argument('--out', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_same_work')
    args = parser.parse_args()
    audit_protocol = json.loads((args.audit / 'protocol.json').read_text())
    audit_verification = json.loads((args.audit / 'verification.json').read_text())
    if sha256(args.audit / 'performances.csv') != audit_verification['csv_sha256']['performances.csv']:
        raise ValueError('Metadata audit changed')
    if sha256(args.features / 'embeddings.csv') != audit_protocol['input_sha256']['feature_embeddings.csv']:
        raise ValueError('Audited 14D export changed')
    audited = read_csv(args.audit / 'performances.csv')
    metadata = unique_rows(audited, 'key')
    feature_rows = unique_rows(read_csv(args.features / 'embeddings.csv'), 'key')
    selected = [metadata[key] for key in KEYS]
    if len({r['piece'] for r in selected}) != 1 or len({r['grid'] for r in selected}) != 1:
        raise ValueError('Selected recordings must have the same score and beat grid')
    for key in KEYS:
        if any(metadata[key][f] != feature_rows[key][f] for f in ('piece', 'grid')):
            raise ValueError('Metadata/feature identity differs')
    vectors = np.array([[float(feature_rows[key][f'coordinate_{j}']) for j in range(14)] for key in KEYS])
    cohort = np.array([[float(r[f'coordinate_{j}']) for j in range(14)] for r in feature_rows.values()
                       if r['piece'] == selected[0]['piece'] and r['grid'] == selected[0]['grid']])
    error = assert_existing_profiles(read_csv(args.original / 'profiles.csv'), vectors, cohort)
    meta_vectors, vocabulary = metadata_vectors(selected, audited)
    meta_distance = np.linalg.norm(meta_vectors[:, None] - meta_vectors[None, :], axis=2)
    expected_meta = np.array([[np.sqrt(np.mean([a[f] != b[f] for f in META_FIELDS])) for b in selected] for a in selected])
    if not np.allclose(meta_distance, expected_meta, atol=1e-12, rtol=0) or np.any(meta_distance != 0):
        raise ValueError('Identical semantic metadata must collapse exactly')
    summary_distance = distance_matrices(vectors)['representative_range']
    manifest = json.loads((args.learned / 'manifest.json').read_text())
    feature_stats = json.loads((args.features / 'stats.json').read_text())
    if manifest['source_provenance'] != feature_stats['source_provenance'] or manifest['cache_sha256'] != feature_stats['raw_cache_sha256']:
        raise ValueError('Learned and summary source provenance differs')
    learned_identities = unique_rows(manifest['identities'], 'key')
    for key in KEYS:
        if any(metadata[key][f] != learned_identities[key][f] for f in ('piece', 'grid')):
            raise ValueError('Learned inference identity differs')
    matrices = [('metadata', 0, meta_distance, 'categorical_equal_groups', meta_vectors.shape[1]),
                ('representative_range', 0, summary_distance, 'five_feature_rms', 14)]
    sources = []
    for arm in ('cnn64', 'bilstm64'):
        entries = [e for e in manifest['exports'] if e['arm'] == arm and e['stage'] == 'best_extended']
        if sorted(e['seed'] for e in entries) != list(SEEDS):
            raise ValueError('Expected every saved model seed once')
        for entry in sorted(entries, key=lambda e: e['seed']):
            path = args.learned / entry['file']
            if sha256(path) != entry['export_sha256']:
                raise ValueError('Learned export changed')
            with np.load(path, allow_pickle=False) as z:
                keys = list(z['keys'])
                if len(keys) != len(set(keys)) or set(keys) != set(metadata):
                    raise ValueError('Learned cohort keys differ')
                lookup = {key: i for i, key in enumerate(keys)}
                values = z['vectors'][[lookup[key] for key in KEYS]].astype(np.float64)
            if values.shape != (3, 128):
                raise ValueError('Expected 128D whole-performance mean+SD vectors')
            matrices.append((arm, entry['seed'], learned_distances(values, 'cosine'), 'cosine', 128))
            sources.append(entry)
    records, independent_error = [], 0.
    for model, seed, matrix, metric, dimensions in matrices:
        for i, j in PAIRS:
            value = float(matrix[i, j])
            if model == 'metadata':
                independent = expected_meta[i, j]
            elif model == 'representative_range':
                weights = np.zeros(14)
                for coords in BLOCKS.values():
                    weights[coords] = 1 / (5 * len(coords))
                independent = np.sqrt(np.dot((vectors[i] - vectors[j]) ** 2, weights))
            else:
                entry = next(e for e in sources if e['arm'] == model and e['seed'] == seed)
                with np.load(args.learned / entry['file'], allow_pickle=False) as z:
                    lookup = {key: k for k, key in enumerate(z['keys'])}
                    x, y = (z['vectors'][lookup[key]].astype(np.float64) for key in (KEYS[i], KEYS[j]))
                independent = np.clip(1 - np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)), 0, 2)
            independent_error = max(independent_error, abs(value - independent))
            if abs(value - independent) > 1e-12:
                raise ValueError('Independent scalar distance differs')
            records.append(dict(model=model, seed=seed, metric=metric, dimensions=dimensions,
                index_a=i, index_b=j, key_a=KEYS[i], key_b=KEYS[j], pair=f'{ALIASES[i]}–{ALIASES[j]}', distance=value))
    summaries = aggregate_pairs(records)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {'selected_performances.csv': [dict(alias=alias, **row) for alias, row in zip(ALIASES, selected)],
               'pair_distances.csv': records, 'pair_summary.csv': summaries,
               'profile_display.csv': [dict(key=key, alias=ALIASES[i], coordinate=j,
                   embedding_value=float(vectors[i, j]), percentile=float(midrank_profile(vectors[i], cohort)[j]))
                   for i, key in enumerate(KEYS) for j in DISPLAY_COORDS]}
    for name, rows in outputs.items():
        write_csv(args.out / name, rows)
    render(records, summaries, vectors, cohort, args.out)
    protocol = dict(selection='Original same_work example unchanged; all three preselected BWV 848 recordings; no distance-based reselection',
        keys=KEYS, cohort_performances=len(cohort), metadata_fields=META_FIELDS,
        metadata='Equal-weight one-hot composer/title/derived work type; scaled by sqrt(2 * number of fields); no filename/recording ID',
        metadata_vocabulary=vocabulary, learned='All three saved seeds; extended validation-selected CNN64/BiLSTM64; whole-performance latent mean+SD 128D',
        summary='Original normalized 14D; equal five music blocks; three pedal channels share one block; no new coordinate fitting',
        profile='Existing display coordinates and within-piece midrank percentile; not inputs to the distance',
        figure='Model-specific distance scales; learned bars=mean, lines=min/max, dots=all seeds; range is not a confidence interval',
        source_provenance=manifest['source_provenance'], selected_model_exports=sources,
        limits=['A fixed three-recording illustration, not a population result or a held-out performance comparison.',
                'Nonzero distance alone does not establish that all intended musical information is reflected; knowledge ablations and feature-distance tests are separate.',
                'Metadata scope is composer/title/work type; performer metadata is not asserted.',
                'Profile values are display-only, and raw distance magnitudes cannot rank different embedding types.'],
        input_sha256={'original_profiles': sha256(args.original / 'profiles.csv'),
            'feature_export': sha256(args.features / 'embeddings.csv'), 'metadata_performances': sha256(args.audit / 'performances.csv'),
            'learned_manifest': sha256(args.learned / 'manifest.json')},
        code_sha256=sha256(__file__))
    (args.out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'verification.json').write_text(json.dumps(dict(original_profile_coordinates_checked=42,
        maximum_original_vector_error=error, independent_pair_distances_checked=len(records),
        maximum_independent_distance_error=independent_error,
        metadata_identical_pairs=sum(r['model'] == 'metadata' and r['distance'] == 0 for r in records),
        verified_learned_exports=len(sources), csv_sha256={name: sha256(args.out / name) for name in outputs}), indent=2) + '\n')
    for row in summaries:
        print(row['model'], row['pair'], row['distance_mean'], row['distance_min'], row['distance_max'])


if __name__ == '__main__':
    main()
