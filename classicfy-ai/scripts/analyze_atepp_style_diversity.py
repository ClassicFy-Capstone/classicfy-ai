"""Measure within-work diversity and cross-composer proximity in normalized ATEPP.

Uses saved standardized beat sequences and the existing 14D feature summaries.
This analysis does not use a learned CNN embedding or listening labels.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from validate_embedding import BLOCKS, CHANNELS, summarize
from validate_tempo import plt, setup_style

NAMES = list(BLOCKS)
COLORS = ['#2878c8', '#8d62b7', '#d76b47', '#148766', '#ba8b26']
GREEN, GRAY = '#148766', '#87919d'
TOL = 1e-12
COMPOSERS = {'Ludwig van Beethoven': '베토벤', 'Wolfgang Amadeus Mozart': '모차르트',
             'Johann Sebastian Bach': '바흐', 'Claude Debussy': '드뷔시',
             'Maurice Ravel': '라벨', 'Franz Schubert': '슈베르트',
             'Franz Liszt': '리스트', 'Robert Schumann': '슈만',
             'Sergei Rachmaninoff': '라흐마니노프', 'Franz Joseph Haydn': '하이든'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def block_squared(x, y):
    """Five block distances; average coordinates within each musical feature."""
    return np.stack([np.mean((x[..., ix] - y[..., ix]) ** 2, axis=-1)
                     for ix in BLOCKS.values()], axis=-1)


def weighted_vectors(vectors):
    weights = np.zeros(14)
    for ix in BLOCKS.values():
        weights[ix] = 1 / (5 * len(ix))
    return vectors * np.sqrt(weights)


def save_figure(fig, out, stem):
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'{stem}.{extension}', dpi=200, facecolor='#fcfcfb')
    plt.close(fig)


def boxplot(ax, samples, labels, colors):
    artists = ax.boxplot(samples, tick_labels=labels, patch_artist=True,
                         showfliers=False, widths=.55)
    for patch, color in zip(artists['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(.5)
    for median in artists['medians']:
        median.set_color('#333333')
    ax.grid(axis='y', alpha=.5); ax.set_axisbelow(True)


def diversity(cache_path, data, out):
    rows, pair_rows, examples = [], [], {}
    expected = {r.key: r for r in data.itertuples()}
    coordinate_columns = list(data.attrs['coordinate_columns'])
    with np.load(cache_path, allow_pickle=False) as cache:
        manifest = json.loads(str(cache['manifest']))
        index = {(r['piece'], r['key'], r['channel']): i
                 for i, r in enumerate(manifest['records'])}
        for group_number, (piece, group) in enumerate(data.groupby('piece', sort=True)):
            keys = group.key.tolist()
            ids = [[index[piece, key, channel] for channel in CHANNELS] for key in keys]
            values = np.array([[cache[f'{i}/standardized_values'] for i in entry] for entry in ids])
            masks = np.array([[cache[f'{i}/standardized_mask'] for i in entry] for entry in ids])
            shared = (masks & np.isfinite(values)).all(axis=(0, 1))
            if shared.sum() < 32 or shared.mean() < .5 or len(keys) < 5:
                raise ValueError(f'{piece}: exported cohort does not satisfy shared-grid criteria')
            values = values[:, :, shared]
            summary = np.array([summarize(v) for v in values])
            np.testing.assert_allclose(summary, group[coordinate_columns].to_numpy(), atol=1e-11, rtol=1e-11)
            pairs = np.array(list(combinations(range(len(keys)), 2)))
            delta = values[pairs[:, 0]] - values[pairs[:, 1]]
            summary_components = block_squared(summary[pairs[:, 0]], summary[pairs[:, 1]])
            for b, (name, ix) in enumerate(BLOCKS.items()):
                channels = np.unique(np.array(ix) // 2)
                distances = np.sqrt(np.mean(delta[:, channels] ** 2, axis=(1, 2)))
                beat_spread = np.std(values[:, channels], axis=0, ddof=1)
                mean_distances = np.sqrt(np.mean((summary[pairs[:, 0]][:, ix[::2]]
                                                 - summary[pairs[:, 1]][:, ix[::2]]) ** 2, axis=1))
                width_distances = np.sqrt(np.mean((summary[pairs[:, 0]][:, ix[1::2]]
                                                  - summary[pairs[:, 1]][:, ix[1::2]]) ** 2, axis=1))
                rows.append({'piece': piece, 'track': group.track.iloc[0], 'composer': group.composer.iloc[0],
                             'feature': name, 'performances': len(keys), 'shared_beats': int(shared.sum()),
                             'pairs': len(pairs), 'beat_pair_rms_median': float(np.median(distances)),
                             'beat_pair_rms_p90': float(np.percentile(distances, 90)),
                             'beat_between_performance_sd_median': float(np.median(beat_spread)),
                             'summary_pair_distance_median': float(np.median(np.sqrt(summary_components[:, b]))),
                             'tendency_pair_distance_median': float(np.median(mean_distances)),
                             'width_pair_distance_median': float(np.median(width_distances))})
                for (a, c), distance, component in zip(pairs, distances, summary_components[:, b]):
                    pair_rows.append({'piece': piece, 'feature': name, 'key_a': keys[a], 'key_b': keys[c],
                                      'beat_pair_rms': float(distance), 'summary_distance': float(np.sqrt(component))})
            overall = np.sqrt(summary_components.mean(axis=1))
            examples[piece] = (float(np.median(overall)), keys, values,
                               cache[f'{ids[0][0]}/score_beats'][:-1][shared], summary)
            if group_number % 40 == 0:
                print(f'Checked standardized sequences: {group_number + 1} works', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'within_work_diversity.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(pair_rows).to_csv(out / 'within_work_pair_distances.csv', index=False, encoding='utf-8-sig')
    fig, ax = plt.subplots(figsize=(12, 7))
    boxplot(ax, [frame.loc[frame.feature == n, 'beat_pair_rms_median'] for n in NAMES], NAMES, COLORS)
    ax.set(title='동일 작품의 연주들은 정규화 후에도 얼마나 다른가?', ylabel='같은 악보 위치의 연주 쌍 RMS 차이 (표준화 단위)')
    for b, name in enumerate(NAMES):
        median = frame.loc[frame.feature == name, 'beat_pair_rms_median'].median()
        ax.text(b + 1, ax.get_ylim()[1] * .94, f'작품 중앙값 {median:.3f}', ha='center', fontsize=10)
    fig.text(.06, .035, f'{frame.piece.nunique()}작품 · {len(data):,}연주. 작품별 모든 쌍의 중앙값을 한 점으로 집계. 상자는 작품 간 IQR, 수염 1.5 IQR; 극단점만 표시 생략.', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .98)); save_figure(fig, out, '01_within_work_beat_diversity')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title in zip(axes, ['tendency_pair_distance_median', 'width_pair_distance_median'],
                                ['A. 연주 전체의 대표 성향 차이', 'B. 연주 내부 변화 폭의 차이']):
        boxplot(ax, [frame.loc[frame.feature == n, metric] for n in NAMES], NAMES, COLORS)
        ax.set(title=title, ylabel='같은 작품 연주 쌍의 요약 좌표 RMS 차이')
        ax.tick_params(axis='x', rotation=15)
    fig.suptitle('피처 값의 차이와 연주 요약의 차이를 구분', weight='bold', fontsize=20)
    fig.text(.04, .025, '대표 성향=평균 (Rubato는 |편차| 중앙값). 변화 폭=p95−p5. Pedaling은 세 지표의 평균 제곱 차이에 제곱근. 작품마다 동일 가중치.', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .93)); save_figure(fig, out, '02_within_work_summary_diversity')
    target = np.median([e[0] for e in examples.values()])
    piece = min(examples, key=lambda p: (abs(examples[p][0] - target), p))
    _, keys, values, grid, summary = examples[piece]
    order = np.argsort(summary[:, 0], kind='stable')
    selected = [int(order[int(q * (len(order) - 1))]) for q in (.25, .5, .75)]
    center = len(grid) // 2; sl = slice(max(0, center - 48), min(len(grid), center + 48))
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    for b, (name, ax) in enumerate(zip(NAMES, axes)):
        channel = b if b < 4 else 4
        for i in selected:
            ax.plot(grid[sl], values[i, channel, sl], label=f'{keys[i]} · {expected[keys[i]].artist}', lw=1.5)
        ax.axhline(0, color=GRAY, lw=.8); ax.set_ylabel(name if b < 4 else 'Pedaling\n(depth)'); ax.grid(alpha=.4)
    axes[0].legend(ncols=3, fontsize=10, loc='upper center', bbox_to_anchor=(.5, 1.4))
    axes[-1].set_xlabel('악보 quarter-note 위치')
    fig.suptitle(f'정규화 후에도 위치별 연주 차이는 남는다\n{expected[keys[0]].track}', fontsize=18, weight='bold')
    fig.text(.035, .018, '전체 14D 연주 쌍 다양성이 작품 중앙값에 가장 가까운 작품. Tempo 성향 25/50/75백분위 연주; 중앙 최대 96개 공통 유효 위치. 페달은 depth만 표시.', fontsize=10)
    fig.tight_layout(rect=(0, .045, 1, .92)); save_figure(fig, out, '03_same_work_sequence_example')
    sequence_rows = []
    for i in selected:
        for ch, name in enumerate(CHANNELS):
            sequence_rows.extend({'piece': piece, 'key': keys[i], 'channel': name, 'score_position': float(p), 'value': float(v)}
                                 for p, v in zip(grid[sl], values[i, ch, sl]))
    pd.DataFrame(sequence_rows).to_csv(out / 'same_work_example_sequences.csv', index=False, encoding='utf-8-sig')
    return frame, {'piece': piece, 'keys': [keys[i] for i in selected], 'median_14d_pair_distance': examples[piece][0]}


def nearest_analysis(data, vectors, out, rng):
    z = weighted_vectors(vectors)
    distances = np.sqrt(np.maximum(np.sum(z*z, axis=1)[:, None] + np.sum(z*z, axis=1)[None, :] - 2*z@z.T, 0))
    pieces, composers = data.piece.to_numpy(), data.composer.to_numpy()
    artists = data.artist.str.casefold().str.replace(r'\s+', ' ', regex=True).str.strip().to_numpy()
    hashes = data.midi_sha256.to_numpy()
    eligible = (pieces[:, None] != pieces[None, :]) & (artists[:, None] != artists[None, :]) & (hashes[:, None] != hashes[None, :])
    full = np.where(eligible, distances, np.inf)
    same = composers[:, None] == composers[None, :]
    same_distance = np.where(same, full, np.inf).min(axis=1)
    cross_distance = np.where(~same, full, np.inf).min(axis=1)
    if not np.isfinite(same_distance).all() or not np.isfinite(cross_distance).all():
        raise ValueError('Every query needs both same-composer and cross-composer comparators')
    nearest = np.isclose(full, full.min(axis=1)[:, None], atol=TOL, rtol=0)
    fractions = (nearest & ~same).sum(axis=1) / nearest.sum(axis=1)
    chance = (eligible & ~same).sum(axis=1) / eligible.sum(axis=1)
    cross_index = np.argmin(np.where(~same, full, np.inf), axis=1)
    same_index = np.argmin(np.where(same, full, np.inf), axis=1)
    queries = data[['key', 'piece', 'composer', 'artist', 'track']].copy()
    queries['closest_cross_key'] = data.key.to_numpy()[cross_index]
    queries['closest_same_composer_key'] = data.key.to_numpy()[same_index]
    queries['cross_distance'], queries['same_composer_distance'] = cross_distance, same_distance
    queries['cross_strictly_closer'] = (cross_distance < same_distance) & ~np.isclose(cross_distance, same_distance, atol=TOL, rtol=0)
    queries['nearest_cross_fraction'], queries['random_candidate_cross_fraction'] = fractions, chance
    queries['candidate_count'] = eligible.sum(axis=1)
    queries.to_csv(out / 'cross_composer_queries.csv', index=False, encoding='utf-8-sig')
    work = queries.groupby('piece')[['nearest_cross_fraction', 'random_candidate_cross_fraction', 'cross_strictly_closer']].mean()
    work.to_csv(out / 'cross_composer_work_summary.csv', encoding='utf-8-sig')
    samples = work.to_numpy()[rng.integers(0, len(work), size=(2000, len(work)))].mean(axis=1)
    ci = np.percentile(samples, [2.5, 97.5], axis=0)
    stats = {'performances': len(data), 'works': len(work), 'composers': data.composer.nunique(),
             'query_cross_nearest_fraction': float(fractions.mean()),
             'work_macro_cross_nearest_fraction': float(work.nearest_cross_fraction.mean()),
             'work_macro_cross_nearest_ci95': ci[:, 0].tolist(),
             'work_macro_random_candidate_cross_fraction': float(work.random_candidate_cross_fraction.mean()),
             'work_macro_strict_cross_closer_fraction': float(work.cross_strictly_closer.mean()),
             'bootstrap_samples': 2000, 'eligible_directed_pairs': int(eligible.sum())}
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax = axes[0]
    ax.scatter(same_distance, cross_distance, c=np.where(queries.cross_strictly_closer, GREEN, GRAY), alpha=.25, s=12)
    bound = max(same_distance.max(), cross_distance.max()) * 1.04
    ax.plot([0, bound], [0, bound], '--', color=GRAY); ax.set_xlim(0, bound); ax.set_ylim(0, bound)
    ax.set(title='A. 모든 연주의 실제 14D 최소 거리', xlabel='같은 작곡가 · 다른 작품의 최근접 거리', ylabel='다른 작곡가의 최근접 거리')
    ax.text(.04, .93, '대각선 아래: 다른 작곡가가 더 가까움', transform=ax.transAxes, fontsize=11, color=GREEN)
    inset = ax.inset_axes([.49, .12, .46, .45])
    zoom = float(np.percentile(np.r_[same_distance, cross_distance], 95) * 1.05)
    inset.scatter(same_distance, cross_distance, c=np.where(queries.cross_strictly_closer, GREEN, GRAY), alpha=.25, s=7)
    inset.plot([0, zoom], [0, zoom], '--', color=GRAY, lw=.8)
    inset.set_xlim(0, zoom); inset.set_ylim(0, zoom)
    inset.set_title('거리 95백분위 구간 확대', fontsize=9)
    inset.tick_params(labelsize=8)
    ax = axes[1]
    heights = work[['nearest_cross_fraction', 'random_candidate_cross_fraction']].mean().to_numpy()*100
    ax.bar([0, 1], heights, color=[GREEN, GRAY], width=.55)
    ax.errorbar([0, 1], heights, yerr=np.abs(ci[:, :2]*100-heights).T, fmt='none', capsize=5, color='#333333')
    ax.set(xticks=[0, 1], xticklabels=['실제 최근접\n다른 작곡가 비율', '후보를 무작위 선택\n다른 작곡가 비율'], ylim=(0, 100), ylabel='작품 균등 평균 (%)', title='B. 전체 비율과 후보 구성 기준선')
    for i, height in enumerate(heights): ax.text(i, height+4, f'{height:.1f}%', ha='center', weight='bold')
    fig.suptitle('작곡가가 달라도 비슷한 피처 요약이 나오는가?', fontsize=21, weight='bold')
    fig.text(.04, .025, '다른 작품·다른 연주자·다른 MIDI 해시만 후보. 동점은 비율 배분. 오차 막대=작품 bootstrap 95% 구간; 후보 구성 기준선은 작곡가 불균형을 반영.', fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .93)); save_figure(fig, out, '04_cross_composer_population')
    # Choose the two most represented composers by the number of works, before looking at distances.
    counts = data.groupby('composer').piece.nunique().sort_values(ascending=False)
    ca, cb = counts.index[:2]
    aa, bb = np.flatnonzero(composers == ca), np.flatnonzero(composers == cb)
    keys = data.key.to_numpy()
    pair_candidates = [(float(full[a, b]), str(keys[a]), str(keys[b]), int(a), int(b))
                       for a in aa for b in bb if np.isfinite(full[a, b])]
    chosen, used_pieces, used_artists = [], set(), set()
    for rank, (distance, _, _, a, b) in enumerate(sorted(pair_candidates), 1):
        if {pieces[a], pieces[b]} & used_pieces or {artists[a], artists[b]} & used_artists: continue
        chosen.append((a, b, distance, rank)); used_pieces.update([pieces[a], pieces[b]]); used_artists.update([artists[a], artists[b]])
        if len(chosen) == 2: break
    if len(chosen) != 2: raise ValueError('Cannot choose two distinct-work/artist pairs')
    return stats, chosen, distances, full, (ca, cb), vectors


def composer_example(data, vectors, selected_pairs, distances, full, composer_pair, out):
    indices = [i for a, b, _, _ in selected_pairs for i in (a, b)]
    selected = data.iloc[indices].reset_index(drop=True).copy()
    codes = ['A1', 'B1', 'A2', 'B2']; selected.insert(0, 'id', codes)
    z = weighted_vectors(vectors[indices]); centered = z-z.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ axes[:2].T; variance = singular[:2]**2 / np.sum(singular**2)
    d = distances[np.ix_(indices, indices)]
    np.testing.assert_allclose(np.linalg.norm(z[:, None]-z[None, :], axis=2), d, atol=1e-7)
    hits = [bool(np.isclose(d[i, i ^ 1], np.min(np.delete(d[i], i)), atol=TOL, rtol=0)) for i in range(4)]
    selected['pc1'], selected['pc2'] = xy.T
    selected['style_partner_nearest_in_four'] = hits
    global_ranks = []
    for i, q in enumerate(indices):
        target_distance = full[q, indices[i ^ 1]]
        tied = np.isclose(full[q], target_distance, atol=TOL, rtol=0)
        lower = (full[q] < target_distance) & ~tied
        global_ranks.append(float(lower.sum() + (tied.sum()+1)/2))
    selected['style_partner_rank_in_entire_gallery'] = global_ranks
    selected.to_csv(out / 'selected_performances.csv', index=False, encoding='utf-8-sig')
    pair_rows = []
    for a, b in combinations(range(4), 2):
        components = np.sqrt(block_squared(vectors[indices[a]], vectors[indices[b]]))
        pair_rows.append({'a': codes[a], 'b': codes[b], 'distance_14d': float(d[a, b]),
                          'same_composer': a % 2 == b % 2, 'selected_style_pair': a//2 == b//2,
                          **dict(zip(NAMES, components.tolist()))})
    pd.DataFrame(pair_rows).to_csv(out / 'selected_pairwise_distances.csv', index=False, encoding='utf-8-sig')
    fig = plt.figure(figsize=(16, 11))
    fig.text(.045, .945, 'ATEPP: 작곡가가 달라도 연주 피처가 가까운가?', fontsize=25, weight='bold')
    fig.text(.045, .90, '정규화된 5개 feature의 성향·변화 폭 → 14차원 실제 거리와 PCA 배치', fontsize=14, color='#555953')
    fig.text(.045, .85, f'선정한 4연주 중 {sum(hits)}연주: 다른 작곡가의 선정 상대가 4개 중 최근접 (동점 포함)', fontsize=17, color=GREEN, weight='bold')
    ax = fig.add_axes((.07, .37, .41, .39))
    for a, b in ((0, 2), (1, 3)): ax.plot(xy[[a, b], 0], xy[[a, b], 1], '--', color=GRAY, lw=1.7)
    for a, b in ((0, 1), (2, 3)): ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=GREEN, lw=3)
    for i, code in enumerate(codes):
        color, marker = (COLORS[2], 'o') if i % 2 == 0 else (COLORS[0], '^')
        ax.scatter(*xy[i], s=220, color=color, marker=marker, edgecolor='white', zorder=3)
        ax.annotate(code, xy[i], xytext=(0, 14), textcoords='offset points', ha='center', color=color, weight='bold', fontsize=13)
    ax.set(title='A. 가중 14D 좌표의 2차원 PCA', xlabel=f'PC1 ({variance[0]:.1%})', ylabel=f'PC2 ({variance[1]:.1%})')
    ax.margins(.3); ax.grid(alpha=.4)
    handles = [Line2D([], [], marker='o', ls='', color=COLORS[2], label=COMPOSERS.get(composer_pair[0], composer_pair[0])),
               Line2D([], [], marker='^', ls='', color=COLORS[0], label=COMPOSERS.get(composer_pair[1], composer_pair[1])),
               Line2D([], [], color=GREEN, lw=3, label='선정 스타일 유사 쌍'), Line2D([], [], color=GRAY, ls='--', label='같은 작곡가')]
    fig.legend(handles=handles, bbox_to_anchor=(.29, .30), loc='center', ncols=2, fontsize=11, frameon=False)
    ax = fig.add_axes((.66, .37, .28, .39))
    comparisons = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (2, 1)]
    labels = ['A1–B1  선정 유사 쌍', 'A2–B2  선정 유사 쌍', 'A1–A2  같은 작곡가', 'B1–B2  같은 작곡가', 'A1–B2  나머지 쌍', 'A2–B1  나머지 쌍']
    values = [d[a, b] for a, b in comparisons]
    ax.barh(range(6), values, color=[GREEN, GREEN, GRAY, GRAY, '#c8cdd3', '#c8cdd3'], height=.6)
    ax.set_yticks(range(6), labels, fontsize=11); ax.invert_yaxis()
    ax.set(title='B. 원래 14차원의 실제 거리', xlabel='5개 feature 균등 RMS 거리\n작을수록 피처 요약이 비슷함', xlim=(0, max(values)*1.23))
    ax.grid(axis='x', alpha=.4); ax.set_axisbelow(True)
    for i, v in enumerate(values): ax.text(v+max(values)*.02, i, f'{v:.3f}', va='center', fontsize=12)
    fig.text(.045, .255, '선정 연주 · 서로 다른 4작품 / 4연주자', fontsize=13, weight='bold')
    for i, row in selected.iterrows():
        label = f'{codes[i]} · {COMPOSERS.get(row.composer, row.composer)} · MIDI {row.key} · {row.artist}\n{textwrap.fill(row.track, width=67)}'
        fig.text(.045 if i < 2 else .52, .218 if i % 2 == 0 else .155, label, fontsize=9.5)
    fig.text(.045, .09, f'작품 수가 가장 많은 두 작곡가를 먼저 선정. 피처 거리순으로 2쌍 선택; 쌍 순위 {selected_pairs[0][3]} / {selected_pairs[1][3]}.', fontsize=10, color='#555953')
    fig.text(.045, .062, f'PCA는 선정한 4개의 가중 14D 좌표에만 적용 (표시 분산 {variance.sum():.1%}); 오른쪽 실제 거리로 판정. CNN 임베딩 아님.', fontsize=10, color='#555953')
    fig.text(.045, .034, '유사성을 기준으로 고른 탐색 사례. 전체 모집단 결과는 04 그림에 별도 제시. 청취 스타일의 동일성이나 모델 일반화 성능을 증명하지 않음.', fontsize=10, color='#555953')
    save_figure(fig, out, '05_composer_vs_style_example')
    fig, ax = plt.subplots(figsize=(14, 5))
    image = ax.imshow(vectors[indices], aspect='auto', cmap='coolwarm')
    labels = [f'{c}\n{part}' for c in CHANNELS for part in ('성향', '폭')]
    ax.set_xticks(range(14), labels, rotation=40, ha='right', fontsize=10)
    ax.set_yticks(range(4), [f'{code} · {selected.artist.iloc[i]}' for i, code in enumerate(codes)])
    ax.set_title('선정 연주의 실제 14개 피처 요약값', weight='bold', pad=15)
    fig.colorbar(image, ax=ax, label='표준화된 성향 / 변화 폭')
    fig.text(.04, .025, '검색 전에 별도 전역 z-score를 적용하지 않음. Rubato 성향=|편차| 중앙값; Pedaling 세 지표는 합쳐서 한 feature 가중치.', fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, 1)); save_figure(fig, out, '06_selected_feature_profiles')
    return {'composers': list(composer_pair), 'selection_pair_ranks': [p[3] for p in selected_pairs],
            'selected_keys': selected.key.tolist(), 'nearest_style_partner_count_in_four': sum(hits),
            'style_partner_ranks_in_entire_gallery': global_ranks,
            'pca_variance_ratio': variance.tolist(), 'distance': '14D five-block equally weighted RMS',
            'selection': 'Two composers with the most eligible works; greedy minimum feature distance, four distinct works and artists; no outcome constraint.'}


def main():
    ai = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ai/'analysis/atepp_normalization')
    parser.add_argument('--out', type=Path, default=ai/'analysis/atepp_style_diversity')
    parser.add_argument('--seed', type=int, default=20261011)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True); setup_style()
    plt.rcParams.update({'axes.titlesize': 15, 'axes.labelsize': 12, 'pdf.fonttype': 42})
    embeddings = args.analysis/'_embedding_validation/embeddings.csv'
    catalog = args.analysis/'performance_catalog.csv'
    summary = pd.read_csv(embeddings, dtype={'key': str})
    coordinate_columns = summary.columns[3:].tolist()
    if len(coordinate_columns) != 14: raise ValueError('Expected existing 14D feature summaries')
    data = summary.merge(pd.read_csv(catalog, dtype={'key': str}), on=['key', 'piece'], validate='one_to_one')
    if len(data) != len(summary) or not data.key.is_unique or data[['artist', 'composer', 'midi_sha256']].isna().any().any():
        raise ValueError('Incomplete or duplicate performance metadata')
    data.attrs['coordinate_columns'] = coordinate_columns
    vectors = data[coordinate_columns].to_numpy()
    if not np.isfinite(vectors).all(): raise ValueError('Nonfinite summaries')
    normalization = json.loads((args.analysis/'stats.json').read_text(encoding='utf-8'))['normalization']
    cache_path = Path(normalization['normalized_cache'])
    print('Verifying normalized cache fingerprint', flush=True)
    cache_hash = sha256(cache_path)
    if cache_hash != normalization['normalized_cache_sha256']: raise ValueError('Normalized cache hash changed')
    frame, same_example = diversity(cache_path, data, args.out)
    print('Comparing all eligible cross-work/composer candidates', flush=True)
    population, pairs, distances, full, composers, vectors = nearest_analysis(data, vectors, args.out, np.random.default_rng(args.seed))
    example = composer_example(data, vectors, pairs, distances, full, composers, args.out)
    diversity_summary = frame.groupby('feature', sort=False)[['beat_pair_rms_median', 'summary_pair_distance_median', 'tendency_pair_distance_median', 'width_pair_distance_median']].median()
    diversity_summary.to_csv(args.out/'diversity_feature_summary.csv', encoding='utf-8-sig')
    stats = {'dataset': 'ATEPP-1.2', 'seed': args.seed, 'population': population, 'same_work_example': same_example,
             'composer_example': example, 'feature_diversity_work_medians': diversity_summary.to_dict(orient='index'),
             'cnn_used': False, 'shared_grid_summaries_verified': len(data),
             'input_sha256': {str(p.resolve()): sha256(p) for p in (embeddings, catalog, args.analysis/'stats.json', Path(__file__))},
             'normalized_cache_sha256': cache_hash}
    (args.out/'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    table = '\n'.join(f'| {name} | {r.beat_pair_rms_median:.4f} | {r.tendency_pair_distance_median:.4f} | {r.width_pair_distance_median:.4f} |' for name, r in diversity_summary.iterrows())
    readme = f'''# ATEPP 정규화 후 연주 다양성과 작곡가 간 피처 유사성

## 동일 작품 내 다섯 feature 차이

품질 승인·정규화 후 5개 이상 연주, 32개 이상 공통 유효 beat, 공통 coverage 50% 이상인 {population['works']}작품·{population['performances']:,}연주를 분석했다. 모든 채널/연주가 함께 유효한 동일 악보 위치만 비교했다. 저장 cache에서 다시 계산한 14차원 요약을 기존 export와 전부 대조했다.

| Feature | 위치별 연주 쌍 RMS 차이 | 대표 성향 차이 | 변화 폭 차이 |
| --- | --- | --- | --- |
{table}

표는 작품 안 모든 연주 쌍의 중앙값을 구한 뒤 작품 간 중앙값을 취한다. 성향=평균, Rubato만 절대값 중앙값; 변화 폭=p95−p5. 위치별 RMS는 정규화된 동일 위치 값의 제곱 차이를 beat와 채널에 걸쳐 평균한 제곱근이다. Pedaling의 depth/down ratio/changes는 평균 제곱 차이로 하나의 feature로 묶는다. 작품 내 변동성으로 scale을 fit했으므로, feature 간 RMS 차이를 음악적 표현의 절대 다양성 순위로 해석할 수 없다. 표준화 단위에서 남아 있는 편차를 나타낸다. 원래 단위와 scale은 [정규화 보고서](../atepp_normalization/README.md)를 확인한다. 작은 모집단·불완전 채널 그룹은 이 비교에서 제외했다.

![위치별 다양성](01_within_work_beat_diversity.png)

![성향과 폭](02_within_work_summary_diversity.png)

![동일 작품 사례](03_same_work_sequence_example.png)

[작품별 결과](within_work_diversity.csv), [모든 연주 쌍](within_work_pair_distances.csv), [대표 작품 위치별 값](same_work_example_sequences.csv).

## 다른 작곡가라도 가까운가?

기존 14차원 정규화 feature 요약에서 다섯 feature의 평균 제곱 거리를 균등 평균한 제곱근을 사용했다. Pedaling 6좌표는 한 묶음으로 평균하므로 세 배 가중되지 않는다. 추가 전역 표준화나 L2 정규화는 적용하지 않는다. 실제 ATEPP CNN 임베딩 파일이 없어 이 결과는 **feature 요약 거리**다.

다른 작품·다른 연주자(메타데이터 이름 정규화)·다른 MIDI 해시만 후보로 삼았다. 같은 작곡가의 다른 작품 후보와 다른 작곡가 후보를 모두 비교했다. 전체 연주 평균의 다른 작곡가 최근접 비율은 {population['query_cross_nearest_fraction']:.2%}, 작품 균등 평균은 **{population['work_macro_cross_nearest_fraction']:.2%}** (작품 bootstrap 95% 구간 {population['work_macro_cross_nearest_ci95'][0]:.2%}–{population['work_macro_cross_nearest_ci95'][1]:.2%})다. 후보를 균등 무작위 선택할 때의 다른 작곡가 비율은 {population['work_macro_random_candidate_cross_fraction']:.2%}다. 작곡가별 후보 수가 불균등하므로 높은 비율 자체만으로 작곡가 영향 제거를 주장하지 않는다. 최근접 동점은 균등 비율 배분; CSV에 표시한 단일 후보 ID는 export 순서로 결정한다. bootstrap은 작품을 표본 단위로 쓰지만 여러 query가 후보를 공유하는 의존성은 남는다.

![전체 작곡가 비교](04_cross_composer_population.png)

첨부 예시와 같은 4연주 그림은 거리와 무관하게 먼저 작품 수가 가장 많은 두 작곡가를 선택하고, 피처 거리가 작은 순서로 4개의 서로 다른 작품·연주자를 선택했다. 선정 상대가 4개 중 최근접인 연주는 {example['nearest_style_partner_count_in_four']}/4개다. **가까운 쌍을 고른 탐색 사례이며 전체 성능 검증이 아니다.** PCA는 선정한 4개의 가중 좌표에만 fit했다. PCA 그림은 분산 일부만 보이므로 원래 14차원 실제 거리로 판정한다. 청취 스타일의 동일성이나 CNN의 작곡가 독립성을 검증한 결과가 아니다.

![작곡가와 스타일 사례](05_composer_vs_style_example.png)

![선정 피처 값](06_selected_feature_profiles.png)

[전체 query 결과](cross_composer_queries.csv), [작품 균등 집계](cross_composer_work_summary.csv), [선정 연주](selected_performances.csv), [여섯 쌍 거리와 feature별 차이](selected_pairwise_distances.csv). 모든 그림은 PNG와 PDF로 저장한다.

## 재현

저장소 루트에서 `.venv/Scripts/python.exe classicfy-ai/scripts/analyze_atepp_style_diversity.py`를 실행한다. `--analysis`, `--out`, `--seed`를 지정할 수 있다. 원본 정규화 cache 해시를 확인하고 결과 수치·입력 SHA256을 [stats.json](stats.json)에, 산출물 해시를 [artifact_manifest.csv](artifact_manifest.csv)에 저장한다.
'''
    (args.out/'README.md').write_text(readme, encoding='utf-8')
    manifest = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(args.out.iterdir()) if p.is_file() and p.name != 'artifact_manifest.csv']
    pd.DataFrame(manifest).to_csv(args.out/'artifact_manifest.csv', index=False)
    print(json.dumps(population, ensure_ascii=False, indent=2), flush=True)
    print(f'Saved six PNG/PDF figures: {args.out}', flush=True)


if __name__ == '__main__':
    main()
