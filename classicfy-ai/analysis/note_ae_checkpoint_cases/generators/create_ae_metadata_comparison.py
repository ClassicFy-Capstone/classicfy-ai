"""Compare one anchor's AE-nearest other-work recordings under three conditions.

Reuses the verified epoch-8 catalogue embeddings; no model fitting, feature-based
selection or new checkpoint selection. Run with the repository Python venv.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/classicfy-presentation-mpl')
from create_composer_metadata_figure import setup_style, box, MUTED
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'presentation/figures/cross_work_checkpoint'
ANCHOR_SOURCE = ROOT / 'presentation/figures/same_work_checkpoint'
OUT = ROOT / 'presentation/figures/ae_metadata_comparison'
BLUE = '#2878C8'
COLORS = ['#148766', '#7E67AC', '#D29B38']
LABELS = ['같은 연주자', '같은 작곡가', '같은 시대']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_and_select():
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    assert sha(ANCHOR_SOURCE / 'checkpoint.pt') == protocol['checkpoint_sha256']
    meta = {r['perf_id']: r for r in csv.DictReader(
        (ROOT / 'datasets/ATEPP/ATEPP-metadata-1.2.csv').open(encoding='utf-8-sig'))}
    anchor_meta = meta['00957']
    eras = protocol['era_mapping']
    anchor_eras = set(eras[anchor_meta['composer']])
    catalogue = np.load(SOURCE / 'catalogue_embeddings.npz', allow_pickle=False)
    assert str(catalogue['protocol_sha']) == protocol['id']
    ids, e = catalogue['performance_id'].copy(), catalogue['e'].copy()
    index = {key: i for i, key in enumerate(ids)}
    original = np.load(ANCHOR_SOURCE / 'example.npz', allow_pickle=False)
    np.testing.assert_allclose(e[index['ATEPP:00957']], original['embedding'][0], atol=2e-6, rtol=2e-6)
    # Exact original A for both distance calculations and PCA.
    e[index['ATEPP:00957']] = original['embedding'][0]
    distances = np.clip(1 - e @ original['embedding'][0], 0, 2)
    all_rows = list(csv.DictReader((SOURCE / 'all_ranked_candidates.csv').open(encoding='utf-8-sig')))
    anchor, candidates = all_rows[0], all_rows[1:]
    assert anchor['performance_id'] == 'ATEPP:00957'
    assert len(candidates) == 2779
    for rank, r in enumerate(candidates, 1):
        r['global_rank'] = int(r['global_rank'])
        r['cosine_distance'] = float(r['cosine_distance'])
        assert r['global_rank'] == rank and r['work'] != anchor['work']
        np.testing.assert_allclose(r['cosine_distance'], distances[index[r['performance_id']]], atol=1e-12, rtol=0)
    assert all(candidates[i]['cosine_distance'] <= candidates[i + 1]['cosine_distance'] for i in range(len(candidates) - 1))
    tests = [lambda r: meta[r['record_key']]['artist_id'] == anchor_meta['artist_id'],
             lambda r: r['composer'] == anchor_meta['composer'],
             lambda r: r['composer'] != anchor_meta['composer'] and bool(set(eras[r['composer']]) & anchor_eras)]
    pools = [[r for r in candidates if condition(r)] for condition in tests]
    chosen = []
    for j, pool in enumerate(pools):
        r = dict(pool[0])
        r['source_metadata_group'] = r.pop('group')
        r.update(label=str(j + 1), condition=LABELS[j], pool_count=len(pool), selected=True,
                 windows=int(r['windows']),
                 same_artist=meta[r['record_key']]['artist_id'] == anchor_meta['artist_id'],
                 same_composer=r['composer'] == anchor_meta['composer'],
                 same_era=bool(set(eras[r['composer']]) & anchor_eras))
        # Independent direct dot-product check for selected minima.
        value = sum(float(a) * float(b) for a, b in zip(original['embedding'][0], e[index[r['performance_id']]]))
        assert abs(r['cosine_distance'] - max(0, 1 - value)) < 1e-12
        assert r['cosine_distance'] == min(v['cosine_distance'] for v in pool)
        chosen.append(r)
    assert len({r['performance_id'] for r in chosen}) == 3
    centered = e - e.mean(0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    for v in vt[:2]:
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
    xy = centered @ vt[:2].T
    variance = singular[:2] ** 2 / (singular ** 2).sum()
    cohort = [index['ATEPP:00957']] + [index[r['performance_id']] for r in chosen]
    background = np.array([index[r['performance_id']] for r in candidates])
    return protocol, anchor, candidates, chosen, ids, e, xy, variance, cohort, background


def render(chosen, xy, variance, cohort, background):
    setup_style()
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.text(.045, .94, '연주자·작곡가·시대로 비교한 AE 거리', fontsize=27, weight='bold')
    fig.text(.955, .943, 'AtEPP AE · epoch 8 · 동일한 기준 A', ha='right', fontsize=11, color=MUTED)
    fig.text(.045, .891, '기준 A  ·  Vladimir Horowitz  ·  슈만 Kreisleriana, Op.16: 6. Sehr langsam  ·  낭만',
             fontsize=13, color=BLUE)
    fig.text(.045, .846, '같은 작품 제외 · 전체 2,779개 연주 검색 · 각 조건에서 AE가 가장 가까운 연주 1개씩 선정',
             fontsize=12, color=MUTED)

    fig.text(.045, .788, '학습된 AE 임베딩에서의 위치', fontsize=17, weight='bold')
    fig.text(.045, .752, '회색 = 다른 작품 연주 · 기준 A와 세 후보 주변 확대', fontsize=10.5, color=MUTED)
    ax = fig.add_axes([.077, .386, .377, .324])
    ax.scatter(*xy[background].T, color='#BAC5D2', s=12, alpha=.25, edgecolors='none', zorder=1)
    for j, i in enumerate(cohort[1:]):
        ax.plot(xy[[cohort[0], i], 0], xy[[cohort[0], i], 1], color=COLORS[j], lw=1.5, alpha=.65, zorder=2)
        ax.scatter(*xy[i], s=95, color=COLORS[j], edgecolor='white', lw=1, zorder=4)
        offset = [(18, -20), (-22, -22), (-15, 17)][j]
        ax.annotate(str(j + 1), xy[i], xytext=offset, textcoords='offset points', fontsize=13,
                    weight='bold', color=COLORS[j], ha='center',
                    arrowprops={'arrowstyle': '-', 'lw': .7, 'color': COLORS[j]})
    ai = cohort[0]
    ax.scatter(*xy[ai], marker='*', s=280, color=BLUE, edgecolor='white', lw=1.1, zorder=5)
    ax.annotate('A', xy[ai], xytext=(15, 17), textcoords='offset points', fontsize=13,
                weight='bold', color=BLUE, arrowprops={'arrowstyle': '-', 'lw': .7, 'color': BLUE})
    bounds = xy[cohort]
    pad = np.maximum(np.ptp(bounds, axis=0) * .30, .065)
    ax.set_xlim(bounds[:, 0].min() - pad[0], bounds[:, 0].max() + pad[0])
    ax.set_ylim(bounds[:, 1].min() - pad[1], bounds[:, 1].max() + pad[1])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'PC1 ({variance[0]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({variance[1]:.1%})', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(color='#EBEFF4', lw=.7); ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.045, .323, f'PCA는 전체 2,801개로 계산 · 설명 분산 {variance.sum():.1%} · 거리 판단은 원래 64D 기준',
             fontsize=10, color=MUTED)

    fig.text(.535, .788, '기준 A와의 원래 64차원 거리', fontsize=17, weight='bold')
    fig.text(.535, .752, 'cosine 거리 · 작을수록 A와 가까움', fontsize=10.5, color=MUTED)
    axd = fig.add_axes([.636, .423, .268, .255])
    values = [r['cosine_distance'] for r in chosen]
    axd.barh(range(3), values, color=COLORS, height=.48)
    axd.set_yticks(range(3), [f'{j + 1}  {LABELS[j]}' for j in range(3)], fontsize=12)
    axd.invert_yaxis()
    axd.set_xlim(0, max(values) * 1.33)
    for j, (r, v) in enumerate(zip(chosen, values)):
        axd.text(v + .004, j, f"{v:.4f}  ·  전체 {r['global_rank']}위", va='center',
                 fontsize=11, weight='bold', color=COLORS[j])
    axd.set_xlabel('기준 A와의 64D cosine 거리', fontsize=10)
    axd.tick_params(axis='y', length=0, pad=8)
    axd.tick_params(axis='x', labelsize=9)
    axd.grid(axis='x', color='#EBEFF4'); axd.set_axisbelow(True)
    for spine in axd.spines.values():
        spine.set_visible(False)
    fig.text(.535, .323, '전체 순위는 작곡가·연주자·시대 조건을 적용하기 전의 검색 순위', fontsize=10, color=MUTED)

    titles = ['12 Études, Op.8\nXI. B-flat minor',
              'Kreisleriana, Op.16\nII. Sehr innig und nicht zu rasch',
              'Harmonies poétiques et religieuses, S.173\nVII. Funérailles']
    for j, r in enumerate(chosen):
        left = .045 + j * .313
        box(fig, left, .108, .284, .179, '#F6F8FB', radius=.008)
        fig.text(left + .013, .259, f"{j + 1}  {LABELS[j]} · 다른 작품", fontsize=13, weight='bold', color=COLORS[j])
        fig.text(left + .013, .227, f"{r['artist']}  ·  {r['composer']}", fontsize=10.5, weight='bold')
        fig.text(left + .013, .198, titles[j], va='top', fontsize=10, linespacing=1.35)
        detail = f"후보 {r['pool_count']:,}개 중 최근접  ·  {r['performance_id']}"
        if j == 2:
            detail = f"같은 시대·다른 작곡가 {r['pool_count']:,}개 중 최근접"
        fig.text(left + .013, .124, detail, fontsize=9, color=MUTED)
    fig.text(.045, .069,
             '이 A에서는 같은 작곡가·같은 시대의 최근접 후보가 같은 연주자의 최근접 후보보다 AE상 가깝다.',
             fontsize=14, weight='bold')
    fig.text(.045, .035,
             '조건별 후보 수는 다르며 일부 조건은 겹침. 시대는 기존 겹침 범주 사용. 작품은 악장 단위. train/validation/test 카탈로그의 단일 기준 사례.',
             fontsize=9, color=MUTED)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'ae_metadata_comparison.{ext}', dpi=200, facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol, anchor, candidates, chosen, ids, e, xy, variance, cohort, background = load_and_select()
    render(chosen, xy, variance, cohort, background)
    with (OUT / 'selected.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(chosen[0])); writer.writeheader(); writer.writerows(chosen)
    np.savez_compressed(OUT / 'projection.npz', performance_id=ids, xy=xy,
                        selected_id=ids[cohort], selected_e=e[cohort], PCA_variance=variance)
    result = {'anchor': anchor, 'selected': chosen, 'source_protocol_id': protocol['id'],
              'checkpoint_epoch': 8, 'checkpoint_sha256': protocol['checkpoint_sha256'],
              'catalogue_embeddings_sha256': sha(SOURCE / 'catalogue_embeddings.npz'),
              'global_candidates': len(candidates), 'PCA_fit_count': len(ids),
              'PCA_variance': variance.tolist(), 'projection_view': 'local crop around A and 3 minima; no refitting on selected points',
              'conditions': ['same artist_id, any composer', 'same composer, any artist',
                             'same era overlap, different composer, any artist'],
              'selection': 'nearest original 64D cosine distance within each condition, global rank reused; no feature selection',
              'era_mapping': protocol['era_mapping'],
              'verification': {'passed': True, 'same_anchor_and_checkpoint': True,
                               'all_2779_saved_distances_recomputed': True, 'global_ranks_validated': True,
                               'selected_group_minima_and_scalar_dot_products_checked': True,
                               'selected_recordings_distinct_and_other_work': True},
              'limits': ['one query, catalogue search including train/validation/test',
                         'unequal group sizes and overlapping group membership',
                         'no causal claim of performer/composer/era influence',
                         'same composer includes another movement of the same cycle',
                         'no listener similarity evaluation']}
    write(OUT / 'provenance.json', result)
    table = '\n'.join(f"| {r['condition']} | {r['artist']} | {r['composer']} | {r['global_rank']} | {r['cosine_distance']:.4f} | {r['pool_count']} |"
                      for r in chosen)
    note = f'''# 같은 A에서 연주자·작곡가·시대 조건별 AE 최근접 비교

![figure](ae_metadata_comparison.png)

A는 기존과 동일한 Horowitz의 슈만 《Kreisleriana》 6번 (`ATEPP:00957`)이다.
앞서 고정한 epoch 8 체크포인트 및 2,801개 녹음의 전체 full-window pooled 64D 임베딩을 그대로 재사용했다.
재학습·체크포인트 재선정·feature 기반 후보 선정은 하지 않았다.

같은 작품을 제외한 전체 2,779개를 먼저 검색하고 세 조건에서 각각 cosine 거리 최소 후보를 골랐다.

| 조건 | 연주자 | 작곡가 | 전체 순위 | AE cosine 거리 | 조건 후보 수 |
|---|---|---|---:|---:|---:|
{table}

같은 연주자는 artist_id로 판단하며 다른 작곡가도 허용한다. 같은 작곡가는 연주자를 제한하지 않는다.
같은 시대 조건은 이전 그림처럼 작곡가는 다르게 하되 기존 시대 범주가 하나 이상 겹쳐야 한다.
일부 후보는 조건 여러 개에 속할 수 있다. 선택한 세 녹음은 서로 다르다.
전체 후보 수와 조건부 후보 수가 다르므로 각 그룹의 최소 거리를 이용해 메타데이터 영향의 크기를 인과적으로 비교할 수는 없다.

왼쪽 PCA는 전체 2,801개 AE 벡터로 한 번 계산했다. A의 벡터는 앞 그림의 원본을 그대로 사용한다.
설명 분산은 {variance.sum():.1%}이며 A와 세 후보 주변을 확대했다. 회색 배경은 같은 작품을 제외한 후보들이다.
배경 중 확대 범위 밖의 점은 보이지 않는다. 세 후보만으로 PCA를 다시 학습하지 않았다.
오른쪽 거리와 전체 순위는 투영 전 원래 64차원 기준이다.

이 기준 A에서는 같은 작곡가·같은 시대의 최근접 후보가 같은 연주자의 최근접 후보보다 AE상 가깝다.
단일 기준 사례이며 전체 데이터의 경향이나 청취 유사성 검증은 아니다.
기존 work_id는 악장 단위여서 같은 작곡가 후보는 《Kreisleriana》의 다른 악장(2번)이다.
train/validation/test를 합친 카탈로그 검색이며 독립적인 heldout 성능 평가가 아니다.
시대 분류는 앞선 cross_work_checkpoint/protocol.json과 동일하며 베토벤 등은 겹치는 범주를 허용한다.

검증: 모든 2,779개 저장 거리를 원본 벡터로 재계산, 전체 순위 정렬과 조건별 최소값,
선정 3개 벡터의 scalar dot product, 같은 A·체크포인트 hash를 확인했다.

PNG / SVG / PDF는 ae_metadata_comparison 이름으로 저장했다. selected.csv는 선택된 후보와
연주자·작곡가·시대 일치 여부, projection.npz는 좌표·선택 벡터, provenance.json은 입력과 검증을 담는다.

```sh
capstone-2026-2-classicfy/.venv/bin/python presentation/create_ae_metadata_comparison.py
```
'''
    (OUT / 'README.md').write_text(note)
    print(json.dumps({'selected': chosen, 'PCA_variance': float(variance.sum()),
                      'figure': str(OUT / 'ae_metadata_comparison.png')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
