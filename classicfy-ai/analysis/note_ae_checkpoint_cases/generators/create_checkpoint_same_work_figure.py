"""One fixed same-work example from a snapshot of the running full-corpus AE.

Run with capstone-2026-2-classicfy/.venv/bin/python. Never modifies training
inputs/checkpoints. Reuses its local snapshot on rerun; --checkpoint selects a
different completed checkpoint into a NEW --out directory.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'capstone-2026-2-classicfy/classicfy-ai/src'))
from embedding.note_ae_indexed_data import IndexedNoteAECorpus
from embedding.note_ae_training import load_note_ae_model
from embedding.note_performance_data import load_note_performance

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/classicfy-presentation-mpl')
from create_composer_metadata_figure import setup_style, TEXT, MUTED
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CORPUS = ROOT / 'diagnostics/full_corpus_reconstruction/run01/corpus'
PROFILES = ROOT / 'diagnostics/work_relative_5d/run01/eligible.json'
FEATURES = ['tempo', 'rubato', 'dynamics', 'articulation', 'pedal']
COLORS = ['#2878C8', '#148766', '#D66B48']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def select(anchor, out, metadata=None, unique_artists=False):
    rows = json.loads(PROFILES.read_text())
    x = np.array([r['summary'] for r in rows], dtype=float)
    groups = np.array([r['cohort_id'] for r in rows])
    medians = {g: np.median(x[groups == g], axis=0) for g in sorted(set(groups))}
    residual = x - np.stack([medians[g] for g in groups])
    train = np.array([r['split'] == 'train' for r in rows])
    scale = residual[train].std(0)
    assert np.isfinite(x).all() and np.all(scale > 0)
    profiles = residual / scale
    a = next(i for i, r in enumerate(rows) if r['performance_id'] == anchor)
    cohort = [i for i, r in enumerate(rows) if r['cohort_id'] == rows[a]['cohort_id']]
    original_cohort = cohort.copy()
    if unique_artists:
        # Choose representatives by ID only, preserving A for its artist.
        # Neither feature distances nor embedding values enter this step.
        representatives = {metadata[rows[a]['record_key']]['artist_id']: a}
        for i in sorted(cohort, key=lambda i: rows[i]['performance_id']):
            representatives.setdefault(metadata[rows[i]['record_key']]['artist_id'], i)
        cohort = sorted(representatives.values(), key=lambda i: rows[i]['performance_id'])
        assert len(cohort) == len({metadata[rows[i]['record_key']]['artist_id'] for i in cohort})
    distances = np.sqrt(np.mean((profiles - profiles[a]) ** 2, axis=1))
    others = sorted((i for i in cohort if i != a),
                    key=lambda i: (distances[i], rows[i]['performance_id']))
    b, c = others[0], others[-1]
    order = [a, b, c] + [i for i in cohort if i not in (a, b, c)]
    chosen = [rows[i] for i in order]
    # Persist feature-only selection BEFORE loading any model or embedding.
    selection = {'anchor_policy': 'fixed previous example ATEPP:00957; no embedding search',
                 'anchor': anchor, 'B': rows[b]['performance_id'], 'C': rows[c]['performance_id'],
                 'policy': 'B/C = minimum/maximum equal-weight RMS 5D distance within same work/score/route cohort',
                 'candidate_count': len(cohort), 'candidate_ids': [r['performance_id'] for r in chosen],
                 'feature_scale': scale.tolist(), 'scale_training_recordings': int(train.sum()),
                 'profiles_source_sha256': sha(PROFILES), 'selection_uses_embedding': False}
    if unique_artists:
        selection.update(unique_artists=True, original_recording_count=len(original_cohort),
                         representative_policy='preserve A; for each other artist_id keep minimum performance_id, before distances',
                         excluded_ids=[rows[i]['performance_id'] for i in original_cohort if i not in cohort],
                         profile_statistics='reuse original cohort medians and train SD; deduplicate displayed/search candidates only')
    if (out / 'selection.json').exists():
        assert json.loads((out / 'selection.json').read_text()) == selection
    else:
        write(out / 'selection.json', selection)
    return chosen, profiles[order], distances[order], selection


def encode(rows, checkpoint, out):
    torch.set_num_threads(1)
    corpus = IndexedNoteAECorpus(CORPUS)
    model, payload, _, _ = load_note_ae_model(checkpoint, corpus=corpus)
    items = list(corpus.items.values())
    ids = {r['performance_id'] for r in rows}
    by_id = {key: sorted([v for v in items if v['window']['performance_id'] == key],
                         key=lambda v: v['window']['q_start']) for key in ids}
    embeddings, counts, per_window, window_ids = [], [], [], []
    with torch.inference_mode():
        for row in rows:
            windows = by_id[row['performance_id']]
            assert windows and all(w['window']['split'] == row['split'] for w in windows)
            # Independently verify saved descriptive summaries against source R.
            raw = Path(row['performance_cache'])
            assert sha(raw / 'performance.npz') == row['R_npz_sha']
            assert sha(raw / 'manifest.json') == row['R_manifest_sha']
            r = load_note_performance(raw)
            tempo = r.beats.log2_qpm[r.beats.valid]
            attack = np.array([bool(m['is_attack']) and not m['is_grace'] for m in r.metadata])
            summary = [np.median(tempo), np.median(np.abs(tempo - np.median(tempo)))]
            summary += [r.values[r.mask[:, j] & attack, j].mean() for j in (2, 3, 4)]
            np.testing.assert_allclose(summary, row['summary'], rtol=1e-12, atol=1e-12)
            es = []
            for j in range(0, len(windows), 8):
                es.extend(model.encode_batch(corpus.collate(windows[j:j + 8]))['e'].numpy())
            es = np.asarray(es, dtype=float)
            pooled = es.mean(0)
            assert np.linalg.norm(pooled) > 0 and np.isfinite(es).all()
            embeddings.append(pooled / np.linalg.norm(pooled))
            counts.append(len(windows))
            per_window.extend(es)
            window_ids.extend(w['window']['window_id'] for w in windows)
            print('ENCODED', row['performance_id'], len(windows), 'windows', flush=True)
    np.savez_compressed(out / 'window_embeddings.npz', e=np.asarray(per_window),
                        window_id=np.array(window_ids), counts=np.array(counts))
    return np.asarray(embeddings), counts, payload


def render(out, rows, x, d5, e, counts, epoch, metadata, unique_artists=False):
    setup_style()
    de = np.clip(1 - e @ e[0], 0, 2)
    centered = e - e.mean(0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    for v in vt[:2]:
        if v[np.argmax(np.abs(v))] < 0:
            v *= -1
    xy = centered @ vt[:2].T
    variance = s[:2] ** 2 / (s ** 2).sum()
    agrees = bool(de[1] < de[2])
    fig = plt.figure(figsize=(14, 7.5), facecolor='white')
    fig.text(.055, .93, '같은 작품, 서로 다른 연주', fontsize=27, weight='bold')
    fig.text(.055, .884, '슈만  Kreisleriana, Op.16 · VI. Sehr langsam', fontsize=14, color=MUTED)
    fig.text(.945, .932, f'AtEPP AE · epoch {epoch}', ha='right', fontsize=11, color=MUTED)
    scope_label = f'같은 작품 · {len(rows)}명 · 연주자당 1개' if unique_artists else f'기준 연주 1개 · 같은 작품 {len(rows)}개 연주'
    fig.text(.945, .889, scope_label, ha='right', fontsize=11, color=MUTED)
    fig.text(.055, .79, '학습된 연주 임베딩', fontsize=17, weight='bold')
    point_label = '연주자 중복 제거 · 점 하나 = 대표 녹음 하나' if unique_artists else '점 하나 = 전곡 연주 하나 · A / B / C 강조'
    fig.text(.055, .751, point_label, fontsize=11, color=MUTED)
    fig.text(.55, .79, '작품 기준의 상대적 5개 feature', fontsize=17, weight='bold')
    fig.text(.55, .751, '0 = 작품 중앙값 · 부호와 차이의 크기를 함께 비교', fontsize=11, color=MUTED)

    ax = fig.add_axes([.085, .34, .36, .36])
    ax.scatter(xy[3:, 0], xy[3:, 1], s=44, color='#CDD5DF', edgecolor='white', lw=.8, zorder=2)
    for i in (1, 2):
        ax.plot(xy[[0, i], 0], xy[[0, i], 1], color=COLORS[i],
                lw=1.6, ls='-' if i == 1 else '--', alpha=.75, zorder=1)
    for i in range(3):
        ax.scatter(*xy[i], s=360 if i == 0 else 100, marker='*' if i == 0 else 'o',
                   facecolor='none' if i == 0 else COLORS[i],
                   edgecolor=COLORS[0] if i == 0 else 'white',
                   lw=1.8 if i == 0 else 1.2, zorder=6 if i == 0 else 4)
        # Separate the very close A/B labels with short leaders.
        offset = [(-22, 19), (20, -23), (10, 12)][i]
        ax.annotate('ABC'[i], xy[i], xytext=offset, textcoords='offset points',
                    color=COLORS[i], fontsize=14, weight='bold', ha='center',
                    arrowprops={'arrowstyle': '-', 'color': COLORS[i], 'lw': .8})
    ax.set_xlabel(f'PC1 ({variance[0]:.1%})', fontsize=10)
    ax.set_ylabel(f'PC2 ({variance[1]:.1%})', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.margins(.23)
    ax.grid(color='#EBEFF4', lw=.7); ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.055, .222, '원래 64D cosine 거리', fontsize=11, color=MUTED)
    fig.text(.24, .222, f'A–B  {de[1]:.4f}', color=COLORS[1], fontsize=13, weight='bold')
    fig.text(.37, .222, f'A–C  {de[2]:.4f}', color=COLORS[2], fontsize=13, weight='bold')

    axf = fig.add_axes([.585, .32, .36, .38])
    pos = np.arange(5)
    for i in range(3):
        axf.bar(pos + (i - 1) * .235, x[i], width=.21, color=COLORS[i], zorder=3)
    axf.axhline(0, color='#8F9AA9', lw=1)
    axf.set_xticks(pos, ['빠르기', '루바토', '강약', '음 유지', '페달'], fontsize=12)
    axf.set_ylabel('작품 중앙값 대비 편차 / train 표준편차', fontsize=10)
    axf.tick_params(axis='x', length=0, pad=10)
    axf.tick_params(axis='y', labelsize=9, length=0)
    axf.grid(axis='y', color='#EBEFF4'); axf.set_axisbelow(True)
    axf.margins(y=.16)
    for spine in axf.spines.values():
        spine.set_visible(False)
    fig.text(.55, .222, '5D feature 거리', fontsize=11, color=MUTED)
    fig.text(.72, .222, f'A–B  {d5[1]:.3f}', color=COLORS[1], fontsize=13, weight='bold')
    fig.text(.85, .222, f'A–C  {d5[2]:.3f}', color=COLORS[2], fontsize=13, weight='bold')

    legend_labels = ['A  기준', 'B  feature 최근접', 'C  feature 최원거리']
    handles = [Line2D([], [], marker='*' if i == 0 else 'o', color=COLORS[i], ls='',
                      markersize=12 if i == 0 else 9,
                      label=f"{legend_labels[i]}  ·  {metadata[rows[i]['record_key']]['artist']} ({rows[i]['record_key']})")
               for i in range(3)]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.048, .137),
               ncol=3, frameon=False, fontsize=11, columnspacing=2)
    conclusion = ('feature가 가까운 B는 임베딩에서도 A에 더 가깝다.' if agrees else
                  '이 기준 연주에서는 feature의 가까움이 임베딩 거리 순서와 일치하지 않는다.')
    fig.text(.055, .092, conclusion, fontsize=15, weight='bold')
    fig.text(.055, .044,
             f'B·C는 임베딩을 보기 전 feature 거리로 선정. PCA는 같은 작품 {len(rows)}개로 계산 ({variance.sum():.1%} 분산). 한 기준 연주의 사례.',
             fontsize=9.5, color=MUTED)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out / f'same_work_checkpoint.{ext}', dpi=200, facecolor='white')
    plt.close(fig)
    return de, xy, variance, agrees


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'diagnostics/full_corpus_reconstruction/run01/training/best.pt')
    parser.add_argument('--out', type=Path, default=ROOT / 'presentation/figures/same_work_checkpoint')
    parser.add_argument('--unique-artists', action='store_true', help='Keep A and the lowest-ID recording of every other artist; reuse original profile statistics')
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    metadata = {r['perf_id']: r for r in csv.DictReader(
        (ROOT / 'datasets/ATEPP/ATEPP-metadata-1.2.csv').open(encoding='utf-8-sig'))}
    rows, x, d5, selection = select('ATEPP:00957', out, metadata, args.unique_artists)
    snapshot = out / 'checkpoint.pt'
    if not snapshot.exists():
        before = sha(args.checkpoint)
        temporary = out / '.checkpoint.tmp'
        shutil.copyfile(args.checkpoint, temporary)
        assert before == sha(temporary) == sha(args.checkpoint), 'Retry after checkpoint write completes'
        temporary.replace(snapshot)
    snapshot_sha = sha(snapshot)
    e, counts, payload = encode(rows, snapshot, out)
    epoch = payload['epoch']
    de, xy, variance, agrees = render(out, rows, x, d5, e, counts, epoch, metadata, args.unique_artists)
    exported = []
    for i, row in enumerate(rows):
        exported.append({'label': 'ABC'[i] if i < 3 else '',
                         'performance_id': row['performance_id'], 'recording_id': row['recording_id'],
                         'artist': metadata[row['record_key']]['artist'], 'work': row['work_key'],
                         'feature_distance_to_A': float(d5[i]), 'cosine_distance_to_A': float(de[i]),
                         'windows': counts[i], 'pc1': float(xy[i, 0]), 'pc2': float(xy[i, 1]),
                         **dict(zip(FEATURES, map(float, x[i])))})
    with (out / 'performances.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0])); writer.writeheader(); writer.writerows(exported)
    np.savez_compressed(out / 'example.npz', performance_id=np.array([r['performance_id'] for r in rows]),
                        feature_profiles=x, embedding=e, xy=xy, cosine_distance_to_A=de)
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1, atol=1e-12)
    assert len({r['recording_id'] for r in rows}) == len(rows)
    assert len({r['cohort_id'] for r in rows}) == 1
    assert d5[1] == min(d5[1:]) and d5[2] == max(d5[1:])
    assert len(rows) == selection['candidate_count'] and sha(snapshot) == snapshot_sha
    if args.unique_artists:
        assert len({metadata[r['record_key']]['artist_id'] for r in rows}) == len(rows)
    provenance = {'checkpoint_source': str(args.checkpoint), 'snapshot_sha256': snapshot_sha,
                  'checkpoint_epoch': epoch, 'best_validation': payload['best_validation'],
                  'checkpoint_policy': 'snapshot of running full-corpus best.pt at invocation; reruns reuse snapshot',
                  'corpus_manifest_sha256': sha(CORPUS / 'manifest.json'),
                  'training_scope': json.loads((CORPUS.parent / 'protocol.json').read_text())['scope'],
                  'selection': selection, 'examples': exported[:3],
                  'embedding': 'equal-window mean of unit e over every full 8-bar window, then L2 normalized; tail excluded',
                  'profile': 'whole-record summary minus same work/score/route median, divided by existing eligible train residual SD',
                  'profile_definitions': ['median valid beat log2 QPM', 'median absolute deviation from own median log2 QPM',
                                          'mean attack velocity_unit', 'mean attack articulation_log2_ratio', 'mean attack sustain_onset_unit'],
                  'PCA_variance': variance.tolist(), 'distance_order_agrees': agrees,
                  'verification': {'passed': True, 'checkpoint_corpus_contract': True,
                                   'source_R_hashes_and_recomputed_summaries': len(rows),
                                   'all_cohort_candidates_plotted': True, 'unit_pooled_embeddings': True,
                                   'distinct_recordings': True, 'feature_only_extrema': True},
                  'limits': ['one fixed anchor, not dataset-wide performance', 'no listening evaluation',
                             'whole-record feature vs full-window-only embedding coverage differs',
                             'one point per recording does not measure repeated-sample cluster separation',
                             'descriptive heldout cohort medians use the collected heldout recordings']}
    write(out / 'provenance.json', provenance)
    if args.unique_artists:
        provenance['verification']['one_recording_per_artist_id'] = True
        write(out / 'provenance.json', provenance)
    note = f'''# 같은 작품의 연주 차이: 전체 적격 AtEPP 학습 중간 체크포인트

![figure](same_work_checkpoint.png)

PNG / SVG / PDF를 같은 이름으로 저장했다. 기준 A는 이전 사례의 Horowitz `ATEPP:00957`로 고정했다.
같은 작품·악보·반복 경로의 적격 {len(rows)}개 녹음을 모두 표시한다. B는 5D feature 최근접, C는 최원거리이며
선정 결과를 `selection.json`에 기록한 다음 모델을 읽었다. 임베딩 결과에 맞춰 사례나 B/C를 바꾸지 않았다.

전체 적격 AtEPP 2,801개 녹음 / 254작품 중 train 2,070개 / 185작품으로 학습 중인 모델의
**epoch {epoch} best validation 체크포인트**를 별도로 복사해 CPU에서 이 작품만 추론했다.
학습은 중단하거나 수정하지 않았다. 원본 전체 AtEPP가 모두 학습에 쓰였다는 의미는 아니다.
`checkpoint.pt`를 고정하므로 이후 학습이 진행되어도 재실행 결과를 유지한다.

| 대상 | 연주자 · ID | feature 거리 | 64D cosine 거리 |
|---|---|---:|---:|
'''
    for r in exported[:3]:
        note += f"| {r['label']} | {r['artist']} · {r['performance_id']} | {r['feature_distance_to_A']:.3f} | {r['cosine_distance_to_A']:.4f} |\n"
    note += f'''
거리 순서 일치: **{agrees}**. 왼쪽 PCA는 {len(rows)}개 전부로 계산했고 설명 분산은 {variance.sum():.1%}다.
판단은 2D 점 간격 대신 원래 64D cosine 거리를 사용한다. 오른쪽 막대는 부호 있는 상대 프로필이다.
프로필 정의와 train 스케일은 기존 `diagnostics/work_relative_5d/run01`과 같다. 0은 작품 cohort 중앙값이며,
작품별 SD로 다시 나누지 않았다. 5D 거리는 다섯 축 차이의 RMS다. 페달은 발음 시점 sustain 깊이만 나타낸다.

각 녹음의 full 8마디 unit `e`를 균등 평균한 뒤 L2 정규화했다. feature는 전곡 요약이라 마지막 tail의
포함 범위가 다르다. 이 그림은 기준 연주 하나의 표현 차이 사례이며, 전수 성능·청취 유사성·연주별 군집 분리의
검증 결과가 아니다. 다른 작품 탐색 주장은 별도 사례가 필요하다.

검증: 체크포인트와 corpus 계약·입력 tensor hash, {len(rows)}개 원본 R hash와 feature 재계산 일치,
고유 녹음·동일 cohort, 모든 full window 사용, pooled 벡터 단위 길이, B/C feature 최소·최대 여부.

재현:
```sh
capstone-2026-2-classicfy/.venv/bin/python presentation/create_checkpoint_same_work_figure.py
```
새 체크포인트는 `--checkpoint PATH --out NEW_DIRECTORY`로 지정한다. `performances.csv`에는 전체 후보와
거리·feature·PCA 좌표, `example.npz`에는 원래 벡터, `window_embeddings.npz`에는 구간 벡터가 있다.
`provenance.json`에는 체크포인트 hash·epoch·검증·제약을 기록했다.
'''
    if args.unique_artists:
        note += f'''
## 연주자 중복 제거 버전

기존 {selection['original_recording_count']}개 녹음에서 **연주자당 1개, 총 {len(rows)}명**으로 줄였다.
artist_id로 중복을 판별한다. Horowitz는 기준 A(00957)를 유지하고, 다른 연주자는 performance_id가
가장 작은 녹음을 유지했다. 대표 선정에 feature나 임베딩 거리를 사용하지 않았다.
중복 제거 후 B/C를 feature 최소/최대 거리로 다시 선정하고, 남은 전체 벡터로 PCA를 다시 계산했다.
원래 작품 중앙값과 train scale은 유지했으므로 A와 남은 녹음의 feature 값·64D 벡터는 이전과 같다.
이 버전에서 '전체 후보'는 중복 제거 후의 {len(rows)}개 대표 녹음을 뜻한다.
앞 그림과 동일한 epoch 8 checkpoint.pt를 별도 출력 폴더로 복사해 사용했다.

재현:
```sh
capstone-2026-2-classicfy/.venv/bin/python presentation/create_checkpoint_same_work_figure.py --unique-artists --checkpoint presentation/figures/same_work_checkpoint/checkpoint.pt --out presentation/figures/same_work_checkpoint_unique_artists
```
'''
    (out / 'README.md').write_text(note)
    print(json.dumps({'epoch': epoch, 'examples': exported[:3], 'distance_order_agrees': agrees,
                      'PCA_variance': float(variance.sum()), 'figure': str(out / 'same_work_checkpoint.png')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
