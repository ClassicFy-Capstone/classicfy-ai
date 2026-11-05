"""Rank the entire available corpus for ONE fixed anchor, then select group minima.

CPU-only frozen inference uses the exact epoch-8 snapshot from the same-work
figure. Chunk caches are tied to checkpoint/corpus hashes and resume safely.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import textwrap

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'capstone-2026-2-classicfy/classicfy-ai/src'))
from embedding.note_ae_indexed_data import IndexedNoteAECorpus
from embedding.note_ae_training import load_note_ae_model
from embedding.note_performance_data import load_note_performance

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/classicfy-presentation-mpl')
from create_composer_metadata_figure import setup_style, box, TEXT, MUTED
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = ROOT / 'presentation/figures/cross_work_checkpoint'
PREVIOUS = ROOT / 'presentation/figures/same_work_checkpoint'
CHECKPOINT = PREVIOUS / 'checkpoint.pt'
CORPUS = ROOT / 'diagnostics/full_corpus_reconstruction/run01/corpus'
PROFILES = ROOT / 'diagnostics/work_relative_5d/run01/eligible.json'
FEATURES = ['tempo', 'rubato', 'dynamics', 'articulation', 'pedal']
# Overlapping eras make the different-era group conservative. In particular,
# Beethoven is NOT treated as unambiguously outside Schumann's Romantic era.
ERAS = {
    'Johann Sebastian Bach': ['Baroque'], 'Domenico Scarlatti': ['Baroque'],
    'Franz Joseph Haydn': ['Classical'], 'Wolfgang Amadeus Mozart': ['Classical'],
    'Ludwig van Beethoven': ['Classical', 'Romantic'],
    'Franz Schubert': ['Classical', 'Romantic'], 'Robert Schumann': ['Romantic'],
    'Franz Liszt': ['Romantic'], 'Johannes Brahms': ['Romantic'],
    'Sergei Rachmaninoff': ['Romantic', 'Modern'],
    'Alexander Scriabin': ['Romantic', 'Modern'],
    'Claude Debussy': ['Modern'], 'Maurice Ravel': ['Modern'],
}
ERA_LABELS = {'Baroque': '바로크', 'Classical': '고전', 'Romantic': '낭만', 'Modern': '근대'}
GROUPS = ['same_composer', 'same_era_other_composer', 'different_era_other_composer']
GROUP_LABELS = ['같은 작곡가 · 다른 작품', '같은 시대 · 다른 작곡가', '다른 시대 · 다른 작곡가']
COLORS = ['#7E67AC', '#D29B38', '#148766']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def metadata_group(composer):
    if composer == 'Robert Schumann':
        return GROUPS[0]
    return GROUPS[1] if set(ERAS[composer]) & set(ERAS['Robert Schumann']) else GROUPS[2]


def infer(corpus, protocol):
    torch.set_num_threads(1)
    model, payload, _, _ = load_note_ae_model(CHECKPOINT, corpus=corpus)
    assert payload['epoch'] == 8
    items = sorted(corpus.items.values(), key=lambda w: (w['window']['N'], w['window']['window_id']))
    chunks = OUT / 'chunks'; chunks.mkdir(exist_ok=True)
    vectors = np.zeros((len(items), 64), dtype=np.float32)
    started = time.perf_counter()
    block = 1024
    with torch.inference_mode():
        for start in range(0, len(items), block):
            stop = min(start + block, len(items))
            path = chunks / f'{start:06d}.npz'
            ids = np.array([w['window']['window_id'] for w in items[start:stop]])
            if path.exists():
                with np.load(path, allow_pickle=False) as f:
                    assert str(f['protocol_sha']) == protocol['id']
                    np.testing.assert_array_equal(f['window_id'], ids)
                    es = f['e'].copy()
            else:
                es = []
                for j in range(start, stop, 32):
                    batch = corpus.collate(items[j:min(j + 32, stop)])
                    es.extend(model.encode_batch(batch)['e'].numpy())
                es = np.asarray(es)
                temporary = path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, e=es, window_id=ids, protocol_sha=np.array(protocol['id']))
                temporary.replace(path)
            assert es.shape == (stop - start, 64) and np.isfinite(es).all()
            vectors[start:stop] = es
            write(OUT / 'progress.json', {'windows': stop, 'total': len(items),
                                         'seconds': round(time.perf_counter() - started, 1)})
            print('ENCODED', stop, '/', len(items), 'seconds', round(time.perf_counter() - started, 1), flush=True)
    grouped = {}
    for i, w in enumerate(items):
        grouped.setdefault(w['window']['performance_id'], []).append(i)
    pooled, rows, counts = [], [], []
    for key in sorted(grouped):
        idx = sorted(grouped[key], key=lambda i: items[i]['window']['q_start'])
        v = vectors[idx].astype(float).mean(0)
        assert np.linalg.norm(v) > 0
        pooled.append(v / np.linalg.norm(v))
        rows.append(items[idx[0]])
        counts.append(len(idx))
    np.savez_compressed(OUT / 'catalogue_embeddings.npz',
                        performance_id=np.array([w['window']['performance_id'] for w in rows]),
                        e=np.array(pooled), counts=np.array(counts), protocol_sha=np.array(protocol['id']))
    return rows, np.asarray(pooled), counts


def select(rows, e, counts, metadata):
    a = next(i for i, w in enumerate(rows) if w['window']['performance_id'] == 'ATEPP:00957')
    old = np.load(PREVIOUS / 'example.npz', allow_pickle=False)
    np.testing.assert_allclose(e[a], old['embedding'][0], atol=2e-6, rtol=2e-6)
    distances = np.clip(1 - e @ old['embedding'][0], 0, 2)
    anchor = rows[a]['window']
    pool = [i for i, w in enumerate(rows) if w['window']['work_id'] != anchor['work_id']
            and w['window']['recording_id'] != anchor['recording_id']]
    order = sorted(pool, key=lambda i: (distances[i], rows[i]['window']['performance_id']))
    ranks = {i: j + 1 for j, i in enumerate(order)}
    selected = [next(i for i in order if metadata_group(metadata[rows[i]['record_key']]['composer']) == g)
                for g in GROUPS]
    output = []
    for i in [a] + order:
        w, m = rows[i]['window'], metadata[rows[i]['record_key']]
        output.append({'performance_id': w['performance_id'], 'record_key': rows[i]['record_key'],
                       'recording_id': w['recording_id'], 'work': w['work_id'], 'split': w['split'],
                       'artist': m['artist'], 'composer': m['composer'], 'title': m['track'],
                       'era': ' / '.join(ERA_LABELS[p] for p in ERAS[m['composer']]),
                       'group': 'anchor' if i == a else metadata_group(m['composer']),
                       'global_rank': ranks.get(i, 0), 'cosine_distance': float(distances[i]),
                       'windows': counts[i], 'selected': i in selected})
    with (OUT / 'all_ranked_candidates.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    by_id = {r['performance_id']: r for r in output}
    chosen = [by_id[rows[i]['window']['performance_id']] for i in [a] + selected]
    result = {'anchor': chosen[0], 'selected': chosen[1:], 'candidate_count': len(order),
              'candidate_works': len({rows[i]['window']['work_id'] for i in order}),
              'all_recordings': len(rows), 'excluded_same_work_count': len(rows) - len(order),
              'candidate_splits': {s: sum(r['split'] == s for r in output[1:]) for s in ['train', 'validation', 'test']},
              'group_sizes': {g: sum(r['group'] == g for r in output[1:]) for g in GROUPS},
              'selection': 'rank ALL different-work recordings first; take first member of each metadata group',
              'anchor_embedding_reused_from_previous_figure': True,
              'anchor_fresh_inference_matches_previous': True,
              'work_definition': 'existing movement-level work_id; other movements of same cycle remain candidates'}
    write(OUT / 'retrieval.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def profiles(result, corpus):
    rows = json.loads(PROFILES.read_text())
    raw = np.array([r['summary'] for r in rows], dtype=float)
    groups = np.array([r['cohort_id'] for r in rows])
    medians = {g: np.median(raw[groups == g], axis=0) for g in set(groups)}
    residual = raw - np.stack([medians[g] for g in groups])
    scale = residual[[r['split'] == 'train' for r in rows]].std(0)
    np.testing.assert_allclose(scale, json.loads((PREVIOUS / 'selection.json').read_text())['feature_scale'], atol=0, rtol=0)
    x = residual / scale
    by_id = {r['performance_id']: i for i, r in enumerate(rows)}
    selected = [result['anchor']] + result['selected']
    anchor_i = by_id[selected[0]['performance_id']]
    np.testing.assert_array_equal(x[anchor_i], np.load(PREVIOUS / 'example.npz')['feature_profiles'][0])
    d5 = np.sqrt(np.mean((x - x[anchor_i]) ** 2, axis=1))
    feature_pool = [i for i, r in enumerate(rows) if r['work_key'] != selected[0]['work']]
    feature_order = sorted(feature_pool, key=lambda i: (d5[i], rows[i]['performance_id']))
    feature_ranks = {i: j + 1 for j, i in enumerate(feature_order)}
    index = json.loads(Path(corpus.manifest['source_index']).read_text())
    current = {r['performance_id']: r for r in index['sources']}
    def summarize(perf):
        tempo = perf.beats.log2_qpm[perf.beats.valid]
        attack = np.array([bool(m['is_attack']) and not m['is_grace'] for m in perf.metadata])
        summary = [np.median(tempo), np.median(np.abs(tempo - np.median(tempo)))]
        coverage = [perf.beats.valid.mean()]
        observed_counts = [len(tempo)]
        for k in (2, 3, 4):
            valid = perf.mask[:, k] & attack
            summary.append(perf.values[valid, k].mean())
            coverage.append(valid.sum() / attack.sum())
            observed_counts.append(valid.sum())
        return np.asarray(summary), min(coverage) >= .8 and min(observed_counts) >= 8

    checks = []
    for r in selected:
        # Current R has new provenance IDs; verify the actual feature values,
        # score and route against the old source used by the first figure.
        source = current[r['performance_id']]
        perf = load_note_performance(source['performance_cache'])
        summary, quality_ok = summarize(perf)
        assert quality_ok
        i = by_id.get(r['performance_id'])
        if i is not None:
            np.testing.assert_allclose(summary, raw[i], rtol=1e-12, atol=1e-12)
            assert perf.score_source_sha256 == rows[i]['score_sha'] and perf.traversal_id == rows[i]['traversal']
            profile, cohort_count = x[i], rows[i]['cohort_count']
        else:
            # Keep the actual retrieval minimum. Its profile uses the SAME
            # formula and train scale but is explicitly provisional below n=10.
            summaries, recordings = [], set()
            for s in index['sources']:
                if s['work_key'] != r['work'] or s['recording_id'] in recordings:
                    continue
                p = load_note_performance(s['performance_cache'])
                if (p.score_source_sha256, p.traversal_id) != (perf.score_source_sha256, perf.traversal_id):
                    continue
                values, ok = summarize(p)
                if ok:
                    summaries.append(values); recordings.add(s['recording_id'])
            cohort_count = len(summaries)
            assert 1 < cohort_count < 10
            profile = (summary - np.median(summaries, axis=0)) / scale
        rank = feature_ranks.get(i) if i is not None else None
        r.update(feature_distance=float(np.sqrt(np.mean((profile - x[anchor_i]) ** 2))), feature_rank=rank,
                 feature_pool_count=len(feature_pool),
                 feature_nearest_fraction=rank / len(feature_pool) if rank else None,
                 feature_profile=profile.tolist(), cohort_recordings=cohort_count,
                 feature_profile_provisional=cohort_count < 10)
        checks.append({'performance_id': r['performance_id'],
                       'current_R_sha256': sha(Path(source['performance_cache']) / 'performance.npz'),
                       'current_summary_matches_previous_feature_source': i is not None,
                       'current_summary_recomputed': True, 'cohort_count': cohort_count})
    result['feature_verification'] = checks
    result['feature_pool_count'] = len(feature_pool)
    result['feature_scale'] = scale.tolist()
    write(OUT / 'retrieval.json', result)
    return np.array([r['feature_profile'] for r in selected])


def render(result, x):
    setup_style()
    n = result['candidate_count']
    chosen = result['selected']
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.text(.045, .935, '다른 작품으로 이어지는 연주 탐색', fontsize=28, weight='bold')
    fig.text(.955, .94, 'AtEPP AE · epoch 8 · 같은 기준 A', ha='right', fontsize=11, color=MUTED)
    fig.text(.045, .887,
             f"기준 A  ·  Vladimir Horowitz  ·  슈만 Kreisleriana, Op.16: 6. Sehr langsam  ·  낭만", fontsize=13, color='#2878C8')
    fig.text(.045, .837,
             f"같은 작품 제외 → 전체 {n:,}개 연주 / {result['candidate_works']}작품 검색 → 각 메타데이터 그룹의 임베딩 최근접 1개",
             fontsize=12, color=MUTED)
    low, high = min(-.15, x.min() - .45), max(.15, x.max() + .45)
    for j, r in enumerate(chosen):
        left = .045 + j * .313
        color = COLORS[j]
        box(fig, left, .505, .284, .286, '#F6F8FB', radius=.009)
        fig.text(left + .015, .754, GROUP_LABELS[j], color=color, fontsize=14, weight='bold')
        fig.text(left + .015, .699, f"전체 {r['global_rank']:,}위", fontsize=25, color=color, weight='bold')
        fig.text(left + .015, .665,
                 f"64D cosine 거리  {r['cosine_distance']:.4f}  ·  상위 {r['global_rank']/n:.2%}",
                 fontsize=10.5, color=MUTED)
        fig.text(left + .015, .623, r['composer'], fontsize=12.5, weight='bold')
        fig.text(left + .015, .594, f"{r['era']}  ·  {r['artist']}", fontsize=10.5, color=MUTED)
        short_titles = {'ATEPP:01008': 'Kreisleriana, Op.16\nII. Sehr innig und nicht zu rasch',
                        'ATEPP:00315': 'Harmonies poétiques et religieuses, S.173\nVII. Funérailles',
                        'ATEPP:03082': 'Suite bergamasque, L.75\nI. Prélude'}
        title = short_titles.get(r['performance_id'], textwrap.fill(r['title'], width=44, break_long_words=False))
        fig.text(left + .015, .558, title, fontsize=10.5, linespacing=1.35, va='top')
        ax = fig.add_axes([left + .025, .235, .245, .215])
        pos = np.arange(5)
        ax.bar(pos - .16, x[0], width=.29, color='#2878C8', zorder=3)
        ax.bar(pos + .16, x[j + 1], width=.29, color=color, zorder=3)
        ax.axhline(0, color='#A1ADBA', lw=.8)
        ax.set_ylim(low, high)
        ax.set_xticks(pos, ['빠르기', '루바토', '강약', '음 유지', '페달'], fontsize=10)
        ax.tick_params(length=0, labelsize=9)
        ax.grid(axis='y', color='#E7ECF2'); ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if j == 0:
            ax.set_ylabel('작품 중앙값 대비 편차 / train 표준편차', fontsize=9)
        ax.legend(handles=[Line2D([], [], color='#2878C8', lw=5, label='기준 A'),
                           Line2D([], [], color=color, lw=5, label='이 후보')],
                  frameon=False, ncol=2, fontsize=9, loc='lower left', bbox_to_anchor=(-.02, 1.03))
        fig.text(left + .015, .175, f"5D feature 거리  {r['feature_distance']:.3f}", fontsize=13, weight='bold', color=color)
        detail = (f"작품 기준 {r['cohort_recordings']}개 · 10개 미만의 참고 프로필" if r['feature_profile_provisional'] else
                  f"feature 근접 순위 {r['feature_rank']:,} / {r['feature_pool_count']:,}  ·  가까운 순")
        fig.text(left + .015, .143, detail, fontsize=10.5, color=MUTED)
    r = chosen[2]
    # Do not describe a feature-distant minimum as successful expression retrieval.
    if r['global_rank'] / n <= .1 and r['feature_nearest_fraction'] is not None and r['feature_nearest_fraction'] <= .2:
        conclusion = '작곡가·시대가 다른 후보도 전체 검색 상위에 있으며, 상대 feature 프로필도 가깝다.'
    elif r['global_rank'] / n <= .1:
        conclusion = f"작곡가·시대가 다른 후보가 전체 {r['global_rank']}위에 등장하지만, 상대 feature 프로필은 A와 멀다."
    else:
        conclusion = '이 기준 A에서는 작곡가·시대가 다른 후보의 전체 검색 순위가 낮다.'
    fig.text(.045, .082, conclusion, fontsize=15, weight='bold')
    fig.text(.045, .047,
             '순위는 그룹 필터 전 전체 검색 기준. 시대는 겹치는 범주 허용(베토벤: 고전/낭만). feature 순위는 10개 이상 기준 적격 후보에서 계산.',
             fontsize=9, color=MUTED)
    fig.text(.045, .023,
             'train/validation/test를 합친 카탈로그의 한 기준 연주 탐색 사례 · 작품은 악장 단위 · 전체 성능이나 청취 유사성의 검증은 아님.',
             fontsize=9, color=MUTED)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'cross_work_checkpoint.{ext}', dpi=200, facecolor='white')
    plt.close(fig)
    result['conclusion'] = conclusion


def document(result, protocol):
    selected = [result['anchor']] + result['selected']
    fields = ['performance_id', 'artist', 'composer', 'title', 'era', 'group', 'split', 'global_rank',
              'cosine_distance', 'feature_distance', 'feature_rank', 'feature_pool_count',
              'feature_nearest_fraction', 'cohort_recordings', 'feature_profile_provisional', *FEATURES]
    with (OUT / 'selected.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for r in selected:
            writer.writerow({**r, **dict(zip(FEATURES, r['feature_profile']))})
    np.savez_compressed(OUT / 'selected_profiles.npz',
                        performance_id=np.array([r['performance_id'] for r in selected]),
                        feature_profile=np.array([r['feature_profile'] for r in selected]))
    result['protocol_id'] = protocol['id']
    result['verification'] = {'passed': True, 'entire_2801_recording_corpus_encoded': True,
                              'checkpoint_and_corpus_hash_checked': True,
                              'checkpoint_corpus_scaler_contract_checked': True,
                              'same_anchor_and_same_checkpoint_as_previous_figure': True,
                              'every_full_window_used_once': True,
                              'metadata_filter_applied_after_global_ranking': True,
                              'three_true_group_minima': True,
                              'no_feature_based_candidate_substitution': True,
                              'selected_current_R_summaries_recomputed': True,
                              'previously_eligible_profiles_matched': True,
                              'low_count_profiles_marked_provisional': True}
    write(OUT / 'retrieval.json', result)
    table = '\n'.join(f"| {GROUP_LABELS[j]} | {r['composer']} · {r['artist']} · {r['performance_id']} | {r['global_rank']} | {r['cosine_distance']:.4f} | {r['feature_distance']:.3f} | " +
                      (f"참고 프로필 n={r['cohort_recordings']}" if r['feature_profile_provisional'] else f"{r['feature_rank']}/{r['feature_pool_count']}") + ' |'
                      for j, r in enumerate(result['selected']))
    note = f'''# 메타데이터 조건과 별개로 다른 작품을 검색하는 단일 기준 사례

![figure](cross_work_checkpoint.png)

## 검색 범위

앞 그림과 정확히 같은 A(Horowitz `ATEPP:00957`, 슈만 《Kreisleriana》 6번) 및 epoch 8 snapshot을 사용했다.
검증 후 적격 AtEPP 2,801개 녹음의 모든 67,542 full 8마디를 CPU에서 추론했다.
원본 AtEPP 전체가 아닌, 현재 전처리와 모델 입력이 준비된 corpus 전체다.
동일 work_id와 동일 recording_id를 제외한 **{result['candidate_count']:,}개 연주 / {result['candidate_works']}작품**을
작곡가·시대·feature 조건 없이 64D cosine 거리순으로 먼저 정렬했다.
그 뒤 각 메타데이터 그룹에서 처음 등장하는 후보를 선택했다. 세 후보 모두 그룹의 실제 임베딩 최근접이며
feature가 더 잘 맞는 후보로 교체하지 않았다. 동률은 performance_id 순이다.

| 그룹 | 연주 | 전체 순위 | 64D 거리 | 5D 거리 | feature 근접 순위 |
|---|---|---:|---:|---:|---:|
{table}

**{result['conclusion']}**

## 계산과 분류

- 구간 unit e를 전곡의 모든 full 8마디에 대해 균등 평균하고 L2 정규화했다. tail은 제외한다.
- 검색 A는 앞 그림에 저장된 원래 벡터를 그대로 사용했다. batch 크기를 바꾼 재추론과도 오차 2e-6 이내에서 일치했다.
- 5D는 앞 그림과 동일한 전곡 feature 정의·작품/악보/반복 경로 중앙값·train residual SD를 재사용했다.
  네 연주의 현재 run02 R에서 summary를 재계산했다. 기존 적격 프로필인 A와 드뷔시 후보는 앞 그림의 run01 값과 일치했다.
  슈만 2번과 리스트 후보는 기준 녹음이 각각 8개와 4개여서 기존 10개 이상 적격 프로필에는 없다.
  이 둘은 동일한 summary·cohort 중앙값·train scale 공식으로 참고 프로필을 계산하고 그림에 표본 수를 명시했다.
  검색 후보를 다른 연주로 교체하거나 기존 적격 기준을 변경하지 않았다.
- feature 순위는 상대 프로필 적격 **{result['feature_pool_count']:,}개 다른 작품 연주**에서 계산한다.
  전체 임베딩 검색 분모와 다르다. 표본 10개 미만인 두 참고 프로필에는 근접 순위를 부여하지 않았다.
  feature 거리와 64D 거리는 서로 다른 단위다.
- 시대는 설명용의 거친 작곡가 범주이며 corpus에 원래 있던 필드가 아니다. 겹치는 범주를 허용한다.
  베토벤·슈베르트는 고전/낭만, 라흐마니노프·스크랴빈은 낭만/근대로 처리했다.
  슈만의 낭만 범주와 하나라도 겹치면 같은 시대 그룹에 넣는다. 베토벤을 다른 시대로 세어 주장을 유리하게 만들지 않았다.
  매핑은 임베딩 추론 전에 protocol.json에 고정했다.
- 시대 구분 참고: [음악 시대 교육 자료](https://www.classicsforkids.com/musical-genres/),
  [Eastman: 베토벤의 과도기적 위치](https://www.esm.rochester.edu/beethoven/transitional-composer-and-heroic-objective/).
  겹침을 허용하는 구현은 이 사례의 명시적 분류 관례다.

## 해석 범위

이는 한 A가 전체 카탈로그를 검색하는 사례다. train/validation/test를 모두 포함하며 독립적인 heldout 성능 평가가 아니다.
각 split의 후보 수는 {result['candidate_splits']}다. A는 test 작품이다.
기존 work_id는 악장 단위이므로 《Kreisleriana》 다른 악장도 다른 작품 후보로 남긴다.
다른 시대 상위 후보가 나온 것만으로 음악적 유사성이나 metadata 영향의 제거를 증명하지 않는다.
프로필 순위는 별도의 수치적 비교이며 사람의 청취 판단은 검증하지 않았다.
figure의 '상위' 결론은 임베딩 순위 상위 10%, 프로필 가까움은 feature 순위 상위 20%라는
설명용 기준으로 작성했다. 순위·거리 원값을 함께 표시하며 유의성 검정이나 사전 등록한 성능 기준이 아니다.

## 산출물 / 재현

PNG / SVG / PDF: cross_work_checkpoint. 같은 이름의 확장자로 저장했다.
all_ranked_candidates.csv는 A와 전체 검색 순위, selected.csv는 선정된 연주와 5D 값이다.
catalogue_embeddings.npz에는 전체 2,801개 벡터, chunks/에는 재개 가능한 구간 캐시가 있다.
retrieval.json에 순위·후보수·검증, protocol.json에 입력 hash·시대 분류를 보존했다.

```sh
capstone-2026-2-classicfy/.venv/bin/python presentation/create_cross_work_checkpoint_figure.py
```
앞 그림의 checkpoint.pt를 사용하므로 진행 중인 학습의 best.pt가 갱신되어도 같은 모델을 유지한다.
'''
    (OUT / 'README.md').write_text(note)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'checkpoint_sha256': sha(CHECKPOINT), 'corpus_sha256': sha(CORPUS / 'manifest.json'),
                'anchor': 'ATEPP:00957', 'era_mapping': ERAS,
                'era_policy': 'sets may overlap; any Romantic overlap counts as same era, including Beethoven; no labels used in embedding ranking',
                'scope': 'one anchor searches every indexed accepted recording across train/validation/test; catalogue demo, not heldout evaluation',
                'aggregation': 'mean ALL full-window unit e in q_start order, then L2 normalize',
                'distance': '64D cosine to the exact anchor vector of the previous figure',
                'selection': 'global different-work ranking, then group minima; no feature filtering',
                'era_sources': ['https://www.classicsforkids.com/musical-genres/',
                                'https://www.esm.rochester.edu/beethoven/transitional-composer-and-heroic-objective/']}
    protocol['id'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (OUT / 'protocol.json').exists():
        assert json.loads((OUT / 'protocol.json').read_text()) == protocol
    else:
        write(OUT / 'protocol.json', protocol)
    corpus = IndexedNoteAECorpus(CORPUS)
    metadata = {r['perf_id']: r for r in csv.DictReader(
        (ROOT / 'datasets/ATEPP/ATEPP-metadata-1.2.csv').open(encoding='utf-8-sig'))}
    composers = {metadata[w['record_key']]['composer'] for w in corpus.items.values()}
    assert composers <= set(ERAS)
    cache = OUT / 'catalogue_embeddings.npz'
    if cache.exists():
        representatives = {}
        for w in corpus.items.values():
            representatives.setdefault(w['window']['performance_id'], w)
        with np.load(cache, allow_pickle=False) as f:
            assert str(f['protocol_sha']) == protocol['id']
            ids, e, counts = f['performance_id'].copy(), f['e'].copy(), f['counts'].tolist()
        rows = [representatives[key] for key in ids]
        assert sum(counts) == len(corpus.items)
    else:
        rows, e, counts = infer(corpus, protocol)
    result = select(rows, e, counts, metadata)
    assert sha(CHECKPOINT) == protocol['checkpoint_sha256']
    assert len(rows) == 2801
    x = profiles(result, corpus)
    render(result, x)
    document(result, protocol)


if __name__ == '__main__':
    main()
