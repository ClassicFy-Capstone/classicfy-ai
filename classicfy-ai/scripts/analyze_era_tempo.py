"""Summarize score-aligned absolute and relative tempo for the frozen 300 works."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
AI = PROJECT / 'classicfy-ai'
WORKSPACE = PROJECT.parent
DIAGNOSTICS = WORKSPACE / 'diagnostics/era_tempo_refine'
os.environ.setdefault('MPLCONFIGDIR', str(DIAGNOSTICS / 'matplotlib-cache'))
sys.path.insert(0, str(AI / 'src'))
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

ERA_NAMES = {'classical_baroque': '고전·바로크', 'romantic': '낭만', 'impressionist_modern': '인상주의·현대'}
COLORS = dict(zip(ERA_NAMES, ['#2979b8', '#d67735', '#268b75']))
PREFIXES = dict(zip(ERA_NAMES, ['B', 'R', 'M']))
QUANTILES = ('min', 'q25', 'median', 'q75', 'max')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                             for k, v in row.items()})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def five(values, prefix):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    result = np.quantile(values, [0, .25, .5, .75, 1], method='linear') if len(values) else np.full(5, np.nan)
    return {prefix + '_' + key: float(value) for key, value in zip(QUANTILES, result)}


def summarize_common(records, minimum=32, minimum_fraction=.5):
    """Use identical quarter intervals for every recording; no silent cohort reduction."""
    starts = [np.asarray(r['q'][:-1], float) for r in records]
    common = starts[0]
    for positions in starts[1:]:
        common = np.intersect1d(common, positions)
    taus, indices = [], []
    for row, positions in zip(records, starts):
        ix = np.searchsorted(positions, common)
        assert np.array_equal(positions[ix], common)
        indices.append(ix)
        taus.append(np.diff(row['seconds'])[ix])
    matrix = np.asarray(taus, float)
    valid = np.isfinite(matrix).all(axis=0) & ((matrix >= .08) & (matrix <= 5.)).all(axis=0)
    count = int(valid.sum())
    fraction = count / max(len(s) for s in starts)
    if count < minimum or fraction < minimum_fraction:
        return {'status': 'insufficient_common_beats', 'shared_beats': count,
                'shared_fraction': fraction}, [], None
    q = common[valid]
    bpm = 60. / matrix[:, valid]
    logs = np.log2(bpm)
    residual = logs - np.median(logs, axis=0)
    absolutes = np.median(bpm, axis=1)
    relative = np.median(residual, axis=1) if len(records) >= 2 else np.full(1, np.nan)
    percent = 100 * np.expm1(np.log(2) * relative)
    perf_rows = []
    for i, row in enumerate(records):
        perf_rows.append({'perf_id': row['perf_id'], 'composition_id': row['composition_id'],
            'absolute_tempo_qpm': float(absolutes[i]), 'relative_tempo_log2': float(relative[i]),
            'relative_tempo_pct': float(percent[i]), 'shared_beats': count,
            'shared_fraction': fraction, 'within_performance_qpm_iqr': float(np.ptp(np.quantile(bpm[i], [.25, .75]))),
            'within_performance_relative_log2_iqr': float(np.ptp(np.quantile(residual[i], [.25, .75])))})
    result = {'status': 'ok' if len(records) >= 2 else 'absolute_only_single_recording',
              'shared_beats': count, 'shared_fraction': fraction,
              **five(absolutes, 'absolute_qpm'), **five(relative, 'relative_log2'), **five(percent, 'relative_pct')}
    result['absolute_tempo_qpm'] = result['absolute_qpm_median']
    result['absolute_tempo_iqr_qpm'] = result['absolute_qpm_q75'] - result['absolute_qpm_q25']
    result['relative_tempo_iqr_pp'] = result['relative_pct_q75'] - result['relative_pct_q25']
    result['relative_tempo_iqr_log2'] = result['relative_log2_q75'] - result['relative_log2_q25']
    arrays = {'q': q, 'qpm': bpm, 'residual_log2': residual,
              'perf_ids': np.array([r['perf_id'] for r in records])}
    return result, perf_rows, arrays


def eta_squared(values, labels):
    total = np.sum((values - values.mean()) ** 2)
    if total == 0:
        return 0.
    between = sum(np.sum(labels == label) * (values[labels == label].mean() - values.mean()) ** 2
                  for label in np.unique(labels))
    return float(between / total)


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(order)); previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., pvalues[index] * (len(order) - rank)))
        adjusted[index] = previous
    return adjusted


def comparisons(works):
    rng = np.random.default_rng(20261106)
    output = []
    for metric in ['absolute_tempo_qpm', 'relative_tempo_iqr_pp']:
        usable = [r for r in works if np.isfinite(r.get(metric, np.nan))]
        groups = [np.array([r[metric] for r in usable if r['era'] == era]) for era in ERA_NAMES]
        if any(len(g) == 0 for g in groups):
            continue
        h, p = stats.kruskal(*groups)
        n = sum(map(len, groups)); k = len(groups)
        by_composer = defaultdict(list)
        for row in usable:
            by_composer[row['composer'], row['era']].append(row[metric])
        values = np.array([np.median(v) for v in by_composer.values()])
        labels = np.array([list(ERA_NAMES).index(key[1]) for key in by_composer])
        effect = eta_squared(values, labels)
        permutation = np.array([eta_squared(values, rng.permutation(labels)) for _ in range(9999)])
        p_composer = (1 + np.sum(permutation >= effect - 1e-12)) / 10000
        output.append({'metric': metric, 'works': n,
            **{f'{era}_n': len(g) for era, g in zip(ERA_NAMES, groups)},
            **{f'{era}_median': float(np.median(g)) for era, g in zip(ERA_NAMES, groups)},
            'work_kruskal_H': float(h), 'work_kruskal_p_exploratory': float(p),
            'work_epsilon_squared': max(0., float((h - k + 1) / (n - k))),
            'composers': len(values),
            **{f'{era}_composers': int(np.sum(labels == i)) for i, era in enumerate(ERA_NAMES)},
            'composer_eta_squared': effect, 'composer_label_permutation_p': float(p_composer),
            'composer_permutations': 9999,
            'scope': 'Observed score-supported subset; composer median aggregation; era labels confounded with composer; no full-300 inference.'})
    if output:
        for field, target in [('work_kruskal_p_exploratory', 'work_kruskal_p_holm'),
                              ('composer_label_permutation_p', 'composer_permutation_p_holm')]:
            for row, p in zip(output, holm([r[field] for r in output])):
                row[target] = float(p)
    return output


def fixed_recording_count_sensitivity(works, arrays):
    """Compare works using five recordings each, without replacing missing works."""
    rows = []
    comparison_rows = []
    for work in works:
        if work['recordings_used'] < 5:
            continue
        bpm = arrays[work['composition_id'] + '/qpm']
        seed = int(hashlib.sha256(('n5|' + work['composition_id']).encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        absolute, relative = [], []
        for _ in range(200):
            selected = bpm[rng.choice(len(bpm), 5, replace=False)]
            absolute.append(float(np.median(np.median(selected, axis=1))))
            logs = np.log2(selected)
            residual = logs - np.median(logs, axis=0)
            percents = 100 * np.expm1(np.log(2) * np.median(residual, axis=1))
            relative.append(float(np.ptp(np.quantile(percents, [.25, .75]))))
        rows.append({'display_id': work['display_id'], 'composition_id': work['composition_id'],
            'era': work['era'], 'composer': work['composer'], 'recordings_available': len(bpm),
            'recordings_per_draw': 5, 'draws': 200,
            'absolute_tempo_qpm': float(np.median(absolute)),
            'relative_tempo_iqr_pp': float(np.median(relative)),
            'relative_iqr_draw_q25': float(np.quantile(relative, .25)),
            'relative_iqr_draw_q75': float(np.quantile(relative, .75))})
    if rows:
        comparison_rows = comparisons(rows)
        for row in comparison_rows:
            row['scope'] = 'Sensitivity: at least five accepted recordings, median over 200 five-recording draws; fixed original common-position mask; no replacement works.'
    return rows, comparison_rows


def draw(works, coverage, out):
    for font in ['AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']:
        try:
            font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams['font.family'] = [font, 'DejaVu Sans']; break
        except ValueError:
            pass
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
        'figure.facecolor': '#fafaf8', 'axes.facecolor': '#fafaf8',
        'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': '#fafaf8'})
    metrics = [('absolute_tempo_qpm', '절대 tempo: 연주별 대표값의 중앙값 (QPM)', 'absolute_qpm'),
               ('relative_tempo_iqr_pp', '상대 tempo: 연주 간 빠르기 분포 폭 (IQR, %p)', None)]
    def save(fig, name):
        for extension in ['png', 'pdf']:
            fig.savefig(out / f'{name}.{extension}', dpi=190, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    for ax, (metric, label, prefix) in zip(axes, metrics):
        counts = []
        for j, era in enumerate(ERA_NAMES):
            selected = [(i, r) for i, r in enumerate(works) if r['era'] == era]
            visible = [(i, r) for i, r in selected if np.isfinite(r.get(metric, np.nan))]
            absent = [i for i, r in selected if not np.isfinite(r.get(metric, np.nan))]
            ax.axvspan(j * 100 - .5, (j + 1) * 100 - .5, color=COLORS[era], alpha=.045)
            x = [i for i, r in visible]; y = [r[metric] for i, r in visible]
            if prefix:
                for i, r in visible:
                    ax.vlines(i, r[prefix + '_q25'], r[prefix + '_q75'], color=COLORS[era], alpha=.5, lw=1.2)
            ax.scatter(x, y, color=COLORS[era], s=22, zorder=3)
            ax.scatter(absent, [-.07] * len(absent), transform=ax.get_xaxis_transform(),
                       marker='x', color='#b4b7ba', s=13, clip_on=False, linewidths=.6)
            counts.append(f'{ERA_NAMES[era]} {len(visible)}/100')
            if j:
                ax.axvline(j * 100 - .5, color='#b6b8b9', lw=.7)
        ax.set_ylabel(label, fontsize=11); ax.set_title(' · '.join(counts), loc='left', fontsize=11)
        ax.grid(axis='y', alpha=.2); ax.set_ylim(bottom=0)
    ticks = [0, 24, 49, 74, 99, 124, 149, 174, 199, 224, 249, 274, 299]
    axes[-1].set_xticks(ticks, [works[i]['display_id'] for i in ticks], fontsize=9)
    axes[-1].set_xlim(-3, 302)
    axes[-1].set_xlabel('고정된 300개 작품·악장 ID (각 시대 안에서 작곡가·composition_id 순)')
    fig.suptitle('작품마다 고유한 tempo 값 — 시대별 100개 작품의 전체 위치', fontsize=19, fontweight='bold')
    fig.text(.06, .015, '점=작품 1개 · 위쪽 선=연주별 절대 tempo의 Q1–Q3 · 회색 ×=미산출 (0이 아님) · QPM=4분음표/분 · 작품별 scale 표준화 없음', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95)); save(fig, '01_all_300_work_tempo')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    rng = np.random.default_rng(20261106)
    for ax, (metric, label, _) in zip(axes, metrics):
        for j, era in enumerate(ERA_NAMES):
            vals = np.array([r[metric] for r in works if r['era'] == era and np.isfinite(r.get(metric, np.nan))])
            if len(vals):
                box = ax.boxplot([vals], positions=[j], widths=.45, patch_artist=True, showfliers=False,
                    medianprops={'color': '#28323c'}, whiskerprops={'color': '#89939a'}, capprops={'color': '#89939a'})
                box['boxes'][0].set(facecolor=COLORS[era], alpha=.18, edgecolor=COLORS[era])
                ax.scatter(j + rng.uniform(-.16, .16, len(vals)), vals, color=COLORS[era], s=27, alpha=.8)
        ax.set_xticks(range(3), [f'{name}\nn={sum(r["era"] == era and np.isfinite(r.get(metric, np.nan)) for r in works)}'
                                for era, name in ERA_NAMES.items()])
        ax.set_ylabel(label); ax.grid(axis='y', alpha=.2); ax.set_ylim(bottom=0)
    fig.suptitle('시대별 작품 tempo 분포 — 값을 확보한 작품만 표시', fontsize=18, fontweight='bold')
    fig.text(.05, .015, '점=작품 1개. 원래 선정한 각 시대 100개 전체를 대표하는 완전한 분포가 아님. 악보·정렬에 따른 관측 편향이 있음.', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .92)); save(fig, '02_era_distributions')

    fig, ax = plt.subplots(figsize=(12, 8))
    annotations = []
    for era, name in ERA_NAMES.items():
        selected = [r for r in works if r['era'] == era and np.isfinite(r.get('relative_tempo_iqr_pp', np.nan))]
        ax.scatter([r['absolute_tempo_qpm'] for r in selected], [r['relative_tempo_iqr_pp'] for r in selected],
                   s=54, color=COLORS[era], alpha=.85, edgecolors='white', linewidths=.5, label=f'{name} (n={len(selected)})')
        for row in selected:
            annotations.append(row)
    ax.set(xlabel='절대 tempo: 연주별 대표값의 중앙값 (QPM, 4분음표/분)',
           ylabel='상대 tempo: 연주 간 빠르기 분포 폭 (IQR, %p)')
    ax.legend(frameon=False); ax.grid(alpha=.18); ax.set_ylim(bottom=0)
    ax.set_title('작품의 절대 빠르기와 연주 간 상대 빠르기 차이', loc='left', fontsize=18, fontweight='bold', pad=15)
    fig.text(.08, .02, '색=시대 · 점=작품 · 표기 ID는 work_tempo_summary.csv와 대응. 같은 악보 위치에서 비교하며 상대 tempo는 공통값 대비 비율이다.', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    # Place short work IDs without changing the plotted values.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer(); placed = []
    for row in sorted(annotations, key=lambda r: (r['absolute_tempo_qpm'], r['relative_tempo_iqr_pp'])):
        annotation = ax.annotate(row['display_id'], (row['absolute_tempo_qpm'], row['relative_tempo_iqr_pp']),
            xytext=(4, 4), textcoords='offset points', fontsize=7, color=COLORS[row['era']])
        options = [(4, 4), (4, -10), (-25, 4), (-25, -10), (4, 13), (4, -20), (-25, 13), (-25, -20)]
        candidates = []
        for offset in options:
            annotation.set_position(offset)
            box = annotation.get_window_extent(renderer).expanded(1.08, 1.2)
            candidates.append((sum(box.overlaps(previous) for previous in placed), offset, box))
        _, offset, box = min(candidates, key=lambda option: option[0])
        annotation.set_position(offset); placed.append(box)
    save(fig, '03_labeled_tempo_scatter')

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(3)
    absolute = np.array([r['absolute_works'] for r in coverage]); relative = np.array([r['relative_works'] for r in coverage])
    ax.bar(x, relative, color='#268b75', label='절대·상대 모두 산출')
    ax.bar(x, absolute - relative, bottom=relative, color='#e4b862', label='절대만 산출 (유효 녹음 1개)')
    ax.bar(x, 100 - absolute, bottom=absolute, color='#d5d9dc', label='절대·상대 미산출')
    for i, row in enumerate(coverage):
        ax.text(i, 103, f'절대 {row["absolute_works"]} / 상대 {row["relative_works"]}', ha='center', fontsize=11)
    ax.set_xticks(x, list(ERA_NAMES.values())); ax.set(ylim=(0, 119), ylabel='선정 작품·악장 수 (시대별 100개)')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncols=3, frameon=False)
    ax.set_title('300개 선정 표본 중 실제 tempo를 산출한 범위', loc='left', fontsize=18, fontweight='bold')
    fig.tight_layout(); save(fig, '04_tempo_coverage')


def analyze(alignment_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    cohort = AI / 'analysis/era_refine_balanced_100'
    chosen = read_csv(cohort / 'selected_works.csv')
    selected_recordings = read_csv(cohort / 'selected_recordings.csv')
    assert len(chosen) == 300 and len(selected_recordings) == 1864
    audit_path = alignment_dir / 'recordings.jsonl'
    audit = {r['perf_id']: r for r in [json.loads(l) for l in audit_path.read_text().splitlines()]}
    assert set(audit) == {r['perf_id'] for r in selected_recordings}
    by_work = defaultdict(list)
    arrays, performance_summary, works = {}, [], []
    for row in audit.values():
        by_work[row['composition_id']].append(row)
    era_index = Counter()
    for index, work in enumerate(chosen):
        era = work['era']; era_index[era] += 1
        result = {'display_id': PREFIXES[era] + f'{era_index[era]:03}',
            'composition_id': work['composition_id'], 'era': era, 'era_ko': ERA_NAMES[era],
            'composer': work['composer'], 'track': work['representative_track'],
            'recordings_total': int(work['recordings']),
            'recordings_accepted': 0, 'recordings_used': 0,
            'absolute_tempo_qpm': np.nan, 'relative_tempo_iqr_pp': np.nan}
        members = by_work[work['composition_id']]
        result.update({f'n_{status}': sum(r['status'] == status for r in members) for status in
                       ['no_score', 'unsupported_score', 'alignment_rejected', 'technical_error', 'resource_limit']})
        accepted = [r for r in members if r['status'] == 'accepted']
        result['recordings_accepted'] = len(accepted)
        if not accepted:
            statuses = Counter(r['status'] for r in members)
            result['status'] = statuses.most_common(1)[0][0]
            result['reason_counts'] = json.dumps(statuses, sort_keys=True)
            works.append(result); continue
        groups = defaultdict(list)
        for row in accepted:
            groups[row['canonical_score_sha256']].append(row)
        # Do not mix different score editions or coordinate traversals.
        main_hash = min(groups, key=lambda key: (-len(groups[key]), key))
        accepted = sorted(groups[main_hash], key=lambda r: r['perf_id'])
        result['score_cohorts'] = len(groups)
        result['recordings_other_score_edition'] = result['recordings_accepted'] - len(accepted)
        result['canonical_score_sha256'] = main_hash
        usable = []
        for row in accepted:
            assert sha(row['alignment_file']) == row['alignment_sha256']
            with np.load(row['alignment_file'], allow_pickle=False) as cache:
                q = cache['score_beats'].astype(float); seconds = cache['performance_beats'].astype(float)
            assert q.shape == seconds.shape and np.all(np.diff(seconds) > 0) and np.allclose(np.diff(q), 1.)
            usable.append(dict(row, q=q, seconds=seconds))
        summary, performances, packed = summarize_common(usable)
        result.update(summary)
        result['recordings_used'] = len(usable) if packed is not None else 0
        result['complete_recording_coverage'] = result['recordings_used'] == result['recordings_total']
        for row in performances:
            source = audit[row['perf_id']]
            performance_summary.append(dict(row, display_id=result['display_id'], era=era,
                artist=source['artist'], quality=source['quality'], alignment_origin=source['origin']))
        if packed is not None:
            arrays.update({f'{work["composition_id"]}/{key}': value for key, value in packed.items()})
        works.append(result)
    np.savez_compressed(alignment_dir / 'common_beat_values.npz', **arrays)
    for row in works:
        row['absolute_available'] = bool(np.isfinite(row['absolute_tempo_qpm']))
        row['relative_available'] = bool(np.isfinite(row['relative_tempo_iqr_pp']))
    coverage = []
    for era in ERA_NAMES:
        subset = [r for r in works if r['era'] == era]
        coverage.append({'era': era, 'era_ko': ERA_NAMES[era], 'selected_works': len(subset),
            'selected_recordings': sum(r['recordings_total'] for r in subset),
            'absolute_works': sum(r['absolute_available'] for r in subset),
            'relative_works': sum(r['relative_available'] for r in subset),
            'recordings_used': sum(r['recordings_used'] for r in subset),
            'status_counts': json.dumps(Counter(r['status'] for r in subset), sort_keys=True)})
    write_csv(out / 'work_tempo_summary.csv', works)
    value_columns = ['display_id', 'composition_id', 'era_ko', 'composer', 'track',
                     'recordings_total', 'recordings_used', 'absolute_tempo_qpm',
                     'relative_tempo_iqr_pp', 'relative_pct_median', 'status']
    write_csv(out / 'work_tempo_values.csv', [{k: r.get(k, np.nan) for k in value_columns} for r in works])
    for era in ERA_NAMES:
        write_csv(out / f'{era}_work_tempo.csv', [{k: r.get(k, np.nan) for k in value_columns}
                                                for r in works if r['era'] == era])
    write_csv(out / 'performance_tempo_summary.csv', performance_summary)
    write_csv(out / 'coverage.csv', coverage)
    status_rows = [dict(row, alignment_metrics=json.dumps(row.get('alignment_metrics', {}), sort_keys=True)) for row in audit.values()]
    write_csv(out / 'recording_status.csv', status_rows)
    comparison = comparisons(works)
    write_csv(out / 'era_statistics.csv', comparison)
    sensitivity, sensitivity_comparisons = fixed_recording_count_sensitivity(works, arrays)
    write_csv(out / 'n5_work_sensitivity.csv', sensitivity)
    write_csv(out / 'n5_era_statistics.csv', sensitivity_comparisons)
    draw(works, coverage, out)
    protocol = {'population': 'Frozen ATEPP_refine 300 works / 1864 recordings; no replacement works.',
        'absolute': 'Median of per-recording median quarter-note QPM, computed on identical shared quarter intervals.',
        'relative_per_beat': 'log2(QPM[p,b]) - median_p(log2(QPM[p,b])); self included in descriptive reference.',
        'relative_per_recording': 'Median across common quarter intervals; display percentage = 100*(2**value-1).',
        'relative_per_work': 'Q75 minus Q25 of recording relative percentages, in percentage points; signed five-number summaries also retained.',
        'shared_mask': 'Exact common score quarter intervals valid for every used recording; tau in [0.08,5] seconds; >=32 quarters and >=50% of largest individual map.',
        'score_editions': 'Use largest accepted identical canonical-score cohort per composition_id, tie by score SHA256; report excluded edition count.',
        'singleton': 'Absolute permitted; relative unavailable, never set to zero.',
        'normalization': 'No per-work scale, z-score, clipping or missing-value imputation.',
        'units': 'QPM = quarter notes/minute. This is not a meter-dependent perceived beat rate.',
        'quantiles': 'NumPy linear quantiles; interpolation when recordings are few; min/max and n retained.',
        'statistics': 'Exploratory work-level Kruskal-Wallis; composer-median eta squared with 9999 composer-label permutations; Holm adjustment across two metrics, separately for each test family. Not evidence for separation of all 300 works.',
        'statistics_reference': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html',
        'coverage': coverage, 'comparisons': comparison, 'n5_comparisons': sensitivity_comparisons,
        'input_sha256': {str(p): sha(p) for p in [cohort/'selected_works.csv', cohort/'selected_recordings.csv', audit_path, alignment_dir/'alignment_protocol.json']},
        'common_beat_values': str(alignment_dir/'common_beat_values.npz'),
        'common_beat_values_sha256': sha(alignment_dir/'common_beat_values.npz'),
        'script_sha256': sha(Path(__file__)), 'versions': {'numpy': np.__version__, 'scipy': __import__('scipy').__version__, 'matplotlib': matplotlib.__version__}}
    (out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    print(json.dumps({'coverage': coverage, 'comparisons': comparison}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignments', type=Path, default=DIAGNOSTICS)
    parser.add_argument('--out', type=Path, default=AI / 'analysis/era_refine_tempo')
    args = parser.parse_args()
    analyze(args.alignments, args.out)
