"""Extract and plot raw/common-relative rubato, dynamics and CC64 for refine.

Uses all currently accepted exact-input alignments, not the former 300-work draw.
No new alignment, per-work scale normalization, or missing-value imputation.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from fractions import Fraction

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
AI = PROJECT / 'classicfy-ai'
DIAG = WORKSPACE / 'diagnostics/era_expression_refine'
sys.path.insert(0, str(AI / 'src'))
from prepare_balanced_era_cohort import ERAS
import numpy as np

CHANNELS = ['rubato', 'dynamics', 'pedal_depth', 'pedal_down_ratio', 'pedal_changes']
NAMES = {'rubato': '루바토', 'dynamics': '다이내믹', 'pedal_depth': '페달 깊이',
         'pedal_down_ratio': '페달 사용 비율', 'pedal_changes': '페달 전환 빈도'}
COLORS = dict(zip(ERAS, ['#2979b8', '#d67735', '#268b75']))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v
                             for k, v in row.items()})


def observed_pedal_mask(base, left_edges, first_event, *, changes=False):
    if first_event is None:
        return np.zeros(len(left_edges), bool)
    # State is known at the first event; a transition from the prior unknown
    # state is not known. Exclude that boundary interval for change counts.
    return np.asarray(base, bool) & (np.asarray(left_edges) > first_event if changes
                                    else np.asarray(left_edges) >= first_event)


def catalog():
    refine = WORKSPACE / 'datasets/ATEPP_refine'
    metadata = refine / 'ATEPP-metadata-1.2.csv'
    receipt = json.loads((refine / 'refinement_manifest.json').read_text())
    assert sha(metadata) == receipt['refined_metadata_sha256']
    rows = read_csv(metadata); by_id = {r['perf_id']: r for r in rows}
    assert len(rows) == len(by_id) == receipt['retained_performance_count']
    checks = {r['refined_file']: r for r in read_csv(refine / 'file_checksums.csv')}
    previous = WORKSPACE / 'diagnostics/atepp_note_training/run01/preprocessing'
    raw = {r['perf_id']: r for r in read_jsonl(previous / 'raw.jsonl')}
    old = {r['perf_id']: r for r in read_jsonl(previous / 'alignments.jsonl')}
    pool = {}
    for key, record in old.items():
        if key not in by_id or record['status'] != 'extracted':
            continue
        path = previous / 'performances' / key / 'alignment.npz'
        bridge = json.loads((path.parent / 'bridge.json').read_text())
        assert bridge['alignment_accepted'] and bridge['score_coordinate_unit'] == 'quarter'
        pool[key] = {'alignment_file': str(path), 'canonical_score_sha256': bridge['score_midi_sha256'],
                     'xml_sha256': bridge['xml_sha256'], 'origin': 'existing_full_corpus_dual_dtw'}
    additions = WORKSPACE / 'diagnostics/era_tempo_refine/recordings.jsonl'
    for record in read_jsonl(additions):
        if record['status'] == 'accepted':
            key = record['perf_id']; assert key in by_id
            pool[key] = {'alignment_file': record['alignment_file'],
                         'canonical_score_sha256': record['canonical_score_sha256'],
                         'xml_sha256': record['score_xml_sha256'], 'origin': record['origin']}
    composer_era = {c: (era, label) for era, (label, composers) in ERAS.items() for c in composers}
    accepted = []
    for key in sorted(pool):
        row = by_id[key]
        midi = refine / 'ATEPP-1.2' / row['midi_path']
        expected = checks['ATEPP-1.2/' + row['midi_path']]['sha256']
        assert expected == raw[key]['sha256']
        era, label = composer_era[row['composer']]
        xml = refine / 'ATEPP-1.2' / row['score_path']
        assert row['score_path'] and sha(xml) == pool[key]['xml_sha256']
        accepted.append(dict(pool[key], perf_id=key, composition_id=row['composition_id'],
            era=era, era_ko=label, composer=row['composer'], artist=row['artist'], quality=row['quality'],
            midi_file=str(midi), midi_sha256=expected, alignment_sha256=sha(pool[key]['alignment_file'])))
    inputs = [metadata, refine/'refinement_manifest.json', refine/'file_checksums.csv',
              previous/'raw.jsonl', previous/'alignments.jsonl', additions]
    return rows, accepted, {str(p): sha(p) for p in inputs}


def extract_one(task):
    row, destination = task
    from preprocessing.performance_midi import load_performance_midi
    from preprocessing.midi_loader import MidiData, NoteEvent, PedalEvent
    from features.dynamics import extract_dynamics
    from features.pedaling import extract_pedaling
    result = dict(row)
    try:
        assert sha(row['midi_file']) == row['midi_sha256']
        assert sha(row['alignment_file']) == row['alignment_sha256']
        with np.load(row['alignment_file'], allow_pickle=False) as cache:
            q = cache['score_beats'].copy(); seconds = cache['performance_beats'].copy()
        assert np.allclose(np.diff(q), 1.) and np.all(np.diff(seconds) > 0)
        performance = load_performance_midi(row['midi_file'])
        notes = [NoteEvent(n.pitch, n.onset_sec, n.key_offset_sec if n.key_offset_sec is not None else n.onset_sec,
                           n.velocity, 0, n.program) for n in performance.notes]
        cc64 = [c for c in performance.controls if c.number == 64]
        midi = MidiData(notes, [PedalEvent(c.time_sec, c.value, 0) for c in cc64], performance.duration_sec)
        dynamics = extract_dynamics(midi, seconds)
        pedal = extract_pedaling(midi, seconds)
        tau = np.diff(seconds)
        tempo_mask = np.isfinite(tau) & (tau >= .08) & (tau <= 5.)
        # ATEPP absent CC64 is unobserved. Do not infer an initial pedal state.
        pedal_mask = observed_pedal_mask(pedal.depth.mask, seconds[:-1], cc64[0].time_sec if cc64 else None)
        path = Path(destination) / 'raw_features' / (row['perf_id'] + '.npz')
        path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, q=q, seconds=seconds, tempo_log2=np.log2(60/tau), tempo_mask=tempo_mask,
            dynamics=dynamics.sequence.values, dynamics_mask=dynamics.sequence.mask,
            pedal_depth=pedal.depth.values, pedal_down_ratio=pedal.down_ratio.values,
            pedal_changes=pedal.changes.values, pedal_mask=pedal_mask,
            midi_sha256=np.array(row['midi_sha256']), alignment_sha256=np.array(row['alignment_sha256']))
        return dict(result, status='extracted', raw_features=str(path), raw_features_sha256=sha(path),
                    note_onsets=len(notes), cc64_events=len(cc64),
                    first_cc64_seconds=cc64[0].time_sec if cc64 else None)
    except Exception as exc:
        return dict(result, status='technical_error', reason=f'{type(exc).__name__}: {exc}')


def five(values, prefix):
    return {prefix + '_' + key: float(value) for key, value in
            zip(['min', 'q25', 'median', 'q75', 'max'], np.quantile(values, [0, .25, .5, .75, 1]))}


def summarize_channel(values, valid, channel, reference_intervals, minimum=32, minimum_fraction=.5):
    """Common-position raw level and magnitude after removing the work pattern.

    Rubato: median absolute log-tempo deviation; D/P: mean levels and mean
    absolute residuals. Every recording uses the same valid score positions.
    """
    shared = np.asarray(valid, bool).all(axis=0) & np.isfinite(values).all(axis=0)
    n = int(shared.sum()); fraction = n / reference_intervals
    count = len(values)
    base = {'channel': channel, 'shared_beats': n, 'shared_fraction': fraction,
            'recordings_used': 0, 'absolute_value': np.nan, 'relative_value': np.nan}
    if n < minimum or fraction < minimum_fraction:
        return dict(base, status='insufficient_common_positions'), None
    raw = values[:, shared].copy()
    if channel == 'rubato':
        raw -= np.median(raw, axis=1)[:, None]
    residual = raw - np.median(raw, axis=0)
    if channel == 'rubato':
        absolute = np.median(np.abs(raw), axis=1)
        relative = np.median(np.abs(residual), axis=1)
    else:
        absolute = raw.mean(axis=1)
        relative = np.abs(residual).mean(axis=1)
    signed_relative = np.median(residual, axis=1) if channel == 'rubato' else residual.mean(axis=1)
    if count < 2:
        relative[:] = np.nan; signed_relative[:] = np.nan
    result = dict(base, status='ok' if count >= 2 else 'absolute_only_single_recording',
        recordings_used=count, absolute_value=float(np.median(absolute)),
        relative_value=float(np.median(relative)) if count >= 2 else np.nan,
        **five(absolute, 'absolute_recording'),
        raw_within_recording_iqr_median=float(np.median(np.ptp(np.quantile(raw, [.25, .75], axis=1), axis=0))))
    if count >= 2:
        result.update(five(relative, 'relative_amount_recording'))
        result.update(five(signed_relative, 'signed_relative_recording'))
    return result, {'raw': raw, 'residual': residual, 'shared_mask': shared,
                    'absolute': absolute, 'relative': relative, 'signed_relative': signed_relative}


def display_value(channel, value):
    if channel == 'rubato':
        return 100 * np.expm1(np.log(2) * np.asarray(value))
    return np.asarray(value) * {'dynamics': 127., 'pedal_depth': 127., 'pedal_down_ratio': 100., 'pedal_changes': 1.}[channel]


def balanced_selection(work_rows, maximum=30):
    by_work = defaultdict(list)
    for row in work_rows:
        by_work[row['composition_id']].append(row)
    eligible = [rows[0] for rows in by_work.values() if len(rows) == len(CHANNELS)
                and all(np.isfinite(row['relative_value']) for row in rows)]
    target = min(maximum, min(sum(row['era'] == era for row in eligible) for era in ERAS))
    selected, allocations = [], []
    for era in ERAS:
        composers = defaultdict(list)
        for row in eligible:
            if row['era'] == era:
                composers[row['composer']].append(row)
        exact = {c: Fraction(target*len(rows), sum(map(len, composers.values()))) for c, rows in composers.items()}
        quotas = {c: max(1, share.numerator//share.denominator) for c, share in exact.items()}
        assert sum(quotas.values()) <= target
        while sum(quotas.values()) < target:
            c = min((c for c in quotas if quotas[c] < len(composers[c])),
                    key=lambda c: (-(exact[c]-quotas[c]), c))
            quotas[c] += 1
        for composer, rows in sorted(composers.items()):
            ranked = sorted(rows, key=lambda r: hashlib.sha256(
                f'20261106|expression-balanced|{era}|{composer}|{r["composition_id"]}'.encode()).hexdigest())
            for row in ranked[:quotas[composer]]:
                selected.append({k: row[k] for k in ['composition_id', 'work_id', 'era', 'era_ko', 'composer', 'track']})
            allocations.append({'era': era, 'composer': composer, 'eligible_works': len(rows), 'selected_works': quotas[composer]})
    assert len(selected) == target*3 and len({r['composition_id'] for r in selected}) == target*3
    return selected, allocations, target


def draw(work_rows, out, tag=''):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    for font in ['AppleGothic', 'NanumGothic', 'Noto Sans CJK KR']:
        try:
            font_manager.findfont(font, fallback_to_default=False); plt.rcParams['font.family'] = [font, 'DejaVu Sans']; break
        except ValueError:
            pass
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
        'figure.facecolor': '#fafaf8', 'axes.facecolor': '#fafaf8',
        'axes.spines.top': False, 'axes.spines.right': False})
    labels = {
        'rubato': ('루바토 변화량 (%)', '공통 루바토 대비 국소 편차 크기 (%)'),
        'dynamics': ('평균 MIDI velocity (0–127)', '공통 강약 곡선 대비 평균 절대편차 (velocity)'),
        'pedal_depth': ('평균 CC64 값 (0–127)', '공통 페달 깊이 대비 평균 절대편차 (CC64)'),
        'pedal_down_ratio': ('페달 사용 시간 비율 (%)', '공통 사용 비율 대비 평균 절대편차 (%p)'),
        'pedal_changes': ('페달 on/off 전환 횟수 / 4분음표', '공통 전환 빈도 대비 평균 절대편차 / 4분음표')}
    def save(fig, name):
        for extension in ['png', 'pdf']:
            fig.savefig(out / f'{name}{tag}.{extension}', dpi=185, bbox_inches='tight', facecolor='#fafaf8')
        plt.close(fig)
    for feature, channels in [('rubato', ['rubato']), ('dynamics', ['dynamics']),
                               ('pedaling', ['pedal_depth', 'pedal_down_ratio', 'pedal_changes'])]:
        panels = len(channels)
        fig, axes = plt.subplots(1, panels, figsize=(11, 6.5) if panels == 1 else (19, 6))
        axes = np.atleast_1d(axes)
        for ax, channel in zip(axes, channels):
            for era, (name, _) in ERAS.items():
                group = [r for r in work_rows if r['channel'] == channel and r['era'] == era and np.isfinite(r['relative_value'])]
                x = display_value(channel, [r['absolute_value'] for r in group])
                y = display_value(channel, [r['relative_value'] for r in group])
                ax.scatter(x, y, s=34 if panels == 1 else 21, color=COLORS[era], alpha=.72,
                           edgecolors='white', linewidths=.3, label=f'{name} (n={len(group)})')
            ax.set(xlabel='절대: ' + labels[channel][0], ylabel='상대: ' + labels[channel][1])
            ax.set_title(NAMES[channel] + ' — 작품별 절대·상대 수치', loc='left', fontsize=15)
            ax.set_ylim(bottom=0); ax.set_xlim(left=0); ax.grid(alpha=.18)
            ax.legend(frameon=False, fontsize=10)
        fig.suptitle({'rubato': '루바토', 'dynamics': '다이내믹', 'pedaling': '페달링'}[feature] +
                     ' | 색 = 시대 · 점 하나 = 작품·악장 하나', fontsize=17)
        fig.text(.06, .012, '동일 악보 위치에서 비교 · 상대값 = 작품 공통 패턴 제거 후 남는 편차의 크기 · 작품별 scale 표준화 없음 · 각 패널의 실제 n 표시', fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .94)); save(fig, f'{feature}_absolute_relative')

        fig, axes = plt.subplots(2, panels, figsize=(10, 9) if panels == 1 else (18, 10), squeeze=False)
        rng = np.random.default_rng(20261106)
        for column, channel in enumerate(channels):
            for axis_row, (field, kind) in enumerate([('absolute_value', '절대'), ('relative_value', '상대')]):
                ax = axes[axis_row, column]
                ticks = []
                for i, (era, (name, _)) in enumerate(ERAS.items()):
                    vals = display_value(channel, [r[field] for r in work_rows if r['channel'] == channel
                                                  and r['era'] == era and np.isfinite(r[field])])
                    ticks.append(f'{name}\nn={len(vals)}')
                    if len(vals):
                        box = ax.boxplot([vals], positions=[i], widths=.45, showfliers=False, patch_artist=True,
                                         medianprops={'color': '#27333b'})
                        box['boxes'][0].set(facecolor=COLORS[era], alpha=.2)
                        ax.scatter(i + rng.uniform(-.16, .16, len(vals)), vals, s=19, color=COLORS[era], alpha=.65)
                ax.set_xticks(range(3), ticks, fontsize=9)
                ax.set_ylabel(labels[channel][axis_row]); ax.set_title(f'{NAMES[channel]} · {kind}')
                ax.grid(axis='y', alpha=.18); ax.set_ylim(bottom=0)
        fig.suptitle({'rubato': '루바토', 'dynamics': '다이내믹', 'pedaling': '페달링'}[feature] + ' | 시대별 작품 값의 분포', fontsize=17)
        fig.text(.05, .01, '점 = 작품 1개 · 상자 = 작품 간 Q1–Q3 · 위 = 공통값 제거 전 · 아래 = 공통값 제거 후 국소 편차 크기 · 산출 조건에 따른 관측 편향 있음', fontsize=9)
        fig.tight_layout(rect=(0, .045, 1, .95)); save(fig, f'{feature}_era_distributions')


def analyze(destination, out, workers):
    destination.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    for name in ['raw_features', 'matplotlib-cache']:
        (destination/name).mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(destination/'matplotlib-cache')
    os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
    metadata, accepted, input_hashes = catalog()
    metadata_by_work = defaultdict(list)
    for row in metadata:
        metadata_by_work[row['composition_id']].append(row)
    audit = destination/'extraction.jsonl'
    old = {r['perf_id']: r for r in read_jsonl(audit)}
    tasks = []
    for row in accepted:
        existing = old.get(row['perf_id'])
        if existing and existing['status'] == 'extracted':
            assert existing['midi_sha256'] == row['midi_sha256']
            assert existing['alignment_sha256'] == row['alignment_sha256']
            assert sha(existing['raw_features']) == existing['raw_features_sha256']
        else:
            tasks.append((row, str(destination)))
    print(f'All refine: {len(metadata)} recordings / {len(metadata_by_work)} works; accepted alignments {len(accepted)}; extraction tasks {len(tasks)}', flush=True)
    with audit.open('a', encoding='utf-8') as stream, ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract_one, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result(); stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()
            if i % 100 == 0 or i == len(tasks):
                print(f'Extracted {i}/{len(tasks)}; last={result["status"]}', flush=True)
    extracted = {r['perf_id']: r for r in read_jsonl(audit)}
    assert set(extracted) == {r['perf_id'] for r in accepted}
    assert all(r['status'] == 'extracted' for r in extracted.values()), 'Technical extraction errors must be resolved before plotting'
    by_work = defaultdict(lambda: defaultdict(list))
    for row in extracted.values():
        by_work[row['composition_id']][row['canonical_score_sha256']].append(row)
    era_map = {c: (era, label) for era, (label, composers) in ERAS.items() for c in composers}
    work_rows, performance_rows, cohort_rows = [], [], []
    packed = {}; order = {era: i for i, era in enumerate(ERAS)}
    work_ids = sorted(metadata_by_work, key=lambda key: (order[era_map[metadata_by_work[key][0]['composer']][0]],
                                                       metadata_by_work[key][0]['composer'], int(key)))
    for step, work_id in enumerate(work_ids, 1):
        source_rows = metadata_by_work[work_id]; composer = source_rows[0]['composer']; era, era_name = era_map[composer]
        titles = Counter(r['track'] for r in source_rows); title = min(titles, key=lambda t: (-titles[t], t))
        base = {'composition_id': work_id, 'work_id': 'ATEPP/' + work_id, 'era': era, 'era_ko': era_name,
                'composer': composer, 'track': title, 'recordings_in_refine': len(source_rows)}
        editions = by_work.get(work_id, {})
        members = sorted(editions[min(editions, key=lambda key: (-len(editions[key]), key))], key=lambda r: r['perf_id']) if editions else []
        cohort_rows.append(dict(base, accepted_recordings=sum(map(len, editions.values())),
                               used_score_cohort_recordings=len(members), score_cohorts=len(editions)))
        if not members:
            for channel in CHANNELS:
                work_rows.append(dict(base, channel=channel, status='no_accepted_alignment', recordings_used=0,
                                      shared_beats=0, shared_fraction=0., absolute_value=np.nan, relative_value=np.nan))
            continue
        loaded = []
        for row in members:
            with np.load(row['raw_features'], allow_pickle=False) as cache:
                loaded.append({k: cache[k].copy() for k in cache.files if k not in ['midi_sha256', 'alignment_sha256']})
        starts = [cache['q'][:-1] for cache in loaded]; common = starts[0]
        for positions in starts[1:]:
            common = np.intersect1d(common, positions)
        indices = [np.searchsorted(s, common) for s in starts]
        reference = max(map(len, starts))
        for channel in CHANNELS:
            field = 'tempo_log2' if channel == 'rubato' else channel
            mask = 'tempo_mask' if channel == 'rubato' else 'dynamics_mask' if channel == 'dynamics' else 'pedal_mask'
            values = np.array([cache[field][ix] for cache, ix in zip(loaded, indices)])
            valid = np.array([cache[mask][ix] & cache['tempo_mask'][ix] for cache, ix in zip(loaded, indices)])
            if channel == 'pedal_changes':
                valid &= np.array([observed_pedal_mask(np.ones(len(ix), bool), cache['seconds'][:-1][ix],
                                      row['first_cc64_seconds'], changes=True)
                                   for row, cache, ix in zip(members, loaded, indices)])
            summary, arrays = summarize_channel(values, valid, channel, reference)
            result = dict(base, **summary, accepted_recordings=len(members))
            result['absolute_display'] = float(display_value(channel, result['absolute_value']))
            result['relative_display'] = float(display_value(channel, result['relative_value']))
            result['display_unit'] = {'rubato': '% log-ratio equivalent', 'dynamics': 'MIDI velocity',
                'pedal_depth': 'CC64', 'pedal_down_ratio': '% (relative: percentage points)',
                'pedal_changes': 'on/off transitions per quarter'}[channel]
            work_rows.append(result)
            if arrays is not None:
                packed.update({f'{work_id}/{channel}/{key}': value for key, value in arrays.items()})
                packed[f'{work_id}/{channel}/q'] = common[arrays['shared_mask']]
                packed[f'{work_id}/{channel}/perf_ids'] = np.array([row['perf_id'] for row in members])
                for i, row in enumerate(members):
                    performance_rows.append(dict(base, channel=channel, perf_id=row['perf_id'], artist=row['artist'],
                        absolute_value=float(arrays['absolute'][i]), relative_value=float(arrays['relative'][i]),
                        signed_relative_value=float(arrays['signed_relative'][i]),
                        absolute_display=float(display_value(channel, arrays['absolute'][i])),
                        relative_display=float(display_value(channel, arrays['relative'][i])),
                        shared_beats=summary['shared_beats']))
        if step % 200 == 0:
            print(f'Summarized {step}/{len(work_ids)} refine work slots', flush=True)
    coverage = []
    for channel in CHANNELS:
        for era, (label, _) in ERAS.items():
            group = [r for r in work_rows if r['channel'] == channel and r['era'] == era]
            absolute = [r for r in group if np.isfinite(r['absolute_value'])]
            relative = [r for r in group if np.isfinite(r['relative_value'])]
            coverage.append({'channel': channel, 'era': era, 'era_ko': label, 'refine_works': len(group),
                'absolute_works': len(absolute), 'relative_works': len(relative),
                'used_recordings': sum(r['recordings_used'] for r in absolute),
                'absolute_display_median': float(np.median([r['absolute_display'] for r in absolute])) if absolute else np.nan,
                'relative_display_median': float(np.median([r['relative_display'] for r in relative])) if relative else np.nan,
                'status_counts': json.dumps(Counter(r['status'] for r in group), sort_keys=True)})
    write_csv(out/'work_feature_values.csv', work_rows)
    write_csv(out/'performance_feature_values.csv', performance_rows)
    write_csv(out/'cohort_works.csv', cohort_rows)
    write_csv(out/'alignment_catalog.csv', accepted)
    write_csv(out/'coverage.csv', coverage)
    for channel in CHANNELS:
        write_csv(out/f'{channel}_work_values.csv', [r for r in work_rows if r['channel'] == channel])
    balanced, allocations, target = balanced_selection(work_rows)
    balanced_ids = {r['composition_id'] for r in balanced}
    balanced_rows = [r for r in work_rows if r['composition_id'] in balanced_ids]
    write_csv(out/'balanced_work_ids.csv', balanced)
    write_csv(out/'balanced_composer_allocation.csv', allocations)
    write_csv(out/'balanced_work_values.csv', balanced_rows)
    np.savez_compressed(destination/'common_values.npz', **packed)
    draw(work_rows, out)
    draw(balanced_rows, out, tag=f'_balanced{target}')
    source_files = [Path(__file__), AI/'src/features/rubato.py', AI/'src/features/dynamics.py',
                    AI/'src/features/pedaling.py', AI/'src/preprocessing/performance_midi.py']
    protocol = {'scope': 'Entire current ATEPP_refine, all currently available accepted exact-input DualDTW maps; not the former 300-work sample.',
        'refine_recordings': len(metadata), 'refine_works': len(metadata_by_work), 'accepted_recordings': len(accepted),
        'aligned_works': len(by_work), 'work_unit': 'Original composition_id (may be a movement).',
        'relative_policy': 'Per-position median over used recordings is subtracted; reference includes self. Relative amount measures typical local residual magnitude, NOT IQR of global signed tempo representatives.',
        'rubato': 'raw=log2(QPM)-own median over common valid positions; per-recording absolute/relative amounts=median(abs(raw/residual)); work=median over recording amounts. Display=100*(2**amount-1).',
        'dynamics': 'Per-quarter mean onset velocity/127. Absolute=median over recording means. Relative=median over recording mean(abs(local residual)). Display multiply 127.',
        'pedaling': 'CC64 depth time-average/127; down ratio >=64 time fraction; on/off transition counts per quarter. Absolute=median recording means; relative=median recording mean(abs(residual)).',
        'missing_pedal': 'Absent CC64 and intervals starting before first CC64 event are unknown; never silently converted to zero usage.',
        'validity': 'Same exact quarter positions across all recordings within each channel; tau in [.08,5]; >=32 quarters and >=50% of largest map. D requires onset in every recording; pedal requires known initial state. No per-recording dropping to inflate coverage.',
        'singletons': 'Absolute may be available; relative remains missing.',
        'normalization': 'No work scale division, z-score, clipping or missing-value imputation; /127 is the established feature definition only.',
        'score_editions': 'Largest identical canonical-score cohort, ties by score hash; report all counts.',
        'balanced_works_per_era': target,
        'balanced_policy': 'Maximum 30 per era, limited by intersection of all five channels with both values; composer-count proportional quotas, min one per represented composer; SHA256 ranking seed 20261106; no value-based selection. Original within-work recordings/reference unchanged.',
        'balanced_allocations': allocations,
        'coverage': coverage, 'input_sha256': input_hashes,
        'code_sha256': {str(p): sha(p) for p in source_files},
        'extraction_audit': str(audit), 'extraction_audit_sha256': sha(audit),
        'packed_values': str(destination/'common_values.npz'), 'packed_values_sha256': sha(destination/'common_values.npz')}
    assert all(sha(Path(p)) == digest for p, digest in input_hashes.items())
    (out/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    print(json.dumps(coverage, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, default=DIAG)
    parser.add_argument('--out', type=Path, default=AI/'analysis/era_refine_expression')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    analyze(args.diagnostics.resolve(), args.out.resolve(), args.workers)
