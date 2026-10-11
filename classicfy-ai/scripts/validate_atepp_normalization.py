"""Analyze feature normalization for every quality-accepted ATEPP DualDTW performance.

Run from any directory: python validate_atepp_normalization.py --help.
Large provenance-checked stage caches remain in ATEPP_dataset/features/dual_dtw.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atepp_feature_analysis import load_records, collect_normalized, sha256
import validate_embedding as embedding
import validate_feature_search_roles as roles
from validate_interpretation_examples import vector_rows, cross_selection, draw_cross_work, profile_plot
from validate_feature_normalization import write_cache, candidate_scales, eta_squared
from validate_tempo import setup_style, chart, save, plt, BLUE, GREEN, ORANGE, GRAY, write_csv
from validate_dynamics import reference_dynamics
from validate_pedaling import reference_pedal
from features import (fit_residual_scale, standardize_sequence, extract_dynamics, extract_pedaling,
                      extract_articulation, extract_piece_tempo_features, extract_piece_rubato_features)
from features.articulation import extract_note_articulation, build_tempo_map
from preprocessing import load_midi, NoteAlignment, ScoreNote, PerformedNote

CHANNELS = embedding.CHANNELS
STAGES = ('raw', 'common', 'relative', 'standardized')
UNITS = ('log2(0.5 reference seconds / performed beat seconds)', 'log2(local speed / own median speed)',
         'mean note-on velocity / 127', 'log2(key-held duration / local expected duration)',
         'time-mean CC64 / 127', 'fraction of time CC64 >= 64', 'CC64 on/off transitions per beat')


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def finite(sequence):
    return sequence.values[sequence.mask & np.isfinite(sequence.values)]


def figure(path, title, subtitle, xlabel, ylabel, draw, note):
    fig, ax = chart(title, subtitle)
    draw(ax)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=9)
    save(fig, path, note)


def normalization_analysis(results, args, provenance):
    out = args.out / 'feature_normalization'
    out.mkdir(parents=True, exist_ok=True)
    summaries, pair_rows, cache_records, extremes, policy_rows, distributions = [], [], [], [], [], []
    pools = defaultdict(list)
    for result in results:
        records = {r['key']: r for r in result['records']}
        scales = {r['channel']: r for r in result['scales']}
        for channel in CHANNELS:
            entries = result['features'][channel]
            scale = scales[channel]
            for key, f in entries.items():
                residual_mask = f['relative'].mask & np.isfinite(f['relative'].values)
                np.testing.assert_array_equal(residual_mask, f['standardized'].mask)
                row = {'piece': result['piece'], 'channel': channel, 'key': key,
                       'composer': records[key]['composer'], 'artist': records[key]['artist'],
                       'beats': len(residual_mask), 'raw_valid': int(f['raw'].mask.sum()),
                       'relative_valid': int(residual_mask.sum()), 'standardized_valid': int(f['standardized'].mask.sum())}
                for stage in STAGES:
                    values = f[stage].values[residual_mask]
                    row[f'{stage}_mean'] = float(values.mean()) if len(values) else np.nan
                    row[f'{stage}_range'] = float(np.ptp(np.percentile(values, [5, 95]))) if len(values) else np.nan
                    pools[channel, stage].append(finite(f[stage]))
                summaries.append(row)
                cache_records.append({'piece': result['piece'], 'key': key, 'channel': channel,
                                      'score_beats': records[key]['score_beats'], 'scale': scale,
                                      'score_beat_types': ['b'] * len(records[key]['score_beats']), 'minimum_support': 2,
                                      'common_support': f['support'],
                                      **{f'{stage}_{part}': getattr(f[stage], part)
                                         for stage in STAGES for part in ('values', 'mask')}})
                indices = np.flatnonzero(residual_mask)
                for i in indices[np.argsort(np.abs(f['standardized'].values[indices]))[-3:]]:
                    extremes.append({'piece': result['piece'], 'key': key, 'channel': channel,
                                     'beat_index': int(i), 'score_position': float(records[key]['score_beats'][i]),
                                     'beat_seconds': float(np.diff(records[key]['performance_beats'])[i]),
                                     'support': int(f['support'][i]), 'scale': scale['value'],
                                     **{s: float(f[s].values[i]) for s in STAGES}})
            keys = list(entries)
            pair_error, scale_error, varying, collapsed, count = 0., 0., 0, 0, 0
            for i, key in enumerate(keys):
                a = entries[key]
                for other in keys[i + 1:]:
                    b = entries[other]
                    shared = a['relative'].mask & b['relative'].mask
                    if not shared.any():
                        continue
                    before = (a['raw'].values - b['raw'].values)[shared]
                    after = (a['relative'].values - b['relative'].values)[shared]
                    scaled = (a['standardized'].values - b['standardized'].values)[shared]
                    pair_error = max(pair_error, float(np.max(np.abs(before - after))))
                    scale_error = max(scale_error, float(np.max(np.abs(before / scale['value'] - scaled))))
                    has_difference = bool(np.any(np.abs(before) > 1e-12))
                    varying += has_difference
                    collapsed += has_difference and not np.any(np.abs(scaled) > 1e-12)
                    count += 1
            pair_rows.append({'piece': result['piece'], 'channel': channel, 'pairs': count,
                              'varying_pairs': varying, 'collapsed_pairs': collapsed,
                              'max_pair_residual_error': pair_error, 'max_pair_scale_error': scale_error})
    table = pd.DataFrame(summaries)
    table.to_csv(out / 'performance_summary.csv', index=False, encoding='utf-8-sig')
    pairs = pd.DataFrame(pair_rows)
    if pairs.max_pair_residual_error.max() > 1e-10 or pairs.max_pair_scale_error.max() > 1e-10 or pairs.collapsed_pairs.sum():
        raise AssertionError('Common removal/scaling did not preserve performance differences')
    pairs.to_csv(out / 'pair_preservation.csv', index=False)
    scales = pd.DataFrame([s for r in results for s in r['scales']])
    scales.to_csv(out / 'piece_scales.csv', index=False)
    extreme_frame = pd.DataFrame(extremes)
    extreme_frame['absolute_standardized'] = extreme_frame.standardized.abs()
    extreme_frame = extreme_frame.sort_values('absolute_standardized', ascending=False).groupby('channel').head(20)
    extreme_frame.to_csv(out / 'extreme_beats.csv', index=False)
    diagnostics = []
    for channel in CHANNELS:
        residuals = [f['relative'] for r in results for f in r['features'][channel].values()]
        global_scales = {m: fit_residual_scale(residuals, method=m) for m in ('mad', 'std')}
        values_by_policy = defaultdict(list)
        for result in results:
            fs = list(result['features'][channel].values())
            values = np.concatenate([finite(f['relative']) for f in fs])
            diagnostics.append({'piece': result['piece'], 'channel': channel, 'valid_residuals': len(values),
                                'zero_fraction': float(np.mean(values == 0)) if len(values) else np.nan,
                                **candidate_scales(values)})
            policies = {f'piece_{m}': fit_residual_scale([f['relative'] for f in fs], method=m)
                        for m in ('mad', 'iqr', 'std')}
            policies.update({f'global_{m}': s for m, s in global_scales.items()})
            for name, scale in policies.items():
                values_by_policy[name].extend(finite(standardize_sequence(f['relative'], scale)) for f in fs)
        for name, chunks in values_by_policy.items():
            values = np.abs(np.concatenate(chunks))
            policy_rows.append({'channel': channel, 'policy': name, 'valid_beats': len(values),
                                **{f'abs_p{q}': float(np.percentile(values, q)) if len(values) else np.nan
                                   for q in (95, 99, 99.9, 100)}})
        pooled = {s: np.concatenate(pools[channel, s]) for s in STAGES}
        for stage, values in pooled.items():
            distributions.append({'channel': channel, 'stage': stage, 'valid_beats': len(values),
                                  **{f'p{q}': float(np.percentile(values, q)) if len(values) else np.nan
                                     for q in (0, 1, 5, 50, 95, 99, 100)}})
        def histogram(ax):
            for stage, color in zip(STAGES, (GRAY, BLUE, ORANGE, GREEN)):
                values = pooled[stage]
                if len(values):
                    lo, hi = np.percentile(values, [.5, 99.5])
                    if lo == hi:
                        lo, hi = lo - .01, hi + .01
                    ax.hist(values, bins=np.linspace(lo, hi, 80), density=True, histtype='step', label=stage, color=color)
        figure(out / f'01_distribution_{channel}.png', f'ATEPP · {channel}: four processing stages',
               'Distribution diagnostics; each stage uses its own units', 'value', 'density', histogram,
               'Central 99% displayed only. Full tails stay in CSV/cache; no clipping or recentering.')
        def tail(ax):
            qs = np.linspace(90, 100, 101)
            for name, chunks in values_by_policy.items():
                values = np.abs(np.concatenate(chunks))
                if len(values):
                    ax.plot(qs, np.percentile(values, qs), label=name)
            ax.set_yscale('symlog', linthresh=1.)
        figure(out / f'02_policy_tails_{channel}.png', f'ATEPP · {channel}: compare scale policies',
               'MAD / IQR / SD per work versus global corpus scale', 'absolute-value percentile', '|scaled residual|', tail,
               'MAD fallback: IQR then SD then unit. Global scales are diagnostics, not held-out training fits.')
    pd.DataFrame(diagnostics).to_csv(out / 'piece_scale_candidates.csv', index=False)
    pd.DataFrame(policy_rows).to_csv(out / 'policy_comparison.csv', index=False)
    pd.DataFrame(distributions).to_csv(out / 'distributions.csv', index=False)
    identity = []
    rng = np.random.default_rng(args.seed)
    for channel, frame in table.groupby('channel'):
        for label in ('piece', 'composer'):
            for summary in ('mean', 'range'):
                for stage in ('raw', 'relative', 'standardized'):
                    values = frame[f'{stage}_{summary}']
                    identity.append({'channel': channel, 'label': label, 'summary': summary, 'stage': stage,
                                     'eta_squared': eta_squared(values, frame[label]),
                                     'shuffled_mean': float(np.mean([eta_squared(values, rng.permutation(frame[label]))
                                                                    for _ in range(100)]))})
    identity = pd.DataFrame(identity)
    identity.to_csv(out / 'identity_variance.csv', index=False)
    duration_chunks = defaultdict(list)
    for result in results:
        for record in result['records']:
            key = record['key']; seconds = np.diff(record['performance_beats'])
            f = result['features']['pedal_changes'][key]
            shared = f['relative'].mask & (seconds > 0)
            duration_chunks['seconds'].append(seconds[shared])
            for stage in ('raw', 'relative', 'standardized'):
                duration_chunks[f'{stage}_counts'].append(f[stage].values[shared])
            duration_chunks['transitions_per_second'].append(f['raw'].values[shared] / seconds[shared])
    duration = pd.DataFrame({k: np.concatenate(v) for k, v in duration_chunks.items()})
    write_csv(out / 'pedal_duration_comparison.csv', [
        {'stage': stage, 'pearson_with_log10_seconds': float(np.corrcoef(np.log10(duration.seconds), duration[stage])[0, 1]),
         'max_abs': float(duration[stage].abs().max())}
        for stage in ('raw_counts', 'relative_counts', 'standardized_counts', 'transitions_per_second')])
    def draw_duration(ax):
        subset = duration.iloc[::max(1, len(duration) // 12000)]
        ax.scatter(np.log10(subset.seconds), subset.raw_counts, s=3, alpha=.12, color=BLUE, label='raw count')
        ax.set_yscale('symlog', linthresh=1.)
    figure(out / '04_pedal_changes_duration.png', 'ATEPP · pedal counts and beat duration',
           'Counts retain original semantics; per-second rate is a separate diagnostic',
           'log10(performed quarter-note seconds)', 'CC64 on/off transitions per beat', draw_duration,
           'Display subsample only; CSV correlations use all valid beats. Short beats can amplify per-second rates.')
    for label in ('piece', 'composer'):
        for summary in ('mean', 'range'):
            def draw(ax):
                subset = identity[(identity.label == label) & (identity.summary == summary)]
                for i, stage in enumerate(('raw', 'relative', 'standardized')):
                    v = subset[subset.stage == stage].set_index('channel').reindex(CHANNELS).eta_squared
                    ax.barh(np.arange(7) + (i - 1) * .24, 100 * v, height=.22,
                            label=stage, color=(GRAY, ORANGE, GREEN)[i])
                ax.set(yticks=np.arange(7), yticklabels=CHANNELS, xlim=(0, 100))
            figure(out / f'03_{label}_{summary}_variance.png', f'ATEPP · {label} information in {summary}',
                   'Same valid residual beats used in every stage', 'variance explained by label (%)', '', draw,
                   'Descriptive eta-squared, not classification accuracy. Target recordings participate in common/scale fits.')
    args.normalized_cache.parent.mkdir(parents=True, exist_ok=True)
    write_cache(args.normalized_cache, cache_records, provenance)
    stats = {'groups': len(results), 'performances': table.key.nunique(),
             'max_pair_residual_error': float(pairs.max_pair_residual_error.max()),
             'max_pair_scale_error': float(pairs.max_pair_scale_error.max()),
             'collapsed_pairs': int(pairs.collapsed_pairs.sum()),
             'normalized_cache': str(args.normalized_cache), 'normalized_cache_sha256': sha256(args.normalized_cache),
             'clipping': None, 'recentering': False,
             'policy': 'Per-work MAD Tempo/Rubato/Dynamics/Articulation; independent SD for three pedal channels',
             'channels': {c: {'valid_beats': int(table[table.channel == c].standardized_valid.sum()),
                              'max_abs_standardized': float(extreme_frame[extreme_frame.channel == c].absolute_standardized.max())}
                          for c in CHANNELS}}
    write_json(out / 'normalization.json', stats)
    return stats, extreme_frame


def alignment_for(record, midi):
    with np.load(record['alignment_path'], allow_pickle=False) as z:
        matches = []
        for i in range(len(z['pitch'])):
            # Alignment cache includes original MIDI note indices, so velocity is traced too.
            note = midi.notes[int(z['performance_ids'][i])]
            matches.append((ScoreNote(str(z['score_ids'][i]), int(z['pitch'][i]),
                                      float(z['score_onsets'][i]), float(z['score_offsets'][i]), ()),
                            PerformedNote(str(z['performance_ids'][i]), int(z['pitch'][i]),
                                          float(z['performance_onsets'][i]), float(z['performance_offsets'][i]), note.velocity)))
    return NoteAlignment(tuple(matches), (), ())


def verify_sources(result, keys, out):
    """Re-extract real MIDI and aligned note features, not just cached arithmetic."""
    out.mkdir(parents=True, exist_ok=True)
    checks, note_rows, velocity_rows, pedal_rows = [], [], [], []
    records = {r['key']: r for r in result['records']}
    for key in keys:
        r = records[key]
        if sha256(r['midi_path']) != r['midi_sha256']:
            raise ValueError(f'{key}: MIDI changed after extraction')
        midi = load_midi(r['midi_path'])
        alignment = alignment_for(r, midi)
        d, a, p = extract_dynamics(midi, r['performance_beats']), extract_articulation(alignment, r['performance_beats']), extract_pedaling(midi, r['performance_beats'])
        _, direct_dynamics = reference_dynamics(midi, r['performance_beats'])
        np.testing.assert_allclose(d.sequence.values, direct_dynamics, atol=1e-10, rtol=0, equal_nan=True)
        direct_pedal = reference_pedal(midi, r['performance_beats'])
        for name, sequence in [('depth', p.depth), ('down_ratio', p.down_ratio), ('changes', p.changes)]:
            np.testing.assert_allclose(sequence.values[sequence.mask], direct_pedal[name][sequence.mask], atol=1e-8, rtol=0)
        actual = [d.sequence, a.sequence, p.depth, p.down_ratio, p.changes]
        for channel, sequence in zip(CHANNELS[2:], actual):
            expected = result['features'][channel][key]['raw']
            if channel.startswith('pedal') and r['pedal_events'] == 0:
                continue
            np.testing.assert_array_equal(sequence.mask, expected.mask)
            np.testing.assert_allclose(sequence.values, expected.values, atol=1e-10, rtol=0, equal_nan=True)
            checks.append({'key': key, 'channel': channel, 'max_raw_reextraction_error':
                           float(np.max(np.abs(sequence.values[sequence.mask] - expected.values[sequence.mask]))) if sequence.mask.any() else 0.})
        notes = extract_note_articulation(alignment)
        positions, times = build_tempo_map(alignment)
        for beat, (left, right) in enumerate(zip(r['performance_beats'][:-1], r['performance_beats'][1:])):
            inside = notes.values[(notes.onsets >= left) & (notes.onsets < right)]
            direct = np.median(inside) if len(inside) and right > left else np.nan
            np.testing.assert_allclose(a.sequence.values[beat], direct, rtol=0, atol=1e-10, equal_nan=True)
        raw_tempo = result['features']['tempo'][key]['raw']
        direct_tempo = np.log2(.5 / np.diff(r['performance_beats'])[raw_tempo.mask])
        np.testing.assert_allclose(raw_tempo.values[raw_tempo.mask], direct_tempo, atol=1e-10, rtol=0)
        raw_rubato = result['features']['rubato'][key]['raw']
        direct_rubato = raw_tempo.values[raw_rubato.mask] - np.median(finite(raw_tempo))
        np.testing.assert_allclose(raw_rubato.values[raw_rubato.mask], direct_rubato, atol=1e-10, rtol=0)
        checks.extend([{'key': key, 'channel': 'tempo', 'max_raw_reextraction_error':
                        float(np.max(np.abs(raw_tempo.values[raw_tempo.mask] - direct_tempo)))},
                       {'key': key, 'channel': 'rubato', 'max_raw_reextraction_error':
                        float(np.max(np.abs(raw_rubato.values[raw_rubato.mask] - direct_rubato)))}])
        for i, value in zip(notes.match_indices, notes.values):
            s, n = alignment.matches[int(i)]
            expected_seconds = float(np.interp(s.offset_beats, positions, times) - np.interp(s.onset_beats, positions, times))
            np.testing.assert_allclose(value, np.log2(n.duration / expected_seconds), atol=1e-10)
            beat = int(np.searchsorted(r['performance_beats'], n.onset, side='right') - 1)
            note_rows.append({'key': key, 'beat_index': beat, 'pitch': n.pitch, 'score_onset': s.onset_beats,
                              'score_offset': s.offset_beats, 'performance_onset': n.onset, 'performance_offset': n.offset,
                              'held_seconds': n.duration, 'expected_seconds': expected_seconds, 'note_log2': float(value)})
        for n in midi.notes:
            beat = int(np.searchsorted(r['performance_beats'], n.start, side='right') - 1)
            if 0 <= beat < len(r['score_beats']) - 1:
                velocity_rows.append({'key': key, 'beat_index': beat, 'onset_seconds': n.start,
                                      'offset_seconds': n.end, 'pitch': n.pitch, 'velocity': n.velocity})
        pedal_rows.extend({'key': key, 'time_seconds': p.time, 'cc64': p.value} for p in midi.pedals)
    write_csv(out / 'source_checks.csv', checks)
    pd.DataFrame(note_rows).to_csv(out / 'note_articulation.csv', index=False)
    pd.DataFrame(velocity_rows).to_csv(out / 'note_velocity.csv', index=False)
    pd.DataFrame(pedal_rows).to_csv(out / 'cc64_events.csv', index=False)
    return checks, pd.DataFrame(note_rows), pd.DataFrame(velocity_rows), pd.DataFrame(pedal_rows)


def draw_pedal_source_examples(result, pedals, args):
    """Choose the 32-beat source window with the most real on/off transitions."""
    candidates = []
    for record in result['records']:
        key = record['key']
        pp = pedals[pedals.key == key]
        if not len(pp):
            continue
        on = (pp.cc64 >= 64).to_numpy()
        transitions = pp.time_seconds.to_numpy()[on != np.r_[False, on[:-1]]]
        pb = record['performance_beats']
        indices = np.searchsorted(pb, transitions, side='right') - 1
        indices = indices[(indices >= 0) & (indices < len(pb) - 1)]
        counts = np.bincount(indices, minlength=len(pb) - 1)
        width = min(32, len(counts))
        windows = np.convolve(counts, np.ones(width, dtype=int), mode='valid')
        start = int(np.argmax(windows))
        candidates.append((-int(windows[start]), key, start, width, record, pp))
    if not candidates:
        raise ValueError('No CC64 source traces available for the example')
    count, key, start, width, record, pp = min(candidates, key=lambda item: item[:3])
    stop = start + width; pb = record['performance_beats']
    events = pp[(pp.time_seconds >= pb[start]) & (pp.time_seconds < pb[stop])]
    initial = pp[pp.time_seconds <= pb[start]].tail(1)
    initial_value = float(initial.cc64.iloc[0]) if len(initial) else 0.
    times = np.r_[pb[start], events.time_seconds.to_numpy(), pb[stop]]
    signal = np.r_[initial_value, events.cc64.to_numpy(), float(events.cc64.iloc[-1]) if len(events) else initial_value]
    for channel in CHANNELS[4:]:
        out = args.out / 'pedaling' / channel.removeprefix('pedal_')
        def draw(ax):
            if channel == 'pedal_changes':
                on = (pp.cc64 >= 64).to_numpy()
                tt = pp.time_seconds.to_numpy()[on != np.r_[False, on[:-1]]]
                tt = tt[(tt >= pb[start]) & (tt < pb[stop])]
                ax.scatter(tt, np.ones(len(tt)), color=GRAY, marker='|', s=120, label='each on/off transition')
            else:
                ss = signal / 127 if channel == 'pedal_depth' else (signal >= 64).astype(float)
                ax.step(times, ss, where='post', color=GRAY,
                        label='actual CC64 / 127' if channel == 'pedal_depth' else 'actual CC64 >= 64')
            values = result['features'][channel][key]['raw'].values[start:stop]
            ax.step(pb[start:stop + 1], np.r_[values, values[-1]], where='post', color=BLUE, label='extracted beat feature')
        figure(out / '01_extraction.png', f'ATEPP · {channel}: signal to beat feature',
               f'{key} · {record["artist"]} | score positions {record["score_beats"][start]:g}–{record["score_beats"][stop]:g}',
               'actual MIDI seconds', UNITS[CHANNELS.index(channel)], draw,
               'Source window maximizes on/off transitions among the three example MIDI traces; curves 02–04 keep their fixed central window.')
    case = {'key': key, 'start_score_position': float(record['score_beats'][start]),
            'stop_score_position': float(record['score_beats'][stop]), 'start_seconds': float(pb[start]),
            'stop_seconds': float(pb[stop]), 'on_off_transitions': -count,
            'selection': 'maximum real on/off transitions over every 32-beat window of the three example MIDI traces; ties by key/window'}
    write_csv(args.out / 'source_midi/pedal_source_window.csv', [case])
    return case


def feature_examples(result, args):
    records = result['records']
    order = sorted(records, key=lambda r: float(finite(result['features']['tempo'][r['key']]['relative']).mean()))
    selected = [order[int(round(q * (len(order) - 1)))]['key'] for q in (.25, .5, .75)]
    # Quartiles are distinct because eligible groups contain at least five recordings.
    labels = {r['key']: f"{r['key']} · {r['artist']}" for r in records}
    total = len(records[0]['score_beats']) - 1
    start = (total - min(32, total)) // 2
    section = slice(start, start + min(32, total))
    x = records[0]['score_beats'][:-1][section]
    subtitle = f"{records[0]['track'][:85]} | {len(records)} performances"
    rows, pair_rows = [], []
    for channel, unit in zip(CHANNELS, UNITS):
        out = args.out / ('pedaling/' + channel.removeprefix('pedal_') if channel.startswith('pedal_') else channel)
        out.mkdir(parents=True, exist_ok=True)
        fs = result['features'][channel]
        for stage, number in [('raw', '02'), ('relative', '03'), ('standardized', '04')]:
            def draw(ax):
                for key, color in zip(selected, (BLUE, GRAY, ORANGE)):
                    f = fs[key][stage]
                    ax.plot(x, np.where(f.mask, f.values, np.nan)[section], color=color, label=labels[key])
                if stage == 'raw':
                    common = fs[selected[0]]['common']
                    ax.plot(x, common.values[section], color='#242b27', ls='--', label='position-wise common median')
                else:
                    ax.axhline(0, color='#242b27', ls=':')
            figure(out / f'{number}_{stage}.png', f'ATEPP · {channel}: {stage}', subtitle,
                   'score quarter-note position', 'dimensionless scaled residual' if stage == 'standardized' else unit, draw,
                   'Same three performances and score window; all performances fit common/scale. No interpolation or clipping.')
        for statistic, number in [('mean', '05'), ('range', '06')]:
            def draw(ax):
                for j, stage in enumerate(('raw', 'relative', 'standardized')):
                    vals = [finite(fs[r['key']][stage]) for r in order]
                    values = [float(v.mean()) if statistic == 'mean' else float(np.ptp(np.percentile(v, [5, 95]))) for v in vals]
                    ax.plot(np.arange(len(order)), values, marker='o', ms=4, label=stage, color=(GRAY, ORANGE, GREEN)[j])
                ax.set(xticks=np.arange(len(order)), xticklabels=[r['key'] for r in order])
                ax.tick_params(axis='x', rotation=60)
            figure(out / f'{number}_performance_{statistic}.png', f'ATEPP · {channel}: performance {statistic}', subtitle,
                   'performance ID (ordered by relative Tempo)', 'mean value' if statistic == 'mean' else 'p95 - p5', draw,
                   'Stage units differ; use curves/CSV for direct interpretation. Lines connect IDs, not musical time.')
        for i, r in enumerate(records):
            f = fs[r['key']]
            for beat in range(total):
                rows.append({'key': r['key'], 'channel': channel, 'beat_index': beat,
                             'score_position': float(r['score_beats'][beat]), 'performance_start': float(r['performance_beats'][beat]),
                             'performance_end': float(r['performance_beats'][beat + 1]),
                             'support': int(f['support'][beat]), **{s: float(f[s].values[beat]) for s in STAGES},
                             **{f'{s}_mask': bool(f[s].mask[beat]) for s in STAGES}})
            for rr in records[i + 1:]:
                g = fs[rr['key']]; shared = f['relative'].mask & g['relative'].mask
                if shared.any():
                    before = float(np.mean(np.abs((f['raw'].values - g['raw'].values)[shared])))
                    after = float(np.mean(np.abs((f['relative'].values - g['relative'].values)[shared])))
                    pair_rows.append({'channel': channel, 'a': r['key'], 'b': rr['key'], 'raw_difference': before, 'relative_difference': after})
        pairs = [r for r in pair_rows if r['channel'] == channel]
        def draw(ax):
            a, b = [r['raw_difference'] for r in pairs], [r['relative_difference'] for r in pairs]
            ax.scatter(a, b, color=BLUE, s=28)
            bound = max([.01, *a, *b]) * 1.1
            ax.plot([0, bound], [0, bound], color=GRAY, ls='--', label='unchanged pair difference')
        figure(out / '07_pair_preservation.png', f'ATEPP · {channel}: pair differences survive', subtitle,
               'mean absolute raw difference', 'mean absolute relative difference', draw,
               'Same valid beat intersection for both stages; diagonal equality is numerical preservation, not preference accuracy.')
    write_csv(args.out / 'example_beat_features.csv', rows)
    write_csv(args.out / 'example_pair_distances.csv', pair_rows)
    checks, notes, velocities, pedals = verify_sources(result, selected, args.out / 'source_midi')
    key = selected[1]; record = next(r for r in records if r['key'] == key)
    pb = record['performance_beats']
    velocity = velocities[(velocities.key == key) & velocities.beat_index.between(start, section.stop - 1)]
    note = notes[(notes.key == key) & notes.beat_index.between(start, section.stop - 1)]
    for channel, unit in zip(CHANNELS, UNITS):
        out = args.out / ('pedaling/' + channel.removeprefix('pedal_') if channel.startswith('pedal_') else channel)
        def draw(ax):
            if channel == 'tempo':
                durations = np.diff(pb)[section]
                ax.scatter(durations, result['features'][channel][key]['raw'].values[section], color=BLUE, label='real aligned beats')
                xx = np.linspace(durations.min(), durations.max(), 100)
                ax.plot(xx, np.log2(.5 / xx), color=ORANGE, label='log2(0.5 / seconds)')
            elif channel == 'rubato':
                raw = result['features']['tempo'][key]['raw']
                own = float(np.median(finite(raw)))
                ax.plot(x, raw.values[section], label='raw Tempo', color=BLUE)
                ax.axhline(own, color=GRAY, ls='--', label='own median Tempo')
                ax.plot(x, result['features']['rubato'][key]['raw'].values[section], label='Tempo - own median', color=ORANGE)
            elif channel == 'dynamics':
                ax.scatter(velocity.beat_index + record['score_beats'][0], velocity.velocity / 127,
                           s=10, alpha=.5, color=GRAY, label='individual note-on velocity / 127')
                ax.plot(x, result['features'][channel][key]['raw'].values[section], color=BLUE, label='beat mean')
            elif channel == 'articulation':
                ax.scatter(note.expected_seconds, note.held_seconds, color=BLUE, s=12, alpha=.5)
                bound = max(note.expected_seconds.max(), note.held_seconds.max())
                ax.plot([0, bound], [0, bound], color=GRAY, ls='--', label='held = expected')
            else:
                pp = pedals[pedals.key == key]
                at = pp[(pp.time_seconds >= pb[start]) & (pp.time_seconds <= pb[section.stop])]
                initial = pp[pp.time_seconds <= pb[start]].tail(1)
                events = pd.concat([initial.assign(time_seconds=pb[start]), at]).sort_values('time_seconds')
                if len(events):
                    if channel == 'pedal_changes':
                        changed = (pp.cc64 >= 64).to_numpy() != np.r_[False, (pp.cc64 >= 64).to_numpy()[:-1]]
                        transitions = pp[changed]
                        transitions = transitions[(transitions.time_seconds >= pb[start]) & (transitions.time_seconds < pb[section.stop])]
                        ax.scatter(transitions.time_seconds, np.ones(len(transitions)), color=GRAY, marker='|', s=120, label='each on/off transition')
                    else:
                        signal = (events.cc64 / 127).to_numpy() if channel == 'pedal_depth' else (events.cc64 >= 64).to_numpy().astype(float)
                        ax.step([*events.time_seconds, pb[section.stop]], [*signal, signal[-1]],
                                where='post', color=GRAY, label='CC64 / 127 signal' if channel == 'pedal_depth' else 'CC64 >= 64 signal')
                values = result['features'][channel][key]['raw'].values[section]
                ax.step(pb[start:section.stop + 1], np.r_[values, values[-1]], where='post', color=BLUE,
                        label='beat mean depth' if channel == 'pedal_depth' else 'beat on-time fraction' if channel == 'pedal_down_ratio' else 'transitions per beat')
        figure(out / '01_extraction.png', f'ATEPP · {channel}: trace the source', f'{key} | {record["artist"]}',
               'performance beat seconds' if channel == 'tempo' else 'expected note seconds' if channel == 'articulation'
               else 'MIDI seconds' if channel.startswith('pedal_') else 'score quarter-note position',
               'key-held seconds' if channel == 'articulation' else unit,
               draw, 'Features were re-extracted from MIDI / full DualDTW note alignment and compared against raw caches.')
    def draw_note_median(ax):
        ax.scatter(note.beat_index + record['score_beats'][0], note.note_log2, s=10, alpha=.5, color=GRAY,
                   label='aligned note log2(held / expected)')
        ax.plot(x, result['features']['articulation'][key]['raw'].values[section], color=BLUE, label='beat median')
    figure(args.out / 'articulation/08_notes_to_beat.png', 'ATEPP · Articulation: notes to beat median', subtitle,
           'score quarter-note position', 'log2(key-held / local expected duration)', draw_note_median,
           'Notes assigned by actual performance onset into half-open beat windows; the median is checked independently.')
    pedal_case = draw_pedal_source_examples(result, pedals, args)
    shifted_tempo = extract_piece_tempo_features([replace(t, performance_beats=np.asarray(t.performance_beats) / 1.2)
                                                   for t in result['tempo_inputs']])
    shifted_rubato = extract_piece_rubato_features(shifted_tempo)
    max_error = 0.
    for key in result['rubato']:
        for stage in ('absolute', 'common', 'relative'):
            a = getattr(result['rubato'][key], f'{stage}_rubato_sequence')
            b = getattr(shifted_rubato[key], f'{stage}_rubato_sequence')
            np.testing.assert_array_equal(a.mask, b.mask)
            np.testing.assert_allclose(a.values, b.values, rtol=0, atol=1e-10, equal_nan=True)
            max_error = max(max_error, float(np.max(np.abs(a.values[a.mask] - b.values[b.mask]))))
    key = selected[1]
    def draw(ax):
        ax.plot(x, result['rubato'][key].absolute_rubato_sequence.values[section], color=BLUE, lw=3, label='original')
        ax.plot(x, shifted_rubato[key].absolute_rubato_sequence.values[section], color=ORANGE, ls='--', label='whole performance 1.2x faster')
    figure(args.out / 'rubato/08_speed_invariance.png', 'ATEPP · Rubato: global speed invariance', subtitle,
           'score quarter-note position', 'absolute Rubato (log2)', draw,
           'Synthetic speed change, not a new recording. Absolute/common/relative Rubato checked for every example performance.')
    return {'piece': result['piece'], 'track': records[0]['track'], 'performances': len(records),
            'selected_keys': selected, 'score_window': [float(x[0]), float(x[-1])],
            'source_checks': checks, 'rubato_speed_invariance_max_error': max_error, 'pedal_source_window': pedal_case}


def embedding_analysis(groups, audits, scales, args):
    out = args.out / '_embedding_validation'; out.mkdir(parents=True, exist_ok=True)
    table, vectors, examples, contributions = embedding.evaluate(groups, args)
    summary = embedding.aggregate(table, args.seed)
    paired = embedding.paired_comparisons(table, args.seed)
    example = embedding.plot_results(table, summary, examples, contributions, out)
    for name, data in [('piece_results', table.to_dict('records')), ('summary', summary.to_dict('records')),
                       ('embeddings', vectors), ('collection', audits), ('scales', scales),
                       ('distance_contributions', contributions), ('paired_comparisons', paired),
                       ('example_distances', example['distances'])]:
        write_csv(out / f'{name}.csv', data)
    return {'groups': len(groups), 'performances': len(vectors), 'dimensions': 14, 'blocks': embedding.BLOCKS,
            'conditions': embedding.CONDITIONS, 'trials': args.trials, 'seed': args.seed,
            'summary': summary.to_dict('records'), 'example': example}


def search_analysis(groups, args, anchor_key):
    out = args.out / '_feature_search_roles'; out.mkdir(parents=True, exist_ok=True)
    rows, vectors = vector_rows(groups)
    primary = roles.evaluate(vectors, rows)
    alternative = roles.evaluate(vectors, rows, means_only=True)
    metrics, held = primary[0] + alternative[0], primary[1] + alternative[1]
    wm, wh, summary = roles.aggregate(metrics, held, args.seed, args.bootstrap_samples)
    roles.draw_global(summary, out / 'global', len(groups), len(rows))
    examples, advantages = roles.draw_examples(vectors, rows, metrics, *primary[2:5], anchor_key, out / 'examples')
    held_examples = roles.draw_heldout_examples(rows, held, primary[2], out / 'heldout')
    for name, data in [('search_changes', metrics), ('heldout_agreement', held), ('summary', summary),
                       ('examples', advantages), ('work_search_changes', wm.to_dict('records')),
                       ('work_heldout_agreement', wh.to_dict('records')),
                       ('embeddings', [{**r, **{f'coordinate_{j}': float(v) for j, v in enumerate(vec)}}
                                       for r, vec in zip(rows, vectors)])]:
        write_csv(out / f'{name}.csv', data)
    a = next(i for i, r in enumerate(rows) if r['key'] == anchor_key)
    pieces = np.array([r['piece'] for r in rows])
    b, c, distances = cross_selection(vectors, pieces, a)
    cross = args.out / '_interpretation_embedding/cross_work'; cross.mkdir(parents=True, exist_ok=True)
    same = args.out / '_interpretation_embedding/same_work'; same.mkdir(parents=True, exist_ok=True)
    source_indices = np.flatnonzero(pieces == pieces[a])
    ordered = sorted(source_indices, key=lambda i: (vectors[i, 0], rows[i]['key']))
    chosen = [ordered[int(round(q * (len(ordered) - 1)))] for q in (.25, .5, .75)]
    profile_plot(chosen, vectors, pieces, [rows[i]['key'] for i in chosen], [BLUE, GRAY, ORANGE],
                 same / '01_profile.png', 'ATEPP: three performances of the same work',
                 'Display percentiles only; retrieval uses original 14 coordinates')
    write_csv(same / 'profiles.csv', [{**rows[i], 'coordinate': j, 'value': float(value)}
                                     for i in chosen for j, value in enumerate(vectors[i])])
    profile_plot([a, b, c], vectors, pieces, [f'{role} · {rows[i]["key"]}' for role, i in zip('ABC', [a, b, c])],
                 [BLUE, GREEN, GRAY], cross / '01_profile.png', 'ATEPP: compare interpretations across works',
                 'A: query; B: nearest other work; C: farthest within B work')
    draw_cross_work([a, b, c], vectors, rows, distances, cross)
    ranking = sorted(np.flatnonzero(pieces != pieces[a]), key=lambda i: (distances[i], rows[i]['key']))
    write_csv(cross / 'retrieval_candidates.csv', [{'rank': rank, **rows[i], 'distance': float(distances[i])}
                                                 for rank, i in enumerate(ranking, 1)])
    return {'works': len(groups), 'queries': len(rows), 'dimensions': 14, 'alternative_dimensions': 7,
            'summary': summary, 'examples': examples, 'heldout_examples': held_examples,
            'anchor_key': anchor_key, 'cross_work_example': [rows[i] for i in (a, b, c)],
            'bootstrap_samples': args.bootstrap_samples, 'seed': args.seed,
            'ties': 'Midranks; fractional Top5; nearest set membership; heldout ties = 0.5',
            'aggregation': 'Source work/grid macro average; bootstrap source work/grid; fixed candidate gallery'}


def reports(args, stats):
    root = args.out
    n, e, s, example = [stats[k] for k in ('normalization', 'embedding', 'search', 'example')]
    identity = pd.read_csv(root / 'feature_normalization/identity_variance.csv')
    candidates = pd.read_csv(root / 'feature_normalization/piece_scale_candidates.csv')
    policies = pd.read_csv(root / 'feature_normalization/policy_comparison.csv')
    fitted_scales = pd.read_csv(root / 'feature_normalization/piece_scales.csv')
    policy_table = []
    for channel in CHANNELS:
        cc = candidates[candidates.channel == channel]
        pp = policies[policies.channel == channel].set_index('policy')
        ff = fitted_scales[fitted_scales.channel == channel]
        chosen = 'mad' if channel in CHANNELS[:4] else 'std'
        policy_table.append(f'| {channel} | {(cc.mad == 0).sum()}/{len(cc)} | {(cc.iqr == 0).sum()}/{len(cc)} | {ff.value.median():.5g} | {pp.loc["piece_mad", "abs_p100"]:.3g} | {pp.loc["piece_std", "abs_p100"]:.3g} | {chosen.upper()} |')
    comparison = []
    for channel in CHANNELS:
        frame = identity[(identity.channel == channel) & (identity.label == 'piece') & (identity.summary == 'mean')].set_index('stage')
        comparison.append(f'| {channel} | {frame.loc["raw", "eta_squared"]:.4f} | {frame.loc["relative", "eta_squared"]:.4f} | {frame.loc["standardized", "eta_squared"]:.4f} |')
    ten = next(r for r in e['summary'] if r['pattern'] == 'random' and r['drop_fraction'] == .1 and r['config'] == 'all')
    search_table = '\n'.join(f'| {r["condition"]} | {100*r["macro_value"]:.2f}% | {100*r["ci_low"]:.2f}–{100*r["ci_high"]:.2f}% |'
                             for r in s['summary'] if r['mode'] == 'full' and r['metric'] == 'neighbor_changed' and r['condition'].startswith('without_'))
    readme = f'''# ATEPP Feature normalization 및 각 Feature 분석

ATEPP 연주 피처의 공통 패턴 제거·표준화·14차원 embedding과 결측 안정성·검색 역할을 분석한다.

## 데이터와 계산 조건

ATEPP-1.2의 품질 승인 DualDTW 연주 {stats['aligned_performances']:,}개에서 {n['groups']}개 동일 작품/악보 그룹, {n['performances']:,}개 연주를 정규화했다. 전체 모집단 상태와 input/code SHA256은 [provenance.json](provenance.json)에 있다. 단독 연주와 정확한 공통 score 구간이 없는 그룹은 [cohort_audit.csv](cohort_audit.csv)에 이유를 남긴다. 서로 다른 악보는 파일 SHA256으로 나눈다. 연주 ID의 앞자리 0을 보존한다.

ASAP의 수동 beat annotation/robust match 대신 DualDTW 자동 note alignment 품질 승인을 사용한다. 악보 quarter-note 위치의 **정확한 교집합**을 사용하며 보간하지 않는다. 반복·박자 종류는 ATEPP 자동 grid에 따로 주어지지 않으므로 regular beat `b`로 취급한다. Tempo 기준은 기존 ATEPP 코드와 동일한 120 BPM(quarter note 0.5초)이다. 이는 악보의 실제 지시 빠르기라고 해석하지 않는다. MIDI에서 CC64가 없으면 세 페달 채널을 결측 처리한다. MIDI velocity·note-off·CC64는 전사 데이터이며 실제 음압·잔향·페달 센서 측정과 구분해야 한다.

```text
D/A/P common[i] = median(valid performances at same score position), support >= 2
relative[p,i] = raw[p,i] - common[i]
Tempo relative = existing individual_tempo_sequence
Rubato raw = Tempo - own median Tempo; relative = raw - common Rubato
standardized = relative / per-work channel scale
MAD: 1.4826 * median(abs(R - median(R))); IQR: (q75-q25)/1.3489795003921634
SD: std(R, ddof=0); MAD degeneracy: IQR -> SD -> unit; no values -> NaN/mask=False
```

Tempo/Rubato/Dynamics/Articulation은 MAD, 세 Pedaling 지표는 각각 SD를 쓴다. [후보 scale](feature_normalization/piece_scale_candidates.csv), [작품별/전체 scale tail 비교](feature_normalization/policy_comparison.csv), [실제 scale/fallback](feature_normalization/piece_scales.csv), [전체 분포](feature_normalization/distributions.csv)에서 ATEPP의 분포와 퇴화 여부를 확인할 수 있다. 재중앙화·clipping·winsorization은 없다. 단위가 다른 단계의 분포 곡선을 직접 같은 수치로 해석하지 않는다.

ATEPP에서 직접 측정한 scale 진단:

| 채널 | MAD=0 그룹 | IQR=0 그룹 | 실제 scale 중앙값 | MAD fallback 최대 절댓값 | SD 최대 절댓값 | 적용 정책 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(policy_table)}

MAD는 중심부의 편차를 맞추지만 긴 꼬리가 남는다. 페달은 경계값·exact zero·정수 count가 많아 MAD가 퇴화하거나 양수여도 작아질 수 있다. 표의 tail 비교로 해당 문제를 확인하며 SD는 작품 내 전체 spread를 반영한다. 선택한 표준화 정책을 ATEPP 데이터에서 평가한다. 큰 표준화값은 삭제하지 않으며 원자료 재추출과 정렬 확인 대상으로 남긴다. 극단값의 재추출 일치는 음악적으로 타당한 해석이라는 판정과 구분한다.

## 다섯 피처의 추출·공통 제거 전후·연주 차이

대표 작품은 **{example['track']}**, {example['performances']}개 연주다. 적격 그룹 중 연주 수의 중앙값에 가장 가까운 그룹을 piece key 순으로 결정했고, Tempo 편차의 25/50/75백분위 연주 {', '.join(example['selected_keys'])}를 표시했다. 평균/변화 폭과 pair 검사는 해당 작품의 모든 연주를 사용한다. 상세 CSV에 artist와 MIDI ID를 대응시킨다.

| 피처 | 추출값 | 분석 |
| --- | --- | --- |
| [Tempo](tempo/README.md) | log2(0.5초 / 실제 quarter-note 시간) | 실제 beat 시간 대조, 공통/편차/표준화, 빠르기와 연주 쌍 차이 |
| [Rubato](rubato/README.md) | 자기 전체 Tempo 중앙값을 뺀 국소 변화 | 공통 변화 제거, 변화 폭, 전체 1.2배 속도 불변성 |
| [Dynamics](dynamics/README.md) | note-on velocity/127의 beat 평균 | note→beat, 공통/편차/표준화, 평균 강약/변화 폭 |
| [Articulation](articulation/README.md) | log2(key-held / tempo-map 기대 음 길이) | 실제/기대 음 길이, note→beat 중앙값, 연결/분리 성향 |
| [Pedaling](pedaling/README.md) | CC64 시간 평균 깊이·on 비율·전환 횟수 | 실제 CC64, 세 하위 지표 각각 공통 제거/SD/연주 차이 |

[example_beat_features.csv](example_beat_features.csv)는 네 단계 값·mask·support·score 위치·연주 시각을 보존한다. [source_midi](source_midi/README.md)의 note/CC64 및 재추출 검산과 [극단값 검산](feature_normalization/extreme_source_checks.csv)을 함께 읽는다. Rubato 전체 속도 불변성 최대 오차는 {example['rubato_speed_invariance_max_error']:.3g}다.

## 공통 제거와 단위 맞춤 결과

모든 같은 작품 연주 쌍의 같은 유효 beat에서 `raw_A-raw_B = relative_A-relative_B` 최대 오차는 **{n['max_pair_residual_error']:.3g}**, 표준화 관계 오차는 **{n['max_pair_scale_error']:.3g}**다. 원래 차이가 있던 연주 쌍의 붕괴는 **{n['collapsed_pairs']}개**다. 마스크와 유효 beat 수가 보존되는 것을 검사했다. [pair_preservation.csv](feature_normalization/pair_preservation.csv).

연주별 평균값의 작품 설명력 η²(0–1):

| 채널 | raw | relative | standardized |
| --- | --- | --- | --- |
{chr(10).join(comparison)}

![작품 평균 설명력](feature_normalization/03_piece_mean_variance.png)

평균 설명력 감소를 작품 정보의 완전 제거로 해석하지 않는다. 변화 폭·작곡가의 설명력 및 label shuffle 100회는 [identity_variance.csv](feature_normalization/identity_variance.csv)에 있다. common과 scale은 대상 연주를 포함해 fit하므로 독립 일반화 결과가 아니다.

## 결측 안정성과 통합 embedding

{e['groups']}개 작품/grid·{e['performances']:,}개 연주에서 모든 후보×7채널의 공통 유효 beat를 사용한다. 작품당 최소 {args.min_performances}개 연주, 최소 {args.min_beats}개 공통 beat, 공통 coverage ≥ {args.min_coverage:.0%}를 적용한다. mask/count/duration은 embedding에 넣지 않는다.

7채널×(signed mean, p95−p5)의 14좌표이며 Rubato의 첫 좌표만 median(abs)다. Tempo·Rubato·Dynamics·Articulation·Pedaling 다섯 묶음은 같은 가중치이고 Pedaling의 6좌표는 묶음 내 평균한다. 결측이 생겨도 common/scale/gallery를 재fit하지 않는다.

원본 복구 실험은 무결측, 무작위 5/10/20/40% 결측, 연속 20% 결측, 조건당 {args.trials}회다. feature 단독/제외, 성향만, D/A 변화 폭 제외를 모두 비교한다. 무작위 10% 결측의 전체 feature 작품 macro Top1은 **{ten['macro_top1']:.2%}** (작품 bootstrap 95% 구간 {ten['ci_low']:.2%}–{ten['ci_high']:.2%}, 무작위 기대 {ten['macro_chance']:.2%})다.

![결측 안정성](_embedding_validation/04_dropout_stability.png)

![거리 기여도](_embedding_validation/07_feature_distance_contribution.png)

[_embedding_validation](_embedding_validation/README.md)의 9개 그림, 조건별 요약·연주별 embedding·작품별 결과·paired ablation·거리 기여 CSV를 제공한다. 같은 녹음의 beat 일부가 빠진 경우의 복구이며 새로운 녹음/청취 선호의 정확도와는 다르다.

## 다른 작품 검색에서 feature별 역할

동일한 후보 gallery로 full 14좌표와 성향만 7좌표를 비교했다. 개별 feature 단독/제외와 Tempo+Rubato 제외, Dynamics+Articulation 제외를 모두 분석한다. 동일 작품/악보 후보는 제외한다. nearest 동점은 집합 포함 여부, 순위는 midrank, Top5 경계는 fractional membership으로 처리한다.

| 제외 조건 | 최근접 후보 변경률(작품 macro) | bootstrap 95% 구간 |
| --- | --- | --- |
{search_table}

![후보 변화](_feature_search_roles/global/01_neighbor_changes.png)

검색에서 제외한 feature로 후보의 같은 작품 내 다른 연주와 비교해 heldout 일치도를 계산한다(동점=0.5, 무작위 기준=0.5). [heldout_agreement.csv](_feature_search_roles/heldout_agreement.csv), [summary.csv](_feature_search_roles/summary.csv), 대표 및 중간 사례 figure를 제공한다. 이 지표는 feature 간 일치이며 음악적 정답 판정이 아니다. 작품 macro bootstrap {args.bootstrap_samples}회이며 후보 작품이 겹치는 의존성이 남는다.

[다른 작품 A/B/C 사례](_interpretation_embedding/cross_work/README.md)는 A와 전체 다른 작품 후보의 최근접 B, B 작품 내 가장 먼 C를 성향·폭·거리 기여로 비교한다. 백분위 profile은 표시용이며 검색 입력은 원래 14좌표다. C는 의도적으로 먼 대조 사례다.

## 재현

저장소 루트 PowerShell:

```powershell
.venv/Scripts/python.exe classicfy-ai/scripts/validate_atepp_normalization.py
$env:PYTHONPATH = "classicfy-ai/src"
.venv/Scripts/python.exe -m unittest discover -s classicfy-ai/tests -q
```

기본 입력 `classicfy-ai/analysis/atepp_research/midi_audit.csv`, 원본/정렬 cache `../ATEPP_dataset/features/dual_dtw/raw`, `../ATEPP_dataset/alignments/dual_dtw`를 사용한다. `--root`, `--audit`, `--out`, `--normalized-cache`, `--seed`, `--trials`, `--bootstrap-samples`, 최소 모집단 조건을 지정할 수 있다. `--stages`로 normalization/features/embedding/search만 다시 실행할 수도 있다. 파일 경로는 script 위치를 기준으로 하므로 CWD와 무관하다.

네 단계 전체 cache는 저장소 밖 `{args.normalized_cache}`에 저장된다. `np.load(..., allow_pickle=False)`의 manifest record index와 `i/raw_values`, `i/raw_mask`, `i/common_values`, `i/common_support`, `i/relative_values`, `i/standardized_values` 등을 대응시킨다. 입력 raw/정렬 cache를 덮어쓰지 않는다. source hash와 생성 artifact hash는 [provenance.json](provenance.json), [artifact_manifest.csv](artifact_manifest.csv)에 기록한다.

공통값·scale은 corpus 내 기술 분석이다. 독립 평가/새 연주에서는 reference와 scale을 학습 집합에 fit해 고정해야 한다. 자동 정렬 오류, note 밀도, 전사 편향, 작품당 연주 수, 극단값, 중복 MIDI의 영향이 남을 수 있다. 검색 후보 변경과 녹음 복구율은 추천의 정확도나 사용자 선호 예측을 증명하지 않는다.
'''
    (root / 'README.md').write_text(readme, encoding='utf-8')
    for channel, unit in zip(CHANNELS, UNITS):
        folder = root / ('pedaling/' + channel.removeprefix('pedal_') if channel.startswith('pedal_') else channel)
        source_link = '../../source_midi/README.md' if channel.startswith('pedal_') else '../source_midi/README.md'
        content = f'''# ATEPP {channel}

대표 작품: {example['track']} ({example['performances']}연주). 선택 연주: {', '.join(example['selected_keys'])}.
원본 단위: `{unit}`. 좌표는 악보 quarter-note 위치다. 02~04는 동일한 연주/구간이며 모든 연주로 공통값과 scale을 fit한다.

1. `01_extraction.png`: 실제 MIDI/정렬 원자료에서 추출값으로 가는 과정. Articulation은 실제/기대 note 길이, Dynamics는 개별 velocity와 beat 평균, Pedaling은 CC64 신호(세 하위 지표의 raw 산출값은 02와 beat CSV).
2. `02_raw.png`: 원본과 작품 위치별 공통값. 상대 feature를 해석하려면 먼저 이 두 곡선을 비교한다.
3. `03_relative.png`: 공통값을 뺀 연주별 편차. 양수/음수는 해당 위치 공통값보다 크거나 작다.
4. `04_standardized.png`: 편차를 작품별 scale로 나눈 값. 단위 없는 비교값이며 같은 숫자가 원본에서 같은 절대 크기를 뜻하지 않는다.
5. `05_performance_mean.png`: 모든 연주의 평균 성향. 단계별 단위가 다르므로 세 단계의 숫자 크기를 곧바로 비교하지 않는다.
6. `06_performance_range.png`: 모든 연주의 p95−p5 변화 폭. 작품 정보가 여기에는 남을 수 있다.
7. `07_pair_preservation.png`: 같은 유효 위치의 연주 쌍 차이. 대각선은 공통 제거 전후 수치 보존을 뜻한다.

![추출](01_extraction.png)

![공통 제거 전](02_raw.png)

![공통 제거 후](03_relative.png)

![표준화](04_standardized.png)

원자료 검산은 [source_midi]({source_link}), 전부의 beat 값·mask·support는 상위 `example_beat_features.csv`, scale/fallback은 `feature_normalization/piece_scales.csv`를 확인한다. 원본 재추출 일치는 자동 정렬의 음악적 정확성 보장이 아니다.
'''
        if channel == 'rubato':
            content += f'\n![전체 빠르기 불변성](08_speed_invariance.png)\n\n전체 시간을 1/1.2로 바꾼 실험에서 absolute/common/relative Rubato 최대 오차 {example["rubato_speed_invariance_max_error"]:.3g}. 새로운 실제 녹음이 아니다.\n'
        if channel == 'articulation':
            content += '\n![note→beat 중앙값](08_notes_to_beat.png)\n\n개별 note의 log2 길이 비율을 연주 onset 기준 half-open beat에 배정한 중앙값이다. 독립 note 필터/중앙값 검산을 수행했다.\n'
        if channel.startswith('pedal_'):
            content += '\n01 추출 그림은 선택한 세 연주의 모든 32-beat 창 중 실제 on/off 전환이 가장 많은 창을 사용한다(동점은 MIDI ID/앞선 위치 순). 이는 신호→beat 집계 과정을 보여주기 위한 선택이다. 02~04의 중앙 고정 창과 구분하며 `source_midi/pedal_source_window.csv`에 정확한 시각과 위치를 기록한다.\n'
        (folder / 'README.md').write_text(content, encoding='utf-8')
    (root / 'pedaling/README.md').write_text('# ATEPP Pedaling\n\n[Depth](depth/README.md), [Down ratio](down_ratio/README.md), [Changes](changes/README.md)는 각각 별도 SD scale이다. CC64 >=64를 on으로 판단한다. 전환 횟수는 beat당 count이며 초당 횟수로 바꾸지 않는다. CC64가 없는 MIDI는 0 대신 결측으로 처리한다.\n', encoding='utf-8')
    details = {
        'feature_normalization': '작품별 MAD/IQR/SD 후보·pooled global scale·tail·분포·mask·연주 쌍 보존·작품/작곡가 mean/range η²를 분석한다. 그림의 중앙 표시 제한은 데이터 clipping이 아니다.',
        'source_midi': 'note_velocity.csv는 실제 note-on velocity, note_articulation.csv는 실제/기대 note 길이와 log2 ratio, cc64_events.csv는 실제 페달 신호다. source_checks.csv는 full alignment로 재추출한 D/A/P와 raw cache의 오차다.',
        '_embedding_validation': '01 예시·02 단독 비교·03 제외 비교·04 무작위 결측·05 작품별 복구·06 자기/다른 거리·07 기여도·08 요약 선택·09 연속 결측의 9개 figure와 CSV를 제공한다. 전체 녹음에서 common/scale을 fit한 후 beat 일부를 삭제하는 동일 녹음 복구 분석이다.',
        '_feature_search_roles': '14좌표와 성향만 포함한 7좌표의 검색 결과를 비교한다. search_changes.csv: 후보 변경·기존 후보 순위·Top5 유지. heldout_agreement.csv: 검색에서 제외한 feature로 B 작품 내 다른 연주와 비교(동점=0.5). source work/grid macro 및 bootstrap. 제거마다 scale을 재fit하지 않는다.',
        '_interpretation_embedding/cross_work': 'A는 대표 작품의 Tempo 중앙 연주, B는 전체 다른 작품 후보의 거리 최소, C는 B의 작품 내 거리 최대. 01 profile은 자기 작품 내 백분위 표시용, 02 성향, 03 변화 폭, 04 B 작품 후보의 거리, 05 feature 거리 기여. 검색 입력은 14차원 원값이다. 자세한 후보는 retrieval_candidates.csv.'}
    details['_interpretation_embedding/same_work'] = '같은 작품의 Tempo 성향 25/50/75백분위 연주 3개의 profile이다. 백분위는 표시용이고 profiles.csv에 실제 14좌표를 제공한다. 개별 raw/common/relative/standardized 곡선은 상위 다섯 feature 폴더에서 확인한다.'
    for folder, description in details.items():
        directory = root / folder
        images = '\n\n'.join(f'![{p.stem}]({p.relative_to(directory).as_posix()})' for p in sorted(directory.rglob('*.png')))
        (directory / 'README.md').write_text(f'# ATEPP {folder}\n\n{description}\n\n전체 조건과 재현은 [상위 보고서]({"../" * len(Path(folder).parts)}README.md).\n\n{images}\n', encoding='utf-8')


def main():
    ai = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ai.parent.parent / 'ATEPP_dataset')
    parser.add_argument('--audit', type=Path, default=ai / 'analysis/atepp_research/midi_audit.csv')
    parser.add_argument('--out', type=Path, default=ai / 'analysis/atepp_normalization')
    parser.add_argument('--normalized-cache', type=Path)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    parser.add_argument('--min-performances', type=int, default=5)
    parser.add_argument('--min-beats', type=int, default=32)
    parser.add_argument('--min-coverage', type=float, default=.5)
    parser.add_argument('--stages', nargs='+', choices=['normalization', 'features', 'embedding', 'search'],
                        default=['normalization', 'features', 'embedding', 'search'])
    args = parser.parse_args()
    if args.min_performances < 5 or args.min_beats < 4 or not 0 < args.min_coverage <= 1 or args.trials < 1 or args.bootstrap_samples < 1:
        parser.error('Invalid cohort/trial/bootstrap settings (example figures require >=5 performances)')
    args.root, args.audit, args.out = args.root.resolve(), args.audit.resolve(), args.out.resolve()
    args.normalized_cache = (args.normalized_cache or args.root / 'features/dual_dtw/atepp_standardized.npz').resolve()
    args.out.mkdir(parents=True, exist_ok=True); setup_style()
    print('Loading all accepted ATEPP raw caches and source fingerprints', flush=True)
    records, provenance = load_records(args.root, args.audit)
    stats_path = args.out / 'stats.json'
    saved_stats = json.loads(stats_path.read_text(encoding='utf-8')) if stats_path.exists() else {}
    config = {'seed': args.seed, 'minimum_performances': args.min_performances,
              'minimum_shared_beats': args.min_beats, 'minimum_shared_coverage': args.min_coverage}
    complete_run = set(args.stages) == {'normalization', 'features', 'embedding', 'search'}
    changed_config = any(saved_stats.get(k, v) != v for k, v in config.items())
    changed_config |= saved_stats.get('embedding', {}).get('trials', args.trials) != args.trials
    changed_config |= saved_stats.get('search', {}).get('bootstrap_samples', args.bootstrap_samples) != args.bootstrap_samples
    old_provenance_path = args.out / 'provenance.json'
    old_provenance = json.loads(old_provenance_path.read_text(encoding='utf-8')) if old_provenance_path.exists() else {}
    changed_inputs = any(old_provenance.get(k, v) != v for k, v in provenance.items()
                         if k in ('audit_sha256', 'source_files', 'analysis_code_sha256'))
    for name, digest in old_provenance.get('analysis_scripts_sha256', {}).items():
        if name not in (Path(__file__).name, 'atepp_feature_analysis.py'):
            changed_inputs |= sha256(Path(__file__).with_name(name)) != digest
    if (changed_config or changed_inputs) and not complete_run:
        raise ValueError('Inputs/cohort/evaluation settings changed: rerun all stages or choose a separate --out')
    print(f'Normalizing {len(records)} performances on exact shared score grids', flush=True)
    normalized, groups, audits, scales = collect_normalized(records, args)
    print(f'{len(normalized)} comparable groups; embedding: {len(groups)} groups / {sum(len(g["keys"]) for g in groups)} performances', flush=True)
    write_csv(args.out / 'cohort_audit.csv', audits)
    write_csv(args.out / 'performance_catalog.csv', [{k: r[k] for k in ('key', 'piece', 'track', 'artist', 'composer', 'midi_sha256')} for r in records])
    duplicates = pd.DataFrame([{'key': r['key'], 'piece': r['piece'], 'midi_sha256': r['midi_sha256']} for r in records])
    duplicates[duplicates.midi_sha256.duplicated(keep=False)].to_csv(args.out / 'duplicate_midi.csv', index=False)
    write_json(args.out / 'provenance.json', provenance)
    eligible = {g['piece'] for g in groups}
    median_count = np.median([len(g['keys']) for g in groups])
    example = min((r for r in normalized if r['piece'] in eligible),
                  key=lambda r: (abs(len(r['records']) - median_count), r['piece']))
    stats = {} if complete_run else saved_stats
    stats.update(dataset='ATEPP-1.2', aligned_performances=len(records), seed=args.seed,
                 minimum_performances=args.min_performances, minimum_shared_beats=args.min_beats, minimum_shared_coverage=args.min_coverage)
    if 'normalization' in args.stages:
        print('Creating normalization diagnostics and four-stage cache', flush=True)
        stats['normalization'], extremes = normalization_analysis(normalized, args, provenance)
        extremes['eligible_embedding_work'] = extremes.piece.isin(eligible)
        extremes.to_csv(args.out / 'feature_normalization/extreme_beats.csv', index=False)
        # Re-extract the source for each channel's largest standardized extreme.
        top = extremes.sort_values('absolute_standardized', ascending=False).groupby('channel').head(1)
        source_rows = []
        for key in sorted(set(top.key)):
            result = next(r for r in normalized if key in r['features']['tempo'])
            checks, _, _, _ = verify_sources(result, [key], args.out / 'feature_normalization/extreme_sources' / key)
            source_rows.extend(checks)
        write_csv(args.out / 'feature_normalization/extreme_source_checks.csv', source_rows)
        write_json(stats_path, stats)
    if 'features' in args.stages:
        print('Drawing five feature extraction and before/after examples', flush=True)
        stats['example'] = feature_examples(example, args); write_json(stats_path, stats)
    if 'embedding' in args.stages:
        print('Running embedding dropout/ablation evaluation', flush=True)
        stats['embedding'] = embedding_analysis(groups, audits, scales, args); write_json(stats_path, stats)
    if 'search' in args.stages:
        print('Running cross-work search and heldout-feature analysis', flush=True)
        anchor = stats.get('example', {}).get('selected_keys', [example['records'][0]['key']])[len(stats.get('example', {}).get('selected_keys', [])) // 2]
        stats['search'] = search_analysis(groups, args, anchor); write_json(stats_path, stats)
    if all(k in stats for k in ('normalization', 'example', 'embedding', 'search')):
        reports(args, stats)
    scripts = [Path(__file__), Path(__file__).with_name('atepp_feature_analysis.py'),
               *[Path(__file__).with_name(name) for name in ('validate_embedding.py', 'validate_feature_search_roles.py',
                 'validate_interpretation_examples.py', 'validate_feature_normalization.py', 'validate_tempo.py')]]
    provenance['analysis_scripts_sha256'] = {p.name: sha256(p) for p in scripts}
    write_json(args.out / 'provenance.json', provenance)
    manifest = [{'path': p.relative_to(args.out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                for p in sorted(args.out.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.csv' and p.suffix != '.log']
    write_csv(args.out / 'artifact_manifest.csv', manifest)
    print(f'Complete: {args.out}', flush=True)


if __name__ == '__main__':
    main()
