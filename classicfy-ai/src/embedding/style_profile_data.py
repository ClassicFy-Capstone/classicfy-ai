"""Validated source manifests, immutable style artifacts and diagnostics."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from features.style_profile import (
    G_NAMES, StyleConfig, StyleRecording, artifact, check_artifact, make_windows,
    reference_delta, scale_profile, summarize_window, validate_recordings,
)
from preprocessing.musicxml_score import _root, read_musicxml
from .note_performance_data import load_note_performance


def save_style_artifact(value, path):
    check_artifact(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrite: derived stages never modify source R or previous runs.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


def load_style_artifact(path, kind=None, config=None):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    check_artifact(value, kind, config)
    return value


def _path(base, value):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _read_window_score(path, traversal):
    root, _ = _root(path)
    for part in root.findall('part'):
        measures = part.findall('measure')
        for i, measure in enumerate(measures):
            for mark in measure.findall('./attributes/measure-style/multiple-rest'):
                count = int(mark.text)
                span = measures[i:i + count]
                if count < 1 or len(span) != count or any(m.find('.//pitch') is not None for m in span):
                    raise ValueError('Multiple-rest notation needs explicitly expanded physical rest measures')
    return read_musicxml(path, traversal=traversal)


def read_recordings(entries, base):
    records, sources = [], []
    for entry in entries:
        rpath, xpath = (_path(base, entry[k]) for k in ('performance_cache', 'xml_path'))
        r = load_note_performance(rpath)
        for field, value in (('split', r.split), ('work_key', r.work_key), ('record_key', r.record_key)):
            if field in entry and entry[field] != value:
                raise ValueError(f'Source manifest disagrees with R {field}')
        if entry.get('R_fingerprint', r.fingerprint) != r.fingerprint:
            raise ValueError('Source R fingerprint changed')
        traversal = tuple(entry['traversal']) if entry.get('traversal') is not None else None
        xml = _read_window_score(xpath, traversal)
        sha = r.provenance.get('performance', {}).get('midi_sha256')
        identity = entry.get('recording_group') or sha
        if not identity:
            raise ValueError('Supply recording_group when raw MIDI checksum is unavailable')
        rec = StyleRecording(r, xml, identity, entry.get('performance_id', ''))
        records.append(rec)
        sources.append({'performance_cache': str(rpath), 'xml_path': str(xpath),
                        'traversal': list(traversal) if traversal is not None else None,
                        'recording_group': identity, 'performance_id': rec.performance_id,
                        'R_fingerprint': r.fingerprint, 'split': r.split, 'work_key': r.work_key,
                        'record_key': r.record_key})
    validate_recordings(records)
    # Identical MIDI bytes must remain aliases even if explicit group names differ.
    by_sha = {}
    for rec in records:
        sha = rec.performance.provenance.get('performance', {}).get('midi_sha256')
        if sha:
            if sha in by_sha and by_sha[sha] != rec.recording_id:
                raise ValueError('Identical MIDI bytes have conflicting recording groups')
            by_sha[sha] = rec.recording_id
    return records, sources


def make_window_artifact(entries, base, config=StyleConfig()):
    records, sources = read_recordings(entries, base)
    sources.sort(key=lambda entry: entry['performance_id'])
    ids = [entry['performance_id'] for entry in sources]
    if len(ids) != len(set(ids)):
        raise ValueError('Input manifest repeats a performance ID')
    windows = [w for rec in records for w in make_windows(rec)]
    windows.sort(key=lambda w: w['window_id'])
    return artifact('windows', config, sources=sources, windows=windows,
                    split_policy='caller supplied work-level splits; unchanged')


def verify_window_sources(windows):
    check_artifact(windows, 'windows')
    records, _ = read_recordings(windows['sources'], Path('.'))
    rebuilt = sorted((w for rec in records for w in make_windows(rec)), key=lambda w: w['window_id'])
    if rebuilt != windows['windows']:
        raise ValueError('Window membership or source content changed')
    return records


def extract_profile_artifact(windows, reference):
    config = check_artifact(windows, 'windows')
    check_artifact(reference, 'reference', config)
    if reference['parents'].get('windows') != windows['id']:
        raise ValueError('Reference/window dependency mismatch')
    records = verify_window_sources(windows)
    profiles = []
    by_performance = {}
    for w in windows['windows']:
        by_performance.setdefault(w['performance_id'], []).append(w)
    for rec in records:
        r = rec.performance
        deltas = reference_delta(r, reference)
        for w in by_performance[rec.performance_id]:
            profiles.append(summarize_window(r, w, reference, config, deltas=deltas))
    profiles.sort(key=lambda p: p['window']['window_id'])
    # Re-read and compare fingerprints after extraction as an explicit source invariant.
    verify_window_sources(windows)
    return artifact('profiles', config, parents={'windows': windows['id'], 'reference': reference['id']},
                    profiles=profiles, raw_R_unchanged=True)


def style_report(windows, profiles, scaler, calibration, triplets, reference):
    config = check_artifact(windows, 'windows')
    for value, kind in ((profiles, 'profiles'), (scaler, 'profile-scaler'),
                        (calibration, 'distance-calibration'), (triplets, 'triplets')):
        check_artifact(value, kind, config)
    if (profiles['parents']['windows'] != windows['id'] or scaler['parents']['profiles'] != profiles['id']
            or calibration['parents'] != {'profiles': profiles['id'], 'scaler': scaler['id']}
            or triplets['parents'] != {'profiles': profiles['id'], 'scaler': scaler['id'], 'calibration': calibration['id']}):
        raise ValueError('Report artifact dependencies disagree')
    check_artifact(reference, 'reference', config)
    if profiles['parents']['reference'] != reference['id']:
        raise ValueError('Report/reference dependency mismatch')
    rows, ps = windows['windows'], profiles['profiles']
    feature_stats = {}
    for j, name in enumerate(G_NAMES):
        values = [p['g_raw'][j] for p in ps if p['g_mask'][j]]
        feature_stats[name] = {'valid': len(values), 'total': len(ps),
                               'active': scaler['trained_feature_mask'][j],
                               'quantiles': np.quantile(values, [0, .05, .5, .95, 1]).tolist() if values else [],
                               'scaler_fit_count': scaler['valid_counts'][j], 'scaler_status': scaler['status'][j]}
    per_work = Counter(t['anchor']['work_id'] for t in triplets['triplets'])
    eligible = [p for p in ps if p['sampling_eligible']]
    metric_counts = Counter()
    if scaler['metric_ready']:
        for p in eligible:
            pedal_active = any(p['g_mask'][j] and scaler['trained_feature_mask'][j] for j in range(9, 23))
            metric_counts['five_blocks' if pedal_active else 'four_blocks'] += 1
    examples = []
    lookup = {p['window']['window_id']: p for p in ps}
    for t in triplets['triplets'][:5]:
        examples.append({**t, 'profiles': {role: lookup[t[f'{role}_window_id']]['g_raw']
                                           for role in ('anchor', 'positive', 'negative')}})
    report = {'works': len({w['work_id'] for w in rows}),
              'performances': len(windows['sources']), 'windows': len(rows),
              'full_windows': sum(not w['tail'] for w in rows), 'tails': sum(w['tail'] for w in rows),
              'note_count_quantiles': np.quantile([w['N'] for w in rows], [0, .5, .95, 1]).tolist() if rows else [],
              'features': feature_stats, 'core_eligible_windows': len(eligible),
              'core_only_windows': sum(not any(p['g_mask'][9:]) for p in eligible),
              'pedal_available_windows': sum(any(p['g_mask'][9:]) for p in eligible),
              'reference_sufficient_fraction_by_channel': [float(np.mean([p['reference_counts'][k]['sufficient_fraction'] for p in ps]))
                                                          if ps else 0. for k in range(11)],
              'coverage_by_channel': [np.quantile([p['per_channel_coverage'][k] for p in ps], [0, .5, 1]).tolist()
                                      if ps else [] for k in range(11)],
              'metric_comparison_counts': dict(metric_counts),
              'calibration_status': calibration['status'],
              'calibration_signatures': {k: {f: v for f, v in val.items() if f != 'pairs'}
                                         for k, val in calibration['signatures'].items()},
              'anchors': len(triplets['anchor_outcomes']), 'triplets': len(triplets['triplets']),
              'outcomes': triplets['reason_counts'],
              'triplets_per_work': {work: per_work[work] for work in sorted({w['work_id'] for w in rows})},
              'candidate_distributions': triplets['anchor_outcomes'], 'triplet_examples': examples,
              'raw_R_unchanged': profiles['raw_R_unchanged'], 'fit_split': 'train',
              'interpretation': 'cohort-relative weak supervision; not an independent listener-style ground truth'}
    # Include one auditable raw -> reference -> delta -> g trace even if no triplets exist.
    records = verify_window_sources(windows)
    if ps:
        p = ps[0]
        rec = next(rec for rec in records if rec.performance_id == p['window']['performance_id'])
        delta, mask, counts, bd, bm, bc = reference_delta(rec.performance, reference)
        indices = p['window']['row_indices'][:3]
        table = reference['tables'].get(p['window']['cohort_key'], {})
        report['profile_example'] = {'reference_id': reference['id'],
                                      'reference_rows': [table.get('notes', {}).get(rec.performance.row_keys[i]) for i in indices],
                                      'delta_rows': delta[indices].tolist(), 'delta_mask': mask[indices].tolist(),
                                      'beat_trace': [{'key': p['window']['beat_keys'][j],
                                                      'R': float(rec.performance.beats.log2_qpm[i]),
                                                      'reference': table.get('beats', {}).get(p['window']['beat_keys'][j]),
                                                      'delta': float(bd[i]), 'valid': bool(bm[i])}
                                                     for j, i in enumerate(p['window']['beat_indices'][:3])],
                                      'window': p['window'], 'g_raw': p['g_raw'], 'g_mask': p['g_mask'],
                                      'g_scaled': scale_profile(p, scaler)['g_scaled'],
                                      'raw_R_rows': rec.performance.values[indices].tolist()}
    return artifact('report', config, parents={'windows': windows['id'], 'profiles': profiles['id'],
                                              'scaler': scaler['id'], 'calibration': calibration['id'], 'triplets': triplets['id']},
                    **report)
