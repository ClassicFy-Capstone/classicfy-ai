"""Stream whole-performance profiles by cohort from a verified AE input index."""

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from features.performance_style_profile import (
    PerformanceStyleConfig, array_digest, fit_performance_reference, fit_performance_scaler,
    performance_style_schema, scale_performance_profile, summarize_performance,
)
from features.style_profile import G_BLOCKS, G_NAMES, StyleRecording, check_artifact, stable_id
from .note_performance_data import load_note_performance
from .style_profile_data import _read_window_score


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _correlation(x, y):
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def profile_report(profiles, scaler, balanced, references):
    result = {'schema': performance_style_schema(), 'total_recordings': len(profiles), 'splits': {},
              'references': references, 'features': {}, 'scaler_id': scaler['id'],
              'work_balanced_scaler_id': balanced['id'],
              'quality_correlations': 'Descriptive Pearson, without causal interpretation',
              'limitations': ['Relative to a peer cohort; target is included in its reference.',
                             'Heldout references generate evaluation labels only; not available for isolated unseen performances.',
                             'Existing 8-bar minimum counts are operational thresholds, not validated whole-performance thresholds.',
                             'Pedal observations come from existing R provenance; confirmed absence differs from unknown.',
                             'Whole-traversal statistics include tail and regions omitted from AE full-window inputs.']}
    for split in ('train', 'validation', 'test'):
        rows = [p for p in profiles if p['split'] == split]
        result['splits'][split] = {
            'recordings': len(rows), 'works': len({p['work_key'] for p in rows}),
            'unique_recordings': len({p['recording_id'] for p in rows}),
            'core_valid': sum(p['core_valid'] for p in rows),
            'all_23_valid': sum(p['all_features_valid'] for p in rows),
            'any_feature_valid': sum(any(p['g_mask']) for p in rows),
            'five_blocks_observed_after_scaling': sum(all(any(p['g_scaled_mask'][j] for j in ids)
                                                        for ids in G_BLOCKS.values()) for p in rows),
            'reason_counts': dict(sorted(Counter(r for p in rows for r in p['reason_codes']).items())),
            'ae_scope_coverage_quantiles': np.quantile([p['scope']['ae_full_window_coverage'] for p in rows],
                                                       [0, .1, .5, .9, 1]).tolist() if rows else [],
            'work_recording_counts': dict(sorted(Counter(p['work_key'] for p in rows).items()))}
    for j, name in enumerate(G_NAMES):
        stats = {'scaler_status': scaler['status'][j], 'trained': scaler['trained_feature_mask'][j],
                 'train_fit_count': scaler['valid_counts'][j], 'splits': {}}
        for split in ('train', 'validation', 'test'):
            rows = [p for p in profiles if p['split'] == split and p['g_mask'][j]]
            v = np.array([p['g_raw'][j] for p in rows])
            channel = 0 if j <= 2 else (j - 1) // 2
            coverage = [p['channels'][channel]['coverage'] for p in rows]
            cohort_size = [references[p['cohort_id']]['unique_recordings'] for p in rows]
            stats['splits'][split] = {
                'valid': len(rows), 'raw_quantiles': np.quantile(v, [0, .05, .25, .5, .75, .95, 1]).tolist() if len(v) else [],
                'correlation_score_length': _correlation(v, [p['scope']['quarter_length'] for p in rows]),
                'correlation_coverage': _correlation(v, coverage),
                'correlation_cohort_size': _correlation(v, cohort_size)}
        if scaler['trained_feature_mask'][j] and balanced['trained_feature_mask'][j]:
            stats['work_balanced_center_shift_in_primary_units'] = (balanced['center'][j] - scaler['center'][j]) / scaler['scale'][j]
            stats['work_balanced_scale_ratio'] = balanced['scale'][j] / scaler['scale'][j]
        result['features'][name] = stats
    return result


def prepare_performance_styles(index_path, out, config=PerformanceStyleConfig()):
    index_path, out = Path(index_path).resolve(), Path(out).resolve()
    index = json.loads(index_path.read_text())
    check_artifact(index, 'ae-training-index')
    sources = sorted(index['sources'], key=lambda s: s['performance_id'])
    if not sources or len({s['performance_id'] for s in sources}) != len(sources):
        raise ValueError('Need unique performance IDs')
    if any(s['split'] not in {'train', 'validation', 'test'} for s in sources):
        raise ValueError('All sources require existing work splits')
    # Read only small identifiers from each manifest; large note metadata is released.
    cohorts, source_audit = defaultdict(list), []
    split_maps = {k: {} for k in ('work', 'score', 'recording')}
    for s in sources:
        path = Path(s['performance_cache'])
        manifest = json.loads((path / 'manifest.json').read_text())
        for key in ('record_key', 'work_key', 'split'):
            if manifest[key] != s[key]:
                raise ValueError(f'Source index/manifest mismatch: {s["performance_id"]} {key}')
        if manifest['fingerprint'] != s['R_fingerprint'] or manifest['score_source_sha256'] != s['score_source_sha256']:
            raise ValueError('Source R/score fingerprint differs from index')
        midi_sha = manifest['provenance'].get('performance', {}).get('midi_sha256')
        if midi_sha and midi_sha != s['recording_id']:
            raise ValueError('Index recording ID differs from MIDI identity')
        for key, value in [('work', s['work_key']), ('score', s['score_source_sha256']), ('recording', s['recording_id'])]:
            previous = split_maps[key].setdefault(value, s['split'])
            if previous != s['split']:
                raise ValueError(f'{key} crosses splits')
        route = manifest['traversal_id']
        cohorts[s['split'], s['work_key'], s['score_source_sha256'], route].append(s)
        source_audit.append({'performance_id': s['performance_id'], 'performance_cache': str(path),
                             'manifest_sha256': file_sha(path / 'manifest.json'),
                             'array_sha256': manifest['array_sha256'], 'R_fingerprint': manifest['fingerprint'],
                             'windows_sha256': s['windows_sha256'], 'xml_sha256': s['score_source_sha256']})
    intervals = defaultdict(list)
    for item in index['items']:
        w = item['window']
        intervals[w['performance_id']].append((w['q_start'], w['q_end']))
    source_by_id = {s['performance_id']: s for s in sources}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'references').mkdir()
    protocol = {'schema': performance_style_schema(), 'config': asdict(config),
                'source_index': str(index_path), 'source_index_sha256': file_sha(index_path),
                'source_index_id': index['id'], 'source_recordings': len(sources),
                'source_cohorts': len(cohorts), 'split_policy': 'unchanged; references strictly within each split',
                'duration_filter': 'no additional filter; preserve the exact AE source index',
                'evaluation_reference': 'heldout peer statistics for labels only; never fit training scaler',
                'quantile_probabilities': [.1, .5, .9], 'target_in_reference': True,
                'source_audit_id': stable_id(source_audit),
                'code_sha256': {str(p.relative_to(Path(__file__).resolve().parents[2])): file_sha(p) for p in (
                    Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1] / 'features/performance_style_profile.py',
                    Path(__file__).resolve().parents[1] / 'features/style_profile.py',
                    Path(__file__).resolve().parents[2] / 'scripts/prepare_performance_style.py')}}
    protocol['id'] = stable_id(protocol)
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'source_audit.json', source_audit)
    profiles, references = [], {}
    started = time.monotonic()
    with (out / 'profiles_raw.jsonl').open('x') as stream:
        for number, (key, entries) in enumerate(sorted(cohorts.items()), 1):
            # The saved windows preserve the exact traversal used to build R.
            ws_path = Path(entries[0]['windows_file'])
            if file_sha(ws_path) != entries[0]['windows_sha256']:
                raise ValueError('Window manifest changed')
            windows = json.loads(ws_path.read_text())
            traversal = [int(v['key'].split('@')[0]) for w in windows for v in w['measure_visits']]
            xml = _read_window_score(Path(entries[0]['xml_path']), tuple(traversal))
            if xml.source_sha256 != key[2] or xml.traversal_id != key[3]:
                raise ValueError('Score/traversal source changed')
            records = []
            for s in entries:
                r = load_note_performance(s['performance_cache'])
                if r.fingerprint != s['R_fingerprint']:
                    raise ValueError('R fingerprint changed')
                records.append(StyleRecording(r, xml, s['recording_id'], s['performance_id']))
            metadata, arrays = fit_performance_reference(records, config)
            cid = metadata['cohort_id']
            np.savez_compressed(out / 'references' / f'{cid}.npz', **arrays)
            write_json(out / 'references' / f'{cid}.json', metadata)
            references[cid] = {'split': key[0], 'work_key': key[1], 'reference_id': metadata['id'],
                               'unique_recordings': len(metadata['members']), 'duplicates': metadata['excluded_duplicate_ids'],
                               'metadata_file': f'references/{cid}.json', 'arrays_file': f'references/{cid}.npz'}
            for rec in records:
                profile = summarize_performance(rec, metadata, arrays, config, ae_intervals=intervals[rec.performance_id])
                profiles.append(profile)
                stream.write(json.dumps(profile, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({'cohort': number, 'cohorts': len(cohorts), 'recordings': len(profiles),
                              'total': len(sources), 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
            del records, arrays
    profiles.sort(key=lambda p: p['performance_id'])
    if {p['performance_id'] for p in profiles} != set(source_by_id):
        raise ValueError('Profile/source membership differs')
    train = [p for p in profiles if p['split'] == 'train']
    scaler = fit_performance_scaler(train, config)
    balanced = fit_performance_scaler(train, config, work_balanced=True)
    write_json(out / 'scaler_train.json', scaler)
    write_json(out / 'scaler_train_work_balanced.json', balanced)
    scaled = [scale_performance_profile(p, scaler) for p in profiles]
    with (out / 'profiles_scaled.jsonl').open('x') as stream:
        for p in scaled:
            stream.write(json.dumps(p, sort_keys=True, allow_nan=False) + '\n')
    np.savez_compressed(out / 'profiles.npz', performance_id=np.array([p['performance_id'] for p in scaled]),
                        split=np.array([p['split'] for p in scaled]), work_key=np.array([p['work_key'] for p in scaled]),
                        names=np.array(G_NAMES), g_raw=np.array([p['g_raw'] for p in scaled]),
                        g_mask=np.array([p['g_mask'] for p in scaled]),
                        g_scaled=np.array([p['g_scaled'] for p in scaled]),
                        g_scaled_mask=np.array([p['g_scaled_mask'] for p in scaled]),
                        scaler_id=np.array(scaler['id']), protocol_id=np.array(protocol['id']))
    report = profile_report(scaled, scaler, balanced, references)
    write_json(out / 'report.json', report)
    # Verify immutable R inputs a second time after extraction, rather than relying on saved IDs alone.
    for audit in source_audit:
        path = Path(audit['performance_cache'])
        if file_sha(path / 'manifest.json') != audit['manifest_sha256'] or file_sha(path / 'performance.npz') != audit['array_sha256']:
            raise ValueError('Source R files changed during extraction')
        s = source_by_id[audit['performance_id']]
        if file_sha(s['xml_path']) != audit['xml_sha256'] or file_sha(s['windows_file']) != audit['windows_sha256']:
            raise ValueError('Source XML/windows changed during extraction')
    if file_sha(index_path) != protocol['source_index_sha256']:
        raise ValueError('Source index changed during extraction')
    for info in references.values():
        metadata = json.loads((out / info['metadata_file']).read_text())
        with np.load(out / info['arrays_file'], allow_pickle=False) as arrays:
            if array_digest(dict(arrays)) != metadata['arrays_sha256']:
                raise ValueError('Saved reference differs from computed arrays')
    verification = {'passed': True, 'source_R_XML_windows_index_unchanged': True, 'split_overlap': {k: 0 for k in split_maps},
                    'every_source_has_one_profile': len(scaled) == len(sources), 'profiles': len(scaled),
                    'reference_array_hashes_verified': len(references),
                    'all_masked_values_are_zero': all(all(v == 0 for v, m in zip(p['g_raw'], p['g_mask']) if not m) for p in scaled),
                    'scaler_train_only': all(p['split'] == 'train' for p in train),
                    'files': {p.name: file_sha(p) for p in out.iterdir() if p.is_file()}}
    if not verification['all_masked_values_are_zero']:
        raise ValueError('Masked payload is nonzero')
    write_json(out / 'verification.json', verification)
    return report
