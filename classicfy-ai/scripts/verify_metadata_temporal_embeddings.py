"""Independent scalar recalculation of the enhanced metadata comparison exports."""

import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_asap_metadata import ROOT, read_csv, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_temporal_embedding')
    parser.add_argument('--learned', type=Path, default=ROOT.parent / 'datasets/metadata_temporal_exports_run01')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_summary_embedding')
    args = parser.parse_args()
    protocol = json.loads((args.out / 'protocol.json').read_text())
    verification = json.loads((args.out / 'verification.json').read_text())
    for name, expected in verification['csv_sha256'].items():
        if sha256(args.out / name) != expected:
            raise ValueError(f'Output changed: {name}')
    family_path = args.learned / 'metadata_family_comparisons.csv'
    if sha256(family_path) != verification['family_detail_sha256']:
        raise ValueError('Family details changed')
    records = read_csv(args.out / 'comparisons.csv') + read_csv(family_path)
    identities = {r['key']: r for r in read_csv(args.baseline / 'pilot_profiles.csv')}
    frozen = {design: {(r['key'], r['key_b'], r['key_c']) for r in read_csv(args.baseline / name)}
              for design, name in (('global', 'query_comparisons.csv'), ('family_pair', 'family_comparisons.csv'))}
    groups = defaultdict(list)
    for r in records:
        a, b, c = (identities[r[f]] for f in ('key', 'key_b', 'key_c'))
        if not (a['composer'] == b['composer'] == c['composer'] and a['work_type'] != b['work_type']
                and a['work_type'] == c['work_type'] and len({a['family_id'], b['family_id'], c['family_id']}) == 3):
            raise ValueError('Candidate constraints failed')
        groups[r['model'], r['seed'], r['metric'], r['design']].append(r)
    for (model, seed, metric, design), rows in groups.items():
        actual = {(r['key'], r['key_b'], r['key_c']) for r in rows}
        if actual != frozen[design] or len(rows) != len(actual):
            raise ValueError('Candidate set changed or duplicated')
    manifest = json.loads((args.learned / 'manifest.json').read_text())
    maximum_error, learned_checks = 0., 0
    for entry in manifest['exports']:
        path = args.learned / entry['file']
        if sha256(path) != entry['export_sha256']:
            raise ValueError('Model export changed')
        with np.load(path, allow_pickle=False) as z:
            values = dict(zip(z['keys'], z['vectors'].astype(np.float64)))
        model, seed = f"{entry['arm']}_{entry['stage']}", str(entry['seed'])
        for metric in ('cosine', 'euclidean'):
            for design in ('global', 'family_pair'):
                for r in groups[model, seed, metric, design]:
                    x = values[r['key']]
                    for side in ('b', 'c'):
                        y = values[r[f'key_{side}']]
                        value = (np.clip(1 - np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)), 0, 2)
                                 if metric == 'cosine' else np.sqrt(np.dot(x - y, x - y) / len(x)))
                        error = abs(value - float(r[f'distance_{side}']))
                        maximum_error = max(maximum_error, error)
                        if error > 1e-12:
                            raise ValueError('Independent learned distance differs')
                    learned_checks += 1
    # Independent baseline scalar calculation with explicit coordinate weights.
    baseline_checks = 0
    weight = np.r_[np.repeat(.1, 8), np.repeat(1 / 30, 6)]
    vectors = {k: np.array([float(r[f'coordinate_{j}']) for j in range(14)]) for k, r in identities.items()}
    for model in ('representative', 'representative_range', 'range_only'):
        coords = np.arange(14) if model == 'representative_range' else np.arange(0 if model == 'representative' else 1, 14, 2)
        weights = weight[coords] * (1 if model == 'representative_range' else 2)
        for design in ('global', 'family_pair'):
            for r in groups[model, '0', 'descriptor', design]:
                for side in ('b', 'c'):
                    delta = vectors[r['key']][coords] - vectors[r[f'key_{side}']][coords]
                    value = np.sqrt(np.dot(delta ** 2, weights))
                    if abs(value - float(r[f'distance_{side}'])) > 1e-12:
                        raise ValueError('Independent baseline distance differs')
                baseline_checks += 1
    # Reconstruct descriptors by explicit field names rather than relying on
    # the production array assembly or masked-distance function.
    def source_csv(name):
        raw = subprocess.check_output(['git', 'show', f"{protocol['hand_source_commit']}:classicfy-ai/analysis/temporal_coupling/{name}"], cwd=ROOT)
        return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    single = {(r['key'], r['channel']): r for r in source_csv('single_temporal_metrics.csv')}
    pairs = {(r['key'], r['channel_a'], r['channel_b']): r for r in source_csv('pair_metrics.csv')}
    scales = defaultdict(dict)
    for r in read_csv(args.out / 'descriptor_scales.csv'):
        scales[r['model']][int(r['coordinate'])] = float(r['scale'])
    channels = ('tempo', 'rubato', 'dynamics', 'articulation', 'pedal_depth', 'pedal_down_ratio', 'pedal_changes')
    def raw_groups(model, key):
        if model == 'single_temporal':
            return [[float(single[key, channel][s]) for s in protocol['single_stats']] for channel in channels]
        result = []
        for a, b in protocol['independent_pairs']:
            r = pairs[key, a, b]
            if model == 'coupling':
                result.append([float(r[f'lag{lag}_correlation']) if r[f'lag{lag}_status'] == 'ok' else np.nan for lag in protocol['pair_lags']])
            else:
                result.append([float(r[f'{x}_{y}_probability']) if r['status'] == 'ok' else np.nan
                               for x in ('fall', 'flat', 'rise') for y in ('fall', 'flat', 'rise')])
        return result
    descriptor_checks, coordinate_scale_checks = 0, 0
    train_keys = sorted({key for (key, channel), r in single.items() if r['split'] == 'train'})
    for model in ('single_temporal', 'coupling', 'joint_direction'):
        training = np.array([np.array(raw_groups(model, key)).ravel() for key in train_keys])
        for j in range(training.shape[1]):
            x = training[:, j]; x = x[np.isfinite(x)]
            value = np.nan
            if len(x) >= 10:
                value = (np.percentile(x, 75) - np.percentile(x, 25)) / 1.3489795003921634
                if value <= 1e-12:
                    value = np.sqrt(np.mean((x - np.mean(x)) ** 2))
                if value <= 1e-12:
                    value = np.nan
            np.testing.assert_allclose(value, scales[model][j], rtol=0, atol=1e-12, equal_nan=True)
            coordinate_scale_checks += 1
        weights = [.2] * 4 + [1 / 15] * 3 if model == 'single_temporal' else [1 / 17] * 17
        for design in ('global', 'family_pair'):
            for r in groups[model, '0', 'descriptor', design]:
                raw = [raw_groups(model, r[field]) for field in ('key', 'key_b', 'key_c')]
                terms, used_weights, coordinate = [], [], 0
                for block, weight in enumerate(weights):
                    block_scale = np.array([scales[model][coordinate + j] for j in range(len(raw[0][block]))])
                    coordinate += len(block_scale)
                    active = np.isfinite(block_scale)
                    supported = active & np.isfinite(np.array([row[block] for row in raw])).all(axis=0)
                    if active.sum() == 0 or supported.sum() < np.ceil(.8 * active.sum()):
                        continue
                    normalized = np.array([row[block] for row in raw])[:, supported] / block_scale[supported]
                    terms.append([np.mean((normalized[0] - normalized[i]) ** 2) for i in (1, 2)])
                    used_weights.append(weight)
                threshold = len(weights) if model == 'single_temporal' else np.ceil(.8 * len(weights))
                values = np.sqrt(np.average(terms, axis=0, weights=used_weights)) if len(terms) >= threshold else [np.nan, np.nan]
                np.testing.assert_allclose(values, [float(r['distance_b']), float(r['distance_c'])], rtol=0, atol=1e-12, equal_nan=True)
                descriptor_checks += 1
    hybrid_checks = 0
    for design in ('global', 'family_pair'):
        lookup = {(model, r['key'], r['key_b'], r['key_c']): r for (model, seed, metric, d), rows in groups.items()
                  if seed == '0' and d == design for r in rows}
        for model, components in (('summary_single', ('representative_range', 'single_temporal')),
                                  ('summary_single_coupling', ('representative_range', 'single_temporal', 'coupling'))):
            for r in groups[model, '0', 'descriptor', design]:
                for side in ('b', 'c'):
                    terms = [(float(lookup[k, r['key'], r['key_b'], r['key_c']][f'distance_{side}']) /
                              protocol['train_distance_calibrations'][k]['median']) ** 2 for k in components]
                    np.testing.assert_allclose(np.sqrt(np.mean(terms)), float(r[f'distance_{side}']), rtol=0, atol=1e-12, equal_nan=True)
                hybrid_checks += 1
    summary_checks = 0
    common_keys = set(identities)
    for model in ('single_temporal', 'coupling', 'joint_direction'):
        common_keys &= {r['key'] for r in groups[model, '0', 'descriptor', 'global'] if np.isfinite(float(r['preference']))}
    for summary in read_csv(args.out / 'summary.csv'):
        model, seed, metric = (summary[k] for k in ('model', 'seed', 'metric'))
        design, subset = summary['design'], summary['subset']
        if design == 'family_balanced':
            by_key = defaultdict(list)
            for r in groups[model, seed, metric, 'family_pair']:
                by_key[r['key']].append(r)
            rows = [dict(rows[0], preference=np.mean([float(r['preference']) for r in rows])) for rows in by_key.values()]
        else:
            rows = groups[model, seed, metric, 'global']
            if design == 'common_hand_support':
                rows = [r for r in rows if r['key'] in common_keys]
        if subset != 'all':
            rows = [r for r in rows if float(r['representative_preference']) == (1 if subset == 'representative_b_closer' else 0)]
        supported = [r for r in rows if np.isfinite(float(r['preference']))]
        by_work = defaultdict(list)
        for r in supported:
            by_work[r['work_id']].append(float(r['preference']))
        query = np.mean([float(r['preference']) for r in supported]) if supported else np.nan
        work = np.mean([np.mean(scores) for scores in by_work.values()]) if by_work else np.nan
        if int(summary['eligible']) != len(rows) or int(summary['supported']) != len(supported):
            raise ValueError('Independent support count differs')
        np.testing.assert_allclose([query, work], [float(summary['query_preference']), float(summary['work_preference'])], atol=1e-12, rtol=0, equal_nan=True)
        summary_checks += 1
    result = dict(candidate_constraint_rows_checked=len(records), frozen_variants_checked=len(groups),
        independent_learned_triplets=learned_checks, independent_baseline_triplets=baseline_checks,
        independent_descriptor_triplets=descriptor_checks, independent_hybrid_triplets=hybrid_checks,
        train_coordinate_scales_checked=coordinate_scale_checks,
        maximum_learned_distance_error=maximum_error, independent_summary_rows=summary_checks,
        common_hand_support=len(common_keys), code_sha256=sha256(__file__),
        checked_input_sha256={name: sha256(args.out / name) for name in ('comparisons.csv', 'summary.csv', 'verification.json')})
    (args.out / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
