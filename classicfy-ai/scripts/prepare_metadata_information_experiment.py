"""Freeze targets, eligibility and splits before metadata-information prediction.

Preflight only: no predictor fitting, encoder training, hyperparameter selection,
or evaluation of model outcomes. Use the original verified residual sequences.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SINGLE_STATS = ('large_ratio', 'roughness', 'rise_ratio', 'reversal_ratio',
                'positive_run', 'negative_run', 'value_lag1', 'value_lag8', 'delta_lag1', 'delta_lag8')
LAGS = (-4, -2, -1, 0, 1, 2, 4)
CHANNELS = ('tempo', 'rubato', 'dynamics', 'articulation', 'pedal_depth', 'pedal_down_ratio', 'pedal_changes')
EPS = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def temporal_targets(tempo, dynamics, mask):
    """Nonoverlapping full original-index blocks, never bridge missing beats."""
    tempo, dynamics, mask = np.asarray(tempo), np.asarray(dynamics), np.asarray(mask)
    if tempo.ndim != 1 or tempo.shape != dynamics.shape or mask.shape != tempo.shape or mask.dtype != bool:
        raise ValueError('Expected aligned one-dimensional original-index sequences')
    roughness, rms_support = [], []
    for start in range(0, len(mask) - 31, 32):
        values, valid = dynamics[start:start + 32], mask[start:start + 32]
        paired = valid[:-1] & valid[1:] & np.isfinite(values[:-1]) & np.isfinite(values[1:])
        if paired.sum() >= 25:
            delta = np.diff(values)[paired]
            roughness.append(float(np.sqrt(np.mean(delta ** 2))))
            rms_support.append(int(paired.sum()))
    q1, q1_status = np.nan, 'insufficient_blocks'
    if len(roughness) >= 3:
        average = float(np.mean(roughness))
        if average > EPS:
            q1, q1_status = float(np.std(roughness) / average), 'ok'
        else:
            q1_status = 'zero_mean_change'
    correlations, correlation_support, constant_blocks = [], [], 0
    for start in range(0, len(mask) - 63, 64):
        t, d, valid = tempo[start:start + 64], dynamics[start:start + 64], mask[start:start + 64]
        paired = valid[:-1] & valid[1:] & np.isfinite(t[:-1]) & np.isfinite(t[1:]) & np.isfinite(d[:-1]) & np.isfinite(d[1:])
        if paired.sum() >= 51:
            x, y = np.diff(t)[paired], np.diff(d)[paired]
            if x.std() <= EPS or y.std() <= EPS:
                constant_blocks += 1
                continue
            correlations.append(float(np.clip(np.corrcoef(x, y)[0, 1], -1, 1)))
            correlation_support.append(int(paired.sum()))
    q2, q3 = (float(np.mean(np.abs(correlations))), float(np.std(correlations))) if len(correlations) >= 3 else (np.nan, np.nan)
    return dict(change_concentration_cv=q1, change_concentration_status=q1_status,
        coupling_local_strength=q2, coupling_local_variation=q3,
        coupling_status='ok' if len(correlations) >= 3 else 'insufficient_nonconstant_blocks',
        rms_blocks=len(roughness), possible_rms_blocks=len(mask) // 32,
        correlation_blocks=len(correlations), possible_correlation_blocks=len(mask) // 64,
        constant_correlation_blocks=constant_blocks,
        rms_minimum_pairs=min(rms_support) if rms_support else 0,
        correlation_minimum_pairs=min(correlation_support) if correlation_support else 0,
        total_beats=len(mask), valid_beat_coverage=float(mask.mean()))


def assigned_roles(rows):
    families = defaultdict(set)
    for row in rows:
        families[row['family_id']].add(row['split'])
    result = {}
    for row in rows:
        source_split, roles = row['split'], families[row['family_id']]
        if source_split == 'train':
            role, reason = 'train', ''
        elif source_split == 'validation' and 'train' not in roles:
            role, reason = 'validation', ''
        elif source_split == 'test' and roles == {'test'}:
            role, reason = 'test', ''
        else:
            role, reason = 'excluded', 'family_overlaps_earlier_model_development_split'
        result[row['key']] = (role, reason)
    return result, {k: sorted(v) for k, v in families.items() if len(v) > 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent / 'datasets')
    parser.add_argument('--audit', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_style_validation')
    parser.add_argument('--learned', type=Path, default=ROOT.parent / 'datasets/metadata_temporal_exports_run01')
    parser.add_argument('--hand-ref', default='feat/#14-temporal-feature-analysis')
    parser.add_argument('--out', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_information_design')
    args = parser.parse_args()
    audit_verification = json.loads((args.audit / 'verification.json').read_text())
    if sha(args.audit / 'performances.csv') != audit_verification['csv_sha256']['performances.csv']:
        raise ValueError('Audited cohort changed')
    rows = sorted(csv_rows((args.audit / 'performances.csv').read_bytes()), key=lambda r: r['key'])
    manifest = json.loads((args.learned / 'manifest.json').read_text())
    reference = json.loads((args.data / 'temporal_embedding_cluster_run01/dataset.json').read_text())
    audit_protocol = json.loads((args.audit / 'protocol.json').read_text())
    if audit_protocol['source_provenance'] != reference['source_provenance'] or manifest['source_provenance'] != reference['source_provenance']:
        raise ValueError('Metadata, model and target source provenance differ')
    if audit_protocol['input_sha256']['dataset.json'] != sha(args.data / 'temporal_embedding_cluster_run01/dataset.json'):
        raise ValueError('Metadata audit used a different reference')
    for row in rows:
        if row['split'] != reference['split'][row['piece']]:
            raise ValueError('Metadata/model split differs')
    if manifest['reference_manifest_sha256'] != sha(args.data / 'temporal_embedding_cluster_run01/dataset.json'):
        raise ValueError('Reference changed')
    if sha(args.data / 'feature_normalization_cluster_raw.npz') != manifest['cache_sha256']:
        raise ValueError('Raw cache changed')
    commit = manifest['source_commit']
    archive_bytes = subprocess.check_output(['git', 'archive', commit, 'classicfy-ai/src', 'classicfy-ai/scripts'], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='classicfy-information-design-') as directory:
        snapshot = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
            archive.extractall(snapshot, filter='data')
        git_dir = subprocess.check_output(['git', 'rev-parse', '--absolute-git-dir'], cwd=ROOT, text=True).strip()
        (snapshot / '.git').write_text(f'gitdir: {git_dir}\n')
        inventory = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '20a4168',
            'classicfy-ai/src/features', 'classicfy-ai/src/preprocessing'], cwd=ROOT, text=True).splitlines()
        original = [p for p in inventory if p.endswith('.py')]
        for name in original:
            if (snapshot / name).read_bytes() != subprocess.check_output(['git', 'show', f'20a4168:{name}'], cwd=ROOT):
                raise ValueError('Original feature/preprocessing code changed')
        source_files = [snapshot / p for p in original if '/features/' in p and Path(p).name not in ('normalization.py', '__init__.py')]
        source_files += [snapshot / p for p in original if '/preprocessing/' in p]
        sys.path[:0] = [str(snapshot / 'classicfy-ai/scripts'), str(snapshot / 'classicfy-ai/src')]
        from validate_embedding import collect_groups
        groups, audits, scales, provenance = collect_groups(SimpleNamespace(
            asap_root=(args.data / 'ASAP').resolve(), nasap_root=(args.data / 'nASAP').resolve(),
            cache=args.data / 'feature_normalization_cluster_raw.npz', **reference['cohort_settings']),
            provenance_source_files=source_files)
        if provenance != reference['source_provenance'] or scales != reference['scales']:
            raise ValueError('Feature provenance/scales differ')
        lengths = {(r['piece'], r['grid']): r['total_intervals'] for r in audits if r['accepted']}
        targets = {}
        for group in groups:
            length = lengths[group['piece'], group['grid']]
            index = np.asarray(group['shared_indices'], dtype=int)
            if len(index) < 2 or np.any(np.diff(index) <= 0) or index[0] < 0 or index[-1] >= length:
                raise ValueError('Invalid original beat indices')
            mask = np.zeros(length, dtype=bool); mask[index] = True
            for key, source in zip(group['keys'], group['values'], strict=True):
                t, d = np.full(length, np.nan), np.full(length, np.nan)
                t[index], d[index] = source[0], source[2]
                if key in targets:
                    raise ValueError('Duplicate reconstructed performance')
                targets[key] = temporal_targets(t, d, mask)
    if set(targets) != {r['key'] for r in rows}:
        raise ValueError('Target and audited cohorts differ')
    hand_commit = subprocess.check_output(['git', 'rev-parse', args.hand_ref], cwd=ROOT, text=True).strip()
    def hand_raw(name):
        return subprocess.check_output(['git', 'show', f'{hand_commit}:classicfy-ai/analysis/temporal_coupling/{name}'], cwd=ROOT)
    hand_protocol = json.loads(hand_raw('protocol.json'))
    if hand_protocol['source_provenance'] != reference['source_provenance'] or hand_protocol['reference_manifest_sha256'] != manifest['reference_manifest_sha256']:
        raise ValueError('#14 descriptor source differs from target/model reference')
    verification = json.loads(hand_raw('verification.json'))
    def hand_csv(name):
        raw = hand_raw(name)
        if hashlib.sha256(raw).hexdigest() != verification['csv_sha256'][name]:
            raise ValueError('Verified #14 descriptor changed')
        return csv_rows(raw)
    singles = {(r['key'], r['channel']): r for r in hand_csv('single_temporal_metrics.csv')}
    relation_rows = [r for r in hand_csv('pair_metrics.csv') if r['related_definition'] == 'False']
    pairs = sorted({(r['channel_a'], r['channel_b']) for r in relation_rows})
    relations = {(r['key'], r['channel_a'], r['channel_b']): r for r in relation_rows}
    values = dict(single=np.array([[float(singles[r['key'], c][s]) for c in CHANNELS for s in SINGLE_STATS] for r in rows]),
        relation=np.array([[float(relations[r['key'], a, b][f'lag{lag}_correlation'])
            if relations[r['key'], a, b][f'lag{lag}_status'] == 'ok' else np.nan
            for a, b in pairs for lag in LAGS] for r in rows]))
    train = np.array([r['split'] == 'train' for r in rows])
    schema, schema_rows, input_complete = {}, [], np.ones(len(rows), dtype=bool)
    for name, matrix in values.items():
        keep = []
        for j in range(matrix.shape[1]):
            column = matrix[train, j]; finite = column[np.isfinite(column)]
            coverage = len(finite) / train.sum()
            active = coverage >= .95 and len(finite) >= 10 and np.std(finite) > EPS
            if active:
                keep.append(j)
            schema_rows.append(dict(block=name, coordinate=j, train_finite_fraction=float(coverage), retained=bool(active)))
        schema[name] = keep
        if not keep:
            raise ValueError('No supported training coordinates')
        input_complete &= np.isfinite(matrix[:, keep]).all(axis=1)
    roles, overlaps = assigned_roles(rows)
    exports = []
    target_names = ('change_concentration_cv', 'coupling_local_strength', 'coupling_local_variation')
    for i, row in enumerate(rows):
        role, reason = roles[row['key']]
        record = row | dict(evaluation_role=role, split_exclusion_reason=reason,
                             complete_main_inputs=bool(input_complete[i])) | targets[row['key']]
        for name in target_names:
            eligible = role != 'excluded' and input_complete[i] and np.isfinite(record[name])
            record[f'{name}_eligible'] = bool(eligible)
        exports.append(record)
    # Same input/target cohort for every condition within each endpoint; require
    # >=3 eligible recordings per piece for within-piece evaluation.
    for name in target_names:
        per_piece = Counter(r['piece'] for r in exports if r[f'{name}_eligible'])
        for row in exports:
            if per_piece[row['piece']] < 3:
                row[f'{name}_eligible'] = False
    fold_records = []
    for name in target_names:
        eligible = [r for r in exports if r[f'{name}_eligible']]
        families = sorted({r['family_id'] for r in eligible if r['evaluation_role'] == 'train'})
        if len(families) < 10:
            raise ValueError('Insufficient training families for the fixed five-fold design')
        order = np.random.default_rng(20261013).permutation(len(families))
        folds = {families[index]: position % 5 for position, index in enumerate(order)}
        for row in eligible:
            fold_records.append(dict(target=name, key=row['key'], piece=row['piece'], family_id=row['family_id'],
                role=row['evaluation_role'], inner_fold=folds[row['family_id']] if row['evaluation_role'] == 'train' else -1))
    selected_models = [e for e in manifest['exports'] if e['arm'] in ('cnn64', 'bilstm64') and e['stage'] == 'best_extended']
    if len(selected_models) != 6:
        raise ValueError('Expected six frozen main model exports')
    for arm in ('cnn64', 'bilstm64'):
        if sorted(e['seed'] for e in selected_models if e['arm'] == arm) != [20261006, 20261007, 20261008]:
            raise ValueError('Missing or duplicate model seed')
    for entry in selected_models:
        path = args.learned / entry['file']
        if sha(path) != entry['export_sha256']:
            raise ValueError('Frozen model export changed')
        with np.load(path, allow_pickle=False) as z:
            if set(z['keys']) != set(targets) or z['vectors'].shape != (len(targets), 128) or not np.isfinite(z['vectors']).all():
                raise ValueError('Invalid frozen learned representation')
    summaries = []
    for name in target_names:
        for role in ('train', 'validation', 'test'):
            pool = [r for r in exports if r['evaluation_role'] == role]
            eligible = [r for r in pool if r[f'{name}_eligible']]
            summaries.append(dict(target=name, role=role, role_performances=len(pool), eligible_performances=len(eligible),
                eligible_works=len({r['work_id'] for r in eligible}), eligible_families=len({r['family_id'] for r in eligible}),
                eligible_composers=len({r['composer'] for r in eligible}),
                incomplete_inputs=sum(not r['complete_main_inputs'] for r in pool),
                undefined_target=sum(not np.isfinite(r[name]) for r in pool)))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in (('target_preflight.csv', exports), ('eligibility_summary.csv', summaries),
                          ('descriptor_schema.csv', schema_rows), ('fold_manifest.csv', fold_records)):
        write_csv(args.out / name, records)
    protocol = dict(version=1, status='design_and_preflight_only_no_prediction_results',
        reference_manifest_sha256=sha(args.data / 'temporal_embedding_cluster_run01/dataset.json'),
        learned_manifest_sha256=sha(args.learned / 'manifest.json'), hand_source_commit=hand_commit,
        learned_source_commit=commit, source_provenance=reference['source_provenance'],
        targets=dict(change_concentration_cv=dict(block_beats=32, minimum_adjacent_pairs=25, minimum_blocks=3,
            definition='population SD of per-block Dynamics delta RMS / mean block RMS', transform='log1p', mean_floor=EPS),
            coupling_local_strength=dict(block_beats=64, minimum_adjacent_pairs=51, minimum_blocks=3,
                definition='mean absolute per-block Pearson(delta Tempo, delta Dynamics)', transform='identity', constant_floor=EPS),
            coupling_local_variation=dict(block_beats=64, minimum_adjacent_pairs=51, minimum_blocks=3,
                definition='population SD of per-block Pearson(delta Tempo, delta Dynamics)', transform='identity', constant_floor=EPS)),
        blocks='Nonoverlapping original-index complete blocks from index zero; discard short remainder; gaps are never joined',
        descriptor_policy='Train-only >=95% finite and nonconstant columns; complete cases across all retained single/relation inputs; no imputation; same cohort for every condition per target',
        retained_descriptor_coordinates=schema, original_family_overlaps=overlaps,
        split='Keep original encoder train/validation/test; exclude validation families overlapping train and test families overlapping train or validation; no encoder re-training',
        sample_policy='At least 3 eligible recordings per piece, separately per target; all methods use identical target-specific cohort',
        predictor='Separate linear ridge per endpoint and condition; all training numeric preprocessing inside each inner fold; no outcome-dependent feature additions',
        metadata='Train vocabulary one-hot composer, work_type, composer×work_type; no work-title ID for unseen-work primary prediction',
        covariates=['log1p(total_beats)', 'valid_beat_coverage'],
        conditions=['M', 'M+Q', 'M+Q+B14', 'M+Q+B14+T', 'M+Q+B14+T+R', 'M+Q+B14+CNN128', 'M+Q+B14+BiLSTM128'],
        main_models='Frozen extended-budget CNN64/BiLSTM64, every seed 20261006/07/08; no seed selected from outcomes',
        selected_model_exports=selected_models,
        ridge_alphas=[.0001, .001, .01, .1, 1., 10., 100., 1000.],
        inner_cv='Five folds of training family groups; weighted recording loss -> work mean -> family mean; outer train-only schema frozen at preflight',
        inner_cv_seed=20261013,
        fitting='Weights sum to one: equal family weight, equal work weight within family, equal recordings within work; unpenalized intercept; weighted train-fold mean/SD preprocessing; ties choose larger ridge alpha within absolute tolerance 1e-12',
        predictions='No clipping; score change_concentration_cv in log1p space; bounded correlation targets in original space',
        global_metric='Squared error mean within work, then within family, then across families; compare paired relative MSE reductions',
        within_work_metric='Subtract each test work mean from observed values and separately from predictions for scoring only, then the same family-balanced squared error; no test-work mean enters fitting or inference',
        relative_reduction='100 * (1 - augmented_MSE/reference_MSE); undefined for reference_MSE<=1e-12; not a success-rate percentage',
        controls=dict(shuffles=20, seeds=list(range(20261101, 20261121)),
            strata='Eligible recordings within identical piece+grid, separately per split; never move records across pieces or splits',
            added_blocks='Shuffle T only for C vs B, and R only for D vs C; keep M/Q/B and preceding blocks unchanged; repeat the same train-only CV procedure',
            matched_noise='Add independent Gaussian blocks of 57 and 119 coordinates using seeds 20261101..20261120, with identical fitting/tuning; random values independent of targets'),
        uncertainty=dict(family_bootstrap_repetitions=2000, seed=20261016,
            definition='Paired resampling of evaluation families for MSE differences and relative reductions; percentile 95% interval; fitted predictors unchanged',
            minimum_families=5, below_minimum='Per-work results only; no interval'),
        evidence_rule='Positive main global and within-work improvements with per-family results, CV controls and model-seed ranges reported; use paired intervals and shuffled/noise ranges to distinguish stable support from descriptive or unresolved differences; do not replace endpoints, blocks or cohorts after outcomes',
        primary_contrasts=['(M+Q+B14) vs (M+Q)', '(M+Q+B14+T) vs (M+Q+B14) on change_concentration_cv',
            '(M+Q+B14+T+R) vs (M+Q+B14+T) on coupling_local_strength'],
        secondary_endpoint='coupling_local_variation; do not substitute it for the primary endpoint after observing outcomes',
        input_sha256={'metadata_performances':sha(args.audit / 'performances.csv'), 'raw_cache':sha(args.data / 'feature_normalization_cluster_raw.npz')},
        code_sha256=sha(__file__))
    (args.out / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'preflight_verification.json').write_text(json.dumps(dict(reconstructed_performances=len(targets),
        train_schema_performances=int(train.sum()), retained_single_coordinates=len(schema['single']),
        retained_relation_coordinates=len(schema['relation']),
        csv_sha256={name: sha(args.out / name) for name in ('target_preflight.csv', 'eligibility_summary.csv', 'descriptor_schema.csv', 'fold_manifest.csv')}), indent=2) + '\n')
    for r in summaries:
        print(r)


if __name__ == '__main__':
    main()
