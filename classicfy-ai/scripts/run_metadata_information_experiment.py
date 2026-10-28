"""Run the frozen metadata-information design, with all planned controls.

Fixed encoders, train-family CV, paired evaluation and whole-performance inputs.
Large input/model/control archives remain in datasets/, outside Git.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'classicfy-ai/src'))
from embedding.information_probe import (RidgeReadout, family_weights, metadata_matrix,
    choose_alphas, work_errors, macro_error, paired_bootstrap, shuffle_within_work)
from audit_asap_metadata import read_csv, write_csv, sha256

TARGETS = ('change_concentration_cv', 'coupling_local_strength', 'coupling_local_variation')
SINGLE_STATS = ('large_ratio', 'roughness', 'rise_ratio', 'reversal_ratio',
                'positive_run', 'negative_run', 'value_lag1', 'value_lag8', 'delta_lag1', 'delta_lag8')
CHANNELS = ('tempo', 'rubato', 'dynamics', 'articulation', 'pedal_depth', 'pedal_down_ratio', 'pedal_changes')
LAGS = (-4, -2, -1, 0, 1, 2, 4)
CONDITIONS = {'A0': (), 'A1': ('Q',), 'B': ('Q', 'B'), 'C': ('Q', 'B', 'T'), 'D': ('Q', 'B', 'T', 'R')}


def source_csv(commit, name, verification):
    raw = subprocess.check_output(['git', 'show', f'{commit}:classicfy-ai/analysis/temporal_coupling/{name}'], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != verification['csv_sha256'][name]:
        raise ValueError('Original descriptor export changed')
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))


def prepare_inputs(args, protocol):
    verification = json.loads((args.design / 'preflight_verification.json').read_text())
    for name, expected in verification['csv_sha256'].items():
        if sha256(args.design / name) != expected:
            raise ValueError(f'Frozen design file changed: {name}')
    rows = read_csv(args.design / 'target_preflight.csv')
    if len(rows) != len({r['key'] for r in rows}):
        raise ValueError('Duplicate design key')
    features = {r['key']: r for r in read_csv(args.features / 'embeddings.csv')}
    feature_audit = json.loads((args.design.parent / 'metadata_style_validation/protocol.json').read_text())
    if sha256(args.features / 'embeddings.csv') != feature_audit['input_sha256']['feature_embeddings.csv']:
        raise ValueError('Frozen summary export changed')
    if sha256(args.learned / 'manifest.json') != protocol['learned_manifest_sha256']:
        raise ValueError('Frozen learned exports manifest changed')
    if set(features) != {r['key'] for r in rows}:
        raise ValueError('Feature cohort differs')
    for row in rows:
        if any(row[f] != features[row['key']][f] for f in ('piece', 'grid')):
            raise ValueError('Feature identity differs')
    commit = protocol['hand_source_commit']
    raw = subprocess.check_output(['git', 'show', f'{commit}:classicfy-ai/analysis/temporal_coupling/verification.json'], cwd=ROOT)
    hand_verification = json.loads(raw)
    singles = {(r['key'], r['channel']): r for r in source_csv(commit, 'single_temporal_metrics.csv', hand_verification)}
    pairs = [r for r in source_csv(commit, 'pair_metrics.csv', hand_verification) if r['related_definition'] == 'False']
    pair_names = sorted({(r['channel_a'], r['channel_b']) for r in pairs})
    relations = {(r['key'], r['channel_a'], r['channel_b']): r for r in pairs}
    single = np.array([[float(singles[r['key'], c][s]) for c in CHANNELS for s in SINGLE_STATS] for r in rows])
    relation = np.array([[float(relations[r['key'], a, b][f'lag{lag}_correlation'])
        if relations[r['key'], a, b][f'lag{lag}_status'] == 'ok' else np.nan
        for a, b in pair_names for lag in LAGS] for r in rows])
    blocks = dict(Q=np.array([[np.log1p(float(r['total_beats'])), float(r['valid_beat_coverage'])] for r in rows]),
        B=np.array([[float(features[r['key']][f'coordinate_{j}']) for j in range(14)] for r in rows]),
        T=single[:, protocol['retained_descriptor_coordinates']['single']],
        R=relation[:, protocol['retained_descriptor_coordinates']['relation']])
    for entry in protocol['selected_model_exports']:
        path = args.learned / entry['file']
        if sha256(path) != entry['export_sha256']:
            raise ValueError('Frozen model export changed')
        with np.load(path, allow_pickle=False) as z:
            keys = list(z['keys'])
            if len(keys) != len(set(keys)) or set(keys) != {r['key'] for r in rows}:
                raise ValueError('Model keys differ')
            lookup = {key: i for i, key in enumerate(keys)}
            blocks[f"{entry['arm']}_{entry['seed']}"] = z['vectors'][[lookup[r['key']] for r in rows]].astype(float)
    input_signature = dict(design_protocol=sha256(args.design / 'protocol.json'),
        design_csv=verification['csv_sha256'], feature_export=sha256(args.features / 'embeddings.csv'),
        learned_manifest=sha256(args.learned / 'manifest.json'),
        hand_csv={name: hand_verification['csv_sha256'][name] for name in ('single_temporal_metrics.csv', 'pair_metrics.csv')},
        runner_code=sha256(__file__), core_code=sha256(ROOT / 'classicfy-ai/src/embedding/information_probe.py'))
    return rows, blocks, input_signature


def fit_case(rows, numeric, targets, fold_by_key, alphas):
    training = np.array([r['evaluation_role'] == 'train' for r in rows])
    train_ids = np.flatnonzero(training)
    train_rows = [rows[i] for i in train_ids]
    folds = np.array([fold_by_key[r['key']] for r in train_rows])
    if set(folds) != set(range(5)):
        raise ValueError('Frozen CV missing fold')
    weights = family_weights(train_rows)
    oof = np.full((len(train_ids), len(alphas), targets.shape[1]), np.nan)
    fold_scores = []
    for fold in range(5):
        fit_ids, val_ids = train_ids[folds != fold], train_ids[folds == fold]
        fit_rows, val_rows = [rows[i] for i in fit_ids], [rows[i] for i in val_ids]
        m_fit, _, _ = metadata_matrix(fit_rows, fit_rows)
        m_val, _, _ = metadata_matrix(fit_rows, val_rows)
        x_fit, x_val = np.column_stack((m_fit, numeric[fit_ids])), np.column_stack((m_val, numeric[val_ids]))
        model = RidgeReadout.prepare(x_fit, targets[fit_ids], family_weights(fit_rows))
        errors = []
        for position, alpha in enumerate(alphas):
            predicted = model.predict(x_val, alpha)
            oof[folds == fold, position] = predicted
            errors.append(family_weights(val_rows) @ ((targets[val_ids] - predicted) ** 2))
        fold_scores.append(errors)
    if not np.isfinite(oof).all():
        raise ValueError('OOF predictions missing')
    scores = np.einsum('n,nak->ak', weights, (oof - targets[train_ids, None, :]) ** 2)
    selected = choose_alphas(scores, alphas)
    metadata, names, _ = metadata_matrix(train_rows, rows)
    x = np.column_stack((metadata, numeric))
    model = RidgeReadout.prepare(x[train_ids], targets[train_ids], weights)
    predictions = model.predict(x, selected)
    return dict(predictions=predictions, selected_alphas=selected, cv_scores=scores,
        cv_fold_scores=np.array(fold_scores), cv_oof_predictions=oof,
        x=x, y=targets, train_ids=train_ids, train_weights=weights,
        mean=model.mean, scale=model.scale, active=model.active, target_mean=model.target_mean,
        coefficients=model.coefficients(selected), metadata_columns=np.array(names),
        raw_metadata_columns=len(names))


def cached_fit(path, signature, rows, numeric, y, folds, alphas, source_indices):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z['signature']) != signature:
                raise ValueError(f'Cached fit uses different source/protocol: {path}')
            return {key: z[key].copy() for key in z.files if key != 'signature'}
    result = fit_case(rows, numeric, y, folds, alphas)
    result.update(keys=np.array([r['key'] for r in rows]), source_indices=source_indices)
    np.savez_compressed(path, signature=np.array(signature), **result)
    return result


def evaluate_fit(rows, y, predicted, meta, target_names):
    prediction_rows, metric_rows, work_rows = [], [], []
    for j, target in enumerate(target_names):
        for role in ('train', 'validation', 'test'):
            ids = [i for i, r in enumerate(rows) if r['evaluation_role'] == role]
            subset = [rows[i] for i in ids]
            local = work_errors(subset, y[ids, j], predicted[ids, j])
            for r in local:
                work_rows.append(meta | dict(target=target, role=role) | r)
            metric_rows.append(meta | dict(target=target, role=role, recordings=len(ids), works=len(local),
                families=len({r['family_id'] for r in subset}),
                global_mse=macro_error(local, 'global_mse'), within_mse=macro_error(local, 'within_mse')))
            for i in ids:
                prediction_rows.append(meta | dict(target=target, role=role, key=rows[i]['key'], piece=rows[i]['piece'],
                    family_id=rows[i]['family_id'], truth=float(y[i, j]), prediction=float(predicted[i, j])))
    return prediction_rows, metric_rows, work_rows


def bootstrap_results(works, protocol):
    lookup = defaultdict(list)
    for row in works:
        if row['kind'] == 'main':
            lookup[row['target'], row['condition'], row['encoder_seed'], row['role']].append(row)
    results = []
    for target in TARGETS:
        for role in ('validation', 'test'):
            for condition, seed in sorted({(k[1], k[2]) for k in lookup if k[0] == target and k[3] == role}):
                if condition == 'A0':
                    continue
                reference = {'A1': 'A0', 'B': 'A1', 'C': 'B', 'D': 'C', 'E_CNN': 'B', 'E_BiLSTM': 'B'}[condition]
                before, after = lookup[target, reference, 0, role], lookup[target, condition, seed, role]
                for metric in ('global_mse', 'within_mse'):
                    a, b = macro_error(before, metric), macro_error(after, metric)
                    count = len({r['family_id'] for r in before})
                    decrease, reduction = paired_bootstrap(before, after, metric,
                        repetitions=protocol['uncertainty']['family_bootstrap_repetitions'], seed=protocol['uncertainty']['seed'])
                    supported = count >= protocol['uncertainty']['minimum_families']
                    results.append(dict(target=target, role=role, reference=reference, condition=condition,
                        encoder_seed=seed, metric=metric, families=count, before_mse=a, after_mse=b,
                        reduction_percent=100 * (1 - b / a) if a > 1e-12 else np.nan,
                        reduction_ci_low=float(np.nanpercentile(reduction, 2.5)) if supported else np.nan,
                        reduction_ci_high=float(np.nanpercentile(reduction, 97.5)) if supported else np.nan,
                        decrease_ci_low=float(np.percentile(decrease, 2.5)) if supported else np.nan,
                        decrease_ci_high=float(np.percentile(decrease, 97.5)) if supported else np.nan,
                        improved_works=sum(bw[metric] < aw[metric] for aw, bw in zip(sorted(before, key=lambda r:r['piece']), sorted(after, key=lambda r:r['piece']))),
                        works=len(before), interval_status='ok' if supported else 'fewer_than_five_families'))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_information_design')
    parser.add_argument('--features', type=Path, default=ROOT / 'classicfy-ai/analysis/_feature_search_roles')
    parser.add_argument('--learned', type=Path, default=ROOT.parent / 'datasets/metadata_temporal_exports_run01')
    parser.add_argument('--runs', type=Path, default=ROOT.parent / 'datasets/metadata_information_run01')
    parser.add_argument('--out', type=Path, default=ROOT / 'classicfy-ai/analysis/metadata_information_results')
    args = parser.parse_args()
    protocol = json.loads((args.design / 'protocol.json').read_text())
    rows, blocks, input_signature = prepare_inputs(args, protocol)
    if protocol['version'] != 1:
        raise ValueError('Unsupported frozen protocol')
    fold_records = read_csv(args.design / 'fold_manifest.csv')
    args.runs.mkdir(parents=True, exist_ok=True); args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.runs / 'inputs.npz', keys=np.array([r['key'] for r in rows]), **blocks)
    inputs_sha = sha256(args.runs / 'inputs.npz')
    all_predictions, all_metrics, all_works, all_cv, audits = [], [], [], [], []
    control_predictions = []
    started = time.monotonic()
    for group, target_names in (('dynamics', TARGETS[:1]), ('coupling', TARGETS[1:])):
        flags = [r[f'{target_names[0]}_eligible'] == 'True' for r in rows]
        if any([r[f'{name}_eligible'] == 'True' for r in rows] != flags for name in target_names):
            raise ValueError('Grouped outputs have different eligible cohorts')
        ids = np.flatnonzero(flags)
        local_rows = [rows[i] for i in ids]
        local_blocks = {name: values[ids] for name, values in blocks.items()}
        y = np.array([[np.log1p(float(r[name])) if protocol['targets'][name]['transform'] == 'log1p'
                      else float(r[name]) for name in target_names] for r in local_rows])
        if not np.isfinite(y).all():
            raise ValueError('Undefined eligible target')
        folds = {r['key']: int(r['inner_fold']) for r in fold_records
                 if r['target'] == target_names[0] and r['role'] == 'train'}
        cases = [(condition, 'main', 0, 0, names, None) for condition, names in CONDITIONS.items()]
        for arm, condition in (('cnn64', 'E_CNN'), ('bilstm64', 'E_BiLSTM')):
            for seed in (20261006, 20261007, 20261008):
                cases.append((condition, 'main', seed, 0, ('Q', 'B', f'{arm}_{seed}'), None))
        for condition, added in (('C', 'T'), ('D', 'R')):
            for kind in ('shuffle', 'noise'):
                for seed in protocol['controls']['seeds']:
                    cases.append((condition, kind, 0, seed, CONDITIONS[condition], added))
        for position, (condition, kind, encoder_seed, control_seed, names, added) in enumerate(cases):
            case_blocks = dict(local_blocks)
            source_indices = np.arange(len(local_rows))
            if kind == 'shuffle':
                case_blocks[added], source_indices = shuffle_within_work(local_rows, local_blocks[added], control_seed)
            elif kind == 'noise':
                case_blocks[added] = np.random.default_rng(control_seed).standard_normal(local_blocks[added].shape)
            numeric = np.column_stack([case_blocks[name] for name in names]) if names else np.empty((len(local_rows), 0))
            if not np.isfinite(numeric).all():
                raise ValueError('Incomplete eligible inputs')
            meta = dict(condition=condition, kind=kind, encoder_seed=encoder_seed, control_seed=control_seed)
            identifier = f'{group}_{condition}_{kind}_{encoder_seed}_{control_seed}'
            signature = hashlib.sha256(json.dumps(input_signature | dict(case=identifier), sort_keys=True).encode()).hexdigest()
            result = cached_fit(args.runs / f'{identifier}.npz', signature, local_rows, numeric, y,
                                folds, protocol['ridge_alphas'], source_indices)
            predictions, metrics, works = evaluate_fit(local_rows, y, result['predictions'], meta, target_names)
            (all_predictions if kind == 'main' else control_predictions).extend(predictions)
            all_metrics.extend(metrics); all_works.extend(works)
            for j, target in enumerate(target_names):
                for k, alpha in enumerate(protocol['ridge_alphas']):
                    all_cv.append(meta | dict(target=target, alpha=alpha, oof_macro_mse=float(result['cv_scores'][k,j]),
                        selected=bool(alpha == result['selected_alphas'][j])))
            audits.append(meta | dict(group=group, file=f'{identifier}.npz', rows=len(local_rows),
                train_rows=len(result['train_ids']), raw_dimensions=result['x'].shape[1],
                active_dimensions=int(result['active'].sum()), selected_alphas=result['selected_alphas'].tolist(),
                archive_sha256=sha256(args.runs / f'{identifier}.npz')))
            if position < 11 or (position + 1) % 10 == 0:
                test = [r for r in metrics if r['role'] == 'test']
                print(group, position + 1, '/', len(cases), condition, kind, encoder_seed, control_seed,
                      'alpha', result['selected_alphas'].tolist(),
                      'test MSE', [(r['target'], round(r['global_mse'], 6), round(r['within_mse'], 6)) for r in test], flush=True)
    bootstrap = bootstrap_results(all_works, protocol)
    exports = {'predictions.csv': all_predictions,
        'metrics.csv': [r for r in all_metrics if r['kind'] == 'main'],
        'controls.csv': [r for r in all_metrics if r['kind'] != 'main'],
        'per_work_metrics.csv': [r for r in all_works if r['kind'] == 'main'],
        'cv_results.csv': all_cv, 'bootstrap.csv': bootstrap}
    for name, values in exports.items():
        write_csv(args.out / name, values)
    write_csv(args.runs / 'control_predictions.csv', control_predictions)
    write_csv(args.runs / 'control_per_work_metrics.csv', [r for r in all_works if r['kind'] != 'main'])
    metadata_unknown = []
    for target in TARGETS:
        eligible = [r for r in rows if r[f'{target}_eligible'] == 'True']
        training = [r for r in eligible if r['evaluation_role'] == 'train']
        for role in ('train', 'validation', 'test'):
            _, _, unknown = metadata_matrix(training, [r for r in eligible if r['evaluation_role'] == role])
            metadata_unknown.append(dict(target=target, role=role, composer=unknown[0], work_type=unknown[1], combination=unknown[2]))
    execution = dict(status='completed_fixed_design', frozen_design_protocol_sha256=sha256(args.design / 'protocol.json'),
        design_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources=input_signature, inputs_archive_sha256=inputs_sha, runtime_seconds=time.monotonic() - started,
        python=sys.version, numpy=np.__version__, numerical_method='Weighted centered ridge via symmetric eigendecomposition; unpenalized intercept; all candidate alphas use the same fold decomposition',
        grouping='Coupling endpoints share only the input matrix decomposition; alpha and coefficient column selected independently per endpoint',
        metadata_unknown=metadata_unknown, runs=audits,
        csv_sha256={name:sha256(args.out/name) for name in exports},
        external_control_predictions_sha256=sha256(args.runs / 'control_predictions.csv'),
        external_control_works_sha256=sha256(args.runs / 'control_per_work_metrics.csv'))
    (args.out / 'execution.json').write_text(json.dumps(execution, ensure_ascii=False, indent=2) + '\n')
    print('Completed',len(audits),'fits;',len(all_predictions),'main predictions;',len(control_predictions),'control predictions',flush=True)


if __name__ == '__main__':
    main()
