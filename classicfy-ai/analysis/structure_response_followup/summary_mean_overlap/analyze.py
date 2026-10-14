"""Read-only diagnostic of the frozen 335-recording structure experiment."""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE / 'capstone-2026-2-classicfy'
sys.path.insert(0, str(REPO / 'classicfy-ai/src'))
from embedding.retrieval_probe import weights, fit_scale, distance, ranked, priorities, weighted_quantile
from embedding.standalone_structure import standalone_distances
from features.structure_response import split_view

OUT = Path(__file__).parent
SOURCE = WORKSPACE / 'datasets/standalone_structure_run01/standalone.npz'
REPORT = REPO / 'classicfy-ai/analysis/standalone_structure'
CHANNELS = ('tempo', 'rubato', 'dynamics', 'articulation', 'pedal_depth', 'pedal_down_ratio', 'pedal_changes')
z = np.load(SOURCE, allow_pickle=False)
keys = z['keys'].tolist()
with (REPO / 'classicfy-ai/analysis/phrase_response/coverage.csv').open() as f:
    lookup = {r['key']: r for r in csv.DictReader(f)}
metadata = [lookup[k] for k in keys]
gids, qids = z['gallery_indices'], z['query_indices']
gallery = [metadata[i] for i in gids]
gw, allw = weights(gallery), weights(metadata)
families = np.array([r['family_id'] for r in gallery])
work_groups = defaultdict(list)
for j, row in enumerate(gallery):
    work_groups[row['piece'], row['grid']].append(j)


def macro(rows, field):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r['family_id']][r['piece']].append(r[field])
    return float(np.mean([np.mean([np.mean(v) for v in works.values()]) for works in grouped.values()]))


def correlation(a, b, w):
    a, b = a - w @ a, b - w @ b
    den = np.sqrt((w @ (a*a)) * (w @ (b*b)))
    return float(w @ (a*b) / den) if den > 1e-12 else None


def within_piece(x):
    result = x.copy()
    for piece in set(r['piece'] for r in metadata):
        ids = np.array([i for i, r in enumerate(metadata) if r['piece'] == piece])
        result[ids] -= x[ids].mean(axis=0)
    return result


def calibrated(x):
    d = distance(x, x[gids], fit_scale(x[gids], gw))
    a, b = np.triu_indices(len(gids), 1)
    different = families[a] != families[b]
    a, b = a[different], b[different]
    median = float(weighted_quantile(d[gids[a], b], gw[a]*gw[b], .5))
    assert median > 1e-12
    return d / median


def save_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


correlations, energy, query_rows, balanced_rows = [], [], [], []
max_reconstruction_error = 0.
for view, direction in enumerate(('A_to_B', 'B_to_A')):
    descriptors = {name: z[f'{direction}_input_{name}'] for name in ('B', 'T', 'F', 'P', 'H')}
    evaluations = {axis: z[f'{direction}_evaluation_{axis}'] for axis in ('E-L', 'E-T', 'E-R', 'E-S')}
    reconstructed, _ = standalone_distances(descriptors, metadata, gids)
    for method, d in reconstructed.items():
        saved = z[f'{direction}_distance_{method}']
        max_reconstruction_error = max(max_reconstruction_error, float(abs(d-saved).max()))
        np.testing.assert_allclose(d, saved, atol=1e-12, rtol=1e-12)
    global_mean = []
    for i in range(len(keys)):
        mask, _, _ = split_view(z[f'{i}/mask'], z[f'{i}/pairs'], view)
        global_mean.append(z[f'{i}/values'][:, mask].mean(axis=1))
    global_mean = np.asarray(global_mean)
    centered_global = within_piece(global_mean)
    distances = {'B': reconstructed['B']}
    for method in ('F', 'P', 'H'):
        x = descriptors[method]
        groups = x.reshape(len(keys), -1, 3, 22)
        levels = groups[..., :7].reshape(len(keys), -1, 7)
        centered_levels = within_piece(levels)
        for context in range(levels.shape[1]):
            for c, channel in enumerate(CHANNELS):
                correlations.append(dict(direction=direction, method=method, context=context, channel=channel,
                    pearson=correlation(levels[:, context, c], global_mean[:, c], allw),
                    within_piece_pearson=correlation(centered_levels[:, context, c], centered_global[:, c], allw)))
        # Orthogonal decomposition of between-recording level variance:
        # shared value across three bins vs. differences between bins.
        level_groups = groups[..., :7]
        channel_sd = np.sqrt(gw @ ((global_mean[gids] - gw @ global_mean[gids])**2))
        assert (channel_sd > 1e-12).all()
        scaled = level_groups / channel_sd
        shifted = scaled - np.einsum('n,ngbc->gbc', allw, scaled)[None]
        common = shifted.mean(axis=2, keepdims=True)
        total = float(np.einsum('n,ngbc->', allw, shifted**2))
        shared = float(3 * np.einsum('n,ngbc->', allw, common**2))
        contrast = float(np.einsum('n,ngbc->', allw, (shifted-common)**2))
        np.testing.assert_allclose(shared+contrast, total, rtol=1e-12)
        energy.append(dict(direction=direction, method=method, shared_level_variance_fraction=shared/total,
                           context_contrast_variance_fraction=contrast/total))
        centered = groups.copy()
        centered[..., :7] -= global_mean[:, None, None, :]
        variants = dict(full=x, means_only=level_groups.reshape(len(keys), -1),
            change_only=groups[..., 7:].reshape(len(keys), -1),
            mean_centered=centered.reshape(len(keys), -1))
        for variant, values in variants.items():
            assert np.isfinite(values).all()
            distances[f'{method}_{variant}'] = calibrated(values)
        np.testing.assert_allclose(distances[f'{method}_full'], reconstructed[method], atol=1e-12, rtol=1e-12)
    for variant in ('full', 'means_only', 'change_only', 'mean_centered'):
        distances[f'P+H_{variant}'] = np.sqrt((distances[f'P_{variant}']**2 + distances[f'H_{variant}']**2)/2)
    np.testing.assert_allclose(distances['P+H_full'], reconstructed['P+H'], atol=1e-12, rtol=1e-12)
    for qi in qids:
        row = metadata[qi]
        priority = priorities(20261014, row['key'], len(gallery))
        base = ranked(distances['B'][qi], families, priority, 5)
        for method, d in distances.items():
            selected = ranked(d[qi], families, priority, 5)
            overlap = len(set(base) & set(selected))/5
            family_overlap = len(set(families[base]) & set(families[selected]))/5
            rho = float(spearmanr(distances['B'][qi], d[qi]).statistic)
            record = dict(direction=direction, key=row['key'], piece=row['piece'], family_id=row['family_id'],
                          role=row['evaluation_role'], method=method, overlap_B=overlap,
                          family_overlap_B=family_overlap, distance_spearman_B=rho)
            for axis in ('E-L', 'E-T', 'E-R', 'E-S'):
                e = evaluations[axis]
                record[axis] = 100*float(e[qi, selected].mean())
                record[axis+'_same_work_expected'] = 100*float(np.mean([
                    e[qi, work_groups[gallery[j]['piece'], gallery[j]['grid']]].mean() for j in selected]))
            query_rows.append(record)
    # Same existing twenty candidate-count controls, applied to every variant.
    for seed in range(20261014, 20261034):
        balanced = z[f'{direction}_balanced_{seed}']
        for method, d in distances.items():
            records = []
            for qi in qids:
                row = metadata[qi]
                restricted = np.full(len(gallery), np.inf)
                restricted[balanced] = d[qi, balanced]
                selected = ranked(restricted, families, priorities(seed, row['key'], len(gallery)), 5)
                records.append(dict(piece=row['piece'], family_id=row['family_id'], role=row['evaluation_role'],
                    **{axis: 100*float(evaluations[axis][qi, selected].mean())
                       for axis in ('E-S', 'E-T')}))
            for role in ('test', 'validation'):
                rs = [r for r in records if r['role'] == role]
                balanced_rows.append(dict(direction=direction, seed=seed, role=role, method=method,
                    **{axis: macro(rs, axis) for axis in ('E-S', 'E-T')}))

summary = []
for role in ('test', 'validation'):
    for method in sorted(set(r['method'] for r in query_rows)):
        rows = [r for r in query_rows if r['role'] == role and r['method'] == method]
        summary.append(dict(role=role, method=method, **{field: macro(rows, field) for field in (
            'overlap_B', 'family_overlap_B', 'distance_spearman_B', 'E-L', 'E-T', 'E-R', 'E-S',
            'E-L_same_work_expected', 'E-T_same_work_expected', 'E-R_same_work_expected', 'E-S_same_work_expected')}))

# Reproduce every displayed baseline/full-method score, not merely rankings.
with (REPORT / 'summary.csv').open() as f:
    previous = {(r['role'], r['method'], r['axis']): float(r['score'])*100 for r in csv.DictReader(f)}
max_score_error = 0.
for r in summary:
    if r['method'] == 'B' or r['method'].endswith('_full'):
        original = r['method'].removesuffix('_full')
        for axis in ('E-L', 'E-T', 'E-R', 'E-S'):
            error = abs(r[axis] - previous[r['role'], original, axis])
            max_score_error = max(max_score_error, error)
            assert error < 1e-10

save_csv('level_correlations.csv', correlations)
save_csv('level_variance.csv', energy)
save_csv('queries.csv', query_rows)
save_csv('summary.csv', summary)
save_csv('balanced.csv', balanced_rows)
overview = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), recordings=len(keys),
    gallery=len(gids), queries=len(qids), max_distance_reconstruction_error=max_reconstruction_error,
    max_original_score_error=max_score_error,
    definitions=dict(means_only='Only the seven context-conditioned level means; drop all 15 change coordinates per bin.',
        change_only='Drop seven level means; keep delta mean, delta RMS and joint Tempo-Dynamics moment.',
        mean_centered='Subtract same-recording, same-search-view global signed mean from each of seven context means; other coordinates unchanged.',
        shared_variance='Between-recording variance of level coordinates decomposed into a shared three-bin component and bin contrasts; channel-scaled using gallery global-mean SD; family-balanced across the frozen 335 recordings.',
        correlations='Signed Pearson correlation between context mean and same-view global signed mean, family-balanced. Within-piece correlations first subtract each piece mean. B uses median absolute rubato, so B is not identical to the seven signed global means.',
        retrieval='Same gallery, per-family top-5 cap, opposite-view saved judges, gallery-only SD and distance-median calibration; no tuning.'),
    limitations=['Exploratory diagnostics chosen after viewing prior results, not a preregistered success test.',
        'Only three test and three validation families; no population inference.',
        'No tension-specific judge, no new phrase truth, no new model.',
        'Delta coordinates may still encode overall change scale; removing levels is not removing all range information.',
        'Shared-level variance fraction concerns level coordinates only, not all 66/198 coordinates.',
        'Main comparisons retain unequal candidate counts by work; exact same-work expectation and existing twenty balanced-gallery seeds are included.'])
(OUT/'verification.json').write_text(json.dumps(overview, indent=2, ensure_ascii=False)+'\n')
for method in ('F', 'P', 'H'):
    cs = [r for r in correlations if r['method'] == method]
    print(method, 'median context/global r', np.median([r['pearson'] for r in cs]),
        'within-piece r', np.median([r['within_piece_pearson'] for r in cs if r['within_piece_pearson'] is not None]),
        'shared level variance', np.mean([r['shared_level_variance_fraction'] for r in energy if r['method'] == method]))
for r in summary:
    if r['method'] == 'B' or r['method'].startswith(('P_', 'H_', 'P+H_')):
        print(r['role'],r['method'], 'E-S',round(r['E-S'],2),'E-T',round(r['E-T'],2),
              'overlap',round(r['overlap_B'],3),'rho',round(r['distance_spearman_B'],3))
print('verification', max_reconstruction_error, max_score_error)
for role in ('test', 'validation'):
    for method in ('P_full','P_change_only','H_full','H_change_only','P+H_full','P+H_change_only'):
        rows = [r for r in balanced_rows if r['role']==role and r['method']==method]
        s = next(r for r in summary if r['role']==role and r['method']==method)
        print('control',role,method,'balanced E-S',round(np.mean([r['E-S'] for r in rows]),2),
              'same-work expected E-S',round(s['E-S_same_work_expected'],2))
