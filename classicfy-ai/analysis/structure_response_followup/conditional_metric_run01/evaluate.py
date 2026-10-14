"""Evaluate all checkpoints only after selection: local fit and opposite-view retrieval."""
from collections import defaultdict
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from data import GroupCatalog, load_mapping, PREVIOUS, WORK, sha
from model import ConditionalMetricEncoder, target_distances
from train import MODES, code_hash
from prepare_judges import AXES

OUT = Path(__file__).resolve().parent
CURVES = WORK / 'diagnostics/curve_comparison'
sys.path.insert(0, str(CURVES))
from compare import set_distances, calibrated, evaluate, macro, weights
from embedding.retrieval_probe import percentile_scores, priorities, ranked
KINDS = ('P', 'H0', 'H1', 'H2')


def write(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def encode(model, catalog, permutation_seed=None):
    d = catalog.data; x = d.x.clone(); vectors = []
    if permutation_seed is not None:
        for i in d.ids:
            for k in range(1, 5):
                rows = np.flatnonzero((d.arrays['owner'] == i) & (d.arrays['type'] == k))
                rng = np.random.default_rng(np.random.SeedSequence([permutation_seed, catalog.view, k, int(i)]))
                x[rows] = x[rows][:, :, rng.permutation(17)]
    with torch.no_grad():
        for start in range(0, len(x), 256): vectors.append(model.encode(x[start:start + 256], d.context[start:start + 256]).numpy())
    vectors = np.concatenate(vectors)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    return vectors


def retrieval(vectors, d, gallery_columns, b, fixed=None, coordinate_scales=None):
    gw = weights([d.metadata[i] for i in d.gids]); families = np.array([d.metadata[i]['family_id'] for i in d.gids])
    blocks = {}; fitted = {}
    for k, kind in enumerate(KINDS, 1):
        sets = {i: vectors[(d.arrays['owner'] == i) & (d.arrays['type'] == k), :, None] for i in d.ids}
        scale = (np.ones(32), np.ones(32, bool)) if coordinate_scales is None else coordinate_scales[kind]
        raw = set_distances(sets, d.ids, d.ids, scale)
        if fixed is None:
            _, median = calibrated(raw[:, gallery_columns], d.gids, gw, families)
        else: median = fixed[kind]
        fitted[kind] = median; blocks[kind] = raw / median
    h = np.sqrt(np.mean([blocks[k] ** 2 for k in ('H0', 'H1', 'H2')], axis=0))
    if fixed is None:
        _, hm = calibrated(h[:, gallery_columns], d.gids, gw, families)
    else: hm = fixed['H']
    fitted['H'] = hm; h /= hm; p = blocks['P']
    return {'P': p, 'H': h, 'P+H': np.sqrt((p * p + h * h) / 2),
            'B+P+H': np.sqrt((b * b + p * p + h * h) / 3)}, fitted


def within_work(method, distances, judgments, d, seed):
    ids = d.ids; columns = {int(i): j for j, i in enumerate(ids)}
    groups = defaultdict(list)
    for i in d.ids:
        row = d.metadata[i]; groups[row['piece'], row['grid']].append(int(i))
    results, ranks = [], []
    for qi in d.qids:
        row = d.metadata[qi]
        candidates = np.array([i for i in groups[row['piece'], row['grid']] if i != qi])
        assert len(candidates) >= 3
        cj = np.array([columns[int(i)] for i in candidates]); values = distances[qi, cj]
        priority = priorities(20261014, row['key'], len(ids))[cj]
        chosen = ranked(values, np.zeros(len(candidates)), priority, 3, cap=False)
        record = dict(seed=seed, method=method, key=row['key'], piece=row['piece'], family_id=row['family_id'],
                      role=row['evaluation_role'], candidate_count=len(candidates))
        for axis, judge in judgments.items():
            e = percentile_scores(judge[qi, cj][None, :], np.ones(len(candidates)) / len(candidates))[0]
            assert abs(float(e.mean()) - .5) < 1e-10
            record[axis] = 100 * float(e[chosen].mean()); record[axis + '_random'] = 50.
        results.append(record)
        ranks.extend(dict(seed=seed, query=row['key'], method=method, rank=rank, candidate=d.metadata[candidates[j]]['key'], distance=float(values[j]))
                     for rank, j in enumerate(chosen, 1))
    return results, ranks


def passage_fit(vectors, catalog, seed, mode):
    result = []
    for role in ('validation', 'test'):
        for g in catalog.by_role[role]:
            js = list(g.rows); targets = {k: v[js] for k, v in catalog.targets.items()}
            wanted, support = target_distances(targets); support = torch.triu(support, diagonal=1).numpy()
            wanted = wanted.numpy()[support]
            z = vectors[js].astype(float); predicted = ((z[:, None] - z[None, :]) ** 2).sum(2)[support]
            assert len(wanted) > 0
            identifiable = np.std(wanted) > 1e-10
            rho = float(spearmanr(wanted, predicted).statistic) if identifiable and np.std(predicted) > 1e-10 else 0.
            result.append(dict(seed=seed, view=catalog.view, mode=mode, role=role, family_id=g.family, piece=g.piece,
                               grid=g.grid, kind=KINDS[g.kind - 1], category='P' if g.kind == 1 else 'H', signature=g.signature,
                               pairs=len(wanted), mae=float(np.mean(abs(predicted - wanted))),
                               rmse=float(np.sqrt(np.mean((predicted - wanted) ** 2))), spearman=rho,
                               target_sd=float(np.std(wanted)), predicted_sd=float(np.std(predicted)),
                               rank_identifiable=identifiable))
    return result


def aggregate_passages(rows, field):
    # Equal directions, then family/work-grid/H metric/passages within each direction.
    views = []
    for view in sorted(set(r['view'] for r in rows)):
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in rows:
            if r['view'] == view: tree[r['family_id']][r['piece'], r['grid']][r['kind']].append(r[field])
        views.append(float(np.mean([np.mean([np.mean([np.mean(v) for v in kinds.values()]) for kinds in works.values()]) for works in tree.values()])))
    return float(np.mean(views))


def summarize(records, within_records, passage_records, controls):
    per_seed = []
    for scope, rows in (('cross_work', records), ('within_work', within_records)):
        for role, method, seed in sorted(set((r['role'], r['method'], r['seed']) for r in rows)):
            rr = [r for r in rows if (r['role'], r['method'], r['seed']) == (role, method, seed)]
            fields = AXES + tuple(a + ('_same_work' if scope == 'cross_work' else '_random') for a in AXES)
            per_seed.append(dict(scope=scope, role=role, method=method, seed=seed, **{a: macro(rr, a) for a in fields}))
    write('per_seed_summary.csv', per_seed)
    summary = []
    for scope, role, method in sorted(set((r['scope'], r['role'], r['method']) for r in per_seed)):
        rr = [r for r in per_seed if (r['scope'], r['role'], r['method']) == (scope, role, method)]
        fields = [k for k in rr[0] if k not in ('scope', 'role', 'method', 'seed')]
        summary.append(dict(scope=scope, role=role, method=method, training_seeds=len(rr),
                            **{a: float(np.mean([r[a] for r in rr])) for a in fields},
                            **{a + '_seed_sd': float(np.std([r[a] for r in rr], ddof=1)) if len(rr) > 1 else 0. for a in fields}))
    write('summary.csv', summary)
    ps = []
    for role, mode, category, seed in sorted(set((r['role'], r['mode'], r['category'], r['seed']) for r in passage_records)):
        rr = [r for r in passage_records if (r['role'], r['mode'], r['category'], r['seed']) == (role, mode, category, seed)]
        ps.append(dict(role=role, mode=mode, category=category, seed=seed,
                       **{f: aggregate_passages(rr, f) for f in ('mae', 'rmse', 'spearman', 'target_sd', 'predicted_sd')}))
    write('passage_per_seed_summary.csv', ps)
    pr = []
    for role, mode, category in sorted(set((r['role'], r['mode'], r['category']) for r in ps)):
        rr = [r for r in ps if (r['role'], r['mode'], r['category']) == (role, mode, category)]
        fields = ('mae', 'rmse', 'spearman', 'target_sd', 'predicted_sd')
        pr.append(dict(role=role, mode=mode, category=category,
                       **{a: float(np.mean([r[a] for r in rr])) for a in fields},
                       **{a + '_seed_sd': float(np.std([r[a] for r in rr], ddof=1)) for a in fields}))
    write('passage_summary.csv', pr)
    cs = []
    for scope, role, mode, kind, control in sorted(set((r['scope'], r['role'], r['mode'], r['kind'], r['control']) for r in controls)):
        rr = [r for r in controls if (r['scope'], r['role'], r['mode'], r['kind'], r['control']) == (scope, role, mode, kind, control)]
        seed_values = {seed: [r for r in rr if r['seed'] == seed] for seed in sorted(set(r['seed'] for r in rr))}
        means = {a: [float(np.mean([r[a] for r in rs])) for rs in seed_values.values()] for a in AXES}
        cs.append(dict(scope=scope, role=role, mode=mode, kind=kind, control=control,
                       **{a: float(np.mean(v)) for a, v in means.items()},
                       **{a + '_seed_sd': float(np.std(v, ddof=1)) for a, v in means.items()}))
    write('control_summary.csv', cs)


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    protocol = json.loads((OUT / 'protocol.json').read_text()); seeds = protocol['training']['seeds']
    for seed in seeds:
        for view in (0, 1):
            for mode in MODES:
                assert (OUT / f'seed{seed}_view{view}_{mode}/training.json').exists(), 'Finish all24 training selections first'
    with np.load(CURVES / 'comparison.npz') as z: reference = {k: z[k] for k in z.files}
    with np.load(OUT / 'within_judges.npz') as z: extended = {k: z[k] for k in z.files}
    with np.load(OUT / 'baseline_distances.npz') as z: baselines = {k: z[k] for k in z.files}
    mapping = load_mapping(); records = []; within_records = []; ranks = []; within_ranks = []; passages = []; controls = []; calibrations = {}; stored = {}
    for view, direction in enumerate(('A_to_B', 'B_to_A')):
        catalog = GroupCatalog(view, mapping); d = catalog.data; cols = extended['gallery_columns']
        judges = {axis: reference[f'{direction}/evaluation/{axis}'] for axis in AXES}
        within_judges = {axis: extended[f'{direction}/{axis}'] for axis in AXES}
        allowed_sets = []
        groups = defaultdict(list)
        for j, i in enumerate(d.gids): groups[d.metadata[i]['piece'], d.metadata[i]['grid']].append(j)
        for control_seed in range(20261014, 20261034):
            rng = np.random.default_rng(control_seed)
            allowed_sets.append(np.array(sorted(j for js in groups.values() if len(js) >= 3 for j in rng.choice(js, 3, replace=False))))

        def score_distances(distances, seed, mode, is_control=False, control_seed=0, control_name=''):
            for kind, distance in distances.items():
                method = mode + '_' + kind
                rr, rk = evaluate(method, distance[:, cols], judges, d.metadata, d.gids, d.qids)
                wr, wk = within_work(method, distance, within_judges, d, seed)
                if is_control:
                    for scope, rows in (('cross_work', rr), ('within_work', wr)):
                        for role in ('test', 'validation'):
                            local = [r for r in rows if r['role'] == role]
                            controls.append(dict(seed=seed, direction=direction, mode=mode, kind=kind, role=role, scope=scope,
                                                  control=control_name, control_seed=control_seed, **{a: macro(local, a) for a in AXES}))
                else:
                    records.extend(dict(seed=seed, direction=direction, **r) for r in rr)
                    within_records.extend(dict(direction=direction, **r) for r in wr)
                    ranks.extend(dict(seed=seed, direction=direction, **r) for r in rk)
                    within_ranks.extend(dict(direction=direction, **r) for r in wk)
                    stored[f'{direction}/seed{seed}/{method}'] = distance

        for seed in seeds:
            for mode in (*MODES, 'fused_untrained', 'separate_untrained'):
                source_mode = mode.replace('_untrained', '_no_metric'); untrained = mode.endswith('untrained')
                directory = OUT / f'seed{seed}_view{view}_{source_mode}'
                cp = torch.load(directory / ('initial.pt' if untrained else 'best.pt'), weights_only=True)
                assert cp['data_sha256'] == sha(PREVIOUS / 'examples.npz') and cp['code_sha256'] == code_hash()
                model = ConditionalMetricEncoder(cp['fused_context']); model.load_state_dict(cp['state_dict']); model.eval()
                vectors = encode(model, catalog)
                np.savez_compressed(OUT / f'embeddings_seed{seed}_view{view}_{mode}.npz', vectors=vectors, owners=d.arrays['owner'], types=d.arrays['type'])
                distances, fitted = retrieval(vectors, d, cols, baselines[f'{direction}/B_summary'])
                calibrations[f'{direction}/seed{seed}/{mode}'] = fitted
                score_distances(distances, seed, mode)
                passages.extend(passage_fit(vectors, catalog, seed, mode))
                if not untrained:
                    for control_seed, allowed in zip(range(20261014, 20261034), allowed_sets):
                        for kind, distance in distances.items():
                            rr, _ = evaluate(mode + '_' + kind, distance[:, cols], judges, d.metadata, d.gids, d.qids, control_seed, allowed)
                            for role in ('test', 'validation'):
                                local = [r for r in rr if r['role'] == role]
                                controls.append(dict(seed=seed, direction=direction, mode=mode, kind=kind, role=role, scope='cross_work',
                                                     control='three_candidates', control_seed=control_seed, **{a: macro(local, a) for a in AXES}))
                    for control_seed in range(20261014, 20261019):
                        shuffled = encode(model, catalog, control_seed)
                        sd, _ = retrieval(shuffled, d, cols, baselines[f'{direction}/B_summary'], fitted)
                        score_distances(sd, seed, mode, True, control_seed, 'performance_order_permuted')
                print('Evaluated', direction, seed, mode, flush=True)

        for name in ('B', 'F', 'P', 'H', 'P+H'):
            score_distances({name: baselines[f'{direction}/{name}_summary']}, 0, 'summary')
        old_scales = json.loads((PREVIOUS / 'retrieval_scales.json').read_text())[direction + '/response_aux']
        with np.load(PREVIOUS / f'view{view}_response_aux/embeddings.npz') as z: vectors = z['vectors']
        fixed = {kind: old_scales[kind]['median'] for kind in KINDS}; fixed['H'] = old_scales['H']['median']
        scales = {kind: (np.array(old_scales[kind]['sd']), np.array(old_scales[kind]['active'])) for kind in KINDS}
        old_d, _ = retrieval(vectors, d, cols, baselines[f'{direction}/B_summary'], fixed, scales)
        with np.load(PREVIOUS / 'distances.npz') as z:
            for kind, values in old_d.items():
                assert np.allclose(values[d.ids][:, cols], z[f'{direction}/response_aux_{kind}'][d.ids], atol=1e-10, rtol=0)
        score_distances(old_d, 0, 'previous_response_aux')
    write('scores.csv', records); write('within_work_scores.csv', within_records)
    write('rankings.csv', ranks); write('within_work_rankings.csv', within_ranks)
    write('passage_metrics.csv', passages); write('controls.csv', controls)
    summarize(records, within_records, passages, controls)
    np.savez_compressed(OUT / 'distances.npz', **stored)
    (OUT / 'retrieval_calibrations.json').write_text(json.dumps(calibrations, indent=2) + '\n')
    print('Completed all24-model evaluation, controls, and old baseline checks.', flush=True)


if __name__ == '__main__': main()
