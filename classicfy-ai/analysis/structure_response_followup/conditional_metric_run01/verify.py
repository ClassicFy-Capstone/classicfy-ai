"""Independent ranking/score aggregation audit and selected checkpoint reconstruction."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import torch
from data import GroupCatalog, load_mapping, PREVIOUS, WORK, sha
from model import ConditionalMetricEncoder, target_distances
from train import MODES, code_hash
from prepare_judges import AXES
from embedding.retrieval_probe import priorities

OUT = Path(__file__).resolve().parent


def read(name):
    with (OUT / name).open() as f: return list(csv.DictReader(f))


def scalar_rank(values, priority, families, k, cap):
    ordered = sorted(range(len(values)), key=lambda j: (values[j], j)); tiers = []; start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and values[ordered[stop]] - values[ordered[start]] <= 1e-12: stop += 1
        tiers.extend(sorted(ordered[start:stop], key=lambda j: priority[j])); start = stop
    chosen = []; seen = set()
    for j in tiers:
        if cap and families[j] in seen: continue
        chosen.append(j); seen.add(families[j])
        if len(chosen) == k: return chosen
    raise AssertionError('Insufficient candidates')


def scalar_macro(rows, field):
    tree = defaultdict(lambda: defaultdict(list))
    for r in rows: tree[r['family_id']][r['piece']].append(float(r[field]))
    return float(np.mean([np.mean([np.mean(v) for v in works.values()]) for works in tree.values()]))


def main():
    torch.set_num_threads(2)
    checks = 0; max_error = 0.; max_embedding_error = 0.
    def close(a, b, tolerance=1e-9):
        nonlocal checks, max_error
        error = abs(float(a) - float(b)); assert error <= tolerance, (a, b, error)
        checks += 1; max_error = max(max_error, error)
    protocol = json.loads((OUT / 'protocol.json').read_text()); seeds = protocol['training']['seeds']
    mapping = load_mapping(); catalogs = [GroupCatalog(v, mapping) for v in (0, 1)]
    with np.load(WORK / 'diagnostics/curve_comparison/comparison.npz') as z: reference = {k: z[k] for k in z.files}
    with np.load(OUT / 'within_judges.npz') as z: extended = {k: z[k] for k in z.files}
    with np.load(OUT / 'distances.npz') as z: distances = {k: z[k] for k in z.files}
    d = catalogs[0].data; key_ids = {key: i for i, key in enumerate(d.keys)}; columns = {int(i): j for j, i in enumerate(d.ids)}
    gallery_columns = extended['gallery_columns']; families = [d.metadata[i]['family_id'] for i in d.gids]
    work_gallery = defaultdict(list); work_all = defaultdict(list)
    for j, i in enumerate(d.gids): work_gallery[d.metadata[i]['piece'], d.metadata[i]['grid']].append(j)
    for i in d.ids: work_all[d.metadata[i]['piece'], d.metadata[i]['grid']].append(int(i))
    for seed in seeds:
        for view in (0, 1):
            initial_states = []; streams = []
            catalog = catalogs[view]
            for mode in MODES:
                folder = OUT / f'seed{seed}_view{view}_{mode}'
                cp = torch.load(folder / 'best.pt', weights_only=True); initial = torch.load(folder / 'initial.pt', weights_only=True)
                assert cp['code_sha256'] == code_hash() and cp['data_sha256'] == sha(PREVIOUS / 'examples.npz')
                assert cp['input_scales'] == catalog.data.scales and cp['amplitude_scales'] == catalog.amplitude_scales
                assert cp['train_owners'] == catalog.data.gids.tolist() and not cp['test_used_for_training_or_selection']
                assert not set(cp['train_owners']) & set(cp['validation_owners'])
                assert not (set(cp['train_owners']) | set(cp['validation_owners'])) & set(catalog.data.arrays['owner'][catalog.data.roles['test']].tolist())
                history = read(f'{folder.name}/history.csv')
                minimum = min(float(r['validation_loss']) for r in history)
                close(cp['validation']['loss'], minimum, 1e-6)
                close(cp['validation']['loss'], next(float(r['validation_loss']) for r in history if int(r['epoch']) == cp['epoch']))
                initial_states.append(initial['state_dict']); streams.append({r['epoch']: r['sampling_sha256'] for r in history if int(r['epoch']) > 0})
                model = ConditionalMetricEncoder(cp['fused_context']); model.load_state_dict(cp['state_dict']); model.eval()
                with np.load(OUT / f'embeddings_seed{seed}_view{view}_{mode}.npz') as z: vectors = z['vectors']
                indices = np.unique(np.rint(np.linspace(0, len(vectors) - 1, 11)).astype(int))
                with torch.no_grad(): predicted = model.encode(catalog.data.x[indices], catalog.data.context[indices]).numpy()
                error = float(abs(predicted - vectors[indices]).max()); assert error < 1e-5
                max_embedding_error = max(max_embedding_error, error); checks += 7
            for state in initial_states[1:]:
                assert all(torch.equal(state[k], initial_states[0][k]) for k in state); checks += 1
            for stream in streams[1:]:
                for epoch in set(stream) & set(streams[0]): assert stream[epoch] == streams[0][epoch]; checks += 1

    for scope, rank_file, score_file in (('cross', 'rankings.csv', 'scores.csv'), ('within', 'within_work_rankings.csv', 'within_work_scores.csv')):
        rank_rows = read(rank_file); score_rows = read(score_file); rank_map = defaultdict(list)
        for r in rank_rows: rank_map[r['direction'], r['seed'], r['method'], r['query']].append(r)
        for r in score_rows:
            direction, seed, method, key = r['direction'], r['seed'], r['method'], r['key']; qi = key_ids[key]
            actual = sorted(rank_map[direction, seed, method, key], key=lambda x: int(x['rank']))
            matrix = distances[f'{direction}/seed{seed}/{method}']
            if scope == 'cross':
                values = matrix[qi, gallery_columns]; priority = priorities(20261014, key, len(d.gids))
                selected = scalar_rank(values, priority, families, 5, True); ids = [int(d.gids[j]) for j in selected]
                assert [key_ids[a['candidate']] for a in actual] == ids; checks += 1
                for a, j in zip(actual, selected): close(a['distance'], values[j])
                for axis in AXES:
                    judge = reference[f'{direction}/evaluation/{axis}']
                    close(r[axis], 100 * float(np.mean([judge[qi, j] for j in selected])))
                    expectation = []
                    for i in ids:
                        meta = d.metadata[i]; js = work_gallery[meta['piece'], meta['grid']]
                        expectation.append(float(np.mean([judge[qi, j] for j in js])))
                    close(r[axis + '_same_work'], 100 * float(np.mean(expectation)))
            else:
                meta = d.metadata[qi]; candidates = [i for i in work_all[meta['piece'], meta['grid']] if i != qi]
                cj = [columns[i] for i in candidates]; values = matrix[qi, cj]
                priority = priorities(20261014, key, len(d.ids))[cj]
                selected = scalar_rank(values, priority, [0] * len(candidates), 3, False)
                assert [key_ids[a['candidate']] for a in actual] == [candidates[j] for j in selected]; checks += 1
                for a, j in zip(actual, selected): close(a['distance'], values[j])
                for axis in AXES:
                    costs = extended[f'{direction}/{axis}'][qi, cj]
                    percentile = []
                    for value in costs:
                        percentile.append(sum(1 if other > value + 1e-12 else .5 if abs(other - value) <= 1e-12 else 0 for other in costs) / len(costs))
                    close(r[axis], 100 * float(np.mean([percentile[j] for j in selected])))
                    close(r[axis + '_random'], 50.); close(np.mean(percentile), .5)
    all_scores = {'cross_work': read('scores.csv'), 'within_work': read('within_work_scores.csv')}
    for r in read('per_seed_summary.csv'):
        rows = [a for a in all_scores[r['scope']] if (a['role'], a['method'], a['seed']) == (r['role'], r['method'], r['seed'])]
        for field in r:
            if field not in ('scope', 'role', 'method', 'seed') and r[field] != '': close(r[field], scalar_macro(rows, field))
    per_seed = read('per_seed_summary.csv')
    for r in read('summary.csv'):
        rows = [a for a in per_seed if (a['scope'], a['role'], a['method']) == (r['scope'], r['role'], r['method'])]
        close(r['training_seeds'], len(rows))
        for field in r:
            if field in ('scope', 'role', 'method', 'training_seeds') or field.endswith('_seed_sd') or r[field] == '': continue
            values = [float(a[field]) for a in rows]; close(r[field], np.mean(values))
            close(r[field + '_seed_sd'], np.std(values, ddof=1) if len(values) > 1 else 0.)
    # Independently recalculate per-passage diagnostics and hierarchical aggregates.
    passage_rows = read('passage_metrics.csv'); vectors_cache = {}; target_cache = {}; weights_cache = {}
    def ranks_average(values):
        order = np.argsort(values, kind='stable'); result = np.zeros(len(values)); start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and values[order[stop]] == values[order[start]]: stop += 1
            result[order[start:stop]] = (start + stop - 1) / 2; start = stop
        return result
    for catalog in catalogs:
        for role in ('test', 'validation'):
            for group, weight in zip(catalog.by_role[role], catalog.group_weights(role)):
                kind = ('P', 'H0', 'H1', 'H2')[group.kind - 1]
                token = (catalog.view, role, group.piece, group.grid, kind, group.signature)
                js = list(group.rows)
                target, supported = target_distances({k: v[js] for k, v in catalog.targets.items()})
                mask = torch.triu(supported, diagonal=1).numpy()
                target_cache[token] = (js, mask, target.numpy()[mask]); weights_cache[token] = float(weight)
    for r in passage_rows:
        view = int(r['view']); seed = int(r['seed']); mode = r['mode']; key = (view, seed, mode)
        if key not in vectors_cache:
            with np.load(OUT / f'embeddings_seed{seed}_view{view}_{mode}.npz') as z: vectors_cache[key] = z['vectors']
        token = (view, r['role'], r['piece'], r['grid'], r['kind'], r['signature'])
        js, mask, wanted = target_cache[token]; z = vectors_cache[key][js].astype(float)
        predicted = ((z[:, None] - z[None, :]) ** 2).sum(2)[mask]
        close(r['pairs'], len(wanted)); close(r['mae'], np.mean(abs(wanted - predicted)))
        close(r['rmse'], np.sqrt(np.mean((wanted - predicted) ** 2)))
        close(r['target_sd'], np.std(wanted)); close(r['predicted_sd'], np.std(predicted))
        rho = float(np.corrcoef(ranks_average(wanted), ranks_average(predicted))[0, 1]) if min(np.std(wanted), np.std(predicted)) > 1e-10 else 0.
        close(r['spearman'], rho)
    passage_seed = read('passage_per_seed_summary.csv')
    fields = ('mae', 'rmse', 'spearman', 'target_sd', 'predicted_sd')
    for r in passage_seed:
        selected = [a for a in passage_rows if (a['role'], a['mode'], a['category'], a['seed']) == (r['role'], r['mode'], r['category'], r['seed'])]
        for field in fields:
            direction_values = []
            for view in (0, 1):
                local = [a for a in selected if int(a['view']) == view]
                ww = [weights_cache[(view, a['role'], a['piece'], a['grid'], a['kind'], a['signature'])] for a in local]
                direction_values.append(np.average([float(a[field]) for a in local], weights=ww))
            close(r[field], np.mean(direction_values))
    for r in read('passage_summary.csv'):
        selected = [a for a in passage_seed if (a['role'], a['mode'], a['category']) == (r['role'], r['mode'], r['category'])]
        for field in fields:
            values = [float(a[field]) for a in selected]
            close(r[field], np.mean(values)); close(r[field + '_seed_sd'], np.std(values, ddof=1))
    control_rows = read('controls.csv')
    for r in read('control_summary.csv'):
        selected = [a for a in control_rows if all(a[k] == r[k] for k in ('scope', 'role', 'mode', 'kind', 'control'))]
        for axis in AXES:
            means = [np.mean([float(a[axis]) for a in selected if int(a['seed']) == seed]) for seed in seeds]
            close(r[axis], np.mean(means)); close(r[axis + '_seed_sd'], np.std(means, ddof=1))
    # Independent scalar target formula on real heldout passages (masks and scale support).
    for catalog in catalogs:
        for g in catalog.by_role['test'][:12]:
            js = list(g.rows); target, _ = target_distances({k: v[js] for k, v in catalog.targets.items()})
            for i, j in ((0, 1), (0, len(js) - 1)):
                numerator = 0.; denominator = 0.
                for name, weight in (('shape', .6), ('amplitude', .2), ('lag', .2)):
                    a = catalog.targets[name][js[i]].numpy().ravel().astype(float); b = catalog.targets[name][js[j]].numpy().ravel().astype(float)
                    valid = catalog.targets[name + '_mask'][js[i]].numpy().ravel() & catalog.targets[name + '_mask'][js[j]].numpy().ravel()
                    if valid.any(): numerator += weight * np.mean((a[valid] - b[valid]) ** 2); denominator += weight
                close(target[i, j], numerator / denominator, 1e-6)
    result = dict(status='passed', checks=checks, max_scalar_error=max_error, max_checkpoint_reencode_error=max_embedding_error,
                  scope='All24 selected checkpoints and input/code/scaler identities; paired initializations and all shared epoch sample/mask streams; every cross-work and within-work TopK rank, score and random-recording expectation; all retrieval and passage metrics/aggregates, every seed mean/SD and control aggregate; scalar target samples.',
                  unit_tests=18, limits='Control aggregate arithmetic checked, not every shuffled ranking independently replayed. Numeric proxies and a reused test do not constitute perceptual or new-data validation.')
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
