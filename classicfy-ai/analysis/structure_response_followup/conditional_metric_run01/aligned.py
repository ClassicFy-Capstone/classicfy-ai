"""Exploratory fixed corresponding-passage inference, without retraining."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from data import GroupCatalog, load_mapping
from train import MODES
from evaluate import within_work, write, macro, AXES

OUT = Path(__file__).resolve().parent


def matrices(vectors, catalog):
    d = catalog.data; columns = {int(i): j for j, i in enumerate(d.ids)}
    work_owners = defaultdict(set)
    for i in d.qids:
        row = d.metadata[i]; work_owners[row['piece'], row['grid']].add(int(i))
    passages = defaultdict(list)
    for g in catalog.groups:
        if g.role == 'train': continue
        owners = set(d.arrays['owner'][list(g.rows)].tolist())
        if owners == work_owners[g.piece, g.grid]: passages[g.piece, g.grid, g.kind].append(g)
    result = {kind: np.full((335, len(d.ids)), np.inf) for kind in ('P', 'H')}
    counts = []
    for (piece, grid), owners in work_owners.items():
        owners = sorted(owners); cols = [columns[i] for i in owners]; blocks = {}
        for kind in range(1, 5):
            groups = passages[piece, grid, kind]
            assert groups, 'Shared passage support required for every recording'
            costs = []
            for g in groups:
                rows = {int(d.arrays['owner'][j]): j for j in g.rows}
                z = vectors[[rows[i] for i in owners]].astype(float)
                costs.append(((z[:, None] - z[None, :]) ** 2).sum(2))
            blocks[kind] = np.mean(costs, axis=0)
            counts.append(dict(view=catalog.view, piece=piece, grid=grid, kind=kind, shared_passages=len(groups), owners=len(owners)))
        result['P'][np.ix_(owners, cols)] = blocks[1]
        result['H'][np.ix_(owners, cols)] = np.mean([blocks[k] for k in range(2, 5)], axis=0)
    return result, counts


def main():
    seeds = json.loads((OUT / 'protocol.json').read_text())['training']['seeds']
    with np.load(OUT / 'within_judges.npz') as z: judgments = {k: z[k] for k in z.files}
    mapping = load_mapping(); records = []; ranks = []; support = []; stored = {}
    for view, direction in enumerate(('A_to_B', 'B_to_A')):
        catalog = GroupCatalog(view, mapping)
        judges = {a: judgments[f'{direction}/{a}'] for a in AXES}
        for seed in seeds:
            for mode in (*MODES, 'fused_untrained', 'separate_untrained'):
                with np.load(OUT / f'embeddings_seed{seed}_view{view}_{mode}.npz') as z: vectors = z['vectors']
                distances, counts = matrices(vectors, catalog)
                if seed == seeds[0] and mode == 'fused_no_metric': support.extend(counts)
                for kind, distance in distances.items():
                    method = mode + '_' + kind
                    rr, rk = within_work(method, distance, judges, catalog.data, seed)
                    records.extend(dict(direction=direction, **r) for r in rr)
                    ranks.extend(dict(direction=direction, **r) for r in rk)
                    stored[f'{direction}/seed{seed}/{method}'] = distance
    write('aligned_scores.csv', records); write('aligned_rankings.csv', ranks); write('aligned_support.csv', support)
    per_seed = []
    for role, method, seed in sorted(set((r['role'], r['method'], r['seed']) for r in records)):
        rows = [r for r in records if (r['role'], r['method'], r['seed']) == (role, method, seed)]
        per_seed.append(dict(role=role, method=method, seed=seed, **{a: macro(rows, a) for a in AXES}))
    write('aligned_per_seed_summary.csv', per_seed)
    summary = []
    for role, method in sorted(set((r['role'], r['method']) for r in per_seed)):
        rows = [r for r in per_seed if (r['role'], r['method']) == (role, method)]
        summary.append(dict(role=role, method=method, **{a: float(np.mean([r[a] for r in rows])) for a in AXES},
                            **{a + '_seed_sd': float(np.std([r[a] for r in rows], ddof=1)) for a in AXES}))
    write('aligned_summary.csv', summary)
    np.savez_compressed(OUT / 'aligned_distances.npz', **stored)
    for r in summary:
        if r['role'] == 'test' and r['method'].startswith('separate_metric_'):
            print(r['method'], 'E-PN', round(r['E-PN'], 2), 'E-HL', round(r['E-HL'], 2), flush=True)


if __name__ == '__main__': main()
