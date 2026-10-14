"""Scalar audit of the additional fixed aligned-passage diagnostic."""
from collections import defaultdict
import json
import importlib.util
from pathlib import Path
import numpy as np
from data import GroupCatalog, load_mapping
from evaluate import AXES
from embedding.retrieval_probe import priorities

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('conditional_run_verifier', OUT / 'verify.py')
local_verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_verify)
read, scalar_rank, scalar_macro = local_verify.read, local_verify.scalar_rank, local_verify.scalar_macro


def main():
    checks = 0; maximum = 0.
    def close(a, b):
        nonlocal checks, maximum
        error = abs(float(a) - float(b)); assert error < 1e-9
        checks += 1; maximum = max(maximum, error)
    catalogs = [GroupCatalog(v, load_mapping()) for v in (0, 1)]
    d = catalogs[0].data; key_ids = {k: i for i, k in enumerate(d.keys)}; columns = {int(i): j for j, i in enumerate(d.ids)}
    with np.load(OUT / 'aligned_distances.npz') as z: distances = {k: z[k] for k in z.files}
    with np.load(OUT / 'within_judges.npz') as z: judges = {k: z[k] for k in z.files}
    rank_map = defaultdict(list)
    for r in read('aligned_rankings.csv'): rank_map[r['direction'], r['seed'], r['method'], r['query']].append(r)
    work = defaultdict(list)
    for i in d.qids:
        m = d.metadata[i]; work[m['piece'], m['grid']].append(int(i))
    scores = read('aligned_scores.csv')
    for r in scores:
        qi = key_ids[r['key']]; meta = d.metadata[qi]; ids = [i for i in work[meta['piece'], meta['grid']] if i != qi]
        cj = [columns[i] for i in ids]; matrix = distances[f"{r['direction']}/seed{r['seed']}/{r['method']}"]
        selected = scalar_rank(matrix[qi, cj], priorities(20261014, r['key'], len(d.ids))[cj], [0] * len(ids), 3, False)
        actual = sorted(rank_map[r['direction'], r['seed'], r['method'], r['key']], key=lambda a: int(a['rank']))
        assert [key_ids[a['candidate']] for a in actual] == [ids[j] for j in selected]; checks += 1
        for a, j in zip(actual, selected): close(a['distance'], matrix[qi, cj[j]])
        for axis in AXES:
            costs = judges[f"{r['direction']}/{axis}"][qi, cj]
            percentile = [sum(1 if other > value + 1e-12 else .5 if abs(other - value) <= 1e-12 else 0 for other in costs) / len(costs) for value in costs]
            close(r[axis], 100 * np.mean([percentile[j] for j in selected])); close(r[axis + '_random'], 50.)
    per_seed = read('aligned_per_seed_summary.csv')
    for r in per_seed:
        rows = [a for a in scores if (a['role'], a['method'], a['seed']) == (r['role'], r['method'], r['seed'])]
        for axis in AXES: close(r[axis], scalar_macro(rows, axis))
    for r in read('aligned_summary.csv'):
        rows = [a for a in per_seed if (a['role'], a['method']) == (r['role'], r['method'])]
        for axis in AXES:
            values = [float(a[axis]) for a in rows]; close(r[axis], np.mean(values)); close(r[axis + '_seed_sd'], np.std(values, ddof=1))
    # Distance formula samples for every work, type, direction, and primary seed.
    for view, catalog in enumerate(catalogs):
        direction = ('A_to_B', 'B_to_A')[view]
        for seed in (20261014, 20261015, 20261016):
            with np.load(OUT / f'embeddings_seed{seed}_view{view}_separate_metric.npz') as z: vectors = z['vectors']
            for (piece, grid), owners in work.items():
                a, b = sorted(owners)[:2]; by_kind = defaultdict(list)
                for g in catalog.groups:
                    if g.piece != piece or g.grid != grid: continue
                    rows = {int(catalog.data.arrays['owner'][j]): j for j in g.rows}
                    if set(rows) != set(owners): continue
                    za = vectors[rows[a]].astype(float); zb = vectors[rows[b]].astype(float)
                    by_kind[g.kind].append(sum(float((za[c] - zb[c]) ** 2) for c in range(32)))
                close(distances[f'{direction}/seed{seed}/separate_metric_P'][a, columns[b]], np.mean(by_kind[1]))
                close(distances[f'{direction}/seed{seed}/separate_metric_H'][a, columns[b]], np.mean([np.mean(by_kind[k]) for k in (2, 3, 4)]))
    result = dict(status='passed', checks=checks, max_scalar_error=maximum,
                  scope='All exploratory aligned rankings, opposite-view scores, random expectations and seed aggregates; scalar corresponding-passage distance samples for all6 works, both views, all3 primary seeds.',
                  limits='Additional inference diagnostic was defined after seeing primary results and remains exploratory.')
    (OUT / 'aligned_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
