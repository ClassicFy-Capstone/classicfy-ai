"""Independent scalar input, distance, ranking and score verification.

Imports only the earlier independent verifier's arithmetic helpers. No
production descriptor, distance, ranking or response function is called.
"""

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np

from verify_structure_response_comparison import read, digest, weights, matrix, ranking, scalar_response

ROOT = Path(__file__).resolve().parents[2]


def macro(records):
    families = defaultdict(lambda: defaultdict(list))
    for row in records:
        families[row['family_id']][row['piece']].append(float(row['score']))
    by_family = {f: float(np.mean([np.mean(v) for v in works.values()])) for f, works in families.items()}
    return float(np.mean(list(by_family.values()))), by_family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'classicfy-ai/analysis/standalone_structure')
    parser.add_argument('--artifacts', type=Path, default=ROOT.parent/'datasets/standalone_structure_run01')
    args = parser.parse_args(); out = args.out
    prov = json.loads((out/'provenance.json').read_text())
    for path, value in prov['inputs'].items():
        assert digest(path) == value, path
    for path, value in prov['code'].items():
        assert digest(ROOT/path) == value, path
    for path, value in prov['results'].items():
        assert digest(out/path) == value, path
    assert digest(args.artifacts/'standalone.npz') == prov['artifacts']['standalone.npz']
    z = np.load(args.artifacts/'standalone.npz', allow_pickle=False)
    keys = z['keys'].tolist(); gids = z['gallery_indices']; qids = z['query_indices']
    coverage = {r['key']: r for r in read(ROOT/'classicfy-ai/analysis/phrase_response/coverage.csv')}
    metadata = [coverage[k] for k in keys]; gallery = [metadata[i] for i in gids]
    family = np.array([r['family_id'] for r in gallery]); w = weights(gallery)
    assert not {metadata[i]['family_id'] for i in qids} & set(family)
    assert keys == json.loads((out/'cohort.json').read_text())['keys']
    scales = json.loads((out/'scales.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    ranks = read(out/'rankings.csv'); scores = read(out/'scores.csv'); controls = read(out/'controls.csv')
    pp = {(r['key'], r['view']): r for r in read(ROOT/'classicfy-ai/analysis/phrase_response/responses.csv')}
    import csv
    with gzip.open(ROOT/'classicfy-ai/analysis/tonal_tension/responses.csv.gz', 'rt', encoding='utf-8-sig') as f:
        hh = {(r['key'], r['view'], r['metric']): r for r in csv.DictReader(f)}
    checks = 0; maximum = 0.
    def equal(a, b, tolerance=1e-10):
        nonlocal checks, maximum
        a, b = np.asarray(a, float), np.asarray(b, float)
        assert a.shape == b.shape and np.array_equal(np.isfinite(a), np.isfinite(b))
        e = float(np.max(abs(a-b))) if a.size else 0.
        assert e <= tolerance, e
        checks += a.size; maximum = max(maximum, e)
    works = defaultdict(list)
    for j, row in enumerate(gallery):
        works[row['piece'], row['grid']].append(j)
    rank_lookup = defaultdict(list)
    for r in ranks:
        rank_lookup[r['direction'], r['query'], r['method']].append(r['candidate'])
    expected_rows = []
    for view, direction in enumerate(('A_to_B', 'B_to_A')):
        for i, key in enumerate(keys):
            x = z[f'{i}/values']; mask = z[f'{i}/mask']; base_pair = z[f'{i}/pairs']
            phase = z[f'{i}/phase']; owner = z[f'{i}/owner']; n = len(mask)
            valid = np.array([mask[j] and j//64%2 == view and j < n//64*64 for j in range(n)])
            pair = np.array([bool(base_pair[j]) and j > 0 and valid[j] and valid[j-1] and j%64 != 0 for j in range(n)])
            xx = x[:, valid]; location = xx.mean(axis=1); location[1] = np.median(abs(xx[1]))
            b = np.column_stack([location, np.ptp(np.percentile(xx, [5, 95], axis=1), axis=0)]).ravel()
            equal(b, z[direction+'_input_B'][i])
            delta = np.diff(x)[:, pair[1:]]
            t = np.r_[delta.mean(axis=1), np.sqrt((delta*delta).mean(axis=1)), np.mean(delta[0]*delta[2])]
            equal(t, z[direction+'_input_T'][i])
            index = np.arange(n); fixed = np.minimum(2, (index%16)*3//16)
            equal(scalar_response(x, valid, pair, fixed, index//16), z[direction+'_input_F'][i])
            equal(scalar_response(x, valid, pair, phase, owner), z[direction+'_input_P'][i])
            equal([float(pp[key, 'AB'[view]][f'p{j}']) for j in range(66)], z[direction+'_input_P'][i])
            equal([float(hh[key, 'AB'[view], metric][f'h{j}']) for metric in
                   ('cloud_diameter', 'cloud_momentum', 'tensile_strain') for j in range(66)], z[direction+'_input_H'][i])
            # All judges use the opposite view, never the search beats.
            v = 1-view
            chosen = np.array([mask[j] and j//64%2 == v and j < n//64*64 for j in range(n)])
            pair = np.array([bool(base_pair[j]) and j > 0 and chosen[j] and chosen[j-1] and j%64 != 0 for j in range(n)])
            blocks = [j for j in range(0, n-63, 64) if j//64%2 == v]
            el, et, correlations, es = [], [], [], []
            for c in range(7):
                values = x[c, chosen]
                el.extend([np.median(abs(values)) if c == 1 else np.median(values), np.ptp(np.percentile(values, [25, 75]))])
                rms = []
                for start in blocks:
                    for offset in (0, 32):
                        ps = [j for j in range(start+offset+1, start+offset+32) if pair[j]]
                        if len(ps) >= 25:
                            rms.append(math.sqrt(math.fsum(float(x[c, j]-x[c, j-1])**2 for j in ps)/len(ps)))
                et.extend(np.log1p(np.percentile(rms, [25, 50, 75])))
            for start in blocks:
                ps = [j for j in range(start+1, start+64) if pair[j]]
                if len(ps) >= 51:
                    a = np.array([x[0, j]-x[0, j-1] for j in ps]); b = np.array([x[2, j]-x[2, j-1] for j in ps])
                    a -= a.mean(); b -= b.mean()
                    if min(a.std(), b.std()) > 1e-12:
                        correlations.append(float(np.dot(a, b)/np.sqrt(np.dot(a, a)*np.dot(b, b))))
            er = np.r_[np.percentile(correlations, [25, 50, 75]), np.std(correlations)]
            for p in range(3):
                ps = [j for j in range(1, n) if pair[j] and phase[j-1] == phase[j] == p and owner[j-1] == owner[j]]
                a = np.array([x[0, j]-x[0, j-1] for j in ps]); b = np.array([x[2, j]-x[2, j-1] for j in ps])
                es.extend([np.mean(a > 1e-6), np.mean(b > 1e-6), np.mean(a*b > 1e-12)])
            for axis, a in [('E-L', el), ('E-T', et), ('E-R', er), ('E-S', es)]:
                equal(a, z[direction+'_judge_'+axis][i])
        medians = {}; blocks = {}
        a, b = np.triu_indices(len(gids), 1); good = family[a] != family[b]; a, b = a[good], b[good]
        for name in ('B', 'T', 'F', 'P', 'H'):
            x = z[direction+'_input_'+name]; scale = scales[direction]['scales'][name]
            if scale:
                mean = w @ x[gids]; sd = np.sqrt(w @ ((x[gids]-mean)**2))
                equal(mean, scale['mean']); equal(sd, scale['scale'])
                assert (sd > 1e-12).tolist() == scale['active']
            d = matrix(x, x[gids], scale, 2 if name == 'B' else None)
            values = d[gids[a], b]; pw = w[a]*w[b]; order = np.argsort(values, kind='stable')
            median = float(values[order][np.searchsorted(np.cumsum(pw[order]), pw.sum()/2)])
            equal(median, scales[direction]['block_medians'][name]); medians[name] = median; blocks[name] = d/median
        dd = {name: np.sqrt(sum(blocks[c]**2 for c in components)/len(components))
              for name, components in protocol['arms'].items() if name != 'M-CT'}
        dd['M-CT'] = np.array([[.5*(r['composer'] != g['composer'])+.5*(r['work_type'] != g['work_type'])
                                for g in gallery] for r in metadata])
        for name, d in dd.items():
            equal(d, z[direction+'_distance_'+name])
        eval_scores = {}
        for axis in ('E-L', 'E-T', 'E-R', 'E-S'):
            x = z[direction+'_judge_'+axis]; scale = scales[direction]['evaluation_scales'][axis]
            mean = w @ x[gids]; sd = np.sqrt(w @ ((x[gids]-mean)**2))
            equal(mean, scale['mean']); equal(sd, scale['scale'])
            d = matrix(x, x[gids], scale, 2 if axis == 'E-L' else 3 if axis == 'E-T' else None)
            e = np.array([np.array([w[v > target+1e-12].sum()+.5*w[abs(v-target) <= 1e-12].sum()
                                    for target in v]) for v in d])
            equal(e, z[direction+'_evaluation_'+axis]); eval_scores[axis] = e
        score_lookup = {(r['key'], r['method'], r['axis']): r for r in scores if r['direction'] == direction}
        for qi in qids:
            for name, d in dd.items():
                selected = ranking(d[qi], family, keys[qi], 20261014)
                assert [gallery[j]['key'] for j in selected] == rank_lookup[direction, keys[qi], name]
                choices = [selected] if name != 'M-CT' else [ranking(d[qi], family, keys[qi], seed)
                    for seed in range(20261014, 20261114)]
                for axis, e in eval_scores.items():
                    value = float(np.mean([e[qi, c].mean() for c in choices]))
                    sw = float(np.mean([np.mean([e[qi, works[gallery[j]['piece'], gallery[j]['grid']]].mean()
                                                 for j in c]) for c in choices]))
                    record = score_lookup[keys[qi], name, axis]
                    equal(value, float(record['score'])); equal(sw, float(record['same_work_expected']))
                    expected_rows.append(dict(**metadata[qi], role=record['role'], method=name, axis=axis, score=value))
        for seed in range(20261014, 20261034):
            rng = np.random.default_rng(seed)
            balanced = np.array(sorted(j for group in sorted(works) for j in rng.choice(works[group], 3, replace=False)))
            equal(balanced, z[f'{direction}_balanced_{seed}'])
            for name, d in dd.items():
                selected = {}
                for qi in qids:
                    limit = np.full(len(gallery), np.inf); limit[balanced] = d[qi, balanced]
                    selected[qi] = ranking(limit, family, keys[qi], seed)
                for axis, e in eval_scores.items():
                    for role in ('test', 'validation'):
                        rs = [dict(**metadata[qi], score=float(e[qi, selected[qi]].mean())) for qi in qids
                              if metadata[qi]['evaluation_role'] == role]
                        value, _ = macro(rs)
                        saved = next(r for r in controls if (r['direction'], int(r['seed']), r['method'], r['axis'], r['role'])
                                     == (direction, seed, name, axis, role))
                        equal(value, float(saved['score']))
    for row in read(out/'summary.csv'):
        rs = [r for r in expected_rows if (r['role'], r['method'], r['axis']) == (row['role'], row['method'], row['axis'])]
        value, _ = macro(rs); equal(value, float(row['score']))
    for row in read(out/'per_family.csv'):
        rs = [r for r in expected_rows if (r['role'], r['method'], r['axis']) == (row['role'], row['method'], row['axis'])]
        _, fs = macro(rs); equal(fs[row['family_id']], float(row['score']))
    result = dict(status='passed', numerical_checks=checks, maximum_absolute_error=maximum,
        scope='Independent scalar B/T/F/P and held-out judges, verified H CSV, train scales, all standalone/blended distances, Top5, metadata ties, same-work expectations, 20 balanced-gallery seeds and macro aggregation',
        limitations='Reuses previously verified score/feature join and H CSV; no musical phrase/key accuracy or new tension-episode judge',
        verifier_sha256=digest(Path(__file__)))
    (out/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
