"""Rebuild verified score tension and retain individual ordered segment curves."""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
REPO = WORK / 'capstone-2026-2-classicfy'
sys.path.insert(0, str(REPO / 'classicfy-ai/src'))
from preprocessing import load_musicxml
from features.tonal_tension import TonalContext, tension_labels
from features.structure_response import response, split_view
POINTS = 17


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain(curves, spans, maximum=32):
    if len(curves) > maximum:
        ids = np.rint(np.linspace(0, len(curves)-1, maximum)).astype(int)
        curves, spans = [curves[j] for j in ids], [spans[j] for j in ids]
    return np.asarray(curves).reshape(-1, 7, POINTS), np.asarray(spans, int).reshape(-1, 2)


def curve(x, a, b, positions=None):
    u = np.linspace(0, 1, b-a) if positions is None else positions[a:b]
    target = np.linspace(u[0], u[-1], POINTS)
    return np.array([np.interp(target, u, row[a:b]) for row in x])


def build(x, mask, pairs, owner, written, tension, key_index, view):
    valid, adjacent, blocks = split_view(mask, pairs, view)
    result, spans = {}, {}
    for kind in ('F', 'P'):
        curves, locations = [], []
        if kind == 'F':
            runs = [(a, a+16) for block in blocks for a in range(block, block+64, 16)]
        else:
            edges = np.r_[0, np.flatnonzero(owner[1:] != owner[:-1])+1, len(owner)]
            runs = list(zip(edges[:-1], edges[1:]))
        for a, b in runs:
            if b-a < 4 or a//64 != (b-1)//64 or not valid[a:b].all() or not adjacent[a+1:b].all():
                continue
            if kind == 'P' and owner[a] < 0:
                continue
            positions = np.mean(written, axis=1) if kind == 'P' else None
            if positions is not None and not (np.diff(positions[a:b]) > 0).all():
                continue
            curves.append(curve(x, a, b, positions))
            locations.append((a, b))
        result[kind], spans[kind] = retain(curves, locations)
    for metric in range(3):
        t = tension[metric]
        finite = t[np.isfinite(t)]
        high = float(np.quantile(finite, 2/3)) if finite.size else np.inf
        curves, locations = [], []
        for block in blocks:
            candidates = []
            for p in range(block+8, block+56):
                a, b = p-8, p+9
                if not valid[a:b].all() or not adjacent[a+1:b].all() or not np.isfinite(t[a:b]).all():
                    continue
                if metric == 2 and not (key_index[a:b] == key_index[p]).all():
                    continue
                # Left edge of a local maximum, above the score-only upper tertile.
                if t[p] >= high and t[p] > t[p-1] and t[p] >= t[p+1] and t[p] > max(t[a], t[b-1]):
                    candidates.append(p)
            # Non-overlapping windows, chronological choice; no performance value enters.
            previous_end = -1
            for p in candidates:
                a, b = p-8, p+9
                if a < previous_end:
                    continue
                curves.append(curve(x, a, b))
                locations.append((a, b))
                previous_end = b
        name = 'H'+str(metric)
        result[name], spans[name] = retain(curves, locations)
    return result, spans, valid


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(date='2026-10-14 KST', status='definitions_frozen_before_new_rankings',
        source='Existing frozen 335 recordings and their exact 7-channel normalized residual arrays',
        split='Same alternating complete 64-original-beat blocks, both directions; no curve spans A/B or missing/structural-pair cuts',
        phrase='Full contiguous runs of the existing provisional owner labels, >=4 original beat intervals, all intervals and pairs valid, contained in one 64-beat block',
        fixed='Complete 16-original-beat windows with all intervals and pairs valid',
        tension='Per metric: score-only local maximum above whole-score upper tertile, higher than both 8-beat-away endpoints; retain original-index window [peak-8,peak+8]; strain never crosses key section; chronologically select non-overlapping windows',
        terminology='Tension-peak windows are provisional contexts, not verified buildup/resolution episodes. Momentum peaks indicate movement, not necessarily tension culmination.',
        representation='Each segment separately interpolated to 17 ordered points; phrase interpolation uses written-quarter midpoint, F/H uses original beat index; H preserves the exact central peak sample; never average different segments into one curve',
        numeric_ties='Round reconstructed tension values to 12 decimal places before peak selection and lag judge, to avoid local maxima created by float-roundoff on flat clouds. Original frozen summary H remains unchanged.',
        cap='At most 32 curves per recording/view/type, uniformly selected by chronological index without using performance values',
        variants=['raw curves','subtract same-recording search-view signed global mean'],
        distance='Gallery-only family-balanced per-channel curve-point SD; symmetric mean nearest-segment squared distance; three H metrics equally weighted; P+H equally weighted after train-other-family median distance calibration',
        cohort='Eligibility based on curve support, fixed before new rankings. Phrase comparison requires F/P in both views; joint comparison additionally all three H metrics in both views. Rerank all old baselines with the same eligible gallery/query cohort.',
        evaluation='Four existing opposite-view judges recomputed against the eligible gallery. Additional E-PC uses opposite-view 24-point per-phrase successive differences in Tempo/Dynamics. E-HL uses opposite-view peak-window delta tension vs delta Tempo/Dynamics correlations at lags -4..4.',
        controls=['same returned work/grid random-recording exact expectation','20 fixed seeds with up to three candidates per work/grid','20 fixed seeds permuting point order with a shared permutation per type/seed across all recordings; preserves per-curve values and pointwise distances but destroys adjacency in delta-based comparison'],
        limits=['No new phrase or tension truth','17-point resampling remains a representation, not lossless storage of every beat','E-PC/E-HL are numeric consistency proxies, not listener ground truth','No outcome tuning, neural training or population inference'])
    # Point permutation cannot change Euclidean curve matching if shared. Use independent
    # per-recording permutations in actual order control, specified here before rankings.
    protocol['controls'][-1] = '20 fixed seeds independently permuting point order per recording/view/type, identically across its curves; preserves curve means/ranges but destroys relative timing across recordings'
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    source = WORK/'datasets/standalone_structure_run01/standalone.npz'
    z = np.load(source, allow_pickle=False)
    expected = json.loads((WORK/'diagnostics/summary_mean_overlap/verification.json').read_text())['source_sha256']
    assert sha(source) == expected
    keys = z['keys'].tolist()
    with (REPO/'classicfy-ai/analysis/phrase_response/coverage.csv').open() as f:
        lookup = {r['key']: r for r in csv.DictReader(f)}
    metadata = [lookup[k] for k in keys]
    key_ids = {key: i for i, key in enumerate(keys)}
    primary = {}
    with (REPO/'classicfy-ai/analysis/phrase_response/phrase_hypotheses.csv').open() as f:
        for row in csv.DictReader(f):
            primary[row['piece']] = row['part_id']
    written = [np.full((z[f'{i}/mask'].size, 2), np.nan) for i in range(len(keys))]
    cuts = [np.zeros(len(w), bool) for w in written]
    map_path = WORK/'datasets/score_structure_full_run01/beat_map.csv.gz'
    map_prov = json.loads((REPO/'classicfy-ai/analysis/score_structure_full/provenance.json').read_text())
    assert sha(map_path) == map_prov['artifacts'][map_path.name]
    with gzip.open(map_path, 'rt') as f:
        for row in csv.DictReader(f):
            if row['key'] not in key_ids:
                continue
            i, j = key_ids[row['key']], int(row['index'])
            cuts[i][j] |= row['break_before'] == 'True'
            if row['part_id'] == primary[metadata[i]['piece']] and row['written_start_quarters'] and row['written_stop_quarters']:
                written[i][j] = float(row['written_start_quarters']), float(row['written_stop_quarters'])
    tonal_prov = json.loads((REPO/'classicfy-ai/analysis/tonal_tension/provenance.json').read_text())
    with (REPO/'classicfy-ai/analysis/tonal_tension/coverage.csv').open() as f:
        tonal_coverage = {(r['key'],r['metric']):r for r in csv.DictReader(f)}
    contexts, input_shas = {}, {}
    for piece in sorted(set(r['piece'] for r in metadata)):
        path = WORK/'datasets/nASAP'/Path(piece).parent/'xml_score.musicxml'
        digest = sha(path)
        assert digest == tonal_prov['inputs'][str(path)]
        contexts[piece] = TonalContext(load_musicxml(path))
        input_shas[str(path)] = digest
    arrays = {'keys': np.asarray(keys)}
    coverage, max_tension_error, threshold_error, changed_summaries = [], 0., 0., []
    for i, row in enumerate(metadata):
        x, mask, pairs, owner = [z[f'{i}/{name}'] for name in ('values','mask','pairs','owner')]
        current_cuts = cuts[i].copy()
        current_cuts[1:] |= ~mask[:-1] | ~mask[1:]
        tension, support = contexts[row['piece']].project(written[i], current_cuts)
        key_index = np.array([s['key_index'] for s in support])
        labels = [tension_labels(t, np.ones(len(mask), bool))[0] for t in tension]
        for metric, name in enumerate(('cloud_diameter','cloud_momentum','tensile_strain')):
            _, thresholds = tension_labels(tension[metric], np.ones(len(mask),bool))
            old = tonal_coverage[row['key'],name]
            error = max(abs(thresholds[0]-float(old['low_mid_threshold'])),abs(thresholds[1]-float(old['mid_high_threshold'])))
            threshold_error = max(threshold_error,error)
            assert error < 1e-10, (row['key'], name, error)
            assert int(np.isfinite(tension[metric]).sum()) == int(old['finite_score_intervals'])
        for view, direction in enumerate(('A_to_B','B_to_A')):
            rounded_tension = np.round(tension,12)
            curves, spans, valid = build(x, mask, pairs, owner, written[i], rounded_tension, key_index, view)
            reconstructed = []
            for metric in range(3):
                mp = pairs.copy()
                mp[1:] &= np.isfinite(tension[metric,:-1]) & np.isfinite(tension[metric,1:])
                if metric == 2:
                    mp[1:] &= key_index[:-1] == key_index[1:]
                m, p, _ = split_view(mask, mp, view)
                h, _ = response(x, m & np.isfinite(tension[metric]), p, labels[metric])
                reconstructed.extend(h)
            saved = z[f'{direction}_input_H'][i]
            err = float(np.max(abs(np.array(reconstructed)-saved)))
            max_tension_error = max(max_tension_error, err)
            if err > 1e-10:
                changed_summaries.append(dict(key=row['key'], view=view,max_coordinate_error=err))
            arrays[f'{i}/{view}/mean'] = x[:,valid].mean(axis=1)
            for kind in curves:
                arrays[f'{i}/{view}/{kind}'] = curves[kind]
                arrays[f'{i}/{view}/{kind}_spans'] = spans[kind]
            arrays[f'{i}/{view}/H_lag'] = lag_judge(x, rounded_tension, spans)
            # Independent evaluation resolution; differences remove level information.
            ph = []
            for a,b in spans['P']:
                u = np.mean(written[i][a:b], axis=1)
                target = np.linspace(u[0], u[-1], 24)
                values = np.array([np.interp(target, u, x[c,a:b]) for c in (0,2)])
                ph.append(np.diff(values, axis=1))
            arrays[f'{i}/{view}/P_judge'] = np.array(ph).reshape(-1,2,23)
            coverage.append(dict(key=row['key'], piece=row['piece'], family_id=row['family_id'], role=row['evaluation_role'],
                view=view, **{kind: len(curves[kind]) for kind in curves}, finite_H_lag=bool(np.isfinite(arrays[f'{i}/{view}/H_lag']).all())))
        arrays[f'{i}/tension'] = tension
        if (i+1)%50 == 0:
            print('Prepared',i+1,'/',len(keys), flush=True)
    np.savez_compressed(OUT/'curves.npz', **arrays)
    with (OUT/'coverage.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage[0])); writer.writeheader(); writer.writerows(coverage)
    by_key = {key: [r for r in coverage if r['key']==key] for key in keys}
    phrase = [i for i,key in enumerate(keys) if all(r['F']>0 and r['P']>0 for r in by_key[key])]
    joint = [i for i in phrase if all(all(r[k]>0 for k in ('H0','H1','H2')) and r['finite_H_lag'] for r in by_key[keys[i]])]
    cohort = dict(phrase=phrase,joint=joint, roles={name: {role: sum(metadata[i]['evaluation_role']==role for i in ids)
        for role in ('train','test','validation')} for name,ids in (('phrase',phrase),('joint',joint))})
    (OUT/'cohort.json').write_text(json.dumps(cohort, indent=2)+'\n')
    (OUT/'preparation_verification.json').write_text(json.dumps(dict(source_sha256=sha(source),map_sha256=sha(map_path),
        xml_sha256=input_shas,max_H_response_reconstruction_error=max_tension_error,
        max_tertile_threshold_error=threshold_error,float_tie_affected_summaries=changed_summaries,
        interpretation='Reconstructed finite coverage and raw score quantile thresholds match within 1e-10. Tiny float differences can change tertile membership on ties, hence original saved H descriptors are used without alteration; rounded raw curves are used only for new peak detection and lag judges.',
        curves_sha256=sha(OUT/'curves.npz')),indent=2)+'\n')
    print(json.dumps(cohort['roles']), 'H reconstruction error', max_tension_error,'threshold error',threshold_error,'affected views',len(changed_summaries), flush=True)


def lag_judge(x, tension, spans):
    coordinates = []
    for metric in range(3):
        locations = spans['H'+str(metric)]
        for c in (0,2):
            for lag in range(-4,5):
                aa, bb = [], []
                for start,stop in locations:
                    dt, dx = np.diff(tension[metric,start:stop]), np.diff(x[c,start:stop])
                    if lag >= 0:
                        aa.extend(dt[:len(dt)-lag]); bb.extend(dx[lag:])
                    else:
                        aa.extend(dt[-lag:]); bb.extend(dx[:len(dx)+lag])
                coordinates.append(float(np.corrcoef(aa,bb)[0,1]) if len(aa)>=8 and min(np.std(aa),np.std(bb))>1e-10 else np.nan)
    return np.array(coordinates)


if __name__ == '__main__':
    main()
