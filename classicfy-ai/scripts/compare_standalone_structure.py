"""Standalone P/H ablation on the exact previously frozen cohort and judges.

Reuses verified response CSVs and the full original-index feature/bridge join.
No new phrase boundaries, tension labels, models, cohorts or outcome tuning.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'classicfy-ai/src'))
from embedding.standalone_structure import ARMS, standalone_distances
from embedding.structure_retrieval import heldout_judges
from embedding.retrieval_probe import (summary, weights, fit_scale, distance,
    percentile_scores, ranked, priorities, overlap)
from features.structure_response import (PhraseHypothesis, phase_labels, response,
    split_view, fixed_labels)
from preprocessing import ASAPLoader
from compare_structure_responses import macro
from calculate_phrase_response import dump
from validate_score_beat_bridge import rows, sha, write


def checked(path, digest):
    if sha(path) != digest:
        raise ValueError('Frozen input differs: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent/'datasets')
    parser.add_argument('--join', type=Path, default=ROOT.parent/'datasets/score_structure_full_run01')
    parser.add_argument('--reference', type=Path, default=ROOT/'classicfy-ai/analysis/structure_response_comparison')
    parser.add_argument('--out', type=Path, default=ROOT/'classicfy-ai/analysis/standalone_structure')
    parser.add_argument('--artifacts', type=Path, default=ROOT.parent/'datasets/standalone_structure_run01')
    args = parser.parse_args()
    for folder in (args.out, args.artifacts):
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Choose empty output folders: ' + str(folder))
        folder.mkdir(parents=True, exist_ok=True)
    phrase_report = ROOT/'classicfy-ai/analysis/phrase_response'
    tonal_report = ROOT/'classicfy-ai/analysis/tonal_tension'
    full_report = ROOT/'classicfy-ai/analysis/score_structure_full'
    full_prov = json.loads((full_report/'provenance.json').read_text())
    phrase_prov = json.loads((phrase_report/'provenance.json').read_text())
    tonal_prov = json.loads((tonal_report/'provenance.json').read_text())
    ref_prov = json.loads((args.reference/'provenance.json').read_text())
    for name in ('feature_join.npz', 'beat_map.csv.gz'):
        checked(args.join/name, full_prov['artifacts'][name])
    for name in ('responses.csv', 'coverage.csv', 'phrase_hypotheses.csv'):
        checked(phrase_report/name, phrase_prov['results'][name])
    checked(tonal_report/'responses.csv.gz', tonal_prov['results']['responses.csv.gz'])
    for name in ('cohort.json', 'scores.csv', 'rankings.csv', 'controls.csv', 'scales.json'):
        checked(args.reference/name, ref_prov['results'][name])
    annotation = args.data/'ASAP/asap_annotations.json'
    expected = next(d for p, d in phrase_prov['inputs'].items() if p.endswith('/ASAP/asap_annotations.json'))
    checked(annotation, expected)
    cohort = json.loads((args.reference/'cohort.json').read_text())
    coverage = {r['key']: r for r in rows(phrase_report/'coverage.csv')}
    metadata = [coverage[k] for k in cohort['keys']]
    gids = np.array([i for i, r in enumerate(metadata) if r['evaluation_role'] == 'train'])
    qids = np.array([i for i, r in enumerate(metadata) if r['evaluation_role'] in ('test', 'validation')])
    if len(metadata) != 335 or len(gids) != 247 or len(qids) != 88:
        raise ValueError('Frozen 335/247/88 cohort changed')
    # Freeze the ablation and unchanged judges before inspecting new rankings.
    protocol = dict(date='2026-10-14', status='frozen_standalone_ablation', K=5,
        reference_cohort_sha256=sha(args.reference/'cohort.json'),
        arms={**ARMS, 'M-CT': ('composer', 'work_type')},
        dimensions=dict(B=14, T=15, F=66, P=66, H=198),
        standalone='Remove the extra global B block; P/H still contain context-conditioned means',
        training='Reuse family-balanced train-only SD and other-family median calibration; equal squared block distances',
        views='Same alternating full 64-original-beat A/B blocks, both directions; original indices and pair cuts retained',
        judges='Unchanged E-L/E-T/E-R/E-S, opposite view only; E-S covers provisional phrase phase Tempo/Dynamics signs, not tension episodes',
        support='Exact prior 335 cohort, including its strict 20-shift support selection; no new exclusion or threshold relaxation',
        controls='Same-work recording exact expectation; 20 balanced-gallery seeds with three recordings per work/grid; metadata ties 100 seeds',
        reconstruction='Verified P/H CSVs plus original-index join; reproduce all six prior B-based arms and M-CT scores/ranks before interpreting new arms',
        limits=['Three test and three validation families; exploratory, no population CI',
                'No new general 57D/119D or CNN/BiLSTM comparison',
                'No tension-conditioned judge or whole-episode holdout added here',
                'Provisional phrase boundaries and signature-based tension unchanged',
                'No standalone structure-shift control; previous controls covered B+P/B+H/B+P+H',
                'Current catalog normalization and disjoint-beat consistency limits remain'])
    dump(args.out/'protocol.json', protocol)
    dump(args.out/'cohort.json', cohort)
    hypotheses = defaultdict(list)
    for r in rows(phrase_report/'phrase_hypotheses.csv'):
        hypotheses[r['piece']].append(PhraseHypothesis(r['phrase_id'], r['part_id'],
            Fraction(r['start']), Fraction(r['stop']), (), ()))
    primary_part = {piece: ps[0].part_id for piece, ps in hypotheses.items()}
    z = np.load(args.join/'feature_join.npz', allow_pickle=False)
    source_ids = {key: i for i, key in enumerate(z['keys'].tolist())}
    written = {r['key']: [None]*len(z[f'{source_ids[r["key"]]}/combined_mask']) for r in metadata}
    with gzip.open(args.join/'beat_map.csv.gz', 'rt', encoding='utf-8-sig') as stream:
        import csv
        for r in csv.DictReader(stream):
            if r['key'] in written and r['part_id'] == primary_part[coverage[r['key']]['piece']]:
                if r['written_start_quarters'] and r['written_stop_quarters']:
                    written[r['key']][int(r['index'])] = (float(r['written_start_quarters']), float(r['written_stop_quarters']))
    pp = {(r['key'], r['view']): np.array([float(r[f'p{j}']) for j in range(66)])
          for r in rows(phrase_report/'responses.csv')}
    with gzip.open(tonal_report/'responses.csv.gz', 'rt', encoding='utf-8-sig') as stream:
        hh = {(r['key'], r['view'], r['metric']): np.array([float(r[f'h{j}']) for j in range(66)])
              for r in csv.DictReader(stream)}
    metrics = ('cloud_diameter', 'cloud_momentum', 'tensile_strain')
    D = {v: defaultdict(list) for v in (0, 1)}
    J = {v: defaultdict(list) for v in (0, 1)}
    arrays = dict(keys=np.array(cohort['keys']), gallery_indices=gids, query_indices=qids)
    loader = ASAPLoader(args.data/'ASAP')
    response_error = 0.
    for i, r in enumerate(metadata):
        source = source_ids[r['key']]
        x = z[f'{source}/values']; mask = z[f'{source}/combined_mask']
        fm = z[f'{source}/feature_mask']; pair = z[f'{source}/adjacent_pair_mask'].copy()
        if sha_bytes(x, fm) != r['feature_sha256']:
            raise ValueError('Prior feature digest differs: ' + r['key'])
        irregular = np.array([b == 'bR' for b in loader.get_sample(r['key']).score_beat_types])
        for j in range(1, len(pair)):
            if irregular[j-1:j+2].any():
                pair[j] = False
        phase, owner = phase_labels(written[r['key']], hypotheses[r['piece']])
        for name, value in [('values', x), ('mask', mask), ('pairs', pair), ('phase', phase), ('owner', owner)]:
            arrays[f'{i}/{name}'] = value
        for view in (0, 1):
            valid, pairs, _ = split_view(mask, pair, view)
            fixed, fixed_owner = fixed_labels(len(mask))
            p, _ = response(x, valid, pairs, phase, owner)
            old_p = pp[r['key'], 'AB'[view]]
            response_error = max(response_error, float(np.max(abs(p-old_p))))
            np.testing.assert_allclose(p, old_p, atol=1e-12, rtol=1e-12)
            f, _ = response(x, valid, pairs, fixed, fixed_owner)
            delta = np.diff(x)[:, pairs[1:]]
            t = np.r_[delta.mean(axis=1), np.sqrt((delta*delta).mean(axis=1)), np.mean(delta[0]*delta[2])]
            h = np.concatenate([hh[r['key'], 'AB'[view], m] for m in metrics])
            for name, value in dict(B=summary(x, valid), T=t, F=f, P=old_p, H=h).items():
                D[view][name].append(value)
            for name, value in heldout_judges(x, mask, pair, phase, owner, view).items():
                J[view][name].append(value)
    D = {v: {n: np.array(a) for n, a in d.items()} for v, d in D.items()}
    J = {v: {n: np.array(a) for n, a in d.items()} for v, d in J.items()}
    for group in (*D.values(), *J.values()):
        if not all(np.isfinite(a).all() for a in group.values()):
            raise ValueError('Reconstructed frozen cohort lost support')
    gallery = [metadata[i] for i in gids]; gw = weights(gallery)
    families = np.array([r['family_id'] for r in gallery])
    works = defaultdict(list)
    for j, r in enumerate(gallery):
        works[r['piece'], r['grid']].append(j)
    old_rank = defaultdict(list)
    for r in rows(args.reference/'rankings.csv'):
        old_rank[r['direction'], r['query'], r['method']].append(r['candidate'])
    old_score = {(r['direction'], r['key'], r['method'], r['axis']): float(r['score'])
                 for r in rows(args.reference/'scores.csv')}
    scores, rankings, controls, scale_records = [], [], [], {}
    old_error = 0.; reproduced_ranks = 0
    for view in (0, 1):
        direction = ('A_to_B', 'B_to_A')[view]
        dd, scales = standalone_distances(D[view], metadata, gids)
        dd['M-CT'] = np.array([[.5*(r['composer'] != g['composer'])+.5*(r['work_type'] != g['work_type'])
                               for g in gallery] for r in metadata])
        evaluation = {}; eval_scales = {}
        for axis, a in J[1-view].items():
            scale = fit_scale(a[gids], gw); eval_scales[axis] = scale
            ed = distance(a, a[gids], scale, group_width=2 if axis == 'E-L' else 3 if axis == 'E-T' else None)
            evaluation[axis] = percentile_scores(ed, gw)
            arrays[direction+'_judge_'+axis] = a
            arrays[direction+'_evaluation_'+axis] = evaluation[axis]
        scales['evaluation_scales'] = eval_scales; scale_records[direction] = scales
        for name, a in D[view].items():
            arrays[direction+'_input_'+name] = a
        for name, a in dd.items():
            arrays[direction+'_distance_'+name] = a
        for qi in qids:
            base = ranked(dd['B'][qi], families, priorities(20261014, metadata[qi]['key'], len(gallery)), 5)
            for name, distances in dd.items():
                selected = ranked(distances[qi], families, priorities(20261014, metadata[qi]['key'], len(gallery)), 5)
                selected_keys = [gallery[j]['key'] for j in selected]
                if name.startswith('B') or name == 'M-CT':
                    if selected_keys != old_rank[direction, metadata[qi]['key'], name]:
                        raise ValueError('Previous ranking differs: ' + name)
                    reproduced_ranks += 1
                choices = [selected] if name != 'M-CT' else [ranked(distances[qi], families,
                    priorities(seed, metadata[qi]['key'], len(gallery)), 5) for seed in range(20261014, 20261114)]
                for rank, j in enumerate(selected, 1):
                    rankings.append(dict(direction=direction, query=metadata[qi]['key'], method=name, rank=rank,
                        candidate=gallery[j]['key'], candidate_family=gallery[j]['family_id'], distance=float(distances[qi, j])))
                for axis, e in evaluation.items():
                    value = float(np.mean([e[qi, c].mean() for c in choices]))
                    sw = float(np.mean([np.mean([e[qi, works[gallery[j]['piece'], gallery[j]['grid']]].mean()
                                                 for j in c]) for c in choices]))
                    if name.startswith('B') or name == 'M-CT':
                        error = abs(value-old_score[direction, metadata[qi]['key'], name, axis])
                        old_error = max(old_error, error)
                        if error > 1e-10:
                            raise ValueError('Previous score differs: ' + name + '/' + axis)
                    scores.append(dict(direction=direction, key=metadata[qi]['key'], piece=metadata[qi]['piece'],
                        family_id=metadata[qi]['family_id'], role=metadata[qi]['evaluation_role'], method=name, axis=axis,
                        score=value, same_work_expected=sw, overlap_with_B=overlap(selected, base, families)['recording_overlap']))
        for seed in range(20261014, 20261034):
            rng = np.random.default_rng(seed)
            balanced = np.array(sorted(j for group in sorted(works) for j in rng.choice(works[group], 3, replace=False)))
            arrays[f'{direction}_balanced_{seed}'] = balanced
            for name, distances in dd.items():
                selected = {}
                for qi in qids:
                    d = np.full(len(gallery), np.inf); d[balanced] = distances[qi, balanced]
                    selected[qi] = ranked(d, families, priorities(seed, metadata[qi]['key'], len(gallery)), 5)
                for axis, e in evaluation.items():
                    for role in ('test', 'validation'):
                        rs = [dict(**metadata[qi], score=float(e[qi, selected[qi]].mean()))
                              for qi in qids if metadata[qi]['evaluation_role'] == role]
                        value, _ = macro(rs)
                        controls.append(dict(direction=direction, control='balanced_gallery_3', seed=seed,
                            method=name, axis=axis, role=role, score=value))
    summary_rows, family_rows = [], []
    for role in ('test', 'validation'):
        for method in (*ARMS, 'M-CT'):
            for axis in ('E-L', 'E-T', 'E-R', 'E-S'):
                rs = [r for r in scores if (r['role'], r['method'], r['axis']) == (role, method, axis)]
                value, fs = macro(rs); sw, _ = macro(rs, 'same_work_expected')
                summary_rows.append(dict(role=role, method=method, axis=axis, score=value,
                    same_work_expected=sw, query_families=len(fs)))
                family_rows.extend(dict(role=role, method=method, axis=axis, family_id=f, score=v) for f, v in fs.items())
    for name, value in [('scores', scores), ('rankings', rankings), ('controls', controls),
                        ('summary', summary_rows), ('per_family', family_rows)]:
        write(args.out/(name+'.csv'), value)
    dump(args.out/'scales.json', scale_records)
    dump(args.out/'reconstruction.json', dict(prior_rankings_reproduced=reproduced_ranks,
        prior_query_score_max_error=old_error, phrase_response_max_error=response_error,
        original_features='Every included original values/feature mask digest matches prior phrase report',
        limits='Verified H CSVs reused; this run does not revalidate musical phrase or key truth'))
    np.savez_compressed(args.artifacts/'standalone.npz', **arrays)
    inputs = [args.join/'feature_join.npz', args.join/'beat_map.csv.gz', annotation,
        phrase_report/'responses.csv', phrase_report/'coverage.csv', phrase_report/'phrase_hypotheses.csv',
        tonal_report/'responses.csv.gz', args.reference/'cohort.json', args.reference/'scores.csv', args.reference/'rankings.csv',
        args.reference/'controls.csv', args.reference/'scales.json', full_report/'provenance.json',
        phrase_report/'provenance.json', tonal_report/'provenance.json', args.reference/'provenance.json']
    code = [Path(__file__), ROOT/'classicfy-ai/src/embedding/standalone_structure.py',
        ROOT/'classicfy-ai/src/embedding/structure_retrieval.py', ROOT/'classicfy-ai/src/embedding/retrieval_probe.py',
        ROOT/'classicfy-ai/src/features/structure_response.py', Path(__file__).with_name('compare_structure_responses.py')]
    dump(args.out/'provenance.json', dict(inputs={str(p): sha(p) for p in inputs},
        code={str(p.relative_to(ROOT)): sha(p) for p in code},
        results={p.name: sha(p) for p in args.out.iterdir() if p.is_file()},
        artifacts={'standalone.npz': sha(args.artifacts/'standalone.npz')}))
    print(json.dumps(dict(included=len(metadata), gallery=len(gids), queries=len(qids),
        prior_rankings_reproduced=reproduced_ranks, prior_score_max_error=old_error)), flush=True)


def sha_bytes(x, mask):
    import hashlib
    return hashlib.sha256(x.tobytes()+mask.tobytes()).hexdigest()


if __name__ == '__main__':
    main()
