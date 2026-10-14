"""Extend frozen train-scaled opposite-view judges to heldout within-work candidates."""
import json
import sys
from pathlib import Path
import numpy as np
from data import previous, sha, PREVIOUS, WORK

OUT = Path(__file__).resolve().parent
CURVES = WORK / 'diagnostics/curve_comparison'
sys.path.insert(0, str(CURVES))
from compare import set_distances, channel_scale, calibrated, weights
from embedding.retrieval_probe import fit_scale, distance, percentile_scores
AXES = ('E-L', 'E-T', 'E-R', 'E-S', 'E-PC', 'E-PN', 'E-HL')


def main():
    d = previous.ViewData(0); ids = d.ids; gids = d.gids
    metadata = d.metadata; gw = weights([metadata[i] for i in gids]); families = np.array([metadata[i]['family_id'] for i in gids])
    columns = np.array([int(np.flatnonzero(ids == i)[0]) for i in gids])
    with np.load(CURVES / 'comparison.npz') as f: reference = {k: f[k] for k in f.files}
    source = np.load(WORK / 'datasets/standalone_structure_run01/standalone.npz')
    curves = np.load(CURVES / 'curves.npz'); arrays = dict(ids=ids, gallery_columns=columns)
    scales = {}; checks = []; baselines = {}
    for view, direction in enumerate(('A_to_B', 'B_to_A')):
        scale_records = {}
        raw = {}
        for axis in AXES[:4]:
            a = source[f'{direction}_judge_{axis}']; scale = fit_scale(a[gids], gw)
            width = 2 if axis == 'E-L' else 3 if axis == 'E-T' else None
            raw[axis] = distance(a, a[ids], scale, group_width=width)
            scale_records[axis] = scale
        for axis in ('E-PC', 'E-PN'):
            data = {i: curves[f'{i}/{1 - view}/P_judge'].copy() for i in ids}
            if axis == 'E-PN':
                for i, x in data.items():
                    rms = np.sqrt(np.mean(x * x, axis=2)); good = (rms > 1e-6).all(axis=1)
                    assert good.any(); data[i] = x[good] / rms[good, :, None]
            scale = channel_scale(data, gids, gw)
            raw[axis] = set_distances(data, ids, ids, scale)
            scale_records[axis] = dict(sd=scale[0].tolist(), active=scale[1].tolist())
        a = np.array([curves[f'{i}/{1 - view}/H_lag'] for i in range(335)])
        scale = fit_scale(a[gids], gw); scale_records['E-HL'] = scale
        raw['E-HL'] = np.full((335, len(ids)), np.inf)
        raw['E-HL'][ids] = distance(a[ids], a[ids], scale)
        for axis, values in raw.items():
            recomputed = percentile_scores(values[ids][:, columns], gw)
            error = float(np.max(abs(recomputed - reference[f'{direction}/evaluation/{axis}'][ids])))
            assert error < 1e-10, (direction, axis, error)
            checks.append(dict(direction=direction, axis=axis, max_frozen_score_error=error))
            arrays[f'{direction}/{axis}'] = values
        scales[direction] = scale_records
        for name in ('B', 'F', 'P', 'H'):
            x = source[f'{direction}_input_{name}']; scale = None if name == 'B' else fit_scale(x[gids], gw)
            raw_d = distance(x, x[ids], scale, group_width=2 if name == 'B' else None)
            _, median = calibrated(raw_d[:, columns], gids, gw, families)
            scaled = raw_d / median
            assert np.allclose(scaled[ids][:, columns], reference[f'{direction}/distance/{name}_summary'][ids], atol=1e-10, rtol=0)
            baselines[f'{direction}/{name}_summary'] = scaled
        p, h = baselines[f'{direction}/P_summary'], baselines[f'{direction}/H_summary']
        baselines[f'{direction}/P+H_summary'] = np.sqrt((p * p + h * h) / 2)
    np.savez_compressed(OUT / 'within_judges.npz', **arrays)
    np.savez_compressed(OUT / 'baseline_distances.npz', **baselines)
    (OUT / 'judge_scales.json').write_text(json.dumps(scales, indent=2) + '\n')
    (OUT / 'judge_verification.json').write_text(json.dumps(dict(status='passed', checks=checks,
        source_comparison_sha256=sha(CURVES / 'comparison.npz'), source_curves_sha256=sha(CURVES / 'curves.npz'),
        within_judges_sha256=sha(OUT / 'within_judges.npz'), baseline_distances_sha256=sha(OUT / 'baseline_distances.npz'),
        interpretation='Heldout candidate extension uses frozen train-only scales; all14 opposite-view percentile arrays match prior gallery results.'), indent=2) + '\n')
    print('Frozen opposite-view gallery judge checks passed:', len(checks), flush=True)


if __name__ == '__main__': main()
