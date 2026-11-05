"""Render archived epoch-8 figures from saved vectors, without raw datasets.

The original inference generators are archived verbatim. This entry point calls
only their render functions and supplies the committed numerical artifacts.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI / 'src'))


def load_generator(directory, name):
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(f'archived_{name}', directory / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=AI / 'analysis/note_ae_checkpoint_cases')
    parser.add_argument('--out', type=Path, required=True, help='Destination for the four rendered figure directories')
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    manifest = json.loads((data / 'archive_manifest.json').read_text())
    for row in manifest:
        assert hashlib.sha256((data / row['path']).read_bytes()).hexdigest() == row['sha256'], row['path']
    generators = data / 'generators'
    same = load_generator(generators, 'create_checkpoint_same_work_figure')
    checks = []
    for name in ('same_work_checkpoint', 'same_work_checkpoint_unique_artists'):
        source = data / name
        dest = out / name; dest.mkdir(parents=True, exist_ok=True)
        rows = csv_rows(source / 'performances.csv')
        metadata = {}
        for r in rows:
            r['record_key'] = r['performance_id'].split(':')[1]
            metadata[r['record_key']] = {'artist': r['artist']}
        provenance = json.loads((source / 'provenance.json').read_text())
        with np.load(source / 'example.npz', allow_pickle=False) as f:
            de, xy, variance, agrees = same.render(
                dest, rows, f['feature_profiles'],
                np.array([float(r['feature_distance_to_A']) for r in rows]),
                f['embedding'], [int(r['windows']) for r in rows],
                provenance['checkpoint_epoch'], metadata, name.endswith('unique_artists'))
            np.testing.assert_allclose(de, f['cosine_distance_to_A'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(xy, f['xy'], atol=1e-12, rtol=0)
        assert agrees == provenance['distance_order_agrees']
        checks.append(name)
    cross = load_generator(generators, 'create_cross_work_checkpoint_figure')
    source = data / 'cross_work_checkpoint'
    cross.OUT = out / source.name; cross.OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads((source / 'retrieval.json').read_text())
    with np.load(source / 'selected_profiles.npz', allow_pickle=False) as f:
        cross.render(result, f['feature_profile'])
    checks.append(source.name)
    comparison = load_generator(generators, 'create_ae_metadata_comparison')
    source = data / 'ae_metadata_comparison'
    comparison.OUT = out / source.name; comparison.OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads((source / 'provenance.json').read_text())
    with np.load(source / 'projection.npz', allow_pickle=False) as f:
        index = {key: i for i, key in enumerate(f['performance_id'])}
        cohort = [index[key] for key in f['selected_id']]
        background = np.array([index[r['performance_id']] for r in
                               csv_rows(data / 'cross_work_checkpoint/all_ranked_candidates.csv')[1:]])
        comparison.render(result['selected'], f['xy'], f['PCA_variance'], cohort, background)
    checks.append(source.name)
    print(json.dumps({'archive_hashes_verified': len(manifest), 'rendered': checks,
                      'raw_datasets_required': False, 'inference_or_training_run': False}, indent=2))


if __name__ == '__main__':
    main()
