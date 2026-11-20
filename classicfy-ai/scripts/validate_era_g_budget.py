"""Compare relaxed G gradient budgets on validation at the existing fixed horizons."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from train_era_note_ae import (
    ERA_COMPOSERS, G_PROFILE_VERSION, Runtime, SelectedNoteCorpus, dump,
    file_hash, fit, tables_for,
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(row, baseline, budget, batch_size, retained_count):
    metrics = row['validation']
    style = row['style']
    steps = math.ceil(retained_count / batch_size)
    return {
        'budget': budget,
        'epoch': row['epoch'],
        'validation': metrics,
        'guard': row['guard'],
        'reconstruction_change_pct': 100 * (
            metrics['reconstruction']['loss'] / baseline['reconstruction']['loss'] - 1),
        'effective_lambda': style['applied_weight'],
        'mean_z_gradient_ratio': style['applied_gradient_ratio'],
        'steps': steps,
        'steps_below_requested_lambda': steps - style['ceiling_active_steps'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--common-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--era', choices=ERA_COMPOSERS, required=True)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    source = args.source_run
    era = args.era
    out = args.out / era
    if out.resolve().is_relative_to(source.resolve()):
        raise ValueError('Sweep output must be outside the existing run')
    out.mkdir(parents=True, exist_ok=True)
    contract = read(source / 'profile_contract.json')
    if contract['version'] != G_PROFILE_VERSION or read(source / 'verification.json')['status'] != 'passed':
        raise ValueError('Verified v2 profile source required')
    prior = read(source / 'ablation/protocol.json')
    control_path = source / 'ablation' / era / 'g_only/result.json'
    control = read(control_path)
    checkpoint = args.common_run / 'evaluated_best.pt'
    initialization = file_hash(checkpoint)
    if initialization != prior['common_checkpoint_sha256']:
        raise ValueError('Initial checkpoint does not match the baseline')
    horizon = prior['eras'][era]['epochs']
    batch_size = prior['batch_clips']
    settings = {**prior['eras'][era]['config'], 'objective_mode': 'g_only'}
    if control['config'] != settings or len(control['history']) != horizon:
        raise ValueError('Existing 25% run is incompatible with the sweep')
    protocol = {
        'version': 'era-g-budget-validation-v1', 'era': era, 'horizon': horizon,
        'batch_clips': batch_size, 'budgets': [.25, .5, 1.],
        'base_config': settings, 'initialization_sha256': initialization,
        'metadata_sha256': file_hash(source / 'metadata.json'),
        'profiles_sha256': file_hash(source / 'profiles.npz'),
        'control_sha256': file_hash(control_path),
        'selection': 'Fixed final validation epoch. Minimize G score among guard-passing budgets above 25%.',
    }
    if (out / 'protocol.json').exists() and read(out / 'protocol.json') != protocol:
        raise ValueError('Sweep protocol changed')
    dump(out / 'protocol.json', protocol)
    metadata = read(source / 'metadata.json')
    corpus = SelectedNoteCorpus(args.common_run / 'corpus')
    if corpus.manifest['id'] != contract['corpus_id']:
        raise ValueError('Corpus does not match the verified profile source')
    with np.load(source / 'profiles.npz', allow_pickle=False) as data:
        profiles = [data[k] for k in ('raw', 'known', 'quality')]
    ids, _, tables, calibration = tables_for(metadata, *profiles, era)
    runtime = Runtime(corpus, checkpoint, source / 'evaluation_cache')
    scale = torch.tensor(np.maximum(runtime.teacher_clip[ids['train']].std(0), .05))
    trials = {'0.25': summarize(control['history'][-1], control['baseline'], .25, batch_size, len(ids['train']))}
    for budget in (.5, 1.):
        config = {**settings, 'gradient_budget': budget}
        trial = fit(runtime, era, ids['train'], ids['validation'], tables, calibration, scale,
                    config, out / f'budget_{budget:g}', batch_size, horizon)
        np.testing.assert_allclose(trial['baseline']['geometry']['score'],
                                   control['baseline']['geometry']['score'], rtol=1e-5)
        trials[str(budget)] = summarize(trial['history'][-1], trial['baseline'],
                                        budget, batch_size, len(ids['train']))
        dump(out / 'trials.json', trials)
    stronger = [row for row in trials.values() if row['budget'] > .25 and row['guard']['passed']]
    if not stronger:
        raise ValueError(f'{era}: no relaxed budget passed preservation')
    winner = min(stronger, key=lambda row: (row['validation']['geometry']['score'], row['budget']))
    dump(out / 'selection.json', {
        'status': 'complete', 'protocol': protocol, 'trials': trials,
        'selected_budget': winner['budget'],
        'improves_validation_G_score_vs_25': (
            winner['validation']['geometry']['score'] < trials['0.25']['validation']['geometry']['score']),
    })
    print('G BUDGET SELECTED', era, winner['budget'], flush=True)


if __name__ == '__main__':
    main()
