"""Evaluate the frozen 25% and validation-selected G caps on the test split."""
import argparse
from pathlib import Path

import numpy as np
import torch

from analyze_era_loss_ablation import diagnose, paired_work_interval
from train_era_note_ae import (
    ERA_COMPOSERS, G_PROFILE_VERSION, Runtime, SelectedNoteCorpus, dump,
    file_hash, fingerprint, restore, tables_for,
)
from validate_era_g_budget import read


def verify_split(metadata, corpus):
    if len(metadata) != len(corpus.manifest['clips']):
        raise ValueError('Corpus and G metadata have different clip counts')
    splits = {}
    for item, clip in zip(metadata, corpus.manifest['clips'], strict=True):
        if item['perf_id'] != clip['perf_id'] or item['split'] != clip['split']:
            raise ValueError('G metadata differs from the corpus split')
        for key in ('work', 'source'):
            token = (key, item[key])
            if splits.setdefault(token, item['split']) != item['split']:
                raise ValueError(f'{key} overlaps between splits')


def evaluate_era(era, runtime, source, selections, metadata, profiles, hashes):
    decision_path = selections / era / 'selection.json'
    decision = read(decision_path)
    protocol = decision['protocol']
    budget = decision['selected_budget']
    selected = decision['trials'][str(budget)]
    if (decision['status'] != 'complete' or protocol['era'] != era
            or not selected['guard']['passed'] or budget not in (.5, 1.)
            or protocol['initialization_sha256'] != runtime.base_hash
            or any(protocol[key] != value for key, value in hashes.items())):
        raise ValueError(f'Unverified validation selection for {era}')
    control = source / 'ablation' / era / 'g_only'
    if file_hash(control / 'result.json') != protocol['control_sha256']:
        raise ValueError(f'Changed 25% validation control for {era}')
    paths = {'control': control / 'last.pt',
             'selected': selections / era / f'budget_{budget:g}' / 'last.pt'}
    ids, scaler, tables, calibration = tables_for(metadata, *profiles, era)
    if read(source / era / 'g_train_scaler.json') != dict(scaler=scaler, calibration=calibration):
        raise ValueError(f'Train-only G calibration changed for {era}')
    expected_scale = torch.tensor(np.maximum(runtime.teacher_clip[ids['train']].std(0), .05))
    results, diagnostics, checkpoint_hashes = {}, {}, {}
    for name, path in paths.items():
        model, head, scale, state = restore(path, runtime)
        cap = .25 if name == 'control' else budget
        expected_config = {**protocol['base_config'], 'gradient_budget': cap}
        if (state['epoch'] != protocol['horizon'] or state['config'] != expected_config
                or state['era'] != era or state['corpus_id'] != runtime.corpus.manifest['id']
                or state['history'][-1]['validation'] != decision['trials'][str(cap)]['validation']):
            raise ValueError(f'Checkpoint differs from fixed validation horizon: {era}/{name}')
        torch.testing.assert_close(scale, expected_scale)
        results[name], diagnostics[name] = diagnose(
            model, head, runtime, ids['test'], scale, tables['test'], calibration,
            protocol['batch_clips'],
        )
        checkpoint_hashes[name] = file_hash(path)

    prior = read(control / 'test_evaluation.json')
    if prior['signature']['checkpoint_sha256'] != checkpoint_hashes['control']:
        raise ValueError(f'Changed test control checkpoint for {era}')
    before, after = results['control'], results['selected']
    if (before['geometry']['pairs'] != after['geometry']['pairs']
            or not np.array_equal(diagnostics['control']['triplets'], diagnostics['selected']['triplets'])):
        raise ValueError(f'Test comparisons differ across models for {era}')
    np.testing.assert_allclose(
        [before['geometry']['pair_mse'], before['geometry']['ranking_accuracy'], before['reconstruction']['loss']],
        [prior['metrics']['geometry']['pair_mse'], prior['metrics']['geometry']['ranking_accuracy'],
         prior['metrics']['reconstruction']['loss']],
        rtol=1e-5, atol=1e-8,
    )

    works = np.array([metadata[i]['work'] for i in ids['test']])
    intervals = {}
    for kind, key in [('distance_error', 'distance_error_per_anchor'),
                      ('accuracy', 'ranking_per_anchor'),
                      ('reconstruction', 'reconstruction_per_clip')]:
        interval = paired_work_interval(diagnostics['selected'][key], diagnostics['control'][key], works, kind=kind)
        intervals[kind] = {k: v for k, v in interval.items() if k not in ('works', 'per_work_left', 'per_work_right')}
    return dict(selected_budget=budget, epoch=protocol['horizon'], seed=protocol['base_config']['seed'],
                clips=len(works), works=len(np.unique(works)), checkpoint_sha256=checkpoint_hashes,
                validation_selection_sha256=file_hash(decision_path), metrics=results,
                paired_work_bootstrap=intervals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--common-run', type=Path, required=True)
    parser.add_argument('--budget-selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    source = args.source_run
    metadata_path, profiles_path = source / 'metadata.json', source / 'profiles.npz'
    metadata = read(metadata_path)
    corpus = SelectedNoteCorpus(args.common_run / 'corpus')
    contract = read(source / 'profile_contract.json')
    if (contract['version'] != G_PROFILE_VERSION or contract['corpus_id'] != corpus.manifest['id']
            or contract['metadata'] != fingerprint(metadata)
            or read(source / 'verification.json')['status'] != 'passed'):
        raise ValueError('Verified v2 corpus and profiles required')
    verify_split(metadata, corpus)
    with np.load(profiles_path, allow_pickle=False) as data:
        profiles = [data[key] for key in ('raw', 'known', 'quality')]
    hashes = dict(metadata_sha256=file_hash(metadata_path), profiles_sha256=file_hash(profiles_path))
    runtime = Runtime(corpus, args.common_run / 'evaluated_best.pt', source / 'evaluation_cache')
    results = {}
    for era in ERA_COMPOSERS:
        results[era] = evaluate_era(era, runtime, source, args.budget_selection, metadata, profiles, hashes)
        print(f'Test complete: {era}', flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump(args.out, dict(
        status='passed', split='test', corpus_id=corpus.manifest['id'], common_checkpoint_sha256=runtime.base_hash,
        **hashes, selection='Validation-selected cap at fixed last epoch; no test retuning',
        limitation='Held-out works; the same test split was inspected in earlier analyses; single training seed',
        eras=results,
    ))


if __name__ == '__main__':
    main()
