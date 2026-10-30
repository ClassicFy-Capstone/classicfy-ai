"""Evaluate fixed channel means from each full performance, without fitting an AE."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from embedding.note_ae_indexed_data import IndexedNoteAECorpus
from embedding.note_ae_loss import reconstruction_loss
from embedding.note_ae_training import _loader, _RecMetrics
from features.note_performance import R_CHANNELS


def performance_channel_mean(record):
    """Mean in the existing scaled R coordinates; tempo counts valid beats once."""
    means, counts = torch.zeros(11), torch.zeros(11, dtype=torch.long)
    for k in range(11):
        values = (record['beat_scaled'][record['beat_valid']] if k == 0
                  else record['R'][:, k][record['M'][:, k]])
        counts[k] = len(values)
        if len(values):
            means[k] = values.double().mean().float()
    return means, counts


@torch.no_grad()
def evaluate(corpus, split, batch_size=128):
    means = {}
    meters = {name: _RecMetrics() for name in ('train_mean', 'performance_mean')}
    sums = {name: np.zeros(11) for name in meters}
    supports = {name: np.zeros(11, dtype=np.int64) for name in meters}
    seen = set()
    loader = _loader(corpus.dataset(split), batch_size, corpus.collate, bucket_by_length=True)
    for step, batch in enumerate(loader, 1):
        average = []
        for window in batch['window_metadata']:
            key = window['performance_id'].split(':', 1)[1]
            if key not in means:
                means[key] = performance_channel_mean(corpus._record(key))
            average.append(means[key][0])
            if window['window_id'] in seen:
                raise ValueError('Repeated window during baseline evaluation')
            seen.add(window['window_id'])
        prediction = torch.stack(average)[:, None].expand_as(batch['R'])
        for name, value in [('train_mean', torch.zeros_like(prediction)), ('performance_mean', prediction)]:
            loss = reconstruction_loss(value, batch['R'], batch['M'], batch['P'], batch['beat_id'])
            meters[name].add(loss)
            valid = batch['M'] & batch['P'][..., None]
            errors = F.huber_loss(torch.where(valid, value, 0.), torch.where(valid, batch['R'], 0.),
                                  reduction='none', delta=1.)
            counts = valid.sum(1)
            channel_losses = errors.sum(1) / counts.clamp_min(1)
            # Channel zero must have the same once-per-beat aggregation as AE loss.
            channel_losses[:, 0] = loss['group_losses'][:, 0]
            counts[:, 0] = loss['channel_counts'][:, 0]
            present = counts > 0
            sums[name] += torch.where(present, channel_losses, 0.).double().sum(0).numpy()
            supports[name] += present.sum(0).numpy()
        if step % 100 == 0 or step == len(loader):
            print(json.dumps({'split': split, 'batch': step, 'batches': len(loader)}), flush=True)
    if len(seen) != len(corpus.dataset(split)):
        raise ValueError('Incomplete split coverage')
    result = {'windows': len(seen), 'recordings': len(means)}
    for name, meter in meters.items():
        result[name] = {**meter.result(), 'channel_losses': {
            k: {'loss': float(sums[name][i] / supports[name][i]) if supports[name][i] else None,
                'valid_windows': int(supports[name][i])} for i, k in enumerate(R_CHANNELS)}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'],
                        default=['train', 'validation', 'test'])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    corpus = IndexedNoteAECorpus(args.corpus)
    started = time.perf_counter()
    result = {'definition': 'each full performance channel mean, including valid tail observations; '
              'tempo once per valid beat, other channels once per observed notation row; '
              'fixed existing train R scaler; original Huber group/window aggregation',
              'source_index_id': corpus.manifest['source_index_id'], 'R_scaler': corpus.r_scaler.fingerprint,
              'model_used': False, 'training_or_checkpoint_selection_changed': False,
              'scope': 'target-derived descriptive baseline; full-performance information exceeds an 8-bar input',
              'results': {}}
    for split in args.splits:
        result['results'][split] = evaluate(corpus, split)
    result['seconds'] = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({split: {name: row[name]['loss'] for name in ('train_mean', 'performance_mean')}
                      for split, row in result['results'].items()}), flush=True)


if __name__ == '__main__':
    main()
