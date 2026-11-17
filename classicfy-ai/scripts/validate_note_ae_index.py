"""Exercise every indexed input through the actual unchanged AE collator."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from embedding.note_ae_source_index import IndexedNoteAECorpus
from embedding.note_ae_loss import reconstruction_loss
from embedding.style_profile_data import load_style_artifact


def validate(path, out):
    torch.set_num_threads(1)
    corpus = IndexedNoteAECorpus(load_style_artifact(path, 'ae-training-index'),
                                path.parent / 'R_scaler.json', path.parent / 'S_scale.json')
    result = {'splits': {}, 'excluded_windows': len(corpus.excluded), 'model_or_optimizer_run': False}
    for split in ('train', 'validation', 'test'):
        dataset = corpus.dataset(split)
        if not len(dataset):
            raise ValueError('All three work-level splits require usable windows')
        count, batches, padding = 0, 0, 0
        for batch in DataLoader(dataset, batch_size=4, collate_fn=corpus.collate, num_workers=0):
            if not torch.isfinite(batch['R']).all() or batch['M'][~batch['P']].any():
                raise ValueError('Nonfinite R or observations on padding')
            if not torch.equal(batch['lengths'], batch['P'].sum(1)):
                raise ValueError('Invalid note lengths')
            zero = reconstruction_loss(batch['R'], batch['R'], batch['M'], batch['P'], batch['beat_id'])
            if float(zero['loss']) != 0. or zero['skip']:
                raise ValueError('Invalid reconstruction targets or tempo ownership')
            count += len(batch['P']); batches += 1; padding += int((~batch['P']).sum())
            if batches % 250 == 0:
                print(f'{split}: checked {count}/{len(dataset)} windows', flush=True)
        if count != len(dataset):
            raise ValueError('Input validation missed windows')
        result['splits'][split] = {'windows': count, 'batches': batches, 'padding': padding}
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    validate(args.index.resolve(), args.out.resolve())
