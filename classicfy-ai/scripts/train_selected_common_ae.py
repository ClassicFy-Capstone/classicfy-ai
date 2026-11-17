"""Train the seven-channel AE on every selected representative segment.

Consumes frozen recovered feature sequences, never MusicXML or whole recordings.
No G supervision. All observed targets, including short/tail windows, are used.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from embedding.model import ModelConfig, TemporalAutoencoder, error_totals, feature_loss
from embedding.training import TrainingConfig, save_checkpoint, load_model
from embedding.temporal_reconstruction_loss import channel_error_totals, weighted_feature_loss


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, seed):
    rows = list(csv.DictReader((root / 'selected_segments.csv').open(encoding='utf-8-sig')))
    assert len({r['perf_id'] for r in rows}) == len(rows)
    parent = {r['composition_id']: r['composition_id'] for r in rows}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    score_hashes = {name: digest(Path(name)) for name in {r['local_score'] for r in rows}}
    seen = {}
    for r in rows:
        w = r['composition_id']
        for identity in [('recording', r['source_sha256']), ('score', score_hashes[r['local_score']])]:
            if identity in seen:
                a, b = find(w), find(seen[identity])
                parent[max(a, b)] = min(a, b)
            seen[identity] = w
    works = sorted({find(w) for w in parent})
    order = np.random.default_rng(seed).permutation(len(works))
    nv = max(1, round(len(works) * .15))
    splits = {works[i]: ('validation' if j < nv else 'test' if j < 2*nv else 'train')
              for j, i in enumerate(order)}
    groups = defaultdict(list)
    for r in rows:
        groups[r['stages_path'], r['masks_path']].append(r)
    sequences, hashes = [], {}
    for (sp, mp), members in groups.items():
        for name in {sp, mp}:
            hashes[name] = digest(root / name)
        with np.load(root / sp, allow_pickle=False) as z, np.load(root / mp, allow_pickle=False) as masks:
            for r in members:
                local = r['embedding_origin'] == 'local_score_fragment_alignment'
                prefix = '' if local else r['perf_id'] + '/'
                mprefix = '' if local else f"{r['perf_id']}/{int(r['candidate_index'])}/"
                x, pb = z[prefix+'standardized_values'], z[prefix+'performance_beats']
                region = (pb[:-1] >= float(r['start_seconds'])) & (pb[1:] <= float(r['end_seconds'])) & (np.diff(pb) > 0)
                valid = region & z[prefix+'standardized_mask'].all(axis=0) & np.isfinite(x).all(axis=0)
                np.testing.assert_array_equal(region, masks[mprefix+'region'])
                np.testing.assert_array_equal(valid, masks[mprefix+'valid'])
                assert int(valid.sum()) == int(float(r['valid_beats'])) >= 2
                positions = np.flatnonzero(region)
                left, right = int(positions[0]), int(positions[-1])+1
                mask = np.broadcast_to(valid[left:right], (7, right-left)).copy()
                values = np.where(mask, x[:, left:right], 0).astype(np.float32)
                assert np.isfinite(values).all()
                sequences.append({'key': r['perf_id'], 'work': find(r['composition_id']),
                    'composer': r['composer'], 'recording': r['source_sha256'],
                    'score_sha256': score_hashes[r['local_score']], 'composition_id': r['composition_id'],
                    'split': splits[find(r['composition_id'])], 'values': values, 'valid': mask,
                    'start_seconds': float(r['start_seconds']), 'end_seconds': float(r['end_seconds']),
                    'origin': r['embedding_origin'], 'first_source_beat': left})
    assert len(sequences) == len(rows)
    return sequences, hashes


class AllObservedWindows(Dataset):
    def __init__(self, sequences, size=64):
        self.sequences, self.size, self.items = sequences, size, []
        for si, s in enumerate(sequences):
            for start in range(0, s['values'].shape[1], size):
                if s['valid'][:, start:start+size].any():
                    self.items.append((si, start))
        assert {si for si, _ in self.items} == set(range(len(sequences)))
        assert sum(int(self[i][1].sum()) for i in range(len(self))) == sum(int(s['valid'].sum()) for s in sequences)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        si, start = self.items[i]
        s = self.sequences[si]
        width = min(self.size, s['values'].shape[1]-start)
        x, m = np.zeros((7, self.size), np.float32), np.zeros((7, self.size), bool)
        x[:, :width], m[:, :width] = s['values'][:, start:start+width], s['valid'][:, start:start+width]
        return torch.from_numpy(x), torch.from_numpy(m)


def evaluate(model, data, *, loss_kind='mse', huber_delta=1., pedal_weight=1.):
    model.eval()
    sums, counts = torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.long)
    objective_sums = torch.zeros(7, dtype=torch.float64)
    with torch.no_grad():
        for x, m in DataLoader(data, batch_size=32):
            y = model(x, m)[0]
            a, b = error_totals(y, x, m)
            sums += a.double(); counts += b
            objective_sums += channel_error_totals(y,x,m,kind=loss_kind,delta=huber_delta)[0].double()
    return {'loss': float(feature_loss(sums, counts)), 'channel_mse': (sums/counts.clamp_min(1)).tolist(),
            'training_objective': float(weighted_feature_loss(objective_sums,counts,pedal_weight=pedal_weight))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--loss', choices=('mse','huber'), default='mse')
    parser.add_argument('--huber-delta', type=float, default=1.)
    parser.add_argument('--pedal-weight', type=float, default=1.)
    parser.add_argument('--skip-test', action='store_true', help='Reserve test evaluation until candidate selection')
    parser.add_argument('--torch-threads', type=int, default=4)
    args = parser.parse_args()
    if not np.isfinite(args.huber_delta) or args.huber_delta<=0 or not np.isfinite(args.pedal_weight) or args.pedal_weight<=0:
        parser.error('delta and pedal weight must be finite and positive')
    if args.torch_threads < 1: parser.error('torch threads must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def status(stage, **fields):
        value = {'stage': stage, 'elapsed_seconds': time.monotonic()-started, **fields}
        dump(args.out/'status.json', value)
        print(json.dumps(value), flush=True)
    try:
        torch.set_num_threads(args.torch_threads)
        seed = 20261112
        torch.manual_seed(seed)
        status('preparing')
        sequences, hashes = prepare(args.dataset, seed)
        for field in ('work', 'recording', 'score_sha256'):
            bindings = defaultdict(set)
            for s in sequences: bindings[s[field]].add(s['split'])
            assert all(len(v)==1 for v in bindings.values()), f'{field} leaks across splits'
        datasets = {split: AllObservedWindows([s for s in sequences if s['split']==split])
                    for split in ('train', 'validation', 'test')}
        manifest = {'selected_segments_sha256': digest(args.dataset/'selected_segments.csv'),
            'source_hashes': hashes, 'seed': seed, 'normalization': 'Frozen supplied per-work reference; works disjoint across splits. No cross-work fitted transform added.',
            'target': 'Every supplied valid beat in selected time boundaries; gaps and padding masked; no synthetic masking; no G loss.',
            'sequences': [{k: v for k, v in s.items() if k not in ('values','valid')} | {'valid_beats': int(s['valid'][0].sum())} for s in sequences],
            'counts': {split: {'segments': len(d.sequences), 'works': len({s['work'] for s in d.sequences}), 'windows': len(d),
                'observed_targets': sum(int(s['valid'].sum()) for s in d.sequences)} for split, d in datasets.items()}}
        dump(args.out/'dataset.json', manifest)
        import shutil
        snapshot = args.out/'code_snapshot'
        snapshot.mkdir()
        for source in (Path(__file__), Path(__file__).resolve().parents[1]/'src/embedding/model.py',
                       Path(__file__).resolve().parents[1]/'src/embedding/training.py',
                       Path(__file__).resolve().parents[1]/'src/embedding/temporal_reconstruction_loss.py'):
            shutil.copyfile(source, snapshot/source.name)
        packed = {}
        for s in sequences:
            packed[s['key']+'/values'], packed[s['key']+'/valid'] = s['values'], s['valid']
        np.savez_compressed(args.out/'input_sequences.npz', **packed)
        del packed
        config = TrainingConfig(epochs=args.epochs, patience=10, seed=seed)
        mc = ModelConfig()
        model = TemporalAutoencoder(mc)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=.5, min_lr=1e-5)
        objective_args = {'loss_kind':args.loss,'huber_delta':args.huber_delta,'pedal_weight':args.pedal_weight}
        def save_model(name, epoch, validation_loss):
            path=args.out/name
            save_checkpoint(path,model,mc,config,epoch,validation_loss)
            payload=torch.load(path,map_location='cpu',weights_only=True)
            payload['reconstruction_objective']=dict(objective_args)
            payload['dataset_manifest_sha256']=digest(args.out/'dataset.json')
            temporary=path.with_suffix('.tmp')
            torch.save(payload,temporary);temporary.replace(path)
        initial = evaluate(model, datasets['validation'], **objective_args)
        best, significant, stale, best_epoch = initial['loss'], initial['loss'], 0, 0
        save_model('initial.pt',0,best)
        save_model('best.pt',0,best)
        dump(args.out/'protocol.json', {'objective': objective_args,
            'checkpoint_selection':'unweighted five-group validation MSE (same metric as baseline)',
            'test_reserved':args.skip_test,
            'torch_threads':args.torch_threads,
            'initial_validation': initial, 'max_epochs': args.epochs, 'patience': 10, 'min_delta': 1e-4,
            'g_loss': False, 'branch': 'feat/#20-g-loss-update', 'script_sha256': digest(Path(__file__))})
        status('training', counts=manifest['counts'], initial_validation=initial)
        rng = torch.Generator().manual_seed(seed+1)
        for epoch in range(1, args.epochs+1):
            model.train()
            sums, counts = torch.zeros(7, dtype=torch.float64), torch.zeros(7, dtype=torch.long)
            objective_sums = torch.zeros(7, dtype=torch.float64)
            for x, m in DataLoader(datasets['train'], batch_size=32, shuffle=True, generator=rng):
                optimizer.zero_grad(set_to_none=True)
                pred = model(x,m)[0]
                a,b = channel_error_totals(pred,x,m,kind=args.loss,delta=args.huber_delta)
                loss = weighted_feature_loss(a,b,pedal_weight=args.pedal_weight)
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                objective_sums += a.detach().double()
                sums += error_totals(pred.detach(),x,m)[0].double(); counts += b
            assert int(counts.sum()) == manifest['counts']['train']['observed_targets']
            val = evaluate(model,datasets['validation'],**objective_args)
            scheduler.step(val['loss'])
            if val['loss'] < best:
                best,best_epoch = val['loss'],epoch
                save_model('best.pt',epoch,best)
            if val['loss'] < significant-1e-4:
                significant,stale = val['loss'],0
            else: stale += 1
            save_model('last.pt',epoch,val['loss'])
            row = {'epoch': epoch, 'train_loss': float(feature_loss(sums,counts)), 'validation': val,
                   'train_objective':float(weighted_feature_loss(objective_sums,counts,pedal_weight=args.pedal_weight)),
                   'best_epoch': best_epoch, 'learning_rate': optimizer.param_groups[0]['lr'], 'stale': stale}
            with (args.out/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
            status('training',**row)
            if stale >= config.patience: break
        model,_ = load_model(args.out/'best.pt')
        test = None if args.skip_test else evaluate(model,datasets['test'],**objective_args)
        if test is not None: dump(args.out/'test.json',test)
        status('complete',best_epoch=best_epoch,best_validation=best,test=test,checkpoint=str(args.out/'best.pt'))
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise


if __name__ == '__main__':
    main()
