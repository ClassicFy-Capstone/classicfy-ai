"""Train the frozen 2x2 design; complete-role validation and paired random streams."""
import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import GroupCatalog, load_mapping, sha, PREVIOUS
from model import ConditionalMetricEncoder, objective, target_distances, previous

OUT = Path(__file__).resolve().parent
MODES = {'fused_no_metric': (True, False), 'separate_no_metric': (False, False),
         'fused_metric': (True, True), 'separate_metric': (False, True)}


def code_hash():
    return hashlib.sha256(b''.join((OUT / p).read_bytes() for p in ('data.py', 'model.py', 'train.py', 'protocol.json'))).hexdigest()


def validation_plan(catalog, seed):
    groups = catalog.by_role['validation']; gw = catalog.group_weights('validation')
    indices, group_ids, weights = [], [], []
    slices = []
    for j, (g, weight) in enumerate(zip(groups, gw)):
        start = len(indices); indices.extend(g.rows); group_ids.extend([j] * len(g.rows))
        weights.extend([weight / len(g.rows)] * len(g.rows)); slices.append(slice(start, len(indices)))
    batch = catalog.batch(indices, group_ids)
    generator = torch.Generator().manual_seed(seed + 100 + catalog.view)
    mask_a = previous.masks(len(indices), generator); mask_b = previous.masks(len(indices), generator)
    pair_targets = []
    for sl in slices:
        target, supported = target_distances({k: v[sl] for k, v in batch['targets'].items()})
        distinct = batch['owners'][sl][:, None] != batch['owners'][sl][None, :]
        pairs = torch.triu(supported & distinct, diagonal=1)
        assert pairs.any()
        pair_targets.append((target, pairs))
    return batch, mask_a, mask_b, torch.tensor(weights, dtype=torch.float32), gw, slices, pair_targets


def validate(model, plan, metric):
    model.eval()
    batch, masks_a, masks_b, w, gw, slices, pair_targets = plan
    parts = {}; raws = [[], []]; units = [[], []]
    with torch.no_grad():
        for start in range(0, len(batch['x']), 256):
            sl = slice(start, start + 256); x = batch['x'][sl]; c = batch['context'][sl]
            targets = {k: v[sl] for k, v in batch['targets'].items()}
            a = model(x, c, masks_a[sl]); b = model(x, c, masks_b[sl])
            ra = previous.response_errors(a['heads'], targets); rb = previous.response_errors(b['heads'], targets)
            local = {k: (ra[k] + rb[k]) / 2 for k in ra}
            local['amplitude'] = (previous.masked_mean((a['heads']['amplitude'] - targets['amplitude']).square(), targets['amplitude_mask']) +
                                  previous.masked_mean((b['heads']['amplitude'] - targets['amplitude']).square(), targets['amplitude_mask'])) / 2
            local['rec'] = (previous.reconstruction(a['heads']['rec'], x, masks_a[sl]) + previous.reconstruction(b['heads']['rec'], x, masks_b[sl])) / 2
            for name, values in local.items(): parts[name] = parts.get(name, 0.) + float((w[sl] * values).sum())
            for j, output in enumerate((a, b)):
                raws[j].append(output['raw']); units[j].append(output['unit'])
        raw_a, raw_b = [torch.cat(v) for v in raws]; unit_a, unit_b = [torch.cat(v) for v in units]
        def moments(raw):
            center = raw - (w[:, None] * raw).sum(0)
            covariance = center.T @ (center * w[:, None])
            return (float(F.relu(1 - torch.sqrt(covariance.diag() + 1e-4)).mean()),
                    float((covariance.square().sum() - covariance.diag().square().sum()) / raw.shape[1]))
        av, ac = moments(raw_a); bv, bc = moments(raw_b)
        parts.update(invariance=float((w[:, None] * (raw_a - raw_b).square()).sum() / 32),
                     variance=(av + bv) / 2, covariance=(ac + bc) / 2)
        metric_loss = 0.
        for weight, sl, (target, pairs) in zip(gw, slices, pair_targets):
            for units in (unit_a, unit_b):
                d = (units[sl][:, None] - units[sl][None, :]).square().sum(2)
                metric_loss += weight * float(F.smooth_l1_loss(d[pairs], target[pairs], beta=.25)) / 2
        parts['metric'] = float(metric_loss)
        parts['loss'] = parts['shape'] + .25 * (parts['amplitude'] + parts['direction'] + parts['peaks'] + parts['lag']) + .1 * parts['rec']
        parts['loss'] += parts['invariance'] + parts['variance'] + .04 * parts['covariance'] + (parts['metric'] if metric else 0)
        parts['median_raw_sd'] = float(torch.sqrt(((raw_a - (w[:, None] * raw_a).sum(0)).square() * w[:, None]).sum(0)).median())
        assert np.isfinite(list(parts.values())).all()
    return parts


def run(catalog, seed, mode, config):
    view = catalog.view; fused, metric = MODES[mode]
    directory = OUT / f'seed{seed}_view{view}_{mode}'; directory.mkdir(exist_ok=True)
    if (directory / 'training.json').exists():
        print('Retaining completed', directory.name, flush=True); return
    if (directory / 'initial.pt').exists(): raise RuntimeError('Incomplete run exists; do not overwrite it silently')
    torch.manual_seed(seed); model = ConditionalMetricEncoder(fused)
    optim = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    plan = validation_plan(catalog, seed)
    initial = validate(model, plan, metric)
    common = dict(view=view, seed=seed, mode=mode, fused_context=fused, metric=metric, config=config,
                  data_sha256=sha(PREVIOUS / 'examples.npz'), code_sha256=code_hash(),
                  train_owners=catalog.data.gids.tolist(), validation_owners=sorted(set(plan[0]['owners'].tolist())),
                  input_scales=catalog.data.scales, amplitude_scales=catalog.amplitude_scales,
                  parameters=sum(p.numel() for p in model.parameters()), test_used_for_training_or_selection=False)
    torch.save(dict(**common, state_dict=model.state_dict(), epoch=0, validation=initial), directory / 'initial.pt')
    best = initial['loss']; best_epoch = 0; best_state = copy.deepcopy(model.state_dict()); best_values = initial; stale = 0
    rows = [dict(epoch=0, seconds=0, sampling_sha256='', **{'validation_' + k: v for k, v in initial.items()})]
    start_time = time.monotonic(); rng = np.random.default_rng(seed + 200 + view)
    generator = torch.Generator().manual_seed(seed + 300 + view)
    train_owners = set(catalog.data.gids.tolist())
    for epoch in range(1, config['epochs'] + 1):
        model.train(); totals = {}; stream = hashlib.sha256()
        for step in range(config['steps_per_epoch']):
            batch = catalog.sample_train(rng, config['batch_passages'], config['max_recordings_per_passage'])
            assert set(batch['owners'].tolist()) <= train_owners
            ma = previous.masks(len(batch['x']), generator); mb = previous.masks(len(batch['x']), generator)
            stream.update(batch['indices'].tobytes()); stream.update(ma.numpy().tobytes()); stream.update(mb.numpy().tobytes())
            optim.zero_grad(set_to_none=True)
            loss, parts, _ = objective(model, batch, ma, mb, metric)
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip']); optim.step()
            for name, value in parts.items(): totals[name] = totals.get(name, 0.) + value
        result = validate(model, plan, metric)
        rows.append(dict(epoch=epoch, seconds=time.monotonic() - start_time, sampling_sha256=stream.hexdigest(),
                         **{'train_' + k: v / config['steps_per_epoch'] for k, v in totals.items()},
                         **{'validation_' + k: v for k, v in result.items()}))
        if result['loss'] < best - 1e-6:
            best = result['loss']; best_epoch = epoch; best_values = result; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else: stale += 1
        fields = sorted(set(k for row in rows for k in row))
        with (directory / 'history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        if epoch == 1 or epoch % 4 == 0:
            print(directory.name, 'epoch', epoch, 'val', round(result['loss'], 4), 'metric', round(result['metric'], 4), 'best', best_epoch, 's', round(time.monotonic() - start_time), flush=True)
        if stale >= config['patience'] and epoch >= config['minimum_epochs']: break
    torch.save(dict(**common, state_dict=best_state, epoch=best_epoch, validation=best_values), directory / 'best.pt')
    summary = dict(**{k: v for k, v in common.items() if k not in ('input_scales', 'amplitude_scales')},
                   initial=initial, best=best_values, best_epoch=best_epoch, epochs_run=epoch, seconds=time.monotonic() - start_time)
    (directory / 'training.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Completed', directory.name, 'best', best_epoch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, required=True); args = parser.parse_args()
    config = json.loads((OUT / 'protocol.json').read_text())['training']
    assert args.seed in config['seeds']
    torch.set_num_threads(config['cpu_threads']); torch.use_deterministic_algorithms(True)
    mapping = load_mapping()
    for view in (0, 1):
        catalog = GroupCatalog(view, mapping)
        for mode in MODES: run(catalog, args.seed, mode, config)
