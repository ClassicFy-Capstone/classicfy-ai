"""Small-run friendly training, validation, checkpointing and note-window export."""

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from features.note_performance import note_performance_schema
from features.note_performance_normalization import NotePerformanceScaler
from features.note_score_context import NoteScoreScale, note_score_schema
from features.style_profile import stable_id
from .note_ae_data import to_device
from .note_ae_loss import RECONSTRUCTION_GROUPS, mean_tempo_loss, reconstruction_loss, triplet_style_loss
from .note_autoencoder import NoteAEConfig, NotePerformanceAutoencoder
from .note_score_encoder import NoteScoreEncoderConfig


@dataclass(frozen=True)
class NoteAETrainingConfig:
    epochs: int = 20
    batch_size: int = 4
    triplet_batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.
    margin: float = .2
    lambda_style: float = .1
    huber_delta: float = 1.
    seed: int = 2026
    device: str = 'cpu'
    torch_threads: int = 1
    max_steps_per_epoch: int = 0
    max_validation_batches: int = 0
    beta_mean_tempo: float = 0.
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.
    evaluate_train_each_epoch: bool = False

    def __post_init__(self):
        for key in ('epochs', 'batch_size', 'triplet_batch_size', 'torch_threads'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Positive integer training sizes required')
        for key in ('seed', 'max_steps_per_epoch', 'max_validation_batches', 'early_stopping_patience'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('Nonnegative integer seed/limits required')
        for key in ('learning_rate', 'gradient_clip', 'huber_delta', 'weight_decay', 'margin', 'lambda_style',
                    'beta_mean_tempo', 'early_stopping_min_delta'):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0 or (key in {'learning_rate', 'gradient_clip', 'huber_delta'} and value == 0):
                raise ValueError('Invalid optimizer or loss setting')
        if self.device not in {'cpu', 'cuda', 'mps'}:
            raise ValueError('Supported devices are cpu, cuda and mps')
        if not isinstance(self.evaluate_train_each_epoch, bool):
            raise ValueError('evaluate_train_each_epoch must be boolean')


def ae_schema():
    return {'version': 'conditional-note-ae-v1', 'R': note_performance_schema(), 'S': note_score_schema(),
            'tempo': 'one prediction per note-associated beat; no artificial notes',
            'loss': 'Huber; observed-channel pedal mean; equal present groups; equal valid windows',
            'style': 'cosine normalized z; fixed manifest only'}


def seed_run(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)


def _loader(dataset, size, collate, *, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=size, shuffle=shuffle, collate_fn=collate, generator=generator if generator is not None else torch.Generator().manual_seed(0), num_workers=0)


class _RecMetrics:
    def __init__(self):
        self.total, self.windows, self.skipped = 0., 0, 0
        self.groups, self.support, self.counts = np.zeros(5), np.zeros(5, int), np.zeros(11, int)

    def add(self, loss):
        good = int(loss['valid_windows'].sum())
        self.total += float(loss['loss'].detach()) * good
        self.windows += good
        self.skipped += int((~loss['valid_windows']).sum())
        g, m = loss['group_losses'].detach().cpu().numpy(), loss['group_mask'].cpu().numpy()
        self.groups += np.where(m, g, 0).sum(0)
        self.support += m.sum(0)
        self.counts += loss['channel_counts'].cpu().numpy().sum(0)

    def result(self):
        return {'loss': self.total / self.windows if self.windows else None, 'windows': self.windows,
                'skipped_windows': self.skipped,
                'groups': {name: {'loss': float(self.groups[i] / self.support[i]) if self.support[i] else None,
                                   'valid_windows': int(self.support[i])} for i, name in enumerate(RECONSTRUCTION_GROUPS)},
                'channel_counts': self.counts.tolist()}


class _MeanTempoMetrics:
    def __init__(self):
        self.total, self.squared, self.windows = 0., 0., 0

    def add(self, loss):
        valid = loss['valid_windows']
        self.windows += int(valid.sum())
        self.total += float(loss['window_losses'][valid].detach().sum())
        self.squared += float((loss['prediction'][valid] - loss['target'][valid]).detach().square().sum())

    def result(self):
        return {'loss': self.total / self.windows if self.windows else None,
                'standardized_RMSE': math.sqrt(self.squared / self.windows) if self.windows else None,
                'valid_windows': self.windows}


def _mean_tempo_output(output, batch, delta):
    if 'mean_tempo_hat' not in output:
        return None
    return mean_tempo_loss(output['mean_tempo_hat'], batch['beat_scaled'], batch['beat_loss_mask'], delta=delta)


@torch.no_grad()
def evaluate_note_ae(model, dataset, collate, config):
    model.eval()
    metrics, latents, mean_metrics = _RecMetrics(), [], _MeanTempoMetrics()
    for step, batch in enumerate(_loader(dataset, config.batch_size, collate)):
        if config.max_validation_batches and step >= config.max_validation_batches:
            break
        batch = to_device(batch, config.device)
        output = model(batch)
        metrics.add(reconstruction_loss(output['R_hat'], batch['R'], batch['M'], batch['P'], batch['beat_id'], delta=config.huber_delta))
        mean = _mean_tempo_output(output, batch, config.huber_delta)
        if mean is not None:
            mean_metrics.add(mean)
        latents.append(output['z'].cpu())
    result = metrics.result()
    result['mean_tempo_auxiliary'] = mean_metrics.result() if model.config.mean_tempo_auxiliary else None
    result['status'] = 'evaluated' if result['windows'] else 'no_validation_windows'
    if latents:
        z = torch.cat(latents)
        result.update(z_norm_mean=float(z.norm(dim=-1).mean()), embedding_variance=float(z.var(dim=0, unbiased=False).mean()))
    return result


def _rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'mps': torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def _restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if state['mps'] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state['mps'])


def save_note_ae_checkpoint(path, model, optimizer, config, corpus, *, epoch, step, triplets=None,
                            generators=None, best_validation=None, early_stopping_state=None):
    payload = {'schema': ae_schema(), 'model_config': asdict(model.config),
               'score_config': asdict(model.score_encoder.config), 'training_config': asdict(config),
               'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'step': step,
               'rng': _rng_state(), 'corpus_contract': corpus.contract,
               'R_scaler': corpus.r_scaler.to_dict(), 'R_scaler_fingerprint': corpus.r_scaler.fingerprint,
               'score_scale': json.loads(json.dumps(asdict(corpus.score_scale))),
               'g_dependencies': triplets.dependencies if triplets is not None else {},
               'generators': {k: g.get_state() for k, g in (generators or {}).items()},
               'best_validation': best_validation, 'checkpoint_boundary': 'completed epoch',
               'early_stopping_state': early_stopping_state or {'significant_best': None, 'bad_epochs': 0}}
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def read_note_ae_checkpoint(path, *, corpus=None, triplets=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != ae_schema():
        raise ValueError('Checkpoint AE/S/R schema mismatch')
    r = payload['R_scaler']
    if r.get('schema') != note_performance_schema():
        raise ValueError('Checkpoint R scaler schema mismatch')
    scaler = NotePerformanceScaler(*(tuple(r[key]) for key in ('mean', 'scale', 'valid_counts', 'trained_channel_mask', 'channel_status')),
                                   r['fit_manifest_id'], tuple(r['training_records']), r['schema_version'])
    score = payload['score_scale']
    score_scale = NoteScoreScale(tuple(score['mean']), tuple(score['scale']), tuple(score['training_sources']), score['schema_version'])
    if (scaler.fingerprint != payload['R_scaler_fingerprint'] or scaler.fingerprint != payload['corpus_contract']['R_scaler']
            or stable_id(asdict(score_scale)) != payload['corpus_contract']['score_scale']):
        raise ValueError('Checkpoint scaler integrity mismatch')
    if corpus is not None and payload['corpus_contract'] != corpus.contract:
        raise ValueError('Checkpoint corpus/split/scaler contract mismatch')
    if triplets is not None and payload['g_dependencies'] != triplets.dependencies:
        raise ValueError('Checkpoint fixed g/Triplet manifest mismatch')
    return payload, scaler, score_scale


def load_note_ae_model(path, *, corpus=None):
    payload, scaler, score = read_note_ae_checkpoint(path, corpus=corpus)
    model = NotePerformanceAutoencoder(NoteAEConfig(**payload['model_config']), NoteScoreEncoderConfig(**payload['score_config']))
    model.load_state_dict(payload['model'])
    model.r_scaler_fingerprint = scaler.fingerprint
    model.score_scale_fingerprint = stable_id(asdict(score))
    model.eval()
    return model, payload, scaler, score


def train_note_ae(corpus, out, *, config=None, model_config=None, score_config=None, triplets=None, resume=None):
    config, model_config = config or NoteAETrainingConfig(), model_config or NoteAEConfig()
    if config.beta_mean_tempo > 0 and not model_config.mean_tempo_auxiliary:
        raise ValueError('Positive beta_mean_tempo requires the mean tempo auxiliary head')
    seed_run(config)
    train, validation = corpus.dataset('train'), corpus.dataset('validation')
    if not len(train):
        raise ValueError('No observed full eight-bar train windows')
    if config.early_stopping_patience and not len(validation):
        raise ValueError('Validation early stopping requires validation windows')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model = NotePerformanceAutoencoder(model_config, score_config).to(config.device)
    model.r_scaler_fingerprint = corpus.r_scaler.fingerprint
    model.score_scale_fingerprint = stable_id(asdict(corpus.score_scale))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate, weight_decay=config.weight_decay)
    generators = {key: torch.Generator().manual_seed(config.seed + i) for i, key in enumerate(('reconstruction', 'style'))}
    start_epoch, step, best = 0, 0, None
    stopping = {'significant_best': None, 'bad_epochs': 0}
    if resume:
        payload, _, _ = read_note_ae_checkpoint(resume, corpus=corpus, triplets=triplets)
        expected_g = triplets.dependencies if triplets is not None else {}
        old_config = {k: v for k, v in asdict(NoteAETrainingConfig(**payload['training_config'])).items() if k != 'epochs'}
        if (asdict(NoteAEConfig(**payload['model_config'])) != asdict(model_config) or payload['score_config'] != asdict(model.score_encoder.config)
                or old_config != {k: v for k, v in asdict(config).items() if k != 'epochs'} or payload['g_dependencies'] != expected_g):
            raise ValueError('Resume model/training/g configuration mismatch')
        model.load_state_dict(payload['model'])
        optimizer.load_state_dict(payload['optimizer'])
        _restore_rng(payload['rng'])
        for k, g in generators.items():
            g.set_state(payload['generators'][k])
        start_epoch, step, best = payload['epoch'], payload['step'], payload['best_validation']
        stopping = payload.get('early_stopping_state', stopping).copy()
        if config.early_stopping_patience and stopping['bad_epochs'] >= config.early_stopping_patience:
            raise ValueError('Checkpoint already reached its validation early stopping limit')
    if config.epochs <= start_epoch:
        raise ValueError('Resume epoch limit must exceed the completed checkpoint epoch')
    rec_loader = _loader(train, config.batch_size, corpus.collate, shuffle=True, generator=generators['reconstruction'])
    style_loader = (_loader(triplets, config.triplet_batch_size, triplets.collate, shuffle=True, generator=generators['style'])
                    if triplets is not None and len(triplets) and config.lambda_style > 0 else None)
    initial = evaluate_note_ae(model, train, corpus.collate, config)
    history = []
    expected_windows = Counter(item['window']['window_id'] for item in train.items)
    stop_reason = 'max_epochs'
    for epoch in range(start_epoch, config.epochs):
        model.train()
        mean_metrics = _MeanTempoMetrics()
        metrics, style_sum, active_sum, dap_sum, dan_sum, margin_sum = _RecMetrics(), 0., 0., 0., 0., 0.
        total_sum, updates, exposed, cycles, skipped = 0., 0, 0, 0, 0
        style_iter = iter(style_loader) if style_loader is not None else None
        norms, latents, exposure_ids = [], [], Counter()
        rec_ids, triplet_ids = Counter(), Counter()
        for batch_index, raw in enumerate(rec_loader):
            if config.max_steps_per_epoch and batch_index >= config.max_steps_per_epoch:
                break
            batch = to_device(raw, config.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            rec = reconstruction_loss(output['R_hat'], batch['R'], batch['M'], batch['P'], batch['beat_id'], delta=config.huber_delta)
            mean = _mean_tempo_output(output, batch, config.huber_delta)
            if mean is not None:
                mean_metrics.add(mean)
            metrics.add(rec)
            rec_ids.update(w['window_id'] for w, valid in zip(raw['window_metadata'], rec['valid_windows']) if valid)
            style = None
            if style_iter is not None:
                try:
                    style_batch = next(style_iter)
                except StopIteration:
                    cycles += 1
                    style_iter = iter(style_loader)
                    style_batch = next(style_iter)
                encoded = [model.encode_batch(to_device(style_batch[role], config.device))['z']
                           for role in ('anchor', 'positive', 'negative')]
                style = triplet_style_loss(*encoded, margin=config.margin)
                count = len(encoded[0])
                exposed += count
                triplet_ids.update(style_batch['triplet_ids'])
                style_sum += float(style['loss'].detach()) * count
                active_sum += float(style['active_ratio']) * count
                dap_sum += float(style['d_ap'].detach().sum())
                dan_sum += float(style['d_an'].detach().sum())
                margin_sum += float(style['margin_satisfied']) * count
                exposure_ids.update(w['window_id'] for w in style_batch['anchor']['window_metadata'])
            if rec['skip'] and style is None and (mean is None or mean['skip'] or config.beta_mean_tempo == 0):
                skipped += 1
                continue
            loss = rec['loss'] + (config.lambda_style * style['loss'] if style is not None else 0.)
            if mean is not None and config.beta_mean_tempo > 0:
                loss = loss + config.beta_mean_tempo * mean['loss']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            norms.append(float(norm))
            latents.append(output['z'].detach().cpu())
            total_sum += float(loss.detach())
            updates += 1
            step += 1
        validation_metrics = evaluate_note_ae(model, validation, corpus.collate, config)
        score = validation_metrics['loss']
        improved = score is not None and (best is None or score < best)
        if improved:
            best = score
        if config.early_stopping_patience:
            if score is None:
                raise ValueError('Early stopping requires an observed validation reconstruction loss')
            # Checkpoint selection always keeps the true minimum. min_delta
            # controls only patience, independently of small best improvements.
            if (stopping['significant_best'] is None
                    or score < stopping['significant_best'] - config.early_stopping_min_delta):
                stopping = {'significant_best': score, 'bad_epochs': 0}
            else:
                stopping['bad_epochs'] += 1
        complete_epoch = rec_ids == expected_windows
        if config.max_steps_per_epoch == 0 and not complete_epoch:
            raise ValueError('Unlimited reconstruction epoch did not use every train window exactly once')
        full_train = (evaluate_note_ae(model, train, corpus.collate, replace(config, max_validation_batches=0))
                      if config.evaluate_train_each_epoch else None)
        train_metrics = metrics.result()
        z = torch.cat(latents) if latents else None
        row = {'epoch': epoch + 1, 'step': step, 'reconstruction': train_metrics, 'validation': validation_metrics,
               'full_train_evaluation': full_train, 'early_stopping_state': stopping.copy(),
               'complete_train_epoch': complete_epoch,
               'mean_tempo_auxiliary': mean_metrics.result() if model_config.mean_tempo_auxiliary else None,
               'total_loss': total_sum / updates if updates else None,
               'style_loss': style_sum / exposed if exposed else None,
               'weighted_style_loss': config.lambda_style * style_sum / exposed if exposed else None,
               'active_triplet_ratio': active_sum / exposed if exposed else None,
               'd_ap': dap_sum / exposed if exposed else None, 'd_an': dan_sum / exposed if exposed else None,
               'train_triplet_margin_satisfied': margin_sum / exposed if exposed else None,
               'style_loader_restarts': cycles, 'triplet_exposures': exposed, 'anchor_exposures': dict(exposure_ids),
               'reconstruction_window_exposures': dict(rec_ids), 'triplet_id_exposures': dict(triplet_ids),
               'updates': updates, 'skipped_batches': skipped, 'skip_reason': 'no_observed_reconstruction' if skipped else None,
               'gradient_norm_max': max(norms) if norms else None,
               'z_norm_mean': float(z.norm(dim=-1).mean()) if z is not None else None,
               'embedding_variance': float(z.var(dim=0, unbiased=False).mean()) if z is not None else None}
        history.append(row)
        save_note_ae_checkpoint(out / 'last.pt', model, optimizer, config, corpus, epoch=epoch + 1, step=step,
                                triplets=triplets, generators=generators, best_validation=best, early_stopping_state=stopping)
        if improved:
            save_note_ae_checkpoint(out / 'best.pt', model, optimizer, config, corpus, epoch=epoch + 1, step=step,
                                    triplets=triplets, generators=generators, best_validation=best, early_stopping_state=stopping)
        print(json.dumps({'epoch': epoch + 1, 'rec': train_metrics['loss'], 'style': row['style_loss'], 'validation': score,
                          'full_train': full_train['loss'] if full_train is not None else None,
                          'complete_train_epoch': complete_epoch, 'bad_epochs': stopping['bad_epochs']}), flush=True)
        if config.early_stopping_patience and stopping['bad_epochs'] >= config.early_stopping_patience:
            stop_reason = 'validation_early_stopping'
            break
    final = evaluate_note_ae(model, train, corpus.collate, config)
    report = {'schema': ae_schema(), 'training_config': asdict(config), 'model_config': asdict(model_config),
              'score_config': asdict(model.score_encoder.config), 'score_context': 'computed on demand; ' + model_config.score_gradient,
              'corpus_contract': corpus.contract, 'g_dependencies': triplets.dependencies if triplets is not None else {},
              'train_windows': len(train), 'validation_windows': len(validation),
              'test_windows_not_used': len(corpus.dataset('test')), 'eligible_triplets': len(triplets) if triplets is not None else 0,
              'style_training_performed': bool(style_loader is not None and any(r['triplet_exposures'] for r in history)),
              'mean_tempo_auxiliary': {'enabled': model_config.mean_tempo_auxiliary, 'beta': config.beta_mean_tempo,
                  'target': 'mean of beat_scaled over window-owned valid beats, including rests; same train tempo scaler',
                  'decoder_receives_target_mean': False},
              'excluded_windows': corpus.excluded, 'initial_train': initial, 'final_train': final, 'history': history,
              'best_validation': best, 'checkpoint_selection': 'validation reconstruction' if best is not None else 'no best selection; last only',
              'stop_reason': stop_reason, 'completed_epoch': history[-1]['epoch'], 'early_stopping_state': stopping,
              'limits': 'initial hyperparameters; train g agreement is not listener-style generalization'}
    (out / 'training.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return model, report


@torch.no_grad()
def export_note_embeddings(model, corpus, path, *, split, checkpoint_id, batch_size=4, device='cpu'):
    model.to(device).eval()
    zs, es, metadata = [], [], []
    for batch in _loader(corpus.dataset(split), batch_size, corpus.collate):
        output = model.encode_batch(to_device(batch, device))
        zs.append(output['z'].cpu().numpy())
        es.append(output['e'].cpu().numpy())
        metadata.extend(batch['window_metadata'])
    manifest = {'schema': ae_schema(), 'checkpoint_id': checkpoint_id, 'corpus_contract': corpus.contract,
                'split': split, 'windows': metadata, 'R_scaler': corpus.r_scaler.fingerprint}
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, z=np.concatenate(zs) if zs else np.zeros((0, model.config.latent_dim)),
                            e=np.concatenate(es) if es else np.zeros((0, model.config.latent_dim)),
                            manifest=np.array(json.dumps(manifest, ensure_ascii=False, allow_nan=False)))
    return manifest


@torch.no_grad()
def latent_ablation(model, corpus, *, split='train', device='cpu'):
    model.to(device).eval()
    groups = {}
    for item in corpus.dataset(split).items:
        groups.setdefault(item['window']['same_window_key'], []).append(item)
    pair = next((items[:2] for items in groups.values() if len({i['window']['performance_id'] for i in items}) >= 2), None)
    if pair is None:
        return {'status': 'no_same_window_performance_pair'}
    batch = to_device(corpus.collate(pair), device)
    C = model.context(batch)
    z = model.ae.encode(C, batch['R'], batch['M'], batch['P'], batch['lengths'])['z']
    results, predictions = {}, {}
    for name, value in (('own_z', z), ('swapped_z', z.flip(0)), ('zero_z', torch.zeros_like(z))):
        output = model.ae.decode(C, value, batch['P'], batch['lengths'], batch['beat_id'])['R_hat']
        predictions[name] = output
        results[name] = float(reconstruction_loss(output, batch['R'], batch['M'], batch['P'], batch['beat_id'])['loss'])
    observed = batch['M']
    common = observed[0] & observed[1]
    target_difference = float((batch['R'][0][common] - batch['R'][1][common]).square().mean()) if common.any() else None
    return {'status': 'evaluated', 'window_ids': [i['window']['window_id'] for i in pair],
            'reconstruction': results, 'target_pair_mean_squared_difference': target_difference,
            'swapped_output_mean_squared_change': float((predictions['own_z'][observed] - predictions['swapped_z'][observed]).square().mean()),
            'zero_output_mean_squared_change': float((predictions['own_z'][observed] - predictions['zero_z'][observed]).square().mean()),
            'interpretation': 'small-run diagnostic; near-zero changes may indicate weak latent use, not a musical-quality evaluation'}
