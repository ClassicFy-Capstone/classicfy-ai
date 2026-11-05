"""Equal-group, equal-window reconstruction and fixed-manifest cosine triplets."""

import math

import torch
from torch.nn import functional as F

from .note_autoencoder import beat_groups, validate_padding


RECONSTRUCTION_GROUPS = ('tempo', 'onset', 'velocity', 'articulation', 'pedal')


def mean_tempo_loss(prediction, beat_targets, beat_mask, *, delta=1.):
    """Huber against the window mean of once-scaled, once-counted valid beats.

    Use full beat tables and window ownership masks, including beats without
    attacks. Neither note density nor repeated note-level tempo copies weight
    this target. All-missing windows contribute no loss or denominator.
    """
    if (beat_targets.ndim != 2 or beat_mask.shape != beat_targets.shape or beat_mask.dtype != torch.bool
            or prediction.shape != beat_targets.shape[:1] or not len(prediction)
            or not math.isfinite(delta) or delta <= 0):
        raise ValueError('Expected mean prediction [B], scaled beat targets/mask [B,K], and positive Huber delta')
    counts = beat_mask.sum(-1)
    valid = counts > 0
    clean = torch.where(beat_mask, beat_targets, 0.)
    pred = torch.where(valid, prediction, 0.)
    if not torch.isfinite(clean).all() or not torch.isfinite(pred).all():
        raise ValueError('Observed mean tempo prediction and beat targets must be finite')
    target = (clean.sum(-1) / counts.clamp_min(1)).detach()
    errors = F.huber_loss(pred, target, reduction='none', delta=delta)
    return {'loss': errors[valid].mean() if valid.any() else pred.sum() * 0.,
            'window_losses': torch.where(valid, errors, 0.), 'valid_windows': valid,
            'target': target, 'prediction': pred, 'beat_counts': counts, 'skip': not bool(valid.any())}


def reconstruction_loss(prediction, target, M, P, beat_id, *, delta=1.):
    if (prediction.shape != target.shape or target.shape != (*P.shape, 11)
            or M.shape != target.shape or M.dtype != torch.bool or beat_id.shape != P.shape
            or beat_id.dtype != torch.long or not math.isfinite(delta) or delta <= 0):
        raise ValueError('Invalid reconstruction shapes, masks, beat IDs or Huber delta')
    validate_padding(P, P.sum(1))
    valid = M & P[..., None]
    if torch.any(valid[..., 0] & (beat_id < 0)):
        raise ValueError('Observed tempo requires beat correspondence')
    pred, truth = torch.where(valid, prediction, 0.), torch.where(valid, target, 0.)
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError('Observed prediction/target must be finite')
    errors = F.huber_loss(pred, truth, reduction='none', delta=delta)
    zero = pred.sum() * 0.
    counts = valid.sum(1)
    channel_losses = errors.sum(1) / counts.clamp_min(1)
    rows, keys, inverse, sizes = beat_groups(beat_id, valid[..., 0])
    beat_count = torch.bincount(keys[:, 0], minlength=len(P))
    tempo_loss = pred.new_zeros(len(P)) + zero
    if len(keys):
        t, p = truth[rows[:, 0], rows[:, 1], 0], pred[rows[:, 0], rows[:, 1], 0]
        first = torch.full((len(keys),), len(rows), device=target.device, dtype=torch.long)
        first.scatter_reduce_(0, inverse, torch.arange(len(rows), device=target.device), reduce='amin')
        if not torch.allclose(t, t[first][inverse], atol=1e-6, rtol=1e-6):
            raise ValueError('Copied tempo targets disagree within one beat')
        if not torch.allclose(p, p[first][inverse], atol=1e-6, rtol=1e-6):
            raise ValueError('Tempo predictions must be shared within one beat')
        means = p.new_zeros(len(keys)).index_add(0, inverse, p) / sizes
        beat_losses = F.huber_loss(means, t[first], reduction='none', delta=delta)
        tempo_loss = tempo_loss.index_add(0, keys[:, 0], beat_losses) / beat_count.clamp_min(1)
    counts[:, 0] = beat_count
    pedal_mask = counts[:, 4:] > 0
    pedal_loss = channel_losses[:, 4:].sum(-1) / pedal_mask.sum(-1).clamp_min(1)
    groups = torch.cat([tempo_loss[:, None], channel_losses[:, 1:4], pedal_loss[:, None]], -1)
    masks = torch.cat([(beat_count > 0)[:, None], counts[:, 1:4] > 0, pedal_mask.any(-1)[:, None]], -1)
    valid_windows = masks.any(-1)
    windows = groups.sum(-1) / masks.sum(-1).clamp_min(1)
    return {'loss': windows[valid_windows].mean() if valid_windows.any() else zero,
            'window_losses': windows, 'valid_windows': valid_windows,
            'group_losses': groups, 'group_mask': masks, 'channel_counts': counts,
            'skip': not bool(valid_windows.any()), 'skip_reason': None if valid_windows.any() else 'no_observed_reconstruction'}


def triplet_style_loss(z_a, z_p, z_n, *, margin=.2):
    if (z_a.ndim != 2 or min(z_a.shape) < 1 or z_a.shape != z_p.shape or z_a.shape != z_n.shape
            or not math.isfinite(margin) or margin < 0 or any(not torch.isfinite(z).all() for z in (z_a, z_p, z_n))):
        raise ValueError('Expected matching finite triplet latents and nonnegative margin')
    a, p, n = (F.normalize(z, dim=-1) for z in (z_a, z_p, z_n))
    dap, dan = 1 - (a * p).sum(-1), 1 - (a * n).sum(-1)
    hinges = F.relu(dap - dan + margin)
    return {'loss': hinges.mean(), 'd_ap': dap, 'd_an': dan,
            'active_ratio': (hinges > 0).float().mean(), 'margin_satisfied': (dan - dap >= margin).float().mean()}
