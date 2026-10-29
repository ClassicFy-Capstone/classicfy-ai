"""Equal-group, equal-window reconstruction and fixed-manifest cosine triplets."""

import math

import torch
from torch.nn import functional as F

from .note_autoencoder import validate_padding


RECONSTRUCTION_GROUPS = ('tempo', 'onset', 'velocity', 'articulation', 'pedal')


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
    window_losses, group_losses, group_masks, counts = [], [], [], []
    zero = pred.sum() * 0.
    for i in range(len(P)):
        losses, flags = [], []
        count = torch.zeros(11, dtype=torch.long, device=target.device)
        beat_losses = []
        for key in torch.unique(beat_id[i, valid[i, :, 0]]):
            rows = valid[i, :, 0] & (beat_id[i] == key)
            t, p = truth[i, rows, 0], pred[i, rows, 0]
            if not torch.allclose(t, t[:1].expand_as(t), atol=1e-6, rtol=1e-6):
                raise ValueError('Copied tempo targets disagree within one beat')
            if not torch.allclose(p, p[:1].expand_as(p), atol=1e-6, rtol=1e-6):
                raise ValueError('Tempo predictions must be shared within one beat')
            beat_losses.append(F.huber_loss(p.mean(), t[0], delta=delta))
        count[0] = len(beat_losses)
        losses.append(torch.stack(beat_losses).mean() if beat_losses else zero)
        flags.append(bool(beat_losses))
        for k in (1, 2, 3):
            count[k] = valid[i, :, k].sum()
            ok = bool(count[k] > 0)
            losses.append(errors[i, valid[i, :, k], k].mean() if ok else zero)
            flags.append(ok)
        pedal = []
        for k in range(4, 11):
            count[k] = valid[i, :, k].sum()
            if count[k] > 0:
                pedal.append(errors[i, valid[i, :, k], k].mean())
        losses.append(torch.stack(pedal).mean() if pedal else zero)
        flags.append(bool(pedal))
        gs = torch.stack(losses)
        gm = torch.tensor(flags, dtype=torch.bool, device=target.device)
        group_losses.append(gs)
        group_masks.append(gm)
        counts.append(count)
        window_losses.append(gs[gm].mean() if gm.any() else zero)
    groups, masks = torch.stack(group_losses), torch.stack(group_masks)
    valid_windows = masks.any(-1)
    windows = torch.stack(window_losses)
    return {'loss': windows[valid_windows].mean() if valid_windows.any() else zero,
            'window_losses': windows, 'valid_windows': valid_windows,
            'group_losses': groups, 'group_mask': masks, 'channel_counts': torch.stack(counts),
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
