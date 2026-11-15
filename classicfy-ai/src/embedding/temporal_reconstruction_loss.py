"""Observed-only MSE/Huber with explicit musical-group weights."""
import math
import torch
from torch.nn import functional as F

GROUPS = ((0,), (1,), (2,), (3,), (4, 5, 6))


def channel_error_totals(prediction, target, valid, *, kind='mse', delta=1.):
    if kind not in ('mse', 'huber') or not math.isfinite(delta) or delta <= 0:
        raise ValueError('Expected mse/huber and a positive finite Huber delta')
    if (prediction.shape != target.shape or valid.shape != target.shape
            or prediction.ndim != 3 or prediction.shape[1] != 7 or valid.dtype != torch.bool):
        raise ValueError('Expected matching [B,7,T] predictions, targets and boolean mask')
    if not valid.any():
        raise ValueError('No observed reconstruction targets')
    pred = torch.where(valid, prediction, 0.)
    truth = torch.where(valid, target, 0.)
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError('Nonfinite observed prediction/target')
    errors = ((pred-truth).square() if kind == 'mse' else
              F.huber_loss(pred, truth, reduction='none', delta=delta))
    return errors.sum((0, 2)), valid.sum((0, 2))


def weighted_feature_loss(sums, counts, *, pedal_weight=1.):
    if not math.isfinite(pedal_weight) or pedal_weight <= 0:
        raise ValueError('Pedal weight must be positive and finite')
    if sums.shape != (7,) or counts.shape != (7,):
        raise ValueError('Expected seven channel sums/counts')
    channel = sums / counts.clamp_min(1)
    terms, weights = [], []
    for i, group in enumerate(GROUPS):
        present = counts[list(group)] > 0
        if present.any():
            weight = pedal_weight if i == 4 else 1.
            terms.append(channel[list(group)][present].mean() * weight)
            weights.append(weight)
    if not terms:
        raise ValueError('No observed feature groups')
    return torch.stack(terms).sum() / sum(weights)
