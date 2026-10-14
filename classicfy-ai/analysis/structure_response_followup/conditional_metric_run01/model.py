"""Separate response retrieval from score conditioning; learn same-passage distances."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location('frozen_response_losses', Path(__file__).resolve().parent.parent / 'response_model/model.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


class ConditionalMetricEncoder(nn.Module):
    def __init__(self, fused_context=False):
        super().__init__()
        self.fused_context = fused_context
        self.performance = nn.Sequential(nn.Conv1d(21, 32, 1), nn.GELU(),
                                         previous.Block(1), previous.Block(2), previous.Block(4))
        self.performance_vector = nn.Linear(32 * 17, 32)
        self.context = nn.Sequential(nn.Conv1d(13, 16, 1), nn.GELU(), nn.Flatten(), nn.Linear(16 * 17, 32), nn.GELU())
        # Same module/init in controls: primary fills the context half with zeros.
        self.search_fusion = nn.Linear(64, 32)
        widths = dict(rec=7 * 17, shape=2 * 16, amplitude=2, direction=2 * 16 * 3, peaks=2 * 17, lag=54)
        self.heads = nn.ModuleDict({name: nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, width))
                                   for name, width in widths.items()})

    def _raw(self, x, hidden=None, context_vector=None):
        if x.shape[1:] != (7, 17):
            raise ValueError('Expected Bx7x17 observed performance')
        hidden = torch.zeros_like(x, dtype=torch.bool) if hidden is None else hidden
        clean = torch.where(hidden, 0., x)
        p = self.performance(torch.cat((clean, torch.ones_like(x), hidden.to(x.dtype)), dim=1))
        p = self.performance_vector(p.flatten(1))
        c = context_vector if self.fused_context else torch.zeros_like(p)
        if c is None:
            raise ValueError('Fused control requires score context')
        return self.search_fusion(torch.cat((p, c), dim=1))

    def encode(self, x, context=None, hidden=None):
        """Primary retrieval does not read score context, even when supplied."""
        c = self.context(context) if self.fused_context else None
        return F.normalize(self._raw(x, hidden, c), dim=1, eps=1e-8)

    def forward(self, x, context, hidden=None):
        c = self.context(context)
        raw = self._raw(x, hidden, c)
        unit = F.normalize(raw, dim=1, eps=1e-8)
        decoder_input = torch.cat((raw, c), dim=1)
        return dict(raw=raw, unit=unit, heads={name: head(decoder_input) for name, head in self.heads.items()})


def target_distances(targets):
    """Continuous 0..4 pair targets; renormalize only genuinely supported blocks."""
    reference = targets['shape']
    n = len(reference)
    numerator = reference.new_zeros((n, n)); denominator = reference.new_zeros((n, n))
    for name, weight in (('shape', .6), ('amplitude', .2), ('lag', .2)):
        values = targets[name].detach().reshape(n, -1)
        valid = targets[name + '_mask'].detach().reshape(n, -1).bool()
        jointly_valid = valid[:, None, :] & valid[None, :, :]
        counts = jointly_valid.sum(2)
        delta = values[:, None, :] - values[None, :, :]
        distances = (delta.square() * jointly_valid).sum(2) / counts.clamp_min(1)
        supported = counts > 0
        numerator += weight * torch.where(supported, distances, 0.)
        denominator += weight * supported
    supported = denominator > 0
    return (numerator / denominator.clamp_min(1e-8)).detach(), supported


def within_passage_loss(unit, targets, groups, owners):
    """No cross-passage or duplicate-owner supervision; equal passage averaging."""
    wanted, supported = target_distances(targets)
    # Avoid sqrt(cdist) derivative issues on identical vectors.
    distance = (unit[:, None, :] - unit[None, :, :]).square().sum(2)
    same_passage = groups[:, None] == groups[None, :]
    other_recording = owners[:, None] != owners[None, :]
    pairs = torch.triu(same_passage & other_recording & supported, diagonal=1)
    group_losses = []
    for group in groups.unique():
        local = pairs & (groups[:, None] == group)
        if local.any():
            group_losses.append(F.smooth_l1_loss(distance[local], wanted[local], beta=.25))
    if not group_losses:
        raise ValueError('No supported distinct-recording pairs within an aligned passage')
    return torch.stack(group_losses).mean(), int(pairs.sum()), len(group_losses)


def balanced_weights(groups, dtype):
    _, inverse, counts = groups.unique(return_inverse=True, return_counts=True)
    return (1 / counts[inverse].to(dtype)) / len(counts)


def balanced_regularization(a, b, groups):
    w = balanced_weights(groups, a.dtype)
    invariance = (w[:, None] * (a - b).square()).sum() / a.shape[1]
    def terms(x):
        center = x - (w[:, None] * x).sum(0)
        cov = center.T @ (center * w[:, None])
        variance = F.relu(1 - torch.sqrt(cov.diag() + 1e-4)).mean()
        off_diagonal = (cov.square().sum() - cov.diag().square().sum()) / x.shape[1]
        return variance, off_diagonal
    av, ac = terms(a); bv, bc = terms(b)
    return dict(invariance=invariance, variance=(av + bv) / 2, covariance=(ac + bc) / 2)


def objective(model, batch, hidden_a, hidden_b, metric=True):
    x, c, targets = batch['x'], batch['context'], batch['targets']
    a, b = model(x, c, hidden_a), model(x, c, hidden_b)
    errors_a = previous.response_errors(a['heads'], targets)
    errors_b = previous.response_errors(b['heads'], targets)
    w = balanced_weights(batch['groups'], x.dtype)
    errors = {name: (w * (errors_a[name] + errors_b[name]) / 2).sum() for name in errors_a}
    amplitude = (previous.masked_mean((a['heads']['amplitude'] - targets['amplitude']).square(), targets['amplitude_mask']) +
                 previous.masked_mean((b['heads']['amplitude'] - targets['amplitude']).square(), targets['amplitude_mask']))
    amplitude = (w * amplitude).sum() / 2
    rec = (previous.reconstruction(a['heads']['rec'], x, hidden_a) +
           previous.reconstruction(b['heads']['rec'], x, hidden_b))
    rec = (w * rec).sum() / 2
    regular = balanced_regularization(a['raw'], b['raw'], batch['groups'])
    if metric:
        ma, count, passages = within_passage_loss(a['unit'], targets, batch['groups'], batch['owners'])
        mb, _, _ = within_passage_loss(b['unit'], targets, batch['groups'], batch['owners'])
        metric_loss = (ma + mb) / 2
    else:
        metric_loss = x.new_zeros(()); count = 0; passages = 0
    loss = metric_loss + errors['shape'] + .25 * (amplitude + errors['direction'] + errors['peaks'] + errors['lag']) + .1 * rec
    loss = loss + regular['invariance'] + regular['variance'] + .04 * regular['covariance']
    parts = dict(errors, amplitude=amplitude, rec=rec, metric=metric_loss, **regular, loss=loss)
    return loss, {k: float(v.detach()) for k, v in parts.items()}, dict(pairs=count, passages=passages)
