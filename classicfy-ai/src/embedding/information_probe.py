"""Weighted linear readout of fixed performance representations."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


def family_weights(rows):
    """Equal families, equal works within family, equal recordings within work."""
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        groups[row['family_id']][row['piece']].append(i)
    if not groups:
        raise ValueError('No rows')
    weights = np.zeros(len(rows))
    for works in groups.values():
        for indices in works.values():
            weights[indices] = 1 / (len(groups) * len(works) * len(indices))
    return weights


def metadata_matrix(training, rows):
    """Vocabulary is fitted on training only; unknown categories are all-zero."""
    def values(row):
        return (row['composer'], row['work_type'], row['composer'] + '::' + row['work_type'])
    vocabulary = [sorted({values(row)[j] for row in training}) for j in range(3)]
    blocks, names, unknown = [], [], []
    for j, labels in enumerate(vocabulary):
        lookup = {label: i for i, label in enumerate(labels)}
        block = np.zeros((len(rows), len(labels)))
        count = 0
        for i, row in enumerate(rows):
            value = values(row)[j]
            if value in lookup:
                block[i, lookup[value]] = 1
            else:
                count += 1
        blocks.append(block)
        names.extend(f'{("composer", "type", "composer_type")[j]}={label}' for label in labels)
        unknown.append(count)
    return np.concatenate(blocks, axis=1), names, unknown


@dataclass
class RidgeReadout:
    mean: np.ndarray
    scale: np.ndarray
    active: np.ndarray
    target_mean: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_rhs: np.ndarray

    @classmethod
    def prepare(cls, x, y, weights):
        x, y, weights = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
        if y.ndim == 1:
            y = y[:, None]
        if (x.ndim != 2 or len(x) != len(y) or weights.shape != (len(x),)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1)):
            raise ValueError('Invalid weighted training inputs')
        mean = weights @ x
        scale = np.sqrt(weights @ ((x - mean) ** 2))
        active = scale > 1e-12
        z = (x[:, active] - mean[active]) / scale[active]
        target_mean = weights @ y
        gram = z.T @ (weights[:, None] * z)
        eigenvalues, eigenvectors = eigh(gram, check_finite=False)
        eigenvalues = np.maximum(eigenvalues, 0)
        rhs = z.T @ (weights[:, None] * (y - target_mean))
        return cls(mean, scale, active, target_mean, eigenvalues, eigenvectors, eigenvectors.T @ rhs)

    def coefficients(self, alpha):
        alpha = np.asarray(alpha, float)
        if alpha.ndim == 0:
            alpha = np.repeat(alpha, len(self.target_mean))
        if alpha.shape != self.target_mean.shape or np.any(alpha <= 0):
            raise ValueError('Expected positive alpha per output')
        return self.eigenvectors @ (self.projected_rhs / (self.eigenvalues[:, None] + alpha[None, :]))

    def predict(self, x, alpha):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Invalid prediction inputs')
        z = (x[:, self.active] - self.mean[self.active]) / self.scale[self.active]
        return z @ self.coefficients(alpha) + self.target_mean


def choose_alphas(scores, alphas, tolerance=1e-12):
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores[:, None]
    if not np.isfinite(scores).all():
        raise ValueError('Invalid CV scores')
    return np.array([max(a for a, loss in zip(alphas, scores[:, j]) if loss <= scores[:, j].min() + tolerance)
                     for j in range(scores.shape[1])])


def work_errors(rows, truth, prediction):
    """Within-work centering is scoring only, never inference preprocessing."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['piece']].append(i)
    result = []
    for piece, ids in sorted(groups.items()):
        a, b = truth[ids], prediction[ids]
        result.append(dict(piece=piece, family_id=rows[ids[0]]['family_id'],
            recordings=len(ids), global_mse=float(np.mean((a - b) ** 2)),
            within_mse=float(np.mean(((a - a.mean()) - (b - b.mean())) ** 2))))
    return result


def macro_error(works, field):
    groups = defaultdict(list)
    for row in works:
        groups[row['family_id']].append(row[field])
    return float(np.mean([np.mean(values) for values in groups.values()]))


def paired_bootstrap(reference, augmented, field, *, repetitions=2000, seed=20261016):
    """Resample the SAME family indices on both sides of each comparison."""
    def family_values(rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row['family_id']].append(row[field])
        return {key: float(np.mean(values)) for key, values in groups.items()}
    a, b = family_values(reference), family_values(augmented)
    if set(a) != set(b):
        raise ValueError('Bootstrap families differ')
    keys = sorted(a)
    a, b = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    indices = np.random.default_rng(seed).integers(0, len(keys), (repetitions, len(keys)))
    before, after = a[indices].mean(axis=1), b[indices].mean(axis=1)
    decrease = before - after
    reduction = np.divide(100 * decrease, before, out=np.full_like(before, np.nan), where=before > 1e-12)
    return decrease, reduction


def shuffle_within_work(rows, values, seed):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['evaluation_role'], row['piece'], row['grid']].append(i)
    rng = np.random.default_rng(seed)
    source_indices = np.arange(len(rows))
    for _, ids in sorted(groups.items()):
        source_indices[ids] = rng.permutation(ids)
    return values[source_indices].copy(), source_indices
