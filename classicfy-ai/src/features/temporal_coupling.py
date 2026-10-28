"""Cross-channel temporal descriptors on a shared original beat grid.

These operate on documented normalized residuals, not audio or causal effects.
Circular delta shifts are a descriptive phase control, not a significance test.
"""

from itertools import product

import numpy as np

from .models import BeatSequence


def contiguous_segments(valid):
    """Return half-open original-index segments; never bridge missing entries."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 1:
        raise ValueError("Validity must be one-dimensional")
    edges = np.diff(np.r_[False, valid, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _settings(min_pairs, std_tolerance):
    if (isinstance(min_pairs, (bool, np.bool_))
            or not isinstance(min_pairs, (int, np.integer)) or min_pairs < 3):
        raise ValueError("min_pairs must be an integer of at least 3")
    if not np.isfinite(std_tolerance) or std_tolerance < 0:
        raise ValueError("std_tolerance must be finite and nonnegative")


def pearson(left, right, *, min_pairs=32, std_tolerance=1e-12):
    """Finite paired Pearson; undefined variance/support are explicit statuses."""
    _settings(min_pairs, std_tolerance)
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not np.isfinite([left, right]).all():
        raise ValueError("Expected equally sized finite one-dimensional paired arrays")
    result = dict(valid_pairs=len(left), correlation=float("nan"), status="insufficient_pairs")
    if len(left) < min_pairs:
        return result
    # Separate rescaling preserves Pearson while preventing avoidable overflow.
    scales = [float(np.abs(v).max()) for v in (left, right)]
    if min(scales) == 0:
        result["status"] = "constant_pairs"
        return result
    x, y = [v / s for v, s in zip((left, right), scales)]
    x, y = x - x.mean(), y - y.mean()
    sx, sy = float(np.sqrt(np.mean(x * x))), float(np.sqrt(np.mean(y * y)))
    if min(sx * scales[0], sy * scales[1]) <= std_tolerance:
        result["status"] = "constant_pairs"
        return result
    result.update(correlation=float(np.clip(np.mean((x / sx) * (y / sy)), -1, 1)), status="ok")
    return result


def cross_channel_correlation(left, right, lag=0, *, min_pairs=32, std_tolerance=1e-12):
    """corr(left[t], right[t+lag]); positive lag means the left channel leads.

    BOTH channels must be valid throughout the intervening original interval.
    For delta inputs this also requires all underlying beat values to be valid.
    """
    _settings(min_pairs, std_tolerance)
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)):
        raise ValueError("lag must be a signed integer")
    if len(left) != len(right):
        raise ValueError("Channels must share the same original beat grid")
    valid = left.mask & right.mask & np.isfinite(left.values) & np.isfinite(right.values)
    width = abs(int(lag))
    possible = max(0, len(left) - width)
    missing = np.r_[0, np.cumsum(~valid, dtype=np.int64)]
    pairs = missing[width + 1:] - missing[:possible] == 0 if possible else np.empty(0, bool)
    a, b = (0, width) if lag >= 0 else (width, 0)
    result = pearson(left.values[a:a + possible][pairs], right.values[b:b + possible][pairs],
                     min_pairs=min_pairs, std_tolerance=std_tolerance)
    return dict(lag=int(lag), possible_pairs=possible,
                pair_coverage=result['valid_pairs'] / possible if possible else float('nan'), **result)


def joint_directions(sequences, *, tolerance=1e-6, min_pairs=32):
    """Joint fall/flat/rise probabilities for two or three delta channels.

    TV from the product of marginals detects dependence; triple dependence can
    be entirely explained by pairwise links. The centered sign third moment is
    also recorded for triples, without claiming general irreducible interaction.
    """
    _settings(min_pairs, 0)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if len(sequences) not in (2, 3) or len({len(s) for s in sequences}) != 1:
        raise ValueError("Expected two or three channels on a common grid")
    values = np.array([s.values for s in sequences])
    valid = np.logical_and.reduce([s.mask & np.isfinite(s.values) for s in sequences])
    signs = np.where(values[:, valid] > tolerance, 1,
                     np.where(values[:, valid] < -tolerance, -1, 0))
    n = signs.shape[1]
    labels = ('fall', 'flat', 'rise')
    result = dict(valid_pairs=n, status='ok' if n >= min_pairs else 'insufficient_pairs',
                  product_marginal_tv=float('nan'), centered_third_moment=float('nan'))
    marginals = np.array([[np.mean(s == k) for k in (-1, 0, 1)] for s in signs]) if n else None
    discrepancy = []
    for codes in product(range(3), repeat=len(sequences)):
        name = '_'.join(labels[k] for k in codes)
        count = int(np.count_nonzero(np.all(signs == np.array(codes)[:, None] - 1, axis=0)))
        prob = count / n if n >= min_pairs else float('nan')
        expected = float(np.prod([marginals[c, k] for c, k in enumerate(codes)])) if n >= min_pairs else float('nan')
        result.update({f'{name}_count': count, f'{name}_probability': prob,
                       f'{name}_independent_probability': expected})
        discrepancy.append(abs(prob - expected))
    if n >= min_pairs:
        result['product_marginal_tv'] = .5 * sum(discrepancy)
        if len(sequences) == 3:
            result['centered_third_moment'] = float(np.mean(np.prod(signs - signs.mean(axis=1, keepdims=True), axis=0)))
    return result


def phase_shift_control(left, right, *, samples=199, seed=20261012,
                        min_segment=64, min_shift=8, min_pairs=32):
    """Shift RIGHT deltas within long joint-valid segments; never across gaps.

    Exact delta distributions and circular autocorrelations are preserved per
    segment; finite-boundary linear autocorrelations may change at one seam.
    The observed correlation is recomputed on the SAME long-segment support.
    A periodic or long-range signal may remain coupled after shifts. No p-value.
    """
    _settings(min_pairs, 0)
    for name, value, minimum in (('samples', samples, 1), ('min_shift', min_shift, 1),
                                 ('min_segment', min_segment, 3)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if min_segment < 2 * min_shift:
        raise ValueError("min_segment must accommodate both shift margins")
    if len(left) != len(right):
        raise ValueError("Channels must share the same original grid")
    valid = left.mask & right.mask & np.isfinite(left.values) & np.isfinite(right.values)
    segments = [(a, b) for a, b in contiguous_segments(valid) if b - a >= min_segment]
    x = np.concatenate([left.values[a:b] for a, b in segments]) if segments else np.empty(0)
    y = np.concatenate([right.values[a:b] for a, b in segments]) if segments else np.empty(0)
    observed = pearson(x, y, min_pairs=min_pairs)
    result = dict(control_pairs=len(x), control_segments=len(segments),
                  control_coverage=len(x) / int(valid.sum()) if valid.any() else float('nan'),
                  observed_correlation=observed['correlation'], status=observed['status'],
                  shift_abs_mean=float('nan'), shift_abs_p95=float('nan'),
                  abs_excess_over_shift_p95=float('nan'))
    if observed['status'] != 'ok':
        return result
    rng = np.random.default_rng(seed)
    shifted = np.concatenate([right.values[a + (np.arange(b - a)[None, :] -
                              rng.integers(min_shift, b - a - min_shift + 1, size=samples)[:, None]) % (b - a)]
                              for a, b in segments], axis=1)
    # All shifts preserve each marginal, so common centering and norms suffice.
    xs, ys = float(np.abs(x).max()), float(np.abs(y).max())
    xc = x / xs - (x / xs).mean()
    yc = shifted / ys - (y / ys).mean()
    denom = np.sqrt(np.mean(xc * xc) * np.var(y / ys))
    absolute = np.abs(np.clip(np.mean(yc * xc, axis=1) / denom, -1, 1))
    result.update(shift_abs_mean=float(absolute.mean()), shift_abs_p95=float(np.percentile(absolute, 95)),
                  abs_excess_over_shift_p95=abs(observed['correlation']) - float(np.percentile(absolute, 95)))
    return result
