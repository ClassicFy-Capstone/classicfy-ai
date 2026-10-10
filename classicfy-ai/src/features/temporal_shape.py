"""Location-independent shape statistics on original, masked beat indices.

Inputs should share a documented normalization. Threshold fitting is separate
from measurement so a performance never defines its own 'large' changes.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .models import BeatSequence


@dataclass(frozen=True)
class ChangeThreshold:
    value: float
    quantile: float
    pair_count: int


def adjacent_changes(sequence: BeatSequence) -> BeatSequence:
    """delta[t] = x[t+1]-x[t]; missing/nonfinite beats break adjacency."""
    values = np.asarray(sequence.values, dtype=float)
    valid = sequence.mask & np.isfinite(values)
    pairs = valid[:-1] & valid[1:]
    delta = np.full(max(0, len(values) - 1), np.nan)
    with np.errstate(over="ignore"):
        delta[pairs] = values[1:][pairs] - values[:-1][pairs]
    if not np.isfinite(delta[pairs]).all():
        raise ValueError("Adjacent subtraction overflowed")
    return BeatSequence(delta, pairs)


def fit_change_threshold(
    reference: Iterable[BeatSequence], *, quantile: float = .95,
) -> ChangeThreshold:
    """Pool valid |delta| from the reference only; long sequences weigh more.

    Zero threshold is retained for degenerate references; no implicit fallback.
    Caller must select reference works before fitting (e.g. train split only).
    """
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be finite and within [0, 1]")
    parts = []
    for sequence in reference:
        changes = adjacent_changes(sequence)
        parts.append(np.abs(changes.values[changes.mask]))
    values = np.concatenate(parts) if parts else np.empty(0)
    if not len(values):
        raise ValueError("Reference contains no valid adjacent pairs")
    return ChangeThreshold(float(np.quantile(values, quantile)), quantile, len(values))


def summarize_temporal_shape(
    sequence: BeatSequence, *, large_threshold: float, direction_tolerance: float = 1e-6,
) -> dict[str, float | int]:
    """Delta quantiles, large-change ratio, directions and adjacent transitions.

    Rise: delta > tolerance; fall: delta < -tolerance; otherwise flat.
    Large: |delta| > threshold (strict). Transition ratios use valid triplets.
    A flat step interrupts a reversal; gaps are never joined. Undefined metrics
    are NaN, with counts provided to distinguish missing data from zero changes.
    """
    if not np.isfinite(large_threshold) or large_threshold < 0:
        raise ValueError("large_threshold must be finite and nonnegative")
    if not np.isfinite(direction_tolerance) or direction_tolerance < 0:
        raise ValueError("direction_tolerance must be finite and nonnegative")
    changes = adjacent_changes(sequence)
    delta = changes.values[changes.mask]
    absolute = np.abs(delta)
    count = len(delta)
    possible = len(changes)
    rows = dict(valid_beats=int(np.count_nonzero(sequence.mask & np.isfinite(sequence.values))),
                possible_pairs=possible, valid_pairs=count,
                pair_coverage=count / possible if possible else float("nan"),
                large_threshold=float(large_threshold), direction_tolerance=float(direction_tolerance))
    for prefix, values in (("delta", delta), ("abs_delta", absolute)):
        quantiles = np.percentile(values, [5, 25, 50, 75, 95]) if count else [float("nan")] * 5
        rows.update({f"{prefix}_p{q}": float(v) for q, v in zip((5, 25, 50, 75, 95), quantiles)})
    rows["mean_abs_delta"] = float(absolute.mean()) if count else float("nan")
    # Scaling first prevents unnecessary overflow when squaring large deltas.
    scale = float(absolute.max()) if count else 0.
    rows["rms_delta"] = scale * float(np.sqrt(np.mean((delta / scale) ** 2))) if scale else (0. if count else float("nan"))
    rows["large_count"] = int(np.count_nonzero(absolute > large_threshold))
    rows["large_ratio"] = rows["large_count"] / count if count else float("nan")
    direction = np.zeros(possible, dtype=np.int8)
    direction[changes.values > direction_tolerance] = 1
    direction[changes.values < -direction_tolerance] = -1
    labels = (("fall", -1), ("flat", 0), ("rise", 1))
    for name, code in labels:
        n = int(np.count_nonzero(changes.mask & (direction == code)))
        rows[f"{name}_count"] = n
        rows[f"{name}_ratio"] = n / count if count else float("nan")
    triplets = changes.mask[:-1] & changes.mask[1:]
    n_triplets = int(triplets.sum())
    rows["valid_triplets"] = n_triplets
    for left, lcode in labels:
        for right, rcode in labels:
            n = int(np.count_nonzero(triplets & (direction[:-1] == lcode) & (direction[1:] == rcode)))
            rows[f"{left}_to_{right}_count"] = n
            rows[f"{left}_to_{right}_ratio"] = n / n_triplets if n_triplets else float("nan")
    moving = triplets & (direction[:-1] != 0) & (direction[1:] != 0)
    rows["moving_triplets"] = int(moving.sum())
    rows["reversal_count"] = int(np.count_nonzero(moving & (direction[:-1] != direction[1:])))
    rows["reversal_ratio"] = rows["reversal_count"] / rows["moving_triplets"] if rows["moving_triplets"] else float("nan")
    return rows
