"""Observed run durations and lagged Pearson correlations on masked beats.

Gaps break runs and lagged pairs. These descriptors do not align works or
establish musical repetition; lengths retain the original beat-grid units.
"""

from dataclasses import dataclass

import numpy as np

from .models import BeatSequence
from .temporal_shape import adjacent_changes


@dataclass(frozen=True)
class ObservedRun:
    state: str
    start: int
    stop: int
    left_censored: bool
    right_censored: bool

    @property
    def length(self) -> int:
        """Number of original array entries, with stop exclusive."""
        return self.stop - self.start


def observed_runs(
    sequence: BeatSequence, *, tolerance: float = 1e-6,
    labels: tuple[str, str, str] = ("negative", "zero", "positive"),
) -> tuple[ObservedRun, ...]:
    """Runs of negative/near-zero/positive entries within valid segments.

    A boundary touching a gap or sequence edge is censored: the run may
    extend beyond the observed segment. A state change is an observed boundary.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if len(labels) != 3 or len(set(labels)) != 3:
        raise ValueError("Three distinct state labels are required")
    values = np.asarray(sequence.values, dtype=float)
    valid = sequence.mask & np.isfinite(values)
    states = np.where(values > tolerance, 2, np.where(values < -tolerance, 0, 1))
    runs = []
    i = 0
    while i < len(values):
        if not valid[i]:
            i += 1
            continue
        start = i
        state = states[i]
        i += 1
        while i < len(values) and valid[i] and states[i] == state:
            i += 1
        runs.append(ObservedRun(labels[state], start, i,
                                start == 0 or not valid[start - 1],
                                i == len(values) or not valid[i]))
    return tuple(runs)


def summarize_runs(runs: tuple[ObservedRun, ...], states: tuple[str, ...]) -> dict[str, float | int]:
    """Run-weighted durations. No runs means undefined length, not length zero."""
    result = {}
    for state in states:
        selected = [run for run in runs if run.state == state]
        lengths = np.array([run.length for run in selected], dtype=float)
        result.update({f"{state}_run_count": len(selected),
                       f"{state}_run_entries": int(lengths.sum()),
                       f"{state}_left_censored_count": sum(r.left_censored for r in selected),
                       f"{state}_right_censored_count": sum(r.right_censored for r in selected)})
        result[f"{state}_run_mean"] = float(lengths.mean()) if len(lengths) else float("nan")
        result[f"{state}_run_median"] = float(np.median(lengths)) if len(lengths) else float("nan")
        result[f"{state}_run_p95"] = float(np.percentile(lengths, 95)) if len(lengths) else float("nan")
        result[f"{state}_run_max"] = int(lengths.max()) if len(lengths) else float("nan")
    return result


def summarize_temporal_duration(
    sequence: BeatSequence, *, direction_tolerance: float = 1e-6, level_tolerance: float = 1e-6,
) -> dict[str, float | int]:
    """Directional runs measure delta steps; level runs measure beat entries.

    For a normalized residual, positive/negative levels are relative to the
    work common pattern. They are not absolute loud/soft or fast/slow states.
    """
    changes = adjacent_changes(sequence)
    directions = ("fall", "flat", "rise")
    levels = ("negative", "zero", "positive")
    result = {f"direction_{k}": v for k, v in summarize_runs(
        observed_runs(changes, tolerance=direction_tolerance, labels=directions), directions).items()}
    result.update({f"level_{k}": v for k, v in summarize_runs(
        observed_runs(sequence, tolerance=level_tolerance, labels=levels), levels).items()})
    result.update(valid_beats=int(np.count_nonzero(sequence.mask & np.isfinite(sequence.values))),
                  valid_pairs=int(changes.mask.sum()), direction_tolerance=float(direction_tolerance),
                  level_tolerance=float(level_tolerance))
    return result


def lagged_correlation(
    sequence: BeatSequence, lag: int, *, min_pairs: int = 32, std_tolerance: float = 1e-12,
) -> dict[str, float | int | str]:
    """Pearson r of x[t],x[t+lag] pooled within fully valid contiguous segments.

    Each lag uses its own paired means and variances. Constant or under-supported
    pairs return NaN with an explicit status. No detrending/smoothing is applied.
    """
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError("lag must be a positive integer")
    if (isinstance(min_pairs, (bool, np.bool_)) or not isinstance(min_pairs, (int, np.integer))
            or min_pairs < 3):
        raise ValueError("min_pairs must be an integer of at least 3")
    if not np.isfinite(std_tolerance) or std_tolerance < 0:
        raise ValueError("std_tolerance must be finite and nonnegative")
    values = np.asarray(sequence.values, dtype=float)
    valid = sequence.mask & np.isfinite(values)
    lag = int(lag)
    possible = max(0, len(values) - lag)
    # A valid lag window contains no missing entry, including interior beats.
    missing = np.r_[0, np.cumsum(~valid, dtype=np.int64)]
    pairs = (missing[lag + 1:] - missing[:possible] == 0) if possible else np.empty(0, dtype=bool)
    count = int(pairs.sum())
    result = dict(lag=lag, possible_pairs=possible, valid_pairs=count,
                  pair_coverage=count / possible if possible else float("nan"),
                  min_pairs=int(min_pairs), std_tolerance=float(std_tolerance),
                  correlation=float("nan"), status="insufficient_pairs")
    if count < min_pairs:
        return result
    left, right = values[:possible][pairs], values[lag:][pairs]
    scale = max(float(np.abs(left).max()), float(np.abs(right).max()))
    if not scale:
        result["status"] = "constant_pairs"
        return result
    left, right = left / scale, right / scale
    left, right = left - left.mean(), right - right.mean()
    left_sd, right_sd = float(np.sqrt(np.mean(left * left))), float(np.sqrt(np.mean(right * right)))
    if min(left_sd, right_sd) * scale <= std_tolerance:
        result["status"] = "constant_pairs"
        return result
    result["correlation"] = float(np.clip(np.mean((left / left_sd) * (right / right_sd)), -1, 1))
    result["status"] = "ok"
    return result
