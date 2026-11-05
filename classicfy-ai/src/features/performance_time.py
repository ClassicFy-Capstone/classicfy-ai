"""Strict beat-scale quarter/second reference maps, independent of note deviations."""

from dataclasses import dataclass, field, replace

import numpy as np


# Existing research-alignment quality bounds, in seconds per quarter note.
# These flag uncertain estimated intervals; they are not universal tempo limits.
ESTIMATED_BEAT_MIN_TAU = .08
ESTIMATED_BEAT_MAX_TAU = 5.


@dataclass(frozen=True)
class BeatTimeMap:
    q: np.ndarray
    seconds: np.ndarray
    valid: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        q, times = (np.array(v, dtype=np.float64, copy=True) for v in (self.q, self.seconds))
        unavailable = (q.shape == (0,) and times.shape == (0,) and self.provenance.get('status') == 'unavailable')
        if (not unavailable and (q.ndim != 1 or len(q) < 2 or times.shape != q.shape
                or not np.isfinite(q).all() or not np.isfinite(times).all()
                or np.any(np.diff(q) <= 0) or np.any(np.diff(times) <= 0) or np.any(times < 0))):
            raise ValueError('Beat anchors must be finite, strictly increasing in quarter and seconds')
        interval_count = max(0, len(q) - 1)
        if self.valid is not None and np.asarray(self.valid).dtype != bool:
            raise ValueError('Beat validity must be boolean')
        valid = np.ones(interval_count, bool) if self.valid is None else np.array(self.valid, dtype=bool, copy=True)
        if valid.shape != (interval_count,):
            raise ValueError('Beat validity must match anchor intervals')
        if self.provenance.get('granularity') != 'beat' or not self.provenance.get('source'):
            raise ValueError('Time map requires explicit beat-scale source provenance')
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            tau = np.diff(times) / np.diff(q)
            log_qpm = np.log2(60.) - np.log2(tau)
        if not np.isfinite(tau).all() or np.any(tau <= 0) or not np.isfinite(log_qpm).all():
            raise ValueError('Beat slopes exceed valid floating point range')
        tau.setflags(write=False)
        log_qpm.setflags(write=False)
        object.__setattr__(self, '_tau', tau)
        object.__setattr__(self, '_log2_qpm', log_qpm)
        for name, value in (('q', q), ('seconds', times), ('valid', valid)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def unavailable(cls, source, reason):
        if not reason:
            raise ValueError('An unavailable time map needs a reason')
        return cls(np.array([]), np.array([]), np.array([], bool),
                   {'granularity': 'beat', 'source': source, 'status': 'unavailable', 'reason': reason})

    @property
    def tau(self):
        return self._tau

    @property
    def log2_qpm(self):
        return self._log2_qpm

    def beat_at(self, q):
        if not len(self.q) or not np.isfinite(q) or q < self.q[0] or q > self.q[-1]:
            return -1
        return min(int(np.searchsorted(self.q, q, side='right') - 1), len(self.valid) - 1)

    def time_at(self, q):
        index = self.beat_at(q)
        if index < 0:
            return None
        at = int(np.searchsorted(self.q, q, side='left'))
        if at < len(self.q) and self.q[at] == q:
            if (at > 0 and self.valid[at - 1]) or (at < len(self.valid) and self.valid[at]):
                return float(self.seconds[at])
        if not self.valid[index]:
            return None
        return float(self.seconds[index] + (q - self.q[index]) * self.tau[index])

    def expected_duration(self, onset_q, duration_q):
        if not np.isfinite(duration_q) or duration_q <= 0:
            return None
        end = onset_q + duration_q
        left = self.beat_at(onset_q)
        right = int(np.searchsorted(self.q, end, side='left') - 1)
        if left < 0 or end > self.q[-1] or right < left or right >= len(self.valid) or not self.valid[left:right + 1].all():
            return None
        start_time, stop_time = self.time_at(onset_q), self.time_at(end)
        if start_time is None or stop_time is None or stop_time <= start_time:
            return None
        return stop_time - start_time


def mask_estimated_beat_intervals(beats: BeatTimeMap) -> BeatTimeMap:
    """Exclude implausible estimated slopes without changing anchors or prior masks.

    Whole-alignment acceptance can tolerate a few extreme intervals. Those
    intervals must still be unknown targets, rather than observed tempo values.
    Annotated reference maps are deliberately outside this policy.
    """
    if beats.provenance.get('estimated') is not True:
        raise ValueError('Interval bounds require an explicitly estimated beat map')
    outside = (beats.tau < ESTIMATED_BEAT_MIN_TAU) | (beats.tau > ESTIMATED_BEAT_MAX_TAU)
    provenance = {**beats.provenance, 'interval_validation': {
        'policy': 'estimated-beat-tau-bounds-v1',
        'min_seconds_per_quarter': ESTIMATED_BEAT_MIN_TAU,
        'max_seconds_per_quarter': ESTIMATED_BEAT_MAX_TAU,
        'out_of_range_indices': np.flatnonzero(outside).tolist(),
        'source_invalid_indices': np.flatnonzero(~beats.valid).tolist(),
    }}
    return replace(beats, valid=beats.valid & ~outside, provenance=provenance)
