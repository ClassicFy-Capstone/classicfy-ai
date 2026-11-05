"""Convert ASAP score-second beat anchors to quarter-note boundaries.

Only boundary times are interpolated; MIDI features must be re-extracted from
events. Repeated/special anchors invalidate every target bin they intersect.
"""
from functools import lru_cache

import mido
import numpy as np


@lru_cache(maxsize=None)
def score_tempo_map(path):
    midi = mido.MidiFile(path)
    tick = 0
    seconds = 0.0
    tempo = 500000
    knots = [(0.0, 0.0, tempo)]
    for message in mido.merge_tracks(midi.tracks):
        seconds += message.time * tempo / 1e6 / midi.ticks_per_beat
        tick += message.time
        if message.type == "set_tempo":
            tempo = message.tempo
            knot = (seconds, tick / midi.ticks_per_beat, tempo)
            if np.isclose(seconds, knots[-1][0], atol=1e-12, rtol=0):
                knots[-1] = knot
            else:
                knots.append(knot)
    return np.asarray(knots, float)


def seconds_to_quarters(seconds, tempo_map):
    seconds = np.asarray(seconds, float)
    index = np.searchsorted(tempo_map[:, 0], seconds, side="right") - 1
    if np.any(index < 0) or not np.isfinite(seconds).all():
        raise ValueError("Invalid score seconds")
    return (tempo_map[index, 1]
            + (seconds - tempo_map[index, 0]) * 1e6 / tempo_map[index, 2])


def quarter_boundaries(score_quarters, performance_seconds, beat_types):
    q = np.asarray(score_quarters, float)
    t = np.asarray(performance_seconds, float)
    # Annotation seconds can be rounded. Snap only micro-scale errors near an
    # integer quarter boundary; do not quantize arbitrary score positions.
    q = np.where(np.abs(q - np.round(q)) <= 2e-5, np.round(q), q)
    if (q.ndim != 1 or q.shape != t.shape or len(q) < 2
            or len(beat_types) != len(q) or not np.isfinite(q).all()
            or not np.isfinite(t).all() or np.any(np.diff(q) <= 1e-10)
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Quarter and performance anchors must increase strictly")
    first, last = int(np.ceil(q[0] - 1e-8)), int(np.floor(q[-1] + 1e-8))
    if last <= first:
        raise ValueError("No full quarter-note interval within annotation coverage")
    grid = np.arange(first, last + 1, dtype=float)
    times = np.interp(grid, q, t)
    exact = np.array([np.any(np.isclose(x, q, atol=1e-8, rtol=0)) for x in grid])
    types = np.asarray(beat_types)
    bad = (types[:-1] == "bR") | (types[1:] == "bR")
    valid = np.diff(times) > 0
    for a, b in zip(q[:-1][bad], q[1:][bad]):
        valid &= ~((grid[:-1] < b - 1e-8) & (grid[1:] > a + 1e-8))
    return grid, times, valid, exact
