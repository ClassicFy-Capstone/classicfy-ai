"""Conservative, approximate ATEPP score/performance alignment.

ATEPP has work metadata, not ASAP beat annotations. Pitch-class DTW supplies
an initial map; unique same-pitch note matches refine it. Quality metrics are
mandatory. This is an automatic alignment, never a ground-truth annotation.
"""
from pathlib import Path
import unicodedata
import re

import numpy as np
from numba import njit

from .midi_loader import MidiData
from .note_alignment import NoteAlignment, ScoreNote, PerformedNote
from .score_midi import load_score_midi


def canonical_path(value):
    # The release ZIP contains UTF-8 names with the ZIP UTF-8 flag unset.
    try:
        value = value.encode('cp437').decode('utf-8')
    except (UnicodeError, LookupError):
        pass
    value = ''.join(c for c in unicodedata.normalize('NFKD', value) if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9/]', '', value.lower().replace('\\', '/'))


def _frames(pitches, onsets, ends, step, last):
    count = int(np.ceil(last / step)) + 1
    x = np.zeros((count, 12), np.float32)
    for pitch, onset, end in zip(pitches, onsets, ends):
        i = int(onset / step)
        x[max(0, i - 1):min(count, i + 2), int(pitch) % 12] += 1
        stop = min(count, int(min(end, onset + step * 4) / step) + 1)
        x[i:stop, int(pitch) % 12] += .2
    norm = np.linalg.norm(x, axis=1)
    x /= np.maximum(norm[:, None], 1e-8)
    return x


@njit(cache=True)
def _dtw(a, b, band=.25):
    n, m = len(a), len(b)
    parent = np.zeros((n, m), np.uint8)
    previous = np.full(m + 1, np.inf)
    previous[0] = 0.
    for i in range(n):
        current = np.full(m + 1, np.inf)
        center = i * (m - 1) / max(n - 1, 1)
        low = max(0, int(center - band * m) - 3)
        high = min(m, int(center + band * m) + 4)
        for j in range(low, high):
            diagonal, vertical, horizontal = previous[j], previous[j + 1] + .08, current[j] + .08
            best, direction = diagonal, 1
            if vertical < best:
                best, direction = vertical, 2
            if horizontal < best:
                best, direction = horizontal, 3
            dot = 0.
            for k in range(a.shape[1]):
                dot += a[i, k] * b[j, k]
            current[j + 1] = best + 1. - dot
            parent[i, j] = direction
        previous = current
    cost = previous[m]
    i, j = n - 1, m - 1
    path = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        direction = parent[i, j]
        if direction == 1:
            i, j = i - 1, j - 1
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            break
    return path, cost / max(len(path), 1)


def _note_pairs(score, midi, positions, times):
    predicted = np.interp([n.onset_beats for n in score], positions, times)
    beat_duration = np.median(np.diff(times) / np.diff(positions))
    tolerance = min(.55, max(.12, .5 * beat_duration))
    by_pitch = {}
    for j, note in enumerate(midi.notes):
        by_pitch.setdefault(note.pitch, []).append(j)
    onset_by_pitch = {pitch: np.array([midi.notes[j].start for j in indices])
                      for pitch, indices in by_pitch.items()}
    proposals = []
    for i, (note, when) in enumerate(zip(score, predicted)):
        indices = by_pitch.get(note.pitch, [])
        if not indices:
            continue
        onset = onset_by_pitch[note.pitch]
        at = np.searchsorted(onset, when)
        for k in range(max(0, at - 2), min(len(indices), at + 2)):
            error = abs(onset[k] - when)
            if error <= tolerance:
                proposals.append((error, i, indices[k]))
    used_s, used_p, pairs = set(), set(), []
    for _, i, j in sorted(proposals):
        if i not in used_s and j not in used_p:
            used_s.add(i)
            used_p.add(j)
            pairs.append((i, j))
    return sorted(pairs), beat_duration


def align_score_performance(score, midi: MidiData):
    from features.articulation import build_tempo_map
    if len(score) < 20 or len(midi.notes) < 20:
        raise ValueError('Too few notes')
    last = max(n.offset_beats for n in score)
    duration = max(n.end for n in midi.notes)
    # Keep the DTW lattice bounded for very long movements.
    s_step, p_step = max(.5, last / 5000), max(.20, duration / 5000)
    s = _frames([n.pitch for n in score], [n.onset_beats for n in score],
                [n.offset_beats for n in score], s_step, last)
    p = _frames([n.pitch for n in midi.notes], [n.start for n in midi.notes],
                [n.end for n in midi.notes], p_step, duration)
    path, cost = _dtw(s, p)
    mapping = {}
    for i, j in path:
        mapping.setdefault(i, []).append(j)
    positions = np.array(sorted(mapping)) * s_step
    times = np.array([np.median(mapping[i]) * p_step for i in sorted(mapping)])
    for _ in range(2):
        pairs, beat_duration = _note_pairs(score, midi, positions, times)
        alignment = NoteAlignment(tuple((score[i], PerformedNote(str(j), midi.notes[j].pitch,
                                 midi.notes[j].start, midi.notes[j].end, midi.notes[j].velocity))
                                 for i, j in pairs), (), ())
        positions, times = build_tempo_map(alignment)
        if len(positions) < 8:
            raise ValueError('Too few monotone alignment anchors')
    # Score quarter-note grid. Never extrapolate the tempo map.
    beats = np.arange(np.ceil(positions[0]), np.floor(positions[-1]) + 1.)
    perf_beats = np.interp(beats, positions, times)
    widths = np.diff(perf_beats)
    errors = np.array([abs(midi.notes[j].start - np.interp(score[i].onset_beats, positions, times))
                       / max(beat_duration, 1e-6) for i, j in pairs])
    unique_score_onsets = len(set(score[i].onset_beats for i, _ in pairs))
    metrics = dict(score_notes=len(score), performance_notes=len(midi.notes), matches=len(pairs),
                   score_recall=len(pairs) / len(score), performance_precision=len(pairs) / len(midi.notes),
                   span_coverage=(positions[-1] - positions[0]) / last,
                   monotone_anchor_ratio=len(positions) / max(unique_score_onsets, 1),
                   timing_p90_beats=float(np.percentile(errors, 90)) if len(errors) else 999.,
                   dtw_cost=float(cost), note_count_ratio=len(midi.notes) / len(score),
                   beat_intervals=len(widths),
                   extreme_beat_fraction=float(np.mean((widths < .08) | (widths > 5.))) if len(widths) else 1.)
    reasons = []
    for name, minimum in [('score_recall', .70), ('performance_precision', .60),
                          ('span_coverage', .90), ('monotone_anchor_ratio', .90)]:
        if metrics[name] < minimum:
            reasons.append(name)
    if not .65 <= metrics['note_count_ratio'] <= 1.6:
        reasons.append('note_count_ratio')
    if metrics['timing_p90_beats'] > .5:
        reasons.append('timing_p90_beats')
    if len(widths) < 16 or np.any(widths <= 0) or metrics['extreme_beat_fraction'] > .02:
        reasons.append('beat_grid')
    metrics['accepted'] = not reasons
    metrics['reject_reasons'] = ';'.join(reasons)
    return alignment, beats, perf_beats, metrics
