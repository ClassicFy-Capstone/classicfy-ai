"""Score-conditioned residual responses on original, uncompressed beat indices."""

from dataclasses import dataclass
from fractions import Fraction
import hashlib

import numpy as np


@dataclass(frozen=True)
class PhraseHypothesis:
    phrase_id: str
    part_id: str
    start: Fraction
    stop: Fraction
    start_evidence: tuple[str, ...]
    stop_evidence: tuple[str, ...]
    status: str = 'provisional_score_segmentation'


def phrase_hypotheses(score, candidates, minimum=Fraction(4), maximum=Fraction(32)):
    """Conservative score-only scaffold, explicitly not phrase ground truth.

    Whole-part rests/fermatas and >=4-quarter same-scope slur arrivals are
    boundary proposals. Short proposals are skipped; long gaps are capped at
    32 quarters with an explicit fixed-length fallback. No performance enters.
    """
    if minimum <= 0 or maximum < minimum:
        raise ValueError('Invalid phrase length limits')
    pid = score.part_names[0][0]
    measures = [m for m in score.measures if m.part_id == pid]
    start = min(m.onset_quarters for m in measures)
    stop = max(m.onset_quarters + m.duration_quarters for m in measures)
    proposals = {start: {'score_start'}, stop: {'score_end'}}
    for candidate in candidates.candidates:
        for cue in candidate.cues:
            if cue.kind == 'global_notated_silence' or (cue.part_id == pid and cue.kind == 'notation_fermata'):
                q = cue.position_quarters
                if start < q < stop:
                    proposals.setdefault(q, set()).add(cue.kind)
    for slur in candidates.slurs:
        if slur.part_id == pid and slur.status == 'paired' and slur.stop_onset_quarters - slur.start_quarters >= minimum:
            q = slur.stop_onset_quarters
            if start < q < stop:
                proposals.setdefault(q, set()).add('long_same_scope_slur_arrival')
    chosen = [(start, proposals[start])]
    for q in sorted(proposals):
        if q == start:
            continue
        while q - chosen[-1][0] > maximum:
            chosen.append((chosen[-1][0] + maximum, {'fixed_length_cap'}))
        if q - chosen[-1][0] >= minimum or q == stop:
            chosen.append((q, proposals[q]))
    result = []
    for (a, left), (b, right) in zip(chosen, chosen[1:]):
        if b > a:
            identifier = hashlib.sha256(f'{pid}/{a}/{b}'.encode()).hexdigest()[:12]
            result.append(PhraseHypothesis(identifier, pid, a, b, tuple(sorted(left)), tuple(sorted(right))))
    return tuple(result)


def phase_labels(written, phrases, shift=0.0):
    """Assign interval midpoints; never snap a boundary to a beat edge.

    Sensitivity shifts only internal boundaries, keeping the score endpoints.
    Undefined or inverted intervals remain -1. Returns bin and phrase identity.
    """
    labels = np.full(len(written), -1, int)
    owners = np.full(len(written), -1, int)
    for i, item in enumerate(written):
        if item is None or item[1] <= item[0]:
            continue
        midpoint = .5 * sum(item)
        for j, phrase in enumerate(phrases):
            a = float(phrase.start) + (shift if j else 0)
            b = float(phrase.stop) + (shift if j + 1 < len(phrases) else 0)
            if a <= midpoint < b:
                labels[i] = min(2, int(3 * (midpoint - a) / (b - a)))
                owners[i] = j
                break
    return labels, owners


def response(x, mask, pair_mask, labels, owners=None, minimum_beats=8, minimum_pairs=4):
    """Three equal-width context bins: level, mean/RMS delta, joint delta.

    Pair i means i-1 -> i; cut at missing values, visit jumps, context-bin and
    phrase/block identity boundaries. NaN denotes unsupported coordinates.
    Joint change is E[deltaTempo * deltaDynamics], not a correlation.
    """
    x = np.asarray(x, float)
    mask = np.asarray(mask, bool)
    pair_mask = np.asarray(pair_mask, bool)
    labels = np.asarray(labels, int)
    n = len(mask)
    if x.shape != (7, n) or pair_mask.shape != (n,) or labels.shape != (n,):
        raise ValueError('Expected seven channels and aligned original-index arrays')
    if not np.isfinite(x[:, mask]).all():
        raise ValueError('Valid features must be finite')
    if minimum_beats < 1 or minimum_pairs < 1:
        raise ValueError('Support minima must be positive')
    pairs = pair_mask.copy()
    if n:
        pairs[0] = False
        pairs[1:] &= mask[:-1] & mask[1:] & (labels[:-1] == labels[1:])
    if owners is not None:
        owners = np.asarray(owners)
        if owners.shape != (n,):
            raise ValueError('Owner length differs')
        pairs[1:] &= owners[:-1] == owners[1:]
    delta = np.zeros_like(x)
    delta[:, 1:] = np.diff(x)
    coordinates, support = [], []
    for bin_id in range(3):
        valid = mask & (labels == bin_id)
        good = pairs & (labels == bin_id)
        nb, npairs = int(valid.sum()), int(good.sum())
        level = x[:, valid].mean(axis=1) if nb >= minimum_beats else np.full(7, np.nan)
        if npairs >= minimum_pairs:
            d = delta[:, good]
            change = np.r_[d.mean(axis=1), np.sqrt((d*d).mean(axis=1)), np.mean(d[0]*d[2])]
        else:
            change = np.full(15, np.nan)
        coordinates.extend(np.r_[level, change])
        support.append(dict(bin=bin_id, beats=nb, pairs=npairs))
    return np.asarray(coordinates), support


def split_view(mask, pair_mask, parity, block_width=64):
    """Disjoint alternating original-index blocks; no cross-block differences."""
    if parity not in (0, 1) or block_width < 2:
        raise ValueError('Invalid view definition')
    valid = np.zeros(len(mask), bool)
    pairs = np.zeros(len(mask), bool)
    blocks = []
    for block, start in enumerate(range(0, len(mask)-block_width+1, block_width)):
        if block % 2 == parity:
            valid[start:start+block_width] = mask[start:start+block_width]
            pairs[start+1:start+block_width] = pair_mask[start+1:start+block_width]
            blocks.append(start)
    return valid, pairs, blocks


def fixed_labels(n, width=16):
    index = np.arange(n)
    return np.minimum(2, (index % width) * 3 // width), index // width
