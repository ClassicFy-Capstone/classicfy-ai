"""Fixed cross-work retrieval, disjoint-block descriptors and numeric judges."""

from collections import defaultdict
import hashlib

import numpy as np
from scipy.spatial.distance import cdist

EPS = 1e-12
CHANNELS = ('tempo', 'rubato', 'dynamics', 'articulation', 'pedal_depth',
            'pedal_down_ratio', 'pedal_changes')
LAGS = (-4, -2, -1, 0, 1, 2, 4)
PAIRS = tuple((a, b) for a in range(7) for b in range(a + 1, 7)
              if (a, b) != (0, 1) and not (a >= 4 and b >= 4))


def weights(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        groups[r['family_id']][r['piece'], r['grid']].append(i)
    result = np.zeros(len(rows))
    for works in groups.values():
        for ids in works.values():
            result[ids] = 1 / (len(groups) * len(works) * len(ids))
    return result


def weighted_quantile(x, w, q):
    order = np.argsort(x, kind='stable')
    return np.asarray(x)[order][np.searchsorted(np.cumsum(np.asarray(w)[order]), q * np.sum(w), side='left')]


def finite_pairs(x, valid, lag):
    width = abs(lag); n = len(valid) - width
    if n <= 0:
        return np.array([], float), np.array([], float)
    missing = np.r_[0, np.cumsum(~valid)]
    good = missing[width + 1:] == missing[:n]
    a, b = (0, width) if lag >= 0 else (width, 0)
    return x[a:a+n][good], x[b:b+n][good]


def pearson(x, y, minimum=32):
    if len(x) < minimum or min(np.std(x), np.std(y)) <= EPS:
        return np.nan
    return float(np.clip(np.corrcoef(x, y)[0, 1], -1, 1))


def run_lengths(x, valid, positive):
    state = (x > 1e-6) if positive else (x < -1e-6)
    good = valid & state
    edges = np.diff(np.r_[False, good, False].astype(int))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def summary(x, valid):
    v = x[:, valid]
    if v.shape[1] < 2:
        return np.full(14, np.nan)
    location = v.mean(axis=1); location[1] = np.median(abs(v[1]))
    return np.column_stack((location, np.ptp(np.percentile(v, [5, 95], axis=1), axis=0))).ravel()


def hand_descriptors(x, valid, thresholds, single_keep):
    singles, changes = [], []
    pair_mask = valid[:-1] & valid[1:]
    for c in range(7):
        v = x[c, valid]; delta = np.diff(x[c]); observed = delta[pair_mask]
        changes.append(delta)
        width = np.ptp(np.percentile(v, [5, 95])) if len(v) else np.nan
        moving = pair_mask[:-1] & pair_mask[1:] & (abs(delta[:-1]) > 1e-6) & (abs(delta[1:]) > 1e-6)
        runs = [run_lengths(x[c], valid, positive) for positive in (True, False)]
        row = [np.mean(abs(observed) > thresholds[c]) if len(observed) else np.nan,
               np.sqrt(np.mean(observed**2)) / width if len(observed) and width > EPS else np.nan,
               np.mean(observed > 1e-6) if len(observed) else np.nan,
               np.mean(np.sign(delta[:-1][moving]) != np.sign(delta[1:][moving])) if moving.any() else np.nan]
        row += [np.log1p(r.mean()) if len(r) >= 20 else np.nan for r in runs]
        for value, mask in ((x[c], valid), (delta, pair_mask)):
            for lag in (1, 8):
                left, right = finite_pairs(value, mask, lag)
                row.append(pearson(left, right))
        singles.extend(row)
    relations = []
    # Original #14 CSV sorts channel *names*, not channel numeric indices.
    for a, b in sorted(PAIRS, key=lambda pair: (CHANNELS[pair[0]], CHANNELS[pair[1]])):
        # combinations in #14 follow channel order; sorting preserves orientation.
        for lag in LAGS:
            width = abs(lag); n = len(pair_mask) - width
            missing = np.r_[0, np.cumsum(~pair_mask)]
            good = missing[width+1:] == missing[:n] if n > 0 else np.array([], bool)
            ia, ib = (0, width) if lag >= 0 else (width, 0)
            relations.append(pearson(changes[a][ia:ia+n][good], changes[b][ib:ib+n][good]))
    return np.asarray(singles)[single_keep], np.asarray(relations)


def partition(x, valid, parity):
    selected = np.zeros(len(valid), bool); blocks = []
    for i, start in enumerate(range(0, len(valid) - 63, 64)):
        m = valid[start:start+64]
        good = int((m[:-1] & m[1:]).sum()) >= 51
        if i % 2 == parity and good:
            selected[start:start+64] = m; blocks.append(start)
    return selected, blocks


def judges(x, valid, blocks):
    level, temporal = [], []
    for c in range(7):
        v = x[c, valid]
        if len(v) < 100:
            level += [np.nan, np.nan]
        else:
            level += [np.median(abs(v)) if c == 1 else np.median(v), np.ptp(np.percentile(v, [25, 75]))]
        rms = []
        for start in blocks:
            for offset in (0, 32):
                a = start + offset; m = valid[a:a+32]; p = m[:-1] & m[1:]
                if p.sum() >= 25:
                    rms.append(np.sqrt(np.mean(np.diff(x[c, a:a+32])[p]**2)))
        temporal.extend(np.log1p(np.percentile(rms, [25, 50, 75])) if len(rms) >= 3 else [np.nan]*3)
    rs = []
    for start in blocks:
        m = valid[start:start+64]; p = m[:-1] & m[1:]
        r = pearson(np.diff(x[0, start:start+64])[p], np.diff(x[2, start:start+64])[p], 51)
        if np.isfinite(r):
            rs.append(r)
    relation = np.r_[np.percentile(rs, [25, 50, 75]), np.std(rs)] if len(rs) >= 2 else np.full(4, np.nan)
    return {'E-L': np.asarray(level), 'E-T': np.asarray(temporal), 'E-R': relation}


def fit_scale(x, w):
    mean = w @ x; scale = np.sqrt(w @ ((x - mean)**2)); active = scale > EPS
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'active': active.tolist()}


def distance(x, y, scale=None, group_width=None):
    if scale is not None:
        active = np.asarray(scale['active']); s = np.asarray(scale['scale'])
        if not active.any():
            raise ValueError('No nonconstant training coordinates')
    else:
        active = np.ones(x.shape[1], bool); s = np.ones(x.shape[1])
    if group_width is None:
        return np.sqrt(np.maximum(cdist(x[:, active]/s[active], y[:, active]/s[active], 'sqeuclidean') / active.sum(), 0))
    sums = []
    for channels in ([0], [1], [2], [3], [4, 5, 6]):
        ids = np.array([c*group_width+j for c in channels for j in range(group_width)])
        ids = ids[active[ids]]
        if len(ids):
            sums.append(cdist(x[:, ids]/s[ids], y[:, ids]/s[ids], 'sqeuclidean') / len(ids))
    if not sums:
        raise ValueError('No supported groups')
    return np.sqrt(np.maximum(np.mean(sums, axis=0), 0))


def priorities(seed, query_key, count):
    derived = int.from_bytes(hashlib.sha256(f'{seed}/{query_key}'.encode()).digest()[:8], 'little')
    return np.random.default_rng(derived).random(count)


def ranked(distances, families, priority, k, cap=True):
    eligible = np.flatnonzero(np.isfinite(distances)); order = eligible[np.argsort(distances[eligible], kind='stable')]
    tiers, i = [], 0
    while i < len(order):
        end = i + 1
        while end < len(order) and distances[order[end]] - distances[order[i]] <= EPS:
            end += 1
        tiers.extend(order[i:end][np.argsort(priority[order[i:end]], kind='stable')]); i = end
    result, seen = [], set()
    for index in tiers:
        if cap and families[index] in seen:
            continue
        result.append(index); seen.add(families[index])
        if len(result) == k:
            return np.asarray(result, int)
    return None


def overlap(a, b, families):
    intersection = set(a) & set(b)
    ranks_a, ranks_b = {v:i+1 for i,v in enumerate(a)}, {v:i+1 for i,v in enumerate(b)}
    return {'recording_overlap': len(intersection)/len(a),
            'family_overlap': len(set(families[a]) & set(families[b]))/len(a),
            'rank_overlap': sum(min(1/ranks_a[v], 1/ranks_b[v]) for v in intersection)/sum(1/np.arange(1,len(a)+1))}


def percentile_scores(d, w):
    result = np.empty(d.shape)
    for i, values in enumerate(d):
        result[i] = ((values[None, :] > values[:, None] + EPS) @ w
                     + .5 * (np.abs(values[None, :] - values[:, None]) <= EPS) @ w)
    return result
