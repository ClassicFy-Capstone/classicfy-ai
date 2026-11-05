"""Anchor-conditional g distances and deterministic train-only triplet manifests."""

from collections import Counter, defaultdict
import heapq
import math
import logging

import numpy as np

from features.style_profile import (
    G_BLOCKS, StyleConfig, artifact, check_artifact, scale_profile, scale_profiles, stable_id, unique_profiles,
)


def anchor_signature(profile, scaler):
    if not scaler['metric_ready'] or not profile['sampling_eligible'] or not all(profile['g_mask'][:9]):
        return ()
    return tuple(j for j, enabled in enumerate(scaler['trained_feature_mask'])
                 if enabled and profile['g_mask'][j])


def signature_key(signature):
    return ','.join(map(str, signature))


def profile_distance(a, b, signature):
    """J is supplied by the anchor, never recomputed using candidate intersections."""
    if not signature or len(set(signature)) != len(signature) or any(j < 0 or j >= 23 for j in signature):
        raise ValueError('Invalid distance signature')
    if any(not p['g_scaled_mask'][j] for p in (a, b) for j in signature):
        raise ValueError('Candidate does not observe the fixed anchor signature')
    diffs = np.asarray(a['g_scaled']) - np.asarray(b['g_scaled'])
    blocks = {}
    for name, indices in G_BLOCKS.items():
        selected = [j for j in indices if j in signature]
        if selected:
            blocks[name] = float(np.mean(diffs[selected] ** 2))
    distance = float(np.mean(list(blocks.values())))
    if not math.isfinite(distance):
        raise ValueError('Profile distance overflow')
    return distance, blocks


def candidate_allowed(a, b, signature):
    wa, wb = a['window'], b['window']
    return (wa['split'] == wb['split'] == 'train' and a['sampling_eligible'] and b['sampling_eligible']
            and not wa['tail'] and not wb['tail']
            and wa['performance_id'] != wb['performance_id'] and wa['recording_id'] != wb['recording_id']
            and all(b['g_mask'][j] for j in signature)
            and (wa['work_id'] != wb['work_id']
                 or (wa['score_version_id'] == wb['score_version_id']
                     and wa['repeat_route_id'] == wb['repeat_route_id']
                     and wa['same_window_key'] == wb['same_window_key'])))


class _CandidateIndex:
    """O(N) typed indices and one distance vector, never an NxN matrix."""
    def __init__(self, profiles, signature):
        self.profiles = [p for p in profiles if p['sampling_eligible'] and all(p['g_mask'][j] for j in signature)]
        self.signature = signature
        self.positions = {p['window']['window_id']: i for i, p in enumerate(self.profiles)}
        self.labels = {}
        for field in ('work_id', 'same_window_key', 'score_version_id', 'repeat_route_id', 'performance_id', 'recording_id'):
            values = [p['window'][field] for p in self.profiles]
            codes = {v: i for i, v in enumerate(sorted(set(values)))}
            self.labels[field] = np.array([codes[v] for v in values], dtype=np.int64)
        self.groups = {}
        for field in ('work_id', 'same_window_key', 'performance_id', 'recording_id'):
            groups = defaultdict(list)
            for i, code in enumerate(self.labels[field]):
                groups[int(code)].append(i)
            self.groups[field] = {k: np.array(v, dtype=np.int64) for k, v in groups.items()}
        self.work_names = {int(self.labels['work_id'][i]): p['window']['work_id'] for i, p in enumerate(self.profiles)}
        self.matrix = np.array([p['g_scaled'] for p in self.profiles], dtype=float).reshape(-1, 23)
        self.block_columns = [[j for j in ids if j in signature] for ids in G_BLOCKS.values()]
        self.block_columns = [ids for ids in self.block_columns if ids]

    def indices(self, anchor):
        at = self.positions[anchor['window']['window_id']]
        labels = self.labels
        same = labels['work_id'] == labels['work_id'][at]
        allowed = (~same | ((labels['same_window_key'] == labels['same_window_key'][at])
                            & (labels['score_version_id'] == labels['score_version_id'][at])
                            & (labels['repeat_route_id'] == labels['repeat_route_id'][at])))
        allowed &= labels['performance_id'] != labels['performance_id'][at]
        allowed &= labels['recording_id'] != labels['recording_id'][at]
        return np.flatnonzero(allowed)

    def count(self, anchor):
        at = self.positions[anchor['window']['window_id']]
        labels = self.labels
        own = labels['work_id'][at]
        same = self.groups['same_window_key'][int(labels['same_window_key'][at])]
        own_valid = ((labels['work_id'][same] == own)
                     & (labels['score_version_id'][same] == labels['score_version_id'][at])
                     & (labels['repeat_route_id'][same] == labels['repeat_route_id'][at]))
        excluded = np.union1d(self.groups['performance_id'][int(labels['performance_id'][at])],
                              self.groups['recording_id'][int(labels['recording_id'][at])])
        removed = ((labels['work_id'][excluded] != own)
                   | ((labels['same_window_key'][excluded] == labels['same_window_key'][at])
                      & (labels['score_version_id'][excluded] == labels['score_version_id'][at])
                      & (labels['repeat_route_id'][excluded] == labels['repeat_route_id'][at])))
        return len(self.profiles) - len(self.groups['work_id'][int(own)]) + int(own_valid.sum()) - int(removed.sum())

    def buckets(self, anchor):
        allowed = np.zeros(len(self.profiles), dtype=bool)
        allowed[self.indices(anchor)] = True
        result = {}
        for work, positions in self.groups['work_id'].items():
            candidates = positions[allowed[positions]]
            if len(candidates):
                result[work] = candidates
        # Preserve the original sorted-work random choice order.
        return dict(sorted(result.items()))

    def draw_context(self, anchor):
        """Candidate work counts without materializing every candidate bucket."""
        at = self.positions[anchor['window']['window_id']]
        labels = self.labels
        own = int(labels['work_id'][at])
        excluded = np.union1d(self.groups['performance_id'][int(labels['performance_id'][at])],
                              self.groups['recording_id'][int(labels['recording_id'][at])])
        removed = np.bincount(labels['work_id'][excluded], minlength=len(self.groups['work_id']))
        available = np.array([len(self.groups['work_id'][k]) for k in range(len(removed))]) - removed
        same = self.groups['same_window_key'][int(labels['same_window_key'][at])]
        valid = ((labels['work_id'][same] == own)
                 & (labels['score_version_id'][same] == labels['score_version_id'][at])
                 & (labels['repeat_route_id'][same] == labels['repeat_route_id'][at])
                 & (labels['performance_id'][same] != labels['performance_id'][at])
                 & (labels['recording_id'][same] != labels['recording_id'][at]))
        own_pool = same[valid]
        available[own] = len(own_pool)
        return np.flatnonzero(available), own, own_pool, removed

    def draw_pool(self, anchor, context, work):
        _, own, own_pool, removed = context
        if work == own:
            return own_pool
        pool = self.groups['work_id'][int(work)]
        if removed[work] == 0:
            return pool
        at = self.positions[anchor['window']['window_id']]
        labels = self.labels
        return pool[(labels['performance_id'][pool] != labels['performance_id'][at])
                    & (labels['recording_id'][pool] != labels['recording_id'][at])]

    def distances(self, anchor, indices):
        delta = self.matrix[indices] - np.asarray(anchor['g_scaled'])
        # Match the scalar reduction order exactly at inclusive near/far cutoffs.
        blocks = [np.mean(np.ascontiguousarray(delta[:, ids] ** 2), axis=1) for ids in self.block_columns]
        return np.mean(np.ascontiguousarray(np.stack(blocks, axis=1)), axis=1)


def _prepare(profiles, scaler, config, profiles_id):
    check_artifact(scaler, 'profile-scaler', config)
    if scaler['parents']['profiles'] != profiles_id:
        raise ValueError('Scaler was fit on a different profile artifact')
    profiles = unique_profiles(profiles, train_only=True)
    return scale_profiles(profiles, scaler)


def _rng(seed, identity):
    return np.random.default_rng(int(stable_id([seed, identity])[:16], 16))


def calibrate_distances(profiles, scaler, config=StyleConfig(), *, profiles_id):
    profiles = _prepare(profiles, scaler, config, profiles_id)
    signatures = defaultdict(list)
    for p in profiles:
        signature = anchor_signature(p, scaler)
        if signature:
            signatures[signature].append(p)
    results = {}
    for si, (signature, anchors) in enumerate(sorted(signatures.items()), 1):
        logging.getLogger(__name__).info('signature %d/%d anchors=%d', si, len(signatures), len(anchors))
        index = _CandidateIndex(profiles, signature)
        anchor_works = defaultdict(list)
        possible_pairs = 0
        for anchor in anchors:
            n = index.count(anchor)
            if n:
                anchor_works[anchor['window']['work_id']].append(anchor)
                possible_pairs += n
        rng = _rng(config.seed, ['calibration', signature])
        selected = set()
        cap = config.max_calibration_pairs
        if possible_pairs <= cap:
            for work_anchors in anchor_works.values():
                for anchor in work_anchors:
                    at = index.positions[anchor['window']['window_id']]
                    selected.update((at, int(i)) for i in index.indices(anchor))
        else:
            # Draw exactly the prescribed hierarchical distribution, rejecting
            # already-seen ordered pairs. Memory and distance computations are capped.
            works = sorted(anchor_works)
            cache = {}
            attempts = 0
            while len(selected) < cap and attempts < max(10000, cap * 100):
                attempts += 1
                work = works[int(rng.integers(len(works)))]
                pool = anchor_works[work]
                anchor = pool[int(rng.integers(len(pool)))]
                at = index.positions[anchor['window']['window_id']]
                if at not in cache:
                    if len(cache) >= 32:
                        cache.pop(next(iter(cache)))
                    cache[at] = index.draw_context(anchor)
                context = cache[at]
                work = context[0][int(rng.integers(len(context[0])))]
                candidates = index.draw_pool(anchor, context, work)
                selected.add((at, int(candidates[int(rng.integers(len(candidates)))])))
            if len(selected) < cap:
                # Rare, highly imbalanced near-exhaustion fallback: exact weighted
                # reservoir on remaining keys, with the same pair probabilities.
                heap = []
                remaining = cap - len(selected)
                for work_anchors in anchor_works.values():
                    for anchor in work_anchors:
                        at = index.positions[anchor['window']['window_id']]
                        buckets = index.buckets(anchor)
                        for candidates in buckets.values():
                            weight = 1. / (len(works) * len(work_anchors) * len(buckets) * len(candidates))
                            for candidate in candidates:
                                pair = (at, int(candidate))
                                if pair in selected:
                                    continue
                                priority = math.log(max(float(rng.random()), np.finfo(float).tiny)) / weight
                                entry = (priority, *pair)
                                if len(heap) < remaining:
                                    heapq.heappush(heap, entry)
                                elif priority > heap[0][0]:
                                    heapq.heapreplace(heap, entry)
                selected.update((a, b) for _, a, b in heap)
        pairs = []
        ordered = sorted(selected, key=lambda pair: (index.profiles[pair[0]]['window']['window_id'],
                                                      index.profiles[pair[1]]['window']['window_id']))
        for ai, bi in ordered:
            anchor, candidate = index.profiles[ai], index.profiles[bi]
            distance, _ = profile_distance(anchor, candidate, signature)
            pairs.append({'anchor_window_id': anchor['window']['window_id'],
                          'candidate_window_id': candidate['window']['window_id'], 'distance': distance})
        distances = [p['distance'] for p in pairs]
        result = {'J_a': list(signature), 'pair_count': len(pairs), 'possible_pair_count': possible_pairs,
                  'pairs': pairs, 'status': 'insufficient_pairs', 'global_near': 0., 'global_far': 0.,
                  'distance_scale': 0., 'delta_g': 0., 'scale_method': 'unavailable'}
        if len(pairs) >= config.min_calibration_pairs:
            q25, q75 = np.quantile(distances, [.25, .75], method='linear')
            # Configurable near/far ranks are kept consistent in local/global cutoffs.
            near, far = np.quantile(distances, [config.near_quantile, config.far_quantile], method='linear')
            spread, method = float(q75 - q25), 'iqr'
            if spread <= config.epsilon:
                spread, method = float(np.std(distances, ddof=0)), 'std_fallback'
            result.update(global_near=float(near), global_far=float(far), distance_scale=spread,
                          delta_g=config.gap_fraction * spread, scale_method=method,
                          status='valid' if spread > config.epsilon else 'degenerate_metric')
        results[signature_key(signature)] = result
        logging.getLogger(__name__).info('signature %d/%d pairs=%d status=%s', si, len(signatures), len(pairs), result['status'])
    return artifact('distance-calibration', config, parents={'profiles': profiles_id, 'scaler': scaler['id']},
                    fit_split='train', seed=config.seed, signatures=results,
                    status='valid' if any(r['status'] == 'valid' for r in results.values()) else
                           'inactive_core' if not scaler['metric_ready'] else 'no_valid_signature',
                    pair_sampling='hierarchical work-uniform unique draws; exact remaining-key reservoir fallback',
                    quantile_method='numpy linear')


def _triplet_valid(a, p, n, signature, near, far, gap):
    if not candidate_allowed(a, p, signature) or not candidate_allowed(a, n, signature):
        return False
    wa, wp, wn = (x['window'] for x in (a, p, n))
    if wa['work_id'] == wp['work_id'] or wa['work_id'] != wn['work_id']:
        return False
    if wp['recording_id'] == wn['recording_id'] or wp['performance_id'] == wn['performance_id']:
        return False
    if wa['same_window_key'] != wn['same_window_key']:
        return False
    dap, _ = profile_distance(a, p, signature)
    dan, _ = profile_distance(a, n, signature)
    return dap <= near and dan >= far and dan - dap >= gap


def validate_triplet(a, p, n, signature, *, near, far, gap):
    if not _triplet_valid(a, p, n, signature, near, far, gap):
        raise ValueError('Triplet violates work/window/recording/cutoff/gap constraints')


def build_triplets(profiles, scaler, calibration, config=StyleConfig(), *, profiles_id):
    profiles = _prepare(profiles, scaler, config, profiles_id)
    check_artifact(calibration, 'distance-calibration', config)
    if calibration['parents'] != {'profiles': profiles_id, 'scaler': scaler['id']}:
        raise ValueError('Calibration/profile/scaler dependency mismatch')
    indexes, triplets, outcomes = {}, [], []
    for anchor in profiles:
        wa = anchor['window']
        signature = anchor_signature(anchor, scaler)
        outcome = {'anchor_window_id': wa['window_id'], 'work_id': wa['work_id'],
                   'reason': None, 'candidate_count': 0, 'candidate_work_counts': {}}
        def skip(reason):
            outcome['reason'] = reason
            outcomes.append(outcome)
        if not anchor['sampling_eligible']:
            skip('reference_insufficient' if 'reference_insufficient' in anchor['ineligible_reason_codes'] else 'core_missing')
            continue
        if not signature:
            skip('inactive_core')
            continue
        calibrated = calibration['signatures'].get(signature_key(signature))
        if not calibrated or calibrated['status'] != 'valid':
            skip('calibration_insufficient' if not calibrated else calibrated['status'])
            continue
        if signature not in indexes:
            indexes[signature] = _CandidateIndex(profiles, signature)
        index = indexes[signature]
        indices = index.indices(anchor)
        at = index.positions[wa['window_id']]
        candidate_works = index.labels['work_id'][indices]
        work_codes, work_counts = np.unique(candidate_works, return_counts=True)
        names = index.work_names
        outcome.update(candidate_count=len(indices),
                       candidate_work_counts={names[int(k)]: int(v) for k, v in zip(work_codes, work_counts)})
        if len(indices) < config.min_candidates:
            skip('candidates_insufficient')
            continue
        cross_work = candidate_works != index.labels['work_id'][at]
        if not cross_work.any():
            skip('cross_work_candidate_missing')
            continue
        distances = index.distances(anchor, indices)
        near, far = np.quantile(distances, [config.near_quantile, config.far_quantile], method='linear')
        near, far = min(float(near), calibrated['global_near']), max(float(far), calibrated['global_far'])
        pmask = cross_work & (distances <= near)
        nmask = ~cross_work & (distances >= far)
        pi, ni = indices[pmask], indices[nmask]
        pd, nd = distances[pmask], distances[nmask]
        if not len(pi):
            skip('positive_missing')
            continue
        if not len(ni):
            skip('same_window_far_negative_missing')
            continue
        gap = calibrated['delta_g']
        potential = nd.max() - pd >= gap
        feasible = np.zeros(len(pi), dtype=bool)
        labels = index.labels
        # Highest negatives only accelerate the existence test. Final p/n choices
        # remain uniform over exactly the same feasible sets and in the same order.
        for npos in np.argsort(-nd, kind='stable'):
            if not potential.any() or feasible[potential].all():
                break
            n = ni[npos]
            feasible |= (potential & (nd[npos] - pd >= gap)
                         & (labels['recording_id'][pi] != labels['recording_id'][n])
                         & (labels['performance_id'][pi] != labels['performance_id'][n]))
        feasible_positions = np.flatnonzero(feasible)
        if not len(feasible_positions):
            skip('gap_insufficient')
            continue
        rng = _rng(config.seed, ['triplet', wa['performance_id'], wa['window_id']])
        ppos = feasible_positions[int(rng.integers(len(feasible_positions)))]
        p = pi[ppos]
        positive, dap = index.profiles[int(p)], float(pd[ppos])
        allowed = np.flatnonzero((nd - dap >= gap)
                                 & (labels['recording_id'][ni] != labels['recording_id'][p])
                                 & (labels['performance_id'][ni] != labels['performance_id'][p]))
        npos = allowed[int(rng.integers(len(allowed)))]
        negative, dan = index.profiles[int(ni[npos])], float(nd[npos])
        _, bp = profile_distance(anchor, positive, signature)
        _, bn = profile_distance(anchor, negative, signature)
        validate_triplet(anchor, positive, negative, signature, near=near, far=far, gap=gap)
        triplet = {'anchor_window_id': wa['window_id'], 'positive_window_id': positive['window']['window_id'],
                   'negative_window_id': negative['window']['window_id'],
                   'anchor': wa, 'positive': positive['window'], 'negative': negative['window'],
                   'D_ap': dap, 'D_an': dan, 'distance_gap': dan - dap, 'tau_near': near, 'tau_far': far,
                   'delta_g': gap, 'J_a': list(signature), 'active_blocks': list(bp),
                   'positive_block_distances': bp, 'negative_block_distances': bn,
                   'profile_ids': [p['profile_id'] for p in (anchor, positive, negative)],
                   'reference_id': anchor['reference_id'], 'scaler_id': scaler['id'],
                   'calibration_id': calibration['id'], 'seed': config.seed, 'split': 'train',
                   'coverage': [p['per_channel_coverage'] for p in (anchor, positive, negative)],
                   'reference_counts': [p['reference_counts'] for p in (anchor, positive, negative)]}
        triplet['triplet_id'] = stable_id(triplet)
        triplets.append(triplet)
        outcome['reason'] = 'selected'
        outcomes.append(outcome)
    return artifact('triplets', config, parents={'profiles': profiles_id, 'scaler': scaler['id'], 'calibration': calibration['id']},
                    split='train', seed=config.seed, triplets=triplets, anchor_outcomes=outcomes,
                    reason_counts=dict(Counter(o['reason'] for o in outcomes)))
