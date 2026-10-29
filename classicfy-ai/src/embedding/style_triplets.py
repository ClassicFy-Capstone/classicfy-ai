"""Anchor-conditional g distances and deterministic train-only triplet manifests."""

from collections import Counter, defaultdict
import heapq
import math

import numpy as np

from features.style_profile import (
    G_BLOCKS, StyleConfig, artifact, check_artifact, scale_profile, stable_id, unique_profiles,
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
    """Only O(N) profile indices are retained; candidate lists live for one anchor."""
    def __init__(self, profiles, signature):
        self.by_work, self.by_window = defaultdict(list), defaultdict(list)
        for p in profiles:
            if p['sampling_eligible'] and all(p['g_mask'][j] for j in signature):
                self.by_work[p['window']['work_id']].append(p)
                self.by_window[p['window']['same_window_key']].append(p)
        self.signature = signature

    def buckets(self, anchor):
        own = anchor['window']['work_id']
        buckets = {}
        for work, profiles in sorted(self.by_work.items()):
            pool = profiles if work != own else self.by_window.get(anchor['window']['same_window_key'], [])
            candidates = [p for p in pool if candidate_allowed(anchor, p, self.signature)]
            if candidates:
                buckets[work] = candidates
        return buckets


def _prepare(profiles, scaler, config, profiles_id):
    check_artifact(scaler, 'profile-scaler', config)
    if scaler['parents']['profiles'] != profiles_id:
        raise ValueError('Scaler was fit on a different profile artifact')
    profiles = unique_profiles(profiles, train_only=True)
    return [scale_profile(p, scaler) for p in profiles]


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
    for signature, anchors in sorted(signatures.items()):
        index = _CandidateIndex(profiles, signature)
        anchor_works = defaultdict(list)
        for anchor in anchors:
            if index.buckets(anchor):
                anchor_works[anchor['window']['work_id']].append(anchor)
        rng = _rng(config.seed, ['calibration', signature])
        heap, possible_pairs = [], 0
        # Exponential-race weighted reservoir: equivalent to sampling the specified
        # hierarchical pair distribution without replacement. Retain <= cap pairs;
        # enumerate keys in a bounded stream, calculate distances only after sampling.
        for work, work_anchors in sorted(anchor_works.items()):
            for anchor in work_anchors:
                buckets = index.buckets(anchor)
                for candidates in buckets.values():
                    weight = 1. / (len(anchor_works) * len(work_anchors) * len(buckets) * len(candidates))
                    for candidate in candidates:
                        possible_pairs += 1
                        priority = math.log(max(float(rng.random()), np.finfo(float).tiny)) / weight
                        entry = (priority, anchor['window']['window_id'], candidate['window']['window_id'],
                                 anchor, candidate)
                        if len(heap) < config.max_calibration_pairs:
                            heapq.heappush(heap, entry)
                        elif priority > heap[0][0]:
                            heapq.heapreplace(heap, entry)
        pairs = []
        for _, aid, bid, anchor, candidate in sorted(heap, key=lambda item: (item[1], item[2])):
            distance, _ = profile_distance(anchor, candidate, signature)
            pairs.append({'anchor_window_id': aid, 'candidate_window_id': bid, 'distance': distance})
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
    return artifact('distance-calibration', config, parents={'profiles': profiles_id, 'scaler': scaler['id']},
                    fit_split='train', seed=config.seed, signatures=results,
                    status='valid' if any(r['status'] == 'valid' for r in results.values()) else
                           'inactive_core' if not scaler['metric_ready'] else 'no_valid_signature',
                    pair_sampling='hierarchical work-uniform exponential reservoir without replacement',
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
        buckets = indexes[signature].buckets(anchor)
        candidates = [p for pool in buckets.values() for p in pool]
        outcome.update(candidate_count=len(candidates), candidate_work_counts={k: len(v) for k, v in buckets.items()})
        if len(candidates) < config.min_candidates:
            skip('candidates_insufficient')
            continue
        if not any(p['window']['work_id'] != wa['work_id'] for p in candidates):
            skip('cross_work_candidate_missing')
            continue
        distances = [(p, *profile_distance(anchor, p, signature)) for p in candidates]
        near, far = np.quantile([d for _, d, _ in distances],
                                [config.near_quantile, config.far_quantile], method='linear')
        near, far = min(float(near), calibrated['global_near']), max(float(far), calibrated['global_far'])
        positives = [(p, d, blocks) for p, d, blocks in distances if p['window']['work_id'] != wa['work_id'] and d <= near]
        negatives = [(p, d, blocks) for p, d, blocks in distances if p['window']['same_window_key'] == wa['same_window_key']
                     and p['window']['work_id'] == wa['work_id'] and d >= far]
        if not positives:
            skip('positive_missing')
            continue
        if not negatives:
            skip('same_window_far_negative_missing')
            continue
        gap = calibrated['delta_g']
        feasible = [(p, d, blocks) for p, d, blocks in positives
                    if any(dn - d >= gap and p['window']['recording_id'] != n['window']['recording_id']
                           and p['window']['performance_id'] != n['window']['performance_id'] for n, dn, _ in negatives)]
        if not feasible:
            skip('gap_insufficient')
            continue
        rng = _rng(config.seed, ['triplet', wa['performance_id'], wa['window_id']])
        positive, dap, bp = feasible[int(rng.integers(len(feasible)))]
        allowed = [(n, dn, blocks) for n, dn, blocks in negatives if dn - dap >= gap
                   and positive['window']['recording_id'] != n['window']['recording_id']
                   and positive['window']['performance_id'] != n['window']['performance_id']]
        negative, dan, bn = allowed[int(rng.integers(len(allowed)))]
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
