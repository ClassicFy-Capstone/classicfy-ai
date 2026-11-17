"""Frozen g supervision contracts; no learned quantities or evaluation fitting."""

import numpy as np

from features.style_profile import G_BLOCKS, style_quality_schema


STYLE_OBJECTIVES = ('legacy', 'confidence', 'adaptive', 'adaptive_confidence')


def triplet_confidence(profiles, signature):
    """Equal blocks of the weakest role's block quality on the fixed signature."""
    qualities = []
    for profile in profiles:
        if profile.get('quality_schema') != style_quality_schema():
            raise ValueError('Confidence training requires regenerated style-quality-v1 profiles')
        q = np.asarray(profile.get('g_quality'), dtype=float)
        mask = np.asarray(profile['g_mask'], dtype=bool)
        if (q.shape != (23,) or mask.shape != (23,) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1))
                or np.any(q[~mask] != 0)):
            raise ValueError('Invalid g quality: expected 23 finite values in [0,1], zero for missing features')
        qualities.append(q)
    if len(qualities) != 3 or not signature or any(not p['g_mask'][j] for p in profiles for j in signature):
        raise ValueError('Confidence requires three profiles observing the fixed signature')
    blocks = []
    for indices in G_BLOCKS.values():
        selected = [j for j in indices if j in signature]
        if selected:
            blocks.append(min(float(q[selected].mean()) for q in qualities))
    return float(np.mean(blocks))


def style_objective_contract(config):
    return {'version': 'g-triplet-objective-v1', 'mode': config.style_objective,
            'margin': config.margin, 'margin_scale': config.margin_scale,
            'margin_min': config.margin_min, 'margin_max': config.margin_max,
            'reduction': 'mean(weight * hinge); batch denominator',
            'quality': style_quality_schema() if 'confidence' in config.style_objective else None}
