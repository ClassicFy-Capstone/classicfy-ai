"""Remove the global summary block while preserving the frozen distance rule."""

import numpy as np

from .retrieval_probe import distance, fit_scale, weighted_quantile, weights


ARMS = {
    'B': ('B',), 'B+T': ('B', 'T'), 'B+F': ('B', 'F'),
    'B+P': ('B', 'P'), 'B+H': ('B', 'H'), 'B+P+H': ('B', 'P', 'H'),
    'T': ('T',), 'F': ('F',), 'P': ('P',), 'H': ('H',), 'P+H': ('P', 'H'),
}


def standalone_distances(descriptors, metadata, gallery_indices):
    """Same training-only scales and equal squared block weighting as before.

    P/H retain their context-conditioned means; standalone means no additional
    14D global B block, not that all level information has been removed.
    """
    ids = np.asarray(gallery_indices, int)
    w = weights([metadata[i] for i in ids])
    family = np.array([metadata[i]['family_id'] for i in ids])
    a, b = np.triu_indices(len(ids), 1)
    different = family[a] != family[b]
    a, b = a[different], b[different]
    if not len(a):
        raise ValueError('Need distinct training families')
    blocks, scales, medians = {}, {}, {}
    for name in ('B', 'T', 'F', 'P', 'H'):
        x = np.asarray(descriptors[name], float)
        if x.ndim != 2 or len(x) != len(metadata) or not np.isfinite(x).all():
            raise ValueError('Unsupported frozen descriptor: ' + name)
        scales[name] = None if name == 'B' else fit_scale(x[ids], w)
        d = distance(x, x[ids], scales[name], group_width=2 if name == 'B' else None)
        median = float(weighted_quantile(d[ids[a], b], w[a]*w[b], .5))
        if median <= 1e-12:
            raise ValueError('No supported training distance: ' + name)
        medians[name] = median
        blocks[name] = d / median
    arms = {name: np.sqrt(np.mean([blocks[c]**2 for c in components], axis=0))
            for name, components in ARMS.items()}
    return arms, dict(scales=scales, block_medians=medians)
