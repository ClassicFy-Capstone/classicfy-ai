"""Selected-clip, reference-relative G supervision and protected era heads.

This is a separate contract from eight-bar G artifacts. References use matching
XML occurrences within a work/split, deduplicate source MIDI, and may include
the target (as in the existing G contract). Held-out cohort references define
evaluation targets only; feature scales and distance calibration fit train only.
"""
from collections import defaultdict
import csv
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from features.style_profile import G_BLOCKS, G_NAMES, weighted_quantile
from .selected_style_coordinates import G_PROFILE_VERSION,QUARTER_TICKS


ERA_COMPOSERS = {
    'baroque': ['Johann Sebastian Bach'],
    'classical_transition': ['Franz Joseph Haydn', 'Ludwig van Beethoven', 'Wolfgang Amadeus Mozart'],
    'romantic_impressionist': ['Alexander Scriabin', 'Edvard Grieg', 'Franz Liszt', 'Franz Schubert',
                 'Frédéric Chopin', 'Johannes Brahms', 'Modest Mussorgsky',
                 'Pyotr Ilyich Tchaikovsky', 'Robert Schumann', 'Sergei Rachmaninoff',
                 'Claude Debussy', 'Maurice Ravel'],
}


def era_metadata(corpus, selected_csv):
    with open(selected_csv, encoding='utf-8-sig') as stream:
        rows = {r['perf_id']: r for r in csv.DictReader(stream)}
    lookup = {c: era for era, composers in ERA_COMPOSERS.items() for c in composers}
    result = []
    for clip in corpus.manifest['clips']:
        row = rows[clip['perf_id']]
        if row['source_sha256'] != clip['source_sha256']:
            raise ValueError('Selected source identity mismatch')
        result.append(dict(perf_id=clip['perf_id'], composer=row['composer'],
                           era=lookup[row['composer']], work=clip['composition_id'],
                           source=clip['source_sha256'], split=clip['split']))
    return result


def selected_profiles(corpus,tempo_cells):
    """23D summaries over every selected clip; G-ineligible clips still train AE.

    Tempo deduplicates local beat IDs but references absolute XML interval cells.
    Other channels use onset-balanced note weights. Require >=3 reference
    recordings/cell, >=80% coverage and >=8 independent beats/onsets.
    """
    clips, arrays = corpus.manifest['clips'], corpus.arrays
    if tempo_cells.shape!=(len(arrays['R']),2):raise ValueError('Absolute tempo interval coordinates are required')
    raw = np.zeros((len(clips), 23), np.float32)
    quality = np.zeros_like(raw)
    known = np.zeros_like(raw, dtype=bool)
    cohorts = defaultdict(list)
    for i, c in enumerate(clips):
        cohorts[(c['split'], c['composition_id'], c['source_xml_sha256'], c['traversal_id'])].append(i)
    for members in cohorts.values():
        data = {}
        for ci in members:
            c = clips[ci]; section = slice(c['start'], c['stop'])
            data[ci] = {key: np.asarray(arrays[key][section]) for key in
                        ('R', 'M', 'original_score_rows', 'note_to_onset', 'beat_id', 'note_q')}
            data[ci]['tempo_cells']=np.asarray(tempo_cells[section])
        distinct, seen = [], set()
        for ci in members:
            source = clips[ci]['source_sha256']
            if source not in seen: distinct.append(ci); seen.add(source)
        for k in range(11):
            values, keys, groups, weights, positions, ends = {}, {}, {}, {}, {}, {}
            for ci, d in data.items():
                if k == 0:
                    # Include unknown beats in coverage; do not bridge missing beats.
                    ids = np.flatnonzero(d['beat_id'] >= 0)
                    # A tied/unmatched first note can be masked while another
                    # attack in that same beat has a valid copied tempo value.
                    ids=ids[np.lexsort((~d['M'][ids,0],d['beat_id'][ids]))]
                    _, first = np.unique(d['beat_id'][ids], return_index=True)
                    ids = ids[first]
                    coordinates=np.ascontiguousarray(d['tempo_cells'][ids],dtype=np.int64)
                    key=coordinates.view(np.dtype([('start','<i8'),('end','<i8')])).reshape(-1)
                    w=(coordinates[:,1]-coordinates[:,0])/QUARTER_TICKS
                    if np.any(w<=0):raise ValueError('Nonpositive absolute tempo reference cell')
                    group=d['beat_id'][ids]
                    positions[ci]=coordinates[:,0]/QUARTER_TICKS
                    ends[ci]=coordinates[:,1]
                else:
                    ids = np.arange(len(d['R']))
                    key = d['original_score_rows']
                    group = d['note_to_onset']
                    _, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
                    w = 1. / counts[inverse]
                keys[ci], groups[ci], weights[ci] = key, group, w
                values[ci] = np.where(d['M'][ids, k], d['R'][ids, k], np.nan)
            union = np.unique(np.concatenate([keys[ci] for ci in members]))
            if not len(union): continue
            table = np.full((len(distinct), len(union)), np.nan)
            for row, ci in enumerate(distinct):
                table[row, np.searchsorted(union, keys[ci])] = values[ci]
            count = np.isfinite(table).sum(0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                reference = np.nanmedian(table, axis=0)
            for ci in members:
                at = np.searchsorted(union, keys[ci]); n = count[at]
                delta = values[ci] - reference[at]
                valid = np.isfinite(delta) & (n >= 3)
                w = weights[ci]
                coverage = w[valid].sum() / max(w.sum(), 1e-12)
                if coverage < .8 or len(np.unique(groups[ci][valid])) < 8: continue
                median, lo, hi = weighted_quantile(delta[valid], w[valid], [.5, .1, .9])
                indices = [0, 1] if k == 0 else [2*k+1, 2*k+2]
                raw[ci, indices] = median, hi-lo
                known[ci, indices] = True
                quality[ci, indices] = coverage**2 * np.average(n[valid]/(n[valid]+3), weights=w[valid])
                if k == 0 and len(delta) > 1:
                    gaps = np.diff(positions[ci])
                    adjacent=ends[ci][:-1]==keys[ci]['start'][1:]
                    edges = valid[:-1] & valid[1:] & adjacent & (gaps>0)
                    possible = max(1e-12,float(positions[ci][-1]-positions[ci][0]))
                    edge_quarters=gaps[edges].sum()
                    cov = edge_quarters/possible
                    if edges.sum() >= 7 and cov >= .8:
                        raw[ci, 2] = np.abs(np.diff(delta))[edges].sum()/edge_quarters
                        known[ci, 2] = True
                        support = np.minimum(n[:-1], n[1:])[edges]
                        quality[ci, 2] = cov**2 * np.average(support/(support+3),weights=gaps[edges])
    return raw, known, quality


def fit_g_scaler(raw, known, fit_indices):
    center, scale, trained = np.zeros(23), np.ones(23), np.zeros(23, bool)
    for k in range(23):
        x = raw[fit_indices, k][known[fit_indices, k]]
        if len(x) < 20: continue
        center[k] = np.median(x)
        spread = np.quantile(x, .75)-np.quantile(x, .25)
        if spread <= 1e-8: spread = x.std()
        if spread > 1e-8: scale[k], trained[k] = spread, True
    return dict(center=center.tolist(), scale=scale.tolist(), trained=trained.tolist(),
                fit_indices=list(map(int, fit_indices)), clip_standardized=5.)


def pair_tables(raw, known, quality, metadata, scaler, ids):
    """Anchor signature is fixed for all its candidates; no cross-era/split pairs.

    Cross-work supervision avoids matching the score identity. At least tempo,
    timing, dynamics and articulation blocks must be observed. Untrained,
    constant dimensions are omitted rather than invented.
    """
    ids = np.asarray(ids, int)
    if len({metadata[i]['era'] for i in ids}) != 1 or len({metadata[i]['split'] for i in ids}) != 1:
        raise ValueError('Pair table must be single era and split')
    trained = np.array(scaler['trained'], bool)
    mask = known[ids] & trained
    x = np.clip((raw[ids]-scaler['center'])/scaler['scale'], -5, 5)
    distance = np.full((len(ids), len(ids)), np.nan, np.float32)
    confidence = np.zeros_like(distance)
    for a, global_a in enumerate(ids):
        blocks = [np.array([k for k in block if mask[a, k]], int) for block in G_BLOCKS.values()]
        if any(not len(b) for b in blocks[:4]): continue
        signature = np.flatnonzero(mask[a])
        candidates = np.flatnonzero(mask[:, signature].all(1))
        candidates = np.array([b for b in candidates if metadata[ids[b]]['work'] != metadata[global_a]['work']
                               and metadata[ids[b]]['source'] != metadata[global_a]['source']], int)
        if not len(candidates): continue
        ds, qs = [], []
        for block in blocks:
            if not len(block): continue
            ds.append(((x[candidates][:, block]-x[a, block])**2).mean(1))
            qa = quality[global_a, block].mean()
            qb = quality[ids[candidates]][:, block].mean(1)
            qs.append(np.minimum(qa, qb))
        distance[a, candidates] = np.mean(ds, axis=0)
        confidence[a, candidates] = np.mean(qs, axis=0)
    return distance, confidence


def distance_calibration(distance):
    x = distance[np.isfinite(distance)]
    if len(x) < 100: raise ValueError('Insufficient cross-work G pairs')
    median = float(np.median(x)); iqr = float(np.quantile(x, .75)-np.quantile(x, .25))
    if median <= 1e-8 or iqr <= 1e-8: raise ValueError('Degenerate G distances')
    return dict(median=median, iqr=iqr, pairs=len(x))


class EraStyleHead(nn.Module):
    """Identity-initialized bounded residual; preserves a direct path from AE z."""
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clamp_min(.05))
        d = len(scale)
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        shift = .5 * self.scale * torch.tanh(self.net(z))
        return dict(z_style=z+shift, e=F.normalize(z+shift, dim=-1),
                    head_penalty=(shift/self.scale).square().mean())


def g_objective(e, distances, confidence, calibration):
    """Confidence-weighted distance matching + gap-adaptive near/far ranking."""
    valid = torch.isfinite(distances)
    distance = torch.where(valid, distances, 0.)
    target = distance/(distance+calibration['median'])
    cosine_distance = 1-e@e.T
    if valid.any():
        pair = (confidence[valid]*(cosine_distance[valid]-target[valid]).square()).mean()
    else: pair = e.sum()*0
    terms = []
    for a in range(len(e)):
        candidates = torch.where(valid[a])[0]
        if len(candidates) < 2: continue
        p = candidates[distance[a, candidates].argmin()]
        n = candidates[distance[a, candidates].argmax()]
        gap = distance[a, n]-distance[a, p]
        if gap < .1*calibration['iqr']: continue
        margin = (.2*gap/calibration['iqr']).clamp(.05, .4).detach()
        weight = torch.minimum(confidence[a, p], confidence[a, n]).detach()
        terms.append(weight*F.relu(cosine_distance[a, p]-cosine_distance[a, n]+margin))
    rank = torch.stack(terms).mean() if terms else e.sum()*0
    return dict(loss=.5*pair+.5*rank, pair=pair, rank=rank,
                pairs=int(valid.sum()), triplets=len(terms))


def capped_style_weight(rec, style, z, requested, budget=.25):
    """Cap style gradient at the actual AE bottleneck, without backpropagating the cap."""
    rec_grad = torch.autograd.grad(rec, z, retain_graph=True)[0]
    style_grad = torch.autograd.grad(style, z, retain_graph=True)[0]
    nr, ng = float(rec_grad.norm()), float(style_grad.norm())
    weight = min(requested, budget*nr/max(ng, 1e-12))
    return weight, dict(rec_z_grad=nr, g_z_grad=ng, applied_weight=weight,
                        applied_gradient_ratio=weight*ng/max(nr, 1e-12))


def compose_era_objective(rec, g, anchor, head_penalty, weight, mode='joint', reconstruction_weight=1.):
    """Controlled loss ablations retain identical preservation regularizers.

    G-only removes the direct reconstruction gradient; the reconstruction-derived
    gradient cap remains a common safeguard, so this is not an unconstrained G run.
    """
    if not np.isfinite(reconstruction_weight) or reconstruction_weight<0:raise ValueError('Invalid reconstruction weight')
    if mode == 'joint':return reconstruction_weight*rec+weight*g+.05*anchor+.01*head_penalty
    if mode == 'reconstruction_only':return rec+.05*anchor+.01*head_penalty
    if mode == 'g_only':return weight*g+.05*anchor+.01*head_penalty
    raise ValueError('Unknown era objective mode: '+mode)


def load_era_checkpoint(path):
    """Load the self-contained inference artifact, including the trained score encoder."""
    from .note_autoencoder import NotePerformanceAutoencoder, NoteAEConfig
    from .note_score_encoder import NoteScoreEncoderConfig
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state.get('format') != 'era-note-ae-head-v2' or state.get('era') not in ERA_COMPOSERS
            or state.get('supervision_version')!=G_PROFILE_VERSION):
        raise ValueError('Expected v2 absolute-beat-cell era checkpoint; v1 G references were invalidated')
    model = NotePerformanceAutoencoder(NoteAEConfig(**{**state['model_config'], 'score_gradient': 'frozen'}),
                                      NoteScoreEncoderConfig(**state['score_config']))
    model.load_state_dict(state['state_dict'])
    model.r_scaler_fingerprint = state['scaler']['id']
    model.score_scale_fingerprint = state['scaler']['id']
    model.score_encoder.requires_grad_(False)
    head = EraStyleHead(state['head_scale'])
    head.load_state_dict(state['head_state_dict'])
    model.eval(); head.eval()
    return model, head, state


def aggregate_clip_latents(z, owner, observed_notes, clip_count):
    """Same differentiable pooling during G training and deployment export."""
    if (z.ndim != 2 or owner.shape != z.shape[:1] or observed_notes.shape != owner.shape
            or (observed_notes < 0).any() or not torch.isfinite(observed_notes).all()
            or (owner < 0).any() or (owner >= clip_count).any()):
        raise ValueError('Invalid clip pooling inputs')
    totals = z.new_zeros(clip_count).index_add(0, owner, observed_notes)
    if (totals <= 0).any(): raise ValueError('Cannot embed a clip without observations')
    return z.new_zeros((clip_count,z.shape[1])).index_add(0, owner,
                                                       z*(observed_notes/totals[owner])[:,None])
