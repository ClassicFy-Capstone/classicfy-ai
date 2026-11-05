"""Persist and batch note-level score features without mixing reconstruction masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from features.note_score_context import (
    CATEGORICAL_FIELDS, FLAG_FIELDS, NUMERIC_FIELDS, SCHEMA_VERSION,
    NoteScoreFeatures, NoteScoreScale, encode_note_numeric, note_score_schema,
)
from preprocessing.score_note_mapping import NoteAlignmentLinks, TieMapping


@dataclass(frozen=True)
class NoteScoreSegment:
    features: NoteScoreFeatures
    start_group: int
    stop_group: int
    loss_start_group: int
    loss_stop_group: int

    def __post_init__(self):
        bounds = (self.start_group, self.stop_group, self.loss_start_group, self.loss_stop_group)
        if any(isinstance(x, bool) or not isinstance(x, int) for x in bounds):
            raise ValueError('Segment group bounds must be integers')
        if not 0 <= self.start_group <= self.loss_start_group < self.loss_stop_group <= self.stop_group <= len(self.features.onset_q):
            raise ValueError('Invalid onset segment bounds')


def split_note_score_context(features: NoteScoreFeatures, *, max_onsets=256, overlap=32):
    """Chunk whole onset groups; loss ownership is disjoint and remains global."""
    if (isinstance(max_onsets, bool) or not isinstance(max_onsets, int) or max_onsets < 1
            or isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0):
        raise ValueError('Expected positive onset count and nonnegative overlap')
    count = len(features.onset_q)
    return [NoteScoreSegment(features, max(0, start - overlap), min(count, start + max_onsets + overlap),
                              start, min(count, start + max_onsets))
            for start in range(0, count, max_onsets)]


def collate_note_score_context(items, *, scale: NoteScoreScale | None = None, alignment_links=None):
    """Return encoder fields and separate score metadata/loss ownership masks."""
    segments = []
    for item in items:
        if isinstance(item, NoteScoreFeatures):
            item = NoteScoreSegment(item, 0, len(item.onset_q), 0, len(item.onset_q))
        if not isinstance(item, NoteScoreSegment):
            raise TypeError('Expected NoteScoreFeatures or NoteScoreSegment')
        segments.append(item)
    if not segments:
        raise ValueError('Cannot collate an empty score batch')
    if alignment_links is not None:
        alignment_links = list(alignment_links)
        if len(alignment_links) != len(segments):
            raise ValueError('Provide one alignment link set per batch item')
    rows = [np.flatnonzero((s.features.note_to_onset >= s.start_group) &
                          (s.features.note_to_onset < s.stop_group)) for s in segments]
    n, g, b = max(map(len, rows)), max(s.stop_group - s.start_group for s in segments), len(segments)
    if n == 0:
        raise ValueError('Cannot encode an empty pitched-note score')
    arrays = {
        'numeric': np.zeros((b, n, len(NUMERIC_FIELDS)), np.float32),
        'categorical': np.zeros((b, n, len(CATEGORICAL_FIELDS)), np.int64),
        'flags': np.zeros((b, n, len(FLAG_FIELDS)), np.float32),
        'field_known': np.zeros((b, n, len(NUMERIC_FIELDS + CATEGORICAL_FIELDS + FLAG_FIELDS)), bool),
        'note_mask': np.zeros((b, n), bool), 'onset_mask': np.zeros((b, g), bool),
        'note_to_onset': np.full((b, n), -1, np.int64), 'delta_q': np.zeros((b, g), np.float32),
        'loss_note_mask': np.zeros((b, n), bool),
        'attack_owner_mask': np.zeros((b, n), bool),
        'duration_owner_mask': np.zeros((b, n), bool),
        'tie_chain_duration_q': np.zeros((b, n), np.float32),
    }
    if alignment_links is not None:
        arrays['attack_target_mask'] = np.zeros((b, n), bool)
        arrays['duration_target_mask'] = np.zeros((b, n), bool)
    metadata = []
    for bi, (segment, indices) in enumerate(zip(segments, rows)):
        features, length = segment.features, len(indices)
        width = segment.stop_group - segment.start_group
        numeric = scale.transform(features) if scale is not None else encode_note_numeric(features)
        arrays['numeric'][bi, :length] = numeric[indices]
        for name in ('categorical', 'flags', 'field_known'):
            arrays[name][bi, :length] = getattr(features, name)[indices]
        arrays['note_mask'][bi, :length] = True
        arrays['onset_mask'][bi, :width] = True
        groups = features.note_to_onset[indices]
        arrays['note_to_onset'][bi, :length] = groups - segment.start_group
        q = features.onset_q[segment.start_group:segment.stop_group]
        arrays['delta_q'][bi, 1:width] = np.diff(q)
        center = (groups >= segment.loss_start_group) & (groups < segment.loss_stop_group)
        arrays['loss_note_mask'][bi, :length] = center
        arrays['attack_owner_mask'][bi, :length] = features.ties.is_attack[indices] & center
        arrays['duration_owner_mask'][bi, :length] = features.ties.is_attack[indices] & features.ties.chain_known[indices] & center
        arrays['tie_chain_duration_q'][bi, :length] = features.ties.chain_duration_q[indices]
        if alignment_links is not None:
            links = alignment_links[bi]
            if not isinstance(links, NoteAlignmentLinks) or links.occurrence_ids != features.ties.occurrence_ids:
                raise ValueError('Alignment links belong to a different score/traversal')
            arrays['attack_target_mask'][bi, :length] = links.attack_mask[indices] & center
            arrays['duration_target_mask'][bi, :length] = links.duration_mask[indices] & center
        metadata.append({'score_ref': features.cache_key, 'note_indices': indices.tolist(),
                         'notes': [features.metadata[int(i)] for i in indices],
                         'start_group': segment.start_group, 'stop_group': segment.stop_group})
    if not np.isfinite(arrays['numeric']).all() or not np.isfinite(arrays['delta_q']).all():
        raise ValueError('Score batch overflows float32')
    result = {name: torch.from_numpy(value) for name, value in arrays.items()}
    result['metadata'] = metadata
    return result


def save_note_score_context(features: NoteScoreFeatures, out, *, links: NoteAlignmentLinks | None = None,
                            source_tables: dict | None = None):
    """Write a new cache directory atomically; never replace existing artifacts."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.note-score-', dir=out.parent))
    try:
        arrays = {name: getattr(features, name) for name in
                  ('numeric', 'categorical', 'flags', 'field_known', 'note_to_onset', 'onset_q')}
        for name in ('owner_indices', 'chain_duration_q', 'chain_known', 'is_attack'):
            arrays['ties_' + name] = getattr(features.ties, name)
        manifest = {'schema': note_score_schema(), 'source_sha256': features.source_sha256,
                    'traversal_id': features.traversal_id, 'cache_key': features.cache_key,
                    'notes': features.metadata, 'diagnostics': features.diagnostics,
                    'tie_occurrence_ids': features.ties.occurrence_ids, 'tie_chain_ids': features.ties.chain_ids}
        if links is not None:
            if links.occurrence_ids != features.ties.occurrence_ids:
                raise ValueError('Alignment and score identities differ')
            manifest['alignment'] = {'performed_ids': links.performed_ids, 'status': links.status,
                                     'unresolved_score_ids': links.unresolved_score_ids}
            arrays['aligned_attack_mask'] = links.attack_mask
            arrays['aligned_duration_mask'] = links.duration_mask
        np.savez_compressed(temporary / 'features.npz', **arrays)
        manifest['features_sha256'] = hashlib.sha256((temporary / 'features.npz').read_bytes()).hexdigest()
        (temporary / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        if source_tables is not None:
            (temporary / 'source_tables.json').write_text(json.dumps(source_tables, default=str, ensure_ascii=False,
                                                                     indent=2, allow_nan=False), encoding='utf-8')
        if out.exists():
            raise FileExistsError(out)
        os.rename(temporary, out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_note_score_context(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != note_score_schema():
        raise ValueError('Note score cache schema mismatch')
    if hashlib.sha256((path / 'features.npz').read_bytes()).hexdigest() != manifest['features_sha256']:
        raise ValueError('Note score cache integrity mismatch')
    with np.load(path / 'features.npz', allow_pickle=False) as cache:
        ties = TieMapping(tuple(manifest['tie_occurrence_ids']), tuple(manifest['tie_chain_ids']),
                          *(cache['ties_' + name] for name in
                            ('owner_indices', 'chain_duration_q', 'chain_known', 'is_attack')))
        features = NoteScoreFeatures(*(cache[name] for name in
                                      ('numeric', 'categorical', 'flags', 'field_known', 'note_to_onset', 'onset_q')),
                                     tuple(manifest['notes']), ties, manifest['source_sha256'], manifest['traversal_id'],
                                     tuple(manifest['diagnostics']))
        links = None
        if 'alignment' in manifest:
            alignment = manifest['alignment']
            attack, duration = cache['aligned_attack_mask'].copy(), cache['aligned_duration_mask'].copy()
            if (attack.shape != (len(features.metadata),) or duration.shape != attack.shape
                    or attack.dtype != bool or duration.dtype != bool
                    or np.any(attack & ~ties.is_attack) or np.any(duration & ~(attack & ties.chain_known))):
                raise ValueError('Invalid cached reconstruction masks')
            attack.setflags(write=False)
            duration.setflags(write=False)
            links = NoteAlignmentLinks(tuple(tuple(ids) for ids in alignment['performed_ids']),
                                       tuple(alignment['status']), attack, duration,
                                       tuple(alignment['unresolved_score_ids']), ties.occurrence_ids)
    if features.cache_key != manifest['cache_key']:
        raise ValueError('Score cache key mismatch')
    return features, links


def save_note_score_scale(scale: NoteScoreScale, path):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(asdict(scale), stream, indent=2, allow_nan=False)


def load_note_score_scale(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Note score scale schema mismatch')
    return NoteScoreScale(tuple(payload['mean']), tuple(payload['scale']), tuple(payload['training_sources']))
