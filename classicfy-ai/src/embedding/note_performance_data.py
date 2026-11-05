"""R caches and identity-checked score/R batches, stopping before the AE."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from features.note_performance import NotePerformanceFeatures, note_performance_schema
from features.note_performance_normalization import NotePerformanceScaler
from features.note_score_context import NoteScoreFeatures
from features.performance_time import BeatTimeMap
from .note_score_context_data import NoteScoreSegment, collate_note_score_context, split_note_score_context


R_ARRAY_FIELDS = ('values', 'mask', 'physical', 'physical_mask', 'reasons', 'beat_id', 'local_tau', 'expected_duration_sec')


@dataclass(frozen=True)
class NotePerformancePair:
    score: NoteScoreFeatures
    performance: NotePerformanceFeatures

    def __post_init__(self):
        r, s = self.performance, self.score
        if (r.score_ref != s.cache_key or r.score_source_sha256 != s.source_sha256 or r.traversal_id != s.traversal_id
                or r.row_keys != tuple(meta['occurrence_id'] for meta in s.metadata)):
            raise ValueError('S/R source, traversal and ordered occurrence keys must all agree')
        if any(meta['score_onset_q'] != float(q) for meta, q in
               zip(r.metadata, (s.onset_q[index] for index in s.note_to_onset))):
            raise ValueError('S/R onset coordinates disagree')


@dataclass(frozen=True)
class NotePerformanceSegment:
    pair: NotePerformancePair
    score_segment: NoteScoreSegment

    def __post_init__(self):
        if self.score_segment.features.cache_key != self.pair.score.cache_key:
            raise ValueError('Chunk belongs to a different score')


def split_note_performance(pair, *, max_onsets=256, overlap=32):
    return [NotePerformanceSegment(pair, segment) for segment in
            split_note_score_context(pair.score, max_onsets=max_onsets, overlap=overlap)]


def collate_note_performance(items, scaler: NotePerformanceScaler, *, score_scale=None):
    """Keep row observations separate from chunk/loss masks and beat tempo targets.

    R_loss_mask excludes tempo: beat_loss_mask exposes one target per owned beat.
    These are data masks only; no reconstruction objective is implemented here.
    Full beat tables remain attached to every chunk, including beats without notes.
    """
    segments = []
    for item in items:
        if isinstance(item, NotePerformancePair):
            item = NotePerformanceSegment(item, NoteScoreSegment(item.score, 0, len(item.score.onset_q), 0, len(item.score.onset_q)))
        if not isinstance(item, NotePerformanceSegment):
            raise TypeError('Expected NotePerformancePair or NotePerformanceSegment')
        segments.append(item)
    if not segments:
        raise ValueError('Cannot collate an empty S/R batch')
    batch = collate_note_score_context([item.score_segment for item in segments], scale=score_scale)
    b, n = batch['note_mask'].shape
    beat_count = max(len(item.pair.performance.beats.valid) for item in segments)
    arrays = {'R': np.zeros((b, n, 11), np.float32), 'R_scaled': np.zeros((b, n, 11), np.float32),
              'R_mask': np.zeros((b, n, 11), bool), 'R_loss_mask': np.zeros((b, n, 11), bool),
              'beat_id': np.full((b, n), -1, np.int64), 'local_tau': np.zeros((b, n), np.float32),
              'expected_duration_sec': np.zeros((b, n), np.float32),
              'beat_values': np.zeros((b, beat_count), np.float32),
              'beat_scaled': np.zeros((b, beat_count), np.float32),
              'beat_mask': np.zeros((b, beat_count), bool), 'beat_table_mask': np.zeros((b, beat_count), bool),
              'beat_loss_mask': np.zeros((b, beat_count), bool), 'beat_tau': np.zeros((b, beat_count), np.float32),
              'beat_q_edges': np.zeros((b, beat_count + 1), np.float64)}
    record_metadata = []
    for bi, item in enumerate(segments):
        segment, pair = item.score_segment, item.pair
        r = pair.performance
        rows = np.asarray(batch['metadata'][bi]['note_indices'], np.int64)
        count, beats = len(rows), len(r.beats.valid)
        arrays['R'][bi, :count] = r.values[rows]
        arrays['R_scaled'][bi, :count] = scaler.transform(r)[rows]
        arrays['R_mask'][bi, :count] = r.mask[rows]
        center = batch['loss_note_mask'][bi, :count].numpy()
        arrays['R_loss_mask'][bi, :count] = r.mask[rows] & center[:, None] & np.asarray(scaler.trained_channel_mask)[None, :]
        arrays['R_loss_mask'][bi, :, 0] = False
        arrays['beat_id'][bi, :count] = r.beat_id[rows]
        arrays['local_tau'][bi, :count] = r.local_tau[rows]
        arrays['expected_duration_sec'][bi, :count] = r.expected_duration_sec[rows]
        arrays['beat_values'][bi, :beats] = np.where(r.beats.valid, r.beats.log2_qpm, 0)
        arrays['beat_scaled'][bi, :beats] = scaler.transform_beats(r)
        arrays['beat_mask'][bi, :beats] = r.beats.valid
        arrays['beat_table_mask'][bi, :beats] = True
        arrays['beat_tau'][bi, :beats] = r.beats.tau
        if len(r.beats.q):
            arrays['beat_q_edges'][bi, :beats + 1] = r.beats.q
        left = -np.inf if segment.loss_start_group == 0 else pair.score.onset_q[segment.loss_start_group]
        right = np.inf if segment.loss_stop_group == len(pair.score.onset_q) else pair.score.onset_q[segment.loss_stop_group]
        arrays['beat_loss_mask'][bi, :beats] = (r.beats.valid & (r.beats.q[:-1] >= left)
                                               & (r.beats.q[:-1] < right) & scaler.trained_channel_mask[0])
        record_metadata.append({'dataset': r.dataset, 'record_key': r.record_key, 'work_key': r.work_key,
                                'split': r.split, 'score_ref': r.score_ref, 'R_fingerprint': r.fingerprint,
                                'row_keys': [r.row_keys[int(index)] for index in rows],
                                'rows': [r.metadata[int(index)] for index in rows],
                                'beat_provenance': r.beats.provenance, 'R_provenance': r.provenance})
    for name, array in arrays.items():
        if array.dtype.kind == 'f' and not np.isfinite(array).all():
            raise ValueError(f'Batch {name} overflows model floating point range')
        batch[name] = torch.from_numpy(array)
    batch['trained_channel_mask'] = torch.tensor(scaler.trained_channel_mask, dtype=torch.bool)
    batch['R_metadata'] = record_metadata
    batch['R_scaler'] = {'fingerprint': scaler.fingerprint, 'fit_manifest_id': scaler.fit_manifest_id,
                          'schema': note_performance_schema()}
    return batch


def save_note_performance(features, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.note-performance-', dir=out.parent))
    try:
        arrays = {name: getattr(features, name) for name in R_ARRAY_FIELDS}
        arrays.update(note_mask=features.note_mask, beat_q=features.beats.q,
                      beat_seconds=features.beats.seconds, beat_valid=features.beats.valid)
        np.savez_compressed(temporary / 'performance.npz', **arrays)
        manifest = {'schema': note_performance_schema(),
                    **{name: getattr(features, name) for name in ('record_key', 'work_key', 'split', 'dataset', 'score_ref',
                                                                  'score_source_sha256', 'traversal_id', 'row_keys', 'metadata', 'provenance')},
                    'beat_provenance': features.beats.provenance, 'fingerprint': features.fingerprint,
                    'array_sha256': hashlib.sha256((temporary / 'performance.npz').read_bytes()).hexdigest()}
        (temporary / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        if out.exists():
            raise FileExistsError(out)
        os.rename(temporary, out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_note_performance(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != note_performance_schema():
        raise ValueError('R cache schema mismatch')
    if hashlib.sha256((path / 'performance.npz').read_bytes()).hexdigest() != manifest['array_sha256']:
        raise ValueError('R cache array integrity mismatch')
    with np.load(path / 'performance.npz', allow_pickle=False) as arrays:
        if (arrays['note_mask'].shape != (len(manifest['row_keys']),) or arrays['note_mask'].dtype != bool
                or not arrays['note_mask'].all()):
            raise ValueError('Stored full-score R rows must all have note_mask=True')
        beats = BeatTimeMap(arrays['beat_q'], arrays['beat_seconds'], arrays['beat_valid'], manifest['beat_provenance'])
        features = NotePerformanceFeatures(*(manifest[name] for name in
                                            ('record_key', 'work_key', 'split', 'dataset', 'score_ref', 'score_source_sha256', 'traversal_id')),
                                           tuple(manifest['row_keys']), *(arrays[name] for name in R_ARRAY_FIELDS),
                                           beats, tuple(manifest['metadata']), manifest['provenance'])
    if features.fingerprint != manifest['fingerprint']:
        raise ValueError('R cache record integrity mismatch')
    return features


def export_scaled_note_performance(features, scaler, out):
    """Portable pre-AE tensors plus immutable raw-cache/scaler references."""
    path = Path(out)
    if path.exists():
        raise FileExistsError(path)
    manifest = {'schema': note_performance_schema(), 'record_key': features.record_key,
                'work_key': features.work_key, 'split': features.split, 'dataset': features.dataset,
                'score_ref': features.score_ref, 'row_keys': features.row_keys,
                'raw_fingerprint': features.fingerprint, 'scaler': scaler.to_dict(), 'scaler_fingerprint': scaler.fingerprint,
                'metadata': features.metadata, 'beat_provenance': features.beats.provenance}
    with path.open('xb') as stream:
        np.savez_compressed(stream, R_scaled=scaler.transform(features), R_mask=features.mask,
                            note_mask=np.ones(len(features.row_keys), bool), beat_id=features.beat_id,
                            local_tau=features.local_tau, expected_duration_sec=features.expected_duration_sec,
                            beat_scaled=scaler.transform_beats(features), beat_mask=features.beats.valid,
                            beat_q=features.beats.q, beat_seconds=features.beats.seconds,
                            trained_channel_mask=np.asarray(scaler.trained_channel_mask, bool),
                            manifest=np.array(json.dumps(manifest, ensure_ascii=False, allow_nan=False)))
