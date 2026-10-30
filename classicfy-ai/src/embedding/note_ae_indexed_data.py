"""Bounded-memory AE inputs from a verified, per-record window index.

Only deterministic input tensors are cached; score context remains trainable.
The source R, split, masks and train-only scalers are never modified.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from features.note_performance_normalization import load_note_performance_scaler
from features.style_profile import check_artifact, stable_id
from .note_ae_data import NoteAEWindows
from .note_performance_data import NotePerformancePair, load_note_performance
from .note_score_context_data import load_note_score_context, load_note_score_scale


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _source_score(path):
    return load_note_score_context(path)[0]


def pack_record(task):
    """Validate one source and persist compact tensors, retaining every full window."""
    entry, score_path, destination, r_scale_path, s_scale_path = task
    torch.set_num_threads(1)
    destination = Path(destination)
    scaler = load_note_performance_scaler(r_scale_path)
    score_scale = load_note_score_scale(s_scale_path)
    source_bytes = Path(entry['windows_file']).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != entry['windows_sha256']:
        raise ValueError('Window source changed')
    windows = json.loads(source_bytes)
    score = _source_score(score_path)
    r = load_note_performance(entry['performance_cache'])
    NotePerformancePair(score, r)
    if (r.fingerprint != entry['R_fingerprint'] or r.split != entry['split']
            or r.work_key != entry['work_key'] or file_sha(entry['xml_path']) != score.source_sha256):
        raise ValueError('Indexed source identity changed')
    score_file = destination / 'scores' / (score.cache_key + '.pt')
    if not score_file.exists():
        # Parallel records may share a score. Unique temporary names and atomic
        # replacement ensure readers never see a partial tensor file.
        import os
        temporary = score_file.with_suffix(f'.{os.getpid()}.tmp')
        torch.save({
            'numeric': torch.from_numpy(score_scale.transform(score).copy()),
            **{k: torch.from_numpy(getattr(score, k).copy()) for k in
               ('categorical', 'flags', 'field_known', 'note_to_onset', 'onset_q')},
        }, temporary)
        temporary.replace(score_file)
    packed, excluded = [], []
    keep = ('window_id', 'work_id', 'performance_id', 'recording_id', 'split',
            'same_window_key', 'q_start', 'q_end', 'N', 'score_version_id')
    for w in windows:
        reason = ('tail' if w['tail'] else 'empty_window' if not w['N'] else
                  'no_observed_reconstruction' if not r.mask[w['row_indices']].any() else None)
        if reason:
            excluded.append({'window_id': w['window_id'], 'split': w['split'], 'reason': reason})
            continue
        start, stop = map(int, np.searchsorted(score.onset_q, [w['q_start'], w['q_end']], side='left'))
        rows = np.flatnonzero((score.note_to_onset >= start) & (score.note_to_onset < stop))
        if (rows.tolist() != w['row_indices'] or len(rows) != w['N']
                or [r.row_keys[i] for i in rows] != w['score_note_keys']
                or w['R_fingerprint'] != r.fingerprint or w['score_version_id'] != score.source_sha256
                or w['repeat_route_id'] != score.traversal_id or w['split'] != r.split
                or w['work_id'] != r.work_key or w['performance_id'] != entry['performance_id']):
            raise ValueError('Window, score and R occurrence correspondence mismatch')
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError('Expected contiguous notation rows')
        packed.append({'window': {k: w[k] for k in keep}, 'score_key': score.cache_key,
                       'record_key': entry['record_key'], 'row_start': int(rows[0]),
                       'row_stop': int(rows[-1]) + 1, 'start_group': start, 'stop_group': stop})
    record_file = destination / 'records' / (entry['record_key'] + '.pt')
    torch.save({
        'R': torch.from_numpy(scaler.transform(r)),
        'R_raw': torch.from_numpy(r.values.astype(np.float32)),
        'M': torch.from_numpy(r.mask.copy()),
        'beat_id': torch.from_numpy(r.beat_id.copy()),
        'local_tau': torch.from_numpy(r.local_tau.astype(np.float32)),
        'beat_scaled': torch.from_numpy(scaler.transform_beats(r)),
        'beat_valid': torch.from_numpy(r.beats.valid.copy()),
        'beat_q': torch.from_numpy(r.beats.q.copy()),
    }, record_file)
    return {'items': packed, 'excluded': excluded, 'source': entry,
            'score_sha': score.source_sha256, 'score_key': score.cache_key,
            'record_sha': file_sha(record_file)}


class IndexedNoteAECorpus:
    """Duck-compatible with NoteAECorpus; memory is bounded by tensor LRU caches."""

    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest['format'] != 'indexed-note-ae-tensors-v1':
            raise ValueError('Unknown indexed AE corpus format')
        self.r_scaler = load_note_performance_scaler(self.path / 'R_scaler.json')
        self.score_scale = load_note_score_scale(self.path / 'S_scale.json')
        self._r_id = self.r_scaler.fingerprint
        self._s_id = stable_id(self.score_scale.__dict__)
        if (self._r_id != self.manifest['R_scaler'] or self._s_id != self.manifest['score_scale']):
            raise ValueError('Indexed corpus scaler changed')
        self.items = {v['window']['window_id']: v for v in self.manifest['items']}
        if len(self.items) != len(self.manifest['items']):
            raise ValueError('Duplicate indexed window')
        self.excluded = self.manifest['excluded']
        self.windows = {'id': self.manifest['source_index_id']}
        for kind in ('records', 'scores'):
            for key, digest in self.manifest['tensor_hashes'][kind].items():
                if file_sha(self.path / kind / (key + '.pt')) != digest:
                    raise ValueError('Indexed input tensor integrity mismatch')

    @lru_cache(maxsize=128)
    def _record(self, key):
        return torch.load(self.path / 'records' / (key + '.pt'), weights_only=True, mmap=True)

    @lru_cache(maxsize=64)
    def _score(self, key):
        return torch.load(self.path / 'scores' / (key + '.pt'), weights_only=True, mmap=True)

    def dataset(self, split):
        return NoteAEWindows([v for _, v in sorted(self.items.items()) if v['window']['split'] == split])

    @property
    def contract(self):
        return {'windows_id': self.manifest['source_index_id'], 'validation_windows_id': None,
                'R_scaler': self._r_id, 'score_scale': self._s_id,
                'split': self.manifest['splits'], 'eligible_windows': sorted(self.items),
                'tensor_manifest': self.manifest['id']}

    def collate(self, items):
        values = defaultdict(list)
        for item in items:
            r, s = self._record(item['record_key']), self._score(item['score_key'])
            a, b = item['row_start'], item['row_stop']
            g, h = item['start_group'], item['stop_group']
            for k in ('numeric', 'categorical', 'flags', 'field_known'):
                v = s[k][a:b]
                values[k].append(v.to(torch.float32) if k == 'flags' else v)
            for k in ('R', 'R_raw', 'M', 'beat_id', 'local_tau'):
                values[k].append(r[k][a:b])
            values['note_to_onset'].append(s['note_to_onset'][a:b] - g)
            values['note_mask'].append(torch.ones(b - a, dtype=torch.bool))
            values['onset_mask'].append(torch.ones(h - g, dtype=torch.bool))
            q = s['onset_q'][g:h]
            values['delta_q'].append(torch.cat([q.new_zeros(1), q.diff()]).float())
            values['beat_scaled'].append(r['beat_scaled'])
            w = item['window']
            values['beat_loss_mask'].append(r['beat_valid'] & (r['beat_q'][:-1] >= w['q_start'])
                                            & (r['beat_q'][:-1] < w['q_end'])
                                            & self.r_scaler.trained_channel_mask[0])
        batch = {k: pad_sequence(v, batch_first=True, padding_value=-1 if k in
                                 ('note_to_onset', 'beat_id') else 0) for k, v in values.items()}
        batch.update(P=batch['note_mask'], lengths=batch['note_mask'].sum(1),
                     window_metadata=[v['window'] for v in items],
                     R_state='train-channel-scaled-once', score_scale_id=self._s_id,
                     R_scaler={'fingerprint': self._r_id})
        return batch
