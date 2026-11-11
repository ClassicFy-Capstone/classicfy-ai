"""Bounded-memory loading of the same full eight-bar S/R AE inputs."""
from collections import OrderedDict, defaultdict
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from features.note_performance_normalization import NotePerformanceScaler, load_note_performance_scaler
from features.note_score_context import NoteScoreScale
from features.style_profile import check_artifact, stable_id
from .note_ae_data import collate_ae_windows
from .note_performance_data import NotePerformancePair, NotePerformanceSegment, load_note_performance
from .note_score_context_data import NoteScoreSegment, load_note_score_context, load_note_score_scale


class RecordBlockSampler(Sampler):
    """Shuffle records, then windows within cache-sized blocks; use each once."""
    def __init__(self, items, generator, block_records=8):
        if block_records < 1:
            raise ValueError('Positive record block size required')
        groups = defaultdict(list)
        for index, item in enumerate(items):
            groups[item['window']['performance_id']].append(index)
        self.groups = [groups[key] for key in sorted(groups)]
        self.generator, self.block_records = generator, block_records

    def __len__(self):
        return sum(map(len, self.groups))

    def __iter__(self):
        order = torch.randperm(len(self.groups), generator=self.generator).tolist()
        for start in range(0, len(order), self.block_records):
            indices = [i for group in order[start:start + self.block_records] for i in self.groups[group]]
            for offset in torch.randperm(len(indices), generator=self.generator).tolist():
                yield indices[offset]


class IndexedNoteWindows(Dataset):
    def __init__(self, corpus, items):
        self.corpus, self.items = corpus, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.corpus.resolve(self.items[index])


class IndexedNoteAECorpus:
    """Keep only a small recording/S cache, not every raw note in the corpus."""
    def __init__(self, index, r_scaler, score_scale, *, cache_records=8):
        if cache_records < 1:
            raise ValueError('Positive recording cache size required')
        check_artifact(index, 'ae-training-index')
        self.index, self.cache_records = index, cache_records
        self.r_scaler = r_scaler if isinstance(r_scaler, NotePerformanceScaler) else load_note_performance_scaler(r_scaler)
        self.score_scale = score_scale if isinstance(score_scale, NoteScoreScale) else load_note_score_scale(score_scale)
        if self.r_scaler.fit_manifest_id != index['id']:
            raise ValueError('R scaler and input index differ')
        self.sources = {entry['performance_id']: entry for entry in index['sources']}
        if len(self.sources) != len(index['sources']):
            raise ValueError('Duplicate performance source')
        work_splits, score_splits, recording_splits, verified_xml = {}, {}, {}, {}
        for entry in self.sources.values():
            path = entry['xml_path']
            if path not in verified_xml:
                verified_xml[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if verified_xml[path] != entry['score_source_sha256']:
                raise ValueError('XML source changed after preprocessing')
            for mapping, key in ((work_splits, entry['work_key']),
                                 (score_splits, entry['score_source_sha256']),
                                 (recording_splits, entry['recording_id'])):
                if key in mapping and mapping[key] != entry['split']:
                    raise ValueError('Work, score or recording crosses dataset splits')
                mapping[key] = entry['split']
        for entry in self.r_scaler.training_records:
            if work_splits.get(entry['work_key'], 'train') != 'train':
                raise ValueError('R scaler training work crosses evaluation split')
        if any(score_splits.get(sha, 'train') != 'train' for sha in self.score_scale.training_sources):
            raise ValueError('Score scale training source crosses evaluation split')
        self.items, self.excluded = index['items'], index['excluded']
        ids = [item['window']['window_id'] for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate AE window')
        self._records, self._scores = OrderedDict(), OrderedDict()
        self.collate = partial(collate_ae_windows, r_scaler=self.r_scaler, score_scale=self.score_scale)

    def dataset(self, split):
        return IndexedNoteWindows(self, [item for item in self.items if item['window']['split'] == split])

    def training_sampler(self, dataset, generator):
        return RecordBlockSampler(dataset.items, generator, self.cache_records)

    def _record(self, pid):
        if pid in self._records:
            self._records.move_to_end(pid)
            return self._records[pid]
        entry = self.sources[pid]
        path = Path(entry['windows_file'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['windows_sha256']:
            raise ValueError('Window file changed')
        windows = json.loads(path.read_text())
        r = load_note_performance(entry['performance_cache'])
        if (r.fingerprint != entry['R_fingerprint'] or r.split != entry['split']
                or r.work_key != entry['work_key'] or r.record_key != entry['record_key']):
            raise ValueError('Recording source or split changed')
        score_path = entry['score_cache']
        if score_path not in self._scores:
            self._scores[score_path] = load_note_score_context(score_path)[0]
        self._scores.move_to_end(score_path)
        while len(self._scores) > self.cache_records:
            self._scores.popitem(last=False)
        pair = NotePerformancePair(self._scores[score_path], r)
        self._records[pid] = pair, windows
        while len(self._records) > self.cache_records:
            self._records.popitem(last=False)
        return pair, windows

    def resolve(self, item):
        identity = item['window']
        pair, windows = self._record(identity['performance_id'])
        window = windows[item['local_index']]
        if any(window[key] != value for key, value in identity.items()):
            raise ValueError('Indexed window identity changed')
        start, stop = np.searchsorted(pair.score.onset_q, [window['q_start'], window['q_end']], side='left')
        rows = np.flatnonzero((pair.score.note_to_onset >= start) & (pair.score.note_to_onset < stop))
        if (rows.tolist() != window['row_indices']
                or [pair.performance.row_keys[i] for i in rows] != window['score_note_keys']
                or window['R_fingerprint'] != pair.performance.fingerprint
                or window['score_version_id'] != pair.score.source_sha256
                or window['repeat_route_id'] != pair.score.traversal_id):
            raise ValueError('AE window/S/R correspondence changed')
        segment = NotePerformanceSegment(pair, NoteScoreSegment(pair.score, int(start), int(stop), int(start), int(stop)))
        return {'window': window, 'segment': segment}

    @property
    def contract(self):
        return {'windows_id': self.index['id'], 'validation_windows_id': None,
                'R_scaler': self.r_scaler.fingerprint, 'score_scale': stable_id(self.score_scale.__dict__),
                'split': {key: entry['split'] for key, entry in sorted(self.sources.items())},
                'eligible_windows': sorted(item['window']['window_id'] for item in self.items),
                'input_loading': 'indexed full windows; unchanged collate_ae_windows',
                'record_block_shuffle': self.cache_records}
