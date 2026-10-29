"""Identity-checked, variable-note full eight-bar AE and fixed triplet datasets."""

from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset

from features.note_performance_normalization import fit_note_performance_scaler
from features.note_score_context import extract_note_score_context, fit_note_score_scale
from features.style_profile import check_artifact, stable_id, validate_recordings
from .note_performance_data import NotePerformancePair, NotePerformanceSegment, collate_note_performance
from .note_score_context_data import NoteScoreSegment
from .style_profile_data import verify_window_sources
from .style_triplets import anchor_signature, profile_distance, scale_profile, signature_key, validate_triplet


class NoteAECorpus:
    def __init__(self, windows, *, validation_windows=None, r_scaler=None, score_scale=None):
        check_artifact(windows, 'windows')
        self.windows = windows
        self.validation_windows = validation_windows
        recordings = verify_window_sources(windows)
        all_windows = list(windows['windows'])
        if validation_windows is not None:
            check_artifact(validation_windows, 'windows')
            if any(w['split'] != 'validation' for w in validation_windows['windows']):
                raise ValueError('Additional validation windows must be validation only')
            recordings += verify_window_sources(validation_windows)
            all_windows += validation_windows['windows']
        validate_recordings(recordings)
        ids = [rec.performance_id for rec in recordings]
        if len(ids) != len(set(ids)):
            raise ValueError('Corpus repeats a performance ID')
        train = [r.performance for r in recordings if r.performance.split == 'train']
        self.r_scaler = r_scaler or fit_note_performance_scaler(train, fit_manifest_id=windows['id'])
        # Reject known split leakage even when an external train scaler is supplied.
        splits = {r.performance.work_key: r.performance.split for r in recordings}
        for row in self.r_scaler.training_records:
            if splits.get(row['work_key'], 'train') != 'train':
                raise ValueError('R scaler train work appears in an evaluation split')
        scores, self.pairs = {}, {}
        for rec in recordings:
            key = rec.performance.score_ref
            if key not in scores:
                scores[key] = extract_note_score_context(rec.xml)
            self.pairs[rec.performance_id] = NotePerformancePair(scores[key], rec.performance)
        self.score_scale = score_scale or fit_note_score_scale(
            pair.score for pair in self.pairs.values() if pair.performance.split == 'train')
        if any(pair.score.source_sha256 in self.score_scale.training_sources and pair.performance.split != 'train'
               for pair in self.pairs.values()):
            raise ValueError('Score scaler train source appears in an evaluation split')
        self.items, self.excluded = {}, []
        for w in all_windows:
            reason = None
            if w['tail']:
                reason = 'tail'
            elif not w['N']:
                reason = 'empty_window'
            pair = self.pairs[w['performance_id']]
            rows = np.asarray(w['row_indices'], int)
            if not reason and not pair.performance.mask[rows].any():
                reason = 'no_observed_reconstruction'
            if reason:
                self.excluded.append({'window_id': w['window_id'], 'split': w['split'], 'reason': reason})
                continue
            start, stop = np.searchsorted(pair.score.onset_q, [w['q_start'], w['q_end']], side='left')
            segment = NotePerformanceSegment(pair, NoteScoreSegment(pair.score, int(start), int(stop), int(start), int(stop)))
            actual = np.flatnonzero((pair.score.note_to_onset >= start) & (pair.score.note_to_onset < stop))
            if (actual.tolist() != w['row_indices'] or [pair.performance.row_keys[i] for i in actual] != w['score_note_keys']
                    or w['R_fingerprint'] != pair.performance.fingerprint or pair.score.source_sha256 != w['score_version_id']
                    or pair.score.traversal_id != w['repeat_route_id']):
                raise ValueError('AE window/S/R occurrence or version correspondence mismatch')
            if w['window_id'] in self.items:
                raise ValueError('Corpus repeats a window ID')
            self.items[w['window_id']] = {'window': w, 'segment': segment}
        self.collate = partial(collate_ae_windows, r_scaler=self.r_scaler, score_scale=self.score_scale)

    def dataset(self, split):
        return NoteAEWindows([v for _, v in sorted(self.items.items()) if v['window']['split'] == split])

    @property
    def contract(self):
        return {'windows_id': self.windows['id'],
                'validation_windows_id': self.validation_windows['id'] if self.validation_windows else None,
                'R_scaler': self.r_scaler.fingerprint, 'score_scale': stable_id(self.score_scale.__dict__),
                'split': {pid: pair.performance.split for pid, pair in sorted(self.pairs.items())},
                'eligible_windows': sorted(self.items)}


class NoteAEWindows(Dataset):
    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_ae_windows(items, *, r_scaler, score_scale):
    batch = collate_note_performance([item['segment'] for item in items], r_scaler, score_scale=score_scale)
    batch['R_raw'] = batch.pop('R')
    batch['R'] = batch.pop('R_scaled')
    batch['P'] = batch['note_mask']
    batch['M'] = batch['R_mask'] & batch['P'][..., None]
    batch['lengths'] = batch['P'].sum(1)
    batch['window_metadata'] = [item['window'] for item in items]
    batch['R_state'] = 'train-channel-scaled-once'
    batch['score_scale_id'] = stable_id(score_scale.__dict__)
    batch['beat_occurrence_keys'] = []
    for i, item in enumerate(items):
        if batch['R_metadata'][i]['row_keys'] != item['window']['score_note_keys']:
            raise ValueError('Padded AE rows disagree with window note keys')
        batch['beat_occurrence_keys'].append(item['window']['beat_keys'])
    return batch


class FixedNoteTriplets(Dataset):
    def __init__(self, corpus, triplets, profiles, scaler, calibration):
        config = check_artifact(triplets, 'triplets')
        for obj, kind in ((profiles, 'profiles'), (scaler, 'profile-scaler'), (calibration, 'distance-calibration')):
            check_artifact(obj, kind, config)
        if (profiles['parents']['windows'] != corpus.windows['id']
                or scaler['parents']['profiles'] != profiles['id']
                or calibration['parents'] != {'profiles': profiles['id'], 'scaler': scaler['id']}
                or triplets['parents'] != {'profiles': profiles['id'], 'scaler': scaler['id'], 'calibration': calibration['id']}):
            raise ValueError('Fixed g/Triplet provenance does not match the AE corpus')
        lookup = {p['window']['window_id']: scale_profile(p, scaler) for p in profiles['profiles']}
        self.rows, self.corpus = [], corpus
        for t in triplets['triplets']:
            ids = [t[role + '_window_id'] for role in ('anchor', 'positive', 'negative')]
            if any(key not in corpus.items or key not in lookup for key in ids):
                raise ValueError('Triplet references an unavailable full AE window')
            a, p, n = [lookup[key] for key in ids]
            if (any(corpus.items[key]['window'] != profile['window'] for key, profile in zip(ids, (a, p, n)))
                    or [a['profile_id'], p['profile_id'], n['profile_id']] != t['profile_ids']
                    or any(profile['reference_id'] != t['reference_id'] for profile in (a, p, n))
                    or t['scaler_id'] != scaler['id'] or t['calibration_id'] != calibration['id'] or t['split'] != 'train'):
                raise ValueError('Triplet identities or frozen provenance disagree')
            signature = anchor_signature(a, scaler)
            fixed = calibration['signatures'].get(signature_key(signature))
            dap, _ = profile_distance(a, p, signature)
            dan, _ = profile_distance(a, n, signature)
            if (list(signature) != t['J_a'] or not fixed or fixed['status'] != 'valid'
                    or t['delta_g'] != fixed['delta_g'] or t['tau_near'] > fixed['global_near']
                    or t['tau_far'] < fixed['global_far'] or not np.isclose(dap, t['D_ap'])
                    or not np.isclose(dan, t['D_an']) or not np.isclose(dan - dap, t['distance_gap'])):
                raise ValueError('Triplet metric signature, distances or calibration disagree')
            validate_triplet(a, p, n, signature, near=t['tau_near'], far=t['tau_far'], gap=t['delta_g'])
            self.rows.append(t)
        self.manifest_id = triplets['id']
        self.dependencies = {**triplets['parents'], 'reference': profiles['parents']['reference'], 'triplets': triplets['id']}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        t = self.rows[index]
        return {**{role: self.corpus.items[t[role + '_window_id']] for role in ('anchor', 'positive', 'negative')},
                'triplet_id': t['triplet_id']}

    def collate(self, items):
        return {**{role: self.corpus.collate([item[role] for item in items]) for role in ('anchor', 'positive', 'negative')},
                'triplet_ids': [item['triplet_id'] for item in items]}


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
