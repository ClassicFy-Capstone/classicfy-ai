"""Train-only global scaling of deterministic R; tempo is counted once per beat."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .note_performance import R_CHANNELS, R_SCHEMA_VERSION, NotePerformanceFeatures, note_performance_schema


@dataclass(frozen=True)
class NotePerformanceScaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    valid_counts: tuple[int, ...]
    trained_channel_mask: tuple[bool, ...]
    channel_status: tuple[str, ...]
    fit_manifest_id: str
    training_records: tuple[dict, ...]
    schema_version: str = R_SCHEMA_VERSION

    def __post_init__(self):
        fields = (self.mean, self.scale, self.valid_counts, self.trained_channel_mask, self.channel_status)
        if self.schema_version != R_SCHEMA_VERSION or any(len(field) != 11 for field in fields):
            raise ValueError('Invalid R scaler schema')
        if (not isinstance(self.fit_manifest_id, str) or not self.fit_manifest_id or not self.training_records or not np.isfinite(self.mean).all()
                or not np.isfinite(self.scale).all() or np.any(np.asarray(self.scale) <= 0)):
            raise ValueError('R scaler needs explicit train provenance and finite positive scales')
        for index, (count, trained, status) in enumerate(zip(self.valid_counts, self.trained_channel_mask, self.channel_status)):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0 or not isinstance(trained, bool) or trained != (count > 0):
                raise ValueError('R scaler counts and trained-channel mask disagree')
            if status not in {'fitted', 'constant', 'unobserved'} or (status == 'unobserved') != (count == 0):
                raise ValueError('Invalid R scaler channel status')
            if count == 0 and (self.mean[index] != 0 or self.scale[index] != 1):
                raise ValueError('Unobserved train channels require mean=0 and scale=1')
        if any(record.get('split') != 'train' for record in self.training_records):
            raise ValueError('R scaler provenance contains non-train records')

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def _convert(self, values, mask, *, inverse):
        values, mask = np.asarray(values, dtype=float), np.asarray(mask)
        if values.ndim < 2 or values.shape[-1] != 11 or mask.shape != values.shape or mask.dtype != bool:
            raise ValueError('Expected matching R[...,11] and boolean observation mask')
        if not np.isfinite(values[mask]).all():
            raise ValueError('Valid R values must be finite')
        clean = np.where(mask, values, 0)
        with np.errstate(over='ignore', invalid='ignore'):
            result = clean * self.scale + self.mean if inverse else (clean - self.mean) / self.scale
        result = np.where(mask, result, 0)
        if not inverse:
            with np.errstate(over='ignore'):
                result = result.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('R scaling exceeds model floating point range')
        return result

    def transform(self, values, mask=None):
        if isinstance(values, NotePerformanceFeatures):
            if mask is not None:
                raise ValueError('A feature record already provides its R mask')
            values, mask = values.values, values.mask
        return self._convert(values, mask, inverse=False)

    def inverse_transform(self, values, mask):
        return self._convert(values, mask, inverse=True)

    def transform_beats(self, features):
        result = np.zeros(len(features.beats.valid), np.float32)
        valid = features.beats.valid
        result[valid] = ((features.beats.log2_qpm[valid] - self.mean[0]) / self.scale[0]).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Beat scaling overflows float32')
        return result

    def to_dict(self):
        return {'schema': note_performance_schema(), 'schema_version': self.schema_version,
                'channels': list(R_CHANNELS), 'mean': list(self.mean), 'scale': list(self.scale),
                'valid_counts': list(self.valid_counts), 'trained_channel_mask': list(self.trained_channel_mask),
                'channel_status': list(self.channel_status), 'fit_manifest_id': self.fit_manifest_id,
                'fit_split': 'train', 'training_records': list(self.training_records),
                'variance': 'population', 'tempo_weight': 'one valid beat per recording',
                'constant_policy': 'exact zero variance -> scale 1', 'unobserved_policy': 'mean 0, scale 1, trained false'}


def validate_work_splits(records):
    by_work = {}
    for record in records:
        work = record.work_key if isinstance(record, NotePerformanceFeatures) else record['work_key']
        split = record.split if isinstance(record, NotePerformanceFeatures) else record['split']
        if not work or split not in {'train', 'validation', 'test', 'unassigned'}:
            raise ValueError('Explicit work and recognized split are required')
        if split == 'unassigned':
            continue
        if work in by_work and by_work[work] != split:
            raise ValueError(f'Work crosses dataset splits: {work}')
        by_work[work] = split


def fit_note_performance_scaler(training_records, *, fit_manifest_id):
    records = list(training_records)
    if not records or not isinstance(fit_manifest_id, str) or not fit_manifest_id:
        raise ValueError('Supply explicit train records and a fit manifest identifier')
    if any(not isinstance(record, NotePerformanceFeatures) or record.split != 'train' for record in records):
        raise ValueError('Only explicitly labeled train records may fit the R scaler')
    validate_work_splits(records)
    unique = {}
    for record in records:
        key = record.dataset, record.record_key
        if key in unique and unique[key].fingerprint != record.fingerprint:
            raise ValueError(f'Conflicting duplicate train recording: {key}')
        unique[key] = record
    count, mean, m2 = np.zeros(11, np.int64), np.zeros(11), np.zeros(11)
    provenance = []
    for record in unique.values():
        for channel in range(11):
            values = (record.beats.log2_qpm[record.beats.valid] if channel == 0
                      else record.values[record.mask[:, channel], channel])
            if not len(values):
                continue
            block_mean = float(np.mean(values))
            block_m2 = float(np.sum((values - block_mean) ** 2))
            total = int(count[channel]) + len(values)
            delta = block_mean - mean[channel]
            m2[channel] += block_m2 + delta * delta * int(count[channel]) * len(values) / total
            mean[channel] += delta * len(values) / total
            count[channel] = total
        provenance.append({'dataset': record.dataset, 'record_key': record.record_key, 'work_key': record.work_key,
                           'split': record.split, 'fingerprint': record.fingerprint,
                           'score_ref': record.score_ref, 'midi_sha256': record.provenance.get('performance', {}).get('midi_sha256')})
    variance = np.divide(m2, count, out=np.zeros(11), where=count > 0)
    scale = np.sqrt(variance)
    status = tuple('unobserved' if n == 0 else 'constant' if sd == 0 else 'fitted' for n, sd in zip(count, scale))
    scale[scale == 0] = 1
    return NotePerformanceScaler(tuple(map(float, mean)), tuple(map(float, scale)), tuple(map(int, count)),
                                  tuple(bool(n) for n in count), status, fit_manifest_id,
                                  tuple(sorted(provenance, key=lambda record: (record['dataset'], record['record_key']))))


def save_note_performance_scaler(scaler, path):
    payload = scaler.to_dict()
    payload['fingerprint'] = scaler.fingerprint
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load_note_performance_scaler(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('schema') != note_performance_schema() or payload.get('channels') != list(R_CHANNELS) or payload.get('fit_split') != 'train':
        raise ValueError('R scaler schema/channels or fit split mismatch')
    scaler = NotePerformanceScaler(*(tuple(payload[name]) for name in
                                    ('mean', 'scale', 'valid_counts', 'trained_channel_mask', 'channel_status')),
                                   payload['fit_manifest_id'], tuple(payload['training_records']), payload['schema_version'])
    if payload.get('fingerprint') != scaler.fingerprint:
        raise ValueError('R scaler integrity mismatch')
    return scaler
