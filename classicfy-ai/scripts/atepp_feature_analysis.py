"""ATEPP adapter for feature normalization and embedding analysis.

Quarter-note score positions are retained; Tempo uses a fixed 120 BPM reference
as in extend_atepp.py. Automatic alignment quality is not ASAP's robust flag.
"""
from collections import defaultdict
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from features import (BeatSequence, PieceFeatureInput, TempoInput,
                      extract_piece_tempo_features, extract_piece_rubato_features,
                      separate_piece_feature, standardize_piece_feature,
                      fit_residual_scale, standardize_sequence)
from validate_embedding import CHANNELS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_source(value, root):
    """Rebase a cache from another checkout without guessing a filename."""
    path = Path(value)
    if path.is_file():
        return path
    parts = str(value).replace('\\', '/').split('/')
    if root.name in parts:
        candidate = root.joinpath(*parts[parts.index(root.name) + 1:])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(value)


def load_records(root, audit_path):
    audit = pd.read_csv(audit_path, dtype={'perf_id': str, 'composition_id': str}).fillna('')
    accepted = audit[audit.status == 'aligned']
    if 'aligner' not in accepted or not (accepted.aligner == 'dual_dtw').all():
        raise ValueError('This analysis requires a homogeneous DualDTW audit')
    records, manifest = [], []
    score_hashes = {}
    for row in accepted.to_dict('records'):
        key = row['perf_id']
        raw_path = resolve_source(row['raw_cache'], root)
        score_path = resolve_source(row['local_score'], root)
        if str(score_path) not in score_hashes:
            score_hashes[str(score_path)] = sha256(score_path)
        score_hash = score_hashes[str(score_path)]
        with np.load(raw_path, allow_pickle=False) as z:
            record = {name: z[name].copy() for name in z.files}
        grid = record['score_beats']
        if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
                or not np.allclose(np.diff(grid), 1., rtol=0, atol=1e-10)):
            raise ValueError(f'{key}: expected strictly increasing quarter-note grid')
        if record['dpa'].shape != (len(grid) - 1, 5):
            raise ValueError(f'{key}: unexpected feature shape')
        if record['performance_beats'].shape != grid.shape:
            raise ValueError(f'{key}: performance grid shape mismatch')
        if not np.isfinite(record['performance_beats']).all():
            raise ValueError(f'{key}: nonfinite performance anchors')
        if int(row['pedal_events']) == 0:
            record['dpa'][:, 2:] = np.nan
            record['dpa_mask'][:, 2:] = False
        record.update(key=key, piece=f"ATEPP/{row['composition_id']}/{score_hash[:12]}",
                      composer=row['composer'], artist=row['artist'], track=row['track'],
                      midi_path=str(resolve_source(row['local_midi'], root)),
                      raw_path=str(raw_path), source_score_sha256=score_hash,
                      alignment_path=str(resolve_source(row['alignment_cache'], root)),
                      midi_sha256=row['sha256'], pedal_events=int(row['pedal_events']))
        records.append(record)
        manifest.append({'key': key, 'piece': record['piece'], 'raw_sha256': sha256(raw_path),
                         'alignment_sha256': sha256(record['alignment_path']),
                         'score_sha256': score_hash, 'midi_sha256': row['sha256']})
    if not records:
        raise ValueError('No quality-accepted ATEPP alignments')
    ai = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *sorted((ai / 'src/features').glob('*.py')),
               *sorted((ai / 'src/preprocessing').glob('*.py'))]
    provenance = {'dataset': 'ATEPP-1.2', 'aligner': 'dual_dtw',
                  'audit_sha256': sha256(audit_path), 'source_files': manifest,
                  'analysis_code_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
                  'reference_tempo_bpm': 120, 'alignment_is_ground_truth': False,
                  'statuses': audit.status.value_counts().to_dict()}
    return records, provenance


def align_group(records):
    """Exact intersection of score positions, with no interpolation/truncation."""
    if len({r['piece'] for r in records}) != 1:
        raise ValueError('Different score identities cannot share a common pattern')
    start = max(r['score_beats'][0] for r in records)
    stop = min(r['score_beats'][-1] for r in records)
    reference = records[0]['score_beats']
    grid = reference[(reference >= start) & (reference <= stop)]
    if len(grid) < 2:
        raise ValueError('No shared score interval')
    trimmed = []
    for r in records:
        left, right = np.searchsorted(r['score_beats'], [grid[0], grid[-1]])
        np.testing.assert_array_equal(r['score_beats'][left:right + 1], grid)
        rr = dict(r)
        rr.update(score_beats=grid.copy(), performance_beats=r['performance_beats'][left:right + 1].copy(),
                  dpa=r['dpa'][left:right].copy(), dpa_mask=r['dpa_mask'][left:right].copy(),
                  retained_grid_fraction=(len(grid) - 1) / (len(r['score_beats']) - 1))
        trimmed.append(rr)
    return trimmed


def normalize_group(records):
    """Return all four stages, masks/support and fitted scales for seven channels."""
    rs = align_group(records)
    grid, piece = rs[0]['score_beats'], rs[0]['piece']
    inputs = [TempoInput(r['key'], grid * .5, r['performance_beats'],
                         ['b'] * len(grid), ['b'] * len(grid)) for r in rs]
    tempo = extract_piece_tempo_features(inputs)
    rubato = extract_piece_rubato_features(tempo)
    features, scales = {}, []
    for index, channel in enumerate(CHANNELS):
        if index < 2:
            fields = {}
            for r in rs:
                key = r['key']
                t, rb = tempo[key], rubato[key]
                fields[key] = ((BeatSequence([i.score_relative_tempo if i.mask else np.nan for i in t.intervals],
                                             [i.mask for i in t.intervals]), t.common_tempo_sequence,
                                t.individual_tempo_sequence, t.common_tempo_support) if index == 0 else
                               (rb.absolute_rubato_sequence, rb.common_rubato_sequence,
                                rb.relative_rubato_sequence, rb.common_rubato_support))
            scale = fit_residual_scale([v[2] for v in fields.values()], method='mad')
            features[channel] = {k: dict(raw=v[0], common=v[1], relative=v[2],
                                        standardized=standardize_sequence(v[2], scale), support=v[3])
                                 for k, v in fields.items()}
        else:
            # dpa stores D,A,depth,down_ratio,changes; masks must survive trimming.
            separated = separate_piece_feature([
                PieceFeatureInput(r['key'], piece, grid,
                                  BeatSequence(r['dpa'][:, index - 2], r['dpa_mask'][:, index - 2])) for r in rs])
            fitted = standardize_piece_feature(list(separated.values()), feature_name=channel)
            scale = next(iter(fitted.values())).scale
            features[channel] = {k: {**{s: getattr(v, s) for s in ('raw', 'common', 'relative', 'standardized')},
                                     'support': v.common_support} for k, v in fitted.items()}
        scales.append({'piece': piece, 'channel': channel, **asdict(scale)})
    return {'records': rs, 'features': features, 'scales': scales, 'tempo': tempo, 'rubato': rubato,
            'piece': piece, 'grid': sha256_bytes(grid.tobytes())[:12], 'tempo_inputs': inputs}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def collect_normalized(records, args):
    by_piece = defaultdict(list)
    for r in records:
        by_piece[r['piece']].append(r)
    normalized, groups, audits, scales = [], [], [], []
    for piece, rs in sorted(by_piece.items()):
        rs = sorted(rs, key=lambda r: r['key'])
        audit = {'piece': piece, 'grid': '', 'aligned_candidates': len(rs), 'accepted': False,
                 'reason': '', 'shared_beats': 0, 'total_intervals': 0, 'shared_fraction': 0.}
        if len(rs) < 2:
            audit['reason'] = 'single_performance'
            audits.append(audit)
            continue
        try:
            result = normalize_group(rs)
        except ValueError as exc:
            if str(exc) != 'No shared score interval':
                raise
            audit['reason'] = 'no_shared_score_interval'
            audits.append(audit)
            continue
        normalized.append(result)
        scales.extend(result['scales'])
        audit.update(grid=result['grid'], total_intervals=len(result['records'][0]['score_beats']) - 1)
        if len(rs) < args.min_performances:
            audit['reason'] = 'too_few_performances'
        else:
            shared = np.logical_and.reduce([v['standardized'].mask & np.isfinite(v['standardized'].values)
                                            for f in result['features'].values() for v in f.values()])
            audit.update(shared_beats=int(shared.sum()), shared_fraction=float(shared.mean()))
            if shared.sum() < args.min_beats or shared.mean() < args.min_coverage:
                audit['reason'] = 'too_few_shared_valid_beats'
            else:
                audit['accepted'] = True
                groups.append({'piece': piece, 'grid': result['grid'], 'keys': [r['key'] for r in rs],
                               'values': np.array([[result['features'][c][r['key']]['standardized'].values[shared]
                                                    for c in CHANNELS] for r in rs]),
                               'shared_indices': np.flatnonzero(shared)})
        audits.append(audit)
    if len(groups) < 2:
        raise ValueError('Need at least two eligible works for cross-work analysis')
    return normalized, groups, audits, scales
