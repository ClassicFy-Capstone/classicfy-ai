"""Original note R11/S extraction restricted to an entire selected clip."""
from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
import pickle

import numpy as np

from features.note_performance import extract_note_performance
from features.note_score_context import NoteScoreFeatures, encode_note_numeric
from features.performance_time import BeatTimeMap, mask_estimated_beat_intervals
from .note_alignment import NoteAlignment, ScoreNote, PerformedNote
from .note_performance_adapters import bind_alignment_to_raw_midi
from .performance_midi import load_performance_midi
from .score_midi import read_score_midi
from .score_note_mapping import TieMapping, NoteAlignmentLinks


@lru_cache(maxsize=2)
def load_bridge(path):
    # Only locally generated, hash-checked preparation artifacts are accepted.
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


@lru_cache(maxsize=2)
def midi_lookup(path):
    return {n.note_id:n for n in read_score_midi(path)[0]}


def mapped_quarters(q, source, target):
    q = np.asarray(q, float)
    result = np.interp(q, source, target)
    for side, i, j in [('left', 0, 1), ('right', -1, -2)]:
        select = q < source[i] if side == 'left' else q > source[i]
        result[select] = target[i] + (q[select] - source[i]) * (target[j] - target[i]) / (source[j] - source[i])
    return result


def slice_score_and_links(score, links, indices):
    indices = np.asarray(indices, int)
    first, last = int(score.note_to_onset[indices[0]]), int(score.note_to_onset[indices[-1]]) + 1
    local = {int(old):new for new, old in enumerate(indices)}
    owners = np.array([local.get(int(score.ties.owner_indices[i]), j) for j, i in enumerate(indices)])
    known = score.ties.chain_known[indices] & np.array([int(score.ties.owner_indices[i]) in local for i in indices])
    keys = tuple(score.ties.occurrence_ids[i] for i in indices)
    ties = TieMapping(keys, tuple(score.ties.chain_ids[i] for i in indices), owners,
                      score.ties.chain_duration_q[indices], known, score.ties.is_attack[indices])
    cut = NoteScoreFeatures(score.numeric[indices], score.categorical[indices], score.flags[indices],
        score.field_known[indices], score.note_to_onset[indices] - first, score.onset_q[first:last],
        tuple(score.metadata[i] for i in indices), ties, score.source_sha256, score.traversal_id,
        score.diagnostics)
    linked = NoteAlignmentLinks(tuple(links.performed_ids[i] for i in indices),
        tuple(links.status[i] for i in indices), links.attack_mask[indices],
        links.duration_mask[indices] & known, links.unresolved_score_ids, keys)
    return cut, linked


def prepare_clip(row, bridge_path, performance_path, alignment_path):
    xml, score, bridge = load_bridge(str(bridge_path))
    if hashlib.sha256(Path(performance_path).read_bytes()).hexdigest() != row['source_sha256']:
        raise ValueError('Original performance hash differs from selected dataset')
    performance = load_performance_midi(performance_path)
    start, stop = float(row['start_seconds']), float(row['end_seconds'])
    if not 0 <= start < stop <= performance.duration_sec + .01:
        raise ValueError('Selected clip bounds lie outside original performance')
    with np.load(alignment_path, allow_pickle=False) as z:
        cache = {k:z[k].copy() for k in z.files}
    q, seconds = cache['score_beats'], cache['performance_beats']
    converted = mapped_quarters(q, bridge['midi_anchor_q'], bridge['xml_anchor_q'])
    valid = cache.get('beat_valid', np.ones(len(q) - 1, bool)).copy()
    # Do not assign tempo through a large unsupported score-to-score gap.
    anchor_indices = np.searchsorted(bridge['midi_anchor_q'], (q[:-1] + q[1:]) / 2).clip(1, len(bridge['midi_anchor_q']) - 1)
    mq, xq = bridge['midi_anchor_q'], bridge['xml_anchor_q']
    gaps = np.maximum(np.diff(mq), np.diff(xq))[anchor_indices - 1]
    valid &= (gaps <= 16) & (q[:-1] >= mq[0] - 1) & (q[1:] <= mq[-1] + 4)
    beats = mask_estimated_beat_intervals(BeatTimeMap(converted, seconds, valid, {
        'granularity':'beat', 'source':'selected dataset alignment, verified symbolic MIDI/XML bridge',
        'estimated':True, 'alignment_sha256':hashlib.sha256(Path(alignment_path).read_bytes()).hexdigest(),
        'score_bridge_source_xml':bridge['source_xml_sha256'], 'clip_seconds':[start, stop]}))
    midi = midi_lookup(row['local_score'])
    matches, mapped_clip_rows, original_clip_matches = [], [], 0
    fields = ('score_ids', 'performance_ids', 'score_onsets', 'score_offsets',
              'performance_onsets', 'performance_offsets', 'pitch')
    for sid, pid, onset_q, offset_q, onset, offset, pitch in zip(*(cache[k] for k in fields)):
        if not start <= onset < stop:
            continue
        original_clip_matches += 1
        sid = str(sid)
        original = midi.get(sid)
        if original is None or original.pitch != pitch or abs(original.onset_beats - onset_q) > 1e-4:
            raise ValueError(f'Alignment cache does not match selected score MIDI at {sid}')
        xi = bridge['mapped_rows'].get(sid)
        if xi is None:
            continue
        note = xml.notes[xi]
        if note.pitch != pitch:
            raise ValueError('Pitch mismatch across score bridge')
        matches.append((ScoreNote(note.occurrence_id, int(pitch), float(note.onset), float(note.offset), ()),
                        PerformedNote(str(pid), int(pitch), float(onset), float(offset), 64)))
        mapped_clip_rows.append(xi)
    if not mapped_clip_rows:
        raise ValueError('No original aligned note in selected clip maps to MusicXML')
    links, binding = bind_alignment_to_raw_midi(xml, score, performance, NoteAlignment(tuple(matches), (), ()),
                                               verify_velocity=False)
    lower, upper = np.interp([start, stop], seconds, converted)
    note_q = score.onset_q[score.note_to_onset]
    in_range = np.flatnonzero((note_q >= lower) & (note_q < upper))
    indices = np.unique(np.r_[in_range, mapped_clip_rows])
    # Full onset groups, including tied/unmatched written notes and partial tails.
    lo, hi = note_q[indices].min(), note_q[indices].max()
    indices = np.flatnonzero((note_q >= lo) & (note_q <= hi))
    cut, linked = slice_score_and_links(score, links, indices)
    r = extract_note_performance(cut, performance, linked, beats, record_key=row['perf_id'],
        work_key='ATEPP/'+row['composition_id'], split='unassigned', dataset='ATEPP',
        provenance={'adapter':'selected-musicxml-note-r-v1', 'clip_seconds':[start, stop],
                    'binding':binding, 'no_performance_targets_outside_clip':True})
    mask=r.mask.copy()
    for i,meta in enumerate(r.metadata):
        onset=meta.get('onset_sec')
        if onset is not None and not start<=onset<stop:
            mask[i]=False
        if meta.get('key_offset_sec') is not None and meta['key_offset_sec']>stop:
            mask[i,[3,5,6,7,8,9]]=False
        if meta.get('next_onset_sec') is not None and meta['next_onset_sec']>stop:
            mask[i,[8,9]]=False
        bi=int(r.beat_id[i])
        if bi>=0 and (beats.seconds[bi]<start or beats.seconds[bi+1]>stop):
            mask[i,0]=False
    if not mask.any():
        raise ValueError('Selected clip has no observed note reconstruction targets')
    arrays = dict(numeric=encode_note_numeric(cut), categorical=cut.categorical, flags=cut.flags,
        field_known=cut.field_known, note_to_onset=cut.note_to_onset, onset_q=cut.onset_q,
        R=np.where(mask,r.values,0).astype(np.float32), M=mask, beat_id=r.beat_id,
        local_tau=r.local_tau, expected_duration_sec=r.expected_duration_sec,
        original_score_rows=indices)
    report = dict(perf_id=row['perf_id'], composition_id=row['composition_id'],
        source_sha256=row['source_sha256'], source_xml_sha256=score.source_sha256,
        score_midi_sha256=bridge['source_midi_sha256'], start_seconds=start, end_seconds=stop,
        notes=len(indices), onsets=len(cut.onset_q), original_clip_alignment_matches=original_clip_matches,
        xml_mapped_clip_matches=len(mapped_clip_rows), xml_mapping_fraction=len(mapped_clip_rows)/max(1,original_clip_matches),
        observed_notes=int(mask.any(1).sum()), observed_by_channel=mask.sum(0).tolist(),
        boundary_masked_by_channel=(r.mask&~mask).sum(0).tolist(),
        beat_intervals=len(beats.valid), valid_beat_intervals=int(beats.valid.sum()),
        binding=binding, source_score_cache=str(bridge_path), alignment_path=str(alignment_path),
        raw_performance_path=str(performance_path), traversal_id=score.traversal_id)
    return arrays, report
