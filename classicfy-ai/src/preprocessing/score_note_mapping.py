"""Shared written-note / sounding-event ownership and tie duration contract."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .musicxml_score import MusicXMLScore
from .note_alignment import NoteAlignment


@dataclass(frozen=True)
class TieMapping:
    occurrence_ids: tuple[str, ...]
    chain_ids: tuple[str, ...]
    owner_indices: np.ndarray
    chain_duration_q: np.ndarray
    chain_known: np.ndarray
    is_attack: np.ndarray

    def __post_init__(self):
        n = len(self.occurrence_ids)
        if len(set(self.occurrence_ids)) != n or len(self.chain_ids) != n:
            raise ValueError('Tie mapping requires unique occurrence IDs')
        for name, dtype in (('owner_indices', np.int64), ('chain_duration_q', float),
                            ('chain_known', bool), ('is_attack', bool)):
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if value.shape != (n,):
                raise ValueError('Tie mapping arrays must match note count')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if n and (np.any(self.owner_indices < 0) or np.any(self.owner_indices >= n)):
            raise ValueError('Invalid tie owner indices')
        if not np.isfinite(self.chain_duration_q).all() or np.any(self.chain_duration_q < 0):
            raise ValueError('Tie duration must be finite and nonnegative')
        if n and (np.any(self.owner_indices[self.owner_indices] != self.owner_indices)
                  or np.any(self.is_attack & (self.owner_indices != np.arange(n)))):
            raise ValueError('Tie owners must own themselves and all attack targets')


def build_tie_mapping(score: MusicXMLScore) -> TieMapping:
    notes = score.notes
    chains, active, invalid = {}, {}, set()
    for i, note in enumerate(notes):
        key = (note.part, note.voice, note.pitch)
        starts, stops = 'start' in note.ties, 'stop' in note.ties
        malformed = any(t not in {'start', 'stop'} for t in note.ties)
        if stops and key in active:
            owner = active.pop(key)
            previous = notes[chains[owner][-1]]
            if previous.offset != note.onset or not note.voice_known:
                invalid.add(owner)
        else:
            owner = i
            if stops:
                invalid.add(owner)
            if key in active:
                invalid.add(active.pop(key))
                invalid.add(owner)
        chains.setdefault(owner, []).append(i)
        if malformed:
            invalid.add(owner)
        if starts:
            if key in active:
                invalid.add(active[key])
                invalid.add(owner)
            active[key] = owner
    invalid.update(active.values())
    owners = np.full(len(notes), -1, np.int64)
    durations = np.zeros(len(notes))
    known = np.zeros(len(notes), bool)
    attack = np.zeros(len(notes), bool)
    chain_ids = [''] * len(notes)
    for owner, members in chains.items():
        duration = sum((notes[i].duration for i in members), Fraction(0))
        for i in members:
            owners[i] = owner
            durations[i] = float(duration)
            known[i] = owner not in invalid
            attack[i] = i == owner and 'stop' not in notes[i].ties
            chain_ids[i] = notes[owner].occurrence_id
    return TieMapping(tuple(n.occurrence_id for n in notes), tuple(chain_ids),
                      owners, durations, known, attack)


@dataclass(frozen=True)
class NoteAlignmentLinks:
    performed_ids: tuple[tuple[str, ...], ...]
    status: tuple[str, ...]
    attack_mask: np.ndarray
    duration_mask: np.ndarray
    unresolved_score_ids: tuple[str, ...]
    occurrence_ids: tuple[str, ...]

    def __post_init__(self):
        n = len(self.occurrence_ids)
        if len(self.performed_ids) != n or len(self.status) != n or len(set(self.occurrence_ids)) != n:
            raise ValueError('Alignment link identities and rows must agree')
        for name in ('attack_mask', 'duration_mask'):
            value = np.array(getattr(self, name), dtype=bool, copy=True)
            if value.shape != (n,):
                raise ValueError('Reconstruction masks must match written note count')
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.duration_mask & ~self.attack_mask):
            raise ValueError('Duration targets must belong to matched attacks')


def link_note_alignment(score: MusicXMLScore, ties: TieMapping, alignment: NoteAlignment,
                        *, score_id_to_occurrence: dict[str, str] | None = None) -> NoteAlignmentLinks:
    """Link exact IDs only; callers supply aliases for dataset-specific ID conventions.

    Attack masks belong to the attack owner only. Tied rows retain their score
    features, regardless of their reconstruction masks.
    """
    if ties.occurrence_ids != tuple(n.occurrence_id for n in score.notes):
        raise ValueError('Tie mapping belongs to a different score/traversal')
    lookup = {n.occurrence_id: i for i, n in enumerate(score.notes)}
    sources = defaultdict(list)
    for i, note in enumerate(score.notes):
        sources[note.source_id].append(i)
    for source, rows in sources.items():
        if len(rows) == 1:
            if source in lookup and lookup[source] != rows[0]:
                raise ValueError(f'A source ID collides with an occurrence ID: {source}')
            lookup[source] = rows[0]
    if score_id_to_occurrence is not None and not isinstance(score_id_to_occurrence, dict):
        raise ValueError('Score ID aliases must be a dictionary')
    if score_id_to_occurrence:
        for source, occurrence in score_id_to_occurrence.items():
            if occurrence not in lookup or occurrence not in ties.occurrence_ids:
                raise ValueError(f'Unknown occurrence alias target: {occurrence}')
            if source in lookup and lookup[source] != lookup[occurrence]:
                raise ValueError(f'Conflicting score ID alias: {source}')
            lookup[source] = lookup[occurrence]
    candidates, owner_candidates, unresolved = defaultdict(set), defaultdict(set), set()
    performed_notes = {}
    for snote, pnote in alignment.matches:
        if (not np.isfinite([pnote.onset, pnote.offset]).all() or pnote.offset < pnote.onset
                or not 0 <= pnote.velocity <= 127):
            raise ValueError(f'Invalid performed note: {pnote.note_id}')
        if pnote.note_id in performed_notes and performed_notes[pnote.note_id] != pnote:
            raise ValueError(f'Conflicting performed note ID: {pnote.note_id}')
        performed_notes[pnote.note_id] = pnote
        row = lookup.get(snote.note_id)
        if row is None:
            unresolved.add(snote.note_id)
            continue
        if snote.pitch != score.notes[row].pitch or pnote.pitch != score.notes[row].pitch:
            raise ValueError(f'Pitch mismatch at score ID {snote.note_id}')
        owner = int(ties.owner_indices[row])
        candidates[owner].add(pnote.note_id)
        if row == owner:
            owner_candidates[owner].add(pnote.note_id)
    event_owners = defaultdict(set)
    for owner, events in candidates.items():
        for event in events:
            event_owners[event].add(owner)
    masks = np.zeros(len(score.notes), bool)
    statuses, performed = [], []
    for i in range(len(score.notes)):
        owner = int(ties.owner_indices[i])
        events = tuple(sorted(candidates[owner]))
        performed.append(events)
        ambiguous = len(events) > 1 or any(len(event_owners[e]) > 1 for e in events)
        if ambiguous:
            status = 'ambiguous'
        elif not events:
            status = 'unmatched'
        elif not owner_candidates[owner]:
            status = 'unmatched_attack_owner'
        elif not ties.is_attack[i]:
            status = 'tie_continuation'
        else:
            status = 'matched_attack'
            masks[i] = True
        statuses.append(status)
    duration_mask = masks & ties.chain_known
    masks.setflags(write=False)
    duration_mask.setflags(write=False)
    return NoteAlignmentLinks(tuple(performed), tuple(statuses), masks, duration_mask, tuple(sorted(unresolved)), ties.occurrence_ids)
