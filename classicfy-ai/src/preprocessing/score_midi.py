"""Symbolic score MIDI reading, independent of optional alignment engines."""

from pathlib import Path

import mido

from .note_alignment import ScoreNote


def read_score_midi(path: str | Path):
    """Return quarter-position notes, encoded extent, and reading diagnostics.

    Keep the ATEPP loader's FIFO same-pitch pairing and exact cross-track
    duplicate removal. Unclosed and zero-length notes are reported, not given
    invented offsets. MIDI cannot recover XML tie/grace/voice semantics.
    """
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError('Expected a synchronous score MIDI with positive PPQ')
    notes = []
    end_tick = zero_length = unclosed = orphan_offsets = 0
    for track in midi.tracks:
        tick, active = 0, {}
        for msg in track:
            tick += msg.time
            if msg.type not in ('note_on', 'note_off') or getattr(msg, 'channel', 0) == 9:
                continue
            key = (getattr(msg, 'channel', 0), msg.note)
            if msg.type == 'note_on' and msg.velocity > 0:
                active.setdefault(key, []).append(tick)
            elif active.get(key):
                start = active[key].pop(0)
                if tick > start:
                    notes.append((msg.note, start / midi.ticks_per_beat, tick / midi.ticks_per_beat))
                else:
                    zero_length += 1
            else:
                orphan_offsets += 1
        unclosed += sum(len(values) for values in active.values())
        end_tick = max(end_tick, tick)
    notes.sort(key=lambda n: (n[1], n[0], n[2]))
    original = len(notes)
    notes = list(dict.fromkeys(notes))
    result = tuple(ScoreNote(str(i), p, a, b, ()) for i, (p, a, b) in enumerate(notes))
    diagnostics = dict(ppq=midi.ticks_per_beat, positive_notes_before_deduplication=original,
        duplicate_notes_removed=original-len(notes), zero_length_notes=zero_length,
        unclosed_notes=unclosed, orphan_offsets=orphan_offsets)
    return result, end_tick / midi.ticks_per_beat, diagnostics


def load_score_midi(path: str | Path):
    """Read symbolic quarter-note positions from ticks, ignoring tempo/velocity."""
    return read_score_midi(path)[0]
