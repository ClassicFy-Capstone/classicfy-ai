"""Exact symbolic XML-to-MIDI bridge for automatic alignment validation.

This creates a new score representation, never asserts equivalence to a bundled
MIDI of another edition. Ambiguous unisons and unknown/grace durations have no
alias. Every alias is verified against MIDI ticks after a round trip.
"""
from collections import defaultdict
from fractions import Fraction
import hashlib
import math
from pathlib import Path

import mido

from .score_midi import read_score_midi
from .score_note_mapping import build_tie_mapping


def render_xml_score_bridge(xml, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    ties = build_tie_mapping(xml)
    events, skipped = [], defaultdict(int)
    durations = defaultdict(Fraction)
    for i, note in enumerate(xml.notes):
        durations[int(ties.owner_indices[i])] += note.duration
    for i, note in enumerate(xml.notes):
        if not ties.is_attack[i]:
            continue
        if note.grace or not ties.chain_known[i]:
            skipped['grace' if note.grace else 'unknown_tie_duration'] += 1
            continue
        # Sum source Fractions rather than round the floating-point tie table.
        duration = durations[i]
        events.append((note.pitch, note.onset, note.onset + duration, note.occurrence_id))
    if not events:
        raise ValueError('No known positive-length score attacks')
    ppq = math.lcm(480, *(q.denominator for event in events for q in event[1:3]))
    if ppq > 32767:
        raise ValueError('Exact XML MIDI bridge requires PPQ beyond the MIDI range')
    midi = mido.MidiFile(type=1, ticks_per_beat=ppq)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=500000)]))
    # Per-voice tracks prevent cross-voice same-pitch FIFO from changing offsets.
    by_voice, lookup = defaultdict(list), defaultdict(list)
    notes_by_key = {n.occurrence_id: n for n in xml.notes}
    for pitch, onset, end, key in events:
        n = notes_by_key[key]
        by_voice[n.part, n.voice].append((pitch, onset, end))
        lookup[pitch, float(onset), float(end)].append(key)
    for items in by_voice.values():
        timed = []
        for pitch, onset, end in items:
            timed.extend([(int(onset * ppq), 1, pitch), (int(end * ppq), 0, pitch)])
        track, previous = mido.MidiTrack(), 0
        for tick, on, pitch in sorted(timed):
            track.append(mido.Message('note_on' if on else 'note_off', note=pitch,
                                      velocity=64 if on else 0, time=tick - previous))
            previous = tick
        midi.tracks.append(track)
    out.parent.mkdir(parents=True, exist_ok=True)
    midi.save(out)
    notes, _, diagnostics = read_score_midi(out)
    if diagnostics['unclosed_notes'] or diagnostics['orphan_offsets'] or diagnostics['zero_length_notes']:
        raise ValueError('XML score MIDI round trip produced malformed events')
    aliases = {}
    for note in notes:
        matches = lookup.get((note.pitch, note.onset_beats, note.offset_beats), ())
        if len(matches) == 1:
            aliases[note.note_id] = matches[0]
    if not aliases:
        raise ValueError('XML score MIDI round trip has no unambiguous exact aliases')
    # Same-pitch overlapping voices can change FIFO durations; do not silently accept.
    encoded = {(n.pitch, n.onset_beats, n.offset_beats) for n in notes}
    if encoded != set(lookup):
        raise ValueError('XML score MIDI changed pitch/onset/offset during its round trip')
    return {'xml_sha256': xml.source_sha256, 'score_midi_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
            'traversal_id': xml.traversal_id, 'score_coordinate_unit': 'quarter',
            'score_id_to_occurrence': aliases, 'alignment_accepted': False,
            'bridge_policy': 'new canonical MIDI rendered from XML; exact ticks/pitch/offset roundtrip',
            'ambiguous_unisons': sum(len(v) != 1 for v in lookup.values()), 'excluded': dict(skipped),
            'midi_diagnostics': diagnostics}
