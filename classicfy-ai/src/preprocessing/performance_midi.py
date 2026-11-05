"""Raw piano note-off and CC64/67 streams, with explicit MIDI port/channel scope."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path

import mido


@dataclass(frozen=True)
class PerformanceNote:
    note_id: str
    pitch: int
    onset_sec: float
    key_offset_sec: float | None
    velocity: int
    port: int = 0
    channel: int = 0
    track: int = 0
    order: int = 0
    program: int = 0

    def __post_init__(self):
        if (not self.note_id or not math.isfinite(self.onset_sec) or self.onset_sec < 0
                or not 0 <= self.pitch <= 127 or not 1 <= self.velocity <= 127
                or not 0 <= self.channel <= 15 or self.port < 0):
            raise ValueError('Invalid raw performed note')
        if self.key_offset_sec is not None and (not math.isfinite(self.key_offset_sec) or self.key_offset_sec < self.onset_sec):
            raise ValueError('Key offset must be finite and not precede onset')


@dataclass(frozen=True)
class ControlEvent:
    time_sec: float
    number: int
    value: int
    port: int = 0
    channel: int = 0
    track: int = 0
    order: int = 0

    def __post_init__(self):
        if (not math.isfinite(self.time_sec) or self.time_sec < 0 or self.number not in (64, 67)
                or not 0 <= self.value <= 127 or not 0 <= self.channel <= 15 or self.port < 0):
            raise ValueError('Invalid piano pedal event')


@dataclass(frozen=True)
class PedalObservation:
    status: str = 'unknown'
    complete: bool = False
    initial_value: int | None = None
    source: str = 'not established'
    start_sec: float = 0.
    end_sec: float | None = None

    def __post_init__(self):
        if self.status not in {'observed', 'confirmed_absent', 'unknown'}:
            raise ValueError('Invalid pedal observation status')
        if not isinstance(self.complete, bool) or not self.source:
            raise ValueError('Pedal completeness and provenance must be explicit')
        if self.initial_value is not None and (isinstance(self.initial_value, bool) or not isinstance(self.initial_value, int) or not 0 <= self.initial_value <= 127):
            raise ValueError('Invalid initial CC value')
        if not math.isfinite(self.start_sec) or self.start_sec < 0:
            raise ValueError('Invalid pedal observation start')
        if self.end_sec is not None and (not math.isfinite(self.end_sec) or self.end_sec < self.start_sec):
            raise ValueError('Invalid pedal observation end')
        if self.status == 'confirmed_absent' and (not self.complete or self.initial_value not in (None, 0)):
            raise ValueError('Confirmed pedal absence requires complete observation and zero state')
        if self.status == 'confirmed_absent' and self.source == 'not established':
            raise ValueError('Confirmed pedal absence needs recording provenance')


@dataclass(frozen=True)
class PerformanceMidi:
    notes: tuple[PerformanceNote, ...]
    controls: tuple[ControlEvent, ...]
    duration_sec: float
    selected_channels: tuple[tuple[int, int], ...]
    pedal_observations: dict[tuple[int, int, int], PedalObservation] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        if not math.isfinite(self.duration_sec) or self.duration_sec < 0:
            raise ValueError('Invalid MIDI extent')
        scopes = set(self.selected_channels)
        if not scopes or len(scopes) != len(self.selected_channels):
            raise ValueError('Select unique MIDI port/channel scopes')
        if len({n.note_id for n in self.notes}) != len(self.notes):
            raise ValueError('Performed note IDs must be unique')
        if any((n.port, n.channel) not in scopes for n in self.notes) or any((c.port, c.channel) not in scopes for c in self.controls):
            raise ValueError('A note/controller lies outside selected piano channels')
        if any(n.onset_sec > self.duration_sec or n.key_offset_sec is not None and n.key_offset_sec > self.duration_sec for n in self.notes):
            raise ValueError('Note lies outside MIDI extent')
        if any(c.time_sec > self.duration_sec for c in self.controls):
            raise ValueError('Controller lies outside MIDI extent')
        for key, observation in self.pedal_observations.items():
            if key[:2] not in scopes or key[2] not in (64, 67) or not isinstance(observation, PedalObservation):
                raise ValueError('Invalid pedal observation scope')
            if observation.status == 'confirmed_absent' and any(
                    (c.port, c.channel, c.number) == key and c.value != 0 for c in self.controls):
                raise ValueError('Nonzero CC contradicts confirmed pedal absence')

    def observation(self, port, channel, number):
        return self.pedal_observations.get((port, channel, number), PedalObservation())


def _events(path):
    midi = mido.MidiFile(path)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError('Need a synchronous, PPQ-based MIDI file')
    events = []
    for track_index, track in enumerate(midi.tracks):
        tick, port = 0, 0
        for event_index, message in enumerate(track):
            tick += message.time
            if message.type == 'midi_port':
                port = message.port
            events.append((tick, track_index, event_index, port, message))
    events.sort(key=lambda item: item[:3])
    return midi.ticks_per_beat, events


def load_performance_midi(path, *, selected_channels=None, pedal_observations=None) -> PerformanceMidi:
    """Read raw key events; never lengthen note-off by pedal or merge distinct channels.

    With no explicit selection, exactly one piano port/channel must be present.
    Existing CC events establish an observed file stream, but never an initial
    pre-event value. An absent CC stream stays unknown unless explicitly confirmed.
    """
    path = Path(path)
    ppq, events = _events(path)
    tempo, last_tick, seconds = 500000, 0, 0.
    programs, active, notes, controls, diagnostics = {}, defaultdict(deque), [], [], []
    ambiguous_offsets = set()
    for order, (tick, track, event_index, port, message) in enumerate(events):
        seconds += mido.tick2second(tick - last_tick, ppq, tempo)
        last_tick = tick
        if message.type == 'set_tempo':
            if message.tempo <= 0:
                raise ValueError('Invalid MIDI tempo')
            tempo = message.tempo
        if not hasattr(message, 'channel'):
            continue
        channel = message.channel
        scope = (port, channel)
        if message.type == 'program_change':
            programs[scope] = message.program
        elif message.type == 'control_change' and message.control in (64, 67):
            controls.append(ControlEvent(seconds, message.control, message.value, port, channel, track, order))
        elif message.type == 'note_on' and message.velocity > 0:
            event = PerformanceNote(f'p{port}:c{channel}:t{track}:e{event_index}', message.note, seconds,
                                    None, message.velocity, port, channel, track, order, programs.get(scope, 0))
            queue = active[port, channel, message.note]
            if queue:
                ambiguous_offsets.update(n.note_id for n in queue)
                ambiguous_offsets.add(event.note_id)
                diagnostics.append(f'ambiguous_same_pitch_note_off:{event.note_id}')
            active[port, channel, message.note].append(event)
        elif message.type == 'note_off' or message.type == 'note_on' and message.velocity == 0:
            queue = active[port, channel, message.note]
            if queue:
                event = queue.popleft()
                notes.append(PerformanceNote(**{**event.__dict__, 'key_offset_sec': None if event.note_id in ambiguous_offsets else seconds}))
            else:
                diagnostics.append(f'orphan_note_off:p{port}:c{channel}:pitch{message.note}:order{order}')
    for queue in active.values():
        for note in queue:
            notes.append(note)
            diagnostics.append(f'unknown_key_offset:{note.note_id}')
    if selected_channels is None:
        scopes = sorted({(n.port, n.channel) for n in notes if n.channel != 9 and 0 <= n.program <= 7})
        if len(scopes) != 1:
            raise ValueError('Select piano port/channel scopes explicitly when zero or multiple piano channels exist')
        if any((n.port, n.channel) == scopes[0] and not 0 <= n.program <= 7 for n in notes):
            raise ValueError('Selected MIDI channel changes instrument family; select an explicit supported piano part')
    else:
        scopes = [tuple(scope) for scope in selected_channels]
        if any(len(scope) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in scope)
               or scope[0] < 0 or not 0 <= scope[1] <= 15 for scope in scopes):
            raise ValueError('Expected MIDI (port, channel) pairs')
    selected = set(scopes)
    notes = tuple(sorted((n for n in notes if (n.port, n.channel) in selected), key=lambda n: (n.onset_sec, n.order)))
    controls = tuple(c for c in controls if (c.port, c.channel) in selected)
    if not notes:
        raise ValueError('No note events in selected piano channels')
    observations = dict(pedal_observations or {})
    for port, channel in scopes:
        for number in (64, 67):
            key = (port, channel, number)
            if key not in observations and any((c.port, c.channel, c.number) == key for c in controls):
                observations[key] = PedalObservation('observed', True, None,
                                                      'complete raw MIDI event stream; initial state unestablished',
                                                      end_sec=seconds)
    return PerformanceMidi(notes, controls, seconds, tuple(scopes), observations,
                           {'midi_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'path': str(path),
                            'reader': 'mido_raw_note_off_v1', 'offset_semantics': 'key_release',
                            'pedal_extended_offsets': False, 'same_pitch_pairing': 'FIFO',
                            'selected_channels': [list(scope) for scope in scopes],
                            'channel_selection': 'single_piano_scope' if selected_channels is None else 'explicit',
                            'equal_tick_order': 'track_index_then_original_event_index'}, tuple(diagnostics))


def score_seconds_to_quarters(path, seconds):
    """Convert score-MIDI annotation seconds using its actual tempo events and PPQ."""
    import numpy as np
    ppq, events = _events(path)
    tempo, tick, time = 500000, 0, 0.
    anchors = [(0., 0., 60. / 120)]
    for at, _, _, _, message in events:
        if message.type != 'set_tempo':
            continue
        time += mido.tick2second(at - tick, ppq, tempo)
        tick, tempo = at, message.tempo
        if tempo <= 0:
            raise ValueError('Invalid score MIDI tempo')
        anchor = (time, tick / ppq, tempo / 1e6)
        if time == anchors[-1][0]:
            anchors[-1] = anchor
        else:
            anchors.append(anchor)
    source = np.asarray(seconds, dtype=np.float64)
    if source.ndim != 1 or not np.isfinite(source).all() or np.any(source < 0):
        raise ValueError('Expected nonnegative score annotation seconds')
    table = np.asarray(anchors)
    index = np.searchsorted(table[:, 0], source, side='right') - 1
    return table[index, 1] + (source - table[index, 0]) / table[index, 2]
