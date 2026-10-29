"""Read symbolic MusicXML without inferring performance or expression values."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import math
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile


PARSER_VERSION = 'musicxml-notes-v1'
_PITCH = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


@dataclass(frozen=True)
class Meter:
    groups: tuple[int, ...]
    denominator: int

    @property
    def numerator(self):
        return sum(self.groups)

    @property
    def duration(self):
        return Fraction(4 * self.numerator, self.denominator)


@dataclass(frozen=True)
class XMLNote:
    source_id: str
    xml_id: str | None
    part: str
    voice: str
    staff: str
    measure: int
    measure_label: str
    local_onset: Fraction
    duration: Fraction
    pitch: int | None
    grace: bool
    grace_after: bool
    flags: tuple[str, ...]
    ties: tuple[str, ...]
    slurs: tuple[tuple[str, str], ...]
    beams: tuple[tuple[str, str], ...]
    meter: Meter | None = None
    fifths: int | None = None
    mode: str | None = None
    occurrence_id: str = ''
    occurrence: int = 0
    visit: int = 0
    onset: Fraction = Fraction(0)
    written_onset: Fraction = Fraction(0)
    voice_known: bool = True
    pitch_known: bool = True
    document_order: int = 0
    written_pitch: int | None = None
    chord_group: int = 0
    source_locator: str = ''

    @property
    def offset(self):
        return self.onset + self.duration


@dataclass(frozen=True)
class XMLDirection:
    part: str
    staff: str | None
    voice: str | None
    measure: int
    position: Fraction
    kind: str
    token: str
    number: str = '1'
    value: float | None = None
    source_id: str = ''
    system: bool = False


@dataclass(frozen=True)
class MeasureVisit:
    measure: int
    occurrence: int
    onset: Fraction
    duration: Fraction
    reset: bool


@dataclass(frozen=True)
class MusicXMLScore:
    notes: tuple[XMLNote, ...]
    rests: tuple[XMLNote, ...]
    directions: tuple[XMLDirection, ...]
    visits: tuple[MeasureVisit, ...]
    measure_durations: tuple[Fraction, ...]
    implicit_measures: tuple[bool, ...]
    source_sha256: str
    traversal_id: str
    diagnostics: tuple[str, ...]


def _text(node, path, default=None):
    child = node.find(path)
    return default if child is None or child.text is None else child.text.strip()


def _fraction(text, name):
    try:
        result = Fraction(text)
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f'Invalid {name}: {text!r}') from exc
    return result


def _root(path):
    data = path.read_bytes()
    # Some released ATEPP .mxl files contain ordinary, uncompressed XML.
    if path.suffix.lower() == '.mxl' and data.startswith(b'PK'):
        with ZipFile(path) as archive:
            container = ET.fromstring(archive.read('META-INF/container.xml'))
            roots = [n.get('full-path') for n in container.iter()
                     if n.tag.rsplit('}', 1)[-1] == 'rootfile'
                     and n.get('media-type') in (None, 'application/vnd.recordare.musicxml+xml')]
            if len(roots) != 1:
                raise ValueError('Expected one MusicXML rootfile in MXL')
            member = PurePosixPath(roots[0])
            if member.is_absolute() or '..' in member.parts:
                raise ValueError('Invalid MXL rootfile path')
            xml = archive.read(str(member))
    else:
        xml = data
    root = ET.fromstring(xml)
    for element in root.iter():
        element.tag = element.tag.rsplit('}', 1)[-1]
    if root.tag != 'score-partwise':
        raise ValueError('Only score-partwise MusicXML is supported')
    return root, hashlib.sha256(data).hexdigest()


def _meter(node):
    beats, types = node.findall('beats'), node.findall('beat-type')
    if len(beats) != 1 or len(types) != 1:
        return None
    try:
        groups = tuple(int(x) for x in beats[0].text.split('+'))
        denominator = int(types[0].text)
    except (ValueError, TypeError, AttributeError):
        return None
    if not groups or min(groups) < 1 or denominator < 1:
        raise ValueError('Invalid MusicXML meter')
    return Meter(groups, denominator)


def _directions(node, part, measure, position, divisions, source_id):
    offset = _fraction(_text(node, 'offset', '0'), 'direction offset') / divisions
    at = position + offset
    scope = dict(part=part, staff=_text(node, 'staff'), voice=_text(node, 'voice'),
                 measure=measure, position=at, source_id=source_id,
                 system=node.get('system') is not None)
    result = []
    for direction in node.findall('direction-type'):
        for item in direction:
            if item.tag == 'words':
                text = ''.join(item.itertext()).strip()
                if re.search(r'\b(?:d\s*\.\s*[cs]\b|da\s+capo\b|dal\s+segno\b|al\s+coda\b|fine\b)', text, re.IGNORECASE):
                    raise ValueError('Text navigation requires a supported unfolded score edition')
                result.append(XMLDirection(**scope, kind='words', token=text))
            elif item.tag == 'dynamics':
                for mark in item:
                    token = ''.join(mark.itertext()).strip() if mark.tag == 'other-dynamics' else mark.tag
                    result.append(XMLDirection(**scope, kind='dynamic', token=token))
            elif item.tag == 'wedge':
                result.append(XMLDirection(**scope, kind='wedge', token=item.get('type', ''),
                                           number=item.get('number', '1')))
            elif item.tag == 'metronome':
                units = {'whole': 4, 'half': 2, 'quarter': 1, 'eighth': .5,
                         '16th': .25, '32nd': .125}
                unit = units.get(_text(item, 'beat-unit'))
                try:
                    bpm = float(_text(item, 'per-minute'))
                except (TypeError, ValueError):
                    bpm = None
                if unit is not None and bpm is not None and math.isfinite(bpm) and bpm > 0:
                    dots = len(item.findall('beat-unit-dot'))
                    bpm *= unit * (2 - 2 ** (-dots))
                else:
                    bpm = None
                result.append(XMLDirection(**scope, kind='metronome', token='metronome', value=bpm))
            elif item.tag in {'segno', 'coda'}:
                raise ValueError(f'Unsupported navigation mark: {item.tag}')
    sound = node.find('sound')
    if sound is not None:
        if any(k in sound.attrib for k in ('dacapo', 'dalsegno', 'tocoda', 'fine')):
            raise ValueError('D.C./D.S./coda navigation requires an explicit supported score edition')
        # Playback dynamics/tempo are not substitutes for printed expression.
    return result


def _repeat_path(count, forwards, backwards, endings):
    """Unfold non-nested repeats and numbered endings; reject ambiguous nesting."""
    pairs, start = {}, None
    for index in range(count):
        if index in forwards:
            if start is not None:
                raise ValueError('Nested repeats require an explicit measure traversal')
            start = index
        if index in backwards:
            left = 0 if start is None else start
            pairs[index] = (left, backwards[index])
            start = None
    if start is not None:
        raise ValueError('Unclosed forward repeat')
    path, passes = [], Counter()
    index, active_pass = 0, 1
    while index < count:
        if len(path) > count * 32:
            raise ValueError('Repeat traversal exceeds supported bound')
        allowed = endings.get(index)
        if allowed is None or active_pass in allowed:
            path.append(index)
        if index in pairs:
            left, total = pairs[index]
            passes[index] += 1
            if passes[index] < total:
                active_pass = passes[index] + 1
                index = left
                continue
        index += 1
        if index in forwards:
            active_pass = 1
    return path


def read_musicxml(path: str | Path, *, traversal: tuple[int, ...] | None = None) -> MusicXMLScore:
    """Parse source notes and unfold a zero-based measure traversal, preserving IDs."""
    path = Path(path)
    root, digest = _root(path)
    parts = root.findall('part')
    if not parts:
        raise ValueError('MusicXML contains no parts')
    measure_count = len(parts[0].findall('measure'))
    if not measure_count or any(len(p.findall('measure')) != measure_count for p in parts):
        raise ValueError('Parts must have corresponding measures')
    notes, directions, diagnostics, attributes = [], [], [], []
    extents = [Fraction(0)] * measure_count
    implicit = [False] * measure_count
    nominal = [Fraction(0)] * measure_count
    forwards, backwards, endings = set(), {}, {}
    ids, part_ids = set(), set()
    for pi, part in enumerate(parts):
        part_id = part.get('id', f'part-{pi}')
        if part_id in part_ids:
            raise ValueError(f'Duplicate part ID: {part_id}')
        part_ids.add(part_id)
        divisions = Fraction(1)
        current_meter = None
        active_ending = None
        for mi, measure in enumerate(part.findall('measure')):
            cursor, extent, chord_at, chord_group = Fraction(0), Fraction(0), None, 0
            implicit[mi] |= measure.get('implicit') == 'yes'
            for ei, element in enumerate(measure):
                locator = f'{part_id}/measure[{mi}]/element[{ei}]'
                if element.tag == 'attributes':
                    if element.find('divisions') is not None:
                        divisions = _fraction(_text(element, 'divisions'), 'divisions')
                        if divisions <= 0:
                            raise ValueError('Divisions must be positive')
                    for time in element.findall('time'):
                        meter = _meter(time)
                        attributes.append((part_id, mi, cursor, 'meter', time.get('number'), meter))
                        if time.get('number') is None:
                            current_meter = meter
                    for key in element.findall('key'):
                        fifths = _text(key, 'fifths')
                        value = (int(fifths) if fifths is not None else None, _text(key, 'mode'))
                        attributes.append((part_id, mi, cursor, 'key', key.get('number'), value))
                    for trans in element.findall('transpose'):
                        value = int(_text(trans, 'chromatic', '0')) + 12 * int(_text(trans, 'octave-change', '0'))
                        attributes.append((part_id, mi, cursor, 'transpose', trans.get('number'), value))
                elif element.tag in {'backup', 'forward'}:
                    width = _fraction(_text(element, 'duration'), element.tag) / divisions
                    if width < 0:
                        raise ValueError('Negative cursor duration')
                    cursor += width if element.tag == 'forward' else -width
                    if cursor < 0:
                        raise ValueError('MusicXML backup precedes measure start')
                    extent, chord_at = max(extent, cursor), None
                elif element.tag == 'direction':
                    directions.extend(_directions(element, part_id, mi, cursor, divisions, locator))
                elif element.tag == 'sound' and any(k in element.attrib for k in ('dacapo', 'dalsegno', 'tocoda', 'fine')):
                    raise ValueError('Unsupported sound navigation')
                elif element.tag == 'note':
                    xml_id = element.get('id')
                    if xml_id is not None:
                        if xml_id in ids:
                            raise ValueError(f'Duplicate MusicXML note ID: {xml_id}')
                        ids.add(xml_id)
                    grace = element.find('grace') is not None
                    duration = Fraction(0) if grace else _fraction(_text(element, 'duration'), 'note duration') / divisions
                    if duration < 0 or (duration == 0 and not grace):
                        raise ValueError(f'Invalid note duration at {locator}')
                    chord = element.find('chord') is not None
                    if chord and chord_at is None:
                        raise ValueError(f'Chord without preceding note at {locator}')
                    onset = chord_at if chord else cursor
                    if not chord:
                        chord_group = len(notes)
                    staff, voice = _text(element, 'staff', '1'), _text(element, 'voice')
                    pitch_node = element.find('pitch')
                    pitch, pitch_known = None, True
                    if pitch_node is not None:
                        alter = _fraction(_text(pitch_node, 'alter', '0'), 'pitch alter')
                        if alter.denominator != 1:
                            raise ValueError('Microtonal MIDI pitch is unsupported')
                        step = _text(pitch_node, 'step')
                        if step not in _PITCH:
                            raise ValueError('Invalid pitch step')
                        pitch = 12 * (int(_text(pitch_node, 'octave')) + 1) + _PITCH[step] + int(alter)
                        if not 0 <= pitch <= 127:
                            raise ValueError('MIDI pitch outside 0..127')
                    elif element.find('rest') is None:
                        raise ValueError('Unpitched notes are unsupported')
                    flags = tuple(n.tag for n in element.findall('notations/articulations/*'))
                    flags += tuple(n.tag for n in element.findall('notations/ornaments/*'))
                    flags += tuple(n.tag for n in element.findall('notations/*')
                                   if n.tag in {'fermata', 'arpeggiate'})
                    ties = tuple(n.get('type', '') for n in element.findall('tie'))
                    if any(n.get('time-only') is not None for n in element.findall('tie')):
                        raise ValueError('Repeat-dependent tie time-only is unsupported')
                    if not ties:
                        ties = tuple(n.get('type', '') for n in element.findall('notations/tied'))
                    slurs = tuple((n.get('number', '1'), n.get('type', '')) for n in element.findall('notations/slur'))
                    beams = tuple((n.get('number', '1'), (n.text or '').strip()) for n in element.findall('beam'))
                    grace_node = element.find('grace')
                    after = grace and grace_node.get('steal-time-previous') is not None
                    notes.append(XMLNote(xml_id or locator, xml_id, part_id, voice or '1', staff,
                                         mi, measure.get('number', str(mi + 1)), onset, duration, pitch,
                                         grace, after, flags, ties, slurs, beams,
                                         voice_known=voice is not None, pitch_known=pitch_known,
                                         document_order=len(notes), written_pitch=pitch, chord_group=chord_group,
                                         source_locator=locator))
                    if not chord:
                        chord_at = onset
                        cursor += duration
                    extent = max(extent, onset + duration, cursor)
                elif element.tag == 'barline' and pi == 0:
                    ending = element.find('ending')
                    if ending is not None:
                        if ending.get('type') == 'start':
                            try:
                                active_ending = frozenset(int(x.strip()) for x in ending.get('number', '').split(','))
                            except ValueError as exc:
                                raise ValueError('Unsupported ending numbers') from exc
                        elif ending.get('type') in {'stop', 'discontinue'}:
                            if active_ending:
                                endings[mi] = active_ending
                            active_ending = None
                    repeat = element.find('repeat')
                    if repeat is not None:
                        if repeat.get('direction') == 'forward':
                            forwards.add(mi)
                        elif repeat.get('direction') == 'backward':
                            total = int(repeat.get('times', '2'))
                            if not 2 <= total <= 16:
                                raise ValueError('Unsupported repeat count')
                            backwards[mi] = total
            if pi == 0 and active_ending:
                endings[mi] = active_ending
            extents[mi] = max(extents[mi], extent)
            if current_meter:
                nominal[mi] = max(nominal[mi], current_meter.duration)
    durations = tuple(e if e > 0 else n for e, n in zip(extents, nominal))
    if any(d <= 0 for d in durations):
        raise ValueError('Cannot determine an empty measure duration')
    starts, total = [], Fraction(0)
    for duration in durations:
        starts.append(total)
        total += duration
    # Resolve attributes by musical position, independently of voice document order.
    context = {}
    for part_id, mi, at, kind, staff, value in attributes:
        context.setdefault((part_id, kind, staff), []).append((starts[mi] + at, value))
    for events in context.values():
        events.sort(key=lambda item: item[0])
    def active(note, kind):
        q = starts[note.measure] + note.local_onset
        candidates = []
        for staff in (None, note.staff):
            for at, value in context.get((note.part, kind, staff), ()):
                if at <= q:
                    candidates.append((at, staff is not None, value))
        return max(candidates, key=lambda item: item[:2])[2] if candidates else None
    source_notes = []
    voice_defaults = {}
    for part_id in part_ids:
        part_notes = [note for note in notes if note.part == part_id]
        voices = {note.voice for note in part_notes if note.voice_known}
        if len({note.staff for note in part_notes}) == 1 and len(voices) <= 1:
            voice_defaults[part_id] = next(iter(voices), '1')
    for note in notes:
        if not note.voice_known and note.part in voice_defaults:
            note = replace(note, voice=voice_defaults[note.part], voice_known=True)
        key = active(note, 'key')
        transpose = active(note, 'transpose') or 0
        pitch = note.pitch + transpose if note.pitch is not None else None
        if pitch is not None and not 0 <= pitch <= 127:
            raise ValueError('Transposed MIDI pitch outside 0..127')
        source_notes.append(replace(note, meter=active(note, 'meter'),
                                    pitch=pitch,
                                    fifths=key[0] if key else None, mode=key[1] if key else None,
                                    written_onset=starts[note.measure] + note.local_onset))
    path_indices = list(traversal) if traversal is not None else _repeat_path(measure_count, forwards, backwards, endings)
    if not path_indices or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < measure_count for i in path_indices):
        raise ValueError('Traversal must contain valid zero-based measure indices')
    if path_indices[0] != 0:
        raise ValueError('A full traversal must start at measure zero; crop after feature extraction')
    counts, visits, unfolded, q = Counter(), [], [], Fraction(0)
    by_measure = {}
    for note in source_notes:
        by_measure.setdefault(note.measure, []).append(note)
    for vi, mi in enumerate(path_indices):
        counts[mi] += 1
        reset = vi > 0 and mi <= path_indices[vi - 1]
        visits.append(MeasureVisit(mi, counts[mi], q, durations[mi], reset))
        for note in by_measure.get(mi, ()):
            unfolded.append(replace(note, onset=q + note.local_onset, visit=vi,
                                    occurrence=counts[mi], occurrence_id=f'{note.source_id}@{counts[mi]}'))
        q += durations[mi]
    unfolded.sort(key=lambda n: (n.onset, n.visit, n.part, n.document_order))
    traversal_id = hashlib.sha256(','.join(map(str, path_indices)).encode()).hexdigest()
    return MusicXMLScore(tuple(n for n in unfolded if n.pitch is not None),
                         tuple(n for n in unfolded if n.pitch is None), tuple(directions),
                         tuple(visits), durations, tuple(implicit), digest, traversal_id, tuple(diagnostics))
