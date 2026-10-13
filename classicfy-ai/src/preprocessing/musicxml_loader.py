"""Read written score structure; do not unfold repeats or infer phrases.

Positions are exact quarter-note units on each part's written timeline, not
ASAP beat indices or performed seconds. Measures advance by their encoded
extent, including forward movements, without filling omitted durations.
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as ET


Attributes = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class PitchSpelling:
    step: str
    alter: Fraction
    octave: int

    @property
    def midi_pitch(self) -> Fraction:
        """Retain fractional alterations rather than rounding microtones."""
        return 12 * (self.octave + 1) + dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)[self.step] + self.alter


@dataclass(frozen=True)
class SlurMark:
    note_id: str
    kind: str
    number: str
    attributes: Attributes


@dataclass(frozen=True)
class MusicXMLNote:
    note_id: str
    xml_id: str | None
    part_id: str
    measure_index: int
    measure_number: str
    note_index: int
    onset_quarters: Fraction
    measure_offset_quarters: Fraction
    duration_quarters: Fraction
    pitch: PitchSpelling | None
    voice: str | None
    staff: int | None
    is_rest: bool
    is_measure_rest: bool
    is_grace: bool
    is_chord: bool
    is_cue: bool
    note_type: str | None
    dots: int
    accidental: str | None
    grace_attributes: Attributes
    ties: tuple[Attributes, ...]
    slurs: tuple[SlurMark, ...]
    notation_xml: tuple[str, ...]

    @property
    def offset_quarters(self) -> Fraction:
        return self.onset_quarters + self.duration_quarters


@dataclass(frozen=True)
class ScoreMeasure:
    part_id: str
    measure_index: int
    number: str
    implicit: bool
    non_controlling: bool
    onset_quarters: Fraction
    duration_quarters: Fraction
    nominal_duration_quarters: Fraction | None
    attributes: Attributes


@dataclass(frozen=True)
class KeySignatureChange:
    part_id: str
    measure_index: int
    onset_quarters: Fraction
    staff: int | None
    fifths: int | None
    mode: str | None
    alterations: tuple[tuple[str, Fraction], ...]
    source_xml: str


@dataclass(frozen=True)
class TimeSignatureChange:
    part_id: str
    measure_index: int
    onset_quarters: Fraction
    staff: int | None
    groups: tuple[tuple[str, int], ...]
    senza_misura: bool
    source_xml: str

    @property
    def duration_quarters(self) -> Fraction | None:
        if self.senza_misura:
            return None
        return sum((Fraction(sum(_integer(x, 'beats', positive=True) for x in beats.split('+')) * 4, unit)
                    for beats, unit in self.groups), Fraction(0))


@dataclass(frozen=True)
class BarlineMark:
    part_id: str
    measure_index: int
    onset_quarters: Fraction
    location: str
    kind: str
    attributes: Attributes
    text: str | None


@dataclass(frozen=True)
class StructureDiagnostic:
    code: str
    part_id: str
    measure_index: int
    detail: str


@dataclass(frozen=True)
class ScoreStructure:
    source_path: Path
    version: str | None
    part_names: tuple[tuple[str, str | None], ...]
    notes: tuple[MusicXMLNote, ...]
    measures: tuple[ScoreMeasure, ...]
    key_signatures: tuple[KeySignatureChange, ...]
    time_signatures: tuple[TimeSignatureChange, ...]
    barline_marks: tuple[BarlineMark, ...]
    diagnostics: tuple[StructureDiagnostic, ...]

    @property
    def slurs(self) -> tuple[SlurMark, ...]:
        return tuple(mark for note in self.notes for mark in note.slurs)


def _attrs(node: ET.Element | None) -> Attributes:
    return tuple(sorted(node.attrib.items())) if node is not None else ()


def _text(node: ET.Element, name: str) -> str | None:
    value = node.findtext(name)
    return value.strip() if value is not None else None


def _fraction(value: str | None, label: str, *, positive: bool = False) -> Fraction:
    try:
        result = Fraction(value) if value is not None else None
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if result is None or (positive and result <= 0):
        raise ValueError(f"Invalid {label}: {value!r}")
    return result


def _integer(value: str | None, label: str, *, positive: bool = False) -> int:
    number = _fraction(value, label, positive=positive)
    if number.denominator != 1:
        raise ValueError(f"Invalid integer {label}: {value!r}")
    return int(number)


def _staff(node: ET.Element, *, attribute: bool = False) -> int | None:
    value = node.get('number') if attribute else _text(node, 'staff')
    return _integer(value, 'staff', positive=True) if value is not None else None


def _read_pitch(node: ET.Element) -> PitchSpelling:
    step = _text(node, 'step')
    octave = _integer(_text(node, 'octave'), 'octave')
    if step not in 'C D E F G A B'.split() or not 0 <= octave <= 9:
        raise ValueError('Invalid spelled pitch')
    alter = _fraction(_text(node, 'alter') or '0', 'alter')
    return PitchSpelling(step, alter, octave)


def _read_key(node: ET.Element, part_id: str, index: int, onset: Fraction) -> KeySignatureChange:
    fifths = _text(node, 'fifths')
    steps = [_text(n, '.') for n in node.findall('key-step')]
    alters = [_text(n, '.') for n in node.findall('key-alter')]
    if len(steps) != len(alters) or any(s not in 'C D E F G A B'.split() for s in steps):
        raise ValueError('Invalid non-traditional key signature')
    if fifths is None and not steps:
        raise ValueError('Key signature has neither fifths nor key-step')
    return KeySignatureChange(part_id, index, onset, _staff(node, attribute=True),
        _integer(fifths, 'fifths') if fifths is not None else None, _text(node, 'mode'),
        tuple((s, _fraction(a, 'key-alter')) for s, a in zip(steps, alters)),
        ET.tostring(node, encoding='unicode'))


def _read_time(node: ET.Element, part_id: str, index: int, onset: Fraction) -> TimeSignatureChange:
    beats = [_text(n, '.') for n in node.findall('beats')]
    units = [_text(n, '.') for n in node.findall('beat-type')]
    senza = node.find('senza-misura') is not None
    if len(beats) != len(units) or (not beats and not senza):
        raise ValueError('Invalid time signature')
    for group in beats:
        for count in group.split('+'):
            _integer(count, 'beats', positive=True)
    return TimeSignatureChange(part_id, index, onset, _staff(node, attribute=True),
        tuple((b, _integer(u, 'beat-type', positive=True)) for b, u in zip(beats, units)),
        senza, ET.tostring(node, encoding='unicode'))


def _read_note(node: ET.Element, part_id: str, index: int, number: str, ordinal: int,
               measure_start: Fraction, cursor: Fraction, divisions: Fraction | None,
               anchor: MusicXMLNote | None) -> MusicXMLNote:
    grace = node.find('grace')
    chord = node.find('chord') is not None
    rest = node.find('rest')
    pitch = node.find('pitch')
    if (rest is None) == (pitch is None) or node.find('unpitched') is not None:
        raise ValueError('Expected exactly one pitched note or rest; unpitched notes are unsupported')
    if grace is not None:
        if node.find('duration') is not None:
            raise ValueError('Grace note must not encode a duration')
        duration = Fraction(0)
    else:
        if divisions is None:
            raise ValueError('Missing divisions before note duration')
        duration = _fraction(_text(node, 'duration'), 'duration', positive=True) / divisions
    voice = _text(node, 'voice')
    if chord:
        if anchor is None or anchor.is_rest or rest is not None:
            raise ValueError('Chord has no preceding pitched anchor')
        if anchor.is_grace != (grace is not None) or (
                anchor.voice is not None and voice is not None and anchor.voice != voice):
            raise ValueError('Chord anchor has a different grace state or voice')
        if duration > anchor.duration_quarters:
            raise ValueError('Chord duration exceeds its anchor duration')
        offset = anchor.measure_offset_quarters
    else:
        offset = cursor
    note_id = f'{part_id}:m{index}:n{ordinal}'
    slurs = []
    for mark in node.findall('notations/slur'):
        kind = mark.get('type')
        if kind not in ('start', 'stop', 'continue'):
            raise ValueError('Invalid slur type')
        number_value = mark.get('number', '1')
        _integer(number_value, 'slur number', positive=True)
        slurs.append(SlurMark(note_id, kind, number_value, _attrs(mark)))
    return MusicXMLNote(
        note_id=note_id, xml_id=node.get('id'), part_id=part_id,
        measure_index=index, measure_number=number, note_index=ordinal,
        onset_quarters=measure_start + offset, measure_offset_quarters=offset,
        duration_quarters=duration, pitch=_read_pitch(pitch) if pitch is not None else None,
        voice=voice, staff=_staff(node), is_rest=rest is not None,
        is_measure_rest=rest is not None and rest.get('measure') == 'yes',
        is_grace=grace is not None, is_chord=chord, is_cue=node.find('cue') is not None,
        note_type=_text(node, 'type'), dots=len(node.findall('dot')),
        accidental=_text(node, 'accidental'), grace_attributes=_attrs(grace),
        ties=tuple(_attrs(t) for t in node.findall('tie')), slurs=tuple(slurs),
        notation_xml=tuple(ET.tostring(n, encoding='unicode') for n in node.findall('notations')),
    )


def load_musicxml(path: str | Path) -> ScoreStructure:
    """Load uncompressed score-partwise MusicXML, preserving written order.

    Missing voice/staff/mode remain None. Slurs are raw marks, not inferred
    phrases or paired spans. Repeat/ending marks are stored without unfolding.
    Malformed or unsupported timing raises ValueError with file/part/measure.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file():
        raise ValueError(f'Not a MusicXML file: {source}')
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise ValueError(f'Could not parse uncompressed MusicXML: {source}: {exc}') from exc
    for node in root.iter():
        node.tag = node.tag.rsplit('}', 1)[-1]
    if root.tag != 'score-partwise':
        raise ValueError(f'Unsupported MusicXML root {root.tag!r}: {source}; expected score-partwise')
    parts = root.findall('part')
    if not parts:
        raise ValueError(f'No score parts: {source}')
    part_ids = [p.get('id') for p in parts]
    if any(not p for p in part_ids) or len(set(part_ids)) != len(part_ids):
        raise ValueError(f'Missing or duplicate part ID: {source}')
    names = {p.get('id'): _text(p, 'part-name') for p in root.findall('part-list/score-part')}
    notes, measures, keys, times, barlines, diagnostics = [], [], [], [], [], []
    xml_ids = set()
    for part in parts:
        part_id = part.get('id')
        divisions = None
        nominal = None
        measure_start = Fraction(0)
        part_measures = part.findall('measure')
        if not part_measures:
            raise ValueError(f'No measures: {source}, part {part_id}')
        for index, measure in enumerate(part_measures):
            number = measure.get('number')
            context = f'{source}, part {part_id}, measure {number!r} (index {index})'
            try:
                if number is None:
                    raise ValueError('Missing measure number')
                cursor, extent = Fraction(0), Fraction(0)
                anchor, ordinal = None, 0
                measure_times, pending_barlines = [], []
                for child in measure:
                    if cursor < 0 and child.tag != 'forward':
                        raise ValueError('Backup crosses the beginning of the measure without a compensating forward')
                    if child.tag == 'attributes':
                        value = _text(child, 'divisions')
                        if value is not None:
                            divisions = _fraction(value, 'divisions', positive=True)
                        for key in child.findall('key'):
                            keys.append(_read_key(key, part_id, index, measure_start + cursor))
                        for time in child.findall('time'):
                            signature = _read_time(time, part_id, index, measure_start + cursor)
                            times.append(signature)
                            measure_times.append(signature)
                            # Staff-specific polymeter is preserved but is not a shared grid.
                            if signature.staff in (None, 1):
                                nominal = signature.duration_quarters
                    elif child.tag in ('backup', 'forward'):
                        if divisions is None:
                            raise ValueError('Missing divisions before cursor movement')
                        distance = _fraction(_text(child, 'duration'), child.tag, positive=True) / divisions
                        cursor += -distance if child.tag == 'backup' else distance
                        if cursor < 0 and child.tag == 'backup':
                            # Some ASAP exports round tuplets down, then encode a
                            # larger backup immediately followed by a corrective
                            # forward. Retain both movements; never clamp to zero.
                            diagnostics.append(StructureDiagnostic('compensated_backup_underflow', part_id, index,
                                f'Backup cursor {cursor}; must be restored by forward before any event'))
                        extent = max(extent, cursor)
                        anchor = None
                    elif child.tag == 'note':
                        note = _read_note(child, part_id, index, number, ordinal,
                                          measure_start, cursor, divisions, anchor)
                        if note.xml_id is not None:
                            if note.xml_id in xml_ids:
                                raise ValueError(f'Duplicate XML note ID: {note.xml_id}')
                            xml_ids.add(note.xml_id)
                        notes.append(note)
                        ordinal += 1
                        extent = max(extent, note.measure_offset_quarters + note.duration_quarters)
                        if not note.is_chord:
                            cursor += note.duration_quarters
                            anchor = note
                    elif child.tag == 'barline':
                        location = child.get('location', 'right')
                        if location not in ('left', 'middle', 'right'):
                            raise ValueError('Invalid barline location')
                        for mark in child:
                            if mark.tag in ('repeat', 'ending', 'segno', 'coda'):
                                pending_barlines.append((location, cursor, mark))
                if cursor < 0:
                    raise ValueError('Backup crosses the beginning of the measure without a compensating forward')
                implicit = measure.get('implicit') == 'yes'
                non_controlling = measure.get('non-controlling') == 'yes'
                if extent == 0:
                    diagnostics.append(StructureDiagnostic('empty_measure', part_id, index,
                        'No encoded duration; the next written measure starts at the same position'))
                if nominal is not None and extent != nominal and not implicit:
                    diagnostics.append(StructureDiagnostic('measure_duration_differs_from_meter', part_id, index,
                        f'Encoded extent {extent}, nominal meter length {nominal}; no padding applied'))
                if any(t.onset_quarters != measure_start for t in measure_times):
                    diagnostics.append(StructureDiagnostic('mid_measure_meter_change', part_id, index,
                        'Meter changes inside the measure; consult the time signature events'))
                if len({t.duration_quarters for t in measure_times}) > 1:
                    diagnostics.append(StructureDiagnostic('multiple_meter_lengths', part_id, index,
                        'Different meter lengths are encoded; no shared beat grid inferred'))
                measures.append(ScoreMeasure(part_id, index, number, implicit, non_controlling,
                    measure_start, extent, nominal, _attrs(measure)))
                for location, position, mark in pending_barlines:
                    offset = Fraction(0) if location == 'left' else extent if location == 'right' else position
                    barlines.append(BarlineMark(part_id, index, measure_start + offset, location,
                        mark.tag, _attrs(mark), (mark.text or '').strip() or None))
                measure_start += extent
            except ValueError as exc:
                raise ValueError(f'{context}: {exc}') from exc
    layouts = {tuple((m.number, m.onset_quarters, m.duration_quarters) for m in measures if m.part_id == pid)
               for pid in part_ids}
    if len(layouts) > 1:
        diagnostics.append(StructureDiagnostic('part_measure_layout_differs', '', -1,
            'Part timelines differ; positions are part-local and need validation before combining'))
    return ScoreStructure(source.resolve(), root.get('version'),
        tuple((pid, names.get(pid)) for pid in part_ids), tuple(notes), tuple(measures),
        tuple(keys), tuple(times), tuple(barlines), tuple(diagnostics))
