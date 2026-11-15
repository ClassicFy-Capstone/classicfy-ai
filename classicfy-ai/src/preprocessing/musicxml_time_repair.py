"""Opt-in, audited repair of quantized MusicXML tuplet timing.

The ordinary reader deliberately remains strict.  This helper produces a separate
in-memory tree; it never changes the source file or silently clamps a negative
cursor.  Only a measure that actually underflows may be repaired.  Repairs use
written tuplet ratios or a bounded full-voice backup corroborated by another voice.
"""

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from math import lcm


_TYPES = {
    'maxima': Fraction(32), 'long': Fraction(16), 'breve': Fraction(8),
    'whole': Fraction(4), 'half': Fraction(2), 'quarter': Fraction(1),
    'eighth': Fraction(1, 2), '16th': Fraction(1, 4),
    '32nd': Fraction(1, 8), '64th': Fraction(1, 16),
    '128th': Fraction(1, 32), '256th': Fraction(1, 64),
    '512th': Fraction(1, 128), '1024th': Fraction(1, 256),
}
_MAX_UNDERFLOW_Q = Fraction(1, 16)
_MAX_DIVISIONS_MULTIPLIER = 1000000


def _written_tuplet_duration(note):
    if note.find('grace') is not None:
        return None
    modification = note.find('time-modification')
    width = _TYPES.get(note.findtext('type'))
    if modification is None or width is None:
        return None
    try:
        actual = int(modification.findtext('actual-notes'))
        normal = int(modification.findtext('normal-notes'))
    except (TypeError, ValueError):
        return None
    if actual <= 0 or normal <= 0:
        return None
    dots = len(note.findall('dot'))
    return width * (2 - Fraction(1, 2 ** dots)) * Fraction(normal, actual)


def _scan(entries, replacement=None):
    """Replay MusicXML time in quarter units, without tolerance or clamping."""
    replacement = replacement or {}
    cursor = extent = Fraction(0)
    chord_at = None
    underflows, backup_positions = [], {}
    for index, (element, divisions) in enumerate(entries):
        if element.tag == 'note':
            grace = element.find('grace') is not None
            duration = (Fraction(0) if grace else replacement.get(
                index, Fraction(element.findtext('duration', '0')) / divisions))
            if element.find('chord') is None:
                chord_at = cursor
                cursor += duration
            elif chord_at is None:
                raise ValueError('Timing repair cannot interpret an orphan chord')
            extent = max(extent, cursor, (chord_at or Fraction(0)) + duration)
        elif element.tag in {'backup', 'forward'}:
            duration = Fraction(element.findtext('duration')) / divisions
            cursor += duration if element.tag == 'forward' else -duration
            if element.tag == 'backup':
                backup_positions[index] = cursor
            if cursor < 0:
                underflows.append((index, cursor))
            extent = max(extent, cursor)
            chord_at = None
    return extent, underflows, backup_positions


def _bounded_full_voice_backup(entries, nominal):
    """Repair a full-measure backup only with independent voice corroboration.

    Some exports have unusable graphical note types or tiny extra adjustments.
    In that case keep every performed score duration and correct only a backup
    intended to reset a complete voice.  A separate voice must sum exactly to
    the declared meter, and the deficit must fit one source tick per preceding
    tuplet note as well as the absolute quarter-unit bound.
    """
    if nominal is None or nominal <= 0:
        return None
    totals = defaultdict(Fraction)
    tuplet_counts = defaultdict(int)
    quantization_budget = defaultdict(Fraction)
    for element, divisions in entries:
        if (element.tag == 'note' and element.find('chord') is None
                and element.find('grace') is None):
            voice = element.findtext('voice')
            if not voice:
                return None
            totals[voice] += Fraction(element.findtext('duration')) / divisions
    cursor = Fraction(0)
    last_voice = None
    changes = []
    for index, (element, divisions) in enumerate(entries):
        if element.tag == 'note' and element.find('chord') is None:
            last_voice = element.findtext('voice')
            if element.find('grace') is None:
                cursor += Fraction(element.findtext('duration')) / divisions
                if _written_tuplet_duration(element) is not None:
                    tuplet_counts[last_voice] += 1
                    quantization_budget[last_voice] += Fraction(1, 1) / divisions
        elif element.tag in {'backup', 'forward'}:
            width = Fraction(element.findtext('duration')) / divisions
            if element.tag == 'forward':
                cursor += width
                continue
            if cursor - width < 0:
                deficit = width - cursor
                corroborating = [v for v, duration in totals.items()
                                 if v != last_voice and duration == nominal]
                if (width != nominal or cursor <= 0 or deficit > _MAX_UNDERFLOW_Q
                        or not corroborating or tuplet_counts[last_voice] < 2
                        or deficit > quantization_budget[last_voice]):
                    return None
                changes.append({'element_index': index, 'original_ticks': str(width * divisions),
                                'corrected_ticks': str(cursor * divisions),
                                'original_divisions': str(divisions), 'deficit_q': str(deficit),
                                'quantization_budget_q': str(quantization_budget[last_voice]),
                                'corroborating_voices': corroborating})
                cursor = Fraction(0)
            else:
                cursor -= width
    if not changes:
        return None
    check = [(deepcopy(e), d) for e, d in entries]
    for change in changes:
        check[change['element_index']][0].find('duration').text = change['corrected_ticks']
    extent, underflows, _ = _scan(check)
    return changes if not underflows and extent == nominal else None


def normalize_quantized_timing(root, *, prefer_preserve_durations=False):
    """Return ``(deep_copied_root, audit_records)`` for a namespace-free tree.

    Exact-duration repair requires a real backup underflow, small source error,
    written tuplet durations within one source tick, exact repaired backup anchors,
    and an exact declared-meter or existing full-voice duration anchor.  The
    alternative full-voice backup correction requires an independent voice equal
    to the meter and a tuplet-derived quantization budget.  A rejected measure's
    timing is unchanged and receives a ``quantized_timing_rejected`` audit record.
    Applied changes preserve note identity and use an integer divisions grid;
    normal parsing/alignment must still validate the resulting score.

    ``prefer_preserve_durations`` first tries the independently corroborated
    full-voice backup correction.  It leaves source note durations and divisions
    unchanged where that correction is justified, avoiding unnecessarily large
    MIDI PPQ denominators.  It does not weaken either correction's evidence checks;
    measures without a full-voice witness still use the exact-duration method.
    """
    normalized = deepcopy(root)
    audit = []
    for part in normalized.findall('part'):
        divisions = Fraction(1)
        nominal = None
        updates = []
        part_audit = []
        for measure_index, measure in enumerate(part.findall('measure')):
            entries = []
            for index, element in enumerate(measure):
                if element.tag == 'attributes':
                    if element.find('divisions') is not None:
                        divisions = Fraction(element.findtext('divisions'))
                        if divisions <= 0:
                            raise ValueError('Timing repair requires positive divisions')
                    for time in element.findall('time'):
                        if time.get('number') is not None:
                            continue
                        beats, beat_type = time.findtext('beats'), time.findtext('beat-type')
                        try:
                            candidate = sum(Fraction(v) for v in beats.split('+')) * 4 / Fraction(beat_type)
                        except (AttributeError, ValueError, ZeroDivisionError):
                            candidate = None
                        nominal = candidate
                entries.append((element, divisions))
            _, underflows, _ = _scan(entries)
            if not underflows:
                continue
            record = {
                'part': part.get('id'), 'measure_index': measure_index,
                'measure_number': measure.get('number'),
                'first_underflow_q': str(underflows[0][1]),
            }
            if -underflows[0][1] > _MAX_UNDERFLOW_Q:
                audit.append({**record, 'action': 'quantized_timing_rejected',
                              'reason': 'large_cursor_inconsistency'})
                continue
            if prefer_preserve_durations:
                backup_changes = _bounded_full_voice_backup(entries, nominal)
                if backup_changes is not None:
                    for change in backup_changes:
                        updates.append((entries[change['element_index']][0].find('duration'),
                                        Fraction(change['corrected_ticks'])))
                    part_audit.append({**record, 'action': 'quantized_full_voice_backup_repair',
                                       'changes': backup_changes, 'measure_duration_q': str(nominal),
                                       'repair_preference': 'preserve_source_note_durations'})
                    continue
            replacements, changes = {}, []
            for index, (element, div) in enumerate(entries):
                if element.tag != 'note':
                    continue
                exact = _written_tuplet_duration(element)
                if exact is None:
                    continue
                ticks = Fraction(element.findtext('duration'))
                exact_ticks = exact * div
                if ticks != exact_ticks and ticks > 0 and abs(ticks - exact_ticks) <= 1:
                    replacements[index] = exact
                    changes.append({'element_index': index, 'note_id': element.get('id'),
                                    'original_ticks': str(ticks), 'exact_ticks': str(exact_ticks),
                                    'original_divisions': str(div), 'exact_duration_q': str(exact)})
            if not replacements:
                reason = 'no_written_tuplet_explains_source_quantization'
            else:
                new_extent, new_underflows, anchors = _scan(entries, replacements)
                # The bad backups must return exactly to the beginning.  Tiny
                # remaining discrepancies are not sufficient evidence to repair.
                explained = all(anchors.get(index) == 0 for index, _ in underflows)
                full_voice_anchors = {
                    Fraction(entries[i][0].findtext('duration')) / entries[i][1]
                    for i, position in anchors.items() if position == 0
                }
                anchored = (nominal is not None and new_extent == nominal
                            or new_extent in full_voice_anchors)
                reason = ('remaining_negative_cursor' if new_underflows else
                          'backup_not_exactly_explained' if not explained else
                          'no_exact_measure_duration_anchor' if not anchored else None)
            if reason:
                backup_changes = _bounded_full_voice_backup(entries, nominal)
                if backup_changes is None:
                    audit.append({**record, 'action': 'quantized_timing_rejected', 'reason': reason})
                    continue
                for change in backup_changes:
                    updates.append((entries[change['element_index']][0].find('duration'),
                                    Fraction(change['corrected_ticks'])))
                part_audit.append({**record, 'action': 'quantized_full_voice_backup_repair',
                                   'changes': backup_changes, 'measure_duration_q': str(nominal),
                                   'exact_tuplet_repair_unavailable': reason})
                continue
            for index, exact in replacements.items():
                element, div = entries[index]
                updates.append((element.find('duration'), exact * div))
            part_audit.append({**record, 'action': 'quantized_tuplet_duration_repair',
                               'changes': changes, 'measure_duration_q': str(new_extent),
                               'verified_backup_elements': [i for i, _ in underflows]})
        if not updates:
            continue
        factor = 1
        for _, ticks in updates:
            factor = lcm(factor, ticks.denominator)
        for element in part.iter():
            if element.tag in {'divisions', 'duration', 'offset'} and element.text:
                factor = lcm(factor, Fraction(element.text).denominator)
            for attr in (('attack', 'release') if element.tag == 'note' else
                         ('divisions',) if element.tag == 'sound' else ()):
                if attr in element.attrib:
                    factor = lcm(factor, Fraction(element.get(attr)).denominator)
        if factor > _MAX_DIVISIONS_MULTIPLIER:
            audit.extend({**r, 'action': 'quantized_timing_rejected',
                          'reason': 'divisions_multiplier_limit'} for r in part_audit)
            continue
        for element, ticks in updates:
            element.text = str(ticks)
        # Every time in a part remains physically unchanged except the audited
        # duration corrections.  Sound tempo and pitch values are never scaled.
        if factor != 1:
            for element in part.iter():
                if element.tag in {'divisions', 'duration', 'offset'} and element.text:
                    scaled = Fraction(element.text) * factor
                    assert scaled.denominator == 1
                    element.text = str(scaled.numerator)
                if element.tag == 'note':
                    for attr in ('attack', 'release'):
                        if attr in element.attrib:
                            element.set(attr, str(Fraction(element.get(attr)) * factor))
                if element.tag == 'sound' and 'divisions' in element.attrib:
                    element.set('divisions', str(Fraction(element.get('divisions')) * factor))
        audit.extend({**r, 'part_divisions_multiplier': factor} for r in part_audit)
    return normalized, audit
