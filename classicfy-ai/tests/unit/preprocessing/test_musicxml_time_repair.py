from fractions import Fraction
import xml.etree.ElementTree as ET

from preprocessing.musicxml_time_repair import normalize_quantized_timing
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, direction, measure, note, score_xml


def tuplet(actual=7, normal=2, kind='eighth', dots=0):
    return (f'<type>{kind}</type>' + '<dot/>' * dots
            + f'<time-modification><actual-notes>{actual}</actual-notes>'
            + f'<normal-notes>{normal}</normal-notes></time-modification>')


def seven_notes(*, chord=False, grace=False):
    result = note('grace', grace=True) if grace else ''
    for i in range(7):
        result += note(f't{i}', duration=68, extra=tuplet())
        if i == 0 and chord:
            result += note('chord', 'E', duration=68, extra=tuplet(), chord=True)
    return result


class MusicXMLTimeRepairTest(XMLFixture):
    def normalized(self, xml):
        source = ET.fromstring(xml)
        before = ET.tostring(source)
        normalized, audit = normalize_quantized_timing(source)
        self.assertEqual(ET.tostring(source), before, 'source tree was modified')
        return normalized, audit

    def test_exact_tuplet_repair_preserves_identity_and_rescales_offsets(self):
        xml = score_xml(measure(attributes(beats='1', divisions=480)
                    + direction('<dynamics><p/></dynamics>', offset=24)
                    + seven_notes() + '<backup><duration>480</duration></backup>'
                    + note('bass', duration=480, voice='2')))
        with self.assertRaisesRegex(ValueError, 'backup precedes'):
            self.read(xml)
        normalized, audit = self.normalized(xml)
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]['action'], 'quantized_tuplet_duration_repair')
        self.assertEqual(audit[0]['part_divisions_multiplier'], 7)
        score = self.read(ET.tostring(normalized, encoding='unicode'))
        upper = [n for n in score.notes if n.voice == '1']
        self.assertEqual([n.onset for n in upper], [Fraction(i, 7) for i in range(7)])
        self.assertEqual([n.duration for n in upper], [Fraction(1, 7)] * 7)
        self.assertEqual(next(n.onset for n in score.notes if n.source_id == 'bass'), 0)
        self.assertEqual(score.measure_durations, (Fraction(1),))
        self.assertEqual(score.directions[0].position, Fraction(1, 20))
        again, again_audit = normalize_quantized_timing(normalized)
        self.assertEqual(again_audit, [])
        self.assertEqual(ET.tostring(again), ET.tostring(normalized))

    def test_chord_does_not_advance_time_and_grace_keeps_zero_duration(self):
        xml = score_xml(measure(attributes(beats='1', divisions=480)
                    + seven_notes(chord=True, grace=True)
                    + '<backup><duration>480</duration></backup>'
                    + note('bass', duration=480, voice='2')))
        normalized, audit = self.normalized(xml)
        self.assertEqual(len(audit), 1)
        score = self.read(ET.tostring(normalized, encoding='unicode'))
        notes = {n.source_id: n for n in score.notes}
        self.assertEqual(notes['grace'].duration, 0)
        self.assertEqual(notes['chord'].onset, notes['t0'].onset)
        self.assertEqual(notes['chord'].duration, Fraction(1, 7))
        self.assertEqual(notes['t1'].onset, Fraction(1, 7))

    def test_large_inconsistency_is_left_rejected(self):
        xml = score_xml(measure(attributes(beats='4', divisions=480)
                    + seven_notes() + '<backup><duration>1920</duration></backup>'
                    + note('bass', duration=1920, voice='2')))
        normalized, audit = self.normalized(xml)
        self.assertEqual(audit[0]['reason'], 'large_cursor_inconsistency')
        self.assertEqual(ET.tostring(normalized), ET.tostring(ET.fromstring(xml)))
        with self.assertRaisesRegex(ValueError, 'backup precedes'):
            self.read(ET.tostring(normalized, encoding='unicode'))

    def test_small_error_without_tuplet_evidence_is_not_clamped(self):
        xml = score_xml(measure(attributes(beats='1', divisions=480)
                    + note('upper', duration=479)
                    + '<backup><duration>480</duration></backup>'
                    + note('bass', duration=480, voice='2')))
        normalized, audit = self.normalized(xml)
        self.assertEqual(audit[0]['action'], 'quantized_timing_rejected')
        self.assertEqual(ET.tostring(normalized), ET.tostring(ET.fromstring(xml)))

    def test_bounded_backup_uses_independent_voice_and_keeps_note_durations(self):
        # A real export pattern: rounded 18:8 tuplets plus an extraneous 3-tick
        # rewind.  Exact note normalization alone cannot explain the backup.
        upper = ''.join(note(f't{i}', duration=53, extra=tuplet(18, 8, '16th'))
                        for i in range(18))
        xml = score_xml(measure(attributes(beats='2', divisions=480) + upper
                    + '<backup><duration>3</duration></backup>'
                    + '<backup><duration>960</duration></backup>'
                    + note('bass', duration=960, voice='2')))
        normalized, audit = self.normalized(xml)
        self.assertEqual(audit[0]['action'], 'quantized_full_voice_backup_repair')
        self.assertEqual(audit[0]['changes'][0]['corrected_ticks'], '951')
        score = self.read(ET.tostring(normalized, encoding='unicode'))
        self.assertEqual(score.notes[0].duration, Fraction(53, 480))
        self.assertEqual(next(n.onset for n in score.notes if n.source_id == 'bass'), 0)
        self.assertEqual(score.measure_durations, (Fraction(2),))

    def test_bounded_backup_without_independent_complete_voice_is_rejected(self):
        upper = ''.join(note(f't{i}', duration=53, extra=tuplet(18, 8, '16th'))
                        for i in range(18))
        xml = score_xml(measure(attributes(beats='2', divisions=480) + upper
                    + '<backup><duration>3</duration></backup>'
                    + '<backup><duration>960</duration></backup>'
                    + note('bass', duration=240, voice='2')))
        normalized, audit = self.normalized(xml)
        self.assertEqual(audit[0]['action'], 'quantized_timing_rejected')
        self.assertEqual(ET.tostring(normalized), ET.tostring(ET.fromstring(xml)))

    def test_scores_without_underflow_are_never_retimed(self):
        xml = score_xml(measure(attributes(beats='1', divisions=480) + seven_notes()))
        normalized, audit = self.normalized(xml)
        self.assertEqual(audit, [])
        self.assertEqual(ET.tostring(normalized), ET.tostring(ET.fromstring(xml)))

    def test_prefer_preserve_durations_keeps_source_grid_with_same_witness_checks(self):
        xml = score_xml(measure(attributes(beats='1', divisions=480)
                    + seven_notes() + '<backup><duration>480</duration></backup>'
                    + note('bass', duration=480, voice='2')))
        source = ET.fromstring(xml)
        source_bytes = ET.tostring(source)
        normalized, audit = normalize_quantized_timing(source, prefer_preserve_durations=True)
        self.assertEqual(ET.tostring(source), source_bytes)
        self.assertEqual(audit[0]['action'], 'quantized_full_voice_backup_repair')
        self.assertEqual(audit[0]['part_divisions_multiplier'], 1)
        self.assertEqual(normalized.findtext('part/measure/attributes/divisions'), '480')
        score = self.read(ET.tostring(normalized, encoding='unicode'))
        self.assertEqual(next(n.duration for n in score.notes if n.source_id == 't0'), Fraction(68, 480))
        self.assertEqual(next(n.onset for n in score.notes if n.source_id == 'bass'), 0)
        self.assertEqual(score.measure_durations, (Fraction(1),))
        # The option is only an ordering preference, never a permissive clamp.
        source.find('part/measure/backup/duration').text = '1920'
        unchanged, rejected = normalize_quantized_timing(source, prefer_preserve_durations=True)
        self.assertEqual(rejected[0]['reason'], 'large_cursor_inconsistency')
        self.assertEqual(ET.tostring(unchanged), ET.tostring(source))
