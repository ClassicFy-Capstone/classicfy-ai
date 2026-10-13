import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from preprocessing import load_musicxml


SCORE = '''<score-partwise version="4.0">
<part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1">
<measure number="0" implicit="yes">
 <attributes><divisions>4</divisions><key><fifths>1</fifths></key>
 <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
 <barline location="left"><repeat direction="forward"/></barline>
 <note id="n0"><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration>
 <voice>1</voice><staff>1</staff><tie type="start"/>
 <notations><slur type="start" number="2" placement="above"/><fermata/></notations></note>
</measure>
<measure number="1">
 <note id="n1"><pitch><step>C</step><alter>1</alter><octave>4</octave></pitch>
 <duration>4</duration><voice>1</voice><staff>1</staff><accidental>sharp</accidental>
 <notations><slur type="continue" number="2"/></notations></note>
 <note><chord/><pitch><step>E</step><alter>-1</alter><octave>4</octave></pitch>
 <duration>4</duration><voice>1</voice><staff>1</staff></note>
 <note><grace slash="yes"/><pitch><step>B</step><octave>3</octave></pitch>
 <type>16th</type><voice>1</voice><staff>1</staff></note>
 <note><grace/><chord/><pitch><step>D</step><octave>4</octave></pitch>
 <type>16th</type><voice>1</voice><staff>1</staff></note>
 <note><pitch><step>F</step><octave>4</octave></pitch><duration>8</duration>
 <voice>1</voice><staff>1</staff><notations><slur type="stop" number="2"/></notations></note>
 <forward><duration>4</duration></forward>
 <backup><duration>16</duration></backup>
 <note><rest/><duration>8</duration><voice>2</voice><staff>2</staff></note>
 <note><pitch><step>G</step><octave>3</octave></pitch><duration>8</duration>
 <voice>2</voice><staff>2</staff></note>
 <barline><ending number="1" type="stop"/><repeat direction="backward" times="2"/></barline>
</measure>
<measure number="1">
 <attributes><divisions>6</divisions><key number="2"><fifths>-3</fifths><mode>minor</mode></key>
 <time><beats>3+2</beats><beat-type>8</beat-type></time></attributes>
 <note><pitch><step>A</step><alter>0.5</alter><octave>4</octave></pitch><duration>2</duration></note>
 <note><rest/><duration>13</duration></note>
</measure>
</part></score-partwise>'''


def score_with(body: str) -> str:
    return f'<score-partwise><part id="P1"><measure number="1">{body}</measure></part></score-partwise>'


def pitched(duration='4', extra='') -> str:
    return f'<note>{extra}<pitch><step>C</step><octave>4</octave></pitch><duration>{duration}</duration></note>'


class LoadMusicXMLTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def load(self, text=SCORE):
        path = Path(self.temp.name) / 'score.musicxml'
        path.write_text(text, encoding='utf-8')
        return load_musicxml(path)

    def test_written_timing_with_pickup_chords_grace_and_multiple_voices(self):
        score = self.load()
        self.assertEqual(score.part_names, (('P1', 'Piano'),))
        self.assertEqual([m.onset_quarters for m in score.measures], [0, 1, 5])
        self.assertEqual([m.duration_quarters for m in score.measures], [1, 4, Fraction(5, 2)])
        self.assertEqual([n.onset_quarters for n in score.notes],
                         [0, 1, 1, 2, 2, 2, 1, 3, 5, Fraction(16, 3)])
        self.assertEqual(score.notes[8].duration_quarters, Fraction(1, 3))
        self.assertEqual(score.notes[-1].offset_quarters, Fraction(15, 2))
        self.assertTrue(score.notes[3].is_grace)
        self.assertEqual(score.notes[3].grace_attributes, (('slash', 'yes'),))
        self.assertTrue(score.notes[4].is_chord)
        self.assertTrue(score.notes[6].is_rest)
        self.assertIsNone(score.notes[6].pitch)
        self.assertEqual((score.notes[6].voice, score.notes[6].staff), ('2', 2))
        self.assertFalse(score.diagnostics)

    def test_pitch_spelling_and_fractional_alterations_are_retained(self):
        score = self.load()
        self.assertEqual((score.notes[1].pitch.step, score.notes[1].pitch.alter), ('C', 1))
        self.assertEqual(score.notes[1].pitch.midi_pitch, 61)
        self.assertEqual(score.notes[2].pitch.midi_pitch, 63)
        self.assertEqual(score.notes[8].pitch.midi_pitch, Fraction(139, 2))
        self.assertEqual(score.notes[1].accidental, 'sharp')

    def test_source_ids_and_internal_ids_survive_duplicate_measure_numbers(self):
        score = self.load()
        self.assertEqual(score.notes[0].xml_id, 'n0')
        self.assertEqual(score.notes[0].note_id, 'P1:m0:n0')
        self.assertIsNone(score.notes[-1].xml_id)
        self.assertEqual([m.number for m in score.measures], ['0', '1', '1'])
        self.assertEqual(len({n.note_id for n in score.notes}), len(score.notes))
        self.assertEqual(score.notes, self.load().notes)

    def test_raw_slurs_ties_and_repeat_marks_are_preserved_without_unfolding(self):
        score = self.load()
        self.assertEqual([(s.kind, s.number) for s in score.slurs],
                         [('start', '2'), ('continue', '2'), ('stop', '2')])
        self.assertEqual(dict(score.slurs[0].attributes)['placement'], 'above')
        self.assertEqual(score.slurs[0].note_id, score.notes[0].note_id)
        self.assertEqual(score.notes[0].ties, ((('type', 'start'),),))
        self.assertIn('fermata', score.notes[0].notation_xml[0])
        self.assertEqual([b.kind for b in score.barline_marks], ['repeat', 'ending', 'repeat'])
        self.assertEqual([b.onset_quarters for b in score.barline_marks], [0, 5, 5])
        self.assertEqual(dict(score.barline_marks[-1].attributes)['times'], '2')
        self.assertEqual(len(score.measures), 3)

    def test_key_meter_changes_and_unknown_mode_are_not_inferred(self):
        score = self.load()
        first, second = score.key_signatures
        self.assertEqual((first.fifths, first.mode, first.staff), (1, None, None))
        self.assertEqual((second.fifths, second.mode, second.staff, second.onset_quarters), (-3, 'minor', 2, 5))
        self.assertIsNone(score.notes[-1].voice)
        self.assertIsNone(score.notes[-1].staff)
        self.assertEqual(score.time_signatures[-1].groups, (('3+2', 8),))
        self.assertEqual(score.time_signatures[-1].duration_quarters, Fraction(5, 2))

    def test_namespace_does_not_change_structure(self):
        plain = self.load()
        namespaced = self.load(SCORE.replace('<score-partwise ', '<score-partwise xmlns="urn:musicxml" ', 1))
        self.assertEqual(namespaced.notes, plain.notes)
        self.assertEqual(namespaced.measures, plain.measures)
        self.assertEqual(namespaced.key_signatures, plain.key_signatures)

    def test_nontraditional_key_and_composite_meter(self):
        text = score_with('''<attributes><divisions>2</divisions>
        <key><key-step>F</key-step><key-alter>0.5</key-alter><key-octave number="1">4</key-octave></key>
        <time><beats>2</beats><beat-type>4</beat-type><beats>3</beats><beat-type>8</beat-type></time>
        </attributes><note><rest measure="yes"/><duration>7</duration></note>''')
        score = self.load(text)
        self.assertEqual(score.key_signatures[0].alterations, (('F', Fraction(1, 2)),))
        self.assertIsNone(score.key_signatures[0].fifths)
        self.assertIn('key-octave', score.key_signatures[0].source_xml)
        self.assertEqual(score.time_signatures[0].duration_quarters, Fraction(7, 2))
        self.assertTrue(score.notes[0].is_measure_rest)

    def test_chord_with_omitted_voice_keeps_unknown_voice_and_anchor_position(self):
        text = score_with('<attributes><divisions>4</divisions></attributes>' +
            pitched().replace('</note>', '<voice>1</voice></note>') + pitched(extra='<chord/>'))
        score = self.load(text)
        self.assertEqual([n.onset_quarters for n in score.notes], [0, 0])
        self.assertIsNone(score.notes[1].voice)
        self.assertEqual(score.measures[0].duration_quarters, 1)

    def test_empty_and_underfull_measures_are_diagnosed_without_padding(self):
        text = '''<score-partwise><part id="P1"><measure number="1">
        <attributes><divisions>4</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
        <note><rest/><duration>4</duration></note></measure><measure number="2"/>
        <measure number="3"><note><rest/><duration>4</duration></note></measure></part></score-partwise>'''
        score = self.load(text)
        self.assertEqual([m.onset_quarters for m in score.measures], [0, 1, 1])
        self.assertIn('empty_measure', [d.code for d in score.diagnostics])
        self.assertIn('measure_duration_differs_from_meter', [d.code for d in score.diagnostics])

    def test_compensating_forward_restores_exact_cursor_without_clamping(self):
        text = score_with('<attributes><divisions>4</divisions></attributes>' + pitched('3') +
            '<backup><duration>4</duration></backup><forward><duration>1</duration></forward>' + pitched('4'))
        score = self.load(text)
        self.assertEqual([n.onset_quarters for n in score.notes], [0, 0])
        self.assertEqual(score.measures[0].duration_quarters, 1)
        self.assertEqual([d.code for d in score.diagnostics], ['compensated_backup_underflow'])

    def test_part_timelines_are_kept_separate(self):
        text = '''<score-partwise><part id="P1"><measure number="1">
        <attributes><divisions>4</divisions></attributes><note><rest/><duration>4</duration></note></measure></part>
        <part id="P2"><measure number="1"><attributes><divisions>4</divisions></attributes>
        <note><rest/><duration>8</duration></note></measure></part></score-partwise>'''
        score = self.load(text)
        self.assertEqual([m.duration_quarters for m in score.measures], [1, 2])
        self.assertIn('part_measure_layout_differs', [d.code for d in score.diagnostics])
        self.assertEqual(len({n.note_id for n in score.notes}), 2)

    def test_senza_misura_and_mid_measure_attributes(self):
        text = score_with('<attributes><divisions>4</divisions><time><senza-misura/></time></attributes>' +
            pitched() + '<attributes><key><fifths>2</fifths></key><time><beats>3</beats><beat-type>4</beat-type></time></attributes>' + pitched())
        score = self.load(text)
        self.assertIsNone(score.time_signatures[0].duration_quarters)
        self.assertEqual(score.key_signatures[0].onset_quarters, 1)
        self.assertIn('mid_measure_meter_change', [d.code for d in score.diagnostics])

    def test_invalid_time_or_ids_fail_with_file_and_measure_context(self):
        div = '<attributes><divisions>4</divisions></attributes>'
        cases = [
            (pitched(), 'Missing divisions'),
            (div + pitched('0'), 'Invalid duration'),
            (div + pitched('-1'), 'Invalid duration'),
            (div + pitched(extra='<chord/>'), 'Chord has no preceding'),
            (div + pitched('2') + pitched('4', '<chord/>'), 'Chord duration exceeds'),
            (div + pitched().replace('</note>', '<voice>1</voice></note>') +
             pitched(extra='<chord/><voice>2</voice>'), 'different grace state or voice'),
            (div + pitched(extra='<grace/>'), 'Grace note must not'),
            (div + '<backup><duration>1</duration></backup>', 'Backup crosses'),
            (div + '<backup><duration>1</duration></backup>' + pitched(), 'Backup crosses'),
            ('<attributes><divisions>0</divisions></attributes>', 'Invalid divisions'),
            (div + pitched(extra='<notations><slur type="unknown"/></notations>'), 'Invalid slur'),
            ('<attributes><key><key-step>F</key-step></key></attributes>', 'Invalid non-traditional'),
            ('<attributes><time><beats>0</beats><beat-type>4</beat-type></time></attributes>', 'Invalid beats'),
            (div + '<note><unpitched/><duration>4</duration></note>', 'unpitched'),
            (div + pitched().replace('<note>', '<note id="same">') * 2, 'Duplicate XML note ID'),
        ]
        for body, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message) as caught:
                    self.load(score_with(body))
                self.assertIn('score.musicxml', str(caught.exception))
                self.assertIn('measure', str(caught.exception))

    def test_missing_file_and_unsupported_document_fail_explicitly(self):
        with self.assertRaises(FileNotFoundError):
            load_musicxml(Path(self.temp.name) / 'missing.musicxml')
        for text, message in [('<score-timewise/>', 'Unsupported MusicXML root'),
                              ('<score-partwise/>', 'No score parts'),
                              ('not xml', 'Could not parse')]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                self.load(text)


if __name__ == '__main__':
    unittest.main()
