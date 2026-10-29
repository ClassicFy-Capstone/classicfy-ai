from fractions import Fraction
from zipfile import ZipFile

from preprocessing.musicxml_score import read_musicxml
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, direction, example_xml, measure, note, score_xml


class MusicXMLScoreTest(XMLFixture):
    def test_exact_positions_chords_grace_and_context_changes(self):
        score = self.read()
        self.assertEqual([n.source_id for n in score.notes], ['c1', 'e1', 'c2', 'grace1', 'grace2', 'f1', 'g1', 'a1'])
        self.assertEqual([n.onset for n in score.notes], list(map(Fraction, [0, 0, 1, 4, 4, 4, 5, 6])))
        self.assertEqual([n.duration for n in score.notes], list(map(Fraction, [1, 1, 2, 0, 0, 1, 1, 1])))
        self.assertEqual(score.measure_durations, (Fraction(4), Fraction(3)))
        self.assertEqual(score.notes[0].chord_group, score.notes[1].chord_group)
        self.assertNotEqual(score.notes[3].chord_group, score.notes[4].chord_group)
        self.assertEqual(score.notes[0].meter.numerator, 4)
        self.assertEqual(score.notes[-1].meter.numerator, 3)
        self.assertIsNone(score.notes[0].mode)
        self.assertEqual((score.notes[-1].fifths, score.notes[-1].mode), (2, 'minor'))
        self.assertEqual(score.notes[5].pitch, 66)

    def test_divisions_change_and_tuplets_are_not_applied_twice(self):
        xml = score_xml(measure(attributes(divisions=3)
                        + note('t', duration=1, extra='<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>')
                        + '<attributes><divisions>6</divisions></attributes>' + note('u', 'D', duration=4)))
        score = self.read(xml)
        self.assertEqual([n.onset for n in score.notes], [Fraction(0), Fraction(1, 3)])
        self.assertEqual([n.duration for n in score.notes], [Fraction(1, 3), Fraction(2, 3)])

    def test_voice_backup_and_cross_staff_identity(self):
        xml = score_xml(measure(attributes() + note('upper', duration=16, voice='7')
                        + '<backup><duration>16</duration></backup>'
                        + note('lower', 'E', duration=8, voice='9', staff='2')
                        + note('cross', 'F', duration=8, voice='9', staff='1')))
        score = self.read(xml)
        self.assertEqual([(n.source_id, n.onset, n.voice) for n in score.notes],
                         [('upper', Fraction(0), '7'), ('lower', Fraction(0), '9'), ('cross', Fraction(2), '9')])
        self.assertEqual(score.measure_durations, (Fraction(4),))

    def test_repeat_endings_and_unique_occurrences(self):
        xml = score_xml(
            measure(attributes(beats='1') + '<barline location="left"><repeat direction="forward"/></barline>' + note('a')),
            measure('<barline location="left"><ending number="1" type="start"/></barline>' + note('b')
                    + '<barline><ending number="1" type="stop"/><repeat direction="backward"/></barline>', 2),
            measure('<barline location="left"><ending number="2" type="start"/></barline>' + note('c')
                    + '<barline><ending number="2" type="stop"/></barline>', 3))
        score = self.read(xml)
        self.assertEqual([v.measure for v in score.visits], [0, 1, 0, 2])
        self.assertEqual([n.occurrence_id for n in score.notes], ['a@1', 'b@1', 'a@2', 'c@1'])
        self.assertEqual([n.onset for n in score.notes], list(map(Fraction, range(4))))
        self.assertTrue(score.visits[2].reset)

    def test_compressed_xml_and_namespace_keep_same_note_semantics(self):
        path = self.folder / 'score.mxl'
        xml = example_xml().replace('<score-partwise version="4.0">', '<score-partwise xmlns="urn:musicxml" version="4.0">')
        with ZipFile(path, 'w') as archive:
            archive.writestr('META-INF/container.xml', '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="score.xml" media-type="application/vnd.recordare.musicxml+xml"/></rootfiles></container>')
            archive.writestr('score.xml', xml)
        compressed, plain = read_musicxml(path), self.read()
        self.assertEqual(compressed.notes, plain.notes)
        self.assertNotEqual(compressed.source_sha256, plain.source_sha256)

    def test_transpose_resolves_by_position_after_backup(self):
        xml = score_xml(measure(attributes() + note('a', duration=8)
                        + '<attributes><transpose><chromatic>2</chromatic></transpose></attributes>'
                        + note('b', duration=8) + '<backup><duration>16</duration></backup>'
                        + note('c', duration=16, voice='2')))
        score = self.read(xml)
        self.assertEqual({n.source_id: n.pitch for n in score.notes}, {'a': 60, 'b': 62, 'c': 60})

    def test_missing_id_gets_stable_locator_and_missing_single_voice_is_known(self):
        score = self.read(score_xml(measure(attributes() + note(None, duration=16, voice=None))))
        self.assertIsNone(score.notes[0].xml_id)
        self.assertIn('element[1]', score.notes[0].source_locator)
        self.assertTrue(score.notes[0].voice_known)

    def test_rejects_ambiguous_or_invalid_structure(self):
        invalid = [
            score_xml(measure(attributes() + note('x') + note('x'))),
            score_xml(measure(attributes() + '<backup><duration>1</duration></backup>' + note('x'))),
            score_xml(measure(attributes() + note('x', chord=True))),
            score_xml(measure(attributes() + direction('<words>D.C. al Fine</words>') + note('x'))),
            score_xml(measure(attributes() + note('x', extra='<tie type="start" time-only="1"/>'))),
        ]
        for xml in invalid:
            with self.subTest(xml=xml), self.assertRaises(ValueError):
                self.read(xml)
        with self.assertRaisesRegex(ValueError, 'measure zero'):
            self.read(traversal=(1,))
