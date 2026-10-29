import json
from zipfile import ZipFile

from preprocessing.musicxml_score import read_musicxml
from preprocessing.score_midi import read_score_midi
from preprocessing.xml_score_midi_bridge import render_xml_score_bridge
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, example_xml, measure, note, score_xml


class XMLScoreMidiBridgeTest(XMLFixture):
    def test_exact_tied_event_and_grace_exclusion(self):
        xml = self.read()
        out = self.folder / 'canonical.mid'
        bridge = render_xml_score_bridge(xml, out)
        notes, _, _ = read_score_midi(out)
        owners = {bridge['score_id_to_occurrence'][n.note_id]: n for n in notes}
        self.assertEqual(owners['c1@1'].duration_beats, 3)
        self.assertNotIn('c2@1', owners)
        self.assertEqual(bridge['excluded']['grace'], 2)
        self.assertFalse(bridge['alignment_accepted'])
        with self.assertRaises(FileExistsError):
            render_xml_score_bridge(xml, out)

    def test_unison_does_not_choose_an_arbitrary_written_note(self):
        xml = self.read(score_xml(measure(attributes() + note('a') + note('b', chord=True) + note('c', 'D'))))
        bridge = render_xml_score_bridge(xml, self.folder / 'u.mid')
        self.assertEqual(bridge['ambiguous_unisons'], 1)
        self.assertEqual(set(bridge['score_id_to_occurrence'].values()), {'c@1'})

    def test_tuplet_coordinates_roundtrip_without_quantization(self):
        xml = self.read(score_xml(measure(attributes(divisions=7) + note('a', duration=1) + note('b', duration=2))))
        bridge = render_xml_score_bridge(xml, self.folder / 'tuple.mid')
        notes, _, _ = read_score_midi(self.folder / 'tuple.mid')
        self.assertEqual(notes[1].onset_beats, 1 / 7)
        self.assertEqual(notes[1].duration_beats, 2 / 7)
        self.assertEqual(len(bridge['score_id_to_occurrence']), 2)

    def test_atepp_mxl_missing_media_type_and_ambiguous_roots(self):
        path = self.folder / 'score.mxl'
        with ZipFile(path, 'w') as z:
            z.writestr('META-INF/container.xml', '<container><rootfiles><rootfile full-path="score.xml"/></rootfiles></container>')
            z.writestr('score.xml', example_xml())
        self.assertEqual(read_musicxml(path).notes, self.read().notes)
        with ZipFile(path, 'w') as z:
            z.writestr('META-INF/container.xml', '<container><rootfiles><rootfile full-path="a.xml"/><rootfile full-path="b.xml"/></rootfiles></container>')
        with self.assertRaisesRegex(ValueError, 'one MusicXML'):
            read_musicxml(path)

    def test_uncompressed_xml_with_mxl_extension(self):
        path = self.folder / 'plain.mxl'
        path.write_text(example_xml())
        self.assertEqual(read_musicxml(path).notes, self.read().notes)
