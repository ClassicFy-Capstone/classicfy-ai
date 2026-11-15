"""No coordinate/edition guessing in the local real-data training bridge."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from unit.preprocessing.musicxml_fixtures import XMLFixture, score_xml, measure, attributes, note

spec = importlib.util.spec_from_file_location('local_ae', Path(__file__).resolve().parents[3] / 'scripts/run_local_atepp_ae.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LocalATEPPAETest(XMLFixture):
    def test_bridge_requires_unique_exact_pitch_and_quarter(self):
        xml = self.read(score_xml(measure(attributes() + note('a') + note('b', 'D'))))
        notes = [SimpleNamespace(note_id='good', pitch=60, onset_beats=0.),
                 SimpleNamespace(note_id='wrong-pitch', pitch=61, onset_beats=0.),
                 SimpleNamespace(note_id='wrong-time', pitch=62, onset_beats=1.01)]
        aliases = runner.verified_aliases(xml, notes)
        self.assertEqual(aliases, {'good': xml.notes[0].occurrence_id})
        notes.append(SimpleNamespace(note_id='duplicate', pitch=60, onset_beats=0.))
        self.assertEqual(runner.verified_aliases(xml, notes), {})
