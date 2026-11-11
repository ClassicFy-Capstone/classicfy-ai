"""Protect cross-work candidate selection and written unison passage boundaries."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

AI = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AI / "scripts"))
from prepare_cross_work_theme_study import select_recordings
from prepare_theme_comparison import boundary_ids


def score_xml():
    measures = []
    for number in range(1, 10):
        measures.append(f'''<measure number="{number}">
          <attributes><divisions>1</divisions></attributes>
          <note id="a{number}"><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
          <backup><duration>4</duration></backup>
          <note id="duplicate{number}"><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
          <backup><duration>4</duration></backup>
          <note id="bass{number}"><pitch><step>F</step><octave>3</octave></pitch><duration>4</duration></note>
          <backup><duration>4</duration></backup>
          <note id="tied{number}"><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><tie type="stop"/></note>
        </measure>''')
    return '<score-partwise><part>' + ''.join(measures) + '</part></score-partwise>'


class CrossWorkThemeTest(unittest.TestCase):
    def test_voice_unisons_and_tie_continuations_do_not_require_extra_attacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            score = Path(tmp) / "score.musicxml"
            score.write_text(score_xml())
            self.assertEqual(boundary_ids(score, 4), {"1": ["a1-1", "bass1-1"], "9": ["a9-1", "bass9-1"]})

    def test_wrong_measure_length_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            score = Path(tmp) / "score.musicxml"
            score.write_text(score_xml())
            with self.assertRaisesRegex(ValueError, "Expected 8"):
                boundary_ids(score)

    def test_six_pedal_coordinates_still_count_as_one_feature_block(self):
        anchor = np.zeros(14)
        pedal, tempo = np.zeros(14), np.zeros(14)
        pedal[8:] = 1
        tempo[:2] = .8
        rows = select_recordings(anchor, ["pedal", "tempo"], np.array([pedal, tempo]))
        self.assertEqual([r["key"] for r in rows], ["tempo", "pedal"])
        self.assertAlmostEqual(rows[0]["distance"], np.sqrt(.8 ** 2 / 5))
        self.assertAlmostEqual(rows[-1]["distance"], np.sqrt(1 / 5))

    def test_nearest_ties_have_stable_key_order(self):
        rows = select_recordings(np.zeros(14), ["z", "a", "far"], np.array([[1.] * 14, [1.] * 14, [2.] * 14]))
        self.assertEqual([r["key"] for r in rows], ["a", "z", "far"])

    def test_identical_candidates_cannot_be_presented_as_near_and_far(self):
        with self.assertRaisesRegex(ValueError, "tied"):
            select_recordings(np.zeros(14), ["a", "b"], np.ones((2, 14)))


if __name__ == "__main__":
    unittest.main()
