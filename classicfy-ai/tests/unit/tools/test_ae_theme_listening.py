"""Guard against wrong-work retrieval and off-by-one theme/audio excerpts."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

import numpy as np

AI = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AI / "scripts"))
from prepare_ae_theme_listening import (file_sha, public_pack, rank_candidates, select_near_far,
                                        theme_bounds, time_at_quarter, youtube_id)


class AEThemeListeningTest(unittest.TestCase):
    def test_far_is_measured_from_anchor_not_from_the_near_recording(self):
        rows = [dict(performance_id=p, work_id=w, recording_id=r) for p, w, r in
                [("A", "work1", "midi1"), ("B", "work2", "midi2"),
                 ("C", "work2", "midi3"), ("D", "work2", "midi4"),
                 ("B-duplicate", "work2", "midi2"), ("other", "work3", "midi5")]]
        angles = np.array([0., .5, -1., 1.2, 3.1, 2.])
        embeddings = np.column_stack([np.cos(angles), np.sin(angles)])
        near, far, pool, _, distances = select_near_far(rows, embeddings, 0)
        self.assertEqual(rows[near]["performance_id"], "B")
        self.assertEqual(rows[far]["performance_id"], "D")
        self.assertEqual({rows[i]["performance_id"] for i in pool}, {"C", "D"})
        self.assertGreater(distances[far], distances[near])
        # Measuring from B would wrongly select C, which must not happen.
        self.assertGreater(1 - embeddings[near] @ embeddings[2], 1 - embeddings[near] @ embeddings[far])

    def test_missing_same_work_comparison_is_not_replaced_by_another_work(self):
        rows = [dict(performance_id=p, work_id=w, recording_id=r) for p, w, r in
                [("A", "work1", "midi1"), ("B", "work2", "midi2"), ("C", "work3", "midi3")]]
        with self.assertRaisesRegex(ValueError, "no distinct"):
            select_near_far(rows, np.array([[1., 0.], [.8, .6], [0., 1.]]), 0)

    def test_global_rank_excludes_same_work_and_identical_recording_and_breaks_ties(self):
        rows = [dict(performance_id=p, work_id=w, recording_id=r) for p, w, r in
                [("anchor", "work1", "midi1"), ("same-work", "work1", "other"),
                 ("duplicate", "work2", "midi1"), ("candidate-b", "work3", "midi3"),
                 ("candidate-a", "work4", "midi4")]]
        order, distances = rank_candidates(rows, np.array([[1., 0.]] * len(rows)), 0)
        self.assertEqual([rows[i]["performance_id"] for i in order], ["candidate-a", "candidate-b"])
        np.testing.assert_array_equal(distances, 0)

    def test_pickup_xml_label_is_not_literature_bar_or_list_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / "score.xml"
            xml.write_text('<score-partwise><part id="P1"><measure number="1"/>'
                           '<measure number="2"/><measure number="3"/></part></score-partwise>')
            windows = root / "windows.json"
            windows.write_text(json.dumps([{"measure_visits": [
                {"key": "0@1", "q_start": "0", "duration_q": "1/2"},
                {"key": "1@1", "q_start": "1/2", "duration_q": "6"},
                {"key": "2@1", "q_start": "13/2", "duration_q": "6"},
                {"key": "0@2", "q_start": "25/2", "duration_q": "1/2"}]}]))
            source = dict(xml_path=xml, windows_file=windows, windows_sha256=file_sha(windows))
            annotation = dict(score_sha256=file_sha(xml), xml_measure_start="1", xml_measure_end_inclusive="2")
            self.assertEqual(theme_bounds(source, annotation), (0., 6.5))
            annotation["xml_measure_start"] = "2"
            self.assertEqual(theme_bounds(source, annotation), (.5, 6.5))
            xml.write_text(xml.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                theme_bounds(source, annotation)

    def test_time_mapping_keeps_chord_center_and_first_attack(self):
        performance = SimpleNamespace(metadata=[
            dict(is_attack=True, score_onset_q=0, onset_sec=.2),
            dict(is_attack=True, score_onset_q=0, onset_sec=.3),
            dict(is_attack=True, score_onset_q=2, onset_sec=3.8),
            dict(is_attack=True, score_onset_q=2, onset_sec=4.2)])
        self.assertEqual(time_at_quarter(performance, 0), (.2, "aligned-note-onset"))
        self.assertEqual(time_at_quarter(performance, 2), (4., "aligned-note-onset"))

    def test_compressed_score_preserves_annotated_measure_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score = root / "score.mxl"
            with ZipFile(score, "w") as archive:
                archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="score.xml"/></rootfiles></container>')
                archive.writestr("score.xml", '<score-partwise><part id="P1"><measure number="1"/><measure number="2"/></part></score-partwise>')
            windows = root / "windows.json"
            windows.write_text(json.dumps([{"measure_visits": [
                {"key": "0@1", "q_start": "0", "duration_q": "3"},
                {"key": "1@1", "q_start": "3", "duration_q": "3"}]}]))
            source = dict(xml_path=score, windows_file=windows, windows_sha256=file_sha(windows))
            annotation = dict(score_sha256=file_sha(score), xml_measure_start="1", xml_measure_end_inclusive="2")
            self.assertEqual(theme_bounds(source, annotation), (0., 6.))

    def test_initial_rest_omission_is_explicit_and_uses_the_score_attack(self):
        themes = json.loads((AI / "analysis/ae_theme_listening/themes.json").read_text())
        pack = json.loads((AI / "analysis/ae_theme_listening/listening.json").read_text())
        self.assertTrue(themes["works"]["ATEPP/655"]["omit_initial_rests"])
        self.assertTrue(themes["works"]["ATEPP/667"]["omit_initial_rests"])
        self.assertEqual(pack["clips"]["ATEPP:03108"]["q_start"], .5)
        self.assertEqual(pack["clips"]["ATEPP:03190"]["q_start"], 2.5)
        # The Moonlight melody follows the accompaniment introduction.
        self.assertEqual(pack["clips"]["ATEPP:08945"]["q_start"], 16.)

    def test_no_silent_clamping_extrapolation_or_invalid_beat_interpolation(self):
        performance = SimpleNamespace(metadata=[], beats=SimpleNamespace(
            q=np.array([1., 2., 3.]), seconds=np.array([2., 4., 8.]), valid=np.array([True, False])))
        self.assertEqual(time_at_quarter(performance, 1.5), (3., "valid-beat-interpolation"))
        for quarter in (0., 4., 2.5):
            with self.assertRaises(ValueError):
                time_at_quarter(performance, quarter)

    def test_youtube_id_stays_on_original_recording(self):
        self.assertEqual(youtube_id("https://www.youtube.com/watch?v=JuFd49TCYdE&t=10"), "JuFd49TCYdE")
        self.assertEqual(youtube_id("https://youtu.be/JuFd49TCYdE"), "JuFd49TCYdE")
        for link in ("https://example.com/?v=JuFd49TCYdE", "https://youtube.com/watch?v=short"):
            with self.assertRaises(ValueError):
                youtube_id(link)

    def test_prepared_pack_matches_fixed_epoch_and_real_theme_bounds(self):
        pack = json.loads((AI / "analysis/ae_theme_listening/listening.json").read_text())
        self.assertEqual(pack["model"]["epoch"], 100)
        self.assertEqual(pack["catalogue"]["full_windows"], 67542)
        self.assertEqual(len(pack["cases"]), 10)
        self.assertEqual(len(pack["clips"]), 30)
        for case in pack["cases"]:
            anchor = pack["clips"][case["anchor"]]
            near, far = [pack["clips"][case[k]] for k in ("near", "far")]
            self.assertEqual(case["candidates"][0]["global_rank"], 1)
            self.assertEqual(near["work_id"], far["work_id"])
            self.assertNotEqual(near["recording_id"], far["recording_id"])
            self.assertEqual(near["q_start"], far["q_start"])
            self.assertEqual(near["q_end"], far["q_end"])
            self.assertLess(case["candidates"][0]["cosine_distance_to_anchor"],
                            case["candidates"][1]["cosine_distance_to_anchor"])
            for candidate in case["candidates"]:
                clip = pack["clips"][candidate["clip"]]
                self.assertNotEqual(anchor["work_id"], clip["work_id"])
                self.assertNotEqual(anchor["recording_id"], clip["recording_id"])
        # The Rachmaninoff theme starts AFTER the two-measure introduction.
        self.assertEqual(pack["clips"]["ATEPP:00239"]["q_start"], 6.)
        # Schumann's separately numbered pickup must not consume a full bar.
        self.assertEqual(pack["clips"]["ATEPP:00983"]["q_end"], 30.5)
        for clip in pack["clips"].values():
            self.assertLess(clip["start_seconds"], clip["end_seconds"])
            self.assertFalse(clip["time_mapping"]["youtube_offset_verified"])

    def test_twenty_works_are_not_reused_across_questions(self):
        pack = json.loads((AI / "analysis/ae_theme_listening/listening.json").read_text())
        works = [pack["clips"][case[role]]["work_id"]
                 for case in pack["cases"] for role in ("anchor", "near")]
        self.assertEqual(len(works), 20)
        self.assertEqual(len(set(works)), 20)
        self.assertEqual(len({c["recording_id"] for c in pack["clips"].values()}), 30)
        verification = json.loads((AI / "analysis/ae_theme_listening/verification.json").read_text())
        self.assertEqual(verification["maximum_questions_per_work"], 1)

    def test_initial_pack_does_not_include_roles_distances_or_candidate_artist(self):
        organizer = json.loads((AI / "analysis/ae_theme_listening/listening.json").read_text())
        public, results = public_pack(organizer)
        forbidden = {"near", "far", "near_clip", "far_clip", "role", "artist", "performance_id",
                     "global_rank", "cosine_distance_to_anchor"}
        def check(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(public)
        for case in public["cases"]:
            answer = results[case["id"]]
            self.assertEqual(answer["pack_id"], public["id"])
            self.assertEqual(set(case["candidates"]), {answer["near_clip"], answer["far_clip"]})


if __name__ == "__main__":
    unittest.main()
