"""Physical-unit and shared-position checks for era tempo summaries."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from analyze_era_tempo import summarize_common


def recording(key, bpm, start=0, stop=64, time_shift=0):
    q = np.arange(start, stop + 1, dtype=float)
    return {'perf_id': key, 'composition_id': 'toy', 'q': q,
            'seconds': time_shift + q * 60 / bpm}


class EraTempoSummaryTests(unittest.TestCase):
    def test_known_absolute_units_and_relative_percentages(self):
        result, performances, _ = summarize_common([recording('a', 90), recording('b', 100), recording('c', 110)])
        self.assertAlmostEqual(result['absolute_tempo_qpm'], 100)
        self.assertAlmostEqual(result['absolute_qpm_q25'], 95)
        self.assertAlmostEqual(result['absolute_qpm_q75'], 105)
        self.assertAlmostEqual(result['relative_tempo_iqr_pp'], 10)
        np.testing.assert_allclose([r['relative_tempo_pct'] for r in performances], [-10, 0, 10], atol=1e-10)

    def test_absolute_scales_with_speed_but_relative_does_not(self):
        slow, _, _ = summarize_common([recording('a', 80), recording('b', 100), recording('c', 120)])
        fast, _, _ = summarize_common([recording('a', 160), recording('b', 200), recording('c', 240)])
        self.assertAlmostEqual(fast['absolute_tempo_qpm'], 2 * slow['absolute_tempo_qpm'])
        self.assertAlmostEqual(fast['relative_tempo_iqr_pp'], slow['relative_tempo_iqr_pp'])

    def test_recording_start_time_does_not_change_tempo(self):
        before, _, _ = summarize_common([recording('a', 90), recording('b', 110)])
        after, _, _ = summarize_common([recording('a', 90, time_shift=1.7), recording('b', 110, time_shift=12.3)])
        self.assertAlmostEqual(before['absolute_tempo_qpm'], after['absolute_tempo_qpm'])
        self.assertAlmostEqual(before['relative_tempo_iqr_pp'], after['relative_tempo_iqr_pp'])

    def test_singleton_has_no_relative_value(self):
        result, performances, _ = summarize_common([recording('a', 90)])
        self.assertEqual(result['status'], 'absolute_only_single_recording')
        self.assertAlmostEqual(result['absolute_tempo_qpm'], 90)
        self.assertTrue(np.isnan(result['relative_tempo_iqr_pp']))
        self.assertTrue(np.isnan(performances[0]['relative_tempo_pct']))

    def test_only_identical_score_intervals_are_compared(self):
        result, _, packed = summarize_common([recording('a', 90, 0, 64), recording('b', 110, 16, 80)])
        self.assertEqual(result['shared_beats'], 48)
        np.testing.assert_array_equal(packed['q'], np.arange(16, 64))

    def test_missing_common_positions_are_not_interpolated(self):
        result, performances, packed = summarize_common([recording('a', 90, 0, 64), recording('b', 110, 100, 164)])
        self.assertEqual(result['status'], 'insufficient_common_beats')
        self.assertEqual(performances, [])
        self.assertIsNone(packed)

    def test_invalid_interval_is_removed_for_all_recordings(self):
        a, b = recording('a', 90), recording('b', 110)
        tau = np.diff(a['seconds']); tau[3] = .01
        a['seconds'] = np.r_[0, np.cumsum(tau)]
        result, _, packed = summarize_common([a, b])
        self.assertEqual(result['shared_beats'], 63)
        self.assertNotIn(3, packed['q'])
        self.assertAlmostEqual(result['absolute_tempo_qpm'], 100)


if __name__ == '__main__':
    unittest.main()
