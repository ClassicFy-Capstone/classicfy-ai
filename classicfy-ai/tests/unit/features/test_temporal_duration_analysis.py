from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from analyze_temporal_duration import analyze
from analyze_temporal_shape import restore_channels


class TemporalDurationAnalysisTests(unittest.TestCase):
    def fixture(self):
        group = dict(piece='work', grid='grid', keys=['performance'],
                     values=np.tile([0, 1, 2, 4, 5, 6, 7], (1, 7, 1)),
                     shared_indices=np.array([0, 1, 2, 4, 5, 6, 7]))
        return restore_channels([group], [dict(piece='work', grid='grid', total_intervals=8)],
                                {'work': 'test'})

    def test_duration_and_correlation_keep_original_adjacency(self):
        durations, correlations = analyze(self.fixture(), lags=(1, 2), min_pairs=3)
        self.assertEqual(len(durations), 7)
        self.assertEqual(len(correlations), 7)
        self.assertEqual(durations[0]['direction_rise_run_count'], 2)
        self.assertEqual(durations[0]['direction_rise_run_max'], 3)
        self.assertEqual(correlations[0]['value_lag2_valid_pairs'], 3)
        self.assertEqual(correlations[0]['delta_lag2_valid_pairs'], 1)
        self.assertEqual(correlations[0]['delta_lag2_status'], 'insufficient_pairs')
        self.assertEqual(correlations[0]['split'], 'test')

    def test_positive_scaling_preserves_run_and_correlation_metrics(self):
        items = self.fixture()
        durations, correlations = analyze(items, lags=(1, 2), min_pairs=3)
        from features.models import BeatSequence
        items[0]['channels'] = {name: BeatSequence(seq.values * 4, seq.mask)
                                for name, seq in items[0]['channels'].items()}
        scaled_durations, scaled_correlations = analyze(items, lags=(1, 2), min_pairs=3)
        for a, b in zip(durations, scaled_durations):
            for field in a:
                if isinstance(a[field], (int, float)):
                    np.testing.assert_allclose(a[field], b[field], equal_nan=True)
        for a, b in zip(correlations, scaled_correlations):
            for field in a:
                if isinstance(a[field], (int, float)):
                    np.testing.assert_allclose(a[field], b[field], equal_nan=True)

    def test_empty_or_duplicate_lags_rejected(self):
        for lags in ((), (1, 1)):
            with self.assertRaises(ValueError):
                analyze(self.fixture(), lags=lags)


if __name__ == '__main__':
    unittest.main()
