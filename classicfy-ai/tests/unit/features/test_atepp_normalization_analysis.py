"""Protect ATEPP grid and mask semantics at the adapter boundary."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from atepp_feature_analysis import align_group, normalize_group, collect_normalized


def record(key, grid=None, piece='work', speed=1., offset=0.):
    grid = np.arange(65, dtype=float) if grid is None else np.asarray(grid, dtype=float)
    values = np.column_stack([.5 + offset + .03 * np.sin(grid[:-1]),
                              offset + np.cos(grid[:-1]) * .1,
                              np.full(len(grid) - 1, .4 + offset),
                              np.full(len(grid) - 1, .5 + offset),
                              (grid[:-1] % 4 == 0).astype(float)])
    return dict(key=key, piece=piece, score_beats=grid, performance_beats=grid * .5 / speed,
                dpa=values, dpa_mask=np.ones_like(values, dtype=bool))


class AteppNormalizationAnalysisTest(unittest.TestCase):
    def test_shared_positions_trim_the_corresponding_performance_and_feature_intervals(self):
        a, b = record('00001', np.arange(10.)), record('00002', np.arange(3., 12.))
        trimmed = align_group([a, b])
        np.testing.assert_array_equal(trimmed[0]['score_beats'], np.arange(3., 10.))
        np.testing.assert_array_equal(trimmed[0]['dpa'], a['dpa'][3:9])
        np.testing.assert_array_equal(trimmed[1]['dpa'], b['dpa'][:6])
        np.testing.assert_array_equal(trimmed[1]['performance_beats'], b['performance_beats'][:7])
        self.assertEqual(a['dpa'].shape, (9, 5))
        self.assertFalse(np.shares_memory(a['dpa'], trimmed[0]['dpa']))

    def test_phase_shifted_grids_are_rejected_instead_of_interpolated(self):
        with self.assertRaises(AssertionError):
            align_group([record('a', np.arange(10.)), record('b', np.arange(.5, 10.5))])
        with self.assertRaisesRegex(ValueError, 'Different score identities'):
            align_group([record('a'), record('b', piece='other')])

    def test_missing_masks_survive_common_and_scaling_and_support_is_at_least_two(self):
        a, b, c = [record(k, offset=o) for k, o in zip(['a', 'b', 'c'], [0, .1, -.1])]
        b['dpa_mask'][0, 0] = False
        c['dpa_mask'][0, 0] = False
        # Finite values with false masks must not enter common estimates.
        result = normalize_group([a, b, c])
        fs = result['features']['dynamics']
        self.assertEqual(fs['a']['support'][0], 1)
        for f in fs.values():
            self.assertFalse(f['common'].mask[0])
            self.assertFalse(f['relative'].mask[0])
            self.assertFalse(f['standardized'].mask[0])
            self.assertTrue(np.isnan(f['standardized'].values[0]))
        self.assertTrue(fs['a']['raw'].mask[0])
        self.assertFalse(fs['b']['raw'].mask[0])

    def test_channel_order_and_shared_scale_preserve_pairs(self):
        a, b = record('00001'), record('00002', offset=.1, speed=1.1)
        result = normalize_group([a, b])
        methods = {s['channel']: s['requested_method'] for s in result['scales']}
        self.assertEqual(methods['dynamics'], 'mad')
        self.assertEqual(methods['articulation'], 'mad')
        self.assertEqual(methods['pedal_changes'], 'std')
        for index, name in enumerate(('dynamics', 'articulation', 'pedal_depth', 'pedal_down_ratio', 'pedal_changes')):
            np.testing.assert_array_equal(result['features'][name]['00001']['raw'].values, a['dpa'][:, index])
        for channel, fs in result['features'].items():
            af, bf = fs['00001'], fs['00002']
            shared = af['relative'].mask & bf['relative'].mask
            np.testing.assert_allclose((af['relative'].values - bf['relative'].values)[shared],
                                       (af['raw'].values - bf['raw'].values)[shared], atol=1e-12)

    def test_embedding_excludes_missing_channels_without_zero_imputation(self):
        records = [record(f'{p}/{i:05}', piece=p, speed=1 + .01 * i, offset=.01*i)
                   for p in ('workA', 'workB', 'missing') for i in range(5)]
        for r in records:
            if r['piece'] == 'missing':
                r['dpa'][:, 2:] = np.nan
                r['dpa_mask'][:, 2:] = False
        args = SimpleNamespace(min_performances=5, min_beats=32, min_coverage=.5)
        normalized, groups, audits, scales = collect_normalized(records, args)
        self.assertEqual(len(normalized), 3)
        self.assertEqual(len(groups), 2)
        self.assertEqual(next(a for a in audits if a['piece'] == 'missing')['reason'], 'too_few_shared_valid_beats')
        self.assertEqual(groups[0]['values'].shape, (5, 7, 64))
        self.assertTrue(np.isfinite(groups[0]['values']).all())


if __name__ == '__main__':
    unittest.main()
