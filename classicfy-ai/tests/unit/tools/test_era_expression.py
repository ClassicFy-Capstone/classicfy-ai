"""Check physical units, local relative variation, and observed/missing handling."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from analyze_era_expression import summarize_channel, display_value, observed_pedal_mask, balanced_selection, CHANNELS, ERAS


class EraExpressionTests(unittest.TestCase):
    def summarize(self, values, channel, mask=None):
        values = np.asarray(values, float)
        if mask is None:
            mask = np.ones(values.shape, bool)
        return summarize_channel(values, mask, channel, values.shape[1])

    def test_dynamics_level_and_common_pattern_deviation_in_velocity_units(self):
        result, arrays = self.summarize(np.array([[50]*64, [100]*64])/127, 'dynamics')
        self.assertAlmostEqual(float(display_value('dynamics', result['absolute_value'])), 75)
        self.assertAlmostEqual(float(display_value('dynamics', result['relative_value'])), 25)
        np.testing.assert_allclose(arrays['signed_relative']*127, [-25, 25])

    def test_crossing_dynamics_curves_are_not_lost_by_signed_averaging(self):
        values = np.array([[40, 80]*32, [80, 40]*32])/127
        result, arrays = self.summarize(values, 'dynamics')
        np.testing.assert_allclose(arrays['signed_relative'], [0, 0], atol=1e-12)
        self.assertAlmostEqual(float(display_value('dynamics', result['relative_value'])), 20)

    def test_global_speed_differences_alone_are_not_rubato(self):
        result, _ = self.summarize(np.log2([[100]*64, [200]*64]), 'rubato')
        self.assertAlmostEqual(result['absolute_value'], 0)
        self.assertAlmostEqual(result['relative_value'], 0)

    def test_identical_rubato_shapes_have_absolute_but_no_relative_amount(self):
        pattern = np.array([-.2, .2]*32)
        result, _ = self.summarize(np.array([pattern+7, pattern+8]), 'rubato')
        self.assertAlmostEqual(result['absolute_value'], .2)
        self.assertAlmostEqual(result['relative_value'], 0)
        self.assertAlmostEqual(float(display_value('rubato', result['absolute_value'])), 100*(2**.2-1))

    def test_opposite_rubato_shapes_retain_local_relative_amount(self):
        pattern = np.array([-.2, .2]*32)
        result, _ = self.summarize(np.array([pattern+7, -pattern+7]), 'rubato')
        self.assertAlmostEqual(result['absolute_value'], .2)
        self.assertAlmostEqual(result['relative_value'], .2)

    def test_unknown_pedal_is_missing_but_observed_zero_is_zero(self):
        values = np.zeros((2, 64))
        unknown, _ = self.summarize(values, 'pedal_depth', np.zeros_like(values, bool))
        observed, _ = self.summarize(values, 'pedal_depth')
        self.assertTrue(np.isnan(unknown['absolute_value']))
        self.assertEqual(observed['absolute_value'], 0)
        self.assertEqual(observed['relative_value'], 0)

    def test_singleton_does_not_invent_relative_value(self):
        result, arrays = self.summarize([[.4]*64], 'pedal_down_ratio')
        self.assertAlmostEqual(float(display_value('pedal_down_ratio', result['absolute_value'])), 40)
        self.assertTrue(np.isnan(result['relative_value']))
        self.assertTrue(np.isnan(arrays['relative'][0]))

    def test_same_positions_are_required_across_every_recording(self):
        values = np.array([[.2]*64, [.8]*64]); values[0, 0] = 100
        mask = np.ones(values.shape, bool); mask[1, 0] = False
        result, arrays = self.summarize(values, 'pedal_depth', mask)
        self.assertEqual(result['shared_beats'], 63)
        self.assertFalse(arrays['shared_mask'][0])
        self.assertAlmostEqual(result['absolute_value'], .5)

    def test_sparse_common_positions_are_not_imputed(self):
        values = np.ones((2, 64)); mask = np.zeros_like(values, bool); mask[:, :16] = True
        result, arrays = self.summarize(values, 'dynamics', mask)
        self.assertEqual(result['status'], 'insufficient_common_positions')
        self.assertIsNone(arrays)

    def test_original_units_are_preserved_under_level_shift_and_scale(self):
        values = np.array([[40]*64, [60]*64])/127
        before, _ = self.summarize(values, 'dynamics')
        shifted, _ = self.summarize(values+20/127, 'dynamics')
        scaled, _ = self.summarize(values*1.5, 'dynamics')
        self.assertAlmostEqual(shifted['absolute_value']-before['absolute_value'], 20/127)
        self.assertAlmostEqual(shifted['relative_value'], before['relative_value'])
        self.assertAlmostEqual(scaled['relative_value'], before['relative_value']*1.5)

    def test_first_pedal_event_establishes_state_but_not_a_transition(self):
        states = observed_pedal_mask([True]*3, np.array([0., 1., 2.]), 1.)
        changes = observed_pedal_mask([True]*3, np.array([0., 1., 2.]), 1., changes=True)
        np.testing.assert_array_equal(states, [False, True, True])
        np.testing.assert_array_equal(changes, [False, False, True])
        self.assertFalse(observed_pedal_mask([True]*3, np.array([0., 1., 2.]), None).any())

    def balanced_rows(self):
        rows = []
        for era in ERAS:
            for i in range(4):
                for channel in CHANNELS:
                    rows.append({'composition_id': f'{era}/{i}', 'work_id': f'work/{era}/{i}',
                        'era': era, 'era_ko': era, 'composer': f'composer/{i//2}', 'track': str(i),
                        'channel': channel, 'absolute_value': i, 'relative_value': i/10})
        return rows

    def test_balanced_selection_ignores_values_and_input_order(self):
        rows = self.balanced_rows()
        first, _, target = balanced_selection(rows, maximum=2)
        changed = [dict(r, absolute_value=999, relative_value=999) for r in reversed(rows)]
        second, _, _ = balanced_selection(changed, maximum=2)
        self.assertEqual(target, 2)
        self.assertEqual({r['composition_id'] for r in first}, {r['composition_id'] for r in second})
        self.assertEqual(len(first), 6)

    def test_balanced_selection_requires_all_channels(self):
        rows = self.balanced_rows()
        rows[0]['relative_value'] = np.nan
        selected, _, target = balanced_selection(rows)
        self.assertEqual(target, 3)
        self.assertNotIn(rows[0]['composition_id'], {r['composition_id'] for r in selected})
        self.assertEqual(len(selected), 9)


if __name__ == '__main__':
    unittest.main()
