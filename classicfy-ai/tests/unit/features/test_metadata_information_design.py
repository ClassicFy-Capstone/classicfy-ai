import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from prepare_metadata_information_experiment import temporal_targets, assigned_roles


class MetadataInformationDesignTests(unittest.TestCase):
    def test_change_concentration_distinguishes_equal_mean_local_changes(self):
        t = np.arange(192, dtype=float)
        uniform = np.tile(np.arange(32, dtype=float), 6)
        concentrated = np.concatenate([np.arange(32) * s for s in (.2, .2, .2, 1.8, 1.8, 1.8)])
        mask = np.ones(192, dtype=bool)
        a, b = temporal_targets(t, uniform, mask), temporal_targets(t, concentrated, mask)
        self.assertAlmostEqual(a['change_concentration_cv'], 0)
        self.assertAlmostEqual(b['change_concentration_cv'], .8)
        self.assertEqual(b['rms_blocks'], 6)
        scaled = temporal_targets(t, concentrated * 10 + 5, mask)
        self.assertAlmostEqual(b['change_concentration_cv'], scaled['change_concentration_cv'])

    def test_local_relations_preserve_strength_when_signs_cancel(self):
        wave = np.sin(2 * np.pi * np.arange(64) / 16)
        t = np.tile(wave, 3)
        d = np.concatenate((wave, wave, -wave))
        result = temporal_targets(t, d, np.ones(192, dtype=bool))
        self.assertAlmostEqual(result['coupling_local_strength'], 1)
        self.assertAlmostEqual(result['coupling_local_variation'], np.sqrt(8 / 9))
        self.assertEqual(result['correlation_blocks'], 3)

    def test_missing_beats_are_not_joined_or_used_as_outliers(self):
        t = np.arange(96, dtype=float)
        d = np.tile(np.arange(32, dtype=float), 3)
        mask = np.ones(96, dtype=bool)
        for start in (0, 32, 64):
            mask[start + 5] = mask[start + 10] = False
            d[start + 5] = 1e9
            d[start + 10] = -1e9
        result = temporal_targets(t, d, mask)
        self.assertEqual(result['rms_minimum_pairs'], 27)
        self.assertAlmostEqual(result['change_concentration_cv'], 0)

    def test_constant_relations_and_zero_change_stay_undefined(self):
        t = np.arange(192, dtype=float)
        result = temporal_targets(t, np.zeros(192), np.ones(192, dtype=bool))
        self.assertEqual(result['change_concentration_status'], 'zero_mean_change')
        self.assertEqual(result['constant_correlation_blocks'], 3)
        self.assertTrue(np.isnan(result['change_concentration_cv']))
        self.assertTrue(np.isnan(result['coupling_local_strength']))

    def test_short_final_block_is_not_padded_or_reused(self):
        t = np.arange(191, dtype=float)
        result = temporal_targets(t, np.sin(t), np.ones(191, dtype=bool))
        self.assertEqual(result['possible_rms_blocks'], 5)
        self.assertEqual(result['possible_correlation_blocks'], 2)
        self.assertTrue(np.isnan(result['coupling_local_strength']))

    def test_family_overlap_is_excluded_without_moving_encoder_training(self):
        rows = [dict(key='a', family_id='F1', split='train'), dict(key='b', family_id='F1', split='validation'),
                dict(key='c', family_id='F2', split='validation'), dict(key='d', family_id='F2', split='test'),
                dict(key='e', family_id='F3', split='test')]
        roles, overlap = assigned_roles(rows)
        self.assertEqual(roles['a'][0], 'train')
        self.assertEqual(roles['b'][0], 'excluded')
        self.assertEqual(roles['c'][0], 'validation')
        self.assertEqual(roles['d'][0], 'excluded')
        self.assertEqual(roles['e'][0], 'test')
        self.assertEqual(len(overlap), 2)


if __name__ == '__main__':
    unittest.main()
