from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from analyze_temporal_shape import analyze, restore_channels, verified_source_files


class TemporalShapeAnalysisTests(unittest.TestCase):
    def fixture(self):
        groups = []
        audits = []
        for piece, values in [('train', [0, 1, 2]), ('test', [0, 100, 200])]:
            groups.append(dict(piece=piece, grid='g', keys=[piece],
                               values=np.tile(values, (1, 7, 1)), shared_indices=np.array([0, 1, 3])))
            audits.append(dict(piece=piece, grid='g', total_intervals=4))
        return groups, audits, {'train': 'train', 'test': 'test'}

    def test_restore_does_not_create_false_adjacent_pairs(self):
        items = restore_channels(*self.fixture())
        seq = items[0]['channels']['tempo']
        np.testing.assert_array_equal(seq.mask, [1, 1, 0, 1])
        self.assertTrue(np.isnan(seq.values[2]))
        thresholds, rows = analyze(items)
        self.assertEqual(len(rows), 14)
        self.assertEqual(rows[0]['valid_pairs'], 1)
        self.assertEqual(rows[0]['valid_triplets'], 0)

    def test_held_out_values_never_fit_threshold(self):
        items = restore_channels(*self.fixture())
        thresholds, rows = analyze(items)
        self.assertTrue(all(t['value'] == 1 and t['pair_count'] == 1 for t in thresholds))
        self.assertTrue(all(t['reference_works'] == 1 for t in thresholds))
        self.assertEqual(rows[0]['large_ratio'], 0)
        self.assertEqual(rows[1]['large_ratio'], 1)

    def test_bad_indices_and_duplicate_keys_rejected(self):
        for indices in ([1, 0, 3], [0, 0, 3], [0, 1, 4]):
            groups, audits, split = self.fixture()
            groups[0]['shared_indices'] = np.array(indices)
            with self.assertRaises(ValueError):
                restore_channels(groups, audits, split)
        groups, audits, split = self.fixture()
        groups[1]['keys'] = groups[0]['keys']
        with self.assertRaises(ValueError):
            restore_channels(groups, audits, split)

    def test_reference_source_verification_rejects_edits_but_allows_additions(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            original = repo / 'classicfy-ai/src/features/models.py'
            original.parent.mkdir(parents=True)
            original.write_text('original\n')
            subprocess.run(['git', 'add', '.'], cwd=repo, check=True)
            subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            'commit', '-qm', 'fixture'], cwd=repo, check=True)
            (original.parent / 'temporal_shape.py').write_text('added\n')
            self.assertEqual(verified_source_files(repo, 'HEAD'), [original])
            original.write_text('modified\n')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                verified_source_files(repo, 'HEAD')


if __name__ == '__main__':
    unittest.main()
