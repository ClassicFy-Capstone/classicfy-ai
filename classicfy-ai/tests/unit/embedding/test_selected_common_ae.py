import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('selected_common_ae',
    Path(__file__).resolve().parents[3] / 'scripts/train_selected_common_ae.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SelectedCommonAETest(unittest.TestCase):
    def test_short_tail_and_sparse_windows_preserve_every_observation_once(self):
        sequences = []
        for length in (2, 65, 140):
            valid = np.zeros((7, length), bool)
            valid[:, [0, length-1]] = True
            values = np.broadcast_to(np.arange(length), valid.shape).astype(np.float32)
            sequences.append({'values': values, 'valid': valid})
        data = runner.AllObservedWindows(sequences)
        self.assertEqual(len(data), 5)
        self.assertEqual(sum(int(data[i][1].sum()) for i in range(len(data))), 42)
        for si, seq in enumerate(sequences):
            recovered = np.zeros_like(seq['valid'], dtype=np.int32)
            for i, (owner, start) in enumerate(data.items):
                if owner != si:
                    continue
                x, mask = data[i]
                width = min(64, seq['values'].shape[1]-start)
                recovered[:, start:start+width] += mask[:, :width].numpy()
                np.testing.assert_array_equal(x[:, :width].numpy()[mask[:, :width]],
                    seq['values'][:, start:start+width][seq['valid'][:, start:start+width]])
                self.assertFalse(mask[:, width:].any())
            np.testing.assert_array_equal(recovered, seq['valid'].astype(np.int32))

    def test_unobserved_targets_do_not_change_loss_or_gradient(self):
        prediction = torch.zeros(1, 7, 64, requires_grad=True)
        target = torch.ones_like(prediction)
        valid = torch.zeros_like(prediction, dtype=torch.bool)
        valid[:, :, 0] = True
        target[:, :, 1:] = float('nan')
        sums, counts = runner.error_totals(prediction, target, valid)
        loss = runner.feature_loss(sums, counts)
        self.assertEqual(float(loss), 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertFalse(prediction.grad[:, :, 1:].any())


if __name__ == '__main__':
    unittest.main()
