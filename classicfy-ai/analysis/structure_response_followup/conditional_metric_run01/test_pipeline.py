"""Check evaluation meaning and fair complete-group validation."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from data import GroupCatalog, load_mapping
from model import ConditionalMetricEncoder, previous
from train import validation_plan, validate, MODES
from evaluate import within_work, AXES


class WithinWorkTests(unittest.TestCase):
    def test_leave_self_out_and_uniform_random_expectation(self):
        metadata = [dict(key=f'r{i}', piece='p', grid='g', family_id='f', evaluation_role='test') for i in range(6)]
        d = SimpleNamespace(ids=np.arange(6), qids=np.array([0]), metadata=metadata)
        # The query's own distance is smallest but must never be returned.
        distances = np.array([[0., 1., 2., 3., 4., 5.]])
        judges = {axis: np.array([[0., 1., 2., 3., 4., 5.]]) for axis in AXES}
        scores, ranks = within_work('x', distances, judges, d, 1)
        self.assertEqual([r['candidate'] for r in ranks], ['r1', 'r2', 'r3'])
        self.assertGreater(scores[0]['E-PN'], 50.)
        self.assertEqual(scores[0]['E-PN_random'], 50.)
        self.assertEqual(scores[0]['candidate_count'], 5)

    def test_tied_score_only_predictions_do_not_create_above_chance_scores(self):
        metadata = [dict(key=f'r{i}', piece='p', grid='g', family_id='f', evaluation_role='test') for i in range(6)]
        d = SimpleNamespace(ids=np.arange(6), qids=np.array([0]), metadata=metadata)
        distances = np.zeros((1, 6)); judges = {axis: np.zeros((1, 6)) for axis in AXES}
        scores, ranks = within_work('x', distances, judges, d, 1)
        self.assertTrue(all(scores[0][axis] == 50 for axis in AXES))
        self.assertNotIn('r0', [r['candidate'] for r in ranks])


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = GroupCatalog(0, load_mapping())

    def test_full_validation_group_weights_match_hierarchy(self):
        batch, _, _, w, gw, slices, _ = validation_plan(self.catalog, 20261014)
        self.assertAlmostEqual(float(w.sum()), 1., places=6)
        for weight, sl in zip(gw, slices): self.assertAlmostEqual(float(w[sl].sum()), float(weight), places=7)
        self.assertFalse(set(batch['owners'].tolist()) & set(self.catalog.data.gids.tolist()))
        test = set(self.catalog.data.arrays['owner'][self.catalog.data.roles['test']].tolist())
        self.assertFalse(set(batch['owners'].tolist()) & test)

    def test_metric_on_changes_only_checkpoint_objective_term(self):
        torch.manual_seed(1); model = ConditionalMetricEncoder().eval()
        plan = validation_plan(self.catalog, 20261014)
        off = validate(model, plan, False); on = validate(model, plan, True)
        self.assertAlmostEqual(on['loss'] - off['loss'], on['metric'], places=7)
        for name in ('shape', 'amplitude', 'direction', 'peaks', 'lag', 'metric', 'rec'):
            self.assertEqual(on[name], off[name])

    def test_all_control_initializations_are_identical(self):
        states = []
        for fused, _ in MODES.values():
            torch.manual_seed(20261014); states.append(ConditionalMetricEncoder(fused).state_dict())
        for state in states[1:]:
            self.assertTrue(all(torch.equal(states[0][key], state[key]) for key in state))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
