import unittest

import numpy as np

from embedding.standalone_structure import standalone_distances


class StandaloneStructureTest(unittest.TestCase):
    def inputs(self):
        rng = np.random.default_rng(4)
        d = {name: rng.normal(size=(6, size)) for name, size in
             [('B', 14), ('T', 15), ('F', 66), ('P', 66), ('H', 198)]}
        rows = [dict(family_id=str(i//2), piece=str(i//2), grid='g') for i in range(6)]
        return d, rows, np.arange(4)

    def test_removing_b_really_removes_its_effect(self):
        d, rows, ids = self.inputs()
        first, _ = standalone_distances(d, rows, ids)
        d['B'][4:] += 1000
        second, _ = standalone_distances(d, rows, ids)
        for name in ('T', 'F', 'P', 'H', 'P+H'):
            np.testing.assert_array_equal(first[name], second[name])
        self.assertFalse(np.allclose(first['B+P'][4:], second['B+P'][4:]))

    def test_equal_block_combination_is_not_coordinate_concatenation(self):
        d, rows, ids = self.inputs()
        arms, _ = standalone_distances(d, rows, ids)
        np.testing.assert_allclose(arms['P+H']**2, (arms['P']**2+arms['H']**2)/2)
        np.testing.assert_allclose(arms['B+P']**2, (arms['B']**2+arms['P']**2)/2)

    def test_training_scales_do_not_read_queries(self):
        d, rows, ids = self.inputs()
        _, first = standalone_distances(d, rows, ids)
        for x in d.values():
            x[4:] *= 500
        _, second = standalone_distances(d, rows, ids)
        self.assertEqual(first, second)

    def test_undefined_inputs_are_not_imputed(self):
        d, rows, ids = self.inputs()
        d['H'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            standalone_distances(d, rows, ids)


if __name__ == '__main__':
    unittest.main()
