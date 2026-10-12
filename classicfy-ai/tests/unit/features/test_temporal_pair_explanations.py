from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3] / 'scripts'))
from analyze_temporal_pair_explanations import classify_lag_contrast, dynamics_summary_close


class TemporalPairExplanationTests(unittest.TestCase):
    def classify(self, gaps):
        return classify_lag_contrast({'lag1':gaps[0]}, {'lag1':0.},
            [{'lag1':gaps[1]},{'lag1':gaps[2]}], [{'lag1':0.},{'lag1':0.}])

    def test_summary_bounds_apply_to_both_mean_and_width(self):
        self.assertTrue(dynamics_summary_close(0.,2.,.05,2.05))
        self.assertFalse(dynamics_summary_close(0.,2.,.2,2.05))
        self.assertFalse(dynamics_summary_close(0.,2.,.01,2.5))
        self.assertFalse(dynamics_summary_close(0.,0.,0.,0.))
        self.assertFalse(dynamics_summary_close(np.nan,2.,0.,2.))

    def test_full_gap_requires_same_direction_in_both_halves(self):
        self.assertTrue(self.classify([.3,.2,.15])[1])
        self.assertTrue(self.classify([-.3,-.2,-.15])[1])
        self.assertFalse(self.classify([.3,.2,-.15])[1])
        self.assertFalse(self.classify([.3,.2,.05])[1])

    def test_missing_half_cannot_qualify_for_either_case(self):
        _,divergent,matched=self.classify([.3,np.nan,.15])
        self.assertFalse(divergent)
        self.assertFalse(matched)
        self.assertEqual(self.classify([.02,.02,np.nan])[1:],(False,False))

    def test_matching_full_average_cannot_hide_opposing_halves(self):
        self.assertTrue(self.classify([.02,.04,-.03])[2])
        self.assertFalse(self.classify([.02,.3,-.3])[2])


if __name__=='__main__':
    unittest.main()
