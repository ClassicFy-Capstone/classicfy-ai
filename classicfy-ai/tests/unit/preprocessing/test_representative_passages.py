import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from preprocessing.representative_passages import (PassageConfig,propose_passages,
    subsequence_match,map_passages,choose_passage)


def music():
    rng=np.random.default_rng(61)
    pitches=np.r_[rng.integers(48,75,48),rng.integers(60,84,48)]
    return np.array([[int(p),i*.5,i*.5+.3,80] for i,p in enumerate(pitches)],float)


class RepresentativePassageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):threadpool_limits(1)

    def test_elastic_repetition_matches_a_stretched_pattern(self):
        query=np.eye(8,dtype=np.float32)
        target=np.repeat(query,2,axis=0)
        scores,starts=subsequence_match(query@target.T)
        self.assertGreater(scores[-1],.9)
        self.assertLessEqual(starts[-1],1)

    def test_content_candidates_ignore_velocity_and_cover_whole_piece(self):
        a=music();b=a.copy();b[:,3]=np.arange(len(b))%80+20
        config=PassageConfig(max_frames=100,stride_seconds=4.)
        ca=propose_passages(a,config);cb=propose_passages(b,config)
        self.assertEqual(ca,cb)
        self.assertLessEqual(len(ca),4)
        self.assertTrue(all(c['boundary_type'].endswith('not_cadence') for c in ca))
        self.assertTrue(any(c['end_seconds']>24 for c in ca))

    def test_mapping_recovers_known_tempo_change_and_offset(self):
        a=music();b=a.copy();b[:,1:3]=b[:,1:3]*1.5+4.
        candidates=[dict(start_seconds=12.,end_seconds=32.)]
        mapped=map_passages(a,b,candidates,PassageConfig(max_frames=150))[0]
        self.assertTrue(mapped['alignment_accepted'])
        self.assertAlmostEqual(mapped['start_seconds'],22.,delta=1.6)
        self.assertAlmostEqual(mapped['end_seconds'],52.,delta=1.6)

    def test_single_performance_uses_representativeness_not_fake_variance(self):
        result=choose_passage([{},{}],[[np.ones(3),np.ones(3)]])
        self.assertFalse(result['style_selection_available'])
        self.assertEqual(result['selected_candidate'],0)
        self.assertIsNone(result['selection_stability'])

    def test_median_pair_distance_resists_single_extreme_recording(self):
        vectors=[[np.array([v]),np.array([w])] for v,w in [(0,0),(0,1),(0,2),(1000,3),(0,4)]]
        result=choose_passage([{},{}],vectors)
        self.assertEqual(result['selected_candidate'],1)
        self.assertTrue(result['style_selection_available'])


if __name__=='__main__':unittest.main()
