import sys
from pathlib import Path
import unittest

import numpy as np

AI = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(AI / 'src'), str(AI / 'scripts')]
from embedding.information_probe import (RidgeReadout, family_weights, metadata_matrix,
    choose_alphas, work_errors, macro_error, paired_bootstrap, shuffle_within_work)
from run_metadata_information_experiment import fit_case


class InformationProbeTests(unittest.TestCase):
    def row(self, key, piece, family, role='train', composer='Bach', kind='fugue'):
        return dict(key=key, piece=piece, family_id=family, evaluation_role=role,
                    grid='g', composer=composer, work_type=kind)

    def test_family_work_recording_weights(self):
        rows = [self.row('a','w1','f1'), self.row('b','w1','f1'), self.row('c','w2','f1'),
                self.row('d','w3','f2')]
        np.testing.assert_allclose(family_weights(rows), [.125,.125,.25,.5])

    def test_weighted_ridge_matches_augmented_least_squares(self):
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(25,7)), rng.normal(size=(25,2))
        x[:,6] = 3
        w = np.arange(1,26,dtype=float); w /= w.sum()
        model = RidgeReadout.prepare(x,y,w)
        alpha = np.array([.01,10.])
        query = rng.normal(size=(4,7))
        z = (x[:,model.active]-model.mean[model.active])/model.scale[model.active]
        expected = []
        for j in range(2):
            design = np.vstack((np.sqrt(w)[:,None]*z, np.sqrt(alpha[j])*np.eye(z.shape[1])))
            target = np.r_[np.sqrt(w)*(y[:,j]-w@y[:,j]), np.zeros(z.shape[1])]
            beta = np.linalg.lstsq(design,target,rcond=None)[0]
            expected.append(((query[:,model.active]-model.mean[model.active])/model.scale[model.active])@beta+w@y[:,j])
        np.testing.assert_allclose(model.predict(query,alpha),np.array(expected).T,atol=1e-11)

    def test_constant_target_keeps_unpenalized_intercept(self):
        x=np.arange(20).reshape(10,2);y=np.repeat(7.,10)
        model=RidgeReadout.prepare(x,y,np.repeat(.1,10))
        np.testing.assert_allclose(model.predict(np.array([[1e6,-1e6]]),1000),[[7.]])

    def test_unknown_metadata_cannot_expand_training_vocabulary(self):
        training=[self.row('a','w','f')]
        rows=[self.row('b','v','g','test','Unknown','etude')]
        x,names,counts=metadata_matrix(training,rows)
        self.assertEqual(x.shape,(1,3));self.assertEqual(counts,[1,1,1])
        self.assertEqual(x.sum(),0);self.assertFalse(any('Unknown' in name for name in names))

    def test_alpha_tie_prefers_stronger_regularization(self):
        scores=np.array([[1.,3.],[1.+1e-13,2.],[2.,2.]])
        np.testing.assert_equal(choose_alphas(scores,[.01,1.,10.]),[1.,10.])

    def test_within_work_metric_separates_mean_and_recording_error(self):
        rows=[self.row('a','w1','f1'),self.row('b','w1','f1'),self.row('c','w2','f2'),self.row('d','w2','f2')]
        y=np.array([101.,99.,13.,7.])
        constant=work_errors(rows,y,np.array([100.,100.,10.,10.]))
        self.assertEqual(macro_error(constant,'within_mse'),5.)
        shifted=work_errors(rows,y,y+100)
        self.assertEqual(macro_error(shifted,'within_mse'),0.)
        self.assertEqual(macro_error(shifted,'global_mse'),10000.)

    def test_paired_bootstrap_preserves_identical_errors(self):
        rows=[dict(piece='w1',family_id='f1',global_mse=1.),dict(piece='w2',family_id='f2',global_mse=2.)]
        a,b=paired_bootstrap(rows,rows,'global_mse',repetitions=50)
        np.testing.assert_equal(a,np.zeros(50));np.testing.assert_equal(b,np.zeros(50))

    def test_shuffle_never_crosses_piece_grid_or_split(self):
        rows=[self.row(str(i),'w'+str(i//3),'f','train' if i<3 else 'test') for i in range(6)]
        values=np.arange(12).reshape(6,2)
        shuffled,source=shuffle_within_work(rows,values,20261101)
        for i,j in enumerate(source):
            self.assertEqual(rows[i]['piece'],rows[j]['piece'])
            self.assertEqual(rows[i]['evaluation_role'],rows[j]['evaluation_role'])
            np.testing.assert_equal(shuffled[i],values[j])

    def test_eval_targets_never_change_cv_or_prediction(self):
        rows=[self.row(str(i),'w'+str(i//2),'f'+str(i//2)) for i in range(10)]
        rows += [self.row('v','wv','fv','validation'),self.row('t','wt','ft','test')]
        rng=np.random.default_rng(6);x=rng.normal(size=(12,3));y=(x[:,0]*2+x[:,1])[:,None]
        folds={str(i):i//2 for i in range(10)}
        first=fit_case(rows,x,y,folds,[.001,.1,10.])
        y[10:]=1e9
        second=fit_case(rows,x,y,folds,[.001,.1,10.])
        np.testing.assert_allclose(first['predictions'],second['predictions'])
        np.testing.assert_allclose(first['cv_scores'],second['cv_scores'])
        np.testing.assert_equal(first['selected_alphas'],second['selected_alphas'])


if __name__ == '__main__':
    unittest.main()
