import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('pedal_probe',
    Path(__file__).resolve().parents[3]/'scripts/diagnose_common_ae_pedal.py')
probe=importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class PedalProbeTest(unittest.TestCase):
    def test_linear_signal_and_no_test_label_tuning(self):
        rng=np.random.default_rng(32)
        x=rng.normal(size=(120,3))
        y=x@np.array([[2.,1.],[-1.,3.],[.5,-2.]])+4
        split=np.array(['train']*80+['validation']*20+['test']*20)
        first=probe.regress(x,y,split)
        self.assertTrue(all(v>.99 for v in first['test_r2']))
        y[split=='test']+=1000
        second=probe.regress(x,y,split)
        self.assertEqual(first['selected_alpha'],second['selected_alpha'])


if __name__=='__main__':unittest.main()
