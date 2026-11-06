import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from evaluate_representative_passages import overlap,summarize


class ThemeOverlapTests(unittest.TestCase):
    def test_short_preview_inside_long_theme_is_not_full_theme_recovery(self):
        m=overlap((10,30),[(0,100)])
        self.assertEqual(m['inside_fraction'],1.)
        self.assertAlmostEqual(m['iou'],.2)
        self.assertFalse(m['iou_hit'])
        self.assertTrue(m['inside_80'])

    def test_overlap_union_does_not_double_count_annotations(self):
        self.assertEqual(overlap((0,20),[(0,15),(5,20)])['inside_fraction'],1.)

    def test_rejection_does_not_disappear_from_denominator(self):
        rows=[]
        for pid,interval in [('a',(0,20)),('b',None)]:
            rows.append(dict(perf_id=pid,composition_id='work',source_sha256=pid,reference_task='first',method='v2',
                historical_split='all',available=interval is not None,**overlap(interval,[(0,20)])))
        r=summarize(pd.DataFrame(rows))[0]
        self.assertEqual(r['unconditional_work_macro']['iou_hit'],.5)
        self.assertEqual(r['accepted_only_work_macro']['iou_hit'],1.)
        self.assertEqual(r['performance_coverage'],.5)


if __name__=='__main__':unittest.main()
