"""Check the adapter against the existing real ATEPP pipeline."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

AI = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI / 'scripts'))
from atepp_feature_analysis import normalize_group, resolve_source, sha256
from validate_embedding import CHANNELS


class RealAteppNormalizationTest(unittest.TestCase):
    def test_real_dualdtw_cache_preserves_existing_raw_residual_and_standardized_values(self):
        root = Path(os.environ.get('ATEPP_ROOT', AI.parent.parent / 'ATEPP_dataset'))
        audit_path = AI / 'analysis/atepp_research/midi_audit.csv'
        summary_path = AI / 'analysis/atepp_research/performance_summary.csv'
        if not root.is_dir() or not audit_path.is_file():
            self.skipTest('ATEPP DualDTW caches are not available')
        audit = pd.read_csv(audit_path, dtype={'perf_id': str, 'composition_id': str}).fillna('')
        summary = pd.read_csv(summary_path, dtype={'perf_id': str, 'composition_id': str})
        candidates = sorted((g for _, g in summary.groupby('norm_group') if len(g) >= 5),
                            key=lambda g: (abs(len(g) - 8), g.iloc[0].norm_group))
        selected = candidates[0]
        rows = audit[audit.perf_id.isin(selected.perf_id)]
        self.assertEqual(len(rows), len(selected))
        records = []
        for row in rows.to_dict('records'):
            with np.load(resolve_source(row['raw_cache'], root), allow_pickle=False) as z:
                record = {k: z[k].copy() for k in z.files}
            record.update(key=row['perf_id'], piece='ATEPP/' + row['composition_id'] + '/' +
                          sha256(resolve_source(row['local_score'], root))[:12])
            records.append(record)
        result = normalize_group(records)
        for row in selected.to_dict('records'):
            with np.load(resolve_source(row['normalized_cache'], root), allow_pickle=False) as z:
                np.testing.assert_array_equal(z['score_beats'], result['records'][0]['score_beats'])
                np.testing.assert_array_equal(z['channels'], np.array(CHANNELS))
                for index, channel in enumerate(CHANNELS):
                    f = result['features'][channel][row['perf_id']]
                    for stage, cached_name in [('raw', 'raw'), ('relative', 'relative'), ('standardized', 'normalized')]:
                        np.testing.assert_allclose(f[stage].values, z[cached_name][:, index],
                                                   rtol=0, atol=1e-10, equal_nan=True)
                    np.testing.assert_array_equal(f['standardized'].mask, z['mask'][:, index])


if __name__ == '__main__':
    unittest.main()
