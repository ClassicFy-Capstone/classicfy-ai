"""Whole-performance statistics, cohort boundaries, and immutable CLI outputs."""

from dataclasses import replace
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from integration import test_style_triplets as fixtures
from unit.preprocessing.musicxml_fixtures import XMLFixture
from embedding.note_performance_data import save_note_performance
from embedding.performance_style_data import prepare_performance_styles
from features.note_performance import R_REASONS, inverse_r_physical
from features.performance_time import BeatTimeMap
from features.performance_style_profile import (
    PerformanceStyleConfig, fit_performance_reference, fit_performance_scaler,
    scale_performance_profile, summarize_performance,
)
from features.style_profile import artifact, StyleConfig, fit_reference, make_windows, stable_id, summarize_window


class PerformanceStyleTest(XMLFixture):
    make_score = fixtures.StyleTripletsTest.make_score
    record = fixtures.StyleTripletsTest.record

    def records(self, bars=8, size=1, split='train'):
        xml = self.make_score(bars, size)
        return [self.record(xml, f'{split}-p{i}', i - 1, split=split) for i in range(3)]

    def profile(self, records, index=-1, config=PerformanceStyleConfig()):
        ref, arrays = fit_performance_reference(records, config)
        return summarize_performance(records[index], ref, arrays, config), ref, arrays

    def test_eight_bar_compatibility_and_chord_invariance(self):
        results = []
        for size in (1, 3):
            records = self.records(size=size)
            profile, _, _ = self.profile(records)
            old = summarize_window(records[-1].performance, make_windows(records[-1])[0], fit_reference(records))
            np.testing.assert_allclose(profile['g_raw'], old['g_raw'])
            self.assertEqual(profile['g_mask'], old['g_mask'])
            results.append(profile)
        np.testing.assert_allclose(results[0]['g_raw'], results[1]['g_raw'])

    def test_tail_is_included_and_statistics_are_not_window_average(self):
        records = self.records(bars=9)
        target = records[-1]
        r = target.performance
        values = r.values.copy()
        # Two zero-residual peer baselines and one recording whose tail differs strongly.
        records[0] = self.record(target.xml, 'baseline-a', 0)
        records[1] = self.record(target.xml, 'baseline-b', 0)
        values[:, 2] = np.r_[np.full(32, .3), np.full(4, .9)]
        physical, pm = inverse_r_physical(values, r.mask, r.local_tau)
        records[-1] = replace(target, performance=replace(r, values=values, physical=physical, physical_mask=pm))
        p, ref, arrays = self.profile(records)
        # Equal-value masses at -.2 and .4 have midpoint positions 4/9 and 17/18.
        expected = np.interp([.1, .5, .9], [4 / 9, 17 / 18], [-.2, .4])
        np.testing.assert_allclose([p['g_raw'][5], p['g_raw'][6]], [expected[1], expected[2] - expected[0]])
        self.assertTrue(p['all_features_valid'])
        self.assertEqual(p['scope']['tail_measures'], 1)
        p = summarize_performance(records[-1], ref, arrays, ae_intervals=[(0, 32)])
        self.assertAlmostEqual(p['scope']['ae_full_window_coverage'], 8 / 9)
        self.assertNotAlmostEqual(p['g_raw'][5], -.2)

    def test_missing_chord_notes_keep_original_weights(self):
        records = self.records(size=3)
        r = records[-1].performance
        mask, values, pm, physical, reasons = [v.copy() for v in (r.mask, r.values, r.physical_mask, r.physical, r.reasons)]
        mask[::3, 2], pm[::3, 2] = False, False
        values[::3, 2], physical[::3, 2] = 0, 0
        reasons[::3, 2] = R_REASONS.index('unmatched')
        records[-1] = replace(records[-1], performance=replace(r, mask=mask, values=values, physical_mask=pm,
                                                              physical=physical, reasons=reasons))
        p, _, _ = self.profile(records)
        self.assertAlmostEqual(p['channels'][2]['coverage'], 2 / 3)
        self.assertFalse(p['g_mask'][5])
        self.assertEqual(p['g_raw'][5], 0)

    def test_roughness_never_bridges_missing_beats(self):
        records = self.records()
        _, ref, arrays = self.profile(records)
        r = records[-1].performance
        valid = np.ones(32, bool)
        valid[10] = False
        beats = BeatTimeMap(r.beats.q, np.cumsum(np.r_[0, np.tile([.4, .5], 16)]), valid, r.beats.provenance)
        values, mask, reasons = r.values.copy(), r.mask.copy(), r.reasons.copy()
        values[:, 0] = np.where(valid, beats.log2_qpm, 0)
        mask[:, 0] = valid
        reasons[10, 0] = R_REASONS.index('invalid_beat')
        physical, pm = inverse_r_physical(values, mask, r.local_tau)
        target = replace(records[-1], performance=replace(r, values=values, mask=mask, reasons=reasons,
                                                        physical=physical, physical_mask=pm, beats=beats))
        p = summarize_performance(target, ref, arrays)
        self.assertEqual(p['roughness']['valid_edges'], 29)
        self.assertAlmostEqual(p['g_raw'][2], abs(np.log2(.5 / .4)))

    def test_split_specific_references_and_duplicate_recordings(self):
        records = self.records()
        alias = replace(records[0], performance_id='alias')
        _, ref, _ = self.profile([*records, alias])
        self.assertEqual(len(ref['members']), 3)
        self.assertEqual(len(ref['excluded_duplicate_ids']), 1)
        p, ref, _ = self.profile(self.records(split='validation'))
        self.assertEqual(ref['split'], 'validation')
        self.assertIn('evaluation-labels-only', ref['reference_role'])
        self.assertTrue(p['all_features_valid'])
        with self.assertRaises(ValueError):
            fit_performance_reference([records[0], self.records(split='test')[0]])
        missing, _, _ = self.profile(records[:2])
        self.assertFalse(any(missing['g_mask']))
        self.assertIn('reference_insufficient', missing['reason_codes'])

    def test_different_beat_extents_match_exact_intervals(self):
        records = self.records()
        r = records[0].performance
        # Only its tempo observations lose the initial/final beats; note rows stay intact.
        beats = BeatTimeMap(r.beats.q[1:-1], r.beats.seconds[1:-1], provenance=r.beats.provenance)
        mask, values, reasons = r.mask.copy(), r.values.copy(), r.reasons.copy()
        mask[[0, -1], 0], values[[0, -1], 0] = False, 0.
        reasons[[0, -1], 0] = R_REASONS.index('invalid_beat')
        ids = r.beat_id - 1
        ids[[0, -1]] = -1
        physical, pm = inverse_r_physical(values, mask, r.local_tau)
        records[0] = replace(records[0], performance=replace(r, beats=beats, mask=mask, values=values,
                                                           physical=physical, physical_mask=pm, reasons=reasons, beat_id=ids))
        p, ref, arrays = self.profile(records)
        self.assertEqual(arrays['beat_counts'].tolist(), [2] + [3] * 30 + [2])
        self.assertEqual(p['channels'][0]['valid_count'], 30)
        self.assertAlmostEqual(p['channels'][0]['coverage'], 30 / 32)
        self.assertTrue(p['g_mask'][0])
        self.assertEqual(p['roughness']['valid_edges'], 29)

    def test_scaler_train_only_constants_and_minimum_samples(self):
        records = self.records()
        ref, arrays = fit_performance_reference(records)
        profiles = [summarize_performance(rec, ref, arrays) for rec in records]
        cfg = PerformanceStyleConfig(min_scaler_samples=3)
        scaler = fit_performance_scaler(profiles, cfg)
        self.assertTrue(scaler['trained_feature_mask'][0])
        self.assertFalse(scaler['trained_feature_mask'][2])
        self.assertEqual(scaler['status'][2], 'constant')
        result = scale_performance_profile(profiles[-1], scaler)
        self.assertEqual(result['g_scaled'][2], 0.)
        with self.assertRaisesRegex(ValueError, 'train'):
            fit_performance_scaler([{**profiles[0], 'split': 'test'}], cfg)
        disabled = fit_performance_scaler(profiles)
        self.assertFalse(any(disabled['trained_feature_mask']))

    def test_unknown_pedal_and_observed_zero_are_distinct(self):
        records = self.records()
        r = records[-1].performance
        values = r.values.copy()
        values[:, 4:] = 0.
        physical, pm = inverse_r_physical(values, r.mask, r.local_tau)
        observed = replace(records[-1], performance=replace(r, values=values, physical=physical, physical_mask=pm))
        ref, arrays = fit_performance_reference(records)
        known = summarize_performance(observed, ref, arrays)
        self.assertTrue(all(known['g_mask'][9:]))
        mask, reasons = r.mask.copy(), r.reasons.copy()
        mask[:, 4:] = False
        reasons[:, 4:] = R_REASONS.index('unknown_pedal')
        physical, pm = inverse_r_physical(values, mask, r.local_tau)
        unknown = replace(observed, performance=replace(r, values=values, mask=mask, reasons=reasons,
                                                        physical=physical, physical_mask=pm))
        missing = summarize_performance(unknown, ref, arrays)
        self.assertFalse(any(missing['g_mask'][9:]))
        self.assertEqual(missing['g_raw'][9:], [0.] * 14)

    def test_cli_pipeline_reproducible_and_sources_preserved(self):
        records = self.records(bars=9)
        entries, items = [], []
        for rec in records:
            cache = self.folder / rec.performance_id
            manifest = save_note_performance(rec.performance, cache)
            windows = make_windows(rec)
            wp = self.folder / f'{rec.performance_id}-windows.json'
            wp.write_text(json.dumps(windows))
            entries.append({'performance_cache': str(cache), 'performance_id': rec.performance_id,
                            'recording_id': rec.recording_id, 'record_key': rec.performance.record_key,
                            'work_key': rec.performance.work_key, 'split': 'train',
                            'score_source_sha256': rec.xml.source_sha256, 'R_fingerprint': manifest['fingerprint'],
                            'xml_path': str(self.folder / 'score-1.musicxml'), 'windows_file': str(wp),
                            'windows_sha256': hashlib.sha256(wp.read_bytes()).hexdigest()})
            items.extend({'window': {k: w[k] for k in ('performance_id', 'q_start', 'q_end')}}
                         for w in windows if not w['tail'])
        payload = artifact('ae-training-index', StyleConfig(), sources=entries, items=items)
        index = self.folder / 'index.json'
        index.write_text(json.dumps(payload))
        a, b = self.folder / 'a', self.folder / 'b'
        prepare_performance_styles(index, a)
        command = [sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/prepare_performance_style.py'),
                   '--index', str(index), '--out', str(b)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/validate_performance_style.py'),
                        '--directory', str(b)], check=True, capture_output=True, text=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/report_performance_style.py'),
                        '--directory', str(b), '--out', str(self.folder / 'report')],
                       check=True, capture_output=True, text=True)
        import csv
        with (self.folder / 'report/feature_statistics.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 69)
        for name in ('profiles_raw.jsonl', 'profiles_scaled.jsonl', 'scaler_train.json', 'report.json'):
            self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
        self.assertTrue(json.loads((a / 'verification.json').read_text())['passed'])
        with self.assertRaises(FileExistsError):
            prepare_performance_styles(index, a)
        # An index with a changed source fingerprint is rejected before creating output.
        payload['sources'][0]['R_fingerprint'] = 'changed'
        payload['id'] = stable_id({k: v for k, v in payload.items() if k != 'id'})
        index.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            prepare_performance_styles(index, self.folder / 'invalid')


if __name__ == '__main__':
    unittest.main()
