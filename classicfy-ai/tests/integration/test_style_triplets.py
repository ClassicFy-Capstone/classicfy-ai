"""Semantic contracts for the pre-AE eight-bar weak supervision pipeline."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from embedding.note_performance_data import save_note_performance
from embedding.style_profile_data import (extract_profile_artifact, load_style_artifact,
    make_window_artifact, save_style_artifact, style_report)
from embedding.style_triplets import (anchor_signature, build_triplets, calibrate_distances,
    candidate_allowed, profile_distance, validate_triplet)
from features.note_performance import NotePerformanceFeatures, R_REASONS, inverse_r_physical
from features.note_score_context import extract_note_score_context
from features.performance_time import BeatTimeMap
from features.style_profile import (G_NAMES, StyleConfig, StyleRecording, artifact, fit_profile_scaler,
    fit_reference, make_windows, reference_delta, scale_profile, summarize_window, weighted_quantile)
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, measure, note, score_xml


class StyleTripletsTest(XMLFixture):
    def make_score(self, bars=8, size=1, prefix='n'):
        return self.read(score_xml(*(measure((attributes() if b == 0 else '') + ''.join(
            note(f'{prefix}{b}-{q}-{c}', step=('C', 'E', 'G')[c % 3], chord=c > 0)
            for q in range(4) for c in range(size)), b + 1) for b in range(bars))))

    def record(self, xml, pid, style=0., work='work', split='train', recording=None):
        s = extract_note_score_context(xml)
        end = float(xml.visits[-1].onset + xml.visits[-1].duration)
        edges = np.arange(0, end + 1.)
        tau = .5 * 2 ** (-style * .3)
        beats = BeatTimeMap(edges, edges * tau, provenance={'granularity': 'beat', 'source': 'synthetic'})
        n = len(xml.notes)
        q = np.array([float(note.onset) for note in xml.notes])
        v, mask = np.zeros((n, 11)), np.ones((n, 11), bool)
        v[:, 0] = beats.log2_qpm[np.minimum(q.astype(int), len(beats.valid) - 1)]
        v[:, 1] = style * (.03 + .002 * np.sin(q))
        v[:, 2] = .5 + style * (.04 + .005 * np.cos(q))
        v[:, 3] = style * (.05 + .01 * np.sin(q))
        for k in range(4, 11):
            v[:, k] = .3 + style * (.02 + .003 * np.cos(q + k))
        mask[~s.ties.is_attack] = False
        v[~mask] = 0
        physical, pm = inverse_r_physical(v, mask, np.full(n, tau))
        reasons = np.where(mask, 0, R_REASONS.index('tie_continuation')).astype(np.int64)
        meta = tuple({'score_note_key': note.occurrence_id, 'score_onset_q': float(note.onset),
            'score_onset_group_id': int(s.note_to_onset[i]), 'is_attack': bool(s.ties.is_attack[i]),
            'is_grace': note.grace, 'performed_note_id': f'p{i}', 'alignment_status': 'matched'}
            for i, note in enumerate(xml.notes))
        r = NotePerformanceFeatures(pid, work, split, 'synthetic', s.cache_key, xml.source_sha256, xml.traversal_id,
            tuple(n.occurrence_id for n in xml.notes), v, mask, physical, pm, reasons,
            np.minimum(q.astype(int), len(beats.valid) - 1), np.full(n, tau), np.full(n, tau), beats, meta, {'fixture': pid})
        return StyleRecording(r, xml, recording or pid)

    def profile_set(self):
        records = []
        for w in range(4):
            xml = self.make_score(prefix=f'w{w}')
            records += [self.record(xml, f'w{w}-p{i}', style, work=f'w{w}')
                        for i, style in enumerate((-2., -1., 0., .2, 1., 2.))]
        ref = fit_reference(records)
        return [summarize_window(r.performance, make_windows(r)[0], ref) for r in records]

    def test_pickup_meter_rest_repeat_and_tail(self):
        bodies = [measure(attributes() + note('pickup'), 0, implicit=True)]
        for b in range(1, 10):
            bodies.append(measure((attributes(beats='3') if b == 4 else '') +
                                  note(f'r{b}', duration=12 if b >= 4 else 16, rest=True), b))
        rec = self.record(self.read(score_xml(*bodies)), 'p')
        ws = make_windows(rec)
        self.assertEqual([w['actual_measure_count'] for w in ws], [8, 2])
        self.assertEqual((ws[0]['q_start'], ws[0]['q_end'], ws[0]['N']), (0., 25., 1))
        self.assertTrue(ws[1]['tail'])
        xml = self.read(score_xml(*(measure(attributes() + note(f'n{i}', duration=16), i) for i in range(8))),
                        traversal=list(range(8)) * 2)
        ws = make_windows(self.record(xml, 'repeat'))
        self.assertNotEqual(ws[0]['same_window_key'], ws[1]['same_window_key'])
        self.assertEqual(ws[1]['ordered_measure_occurrence_keys'][0], '0@2')
        short = self.record(self.make_score(2), 'short')
        p = summarize_window(short.performance, make_windows(short)[0], fit_reference([short]))
        self.assertFalse(any(p['g_mask']))
        self.assertFalse(fit_profile_scaler([p], profiles_id='short')['metric_ready'])

    def test_reference_counts_train_duplicates_and_unknown_work(self):
        xml = self.make_score()
        records = [self.record(xml, f'p{i}', i - 1) for i in range(3)]
        self.assertFalse(reference_delta(records[0].performance, fit_reference(records[:2]))[1].any())
        ref = fit_reference([*records, self.record(xml, 'copy', -1, recording='p0')])
        table = next(iter(ref['tables'].values()))
        self.assertEqual(table['notes']['n0-0-0@1']['counts'][0], 3)
        with self.assertRaisesRegex(ValueError, 'train'):
            fit_reference([replace(records[0], performance=replace(records[0].performance, split='test'))])
        other = self.record(self.make_score(prefix='other'), 'new', work='unseen', split='test')
        p = summarize_window(other.performance, make_windows(other)[0], ref)
        self.assertFalse(any(p['g_mask']))
        self.assertTrue(np.isfinite(p['g_raw']).all())
        self.assertIn('reference_insufficient', p['ineligible_reason_codes'])

    def test_quantile_and_tempo_ratio(self):
        np.testing.assert_allclose(weighted_quantile([0, 0, 10], [1, 1, 2], [.1, .5, .9]), [0, 5, 10])
        self.assertEqual(float(weighted_quantile([3], [1], .9)), 3.)
        self.assertEqual(len(G_NAMES), 23)
        self.assertAlmostEqual(np.log2(110) - np.log2(100), np.log2(77) - np.log2(70))
        self.assertLess(np.log2(63) - np.log2(70), 0)

    def test_chord_duplication_and_missing_rows(self):
        profiles = []
        for size in (1, 3):
            xml = self.make_score(size=size)
            records = [self.record(xml, f's{size}-{i}', i - 1) for i in range(3)]
            ref = fit_reference(records)
            profiles.append(summarize_window(records[-1].performance, make_windows(records[-1])[0], ref))
        np.testing.assert_allclose(profiles[0]['g_raw'], profiles[1]['g_raw'])
        self.assertTrue(all(profiles[0]['g_mask']))
        r = records[0].performance
        mask, values, phys, pm, reasons = [a.copy() for a in (r.mask, r.values, r.physical, r.physical_mask, r.reasons)]
        mask[::3, 2], pm[::3, 2] = False, False
        values[::3, 2], phys[::3, 2], reasons[::3, 2] = 0, 0, R_REASONS.index('unmatched')
        rec = replace(records[0], performance=replace(r, mask=mask, values=values, physical=phys, physical_mask=pm, reasons=reasons))
        p = summarize_window(rec.performance, make_windows(rec)[0], ref)
        self.assertAlmostEqual(p['per_channel_coverage'][2], 2 / 3)
        self.assertFalse(p['g_mask'][5])
        self.assertEqual(p['g_raw'][5], 0)

    def test_roughness_no_bridging_missing_edges(self):
        xml = self.make_score()
        records = [self.record(xml, f'p{i}', i - 1) for i in range(3)]
        ref = fit_reference(records)
        r = records[-1].performance
        beats = BeatTimeMap(r.beats.q, np.cumsum(np.r_[0, np.tile([.4, .5], 16)]), provenance=r.beats.provenance)
        values, physical = r.values.copy(), r.physical.copy()
        values[:, 0] = beats.log2_qpm[r.beat_id]
        physical[:, 0] = 2 ** values[:, 0]
        r = replace(r, beats=beats, values=values, physical=physical)
        p = summarize_window(r, make_windows(replace(records[-1], performance=r))[0], ref)
        self.assertEqual(p['roughness_edges'], 31)
        self.assertAlmostEqual(p['g_raw'][2], abs(np.log2(.5 / .4)))
        valid = np.ones(32, bool)
        valid[10] = False
        mask, reasons = r.mask.copy(), r.reasons.copy()
        mask[10, 0], values[10, 0], reasons[10, 0] = False, 0, R_REASONS.index('invalid_beat')
        r = replace(r, beats=replace(beats, valid=valid), mask=mask, values=values, reasons=reasons)
        p = summarize_window(r, make_windows(replace(records[-1], performance=r))[0], ref)
        self.assertEqual(p['roughness_edges'], 29)
        self.assertTrue(p['g_mask'][2])

    def test_scaler_masks_constants_and_block_balance(self):
        ps = self.profile_set()
        scaler = fit_profile_scaler(ps, profiles_id='p')
        self.assertTrue(scaler['metric_ready'])
        self.assertFalse(scaler['trained_feature_mask'][2])
        self.assertEqual(scaler['status'][2], 'constant')
        with self.assertRaisesRegex(ValueError, 'train'):
            fit_profile_scaler([{**ps[0], 'window': {**ps[0]['window'], 'split': 'test'}}], profiles_id='p')
        a = {'g_scaled': [0.] * 23, 'g_scaled_mask': [True] * 23}
        b = {'g_scaled': [2.] * 23, 'g_scaled_mask': [True] * 23}
        self.assertEqual(profile_distance(a, b, tuple(range(23)))[0], 4.)
        self.assertEqual(profile_distance(a, b, tuple(range(9)))[0], 4.)
        sparse = {**ps[0], 'g_mask': ps[0]['g_mask'][:9] + [False] * 14}
        self.assertTrue(all(i < 9 for i in anchor_signature(sparse, scaler)))
        with self.assertRaisesRegex(ValueError, 'signature'):
            profile_distance(a, {**b, 'g_scaled_mask': [True] * 22 + [False]}, tuple(range(23)))

    def test_calibration_triplets_determinism_gap_and_pair_cap(self):
        ps = self.profile_set()
        scaler = fit_profile_scaler(ps, profiles_id='p')
        cal = calibrate_distances(ps, scaler, profiles_id='p')
        self.assertEqual(cal, calibrate_distances(list(reversed(ps)), scaler, profiles_id='p'))
        self.assertEqual(cal['status'], 'valid')
        ts = build_triplets(ps, scaler, cal, profiles_id='p')
        self.assertGreater(len(ts['triplets']), 0)
        self.assertEqual(ts, build_triplets(list(reversed(ps)), scaler, cal, profiles_id='p'))
        lookup = {p['window']['window_id']: scale_profile(p, scaler) for p in ps}
        for t in ts['triplets']:
            a, p, n = (lookup[t[k + '_window_id']] for k in ('anchor', 'positive', 'negative'))
            validate_triplet(a, p, n, tuple(t['J_a']), near=t['tau_near'], far=t['tau_far'], gap=t['delta_g'])
            with self.assertRaisesRegex(ValueError, 'Triplet'):
                validate_triplet(a, a, n, tuple(t['J_a']), near=t['tau_near'], far=t['tau_far'], gap=t['delta_g'])
            with self.assertRaisesRegex(ValueError, 'Triplet'):
                validate_triplet(a, p, n, tuple(t['J_a']), near=t['tau_near'], far=t['tau_far'], gap=t['distance_gap'] + 1)
        cfg = replace(StyleConfig(), max_calibration_pairs=100)
        s = fit_profile_scaler(ps, cfg, profiles_id='p')
        limited = calibrate_distances(ps, s, cfg, profiles_id='p')
        self.assertTrue(all(x['pair_count'] <= 100 for x in limited['signatures'].values()))

    def test_candidates_and_zero_triplet_failure(self):
        ps = self.profile_set()
        scaler = fit_profile_scaler(ps, profiles_id='p')
        a, b = ps[:2]
        sig = anchor_signature(a, scaler)
        self.assertFalse(candidate_allowed(a, {**b, 'window': {**b['window'], 'recording_id': a['window']['recording_id']}}, sig))
        wrong = {**b, 'window': {**a['window'], 'performance_id': 'other', 'recording_id': 'other', 'same_window_key': 'other'}}
        self.assertFalse(candidate_allowed(a, wrong, sig))
        self.assertFalse(candidate_allowed(a, {**b, 'g_mask': [True] * 9 + [False] * 14}, sig))
        rec = self.record(self.make_score(), 'only')
        ref = fit_reference([rec])
        p = summarize_window(rec.performance, make_windows(rec)[0], ref)
        s = fit_profile_scaler([p], profiles_id='p')
        cal = calibrate_distances([p], s, profiles_id='p')
        result = build_triplets([p], s, cal, profiles_id='p')
        self.assertEqual(result['triplets'], [])
        self.assertEqual(result['reason_counts'], {'reference_insufficient': 1})

    def test_artifacts_cli_and_source_preservation(self):
        xml, entries = self.make_score(), []
        for i in range(3):
            rec = self.record(xml, f'p{i}', i - 1)
            path = self.folder / f'r{i}'
            save_note_performance(rec.performance, path)
            entries.append({'performance_cache': str(path), 'xml_path': str(self.folder / 'score-1.musicxml'), 'recording_group': rec.recording_id})
        before = {str(p): p.read_bytes() for p in self.folder.glob('r*/*')}
        ws = make_window_artifact(entries, self.folder)
        ref = fit_reference([self.record(xml, f'p{i}', i - 1) for i in range(3)])
        ref = artifact('reference', StyleConfig(), parents={'windows': ws['id']},
                       **{k: v for k, v in ref.items() if k not in {'id', 'kind', 'schema', 'config', 'config_id', 'parents'}})
        ps = extract_profile_artifact(ws, ref)
        scaler = fit_profile_scaler(ps['profiles'], profiles_id=ps['id'])
        cal = calibrate_distances(ps['profiles'], scaler, profiles_id=ps['id'])
        ts = build_triplets(ps['profiles'], scaler, cal, profiles_id=ps['id'])
        report = style_report(ws, ps, scaler, cal, ts, ref)
        self.assertTrue(report['raw_R_unchanged'])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.folder.glob('r*/*')})
        out = self.folder / 'windows.json'
        save_style_artifact(ws, out)
        self.assertEqual(load_style_artifact(out), ws)
        with self.assertRaises(FileExistsError):
            save_style_artifact(ws, out)
        with self.assertRaisesRegex(ValueError, 'different profile'):
            calibrate_distances(ps['profiles'], scaler, profiles_id='other')
        manifest = self.folder / 'input.json'
        manifest.write_text(json.dumps({'records': entries}))
        result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/prepare_style_triplets.py'),
            'make-windows', '--manifest', str(manifest), '--out', str(self.folder / 'cli.json')],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(load_style_artifact(self.folder / 'cli.json'), ws)
        script = str(Path(__file__).resolve().parents[2] / 'scripts/prepare_style_triplets.py')
        stages = [
            ('reference', ['fit-reference', '--windows', str(out), '--split', 'train']),
            ('profiles', ['extract-profiles', '--windows', str(out), '--reference', str(self.folder / 'reference.json')]),
            ('scaler', ['fit-profile-scaler', '--profiles', str(self.folder / 'profiles.json'), '--split', 'train']),
            ('scaled', ['scale-profiles', '--profiles', str(self.folder / 'profiles.json'), '--scaler', str(self.folder / 'scaler.json')]),
            ('calibration', ['calibrate-distances', '--profiles', str(self.folder / 'profiles.json'), '--scaler', str(self.folder / 'scaler.json'), '--split', 'train']),
            ('triplets', ['build-triplets', '--profiles', str(self.folder / 'profiles.json'), '--scaler', str(self.folder / 'scaler.json'), '--calibration', str(self.folder / 'calibration.json'), '--split', 'train']),
            ('report', ['report', '--windows', str(out), '--reference', str(self.folder / 'reference.json'), '--profiles', str(self.folder / 'profiles.json'), '--scaler', str(self.folder / 'scaler.json'), '--calibration', str(self.folder / 'calibration.json'), '--triplets', str(self.folder / 'triplets.json')]),
        ]
        for name, args in stages:
            result = subprocess.run([sys.executable, script, *args, '--out', str(self.folder / (name + '.json'))], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(load_style_artifact(self.folder / 'report.json')['triplets'], 0)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.folder.glob('r[0-9]/*')})


    def test_degenerate_distances_and_core_only_pedal_missing(self):
        ps = self.profile_set()
        # Fixed scaler from diverse train profiles, followed by equal inputs.
        scaler = fit_profile_scaler(ps, profiles_id='p')
        constant = [{**p, 'g_raw': [0.] * 23} for p in ps]
        cal = calibrate_distances(constant, scaler, profiles_id='p')
        self.assertEqual({v['status'] for v in cal['signatures'].values()}, {'degenerate_metric'})
        self.assertEqual(build_triplets(constant, scaler, cal, profiles_id='p')['triplets'], [])
        core = [{**p, 'g_raw': p['g_raw'][:9] + [0.] * 14, 'g_mask': p['g_mask'][:9] + [False] * 14} for p in ps]
        scaler = fit_profile_scaler(core, profiles_id='core')
        self.assertTrue(scaler['metric_ready'])
        self.assertFalse(any(scaler['trained_feature_mask'][9:]))
        cal = calibrate_distances(core, scaler, profiles_id='core')
        ts = build_triplets(core, scaler, cal, profiles_id='core')
        self.assertGreater(len(ts['triplets']), 0)
        self.assertTrue(all(len(t['active_blocks']) == 4 for t in ts['triplets']))

    def test_global_cutoff_does_not_force_distant_positives(self):
        ps = self.profile_set()
        # Work offsets make every cross-work distance strictly positive.
        ps = [{**p, 'g_raw': [v + int(p['window']['work_id'][1:]) * .001 for v in p['g_raw']]} for p in ps]
        scaler = fit_profile_scaler(ps, profiles_id='p')
        cal = calibrate_distances(ps, scaler, profiles_id='p')
        signatures = {key: {**v, 'global_near': 0.} for key, v in cal['signatures'].items()}
        cal = artifact('distance-calibration', StyleConfig(),
            **{k: v for k, v in cal.items() if k not in {'id', 'schema', 'config', 'config_id', 'kind', 'signatures'}}, signatures=signatures)
        ts = build_triplets(ps, scaler, cal, profiles_id='p')
        self.assertEqual(ts['triplets'], [])
        self.assertIn('positive_missing', ts['reason_counts'])

    def test_tie_grace_population_and_partial_beat_weights(self):
        body = attributes() + note('a', extra='<tie type="start"/>') + note('b', extra='<tie type="stop"/>')
        body += note('grace', grace=True) + note('c') + note('d')
        xml = self.read(score_xml(measure(body), *(measure(note(f'rest{i}', duration=16, rest=True), i) for i in range(1, 8))))
        rec = self.record(xml, 'p')
        w = make_windows(rec)[0]
        self.assertEqual(w['note_weights'], [1., 0., 0., 1., 1.])
        # One long interval crosses the window end: its weight must be clipped.
        edges = np.array([0., 10., 20., 40.])
        beats = BeatTimeMap(edges, edges * .5, provenance=rec.performance.beats.provenance)
        r = rec.performance
        ids = np.array([beats.beat_at(m['score_onset_q']) for m in r.metadata], dtype=np.int64)
        v = r.values.copy()
        v[r.mask[:, 0], 0] = beats.log2_qpm[ids[r.mask[:, 0]]]
        r = replace(r, beats=beats, beat_id=ids, values=v)
        rec = replace(rec, performance=r)
        cfg = replace(StyleConfig(), min_valid_beats=1)
        ref = fit_reference([rec], cfg)
        d, m, c, bd, bm, bc = reference_delta(r, ref)
        p = summarize_window(r, make_windows(rec)[0], ref, cfg,
            deltas=(d, m, c, np.array([0., 0., 10.]), np.ones(3, bool), np.full(3, 3)))
        self.assertEqual(p['per_channel_coverage'][0], 1.)
        self.assertAlmostEqual(p['g_raw'][0], float(weighted_quantile([0, 0, 10], [10, 10, 12], .5)))

    def test_vectorized_candidate_distances_equal_scalar_contract(self):
        from embedding.style_triplets import _CandidateIndex
        ps = self.profile_set()
        scaler = fit_profile_scaler(ps, profiles_id='p')
        scaled = [scale_profile(p, scaler) for p in ps]
        for anchor in scaled[:3]:
            signature = anchor_signature(anchor, scaler)
            index = _CandidateIndex(scaled, signature)
            indices = index.indices(anchor)
            self.assertEqual(index.count(anchor), len(indices))
            context = index.draw_context(anchor)
            buckets = index.buckets(anchor)
            self.assertEqual(list(context[0]), list(buckets))
            for work in context[0]:
                np.testing.assert_array_equal(index.draw_pool(anchor, context, work), buckets[work])
            expected = [p['window']['window_id'] for p in scaled if candidate_allowed(anchor, p, signature)]
            self.assertEqual([index.profiles[i]['window']['window_id'] for i in indices], expected)
            direct = [profile_distance(anchor, index.profiles[i], signature)[0] for i in indices]
            np.testing.assert_array_equal(index.distances(anchor, indices), direct)
        cfg = replace(StyleConfig(), max_calibration_pairs=100)
        scaler = fit_profile_scaler(ps, cfg, profiles_id='p')
        calibration = calibrate_distances(ps, scaler, cfg, profiles_id='p')
        for signature in calibration['signatures'].values():
            keys = [(pair['anchor_window_id'], pair['candidate_window_id']) for pair in signature['pairs']]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(len(keys), min(signature['possible_pair_count'], 100))

    def test_verified_delta_snapshot_is_readonly_and_bound_to_its_record(self):
        xml = self.make_score()
        records = [self.record(xml, f'p{i}', i - 1) for i in range(3)]
        reference = fit_reference(records)
        r = records[0].performance
        delta = reference_delta(r, reference)
        self.assertEqual(len(delta), 6)
        with self.assertRaises(ValueError):
            delta[0][0, 1] = 9
        first = summarize_window(r, make_windows(records[0])[0], reference, deltas=delta)
        self.assertEqual(first, summarize_window(r, make_windows(records[0])[0], reference))
        other = replace(r, record_key='different')
        window = make_windows(replace(records[0], performance=other))[0]
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            summarize_window(other, window, reference, deltas=delta)

    def test_compact_scaler_snapshot_preserves_bulk_values_and_masks(self):
        from features.style_profile import FrozenProfileScaler, scale_profiles
        profiles = self.profile_set()
        scaler = fit_profile_scaler(profiles, profiles_id='p')
        snapshot = FrozenProfileScaler.from_artifact(scaler)
        expected = [scale_profile(p, scaler) for p in profiles[:3]]
        self.assertEqual(scale_profiles(profiles[:3], snapshot), expected)
        self.assertEqual(scale_profiles(profiles[:3], scaler), expected)
        with self.assertRaises(ValueError):
            snapshot.scale[0] = 2
        corrupt = {**scaler, 'center': [999.] * 23}
        with self.assertRaisesRegex(ValueError, 'integrity'):
            FrozenProfileScaler.from_artifact(corrupt)
