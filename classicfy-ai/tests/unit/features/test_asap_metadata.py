import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from audit_asap_metadata import (classify_work, work_family, join_cohort,
                                 candidate_availability, pilot_options, validate_vectors)


class ASAPMetadataTests(unittest.TestCase):
    def row(self, key, composer, title, work_type, family, split='train'):
        return dict(key=key, composer=composer, title=title, work_type=work_type,
                    work_id=f'{composer}::{title}', family_id=family, split=split)

    def test_explicit_singular_plural_and_collection_labels(self):
        cases = [('Piano_Sonatas_31-1', 'Piano_Sonatas', 'sonata'),
                 ('Keyboard_Sonatas_50-1', 'Keyboard_Sonatas', 'sonata'),
                 ('Sonata_3_2nd', 'Sonata_3', 'sonata'),
                 ('Sonatas_5', 'Sonatas', 'sonata'),
                 ('Etudes_op_10_4', 'Etudes_op_10', 'etude'),
                 ('Gran_Etudes_de_Paganini_2_La_campanella', 'Gran_Etudes_de_Paganini', 'etude'),
                 ('Transcendental_Etudes_10', 'Transcendental_Etudes', 'etude'),
                 ('Prelude_bwv_854', 'Prelude', 'prelude'),
                 ('Fugue_bwv_854', 'Fugue', 'fugue'),
                 ('Ballade_2', 'Ballade_2', 'ballade'),
                 ('Ballades_4', 'Ballades', 'ballade'),
                 ('Impromptu_op.90_D.899_3', 'Impromptu_op.90_D.899', 'impromptu'),
                 ('Scherzos_31', 'Scherzos', 'scherzo'),
                 ('Barcarolle', 'Barcarolle', 'barcarolle')]
        for title, subgroup, expected in cases:
            with self.subTest(title=title):
                result = classify_work(title, f'Composer/{subgroup}/1')
                self.assertEqual(result['work_type'], expected)
                self.assertEqual(result['classification_status'], 'title_and_folder_agree')

    def test_unknown_names_and_conflicts_are_not_guessed(self):
        for title, folder in [('Islamey', 'Balakirev/Islamey'),
                              ('Miroirs', 'Ravel/Miroirs'),
                              ('Sonata-like', 'Someone/Unsorted'),
                              ('Not_A_Prelude', 'Someone/Misc')]:
            self.assertEqual(classify_work(title, folder)['work_type'], 'unclassified')
        conflict = classify_work('Etudes_op_10_4', 'Chopin/Piano_Sonatas/4')
        self.assertEqual(conflict['work_type'], 'unclassified')
        self.assertEqual(conflict['classification_status'], 'conflicting_title_folder')

    def test_single_source_evidence_is_retained(self):
        result = classify_work('Unnamed_1', 'Someone/Prelude/1')
        self.assertEqual(result['work_type'], 'prelude')
        self.assertEqual(result['classification_status'], 'folder_only')
        self.assertEqual(result['title_type'], '')
        self.assertEqual(classify_work('Prelude_1', 'Someone/Unsorted/1')['classification_status'], 'title_only')

    def test_family_exclusion_merges_movements_not_opus_numbers(self):
        self.assertEqual(work_family('Beethoven', 'Piano_Sonatas_31-1'),
                         work_family('Beethoven', 'Piano_Sonatas_31-2'))
        self.assertEqual(work_family('Chopin', 'Sonata_3_2nd'), work_family('Chopin', 'Sonata_3_4th'))
        self.assertEqual(work_family('Bach', 'Prelude_bwv_848'), work_family('Bach', 'Fugue_bwv_848'))
        self.assertNotEqual(work_family('Chopin', 'Etudes_op_10_2'), work_family('Chopin', 'Etudes_op_10_4'))
        self.assertNotEqual(work_family('Bach', 'Prelude_bwv_848'), work_family('Bach', 'Prelude_bwv_854'))

    def test_exact_join_split_and_score_protection(self):
        c = dict(key='C/Etudes/1/X.mid', piece='C/Etudes/1/midi_score.mid', grid='g', windows='2', split='train')
        m = dict(midi_performance=c['key'], midi_score=c['piece'], composer='C',
                 title='Etudes_1', folder='C/Etudes/1')
        rows = join_cohort([c], [m], {c['piece']: 'train'})
        self.assertEqual(rows[0]['work_type'], 'etude')
        self.assertEqual(rows[0]['windows'], 2)
        for meta, split in [([], {c['piece']: 'train'}),
                            ([m, m], {c['piece']: 'train'}),
                            ([m | {'midi_score': 'other'}], {c['piece']: 'train'}),
                            ([m], {c['piece']: 'test'})]:
            with self.assertRaises(ValueError):
                join_cohort([c], meta, split)

    def test_candidate_counts_exclude_query_family_and_control_composer(self):
        rows = [self.row('a', 'C', 'Sonata_1-1', 'sonata', 'S1'),
                self.row('a2', 'C', 'Sonata_1-2', 'sonata', 'S1'),
                self.row('c', 'C', 'Sonata_2-1', 'sonata', 'S2'),
                self.row('b', 'C', 'Etude_1', 'etude', 'E1'),
                self.row('other_composer', 'D', 'Etude_1', 'etude', 'DE1')]
        queries, directed = candidate_availability(rows)
        a = next(r for r in queries if r['key'] == 'a' and r['scope'] == 'train')
        self.assertEqual(a['same_type_candidates'], 1)
        self.assertEqual(a['different_type_candidates'], 1)
        self.assertEqual(a['distinct_family_B_C_pairs'], 1)
        self.assertTrue(a['metadata_supported'])
        self.assertEqual(next(r for r in directed if r['key'] == 'a' and r['scope'] == 'train')['other_type'], 'etude')

    def test_B_and_C_cannot_share_a_conservative_family(self):
        rows = [self.row('a', 'Bach', 'Prelude_1', 'prelude', 'BWV1'),
                self.row('c', 'Bach', 'Prelude_2', 'prelude', 'BWV2'),
                self.row('b', 'Bach', 'Fugue_2', 'fugue', 'BWV2')]
        queries, _ = candidate_availability(rows)
        a = next(r for r in queries if r['scope'] == 'train' and r['key'] == 'a')
        self.assertFalse(a['metadata_supported'])
        self.assertEqual(a['reason'], 'no_distinct_B_C_families')

    def test_unknown_type_and_held_out_candidates_do_not_enable_train_queries(self):
        rows = [self.row('a', 'C', 'Etude_1', 'etude', 'E1'),
                self.row('c', 'C', 'Etude_2', 'etude', 'E2', 'test'),
                self.row('b', 'C', 'Sonata_1', 'sonata', 'S1', 'validation'),
                self.row('unknown', 'C', 'Islamey', 'unclassified', 'U1')]
        queries, _ = candidate_availability(rows)
        a = next(r for r in queries if r['key'] == 'a' and r['scope'] == 'train')
        self.assertFalse(a['metadata_supported'])
        self.assertEqual(a['different_type_candidates'], 0)
        self.assertEqual(next(r for r in queries if r['key'] == 'unknown')['reason'], 'unclassified_query')
        self.assertTrue(next(r for r in queries if r['key'] == 'a' and r['scope'] == 'all')['metadata_supported'])

    def test_pilot_requires_two_families_each_and_both_directions(self):
        rows = [self.row(f'e{i}', 'C', f'Etude_{i}', 'etude', f'E{i}') for i in (1, 2)]
        rows += [self.row(f's{i}', 'C', f'Sonata_{i}', 'sonata', f'S{i}') for i in (1, 2)]
        _, directed = candidate_availability(rows)
        result = pilot_options(rows, directed)[0]
        self.assertTrue(result['bidirectional_pilot'])
        self.assertEqual(result['total_supported_queries'], 4)
        shortened = rows[:-1]
        _, directed = candidate_availability(shortened)
        self.assertFalse(pilot_options(shortened, directed)[0]['bidirectional_pilot'])

    def test_vector_and_identity_exports_are_validated(self):
        vector = np.ones((1, 128))
        identity = dict(key='k', piece='p', grid='g', windows=1, split='train')
        row = identity | dict(windows='1', **{f'z{i}': '1' for i in range(128)})
        self.assertEqual(validate_vectors([row], vector, np.array(['k']), [identity]), 0)
        for values, keys, manifest in [(vector, np.array(['other']), [identity]),
                                       (vector * 2, np.array(['k']), [identity]),
                                       (vector, np.array(['k']), [identity | {'split': 'test'}])]:
            with self.assertRaises(ValueError):
                validate_vectors([row], values, keys, manifest)


if __name__ == '__main__':
    unittest.main()
