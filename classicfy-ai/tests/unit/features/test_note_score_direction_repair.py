"""Opt-in direction repair preserves exact timing and repeat occurrence counts."""

from dataclasses import replace
from fractions import Fraction

import numpy as np

from features.note_score_context import (
    CATEGORICAL_FIELDS, VOCABULARIES, extract_note_score_context,
    normalize_direction_offsets,
)
from unit.preprocessing.musicxml_fixtures import (
    XMLFixture, attributes, direction, measure, note, score_xml,
)


class DirectionOffsetRecoveryTest(XMLFixture):
    def score(self, *, offset=1, traversal=None):
        xml = score_xml(
            measure(attributes(beats='1', divisions=1000)
                    + direction('<wedge type="crescendo" number="1"/>')
                    + note('a', duration=1000)
                    + direction('<wedge type="stop" number="1"/>', offset=offset)),
            measure(note('b', duration=500) + note('c', duration=500), 2))
        return self.read(xml, traversal=traversal)

    def gradual(self, features):
        column = CATEGORICAL_FIELDS.index('dynamic_gradual')
        return [VOCABULARIES['dynamic_gradual'][x] for x in features.categorical[:, column]]

    def test_small_positive_offset_keeps_exact_time_and_note_identity(self):
        original = self.score()
        with self.assertRaisesRegex(ValueError, 'Cross-measure'):
            extract_note_score_context(original)
        repaired, audit = normalize_direction_offsets(original)
        self.assertEqual(repaired.notes, original.notes)
        self.assertEqual(repaired.visits, original.visits)
        self.assertEqual(repaired.source_sha256, original.source_sha256)
        self.assertEqual(repaired.traversal_id, original.traversal_id)
        stop = repaired.directions[-1]
        self.assertEqual((stop.measure, stop.position), (1, Fraction(1, 1000)))
        self.assertEqual(original.directions[-1].position,
                         original.measure_durations[0] + stop.position)
        self.assertEqual(len(audit), 1)
        self.assertFalse(audit[0]['absolute_time_changed'])
        features = extract_note_score_context(repaired)
        # The next measure's first note still precedes the stop by 1/1000 q.
        # Clipping to the boundary would incorrectly remove this crescendo.
        self.assertEqual(self.gradual(features), ['crescendo', 'crescendo', 'NONE'])
        np.testing.assert_array_equal(features.onset_q, [0., 1., 1.5])
        self.assertNotEqual(features.cache_key, replace(features, diagnostics=()).cache_key)

    def test_negative_offset_moves_to_previous_measure_without_clipping(self):
        xml = score_xml(
            measure(attributes(beats='1', divisions=1000) + note('a', duration=1000)),
            measure(direction('<dynamics><p/></dynamics>', offset=-1)
                    + note('b', duration=1000), 2))
        score = self.read(xml)
        fixed, audit = normalize_direction_offsets(score)
        self.assertEqual((fixed.directions[0].measure, fixed.directions[0].position),
                         (0, Fraction(999, 1000)))
        self.assertEqual(len(audit), 1)
        features = extract_note_score_context(fixed)
        column = CATEGORICAL_FIELDS.index('dynamic_base')
        self.assertEqual(VOCABULARIES['dynamic_base'][features.categorical[1, column]], 'p')

    def test_exact_boundary_and_unaffected_scores_keep_identity_and_cache(self):
        score = self.score(offset=0)
        expected = extract_note_score_context(score).cache_key
        fixed, audit = normalize_direction_offsets(score)
        self.assertIs(fixed, score)
        self.assertEqual(audit, ())
        self.assertEqual(extract_note_score_context(fixed).cache_key, expected)
        changed, _ = normalize_direction_offsets(self.score())
        again, second_audit = normalize_direction_offsets(changed)
        self.assertIs(again, changed)
        self.assertEqual(second_audit, ())

    def test_repeated_boundary_requires_one_to_one_adjacent_occurrences(self):
        valid = self.score(traversal=(0, 1, 0, 1))
        fixed, audit = normalize_direction_offsets(valid)
        self.assertEqual(audit[0]['occurrences'], 2)
        self.assertEqual(self.gradual(extract_note_score_context(fixed)),
                         ['crescendo', 'crescendo', 'NONE', 'crescendo', 'crescendo', 'NONE'])
        for traversal in ((0, 1, 1), (0, 0, 1), (0, 1, 0)):
            with self.subTest(traversal=traversal), self.assertRaisesRegex(ValueError, 'ambiguous'):
                normalize_direction_offsets(self.score(traversal=traversal))

    def test_large_or_terminal_offsets_are_not_silently_repaired(self):
        with self.assertRaisesRegex(ValueError, 'recovery limit'):
            normalize_direction_offsets(self.score(offset=40))
        score = self.score()
        terminal = replace(score, directions=(replace(score.directions[-1], measure=1),))
        with self.assertRaisesRegex(ValueError, 'leaves the score'):
            normalize_direction_offsets(terminal)
        with self.assertRaisesRegex(ValueError, 'at most'):
            normalize_direction_offsets(score, max_overflow_q=Fraction(1, 8))
