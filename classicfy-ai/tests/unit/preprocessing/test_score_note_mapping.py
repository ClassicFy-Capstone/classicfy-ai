from dataclasses import replace

import numpy as np

from preprocessing.note_alignment import NoteAlignment, PerformedNote, ScoreNote
from preprocessing.score_note_mapping import build_tie_mapping, link_note_alignment
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, measure, note, score_xml


def alignment_for(score, identifiers=None):
    matches = []
    for i, n in enumerate(score.notes):
        if 'stop' in n.ties:
            continue
        identifier = n.occurrence_id if identifiers is None else identifiers[i]
        matches.append((ScoreNote(identifier, n.pitch, float(n.onset), float(n.offset), ()),
                        PerformedNote(f'perf-{i}', n.pitch, float(n.onset) + .1, float(n.offset) + .2, 64)))
    return NoteAlignment(tuple(matches), (), ())


class ScoreNoteMappingTest(XMLFixture):
    def test_tie_whole_duration_single_attack_and_end_mask(self):
        score = self.read()
        ties = build_tie_mapping(score)
        np.testing.assert_array_equal(ties.owner_indices[:3], [0, 1, 0])
        np.testing.assert_array_equal(ties.chain_duration_q[:3], [3, 1, 3])
        np.testing.assert_array_equal(ties.is_attack[:3], [True, True, False])
        self.assertTrue(ties.chain_known.all())
        links = link_note_alignment(score, ties, alignment_for(score))
        self.assertTrue(links.attack_mask[0])
        self.assertFalse(links.attack_mask[2])
        self.assertFalse(links.duration_mask[2])
        self.assertEqual(links.performed_ids[0], links.performed_ids[2])
        self.assertEqual(links.status[2], 'tie_continuation')

    def test_unclosed_or_gapped_tie_disables_chain_duration(self):
        for body in (note('a', extra='<tie type="start"/>'),
                     note('a', extra='<tie type="start"/>') + note('r', rest=True) + note('b', extra='<tie type="stop"/>')):
            with self.subTest(body=body):
                score = self.read(score_xml(measure(attributes() + body)))
                ties = build_tie_mapping(score)
                self.assertFalse(ties.chain_known[0])
                links = link_note_alignment(score, ties, alignment_for(score))
                self.assertTrue(links.attack_mask[0])
                self.assertFalse(links.duration_mask[0])

    def test_continuation_match_does_not_create_missing_owner_target(self):
        score = self.read()
        n = score.notes[2]
        only_end = NoteAlignment(((ScoreNote(n.occurrence_id, 60, 1, 3, ()), PerformedNote('p', 60, 1.1, 3.2, 60)),), (), ())
        links = link_note_alignment(score, build_tie_mapping(score), only_end)
        self.assertFalse(links.attack_mask.any())
        self.assertEqual(links.status[0], 'unmatched_attack_owner')

    def test_explicit_aliases_and_ambiguous_event_ownership(self):
        score = self.read()
        ties = build_tie_mapping(score)
        original = alignment_for(score)
        renamed = NoteAlignment(tuple((replace(s, note_id='dataset-' + s.note_id), p) for s, p in original.matches), (), ())
        unlinked = link_note_alignment(score, ties, renamed)
        self.assertFalse(unlinked.attack_mask.any())
        aliases = {'dataset-' + n.occurrence_id: n.occurrence_id for n in score.notes}
        linked = link_note_alignment(score, ties, renamed, score_id_to_occurrence=aliases)
        self.assertTrue(linked.attack_mask[0])
        first = original.matches[0]
        duplicate = (replace(first[0], note_id=score.notes[-1].occurrence_id, pitch=score.notes[-1].pitch),
                     replace(first[1], pitch=score.notes[-1].pitch))
        with self.assertRaisesRegex(ValueError, 'Conflicting performed note ID'):
            link_note_alignment(score, ties, replace(original, matches=original.matches + (duplicate,)))

    def test_one_to_many_targets_remain_ambiguous(self):
        score = self.read()
        source = alignment_for(score)
        first = source.matches[0]
        extra = (first[0], replace(first[1], note_id='ornament-extra'))
        links = link_note_alignment(score, build_tie_mapping(score), replace(source, matches=source.matches + (extra,)))
        self.assertEqual(links.status[0], 'ambiguous')
        self.assertFalse(links.attack_mask[0])

    def test_performance_values_do_not_enter_score_features(self):
        from features.note_score_context import extract_note_score_context
        score = self.read()
        before = extract_note_score_context(score)
        original = alignment_for(score)
        changed = replace(original, matches=tuple((s, replace(p, velocity=100, onset=p.onset + 2, offset=p.offset + 3)) for s, p in original.matches))
        link_note_alignment(score, before.ties, changed)
        after = extract_note_score_context(score)
        self.assertEqual(before.cache_key, after.cache_key)
        np.testing.assert_array_equal(before.numeric, after.numeric)
        np.testing.assert_array_equal(before.field_known, after.field_known)
