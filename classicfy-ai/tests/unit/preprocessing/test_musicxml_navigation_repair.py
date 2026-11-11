"""Route semantics and refusal cases for opt-in navigation recovery."""

import unittest
import xml.etree.ElementTree as ET

from preprocessing.musicxml_navigation_repair import (
    prepare_navigation_recovery, UnsupportedNavigation,
)
from preprocessing.musicxml_score import read_musicxml
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, direction, measure, note, score_xml


def tree(contents):
    return ET.fromstring(score_xml(*(measure((attributes(beats='1') if i == 0 else '')
                                         + note(f'n{i}') + content, i + 1)
                                   for i, content in enumerate(contents))))


FWD = '<barline location="left"><repeat direction="forward"/></barline>'
BACK = '<barline location="right"><repeat direction="backward"/></barline>'


class NavigationRecoveryTest(XMLFixture):
    def check_parse(self, root, recovery):
        """Each proposed route must retain exact notes and be accepted by the parser."""
        self.assertEqual([ET.tostring(n) for n in root.findall('.//note')],
                         [ET.tostring(n) for n in recovery.cleaned_root.findall('.//note')])
        path = self.folder / 'derived.musicxml'
        ET.ElementTree(recovery.cleaned_root).write(path, encoding='utf-8')
        for candidate in recovery.candidates:
            result = read_musicxml(path, traversal=candidate.traversal)
            self.assertEqual(tuple(v.measure for v in result.visits), candidate.traversal)

    def test_noop_deduplicates_and_preserves_source_tree(self):
        root = tree(['', ''])
        original = ET.tostring(root)
        recovery = prepare_navigation_recovery(root)
        self.assertEqual(ET.tostring(root), original)
        self.assertEqual(len(recovery.candidates), 1)
        self.assertEqual(recovery.candidates[0].traversal, (0, 1))
        self.assertEqual(recovery.audit['selection'], 'original-performance-alignment-required')
        self.check_parse(root, recovery)

    def test_expression_fine_is_not_a_jump(self):
        root = tree(['', direction('<words>accelerando sin al fine</words>')])
        recovery = prepare_navigation_recovery(root)
        self.assertEqual([e['kind'] for e in recovery.audit['events']], ['expression_not_jump'])
        self.assertEqual(len(recovery.candidates), 1)
        self.assertEqual(recovery.cleaned_root.find('.//words').text, 'accelerando sin al end')
        self.check_parse(root, recovery)

    def test_pedal_al_fine_requires_actual_pedal_context(self):
        content = direction('<words>al Fine</words>')
        with self.assertRaisesRegex(UnsupportedNavigation, 'Unclassified'):
            prepare_navigation_recovery(tree(['', content]))
        content += direction('<pedal type="start"/>') + direction('<words font-family="MScore Text">\ue650</words>')
        root = tree(['', content])
        recovery = prepare_navigation_recovery(root)
        self.assertEqual(len(recovery.candidates), 1)
        self.check_parse(root, recovery)

    def test_dc_al_fine_stops_only_on_return_and_keeps_written_candidate(self):
        root = tree(['', direction('<words>Fine</words>'), '', direction('<words>D.C. al Fine</words>')])
        recovery = prepare_navigation_recovery(root)
        self.assertEqual({c.traversal for c in recovery.candidates}, {(0, 1, 2, 3), (0, 1, 2, 3, 0, 1)})
        self.check_parse(root, recovery)

    def test_dc_coda_uses_matching_trigger_and_destination(self):
        root = tree(['', '<direction><direction-type><words>To Coda</words></direction-type><sound tocoda="end"/></direction>',
                     '', '<direction><direction-type><words>D.C. al Coda</words></direction-type><sound dacapo="yes"/></direction>',
                     '<direction><direction-type><coda/></direction-type><sound coda="end"/></direction>', ''])
        recovery = prepare_navigation_recovery(root)
        self.assertIn((0, 1, 2, 3, 0, 1, 4, 5), {c.traversal for c in recovery.candidates})
        self.check_parse(root, recovery)

    def test_ds_named_target(self):
        root = tree(['', '<direction><direction-type><segno/></direction-type><sound segno="s1"/></direction>',
                     direction('<words>Fine</words>'),
                     '<direction><direction-type><words>D.S. al Fine</words></direction-type><sound dalsegno="s1"/></direction>'])
        recovery = prepare_navigation_recovery(root)
        self.assertIn((0, 1, 2, 3, 1, 2), {c.traversal for c in recovery.candidates})
        self.check_parse(root, recovery)

    def test_unmapped_or_multiple_jumps_are_not_guessed(self):
        examples = [
            ['', direction('<words>D.S.</words>')],
            ['', direction('<words>D.C. al Coda</words>')],
            ['', direction('<words>D.C. al Fine</words>')],
            ['', direction('<words>D.C.</words>'), direction('<words>D.C.</words>')],
            [direction('<words>To Coda</words>'), ''],
            ['', direction('<words>Da capo il Scherzino,</words>')
             + direction('<words>senza Trio, e poi Coda</words>'), direction('<words>Coda</words>')],
        ]
        for contents in examples:
            with self.subTest(contents=contents), self.assertRaises(UnsupportedNavigation):
                prepare_navigation_recovery(tree(contents))

    def test_bare_jump_before_tail_does_not_invent_rejoin(self):
        root = tree(['', direction('<words>D.C.</words>'), ''])
        recovery = prepare_navigation_recovery(root)
        self.assertEqual([c.traversal for c in recovery.candidates], [(0, 1, 2)])
        self.assertIn('unsupported_jump_return', recovery.audit['repeat_structure'])

    def test_named_menuetto_return_ends_before_trio(self):
        root = tree([direction('<words>Menuetto</words>'), '',
                     direction('<words>Trio</words>'), direction('<words>Menuetto da capo</words>')])
        recovery = prepare_navigation_recovery(root)
        self.assertIn((0, 1, 2, 3, 0, 1), {c.traversal for c in recovery.candidates})
        self.assertNotIn((0, 1, 2, 3) * 2, {c.traversal for c in recovery.candidates})
        self.check_parse(root, recovery)

    def test_menuetto_roman_section_and_observed_menutto_spelling(self):
        root = tree([direction('<words>Menuetto I</words>'), '',
                     direction('<words>Menuetto II</words>'), direction('<words>Menutto I da capo</words>')])
        recovery = prepare_navigation_recovery(root)
        self.assertIn((0, 1, 2, 3, 0, 1), {c.traversal for c in recovery.candidates})

    def test_unclosed_repeat_hypotheses_are_explicitly_audited(self):
        root = tree(['', FWD, ''])
        recovery = prepare_navigation_recovery(root)
        self.assertEqual(recovery.audit['repeat_structure']['unclosed_forward_measures'], [1])
        self.assertEqual({c.traversal for c in recovery.candidates}, {(0, 1, 2), (0, 1, 2, 1, 2)})
        inferred = next(c for c in recovery.candidates if c.name == 'unclosed_to_score_end')
        self.assertIn('close at final measure', inferred.assumptions[0])
        self.check_parse(root, recovery)

    def test_nested_repeats_reset_inner_repeat_on_every_outer_pass(self):
        root = tree([FWD, FWD, BACK, BACK])
        recovery = prepare_navigation_recovery(root)
        self.assertIn((0, 1, 2, 1, 2, 3) * 2, {c.traversal for c in recovery.candidates})
        self.check_parse(root, recovery)

    def test_numbered_endings(self):
        root = tree([FWD,
                     '<barline location="left"><ending number="1" type="start"/></barline>'
                     '<barline location="right"><ending number="1" type="stop"/><repeat direction="backward"/></barline>',
                     '<barline location="left"><ending number="2" type="start"/></barline>'
                     '<barline location="right"><ending number="2" type="stop"/></barline>'])
        recovery = prepare_navigation_recovery(root)
        self.assertIn((0, 1, 0, 2), {c.traversal for c in recovery.candidates})
        self.check_parse(root, recovery)

    def test_senza_replica_prevents_repeated_return_candidate(self):
        root = tree([FWD, BACK + direction('<words>Fine</words>'), '',
                     direction('<words>Menuetto da capo senza replica</words>')])
        recovery = prepare_navigation_recovery(root)
        self.assertFalse(any('return_with_repeats' in c.name for c in recovery.candidates))
        self.assertIn((0, 1, 0, 1, 2, 3, 0, 1), {c.traversal for c in recovery.candidates})
        self.check_parse(root, recovery)

    def test_bound_holds_oversized_repeat_and_reports_it(self):
        root = tree([FWD, BACK])
        recovery = prepare_navigation_recovery(root, max_visits=2)
        self.assertEqual([c.traversal for c in recovery.candidates], [(0, 1)])
        self.assertTrue(recovery.audit['repeat_structure']['unsupported_repeat_candidates'])


if __name__ == '__main__':
    unittest.main()
