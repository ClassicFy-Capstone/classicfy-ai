from dataclasses import replace

import numpy as np

from features.note_score_context import (
    CATEGORICAL_FIELDS as C, FLAG_FIELDS as F, NUMERIC_FIELDS as N, NONE, UNK,
    VOCABULARIES, encode_note_numeric, extract_note_score_context, fit_note_score_scale,
)
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, direction, measure, note, score_xml


class NoteScoreContextTest(XMLFixture):
    def features(self, xml=None):
        return extract_note_score_context(self.read(xml))

    def category(self, features, row, name):
        return VOCABULARIES[name][features.categorical[row, C.index(name)]]

    def test_hand_computed_note_values_and_distances(self):
        f = self.features()
        np.testing.assert_array_equal(f.numeric[:, N.index('midi_pitch')], [60, 64, 60, 62, 64, 66, 67, 69])
        np.testing.assert_array_equal(f.numeric[:, N.index('duration_q')], [1, 1, 2, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(f.note_to_onset, [0, 0, 1, 2, 2, 2, 3, 4])
        np.testing.assert_array_equal(f.onset_q, [0, 1, 4, 5, 6])
        self.assertEqual(f.numeric[2, N.index('following_rest_q')], 1)
        self.assertEqual(f.numeric[0, N.index('following_rest_q')], 0)
        self.assertEqual(f.numeric[5, N.index('tempo_base_distance_q')], 4)
        self.assertEqual(f.numeric[5, N.index('quarter_bpm')], 120)
        self.assertEqual(self.category(f, 5, 'tempo_base'), 'allegro')
        self.assertEqual(self.category(f, 0, 'meter_numerator'), '4')
        self.assertEqual(self.category(f, 5, 'meter_numerator'), '3')

    def test_none_unknown_and_explicit_mode_are_distinct(self):
        f = self.features()
        offset = len(N)
        self.assertEqual(f.categorical[0, C.index('mode')], UNK)
        self.assertFalse(f.field_known[0, offset + C.index('mode')])
        self.assertEqual(f.categorical[0, C.index('tempo_relative')], NONE)
        self.assertTrue(f.field_known[0, offset + C.index('tempo_relative')])
        self.assertEqual(self.category(f, 0, 'key_fifths'), '0')
        self.assertEqual(self.category(f, 5, 'mode'), 'minor')
        self.assertEqual(f.flags[1, F.index('staccato')], 0)
        self.assertTrue(f.field_known[1, len(N) + len(C) + F.index('staccato')])

    def test_dynamic_base_wedge_and_local_emphasis_are_independent(self):
        f = self.features()
        self.assertEqual(self.category(f, 5, 'dynamic_base'), 'p')
        self.assertEqual(self.category(f, 5, 'dynamic_gradual'), 'crescendo')
        self.assertEqual(self.category(f, 6, 'dynamic_local'), 'sfz')
        self.assertEqual(self.category(f, 6, 'dynamic_base'), 'p')
        self.assertEqual(self.category(f, 7, 'dynamic_local'), 'NONE')
        self.assertEqual(self.category(f, 7, 'dynamic_gradual'), 'NONE')
        self.assertEqual(f.numeric[6, N.index('dynamic_gradual_distance_q')], 1)

    def test_grace_order_anchor_and_beam_states(self):
        f = self.features()
        np.testing.assert_array_equal(f.numeric[3:5, N.index('grace_order')], [1, 2])
        np.testing.assert_array_equal(f.numeric[3:5, N.index('grace_group_size')], [2, 2])
        self.assertEqual(f.metadata[3]['grace_anchor_occurrence_ids'], ('f1@1',))
        self.assertEqual(self.category(f, 3, 'grace_anchor'), 'before')
        np.testing.assert_array_equal(f.flags[3:5, F.index('beam_start')], [1, 0])
        np.testing.assert_array_equal(f.flags[3:5, F.index('beam_end')], [0, 1])
        self.assertTrue(f.field_known[3:5, len(N) + len(C) + F.index('beam_start')].all())

    def test_slur_chord_and_note_notation_flags(self):
        f = self.features()
        np.testing.assert_array_equal(f.flags[:3, F.index('slur_start')], [1, 1, 0])
        np.testing.assert_array_equal(f.flags[:3, F.index('slur_end')], [0, 0, 1])
        self.assertEqual(f.flags[0, F.index('staccato')], 1)
        self.assertEqual(f.flags[1, F.index('tenuto')], 1)

    def test_following_rest_uses_same_voice_even_when_other_voice_plays(self):
        xml = score_xml(measure(attributes() + note('a') + note('r', duration=8, rest=True) + note('b')
                        + '<backup><duration>16</duration></backup>'
                        + note('other', 'G', duration=16, voice='2', staff='2')))
        f = self.features(xml)
        row = next(i for i, m in enumerate(f.metadata) if m['source_id'] == 'a')
        self.assertEqual(f.numeric[row, N.index('following_rest_q')], 2)

    def test_staff_direction_does_not_leak_to_other_staff(self):
        xml = score_xml(measure(attributes() + direction('<dynamics><p/></dynamics>')
                        + direction('<dynamics><f/></dynamics>', staff='1')
                        + note('a', duration=16) + '<backup><duration>16</duration></backup>'
                        + note('b', duration=16, voice='2', staff='2')))
        f = self.features(xml)
        self.assertEqual([self.category(f, i, 'dynamic_base') for i in range(2)], ['f', 'p'])

    def test_direction_offset_and_complementary_tempo_order(self):
        metronome = direction('<metronome><beat-unit>half</beat-unit><beat-unit-dot/><per-minute>40</per-minute></metronome>')
        for header in (direction('<words>Allegro</words>') + metronome,
                       metronome + direction('<words>Allegro</words>')):
            xml = score_xml(measure(attributes() + header + direction('<dynamics><f/></dynamics>', offset=4)
                            + note('a') + note('b', duration=12)))
            with self.subTest(header=header):
                f = self.features(xml)
                self.assertEqual(self.category(f, 0, 'tempo_base'), 'allegro')
                self.assertEqual(f.numeric[0, N.index('quarter_bpm')], 120)
                self.assertEqual(self.category(f, 0, 'dynamic_base'), 'UNK')
                self.assertEqual(self.category(f, 1, 'dynamic_base'), 'f')

    def test_tempo_relative_gradual_and_return(self):
        xml = score_xml(measure(attributes() + direction('<words>Andante</words>') + note('a')
                        + direction('<words>più mosso</words>') + note('b')
                        + direction('<words>rit.</words>') + note('c')
                        + direction('<words>a tempo</words>') + note('d')))
        f = self.features(xml)
        self.assertEqual(self.category(f, 1, 'tempo_relative'), 'piu_mosso')
        self.assertEqual(self.category(f, 2, 'tempo_gradual'), 'ritardando')
        self.assertEqual(self.category(f, 3, 'tempo_base'), 'andante')
        self.assertEqual(self.category(f, 3, 'tempo_relative'), 'NONE')
        self.assertEqual(self.category(f, 3, 'tempo_gradual'), 'NONE')

    def test_simultaneous_base_and_gradual_tempo_are_order_independent(self):
        for header in (direction('<words>rit.</words>') + direction('<words>Andante</words>'),
                       direction('<words>Andante</words>') + direction('<words>rit.</words>'),
                       direction('<words>Andante rit.</words>')):
            with self.subTest(header=header):
                f = self.features(score_xml(measure(attributes() + header + note('a', duration=16))))
                self.assertEqual(self.category(f, 0, 'tempo_base'), 'andante')
                self.assertEqual(self.category(f, 0, 'tempo_gradual'), 'ritardando')

    def test_new_base_clears_old_change_but_retains_simultaneous_change(self):
        xml = score_xml(measure(attributes() + direction('<words>Andante rit.</words>') + note('a')
                        + direction('<words>Moderato</words>') + note('b')
                        + direction('<words>rit.</words>') + direction('<words>Allegro</words>')
                        + note('c', duration=8)))
        f = self.features(xml)
        self.assertEqual([self.category(f, i, 'tempo_gradual') for i in range(3)], ['ritardando', 'NONE', 'ritardando'])

    def test_repeat_restores_entry_state_and_distances(self):
        xml = score_xml(
            measure(attributes(beats='1') + direction('<dynamics><p/></dynamics>') + note('a')),
            measure('<barline location="left"><repeat direction="forward"/></barline>' + note('b'), 2),
            measure(direction('<dynamics><f/></dynamics>') + note('c')
                    + '<barline><repeat direction="backward"/></barline>', 3))
        f = self.features(xml)
        self.assertEqual([self.category(f, i, 'dynamic_base') for i in range(5)], ['p', 'p', 'f', 'p', 'f'])
        self.assertEqual(f.numeric[3, N.index('dynamic_base_distance_q')], 1)

    def test_pickup_compound_and_ambiguous_additive_meter(self):
        pickup = self.features(score_xml(measure(attributes() + note('a'), implicit=True)))
        self.assertEqual(pickup.numeric[0, N.index('measure_phase')], .75)
        self.assertEqual(pickup.numeric[0, N.index('beat_strength')], 2)
        compound = self.features(score_xml(measure(attributes(beats='6', denominator=8)
                                  + note('a', duration=6) + note('b', duration=6))))
        self.assertEqual(compound.numeric[1, N.index('beat_strength')], 3)
        ambiguous = self.features(score_xml(measure(attributes(beats='7', denominator=8) + note('a', duration=14))))
        self.assertFalse(ambiguous.field_known[0, N.index('beat_strength')])
        additive = self.features(score_xml(measure(attributes(beats='3+2+2', denominator=8) + note('a', duration=6) + note('b', duration=8))))
        self.assertEqual(additive.numeric[1, N.index('beat_strength')], 3)

    def test_conflicts_and_unsupported_words_remain_unknown(self):
        xml = score_xml(measure(attributes() + direction('<dynamics><p/><f/></dynamics>')
                        + direction('<words>mystery direction</words>') + note('a', duration=16)))
        f = self.features(xml)
        self.assertEqual(self.category(f, 0, 'dynamic_base'), 'UNK')
        self.assertEqual(self.category(f, 0, 'tempo_local'), 'UNK')
        self.assertGreaterEqual(len(f.diagnostics), 2)

    def test_scale_uses_known_training_values_and_deduplicates_score(self):
        f = self.features()
        raw = f.numeric.copy()
        raw[-1, N.index('quarter_bpm')] = 1e20
        known = f.field_known.copy()
        known[-1, N.index('quarter_bpm')] = False
        altered = replace(f, numeric=raw, field_known=known)
        scale = fit_note_score_scale([altered, altered])
        single = fit_note_score_scale([altered])
        self.assertEqual(scale, single)
        self.assertAlmostEqual(scale.mean[N.index('quarter_bpm')], np.log1p(120), places=6)
        self.assertEqual(scale.transform(altered)[-1, N.index('quarter_bpm')], 0)
        self.assertEqual(scale.training_sources, (f.source_sha256,))
        self.assertAlmostEqual(encode_note_numeric(f)[0, N.index('midi_pitch')], 60 / 127, places=6)
