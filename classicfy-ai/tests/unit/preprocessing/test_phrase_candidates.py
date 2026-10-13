from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

from preprocessing import load_musicxml
from preprocessing.phrase_candidates import extract_phrase_candidates,pair_written_slurs,phrase_position


def note(identifier,step='C',duration=1,voice='1',staff=1,notations='',extra='',octave=4):
    return f'<note id="{identifier}">{extra}<pitch><step>{step}</step><octave>{octave}</octave></pitch><duration>{duration}</duration><voice>{voice}</voice><staff>{staff}</staff><notations>{notations}</notations></note>'


def rest(identifier,duration=1,voice='1',staff=1):
    return f'<note id="{identifier}"><rest/><duration>{duration}</duration><voice>{voice}</voice><staff>{staff}</staff></note>'


class PhraseCandidatesTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)

    def score(self,body,mode='major',divisions=1):
        p=Path(self.temp.name)/'score.musicxml'
        mode_xml=f'<mode>{mode}</mode>' if mode else ''
        p.write_text('<score-partwise><part id="P1"><measure number="1"><attributes>'+
            f'<divisions>{divisions}</divisions><key><fifths>0</fifths>{mode_xml}</key>'+
            '<time><beats>4</beats><beat-type>4</beat-type></time></attributes>'+body+
            '</measure></part></score-partwise>')
        return load_musicxml(p)

    def cues(self,result,kind):
        return [e for c in result.candidates for e in c.cues if e.kind==kind]

    def test_slur_continue_is_not_an_extra_boundary_and_stop_release_is_used(self):
        score=self.score(note('a',notations='<slur type="start" number="1"/>')+
            note('b',notations='<slur type="continue" number="1"/>')+
            note('c',notations='<slur type="stop" number="1"/>'))
        result=extract_phrase_candidates(score);cues=self.cues(result,'slur_end')
        self.assertEqual(len(cues),1);self.assertEqual(cues[0].position_quarters,3)
        self.assertEqual(cues[0].details['source_stop_onset'],'2')
        self.assertEqual(result.slurs[0].status,'paired')
        self.assertTrue(all(c.status=='pending_score_review' for c in result.candidates))

    def test_cross_staff_slur_is_preserved_at_part_scope(self):
        score=self.score(note('a',notations='<slur type="start" number="1"/>')+
            note('b',staff=2,voice='5',notations='<slur type="stop" number="1"/>'))
        result=extract_phrase_candidates(score)
        self.assertEqual(result.slurs[0].status,'paired_cross_scope')
        self.assertEqual(self.cues(result,'slur_end')[0].scope,'P1:part')

    def test_reused_number_and_stop_start_shared_endpoint_pair_in_document_order(self):
        score=self.score(note('a',notations='<slur type="start" number="1"/>')+
            note('b',notations='<slur type="start" number="1"/><slur type="stop" number="1"/>')+
            note('c',notations='<slur type="stop" number="1"/>'))
        pairs=pair_written_slurs(score)
        self.assertEqual([(s.start_note_id,s.stop_note_id) for s in pairs],
                         [('P1:m0:n0','P1:m0:n1'),('P1:m0:n1','P1:m0:n2')])

    def test_conflicting_starts_and_orphan_stops_are_not_silently_paired(self):
        score=self.score(note('a',notations='<slur type="start" number="1"/>')+
            note('b',notations='<slur type="start" number="1"/>')+
            note('c',notations='<slur type="stop" number="1"/>'))
        result=extract_phrase_candidates(score)
        self.assertFalse(self.cues(result,'slur_end'))
        self.assertEqual([s.status for s in result.slurs],['conflicting_start','conflicting_start','ambiguous_stop'])

    def test_same_onset_slur_is_a_diagnostic_not_a_phrase(self):
        score=self.score(note('a',notations='<slur type="start" number="1"/>')+
            '<backup><duration>1</duration></backup>'+note('b',voice='2',notations='<slur type="stop" number="1"/>'))
        result=extract_phrase_candidates(score)
        self.assertEqual(result.slurs[0].status,'nonpositive_onset_span')
        self.assertFalse(self.cues(result,'slur_end'))

    def test_voice_rest_does_not_mean_global_silence_over_accompaniment(self):
        body=note('a')+rest('r')+note('b')+'<backup><duration>3</duration></backup>'+note('bass','G',duration=3,voice='5',staff=2,octave=3)
        result=extract_phrase_candidates(self.score(body))
        cues=self.cues(result,'voice_rest');self.assertEqual(len(cues),1)
        self.assertEqual((cues[0].uncertainty_start_quarters,cues[0].uncertainty_stop_quarters),(1,2))
        self.assertFalse(self.cues(result,'global_notated_silence'))

    def test_overlapping_pitch_in_same_voice_prevents_rest_candidate(self):
        body=note('held',duration=3)+'<backup><duration>3</duration></backup>'+rest('r')+note('b',duration=2)
        result=extract_phrase_candidates(self.score(body))
        self.assertFalse(self.cues(result,'voice_rest'))
        self.assertTrue(any(d['kind']=='rest_overlaps_pitched_scope' for d in result.diagnostics))

    def test_short_rest_threshold_and_exact_fraction_positions(self):
        result=extract_phrase_candidates(self.score(note('a')+rest('r')+note('b'),divisions=4))
        self.assertFalse(self.cues(result,'voice_rest'));self.assertFalse(self.cues(result,'global_notated_silence'))
        with self.assertRaises(ValueError):extract_phrase_candidates(self.score(note('a')),minimum_rest_quarters=0)

    def test_slur_plus_rest_merge_without_fake_numeric_confidence(self):
        score=self.score(note('a',notations='<slur type="start" number="1"/>')+
            note('b',notations='<slur type="stop" number="1"/>')+rest('r')+note('c'))
        result=extract_phrase_candidates(score)
        c=next(c for c in result.candidates if c.scope=='P1:staff=1:voice=1' and c.position_quarters==2)
        self.assertEqual({e.kind for e in c.cues},{'slur_end','voice_rest'})
        self.assertEqual(c.review_priority,'high_review');self.assertEqual(c.uncertainty_stop_quarters,3)

    def chord(self,prefix,steps,octaves=None,duration=1):
        octaves=octaves or [3,4,4]
        return ''.join(note(prefix+str(i),step,duration=duration,octave=octaves[i],extra='<chord/>' if i else '')
                       for i,step in enumerate(steps))

    def test_major_v_i_hint_has_pitch_evidence_without_claiming_cadence(self):
        result=extract_phrase_candidates(self.score(self.chord('v',['G','B','D'])+self.chord('i',['C','E','G'])))
        cue=self.cues(result,'harmonic_closure_hint')[0]
        self.assertEqual(cue.position_quarters,2);self.assertEqual(cue.details['mode_source'],'explicit_musicxml')
        self.assertEqual(cue.uncertainty_start_quarters,1)
        self.assertEqual(cue.uncertainty_stop_quarters,2)
        self.assertTrue(cue.details['tonic_prolongation_possible']);self.assertTrue(cue.details['not_phrase_ground_truth'])

    def test_missing_mode_remains_a_named_hypothesis(self):
        result=extract_phrase_candidates(self.score(self.chord('v',['G','B','D'])+self.chord('i',['C','E','G']),mode=None))
        cue=self.cues(result,'harmonic_closure_hint')[0]
        self.assertEqual(cue.details['mode'],'major')
        self.assertEqual(cue.details['mode_source'],'unconfirmed_key_signature_hypothesis')

    def test_inverted_bass_or_unsupported_mode_do_not_receive_root_cadence_hint(self):
        inverted=self.chord('v',['G','B','D'],octaves=[4,3,4])+self.chord('i',['C','E','G'])
        self.assertFalse(self.cues(extract_phrase_candidates(self.score(inverted)),'harmonic_closure_hint'))
        modal=self.chord('v',['G','B','D'])+self.chord('i',['C','E','G'])
        self.assertFalse(self.cues(extract_phrase_candidates(self.score(modal,mode='dorian')),'harmonic_closure_hint'))

    def test_navigation_symbol_alone_is_not_high_phrase_evidence(self):
        result=extract_phrase_candidates(self.score(note('a')+note('b')+'<barline><repeat direction="backward"/></barline>'))
        c=next(c for c in result.candidates if any(e.kind=='navigation_boundary' for e in c.cues))
        self.assertEqual(c.review_priority,'low_review')

    def test_phrase_relative_position_requires_explicit_positive_interval(self):
        self.assertEqual(phrase_position(0,9,0),(Fraction(0),'start'))
        self.assertEqual(phrase_position(0,9,3),(Fraction(1,3),'middle'))
        self.assertEqual(phrase_position(0,9,6),(Fraction(2,3),'end'))
        self.assertIsNone(phrase_position(0,9,9));self.assertIsNone(phrase_position(0,9,-1))
        with self.assertRaises(ValueError):phrase_position(1,1,1)


if __name__=='__main__':unittest.main()
