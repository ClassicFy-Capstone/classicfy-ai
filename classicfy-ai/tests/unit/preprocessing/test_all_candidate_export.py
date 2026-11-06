"""All-candidate delivery retains every candidate and explicitly marks bad maps."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import mido
import pretty_midi

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from export_all_representative_candidates import export_performance, validate_clip


class AllCandidateExportTests(unittest.TestCase):
    def fixture(self, folder):
        source=Path(folder)/'source.mid'
        midi=mido.MidiFile(ticks_per_beat=1000)
        track=mido.MidiTrack([mido.MetaMessage('set_tempo',tempo=1000000)])
        midi.tracks.append(track)
        for pitch in range(60,66):
            track.append(mido.Message('note_on',note=pitch,velocity=80,time=0))
            track.append(mido.Message('note_off',note=pitch,time=1000))
        midi.save(source)
        row=dict(perf_id='00001',composition_id='work',local_midi=str(source),quality='',
                 sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        candidates=[dict(start_seconds=a,end_seconds=b,representation_score=.8) for a,b in [(0.,2.),(3.,5.),(4.,4.)]]
        mappings=[dict(candidate_index=i,start_seconds=c['start_seconds'],end_seconds=c['end_seconds'],
                 alignment_accepted=i==0,alignment_similarity=.9 if i==0 else .3) for i,c in enumerate(candidates)]
        work=dict(reference_perf_id='00001',candidates=candidates,mappings={'00001':mappings},selection={'selected_candidate':0})
        return row,work

    def test_every_candidate_is_retained_including_uncertain_and_empty(self):
        with tempfile.TemporaryDirectory() as td:
            row,work=self.fixture(td);out=Path(td)/'out'
            rows=export_performance(row,work,out)
            self.assertEqual([r['status'] for r in rows],['exported','exported_review','invalid_interval'])
            self.assertEqual([r['candidate_number'] for r in rows],[1,2,3])
            self.assertTrue((out/'midi/work/00001/candidate_01.mid').exists())
            self.assertTrue((out/'midi_review/work/00001/candidate_02.mid').exists())
            self.assertTrue(all(not r['automatic_selection_applied'] for r in rows))
            self.assertEqual(hashlib.sha256(Path(row['local_midi']).read_bytes()).hexdigest(),row['sha256'])

    def test_missing_mapping_is_counted_without_inventing_a_time(self):
        with tempfile.TemporaryDirectory() as td:
            row,work=self.fixture(td);work['mappings']['00001']=[]
            rows=export_performance(row,work,Path(td)/'out')
            self.assertEqual(len(rows),3)
            self.assertTrue(all(r['status']=='mapping_missing' for r in rows))
            self.assertTrue(all(r['clip_relative_path']=='' for r in rows))

    def test_source_change_prevents_export(self):
        with tempfile.TemporaryDirectory() as td:
            row,work=self.fixture(td);row['sha256']='wrong'
            rows=export_performance(row,work,Path(td)/'out')
            self.assertTrue(all(r['status']=='source_mismatch' for r in rows))
            self.assertFalse(list((Path(td)/'out').rglob('*.mid')))

    def test_validator_detects_changed_note_content(self):
        with tempfile.TemporaryDirectory() as td:
            row,work=self.fixture(td);out=Path(td)/'out'
            first=export_performance(row,work,out)[0]
            original=pretty_midi.PrettyMIDI(row['local_midi'])
            original.instruments[0].notes[0].velocity=42
            with self.assertRaisesRegex(ValueError,'independent_note_mismatch'):
                validate_clip(out/first['clip_relative_path'],original,0.,2.,2)


if __name__=='__main__':unittest.main()
