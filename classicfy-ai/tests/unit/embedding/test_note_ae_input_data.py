"""Pre-AE input contracts only: no AE model or loss is imported."""
import numpy as np

from embedding.note_ae_data import collate_ae_windows
from embedding.note_performance_data import NotePerformancePair, NotePerformanceSegment
from embedding.note_score_context_data import NoteScoreSegment
from features.note_performance_normalization import fit_note_performance_scaler
from features.note_score_context import extract_note_score_context, fit_note_score_scale
from features.style_profile import make_windows
from integration import test_style_triplets as helpers
from unit.preprocessing.musicxml_fixtures import XMLFixture, attributes, measure, note, score_xml


class NoteAEInputDataTest(XMLFixture):
    def test_rest_at_eight_bar_boundary_keeps_tempo_in_its_actual_window(self):
        bars=[]
        for i in range(16):
            body=attributes() if i==0 else ''
            if i==8:
                body+=note('rest',rest=True)+note('n8',duration=12)
            else:
                body+=note(f'n{i}',duration=16)
            bars.append(measure(body,i+1))
        xml=self.read(score_xml(*bars))
        rec=helpers.StyleTripletsTest.record(self,xml,'input',1.)
        s=extract_note_score_context(xml);r=rec.performance
        pair=NotePerformancePair(s,r);items=[]
        for w in make_windows(rec):
            a,b=np.searchsorted(s.onset_q,[w['q_start'],w['q_end']],side='left')
            items.append({'window':w,'segment':NotePerformanceSegment(pair,NoteScoreSegment(s,int(a),int(b),int(a),int(b)))})
        scaler=fit_note_performance_scaler([r],fit_manifest_id='input')
        batch=collate_ae_windows(items,r_scaler=scaler,score_scale=fit_note_score_scale([s]))
        self.assertEqual(batch['beat_loss_mask'].sum(1).tolist(),[32,32])
        self.assertFalse(batch['beat_loss_mask'][0,32])
        self.assertTrue(batch['beat_loss_mask'][1,32])
        for i,item in enumerate(items):
            rows=item['window']['row_indices']
            np.testing.assert_array_equal(batch['R'][i,:len(rows)].numpy(),scaler.transform(r)[rows])
