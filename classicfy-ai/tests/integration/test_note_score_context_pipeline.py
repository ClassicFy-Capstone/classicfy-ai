from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

from embedding.note_score_context_data import collate_note_score_context, load_note_score_context, save_note_score_context, split_note_score_context
from embedding.note_score_encoder import NoteScoreEncoder
from features.note_score_context import extract_note_score_context
from preprocessing.musicxml_score import read_musicxml
from preprocessing.note_alignment import load_match
from preprocessing.score_note_mapping import link_note_alignment
from unit.preprocessing.musicxml_fixtures import XMLFixture, example_xml


ROOT = Path(__file__).resolve().parents[3]


class NoteScorePipelineTest(XMLFixture):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_export_cli_cache_batch_and_context(self):
        path = self.folder / 'source.musicxml'
        path.write_text(example_xml())
        out = self.folder / 'export'
        command = [sys.executable, str(ROOT / 'classicfy-ai/scripts/export_note_score_context.py'),
                   '--xml', str(path), '--out', str(out)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('8 written notes, 5 onset groups', result.stdout)
        features, links = load_note_score_context(out)
        self.assertIsNone(links)
        self.assertEqual(len(json.loads((out / 'source_tables.json').read_text())['rests']), 1)
        output = NoteScoreEncoder()(collate_note_score_context([features]))
        self.assertEqual(output.shape, (1, 8, 128))
        self.assertTrue(torch.isfinite(output).all())
        repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn('already exists', repeated.stderr)

    def test_real_nasap_raw_duration_ledger_alignment_and_encoder(self):
        root = Path(os.environ.get('NASAP_ROOT', ROOT.parent / 'datasets/nASAP'))
        path = root / 'Bach/Prelude/bwv_846/xml_score.musicxml'
        match = path.parent / 'Shi05M.match'
        if not path.is_file() or not match.is_file():
            self.skipTest('Actual nASAP XML and .match are required')
        score = read_musicxml(path)
        features = extract_note_score_context(score)
        # An independent source ledger checks durations without the feature/parser helpers.
        raw = ET.parse(path).getroot()
        ledger = {}
        for part in raw.findall('part'):
            divisions = 1
            for measure in part.findall('measure'):
                for element in measure:
                    if element.tag == 'attributes' and element.find('divisions') is not None:
                        divisions = int(element.findtext('divisions'))
                    if element.tag == 'note' and element.find('pitch') is not None:
                        ledger[element.get('id')] = Fraction(0) if element.find('grace') is not None else Fraction(int(element.findtext('duration')), divisions)
        for note in score.notes:
            self.assertEqual(note.duration, ledger[note.xml_id])
        alignment = load_match(match)
        # Explicit dataset convention, verified against pitch and quarter positions before use.
        by_dataset_id = {f'{n.source_id}-{n.occurrence}': n for n in score.notes}
        aliases = {}
        for snote, _ in alignment.matches:
            if snote.note_id in by_dataset_id:
                n = by_dataset_id[snote.note_id]
                self.assertEqual(snote.pitch, n.pitch)
                self.assertAlmostEqual(snote.onset_beats, float(n.onset), places=4)
                aliases[snote.note_id] = n.occurrence_id
        self.assertGreater(len(aliases), .9 * len(alignment.matches))
        links = link_note_alignment(score, features.ties, alignment, score_id_to_occurrence=aliases)
        self.assertGreater(int(links.attack_mask.sum()), 400)
        self.assertFalse(links.attack_mask[~features.ties.is_attack].any())
        self.assertTrue(np.all(features.ties.chain_known))
        out = self.folder / 'actual'
        save_note_score_context(features, out, links=links, source_tables=asdict(score))
        restored, restored_links = load_note_score_context(out)
        chunks = split_note_score_context(restored, max_onsets=64, overlap=8)[:2]
        batch = collate_note_score_context(chunks, alignment_links=[restored_links] * len(chunks))
        with torch.no_grad():
            context = NoteScoreEncoder().eval()(batch)
        self.assertEqual(context.shape[:2], batch['note_mask'].shape)
        self.assertTrue(torch.isfinite(context).all())
        self.assertEqual(torch.count_nonzero(context[~batch['note_mask']]), 0)
