"""Audit local MusicXML features against raw note arithmetic and run every onset chunk."""

import argparse
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from embedding.note_score_context_data import collate_note_score_context, split_note_score_context
from embedding.note_score_encoder import NoteScoreEncoder
from features.note_score_context import CATEGORICAL_FIELDS, FLAG_FIELDS, NUMERIC_FIELDS, extract_note_score_context, note_score_schema
from preprocessing.musicxml_score import read_musicxml


def raw_note_ledger(path):
    """Check source pitch/duration directly, without using parser/feature helpers."""
    root = ET.parse(path).getroot()
    for element in root.iter():
        element.tag = element.tag.rsplit('}', 1)[-1]
    ledger = {}
    for part in root.findall('part'):
        divisions = Fraction(1)
        for mi, measure in enumerate(part.findall('measure')):
            for ei, element in enumerate(measure):
                if element.tag == 'attributes' and element.find('divisions') is not None:
                    divisions = Fraction(element.findtext('divisions'))
                if element.tag == 'note' and element.find('pitch') is not None:
                    pitch = element.find('pitch')
                    written = 12 * (int(pitch.findtext('octave')) + 1)
                    written += {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[pitch.findtext('step')]
                    written += Fraction(pitch.findtext('alter', '0'))
                    duration = Fraction(0) if element.find('grace') is not None else Fraction(element.findtext('duration')) / divisions
                    identifier = element.get('id') or f'{part.get("id")}/measure[{mi}]/element[{ei}]'
                    ledger[identifier] = (written, duration)
    return ledger


def parser_rejection(error):
    message = str(error)
    if ('unsupported' in message.lower() or 'supported unfolded score edition' in message
            or 'require an explicit measure traversal' in message):
        return 'unsupported'
    if (message.startswith('MusicXML backup precedes measure start')
            or message.startswith('Invalid note duration') or message.startswith('Duplicate MusicXML note ID')):
        return 'invalid_source'
    return 'failed'


def validate(root, out, *, max_onsets=128):
    paths = sorted(Path(root).rglob('*.musicxml'))
    if not paths:
        raise ValueError('No local MusicXML scores found')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    threads = torch.get_num_threads()
    rows, support = [], np.zeros(len(NUMERIC_FIELDS + CATEGORICAL_FIELDS + FLAG_FIELDS), np.int64)
    torch.set_num_threads(1)
    start = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261029)
            encoder = NoteScoreEncoder().eval()
            for pi, path in enumerate(paths):
                row = {'path': str(path), 'status': 'failed'}
                try:
                    score = read_musicxml(path)
                except ValueError as exc:
                    row.update(status=parser_rejection(exc), message=str(exc))
                except Exception as exc:
                    row.update(message=f'{type(exc).__name__}: {exc}')
                else:
                    try:
                        features = extract_note_score_context(score)
                        ledger = raw_note_ledger(path)
                        for note in score.notes:
                            if (note.written_pitch, note.duration) != ledger[note.source_id]:
                                raise AssertionError(f'Independent pitch/duration mismatch at {note.occurrence_id}')
                        chunks = split_note_score_context(features, max_onsets=max_onsets, overlap=8)
                        valid_outputs = 0
                        with torch.no_grad():
                            for start_chunk in range(0, len(chunks), 8):
                                batch = collate_note_score_context(chunks[start_chunk:start_chunk + 8])
                                context = encoder(batch)
                                if not torch.isfinite(context).all():
                                    raise AssertionError('Nonfinite score context')
                                if torch.count_nonzero(context[~batch['note_mask']]):
                                    raise AssertionError('Nonzero padded context')
                                valid_outputs += int(batch['loss_note_mask'].sum())
                        if valid_outputs != len(score.notes):
                            raise AssertionError('Chunk centers do not cover every written note exactly once')
                        row.update(status='ok', notes=len(score.notes), onsets=len(features.onset_q),
                                   chunks=len(chunks), independent_note_checks=len(score.notes),
                                   tie_continuations=int((~features.ties.is_attack).sum()),
                                   tie_unknown=int((~features.ties.chain_known).sum()),
                                   context_rows=valid_outputs, direction_diagnostics=len(features.diagnostics),
                                   source_sha256=features.source_sha256, score_ref=features.cache_key)
                        support += features.field_known.sum(axis=0)
                    except Exception as exc:
                        if isinstance(exc, ValueError) and str(exc) == 'Cross-measure direction offset is unsupported':
                            row.update(status='unsupported', message=str(exc))
                        else:
                            row.update(message=f'{type(exc).__name__}: {exc}')
                rows.append(row)
                if (pi + 1) % 40 == 0:
                    print(f'Checked {pi + 1}/{len(paths)} XML files', flush=True)
    finally:
        torch.set_num_threads(threads)
    counts = Counter(row['status'] for row in rows)
    accepted = [row for row in rows if row['status'] == 'ok']
    total_notes = sum(row['notes'] for row in accepted)
    sources = [Path(__file__), *[Path(__file__).parents[1] / 'src' / name for name in
               ('preprocessing/musicxml_score.py', 'preprocessing/score_note_mapping.py',
                'features/note_score_context.py', 'embedding/note_score_context_data.py', 'embedding/note_score_encoder.py')]]
    summary = {'schema': note_score_schema(), 'encoder_config': asdict(encoder.config),
               'seed': 20261029, 'weights': 'untrained; validates contracts, not musical quality',
               'scores': len(rows), 'statuses': dict(counts), 'written_notes': total_notes,
               'onsets': sum(row['onsets'] for row in accepted),
               'context_rows': sum(row['context_rows'] for row in accepted),
               'independent_note_checks': sum(row['independent_note_checks'] for row in accepted),
               'chunks': sum(row['chunks'] for row in accepted),
               'field_support': {name: {'known': int(count), 'total': total_notes} for name, count in
                                 zip(NUMERIC_FIELDS + CATEGORICAL_FIELDS + FLAG_FIELDS, support)},
               'rejections': dict(Counter(row.get('message') for row in rows if row['status'] != 'ok')),
               'code_sha256': hashlib.sha256(b''.join(path.read_bytes() for path in sources)).hexdigest(),
               'elapsed_seconds': time.perf_counter() - start}
    (out / 'scores.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-onsets', type=int, default=128)
    args = parser.parse_args()
    if args.max_onsets < 1:
        parser.error('--max-onsets must be positive')
    summary = validate(args.xml_root, args.out, max_onsets=args.max_onsets)
    print(json.dumps({key: summary[key] for key in ('scores', 'statuses', 'written_notes', 'context_rows', 'chunks')}, ensure_ascii=False))
    if summary['statuses'].get('failed', 0):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
