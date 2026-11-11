"""Prepare bounded-memory full-corpus inputs and train the unchanged note AE."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from embedding.note_ae_indexed_data import IndexedNoteAECorpus, file_sha, pack_record
from embedding.note_ae_training import NoteAETrainingConfig, train_note_ae
from embedding.note_autoencoder import NoteAEConfig
from embedding.note_score_encoder import NoteScoreEncoderConfig
from embedding.note_score_context_data import load_note_score_scale
from features.note_performance_normalization import load_note_performance_scaler
from features.style_profile import check_artifact, stable_id


def prepare(source, destination, workers):
    index = json.loads((source / 'window_index.json').read_text())
    check_artifact(index, 'window-index')
    if file_sha(source / 'splits.json') != index['split_protocol_sha256']:
        raise ValueError('Source split protocol changed')
    scores = {}
    for line in (source / 'scores.jsonl').open():
        row = json.loads(line)
        if row['status'] == 'prepared':
            scores[row['score_path']] = row['score_cache']
    destination.mkdir(parents=True, exist_ok=False)
    for kind in ('records', 'scores'):
        (destination / kind).mkdir()
    for name in ('R_scaler.json', 'S_scale.json'):
        shutil.copyfile(source / name, destination / name)
    scaler = load_note_performance_scaler(destination / 'R_scaler.json')
    score_scale = load_note_score_scale(destination / 'S_scale.json')
    expected_train = {s['record_key'] for s in index['sources'] if s['split'] == 'train'}
    if {r['record_key'] for r in scaler.training_records} != expected_train:
        raise ValueError('Train scaler provenance differs from indexed training recordings')
    tasks = [(e, scores[e['score_path']], str(destination), str(destination / 'R_scaler.json'),
              str(destination / 'S_scale.json')) for e in index['sources']]
    items, excluded, records, score_splits = [], [], {}, defaultdict(set)
    split_keys = {k: defaultdict(set) for k in ('work', 'score', 'recording')}
    summary = defaultdict(lambda: {'works': set(), 'recordings': 0, 'windows': 0, 'notes': 0})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(pack_record, tasks, chunksize=1), 1):
            items.extend(result['items'])
            excluded.extend(result['excluded'])
            entry = result['source']
            sp = entry['split']
            if sp not in {'train', 'validation', 'test'}:
                raise ValueError('Unassigned recording in full-corpus run')
            for kind, key in [('work', entry['work_key']), ('score', result['score_sha']),
                              ('recording', entry['recording_id'])]:
                split_keys[kind][key].add(sp)
            score_splits[result['score_sha']].add(sp)
            records[entry['record_key']] = result['record_sha']
            summary[sp]['works'].add(entry['work_key'])
            summary[sp]['recordings'] += 1
            summary[sp]['windows'] += len(result['items'])
            summary[sp]['notes'] += sum(v['window']['N'] for v in result['items'])
            if i % 100 == 0 or i == len(tasks):
                print(json.dumps({'prepared_recordings': i, 'total_recordings': len(tasks),
                                  'full_windows': len(items)}), flush=True)
    for kind, keys in split_keys.items():
        if any(len(splits) != 1 for splits in keys.values()):
            raise ValueError(f'{kind} crosses train/validation/test boundaries')
    if set(score_scale.training_sources) != {k for k, v in score_splits.items() if v == {'train'}}:
        raise ValueError('Score scaler provenance differs from indexed training scores')
    summary = {k: {**v, 'works': len(v['works'])} for k, v in summary.items()}
    manifest = {'format': 'indexed-note-ae-tensors-v1', 'source_index_id': index['id'],
                'source_index': str(source / 'window_index.json'),
                'source_index_sha256': file_sha(source / 'window_index.json'),
                'split_protocol_sha256': index['split_protocol_sha256'],
                'R_scaler': scaler.fingerprint, 'score_scale': stable_id(asdict(score_scale)),
                'splits': {e['performance_id']: e['split'] for e in index['sources']},
                'items': sorted(items, key=lambda v: v['window']['window_id']), 'excluded': excluded,
                'summary': summary, 'split_overlap': {k: 0 for k in split_keys},
                'tensor_hashes': {'records': records, 'scores': {
                    p.stem: file_sha(p) for p in (destination / 'scores').glob('*.pt')}}}
    manifest['id'] = stable_id(manifest)
    with (destination / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, allow_nan=False)
    print(json.dumps({'prepared': summary, 'excluded': dict(Counter(v['reason'] for v in excluded)),
                      'split_overlap': manifest['split_overlap']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--workers', type=int, default=3)
    train = sub.add_parser('train')
    train.add_argument('--corpus', type=Path, required=True)
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.source.resolve(), args.out.resolve(), args.workers)
    else:
        raw = json.loads(args.config.read_text())
        if set(raw) - {'model', 'score', 'training'}:
            raise ValueError('Unknown config section')
        model_config = NoteAEConfig(**raw.get('model', {}))
        score_config = NoteScoreEncoderConfig(**raw.get('score', {'context_dim': model_config.context_dim}))
        corpus = IndexedNoteAECorpus(args.corpus)
        train_note_ae(corpus, args.out, config=NoteAETrainingConfig(**raw['training']),
                      model_config=model_config, score_config=score_config, resume=args.resume)


if __name__ == '__main__':
    main()
