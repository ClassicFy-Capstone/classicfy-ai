"""Resume full local ATEPP preparation and CPU AE training from approved sources.

No source MIDI, alignment, score or selected representative clip is modified.
Rejected inputs are audited; no synthetic data or relaxed alignment fallback.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def save_derived(value, path):
    """Resume deterministic derived stages without replacing a different run."""
    from embedding.style_profile_data import save_style_artifact
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'Existing derived artifact differs: {path}')
    else:
        save_style_artifact(value, path)


def log_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


@lru_cache(maxsize=3)
def symbolic(path):
    from preprocessing.musicxml_score import read_musicxml
    from features.note_score_context import extract_note_score_context
    xml = read_musicxml(path)
    return xml, extract_note_score_context(xml)


def verified_aliases(xml, midi_notes, tolerance=1e-4):
    """Verify exact pitch/quarter coordinates; never infer an edition by filename."""
    from preprocessing.score_note_mapping import build_tie_mapping
    ties = build_tie_mapping(xml)
    by_pitch = defaultdict(list)
    for i, note in enumerate(xml.notes):
        if ties.is_attack[i] and not note.grace:
            by_pitch[note.pitch].append(note)
    aliases = {}
    for note in midi_notes:
        matches = [n for n in by_pitch[note.pitch] if abs(float(n.onset) - note.onset_beats) <= tolerance]
        if len(matches) == 1:
            aliases[note.note_id] = matches[0].occurrence_id
    # Ambiguous source unisons must not count as independent score observations.
    counts = Counter(aliases.values())
    return {k: v for k, v in aliases.items() if counts[v] == 1}


def extract_worker(task):
    row, out = task
    from preprocessing.score_midi import read_score_midi
    from preprocessing.note_performance_adapters import prepare_atepp_note_performance
    from embedding.note_performance_data import save_note_performance, load_note_performance
    from embedding.note_score_context_data import save_note_score_context
    from features.note_performance import summarize_note_performance
    pid = row['perf_id']
    result = {'perf_id': pid, 'composition_id': row['composition_id'], 'split': row['split'],
              'work_key': row['work_key'], 'xml_path': row['xml_path']}
    destination = Path(out) / 'performances' / pid
    try:
        xml, score = symbolic(row['xml_path'])
        score_notes, _, diagnostics = read_score_midi(row['local_score'])
        aliases = verified_aliases(xml, score_notes)
        fraction = len(aliases) / max(1, len(score_notes))
        result.update(score_alias_fraction=fraction, score_aliases=len(aliases), score_notes=len(score_notes))
        if fraction < .8:
            raise ValueError('Verified XML/score-MIDI pitch/quarter aliases below 80%')
        destination.mkdir(parents=True, exist_ok=True)
        with np.load(row['alignment_cache'], allow_pickle=False) as original:
            ids = original['score_ids']
            keep = np.array([str(key) in aliases for key in ids])
            if keep.sum() < 20 or keep.mean() < .8:
                raise ValueError('Verified XML identity covers less than 80% or 20 alignment matches')
            fields = ('score_ids', 'performance_ids', 'score_onsets', 'score_offsets',
                      'performance_onsets', 'performance_offsets', 'pitch')
            arrays = {key: original[key][keep].copy() for key in fields}
            arrays.update({key: original[key].copy() for key in ('score_beats', 'performance_beats')})
            if 'beat_valid' in original:
                arrays['beat_valid'] = original['beat_valid'].copy()
        cache = destination / 'verified_alignment.npz'
        if not cache.exists():
            np.savez_compressed(cache, **arrays)
        bridge = {'xml_sha256': xml.source_sha256, 'score_midi_sha256': sha(row['local_score']),
                  'score_coordinate_unit': 'quarter', 'alignment_accepted': True,
                  'score_id_to_occurrence': aliases,
                  'bridge_policy': 'unique attack pitch/onset within 1e-4 quarter; >=80% aliases and matched rows',
                  'source_alignment_sha256': sha(row['alignment_cache']),
                  'source_performance_sha256': row['recording_id'],
                  'source_score_diagnostics': diagnostics,
                  'retained_matches': int(keep.sum()), 'excluded_unverified_matches': int((~keep).sum())}
        dump(destination / 'bridge.json', bridge)
        s, r = prepare_atepp_note_performance(row['xml_path'], row['local_score'], row['local_midi'],
            cache, bridge, record_key=pid, work_key=row['work_key'], split=row['split'], score_features=score)
        coverage = summarize_note_performance(r)
        # A successful bridge is not sufficient if raw-MIDI binding yields no useful observations.
        if not r.mask[:, 1:4].any() or not r.beats.valid.any():
            raise ValueError('No usable core raw-MIDI/beat observations after identity binding')
        if not (destination / 'S').exists():
            save_note_score_context(s, destination / 'S')
        if not (destination / 'R').exists():
            save_note_performance(r, destination / 'R')
        saved = load_note_performance(destination / 'R')
        if saved.fingerprint != r.fingerprint:
            raise ValueError('Resumed source R fingerprint changed')
        result.update(status='extracted', performance_cache=str(destination / 'R'),
                      score_cache=str(destination / 'S'), score_source_sha256=r.score_source_sha256,
                      recording_id=row['recording_id'], R_fingerprint=r.fingerprint, coverage=coverage)
    except Exception as exc:
        result.update(status='rejected', error_type=type(exc).__name__, error=str(exc))
    return result


def inventory(dataset):
    import csv
    selected = {r['perf_id']: r for r in csv.DictReader((dataset / 'selected_segments.csv').open(encoding='utf-8-sig'))}
    choices = defaultdict(list)
    for row in csv.DictReader((dataset / 'alignment_choices.csv').open(encoding='utf-8-sig')):
        if row['perf_id'] not in selected or row['accepted'].lower() != 'true':
            continue
        score = Path(row['local_score'])
        xmls = ([Path(str(score)[:-5])] if str(score).endswith('.musicxml.midi') else [])
        xmls.append(Path(row['local_midi']).with_name('musicxml_cleaned.musicxml'))
        xml = next((x for x in xmls if x.is_file()), None)
        if xml and score.is_file() and Path(row['alignment_cache']).is_file() and Path(row['local_midi']).is_file():
            choices[row['perf_id']].append({**row, 'xml_path': str(xml)})
    tasks, rejected = [], []
    for pid, selected_row in sorted(selected.items()):
        candidates = choices[pid]
        if not candidates:
            rejected.append({'perf_id': pid, 'status': 'unavailable', 'reason': 'No approved alignment with a local XML score'})
            continue
        candidates.sort(key=lambda r: (not r['local_score'].endswith('.musicxml.midi'),
                                      r['alignment_cache'] != selected_row['alignment_cache'],
                                      r['local_score'] != selected_row['local_score'], r['xml_path'], r['alignment_cache']))
        row = candidates[0]
        row['recording_id'], row['xml_sha256'] = sha(row['local_midi']), sha(row['xml_path'])
        tasks.append(row)
    # Union work IDs before splitting, including duplicate XMLs and identical recordings.
    parent = {r['composition_id']: r['composition_id'] for r in tasks}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = [dict(), dict()]
    for row in tasks:
        for mapping, key in zip(seen, (row['xml_sha256'], row['recording_id'])):
            if key in mapping:
                a, b = find(row['composition_id']), find(mapping[key])
                parent[max(a, b)] = min(a, b)
            mapping[key] = row['composition_id']
    duplicates, unique = set(), []
    for row in tasks:
        work = find(row['composition_id'])
        row['work_key'] = 'ATEPP/' + work
        rank = int(hashlib.sha256(('2026:ATEPP:' + work).encode()).hexdigest()[:8], 16) % 100
        row['split'] = 'train' if rank < 70 else 'validation' if rank < 85 else 'test'
        if row['recording_id'] in duplicates:
            rejected.append({'perf_id': row['perf_id'], 'status': 'duplicate', 'reason': 'Identical original MIDI already included'})
        else:
            duplicates.add(row['recording_id'])
            unique.append(row)
    return unique, rejected


def prepare_inputs(out, results, progress):
    from embedding.note_performance_data import load_note_performance
    from embedding.note_score_context_data import load_note_score_context, save_note_score_scale
    from embedding.style_profile_data import save_style_artifact
    from features.note_performance_normalization import NotePerformanceScaler, fit_note_performance_scaler, save_note_performance_scaler
    from features.note_score_context import fit_note_score_scale
    from features.style_profile import StyleConfig, StyleRecording, artifact, make_windows, stable_id
    inputs = out / 'inputs'
    inputs.mkdir(exist_ok=True)
    sources, items, excluded, provenance, score_paths = [], [], [], [], set()
    count, mean, m2 = np.zeros(11, np.int64), np.zeros(11), np.zeros(11)
    for i, result in enumerate(results, 1):
        r = load_note_performance(result['performance_cache'])
        xml, _ = symbolic(result['xml_path'])
        windows = make_windows(StyleRecording(r, xml, result['recording_id']))
        path = inputs / 'windows' / (r.record_key + '.json')
        dump(path, windows)
        source = dict(performance_id=r.dataset + ':' + r.record_key, record_key=r.record_key, work_key=r.work_key,
            split=r.split, recording_id=result['recording_id'], performance_cache=result['performance_cache'],
            xml_path=result['xml_path'], score_path=result['xml_path'], score_cache=result['score_cache'],
            score_source_sha256=r.score_source_sha256, R_fingerprint=r.fingerprint,
            windows_file=str(path), windows_sha256=sha(path))
        sources.append(source)
        if r.split == 'train':
            local = fit_note_performance_scaler([r], fit_manifest_id='streaming-single-record')
            n, mu = np.array(local.valid_counts), np.array(local.mean)
            variance = np.array([sd * sd if status == 'fitted' else 0. for sd, status in zip(local.scale, local.channel_status)])
            total, delta, valid = count + n, mu - mean, n > 0
            m2[valid] += variance[valid] * n[valid] + delta[valid] ** 2 * count[valid] * n[valid] / total[valid]
            mean[valid] += delta[valid] * n[valid] / total[valid]
            count = total
            provenance.extend(local.training_records)
            score_paths.add(result['score_cache'])
        if i % 100 == 0 or i == len(results):
            progress('input_index', recordings=i, total=len(results))
    index = artifact('window-index', StyleConfig(), sources=sources,
        scope='all accepted full eight-bar observed windows; no downsampling', split_protocol_sha256=sha(out / 'inventory.json'))
    save_derived(index, inputs / 'window_index.json')
    sd = np.sqrt(np.divide(m2, count, out=np.zeros(11), where=count > 0))
    statuses = tuple('unobserved' if n == 0 else 'constant' if s == 0 else 'fitted' for n, s in zip(count, sd))
    sd[sd == 0] = 1.
    scaler = NotePerformanceScaler(tuple(map(float, mean)), tuple(map(float, sd)), tuple(map(int, count)),
        tuple(bool(n) for n in count), statuses, index['id'], tuple(provenance))
    if not (inputs / 'R_scaler.json').exists():
        save_note_performance_scaler(scaler, inputs / 'R_scaler.json')
    # De-duplicate scores before fitting S; fitting accepts an iterable.
    unique_scores = {}
    for path in sorted(score_paths):
        s = load_note_score_context(path)[0]
        unique_scores.setdefault(s.source_sha256, path)
    del s
    scale = fit_note_score_scale(load_note_score_context(path)[0] for path in unique_scores.values())
    if not (inputs / 'S_scale.json').exists():
        save_note_score_scale(scale, inputs / 'S_scale.json')
    dump(inputs / 'ready.json', {'index_id': index['id'], 'R_scaler': scaler.fingerprint})
    return index


def prepare_style(out, index, progress):
    from embedding.note_performance_data import load_note_performance
    from embedding.style_profile_data import save_style_artifact
    from embedding.style_triplets import calibrate_distances, build_triplets
    from features.style_profile import (StyleConfig, StyleRecording, artifact, fit_reference, summarize_window, reference_delta,
                                       fit_profile_scaler, stable_id)
    config = StyleConfig()
    directory = out / 'style'
    directory.mkdir(exist_ok=True)
    cohorts = defaultdict(list)
    for entry in index['sources']:
        if entry['split'] == 'train':
            cohorts[entry['work_key'], entry['score_source_sha256']].append(entry)
    reference_ids, profiles = [], []
    for ci, (key, entries) in enumerate(sorted(cohorts.items()), 1):
        records = [StyleRecording(load_note_performance(e['performance_cache']), symbolic(e['xml_path'])[0], e['recording_id']) for e in entries]
        reference = fit_reference(records, config)
        ref_path = directory / 'references' / (stable_id(key) + '.json')
        dump(ref_path, reference)
        reference_ids.append({'cohort': key, 'reference_id': reference['id'], 'path': str(ref_path), 'sha256': sha(ref_path)})
        for rec, entry in zip(records, entries):
            deltas = reference_delta(rec.performance, reference)
            for window in json.loads(Path(entry['windows_file']).read_text(encoding='utf-8')):
                if window['tail'] or not window['N']:
                    continue
                p = summarize_window(rec.performance, window, reference, config, deltas=deltas)
                # Keep g input metadata compact; exact note membership remains in hashed window files.
                p['window'] = {k: v for k, v in window.items() if k not in {
                    'score_note_keys', 'row_indices', 'note_weights', 'onset_group_keys', 'beat_keys', 'beat_indices',
                    'measure_visits', 'ordered_measure_occurrence_keys'}}
                p['cohort_reference_id'] = reference['id']
                profiles.append(p)
        del records, reference
        progress('style_profiles', cohorts=ci, total_cohorts=len(cohorts), windows=len(profiles))
    master = artifact('reference-index', config, references=reference_ids, fit_split='train')
    save_derived(master, directory / 'reference_index.json')
    for p in profiles:
        p['reference_id'] = master['id']
        p['profile_id'] = stable_id({k: v for k, v in p.items() if k != 'profile_id'})
    ps = artifact('profiles', config, profiles=profiles, parents={'windows': index['id'], 'reference': master['id']},
                  scope='all full eight-bar train windows; compact identities; cohort references audited separately')
    save_derived(ps, directory / 'profiles.json')
    scaler = fit_profile_scaler(profiles, config, profiles_id=ps['id'])
    save_derived(scaler, directory / 'scaler.json')
    progress('calibration', train_windows=len(profiles))
    calibration = calibrate_distances(profiles, scaler, config, profiles_id=ps['id'])
    save_derived(calibration, directory / 'calibration.json')
    progress('triplet_selection', train_windows=len(profiles))
    triplets = build_triplets(profiles, scaler, calibration, config, profiles_id=ps['id'])
    save_derived(triplets, directory / 'triplets.json')
    progress('style_ready', triplets=len(triplets['triplets']), reasons=triplets['reason_counts'])


def pack_inputs(out, index, progress):
    import torch
    from dataclasses import asdict
    from embedding.note_ae_indexed_data import pack_record, file_sha
    from embedding.note_score_context_data import load_note_score_scale
    from features.note_performance_normalization import load_note_performance_scaler
    from features.style_profile import stable_id
    import shutil
    destination = out / 'packed'
    for kind in ('records', 'scores'):
        (destination / kind).mkdir(parents=True, exist_ok=True)
    for name in ('R_scaler.json', 'S_scale.json'):
        shutil.copyfile(out / 'inputs' / name, destination / name)
    items, excluded, records = [], [], {}
    for i, entry in enumerate(index['sources'], 1):
        result = pack_record((entry, entry['score_cache'], str(destination), str(destination / 'R_scaler.json'), str(destination / 'S_scale.json')))
        items.extend(result['items']); excluded.extend(result['excluded'])
        records[entry['record_key']] = result['record_sha']
        if i % 100 == 0 or i == len(index['sources']):
            progress('pack_inputs', recordings=i, total=len(index['sources']), windows=len(items))
    r = load_note_performance_scaler(destination / 'R_scaler.json')
    s = load_note_score_scale(destination / 'S_scale.json')
    guards = [defaultdict(set) for _ in range(3)]
    for e in index['sources']:
        for guard, key in zip(guards, (e['work_key'], e['score_source_sha256'], e['recording_id'])):
            guard[key].add(e['split'])
    if any(len(splits) > 1 for guard in guards for splits in guard.values()):
        raise ValueError('Work/score/recording leakage across splits')
    manifest = {'format': 'indexed-note-ae-tensors-v1', 'source_index_id': index['id'],
        'source_index': str(out / 'inputs/window_index.json'),
        'source_index_sha256': file_sha(out / 'inputs/window_index.json'),
        'R_scaler': r.fingerprint, 'score_scale': stable_id(asdict(s)),
        'splits': {e['performance_id']: e['split'] for e in index['sources']},
        'items': sorted(items, key=lambda v: v['window']['window_id']), 'excluded': excluded,
        'source_R_fingerprints': {e['performance_id']: e['R_fingerprint'] for e in index['sources']},
        'split_overlap': {'work': 0, 'score': 0, 'recording': 0},
        'tensor_hashes': {'records': records, 'scores': {p.stem: file_sha(p) for p in (destination / 'scores').glob('*.pt')}}}
    manifest['id'] = stable_id(manifest)
    dump(destination / 'manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def progress(stage, **fields):
        value = {'stage': stage, 'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started, **fields}
        dump(out / 'status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    try:
        if args.train_only:
            train(out, progress)
            return
        if not (out / 'inventory.json').exists():
            tasks, rejected = inventory(args.dataset.resolve())
            dump(out / 'inventory.json', {'tasks': tasks, 'not_attempted': rejected,
                'dataset': str(args.dataset.resolve()), 'selected_segments_sha256': sha(args.dataset / 'selected_segments.csv'),
                'split_policy': 'union work/XML/MIDI identities; SHA256 2026:ATEPP:group 70/15/15',
                'scope': 'all selected recordings with approved alignment and local XML; no sampling'})
        source = json.loads((out / 'inventory.json').read_text(encoding='utf-8'))
        done = {r['perf_id']: r for r in log_rows(out / 'extraction.jsonl')}
        pending = [r for r in source['tasks'] if r['perf_id'] not in done]
        progress('extraction', pending=len(pending), total=len(source['tasks']), unavailable=len(source['not_attempted']))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(extract_worker, [(r, str(out)) for r in pending], chunksize=1), 1):
                append(out / 'extraction.jsonl', result)
                done[result['perf_id']] = result
                if i % 10 == 0 or i == len(pending):
                    progress('extraction', processed=len(done), total=len(source['tasks']), statuses=dict(Counter(r['status'] for r in done.values())))
        accepted = sorted((r for r in done.values() if r['status'] == 'extracted'), key=lambda r: r['perf_id'])
        dump(out / 'extraction_summary.json', {'accepted': len(accepted), 'unavailable': source['not_attempted'],
            'rejected': [r for r in done.values() if r['status'] != 'extracted']})
        if not accepted:
            raise ValueError('No real aligned records passed; training not started')
        if not (out / 'inputs/ready.json').exists():
            index = prepare_inputs(out, accepted, progress)
        else:
            index = json.loads((out / 'inputs/window_index.json').read_text(encoding='utf-8'))
        if not (out / 'packed/manifest.json').exists():
            pack_inputs(out, index, progress)
        if not (out / 'style/triplets.json').exists():
            prepare_style(out, index, progress)
        progress('prepared', recordings=len(accepted))
        if not args.prepare_only:
            train(out, progress)
    except BaseException as exc:
        progress('failed', error_type=type(exc).__name__, error=str(exc))
        raise


def train(out, progress):
    import torch
    from dataclasses import asdict
    from embedding.note_ae_indexed_data import IndexedNoteAECorpus
    from embedding.note_ae_data import FixedNoteTriplets
    from embedding.note_ae_training import NoteAETrainingConfig, train_note_ae, load_note_ae_model, evaluate_note_ae
    from embedding.style_profile_data import load_style_artifact
    config = NoteAETrainingConfig(epochs=100, batch_size=8, triplet_batch_size=8,
        style_objective='adaptive_confidence', lambda_style=.1, device='cpu', torch_threads=4,
        early_stopping_patience=10, early_stopping_min_delta=1e-4, bucket_by_length=True)
    dump(out / 'training_config.json', {'training': asdict(config)})
    corpus = IndexedNoteAECorpus(out / 'packed')
    style = out / 'style'
    triples = FixedNoteTriplets(corpus, *(load_style_artifact(style / name) for name in
        ('triplets.json', 'profiles.json', 'scaler.json', 'calibration.json')))
    progress('training', train_windows=len(corpus.dataset('train')), validation_windows=len(corpus.dataset('validation')),
             test_windows=len(corpus.dataset('test')), triplets=len(triples), config=asdict(config))
    training = out / 'training'
    resume = None
    previous = [p for p in out.glob('training*/last.pt') if p.is_file()]
    if previous:
        resume = max(previous, key=lambda p: p.stat().st_mtime)
        i = 1
        while (out / f'training_resume{i:02}').exists():
            i += 1
        training = out / f'training_resume{i:02}'
    _, report = train_note_ae(corpus, training, config=config, triplets=triples, resume=resume)
    choices = list(out.glob('training*/best.pt'))
    if not choices:
        raise ValueError('No validation-selected checkpoint; test evaluation not run')
    selected = min(choices, key=lambda p: torch.load(p, map_location='cpu', weights_only=True)['best_validation'])
    model, _, _, _ = load_note_ae_model(selected, corpus=corpus)
    result = evaluate_note_ae(model, corpus.dataset('test'), corpus.collate, config)
    dump(out / 'test.json', result)
    progress('complete', checkpoint=str(selected), test=result)


if __name__ == '__main__':
    main()
