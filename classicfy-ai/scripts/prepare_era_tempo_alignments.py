"""Audit all selected refine recordings and obtain score-quarter tempo maps.

Reuses verified exact-input DualDTW results; aligns previously quality-excluded
recordings without excluding quality labels. Does not alter existing caches.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT.parent
AI = PROJECT / 'classicfy-ai'
sys.path.insert(0, str(AI / 'src'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def align_one(task):
    row, canonical, out = task
    import numpy as np
    from preprocessing.performance_midi import load_performance_midi
    from preprocessing.midi_loader import MidiData, NoteEvent
    from preprocessing.score_midi import load_score_midi
    from preprocessing.research_alignment import align_research
    result = dict(row)
    try:
        performance = load_performance_midi(row['midi_file'])
        usable = [n for n in performance.notes if n.key_offset_sec is not None and n.key_offset_sec > n.onset_sec]
        midi = MidiData([NoteEvent(n.pitch, n.onset_sec, n.key_offset_sec, n.velocity, 0, n.program)
                         for n in usable], [], performance.duration_sec)
        score = load_score_midi(canonical)
        cells = len(score) * len(usable)
        result['note_product'] = cells
        if cells > 500_000_000:
            return dict(result, status='resource_limit', reason='note_product_over_500million')
        alignment, q, seconds, metrics = align_research(score, midi, 'dual_dtw')
        result['alignment_metrics'] = metrics
        if not metrics['accepted']:
            return dict(result, status='alignment_rejected', reason=metrics['reject_reasons'])
        dest = Path(out) / 'alignments' / (row['perf_id'] + '.npz')
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest, score_beats=q, performance_beats=seconds,
            score_onsets=np.array([s.onset_beats for s, p in alignment.matches]),
            performance_onsets=np.array([p.onset for s, p in alignment.matches]),
            pitch=np.array([s.pitch for s, p in alignment.matches]))
        return dict(result, status='accepted', origin='new_dual_dtw', alignment_file=str(dest),
                    alignment_sha256=sha(dest), reason='')
    except Exception as exc:
        return dict(result, status='technical_error', reason=f'{type(exc).__name__}: {exc}')


def prepare(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    for name in ['matplotlib-cache', 'numba-cache']:
        (out / name).mkdir(exist_ok=True)
    for key, value in {'MPLCONFIGDIR': out / 'matplotlib-cache', 'NUMBA_CACHE_DIR': out / 'numba-cache',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CLASSICFY_TORCH_THREADS': '1'}.items():
        os.environ[key] = str(value)
    import numpy as np
    refine = WORKSPACE / 'datasets/ATEPP_refine'
    cohort = AI / 'analysis/era_refine_balanced_100'
    previous = WORKSPACE / 'diagnostics/atepp_note_training/run01/preprocessing'
    selected = csv_rows(cohort / 'selected_recordings.csv')
    checks = {r['refined_file']: r for r in csv_rows(refine / 'file_checksums.csv')}
    old = {r['perf_id']: r for r in jsonl(previous / 'alignments.jsonl')}
    raw = {r['perf_id']: r for r in jsonl(previous / 'raw.jsonl')}
    scores = {r['source_sha256']: r for r in jsonl(previous / 'scores.jsonl') if r['status'] == 'prepared'}
    # Reuse canonical scores and accepted caches only under the same method implementation.
    frozen = WORKSPACE / 'diagnostics/atepp_note_training/run01/source/src'
    relevant = ['preprocessing/research_alignment.py', 'features/articulation.py',
                'preprocessing/score_midi.py', 'preprocessing/performance_midi.py']
    for name in relevant:
        assert sha(AI / 'src' / name) == sha(frozen / name), name
    sources = [cohort / 'selected_recordings.csv', cohort / 'selected_works.csv',
               refine / 'ATEPP-metadata-1.2.csv', refine / 'refinement_manifest.json',
               previous / 'alignments.jsonl', previous / 'scores.jsonl', previous / 'raw.jsonl']
    source_hashes = {str(p): sha(p) for p in sources}
    done_path = out / 'recordings.jsonl'
    prior = {r['perf_id']: r for r in jsonl(done_path)}
    tasks = []
    with done_path.open('a', encoding='utf-8') as stream:
        def save(result):
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
        for r in selected:
            midi_file = refine / r['local_midi_file']
            midi_sha = sha(midi_file)
            assert midi_sha == checks[r['local_midi_file']]['sha256'] == raw[r['perf_id']]['sha256']
            row = {'perf_id': r['perf_id'], 'composition_id': r['composition_id'], 'era': r['era'],
                   'composer': r['composer'], 'artist': r['artist'], 'quality': r['quality'],
                   'midi_file': str(midi_file), 'midi_sha256': midi_sha,
                   'previous_status': old[r['perf_id']]['status']}
            if r['perf_id'] in prior:
                prior_row = prior[r['perf_id']]
                assert prior_row['midi_sha256'] == midi_sha and prior_row['composition_id'] == r['composition_id']
                if prior_row['status'] == 'accepted':
                    assert sha(prior_row['alignment_file']) == prior_row['alignment_sha256']
                continue
            if not r['local_score_file']:
                save(dict(row, status='no_score', reason='official_score_not_provided'))
                continue
            xml_sha = sha(refine / r['local_score_file'])
            row['score_xml_sha256'] = xml_sha
            if xml_sha not in scores:
                save(dict(row, status='unsupported_score', reason='XML_score_cannot_be_unfolded_by_current_reader'))
                continue
            source = scores[xml_sha]
            canonical = Path(source['score_midi'])
            bridge = json.loads(Path(source['bridge']).read_text())
            assert bridge['xml_sha256'] == xml_sha and bridge['score_coordinate_unit'] == 'quarter'
            assert sha(canonical) == bridge['score_midi_sha256']
            row['canonical_score_midi'] = str(canonical)
            row['canonical_score_sha256'] = sha(canonical)
            row['traversal_id'] = bridge['traversal_id']
            previous_row = old[r['perf_id']]
            if previous_row['status'] == 'extracted':
                path = previous / 'performances' / r['perf_id'] / 'alignment.npz'
                perf_bridge = json.loads((path.parent / 'bridge.json').read_text())
                assert perf_bridge['xml_sha256'] == xml_sha
                assert perf_bridge['traversal_id'] == row['traversal_id']
                assert perf_bridge['alignment_accepted']
                with np.load(path, allow_pickle=False) as cache:
                    assert cache['score_beats'].shape == cache['performance_beats'].shape
                    assert np.allclose(np.diff(cache['score_beats']), 1.)
                    assert np.all(np.diff(cache['performance_beats']) > 0)
                save(dict(row, status='accepted', origin='verified_existing_dual_dtw',
                          alignment_metrics=previous_row['alignment_metrics'], alignment_file=str(path),
                          alignment_sha256=sha(path), reason=''))
            elif previous_row['status'] == 'alignment_rejected':
                save(dict(row, status='alignment_rejected', reason=previous_row.get('error', 'existing_quality_gate_failed'),
                          alignment_metrics=previous_row.get('alignment_metrics', {})))
            else:
                tasks.append((row, str(canonical), str(out)))
        print(f'New DualDTW tasks: {len(tasks)}; existing metadata audit written', flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(align_one, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                save(result)
                if i % 10 == 0 or i == len(tasks):
                    print(f'New alignments: {i}/{len(tasks)}; last={result["status"]}', flush=True)
    records = {r['perf_id']: r for r in jsonl(done_path)}
    assert set(records) == {r['perf_id'] for r in selected}
    assert all(sha(Path(p)) == digest for p, digest in source_hashes.items())
    protocol = {'selected_recordings': len(selected), 'statuses': dict(Counter(r['status'] for r in records.values())),
                'accepted_origins': dict(Counter(r.get('origin') for r in records.values() if r['status'] == 'accepted')),
                'quality_label_exclusion': False, 'source_sha256': source_hashes,
                'method_sources_sha256': {name: sha(AI / 'src' / name) for name in relevant},
                'script_sha256': sha(Path(__file__)),
                'beat_unit': 'score quarter note', 'normalization': 'none',
                'status_note': 'Automatic score-aligned estimates; no fabricated MIDI-header BPM or score-free fallback.'}
    (out / 'alignment_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: protocol[k] for k in ['selected_recordings', 'statuses', 'accepted_origins']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=WORKSPACE / 'diagnostics/era_tempo_refine')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    prepare(args.out.resolve(), args.workers)
