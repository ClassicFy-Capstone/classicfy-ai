"""Prepare every selected clip for the MusicXML-conditioned note AE."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def bridge_worker(task):
    from preprocessing.selected_score_bridge import build_selected_score_bridge
    row, directory, signature = task
    directory = Path(directory)
    key = hashlib.sha256((row['xml_sha256'] + sha(row['local_score'])).encode()).hexdigest()
    report = directory / (key + '.json')
    if report.exists():
        old = json.loads(report.read_text(encoding='utf-8'))
        if old.get('signature') == signature and old.get('status') == 'passed':
            return old
    start = time.time()
    result = dict(key=key, signature=signature, xml_path=row['source_xml_path'],
                  score_midi_path=row['local_score'], composition_id=row['composition_id'])
    try:
        xml, features, bridge = build_selected_score_bridge(Path(row['source_xml_path']), Path(row['local_score']))
        target = directory / (key + '.pickle')
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            pickle.dump((xml, features, bridge), stream, protocol=5)
        temporary.replace(target)
        result.update(status='passed', cache_path=str(target), cache_sha256=sha(target),
                      notes=len(xml.notes), midi_coverage=bridge['midi_coverage'],
                      xml_coverage=bridge['xml_coverage'], score=bridge['score'],
                      candidate=bridge['selected_candidate'],
                      time_repairs=len(bridge['time_repair']),
                      navigation_repair=bool(bridge['navigation_repair']),
                      direction_repairs=len(bridge['direction_repair']))
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    result['seconds'] = time.time() - start
    dump(report, result)
    return result


def clip_worker(task):
    from preprocessing.selected_note_clip import prepare_clip
    row, directory, bridge_report, source_path, alignment_path, signature = task
    directory = Path(directory)
    report = directory / (row['perf_id'] + '.json')
    if report.exists():
        old = json.loads(report.read_text(encoding='utf-8'))
        if old.get('signature') == signature and old.get('status') == 'passed':
            return old
    started = time.time()
    result = dict(perf_id=row['perf_id'], composition_id=row['composition_id'], signature=signature)
    try:
        if not source_path or not Path(source_path).is_file():
            raise ValueError('Selected original performance MIDI was not found')
        if bridge_report.get('status') != 'passed':
            raise ValueError('Score bridge unavailable: '+bridge_report.get('error', 'not prepared'))
        if sha(bridge_report['cache_path']) != bridge_report['cache_sha256']:
            raise ValueError('Score bridge artifact hash mismatch')
        arrays, details = prepare_clip(row, bridge_report['cache_path'], source_path, alignment_path)
        target = directory / (row['perf_id'] + '.npz')
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(target)
        result.update(details, status='passed', cache_path=str(target), cache_sha256=sha(target))
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    result['seconds'] = time.time() - started
    dump(report, result)
    return result


def prepare_clips(args, xml_rows):
    base = args.workspace / 'outputs/atepp_candidate_style_selected_recovered_final'
    with (base / 'selected_segments.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if args.limit:
        rows = rows[:args.limit]
    xml_by_perf = {r['perf_id']:r for r in xml_rows}
    paths = {}
    for path in (args.workspace / 'ATEPP_dataset/ATEPP-1.2').rglob('*.mid'):
        if path.stem.split('-')[0].isdigit():
            paths.setdefault(path.stem, []).append(path)
    directory = args.out / 'clips'
    directory.mkdir(exist_ok=True)
    dependencies = [SRC/p for p in ('preprocessing/selected_note_clip.py',
        'preprocessing/note_performance_adapters.py', 'preprocessing/performance_midi.py',
        'features/note_performance.py', 'features/note_score_context.py', 'features/performance_time.py')]
    signature = hashlib.sha256(''.join(sha(p) for p in dependencies).encode()).hexdigest()
    tasks = []
    score_hashes, bridge_reports = {}, {}
    for row in rows:
        matches = paths.get(row['perf_id'], [])
        # The worker checks bytes against the selected source SHA before use.
        source = str(matches[0]) if len(matches)==1 else next((str(p) for p in matches if sha(p)==row['source_sha256']), '')
        xml_row = xml_by_perf[row['perf_id']]
        if row['local_score'] not in score_hashes:score_hashes[row['local_score']]=sha(row['local_score'])
        key = hashlib.sha256((xml_row['xml_sha256'] + score_hashes[row['local_score']]).encode()).hexdigest()
        bridge_file = args.out/'score_bridges'/(key+'.json')
        if key not in bridge_reports:
            bridge_reports[key]=json.loads(bridge_file.read_text(encoding='utf-8')) if bridge_file.exists() else {}
        bridge = bridge_reports[key]
        alignment = row['local_alignment_cache'] or row['alignment_cache']
        if row['local_alignment_cache']:
            alignment = str(base/'local_alignment_cache'/Path(alignment).name)
        task_signature = hashlib.sha256((signature+row['source_sha256']+bridge.get('cache_sha256','')
             + (sha(alignment) if alignment and Path(alignment).is_file() else '')
             +row['start_seconds']+row['end_seconds']).encode()).hexdigest()
        tasks.append((row, str(directory), bridge, source, alignment, task_signature))
        if len(tasks)%1000==0:print(f'Indexed {len(tasks)}/{len(rows)} clip sources',flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(clip_worker, task) for task in tasks]
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            if result['status']=='failed' or len(results)%50==0 or args.limit:
                print(f"{len(results)}/{len(jobs)} {result['status']} perf={result['perf_id']} "
                      f"notes={result.get('notes',0)} mapped={result.get('xml_mapping_fraction',0):.3f} "
                      f"{result.get('error','')}", flush=True)
    dump(args.out/'clip_preparation_summary.json', dict(signature=signature, results=results,
        passed=sum(r['status']=='passed' for r in results), failed=sum(r['status']=='failed' for r in results)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--stage', choices=('bridges', 'clips'), default='bridges')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    directory = args.out / 'score_bridges'
    directory.mkdir(exist_ok=True)
    manifest = args.workspace / 'outputs/selected_musicxml_discovery_20261113/run02/selected_musicxml_manifest.csv'
    with manifest.open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if args.stage == 'clips':
        prepare_clips(args, rows)
        return
    sources = [SRC / name for name in ('preprocessing/selected_score_bridge.py',
        'preprocessing/musicxml_score.py', 'preprocessing/musicxml_time_repair.py',
        'preprocessing/musicxml_navigation_repair.py', 'features/note_score_context.py')]
    signature = hashlib.sha256(''.join(sha(p) for p in sources).encode()).hexdigest()
    unique = list({(r['xml_sha256'], r['local_score']):r for r in rows}.values())
    if args.limit:
        unique = unique[:args.limit]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(bridge_worker, (row, str(directory), signature)) for row in unique]
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            print(f"{len(results)}/{len(jobs)} {result['status']} work={result['composition_id']} "
                  f"coverage={result.get('midi_coverage', 0):.3f} {result.get('error', '')}", flush=True)
    dump(args.out / 'score_bridge_summary.json', dict(signature=signature, results=results,
         passed=sum(r['status']=='passed' for r in results), failed=sum(r['status']=='failed' for r in results)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
