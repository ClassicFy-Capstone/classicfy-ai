"""Build all accepted eight-bar inputs and train-only scalers in bounded memory."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from embedding.note_performance_data import load_note_performance
from embedding.note_score_context_data import load_note_score_context, save_note_score_scale
from embedding.style_profile_data import save_style_artifact
from features.note_performance_normalization import NotePerformanceScaler, fit_note_performance_scaler, save_note_performance_scaler
from features.note_score_context import fit_note_score_scale
from features.style_profile import StyleConfig, StyleRecording, artifact, make_windows
from preprocessing.musicxml_score import read_musicxml


@lru_cache(maxsize=4)
def xml_score(path):
    return read_musicxml(path)


def prepare(preprocessing, out):
    out.mkdir(parents=True, exist_ok=False)
    results = {r['perf_id']: r for r in (json.loads(line) for line in (preprocessing / 'alignments.jsonl').read_text().splitlines())}
    accepted = sorted((r for r in results.values() if r['status'] == 'extracted'), key=lambda r: r['perf_id'])
    if not accepted:
        raise ValueError('No accepted aligned recordings')
    scores = {r['source_sha256']: r for r in (json.loads(line) for line in (preprocessing / 'scores.jsonl').read_text().splitlines()) if r['status'] == 'prepared'}
    sources, items, excluded, training, score_paths = [], [], [], [], set()
    count, mean, m2 = np.zeros(11, np.int64), np.zeros(11), np.zeros(11)
    split_guards = [dict(), dict(), dict()]
    for i, result in enumerate(accepted, 1):
        r = load_note_performance(result['performance_cache'])
        xml = xml_score(result['xml_path'])
        recording = r.provenance['performance']['midi_sha256']
        for mapping, key in zip(split_guards, (r.work_key, r.score_source_sha256, recording)):
            if key in mapping and mapping[key] != r.split:
                raise ValueError('Work, XML or recording crosses splits')
            mapping[key] = r.split
        windows = make_windows(StyleRecording(r, xml, recording))
        path = out / 'windows' / (r.record_key + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(windows, ensure_ascii=False, separators=(',', ':')))
        score_cache = scores[r.score_source_sha256]['score_cache']
        pid = r.dataset + ':' + r.record_key
        sources.append(dict(performance_id=pid, record_key=r.record_key, work_key=r.work_key, split=r.split,
                            recording_id=recording, performance_cache=result['performance_cache'],
                            xml_path=result['xml_path'], score_source_sha256=r.score_source_sha256,
                            score_cache=score_cache, R_fingerprint=r.fingerprint, windows_file=str(path),
                            windows_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for local, w in enumerate(windows):
            reason = 'tail' if w['tail'] else 'empty_window' if not w['N'] else 'no_observed_reconstruction' if not r.mask[w['row_indices']].any() else None
            if reason:
                excluded.append(dict(window_id=w['window_id'], split=w['split'], reason=reason))
            else:
                identity = {k: w[k] for k in ('window_id', 'performance_id', 'split', 'N', 'q_start', 'q_end')}
                items.append(dict(window=identity, local_index=local))
        if r.split == 'train':
            local = fit_note_performance_scaler([r], fit_manifest_id='streaming-single-record')
            n, mu = np.asarray(local.valid_counts), np.asarray(local.mean)
            variance = np.array([sd * sd if status == 'fitted' else 0. for sd, status in zip(local.scale, local.channel_status)])
            total, delta, valid = count + n, mu - mean, n > 0
            m2[valid] += variance[valid] * n[valid] + delta[valid] ** 2 * count[valid] * n[valid] / total[valid]
            mean[valid] += delta[valid] * n[valid] / total[valid]
            count = total
            training.extend(local.training_records)
            score_paths.add(score_cache)
        if i % 100 == 0 or i == len(accepted):
            print(f'AE index: {i}/{len(accepted)} recordings, {len(items)} full observed windows', flush=True)
    items.sort(key=lambda item: (item['window']['performance_id'], item['window']['q_start'], item['window']['window_id']))
    index = artifact('ae-training-index', StyleConfig(), sources=sources, items=items, excluded=excluded,
                     parents={'alignment_summary_sha256': hashlib.sha256((preprocessing / 'alignment_summary.json').read_bytes()).hexdigest()},
                     split_policy=json.loads((preprocessing / 'splits.json').read_text()),
                     scope='All accepted full observed windows; no g/profile selection, no downsampling')
    save_style_artifact(index, out / 'index.json')
    sd = np.sqrt(np.divide(m2, count, out=np.zeros(11), where=count > 0))
    status = tuple('unobserved' if n == 0 else 'constant' if s == 0 else 'fitted' for n, s in zip(count, sd))
    sd[sd == 0] = 1
    scaler = NotePerformanceScaler(tuple(map(float, mean)), tuple(map(float, sd)), tuple(map(int, count)),
                                   tuple(bool(n) for n in count), status, index['id'],
                                   tuple(sorted(training, key=lambda t: (t['dataset'], t['record_key']))))
    save_note_performance_scaler(scaler, out / 'R_scaler.json')
    score_scale = fit_note_score_scale(load_note_score_context(path)[0] for path in sorted(score_paths))
    save_note_score_scale(score_scale, out / 'S_scale.json')
    summary = {'recordings': len(sources), 'full_windows': len(items), 'excluded': len(excluded),
               'splits': {split: sum(w['window']['split'] == split for w in items) for split in ('train', 'validation', 'test')},
               'train_scaler_counts': count.tolist(), 'index_id': index['id']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preprocessing', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.preprocessing.resolve(), args.out.resolve())
