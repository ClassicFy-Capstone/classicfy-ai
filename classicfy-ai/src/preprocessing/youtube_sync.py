"""Resumable ATEPP-to-YouTube alignment primitives.

Scientific dependencies are imported only inside the worker. Inventory and resume
checks use the standard library. Automatic timestamps are never human ground truth.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

VERSION = 1
FS = 22050
RATE = 50
ELIGIBLE = {'exported', 'exported_review'}
TERMINAL_SUCCESS = {'completed'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def video_id(url):
    parsed = urlparse(str(url).strip())
    host = (parsed.hostname or '').lower()
    if host == 'youtu.be':
        value = parsed.path.strip('/').split('/')[0]
    elif host in {'youtube.com', 'www.youtube.com', 'music.youtube.com', 'm.youtube.com'}:
        value = parse_qs(parsed.query).get('v', [''])[0]
        if not value and parsed.path.startswith(('/embed/', '/shorts/')):
            value = parsed.path.split('/')[2]
    else:
        value = ''
    if parsed.scheme not in {'http', 'https'} or not re.fullmatch(r'[A-Za-z0-9_-]{11}', value):
        raise ValueError('missing_or_invalid_youtube_url')
    return value


def valid_interval(row):
    try:
        start, end = float(row['start_seconds']), float(row['end_seconds'])
        return math.isfinite(start) and math.isfinite(end) and 0 <= start < end
    except (KeyError, TypeError, ValueError):
        return False


def candidate_reason(row, include_excluded=False):
    if not valid_interval(row):
        return 'invalid_interval'
    if row.get('status') not in ELIGIBLE and not (include_excluded and row.get('status') == 'excluded_duration'):
        return 'excluded_input_status'
    return ''


def load_inventory(source, midi_root=None, include_excluded=False):
    source = Path(source).resolve()
    with (source / 'segments.csv').open(encoding='utf-8-sig', newline='') as stream:
        segments = list(csv.DictReader(stream))
    with (source / 'input_midi_audit.csv').open(encoding='utf-8-sig', newline='') as stream:
        audit = list(csv.DictReader(stream))
    metadata = {}
    for row in audit:
        pid = row['perf_id']
        if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', pid) or pid in metadata:
            raise ValueError(f'duplicate_or_unsafe_perf_id:{pid}')
        metadata[pid] = row
    seen, groups = set(), defaultdict(list)
    for row in segments:
        key = (row['perf_id'], row['candidate_number'])
        if key in seen:
            raise ValueError(f'duplicate_candidate:{key}')
        seen.add(key)
        if row['perf_id'] not in metadata:
            raise ValueError(f'candidate_missing_audit_metadata:{row["perf_id"]}')
        if not candidate_reason(row, include_excluded):
            groups[row['perf_id']].append(row)
    # Original absolute audit paths work here. A supplied MIDI root also supports relocation.
    index = None
    for pid in groups:
        row = metadata[pid]
        local = Path(row.get('local_midi') or '__missing__')
        if midi_root:
            relative = Path(midi_root) / row.get('midi_path', '')
            if relative.is_file():
                local = relative
            elif not local.is_file():
                if index is None:
                    index = defaultdict(list)
                    for path in Path(midi_root).rglob('*.mid'):
                        index[path.stem].append(path)
                found = index.get(pid, [])
                if len(found) == 1:
                    local = found[0]
        row['resolved_midi'] = str(local.resolve())
    return source, segments, metadata, dict(groups)


def configuration(options):
    """Only output-affecting settings invalidate completed work, not worker count."""
    return dict(version=VERSION, feature_rate=RATE, sample_rate=FS,
                engine_sha256=sha(__file__), save_clips=bool(options.get('save_clips')),
                include_excluded=bool(options.get('include_excluded')),
                tuning_max_seconds=60, step_weights=[1.5, 1.5, 2.0], threshold_rec=1000000,
                method='quantized_chroma_DLNCO_MrMsDTW_plus_chroma_crosscheck')


def request_fingerprint(row, segments, config):
    # Include path stat to detect local file changes before skipping completed work.
    path = Path(row['resolved_midi'])
    stat = path.stat() if path.is_file() else None
    return digest(dict(metadata=row, candidates=segments, config=config,
                       midi_stat=[stat.st_size, stat.st_mtime_ns] if stat else None))


def reusable_result(path, fingerprint, save_clips=False, output=None):
    try:
        result = read_json(path)
        if result.get('fingerprint') != fingerprint or result.get('status') != 'completed':
            return False
        if not result.get('rows') or not (Path(path).parent / 'mapping.npz').is_file():
            return False
        if sha(Path(path).parent / 'mapping.npz') != result.get('mapping_sha256'):
            return False
        if save_clips:
            for row in result['rows']:
                relative = row.get('youtube_clip_relative_path')
                if relative and (not (Path(output)/relative).is_file() or
                                 sha(Path(output)/relative) != row['youtube_clip_sha256']):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def empty_match(row, status, error='', url=''):
    return dict(row, youtube_url=url, youtube_match_status=status, youtube_error=error,
                youtube_start_seconds='', youtube_end_seconds='', youtube_duration_seconds='',
                youtube_start_link='', youtube_embed_url='', youtube_review_reasons='',
                youtube_chroma_similarity='', youtube_shifted_control_margin='',
                youtube_start_method_difference_seconds='', youtube_end_method_difference_seconds='',
                youtube_clip_relative_path='', youtube_clip_sha256='',
                youtube_human_verified=False, youtube_exact_accuracy_measured=False)


def classify_error(message):
    text = message.lower()
    if any(s in text for s in ['429', 'too many requests', 'sign in to confirm', 'not a bot', 'rate-limit']):
        return 'access_blocked'
    if any(s in text for s in ['video unavailable', 'private video', 'has been removed',
                               'not available', 'copyright', 'members-only']):
        return 'video_unavailable'
    if 'timed out' in text or 'timeout' in text:
        return 'timeout'
    if 'missing_or_invalid_youtube_url' in text:
        return 'missing_youtube_url'
    if 'hash_mismatch' in text:
        return 'source_mismatch'
    if 'audio_not_cached' in text:
        return 'audio_not_cached'
    return 'failed'


def acquire_audio(job, work):
    """Return a full, untrimmed mono WAV, preserving video time origin at zero."""
    import soundfile as sf
    import imageio_ffmpeg
    options, row = job['options'], job['metadata']
    vid = video_id(row['youtube_links'])
    pid = row['perf_id']
    caches = []
    if options.get('audio_cache'):
        cache = Path(options['audio_cache'])
        caches += [cache/pid, cache/'audio'/pid]
    caches.append(work)
    for cache in caches:
        receipt_path = cache/'receipt.json'
        wav = cache/'source.wav'
        if not receipt_path.is_file() or not wav.is_file():
            continue
        receipt = read_json(receipt_path)
        if receipt.get('video_id') != vid or receipt.get('timeline_origin_seconds') != 0:
            continue
        if sha(wav) != receipt.get('wav_sha256'):
            raise ValueError('audio_cache_hash_mismatch')
        info = sf.info(wav)
        if info.samplerate != FS or info.channels != 1:
            raise ValueError('audio_cache_requires_22050Hz_mono')
        return wav, dict(receipt, audio_source='local_cache', download_seconds=0., decode_seconds=0.)
    if options.get('offline'):
        raise FileNotFoundError('audio_not_cached')
    timeout = int(options.get('download_timeout', 600))
    args = [sys.executable, '-m', 'yt_dlp', '--ignore-config', '--no-playlist', '--no-progress',
            '--no-warnings', '--socket-timeout', '30', '--retries', '1', '--extractor-retries', '1',
            '--js-runtimes', 'node', '-f', 'bestaudio/best', '--write-info-json',
            '--max-filesize', '500M', '-o', str(work/'source.%(ext)s'),
            f'https://www.youtube.com/watch?v={vid}']
    started = time.perf_counter()
    with (work/'download.log').open('w', encoding='utf-8') as stream:
        proc = subprocess.run(args, stdout=stream, stderr=stream, timeout=timeout)
    if proc.returncode:
        error = (work/'download.log').read_text(encoding='utf-8', errors='replace')[-5000:]
        raise RuntimeError(error)
    download_seconds = time.perf_counter()-started
    info = read_json(work/'source.info.json')
    if info.get('id') != vid:
        raise ValueError('downloaded_video_id_mismatch')
    candidates = [p for p in work.glob('source.*') if p.suffix in {'.webm','.m4a','.opus','.mp3','.mp4','.ogg'}]
    if len(candidates) != 1:
        raise ValueError('missing_or_ambiguous_downloaded_audio')
    media = candidates[0]
    wav = work/'source.wav'
    started = time.perf_counter()
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(media), '-vn', '-ac', '1', '-ar', str(FS), '-c:a', 'pcm_s16le', str(wav)],
                   check=True, timeout=timeout)
    receipt = dict(video_id=vid, url=f'https://www.youtube.com/watch?v={vid}', title=info.get('title'),
                   uploader=info.get('uploader'), timeline_origin_seconds=0,
                   fetched_at_utc=datetime.now(timezone.utc).isoformat(),
                   wav_sha256=sha(wav), encoded_sha256=sha(media), download_bytes=media.stat().st_size,
                   audio_source='download', download_seconds=download_seconds,
                   decode_seconds=time.perf_counter()-started)
    atomic_json(work/'receipt.json', receipt)
    return wav, receipt


def path_mapping(wp, rate=RATE):
    """O(path length) median aggregation of monotone vertical DTW runs."""
    import numpy as np
    wp = np.asarray(wp)
    if wp.ndim != 2 or wp.shape[0] != 2 or not wp.shape[1] or np.any(np.diff(wp, axis=1) < 0):
        raise ValueError('invalid_or_nonmonotonic_DTW_path')
    starts = np.r_[0, np.flatnonzero(np.diff(wp[1]))+1]
    counts = np.diff(np.r_[starts, wp.shape[1]])
    left = starts+(counts-1)//2
    right = starts+counts//2
    return wp[1, starts]/rate, (wp[0,left]+wp[0,right])/(2*rate)


def align_audio(job, wav, receipt, work):
    import numpy as np
    import pandas as pd
    import pretty_midi
    import soundfile as sf
    from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
    from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
    from synctoolbox.feature.csv_tools import df_to_pitch_features, df_to_pitch_onset_features
    from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
    from synctoolbox.feature.pitch import audio_to_pitch_features
    from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
    from synctoolbox.feature.utils import estimate_tuning
    started = time.perf_counter()
    row = job['metadata']
    source = Path(row['resolved_midi'])
    if sha(source) != row['sha256']:
        raise ValueError('original_midi_hash_mismatch')
    if any(s.get('source_sha256') and s['source_sha256'] != row['sha256'] for s in job['segments']):
        raise ValueError('candidate_audit_source_hash_mismatch')
    midi = pretty_midi.PrettyMIDI(str(source))
    notes = [dict(start=n.start, duration=n.end-n.start, pitch=n.pitch, velocity=n.velocity, instrument='piano')
             for instrument in midi.instruments if not instrument.is_drum for n in instrument.notes if n.end>n.start]
    if not notes:
        raise ValueError('no_valid_midi_notes')
    df = pd.DataFrame(notes).sort_values('start')
    audio, fs = sf.read(wav, dtype='float64')
    if fs != FS or audio.ndim != 1 or len(audio) < FS:
        raise ValueError('invalid_audio_format')
    if len(audio)/FS > 7200:
        raise ValueError('audio_over_2_hours_requires_manual_review')
    tuning = int(estimate_tuning(audio[:min(len(audio),60*FS)], Fs=FS))
    ac = quantize_chroma(pitch_to_chroma(audio_to_pitch_features(audio, Fs=FS,
                        tuning_offset=tuning, feature_rate=RATE, verbose=False)))
    ap = audio_to_pitch_onset_features(audio, Fs=FS, tuning_offset=tuning, verbose=False)
    ao = pitch_onset_features_to_DLNCO(ap, feature_rate=RATE, feature_sequence_length=ac.shape[1], visualize=False)
    mc = quantize_chroma(pitch_to_chroma(df_to_pitch_features(df.copy(), feature_rate=RATE)))
    mo = pitch_onset_features_to_DLNCO(df_to_pitch_onset_features(df), feature_rate=RATE,
                                    feature_sequence_length=mc.shape[1], visualize=False)
    feature_seconds = time.perf_counter()-started
    started = time.perf_counter()
    # Duration mismatch or shared movement video: locate the MIDI inside the full
    # audio first. Cropping FEATURES retains the corresponding video time offset.
    audio_offset_frames = 0
    length_ratio = ac.shape[1]/mc.shape[1]
    subsequence = bool(job.get('shared_video') or length_ratio > 1.25)
    if subsequence and ac.shape[1] > mc.shape[1]*1.05:
        import librosa
        from scipy.ndimage import uniform_filter1d
        stride = 25  # 2 Hz coarse location, followed by full 50 Hz MrMsDTW.
        aa = uniform_filter1d(ac, size=stride, axis=1)[:,::stride]
        mm = uniform_filter1d(mc, size=stride, axis=1)[:,::stride]
        # Cosine distance is undefined for silent/zero vectors.
        for values in (aa,mm):
            silent=np.linalg.norm(values,axis=0)<1e-12
            values[:,silent]=1/np.sqrt(12)
        _, coarse = librosa.sequence.dtw(X=mm, Y=aa, metric='cosine', subseq=True, backtrack=True)
        lo = max(0, int(coarse[:,1].min()*stride)-2*RATE)
        hi = min(ac.shape[1], int((coarse[:,1].max()+1)*stride)+2*RATE)
        audio_offset_frames = lo
        ac_local, ao_local = ac[:,lo:hi], ao[:,lo:hi]
    else:
        ac_local, ao_local = ac, ao
    options = dict(f_chroma1=ac_local, f_chroma2=mc, input_feature_rate=RATE,
                   step_weights=np.array([1.5,1.5,2.0]), threshold_rec=1000000, verbose=False)
    wp = sync_via_mrmsdtw(**options, f_onset1=ao_local, f_onset2=mo)
    cp = sync_via_mrmsdtw(**options)
    wp[0] += audio_offset_frames
    cp[0] += audio_offset_frames
    mx, my = path_mapping(wp)
    cx, cy = path_mapping(cp)
    alignment_seconds = time.perf_counter()-started
    started = time.perf_counter()
    ai = np.clip(np.rint(my*RATE).astype(int),0,ac.shape[1]-1)
    mi = np.clip(np.rint(mx*RATE).astype(int),0,mc.shape[1]-1)
    def cos(a,b):
        return np.sum(a*b,axis=0)/np.maximum(np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0),1e-12)
    similarity = cos(ac[:,ai],mc[:,mi])
    control = np.median([cos(ac[:,(ai+int(t*RATE))%ac.shape[1]],mc[:,mi]) for t in [10,30,60]],axis=0)
    active = np.std(mc[:,mi],axis=0)>.05
    np.savez_compressed(work/'mapping.npz', midi_seconds=mx, youtube_seconds=my,
                        crosscheck_midi_seconds=cx, crosscheck_youtube_seconds=cy)
    vid = video_id(row['youtube_links'])
    url = f'https://www.youtube.com/watch?v={vid}'
    result_rows = []
    for original in job['segments']:
        result = empty_match(original, 'review_required', url=url)
        start, end = float(original['start_seconds']),float(original['end_seconds'])
        # No fabricated/clamped endpoints when the input interval is outside alignment coverage.
        if not mx[0] <= start < end <= mx[-1]:
            result.update(youtube_match_status='boundary_out_of_range',
                          youtube_error='original_candidate_outside_MIDI_feature_coverage')
            result_rows.append(result)
            continue
        vs,ve = np.interp([start,end],mx,my)
        if not 0 <= vs < ve <= len(audio)/FS:
            result.update(youtube_match_status='invalid_mapped_interval',youtube_error='nonpositive_or_outside_video')
            result_rows.append(result)
            continue
        other = np.interp([start,end],cx,cy)
        difference = np.abs([vs-other[0],ve-other[1]])
        region = active & (mx>=start) & (mx<=end)
        score = float(np.median(similarity[region])) if region.any() else None
        margin = float(np.median(similarity[region]-control[region])) if region.any() else None
        reasons = []
        if score is None or score<.75: reasons.append('low_chroma_similarity')
        if margin is None or margin<.08: reasons.append('weak_shifted_control_margin')
        if difference.max()>.3: reasons.append('boundary_methods_disagree_over_0.3s')
        if not .95<=(ve-vs)/(end-start)<=1.05: reasons.append('duration_ratio_outside_0.95_1.05')
        if original.get('alignment_accepted')!='True': reasons.append('upstream_MIDI_mapping_uncertain')
        if original.get('status')=='exported_review': reasons.append('upstream_review_candidate')
        if original.get('source_quality') not in {'','high quality',None}: reasons.append('source_quality_warning')
        if job.get('shared_video'): reasons.append('shared_video_contains_multiple_MIDI_sources')
        if subsequence: reasons.append('subsequence_location_requires_review')
        if length_ratio<.8: reasons.append('video_much_shorter_than_MIDI')
        result.update(youtube_match_status='review_required' if reasons else 'matched_auto',
            youtube_start_seconds=round(float(vs),4), youtube_end_seconds=round(float(ve),4),
            youtube_duration_seconds=round(float(ve-vs),4), youtube_review_reasons=';'.join(reasons),
            youtube_chroma_similarity=round(score,4) if score is not None else '',
            youtube_shifted_control_margin=round(margin,4) if margin is not None else '',
            youtube_start_method_difference_seconds=round(float(difference[0]),4),
            youtube_end_method_difference_seconds=round(float(difference[1]),4),
            youtube_start_link=url+f'&t={math.floor(vs)}s',
            youtube_embed_url=f'https://www.youtube.com/embed/{vid}?start={math.floor(vs)}&end={math.ceil(ve)}')
        if job['options'].get('save_clips'):
            clip = work/f'candidate_{int(original["candidate_number"]):02d}.wav'
            sf.write(clip,audio[round(vs*FS):round(ve*FS)],FS,subtype='PCM_16')
            result.update(youtube_clip_relative_path=clip.relative_to(Path(job['output'])).as_posix(),
                          youtube_clip_sha256=sha(clip))
        result_rows.append(result)
    return result_rows, dict(feature_seconds=feature_seconds, alignment_seconds=alignment_seconds,
        candidate_seconds=time.perf_counter()-started, audio_duration_seconds=len(audio)/FS,
        midi_duration_seconds=float((df.start+df.duration).max()), tuning_cents=tuning,
        subsequence_used=subsequence, video_feature_crop_start_seconds=audio_offset_frames/RATE,
        frame_resolution_seconds=1/RATE)


def execute_job(request_path):
    """Single-process crash-isolated job. Atomic result is the commit/checkpoint."""
    started = time.perf_counter()
    job = read_json(request_path)
    work = Path(request_path).parent
    result = dict(perf_id=job['metadata']['perf_id'], fingerprint=job['fingerprint'],
                  status='failed', rows=[], timings={}, error='')
    try:
        if not Path(job['metadata']['resolved_midi']).is_file():
            raise FileNotFoundError('original_midi_missing; supply --midi-root')
        if sha(job['metadata']['resolved_midi']) != job['metadata']['sha256']:
            raise ValueError('original_midi_hash_mismatch')
        wav, receipt = acquire_audio(job,work)
        result['audio_receipt'] = receipt
        result['rows'], timings = align_audio(job,wav,receipt,work)
        timings.update(download_seconds=receipt['download_seconds'], decode_seconds=receipt['decode_seconds'],
                       audio_source=receipt['audio_source'], download_bytes=receipt.get('download_bytes',0))
        result.update(status='completed',timings=timings,mapping_sha256=sha(work/'mapping.npz'),
                      package_versions={p:importlib.metadata.version(p) for p in
                          ['synctoolbox','numpy','librosa','pretty_midi','yt-dlp']})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result['error'] = str(exc)[-5000:]
        result['status'] = classify_error(str(exc))
        result['rows'] = [empty_match(s,result['status'],result['error'],job['metadata'].get('youtube_links',''))
                          for s in job['segments']]
    result['timings']['worker_seconds'] = time.perf_counter()-started
    result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(work/'result.json',result)
    # Remove only owned, explicitly named media files; never external cache inputs.
    if not job['options'].get('keep_audio'):
        for path in work.glob('source.*'):
            if (path.suffix in {'.wav','.webm','.m4a','.opus','.mp3','.mp4','.ogg','.part','.ytdl'}
                    or path.name=='source.info.json' or re.search(r'\.part-Frag[0-9]+$',path.name)):
                path.unlink(missing_ok=True)
    return 0 if result['status']=='completed' else 2
