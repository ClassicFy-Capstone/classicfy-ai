"""Published Parangonar aligners, with the same MIDI inputs and tempo-map policy.

Neither ASAP reference beats nor reference note pairs are passed to a matcher.
Score coordinates are quarter notes; performance coordinates are seconds.
"""
from functools import lru_cache
import time
import os
import numpy as np
from .note_alignment import NoteAlignment, PerformedNote
from features.articulation import build_tempo_map
from features.performance_time import ESTIMATED_BEAT_MIN_TAU, ESTIMATED_BEAT_MAX_TAU

MODELS = ('dual_dtw', 'gluenote')


def note_arrays(score, midi):
    sd = [('id', 'U64'), ('pitch', 'i4'), ('onset_beat', 'f4'),
          ('duration_beat', 'f4'), ('onset_quarter', 'f4'),
          ('duration_quarter', 'f4'), ('is_grace', '?'), ('voice', 'i4')]
    pd = [('id', 'U64'), ('pitch', 'i4'), ('onset_sec', 'f4'),
          ('duration_sec', 'f4'), ('velocity', 'i4')]
    sna = np.array([(s.note_id, s.pitch, s.onset_beats, s.duration_beats,
                     s.onset_beats, s.duration_beats, s.is_grace, 1) for s in score], dtype=sd)
    pna = np.array([(str(i), n.pitch, n.start, n.duration, n.velocity)
                    for i, n in enumerate(midi.notes)], dtype=pd)
    return sna, pna


@lru_cache(maxsize=2)
def matcher(model):
    # Parangonar 3.3.3 still uses the alias removed by NumPy 2.4. Its behavior
    # is exactly vstack; keep the published algorithm unchanged.
    if not hasattr(np, 'row_stack'):
        np.row_stack = np.vstack
    import torch
    import parangonar
    from .exact_dtw_kernel import install_exact_kernel
    install_exact_kernel()
    torch.set_num_threads(int(os.environ.get('CLASSICFY_TORCH_THREADS','2')))
    if model == 'dual_dtw':
        return parangonar.DualDTWNoteMatcher()
    if model == 'gluenote':
        engine=parangonar.TheGlueNoteMatcher()
        backend=os.environ.get('CLASSICFY_GLUE_BACKEND','torch')
        if backend!='torch':
            if backend not in ('openvino_cpu','openvino_gpu'): raise ValueError('Unknown GlueNote backend')
            from .gluenote_openvino import accelerate
            engine=accelerate(engine,backend)
        return engine
    raise ValueError(f'Unknown research aligner: {model}')


def prediction_conflicts(predictions):
    edges={(str(item['score_id']),str(item['performance_id'])) for item in predictions if item['label']=='match'}
    score_counts,perf_counts={},{}
    for si,pi in edges:
        score_counts[si]=score_counts.get(si,0)+1
        perf_counts[pi]=perf_counts.get(pi,0)+1
    conflicts={(si,pi) for si,pi in edges if score_counts[si]>1 or perf_counts[pi]>1}
    return conflicts


def alignment_from_predictions(predictions, score, midi):
    scores = {s.note_id: s for s in score}
    perf = {str(i): PerformedNote(str(i), n.pitch, n.start, n.end, n.velocity)
            for i, n in enumerate(midi.notes)}
    conflicts=prediction_conflicts(predictions)
    pairs, used_s, used_p = [], set(), set()
    for item in predictions:
        if item['label'] != 'match':
            continue
        si, pi = str(item['score_id']), str(item['performance_id'])
        if si not in scores or pi not in perf:
            raise ValueError('Matcher returned an unknown note ID')
        # Do not guess which edge is correct. Remove ALL conflicting edges for
        # either matcher. Exact duplicate edges are simply counted once.
        if (si,pi) in conflicts or si in used_s or pi in used_p:
            continue
        pairs.append((scores[si], perf[pi])); used_s.add(si); used_p.add(pi)
    return NoteAlignment(tuple(pairs), tuple(s for si, s in scores.items() if si not in used_s),
                         tuple(p for pi, p in perf.items() if pi not in used_p))


def tempo_grid(alignment):
    positions, times = build_tempo_map(alignment)
    if len(positions) < 2:
        raise ValueError('Fewer than two monotone non-grace anchors')
    beats = np.arange(np.ceil(positions[0]), np.floor(positions[-1]) + 1.)
    return positions, times, beats, np.interp(beats, positions, times)


def align_research(score, midi, model):
    sna, pna = note_arrays(score, midi)
    engine = matcher(model)  # initialization/model loading excluded from per-file runtime
    start = time.perf_counter()
    if model == 'dual_dtw':
        predictions = engine(sna.copy(), pna.copy(), process_ornaments=False)
    else:
        with engine.torch.inference_mode():
            predictions = engine(sna.copy(), pna.copy())
    elapsed = time.perf_counter() - start
    alignment = alignment_from_predictions(predictions, score, midi)
    positions, times, beats, perf_beats = tempo_grid(alignment)
    widths = np.diff(perf_beats)
    median_width = np.median(widths[widths > 0]) if np.any(widths > 0) else 1.
    errors = [abs(p.onset - np.interp(s.onset_beats, positions, times))/median_width
              for s, p in alignment.matches if not s.is_grace]
    last = max(s.offset_beats for s in score)
    unique_onsets = len({s.onset_beats for s, _ in alignment.matches if not s.is_grace})
    metrics = dict(aligner=model, alignment_runtime_seconds=elapsed,
                   inference_backend=os.environ.get('CLASSICFY_GLUE_BACKEND','torch') if model=='gluenote' else 'not_neural',
                   conflicting_prediction_pairs_removed=len(prediction_conflicts(predictions)),
                   score_notes=len(score), performance_notes=len(midi.notes), matches=len(alignment.matches),
                   score_recall=len(alignment.matches)/len(score),
                   performance_precision=len(alignment.matches)/len(midi.notes),
                   span_coverage=(positions[-1]-positions[0])/max(last-min(s.onset_beats for s in score), 1e-6),
                   monotone_anchor_ratio=len(positions)/max(unique_onsets, 1),
                   timing_p90_beats=float(np.percentile(errors, 90)) if errors else 999.,
                   note_count_ratio=len(midi.notes)/len(score), beat_intervals=len(widths),
                   extreme_beat_fraction=float(np.mean((widths < ESTIMATED_BEAT_MIN_TAU)
                                                       | (widths > ESTIMATED_BEAT_MAX_TAU))) if len(widths) else 1.)
    reasons = []
    for name, threshold in [('score_recall', .70), ('performance_precision', .60),
                            ('span_coverage', .90), ('monotone_anchor_ratio', .90)]:
        if metrics[name] < threshold: reasons.append(name)
    if not .65 <= metrics['note_count_ratio'] <= 1.6: reasons.append('note_count_ratio')
    if metrics['timing_p90_beats'] > .5: reasons.append('timing_p90_beats')
    if len(widths) < 16 or np.any(widths <= 0) or metrics['extreme_beat_fraction'] > .02:
        reasons.append('beat_grid')
    metrics.update(accepted=not reasons, reject_reasons=';'.join(reasons))
    return alignment, beats, perf_beats, metrics
