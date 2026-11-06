"""Transfer explicit BPSD theme labels to ATEPP and export performance MIDI clips.

No audio, YouTube access, or learned theme inference. Existing accepted DualDTW
alignments are required. Score-label transfer is approximate and not ground truth.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import re
import sys

import mido
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from preprocessing.score_midi import load_score_midi
from analyze_annotated_structure import events, align_positions, bpsd_name
from atepp_feature_analysis import resolve_source


def read_sequence(source):
    midi = mido.MidiFile(source)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError('Expected synchronous PPQ MIDI')
    now, tempo, sequence = 0., 500000, []
    # Same tick-to-second integration as MidiFile.__iter__, retaining immutable
    # parsed messages instead of cloning every message twice. clip_midi never
    # mutates these objects; output delta times are set on a separate copy.
    for msg in midi.merged_track:
        now += mido.tick2second(msg.time, midi.ticks_per_beat, tempo) if msg.time else 0.
        sequence.append((now, msg))
        if msg.type == 'set_tempo':
            tempo = msg.tempo
    return sequence


def clip_midi(source, destination, start, end, sequence=None):
    """Clip in seconds, carrying channel state/held notes and closing notes/pedals.

    Fixed 1000 PPQ and 60 BPM encode original event seconds at 1 ms resolution;
    original tempo changes are integrated before conversion, never ignored.
    The half-open source interval is [start, end).
    """
    if not np.isfinite([start, end]).all() or not 0 <= start < end:
        raise ValueError('Invalid MIDI clip interval')
    sequence = read_sequence(source) if sequence is None else sequence
    if not sequence or end > sequence[-1][0] + .05:
        raise ValueError('Clip extends beyond source MIDI')
    active, notes, states, output = defaultdict(deque), [], {}, []
    for t, msg in sequence:
        if msg.type == 'note_on' and msg.velocity > 0:
            active[(msg.channel, msg.note)].append((t, msg))
        elif msg.type in ('note_off', 'note_on'):
            key = (msg.channel, msg.note)
            if active[key]:
                # Match the project's pretty_midi performance loader: one
                # offset closes prior onsets, while a simultaneous reattack
                # survives only when an older note was actually closed.
                close = [(a,m) for a,m in active[key] if t-a > 1e-9]
                keep = [(a,m) for a,m in active[key] if t-a <= 1e-9]
                for onset, original in close:
                    if onset < end-1e-9 and t > start+1e-9:
                        notes.append((max(onset, start)-start, min(t, end)-start, original))
                active[key] = deque(keep if close and keep else [])
        elif msg.type not in ('end_of_track', 'set_tempo'):
            if t < start:
                if msg.type == 'control_change':
                    states[(msg.type, msg.channel, msg.control)] = msg
                elif msg.type in ('program_change', 'pitchwheel', 'aftertouch'):
                    states[(msg.type, msg.channel)] = msg
                elif msg.is_meta and msg.type in ('time_signature', 'key_signature'):
                    states[(msg.type,)] = msg
            elif t < end:
                output.append((t-start, 2, msg))
    # Unclosed source onsets are invalid and ignored, as by pretty_midi.
    for msg in states.values():
        output.append((0., 0, msg))
    channels, exported_notes = set(), 0
    for a, b, msg in notes:
        if round(b*1000) <= round(a*1000):
            continue
        exported_notes += 1
        channels.add(msg.channel)
        output.append((a, 3, msg))
        output.append((b, 1, mido.Message('note_off', channel=msg.channel, note=msg.note, velocity=0)))
    if not exported_notes:
        raise ValueError('No notes in clip')
    for ch in channels:
        output.append((end-start, 4, mido.Message('control_change', channel=ch, control=64, value=0)))
    result = mido.MidiFile(type=0, ticks_per_beat=1000)
    track = mido.MidiTrack([mido.MetaMessage('set_tempo', tempo=1000000, time=0)])
    result.tracks.append(track)
    previous = 0
    for t, priority, msg in sorted(output, key=lambda x: (round(x[0]*1000), x[1])):
        tick = round(t*1000)
        track.append(msg.copy(skip_checks=True,time=tick-previous))
        previous = tick
    track.append(mido.MetaMessage('end_of_track', time=max(0, round((end-start)*1000)-previous)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination)
    return exported_notes


def local_boundary(target, q, pos, edge):
    """Invert only close, local matched events; no extrapolation over repeats."""
    valid = np.flatnonzero(np.isfinite(pos) & (np.abs(q-edge) <= 4.))
    exact = valid[np.abs(pos[valid]-target) <= .002]
    if len(exact):
        i = exact[np.argmin(np.abs(q[exact]-edge))]
        return float(q[i]), float(abs(q[i]-edge)), 'matched_event'
    for a, b in zip(valid[:-1], valid[1:]):
        if pos[a] < target < pos[b] and pos[b]-pos[a] <= .5 and q[b]-q[a] <= 2:
            return float(np.interp(target, pos[[a,b]], q[[a,b]])), float(q[b]-q[a]), 'local_interpolation'
    raise ValueError('boundary_not_locally_bracketed')


def score_segments(score_path, annotation_root, name):
    notes = load_score_midi(score_path)
    q, rows = events([n.onset_beats for n in notes], [n.pitch for n in notes])
    ref = pd.read_csv(annotation_root/'ann_score_note'/f'{name}.csv', sep=';')
    ref = ref[ref.grace == 0]
    rq, rr = events(ref.start_meas.to_numpy(), ref.pitch.to_numpy())
    pos, match = align_positions(q, rows, rq, rr, rq)
    fine = pd.read_csv(annotation_root/'ann_score_structureFine'/f'{name}.csv', sep=';')
    labels = np.full(len(q), -1, dtype=int)
    for idx, row in fine.iterrows():
        labels[(pos >= row.start-.002) & (pos < row.end-.002)] = idx
    # Fill only short holes bounded by the same structural label.
    good = np.flatnonzero(labels >= 0)
    for a, b in zip(good[:-1], good[1:]):
        if b > a+1 and labels[a] == labels[b] and q[b]-q[a] <= 2:
            labels[a+1:b] = labels[a]
    segments, reviews = [], []
    for idx, row in fine.iterrows():
        theme = row.structure.split(':')[-1]
        if theme not in ('FirstGroup', 'SecondGroup', 'Theme'):
            continue
        members = np.flatnonzero(labels == idx)
        if not len(members):
            continue  # ATEPP may omit the repeat written out by BPSD.
        blocks = np.split(members, np.flatnonzero(np.diff(members) > 1)+1)
        for block in blocks:
            record = dict(annotation_row=int(idx), label=row.structure,
                          reference_start_measure=float(row.start), reference_end_measure=float(row.end),
                          theme_type=theme, score_transfer_match=match)
            try:
                if match < .8:
                    raise ValueError('score_transfer_match_below_0.8')
                a, ga, ma = local_boundary(row.start, q, pos, q[block[0]])
                b, gb, mb = local_boundary(row.end, q, pos, q[block[-1]])
                inside = (q >= a) & (q < b)
                coverage = np.mean((labels[inside] == idx) & np.isfinite(pos[inside])) if inside.any() else 0.
                if b-a < 4 or coverage < .85:
                    raise ValueError('fragmented_or_low_local_coverage')
                # Reject reversed reference motion within a claimed theme occurrence.
                pp = pos[inside & np.isfinite(pos)]
                if np.any(np.diff(pp) < -.01):
                    raise ValueError('nonmonotone_reference_inside_theme')
                record.update(score_start_beat=a, score_end_beat=b, local_label_coverage=float(coverage),
                              start_boundary_method=ma, end_boundary_method=mb,
                              max_boundary_bracket_beats=max(ga, gb))
                segments.append(record)
            except ValueError as exc:
                reviews.append(dict(**record, reason=str(exc)))
    # De-duplicate any fragments that resolve to the same boundaries.
    unique = {(s['theme_type'], round(s['score_start_beat'],4), round(s['score_end_beat'],4)): s for s in segments}
    segments = sorted(unique.values(), key=lambda s:s['score_start_beat'])
    primary_rows = fine.index[fine.structure.isin(['Exposition:FirstGroup', 'Theme'])]
    primary_row = int(primary_rows[0]) if len(primary_rows) else -1
    counts = defaultdict(int)
    for s in segments:
        counts[s['theme_type']] += 1
        s['occurrence'] = counts[s['theme_type']]
        s['is_primary'] = s['annotation_row'] == primary_row and s['occurrence'] == 1
    return segments, reviews, q, pos, match


def main():
    workspace = HERE.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=workspace/'ATEPP_dataset')
    parser.add_argument('--annotations', type=Path, default=workspace/'annotations/bpsd/Beethoven_Piano_Sonata_Dataset_v2/2_Annotations')
    parser.add_argument('--out', type=Path, default=workspace/'outputs/atepp_theme_extraction')
    parser.add_argument('--limit-works', type=int)
    parser.add_argument('--reexport', action='store_true', help='Re-export existing segment manifest without recomputing score mappings')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.reexport:
        data = pd.read_csv(args.out/'segments.csv', dtype={'perf_id':str,'composition_id':str})
        for source, group in data.groupby('source_midi'):
            sequence = read_sequence(source)
            for index,row in group.iterrows():
                data.loc[index,'note_count'] = clip_midi(source,Path(row.clip_path),row.start_seconds,row.end_seconds,sequence)
        data.to_csv(args.out/'segments.csv',index=False,encoding='utf-8-sig')
        data[data.is_primary].to_csv(args.out/'primary_segments.csv',index=False,encoding='utf-8-sig')
        summary=json.loads((args.out/'summary.json').read_text(encoding='utf-8'))
        summary['reexport_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'Re-exported {len(data)} clips',flush=True)
        return
    audit = pd.read_csv(HERE.parent/'analysis/atepp_research/midi_audit.csv', dtype={'perf_id':str,'composition_id':str}).fillna('')
    extracted, reviews, coverage, score_summary = [], [], [], []
    cache, processed = {}, set()
    for row in audit.to_dict('records'):
        pid = row['perf_id']
        status = dict(perf_id=pid, composition_id=row['composition_id'], artist=row['artist'],
                      track=row['track'], source_alignment_status=row['status'])
        if row['status'] != 'aligned':
            coverage.append(dict(**status, extraction_status='excluded_source_alignment', reason=row['status']))
            continue
        name = None
        if 'Beethoven' in row['composer'] and re.search(r':\s*(?:I\.|1\.)\s', row['track']):
            try:
                name = bpsd_name(row['track'])
            except (AttributeError, ValueError):
                pass
        if not name or not (args.annotations/'ann_score_structureFine'/f'{name}.csv').exists():
            coverage.append(dict(**status, extraction_status='no_explicit_theme_annotation', reason='no_supported_BPSD_theme_label'))
            continue
        score_path = resolve_source(row['local_score'], args.root)
        key = str(score_path)
        if key not in cache:
            if args.limit_works and len(processed) >= args.limit_works:
                continue
            processed.add(key)
            print(f'Transferring {name}: {score_path.name}', flush=True)
            segs, rev, q, pos, match = score_segments(score_path, args.annotations, name)
            cache[key] = (segs, match)
            digest = hashlib.sha256(score_path.read_bytes()).hexdigest()[:12]
            np.savez_compressed(args.out/f'score_map_{name}_{digest}.npz', score_beats=q, reference_measures=pos)
            score_summary.append(dict(bpsd_id=name, score_path=key, score_transfer_match=match, segments=len(segs), review_items=len(rev)))
            reviews.extend(dict(bpsd_id=name, score_path=key, **r) for r in rev)
        segs, match = cache[key]
        count = 0
        with np.load(resolve_source(row['alignment_cache'], args.root), allow_pickle=False) as z:
            anchors, times = z['anchors'], z['times']
            matched_q = z['score_onsets']
            source = resolve_source(row['local_midi'], args.root)
            sequence = read_sequence(source) if segs else None
            for s in segs:
                info = dict(perf_id=pid, composition_id=row['composition_id'], artist=row['artist'], track=row['track'],
                            bpsd_id=name, source_midi=str(source), source_alignment=str(resolve_source(row['alignment_cache'], args.root)), **s)
                a,b = s['score_start_beat'],s['score_end_beat']
                try:
                    if a < anchors[0] or b > anchors[-1]:
                        raise ValueError('theme_outside_alignment_anchors')
                    distances = [float(np.min(np.abs(matched_q-edge))) for edge in (a,b)]
                    if max(distances) > 1.:
                        raise ValueError('no_performance_match_within_one_beat_of_boundary')
                    ta,tb = np.interp([a,b], anchors,times)
                    if not 0 <= ta < tb:
                        raise ValueError('invalid_time_order')
                    tag = f"{pid}_{s['theme_type']}_{s['occurrence']:02d}"
                    destination = args.out/'midi'/name/f'{tag}.mid'
                    n = clip_midi(source,destination,float(ta),float(tb),sequence=sequence)
                    # Reopen every output; duration and note balance checked separately below.
                    duration = mido.MidiFile(destination).length
                    if abs(duration-(tb-ta)) > .002:
                        raise ValueError('export_duration_mismatch')
                    extracted.append(dict(**info, start_seconds=float(ta), end_seconds=float(tb), duration_seconds=float(tb-ta),
                                          boundary_match_distance_beats=max(distances), note_count=n, clip_path=str(destination),
                                          cadence_status='not_annotated', human_verified=False))
                    count += 1
                except ValueError as exc:
                    reviews.append(dict(**info, reason=str(exc)))
        coverage.append(dict(**status, extraction_status='extracted' if count else 'no_accepted_theme_segment',
                             reason='' if count else 'no_explicit_theme_or_transfer_failed', clips=count))
    for name, records in [('segments',extracted),('review_queue',reviews),('coverage',coverage),('score_transfer',score_summary)]:
        pd.DataFrame(records).to_csv(args.out/f'{name}.csv', index=False, encoding='utf-8-sig')
    primary = [r for r in extracted if r['is_primary']]
    pd.DataFrame(primary).to_csv(args.out/'primary_segments.csv', index=False, encoding='utf-8-sig')
    summary = dict(source_performances=len(audit), considered_performances=len(coverage), clips=len(extracted),
                   extracted_performances=len({r['perf_id'] for r in extracted}),
                   extracted_compositions=len({r['composition_id'] for r in extracted}),
                   primary_clips=len(primary), score_versions=len(cache), review_items=len(reviews),
                   coverage_status=pd.Series([r['extraction_status'] for r in coverage]).value_counts().to_dict(),
                   theme_counts=pd.Series([r['theme_type'] for r in extracted]).value_counts().to_dict(),
                   annotation_source='BPSD v2 structureFine', annotation_license='CC BY 3.0',
                   alignment='existing quality-accepted DualDTW; approximate automatic mapping',
                   cadence_extracted=False, youtube_accessed=False, human_verified=False,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
