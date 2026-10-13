"""BWV 848 non-repeating XML/MIDI/ASAP bridge and unchanged Dynamics overlay.

All nine performances are fixed by the existing score input inventory.
Outputs stay separate from the source features and previous reports.
"""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'classicfy-ai/src'))
from preprocessing import ASAPLoader, load_match, load_midi, load_musicxml
from preprocessing.score_beat_bridge import build_score_beat_bridge, load_midi_score_timeline
from features import extract_dynamics
from validate_tempo import plt, setup_style

WORK = 'Bach/Fugue/bwv_848'
VERIFIED = {'verified', 'verified_onset_duration_differs', 'verified_onset_duration_ambiguous'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, values):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def frozen_xml(directory, inventory):
    matches = [r for r in rows(inventory) if r['xml_score'] == WORK+'/xml_score.musicxml']
    if len(matches) != 1:
        raise ValueError('Expected one fixed BWV 848 XML inventory row')
    row = matches[0]
    path = directory / row['xml_score']
    if sha(path) != row['xml_sha256']:
        raise ValueError('XML differs from the frozen input inventory: '+str(path))
    return path


def note_rows(score):
    result = []
    for n in score.notes:
        result.append(dict(note_id=n.note_id, xml_id=n.xml_id, measure_index=n.measure_index,
            measure_number=n.measure_number, onset_quarters=str(n.onset_quarters),
            duration_quarters=str(n.duration_quarters), pitch=str(n.pitch.midi_pitch) if n.pitch else '',
            voice=n.voice, staff=n.staff, is_rest=n.is_rest, is_grace=n.is_grace,
            ties=json.dumps([dict(t) for t in n.ties], sort_keys=True),
            slurs=json.dumps([dict(kind=s.kind, number=s.number) for s in n.slurs], sort_keys=True)))
    return result


def plot_overlay(score, beat_rows, features, diagnostics, out):
    setup_style()
    # The central 32 original intervals, fixed by count rather than any feature.
    count = len(beat_rows)
    start, stop = (count-32)//2, (count-32)//2+32
    left, right = beat_rows[start].xml_start_quarters, beat_rows[stop-1].xml_stop_quarters
    notes = [n for n in score.notes if n.pitch is not None and
             float(n.onset_quarters) < right and float(n.offset_quarters) >= left]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1.2, 1.5]})
    x = lambda q: start + (float(q)-left) / (right-left) * (stop-start)
    for n in notes:
        color = '#c9702c' if n.is_grace else '#3279a8'
        axes[0].plot([x(n.onset_quarters), x(n.offset_quarters)],
                     [float(n.pitch.midi_pitch)]*2, color=color, linewidth=2, alpha=.7)
    axes[0].set_ylabel('Score MIDI pitch')
    axes[0].set_title('BWV 848: written score and unchanged Dynamics on original beat indices',
                      loc='left', fontsize=15, pad=24)
    axes[0].text(0, 1.025, 'XML positions → score MIDI quarters → ASAP intervals; raw slurs are not phrase labels',
                 transform=axes[0].transAxes, fontsize=10)
    voice_ids = sorted({n.voice for n in score.notes if n.voice is not None})
    for n in score.notes:
        if not left <= float(n.onset_quarters) < right or n.voice is None:
            continue
        y = voice_ids.index(n.voice)
        if n.is_rest:
            axes[1].plot([x(n.onset_quarters), x(n.offset_quarters)], [y, y],
                         color='#888888', linewidth=6, alpha=.7)
        for slur in n.slurs:
            axes[1].scatter(x(n.onset_quarters), y, marker='>' if slur.kind=='start' else '<',
                            color='#ad3e66', s=45)
    axes[1].set_yticks(range(len(voice_ids)), ['voice '+v for v in voice_ids])
    axes[1].set_ylabel('Rests / slur marks')
    for key in sorted(features):
        rs = sorted(features[key], key=lambda r:int(r['beat_index']))[start:stop]
        values = [float(r['standardized']) if r['standardized_mask']=='True' else np.nan for r in rs]
        axes[2].plot(np.arange(start, stop)+.5, values, linewidth=1.1, alpha=.85,
                     label=Path(key).stem)
    axes[2].set_ylabel('Dynamics residual / scale')
    axes[2].set_xlabel('Original ASAP beat interval index (0-based)')
    axes[2].legend(ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(.5, -.28))
    for interval in beat_rows[start:stop]:
        for ax in axes:
            ax.axvline(interval.index, color='#d9d9d9', linewidth=.5, zorder=0)
        if interval.index==start or interval.measure_index != beat_rows[interval.index-1].measure_index:
            axes[0].text(interval.index+.1, 86, 'm'+str(interval.measure_number), fontsize=8)
        if not interval.valid:
            for ax in axes:
                ax.axvspan(interval.index, interval.index+1, color='#e4aba0', alpha=.4)
    axes[0].set_ylim(35, 89)
    axes[2].set_xlim(start, stop)
    fig.subplots_adjust(left=.09, right=.98, top=.9, bottom=.18, hspace=.16)
    fig.savefig(out/'01_structure_feature_alignment.png', dpi=160)
    fig.savefig(out/'01_structure_feature_alignment.pdf')
    plt.close(fig)
    # Full score, all nine curves: also expose the ends omitted by the detail.
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    axes[0].plot([i.index+.5 for i in beat_rows], [i.verified_onsets for i in beat_rows],
                 color='#2878c8', linewidth=1)
    axes[0].set_ylabel('Verified pitched onsets')
    axes[0].set_title('All 216 original beat intervals × 9 performances', loc='left', fontsize=16)
    for row in diagnostics:
        if row['status']!='missing_midi':
            continue
        index=row['beat_index']
        axes[0].axvline(index,color='#bd6157',linewidth=.8,linestyle='--')
        label=row['xml_id']+(' / raw slur marks' if row['xml_id']=='n297' else '')
        axes[0].text(index+.8,.92,label,transform=axes[0].get_xaxis_transform(),fontsize=8,color='#984b44')
    for key in sorted(features):
        rs=sorted(features[key], key=lambda r:int(r['beat_index']))
        axes[1].plot([int(r['beat_index'])+.5 for r in rs],
                     [float(r['standardized']) if r['standardized_mask']=='True' else np.nan for r in rs],
                     linewidth=.8, alpha=.75, label=Path(key).stem)
    axes[1].set_ylabel('Unchanged Dynamics')
    axes[1].set_xlabel('Original ASAP beat interval index (0-based)')
    axes[1].legend(ncol=5, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.2))
    axes[1].set_xlim(0,count)
    fig.subplots_adjust(left=.08,right=.98,top=.92,bottom=.2,hspace=.1)
    fig.savefig(out/'02_full_beat_coverage.png',dpi=160)
    plt.close(fig)
    return [start, stop]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asap-root',type=Path,default=ROOT.parent/'datasets/ASAP')
    parser.add_argument('--nasap-root',type=Path,default=ROOT.parent/'datasets/nASAP')
    parser.add_argument('--inventory',type=Path,default=ROOT/'classicfy-ai/analysis/score_structure_inputs')
    parser.add_argument('--dynamics',type=Path,default=ROOT/'classicfy-ai/analysis/dynamics/beat_features.csv')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/score_beat_bridge')
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Choose an empty --out directory; previous results are preserved')
    args.out.mkdir(parents=True,exist_ok=True)
    original_path=frozen_xml(args.asap_root,args.inventory/'works.csv')
    xml_path=frozen_xml(args.nasap_root,args.inventory/'nasap_works.csv')
    score=load_musicxml(xml_path)
    original=load_musicxml(original_path)
    if tuple(replace(n,xml_id=None) for n in score.notes)!=original.notes or score.measures!=original.measures:
        raise ValueError('ASAP/nASAP written structure differs beyond XML IDs')
    fixed=[r for r in rows(args.inventory/'performances.csv') if r['piece']==WORK+'/midi_score.mid']
    if len(fixed)!=9 or len({r['key'] for r in fixed})!=9:
        raise ValueError('Expected the fixed nine BWV 848 performances')
    features=defaultdict(list)
    for row in rows(args.dynamics):
        features[row['key']].append(row)
    if set(features)!={r['key'] for r in fixed}:
        raise ValueError('Dynamics keys differ from the fixed cohort')
    loader=ASAPLoader(args.asap_root,args.nasap_root)
    midi=load_midi_score_timeline(args.asap_root/WORK/'midi_score.mid')
    link_rows,beat_rows,joins,coverage=[],[],[],[]
    def input_name(path):
        path=path.resolve()
        if path==args.dynamics.resolve():
            return 'feature/beat_features.csv'
        for prefix,directory in [('inventory',args.inventory),('ASAP',args.asap_root),('nASAP',args.nasap_root)]:
            if path.is_relative_to(directory.resolve()):
                return prefix+'/'+path.relative_to(directory.resolve()).as_posix()
        raise ValueError('Input is outside its declared root')
    inputs={input_name(p):sha(p) for p in [original_path,xml_path,args.dynamics,
        args.inventory/'works.csv',args.inventory/'nasap_works.csv',args.inventory/'performances.csv',
        args.asap_root/'metadata.csv',args.asap_root/'asap_annotations.json',args.nasap_root/'metadata.csv',
        args.asap_root/WORK/'midi_score.mid']}
    by_xml={n.note_id:n for n in score.notes}
    for row in sorted(fixed,key=lambda r:r['key']):
        sample=loader.get_sample(row['key'])
        if not sample.aligned or sample.robust_note_alignment is not True:
            raise ValueError('Fixed cohort requires aligned robust performances')
        if sha(sample.note_alignment_path)!=row['note_alignment_sha256']:
            raise ValueError('Match differs from frozen input inventory')
        for path in (sample.note_alignment_path,sample.performance_path):
            inputs[input_name(path)]=sha(path)
        grid=hashlib.sha256(np.asarray(sample.score_beats).tobytes()+json.dumps(sample.score_beat_types).encode()).hexdigest()[:12]
        if grid!=row['grid']:
            raise ValueError('Existing feature grid changed')
        alignment=load_match(sample.note_alignment_path)
        bridge=build_score_beat_bridge(score,alignment,midi,sample.score_beats,
                                      sample.performance_beats,sample.score_beat_types)
        performed={s.note_id:p for s,p in alignment.matches}
        match_text=sample.note_alignment_path.read_text()
        clock_rate=int(re.search(r'info\(midiClockRate,(\d+)\)',match_text)[1])
        clock_units=int(re.search(r'info\(midiClockUnits,(\d+)\)',match_text)[1])
        boundary_tolerance=clock_rate/clock_units/1_000_000/2 + .5e-6 + 1e-10
        for link in bridge.note_links:
            n=by_xml[link.note_id]
            p=performed.get(link.match_id)
            index = (next((i for i,(left,right) in enumerate(zip(sample.performance_beats[:-1],
                sample.performance_beats[1:])) if left<=p.onset<right), None) if p else None)
            boundary_uncertain=(min(abs(p.onset-t) for t in sample.performance_beats)<=boundary_tolerance) if p else None
            link_rows.append(dict(key=row['key'], **asdict(link),
                pitch=str(n.pitch.midi_pitch) if n.pitch else '',
                performed_onset_seconds=p.onset if p else None,
                performed_onset_beat_index=index,
                performed_boundary_tolerance_seconds=boundary_tolerance,
                performed_boundary_uncertain=boundary_uncertain,
                performed_assignment_agrees=(index==link.beat_index) if p and not boundary_uncertain else None))
        beat_rows.extend(dict(key=row['key'],**asdict(i)) for i in bridge.intervals)
        sequence=extract_dynamics(load_midi(sample.performance_path),sample.performance_beats)
        table=sorted(features[row['key']],key=lambda r:int(r['beat_index']))
        if [int(r['beat_index']) for r in table]!=list(range(len(bridge.intervals))):
            raise ValueError('Dynamics indices are not the original B-1 intervals')
        for interval,feature in zip(bridge.intervals,table):
            i=interval.index
            checks=[np.isclose(float(feature['raw']),sequence.sequence.values[i],rtol=0,atol=1e-12,equal_nan=True),
                (feature['raw_mask']=='True')==bool(sequence.sequence.mask[i]),
                int(feature['onset_count'])==sequence.onset_counts[i],
                abs(float(feature['performance_start_seconds'])-sample.performance_beats[i])<1e-9,
                abs(float(feature['performance_end_seconds'])-sample.performance_beats[i+1])<1e-9]
            if not all(checks):
                raise ValueError('Existing Dynamics differs from direct source extraction')
            if feature['standardized_mask']=='True':
                np.testing.assert_allclose(float(feature['standardized']),
                    (float(feature['raw'])-float(feature['common']))/float(feature['scale']),rtol=0,atol=1e-12)
            joins.append(dict(key=row['key'],beat_index=i,xml_start_quarters=interval.xml_start_quarters,
                xml_stop_quarters=interval.xml_stop_quarters,measure_number=interval.measure_number,
                bridge_status=interval.status,bridge_valid=interval.valid,
                dynamics_raw=feature['raw'],dynamics_standardized=feature['standardized'],
                dynamics_mask=feature['standardized_mask'],scale=feature['scale'],
                combined_mask=interval.valid and feature['standardized_mask']=='True'))
        counts=Counter(l.status for l in bridge.note_links)
        coverage.append(dict(key=row['key'],grid=grid,original_intervals=len(bridge.intervals),
            valid_intervals=sum(i.valid for i in bridge.intervals),
            minimum_verified_onsets=min(i.verified_onsets for i in bridge.intervals),
            xml_to_score_offset_quarters=bridge.xml_to_score_offset_quarters,offset_support=bridge.offset_support,
            note_statuses=json.dumps(dict(counts),sort_keys=True),
            outside_onsets=sum(l.beat_index is None for l in bridge.note_links),
            performed_boundary_uncertain=sum(r['performed_boundary_uncertain'] is True for r in link_rows if r['key']==row['key']),
            performed_assignment_agrees=sum(r['performed_assignment_agrees'] is True for r in link_rows if r['key']==row['key']),
            performed_assignment_available=sum(r['performed_assignment_agrees'] is not None for r in link_rows if r['key']==row['key'])))
        print(row['key'],dict(counts),flush=True)
    write(args.out/'xml_notes.csv',note_rows(score))
    write(args.out/'note_links.csv',link_rows)
    write(args.out/'beat_map.csv',beat_rows)
    write(args.out/'feature_join.csv',joins)
    write(args.out/'coverage.csv',coverage)
    diagnostics=[]
    for link in bridge.note_links:
        if link.status not in ('missing_midi','verified_onset_duration_differs','tie_head_unverified',
                                'ambiguous_midi','verified_onset_duration_ambiguous'):
            continue
        n=by_xml[link.note_id]
        same_pitch=[m for m in midi.notes if m.pitch==n.pitch.midi_pitch and m.offset_quarters>m.onset_quarters]
        nearest=min(same_pitch,key=lambda m:abs(float(m.onset_quarters)-link.score_onset_quarters)) if same_pitch else None
        diagnostics.append(dict(xml_id=link.xml_id,status=link.status,pitch=str(n.pitch.midi_pitch),
            beat_index=link.beat_index,xml_onset_quarters=str(n.onset_quarters),
            match_duration_quarters=link.match_duration_quarters,midi_duration_quarters=link.midi_duration_quarters,
            nearest_same_pitch_midi_onset_quarters=float(nearest.onset_quarters) if nearest else None,
            nearest_onset_delta_quarters=float(nearest.onset_quarters)-link.score_onset_quarters if nearest else None,
            scope='Same score-MIDI evidence for all nine performances; nearest note is diagnostic only'))
    if diagnostics:
        write(args.out/'note_diagnostics.csv',diagnostics)
    detail=plot_overlay(score,bridge.intervals,features,diagnostics,args.out)
    protocol=dict(work=WORK,performances=9,scope='Non-repeating single-part XML/score-MIDI/ASAP bridge',
        symbolic_mapping='XML plus score side of match (including deletions); no performed timing fit',
        match_rounding_tolerance_quarters=1e-4,midi_tolerance_quarters=1/midi.ppq+1e-4,
        ppq=midi.ppq,note_duration_policy='Overlapping same-channel/pitch durations remain ambiguous; zero-length grace retained; ties explicitly linked',
        libraries={name:importlib.metadata.version(name) for name in ('mido','pretty_midi','numpy','matplotlib')},
        feature='Existing normalized Dynamics, unchanged CSV; direct source raw/mask/count verification',
        performed_boundary_policy='Within half a match MIDI tick plus 0.5 microseconds and 1e-10 second float guard of any performed beat: uncertain; excluded from assignment agreement denominator',
        figure_selection='Central 32 original intervals, all nine performances; full coverage figure',
        figure_detail_original_indices=detail,inputs=inputs,
        xml_sources={'ASAP':'https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/'+WORK+'/xml_score.musicxml',
                     'nASAP':'https://raw.githubusercontent.com/CPJKU/asap-dataset/main/'+WORK+'/xml_score.musicxml'},
        limitations=['Not repeat unfolding or phrase/tension inference',
            'Coordinate coverage is distinct from verification of every MIDI note',
            'Performed onset assignment is a diagnostic, not the definition of symbolic structure',
            'Voice-specific rests are not silence of the whole score',
            'A single work does not establish other-work alignment or retrieval quality'])
    (args.out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    summary=dict(work=WORK,performances=9,xml_events=len(score.notes),
        original_intervals=len(bridge.intervals),beat_rows=len(beat_rows),feature_rows=len(joins),
        valid_intervals=sum(r['valid_intervals'] for r in coverage),
        per_performance=coverage,scope=protocol['scope'])
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    paths=[Path(__file__),ROOT/'classicfy-ai/src/preprocessing/score_beat_bridge.py',
           ROOT/'classicfy-ai/src/preprocessing/musicxml_loader.py',
           ROOT/'classicfy-ai/src/preprocessing/asap_loader.py',
           ROOT/'classicfy-ai/src/preprocessing/note_alignment.py',
           ROOT/'classicfy-ai/src/preprocessing/midi_loader.py',
           ROOT/'classicfy-ai/src/preprocessing/__init__.py',
           ROOT/'classicfy-ai/src/features/dynamics.py',
           ROOT/'classicfy-ai/src/features/beat_grid.py',
           ROOT/'classicfy-ai/src/features/models.py',
           ROOT/'classicfy-ai/scripts/verify_score_beat_bridge.py']
    provenance=dict(code={str(p.relative_to(ROOT)):sha(p) for p in paths},
        results={p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()})
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Saved',args.out,flush=True)


if __name__=='__main__':
    main()
