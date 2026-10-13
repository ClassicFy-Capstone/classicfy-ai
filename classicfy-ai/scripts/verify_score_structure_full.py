"""Independent raw XML/match/pretty_midi audit of the full bridge artifacts.

Does not import any production bridge, XML loader or match loader. Original
seven-channel features are separately reconstructed from the frozen collector.
"""

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from fractions import Fraction
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pretty_midi

from verify_score_beat_bridge import raw_match,sha,rows

ROOT=Path(__file__).resolve().parents[2]


def records(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as stream:
        yield from csv.DictReader(stream)


def symbolic(path):
    root=ET.parse(path).getroot()
    for n in root.iter():n.tag=n.tag.rsplit('}',1)[-1]
    notes={};measures={}
    pitch_classes={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
    for part in root.findall('part'):
        pid=part.get('id');division=None;position=Fraction(0)
        for index,measure in enumerate(part.findall('measure')):
            cursor=extent=Fraction(0);previous=None;ordinal=0
            for item in measure:
                if item.tag=='attributes' and item.findtext('divisions') is not None:
                    division=Fraction(item.findtext('divisions'))
                elif item.tag in ('backup','forward'):
                    cursor+=Fraction(item.findtext('duration'))/division*(-1 if item.tag=='backup' else 1)
                    extent=max(extent,cursor);previous=None
                elif item.tag=='note':
                    grace=item.find('grace') is not None
                    duration=Fraction(0) if grace else Fraction(item.findtext('duration'))/division
                    offset=previous if item.find('chord') is not None else cursor
                    pitch=item.find('pitch')
                    value=(12*(int(pitch.findtext('octave'))+1)+pitch_classes[pitch.findtext('step')]+
                           Fraction(pitch.findtext('alter') or '0')) if pitch is not None else None
                    notes[f'{pid}:m{index}:n{ordinal}']=dict(xml_id=item.get('id'),part_id=pid,measure_index=index,
                        offset=offset,onset=position+offset,duration=duration,pitch=value,grace=grace,
                        rest=item.find('rest') is not None)
                    extent=max(extent,offset+duration)
                    if item.find('chord') is None:previous=cursor;cursor+=duration
                    ordinal+=1
            measures[pid,index]=dict(number=measure.get('number'),onset=position,duration=extent)
            position+=extent
    return notes,measures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/score_structure_full')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/score_structure_full_run01')
    parser.add_argument('--data',type=Path,default=ROOT.parent/'datasets')
    args=parser.parse_args();protocol=json.loads((args.out/'protocol.json').read_text())
    provenance=json.loads((args.out/'provenance.json').read_text())
    for p,digest in protocol['inputs'].items():
        if sha(Path(p))!=digest:raise ValueError('Input changed: '+p)
    for p,digest in provenance['code'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Code changed: '+p)
    for p,digest in provenance['artifacts'].items():
        if sha(args.artifacts/p)!=digest:raise ValueError('Artifact changed: '+p)
    for p,digest in provenance['results'].items():
        if sha(args.out/p)!=digest:raise ValueError('Report changed: '+p)
    coverage=rows(args.out/'coverage.csv');by_key={r['key']:r for r in coverage}
    xml={};measure={};midis={};midi_notes={};checks=0;maximum=0.
    def equal(a,b,tolerance=1e-10):
        nonlocal checks,maximum
        error=abs(float(a)-float(b))
        if not np.isfinite(error) or error>tolerance:raise ValueError(f'Independent mismatch: {a}/{b}; tol={tolerance}')
        checks+=1;maximum=max(maximum,error)
    for piece in sorted({r['piece'] for r in coverage}):
        work=Path(piece).parent;xml[piece],measure[piece]=symbolic(args.data/'nASAP'/work/'xml_score.musicxml')
        midi=pretty_midi.PrettyMIDI(str(args.data/'ASAP'/piece));midis[piece]=midi
        notes=defaultdict(list)
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for n in instrument.notes:notes[n.pitch].append((midi.time_to_tick(n.start)/midi.resolution,
                                                               midi.time_to_tick(n.end)/midi.resolution))
        midi_notes[piece]=notes
    xml_rows=0
    for row in records(args.artifacts/'xml_notes.csv.gz'):
        original=xml[row['piece']][row['note_id']]
        if row['xml_id']!=original['xml_id']:raise ValueError('Written XML ID differs')
        equal(Fraction(row['onset_quarters']),original['onset'])
        equal(Fraction(row['duration_quarters']),original['duration'])
        if original['pitch'] is not None:equal(Fraction(row['pitch']),original['pitch'])
        if row['is_rest']!=str(original['rest']) or row['is_grace']!=str(original['grace']):raise ValueError('XML rest/grace differs')
        xml_rows+=1
    equal(xml_rows,sum(len(ns) for ns in xml.values()))
    visits=defaultdict(list)
    for row in records(args.artifacts/'measure_visits.csv.gz'):visits[row['key'],row['part_id']].append(row)
    annotations=json.loads((args.data/'ASAP/asap_annotations.json').read_text());alignments={};note_rows=0
    inventory={r['key']:r for r in rows(ROOT/'classicfy-ai/analysis/score_structure_inputs/performances.csv')}
    quarter_cache={}
    for row in records(args.artifacts/'note_occurrences.csv.gz'):
        key=row['key'];piece=by_key[key]['piece'];n=xml[piece][row['written_note_id']]
        visit=visits[key,row['part_id']][int(row['visit_index'])]
        if key not in alignments:
            sample_row=inventory[key]
            alignments={key:raw_match(args.data/'nASAP'/sample_row['note_alignment'])}
        q=float(visit['score_start_quarters'])+float(n['offset'])
        equal(row['score_onset_quarters'],q);equal(row['written_onset_quarters'],n['onset'])
        if row['match_id']:
            source=alignments[key][row['match_id']]
            equal(source['pitch'],n['pitch']);equal(row['raw_match_onset_beats'],source['onset'])
            equal(row['raw_match_duration_beats'],source['duration'])
            if row['match_onset_quarters']:equal(row['match_onset_quarters'],q,1e-4)
            if row['match_duration_quarters']:raise ValueError('Unverified duration unit was converted')
            if row['written_xml_id']!=row['match_id'].rpartition('-')[0] and row['written_xml_id']!=row['match_id']:
                raise ValueError('Occurrence ID lost')
        if row['status'].startswith('verified'):
            equal(row['midi_onset_quarters'],q,1/midis[piece].resolution+1e-4)
            candidates=[(start,stop) for start,stop in midi_notes[piece][int(n['pitch'])]
                        if abs(start-float(row['midi_onset_quarters']))<=1e-10]
            if not candidates:raise ValueError('Verified onset is absent from independent MIDI reader')
            if row['status']=='verified':
                if not any(abs(stop-start-float(row['midi_duration_quarters']))<=1e-10 for start,stop in candidates):
                    raise ValueError('MIDI duration differs')
        edges=annotations[key]['midi_score_beats']
        midi=midis[piece]
        if key not in quarter_cache:quarter_cache[key]=[midi.time_to_tick(t)/midi.resolution for t in edges]
        quarter_edges=quarter_cache[key]
        expected=bisect_right(quarter_edges,q)-1
        if 0<=expected<len(edges)-1:
            equal(row['beat_index'],expected)
        elif row['beat_index']:raise ValueError('Out-of-range note assigned to beat')
        note_rows+=1
    seen=defaultdict(int);beats=defaultdict(list)
    for row in records(args.artifacts/'beat_map.csv.gz'):
        key=row['key'];pid=row['part_id'];piece=by_key[key]['piece'];index=seen[key,pid];seen[key,pid]+=1
        equal(row['index'],index);a=annotations[key];midi=midis[piece]
        equal(row['score_start_seconds'],a['midi_score_beats'][index]);equal(row['score_stop_seconds'],a['midi_score_beats'][index+1])
        equal(row['performance_start_seconds'],a['performance_beats'][index]);equal(row['performance_stop_seconds'],a['performance_beats'][index+1])
        equal(row['midi_start_quarters'],midi.time_to_tick(a['midi_score_beats'][index])/midi.resolution)
        equal(row['midi_stop_quarters'],midi.time_to_tick(a['midi_score_beats'][index+1])/midi.resolution)
        local=visits[key,pid];q=float(row['midi_start_quarters']);stop=float(row['midi_stop_quarters'])
        expected=bisect_right([float(v['score_start_quarters']) for v in local],q)-1
        visit=local[expected] if 0<=expected<len(local) and q<float(local[expected]['score_stop_quarters']) else None
        if visit is not None:
            equal(row['visit_index'],expected);equal(row['measure_index'],visit['measure_index'])
            equal(row['written_start_quarters'],float(visit['written_start_quarters'])+q-float(visit['score_start_quarters']))
        covered=[v for v in local if float(v['score_start_quarters'])<stop and float(v['score_stop_quarters'])>q]
        crossing=any(v['break_before']=='True' and q<float(v['score_start_quarters'])<stop for v in local)
        valid=bool(visit and stop>q and a['performance_beats'][index+1]>a['performance_beats'][index] and not crossing and
                   stop<=float(local[-1]['score_stop_quarters'])+1/midi.resolution+1e-4 and
                   all(v['status'] in ('match_midi_anchored','rest_bracketed') for v in covered))
        if (row['valid']=='True')!=valid:raise ValueError('Independent beat mask differs')
        beats[key,pid].append(row)
    # Rebuild original features without importing any bridge. This uses the
    # archived feature collector and its pre-existing normalization provenance.
    from prepare_metadata_retrieval_experiment import reconstruct
    original={}
    legacy=json.loads((ROOT/'classicfy-ai/analysis/metadata_information_design/protocol.json').read_text())
    for sequences,snapshot in reconstruct(args.data,legacy):original.update(sequences)
    feature_checks=0
    with np.load(args.artifacts/'feature_join.npz',allow_pickle=False) as cache:
        if cache['keys'].tolist()!=[r['key'] for r in coverage]:raise ValueError('Feature order changed')
        for index,row in enumerate(coverage):
            key=row['key'];x,m=original[key]
            np.testing.assert_array_equal(cache[f'{index}/values'],x)
            np.testing.assert_array_equal(cache[f'{index}/feature_mask'],m);feature_checks+=x.size+m.size
            parts=[bs for (k,p),bs in beats.items() if k==key]
            if row['status']=='failed':valid=np.zeros(len(m),bool);cuts=np.zeros(len(m),bool)
            else:
                if len(parts)!=int(row['parts']):raise ValueError('Part coverage differs')
                if any(len(bs)!=len(m) for bs in parts):raise ValueError('Original index length changed')
                valid=np.logical_and.reduce([[b['valid']=='True' for b in bs] for bs in parts])
                cuts=np.logical_or.reduce([[b['break_before']=='True' for b in bs] for bs in parts])
            np.testing.assert_array_equal(cache[f'{index}/bridge_mask'],valid)
            combined=valid&m;np.testing.assert_array_equal(cache[f'{index}/combined_mask'],combined)
            pairs=np.zeros(len(m),bool)
            type_by_time={float(t):kind for t,kind in annotations[key]['midi_score_beats_type'].items()}
            types=[type_by_time[t] for t in annotations[key]['midi_score_beats']]
            for i in range(1,len(m)):
                pairs[i]=combined[i-1] and combined[i] and not cuts[i] and 'bR' not in types[i-1:i+2]
            np.testing.assert_array_equal(cache[f'{index}/adjacent_pair_mask'],pairs)
            equal(row['valid_intervals'],sum(valid));equal(row['combined_intervals'],sum(combined))
            feature_checks+=4*len(m)
    result=dict(status='passed',performances=len(coverage),work_grids=len(xml),xml_events=xml_rows,
        note_occurrences=note_rows,beat_rows=sum(len(v) for v in beats.values()),numerical_checks=checks,
        feature_and_mask_checks=feature_checks,max_absolute_error=maximum,
        scope='Independent raw XML/match and pretty_midi tick checks; original seven-channel feature preservation; no musical style ground truth')
    (args.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
