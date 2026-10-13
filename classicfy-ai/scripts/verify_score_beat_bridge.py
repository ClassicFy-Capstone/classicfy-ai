"""Independently verify BWV 848 bridge CSVs using raw XML/regex/pretty_midi.

Does not import the bridge, MusicXML loader, match loader or feature extractor.
The original Dynamics values/masks must survive the join exactly.
"""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pretty_midi

ROOT=Path(__file__).resolve().parents[2]

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_xml(path):
    root=ET.parse(path).getroot()
    for node in root.iter():
        node.tag=node.tag.rsplit('}',1)[-1]
    notes,measures=[],[]
    step_pitch={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
    divisions=None
    position=Fraction(0)
    for index,measure in enumerate(root.find('part').findall('measure')):
        cursor,extent=Fraction(0),Fraction(0)
        previous=None
        ordinal=0
        for item in measure:
            if item.tag=='attributes' and item.findtext('divisions') is not None:
                divisions=Fraction(item.findtext('divisions'))
            elif item.tag in ('backup','forward'):
                cursor += Fraction(item.findtext('duration'))/divisions * (-1 if item.tag=='backup' else 1)
                extent=max(extent,cursor)
            elif item.tag=='note':
                grace=item.find('grace') is not None
                duration=Fraction(0) if grace else Fraction(item.findtext('duration'))/divisions
                chord=item.find('chord') is not None
                onset=previous if chord else cursor
                pitch=item.find('pitch')
                pitch_value=(12*(int(pitch.findtext('octave'))+1)+step_pitch[pitch.findtext('step')]+
                    Fraction(pitch.findtext('alter') or '0')) if pitch is not None else None
                notes.append(dict(note_id=f'P1:m{index}:n{ordinal}',xml_id=item.get('id'),
                    onset=position+onset,duration=duration,pitch=pitch_value,
                    voice=item.findtext('voice'),staff=item.findtext('staff'),
                    grace=grace,rest=item.find('rest') is not None,
                    ties={t.get('type') for t in item.findall('tie')}))
                extent=max(extent,onset+duration)
                if not chord:
                    previous=cursor
                    cursor+=duration
                ordinal+=1
        measures.append((index,measure.get('number'),position,position+extent))
        position+=extent
    return notes,measures


def raw_match(path):
    text=path.read_text()
    rate=int(re.search(r'info\(midiClockRate,(\d+)\)',text)[1])
    units=int(re.search(r'info\(midiClockUnits,(\d+)\)',text)[1])
    records={}
    pattern=re.compile(r'^snote\(([^,]+),\[([A-G]),([^\]]+)\],(\d+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),\[([^\]]*)\]\)-(.*)\.$',re.M)
    pitch_class={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
    alterations={'n':0,'#':1,'b':-1,'x':2,'##':2,'bb':-2}
    for match in pattern.finditer(text):
        identifier,step,alter,octave,bar,offset,duration,onset,stop,attrs,tail=match.groups()
        performed=None
        if tail!='deletion':
            parts=re.fullmatch(r'note\(([^,]+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\)',tail)
            if parts:
                performed=int(parts[3])*rate/units/1_000_000
            else:
                parts=re.fullmatch(r'note\(([^,]+),\[([A-G]),([^\]]+)\],(\d+),(\d+),(\d+),(\d+),(\d+)\)',tail)
                if not parts:
                    raise ValueError('Unexpected performed-note format in independent verifier')
                performed=int(parts[5])*rate/units/1_000_000
        records[identifier]=dict(pitch=12*(int(octave)+1)+pitch_class[step]+alterations[alter],
            onset=float(onset),duration=float(stop)-float(onset),performed=performed,bar=bar)
    if len(records)!=text.count('snote('):
        raise ValueError('Independent regex did not cover every score note')
    return records


def verify(out,asap_root=ROOT.parent/'datasets/ASAP',nasap_root=ROOT.parent/'datasets/nASAP',
           inventory=ROOT/'classicfy-ai/analysis/score_structure_inputs',
           dynamics=ROOT/'classicfy-ai/analysis/dynamics/beat_features.csv'):
    protocol=json.loads((out/'protocol.json').read_text())
    roots={'ASAP':asap_root,'nASAP':nasap_root,'inventory':inventory}
    inputs={}
    for label,digest in protocol['inputs'].items():
        prefix,relative=label.split('/',1)
        path=dynamics if prefix=='feature' else roots[prefix]/relative
        inputs[label]=path
        if sha(path)!=digest:
            raise ValueError('Input changed: '+label)
    work=protocol['work']
    xml_path=inputs['nASAP/'+work+'/xml_score.musicxml']
    score_path=inputs['ASAP/'+work+'/midi_score.mid']
    dynamics_path=inputs['feature/beat_features.csv']
    annotations_path=inputs['ASAP/asap_annotations.json']
    notes,measures=raw_xml(xml_path)
    by_id={n['note_id']:n for n in notes}
    by_xml={n['xml_id']:n for n in notes}
    source=pretty_midi.PrettyMIDI(str(score_path))
    midi_notes=[(n.pitch,source.time_to_tick(n.start)/source.resolution,
        (source.time_to_tick(n.end)-source.time_to_tick(n.start))/source.resolution)
        for instrument in source.instruments if not instrument.is_drum for n in instrument.notes]
    annotations=json.loads(annotations_path.read_text())
    numerical_checks=0
    max_error=0.

    def equal(a,b,tol=1e-10):
        nonlocal numerical_checks,max_error
        difference=abs(float(a)-float(b))
        if difference>tol:
            raise ValueError(f'Independent numeric mismatch: {a} != {b}; tolerance {tol}')
        numerical_checks+=1
        max_error=max(max_error,difference)

    xml_rows=rows(out/'xml_notes.csv')
    if len(xml_rows)!=len(notes):
        raise ValueError('XML event count changed')
    for row in xml_rows:
        raw=by_id[row['note_id']]
        equal(Fraction(row['onset_quarters']),raw['onset'])
        equal(Fraction(row['duration_quarters']),raw['duration'])
        if raw['pitch'] is not None:equal(Fraction(row['pitch']),raw['pitch'])
    beat_rows=rows(out/'beat_map.csv')
    edges_by_key={}
    for key in sorted({r['key'] for r in beat_rows}):
        a=annotations[key]
        edges_by_key[key]=[source.time_to_tick(t)/source.resolution for t in a['midi_score_beats']]
        rs=[r for r in beat_rows if r['key']==key]
        if [int(r['index']) for r in rs]!=list(range(len(a['midi_score_beats'])-1)):
            raise ValueError('Original beat indices changed')
        for i,row in enumerate(rs):
            # Quarter offset is 0 in this fixed work, independently established below.
            for field,want in [('score_start_seconds',a['midi_score_beats'][i]),
                ('score_stop_seconds',a['midi_score_beats'][i+1]),
                ('midi_start_quarters',edges_by_key[key][i]),
                ('midi_stop_quarters',edges_by_key[key][i+1]),
                ('xml_start_quarters',edges_by_key[key][i]),
                ('xml_stop_quarters',edges_by_key[key][i+1]),
                ('performance_start_seconds',a['performance_beats'][i]),
                ('performance_stop_seconds',a['performance_beats'][i+1])]:
                equal(row[field],want,1/source.resolution+1e-10 if 'quarters' in field else 1e-10)
            expected=next((m for m in measures if float(m[2])<=edges_by_key[key][i]<float(m[3])),None)
            if expected is None or row['measure_number']!=expected[1] or int(row['measure_index'])!=expected[0]:
                raise ValueError('Wrong written measure assignment')
    refs={}
    for label,path in inputs.items():
        if label.endswith('.match'):
            refs[work+'/'+path.stem+'.mid']=raw_match(path)
    head_for={}
    spans={}
    for raw in notes:
        if raw['pitch'] is None or raw['grace'] or 'stop' in raw['ties']:
            continue
        current=raw
        while 'start' in current['ties']:
            next_notes=[n for n in notes if n['pitch']==current['pitch'] and
                n['voice']==current['voice'] and n['staff']==current['staff'] and
                n['onset']==current['onset']+current['duration'] and 'stop' in n['ties']]
            if len(next_notes)!=1:raise ValueError('Unexpected ambiguous tie in fixed BWV 848')
            current=next_notes[0]
            head_for[current['note_id']]=raw['note_id']
        spans[raw['note_id']]=current['onset']+current['duration']-raw['onset']
    links=rows(out/'note_links.csv')
    if len(links)!=len(notes)*9:
        raise ValueError('Expected every XML event for all nine performances')
    links_by_key_id={(r['key'],r['note_id']):r for r in links}
    statuses=Counter()
    for row in links:
        key=row['key'];raw=by_id[row['note_id']]
        equal(row['score_onset_quarters'],raw['onset'])
        index=next((i for i,(left,right) in enumerate(zip(edges_by_key[key][:-1],edges_by_key[key][1:]))
                    if left<=float(raw['onset'])<right),None)
        if row['beat_index']!=('' if index is None else str(index)):
            raise ValueError('Incorrect original beat assignment for XML note')
        ref=refs[key].get(raw['xml_id']+'-1')
        if ref:
            equal(ref['onset'],raw['onset'],1e-4)
            equal(row['match_onset_quarters'],ref['onset'])
            equal(row['match_duration_quarters'],ref['duration'])
            equal(ref['pitch'],raw['pitch'])
            if not raw['grace']:
                equal(ref['duration'],spans[raw['note_id']],1e-4)
            if ref['performed'] is not None:
                equal(row['performed_onset_seconds'],ref['performed'])
                perf=annotations[key]['performance_beats']
                perf_index=next((i for i,(left,right) in enumerate(zip(perf[:-1],perf[1:]))
                                 if left<=ref['performed']<right),None)
                tolerance=float(row['performed_boundary_tolerance_seconds'])
                uncertain=min(abs(ref['performed']-t) for t in perf)<=tolerance
                if row['performed_boundary_uncertain']!=str(uncertain):
                    raise ValueError('Incorrect performed boundary uncertainty')
                if uncertain:
                    if row['performed_assignment_agrees']!='':
                        raise ValueError('Boundary-uncertain assignment must not be scored')
                elif row['performed_onset_beat_index']!=('' if perf_index is None else str(perf_index)):
                    raise ValueError('Incorrect performed assignment diagnostic')
                elif row['performed_assignment_agrees']!=str(perf_index==index):
                    raise ValueError('Incorrect performed assignment agreement')
        if raw['rest']:expected='rest_position_only'
        elif raw['grace']:expected='grace_unverified'
        elif raw['note_id'] in head_for:
            head=links_by_key_id[key,head_for[raw['note_id']]]
            expected='tie_continuation_verified' if head['status'].startswith('verified') else 'tie_head_unverified'
            if row['tie_head_id']!=head_for[raw['note_id']]:raise ValueError('Wrong tie head')
        elif ref is None:expected='missing_match'
        else:
            candidates=[n for n in midi_notes if n[0]==ref['pitch'] and
                        abs(n[1]-ref['onset'])<=protocol['midi_tolerance_quarters']]
            duration_candidates=[n for n in candidates if
                        abs(n[2]-ref['duration'])<=protocol['midi_tolerance_quarters']]
            selected=duration_candidates or candidates
            expected=('verified' if len(duration_candidates)==1 else
                      'verified_onset_duration_differs' if len(selected)==1 else
                      'missing_midi' if not selected else 'ambiguous_midi')
            if len(selected)==1:
                equal(row['midi_onset_quarters'],selected[0][1])
                equal(row['midi_duration_quarters'],selected[0][2])
        if row['status']!=expected:
            raise ValueError('Independent note status mismatch: '+row['xml_id']+' '+row['status']+' != '+expected)
        statuses[expected]+=1
    original={(r['key'],r['beat_index']):r for r in rows(dynamics_path)}
    joins=rows(out/'feature_join.csv')
    for row in joins:
        key=row['key'];i=int(row['beat_index']);source_row=original[key,row['beat_index']]
        if any(row[a]!=source_row[b] for a,b in [('dynamics_raw','raw'),('dynamics_standardized','standardized'),
                                                 ('dynamics_mask','standardized_mask'),('scale','scale')]):
            raise ValueError('Existing feature or mask changed')
        equal(row['xml_start_quarters'],edges_by_key[key][i])
        equal(row['xml_stop_quarters'],edges_by_key[key][i+1])
    # Recompute raw Dynamics with scalar MIDI-note loops, without the extractor.
    for key in refs:
        path=inputs['ASAP/'+key]
        performance=pretty_midi.PrettyMIDI(str(path))
        notes_performed=[n for inst in performance.instruments if not inst.is_drum for n in inst.notes]
        edges=annotations[key]['performance_beats']
        for i,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
            velocity=[n.velocity for n in notes_performed if left<=n.start<right]
            saved=original[key,str(i)]
            equal(saved['onset_count'],len(velocity))
            equal(saved['raw'],sum(velocity)/len(velocity)/127)
    coverage=rows(out/'coverage.csv')
    if len(coverage)!=9 or {r['key'] for r in coverage}!=set(refs):
        raise ValueError('Coverage table cohort changed')
    for row in coverage:
        local_links=[r for r in links if r['key']==row['key']]
        local_beats=[r for r in beat_rows if r['key']==row['key']]
        equal(row['original_intervals'],len(local_beats))
        equal(row['valid_intervals'],len(local_beats))
        expected_counts=Counter(r['status'] for r in local_links)
        if json.loads(row['note_statuses'])!=dict(expected_counts):
            raise ValueError('Coverage note counts changed')
        counts=[]
        for beat in local_beats:
            supported=sum(r['status'].startswith('verified') and r['beat_index']==beat['index'] for r in local_links)
            equal(beat['verified_onsets'],supported)
            if beat['valid']!='True' or beat['status']!='coordinate_verified':
                raise ValueError('Unexpected invalid/irregular interval in fixed BWV 848')
            counts.append(supported)
        equal(row['minimum_verified_onsets'],min(counts))
        equal(row['outside_onsets'],sum(r['beat_index']=='' for r in local_links))
        equal(row['performed_boundary_uncertain'],sum(r['performed_boundary_uncertain']=='True' for r in local_links))
        equal(row['performed_assignment_agrees'],sum(r['performed_assignment_agrees']=='True' for r in local_links))
        equal(row['performed_assignment_available'],sum(r['performed_assignment_agrees']!='' for r in local_links))
    result=dict(status='passed',numerical_checks=numerical_checks,max_absolute_error=max_error,
        input_hashes_checked=len(inputs),xml_events=len(notes),note_link_rows=len(links),
        beat_map_rows=len(beat_rows),feature_join_rows=len(joins),note_status_counts=dict(statuses),
        method='Independent raw XML quarters, regex match, pretty_midi ticks and scalar half-open assignment',
        scope='Fixed BWV 848; does not validate repeat handling or other works',
        code_sha256=sha(Path(__file__)),
        csv_sha256={name:sha(out/name) for name in ('xml_notes.csv','note_links.csv','beat_map.csv','feature_join.csv','coverage.csv')})
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--asap-root',type=Path,default=ROOT.parent/'datasets/ASAP')
    parser.add_argument('--nasap-root',type=Path,default=ROOT.parent/'datasets/nASAP')
    parser.add_argument('--inventory',type=Path,default=ROOT/'classicfy-ai/analysis/score_structure_inputs')
    parser.add_argument('--dynamics',type=Path,default=ROOT/'classicfy-ai/analysis/dynamics/beat_features.csv')
    args=parser.parse_args()
    print(json.dumps(verify(args.out,args.asap_root,args.nasap_root,args.inventory,args.dynamics),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
