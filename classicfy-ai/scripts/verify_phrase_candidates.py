"""Independent raw-score evidence and original-index boundary mapping checks.

No production phrase extractor, MusicXML loader or score bridge is imported.
Checks evidence arithmetic; does not measure musical phrase recognition.
"""

import argparse
from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pretty_midi

from verify_score_structure_full import symbolic
from verify_score_beat_bridge import rows,sha

ROOT=Path(__file__).resolve().parents[2]


def source(path):
    notes,measures=symbolic(path);root=ET.parse(path).getroot()
    for n in root.iter():n.tag=n.tag.rsplit('}',1)[-1]
    keys=[];marks=[]
    for part in root.findall('part'):
        pid=part.get('id');division=None
        for index,measure in enumerate(part.findall('measure')):
            cursor=Fraction(0);ordinal=0
            for item in measure:
                if item.tag=='attributes':
                    if item.findtext('divisions') is not None:division=Fraction(item.findtext('divisions'))
                    for key in item.findall('key'):
                        keys.append(dict(part_id=pid,onset=measures[pid,index]['onset']+cursor,
                            fifths=int(key.findtext('fifths')) if key.findtext('fifths') is not None else None,
                            mode=key.findtext('mode'),staff=key.get('number')))
                elif item.tag in ('backup','forward'):
                    cursor+=Fraction(item.findtext('duration'))/division*(-1 if item.tag=='backup' else 1)
                elif item.tag=='note':
                    n=notes[f'{pid}:m{index}:n{ordinal}'];ordinal+=1
                    n.update(voice=item.findtext('voice'),staff=int(item.findtext('staff')) if item.findtext('staff') else None,
                        cue=item.find('cue') is not None,notations={e.tag for e in item.findall('notations/*')},
                        nested_notations={e.tag for node in item.findall('notations') for e in node.iter()},
                        slurs=[(s.get('type'),s.get('number','1')) for s in item.findall('notations/slur')])
                    if item.find('chord') is None:cursor+=n['duration']
                elif item.tag=='barline':
                    q=(measures[pid,index]['onset'] if item.get('location')=='left' else
                       measures[pid,index]['onset']+cursor if item.get('location')=='middle' else
                       measures[pid,index]['onset']+measures[pid,index]['duration'])
                    marks.extend((pid,q,n.tag) for n in item if n.tag in ('repeat','ending','bar-style'))
    return notes,measures,keys,marks


def scope(note):return f'{note["part_id"]}:staff={note["staff"]}:voice={note["voice"]}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent/'datasets')
    parser.add_argument('--inventory',type=Path,default=ROOT/'classicfy-ai/analysis/score_structure_inputs')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/phrase_candidates')
    parser.add_argument('--artifacts',type=Path)
    args=parser.parse_args();protocol=json.loads((args.out/'protocol.json').read_text())
    if args.artifacts is None:
        location=protocol.get('artifact_location',{})
        args.artifacts=(args.out/location['relative_to_report'] if 'relative_to_report' in location else
                        ROOT.parent/'datasets/phrase_candidates_run01')
    provenance=json.loads((args.out/'provenance.json').read_text())
    for label,digest in protocol['inputs'].items():
        prefix,path=label.split('/',1);base=args.inventory if prefix=='inventory' else args.data/prefix
        if sha(base/path)!=digest:raise ValueError('Input changed: '+label)
    for path,digest in provenance['code'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Code changed: '+path)
    for path,digest in provenance['results'].items():
        if sha(args.out/path)!=digest:raise ValueError('Result changed: '+path)
    for path,digest in provenance['artifacts'].items():
        if sha(args.artifacts/path)!=digest:raise ValueError('Artifact changed: '+path)
    checks=0;count=0;cue_count=0;maximum=0.;pilot_candidates={};sources={}
    def equal(a,b,tol=1e-10):
        nonlocal checks,maximum
        error=abs(float(Fraction(a))-float(Fraction(b)))
        if error>tol:raise ValueError(f'Independent mismatch: {a}/{b}')
        checks+=1;maximum=max(maximum,error)
    with gzip.open(args.artifacts/'all_scores.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            result=json.loads(line);piece=result['piece'];work=Path(piece).parent
            notes,measures,keys,marks=source(args.data/'nASAP'/work/'xml_score.musicxml')
            positive=[n for n in notes.values() if n['pitch'] is not None and not n['grace'] and not n['cue']]
            if piece in protocol['pilot_works']:
                sources[piece]=(notes,measures)
                pilot_candidates[piece]={c['candidate_id']:c for c in result['candidates']}
            for s in result['slurs']:
                if s['status'].startswith('paired'):
                    first,last=notes[s['start_note_id']],notes[s['stop_note_id']]
                    if ('start',s['number']) not in first['slurs'] or ('stop',s['number']) not in last['slurs']:
                        raise ValueError('Slur references do not match raw symbols')
                    equal(s['start_quarters'],first['onset']);equal(s['stop_onset_quarters'],last['onset'])
                    equal(s['stop_release_quarters'],last['onset']+last['duration'])
                    if last['onset']<=first['onset']:raise ValueError('Nonpositive slur became paired')
            for c in result['candidates']:
                if c['status']!='pending_score_review':raise ValueError('Unreviewed boundary became active')
                q=Fraction(c['position_quarters']);lo=Fraction(c['uncertainty_start_quarters']);hi=Fraction(c['uncertainty_stop_quarters'])
                if not lo<=q<=hi:raise ValueError('Candidate outside its uncertainty interval')
                for cue in c['cues']:
                    cue_count+=1;equal(cue['position_quarters'],q)
                    local=[notes[n] for n in cue['note_ids']];kind=cue['kind'];details=cue['details']
                    if kind=='slur_end':
                        equal(q,local[-1]['onset']+local[-1]['duration'])
                        equal(details['source_stop_onset'],local[-1]['onset'])
                    elif kind in ('voice_rest','voice_entry_rest','voice_exit_rest'):
                        rest=local[0]
                        if not rest['rest'] or rest['duration']<Fraction(1,2):raise ValueError('Invalid rest cue')
                        equal(q,rest['onset']);equal(cue['uncertainty_stop_quarters'],rest['onset']+rest['duration'])
                        if any(scope(n)==cue['scope'] and n['onset']<rest['onset']+rest['duration'] and
                               n['onset']+n['duration']>rest['onset'] for n in positive):
                            raise ValueError('Voice rest overlaps a pitched note')
                    elif kind=='global_notated_silence':
                        stop=Fraction(cue['uncertainty_stop_quarters'])
                        if stop-q<Fraction(1,2) or any(n['onset']<stop and n['onset']+n['duration']>q for n in positive):
                            raise ValueError('Global silence overlaps another voice')
                    elif kind.startswith('notation_'):
                        symbol=kind[len('notation_'):].replace('_','-')
                        if symbol not in local[0]['nested_notations']:raise ValueError('Missing raw notation symbol')
                        equal(q,local[0]['onset']+local[0]['duration'])
                    elif kind in ('navigation_boundary','section_barline'):
                        if not any(pid==cue['part_id'] and position==q for pid,position,tag in marks):
                            raise ValueError('Missing raw barline evidence')
                    elif kind=='harmonic_closure_hint':
                        arrival=Fraction(details['arrival_onset_quarters']);before=Fraction(details['previous_onset_quarters'])
                        now=[n for n in positive if n['part_id']==cue['part_id'] and n['onset']<=arrival<n['onset']+n['duration']]
                        prev=[n for n in positive if n['part_id']==cue['part_id'] and n['onset']<=before<n['onset']+n['duration']]
                        pcs={int(n['pitch'])%12 for n in now};old={int(n['pitch'])%12 for n in prev}
                        if pcs!=set(details['arrival_pitch_classes']) or old!=set(details['previous_pitch_classes']):
                            raise ValueError('Raw harmonic sonority differs')
                        equal(min(n['pitch'] for n in prev),details['bass_from']);equal(min(n['pitch'] for n in now),details['bass_to'])
                        tonic=details['tonic_pitch_class'];dominant=(tonic+7)%12
                        if not {dominant,(dominant+4)%12,(dominant+7)%12}<=old:raise ValueError('Dominant triad absent')
                        if not {tonic,(tonic+(3 if details['mode']=='minor' else 4))%12,(tonic+7)%12}<=pcs:raise ValueError('Tonic triad absent')
                        current=[k for k in keys if k['part_id']==cue['part_id'] and k['onset']<=arrival and k['staff'] in (None,'1')][-1]
                        expected_mode_source='explicit_musicxml' if current['mode'] else 'unconfirmed_key_signature_hypothesis'
                        if details['mode_source']!=expected_mode_source:raise ValueError('Unknown mode silently confirmed')
                        equal(tonic,(7*current['fifths']+(9 if details['mode']=='minor' else 0))%12)
                        equal(q,min(n['onset']+n['duration'] for n in now if n['onset']==arrival))
                    else:raise ValueError('Unsupported cue in verifier: '+kind)
                count+=1
    annotations=json.loads((args.data/'ASAP/asap_annotations.json').read_text());midis={};visits=defaultdict(list)
    for row in rows(args.out/'pilot_measure_visits.csv'):visits[row['key'],row['part_id']].append(row)
    for piece in protocol['pilot_works']:midis[piece]=pretty_midi.PrettyMIDI(str(args.data/'ASAP'/piece))
    for (key,pid),local in visits.items():
        piece=local[0]['piece'];midi=midis[piece];mapping=annotations[key]['downbeats_score_map']
        dbq=[midi.time_to_tick(t)/midi.resolution for t in annotations[key]['midi_score_downbeats']]
        if isinstance(mapping,list):
            lookup={int(i):q for i,q in zip(mapping,dbq)}
            for v in local:
                index=int(v['measure_index'])
                if index in lookup:equal(v['score_start_quarters'],lookup[index])
        else:
            equal(len(local),len(dbq))
            for v,q in zip(local,dbq):equal(v['score_start_quarters'],q)
    occurrence_count=0
    for row in rows(args.out/'boundary_occurrences.csv'):
        piece=row['piece'];c=pilot_candidates[piece][row['candidate_id']]
        local=visits[row['key'],row['part_id']];visit=local[int(row['visit_index'])]
        expected=float(visit['score_start_quarters'])+float(Fraction(c['position_quarters']))-float(visit['written_start_quarters'])
        equal(row['observed_score_quarters'],expected);equal(row['visit_number'],visit['visit_number'])
        if row['source_occurrence_label']!=visit['source_occurrence_label']:raise ValueError('Occurrence label lost')
        midi=midis[piece];edges=[midi.time_to_tick(t)/midi.resolution for t in annotations[row['key']]['midi_score_beats']]
        nearest=min(range(len(edges)),key=lambda i:abs(edges[i]-expected));on_edge=abs(edges[nearest]-expected)<=1/midi.resolution+1e-4
        equal(row['edge_error_quarters'],abs(edges[nearest]-expected))
        if on_edge:equal(row['original_edge_index'],nearest)
        elif row['original_edge_index']:raise ValueError('Inside-interval boundary was snapped to an edge')
        index=nearest if on_edge else bisect_right(edges,expected)-1
        if 0<=index<len(edges)-1:equal(row['original_interval_index'],index)
        elif row['original_interval_index']:raise ValueError('Out-of-range original index')
        if row['eligible_for_embedding']!='False' or row['boundary_status']!='pending_score_review':
            raise ValueError('Unconfirmed phrase became eligible')
        occurrence_count+=1
    verification=dict(status='passed',work_grids=55,candidates=count,cues=cue_count,
        pilot_boundary_occurrences=occurrence_count,numerical_checks=checks,max_absolute_error=maximum,
        scope='Independent score evidence and original-index mapping arithmetic; no phrase ground-truth accuracy')
    (args.out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
