"""Check repeat routing against raw match Bar:Beat, XML and pretty_midi ticks.

No occurrence/beat bridge or production loaders are imported. Existing
independent XML/match readers are reused, along with the fixed source cache.
"""

import argparse
from collections import Counter,defaultdict
from itertools import groupby
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pretty_midi

from verify_score_beat_bridge import raw_xml,raw_match,rows,sha

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--asap-root',type=Path,default=ROOT.parent/'datasets/ASAP')
    parser.add_argument('--nasap-root',type=Path,default=ROOT.parent/'datasets/nASAP')
    parser.add_argument('--inventory',type=Path,default=ROOT/'classicfy-ai/analysis/score_structure_inputs')
    parser.add_argument('--cache',type=Path,default=ROOT.parent/'datasets/feature_normalization_standardized.npz')
    args=parser.parse_args()
    protocol=json.loads((args.out/'protocol.json').read_text());work=protocol['work']
    roots={'ASAP':args.asap_root,'nASAP':args.nasap_root,'inventory':args.inventory}
    for label,digest in protocol['inputs'].items():
        prefix,name=label.split('/',1)
        path=args.cache if prefix=='cache' else roots[prefix]/name
        assert sha(path)==digest,label
    notes,measures=raw_xml(args.nasap_root/work/'xml_score.musicxml')
    by_xml={n['xml_id']:n for n in notes};by_id={n['note_id']:n for n in notes}
    xml_rows=rows(args.out/'xml_notes.csv')
    assert len(xml_rows)==len(notes)
    visits=rows(args.out/'measure_visits.csv');links=rows(args.out/'note_occurrences.csv')
    beats=rows(args.out/'beat_map.csv');features=rows(args.out/'feature_join.csv')
    keys=sorted({v['key'] for v in visits})
    assert len(keys)==protocol['performances']==6
    annotations=json.loads((args.asap_root/'asap_annotations.json').read_text())
    midi=pretty_midi.PrettyMIDI(str(args.asap_root/work/'midi_score.mid'))
    numeric=0;error=0.
    def equal(a,b):
        nonlocal numeric,error
        x,y=float(a),float(b)
        if np.isnan(x) and np.isnan(y):
            numeric+=1;return
        diff=abs(x-y)
        assert diff<=1e-10,(a,b)
        error=max(error,diff);numeric+=1
    # Validate jump directions from raw barline elements, separate from the
    # production repeat resolver. Direct Bar:Beat traversal checks visit order.
    raw_root=ET.parse(args.nasap_root/work/'xml_score.musicxml').getroot()
    for node in raw_root.iter():node.tag=node.tag.rsplit('}',1)[-1]
    marks=[]
    for index,m in enumerate(raw_root.find('part').findall('measure')):
        for n in m.findall('barline/repeat'):marks.append((index,n.get('direction')))
    repeat_edges={(i,max(j for j,d in marks if d=='forward' and j<=i)) for i,d in marks if d=='backward'}
    cache_records={}
    with np.load(args.cache,allow_pickle=False) as cache:
        manifest=json.loads(str(cache['manifest']))
        for index,r in enumerate(manifest['records']):
            if r['key'] in keys and r['channel']=='dynamics':
                cache_records[r['key']]={name:cache[f'{index}/{name}'].copy() for name in
                    ('raw_values','raw_mask','common_values','standardized_values','standardized_mask')}
    assert set(cache_records)==set(keys)
    transitions=Counter()
    for key in keys:
        local=[v for v in visits if v['key']==key]
        refs=raw_match(args.nasap_root/work/(Path(key).stem+'.match'))
        source_route=[]
        for source_id,ref in sorted(refs.items(),key=lambda pair:(pair[1]['onset'],pair[0])):
            identifier,label=source_id.rsplit('-',1);n=by_xml[identifier]
            equal(ref['pitch'],n['pitch'])
            measure_index=int(n['note_id'].split(':')[1][1:])
            assert ref['bar'].split(':')[0]==measures[measure_index][1],source_id
            source_route.append((ref['bar'].split(':')[0],label))
        route=[k for k,g in groupby(source_route)]
        anchored=[(v['measure_number'],v['source_occurrence_label']) for v in local if v['status']=='match_midi_anchored']
        assert anchored==route,'Observed raw Bar:Beat route differs'
        seen=Counter()
        for i,v in enumerate(local):
            equal(v['index'],i)
            index=int(v['measure_index']);seen[index]+=1
            equal(v['visit_number'],seen[index])
            equal(v['written_start_quarters'],measures[index][2])
            equal(float(v['score_stop_quarters'])-float(v['score_start_quarters']),measures[index][3]-measures[index][2])
            if v['status']=='rest_bracketed':
                assert all(n['rest'] for n in notes if int(n['note_id'].split(':')[1][1:])==index)
                assert 0<i<len(local)-1
                equal(v['score_start_quarters'],local[i-1]['score_stop_quarters'])
                equal(v['score_stop_quarters'],local[i+1]['score_start_quarters'])
            if i:
                before=int(local[i-1]['measure_index']);transition=v['transition']
                if index<=before:assert transition=='repeat_jump' and (before,index) in repeat_edges
                elif index>before+1:
                    assert transition=='ending_skip'
                    raw_measures=raw_root.find('part').findall('measure')
                    assert any(n.get('number')=='1' and n.get('type')=='start' for n in raw_measures[before+1].findall('barline/ending'))
                    assert any(n.get('number')=='2' and n.get('type')=='start' for n in raw_measures[index].findall('barline/ending'))
                else:assert transition=='consecutive'
                assert v['break_before']==str(index!=before+1)
            transitions[v['transition']]+=1
        local_links=[r for r in links if r['key']==key]
        assert len({r['note_id'] for r in local_links})==len(local_links)
        expected=sum(sum(int(n['note_id'].split(':')[1][1:])==int(v['measure_index']) for n in notes) for v in local)
        equal(len(local_links),expected)
        for r in local_links:
            n=by_id[r['written_note_id']];v=local[int(r['visit_index'])]
            equal(r['written_onset_quarters'],n['onset'])
            position=float(v['score_start_quarters'])+float(n['onset']-measures[int(v['measure_index'])][2])
            equal(r['score_onset_quarters'],position)
            if r['match_id']:
                ref=refs[r['match_id']]
                assert r['written_xml_id']==r['match_id'].rsplit('-',1)[0]
                equal(ref['onset'],position)
                equal(ref['duration'],r['match_duration_quarters'])
        edges=[midi.time_to_tick(t)/midi.resolution for t in annotations[key]['midi_score_beats']]
        local_beats=[r for r in beats if r['key']==key]
        assert [int(r['index']) for r in local_beats]==list(range(len(edges)-1))
        computed_breaks=[]
        for i,r in enumerate(local_beats):
            equal(r['midi_start_quarters'],edges[i]);equal(r['midi_stop_quarters'],edges[i+1])
            v=next(v for v in local if float(v['score_start_quarters'])<=edges[i]<float(v['score_stop_quarters']))
            equal(r['visit_index'],v['index']);equal(r['written_measure_index'],v['measure_index'])
            equal(r['visit_number'],v['visit_number'])
            equal(r['written_start_quarters'],float(v['written_start_quarters'])+edges[i]-float(v['score_start_quarters']))
            boundary=v['break_before']=='True' and abs(edges[i]-float(v['score_start_quarters']))<1e-10
            assert r['break_before']==str(boundary)
            if boundary:computed_breaks.append(i)
            assert r['valid']=='True'
            equal(r['performance_start_seconds'],annotations[key]['performance_beats'][i])
            equal(r['performance_stop_seconds'],annotations[key]['performance_beats'][i+1])
        assert computed_breaks==protocol['original_break_indices']
        saved=cache_records[key]
        for r in (q for q in features if q['key']==key):
            i=int(r['beat_index'])
            for field,name in [('dynamics_raw','raw_values'),('dynamics_common','common_values'),
                               ('dynamics_standardized','standardized_values')]:equal(r[field],saved[name][i])
            assert r['dynamics_mask']==str(bool(saved['standardized_mask'][i]))
            expected_pair=i>0 and i not in computed_breaks and saved['standardized_mask'][i] and saved['standardized_mask'][i-1]
            assert r['adjacent_pair_valid']==str(bool(expected_pair))
    result=dict(status='passed',scope='Independent repeat route/occurrences/original beats/cache preservation for 31-2',
        performances=len(keys),measure_visits=len(visits),note_occurrences=len(links),beat_rows=len(beats),
        numeric_checks=numeric,max_absolute_error=error,transition_counts=dict(transitions),
        code_sha256=sha(Path(__file__)),csv_sha256={p.name:sha(p) for p in args.out.glob('*.csv')})
    (args.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
