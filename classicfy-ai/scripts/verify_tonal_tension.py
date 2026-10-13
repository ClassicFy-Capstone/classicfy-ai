"""Independent scalar Spiral Array check from raw XML and saved intervals."""

import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from verify_score_structure_full import symbolic
from validate_score_beat_bridge import rows,sha

ROOT=Path(__file__).resolve().parents[2]
H=math.sqrt(2/15)*math.pi**2/4
W=[516/999,315/999,168/999]


def point(k):return (math.sin(k*math.pi/2),math.cos(k*math.pi/2),H*k)


def norm(a,b):return math.sqrt(math.fsum((u-v)**2 for u,v in zip(a,b)))


def blend(points,weights):return tuple(math.fsum(w*p[j] for w,p in zip(weights,points)) for j in range(3))


def chord(k,minor):return blend([point(k),point(k+1),point(k-3 if minor else k+4)],W)


def key_center(fifths,minor):
    k=fifths+3 if minor else fifths
    if minor:
        dominant=blend([chord(k+1,False),chord(k+1,True)],[.75,.25])
        subdominant=blend([chord(k-1,True),chord(k-1,False)],[.75,.25])
    else:dominant=chord(k+1,False);subdominant=chord(k-1,False)
    return blend([chord(k,minor),dominant,subdominant],W)


def correlation(a,b):
    am=math.fsum(a)/len(a);bm=math.fsum(b)/len(b)
    aa=[x-am for x in a];bb=[x-bm for x in b]
    return math.fsum(x*y for x,y in zip(aa,bb))/math.sqrt(math.fsum(x*x for x in aa)*math.fsum(x*x for x in bb))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/tonal_tension')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/tonal_tension_run01')
    parser.add_argument('--source',type=Path,default=ROOT.parent/'datasets/phrase_response_run02')
    parser.add_argument('--data',type=Path,default=ROOT.parent/'datasets')
    args=parser.parse_args();prov=json.loads((args.out/'provenance.json').read_text())
    for name,digest in prov['inputs'].items():
        if sha(Path(name))!=digest:raise AssertionError('Input hash differs: '+name)
    for name,digest in prov['code'].items():
        if sha(ROOT/name)!=digest:raise AssertionError('Code hash differs: '+name)
    for name,digest in prov['results'].items():
        if sha(args.out/name)!=digest:raise AssertionError('Result hash differs: '+name)
    for name,digest in prov['artifacts'].items():
        if sha(args.artifacts/name)!=digest:raise AssertionError('Artifact hash differs: '+name)
    with gzip.open(args.artifacts/'clouds.csv.gz','rt') as stream:clouds=list(csv.DictReader(stream))
    contexts=defaultdict(list)
    for row in rows(args.out/'key_contexts.csv'):contexts[row['piece']].append(row)
    raw={};checks=0;maximum=0.;verified_clouds={}
    def equal(a,b):
        nonlocal checks,maximum
        checks+=1
        if math.isnan(a) and math.isnan(b):return
        if not math.isfinite(a) or not math.isfinite(b):raise AssertionError((a,b))
        difference=abs(a-b);maximum=max(maximum,difference)
        if difference>1e-10:raise AssertionError((a,b,difference))
    for piece,key_rows in contexts.items():
        path=args.data/'nASAP'/Path(piece).parent/'xml_score.musicxml'
        notes,measures=symbolic(path);tree=ET.parse(path).getroot()
        for node in tree.iter():node.tag=node.tag.rsplit('}',1)[-1]
        for part in tree.findall('part'):
            for m,measure in enumerate(part.findall('measure')):
                for n,node in enumerate(measure.findall('note')):
                    note=notes[f'{part.get("id")}:m{m}:n{n}'];pitch=node.find('pitch')
                    note['cue']=node.find('cue') is not None
                    if pitch is not None:
                        note['fifth']=dict(F=-1,C=0,G=1,D=2,A=3,E=4,B=5)[pitch.findtext('step')]+7*float(pitch.findtext('alter') or '0')
        raw[piece]=[n for n in notes.values() if n['pitch'] is not None and n['duration']>0 and not n['cue']]
        for note in raw[piece]:
            note['on_float']=float(note['onset']);note['off_float']=float(note['onset']+note['duration'])
        # Independent duration histogram and signature-compatible mode scores.
        for key in key_rows:
            if not key['major_correlation']:continue
            a,b=float(key['start']),float(key['stop']);f=int(key['fifths']);hist=[0.]*12
            for note in raw[piece]:
                length=max(0.,min(b,float(note['onset']+note['duration']))-max(a,float(note['onset'])))
                hist[int(note['pitch'])%12]+=length
            major=[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
            minor=[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
            root=7*f%12
            cm=correlation(hist,[major[(j-root)%12] for j in range(12)])
            cn=correlation(hist,[minor[(j-root-9)%12] for j in range(12)])
            equal(cm,float(key['major_correlation']));equal(cn,float(key['minor_correlation']))
            margin=abs(cm-cn);equal(margin,float(key['margin']))
            expected=('major' if cm>cn else 'minor') if margin>=.1 else ''
            if expected!=key['mode']:raise AssertionError('Mode hypothesis differs')
    for row in clouds:
        piece=row['piece'];a,b=float(row['start']),float(row['stop']);active=[]
        for note in raw[piece]:
            overlap=min(b,note['off_float'])-max(a,note['on_float'])
            if overlap>0:active.append((point(note['fifth']),overlap))
        diameter=strain=strain_major=strain_minor=math.nan;center=None
        if active:
            points,weights=zip(*active);total=math.fsum(weights)
            center=blend(points,[w/total for w in weights])
            diameter=max((norm(points[i],points[j]) for i in range(len(points)) for j in range(i)),default=0.)/(12*H)
            for j,name in enumerate(['center_x','center_y','center_z']):equal(center[j],float(row[name]))
            equal(total,float(row['weight']));equal(len(active),float(row['notes']))
            ki=int(row['key_index'])
            if ki>=0:
                key=contexts[piece][ki]
                if key['fifths']:
                    fifths=int(key['fifths']);strain_major=norm(center,key_center(fifths,False))/(12*H)
                    strain_minor=norm(center,key_center(fifths,True))/(12*H)
                    if key['mode']:strain=strain_major if key['mode']=='major' else strain_minor
        for name,value in [('diameter',diameter),('strain',strain),('strain_major',strain_major),('strain_minor',strain_minor)]:equal(value,float(row[name]))
        verified_clouds[piece,a,b]=(center,diameter,strain)
    z=np.load(args.source/'sequences.npz',allow_pickle=False);t=np.load(args.artifacts/'tension.npz',allow_pickle=False)
    coverage=rows(ROOT/'classicfy-ai/analysis/phrase_response/coverage.csv')
    with gzip.open(args.out/'responses.csv.gz','rt',encoding='utf-8-sig') as stream:
        responses={(r['key'],r['metric'],r['view']):r for r in csv.DictReader(stream)}
    metric_names=['cloud_diameter','cloud_momentum','tensile_strain']
    for i,row in enumerate(coverage):
        previous=None;mask=z[f'{i}/mask'];cuts=z[f'{i}/breaks']
        for j,(a,b) in enumerate(z[f'{i}/written']):
            if math.isfinite(a) and b>a:
                center,diameter,strain=verified_clouds[row['piece'],float(a),float(b)]
            else:center=None;diameter=strain=math.nan
            momentum=math.nan
            if center is not None and previous is not None and not cuts[j] and j>0 and mask[j-1] and mask[j]:
                momentum=norm(center,previous)/(12*H)
            for c,v in enumerate([diameter,momentum,strain]):equal(v,float(t[f'{i}/tension'][c,j]))
            previous=center
        x=z[f'{i}/values'];base_pairs=z[f'{i}/pairs'];key_ids=t[f'{i}/key_index']
        for c,name in enumerate(metric_names):
            values=t[f'{i}/tension'][c];finite=sorted(float(v) for v in values if math.isfinite(v))
            thresholds=[]
            for q in (1/3,2/3):
                at=(len(finite)-1)*q;lo=math.floor(at);hi=math.ceil(at)
                thresholds.append(finite[lo]+(finite[hi]-finite[lo])*(at-lo) if finite else math.nan)
            labels=[-1 if not math.isfinite(v) else 0 if v<=thresholds[0] else 1 if v<=thresholds[1] else 2 for v in values]
            if labels!=t[f'{i}/labels'][c].tolist():raise AssertionError('Tension tertile assignment differs')
            for view in ('whole','A','B'):
                selected=[bool(mask[j]) and labels[j]>=0 and (view=='whole' or (j//64%2=='AB'.index(view) and j<len(mask)//64*64)) for j in range(len(mask))]
                expected=[]
                for label in range(3):
                    ids=[j for j in range(len(mask)) if selected[j] and labels[j]==label]
                    ps=[j for j in ids if j and selected[j-1] and base_pairs[j] and labels[j-1]==label
                        and (c!=2 or key_ids[j-1]==key_ids[j]) and (view=='whole' or j%64!=0)]
                    expected.extend([math.fsum(float(x[ch,j]) for j in ids)/len(ids) if len(ids)>=8 else math.nan for ch in range(7)])
                    expected.extend([math.fsum(float(x[ch,j]-x[ch,j-1]) for j in ps)/len(ps) if len(ps)>=4 else math.nan for ch in range(7)])
                    expected.extend([math.sqrt(math.fsum(float(x[ch,j]-x[ch,j-1])**2 for j in ps)/len(ps)) if len(ps)>=4 else math.nan for ch in range(7)])
                    expected.append(math.fsum(float((x[0,j]-x[0,j-1])*(x[2,j]-x[2,j-1])) for j in ps)/len(ps) if len(ps)>=4 else math.nan)
                for j,v in enumerate(expected):equal(v,float(responses[row['key'],name,view][f'h{j}']))
    result=dict(status='passed',score_clouds=len(clouds),performances=len(coverage),numeric_checks=checks,
        maximum_absolute_error=maximum,scope='Independent raw XML spelling/positive duration overlap, scalar helix/chord/key math, duration-profile mode correlations, chronological reset momentum, all projected tension values/tertiles and scalar responses on both disjoint views',
        code_sha256=sha(Path(__file__)))
    (args.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
