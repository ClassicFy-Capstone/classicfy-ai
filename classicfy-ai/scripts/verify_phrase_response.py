"""Scalar verification of saved phrase coordinates and residual responses.

Does not import the production phrase or response implementation. Original
feature digests are compared to the preceding full-cohort connection report.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pretty_midi

from validate_score_beat_bridge import rows, sha
from verify_score_structure_full import symbolic

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/phrase_response')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/phrase_response_run02')
    parser.add_argument('--data',type=Path,default=ROOT.parent/'datasets')
    args=parser.parse_args()
    provenance=json.loads((args.out/'provenance.json').read_text())
    for name,digest in provenance['inputs'].items():
        if sha(Path(name))!=digest:raise ValueError('Input hash changed: '+name)
    for name,digest in provenance['code'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Code hash changed: '+name)
    for name,digest in provenance['results'].items():
        if sha(args.out/name)!=digest:raise ValueError('Result hash changed: '+name)
    if sha(args.artifacts/'sequences.npz')!=provenance['artifact_sha256']:raise ValueError('Sequence hash changed')
    if sha(args.artifacts/'visit_geometry.json')!=provenance['visit_geometry_sha256']:raise ValueError('Visit hash changed')
    z=np.load(args.artifacts/'sequences.npz',allow_pickle=False)
    visits=json.loads((args.artifacts/'visit_geometry.json').read_text())
    old=json.loads((ROOT/'classicfy-ai/analysis/score_structure_full/protocol.json').read_text())['feature_digests']
    old_counts={r['key']:r for r in rows(ROOT/'classicfy-ai/analysis/score_structure_full/coverage.csv')}
    coverage=rows(args.out/'coverage.csv');segments={};actual={}
    for r in rows(args.out/'phrase_hypotheses.csv'):
        segments.setdefault(r['piece'],[]).append((float(Fraction(r['start'])),float(Fraction(r['stop']))))
    for r in rows(args.out/'responses.csv'):actual[r['key'],r['view']]=r
    checks=0;error=0.;midi={};xml={};duration_differences=set()
    def equal(a,b):
        nonlocal checks,error
        checks+=1
        if math.isnan(a) and math.isnan(b):return
        if not math.isfinite(a) or not math.isfinite(b):raise AssertionError((a,b))
        difference=abs(a-b);error=max(error,difference)
        if difference>1e-10:raise AssertionError((a,b,difference))
    for i,r in enumerate(coverage):
        key=r['key'];piece=r['piece'];x=z[f'{i}/values'];mask=z[f'{i}/mask'];pairs=z[f'{i}/pairs'];written=z[f'{i}/written']
        digest=hashlib.sha256(x.tobytes()+z[f'{i}/feature_mask'].tobytes()).hexdigest()
        if digest!=old[key]:raise AssertionError('Original seven-channel feature changed: '+key)
        if int(mask.sum())!=int(old_counts[key]['combined_intervals']):raise AssertionError('Combined mask count differs')
        if piece not in midi:
            midi[piece]=pretty_midi.PrettyMIDI(str(args.data/'ASAP'/piece))
            xml[piece]=symbolic(args.data/'nASAP'/Path(piece).parent/'xml_score.musicxml')[1]
        geometry=visits[str(z[f'{i}/geometry'])];vs=geometry['visits'];starts=[v['score_start_quarters'] for v in vs]
        pid=geometry['part_id'];edges=z[f'{i}/score_quarters']
        for v in vs:
            measure=xml[piece][pid,v['measure_index']]
            equal(v['written_start_quarters'],float(measure['onset']))
            duration=v['score_stop_quarters']-v['score_start_quarters']
            if abs(duration-float(measure['duration']))>1e-10:
                duration_differences.add((piece,v['measure_index'],v['score_start_quarters'],duration))
        for j,(left,right) in enumerate(written):
            if not math.isfinite(left):continue
            vi=bisect_right(starts,edges[j])-1
            equal(left,vs[vi]['written_start_quarters']+edges[j]-starts[vi])
            equal(right,left+edges[j+1]-edges[j])
        labels=[];owners=[]
        for left,right in written:
            label=owner=-1
            if math.isfinite(left) and right>left:
                q=(left+right)/2
                for j,(a,b) in enumerate(segments[piece]):
                    if a<=q<b:
                        label=min(2,int(3*(q-a)/(b-a)));owner=j;break
            labels.append(label);owners.append(owner)
        if labels!=z[f'{i}/phase'].tolist() or owners!=z[f'{i}/owner'].tolist():raise AssertionError('Phase assignment differs')
        for view in ('whole','A','B'):
            chosen=[]
            for j in range(len(mask)):
                use=bool(mask[j])
                if view!='whole':use=use and j//64%2==('AB'.index(view)) and j<len(mask)//64*64
                chosen.append(use)
            expected=[]
            for label in range(3):
                ids=[j for j in range(len(mask)) if chosen[j] and labels[j]==label]
                ps=[j for j in ids if j>0 and chosen[j-1] and pairs[j] and labels[j-1]==label and owners[j-1]==owners[j]
                    and (view=='whole' or j%64!=0)]
                expected.extend([math.fsum(float(x[c,j]) for j in ids)/len(ids) if len(ids)>=8 else math.nan for c in range(7)])
                expected.extend([math.fsum(float(x[c,j]-x[c,j-1]) for j in ps)/len(ps) if len(ps)>=4 else math.nan for c in range(7)])
                expected.extend([math.sqrt(math.fsum(float(x[c,j]-x[c,j-1])**2 for j in ps)/len(ps)) if len(ps)>=4 else math.nan for c in range(7)])
                expected.append(math.fsum(float((x[0,j]-x[0,j-1])*(x[2,j]-x[2,j-1])) for j in ps)/len(ps) if len(ps)>=4 else math.nan)
            for j,value in enumerate(expected):equal(value,float(actual[key,view][f'p{j}']))
    result=dict(status='passed',performances=len(coverage),numeric_checks=checks,maximum_absolute_error=error,
        original_feature_digests_equal=True,prior_combined_counts_equal=True,
        observed_visit_durations_differing_from_xml_extent=len(duration_differences),
        independent_scope='Raw XML written measure origins, saved observed visit projection, scalar phase membership and original-index response arithmetic; bridge route correctness relies on prior independently verified bridge; observed downbeat duration is not assumed equal to XML encoded extent',
        code_sha256=sha(Path(__file__)),input_artifact_sha256=provenance['artifact_sha256'])
    (args.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
