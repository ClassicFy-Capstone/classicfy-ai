"""Independent scalar distance, ranking, score, and span checks."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1]
REPO=WORK/'capstone-2026-2-classicfy'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prov=json.loads((OUT/'provenance.json').read_text())
    for name,h in prov['code'].items():assert digest(OUT/name)==h
    for name,h in prov['results'].items():
        if name not in ('verify.py','verification.json'):assert digest(OUT/name)==h,name
    z=np.load(WORK/'datasets/standalone_structure_run01/standalone.npz')
    c=np.load(OUT/'curves.npz');d=np.load(OUT/'comparison.npz')
    scales=json.loads((OUT/'scales.json').read_text())
    metadata={r['key']:r for r in csv.DictReader((REPO/'classicfy-ai/analysis/phrase_response/coverage.csv').open())}
    keys=z['keys'].tolist();gids=d['gallery_indices'];ids=d['eligible'];qids=d['query_indices']
    gallery=[metadata[keys[i]] for i in gids]
    works=defaultdict(lambda:defaultdict(list))
    for j,r in enumerate(gallery):works[r['family_id']][r['piece'],r['grid']].append(j)
    w=np.zeros(len(gids))
    for family in works.values():
        for recordings in family.values():
            for j in recordings:w[j]=1/(len(works)*len(family)*len(recordings))
    assert abs(w.sum()-1)<1e-12
    checks=0;maximum=0.
    for i in ids:
        mask=z[f'{i}/mask'];pairs=z[f'{i}/pairs']
        for view in (0,1):
            for kind in ('F','P','H0','H1','H2'):
                spans=c[f'{i}/{view}/{kind}_spans'];curves=c[f'{i}/{view}/{kind}']
                assert len(spans)==len(curves)>0
                assert curves.shape[1:]==(7,17) and np.isfinite(curves).all()
                for a,b in spans:
                    assert a//64==(b-1)//64 and a//64%2==view
                    assert b<=len(mask)//64*64 and mask[a:b].all() and pairs[a+1:b].all()
                    checks+=b-a
    # Scalar nearest-curve distances on an outcome-independent sample.
    sample=ids[np.rint(np.linspace(0,len(ids)-1,6)).astype(int)]
    gallery_sample=np.rint(np.linspace(0,len(gids)-1,6)).astype(int)
    for view,direction in enumerate(('A_to_B','B_to_A')):
        for variant in ('raw','centered'):
            for kind in ('F','P'):
                scale=scales[direction][kind+'_'+variant]
                active=np.flatnonzero(scale['active']);sd=np.array(scale['sd'])
                for i in sample:
                    a=c[f'{i}/{view}/{kind}'].copy()
                    if variant=='centered':a-=c[f'{i}/{view}/mean'][None,:,None]
                    for j in gallery_sample:
                        gi=gids[j];b=c[f'{gi}/{view}/{kind}'].copy()
                        if variant=='centered':b-=c[f'{gi}/{view}/mean'][None,:,None]
                        costs=[]
                        for ac in a:
                            costs.append([math.fsum(((float(ac[ch,p])-float(bc[ch,p]))/float(sd[ch]))**2
                                for ch in active for p in range(17))/(len(active)*17) for bc in b])
                        costs=np.array(costs)
                        value=math.sqrt(.5*(math.fsum(costs.min(axis=1))/len(a)+math.fsum(costs.min(axis=0))/len(b)))/scale['median']
                        error=abs(value-float(d[f'{direction}/distance/{kind}_curve_{variant}'][i,j]))
                        maximum=max(maximum,error);assert error<1e-10;checks+=1
        # Scalar B distance and training pair median independently reconstructed.
        x=z[f'{direction}_input_B']; raw=np.empty((335,len(gids)))
        for i in range(335):
            for j,gi in enumerate(gids):
                blocks=[]
                for channels in ((0,),(1,),(2,),(3,),(4,5,6)):
                    blocks.append(math.fsum((float(x[i,ch*2+p])-float(x[gi,ch*2+p]))**2
                        for ch in channels for p in range(2))/(len(channels)*2))
                raw[i,j]=math.sqrt(math.fsum(blocks)/5)
        pairs=[]
        for a in range(len(gids)):
            for b in range(a+1,len(gids)):
                if gallery[a]['family_id']!=gallery[b]['family_id']:pairs.append((raw[gids[a],b],w[a]*w[b]))
        pairs.sort();half=math.fsum(weight for value,weight in pairs)/2;cum=0.
        for median,weight in pairs:
            cum+=weight
            if cum>=half:break
        error=float(np.max(abs(raw/median-d[f'{direction}/distance/B_summary'])))
        maximum=max(maximum,error);assert error<1e-10;checks+=raw.size
    ranks=list(csv.DictReader((OUT/'rankings.csv').open()))
    scores={(r['direction'],r['key'],r['method']):r for r in csv.DictReader((OUT/'scores.csv').open())}
    grouped=defaultdict(list)
    for r in ranks:grouped[r['direction'],r['query'],r['method']].append(r)
    key_to_i={key:i for i,key in enumerate(keys)};g_to_j={keys[i]:j for j,i in enumerate(gids)}
    for (direction,key,method),rows in grouped.items():
        qi=key_to_i[key];matrix=d[f'{direction}/distance/{method}'];selected=[g_to_j[r['candidate']] for r in rows]
        assert len(selected)==len(set(selected))==5
        assert len(set(gallery[j]['family_id'] for j in selected))==5
        assert np.isfinite(matrix[qi,selected]).all()
        # No earlier unselected candidate from a new family can outrank a selected one.
        seen=set()
        for j in selected:
            candidates=[k for k in range(len(gids)) if gallery[k]['family_id'] not in seen]
            assert matrix[qi,j]<=min(matrix[qi,k] for k in candidates)+1e-12
            seen.add(gallery[j]['family_id'])
        record=scores[direction,key,method]
        for axis in ('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL'):
            e=d[f'{direction}/evaluation/{axis}']
            value=100*math.fsum(float(e[qi,j]) for j in selected)/5
            error=abs(value-float(record[axis]));maximum=max(maximum,error);assert error<1e-10;checks+=1
    # Verify macro aggregation independently from query CSV, including both directions.
    score_rows=list(scores.values())
    for r in csv.DictReader((OUT/'summary.csv').open()):
        for axis in ('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL'):
            groups=defaultdict(lambda:defaultdict(list))
            for q in score_rows:
                if q['role']==r['role'] and q['method']==r['method']:groups[q['family_id']][q['piece']].append(float(q[axis]))
            value=math.fsum(math.fsum(math.fsum(v)/len(v) for v in works.values())/len(works)
                for works in groups.values())/len(groups)
            error=abs(value-float(r[axis]));maximum=max(maximum,error);assert error<1e-10;checks+=1
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',checks=int(checks),max_absolute_error=float(maximum),
        scope='Artifact digests, all curve span validity and A/B separation, scalar sampled F/P set distances, full scalar B distance/calibration, every top-5 rank/score and all macro aggregates. Five targeted functional tests separately passed.',
        limits='Does not establish musical phrase truth, listener tension truth or preferred-recording similarity.'),indent=2)+'\n')
    print('Independent verification passed',checks,'checks; maximum error',maximum)


if __name__=='__main__':main()
