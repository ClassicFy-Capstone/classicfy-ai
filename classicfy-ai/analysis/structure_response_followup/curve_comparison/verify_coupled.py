"""Independently rebuild paired cost blocks and check all saved scores."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

OUT=Path(__file__).resolve().parent;WORK=OUT.parents[1];REPO=WORK/'capstone-2026-2-classicfy'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    provenance=json.loads((OUT/'coupled_provenance.json').read_text())
    assert digest(OUT/'coupled.py')==provenance['code_sha256']
    assert digest(OUT/'comparison.npz')==provenance['source_comparison']
    assert digest(OUT/'curves.npz')==provenance['source_curves']
    for name,h in provenance['results'].items():assert digest(OUT/name)==h
    c=np.load(OUT/'curves.npz');base=np.load(OUT/'comparison.npz');saved=np.load(OUT/'coupled_distances.npz')
    ids=base['eligible'];gids=base['gallery_indices'];qids=base['query_indices'];keys=c['keys'].tolist()
    lookup={r['key']:r for r in csv.DictReader((REPO/'classicfy-ai/analysis/phrase_response/coverage.csv').open())}
    gallery=[lookup[keys[i]] for i in gids];by_family=defaultdict(lambda:defaultdict(list))
    for j,r in enumerate(gallery):by_family[r['family_id']][r['piece'],r['grid']].append(j)
    w=np.zeros(len(gids))
    for works in by_family.values():
        for js in works.values():
            for j in js:w[j]=1/(len(by_family)*len(works)*len(js))
    scales=json.loads((OUT/'coupled_scales.json').read_text());checks=0;maximum=0.

    def norm(d):
        pairs=[]
        for a in range(len(gids)):
            for b in range(a+1,len(gids)):
                if gallery[a]['family_id']!=gallery[b]['family_id']:pairs.append((d[gids[a],b],w[a]*w[b]))
        pairs.sort();half=math.fsum(weight for value,weight in pairs)/2;cum=0.
        for median,weight in pairs:
            cum+=weight
            if cum>=half:break
        return d/median

    for view,direction in enumerate(('A_to_B','B_to_A')):
        blocks=[];context_blocks=[]
        for metric in range(3):
            kind='H'+str(metric);scale=scales[direction+'/'+kind]
            ps=np.array(scale['play_sd']);ts=float(scale['tone_sd'][0])
            assert (ps>1e-6).all()
            play={i:(c[f'{i}/{view}/{kind}']-c[f'{i}/{view}/mean'][None,:,None])/ps[None,:,None] for i in ids}
            tone={i:np.array([np.round(c[f'{i}/tension'][metric,a:b],12) for a,b in c[f'{i}/{view}/{kind}_spans']])/ts for i in ids}
            offsets=np.r_[0,np.cumsum([len(play[i]) for i in gids])]
            pp=np.concatenate([play[i].reshape(len(play[i]),-1) for i in gids]);tp=np.concatenate([tone[i] for i in gids])
            d=np.full((335,len(gids)),np.inf);td=d.copy()
            for i in ids:
                pc=cdist(play[i].reshape(len(play[i]),-1),pp,'sqeuclidean')/(7*17)
                tc=cdist(tone[i],tp,'sqeuclidean')/17
                costs=.5*(pc+tc)
                for j in range(len(gids)):
                    window=costs[:,offsets[j]:offsets[j+1]];tw=tc[:,offsets[j]:offsets[j+1]]
                    d[i,j]=math.sqrt(.5*(float(window.min(axis=1).mean())+float(window.min(axis=0).mean())))
                    td[i,j]=math.sqrt(.5*(float(tw.min(axis=1).mean())+float(tw.min(axis=0).mean())))
                # A scalar point-pair check independent of vector cost assembly.
                if i in (ids[0],qids[0],qids[-1]):
                    manual=.5*(math.fsum((float(play[i][0,ch,p])-float(pp[0,ch*17+p]))**2 for ch in range(7) for p in range(17))/(7*17)
                        +math.fsum((float(tone[i][0,p])-float(tp[0,p]))**2 for p in range(17))/17)
                    error=abs(manual-costs[0,0]);maximum=max(maximum,error);assert error<1e-10;checks+=1
            blocks.append(norm(d));context_blocks.append(norm(td))
        h=norm(np.sqrt(sum(b*b for b in blocks)/3));context=norm(np.sqrt(sum(b*b for b in context_blocks)/3))
        ph=np.sqrt((h*h+base[f'{direction}/distance/P_curve_centered']**2)/2)
        for method,a in (('H_paired_curve',h),('H_context_only',context),('P+H_paired_curve',ph)):
            error=float(np.max(abs(a[ids]-saved[direction+'/'+method][ids])))
            maximum=max(maximum,error);assert error<1e-10;checks+=int(a[ids].size)
    ranks=defaultdict(list)
    for r in csv.DictReader((OUT/'coupled_rankings.csv').open()):ranks[r['direction'],r['query'],r['method']].append(r)
    glookup={keys[i]:j for j,i in enumerate(gids)};ilookup={key:i for i,key in enumerate(keys)}
    score_rows=list(csv.DictReader((OUT/'coupled_scores.csv').open()))
    for r in score_rows:
        qi=ilookup[r['key']];selected=[glookup[x['candidate']] for x in ranks[r['direction'],r['key'],r['method']]]
        assert len(set(gallery[j]['family_id'] for j in selected))==5
        for axis in ('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL'):
            e=base[f"{r['direction']}/evaluation/{axis}"]
            value=100*math.fsum(float(e[qi,j]) for j in selected)/5
            error=abs(value-float(r[axis]));maximum=max(maximum,error);assert error<1e-10;checks+=1
    (OUT/'coupled_verification.json').write_text(json.dumps(dict(status='passed',checks=int(checks),max_absolute_error=float(maximum),
        scope='Input/result digests, separately assembled performance/tension equal-weight window costs, train calibration, all coupled distance blocks and all query scores.'),indent=2)+'\n')
    print('Paired verification passed',checks,'checks; maximum error',maximum)


if __name__=='__main__':main()
