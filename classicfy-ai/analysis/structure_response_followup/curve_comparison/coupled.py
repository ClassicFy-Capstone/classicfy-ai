"""Supplement: pair each tension curve with its performance response curve."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from compare import (OUT, WORK, REPO, channel_scale, set_distances, calibrated,
                     control_permutation, evaluate, macro, write, sha, weights)


def main():
    protocol=dict(status='supplemental_search_defined_after_initial_curve_results_before_its_own_results',
        reason='Peak-aligned performance curves alone do not preserve the detailed tension trajectory. Pair the score tension curve and performance curve from the same window.',
        pairing='Every 17-beat window has its own rounded three-metric score-tension curve and seven-channel performance curve. Performance is centered by its own search-view global mean.',
        cost='For each pair of windows, equally weight mean channel-scaled performance squared distance and scaled tension squared distance. Match paired windows, not independently chosen tension/performance matches. Use symmetric nearest-window mean and same train-only block median calibration.',
        controls=['H_context_only: compare score tension curves without any performance input','same returned work random recording expectation','20 three-candidate seeds','20 independently permuted performance point-order seeds with score-tension curves held fixed'],
        unchanged='Same 329 recordings, 241 gallery, 45 test and 43 validation queries, opposite-view judges and no new fitted neural model.',
        limitation='Peak windows are provisional, not verified full tension episodes; equal weights are fixed before these outcomes, not selected by results.')
    (OUT/'coupled_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    z=np.load(WORK/'datasets/standalone_structure_run01/standalone.npz');c=np.load(OUT/'curves.npz');old=np.load(OUT/'comparison.npz')
    with (REPO/'classicfy-ai/analysis/phrase_response/coverage.csv').open() as f:lookup={r['key']:r for r in csv.DictReader(f)}
    metadata=[lookup[k] for k in z['keys']];ids=old['eligible'];gids=old['gallery_indices'];qids=old['query_indices']
    gallery=[metadata[i] for i in gids];gw=weights(gallery);families=np.array([r['family_id'] for r in gallery])
    scores,rankings,controls=[],[],[];saved={};scales={}
    for view,direction in enumerate(('A_to_B','B_to_A')):
        evaluations={axis:old[f'{direction}/evaluation/{axis}'] for axis in ('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL')}
        play={};tone={};fits={};context_blocks={}
        for metric in range(3):
            kind='H'+str(metric)
            play[kind]={i:c[f'{i}/{view}/{kind}']-c[f'{i}/{view}/mean'][None,:,None] for i in ids}
            tone[kind]={i:np.array([np.round(c[f'{i}/tension'][metric,a:b],12) for a,b in c[f'{i}/{view}/{kind}_spans']])[:,None,:] for i in ids}
            fits[kind]=(channel_scale(play[kind],gids,gw),channel_scale(tone[kind],gids,gw))
            context_blocks[kind],_=calibrated(set_distances(tone[kind],ids,gids,fits[kind][1]),gids,gw,families)
            scales[direction+'/'+kind]=dict(play_sd=fits[kind][0][0].tolist(),tone_sd=fits[kind][1][0].tolist())

        def joint_distances(seed=None):
            blocks={}
            for kind in play:
                x=play[kind] if seed is None else control_permutation(play[kind],seed,view,kind)
                ps,ts=fits[kind];pa,ta=ps[1],ts[1]
                nplay=int(pa.sum());ntone=int(ta.sum())
                packed={i:np.concatenate((x[i][:,pa,:]/ps[0][None,pa,None]/np.sqrt(nplay),
                    tone[kind][i][:,ta,:]/ts[0][None,ta,None]/np.sqrt(ntone)),axis=1) for i in ids}
                channels=nplay+ntone
                raw=set_distances(packed,ids,gids,(np.ones(channels),np.ones(channels,bool)))
                blocks[kind],_=calibrated(raw,gids,gw,families)
            h=np.sqrt(np.mean([blocks[k]**2 for k in blocks],axis=0));h,_=calibrated(h,gids,gw,families)
            p=old[f'{direction}/distance/P_curve_centered']
            return dict(H_paired_curve=h,**{'P+H_paired_curve':np.sqrt((p*p+h*h)/2)})

        distances=joint_distances()
        context=np.sqrt(np.mean([v*v for v in context_blocks.values()],axis=0));context,_=calibrated(context,gids,gw,families)
        distances['H_context_only']=context
        for method,d in distances.items():
            r,rr=evaluate(method,d,evaluations,metadata,gids,qids)
            scores.extend(dict(direction=direction,**x) for x in r);rankings.extend(dict(direction=direction,**x) for x in rr)
            saved[direction+'/'+method]=d
        groups=defaultdict(list)
        for j,r in enumerate(gallery):groups[r['piece'],r['grid']].append(j)
        for seed in range(20261014,20261034):
            rng=np.random.default_rng(seed)
            allowed=np.array(sorted(j for group in groups.values() if len(group)>=3 for j in rng.choice(group,3,replace=False)))
            for method,d in distances.items():
                r,_=evaluate(method,d,evaluations,metadata,gids,qids,seed,allowed)
                for role in ('test','validation'):
                    rows=[x for x in r if x['role']==role]
                    controls.append(dict(direction=direction,control='three_candidates',seed=seed,method=method,role=role,
                        **{a:macro(rows,a) for a in evaluations}))
        for seed in range(20261014,20261034):
            for method,d in joint_distances(seed).items():
                r,_=evaluate(method,d,evaluations,metadata,gids,qids)
                for role in ('test','validation'):
                    rows=[x for x in r if x['role']==role]
                    controls.append(dict(direction=direction,control='performance_order_permuted',seed=seed,method=method,role=role,
                        **{a:macro(rows,a) for a in evaluations}))
            if (seed-20261014+1)%5==0:print(direction,'paired controls',seed-20261014+1,'/20',flush=True)
    axes=('E-L','E-T','E-R','E-S','E-PC','E-PN','E-HL');summary=[];control_summary=[]
    for role in ('test','validation'):
        for method in sorted(set(r['method'] for r in scores)):
            rows=[r for r in scores if r['role']==role and r['method']==method]
            summary.append(dict(role=role,method=method,**{a:macro(rows,a) for a in axes},**{a+'_same_work':macro(rows,a+'_same_work') for a in axes}))
        for control in ('three_candidates','performance_order_permuted'):
            for method in sorted(set(r['method'] for r in controls if r['control']==control)):
                rows=[r for r in controls if r['role']==role and r['method']==method and r['control']==control]
                control_summary.append(dict(role=role,control=control,method=method,**{a:float(np.mean([r[a] for r in rows])) for a in axes}))
    write('coupled_scores.csv',scores);write('coupled_rankings.csv',rankings);write('coupled_summary.csv',summary)
    write('coupled_controls.csv',controls);write('coupled_control_summary.csv',control_summary)
    np.savez_compressed(OUT/'coupled_distances.npz',**saved)
    (OUT/'coupled_scales.json').write_text(json.dumps(scales,indent=2)+'\n')
    (OUT/'coupled_provenance.json').write_text(json.dumps(dict(code_sha256=sha(Path(__file__)),source_comparison=sha(OUT/'comparison.npz'),
        source_curves=sha(OUT/'curves.npz'),results={p.name:sha(p) for p in OUT.glob('coupled_*') if p.name!='coupled_provenance.json'}),indent=2)+'\n')
    for r in summary:print(r['role'],r['method'],{a:round(r[a],2) for a in axes},flush=True)


if __name__=='__main__':main()
