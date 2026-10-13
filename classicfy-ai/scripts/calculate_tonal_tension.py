"""Compute three score-only tension measures and seven-channel responses."""

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from preprocessing import load_musicxml
from features.tonal_tension import TonalContext,tension_labels,HEIGHT,SCALE
from features.structure_response import response,split_view
from calculate_phrase_response import dump,provenance
from validate_score_beat_bridge import rows,sha,write

METRICS=('cloud_diameter','cloud_momentum','tensile_strain')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT.parent/'datasets')
    parser.add_argument('--phrases',type=Path,default=ROOT/'classicfy-ai/analysis/phrase_response')
    parser.add_argument('--source',type=Path,default=ROOT.parent/'datasets/phrase_response_run02')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/tonal_tension')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/tonal_tension_run01')
    args=parser.parse_args()
    for folder in (args.out,args.artifacts):
        if folder.exists() and any(folder.iterdir()):raise ValueError('Choose empty output folder: '+str(folder))
        folder.mkdir(parents=True,exist_ok=True)
    source_prov=json.loads((args.phrases/'provenance.json').read_text())
    if sha(args.source/'sequences.npz')!=source_prov['artifact_sha256']:raise ValueError('Verified phrase input changed')
    protocol=dict(version=1,date='2026-10-14',source_phrase_provenance=sha(args.phrases/'provenance.json'),
        source='Written nASAP MusicXML with frozen SHA; source spelling, all parts, positive written overlap duration; no performance input',
        reference='https://partitura.readthedocs.io/en/latest/_modules/partitura/musicanalysis/tonal_tension.html',
        key_reference='https://partitura.readthedocs.io/en/latest/_modules/partitura/musicanalysis/key_identification.html',
        pitch='fifth index F=-1,C=0,G=1,D=2,A=3,E=4,B=5; alteration adds 7 steps; octave invariant',
        helix=dict(radius=1.,height_per_fifth=HEIGHT,scale_C_to_Bsharp=SCALE),
        chord_key_weights=[516/999,315/999,168/999],minor_dominant_major_weight=.75,minor_subdominant_minor_weight=.75,
        window='Exact half-open written projection of each original score interval, positive overlap only; grace/cue omitted; unsupported microtones invalidate cloud',
        diameter='Maximum pairwise Euclidean pitch-point distance; one-note cloud is 0; silent is NaN',
        momentum='Distance between consecutive duration-weighted centers; first/silent/structural jump/missing combined mask resets, never fill unknown with 0',
        strain='Distance from cloud center to key center; signature section + explicit or estimated mode; missing/ambiguous/conflicting key -> NaN; major/minor alternative values retained',
        mode='Use explicit XML mode when present. Otherwise correlate signature-section duration pitch histogram with Krumhansl-Kessler major/relative-minor profiles. Require absolute correlation margin >=0.1. This is a hypothesis, not expert key truth or local modulation detection.',
        labels='Score-value tertiles using all finite projected score intervals, independent of performance feature values; ties stay together. 3 metrics x 3 bins x 22 responses =198D.',
        sensitivity='Both signature-compatible major/minor strain curves retained; no selection by retrieval outcome',
        response_support=dict(beats_per_bin=8,pairs_per_bin=4),
        limitations=['Signature sections do not detect unnotated modulation','Duration profile and correlation margin are not calibrated key probabilities','No empirical listener tension ground truth','Exact ASAP interval windows differ from Partitura default centered one-beat windows'])
    dump(args.out/'protocol.json',protocol)
    coverage=rows(args.phrases/'coverage.csv');z=np.load(args.source/'sequences.npz',allow_pickle=False)
    contexts={};key_rows=[];inputs=[args.source/'sequences.npz',args.phrases/'provenance.json',args.phrases/'coverage.csv']
    for piece in sorted({r['piece'] for r in coverage}):
        path=args.data/'nASAP'/Path(piece).parent/'xml_score.musicxml'
        if sha(path)!=source_prov['inputs'][str(path)]:raise ValueError('Frozen score changed')
        context=TonalContext(load_musicxml(path));contexts[piece]=context;inputs.append(path)
        key_rows.extend(dict(piece=piece,key_index=i,**asdict(k)) for i,k in enumerate(context.contexts))
    write(args.out/'key_contexts.csv',key_rows)
    arrays={};support_rows=[];response_rows=[];geometry_cache={}
    for i,r in enumerate(coverage):
        key=r['key'];piece=r['piece'];mask=z[f'{i}/mask'];x=z[f'{i}/values'];written=z[f'{i}/written']
        cuts=z[f'{i}/breaks'].copy();cuts[1:]|=~mask[:-1]|~mask[1:]
        identity=(piece,written.tobytes(),cuts.tobytes())
        if identity not in geometry_cache:
            geometry_cache[identity]=contexts[piece].project(written,cuts)
        values,support=geometry_cache[identity]
        labels=[];thresholds=[]
        keys=np.array([s['key_index'] for s in support])
        for metric in range(3):
            ll,threshold=tension_labels(values[metric],np.ones(len(mask),bool))
            labels.append(ll);thresholds.append(threshold)
        arrays[f'{i}/tension']=values
        arrays[f'{i}/labels']=np.asarray(labels)
        arrays[f'{i}/key_index']=keys
        arrays[f'{i}/strain_major']=np.array([s['strain_major'] for s in support])
        arrays[f'{i}/strain_minor']=np.array([s['strain_minor'] for s in support])
        for metric,name in enumerate(METRICS):
            metric_pairs=z[f'{i}/pairs'].copy()
            metric_pairs[1:]&=np.isfinite(values[metric,:-1])&np.isfinite(values[metric,1:])
            if metric==2:metric_pairs[1:]&=keys[:-1]==keys[1:]
            support_rows.append(dict(key=key,piece=piece,metric=name,original_intervals=len(mask),
                finite_score_intervals=int(np.isfinite(values[metric]).sum()),usable_intervals=int((mask&np.isfinite(values[metric])).sum()),
                low_mid_threshold=thresholds[metric][0],mid_high_threshold=thresholds[metric][1],
                low_beats=int(np.sum(mask&(labels[metric]==0))),middle_beats=int(np.sum(mask&(labels[metric]==1))),high_beats=int(np.sum(mask&(labels[metric]==2)))))
            for view in ('whole','A','B'):
                m,p,b=split_view(mask,metric_pairs,'AB'.index(view)) if view!='whole' else (mask,metric_pairs,[])
                zz,ss=response(x,m&np.isfinite(values[metric]),p,labels[metric])
                response_rows.append(dict(key=key,piece=piece,metric=name,view=view,**{f'h{j}':v for j,v in enumerate(zz)}))
        if (i+1)%100==0:print('Tension responses',i+1,'/570',flush=True)
    arrays['keys']=z['keys'].copy();arrays['metrics']=np.array(METRICS)
    np.savez_compressed(args.artifacts/'tension.npz',**arrays)
    write(args.out/'coverage.csv',support_rows);write(args.out/'responses.csv',response_rows)
    response_path=args.out/'responses.csv'
    (args.out/'responses.csv.gz').write_bytes(gzip.compress(response_path.read_bytes(),mtime=0))
    response_path.unlink()
    cloud_rows=[]
    for piece,context in contexts.items():
        for (a,b),row in sorted(context.cache.items()):
            center=row['center']
            cloud_rows.append(dict(piece=piece,start=a,stop=b,**{k:v for k,v in row.items() if k!='center'},
                center_x=float(center[0]) if center is not None else np.nan,
                center_y=float(center[1]) if center is not None else np.nan,
                center_z=float(center[2]) if center is not None else np.nan))
    with gzip.open(args.artifacts/'clouds.csv.gz','wt',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cloud_rows[0]));writer.writeheader();writer.writerows(cloud_rows)
    summary=dict(works=len(contexts),performances=len(coverage),score_clouds=len(cloud_rows),
        key_contexts=len(key_rows),key_statuses=dict(Counter(k['status'] for k in key_rows)),
        key_modes=dict(Counter(k['mode'] or 'unknown' for k in key_rows)),
        usable_intervals={name:sum(r['usable_intervals'] for r in support_rows if r['metric']==name) for name in METRICS},
        source_phrase_commit='ceee302',artifact=str(args.artifacts.resolve()))
    dump(args.out/'summary.json',summary)
    pilots=['Bach/Fugue/bwv_848/midi_score.mid','Beethoven/Piano_Sonatas/31-2/midi_score.mid','Mozart/Piano_Sonatas/8-1/midi_score.mid']
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    for ax,piece in zip(axes,pilots):
        i=next(i for i,r in enumerate(coverage) if r['piece']==piece)
        n=min(200,len(z[f'{i}/mask']));values=arrays[f'{i}/tension']
        for c,name in enumerate(METRICS):
            ax.plot(np.arange(n),np.where(z[f'{i}/mask'][:n],values[c,:n],np.nan),lw=1.,label=name)
        ax.set_title(coverage[i]['key'],fontsize=9);ax.set_xlabel('Original beat interval');ax.set_ylabel('Scaled spiral distance');ax.legend(loc='upper right')
    fig.suptitle('Score-only tonal tension; first 200 original intervals\nUnknown mode / silent / unsupported -> missing',fontsize=12)
    for suffix in ('png','pdf'):fig.savefig(args.out/('01_tonal_tension.'+suffix),dpi=160)
    plt.close(fig)
    provenance(args.out,inputs,[Path(__file__),ROOT/'classicfy-ai/src/features/tonal_tension.py',ROOT/'classicfy-ai/src/features/structure_response.py'],
        dict(artifacts={p.name:sha(p) for p in args.artifacts.iterdir()}))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
