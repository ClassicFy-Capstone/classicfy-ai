"""Exercise the score-context export/encoder on the available ASAP handoff.

This is a real-data integration and independent arithmetic audit, not ATEPP
training or a claim that initial random context vectors are learned embeddings.
"""

import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

import mido
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'classicfy-ai/src'))
from preprocessing import ASAPLoader
from embedding.score_context_data import (collate_score_context,dump_json,export_score_context,
    read_score_context_segments,sha256)
from embedding.score_encoder import ScoreEncoder,ScoreEncoderConfig


def independent_notes(path):
    midi=mido.MidiFile(path);notes=set()
    for track in midi.tracks:
        at=0;pending={}
        for event in track:
            at+=event.time
            if event.type not in ('note_on','note_off') or event.channel==9:continue
            ident=event.channel,event.note
            if event.type=='note_on' and event.velocity>0:pending.setdefault(ident,[]).append(at)
            elif pending.get(ident):
                begin=pending[ident].pop(0)
                if at>begin:notes.add((event.note,Fraction(begin,midi.ticks_per_beat),Fraction(at,midi.ticks_per_beat)))
    return [(p,a,b,float(a),float(b)) for p,a,b in sorted(notes)]


def scalar_features(notes,left,right):
    a,b=Fraction(str(left)),Fraction(str(right));result=[0.]*30;active=[];onset_durations=[]
    for pitch,start,stop,ss,ee in notes:
        if ss>=right or ee<=left:continue
        overlap=min(b,stop)-max(a,start)
        if overlap<=0:continue
        active.append((pitch,overlap))
        result[12+pitch%12]+=float(overlap/(b-a))
        if a<=start<b:
            result[pitch%12]+=1;result[24]+=1;onset_durations.append(stop-start)
    if active:
        total=sum((length for p,length in active),Fraction(0))
        result[25]=float(sum((p*length for p,length in active),Fraction(0))/total)
        result[26]=min(p for p,length in active);result[27]=max(p for p,length in active)
    else:result[29]=1.
    if onset_durations:result[28]=float(sum(onset_durations,Fraction(0))/len(onset_durations))
    return np.array(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=ROOT/'classicfy-ai/analysis/dataset_compatibility/asap_atepp_feature_gap'
    parser.add_argument('--performance-cache',type=Path,default=base/'all_asap_quarter_normalized.npz')
    parser.add_argument('--asap-root',type=Path,default=ROOT.parent/'datasets/ASAP')
    parser.add_argument('--nasap-root',type=Path,default=ROOT.parent/'datasets/nASAP')
    parser.add_argument('--out',type=Path,default=ROOT/'classicfy-ai/analysis/score_context_validation')
    parser.add_argument('--artifacts',type=Path,default=ROOT.parent/'datasets/score_context_validation_run02')
    args=parser.parse_args()
    for folder in (args.out,args.artifacts):
        if folder.exists() and any(folder.iterdir()):raise ValueError('Choose empty output folders')
    args.artifacts.mkdir(parents=True,exist_ok=True)
    source=np.load(args.performance_cache,allow_pickle=False);records=json.loads(str(source['manifest']))
    with (base/'all_asap_time_axis_audit.csv').open(encoding='utf-8-sig') as stream:
        source_audit={r['key']:r for r in csv.DictReader(stream)}
    loader=ASAPLoader(args.asap_root,args.nasap_root)
    map_path=args.artifacts/'score_map.csv';by_score={}
    with map_path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['key','work','score_path','score_sha256','split']);writer.writeheader()
        for r in records:
            path=loader.get_sample(r['key']).score_path
            digest=sha256(path)
            if digest!=source_audit[r['key']]['score_sha256']:raise AssertionError('Source score changed')
            by_score[digest]=path
            writer.writerow(dict(key=r['key'],work=r['work'],score_path=path.relative_to(args.asap_root.resolve()).as_posix(),score_sha256=digest,split='unassigned'))
    exported=export_score_context(args.performance_cache,map_path,args.artifacts/'joined',dataset='ASAP',score_root=args.asap_root)
    joined=np.load(args.artifacts/'joined/dataset.npz',allow_pickle=False)
    for i,r in enumerate(exported['records']):
        np.testing.assert_array_equal(joined[f'records/{i}/observation_mask'],source[f'{i}/valid'].T)
        mask=source[f'{i}/valid'].T
        np.testing.assert_array_equal(joined[f'records/{i}/performance'][mask],source[f'{i}/values'].T[mask])
        np.testing.assert_array_equal(joined[f'scores/{r["score_ref"]}/q'],source[f'{i}/q'])
    # Fixed before inspecting outcomes: evenly spaced score/grid identities and bins.
    score_rows=exported['scores'];selected=np.linspace(0,len(score_rows)-1,min(32,len(score_rows)),dtype=int)
    checks=0;maximum=0.;symbolic={};audit_rows=[]
    for index in selected:
        item=score_rows[index];ref=item['score_ref'];digest=item['score_sha256']
        if digest not in symbolic:symbolic[digest]=independent_notes(by_score[digest])
        q=joined[f'scores/{ref}/q'];valid=joined[f'scores/{ref}/valid'];actual=joined[f'scores/{ref}/raw']
        ids=np.flatnonzero(valid)
        chosen=ids[np.linspace(0,len(ids)-1,min(8,len(ids)),dtype=int)] if len(ids) else []
        for j in chosen:
            expected=scalar_features(symbolic[digest],float(q[j]),float(q[j+1]))
            difference=float(np.max(np.abs(actual[j]-expected)));maximum=max(maximum,difference);checks+=30
            if difference>1e-9:raise AssertionError((ref,j,difference))
            audit_rows.append(dict(score_ref=ref,interval_index=int(j),q_start=float(q[j]),q_stop=float(q[j+1]),maximum_absolute_error=difference))
    manifest,segments=read_score_context_segments(args.artifacts/'joined',max_segment_length=64)
    eligible=[i for i,ss in enumerate(segments) if ss]
    probe=[eligible[i] for i in np.linspace(0,len(eligible)-1,min(4,len(eligible)),dtype=int)]
    torch.set_num_threads(2);torch.manual_seed(20261029)
    batch=collate_score_context([segments[i] for i in probe]);model=ScoreEncoder(ScoreEncoderConfig(30))
    with torch.no_grad():context=model(batch['score'],batch['time_mask'],batch['segment_mask'])
    valid=batch['time_mask']&batch['segment_mask'].unsqueeze(-1)
    assert torch.isfinite(context).all() and not torch.count_nonzero(context[~valid])
    # Compare a context encoded in the recording batch against its single segment.
    count=int(batch['time_mask'][0,0].sum())
    with torch.no_grad():single=model(batch['score'][:1,:1,:count],batch['time_mask'][:1,:1,:count],batch['segment_mask'][:1,:1])
    torch.testing.assert_close(single[0,0],context[0,0,:count],rtol=1e-5,atol=1e-6)
    diagnostics=Counter()
    # Unique score identities, not repeated per grid or recording.
    for digest in by_score:
        row=next(s for s in score_rows if s['score_sha256']==digest)
        diagnostics.update({k:v for k,v in row['diagnostics'].items() if k!='ppq'})
    summary=dict(date='2026-10-29',dataset='ASAP integration audit; ATEPP raw inputs unavailable locally',
        recordings=len(records),unique_score_files=len(by_score),score_grid_references=len(score_rows),
        original_intervals=sum(r['original_intervals'] for r in exported['records']),
        supported_score_intervals=sum(r['score_intervals'] for r in exported['records']),
        usable_intervals=sum(r['usable_intervals'] for r in exported['records']),
        eligible_recordings=len(eligible),segments=sum(map(len,segments)),
        performance_values_masks_and_quarters_preserved=True,
        independent_numeric_checks=checks,independent_max_absolute_error=maximum,
        independent_scope='Raw MIDI ticks / exact Fraction overlap scalar arithmetic at 32 evenly spaced score/grid references and <=8 evenly spaced supported intervals each; all records checked for unchanged R/masks/q',
        midi_diagnostics=dict(diagnostics),
        encoder_smoke=dict(initial_random_weights=True,score_shape=list(batch['score'].shape),context_shape=list(context.shape),
            finite=True,padding_zero=True,individual_segment_matches_batch=True),
        runtime=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version.split()[0]),
        limitations=['No actual ATEPP export/training performed','No complete PerformanceAutoencoder or learned context exported','MIDI tie/grace/voice semantics not inferred','Original performance normalization remains corpus-fitted'])
    args.out.mkdir(parents=True,exist_ok=True)
    dump_json(args.out/'summary.json',summary)
    with (args.out/'independent_intervals.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(audit_rows[0]));writer.writeheader();writer.writerows(audit_rows)
    with (args.out/'coverage.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=list(exported['records'][0]);writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(exported['records'])
    code=[Path(__file__),ROOT/'classicfy-ai/src/features/score_context.py',ROOT/'classicfy-ai/src/features/__init__.py',ROOT/'classicfy-ai/src/preprocessing/score_midi.py',
        ROOT/'classicfy-ai/src/embedding/score_context_data.py',ROOT/'classicfy-ai/src/embedding/score_encoder.py',ROOT/'classicfy-ai/scripts/export_atepp_score_context.py']
    dump_json(args.out/'provenance.json',dict(code={str(p.relative_to(ROOT)):sha256(p) for p in code},
        inputs=dict(performance_cache_sha256=sha256(args.performance_cache),source_audit_sha256=sha256(base/'all_asap_time_axis_audit.csv'),
            score_files={p.relative_to(args.asap_root.resolve()).as_posix():digest for digest,p in by_score.items()}),
        artifacts={str(p.relative_to(args.artifacts)):sha256(p) for p in args.artifacts.rglob('*') if p.is_file()},
        results={p.name:sha256(p) for p in args.out.iterdir() if p.is_file()},
        artifact_directory=str(args.artifacts.resolve()),model_policy='No checkpoint or random context-vector cache saved'))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
