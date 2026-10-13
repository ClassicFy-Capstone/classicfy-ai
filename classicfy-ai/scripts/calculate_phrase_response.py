"""Restore the frozen cohort and calculate provisional score-phrase responses."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tarfile
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'classicfy-ai/src'))
from preprocessing import ASAPLoader, load_musicxml, load_match, load_midi_score_timeline
from preprocessing.score_measure_bridge import build_score_measure_bridge
from preprocessing.phrase_candidates import extract_phrase_candidates
from features.structure_response import phrase_hypotheses, phase_labels, response, split_view, fixed_labels
from validate_score_beat_bridge import rows, sha, write


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def provenance(out, inputs, code, extra=None):
    dump(out/'provenance.json', dict(inputs={str(p): sha(p) for p in inputs},
        code={str(p.relative_to(ROOT)): sha(p) for p in code},
        results={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name!='provenance.json'},
        **(extra or {})))


def restore(data):
    """Run the original 20a4168 collector in an isolated child interpreter.

    The later learned-source snapshot is unavailable on origin. Its original
    feature collector is independently archived here and must reproduce every
    frozen scale, provenance field, key and downstream feature digest.
    """
    legacy = json.loads((ROOT/'classicfy-ai/analysis/metadata_information_design/protocol.json').read_text())
    raw = data/'feature_normalization_raw.npz'
    reference = data/'temporal_embedding_run01'
    if sha(raw) != legacy['input_sha256']['raw_cache']:
        raise ValueError('Local raw cache differs from frozen experiment')
    archive = subprocess.check_output(['git','archive','20a4168','classicfy-ai/src','classicfy-ai/scripts'],cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='classicfy-frozen-collector-') as folder:
        snapshot=Path(folder)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(snapshot,filter='data')
        child=snapshot/'restore.py'
        child.write_text('''import sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
root=Path(__file__).parent
sys.path[:0]=[str(root/'classicfy-ai/scripts'),str(root/'classicfy-ai/src')]
from validate_embedding import collect_groups
data=Path(sys.argv[1]);destination=Path(sys.argv[2]);legacy=json.loads(Path(sys.argv[3]).read_text())
reference=json.loads((data/'temporal_embedding_run01/dataset.json').read_text())
groups,audits,scales,prov=collect_groups(SimpleNamespace(asap_root=data/'ASAP',nasap_root=data/'nASAP',cache=data/'feature_normalization_raw.npz',**reference['cohort_settings']))
if scales!=reference['scales'] or prov!=legacy['source_provenance']:raise ValueError('Frozen scales/provenance differ')
lengths={(r['piece'],r['grid']):r['total_intervals'] for r in audits if r['accepted']}
arrays={};keys=[]
for group in groups:
 length=lengths[group['piece'],group['grid']];indices=group['shared_indices']
 for key,values in zip(group['keys'],group['values']):
  i=len(keys);keys.append(key);x=np.zeros((7,length));m=np.zeros(length,bool);x[:,indices]=values;m[indices]=True
  arrays[f'{i}/values']=x;arrays[f'{i}/mask']=m
arrays['keys']=np.array(keys);np.savez_compressed(destination,**arrays)
''')
        destination=snapshot/'restored.npz'
        subprocess.run([sys.executable,str(child),str(data.resolve()),str(destination),
            str(ROOT/'classicfy-ai/analysis/metadata_information_design/protocol.json')],check=True)
        with np.load(destination,allow_pickle=False) as z:
            return {key:(z[f'{i}/values'].copy(),z[f'{i}/mask'].copy()) for i,key in enumerate(z['keys'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT.parent/'datasets')
    parser.add_argument('--out', type=Path, default=ROOT/'classicfy-ai/analysis/phrase_response')
    parser.add_argument('--artifacts', type=Path, default=ROOT.parent/'datasets/phrase_response_run01')
    args = parser.parse_args()
    for folder in (args.out, args.artifacts):
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('Choose empty output folders: '+str(folder))
        folder.mkdir(parents=True, exist_ok=True)
    fixed = rows(ROOT/'classicfy-ai/analysis/score_structure_inputs/performances.csv')
    roles = {r['key']: r for r in rows(ROOT/'classicfy-ai/analysis/metadata_information_design/target_preflight.csv')}
    # Freeze all downstream rules before inspecting response or ranking results.
    protocol = dict(version=1, date='2026-10-14', cohort='Frozen 55 score grids / 570 recordings',
        phrase_policy='P1 score-only provisional scaffold: global notated silence, fermata, same-scope slur >=4 quarters; min gap 4, max gap 32 quarters; cap explicitly marked fixed_length_cap; final short tail retained',
        phrase_truth='No expert phrase ground truth; existing pending candidates remain unchanged',
        feature_policy='Byte-identical raw cache and frozen collector/scales; existing 7-channel normalized residuals; no new common-pattern fit',
        mapping='Original B-1 intervals; first printed part supplies context, all parts required for bridge validity; midpoint phase assignment without snapping; repeat visits retained',
        response='Three bins, each 7 means + 7 mean deltas + 7 delta RMS + mean(deltaTempo*deltaDynamics); 66D; joint moment is not correlation',
        support=dict(minimum_beats_per_bin=8, minimum_pairs_per_bin=4),
        boundary_sensitivity=dict(shifts_quarters=[-.5, .5], exclude_crossing_intervals=True),
        evaluation=dict(views='Alternating full 64-original-beat blocks; no pair across blocks; both A->B and B->A',
            split='Existing evaluation_role with family overlap exclusions unchanged', K=5,
            methods=['B', 'B+T', 'B+F', 'B+P', 'B+H', 'B+P+H', 'structure_only', 'shifted_structure'],
            primary='E-T delta-RMS distribution on held-out blocks, macro by work then family',
            secondary=['E-L median/IQR', 'E-R Tempo-Dynamics delta correlation distribution', 'E-S context conditional delta distributions'],
            controls=['M-CT 100 tie seeds', 'same-work random recording 100 seeds', 'structure circular-shift 20 seeds', 'equal gallery recordings per work (3)', 'length/coverage Q'],
            weights='Equal squared normalized block distances; training family-balanced coordinate SD and other-family median distance',
            tension='Spiral array, exact written duration overlap per original score interval, 3 metrics; key-mode provenance and uncertainty retained; no performance input',
            evidence='Exploratory proxy consistency; do not claim listening/preferences; report all axes, per-family results and support; no rule tuning after outcomes'))
    dump(args.out/'protocol.json', protocol)
    sequences = restore(args.data)
    if set(sequences) != {r['key'] for r in fixed}:
        raise ValueError('Frozen cohort changed')
    print('Restored 570 original-index seven-channel sequences', flush=True)
    loader = ASAPLoader(args.data/'ASAP', args.data/'nASAP')
    annotations = json.loads((args.data/'ASAP/asap_annotations.json').read_text())
    xml_hash = {r['xml_score']: r['xml_sha256'] for r in rows(ROOT/'classicfy-ai/analysis/score_structure_inputs/nasap_works.csv')}
    scores, timelines, phrases, segment_rows = {}, {}, {}, []
    inputs = [args.data/'feature_normalization_raw.npz', args.data/'temporal_embedding_run01/dataset.json',
              ROOT/'classicfy-ai/analysis/score_structure_inputs/performances.csv',
              ROOT/'classicfy-ai/analysis/metadata_information_design/target_preflight.csv']
    for piece in sorted({r['piece'] for r in fixed}):
        path = args.data/'nASAP'/Path(piece).parent/'xml_score.musicxml'
        if sha(path) != xml_hash[str(Path(piece).parent/'xml_score.musicxml')]:
            raise ValueError('Frozen XML changed')
        inputs.extend([path,args.data/'ASAP'/piece])
        score = load_musicxml(path); scores[piece] = score
        timelines[piece] = load_midi_score_timeline(args.data/'ASAP'/piece)
        phrases[piece] = phrase_hypotheses(score, extract_phrase_candidates(score))
        segment_rows.extend(dict(piece=piece, **{k: str(v) if k in ('start','stop') else v
            for k, v in asdict(p).items()}) for p in phrases[piece])
    write(args.out/'phrase_hypotheses.csv', segment_rows)
    arrays, coverage, descriptions, sensitivity = {}, [], [], []
    geometries = {}
    visit_geometry = {}
    for index, row in enumerate(fixed):
        key = row['key']; piece = row['piece']; sample = loader.get_sample(key)
        if sha(sample.note_alignment_path) != row['note_alignment_sha256']:
            raise ValueError('Frozen match changed')
        alignment = load_match(sample.note_alignment_path)
        sources = sorted((s.note_id,s.pitch,s.onset_beats,s.offset_beats,s.attributes)
            for s in [n for n,p in alignment.matches]+list(alignment.deletions))
        geometry = hashlib.sha256(json.dumps([piece,row['grid'],sources,sample.score_downbeats,
            annotations[key]['downbeats_score_map']],sort_keys=True).encode()).hexdigest()
        if geometry not in geometries:
            geometries[geometry] = build_score_measure_bridge(scores[piece],alignment,timelines[piece],
                sample.score_beats,sample.performance_beats,sample.score_beat_types,sample.score_downbeats,
                annotations[key]['downbeats_score_map'])
        bridge = geometries[geometry]
        primary = bridge.parts[0][1]
        visit_geometry[geometry]=dict(piece=piece,part_id=bridge.parts[0][0],visits=[asdict(v) for v in primary.visits])
        arrays[f'{index}/geometry']=np.array(geometry)
        x, old_mask = sequences[key]
        valid = np.logical_and.reduce([[b.valid for b in p.bridge.intervals] for pid,p in bridge.parts])
        mask = old_mask & valid
        pairs = np.logical_and.reduce([p.adjacent_pair_mask(old_mask) for pid,p in bridge.parts])
        irregular=np.array([t=='bR' for t in sample.score_beat_types])
        for i in range(1,len(pairs)):
            if irregular[i-1:i+2].any():pairs[i]=False
        written = primary.interval_written_quarters
        label, owner = phase_labels(written, phrases[piece])
        arrays[f'{index}/values'] = x
        arrays[f'{index}/mask'] = mask
        arrays[f'{index}/feature_mask'] = old_mask
        arrays[f'{index}/pairs'] = np.asarray(pairs, bool)
        arrays[f'{index}/written'] = np.array([w if w is not None else (np.nan,np.nan) for w in written])
        arrays[f'{index}/phase'] = label
        arrays[f'{index}/owner'] = owner
        arrays[f'{index}/breaks'] = np.logical_or.reduce([p.interval_break_before for pid,p in bridge.parts])
        arrays[f'{index}/score_quarters'] = np.array([round(float(timelines[piece].quarters_at_seconds(t))*timelines[piece].ppq)/timelines[piece].ppq for t in sample.score_beats])
        z, support = response(x, mask, pairs, label, owner)
        descriptions.append(dict(key=key,piece=piece,view='whole',**{f'p{j}':v for j,v in enumerate(z)}))
        for view in (0,1):
            m,p,b = split_view(mask, pairs, view)
            z,_ = response(x,m,p,label,owner)
            descriptions.append(dict(key=key,piece=piece,view='AB'[view],**{f'p{j}':v for j,v in enumerate(z)}))
        baseline, _ = response(x,mask,pairs,label,owner)
        for shift in (-.5,.5):
            ll,oo = phase_labels(written,phrases[piece],shift)
            changed,_ = response(x,mask,pairs,ll,oo)
            common = np.isfinite(baseline)&np.isfinite(changed)
            sensitivity.append(dict(key=key,condition='shift_'+str(shift),relabeled_beats=int(np.sum(ll!=label)),
                common_coordinates=int(common.sum()),response_rms=float(np.sqrt(np.mean((changed[common]-baseline[common])**2))) if common.any() else np.nan))
        crossing = np.array([False if w is None else any(w[0]<float(ph.stop)<w[1] for ph in phrases[piece]) for w in written])
        changed,_=response(x,mask&~crossing,pairs,label,owner)
        common=np.isfinite(baseline)&np.isfinite(changed)
        sensitivity.append(dict(key=key,condition='exclude_crossing',relabeled_beats=int(crossing.sum()),common_coordinates=int(common.sum()),
            response_rms=float(np.sqrt(np.mean((changed[common]-baseline[common])**2))) if common.any() else np.nan))
        coverage.append(dict(key=key,piece=piece,**{k:roles[key][k] for k in ('grid','composer','family_id','work_type','evaluation_role')},
            original_intervals=len(mask),feature_intervals=int(old_mask.sum()),combined_intervals=int(mask.sum()),
            phrase_labeled_intervals=int(np.sum(mask&(label>=0))),pairs=int(np.sum(pairs)),hypotheses=len(phrases[piece]),
            finite_response_coordinates=int(np.isfinite(baseline).sum()),feature_sha256=hashlib.sha256(x.tobytes()+old_mask.tobytes()).hexdigest()))
        if (index+1)%50==0:
            print('Phrase responses',index+1,'/570',flush=True)
    arrays['keys']=np.array([r['key'] for r in fixed])
    np.savez_compressed(args.artifacts/'sequences.npz',**arrays)
    dump(args.artifacts/'visit_geometry.json',visit_geometry)
    write(args.out/'coverage.csv',coverage)
    write(args.out/'responses.csv',descriptions)
    write(args.out/'boundary_sensitivity.csv',sensitivity)
    summary=dict(works=len(scores),performances=len(fixed),phrase_hypotheses=len(segment_rows),
        expert_confirmed_phrases=0,original_intervals=sum(r['original_intervals'] for r in coverage),
        combined_intervals=sum(r['combined_intervals'] for r in coverage),
        phrase_labeled_intervals=sum(r['phrase_labeled_intervals'] for r in coverage),
        full_response_performances=sum(r['finite_response_coordinates']==66 for r in coverage),
        boundary_evidence=dict(Counter(e for ps in phrases.values() for p in ps for e in p.stop_evidence)),
        artifact=str((args.artifacts/'sequences.npz').resolve()))
    dump(args.out/'summary.json',summary)
    code=[Path(__file__),ROOT/'classicfy-ai/src/features/structure_response.py']
    code+=sorted((ROOT/'classicfy-ai/src/preprocessing').glob('*.py'))
    inputs += [args.data/'ASAP/asap_annotations.json',args.data/'ASAP/metadata.csv',args.data/'nASAP/metadata.csv']
    inputs += [loader.get_sample(r['key']).note_alignment_path for r in fixed]
    provenance(args.out,inputs,code,dict(artifact_sha256=sha(args.artifacts/'sequences.npz'),
        visit_geometry_sha256=sha(args.artifacts/'visit_geometry.json'),collector_commit='20a4168'))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
