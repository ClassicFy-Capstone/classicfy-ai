"""Verify legacy ranking or v2 G/structure constraints, blinding and YT bounds."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

from prepare_selected_era_listening import build_public_pack,original_helpers,digest,dump,QUOTAS,AI


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--youtube-data',type=Path,required=True)
    p.add_argument('--ui',type=Path,default=AI/'tools/ae_selected_era_listening')
    p.add_argument('--out',type=Path,default=AI/'analysis/ae_selected_era_listening')
    p.add_argument('--common-run',type=Path)
    p.add_argument('--previous-pack',type=Path,help='Original listening.json for a partial replacement')
    args=p.parse_args();read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    pack=read(args.out/'listening.json');manifest=read(args.out/'source_ui_manifest.json')
    source_final=read(args.run/'final_report.json');metadata=read(args.run/'metadata.json')
    with np.load(source_final['embeddings']) as f:
        e=f['e'].astype(np.float64);ids=f['perf_ids'].tolist();periods=f['era'].tolist()
    assert digest(source_final['embeddings'])==pack['selection']['final_embeddings_sha256']
    by_id={m['perf_id']:m for m in metadata};helper,_=original_helpers()
    verified_G=pack['selection'].get('version')=='g-verified-v2'
    if verified_G:
        from selected_era_curation import THRESHOLDS,normalized,pair_is_clear,work_tags,texture_profiles,texture_ratios
        from embedding.era_note_style import pair_tables
        from features.style_profile import G_BLOCKS
        assert args.common_run is not None,'--common-run required for G/score texture verification'
        assert pack['selection']['thresholds']==THRESHOLDS
        assert digest(args.run/'profiles.npz')==pack['selection']['G_profiles_sha256']
        assert digest(args.common_run/'corpus/note_q.npy')==pack['selection']['score_note_q_sha256']
        with np.load(args.run/'profiles.npz') as f:raw,known,quality=[f[k] for k in ('raw','known','quality')]
        md=[{**m,'split':'production'} for m in metadata]
        mi={m['perf_id']:i for i,m in enumerate(metadata)}
        corpus=read(args.common_run/'corpus/manifest.json')
        textures=texture_profiles(corpus['clips'],np.load(args.common_run/'corpus/note_q.npy',mmap_mode='r'))
    with (args.youtube_data/'youtube_matches.csv').open(encoding='utf-8-sig') as f:yt={r['perf_id']:r for r in csv.DictReader(f)}
    if verified_G:tags=work_tags(yt,metadata)
    assert len(pack['cases'])==10 and len(pack['clips'])==30
    assert Counter(c['era'] for c in pack['cases'])==QUOTAS
    all_works=set();checks=[]
    for case in pack['cases']:
        era=case['era'];ix=[i for i,p in enumerate(periods) if p==era]
        rows=[dict(performance_id=ids[i],work_id=by_id[ids[i]]['work'],recording_id=by_id[ids[i]]['source']) for i in ix]
        anchor=case['anchor'].split(':')[1];a=[r['performance_id'] for r in rows].index(anchor)
        if verified_G:
            b=[r['performance_id'] for r in rows].index(case['near'].split(':')[1])
            c=[r['performance_id'] for r in rows].index(case['far'].split(':')[1])
            vectors=e[ix]/np.linalg.norm(e[ix],axis=1,keepdims=True);dist=1-vectors@vectors[a]
            pids=[rows[j]['performance_id'] for j in (a,b,c)];indices=[mi[p] for p in pids]
            assert normalized(yt[pids[1]]['artist']).strip()!=normalized(yt[pids[2]]['artist']).strip()
            assert len({rows[j]['recording_id'] for j in (a,b,c)})==3
            assert len({helper['youtube_id'](yt[pid]['youtube_url']) for pid in pids})==3
            assert tags[by_id[pids[0]]['work']]==tags[by_id[pids[1]]['work']]==case['selection_evidence']['structure_tag']
            ratio=texture_ratios(textures[by_id[pids[0]]['work']],textures[by_id[pids[1]]['work']])
            assert max(ratio)<=THRESHOLDS['max_texture_ratio']
            np.testing.assert_allclose(ratio,case['selection_evidence']['texture_ratios'])
            scaler_path=args.run/era/'g_production_scaler.json'
            assert digest(scaler_path)==pack['selection']['G_scaler_sha256'][era]
            saved=read(scaler_path);sc=saved['scaler'];cal=saved['calibration']
            gd,gq=pair_tables(raw,known,quality,md,sc,indices)
            assert np.isfinite(gd[0,1:]).all()
            # Independent scalar calculation on the anchor's fixed signature.
            mask=known[indices]&sc['trained'];x=np.clip((raw[indices]-sc['center'])/sc['scale'],-5,5)
            independent=[]
            for j in (1,2):
                terms=[]
                for block in G_BLOCKS.values():
                    dims=[k for k in block if mask[0,k]]
                    if dims:terms.append(float(np.mean((x[0,dims]-x[j,dims])**2)))
                independent.append(float(np.mean(terms)))
            np.testing.assert_allclose(independent,gd[0,1:],rtol=1e-6)
            evidence=case['selection_evidence']
            np.testing.assert_allclose(independent,[evidence['G_near'],evidence['G_far']],rtol=1e-6)
            np.testing.assert_allclose(gq[0,1:],[evidence['G_confidence_near'],evidence['G_confidence_far']],rtol=1e-6)
            target=np.asarray(independent)/(np.asarray(independent)+cal['median'])
            np.testing.assert_allclose(target,[evidence['G_target_near'],evidence['G_target_far']],rtol=1e-6)
            np.testing.assert_allclose(target[1]-target[0],evidence['G_target_gap'],rtol=1e-6)
            assert pair_is_clear(dist[b],dist[c],*independent,*gq[0,1:],cal['median'])
            pool=[]
        else:b,c,pool,order,dist=helper['select_near_far'](rows,e[ix],a)
        assert case['near']=='ATEPP:'+rows[b]['performance_id']
        assert case['far']=='ATEPP:'+rows[c]['performance_id']
        assert rows[b]['work_id']==rows[c]['work_id']!=rows[a]['work_id']
        works={rows[a]['work_id'],rows[b]['work_id']};assert not works&all_works;all_works.update(works)
        np.testing.assert_allclose([r['cosine_distance_to_anchor'] for r in case['candidates']],[dist[b],dist[c]],atol=1e-9)
        checks.append(dict(case_id=case['id'],era=era,nearest_B_confirmed=not verified_G,farthest_from_A_C_confirmed=not verified_G,
                           G_both_valid=verified_G,G_order_and_clarity_passed=verified_G,distinct_XY_artists=verified_G,
                           near_distance=float(dist[b]),far_distance=float(dist[c]),gap=float(dist[c]-dist[b])))
    assert len(all_works)==20
    public=json.loads((args.ui/'data.js').read_text(encoding='utf-8').split('=',1)[1].strip().rstrip(';'))
    expected,results=build_public_pack(pack,helper)
    for key in ['id','clips','cases','fixed_xy']:assert public.get(key)==expected.get(key)
    replacement=pack['selection'].get('replacement')
    xy=pack['selection'].get('xy_assignment',{})
    archive_random_id=xy.get('archive_random_pack_id')
    prior_id=archive_random_id or (replacement and replacement['from_pack_id'])
    if prior_id:
        archived=args.out/'archive'/prior_id/'listening.json'
        previous=read(args.previous_pack or archived)
        assert previous['id']==prior_id and previous['id']!=pack['id']
        assert previous['storage_id']==pack['storage_id']==public['storage_id']
        replaced=set(replacement['replaced_case_ids']) if replacement else set()
        assert replaced.isdisjoint(c['id'] for c in pack['cases'])
        kept=[]
        for old,new in zip(previous['cases'],pack['cases'],strict=True):
            assert (old['era'],old['label'])==(new['era'],new['label'])
            if old['id'] in replaced:
                if replacement.get('requested_anchor'):
                    assert new['anchor']==replacement['requested_anchor']
                continue
            assert old==new
            for role in ('anchor','near','far'):
                assert previous['clips'][old[role]]==pack['clips'][new[role]]
            kept.append(old['id'])
        assert len(kept)==len(pack['cases'])-len(replaced)
        assert public['compatible_pack_ids']==[previous['id']]
        assert public['migration_sources']==[dict(pack_id=previous['id'],case_ids=[] if archive_random_id else kept)]
    else:
        assert not public.get('compatible_pack_ids')
    for case in public['cases']:
        assert read(args.ui/case['result_url'])==results[case['id']]
        assert 'near' not in case and 'far' not in case
    if xy:
        sides=['X' if case['candidates'][0]==results[case['id']]['near_clip'] else 'Y' for case in public['cases']]
        assert Counter(sides)=={'X':5,'Y':5}
        assert all(len(set(sides[i:i+3]))>1 for i in range(len(sides)-2))
        for era in QUOTAS:
            assert {side for case,side in zip(pack['cases'],sides,strict=True) if case['era']==era}=={'X','Y'}
    if verified_G:
        encoded=json.dumps(public)
        for name in ['selection_evidence','G_distance_to_anchor','G_near','G_far','selection_score','cosine_distance_to_anchor']:
            assert name not in encoded,'Pre-choice payload leaked '+name
    for key,clip in pack['clips'].items():
        row=yt[key.split(':')[1]]
        assert row['youtube_match_status']=='matched_auto'
        assert clip['start_seconds']==float(row['youtube_start_seconds'])
        assert clip['end_seconds']==float(row['youtube_end_seconds'])
        assert clip['video_id']==helper['youtube_id'](row['youtube_url'])
        assert 0<=clip['start_seconds']<clip['end_seconds']<=clip['track_duration_seconds']
    for name in ['study.js','player.js','serve.py','style.css','player.test.cjs']:
        item=manifest['files'][name]
        assert item['original_sha256']==item['generated_sha256']==digest(args.ui/name)
    assert 'whole-recording mean AE embedding' not in (args.ui/'app.js').read_text(encoding='utf-8')
    assert '전곡 평균' not in (args.ui/'app.js').read_text(encoding='utf-8')
    assert '전곡의 8마디' not in (args.ui/'index.html').read_text(encoding='utf-8')
    dump(args.out/'verification.json',dict(status='passed',questions=10,works=20,recordings=30,
        same_era_only=True,ranked_full_5082_catalogue_within_era=True,no_availability_reranking=not verified_G,
        selection_policy=pack['selection'].get('version','legacy'),
        G_verified_order_and_clarity=verified_G,score_texture_and_title_match=verified_G,distinct_XY_artists=verified_G,
        exact_dataset_youtube_bounds=True,core_UI_engines_identical_to_pinned17=True,
        model_results_not_in_public_prechoice_payload=True,new_pack_does_not_migrate_old_answers=not replacement or bool(archive_random_id),
        retained_cases_migrate_prior_answers=bool(replacement and not archive_random_id),
        fixed_xy_answer_counts=dict(Counter(sides)) if xy else None,
        embeddings_sha256=digest(source_final['embeddings']),cases=checks))
    print('Selected-era AXY catalogue verification passed:10questions/20works/30recordings')


if __name__=='__main__':main()
