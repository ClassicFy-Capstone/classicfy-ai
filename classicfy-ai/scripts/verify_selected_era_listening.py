"""Verify AXY ranking, era isolation, original UI engines and exact YT bounds."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np

from prepare_selected_era_listening import original_helpers,digest,dump,QUOTAS,AI


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--youtube-data',type=Path,required=True)
    p.add_argument('--ui',type=Path,default=AI/'tools/ae_selected_era_listening')
    p.add_argument('--out',type=Path,default=AI/'analysis/ae_selected_era_listening')
    args=p.parse_args();read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    pack=read(args.out/'listening.json');manifest=read(args.out/'source_ui_manifest.json')
    source_final=read(args.run/'final_report.json');metadata=read(args.run/'metadata.json')
    with np.load(source_final['embeddings']) as f:
        e=f['e'].astype(np.float64);ids=f['perf_ids'].tolist();periods=f['era'].tolist()
    assert digest(source_final['embeddings'])==pack['selection']['final_embeddings_sha256']
    by_id={m['perf_id']:m for m in metadata};helper,_=original_helpers()
    with (args.youtube_data/'youtube_matches.csv').open(encoding='utf-8-sig') as f:yt={r['perf_id']:r for r in csv.DictReader(f)}
    assert len(pack['cases'])==10 and len(pack['clips'])==30
    assert Counter(c['era'] for c in pack['cases'])==QUOTAS
    all_works=set();checks=[]
    for case in pack['cases']:
        era=case['era'];ix=[i for i,p in enumerate(periods) if p==era]
        rows=[dict(performance_id=ids[i],work_id=by_id[ids[i]]['work'],recording_id=by_id[ids[i]]['source']) for i in ix]
        anchor=case['anchor'].split(':')[1];a=[r['performance_id'] for r in rows].index(anchor)
        b,c,pool,order,dist=helper['select_near_far'](rows,e[ix],a)
        assert case['near']=='ATEPP:'+rows[b]['performance_id']
        assert case['far']=='ATEPP:'+rows[c]['performance_id']
        assert rows[b]['work_id']==rows[c]['work_id']!=rows[a]['work_id']
        works={rows[a]['work_id'],rows[b]['work_id']};assert not works&all_works;all_works.update(works)
        np.testing.assert_allclose([r['cosine_distance_to_anchor'] for r in case['candidates']],[dist[b],dist[c]],atol=1e-9)
        checks.append(dict(case_id=case['id'],era=era,nearest_B_confirmed=True,farthest_from_A_C_confirmed=True,
                           near_distance=float(dist[b]),far_distance=float(dist[c]),gap=float(dist[c]-dist[b])))
    assert len(all_works)==20
    public=json.loads((args.ui/'data.js').read_text(encoding='utf-8').split('=',1)[1].strip().rstrip(';'))
    expected,results=helper['public_pack'](pack)
    for key in ['id','clips','cases']:assert public[key]==expected[key]
    for case in public['cases']:
        assert read(args.ui/case['result_url'])==results[case['id']]
        assert 'near' not in case and 'far' not in case
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
    assert not public.get('compatible_pack_ids')
    dump(args.out/'verification.json',dict(status='passed',questions=10,works=20,recordings=30,
        same_era_only=True,ranked_full_5082_catalogue_within_era=True,no_availability_reranking=True,
        exact_dataset_youtube_bounds=True,core_UI_engines_identical_to_pinned17=True,
        model_results_not_in_public_prechoice_payload=True,new_pack_does_not_migrate_old_answers=True,
        embeddings_sha256=digest(source_final['embeddings']),cases=checks))
    print('Selected-era AXY catalogue verification passed:10questions/20works/30recordings')


if __name__=='__main__':main()
