"""Replace #17 listening questions with final protected-G selected-clip vectors.

The original ranking and public/blind payload helpers are extracted verbatim
from the pinned #17 commit. Playback failures reject whole A/B/C cases, never
rerank the globally nearest B or the farthest-from-A same-work C.
"""
import argparse
import ast
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
from urllib.parse import urlparse,parse_qs

import numpy as np
from threadpoolctl import threadpool_limits

AI=Path(__file__).resolve().parents[1]
SOURCE_COMMIT='d5311e6247b3d8fff43f5c314b6c817c6ad42a64'
QUOTAS={'baroque':3,'classical_transition':3,'romantic_impressionist':4}
ERA_LABELS={'baroque':'바로크','classical_transition':'고전과 전환','romantic_impressionist':'낭만과 인상주의'}


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def source(path):
    return subprocess.check_output(['git','show',SOURCE_COMMIT+':classicfy-ai/'+path],cwd=AI)


def original_helpers():
    text=source('scripts/prepare_ae_theme_listening.py').decode('utf-8')
    names={'rank_candidates','select_near_far','public_pack','youtube_id'}
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:raise ValueError('Pinned prototype helpers missing')
    ns={'np':np,'hashlib':hashlib,'urlparse':urlparse,'parse_qs':parse_qs,'re':re}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-17-helpers>','exec'),ns)
    return ns,text


def eligible_mapping(row,clip):
    if row['source_sha256']!=clip['source_sha256'] or row['composition_id']!=clip['composition_id']:
        raise ValueError('YouTube mapping identity differs from AE clip: '+row['perf_id'])
    if any(abs(float(row[k])-clip[k])>1e-6 for k in ['start_seconds','end_seconds']):
        raise ValueError('YouTube mapping represents another selected excerpt')
    if row['youtube_match_status']!='matched_auto':return False
    try:start,end=float(row['youtube_start_seconds']),float(row['youtube_end_seconds'])
    except (ValueError,TypeError):return False
    return math.isfinite(start) and math.isfinite(end) and 0<=start<end


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--common-run',type=Path,required=True)
    p.add_argument('--youtube-data',type=Path,required=True)
    p.add_argument('--ui',type=Path,default=AI/'tools/ae_selected_era_listening')
    p.add_argument('--out',type=Path,default=AI/'analysis/ae_selected_era_listening')
    p.add_argument('--unavailable-videos',type=Path,help='JSON list of confirmed unavailable video IDs; reject whole cases only')
    args=p.parse_args();read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    final=read(args.run/'final_report.json');verification=read(args.run/'verification.json')
    if final['status']!='complete' or verification['status']!='passed' or final.get('objective_mode')!='g_only':
        raise ValueError('Complete and verified all5082 protected-G artifacts required')
    if digest(final['embeddings'])!=verification['embeddings_sha256']:raise ValueError('Changed final embeddings')
    with np.load(final['embeddings'],allow_pickle=False) as f:
        e=f['e'].astype(np.float64);perf_ids=f['perf_ids'].tolist();periods=f['era'].tolist()
    np.testing.assert_allclose(np.linalg.norm(e,axis=1),1,atol=1e-6)
    if len(perf_ids)!=5082 or len(set(perf_ids))!=5082:raise ValueError('Incomplete embedding catalogue')
    clips=read(args.common_run/'corpus/manifest.json')['clips'];by_id={c['perf_id']:c for c in clips}
    with (args.youtube_data/'youtube_matches.csv').open(encoding='utf-8-sig',newline='') as f:yt={r['perf_id']:r for r in csv.DictReader(f)}
    if set(yt)!=set(perf_ids):raise ValueError('YouTube and embedding catalogues differ')
    mapping_ok={pid:eligible_mapping(yt[pid],by_id[pid]) for pid in perf_ids}
    signatures=defaultdict(set)
    for r in yt.values():signatures[r['composition_id']].add(tuple(r[k] for k in ['candidate_index','reference_perf_id','reference_start_seconds','reference_end_seconds']))
    if any(len(v)!=1 for v in signatures.values()):raise ValueError('Same-work candidate excerpt differs across performances')
    work_titles={}
    for pid in sorted(perf_ids):
        row=yt[pid];work_titles.setdefault(row['composition_id'],row['composer']+' · '+row['track'])
    helper,helper_text=original_helpers();candidates=[];excluded=Counter()
    canonical={pid:'https://www.youtube.com/watch?v='+helper['youtube_id'](yt[pid]['youtube_url'])
               for pid in perf_ids if mapping_ok[pid]}
    blocked=set(read(args.unavailable_videos)) if args.unavailable_videos else set()
    # #17 has a single legacy source substitution. New dataset timing must
    # never be silently applied to that replacement recording.
    blocked.add('0QIgscF9ezY')
    with threadpool_limits(limits=1):
        for era in QUOTAS:
            global_ids=[i for i,x in enumerate(periods) if x==era];vectors=e[global_ids]
            rows=[dict(performance_id=perf_ids[i],work_id=by_id[perf_ids[i]]['composition_id'],
                       recording_id=by_id[perf_ids[i]]['source_sha256']) for i in global_ids]
            for a,row in enumerate(rows):
                try:b,c,pool,order,dist=helper['select_near_far'](rows,vectors,a)
                except ValueError as error:excluded[str(error)]+=1;continue
                a_id,b_id,c_id=[rows[j]['performance_id'] for j in (a,b,c)]
                if not all(mapping_ok[j] for j in (a_id,b_id,c_id)):excluded['one_or_more_mapping_not_matched_auto']+=1;continue
                video_urls={canonical[j] for j in (a_id,b_id,c_id)}
                if len(video_urls)!=3:excluded['duplicate_video_in_case']+=1;continue
                if any(helper['youtube_id'](url) in blocked for url in video_urls):
                    excluded['confirmed_unavailable_or_legacy_source_substitution']+=1;continue
                candidates.append(dict(era=era,anchor=a_id,near=b_id,far=c_id,gap=float(dist[c]-dist[b]),
                    near_distance=float(dist[b]),far_distance=float(dist[c]),far_rank=order.index(c)+1,
                    candidate_count=len(order),candidate_works=len({rows[j]['work_id'] for j in order}),far_work_pool=len(pool)))
    selected=[];used_works=set();used_videos=set()
    for era,count in QUOTAS.items():
        chosen=0
        for case in sorted((c for c in candidates if c['era']==era),key=lambda c:(-c['gap'],c['anchor'],c['near'],c['far'])):
            ids=[case[k] for k in ('anchor','near','far')]
            works={by_id[j]['composition_id'] for j in ids};videos={canonical[j] for j in ids}
            if works&used_works or videos&used_videos:continue
            selected.append(case);used_works.update(works);used_videos.update(videos);chosen+=1
            if chosen==count:break
        if chosen!=count:raise ValueError(f'Insufficient disjoint valid cases for {era}: {chosen}/{count}')
    policy=dict(source_commit=SOURCE_COMMIT,final_embeddings_sha256=digest(final['embeddings']),
        youtube_matches_sha256=digest(args.youtube_data/'youtube_matches.csv'),era_quotas=QUOTAS,
        ranking='Pinned #17 nearest different-work B then farthest-from-A C within B work; full catalogue within each era; byte-identical MIDI excluded',
        curation='Descending cosine-distance gap within each era; greedy20 disjoint works and30 disjoint video sources; never availability rerank B/C',
        playback='Exact supplied fractional youtube_start_seconds/youtube_end_seconds; no manual re-excerpting',
        unavailable_video_ids=sorted(blocked),
        evidence='Embedding-extreme demonstration cases, not an unbiased listening accuracy sample')
    pack=dict(schema='ae-selected-era-choice-v1',model=dict(epoch='시대별 '+ '/'.join(str(v['production_epoch']) for v in final['eras'].values()),
        objective='protected G only',checkpoint_sha256={era:info['sha256'] for era,info in final['eras'].items()}),
        catalogue=dict(recordings=5082,works=479),selection=policy,
        distance_scope='selected representative clip pooled z plus era head; cosine distance within same era only',cases=[],clips={})
    counters=Counter()
    for case in selected:
        era=case['era'];counters[era]+=1;ids=[case[k] for k in ('anchor','near','far')]
        for pid in ids:
            row=yt[pid];job=args.youtube_data/'jobs'/pid
            req=read(job/'request.json')['metadata'];result=read(job/'result.json');receipt=result['audio_receipt']
            video=receipt['video_id'];url=f'https://www.youtube.com/watch?v={video}'
            if url!=canonical[pid] or receipt.get('timeline_origin_seconds')!=0:raise ValueError('Playback source identity mismatch')
            start,end=float(row['youtube_start_seconds']),float(row['youtube_end_seconds'])
            duration=float(result['timings']['audio_duration_seconds'])
            if not 0<=start<end<=duration:raise ValueError('Mapped YouTube range outside source duration')
            pack['clips']['ATEPP:'+pid]=dict(performance_id='ATEPP:'+pid,artist=row['artist'],title=row['track'],
                album=req.get('album',''),album_date=req.get('album_date',''),video_id=video,youtube_url=url,
                start_seconds=start,end_seconds=end,track_duration_seconds=duration,
                theme=dict(title=work_titles[row['composition_id']],label='데이터셋 선정 대표구간',formal_measures='동일 선정 구간 전체',
                    source_title='제공된 데이터셋 YouTube 대응 구간',source_url=row['youtube_start_link'],
                    evidence='대표구간과 YouTube 재생시간은 제공된 youtube_matches.csv의 자동 매칭 통과 행을 그대로 사용합니다.',
                    mapping_note='X/Y는 같은 작품과 같은 대표구간 기준의 서로 다른 연주입니다. 임베딩도 실제 청취 대표구간 전체에서 계산했습니다. 영상 정렬은 자동 추정이며 청음 검수 전입니다.'),
                source_sha256=row['source_sha256'],midi_start_seconds=float(row['start_seconds']),midi_end_seconds=float(row['end_seconds']),
                youtube_mapping_status=row['youtube_match_status'],youtube_human_verified=row['youtube_human_verified'])
        public_ids=['ATEPP:'+j for j in ids]
        definition=dict(id='selected-'+era+'-'+case['anchor'],label=ERA_LABELS[era]+f' {counters[era]:02d}',
            anchor=public_ids[0],near=public_ids[1],far=public_ids[2],far_work_pool=case['far_work_pool'],
            candidate_count=case['candidate_count'],candidate_works=case['candidate_works'],era=era,gap=case['gap'],
            candidates=[dict(clip=public_ids[j],role='near' if j==1 else 'far',global_rank=1 if j==1 else case['far_rank'],
                cosine_distance_to_anchor=case['near_distance'] if j==1 else case['far_distance'],
                same_artist_as_anchor=yt[ids[j]]['artist']==yt[ids[0]]['artist']) for j in (1,2)])
        pack['cases'].append(definition)
    pack['id']=hashlib.sha256(json.dumps(pack,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    pack['storage_id']=pack['id'];public,results=helper['public_pack'](pack)
    public['distance_scope']=pack['distance_scope'];public['compatible_pack_ids']=[];public['migration_sources']=[]
    args.ui.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    source_files={};changes={}
    for name in ['index.html','style.css','app.js','study.js','player.js','serve.py','study.test.cjs','player.test.cjs']:
        raw=source('tools/ae_theme_listening/'+name);s=raw.decode('utf-8');original=s
        if name=='app.js':
            s=s.replace('distance_scope:"whole-recording mean AE embedding; not theme-only"','distance_scope:data.distance_scope')
            s=s.replace('A와의 전곡 거리','A와의 대표구간 거리')
            s=s.replace('주제 근거와 마디 번호','대표구간 근거와 매핑 정보')
            s=s.replace('가까운 연주는 다른 작품 전체에서','가까운 연주는 같은 시대의 다른 작품 전체에서')
            s=s.replace('주제만의 거리가 아닌 전곡 평균 임베딩 기준입니다.','실제로 청취하는 대표구간 전체의 시대별 임베딩 기준입니다.')
        if name=='index.html':
            s=s.replace('기준 A의 주제를 먼저 들어보세요.','기준 A의 대표구간을 먼저 들어보세요.')
            s=s.replace('페이지 안에서 주제 구간을 재생합니다.','페이지 안에서 대표구간을 재생합니다.')
            s=s.replace('aria-label="주제 구간 재생"','aria-label="대표구간 재생"')
            s=s.replace('선정 방식과 주제 근거','선정 방식과 대표구간 근거')
            s=s.replace('X/Y는 같은 악보의 같은 주제를 듣도록 선정했습니다. 현재 영상 시간은 미검수 추정값입니다.',
                        'X/Y는 같은 작품의 동일 선정 대표구간을 듣도록 선정했습니다. 제공된 YouTube 자동 매핑 시간을 그대로 사용하며 청음 검수 전입니다.')
            s=s.replace('거리는 전곡의 8마디 임베딩을 평균한 cosine 거리이며 발췌만의 거리가 아닙니다.',
                        '거리는 실제 청취하는 대표구간 전체의 임베딩에 시대별 head를 적용한 cosine 거리입니다. 다른 시대의 거리는 비교하지 않습니다.')
            s=s.replace('주제는 음악 형식 분석 문헌과 MusicXML을 대조해 수동 지정했습니다.',
                        '발췌는 제공된 데이터셋의 선정 대표구간과 YouTube 대응 시간을 그대로 사용합니다.')
            s=s.replace('시간은 MIDI 정렬 기준 추정값입니다.','시간은 제공된 YouTube 자동 정렬 추정값입니다.')
            s=s.replace('기준 A와 가장 가까운 다른 작품의 연주 B를 찾은 뒤,','동일 시대 안에서 기준 A와 가장 가까운 다른 작품의 연주 B를 찾은 뒤,')
            s=s.replace('X/Y는 같은 악보의 같은 주제 마디를 비교하도록 지정했습니다. 영상 재생 시간은 미검수 추정값입니다.',
                        'X/Y는 같은 작품의 동일 선정 대표구간을 비교합니다. 제공된 YouTube 자동 매핑 시간을 그대로 사용하며 청음 검수 전입니다.')
            s=s.replace('기준 A에 가장 가까운 다른 작품의 연주 B를 고른 뒤,','동일 시대 안에서 기준 A에 가장 가까운 다른 작품의 연주 B를 고른 뒤,')
            s=s.replace('B/C는 같은 작품의 같은 주제 마디를 사용합니다.','B/C는 같은 작품의 동일 선정 대표구간을 사용합니다.')
            s=s.replace('거리는 전곡의 8마디 임베딩을 평균한 cosine 거리이며 주제만의 거리가 아닙니다.',
                        '거리는 실제 청취하는 대표구간 전체의 임베딩에 시대별 head를 적용한 cosine 거리입니다. 다른 시대의 거리는 비교하지 않습니다.')
            if '전곡의 8마디' in s or '같은 주제 마디' in s:raise ValueError('Stale prototype excerpt description')
        if name=='study.test.cjs':
            replacements={
                'official replacement preserves recording identity, source data, X/Y order and model result':
                "test('new selected clips never use a legacy recording substitution',()=>{\n  const context={window:{}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'data.js'),'utf8'),context);\n  for(const clip of Object.values(context.window.LISTENING_DATA.clips))assert.equal(Choice.sourceVideoId(clip),clip.video_id);\n});",
                'additive pack migration preserves answers, order, notes and timing without changing the previous save':
                "test('new model and excerpts cannot inherit old pack answers',()=>{\n  const context={window:{}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'data.js'),'utf8'),context);\n  const data=context.window.LISTENING_DATA;assert.equal(data.compatible_pack_ids.length,0);\n  assert.equal(Choice.migrateSaved({pack_id:'old-model',orders:{},responses:{},drafts:{},times:{}},data,'now'),null);\n});",
                'all ten questions use background-checked playback sources with audio metadata':
                "test('all ten questions have bounded selected excerpts and separate model result files',()=>{\n  const context={window:{}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'data.js'),'utf8'),context);\n  const data=context.window.LISTENING_DATA;assert.equal(data.cases.length,10);assert.equal(Object.keys(data.clips).length,30);\n  for(const clip of Object.values(data.clips))assert.ok(clip.start_seconds>=0&&clip.end_seconds>clip.start_seconds&&clip.end_seconds<=clip.track_duration_seconds);\n  for(const c of data.cases){assert.equal(c.near,undefined);assert.equal(c.far,undefined);assert.equal(c.candidates.length,2);}\n});"}
            for label,replacement in replacements.items():
                pattern=r"test\('"+re.escape(label)+r"', \(\) => \{[\s\S]*?\n\}\);"
                s,n=re.subn(pattern,lambda _:replacement,s)
                if n!=1:raise ValueError('Pinned data-specific test fixture changed: '+label)
        (args.ui/name).write_bytes(s.encode('utf-8'))
        source_files[name]=dict(original_sha256=hashlib.sha256(raw).hexdigest(),generated_sha256=digest(args.ui/name))
        changes[name]=s!=original
    (args.ui/'data.js').write_text('window.LISTENING_DATA = '+json.dumps(public,ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    (args.ui/'embed_checks.js').write_text('window.EMBED_CHECKS = '+json.dumps(dict(checked_on=None,method='provided-auto-mapping-only; live playback not yet checked',checks={}))+';\n',encoding='utf-8')
    for key,result in results.items():dump(args.ui/'results'/f'{key}.json',result)
    dump(args.out/'listening.json',pack);dump(args.out/'selection_audit.json',dict(policy=policy,eligible_case_count=len(candidates),
        exclusions=dict(excluded),youtube_statuses=dict(Counter(r['youtube_match_status'] for r in yt.values())),selected=selected,
        all_5082_embeddings_used_for_ranking=True,live_playback_verified=False))
    dump(args.out/'source_ui_manifest.json',dict(commit=SOURCE_COMMIT,files=source_files,changed=changes,
        changes_reason='Dataset-specific factual wording, distance scope label and test fixtures; all interaction engines unchanged'))
    (args.out/'source17_prepare.py').write_bytes(helper_text.encode('utf-8'))
    print('Created selected-era AXY pack',pack['id'],'10cases/20works/30recordings',flush=True)


if __name__=='__main__':main()
