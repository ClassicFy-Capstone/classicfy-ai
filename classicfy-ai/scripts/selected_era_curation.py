"""G-verified, title-type-matched demonstration selection (not population evaluation)."""
from collections import Counter, defaultdict
import re
import sys
import unicodedata

import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from embedding.era_note_style import pair_tables


THRESHOLDS = dict(min_embedding_gap=.30, max_near_embedding=.40, min_far_embedding=.65,
                  min_G_target_gap=.15, max_near_G_target=.55, min_far_G_target=.65,
                  min_G_distance_ratio=1.5, min_confidence=.25, max_texture_ratio=1.6)


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text).casefold()
                   if not unicodedata.combining(c))


def structure_tag(title):
    """Conservative title genre/movement proxy, not an analysis of the actual audio."""
    t = normalized(title)
    if re.search(r'\betudes?\b', t): return 'etude'
    forms = {'fugue':r'\bfug(?:ue|a)\b', 'prelude':r'\bprelud(?:e|io)\b',
        'invention':r'\binvention\b','minuet':r'\b(?:minuet(?:to)?|menuet(?:to)?)\b',
        'gigue':r'\bgig(?:ue|a)\b','sarabande':r'\bsaraband[ae]?\b',
        'allemande':r'\ballemande\b','courante':r'\b(?:courante|corrente)\b',
        'gavotte':r'\bgavott[ae]\b','bourree':r'\bbourree\b',
        'rondo':r'\b(?:rondo|rondeau[x]?)\b','nocturne':r'\bnocturne[s]?\b',
        'ballade':r'\bballad[es]?\b','mazurka':r'\bmazurka[s]?\b',
        'polonaise':r'\bpolonaise[s]?\b','waltz':r'\b(?:waltz(?:es)?|valse[s]?)\b',
        'intermezzo':r'\bintermezz[oi]\b','impromptu':r'\bimpromptu[s]?\b',
        'scherzo':r'\bscherz[oi]\b'}
    matches = [(m.start(), name) for name, pattern in forms.items() for m in re.finditer(pattern,t)]
    if matches:
        # An undivided Prelude AND Fugue is ambiguous; movement suffixes resolve it.
        if ':' not in t and re.search(r'(?:prelude.*and.*fugue|fantasia.*and.*fugue)',t): return None
        return max(matches)[1]
    if 'sonata' in t:
        tail=t.rsplit(':',1)[-1]
        movements=re.findall(r':\s*(iv|iii|ii|i|[1-4])\.',t)
        if not movements:return None
        movement={'i':'1','ii':'2','iii':'3','iv':'4'}.get(movements[-1],movements[-1])
        if re.search(r'\b(?:adagio|andante|andantino|largo|lento)\b',tail):return 'sonata_slow_'+movement
        if re.search(r'\b(?:allegro|presto|vivace|allegretto)\b',tail):return 'sonata_fast_'+movement
    return None


def work_tags(rows, metadata):
    votes=defaultdict(Counter)
    for m in metadata:
        tag=structure_tag(rows[m['perf_id']]['track'])
        if tag:votes[m['work']][tag]+=1
    # Conflicting known classifications are excluded rather than guessed.
    return {work:next(iter(counts)) for work,counts in votes.items() if len(counts)==1}


def pair_is_clear(en, ef, gn, gf, confidence_near, confidence_far, median, thresholds=THRESHOLDS):
    if not np.isfinite([en,ef,gn,gf,confidence_near,confidence_far,median]).all() or median<=0:return False
    tn,tf=gn/(gn+median),gf/(gf+median)
    return bool(en<=thresholds['max_near_embedding'] and ef>=thresholds['min_far_embedding']
        and ef-en>=thresholds['min_embedding_gap']
        and tn<=thresholds['max_near_G_target'] and tf>=thresholds['min_far_G_target']
        and tf-tn>=thresholds['min_G_target_gap']
        and gf>=thresholds['min_G_distance_ratio']*max(gn,1e-12)
        and min(confidence_near,confidence_far)>=thresholds['min_confidence'])


def texture_profiles(clips,note_q):
    grouped=defaultdict(list)
    for clip in clips:
        q=np.unique(note_q[clip['start']:clip['stop']]);q=q[np.isfinite(q)]
        if len(q)<2 or q[-1]<=q[0]:continue
        grouped[clip['composition_id']].append(((len(q)-1)/(q[-1]-q[0]),clip['notes']/len(q)))
    return {work:np.median(values,axis=0).tolist() for work,values in grouped.items()}


def texture_ratios(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return (np.maximum(a,b)/np.maximum(np.minimum(a,b),1e-12)).tolist()


def curate(e, perf_ids, periods, metadata, yt, mapping_ok, blocked, profiles, scalers, youtube_id, textures):
    tags=work_tags(yt,metadata); candidates=[];counts={}
    by_id={m['perf_id']:m for m in metadata}
    raw,known,quality=profiles
    for era, saved in scalers.items():
        ids=[i for i,p in enumerate(periods) if p==era]
        md=[{**by_id[pid],'split':'production'} for pid in perf_ids]
        d,q=pair_tables(raw,known,quality,md,saved['scaler'],ids)
        median=saved['calibration']['median']
        vectors=e[ids];vectors=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
        distances=1-vectors@vectors.T
        work_groups=defaultdict(list)
        for local,idx in enumerate(ids):work_groups[md[idx]['work']].append(local)
        eligible={}
        for local,idx in enumerate(ids):
            pid=perf_ids[idx];row=yt[pid]
            if mapping_ok[pid] and youtube_id(row['youtube_url']) not in blocked:
                eligible[local]=dict(pid=pid,work=md[idx]['work'],source=md[idx]['source'],
                    artist=normalized(row['artist']).strip(),video=youtube_id(row['youtube_url']))
        era_cases=[]
        for a,ar in eligible.items():
            tag=tags.get(ar['work'])
            if not tag:continue
            for work,pool in work_groups.items():
                if work==ar['work'] or tags.get(work)!=tag:continue
                if work not in textures or ar['work'] not in textures:continue
                texture=texture_ratios(textures[ar['work']],textures[work])
                if max(texture)>THRESHOLDS['max_texture_ratio']:continue
                valid=[j for j in pool if j in eligible and np.isfinite(d[a,j])
                    and q[a,j]>=THRESHOLDS['min_confidence']
                    and eligible[j]['source']!=ar['source'] and eligible[j]['video']!=ar['video']]
                if len(valid)<2:continue
                # Each B must be G-near and E-near; C must be far under both.
                # Choose the nearest such B in this work with a valid distinct-artist C.
                for b in sorted(valid,key=lambda j:(distances[a,j],eligible[j]['pid'])):
                    far=[c for c in valid if c!=b and eligible[c]['artist']!=eligible[b]['artist']
                         and eligible[c]['source']!=eligible[b]['source'] and eligible[c]['video']!=eligible[b]['video']
                         and pair_is_clear(distances[a,b],distances[a,c],d[a,b],d[a,c],q[a,b],q[a,c],median)]
                    if not far:continue
                    c=max(far,key=lambda j:(distances[a,j],eligible[j]['pid']))
                    en,ef=float(distances[a,b]),float(distances[a,c]);gn,gf=float(d[a,b]),float(d[a,c])
                    tn,tf=gn/(gn+median),gf/(gf+median)
                    full=[j for j in range(len(ids)) if md[ids[j]]['work']!=ar['work'] and md[ids[j]]['source']!=ar['source']]
                    full.sort(key=lambda j:(distances[a,j],perf_ids[ids[j]]))
                    era_cases.append(dict(era=era,anchor=ar['pid'],near=eligible[b]['pid'],far=eligible[c]['pid'],
                        gap=ef-en,near_distance=en,far_distance=ef,
                        near_rank=full.index(b)+1,far_rank=full.index(c)+1,
                        candidate_count=len(full),candidate_works=len({md[ids[j]]['work'] for j in full}),
                        far_work_pool=len(valid)-1,structure_tag=tag,
                        G_near=gn,G_far=gf,G_target_near=tn,G_target_far=tf,G_target_gap=tf-tn,
                        G_confidence_near=float(q[a,b]),G_confidence_far=float(q[a,c]),
                        selection_score=min(ef-en,tf-tn),distinct_candidate_artists=True,texture_ratios=texture,
                        anchor_title=yt[ar['pid']]['track'],candidate_title=yt[eligible[b]['pid']]['track']))
                    break
        candidates.extend(era_cases)
        counts[era]=dict(title_matched_G_verified_cases=len(era_cases),eligible_recordings=len(eligible),
                         classified_works=len({work for work in work_groups if work in tags}))
    return candidates,counts,tags


def select_disjoint(candidates, quotas, by_id, yt, youtube_id):
    """Deterministic quota-constrained selection with backtracking over top cases."""
    grouped={era:sorted([c for c in candidates if c['era']==era],
        key=lambda c:(-c['selection_score'],-c['gap'],c['anchor'],c['near'],c['far'])) for era in quotas}
    selected=[];used_works=set();used_videos=set()
    for era,quota in quotas.items():
        pool=grouped[era]
        def search(start, chosen):
            if len(chosen)==quota:return chosen
            for i in range(start,len(pool)):
                c=pool[i];pids=[c[k] for k in ('anchor','near','far')]
                works={by_id[p]['work'] for p in pids};videos={youtube_id(yt[p]['youtube_url']) for p in pids}
                if works&used_works or videos&used_videos:continue
                used_works.update(works);used_videos.update(videos)
                found=search(i+1,chosen+[c])
                if found:return found
                used_works.difference_update(works);used_videos.difference_update(videos)
            return None
        chosen=search(0,[])
        if not chosen:raise ValueError(f'Not enough disjoint G-verified title-matched cases for {era}: {len(pool)} candidates')
        selected.extend(chosen)
    return selected
