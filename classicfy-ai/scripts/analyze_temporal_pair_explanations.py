"""Explore additional temporal explanations for baseline-close cross-work pairs.

Fixed train-only criteria; numeric explanation study, not recommendation accuracy.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyze_temporal_shape import verified_source_files,restore_channels
from validate_embedding import CHANNELS,collect_groups,summarize,weighted_distances
from features.models import BeatSequence
from features.temporal_shape import fit_change_threshold,summarize_temporal_shape
from features.temporal_duration import summarize_temporal_duration,lagged_correlation
from validate_tempo import plt, setup_style
from analyze_temporal_duration import save_figure

root = Path(__file__).resolve().parents[2]


def dynamics_summary_close(mean_a, width_a, mean_b, width_b):
    """Exploratory 5% bounds, with no perceptual similarity claim."""
    if not np.isfinite([mean_a,width_a,mean_b,width_b]).all() or min(width_a,width_b)<=1e-12:
        return False
    return bool(abs(mean_a-mean_b)<=.05*(width_a+width_b)/2
                and abs(width_a-width_b)<=.05*max(width_a,width_b))


def classify_lag_contrast(a, b, halves_a, halves_b):
    """Require supported contrasts in both original halves; gaps retain signs."""
    gaps=[a['lag1']-b['lag1']]+[halves_a[h]['lag1']-halves_b[h]['lag1'] for h in (0,1)]
    finite=np.isfinite(gaps).all()
    divergent=bool(finite and abs(gaps[0])>=.2
                   and all(np.sign(v)==np.sign(gaps[0]) and abs(v)>=.1 for v in gaps[1:]))
    matched=bool(finite and abs(gaps[0])<=.05 and all(abs(v)<=.1 for v in gaps[1:]))
    return gaps,divergent,matched


def plot_case(kind, pair, items, profiles, metrics, halves, out, pair_count):
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw={'height_ratios': [1.1, 1]})
    indices = [pair['a'], pair['b']]
    colors = ('#2878c8', '#dc6436')
    names = [Path(items[i]['key']).stem for i in indices]
    curves = [items[i]['channels']['dynamics'] for i in indices]
    low = min(float(s.values[s.mask].min()) for s in curves)
    high = max(float(s.values[s.mask].max()) for s in curves)
    for ax, i, sequence, color, name in zip(axes[0], indices, curves, colors, names):
        ax.plot(np.arange(len(sequence)) + 1, sequence.values, color=color, linewidth=.9)
        ax.axhline(profiles[i,4], color=color, linestyle='--', linewidth=1)
        ax.set_ylim(low-.2, high+.2)
        ax.set_xlim(1,len(sequence))
        work = items[i]['piece'].replace('/midi_score.mid','')
        ax.set_title(f'{name}\n{work}', loc='left', fontsize=12)
        ax.set_xlabel('해당 작품의 원래 beat index')
        ax.set_ylabel('정규화 Dynamics 잔차')
    for offset, i, color, name in zip((-.18,.18), indices, colors, names):
        values = [metrics[i]['dynamics']['lag1']] + [h['lag1'] for h in halves[i]['dynamics']]
        axes[1,0].bar(np.arange(3)+offset,values,width=.33,color=color,label=name)
        for x,y in zip(np.arange(3)+offset,values):
            axes[1,0].text(x,y+.015,f'{y:.3f}',ha='center',fontsize=11)
    axes[1,0].set_xticks(np.arange(3),('전곡','전반','후반'))
    axes[1,0].set_ylim(0,1)
    axes[1,0].set_ylabel('인접 값의 Pearson r')
    axes[1,0].set_title('짧은 간격의 지속성: 전곡·전반·후반',loc='left',fontsize=15)
    axes[1,0].legend(frameon=False,loc='upper left')
    data=[]
    definitions=[('평균 잔차',lambda i:profiles[i,4],'.3f'),
                 ('폭 (p95−p5)',lambda i:profiles[i,5],'.3f'),
                 ('큰 변화 비율 (%)',lambda i:100*metrics[i]['dynamics']['large_ratio'],'.2f'),
                 ('양의 수준 run 평균',lambda i:metrics[i]['dynamics']['positive_run'],'.2f'),
                 ('음의 수준 run 평균',lambda i:metrics[i]['dynamics']['negative_run'],'.2f'),
                 ('같은 작품 연주들의 평균 r',lambda i:np.mean([metrics[j]['dynamics']['lag1']
                     for j in range(len(items)) if items[j]['piece']==items[i]['piece']]),'.3f')]
    for label,fn,fmt in definitions:
        data.append([label]+[format(float(fn(i)),fmt) for i in indices])
    axes[1,1].axis('off')
    table=axes[1,1].table(cellText=data,colLabels=['Dynamics 지표']+names,
                         colWidths=[.51,.245,.245],loc='center',cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1,1.8)
    title=('평균·폭이 가까워도, 강약 잔차의 변화 양상은 다를 수 있음' if kind=='divergent'
           else '평균·폭과 강약 잔차의 시간 지표가 함께 가까운 사례')
    fig.suptitle(title,fontsize=20,fontweight='bold',y=.98)
    rank=int(round(pair['baseline_percentile']/100*pair_count))
    fig.text(.5,.92,f"5개 feature 요약 거리 {pair['baseline_distance']:.3f} · 비교 가능 {pair_count:,}쌍 중 가까운 순위 {rank}위 ({pair['baseline_percentile']:.3f}%)",ha='center',fontsize=12)
    fig.text(.5,.035,'각 곡선은 해당 작품의 원래 축 · 무평활 잔차 · 고정 p95 기준의 큰 변화 · 탐색 사례, 청취/추천 검증 없음',ha='center',fontsize=11)
    fig.subplots_adjust(left=.08,right=.98,top=.80,bottom=.12,hspace=.54,wspace=.3)
    save_figure(fig,out,f'{kind}_case')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=root / 'classicfy-ai/analysis/temporal_pair_explanations')
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    ref=json.loads((root.parent/'datasets/temporal_embedding_run01/dataset.json').read_text())
    cache=root.parent/'datasets/feature_normalization_raw.npz'
    if hashlib.sha256(cache.read_bytes()).hexdigest()!=ref['cache_sha256']:
        raise ValueError('Raw cache differs from reference')
    groups,audits,scales,provenance=collect_groups(SimpleNamespace(asap_root=root.parent/'datasets/ASAP',nasap_root=root.parent/'datasets/nASAP',cache=cache,**ref['cohort_settings']),provenance_source_files=verified_source_files())
    if provenance!=ref['source_provenance'] or scales!=ref['scales']:
        raise ValueError('Provenance or normalization scales differ from reference')
    items=[i for i in restore_channels(groups,audits,ref['split']) if i['split']=='train']
    thresholds={r['channel']:float(r['value']) for r in csv.DictReader((root/'classicfy-ai/analysis/temporal_shape/thresholds.csv').open(encoding='utf-8-sig'))}
    for c in CHANNELS:
        fitted = fit_change_threshold((i['channels'][c] for i in items), quantile=.95)
        if not np.isclose(fitted.value,thresholds[c],rtol=0,atol=1e-12):
            raise ValueError(f'Cached threshold differs from train reference: {c}')
    shape={(r['key'],r['channel']):r for r in csv.DictReader((root/'classicfy-ai/analysis/temporal_shape/performance_channels.csv').open(encoding='utf-8-sig'))}
    durations={(r['key'],r['channel']):r for r in csv.DictReader((root/'classicfy-ai/analysis/temporal_duration/performance_durations.csv').open(encoding='utf-8-sig'))}
    correlations={(r['key'],r['channel']):r for r in csv.DictReader((root/'classicfy-ai/analysis/temporal_duration/lag_correlations.csv').open(encoding='utf-8-sig'))}
    items=[i for i in items if i['channels']['tempo'].mask.mean()>=.9]
    profiles=[]; metrics=[]; halves=[]

    def measure(seq,channel):
        s=summarize_temporal_shape(seq,large_threshold=thresholds[channel]);d=summarize_temporal_duration(seq)
        return dict(large_ratio=s['large_ratio'],roughness=s['rms_delta']/np.ptp(np.percentile(seq.values[seq.mask],[5,95])) if np.count_nonzero(seq.mask)>1 and np.ptp(np.percentile(seq.values[seq.mask],[5,95]))>1e-12 else np.nan,
                    positive_run=d['level_positive_run_mean'],negative_run=d['level_negative_run_mean'],positive_count=d['level_positive_run_count'],negative_count=d['level_negative_run_count'],
                    lag1=lagged_correlation(seq,1)['correlation'],lag8=lagged_correlation(seq,8)['correlation'])
    for i,item in enumerate(items):
        valid=item['channels']['tempo'].mask
        profile=summarize(np.array([item['channels'][c].values[valid] for c in CHANNELS]));profiles.append(profile)
        whole={};half={}
        for ci,c in enumerate(CHANNELS):
            key=item['key'];s=shape[key,c];d=durations[key,c];r=correlations[key,c];seq=item['channels'][c]
            whole[c]=dict(large_ratio=float(s['large_ratio']),roughness=float(s['rms_delta'])/profile[2*ci+1] if profile[2*ci+1]>1e-12 else np.nan,positive_run=float(d['level_positive_run_mean']),negative_run=float(d['level_negative_run_mean']),positive_count=int(d['level_positive_run_count']),negative_count=int(d['level_negative_run_count']),lag1=float(r['value_lag1_correlation']),lag8=float(r['value_lag8_correlation']))
            computed=measure(seq,c)
            for field,value in computed.items():
                if not np.isclose(value,whole[c][field],rtol=1e-12,atol=1e-12,equal_nan=True):
                    raise ValueError(f'Cached metric differs from original source: {key}/{c}/{field}')
            middle=len(seq)//2
            half[c]=[measure(BeatSequence(seq.values[a:b],seq.mask[a:b]),c) for a,b in [(0,middle),(middle,len(seq))]]
        metrics.append(whole);halves.append(half)
    profiles=np.array(profiles);dist=weighted_distances(profiles,profiles,'all')
    pi,pj=np.triu_indices(len(items),1)
    lengths=np.array([len(i['channels']['tempo']) for i in items]);pieces=np.array([i['piece'] for i in items])
    eligible=(pieces[pi]!=pieces[pj])&(np.maximum(lengths[pi],lengths[pj])/np.minimum(lengths[pi],lengths[pj])<=2)
    pi,pj=pi[eligible],pj[eligible];pd=dist[pi,pj];cut=float(np.percentile(pd,5));close=pd<=cut
    print('eligible items/works/pairs/close/cut',len(items),len(set(pieces)),len(pi),int(close.sum()),cut,flush=True)

    def write(name,rows):
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

    summary=[];effects=[]
    for ci,c in enumerate(CHANNELS):
        for metric in ['large_ratio','roughness','positive_run','negative_run','lag1','lag8']:
            vals=np.array([m[c][metric] for m in metrics]);finite=np.isfinite(vals)
            if metric in ['positive_run','negative_run']:finite &= np.array([m[c][metric.replace('_run','_count')]>=20 for m in metrics])
            valid=finite[pi]&finite[pj];allgap=np.abs(vals[pi[valid]]-vals[pj[valid]])
            threshold=float(np.percentile(allgap,90)) if len(allgap) else np.nan
            local=valid&close;gap=np.abs(vals[pi[local]]-vals[pj[local]])
            summary.append(dict(channel=c,metric=metric,eligible_pairs=int(valid.sum()),close_pairs=int(local.sum()),all_pair_gap_p90=threshold,close_median_gap=float(np.median(gap)),close_gap_gt_p90_count=int((gap>threshold).sum()),close_gap_gt_p90_fraction=float((gap>threshold).mean())))
            x=vals[finite];p=pieces[finite];total=float(np.sum((x-x.mean())**2));between=sum(len(x[p==piece])*(float(x[p==piece].mean())-float(x.mean()))**2 for piece in set(p))
            effects.append(dict(channel=c,metric=metric,performances=len(x),works=len(set(p)),piece_eta2=between/total if total else np.nan,within_piece_sd_median=float(np.median([np.std(x[p==piece]) for piece in set(p) if np.count_nonzero(p==piece)>1]))))
    write('additional_information.csv',summary);write('piece_effects.csv',effects)
    rows=[];candidates=[];consistent=[]
    for z,(a,b,distance,isclose) in enumerate(zip(pi,pj,pd,close)):
        if not isclose:continue
        a=int(a);b=int(b);ma=metrics[a]['dynamics'];mb=metrics[b]['dynamics'];ha=halves[a]['dynamics'];hb=halves[b]['dynamics']
        mean_gap=abs(profiles[a,4]-profiles[b,4]);widtha=profiles[a,5];widthb=profiles[b,5]
        strict=dynamics_summary_close(profiles[a,4],widtha,profiles[b,4],widthb)
        lags,robust,matched=classify_lag_contrast(ma,mb,ha,hb)
        different_composer=items[a]['piece'].split('/')[0]!=items[b]['piece'].split('/')[0]
        row=dict(a=a,b=b,key_a=items[a]['key'],key_b=items[b]['key'],piece_a=items[a]['piece'],piece_b=items[b]['piece'],baseline_distance=float(distance),baseline_percentile=float(100*np.mean(pd<=distance)),length_ratio=float(max(lengths[a],lengths[b])/min(lengths[a],lengths[b])),dynamics_mean_gap=mean_gap,dynamics_width_relative_gap=abs(widtha-widthb)/max(widtha,widthb),dynamics_lag1_gap=abs(lags[0]),dynamics_first_half_lag1_signed_gap=lags[1],dynamics_second_half_lag1_signed_gap=lags[2],strict_dynamics_summary=bool(strict),robust_temporal_gap=bool(robust),matched_lag1=bool(matched),different_composer=bool(different_composer))
        rows.append(row)
        if strict and robust and different_composer:candidates.append(row)
        if strict and matched and different_composer and abs(ma['large_ratio']-mb['large_ratio'])<=.01 and all(ma[n]>=1 and mb[n]>=1 and max(ma[n],mb[n])/min(ma[n],mb[n])<=1.2 for n in ['positive_run','negative_run']):consistent.append(row)
    write('close_pairs.csv',rows)
    selected=dict(divergent=min(candidates,key=lambda r:(r['baseline_distance'],r['key_a'],r['key_b'])) if candidates else None,consistent=min(consistent,key=lambda r:(r['baseline_distance'],r['key_a'],r['key_b'])) if consistent else None)
    (out/'selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    protocol=dict(source_provenance=provenance,cache_sha256=ref['cache_sha256'],split='train',initial_performances=395,initial_works=39,min_coverage=.9,max_length_ratio=2,eligible_performances=len(items),eligible_works=len(set(pieces)),eligible_pairs=len(pi),close_quantile=.05,close_distance_cut=cut,close_pairs=int(close.sum()),baseline='existing 14D means/Rubato median abs and p95-p5; equal weights for five blocks; pedaling subchannels averaged',strict_dynamics='mean gap <= 5% average width and width gap <= 5% maximum width',divergence='Dynamics lag1 gap >= .2 full; same signed gap >= .1 in each original half; different composers; choose minimum baseline distance among qualifying pairs',consistency='Dynamics lag1 gap <= .05 full and <= .1 each half; large ratio gap <= .01; positive and negative observed run mean ratios <=1.2; different composers; choose minimum baseline distance',robust_case_count=len(candidates),consistent_case_count=len(consistent),undefined='NaN is excluded with denominators; level-run comparison requires >=20 runs per side in main aggregate',limits=['Exploratory matched-pair numeric explanation; no preference/listening/style truth','Existing per-work normalization uses cohort; structure and beat duration differences remain','Pair observations are dependent; no significance claim','Half comparisons are within the same recordings and not independent evaluation','Main figure Dynamics was chosen because previous train shuffle review showed clear order dependence'])
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')

    profile_rows = []
    for i,item in enumerate(items):
        profile_rows.append(dict(index=i, key=item['key'], piece=item['piece'], grid=item['grid'],
            length=int(lengths[i]), coverage=float(item['channels']['tempo'].mask.mean()),
            **{f'{c}_{s}':float(profiles[i,2*ci+j]) for ci,c in enumerate(CHANNELS)
               for j,s in enumerate(('median_abs' if c=='rubato' else 'mean', 'p95_p5'))}))
    write('profiles.csv',profile_rows)
    case_rows, curve_rows = [], []
    for kind,pair in selected.items():
        if pair is None:
            continue
        for side in ('a','b'):
            i=pair[side];seq=items[i]['channels']['dynamics']
            middle=len(seq)//2
            for phase,a,b in [('full',0,len(seq)),('first',0,middle),('second',middle,len(seq))]:
                peers=[(metrics[j]['dynamics'] if phase=='full' else halves[j]['dynamics'][0 if phase=='first' else 1])['lag1']
                       for j in range(len(items)) if items[j]['piece']==items[i]['piece']]
                peers=[v for v in peers if np.isfinite(v)]
                part=BeatSequence(seq.values[a:b],seq.mask[a:b]);m=measure(part,'dynamics')
                correlation=lagged_correlation(part,1)
                values=part.values[part.mask]
                case_rows.append(dict(case=kind,side=side,key=items[i]['key'],piece=items[i]['piece'],phase=phase,
                    valid_beats=len(values),lag1_pairs=correlation['valid_pairs'],
                    mean=float(values.mean()),width=float(np.ptp(np.percentile(values,[5,95]))),
                    peer_lag1_mean=float(np.mean(peers)),peer_count=len(peers),
                    lag1_peer_centered=m['lag1']-float(np.mean(peers)),**m))
            curve_rows.extend(dict(case=kind,side=side,key=items[i]['key'],original_index=t,valid=bool(valid),
                                  dynamics=float(value)) for t,(value,valid) in enumerate(zip(seq.values,seq.mask)))
        plot_case(kind,pair,items,profiles,metrics,halves,out,len(pi))
    if case_rows:
        write('case_metrics.csv',case_rows)
        write('case_values.csv',curve_rows)
    protocol['analysis_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol['input_csv_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        root/'classicfy-ai/analysis/temporal_shape/performance_channels.csv',
        root/'classicfy-ai/analysis/temporal_shape/thresholds.csv',
        root/'classicfy-ai/analysis/temporal_duration/performance_durations.csv',
        root/'classicfy-ai/analysis/temporal_duration/lag_correlations.csv']}
    protocol['close_dynamics_lag1_gap_ge_02'] = sum(float(r['dynamics_lag1_gap'])>=.2 for r in rows)
    protocol['close_dynamics_gap_robust_in_halves'] = sum(r['robust_temporal_gap'] for r in rows)
    protocol['strict_dynamics_summary_pairs'] = sum(r['strict_dynamics_summary'] for r in rows)
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(f"Saved {len(items)} performances, {len(pi)} eligible pairs and {int(close.sum())} close pairs to {out}")


if __name__ == '__main__':
    main()
