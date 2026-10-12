"""Same-work feature differences and transparent cross-work retrieval examples.

Run: .venv/bin/python classicfy-ai/scripts/validate_interpretation_examples.py
Reuses the fixed 14-coordinate embedding and distance from validate_embedding.py.
Percentile profiles are illustrations, never inputs to retrieval.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from validate_embedding import (BLOCKS, CHANNELS, collect_groups, summarize, weighted_distances)
from validate_feature_normalization import read_cache
from validate_tempo import BLUE, GRAY, ORANGE, GREEN, chart, plt, save, setup_style, write_csv
from features import (BeatSequence, PieceFeatureInput, TempoInput, extract_piece_tempo_features,
                      extract_piece_rubato_features, separate_piece_feature)
from features.articulation import build_tempo_map, extract_note_articulation
from preprocessing import ASAPLoader, load_match, load_midi

DISPLAY_COORDS = [0, 3, 4, 6, 8]
DISPLAY_LABELS = ["빠르기\nTempo", "루바토 폭\nRubato", "강약 성향\nDynamics",
                  "음 유지 성향\nArticulation", "페달 깊이\nPedaling"]
CHANNEL_LABELS = ["Tempo", "Rubato", "Dynamics", "Articulation", "페달 깊이", "페달 on 비율", "페달 전환"]
COLORS = {"A": BLUE, "B": GREEN, "C": GRAY}


def piece_label(piece):
    return piece.removesuffix("/midi_score.mid").replace("/", " · ").replace("bwv_", "BWV ")


def percentile_profile(vector, cohort):
    """Midrank percentile; equal values all map to 50. No input mutation."""
    equal = np.isclose(cohort, vector, atol=1e-12, rtol=0)
    less = (cohort < vector) & ~equal
    return 100 * (less.sum(axis=0) + .5 * equal.sum(axis=0)) / len(cohort)


def cross_selection(vectors, pieces, query_index):
    """Global nearest outside query work; farthest within that neighbor's work."""
    distances = weighted_distances(vectors[query_index:query_index+1], vectors, "all")[0]
    eligible = np.flatnonzero(pieces != pieces[query_index])
    if not len(eligible):
        raise ValueError("Need candidates from another piece")
    b = int(eligible[np.argmin(distances[eligible])])
    contrasts = np.flatnonzero(pieces == pieces[b])
    if len(contrasts) < 2:
        raise ValueError("Need multiple performances of the target piece")
    c = int(contrasts[np.argmax(distances[contrasts])])
    if c == b:
        # Completely tied target work still needs a distinct contrast; report tie.
        c = int(next(i for i in contrasts if i != b))
    return b, c, distances


def vector_rows(groups):
    rows, vectors = [], []
    for group in groups:
        for key, values in zip(group["keys"], group["values"]):
            rows.append({"key": key, "piece": group["piece"], "grid": group["grid"]})
            vectors.append(summarize(values))
    return rows, np.array(vectors)


def profile_plot(selected, vectors, pieces, labels, colors, out, title, subtitle):
    fig, ax = chart(title, subtitle)
    markers = ["o", "s", "^"]
    for i, label, color, marker in zip(selected, labels, colors, markers):
        ranks = percentile_profile(vectors[i], vectors[pieces == pieces[i]])
        ax.plot(np.arange(5), ranks[DISPLAY_COORDS], color=color, marker=marker, ms=8, lw=2.2, label=label)
    ax.axhline(50, color=GRAY, ls=":", lw=1)
    ax.set(xticks=np.arange(5), xticklabels=DISPLAY_LABELS, ylim=(0,100), yticks=[0,25,50,75,100],
           ylabel="자기 작품 안에서의 상대 순위 (백분위)")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.55,.07), fontsize=11, ncols=1)
    save(fig,out,"위로 갈수록 해당 작품 안에서 큰 값이다. Pedaling은 깊이를 대표로 표시한다. 검색에는 이 백분위가 아닌 14차원 벡터를 쓴다.",bottom=.27)


def make_raw_sequences(group, records, loader):
    rs = [records[k] for k in group["keys"]]
    samples = [loader.get_sample(r["key"]) for r in rs]
    tempo = extract_piece_tempo_features([TempoInput.from_asap_sample(s) for s in samples])
    rubato = extract_piece_rubato_features(tempo)
    outputs = {}
    for channel in CHANNELS:
        if channel == "tempo":
            outputs[channel] = {k: (BeatSequence([i.score_relative_tempo if i.mask else np.nan for i in f.intervals],
                                                [i.mask for i in f.intervals]), f.common_tempo_sequence,
                                    f.individual_tempo_sequence, f.common_tempo_support) for k,f in tempo.items()}
        elif channel == "rubato":
            outputs[channel] = {k:(f.absolute_rubato_sequence, f.common_rubato_sequence,
                                  f.relative_rubato_sequence, f.common_rubato_support) for k,f in rubato.items()}
        else:
            separated = separate_piece_feature([PieceFeatureInput(r["key"], group["piece"], r["score_beats"],
                       BeatSequence(r[f"{channel}_values"], r[f"{channel}_mask"]), r["score_types"]) for r in rs])
            outputs[channel] = {k:(v.raw, v.common, v.relative, v.common_support) for k,v in separated.items()}
    return outputs, tempo


def draw_same_work(group, selected, anchor, rows, raw, tempo, scales, out, width):
    out.mkdir(parents=True,exist_ok=True)
    reference = next(iter(raw["tempo"].values()))[0]
    start = (len(reference)-min(width,len(reference)))//2
    section = slice(start,start+min(width,len(reference)))
    x = np.arange(start+1, section.stop+1)
    other_colors = iter([GRAY, ORANGE])
    colors = [BLUE if i==anchor else next(other_colors) for i in selected]
    labels = [("기준 A · " if i==anchor else "비교 연주 · ")+Path(rows[i]["key"]).stem for i in selected]
    specs = [("tempo",lambda v:100*np.exp2(v),"같은 악보의 beat도 연주 속도가 다르다","악보 기준 속도 대비 비율 (%)"),
             ("rubato",lambda v:100*np.expm1(np.log(2)*v),"전체 빠르기를 빼도 국소 속도 변화가 다르다","연주 자신의 전체 빠르기 대비 편차 (%)"),
             ("dynamics",lambda v:127*v,"같은 위치에서도 음을 치는 세기가 다르다","beat 평균 MIDI velocity"),
             ("articulation",lambda v:100*np.exp2(v),"같은 음가에서도 건반을 누르는 길이가 다르다","해당 위치 tempo의 기대 음 길이 대비 (%)"),
             ("pedal_depth",lambda v:100*v,"페달을 얼마나 깊게 사용하는지가 다르다","시간 평균 페달 깊이 (%)"),
             ("pedal_down_ratio",lambda v:100*v,"페달을 on으로 유지하는 시간이 다르다","페달 on 시간 비율 (%)"),
             ("pedal_changes",lambda v:v,"페달을 밟고 떼는 횟수가 다르다","beat당 on/off 전환 횟수")]
    beat_rows=[]
    for number,(channel,convert,title,ylabel) in enumerate(specs,2):
        fig,ax=chart(title,f"{piece_label(group['piece'])} | 같은 score 위치 | 중앙 {start+1}~{section.stop}구간")
        common = next(iter(raw[channel].values()))[1]
        ax.plot(x,convert(np.where(common.mask,common.values,np.nan)[section]),color="#242b27",ls="--",lw=2,label="작품 공통 중앙값")
        for i,label,color in zip(selected,labels,colors):
            key=rows[i]["key"]
            original,common,relative,support=raw[channel][key]
            ax.plot(x,convert(np.where(original.mask,original.values,np.nan)[section]),color=color,marker="o",ms=3,lw=1.8,label=label)
            for j in range(len(original)):
                beat_rows.append({"key":key,"channel":channel,"beat_index":j,"display_interval":j+1,
                                  "raw":float(original.values[j]),"common":float(common.values[j]),"relative":float(relative.values[j]),
                                  "standardized":float(relative.values[j]/scales[channel]),"raw_mask":bool(original.mask[j]),
                                  "common_mask":bool(common.mask[j]),"relative_mask":bool(relative.mask[j]),
                                  "support":int(support[j]),"scale":scales[channel],"tempo_status":tempo[key].intervals[j].status})
        ax.set(xlabel="같은 score beat 구간 번호 (1부터)",ylabel=ylabel,xlim=(x[0],x[-1]))
        fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.55,.06),fontsize=10,ncols=2)
        save(fig,out/f"{number:02d}_{channel}_curve.png","raw와 작품 공통값을 함께 표시한다. 모델은 relative/작품 scale을 사용한다. 결측 보간·평활화·clipping 없음.",bottom=.23)
    write_csv(out/"beat_features.csv",beat_rows)
    return [start+1,section.stop]


def draw_cross_work(selected, vectors, rows, distances, out):
    out.mkdir(parents=True,exist_ok=True)
    labels=[f"{role} · {Path(rows[i]['key']).stem}" for role,i in zip("ABC",selected)]
    for offset,title,filename in [(0,"세 연주의 성향 좌표를 직접 비교한다","02_embedding_tendencies.png"),
                                   (1,"검색에 함께 들어간 변화 폭도 비교한다","03_embedding_widths.png")]:
        fig,ax=chart(title,f"A: {piece_label(rows[selected[0]]['piece'])} | B·C: {piece_label(rows[selected[1]]['piece'])}")
        for role,i,label,marker in zip("ABC",selected,labels,["o","s","^"]):
            ax.plot(np.arange(7),vectors[i,offset::2],color=COLORS[role],marker=marker,lw=2,ms=7,label=label)
        if offset==0:ax.axhline(0,color=GRAY,ls=":")
        ax.set(xticks=np.arange(7),xticklabels=CHANNEL_LABELS,ylabel="standardized relative의 5~95백분위 폭" if offset else "standardized relative 평균 (Rubato는 절댓값 중앙값)")
        fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.55,.07),fontsize=11,ncols=3)
        save(fig,out/filename,"7개 좌표는 Tempo·Rubato·Dynamics·Articulation과 페달 3개 지표다. 모든 특징이 일치해야 가까운 후보가 되는 것은 아니다.",bottom=.19)
    target = [i for i,r in enumerate(rows) if r["piece"]==rows[selected[1]]["piece"]]
    target.sort(key=lambda i:(distances[i],rows[i]["key"]))
    fig,ax=chart("다른 작품의 같은 후보군에서 B를 먼저 찾았다",f"기준 A: {Path(rows[selected[0]]['key']).stem} | 후보 작품: {piece_label(rows[selected[1]]['piece'])}",height=max(5.8,2.5+.3*len(target)))
    values=[distances[i] for i in target]
    ax.barh(np.arange(len(target)),values,color=[GREEN if i==selected[1] else GRAY for i in target])
    for j,i in enumerate(target):ax.text(distances[i]+.02,j,f"{distances[i]:.3f}",va="center")
    ax.set(yticks=np.arange(len(target)),yticklabels=[("B · " if i==selected[1] else "C · " if i==selected[2] else "")+Path(rows[i]["key"]).stem for i in target],xlabel="A와의 기존 임베딩 거리 (작을수록 가까움)",xlim=(0,max(values)*1.2))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.invert_yaxis()
    save(fig,out/"04_target_candidates.png","B는 다른 작품의 전체 후보 중 거리 최소, C는 B의 작품 안에서 거리 최대인 대조 예시다. 이 선택 자체가 독립 성능 검증은 아니다.")
    fig,ax=chart("B와 C의 차이는 어떤 feature에서 생겼을까?","각 feature 묶음의 평균 제곱 차이 | 전체 거리²는 아래 5개 값의 평균")
    x=np.arange(5)
    for role,i,shift in [("B",selected[1],-.18),("C",selected[2],.18)]:
        vals=[float(np.mean((vectors[selected[0],indices]-vectors[i,indices])**2)) for indices in BLOCKS.values()]
        ax.bar(x+shift,vals,width=.34,color=COLORS[role],label=f"A와 {role}의 차이")
    ax.set(xticks=x,xticklabels=list(BLOCKS),ylabel="feature 묶음의 평균 제곱 차이 (작을수록 유사)")
    ax.legend(fontsize=11)
    save(fig,out/"05_feature_distance.png","아티큘레이션과 페달 등 여러 축을 함께 보고 검색한 결과다. B가 모든 축에서 C보다 가까운 것은 아니다.")


def source_examples(group, selected, rows, loader, records, out):
    out.mkdir(parents=True,exist_ok=True)
    note_rows,pedal_rows,ratio_rows,beat_rows=[],[],[],[]
    snippets=[]
    count=len(records[group["keys"][0]]["score_beats"])-1
    start=(count-4)//2
    for i in selected:
        key=rows[i]["key"];sample=loader.get_sample(key);midi=load_midi(sample.performance_path)
        left,right=sample.performance_beats[start],sample.performance_beats[start+4]
        notes=[n for n in midi.notes if n.start<right and n.end>left]
        snippets.append((key,sample,midi,left,right,notes))
        alignment=load_match(sample.note_alignment_path)
        positions,times=build_tempo_map(alignment);features=extract_note_articulation(alignment)
        for match_index,onset,value in zip(features.match_indices,features.onsets,features.values):
            if left<=onset<right:
                score,performed=alignment.matches[int(match_index)]
                expected=float(np.interp(score.offset_beats,positions,times)-np.interp(score.onset_beats,positions,times))
                np.testing.assert_allclose(value,np.log2(performed.duration/expected),atol=1e-12)
                matches=[n for n in midi.notes if n.pitch==performed.pitch]
                nearest=min(matches,key=lambda n:abs(n.start-performed.onset)+abs(n.end-performed.offset))
                timing_error=max(abs(nearest.start-performed.onset),abs(nearest.end-performed.offset))
                if timing_error>.002:
                    raise ValueError(f"MIDI/match timestamp discrepancy above 2 ms: {key} / {match_index}")
                ratio_rows.append({"key":key,"match_index":int(match_index),"pitch":performed.pitch,"onset_seconds":performed.onset,
                                   "actual_duration_seconds":performed.duration,"expected_duration_seconds":expected,"note_log2":value,
                                   "midi_match_max_timing_error_seconds":timing_error})
        for note in notes:
            note_rows.append({"key":key,"pitch":note.pitch,"onset_seconds":note.start,"offset_seconds":note.end,
                              "duration_seconds":note.end-note.start,"velocity":note.velocity,"instrument_idx":note.instrument_idx,
                              "snippet_start_seconds":left,"snippet_end_seconds":right})
        for event in midi.pedals:
            pedal_rows.append({"key":key,"time_seconds":event.time,"value":event.value,"instrument_idx":event.instrument_idx})
        for j in range(start,start+4):
            beat_left,beat_right=sample.performance_beats[j:j+2]
            ons=[n.velocity for n in midi.notes if sample.performance_beats[j]<=n.start<sample.performance_beats[j+1]]
            raw=records[key]["dynamics_values"][j]
            np.testing.assert_allclose(np.mean(ons)/127 if ons else np.nan,raw,atol=1e-12)
            used=[r["note_log2"] for r in ratio_rows if r["key"]==key and sample.performance_beats[j]<=r["onset_seconds"]<sample.performance_beats[j+1]]
            np.testing.assert_allclose(np.median(used) if used else np.nan,records[key]["articulation_values"][j],atol=1e-12)
            if len({p.instrument_idx for p in midi.pedals})>1:
                raise ValueError("Source illustration expects one CC64 instrument")
            depth_area,on_area,changes,state,cursor=0.,0.,0,0,beat_left
            for event in midi.pedals:
                if event.time<beat_left:
                    state=event.value
                elif event.time<beat_right:
                    depth_area+=(event.time-cursor)*state/127
                    on_area+=(event.time-cursor)*int(state>=64)
                    changes+=int((state>=64)!=(event.value>=64))
                    state,cursor=event.value,event.time
                else:
                    break
            depth_area+=(beat_right-cursor)*state/127
            on_area+=(beat_right-cursor)*int(state>=64)
            pedal_values=[depth_area/(beat_right-beat_left),on_area/(beat_right-beat_left),changes]
            for channel,value in zip(CHANNELS[4:],pedal_values):
                np.testing.assert_allclose(value,records[key][f"{channel}_values"][j],atol=1e-12)
            beat_rows.append({"key":key,"display_interval":j+1,"start_seconds":sample.performance_beats[j],
                              "end_seconds":sample.performance_beats[j+1],"note_on_count":len(ons),"mean_velocity":127*raw,
                              "valid_matched_note_count":len(used),"articulation_log2":records[key]["articulation_values"][j],
                              **dict(zip(CHANNELS[4:],pedal_values))})
    pitch_min=min(n.pitch for _,_,_,_,_,notes in snippets for n in notes)-2
    pitch_max=max(n.pitch for _,_,_,_,_,notes in snippets for n in notes)+2
    x_max=max(right-left for _,_,_,left,right,_ in snippets)
    for number,(key,sample,midi,left,right,notes) in enumerate(snippets,1):
        fig,ax=chart("원본 MIDI에서 음 시작·유지 시간·세기를 확인한다",f"{piece_label(group['piece'])} | {Path(key).stem} | 같은 score {start+1}~{start+4}구간")
        for n in notes:ax.broken_barh([(n.start-left,n.end-n.start)],(n.pitch-.35,.7),facecolors=plt.cm.viridis(n.velocity/127))
        for j in range(start,start+5):
            time=sample.performance_beats[j]-left
            ax.axvline(time,color=GRAY,ls=":",lw=1)
            if j<start+4:ax.text(time+.02,pitch_max-.5,str(j+1),fontsize=10,va="top")
        ax.set(xlabel="각 연주의 선택 구간 시작부터 실제 시간 (초)",ylabel="MIDI pitch",xlim=(0,x_max),ylim=(pitch_min,pitch_max))
        fig.legend(handles=[Patch(color=plt.cm.viridis(v/127),label=f"velocity {v}") for v in [45,70,95]],loc="lower center",bbox_to_anchor=(.55,.07),fontsize=10,ncols=3)
        save(fig,out/f"{number:02d}_{Path(key).stem}_notes.png","막대 시작=note-on, 길이=note-off까지의 실제 유지 시간, 색=velocity. 세 연주의 pitch·초 눈금 범위를 동일하게 표시한다.",bottom=.21)
        fig,ax=chart("같은 구간의 원본 페달 신호도 확인한다",f"{piece_label(group['piece'])} | {Path(key).stem} | MIDI CC64")
        times=np.array([p.time for p in midi.pedals]);values=np.array([p.value for p in midi.pedals])
        prior=np.flatnonzero(times<=left)
        initial=values[prior[-1]] if len(prior) else 0
        inside=(times>left)&(times<right)
        xs=np.r_[0,times[inside]-left,right-left];ys=np.r_[initial,values[inside],values[inside][-1] if inside.any() else initial]
        ax.step(xs,ys,where="post",color=BLUE,lw=2)
        ax.axhline(64,color=ORANGE,ls="--",label="on 기준 CC64 ≥ 64")
        ax.set(xlabel="각 연주의 선택 구간 시작부터 실제 시간 (초)",ylabel="CC64 원본 값",xlim=(0,x_max),ylim=(-5,132))
        ax.legend(fontsize=10)
        save(fig,out/f"{number:02d}_{Path(key).stem}_cc64.png","CC64가 64 미만이어도 depth에는 기여한다. 이 구간에서 변화가 없으면 수평선이며, 결측이 아니라 원본 신호다.")
    for name,data in [("note_events.csv",note_rows),("cc64_events.csv",pedal_rows),("note_ratios.csv",ratio_rows),("beat_checks.csv",beat_rows)]:
        write_csv(out/name,data)
    return [start+1,start+4], {"checked_beats":len(beat_rows),"checked_matched_notes":len(ratio_rows),
                             "max_midi_match_timing_error_seconds":max(r["midi_match_max_timing_error_seconds"] for r in ratio_rows),
                             "timestamp_tolerance_seconds":.002,
                             "checks":"MIDI velocity mean, matched-note duration ratio/beat median, CC64 integral/on ratio/transitions agree with cache"}


def write_report(out, stats, profiles, variants):
    """Regenerate the reading guide alongside the figures and source tables."""
    selected=stats["selected"]
    a,b,c=[selected[role] for role in "ABC"]
    names={role:Path(selected[role]["key"]).stem for role in "ABC"}
    profile_table=[]
    for role in "ABC":
        rs=[r for r in profiles if r["panel"]=="cross_work" and r["key"]==selected[role]["key"]]
        ranks={r["coordinate"]:r["within_piece_percentile"] for r in rs}
        profile_table.append(f"| {role} · {names[role]} | "+" | ".join(f"{ranks[i]:.1f}" for i in DISPLAY_COORDS)+" |")
    source_links=[]
    for number,key in enumerate(stats["same_work_keys"],1):
        name=Path(key).stem
        source_links.append(f"| {name} | [음표 원자료](source_midi/{number:02d}_{name}_notes.png) | [CC64 원자료](source_midi/{number:02d}_{name}_cc64.png) |")
    checks=stats["source_checks"]
    text=f"""# 같은 작품의 연주 차이와 다른 작품의 유사 특징 검색

현재 다섯 feature를 요약한 임베딩으로 **연주별 차이를 표현하고 다른 작품에서 특징상 가까운 연주를 검색할 수 있는지** 실제 사례를 확인한다.
기존 [원본 연주 복구 실험](../_embedding_validation/README.md)과는 다른 질문이다.
여기서는 기존 14차원 요약 벡터와 거리 함수를 그대로 사용한다. 학습된 신경망 임베딩이나 청취 평가 결과는 아니다.
PNG 파일 하나에 그래프 하나씩 담았다.

## 먼저 볼 그림 세 장

1. [같은 작품의 3연주 profile](same_work/01_profile.png): 악보가 같아도 연주별 성향이 다르다.
2. [다른 작품의 A·B·C profile](cross_work/01_profile.png): 자기 작품 안의 상대적 경향을 비교한다.
3. [B와 같은 작품의 후보 거리](cross_work/04_target_candidates.png): 검색이 실제로 어떤 후보를 골랐는지 확인한다.

## ① 같은 작품에서도 연주별 특징이 다르다

작품은 **{piece_label(a['piece'])}**다. 해당 작품/grid의 유효 후보 전체로 공통값과 scale을 계산했다.
그중 기준 A와 Tempo 평균 순위 25%·75% 근처의 연주를 골랐다. 중복되면 중앙/끝 순위를 사용한다.
선택한 파일은 {', '.join(Path(k).stem for k in stats['same_work_keys'])}다.
이 세 연주만으로 공통값을 다시 계산한 것이 아니다.

profile은 전체 유효 공통 beat의 요약이며, 곡선은 중앙 {stats['curve_display_intervals'][0]}~{stats['curve_display_intervals'][1]}구간이다.
차이가 가장 큰 구간을 찾아 자르지 않았다. 가로축은 같은 score beat **구간 번호**이며 마디 번호가 아니다.
각 곡선의 점선은 작품 내 위치별 공통 중앙값이다. 원래 단위로 표시해 수치를 이해하기 쉽게 했다.
검색에는 공통값을 제거하고 scale로 나눈 값의 요약을 사용한다.

| 그림 | 쉽게 읽는 방법 |
|---|---|
| [Tempo](same_work/02_tempo_curve.png) | 같은 위치에서 위에 있는 연주가 빠르다. 100%는 score MIDI 기준 속도다. |
| [Rubato](same_work/03_rubato_curve.png) | 각 연주 자신의 전체 빠르기를 제거했다. 0 위는 자기 평소보다 빠르고, 아래는 느리다. |
| [Dynamics](same_work/04_dynamics_curve.png) | 같은 위치에서 velocity가 크면 건반을 더 강하게 친 기록이다. |
| [Articulation](same_work/05_articulation_curve.png) | 해당 위치 tempo로 예상한 음 길이에 비해 건반을 유지한 비율이다. |
| [페달 깊이](same_work/06_pedal_depth_curve.png) | 높은 값은 CC64를 깊게 유지한 시간이 많다는 뜻이다. |
| [페달 on 비율](same_work/07_pedal_down_ratio_curve.png) | CC64 ≥ 64인 시간이 beat 안에서 차지하는 비율이다. |
| [페달 전환](same_work/08_pedal_changes_curve.png) | beat 안에서 on/off 경계를 넘은 횟수다. |

같은 beat의 곡선이 다르고, 공통값을 빼고 요약한 profile도 서로 다르다.
따라서 이 사례에서는 **“추출한 특징으로 연주별 차이를 표현할 수 있다”**고 설명할 수 있다.
작품 내 중앙 패턴은 음악적 정답이 아니라 이 데이터의 연주들로 만든 기준이다.

## MIDI 원자료: 이 수치는 어디서 나왔나?

곡선 안의 중앙 {stats['source_display_intervals'][0]}~{stats['source_display_intervals'][1]}구간을 실제 MIDI에서 가져왔다.
세 연주의 pitch 범위와 초 단위 가로축 범위를 같게 했다. beat의 시간 간격이 서로 다른 것도 보인다.

| 연주 | note-on/off·velocity | 페달 신호 |
|---|---|---|
{chr(10).join(source_links)}

음표 그림에서 막대 시작은 note-on, 막대 길이는 note-off까지의 실제 건반 유지 시간, 색은 velocity다.
점선은 score beat가 연주에서 시작하는 실제 시각이다. note-on 간격은 속도에, velocity는 Dynamics에,
실제 길이/tempo 기반 기대 길이는 Articulation에 연결된다. CC64 그림은 페달 지표의 원신호다.
한 beat에서 시작한 여러 음의 velocity는 평균, 유효 정렬 음의 Articulation은 중앙값으로 요약한다.
소리의 데시벨이나 페달로 울리는 길이를 직접 측정한 것은 아니다.

독립 재계산한 {checks['checked_beats']}개 beat에서 velocity 평균·Articulation 중앙값·CC64 적분/전환값이 캐시와 일치했다.
유효 정렬 음 {checks['checked_matched_notes']}개의 note-on/off를 MIDI와 대조한 최대 시간 차이는
{1000*checks['max_midi_match_timing_error_seconds']:.3f} ms다(기존 정렬 검증 허용값 2 ms).
원본 timestamp를 그림에서 수정하지 않았다.

## ② 다른 작품에서도 비슷한 연주 경향을 찾을 수 있다

| 역할 | 작품 | 실제 연주 파일 | A와의 거리 |
|---|---|---|---:|
| A · 기준 | {piece_label(a['piece'])} | `{names['A']}.mid` | — |
| B · 가까운 후보 | {piece_label(b['piece'])} | `{names['B']}.mid` | {stats['distance_AB']:.3f} |
| C · 대조 후보 | {piece_label(c['piece'])} | `{names['C']}.mid` | {stats['distance_AC']:.3f} |

A는 고정한 기준 작품에서 **페달 깊이 평균이 중앙값 이상인 연주 중 Rubato 절댓값 중앙값이 가장 큰 연주**다.
B는 A의 작품을 제외한 {stats['eligible_cross_work_candidates']}개 후보를 모두 비교해 거리 최소로 찾았다.
C는 B와 같은 작품의 후보 중 A와의 거리가 가장 큰 연주다. C를 멀리 있는 대조로 선택한 조건을 숨기지 않는다.
비교가 잘 나오는 A를 찾으려고 전체 작품의 기준 연주를 순회하며 최적화하지 않았다.
전체 {stats['performances']}개 연주의 다른 작품 최근접 결과도 CSV로 저장했다.

쉬운 [A·B·C profile](cross_work/01_profile.png)은 각 요약값을 **자기 작품 후보 안에서의 백분위**로 표시한다.
50 부근은 중간 순위이며, 위로 갈수록 해당 특징의 요약값이 크다. 50이 residual=0이라는 뜻은 아니다.
선은 다섯 범주를 연결한 안내선으로, 범주 사이의 중간 값에는 의미가 없다.

| 연주 | 빠르기 | 루바토 폭 | 강약 성향 | 음 유지 성향 | 페달 깊이 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(profile_table)}

이번 결과에서 A와 B는 자기 작품에서 **루바토 변화 폭이 상위이고 페달 깊이도 큰 편**이다.
C는 페달 깊이가 작은 편이고, 음 유지 성향이 B보다 높다. 이런 차이가 여러 축의 거리에 반영된다.
빠르기와 Dynamics 등 일부 축에서는 C가 A에 더 가깝다. B가 모든 축에서 더 닮은 것은 아니다.

**루바토 폭과 대표 변화량을 구분해야 한다.** 쉬운 profile의 Rubato는 `p95-p5` 변화 폭이다.
검색에 함께 들어가는 `median(abs(relative/scale))`는 B가 A보다 작으며, B의 작품 내 최상위도 아니다.
따라서 “A와 B는 모두 루바토 양이 큰 연주”라고 넓게 설명하면 정확하지 않다.

| 추가 그림 | 확인할 내용 |
|---|---|
| [성향 좌표 7개](cross_work/02_embedding_tendencies.png) | 실제 검색값의 평균과 Rubato 절댓값 중앙값. |
| [변화 폭 좌표 7개](cross_work/03_embedding_widths.png) | `p95-p5`이며, Rubato 폭의 A·B 유사성과 Articulation 폭의 C 차이. |
| [B 작품의 모든 후보 거리](cross_work/04_target_candidates.png) | 같은 곡 후보끼리도 A와의 거리가 다르다. |
| [feature별 거리 성분](cross_work/05_feature_distance.png) | B는 특히 Articulation·Pedaling과 Rubato를 함께 보았을 때 C보다 가깝다. |

백분위는 이해를 돕는 **표시용**이다. 검색에는 아래 14차원 벡터를 쓰므로 profile의 눈으로 보이는 선 간격이 검색 거리와 같지는 않다.

## 정규화와 검색 방법

1. 동일 작품/grid에서 유효 beat의 위치별 중앙값을 common으로 만들고 D/A/P는 `relative = raw - common`을 계산한다.
2. Tempo는 기존 `individual_tempo`, Rubato는 기존 `relative_rubato`를 그대로 쓴다. 공통값 제거를 중복하지 않는다.
3. 각 작품/grid의 모든 유효 residual을 모아 채널별 scale을 계산하고 `standardized = relative / scale`을 사용한다.
4. Tempo·Rubato·Dynamics·Articulation은 MAD, 페달의 depth/down_ratio/changes는 각각 독립 SD다.
   MAD는 `1.4826 × median(|r - median(r)|)`이고, 0이면 IQR/1.349 → SD → unit 1 순으로 fallback한다.
   residual 자체를 다시 중앙값으로 이동하거나 clipping하지 않는다. 최종 선택 scale은 [scales.csv](scales.csv)에 기록했다.
5. 같은 작품/grid의 모든 후보와 7채널에서 동시에 유효한 beat만 요약한다. mask가 검색 단서가 되지 않게 한다.
6. 채널당 성향 1개 + `p95-p5` 폭 1개로 총 14좌표다. 성향은 평균이고 Rubato만 절댓값 중앙값이다.
7. Tempo/Rubato/Dynamics/Articulation/Pedaling을 다섯 묶음으로 비교한다.
   각 묶음의 좌표별 제곱 차이를 평균한 뒤, 다섯 묶음을 같은 가중치로 평균하고 제곱근을 취한다.
   Pedaling은 6좌표를 한 묶음으로 평균한다. 표시용 5축에서는 depth만 대표로 보여준다.

전체 캐시 1,036개 정렬 연주에서 robust note alignment와 후보 수 ≥ 5, 공통 유효 beat ≥ 32,
유효 비율 ≥ 50% 조건을 통과한 {stats['groups']}개 작품/grid의 {stats['performances']}연주를 검색 대상으로 사용했다.
채널 간 scale은 각 작품의 연주 변동을 단위로 만든다. 백분위는 동점에 중간 순위를 준다.

## 이 결과로 말할 수 있는 범위

이 사례는 **“서로 다른 작품에서도 현재 특징상 유사한 연주를 검색할 수 있다”**를 보여준다.
음악적으로 실제로 유사하다는 청취 판정, 추천 성능 향상, 사용자 취향 일치는 아직 측정하지 않았다.
현재 벡터는 연주 전체를 요약하므로 구절 순서를 보존하지 않고, Dynamics/Articulation 폭에는 작품 정보가 남을 수 있다.
페달 전환 횟수는 beat 길이와 CC64 장치 보정에도 영향을 받는다.
작품 공통값·scale에 해당 후보들이 포함되어 있어, 처음 보는 작품이나 독립 연주에 대한 성능 평가도 아니다.

변화 폭을 뺀 7차원 성향 벡터의 최근접 후보는 `{variants[1]['key']}`로 달라진다.
이 비교는 [embedding_variants.csv](embedding_variants.csv)에 남겼다. 차원 수가 다른 두 거리의 숫자는 직접 비교하지 않는다.
검색 표현의 선택이 결과에 영향을 준다는 뜻이며, 예쁜 그림을 위해 이번 14차원 가중치를 다시 맞추지 않았다.
다음 성능 검증에는 청취자가 A에 대해 B와 다른 후보 중 어느 쪽을 더 비슷하게 느끼는지 독립 비교하고,
연주 축을 추가하기 전후의 추천 성능을 비교하는 절차가 필요하다.

## 원자료와 재현

- [profiles.csv](profiles.csv): 14좌표 원값·표시 백분위·후보 수.
- [same_work/beat_features.csv](same_work/beat_features.csv): 전체 구간의 raw/common/relative/standardized·mask/support/scale/status.
- [source_midi/note_events.csv](source_midi/note_events.csv): 선택 구간과 겹치는 원본 음표 이벤트.
- [source_midi/cc64_events.csv](source_midi/cc64_events.csv): 선택한 세 연주의 전체 원본 CC64 이벤트(구간 시작 전 상태 포함).
- [source_midi/note_ratios.csv](source_midi/note_ratios.csv), [beat_checks.csv](source_midi/beat_checks.csv): 기대/실제 음 길이·MIDI 대조 및 beat 검산.
- [retrieval_candidates.csv](retrieval_candidates.csv): A에 대한 다른 작품 전체 후보 거리와 순위.
- [all_query_neighbors.csv](all_query_neighbors.csv): 전체 연주의 다른 작품 최근접과 같은 후보 작품의 대조 결과.
- [stats.json](stats.json): 선정 규칙·데이터 revision·cache/source hash·검증 결과.

저장소 루트에서 실행한다. 기본 데이터 경로는 저장소 밖 `../datasets/ASAP`, `../datasets/nASAP`과 raw cache다.

```bash
.venv/bin/python classicfy-ai/scripts/validate_interpretation_examples.py
```

`--anchor`, `--out`, `--asap-root`, `--nasap-root`, `--cache`로 다른 경로/기준 작품을 지정할 수 있다.
그림과 CSV·JSON·이 문서를 함께 재생성한다. 모든 이전 분석 자료와 원본 feature 구현은 보존한다.
"""
    (out/"README.md").write_text(text,encoding="utf-8")


def main():
    ai=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root",type=Path,default=ai.parent.parent/"datasets/ASAP")
    parser.add_argument("--nasap-root",type=Path,default=ai.parent.parent/"datasets/nASAP")
    parser.add_argument("--cache",type=Path,default=ai.parent.parent/"datasets/feature_normalization_raw.npz")
    parser.add_argument("--out",type=Path,default=ai/"analysis/_interpretation_embedding")
    parser.add_argument("--anchor",default="Bach/Fugue/bwv_848/midi_score.mid")
    parser.add_argument("--min-performances",type=int,default=5)
    parser.add_argument("--min-beats",type=int,default=32)
    parser.add_argument("--min-coverage",type=float,default=.5)
    args=parser.parse_args();args.asap_root=args.asap_root.resolve();args.nasap_root=args.nasap_root.resolve()
    groups,audits,scale_rows,provenance=collect_groups(args)
    rows,vectors=vector_rows(groups);pieces=np.array([r["piece"] for r in rows])
    anchor_indices=np.flatnonzero(pieces==args.anchor)
    if len(anchor_indices)<3:raise ValueError("Anchor must be an eligible piece with at least three performances")
    eligible=[i for i in anchor_indices if vectors[i,8]>=np.median(vectors[anchor_indices,8])]
    a=min(eligible,key=lambda i:(-vectors[i,2],rows[i]["key"]))
    b,c,distances=cross_selection(vectors,pieces,a)
    order=sorted(anchor_indices,key=lambda i:(vectors[i,0],rows[i]["key"]))
    others=[]
    for q in [.25,.75,.5,0,1]:
        i=order[int(round(q*(len(order)-1)))]
        if i!=a and i not in others:others.append(i)
        if len(others)==2:break
    selected=sorted([a,*others],key=lambda i:vectors[i,0])
    anchor_group=next(g for g in groups if rows[a]["key"] in g["keys"])
    args.out.mkdir(parents=True,exist_ok=True);same=args.out/"same_work";cross=args.out/"cross_work";same.mkdir(exist_ok=True);cross.mkdir(exist_ok=True)
    setup_style()
    other_colors=iter([GRAY,ORANGE])
    profile_plot(selected,vectors,pieces,[Path(rows[i]["key"]).stem+(" · 기준 A" if i==a else "") for i in selected],
                 [BLUE if i==a else next(other_colors) for i in selected],same/"01_profile.png","같은 작품에서도 연주 성향은 다르다",
                 f"{piece_label(pieces[a])} | {len(anchor_group['keys'])}연주 중 선택한 3연주 | 작품 내 상대 profile")
    profile_plot([a,b,c],vectors,pieces,[f"{role} · {Path(rows[i]['key']).stem}" for role,i in zip("ABC",[a,b,c])],
                 [BLUE,GREEN,GRAY],cross/"01_profile.png","다른 작품에서도 특징이 비슷한 연주를 찾았다",
                 f"A: {piece_label(pieces[a])} | B·C: {piece_label(pieces[b])}")
    records={r["key"]:r for r in read_cache(args.cache,provenance)};loader=ASAPLoader(args.asap_root,args.nasap_root)
    raw,tempo=make_raw_sequences(anchor_group,records,loader)
    scale_values={r["channel"]:r["value"] for r in scale_rows if r["piece"]==args.anchor and r["grid"]==anchor_group["grid"]}
    curve_range=draw_same_work(anchor_group,selected,a,rows,raw,tempo,scale_values,same,32)
    draw_cross_work([a,b,c],vectors,rows,distances,cross)
    source_range,source_checks=source_examples(anchor_group,selected,rows,loader,records,args.out/"source_midi")
    profile_rows=[]
    for panel,indices in [("same_work",selected),("cross_work",[a,b,c])]:
        for i in indices:
            ranks=percentile_profile(vectors[i],vectors[pieces==pieces[i]])
            for coordinate,value in enumerate(vectors[i]):
                profile_rows.append({"panel":panel,**rows[i],"coordinate":coordinate,"channel":CHANNELS[coordinate//2],
                                     "statistic":"p95_p5" if coordinate%2 else "median_abs" if coordinate==2 else "mean",
                                     "embedding_value":float(value),"within_piece_percentile":float(ranks[coordinate]),
                                     "profile_display":coordinate in DISPLAY_COORDS,"cohort_performances":int((pieces==pieces[i]).sum())})
    write_csv(args.out/"profiles.csv",profile_rows)
    retrieval=[{"rank":rank,**rows[i],"distance":float(distances[i]),"role":"B" if i==b else "C" if i==c else ""}
               for rank,i in enumerate(sorted(np.flatnonzero(pieces!=pieces[a]),key=lambda i:(distances[i],rows[i]["key"])),1)]
    write_csv(args.out/"retrieval_candidates.csv",retrieval)
    all_queries=[]
    for i in range(len(rows)):
        j,k,ds=cross_selection(vectors,pieces,i)
        all_queries.append({"query_key":rows[i]["key"],"query_piece":pieces[i],"nearest_key":rows[j]["key"],
                            "nearest_piece":pieces[j],"nearest_distance":float(ds[j]),"contrast_key":rows[k]["key"],"contrast_distance":float(ds[k])})
    write_csv(args.out/"all_query_neighbors.csv",all_queries)
    write_csv(args.out/"scales.csv",[r for r in scale_rows if r["piece"] in (pieces[a],pieces[b])])
    alternative=[]
    for config in ["all","mean_only"]:
        ds=weighted_distances(vectors[a:a+1],vectors,config)[0];ds[pieces==pieces[a]]=np.inf;j=int(np.argmin(ds))
        alternative.append({"embedding":config,"key":rows[j]["key"],"piece":pieces[j],"distance":float(ds[j])})
    write_csv(args.out/"embedding_variants.csv",alternative)
    stats={"groups":len(groups),"performances":len(rows),"eligible_cross_work_candidates":len(retrieval),
           "embedding":"existing 14-coordinate full embedding, 5 equal musical blocks, no new fitting or weights",
           "selection":{"A":"Within fixed --anchor work: maximum Rubato median abs among performances at or above median pedal depth mean",
                        "same_work":"A plus nearest Tempo mean ranks 25/75%; median/extremes only if duplicates",
                        "B":"minimum full-embedding distance among all candidates outside A piece",
                        "C":"maximum full-embedding distance within B piece, distinct from B"},
           "selected":{"A":rows[a],"B":rows[b],"C":rows[c]},"same_work_keys":[rows[i]["key"] for i in selected],
           "distance_AB":float(distances[b]),"distance_AC":float(distances[c]),"profile_display_coordinates":DISPLAY_COORDS,
           "percentile_policy":"100 * (count strictly lower + 0.5 * count tied) / cohort count; display only, not retrieval input",
           "curve_display_intervals":curve_range,"source_display_intervals":source_range,
           "source_checks":source_checks,
           "source_provenance":provenance,"cache_sha256":hashlib.sha256(args.cache.read_bytes()).hexdigest(),
           "claim_scope":"Extracted features express same-work performance differences; feature-similar performances can be retrieved across works; selected examples, not independent musical similarity or recommendation evaluation",
           "smoothing":None,"clipping":None}
    paths=[Path(__file__),Path(__file__).with_name("validate_embedding.py"),Path(__file__).with_name("validate_tempo.py"),
           Path(__file__).with_name("validate_feature_normalization.py"),
           *[loader.get_sample(rows[i]["key"]).performance_path for i in sorted(set([a,b,c,*selected]))],
           *[loader.get_sample(rows[i]["key"]).note_alignment_path for i in sorted(set([a,b,c,*selected]))]]
    stats["analysis_and_selected_sources_sha256"]=hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
    (args.out/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    write_report(args.out,stats,profile_rows,alternative)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
