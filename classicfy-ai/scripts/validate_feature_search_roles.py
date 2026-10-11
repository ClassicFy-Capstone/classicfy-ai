"""Explain cross-work feature retrieval decisions and held-out-feature concordance.

Uses the existing fixed embedding, never treats its nearest neighbor as a label.
Run: .venv/bin/python classicfy-ai/scripts/validate_feature_search_roles.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from validate_embedding import BLOCKS, collect_groups
from validate_interpretation_examples import piece_label, profile_plot, vector_rows
from validate_tempo import BLUE, GRAY, GREEN, ORANGE, chart, save, setup_style, write_csv

NAMES = list(BLOCKS)
TOL = 1e-12
CONFIGS = {"all": list(range(5)),
           **{f"without_{b}": [j for j in range(5) if j != i] for i,b in enumerate(NAMES)},
           **{f"only_{b}": [i] for i,b in enumerate(NAMES)},
           "without_Tempo_Rubato": [2,3,4], "without_Dynamics_Articulation": [0,1,4]}


def cohort_ids(rows):
    """Same-work comparators and percentile profiles must also share a grid."""
    return np.array([json.dumps([r["piece"],r["grid"]]) for r in rows])


def block_components(vectors, *, means_only=False):
    """Squared distances, one equally weighted musical block per first axis."""
    vectors=np.asarray(vectors)
    if vectors.ndim!=2 or vectors.shape[1]!=14 or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite N x 14 embedding")
    return np.array([np.mean((vectors[:,None,ix]-vectors[None,:,ix])**2,axis=2)
                     for indices in BLOCKS.values() for ix in [indices[::2] if means_only else indices]])


def midrank(distances, index):
    """One-based expected rank in a uniformly ordered tie group."""
    if not np.isfinite(distances[index]):
        raise ValueError("Reference candidate must be eligible")
    equal=np.isclose(distances,distances[index],atol=TOL,rtol=0)
    lower=(distances<distances[index]) & ~equal
    return float(lower.sum()+(equal.sum()+1)/2)


def topk_membership(distances, k=5):
    """Inclusion probabilities for a tie at the top-k boundary; sum equals k."""
    finite=distances[np.isfinite(distances)]
    if not 1<=k<=len(finite):
        raise ValueError("Need at least k eligible candidates")
    cutoff=np.sort(finite)[k-1]
    equal=np.isclose(distances,cutoff,atol=TOL,rtol=0)
    lower=(distances<cutoff)&~equal
    weights=lower.astype(float)
    weights[equal]=(k-lower.sum())/equal.sum()
    return weights


def heldout_score(distances, pieces, neighbor):
    """Pairwise agreement against all other performances of neighbor's work."""
    others=np.flatnonzero((pieces==pieces[neighbor]) & (np.arange(len(pieces))!=neighbor))
    if not len(others):
        raise ValueError("Need a distinct same-work comparator")
    equal=np.isclose(distances[others],distances[neighbor],atol=TOL,rtol=0)
    wins=(distances[others]>distances[neighbor])&~equal
    return float(np.mean(wins+.5*equal)), float(equal.mean()), len(others)


def evaluate(vectors, rows, *, means_only=False):
    pieces=np.array([r["piece"] for r in rows]);count=len(rows)
    cohorts=cohort_ids(rows)
    comp=block_components(vectors,means_only=means_only)
    eligible=pieces[:,None]!=pieces[None,:]
    full=np.where(eligible,np.sqrt(comp.mean(axis=0)),np.inf)
    baseline=np.argmin(full,axis=1)
    mode="means_only" if means_only else "full"
    metric_rows,held_rows=[],[]
    rankings={}
    for config,blocks in CONFIGS.items():
        distances=np.where(eligible,np.sqrt(comp[blocks].mean(axis=0)),np.inf)
        rankings[config]=distances
        for q in range(count):
            b=int(baseline[q]);neighbor=int(np.argmin(distances[q]))
            minimum=distances[q,neighbor]
            nearest_ties=np.isclose(distances[q],minimum,atol=TOL,rtol=0)
            rank=midrank(distances[q],b)
            top5=np.minimum(topk_membership(full[q]),topk_membership(distances[q])).sum()/5
            metric_rows.append({"mode":mode,"config":config,"query_key":rows[q]["key"],"piece":pieces[q],
                "grid":rows[q]["grid"],"candidates":int(eligible[q].sum()),"baseline_key":rows[b]["key"],
                "selected_key":rows[neighbor]["key"],"selected_piece":pieces[neighbor],
                "selected_distance":float(minimum),"nearest_tie_count":int(nearest_ties.sum()),
                "neighbor_changed":float(not nearest_ties[b]),"baseline_rank":rank,
                "baseline_rank_fraction":(rank-1)/(eligible[q].sum()-1),"top5_retained":float(top5)})
    for j,name in enumerate(NAMES):
        distances=rankings[f"without_{name}"]
        for q in range(count):
            b=int(np.argmin(distances[q]));single=np.sqrt(comp[j,q])
            score,ties,n=heldout_score(single,cohorts,b)
            held_rows.append({"mode":mode,"feature":name,"query_key":rows[q]["key"],"piece":pieces[q],
                "grid":rows[q]["grid"],"selected_key":rows[b]["key"],"selected_piece":pieces[b],"selected_grid":rows[b]["grid"],
                "comparison_performances":n,"pairwise_agreement":score,"pairwise_ties":ties,
                "unused_feature_distance":float(single[b]),"selection_distance":float(distances[q,b])})
    return metric_rows,held_rows,comp,full,baseline,rankings


def aggregate(metrics, held, seed, samples):
    """Queries average within work/grid, then works have equal weight."""
    metric_frame=pd.DataFrame(metrics);held_frame=pd.DataFrame(held)
    work_metrics=metric_frame.groupby(["mode","config","piece","grid"],sort=True)[
        ["neighbor_changed","baseline_rank","baseline_rank_fraction","top5_retained"]].mean().reset_index()
    work_held=held_frame.groupby(["mode","feature","piece","grid"],sort=True)[
        ["pairwise_agreement","pairwise_ties"]].mean().reset_index()
    rng=np.random.default_rng(seed)
    works=sorted(set(zip(metric_frame.piece,metric_frame.grid)))
    sampled=rng.integers(0,len(works),(samples,len(works)))
    output=[]
    for frame,category,cols in [(work_metrics,"config",["neighbor_changed","baseline_rank_fraction","top5_retained"]),
                               (work_held,"feature",["pairwise_agreement","pairwise_ties"])]:
        for (mode,name),group in frame.groupby(["mode",category],sort=True):
            group=group.sort_values(["piece","grid"])
            assert list(zip(group.piece,group.grid))==works
            for col in cols:
                values=group[col].to_numpy();boot=values[sampled].mean(axis=1)
                low,high=np.percentile(boot,[2.5,97.5])
                output.append({"mode":mode,"kind":category,"condition":name,"metric":col,
                    "macro_value":float(values.mean()),"ci_low":float(low),"ci_high":float(high),
                    "works":len(works),"queries":int((metric_frame['mode']==mode).sum()/len(CONFIGS)),
                    "bootstrap_samples":samples,"seed":seed})
    return work_metrics,work_held,output


def median_contrast(query, neighbor, pieces, distances):
    """Middle-distance same-work comparator, never the farthest by design."""
    others=np.flatnonzero((pieces==pieces[neighbor])&(np.arange(len(pieces))!=neighbor))
    if not len(others):raise ValueError("No contrast performance")
    order=sorted(others,key=lambda i:(distances[query,i],int(i)))
    return int(order[(len(order)-1)//2])


def representative_query(metrics):
    frame=pd.DataFrame(metrics)
    subset=frame[(frame['mode']=="full")&frame.config.isin([f"without_{n}" for n in NAMES])]
    queries=subset.groupby(["piece","grid","query_key"],sort=True).baseline_rank_fraction.mean().reset_index()
    works=queries.groupby(["piece","grid"],sort=True).baseline_rank_fraction.mean().reset_index()
    target=float(works.baseline_rank_fraction.median())
    work=works.assign(gap=abs(works.baseline_rank_fraction-target)).sort_values(["gap","piece","grid"]).iloc[0]
    pool=queries[(queries.piece==work.piece)&(queries.grid==work.grid)]
    selected=pool.assign(gap=abs(pool.baseline_rank_fraction-work.baseline_rank_fraction)).sort_values(["gap","query_key"]).iloc[0]
    return selected.query_key, {"work_sensitivity":float(work.baseline_rank_fraction),
                                "median_work_sensitivity":target,"query_sensitivity":float(selected.baseline_rank_fraction)}


def summary_item(summary, metric, condition, mode="full"):
    return next(r for r in summary if r["mode"]==mode and r["condition"]==condition and r["metric"]==metric)


def rate_plot(summary, metric, conditions, labels, path, title, subtitle, note, *, baseline=None):
    fig,ax=chart(title,subtitle)
    items=[summary_item(summary,metric,c) for c in conditions]
    values=np.array([100*r["macro_value"] for r in items])
    error=np.array([[100*r["ci_low"] for r in items],[100*r["ci_high"] for r in items]])
    ax.barh(np.arange(len(items)),values,color=BLUE,height=.6)
    ax.errorbar(values,np.arange(len(items)),xerr=[values-error[0],error[1]-values],fmt="none",ecolor="#303830",capsize=4)
    for i,(value,r) in enumerate(zip(values,items)):
        ax.text(min(99,100*r["ci_high"]+1),i,f"{value:.1f}%",va="center",fontsize=11)
    axis_labels={"neighbor_changed":"첫 검색 후보가 달라진 비율 (%)",
                 "top5_retained":"기존 상위 5개 후보의 유지 비율 (%)",
                 "pairwise_agreement":"선택 후보가 같은 작품의 다른 연주보다 가까운 비율 (%)"}
    ax.set(yticks=np.arange(len(items)),yticklabels=labels,xlim=(0,106),xlabel=axis_labels[metric])
    ax.invert_yaxis()
    if baseline is not None:ax.axvline(baseline,color=ORANGE,ls="--",label="무작위 후보 기대값 50%")
    if baseline is not None:ax.legend(loc="lower right",fontsize=10)
    save(fig,path,note+"\n검은 선: 기준 작품/grid를 단위로 재표본한 95% 범위. 평균은 작품마다 같은 가중치.",bottom=.11)


def draw_global(summary,out,works,queries):
    out.mkdir(parents=True,exist_ok=True)
    conditions=[f"without_{n}" for n in NAMES]
    subtitle=f"{works}개 작품/grid · {queries}연주 전체 | 같은 기준 연주·후보·scale 유지"
    rate_plot(summary,"neighbor_changed",conditions,[f"{n} 제외" for n in NAMES],out/"01_neighbor_changes.png",
              "feature 하나를 빼면 선택하는 연주가 바뀔까?",subtitle,
              "높을수록 현재 검색 결정에 영향을 준다. 검색 정확도나 좋은 변화/나쁜 변화의 비율은 아니다.")
    rate_plot(summary,"top5_retained",conditions,[f"{n} 제외" for n in NAMES],out/"02_top5_retained.png",
              "첫 후보가 바뀌어도 상위 후보들은 유지될까?",subtitle,
              "전체 feature의 Top 5 중 몇 %가 해당 feature 제외 후에도 남는가? 100%면 5개 모두 유지된다.")
    rate_plot(summary,"pairwise_agreement",NAMES,NAMES,out/"03_heldout_agreement.png",
              "검색에 안 쓴 feature에서도 후보가 비슷할까?",f"나머지 네 feature로 검색 → 제외한 한 feature로 확인 | {works}작품 평균",
              "선택한 후보가 그 작품의 다른 연주보다 가까운 비율. 동점은 0.5점. 청취 정답이나 추천 정확도가 아니다.",baseline=50)
    pair_conditions=["without_Tempo","without_Rubato","without_Tempo_Rubato","without_Dynamics","without_Articulation","without_Dynamics_Articulation"]
    rate_plot(summary,"neighbor_changed",pair_conditions,["Tempo 제외","Rubato 제외","Tempo + Rubato 제외","Dynamics 제외","Articulation 제외","Dynamics + Articulation 제외"],out/"04_pair_removal.png",
              "연관된 feature를 함께 빼면 어떻게 될까?",subtitle,
              "공유하거나 보완하는 정보를 점검하는 검색 민감도다. 두 개를 빼면 더 많이 바뀌는 것이 자연스러울 수 있다.")
    fig,ax=chart("변화 폭을 포함하는 선택이 결과에 영향을 줄까?","같은 연주·같은 scale·같은 후보 | 검색에 안 쓴 feature의 일치 비율")
    for mode,color,offset,label in [("full",BLUE,-.15,"성향 + 변화 폭 (14좌표)"),("means_only",GREEN,.15,"성향만 (7좌표)")]:
        items=[summary_item(summary,"pairwise_agreement",n,mode) for n in NAMES]
        values=np.array([100*r["macro_value"] for r in items]);low=np.array([100*r["ci_low"] for r in items]);high=np.array([100*r["ci_high"] for r in items])
        ax.errorbar(values,np.arange(5)+offset,xerr=[values-low,high-values],color=color,fmt="o",capsize=3,label=label)
    ax.axvline(50,color=ORANGE,ls="--")
    ax.set(yticks=np.arange(5),yticklabels=NAMES,xlabel="사용하지 않은 feature에서 같은 작품 대조보다 가까운 비율 (%)",xlim=(25,90))
    ax.invert_yaxis();ax.legend(loc="lower right",fontsize=10)
    save(fig,out/"05_summary_sensitivity.png","두 표현의 검색·평가 좌표가 함께 달라지는 민감도 비교다. 음악적 우열을 결정하지 않는다.\n가로 선: 기준 작품/grid 재표본 95% 범위. 세로 점선: 무작위 후보 기대값 50%.",bottom=.11)


def draw_advantage(q,b,c,comp,rows,out,title):
    delta=comp[:,q,c]-comp[:,q,b]
    fig,ax=chart(title,f"기준 {Path(rows[q]['key']).stem} | B: {Path(rows[b]['key']).stem} · C: {Path(rows[c]['key']).stem}")
    ax.barh(np.arange(5),delta,color=[GREEN if d>=0 else ORANGE for d in delta])
    ax.axvline(0,color="#303830",lw=1)
    for i,value in enumerate(delta):
        ax.text(value+.03 if value>=0 else value-.03,i,f"{value:+.2f}",va="center",ha="left" if value>=0 else "right",fontsize=11)
    extent=max(float(np.max(np.abs(delta))),.1)
    ax.set(yticks=np.arange(5),yticklabels=NAMES,xlabel="C와의 제곱 차이 − B와의 제곱 차이 (오른쪽이면 B 지지)",xlim=(-extent*1.35,extent*1.35))
    ax.invert_yaxis()
    save(fig,out,"양수는 B가 가깝고 음수는 C가 가깝다. 다섯 값의 평균이 전체 거리²의 C−B 차이와 정확히 같다.")
    return [{"query_key":rows[q]["key"],"B":rows[b]["key"],"C":rows[c]["key"],"feature":name,
             "AB_squared_difference":float(comp[j,q,b]),"AC_squared_difference":float(comp[j,q,c]),
             "advantage_for_B":float(delta[j])} for j,name in enumerate(NAMES)]


def draw_examples(vectors,rows,metrics,comp,full,baseline,anchor_key,out):
    out.mkdir(parents=True,exist_ok=True)
    keys=[r["key"] for r in rows];pieces=np.array([r["piece"] for r in rows]);cohorts=cohort_ids(rows);frame=pd.DataFrame(metrics)
    q=keys.index(anchor_key);b=int(baseline[q]);c=median_contrast(q,b,cohorts,full)
    anchor_rows=frame[(frame['mode']=="full")&(frame.query_key==anchor_key)].set_index("config")
    configs=["all",*[f"without_{n}" for n in NAMES]]
    values=[anchor_rows.loc[k,"baseline_rank"] for k in configs]
    fig,ax=chart("feature를 빼면 기존 후보 B의 순위가 내려갈까?",f"기준 A: {Path(anchor_key).stem} | 기존 B: {Path(rows[b]['key']).stem} | 다른 작품 {np.isfinite(full[q]).sum()}후보")
    ax.barh(np.arange(6),values,color=[GREEN,*[BLUE]*5])
    for j,value in enumerate(values):ax.text(value+.4,j,f"{value:g}위",va="center")
    ax.set(yticks=np.arange(6),yticklabels=["다섯 feature 전체",*[f"{n} 제외" for n in NAMES]],xlabel="기존 B의 검색 순위 (작을수록 앞에 있음)",xlim=(0,max(values)*1.25))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6));ax.invert_yaxis()
    save(fig,out/"01_anchor_rank.png","B를 정답으로 간주한 정확도는 아니다. 이전 사례의 후보를 고정해 어떤 feature가 순위에 영향을 주는지 확인한다.")
    advantages=draw_advantage(q,b,c,comp,rows,out/"02_anchor_advantage.png","각 feature는 B와 C 중 어느 연주를 지지할까?")
    typical_key,selection=representative_query(metrics);t=keys.index(typical_key);tb=int(baseline[t]);tc=median_contrast(t,tb,cohorts,full)
    profile_plot([t,tb,tc],vectors,cohorts,[f"{role} · {Path(rows[i]['key']).stem}" for role,i in zip("ABC",[t,tb,tc])],
                 [BLUE,GREEN,GRAY],out/"03_typical_profile.png","중간 수준 사례의 연주 성향을 비교한다",
                 f"A: {piece_label(pieces[t])} | B·C: {piece_label(pieces[tb])}")
    target=np.flatnonzero(cohorts==cohorts[tb]);order=sorted(target,key=lambda i:(full[t,i],keys[i]))
    fig,ax=chart("같은 후보 작품에서도 연주별 검색 거리가 다르다",f"기준 A: {Path(typical_key).stem} | 후보 작품: {piece_label(pieces[tb])}",height=max(5.8,2.5+.3*len(order)))
    vals=full[t,order]
    ax.barh(np.arange(len(order)),vals,color=[GREEN if i==tb else ORANGE if i==tc else GRAY for i in order])
    for j,i in enumerate(order):ax.text(full[t,i]+.015,j,f"{full[t,i]:.3f}",va="center",fontsize=10)
    ax.set(yticks=np.arange(len(order)),yticklabels=[("B · " if i==tb else "C · " if i==tc else "")+Path(keys[i]).stem for i in order],xlabel="A와의 전체 임베딩 거리 (작을수록 가까움)",xlim=(0,max(vals)*1.2))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5));ax.invert_yaxis()
    save(fig,out/"04_typical_candidates.png","B는 다른 작품 최근접 후보, C는 B 작품의 나머지 연주 중 중간 거리 후보다. C를 가장 먼 연주로 고르지 않았다.")
    advantages+=draw_advantage(t,tb,tc,comp,rows,out/"05_typical_advantage.png","중간 수준 사례에서 각 feature의 역할은 어떨까?")
    return {"anchor":{"A":rows[q],"B":rows[b],"C":rows[c]},
            "typical":{"A":rows[t],"B":rows[tb],"C":rows[tc],"selection":selection}},advantages


def draw_heldout_examples(rows,held,comp,out):
    out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(held);keys=[r["key"] for r in rows];pieces=np.array([r["piece"] for r in rows]);cohorts=cohort_ids(rows);selections=[]
    for j,name in enumerate(NAMES):
        f=frame[(frame['mode']=="full")&(frame.feature==name)]
        works=f.groupby(["piece","grid"],sort=True).pairwise_agreement.mean().reset_index()
        average=float(works.pairwise_agreement.mean())
        work=works.assign(gap=abs(works.pairwise_agreement-average)).sort_values(["gap","piece","grid"]).iloc[0]
        pool=f[(f.piece==work.piece)&(f.grid==work.grid)]
        representative=pool.assign(gap=abs(pool.pairwise_agreement-work.pairwise_agreement)).sort_values(["gap","query_key"]).iloc[0]
        q=keys.index(representative.query_key);b=keys.index(representative.selected_key)
        values=np.sqrt(comp[j,q]);target=np.flatnonzero((cohorts==cohorts[b])&(np.arange(len(pieces))!=b));order=sorted(target,key=lambda i:(values[i],keys[i]))
        fig,ax=chart(f"검색에 안 쓴 {name}에서도 후보가 가까울까?",f"다른 네 feature로 선택: {Path(keys[b]).stem} | 후보 작품: {piece_label(pieces[b])}")
        # Small deterministic offsets only separate overlapping display markers.
        offsets=(np.arange(len(order))%5-2)*.025
        ax.scatter(values[order],offsets,color=GRAY,s=65,label="같은 후보 작품의 다른 연주들")
        ax.scatter([values[b]],[1],color=GREEN,s=180,label=f"네 feature로 선택 · {Path(keys[b]).stem}")
        ax.text(values[b]+.025,1,f"{values[b]:.3f}",va="center",color=GREEN,fontweight="bold")
        median=float(np.median(values[order]));ax.axvline(median,color=GRAY,ls=":",lw=1.3,label="다른 연주들의 거리 중앙값")
        ax.set(yticks=[0,1],yticklabels=[f"같은 작품의 다른 {len(order)}연주",f"선택한 B · {Path(keys[b]).stem}"],ylim=(-.45,1.5),xlabel=f"기준 A와의 {name} 단독 거리 (왼쪽일수록 유사)",xlim=(0,max(float(max(values[order]))*1.2,float(values[b])*1.2,.1)))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5));ax.legend(loc="upper right",fontsize=9)
        save(fig,out/f"{j+1:02d}_{name}.png",f"A={Path(keys[q]).stem}. 같은 작품 대조보다 가까운 비율 {100*representative.pairwise_agreement:.1f}%.\n회색 점 하나가 연주 하나다. 이 feature는 B 선택에 쓰지 않았다. 점의 작은 세로 이동은 겹침 방지용이다.",bottom=.11)
        selections.append({"feature":name,"A":rows[q],"B":rows[b],"query_agreement":float(representative.pairwise_agreement),
                           "work_agreement":float(work.pairwise_agreement),"macro_agreement":average,
                           "selection":"Work mean closest to macro mean; query agreement closest to selected work mean; lexical ties"})
    return selections


def write_report(out,stats,summary,metrics):
    table=[]
    for name in NAMES:
        changed=summary_item(summary,"neighbor_changed",f"without_{name}")
        retained=summary_item(summary,"top5_retained",f"without_{name}")
        held=summary_item(summary,"pairwise_agreement",name)
        table.append(f"| {name} | {100*changed['macro_value']:.1f}% | {100*retained['macro_value']:.1f}% | "
                     f"{100*held['macro_value']:.1f}% ({100*held['ci_low']:.1f}~{100*held['ci_high']:.1f}) |")
    names={role:Path(stats['examples']['anchor'][role]['key']).stem for role in 'ABC'}
    anchor=stats['examples']['anchor'];typical=stats['examples']['typical']
    held_links=[]
    for j,e in enumerate(stats['heldout_examples'],1):
        held_links.append(f"| [{e['feature']}](heldout/{j:02d}_{e['feature']}.png) | `{Path(e['A']['key']).stem}` → `{Path(e['B']['key']).stem}` | {100*e['query_agreement']:.1f}% |")
    rank_table=[]
    frame=pd.DataFrame(metrics)
    for condition,label in [('all','다섯 feature 전체'),*[(f'without_{n}',f'{n} 제외') for n in NAMES]]:
        r=frame[(frame['mode']=='full')&(frame.query_key==anchor['A']['key'])&(frame.config==condition)].iloc[0]
        rank_table.append(f"| {label} | {r.baseline_rank:g}위 | `{r.selected_key}` |")
    text=f"""# 다른 작품 유사 연주 검색에서 다섯 feature의 역할

## 이번 분석이 답하는 질문

**다른 작품의 후보를 찾을 때, 각 feature가 검색 결정을 바꾸고 연주별 차이를 설명하는가?**
기존 14차원 수치 요약 임베딩·5묶음 동일 가중 거리·작품별 정규화를 고정했다.
학습된 신경망이나 청취 정답은 사용하지 않았다. 낮은 거리를 다시 검색 정확도라고 부르지 않는다.
이 보고서는 **검색 기여/민감도**, **feature 간 일치 정도**를 분리해서 보여준다.

## 청중에게 보여줄 순서

1. [한 feature를 빼면 기존 B의 순위가 어떻게 바뀌는가?](examples/01_anchor_rank.png)
   연주 한 개의 구체적인 예시로 질문을 소개한다. B의 순위가 움직이면 해당 feature가 검색 결정에 참여한다.
2. [B와 C를 구분하는 실제 feature는 무엇인가?](examples/02_anchor_advantage.png)
   오른쪽 막대는 B를 지지, 왼쪽은 C를 지지한다. 모든 feature가 B를 지지해야 할 필요는 없다.
3. [전체 작품에서도 검색 후보가 바뀌는가?](global/01_neighbor_changes.png)
   특정 연주에만 해당하는 현상인지, 55개 작품/grid 전체에서도 나타나는지 확인한다.
4. [검색에 쓰지 않은 feature도 닮았는가?](global/03_heldout_agreement.png)
   후보를 고른 기준과 평가 기준을 나눠, 같은 feature로 고르고 같은 feature로 칭찬하는 순환을 피한다.

위 네 장이 발표 핵심이다. 모든 PNG는 파일당 단일 그래프이고 나머지는 해석을 보충한다.

## 결과부터 보기

아래는 **작품별 평균을 낸 뒤 작품/grid에 같은 가중치**를 준 결과다.
괄호는 기준 작품/grid 재표본 95% 구간이며, 전체 후보 corpus를 고정한 기술통계다.

| feature | 제외하면 첫 후보가 바뀜 | 기존 Top 5가 유지됨 | 다른 네 축으로 고른 후보가 제외한 축에서도 가까움 |
|---|---:|---:|---:|
{chr(10).join(table)}

첫 번째 열이 크면 **검색 결정에 많이 참여한다**. 성능이 좋아졌다는 뜻은 아니다.
두 번째 열이 높으면 첫 후보가 바뀌어도 후보 목록 상당 부분은 유지된다는 뜻이다.
세 번째 열은 선택된 연주가 **그 후보 작품의 다른 연주**보다 사용하지 않은 feature에서도 가까운 비율이다.
무작위로 해당 작품의 연주를 고르면 동점을 포함한 기대값은 50%다.
이 실험에서 Tempo·Rubato의 일치 경향이 상대적으로 크고, 나머지 축은 50%에 더 가깝다.
Dynamics·Pedaling의 재표본 구간은 50%를 포함한다. Articulation은 54.0%로 약한 경향이다.
여러 축을 함께 쓰는 검색은 가능하지만, **모든 축이 하나의 일관된 해석 성향을 잡았다고 결론낼 수는 없다.**

## 분석 기준을 이렇게 정한 이유

### 데이터와 기준 연주

- 기존 원본 캐시 1,036개 정렬 연주에서 robust note alignment·동일 score grid·후보 수 ≥ 5,
  7채널 공통 유효 beat ≥ 32·비율 ≥ 50%를 통과한 **{stats['works']}개 작품/grid, {stats['queries']}개 연주**를 쓴다.
- 일부 좋은 예시만 기준으로 삼지 않고, **모든 연주를 한 번씩 기준 A**로 쓴다.
- A와 같은 작품은 검색 후보에서 모두 제외한다. 다른 반복/grid도 query 작품의 후보로 되돌려 넣지 않는다.
- B의 같은 작품 대조와 표시용 백분위는 B와 동일한 score grid의 연주만 사용한다.
- 공통값·scale·유효 beat·후보 목록은 feature 제거 전후 동일하다. 뺀 뒤 다시 정규화하지 않는다.
- 후보 수가 많은 작품이 결과를 지배하지 않도록 query 평균 → 기준 작품/grid 평균 → 작품 동일 가중으로 집계한다.
  평균 순위 비교에는 후보 수 차이를 보정한 `(B순위−1)/(후보수−1)`도 저장한다.
- 기준 작품/grid를 단위로 {stats['bootstrap']['samples']}회 재표본한다. seed는 {stats['bootstrap']['seed']}다.
  query들은 독립 표본으로 간주하지 않는다. 후보 작품이 중복되는 의존성은 남아 있어 모집단 성능 CI로 해석하지 않는다.

### 검색 표현과 정규화

Tempo의 기존 individual, Rubato의 기존 relative를 유지한다. D/A/P는 위치별 common median을 제거한 residual이다.
각 작품/grid에서 T/R/D/A는 MAD, Pedaling depth/on/changes는 각각 SD로 나눈다.
MAD는 `1.4826 × median(|r−median(r)|)`, 0이면 IQR/1.349 → SD → unit 1이다.
residual 재중앙화·clipping·평활화는 없다. mask와 지원 수는 기존 규칙을 따른다.

7채널 각각의 성향(평균, Rubato만 절댓값 중앙값)과 `p95−p5` 폭을 요약해 14좌표를 만든다.
Tempo/Rubato/Dynamics/Articulation은 각 2좌표, Pedaling은 6좌표지만 각 묶음 내부를 평균한다.
**거리² = 다섯 feature 묶음의 평균 제곱 차이의 평균**이므로 Pedaling 좌표 수가 더 많다고 가중치가 커지지 않는다.
원래 단위가 아니라 작품 안에서의 상대적 편차를 비교한다. 표시용 profile 백분위는 거리 입력이 아니다.

### 1. 제거 실험: 검색에 참여하는가?

전체 feature에서의 최근접 B를 고정하고, 하나씩 뺀 검색에서 B의 순위·첫 후보 변경·Top 5 유지율을 계산한다.
feature 단독 검색도 CSV에 저장했다. Tempo+Rubato, Dynamics+Articulation 동시 제거도 확인한다.
Pedaling 제거는 세 하위 지표/6좌표 전체를 뺀다.

| 그림 | 청중에게 설명할 문장 |
|---|---|
| [첫 후보 변경](global/01_neighbor_changes.png) | “이 특징을 없애면 실제로 다른 연주를 고르는 경우가 이만큼 있습니다.” |
| [Top 5 유지](global/02_top5_retained.png) | “첫 후보의 변화와 후보 목록 전체의 변화는 구분해서 봤습니다.” |
| [두 feature 동시 제거](global/04_pair_removal.png) | “서로 연관된 특징을 함께 없앴을 때 검색이 어떻게 달라지는지도 확인했습니다.” |

변경률을 feature 중요도/정확도 순위로 바꾸지 않는다. 제거하면 거리가 바뀌는 것은 구성상 자연스럽다.
특히 **B는 정답 레이블이 아니다.** B의 순위 하락은 B를 선택한 이유를 보여줄 뿐이다.

### 2. 후보 간 차이 설명: 어떤 축이 B를 지지하는가?

이전 사례의 기준 A를 그대로 이어 쓴다.
A=`{anchor['A']['key']}`, B=`{anchor['B']['key']}`다.
새 대조 C=`{anchor['C']['key']}`는 B와 같은 작품의 나머지 연주를 A와의 전체 거리로 정렬한 **중간 거리** 후보다.
짝수가 되면 아래쪽 중간 순위를 쓴다. 이전 보고서의 최원거리 C와 목적이 다르며 원래 보고서는 보존했다.

| 조건 | 기존 B 순위 | 해당 검색의 첫 후보 |
|---|---:|---|
{chr(10).join(rank_table)}

각 feature f의 막대는 `제곱차이(A,C,f) − 제곱차이(A,B,f)`다.
양수면 B가 더 가깝고, 음수면 C가 더 가깝다. 다섯 막대 평균은 전체 거리²의 C−B 차이와 정확히 같다.
한 후보를 지지하는 축과 전체 후보에서 다른 경쟁자를 걸러내는 축은 다를 수 있다.
이전 기준 A의 중간 거리 대조에서는 Articulation의 B 지지가 가장 크고, 아래 중간 수준 사례에서는 Pedaling이 가장 크다.
어느 feature가 역할을 하는지는 기준 연주와 비교 상대에 따라 달라진다.

새로운 [중간 수준 사례 profile](examples/03_typical_profile.png), [같은 후보 작품의 거리](examples/04_typical_candidates.png),
[feature별 지지 방향](examples/05_typical_advantage.png)도 함께 제시한다.
이 사례는 기준 작품별 “하나 제거 후 B의 상대순위 변화”가 작품 중앙값에 가장 가까운 작품을 먼저 고르고,
그 안에서 해당 작품 평균에 가장 가까운 query를 고른다. 동점은 파일명 순이다.
A=`{typical['A']['key']}`, B=`{typical['B']['key']}`, C=`{typical['C']['key']}`다.
가장 큰 변화나 가장 예쁜 profile을 찾아 선택하지 않았다. 이 선정 자체도 현재 corpus를 이용한 기술적 예시다.

### 3. 사용하지 않은 feature 검사: 다른 축과 일치하는가?

예를 들어 Tempo를 검사한다면 **Rubato·Dynamics·Articulation·Pedaling만으로** 다른 작품의 B를 찾는다.
그다음 Tempo 단독 거리로 A–B를 계산하고, B 작품의 **다른 모든 연주**와 비교한다.
B가 더 가까우면 1, 같으면 0.5, 더 멀면 0점이다. 대조마다 평균 → query 평균 → 작품 동일 가중 순으로 계산한다.
가장 먼 C 하나와만 비교하지 않는다. 동점은 성공으로 올려 세지 않는다.
50%는 정답을 맞힐 확률이 아니라 **같은 작품의 두 후보 중 어느 쪽이 더 가까운가**를 무작위 선택했을 때의 기준이다.

검색에서 제외한 feature가 평가에 사용되므로, 그 feature를 잘 맞춰 놓고 다시 검사하는 문제는 줄어든다.
다만 **feature 간 통계적 연관 검사**다. Tempo와 Rubato는 같은 타이밍 원자료를 공유하고, 네 축이 다섯 번째 축을 대신할 수도 있다.
높으면 feature 필수성이 증명되는 것이 아니고, 낮으면 독립적인 선호 정보일 수도 있어 불필요하다고 단정하지 않는다.
공통값과 scale에는 corpus 후보가 포함되어 있다. 독립 작품/새 연주에 대한 일반화 실험도 아니다.

아래 개별 그림의 초록 연주는 **다른 네 축으로 선택된 연주**다.
회색 점 하나는 같은 후보 작품의 다른 연주 하나다. 왼쪽으로 갈수록 해당 축에서 A와 가깝다.
세로 점선은 다른 연주들의 거리 중앙값이다. 23개 후보가 있는 경우에도 이름을 모두 나열하지 않고 분포를 보여준다.
이 축에서 항상 1위일 필요는 없다. 기준 작품 평균이 전체 평균에 가까운 작품을 고르고,
그 작품 평균에 가까운 query를 고른 사례라 실패나 차이도 그대로 보인다.

| 개별 그림 | 기준 A → 네 축으로 선택한 B | 다른 같은 작품 연주보다 가까운 비율 |
|---|---|---:|
{chr(10).join(held_links)}

[14좌표와 7좌표의 민감도 비교](global/05_summary_sensitivity.png)는 변화 폭 포함 여부도 검사한다.
7좌표는 각 채널의 성향만 쓴다. 두 표현에서 선택과 검사 좌표가 함께 바뀌므로 수치로 표현의 음악적 우열을 결정하지 않는다.

## 무엇이 확인됐고 무엇을 더 확인해야 하나?

**각 feature는 실제 검색 결정과 후보 간 구분에 참여한다.** 연주 차이를 여러 축으로 설명할 수 있다.
그러나 “다섯 개가 모두 필요하다”, “음악적으로 유사한 연주를 정확하게 찾는다”, “추천 성능이 향상됐다”는 결론은 아직 없다.
Dynamics/Articulation 폭에는 작품 정보가 남을 수 있고, MIDI velocity·CC64는 장치 보정, note-off는 실제 울림과 다를 수 있다.
독립 청취자가 유사하다고 고른 후보 또는 사용자의 선호를 평가 기준으로 삼아 최종 제거 실험을 해야 한다.
특징이 달라지면 선택이 달라지는 것과, 선택이 사용자에게 도움이 되는 것을 구분한다.

## 재현과 원수치

저장소 루트에서 실행한다. 기본 데이터 경로는 저장소 밖 `../datasets/ASAP`, `../datasets/nASAP`, raw cache다.

```bash
.venv/bin/python classicfy-ai/scripts/validate_feature_search_roles.py
```

`--anchor-key`, `--seed`, `--bootstrap-samples`, `--out`과 데이터 경로 옵션을 제공한다.
PNG·CSV·JSON·이 문서를 같이 재생성한다. 계산에 필요한 후보/scale도 저장한다.

| 파일 | 내용 |
|---|---|
| [embeddings.csv](embeddings.csv) | 모든 연주 14좌표. |
| [search_changes.csv](search_changes.csv) | 모든 query/구성의 후보·B 순위·동점·Top 5 유지. |
| [heldout_agreement.csv](heldout_agreement.csv) | 검색에 쓰지 않은 축의 query별 비교 결과. |
| [work_search_changes.csv](work_search_changes.csv), [work_heldout_agreement.csv](work_heldout_agreement.csv) | 작품별 집계. |
| [summary.csv](summary.csv) | 작품 동일 가중 평균·재표본 구간. |
| [examples.csv](examples.csv) | B·C 차이의 feature별 정확한 분해. |
| [scales.csv](scales.csv), [cohort_audit.csv](cohort_audit.csv) | 정규화 scale과 작품 포함/제외 조건. |
| [stats.json](stats.json) | 데이터 hash·revision·선정 조건·해석 범위. |

MIDI 원자료와 같은 작품 곡선은 기존 [연주 차이/검색 사례 보고서](../_interpretation_embedding/README.md)에 있다.
이전 그림이나 원본 feature/정규화 정책은 수정하지 않았다.
"""
    (out/"README.md").write_text(text,encoding="utf-8")


def main():
    ai=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root",type=Path,default=ai.parent.parent/"datasets/ASAP")
    parser.add_argument("--nasap-root",type=Path,default=ai.parent.parent/"datasets/nASAP")
    parser.add_argument("--cache",type=Path,default=ai.parent.parent/"datasets/feature_normalization_raw.npz")
    parser.add_argument("--out",type=Path,default=ai/"analysis/_feature_search_roles")
    parser.add_argument("--anchor-key",default="Bach/Fugue/bwv_848/Lou01M.mid")
    parser.add_argument("--min-performances",type=int,default=5)
    parser.add_argument("--min-beats",type=int,default=32)
    parser.add_argument("--min-coverage",type=float,default=.5)
    parser.add_argument("--seed",type=int,default=20261005)
    parser.add_argument("--bootstrap-samples",type=int,default=2000)
    args=parser.parse_args();args.asap_root=args.asap_root.resolve();args.nasap_root=args.nasap_root.resolve()
    if args.bootstrap_samples<1:raise ValueError("Need bootstrap samples")
    groups,audits,scales,provenance=collect_groups(args);rows,vectors=vector_rows(groups)
    if args.anchor_key not in [r["key"] for r in rows]:raise ValueError("Anchor key is not eligible")
    primary=evaluate(vectors,rows);alternative=evaluate(vectors,rows,means_only=True)
    metrics=primary[0]+alternative[0];held=primary[1]+alternative[1]
    work_metrics,work_held,summary=aggregate(metrics,held,args.seed,args.bootstrap_samples)
    args.out.mkdir(parents=True,exist_ok=True);setup_style()
    draw_global(summary,args.out/"global",len(groups),len(rows))
    examples,advantages=draw_examples(vectors,rows,metrics,*primary[2:5],args.anchor_key,args.out/"examples")
    held_examples=draw_heldout_examples(rows,held,primary[2],args.out/"heldout")
    for filename,values in [("search_changes.csv",metrics),("heldout_agreement.csv",held),("summary.csv",summary),
                            ("examples.csv",advantages),("scales.csv",scales),("cohort_audit.csv",audits)]:
        write_csv(args.out/filename,values)
    work_metrics.to_csv(args.out/"work_search_changes.csv",index=False,encoding="utf-8-sig")
    work_held.to_csv(args.out/"work_heldout_agreement.csv",index=False,encoding="utf-8-sig")
    write_csv(args.out/"embeddings.csv",[{**r,**{f"coordinate_{j}":float(value) for j,value in enumerate(v)}} for r,v in zip(rows,vectors)])
    stats={"works":len(groups),"queries":len(rows),"configs":CONFIGS,"coordinates":14,"alternative_coordinates":7,
           "source_provenance":provenance,"raw_cache_sha256":hashlib.sha256(args.cache.read_bytes()).hexdigest(),
           "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "aggregation":"Mean per query, mean queries within source work/grid, equal work/grid macro; candidate gallery fixed",
           "bootstrap":{"unit":"source work/grid","samples":args.bootstrap_samples,"seed":args.seed,
                        "interval":"percentile 2.5-97.5, descriptive conditional on this fixed corpus; overlapping target works remain dependent"},
           "ranking_ties":"Ranks averaged; fractional top-k cutoff; neighbor change requires baseline absent from tied nearest set; heldout tie=0.5; displayed candidate is lexical first",
           "normalization":"Existing residuals; per-work MAD T/R/D/A and separate SD pedal channels; no refitting on removal",
           "examples":examples,"heldout_examples":held_examples,
           "heldout_baseline":"Uniform random performance within selected target work has expected pairwise score 0.5, including ties",
           "heldout_scope":"Cross-feature concordance; not feature necessity, listener agreement, recommendation accuracy, or out-of-sample evaluation",
           "search_scope":"Sensitivity and explanations of this defined feature distance, not nearest neighbor correctness"}
    stats["candidate_count_range"]=[min(r["candidates"] for r in primary[0]),max(r["candidates"] for r in primary[0])]
    stats["nearest_tie_query_counts"]={mode:{config:sum(r["nearest_tie_count"]>1 for r in ms if r["config"]==config)
                                           for config in CONFIGS} for mode,ms in [("full",primary[0]),("means_only",alternative[0])]}
    dependencies=[Path(__file__).with_name(name) for name in ["validate_embedding.py","validate_interpretation_examples.py",
                   "validate_tempo.py","validate_feature_normalization.py"]]
    dependencies += [ai/"src/features"/name for name in ["normalization.py","common_pattern.py","tempo.py","rubato.py"]]
    stats["dependency_sha256"]={str(p.relative_to(ai)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    (args.out/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    write_report(args.out,stats,summary,metrics)
    print(json.dumps({"works":len(groups),"queries":len(rows),"examples":examples,
                      "heldout_summary":[r for r in summary if r["mode"]=="full" and r["metric"]=="pairwise_agreement"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
