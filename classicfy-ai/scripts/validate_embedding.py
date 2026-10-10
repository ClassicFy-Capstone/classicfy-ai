"""Same-piece retrieval under beat dropout; no learned model or preference labels.

Run: .venv/bin/python classicfy-ai/scripts/validate_embedding.py
Outputs readable single-chart figures and CSV/JSON in analysis/_embedding_validation.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_tempo import BLUE, GRAY, ORANGE, GREEN, chart, plt, save, setup_style, write_csv
from validate_feature_normalization import read_cache, source_signature
from features import (BeatSequence, PieceFeatureInput, TempoInput, extract_piece_tempo_features,
                      extract_piece_rubato_features, separate_piece_feature, standardize_piece_feature,
                      fit_residual_scale, standardize_sequence)
from preprocessing import ASAPLoader

CHANNELS = ("tempo", "rubato", "dynamics", "articulation", "pedal_depth", "pedal_down_ratio", "pedal_changes")
BLOCKS = {"Tempo": [0, 1], "Rubato": [2, 3], "Dynamics": [4, 5],
          "Articulation": [6, 7], "Pedaling": [8, 9, 10, 11, 12, 13]}
CONFIGS = {"all": (list(BLOCKS), "full"),
           **{f"only_{b}": ([b], "full") for b in BLOCKS},
           **{f"without_{b}": ([k for k in BLOCKS if k != b], "full") for b in BLOCKS},
           "mean_only": (list(BLOCKS), "first"), "without_DA_range": (list(BLOCKS), "no_DA_range")}
CONDITIONS = [("clean", 0.), ("random", .05), ("random", .10), ("random", .20),
              ("random", .40), ("contiguous", .20)]
LABELS = {"all": "다섯 feature 전체", "mean_only": "성향만 (Rubato는 편차 크기)",
          "without_DA_range": "Dynamics·Articulation 변화 폭 제외",
          **{f"only_{b}": f"{b} 단독" for b in BLOCKS},
          **{f"without_{b}": f"{b} 제외" for b in BLOCKS}}


def summarize(values):
    """14 coordinates: signed mean (Rubato median abs) and p95-p5 per channel."""
    if values.ndim != 2 or values.shape[0] != 7 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected seven finite channels with at least two shared beats")
    first = values.mean(axis=1)
    first[1] = np.median(np.abs(values[1]))
    width = np.diff(np.percentile(values, [5, 95], axis=1), axis=0)[0]
    return np.column_stack([first, width]).ravel()


def weighted_distances(queries, gallery, config):
    """Each musical block contributes its mean squared coordinate distance."""
    blocks, mode = CONFIGS[config]
    distance = np.zeros((len(queries), len(gallery)))
    for b in blocks:
        indices = BLOCKS[b]
        if mode == "first":
            indices = indices[::2]
        elif mode == "no_DA_range" and b in ("Dynamics", "Articulation"):
            indices = indices[:1]
        distance += np.mean((queries[:, None, indices] - gallery[None, :, indices]) ** 2, axis=2)
    return np.sqrt(distance / len(blocks))


def retrieval_metrics(distances, targets):
    """Ties receive expected scores under uniform random tie ordering."""
    own = distances[np.arange(len(targets)), targets]
    equal = np.isclose(distances, own[:, None], rtol=0, atol=1e-12)
    less = ((distances < own[:, None]) & ~equal).sum(axis=1)
    ties = equal.sum(axis=1)
    top1 = np.where(less == 0, 1 / ties, 0.)
    mrr = np.array([np.mean(1 / np.arange(l + 1, l + t + 1)) for l, t in zip(less, ties)])
    other = distances.copy()
    other[np.arange(len(targets)), targets] = np.inf
    nearest = other.min(axis=1)
    margin = np.divide(nearest - own, nearest + own, out=np.zeros(len(own)), where=(nearest + own) > 1e-12)
    return dict(top1=top1, mrr=mrr, self_distance=own, nearest_other_distance=nearest,
                margin=margin, ties=ties, midrank=less + (ties + 1) / 2)


def retained_indices(count, fraction, pattern, rng):
    lost = int(round(count * fraction))
    if lost == 0:
        return np.arange(count)
    if count - lost < 2:
        raise ValueError("Too few retained beats")
    if pattern == "random":
        removed = rng.choice(count, lost, replace=False)
    elif pattern == "contiguous":
        start = rng.integers(0, count - lost + 1)
        removed = np.arange(start, start + lost)
    else:
        raise ValueError("Unknown dropout pattern")
    return np.setdiff1d(np.arange(count), removed)


def collect_groups(args, *, provenance_source_files=None):
    provenance = source_signature(args.asap_root, args.nasap_root, source_files=provenance_source_files)
    records = read_cache(args.cache, provenance)
    loader = ASAPLoader(args.asap_root, args.nasap_root)
    raw_groups = defaultdict(list)
    for r in records:
        grid = hashlib.sha256(r["score_beats"].tobytes() + json.dumps(r["score_types"]).encode()).hexdigest()[:12]
        raw_groups[r["piece"], grid].append(r)
    groups, audits, scales = [], [], []
    for (piece, grid), all_records in sorted(raw_groups.items()):
        rs = sorted((r for r in all_records if r["robust"] is True and r["match_available"]), key=lambda r: r["key"])
        audit = {"piece": piece, "grid": grid, "aligned_candidates": len(all_records),
                 "robust_candidates": len(rs), "accepted": False, "reason": "", "shared_beats": 0,
                 "total_intervals": len(all_records[0]["score_beats"]) - 1, "shared_fraction": 0.}
        if len(rs) < args.min_performances:
            audit["reason"] = "too_few_robust_performances"
            audits.append(audit)
            continue
        samples = [loader.get_sample(r["key"]) for r in rs]
        for r, sample in zip(rs, samples):
            np.testing.assert_array_equal(r["score_beats"], sample.score_beats)
            np.testing.assert_array_equal(r["beat_seconds"], np.diff(sample.performance_beats))
        tempo = extract_piece_tempo_features([TempoInput.from_asap_sample(s) for s in samples])
        rubato = extract_piece_rubato_features(tempo)
        sequences = {}
        for channel in CHANNELS:
            if channel in ("tempo", "rubato"):
                raw = {r["key"]: (tempo[r["key"]].individual_tempo_sequence if channel == "tempo" else
                                 rubato[r["key"]].relative_rubato_sequence) for r in rs}
                scale = fit_residual_scale(list(raw.values()), method="mad")
                sequences[channel] = {k: standardize_sequence(v, scale) for k, v in raw.items()}
            else:
                inputs = [PieceFeatureInput(r["key"], piece, r["score_beats"],
                          BeatSequence(r[f"{channel}_values"], r[f"{channel}_mask"]), r["score_types"]) for r in rs]
                separated = separate_piece_feature(inputs)
                normalized = standardize_piece_feature(list(separated.values()), feature_name=channel)
                sequences[channel] = {k: v.standardized for k, v in normalized.items()}
                scale = next(iter(normalized.values())).scale
            scales.append({"piece": piece, "grid": grid, "channel": channel, **asdict(scale)})
        shared = np.logical_and.reduce([s.mask & np.isfinite(s.values) for items in sequences.values() for s in items.values()])
        audit.update(shared_beats=int(shared.sum()), shared_fraction=float(shared.mean()))
        if shared.sum() < args.min_beats or shared.mean() < args.min_coverage:
            audit["reason"] = "too_few_shared_valid_beats"
        else:
            audit["accepted"] = True
            values = np.array([[sequences[c][r["key"]].values[shared] for c in CHANNELS] for r in rs])
            assert values.shape == (len(rs), 7, int(shared.sum())) and np.isfinite(values).all()
            groups.append({"piece": piece, "grid": grid, "keys": [r["key"] for r in rs],
                           "values": values, "shared_indices": np.flatnonzero(shared)})
        audits.append(audit)
    if not groups:
        raise ValueError("No eligible pieces")
    return groups, audits, scales, provenance


def evaluate(groups, args):
    rows, embedding_rows, examples, contribution_rows = [], [], {}, []
    for gi, g in enumerate(groups):
        n, _, t = g["values"].shape
        gallery = np.array([summarize(v) for v in g["values"]])
        for key, vector in zip(g["keys"], gallery):
            embedding_rows.append({"piece": g["piece"], "grid": g["grid"], "key": key,
                                   **{f"{c}_{stat}": float(vector[2*i+j]) for i, c in enumerate(CHANNELS)
                                      for j, stat in enumerate(("median_abs" if c == "rubato" else "mean", "p95_p5"))}})
        upper = np.triu_indices(n, 1)
        components = np.array([np.mean((gallery[:, None, ix] - gallery[None, :, ix]) ** 2, axis=2)[upper].mean()
                               for ix in BLOCKS.values()])
        if components.sum() > 1e-12:
            contribution_rows.extend({"piece": g["piece"], "block": b, "distance_share": float(v / components.sum())}
                                     for b, v in zip(BLOCKS, components))
        clean_dist = weighted_distances(gallery, gallery, "all")
        norm = float(np.median(clean_dist[upper]))
        for pattern, fraction in CONDITIONS:
            trials = 1 if pattern == "clean" else args.trials
            seed = int.from_bytes(hashlib.sha256(f"{args.seed}/{g['piece']}/{g['grid']}/{pattern}/{fraction}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            queries = np.array([summarize(g["values"][i][:, retained_indices(t, fraction, pattern, rng)])
                                for _ in range(trials) for i in range(n)])
            targets = np.tile(np.arange(n), trials)
            for config in CONFIGS:
                dist = weighted_distances(queries, gallery, config)
                metrics = retrieval_metrics(dist, targets)
                rows.append({"piece": g["piece"], "grid": g["grid"], "candidates": n, "shared_beats": t,
                             "pattern": pattern, "drop_fraction": fraction, "config": config, "queries": len(queries),
                             "chance": 1/n, **{k: float(np.mean(v)) for k,v in metrics.items()},
                             "positive_margin_fraction": float(np.mean(metrics["margin"] > 1e-12))})
                if config == "all" and pattern == "random" and fraction == .10:
                    examples[g["piece"], g["grid"]] = {"distances": dist, "targets": targets, "metrics": metrics,
                                                       "norm": norm, "keys": g["keys"], "queries": queries,
                                                       "gallery": gallery, "shared_indices": g["shared_indices"]}
        if (gi + 1) % 10 == 0:
            print(f"Evaluated {gi + 1}/{len(groups)} groups", flush=True)
    return pd.DataFrame(rows), embedding_rows, examples, contribution_rows


def aggregate(table, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for (pattern, fraction, config), frame in table.groupby(["pattern", "drop_fraction", "config"], sort=True):
        values = frame.top1.to_numpy()
        boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"pattern": pattern, "drop_fraction": fraction, "config": config, "groups": len(frame),
                     "queries": int(frame.queries.sum()), "macro_top1": float(values.mean()),
                     "ci_low": float(lo), "ci_high": float(hi), "macro_mrr": float(frame.mrr.mean()),
                     "macro_chance": float(frame.chance.mean()), "macro_margin": float(frame.margin.mean())})
    return pd.DataFrame(rows)


def paired_comparisons(table, seed):
    frame = table[(table.pattern == "random") & (table.drop_fraction == .10)]
    pivot = frame.pivot(index=["piece", "grid"], columns="config", values="top1")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pivot), (2000, len(pivot)))
    rows = []
    for config in pivot.columns:
        delta = (pivot["all"] - pivot[config]).to_numpy()
        low, high = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
        rows.append({"config": config, "all_minus_variant": float(delta.mean()),
                     "ci_low": float(low), "ci_high": float(high)})
    return rows


def plot_results(table, summary, examples, contributions, out):
    main = summary[(summary.pattern == "random") & (summary.drop_fraction == .10)].set_index("config")
    preferred = [k for k in examples if k[0] == "Bach/Fugue/bwv_848/midi_score.mid"]
    preferred += [k for k in examples if "bwv_848" in k[0]]
    key = preferred[0] if preferred else sorted(examples)[0]
    example = examples[key]
    metrics = example["metrics"]
    qi = int(np.argsort(metrics["margin"], kind="stable")[len(metrics["margin"]) // 2])
    target = int(example["targets"][qi])
    distances = example["distances"][qi]
    order = np.argsort(distances, kind="stable")
    piece_label=key[0].removesuffix("/midi_score.mid").replace("/", " · ")
    fig, ax = chart("beat가 일부 없어도 원래 연주를 찾아낼까?", f"{piece_label} | beat 10% 누락 | 입력의 원래 연주: {Path(example['keys'][target]).stem}")
    colors = [BLUE if i == target else GRAY for i in order]
    ax.barh(np.arange(len(order)), distances[order], color=colors)
    ax.set(yticks=np.arange(len(order)), yticklabels=[Path(example["keys"][i]).stem + (" ← 원래 연주" if i == target else "") for i in order], xlabel="다섯 feature 임베딩 거리 (작을수록 가까움)")
    ax.invert_yaxis()
    save(fig, out / "01_retrieval_example.png", "각 후보까지 거리를 비교한다. 이 예시는 해당 작품의 거리 여유 중앙 순위 시행이며, 가장 잘된 사례를 고르지 않았다.")
    def bars(configs, title, subtitle, filename, note, failure=False):
        frame = main.loc[configs]
        y = np.arange(len(frame))
        fig, ax = chart(title, subtitle, height=max(5.8, 2.5+.45*len(frame)))
        rate = 1000*(1-frame.macro_top1) if failure else 100*frame.macro_top1
        low = 1000*(1-frame.ci_high) if failure else 100*frame.ci_low
        high = 1000*(1-frame.ci_low) if failure else 100*frame.ci_high
        ax.barh(y, rate, color=[BLUE if c == "all" else GRAY for c in configs], height=.55)
        ax.errorbar(rate, y, xerr=np.array([rate-low, high-rate]), fmt="none", color="#222b26", capsize=4)
        bound = max(float(high.max()), 1.) * 1.35 if failure else 110
        for yy, value, upper in zip(y, rate, high):
            ax.text(upper+bound*.01, yy, f"{value:.2f}회" if failure else f"{value:.1f}%", va="center", fontsize=12)
        if not failure:
            ax.axvline(100*frame.macro_chance.iloc[0], color=ORANGE, ls="--", label=f"무작위 선택 {100*frame.macro_chance.iloc[0]:.1f}%")
        ax.set(yticks=y, yticklabels=[LABELS[c] for c in configs], xlabel="검색 1,000회당 실패 횟수 (작을수록 좋음)" if failure else "원래 연주를 1순위로 찾은 비율 (%)", xlim=(0,bound))
        ax.invert_yaxis()
        if not failure:
            fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.58,.07),fontsize=10)
        save(fig, out/filename, note,bottom=.18 if not failure else .07)
    bars(["all"]+[f"only_{b}" for b in BLOCKS], "다섯 개를 합치면 단독 feature보다 잘 찾을까?", "같은 작품 안에서 검색 | beat 10% 무작위 누락 | 작품별 성공률의 평균", "02_combined_vs_single.png", "검은 선은 작품 단위 bootstrap 95% 구간이다. Pedaling의 세 지표는 하나의 feature 묶음으로 계산한다.")
    bars(["all"]+[f"without_{b}" for b in BLOCKS], "하나씩 빼면 검색 오류가 얼마나 늘어날까?", "beat 10% 누락 | 성공률이 모두 높으므로 실패 횟수로 확대해 비교", "03_leave_one_feature_out.png", "검은 선은 작품 bootstrap 95% 구간. 작은 차이만으로 해당 feature의 필요성을 확정하지 않는다.",failure=True)
    fig, ax = chart("beat가 더 많이 사라질수록 찾기 어려워질까?", "무작위 beat 누락 | 동일한 후보 집합과 고정된 공통값·scale")
    for config, color in (("all", BLUE), ("only_Tempo", GRAY), ("mean_only", ORANGE)):
        f = summary[(summary.pattern=="random") & (summary.config==config)].sort_values("drop_fraction")
        ax.plot(100*f.drop_fraction,100*f.macro_top1,marker="o",color=color,label=LABELS[config])
    ax.axhline(100*main.loc["all","macro_chance"],color=GRAY,ls=":",label="무작위 선택")
    ax.set(xlabel="누락 beat 비율 (%)",ylabel="원래 연주 검색 성공률 (%)",ylim=(0,105),xticks=[5,10,20,40])
    ax.legend(fontsize=10,loc="lower left")
    save(fig,out/"04_dropout_stability.png","한 feature의 실패를 다른 feature가 보완하는지 확인한다. 실제 오디오 오류나 정렬 오류를 시뮬레이션한 실험은 아니다.")
    per_piece = table[(table.pattern=="random") & (table.drop_fraction==.10) & (table.config=="all")]
    fig,ax=chart("일부 작품에 검색 오류가 몰려 있을까?", f"beat 10% 누락 | 비교 가능한 {len(per_piece)}작품/grid 그룹 | 작품별 실패 빈도")
    errors=1000*(1-per_piece.top1)
    ax.hist(errors,bins=np.arange(0,max(12,float(errors.max())+2),1),color=BLUE,rwidth=.85)
    ax.set(xlabel="작품별 검색 1,000회당 실패 횟수",ylabel="작품/grid 그룹 수",xlim=(0,max(12,float(errors.max())+2)))
    save(fig,out/"05_per_piece_success.png","그룹마다 후보 수와 beat 수가 다르다. 전체 성공률은 연주 수가 많은 작품에 치우치지 않도록 작품별 평균으로 계산했다.")
    own,other=[],[]
    for item in examples.values():
        if item["norm"]>1e-12:
            own.extend(item["metrics"]["self_distance"]/item["norm"])
            other.extend(item["metrics"]["nearest_other_distance"]/item["norm"])
    fig,ax=chart("작은 결측과 다른 연주의 차이를 구분할 수 있을까?", "beat 10% 누락 | 각 작품의 원본 연주 쌍 거리 중앙값을 1로 맞춘 거리")
    for yy,values,color in ((1,own,BLUE),(0,other,ORANGE)):
        low,med,high=np.percentile(values,[5,50,95])
        ax.plot([low,high],[yy,yy],color=color,lw=5,alpha=.6)
        ax.scatter([med],[yy],color=color,s=120)
        ax.text(high+.03,yy,f"중앙값 {med:.2f}",va="center")
    ax.set(yticks=[1,0],yticklabels=["원래 연주까지 거리","가장 가까운 다른 연주까지 거리"],xlabel="작품별로 단위를 맞춘 거리",ylim=(-.5,1.5),xlim=(0,max(np.percentile(own,95),np.percentile(other,95))*1.35))
    save(fig,out/"06_self_vs_other_distance.png","점은 중앙값, 선은 5~95백분위다. 이 그림은 query들을 모은 분포이며, 각 query의 순위 판정은 별도로 계산한다.")
    frame=pd.DataFrame(contributions).groupby("block").distance_share.mean().reindex(BLOCKS)
    fig,ax=chart("임베딩의 거리를 어떤 feature가 만들고 있을까?", "원본 후보 연주 쌍의 제곱 거리 기여도 | 작품/grid별 비율을 평균")
    ax.bar(np.arange(5),100*frame.values,color=[BLUE,GRAY,ORANGE,GREEN,"#8867ae"])
    for i,v in enumerate(frame):ax.text(i,100*v+1,f"{100*v:.1f}%",ha="center")
    ax.set(xticks=np.arange(5),xticklabels=list(BLOCKS),ylabel="전체 제곱 거리에서 차지하는 비율 (%)",ylim=(0,max(frame)*100+12))
    save(fig,out/"07_feature_distance_contribution.png","각 묶음의 가중치는 같지만 실제 거리 기여도는 다를 수 있다. 가중치를 데이터에 맞춰 학습하거나 조정하지 않았다.")
    bars(["all","mean_only","without_DA_range"],"변화 폭을 넣으면 검색 오류가 줄어들까?","beat 10% 무작위 누락 | 오류가 적으므로 1,000회당 횟수로 비교","08_summary_choice.png","Rubato 첫 좌표는 기존 편차 크기다. 평균만 조합이 이 조건에서 더 안정적이며, 음악적 유사성은 별도 검증이 필요하다.",failure=True)
    f=summary[(summary.config=="all") & (summary.drop_fraction==.20)].set_index("pattern")
    fig,ax=chart("누락이 한 구간에 몰려도 안정적일까?","다섯 feature 전체 | beat 20% 누락 | 무작위와 연속 누락 비교")
    rates=100*f.loc[["random","contiguous"],"macro_top1"].to_numpy()
    ax.bar([0,1],rates,color=[BLUE,ORANGE],width=.5)
    for i,v in enumerate(rates):ax.text(i,v+1,f"{v:.1f}%",ha="center",fontsize=15)
    ax.set(xticks=[0,1],xticklabels=["beat들이 흩어져 누락","연속된 beat 구간이 누락"],ylabel="원래 연주 검색 성공률 (%)",ylim=(0,110))
    save(fig,out/"09_contiguous_dropout.png","연속 누락은 공통 유효 beat 목록에서 연속된 구간이다. 곡 구간에 따라 표현 성향이 달라지는 영향을 포함한다.")
    return {"piece":key[0],"grid":key[1],"query_index":qi,"target_key":example["keys"][target],
            "trial_index":qi//len(example["keys"]),"margin":float(metrics["margin"][qi]),
            "top1":float(metrics["top1"][qi]),"query_vector":example["queries"][qi].tolist(),
            "shared_beat_count":len(example["shared_indices"]),
            "retained_beat_count":len(example["shared_indices"])-int(round(len(example["shared_indices"])*.10)),
            "distances":[{"key":k,"distance":float(d)} for k,d in zip(example["keys"],distances)]}


def write_report(stats, table, summary, paired, audits, contributions, out):
    main=summary[(summary.pattern=="random") & (summary.drop_fraction==.10)].set_index("config")
    all_result=main.loc["all"]
    rates="\n".join(f"| {LABELS[c]} | {100*main.loc[c,'macro_top1']:.2f}% | {100*main.loc[c,'ci_low']:.2f}~{100*main.loc[c,'ci_high']:.2f}% |"
                    for c in ["all"]+[f"only_{b}" for b in BLOCKS])
    deltas="\n".join(f"| {LABELS[r['config']]} | {100*r['all_minus_variant']:+.3f}%p | {100*r['ci_low']:+.3f}~{100*r['ci_high']:+.3f}%p |"
                     for r in paired if r["config"].startswith("without_") and r["config"]!="without_DA_range")
    piece=table[(table.pattern=="random") & (table.drop_fraction==.10) & (table.config=="all")]
    condition=summary[summary.config=="all"]
    stresses="\n".join(f"| {'무작위' if r.pattern=='random' else '연속 구간'} | {100*r.drop_fraction:.0f}% | {100*r.macro_top1:.2f}% |"
                       for r in condition.itertuples() if r.pattern!="clean")
    accepted=[a for a in audits if a["accepted"]]
    shares=pd.DataFrame(contributions).groupby("block").distance_share.mean()
    share_text=" / ".join(f"{b} {100*shares[b]:.1f}%" for b in BLOCKS)
    example=stats["example"]
    other=min(r["distance"] for r in example["distances"] if r["key"]!=example["target_key"])
    own=next(r["distance"] for r in example["distances"] if r["key"]==example["target_key"])
    text=f"""# 다섯 feature 임베딩의 연주 검색·결측 안정성

**현재 설정에서는 다섯 feature를 합친 벡터가 단독 feature보다 원래 연주를 안정적으로 찾았다.**
beat 10% 누락 시 작품별 평균 성공률은 **{100*all_result.macro_top1:.2f}%**,
무작위 후보 선택은 **{100*all_result.macro_chance:.2f}%**다.
다만 이것은 **동일 녹음의 feature 일부가 누락된 상황에서 원본을 찾는 검사**다.
음악적으로 비슷한 다른 해석을 찾거나 사용자가 선호할 연주를 추천하는 성능은 아직 측정하지 않았다.

## 그림을 읽는 순서

PNG마다 그래프 하나만 있다. 01→02→04를 먼저 읽고, 나머지를 보면 된다.

| 그림 | 쉽게 읽는 방법 |
|---|---|
| [01 검색 예시](01_retrieval_example.png) | 입력 연주의 beat 일부를 지우고 각 후보까지 거리를 잰다. 파란 원래 연주의 막대가 가장 짧은지 본다. |
| [02 전체 조합과 단독 비교](02_combined_vs_single.png) | 막대가 길수록 원래 연주를 더 잘 찾는다. 주황 점선은 무작위 선택이다. |
| [03 하나씩 제외](03_leave_one_feature_out.png) | 성공률이 모두 높아서 **1,000회당 실패 횟수**로 확대했다. 짧을수록 좋다. |
| [04 누락 비율 변화](04_dropout_stability.png) | 더 많은 beat가 사라져도 성공률이 유지되는지 본다. |
| [05 작품별 오류 분포](05_per_piece_success.png) | 왼쪽에 작품들이 모이면 대다수 작품에서 오류가 적다는 뜻이다. |
| [06 원래 연주와 다른 연주까지 거리](06_self_vs_other_distance.png) | 파란 거리보다 주황 거리가 큰지 본다. 점=중앙값, 선=5~95백분위다. |
| [07 거리의 feature별 기여도](07_feature_distance_contribution.png) | 특정 feature 하나가 전체 거리를 지배하는지 확인한다. |
| [08 요약값 선택](08_summary_choice.png) | 변화 폭을 추가하는 것이 이 검색에서 도움이 되는지 비교한다. |
| [09 연속 구간 누락](09_contiguous_dropout.png) | 같은 양의 beat가 한 곳에 몰려 없어질 때도 안정적인지 본다. |

## 01. 실제 한 작품에서 무엇을 했나

예시 작품은 `{example['piece'].removesuffix('/midi_score.mid')}`이고,
입력의 원래 연주는 `{Path(example['target_key']).stem}`이다.
원래 연주까지 거리 **{own:.3f}**, 가장 가까운 다른 연주까지 거리 **{other:.3f}**였다.
해당 작품의 전체 query 시행 중 거리 여유가 중앙 순위인 사례를 골랐다.
가장 성공적인 시행을 고른 것이 아니다. 원래 연주는 후보 집합에 포함되어 있다.

## 02~03. 합치면 좋아졌나? 다섯 개가 모두 필요한가?

| 조합 | 검색 성공률 (작품별 평균) | 작품 bootstrap 95% 구간 |
|---|---:|---:|
{rates}

전체 조합은 각 단독 feature보다 높은 성공률을 보였다.
그러나 하나씩 제외해도 성공률이 거의 100%여서 **모든 feature의 필수성을 확정할 수는 없다.**
동일 query로 계산한 작품별 성공률 차이를 bootstrap한 결과는 아래와 같다.
양수는 전체 조합이 더 좋았다는 뜻이다. %p는 성공률의 차이다.

| 비교 조합 | 전체 − 비교 조합 | 차이의 95% 구간 |
|---|---:|---:|
{deltas}

Dynamics·Articulation·Tempo를 제외하면 이번 조건에서 작은 저하가 나타났고,
Rubato·Pedaling 제외의 차이 구간에는 0이 포함됐다.
관찰된 효과는 매우 작고, 여러 비교를 한 탐색 분석이다. 음악적 중요도 순위로 읽지 않는다.
검은 오차 막대는 반복 query를 독립 표본으로 세지 않고 **작품/grid 그룹을 재표집**한 구간이다.
동일 후보 집합과 정해진 누락 시뮬레이션 아래의 불확실성이다.

## 04~06·09. 얼마나 안정적이었나?

| 누락 방식 | 누락 비율 | 전체 조합 성공률 |
|---|---:|---:|
{stresses}

10% 무작위 누락에서 **{int((piece.top1==1).sum())}/{len(piece)}그룹**이 모든 시행에서 원본을 찾았다.
전체 query 수는 {int(piece.queries.sum()):,}개이고, 원본 검색 실패는
{float(np.sum(piece.queries*(1-piece.top1))):.0f}개였다.
연주 수로 가중한 전체 성공률은 {100*float(np.sum(piece.queries*piece.top1)/piece.queries.sum()):.2f}%이며,
그림의 {100*all_result.macro_top1:.2f}%는 작품마다 같은 비중을 준 평균이다.

06번 거리는 작품마다 원본 후보 연주 쌍의 거리 중앙값을 1로 맞춘 표시다.
각 query의 원래 연주까지 거리와 가장 가까운 다른 연주까지 거리를 비교한다.
분포 그림은 query들을 모은 백분위이며, 작품별 평균 성공률과 집계 방식이 다르다.
다른 작품 간 거리나 음악적 유사성을 나타내는 눈금으로 사용하지 않는다.

## 07~08. 거리 비중과 변화 폭

거리의 평균 기여도: **{share_text}**.
어느 한 축이 대부분을 차지하는 결과는 아니었다.
각 묶음은 같은 가중치지만, 데이터 분포에 따라 실제 거리 기여도는 달라진다.

10% 누락에서 전체 14차원은 **{100*main.loc['all','macro_top1']:.2f}%**,
첫 좌표만 쓰는 7차원은 **{100*main.loc['mean_only','macro_top1']:.2f}%**,
Dynamics·Articulation 변화 폭만 제외한 12차원은 **{100*main.loc['without_DA_range','macro_top1']:.2f}%**였다.
7차원은 Tempo/Dynamics/Articulation/Pedaling의 상대 성향과 Rubato의 편차 크기다.
**이 결측 검색에서는 변화 폭 추가가 필요하지 않았고, 오히려 첫 좌표만 쓰는 조합이 더 안정적이었다.**
이 결과를 보고 변화 폭을 원본에서 삭제하지 않는다. 후보 임베딩의 입력 선택 결과로 보존한다.
변화 폭이 음악적 표현 비교나 추천에서 유용한지는 별도 실험 대상이다.
이 보고서는 작품 ID 예측이나 추가 작품 보정 효과를 측정하지 않는다.

## 임베딩과 거리의 정확한 정의

학습 모델 없이 요약 feature를 이어 붙인 벡터다. 기존 추출기는 변경하지 않았다.

| feature 묶음 | 입력 | 전체 조합의 좌표 |
|---|---|---|
| Tempo | 기존 individual_tempo / 작품 내 pooled MAD scale | 평균, 95−5백분위 폭 |
| Rubato | 기존 relative_rubato / 별도 작품 내 pooled MAD scale | 절댓값 중앙값, 95−5백분위 폭 |
| Dynamics | 기존 standardized relative Dynamics | 평균, 95−5백분위 폭 |
| Articulation | 기존 standardized relative Articulation | 평균, 95−5백분위 폭 |
| Pedaling | depth/down_ratio/changes 각각 기존 SD standardized residual | 각 평균·폭, 총 6개 |

전체 14차원이다. Tempo·Rubato의 기존 상대 feature를 다시 공통 제거하지 않았고,
MAD 분모를 추가하는 정책은 **이 분석의 비교용 설정**이다.
D/A는 기존 MAD, 페달은 하위 feature마다 SD를 사용한다.
모든 값은 log2/비율/count 원본을 보존한 상태에서 요약하며 %로 역변환하지 않는다.

```text
feature 묶음 거리² = 그 묶음 좌표별 차이²의 평균
전체 거리 = sqrt(선택한 feature 묶음 거리²의 평균)
```

Pedaling은 좌표가 6개라도 한 묶음이다. 가중치를 학습하거나 결과에 맞춰 조정하지 않았다.
동일 원본 후보들의 공통값과 scale을 먼저 고정한 뒤 query를 만든다.
원본 후보는 기준 계산에 포함되지만 변형 query를 새 연주로 넣어 fit하지 않는다.
이는 주어진 후보 집합 안에서의 분석이며, 독립된 학습/평가 녹음 분할은 아니다.

## 대상·결측·동점 처리

- 원본 aligned 캐시 {stats['aligned_source_performances']}연주를 provenance 대조 후 사용했다.
- 동일 작품·동일 score grid·동일 반복 구조 안에서 robust note alignment 후보가 {stats['minimum_performances']}개 이상인 {stats['groups']}그룹, {stats['performances']}연주를 평가했다.
- 후보 수는 {min(a['robust_candidates'] for a in accepted)}~{max(a['robust_candidates'] for a in accepted)}개다. 결과는 이 모집단에 한정된다.
- 모든 후보·7채널이 함께 유효한 beat를 사용한다. 후보별 결측 mask가 ID 단서가 되지 않도록 동일한 위치로 제한했다.
- 공통 유효 beat는 {min(a['shared_beats'] for a in accepted)}~{max(a['shared_beats'] for a in accepted)}개, 최소 공유 비율은 {100*min(a['shared_fraction'] for a in accepted):.1f}%였다. 채택 기준은 {stats['minimum_shared_beats']}개·{100*stats['minimum_shared_coverage']:.0f}% 이상이다.
- random은 공유 beat 목록에서 비복원 추출로 지운다. contiguous는 이 목록에서 연속 구간을 지우므로 원래 score에 결측 틈이 있을 수 있다.
- 지운 개수는 `round(shared_beats × fraction)`이며, 모든 채널·비교 조합에 같은 query beat 위치를 쓴다.
- 조건마다 연주당 {stats['trials']}회 시행했다. clean 대조군만 1회다. seed={stats['seed']}.
- 후보가 같은 거리에 있으면 순서를 임의로 정해 정답 처리하지 않는다. 동점 중 균일 선택의 기대 성공률과 기대 reciprocal rank를 계산한다. 거리 동점 허용오차는 1e-12다.
- 마스킹된 값의 0 대체·결측 보간·clipping은 하지 않는다. 연주 ID·작품 ID·beat 수·mask 자체는 벡터에 넣지 않는다.

## 결론의 범위와 다음 단계

**이 임베딩은 동일 작품의 원본 후보를 찾는 결측 안정성 검사에서 구별력을 보였다.**
특히 성향 중심의 작은 벡터도 강한 후보였다.
다만 무작위 10% 누락이면 query의 90%는 후보 원본과 같은 beat이고,
곡 전체 평균은 이런 작은 누락에 잘 유지되므로 **쉬운 조건에서 성능이 포화된 결과**다.
독립 녹음·센서 보정·정렬 오류·속도/음 길이 변경을 실험한 결과가 아니다.
요약값이 같은 서로 다른 곡선을 구분하는 능력도 아직 확인하지 않았다.

이후에는 해석 유사성의 기준이나 선호 평가 데이터를 정하고,
기존 추천과 해석 임베딩을 추가한 추천을 같은 평가 조건에서 비교해야 한다.
이번 성공률을 추천 정확도 또는 사용자 만족도로 보고하지 않는다.

## 재현과 파일

저장소 루트에서:

```bash
.venv/bin/python classicfy-ai/scripts/validate_embedding.py
```

원본 캐시는 저장소 밖 `Classicfy/datasets/feature_normalization_raw.npz`다.
없으면 [전체 feature 분석 스크립트](../../scripts/validate_feature_normalization.py)로 먼저 생성한다.
`--asap-root`, `--nasap-root`, `--cache`, `--out`, `--seed`, `--trials`로 경로·시행을 명시할 수 있다.
기본 경로는 스크립트 위치 기준이므로 실행 디렉터리에 의존하지 않는다.

| 파일 | 내용 |
|---|---|
| [summary.csv](summary.csv) | 조건·조합별 작품 평균 성공률/MRR·95% 구간 |
| [piece_results.csv](piece_results.csv) | 모든 작품·조건·조합의 성공률·거리·여유·동점 집계 |
| [paired_comparisons.csv](paired_comparisons.csv) | 10% 누락에서 전체 조합 대비 성공률 차이와 paired CI |
| [embeddings.csv](embeddings.csv) | 원본 {stats['performances']}연주의 14차원 벡터 |
| [collection.csv](collection.csv) | 포함/제외 그룹·후보 수·공유 beat 비율·사유 |
| [scales.csv](scales.csv) | 작품/channel별 고정 scale과 fallback |
| [distance_contributions.csv](distance_contributions.csv) | 작품별 5묶음 제곱 거리 기여도 |
| [example_distances.csv](example_distances.csv) | 01 그림의 실제 후보 거리 |
| [stats.json](stats.json) | cohort·seed·정책·선택 예시·dataset revision·cache/code hash |
| [재현 스크립트](../../scripts/validate_embedding.py) | 모든 그림·CSV·JSON·안내 재생성 |

코드 검증은 `tests/unit/features/test_embedding_validation.py`에서 동점·순위·feature 묶음 가중치·제외·seed·입력 불변을 다룬다.
"""
    (out/"README.md").write_text(text,encoding="utf-8")


def main():
    ai=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root",type=Path,default=ai.parent.parent/"datasets/ASAP")
    parser.add_argument("--nasap-root",type=Path,default=ai.parent.parent/"datasets/nASAP")
    parser.add_argument("--cache",type=Path,default=ai.parent.parent/"datasets/feature_normalization_raw.npz")
    parser.add_argument("--out",type=Path,default=ai/"analysis/_embedding_validation")
    parser.add_argument("--trials",type=int,default=20)
    parser.add_argument("--seed",type=int,default=20261005)
    parser.add_argument("--min-performances",type=int,default=5)
    parser.add_argument("--min-beats",type=int,default=32)
    parser.add_argument("--min-coverage",type=float,default=.5)
    args=parser.parse_args()
    if args.trials<1 or args.min_performances<2 or args.min_beats<4 or not 0<args.min_coverage<=1:
        parser.error("Invalid trial/count/coverage settings")
    args.asap_root,args.nasap_root=args.asap_root.resolve(),args.nasap_root.resolve()
    print("Loading provenance-checked features and forming common valid grids",flush=True)
    groups,audits,scales,provenance=collect_groups(args)
    print(f"Eligible: {len(groups)} groups, {sum(len(g['keys']) for g in groups)} performances",flush=True)
    table,embeddings,examples,contributions=evaluate(groups,args)
    summary=aggregate(table,args.seed)
    paired=paired_comparisons(table,args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    setup_style()
    example=plot_results(table,summary,examples,contributions,args.out)
    for name,data in (("piece_results.csv",table.to_dict("records")),("summary.csv",summary.to_dict("records")),
                      ("embeddings.csv",embeddings),("collection.csv",audits),("scales.csv",scales),
                      ("distance_contributions.csv",contributions),("example_distances.csv",example["distances"]),
                      ("paired_comparisons.csv",paired)):
        write_csv(args.out/name,data)
    src=ai/"src"
    paths=[Path(__file__),Path(__file__).with_name("validate_tempo.py"),Path(__file__).with_name("validate_feature_normalization.py"),
           *sorted((src/"features").glob("*.py")),*sorted((src/"preprocessing").glob("*.py"))]
    stats={"groups":len(groups),"performances":len(embeddings),"aligned_source_performances":sum(a["aligned_candidates"] for a in audits),
           "queries_per_config":int(table[table.config=="all"].queries.sum()),"seed":args.seed,"trials":args.trials,
           "minimum_performances":args.min_performances,"minimum_shared_beats":args.min_beats,"minimum_shared_coverage":args.min_coverage,
           "conditions":CONDITIONS,"configs":CONFIGS,"channels":CHANNELS,"blocks":BLOCKS,"dimensions":14,
           "source_provenance":provenance,"cache_sha256":hashlib.sha256(args.cache.read_bytes()).hexdigest(),
           "analysis_code_sha256":hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest(),
           "example":example,"scope":"Same-recording identity retrieval under beat dropout with known piece; no independent recording, human interpretation labels or recommendation evaluation",
           "normalization":"D/A MAD; each pedal SD; existing Tempo/Rubato residuals independently pooled MAD for this experiment; prototypes and scales fitted to fixed original candidate cohort, never refitted with queries",
           "mask_policy":"All candidates and all seven channels use the same intersection of finite valid score beats; query dropout locations shared across channels/configs; no masks/counts/durations in embedding",
           "aggregation":"Piece/grid macro average; 2000 bootstrap resamples of piece/grid units; all variants share query inputs; ties expected under uniform random tie ordering",
           "clipping":None,"learned_weights":False}
    (args.out/"stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    write_report(stats,table,summary,paired,audits,contributions,args.out)
    print(summary[(summary.pattern=="random") & (summary.drop_fraction==.10)].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
