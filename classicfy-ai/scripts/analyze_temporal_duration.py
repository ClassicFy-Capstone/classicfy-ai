"""Measure observed duration and repetition descriptors without an encoder."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyze_temporal_shape import ROOT, REFERENCE_COMMIT, restore_channels, verified_source_files
from validate_embedding import CHANNELS, collect_groups
from validate_tempo import plt, setup_style
from features.temporal_shape import adjacent_changes
from features.temporal_duration import lagged_correlation, summarize_temporal_duration

DEFAULT_LAGS = (1, 2, 4, 8, 16, 32)


def analyze(items, *, lags=DEFAULT_LAGS, min_pairs=32, direction_tolerance=1e-6,
            level_tolerance=1e-6, std_tolerance=1e-12):
    lags = tuple(lags)
    if not lags or len(set(lags)) != len(lags):
        raise ValueError("At least one unique lag is required")
    durations, correlations = [], []
    for item in items:
        identity = {k: item[k] for k in ("piece", "grid", "key", "split")}
        for channel in CHANNELS:
            sequence = item["channels"][channel]
            durations.append(identity | dict(channel=channel, **summarize_temporal_duration(
                sequence, direction_tolerance=direction_tolerance, level_tolerance=level_tolerance)))
            row = identity | dict(channel=channel)
            for mode, data in (("value", sequence), ("delta", adjacent_changes(sequence))):
                for lag in lags:
                    metrics = lagged_correlation(data, lag, min_pairs=min_pairs, std_tolerance=std_tolerance)
                    for field in ("correlation", "valid_pairs", "possible_pairs", "pair_coverage", "status"):
                        row[f"{mode}_lag{lag}_{field}"] = metrics[field]
            correlations.append(row)
    return durations, correlations


def write_csv(path, rows):
    """UTF-8 BOM for spreadsheet readers; LF keeps generated diffs readable."""
    if not rows:
        raise ValueError(f"No rows to export: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, out, name):
    for extension in ("png", "svg", "pdf"):
        path = out / f"{name}.{extension}"
        fig.savefig(path, dpi=180)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot_duration(rows, out):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, prefix, states, colors, title, units in (
        (axes[0], "direction", ("rise", "fall"), ("#dc6436", "#2878c8"),
         "A. 상승·하강이 이어지는 길이", "연주별 run 평균 (연속 delta step 수)"),
        (axes[1], "level", ("positive", "negative"), ("#dc6436", "#2878c8"),
         "B. 공통 패턴 위·아래 상태의 길이", "연주별 run 평균 (연속 beat 값 수)"),
    ):
        for offset, state, color in zip((-.17, .17), states, colors):
            data = [[r[f"{prefix}_{state}_run_mean"] for r in rows if r["channel"] == c
                     and np.isfinite(r[f"{prefix}_{state}_run_mean"])] for c in CHANNELS]
            box = ax.boxplot(data, positions=np.arange(len(CHANNELS)) + offset, widths=.25,
                             orientation="horizontal", patch_artist=True, showfliers=False,
                             manage_ticks=False)
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(.6)
        ax.set_yticks(np.arange(len(CHANNELS)), CHANNELS)
        ax.set_title(title, loc="left", fontsize=17)
        ax.set_xlabel(units)
        ax.set_xlim(left=0)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color="#dc6436", label="상승"), Patch(color="#2878c8", label="하강")],
                   loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2, frameon=False)
    axes[1].legend(handles=[Patch(color="#dc6436", label="양의 잔차"), Patch(color="#2878c8", label="음의 잔차")],
                   loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2, frameon=False)
    fig.suptitle("연주 특징의 관측된 지속 길이", fontsize=21, fontweight="bold", y=.98)
    fig.text(.5, .90, "작품 공통 제거·정규화 잔차 | 결측은 run을 끊음 | 경계에 닿는 run은 더 길 수 있음", ha="center")
    fig.text(.5, .025, f"{len(rows) // len(CHANNELS)}연주 전체 · 각 색: 연주별 평균 길이의 분포 · box Q1–Q3 / 1.5 IQR 수염 / 범위 밖 점 생략", ha="center", fontsize=11)
    fig.subplots_adjust(left=.12, right=.98, top=.78, bottom=.23, wspace=.6)
    save_figure(fig, out, "run_duration_overview")


def plot_correlations(rows, out, lags, min_pairs):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, mode, title in zip(axes, ("value", "delta"),
                               ("A. 값의 자기상관", "B. 변화량의 자기상관")):
        means = np.full((len(CHANNELS), len(lags)), np.nan)
        support = np.zeros(means.shape, dtype=int)
        for ci, channel in enumerate(CHANNELS):
            selected = [r for r in rows if r["channel"] == channel]
            for li, lag in enumerate(lags):
                finite = [r[f"{mode}_lag{lag}_correlation"] for r in selected
                          if r[f"{mode}_lag{lag}_status"] == "ok"]
                support[ci, li] = len(finite)
                if finite:
                    means[ci, li] = np.mean(finite)
        im = ax.imshow(means, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(lags)), lags)
        ax.set_yticks(np.arange(len(CHANNELS)), CHANNELS)
        ax.set_xlabel("lag (원래 beat-grid 간격)")
        ax.set_title(title, loc="left", fontsize=17)
        for ci in range(len(CHANNELS)):
            for li in range(len(lags)):
                value = means[ci, li]
                label = f"{value:.2f}\nn={support[ci, li]}" if np.isfinite(value) else "미정의\nn=0"
                ax.text(li, ci, label, ha="center", va="center", fontsize=10,
                        color="white" if np.isfinite(value) and abs(value) > .65 else "#222222")
    fig.suptitle("beat 간격별 자기상관: 값과 변화량", fontsize=21, fontweight="bold", y=.98)
    fig.text(.5, .90, f"Pearson r의 연주 동일 가중 평균 | lag별 유효 쌍 ≥ {min_pairs} | n: 계산 가능한 연주 수", ha="center")
    fig.text(.5, .025, "결측을 가로지르는 쌍 제외 · 상수/지원 부족은 NaN · 상관이 높아도 음악적 반복의 증거는 아님", ha="center", fontsize=11)
    fig.subplots_adjust(left=.12, right=.89, top=.78, bottom=.17, wspace=.6)
    fig.colorbar(im, cax=fig.add_axes((.92, .17, .015, .61)), label="평균 r")
    save_figure(fig, out, "lag_correlation_overview")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root", type=Path, default=ROOT.parent / "datasets/ASAP")
    parser.add_argument("--nasap-root", type=Path, default=ROOT.parent / "datasets/nASAP")
    parser.add_argument("--cache", type=Path, default=ROOT.parent / "datasets/feature_normalization_raw.npz")
    parser.add_argument("--reference", type=Path, default=ROOT.parent / "datasets/temporal_embedding_run01/dataset.json")
    parser.add_argument("--out", type=Path, default=ROOT / "classicfy-ai/analysis/temporal_duration")
    parser.add_argument("--lags", type=int, nargs="+", default=DEFAULT_LAGS)
    parser.add_argument("--min-pairs", type=int, default=32)
    parser.add_argument("--direction-tolerance", type=float, default=1e-6)
    parser.add_argument("--level-tolerance", type=float, default=1e-6)
    parser.add_argument("--std-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    cache_hash = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if cache_hash != reference["cache_sha256"]:
        raise ValueError("Raw cache differs from reference manifest")
    files = verified_source_files()
    print("Verified reference sources/cache; collecting unchanged normalized cohorts", flush=True)
    groups, audits, scales, provenance = collect_groups(SimpleNamespace(
        asap_root=args.asap_root.resolve(), nasap_root=args.nasap_root.resolve(), cache=args.cache,
        **reference["cohort_settings"]), provenance_source_files=files)
    if provenance != reference["source_provenance"] or scales != reference["scales"]:
        raise ValueError("Source provenance or normalization scales differ from reference")
    items = restore_channels(groups, audits, reference["split"])
    durations, correlations = analyze(items, lags=args.lags, min_pairs=args.min_pairs,
        direction_tolerance=args.direction_tolerance, level_tolerance=args.level_tolerance,
        std_tolerance=args.std_tolerance)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "performance_durations.csv", durations)
    write_csv(args.out / "lag_correlations.csv", correlations)
    verification_path = args.out / "verification.json"
    if verification_path.exists():
        recorded = json.loads(verification_path.read_text()).get("csv_sha256", {})
        if any(recorded.get(name) != hashlib.sha256((args.out / name).read_bytes()).hexdigest()
               for name in ("performance_durations.csv", "lag_correlations.csv")):
            verification_path.unlink()
    code_paths = [Path(__file__), ROOT / "classicfy-ai/src/features/temporal_duration.py",
                  ROOT / "classicfy-ai/src/features/temporal_shape.py",
                  ROOT / "classicfy-ai/scripts/analyze_temporal_shape.py",
                  ROOT / "classicfy-ai/scripts/validate_embedding.py",
                  ROOT / "classicfy-ai/scripts/validate_feature_normalization.py"]
    protocol = dict(source_provenance=provenance, cache_sha256=cache_hash,
        reference_manifest_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        verified_source_reference=REFERENCE_COMMIT,
        analysis_code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        works=len({item["piece"] for item in items}), performances=len(items), channels=CHANNELS,
        split=reference["split"], cohort_settings=reference["cohort_settings"],
        normalization="unchanged per-work common removal and MAD/SD residual scale",
        mask="shared work/grid cohort mask restored to original indices",
        lags=args.lags, min_pairs=args.min_pairs, std_tolerance=args.std_tolerance,
        direction_tolerance=args.direction_tolerance, level_tolerance=args.level_tolerance,
        direction_duration_unit="consecutive delta steps (R steps span R+1 observed beat values)",
        level_duration_unit="consecutive original beat value entries",
        censoring="gap/sequence-edge boundaries may truncate runs; counts report both edges separately",
        correlation="lag-specific paired Pearson r pooled across fully valid lag+1 entry windows; delta mode requires lag+2 original beat values",
        figure_weighting="equal performance weights; duration uses per-performance run means; correlation excludes undefined entries and reports n",
        limits=["Exploratory settings and numeric descriptors; no recommendation/listening validation",
                "Beat grid/score structure influences length; no equivalent phrases across works assumed",
                "High value autocorrelation may mean smoothness or drift, not musical repetition",
                "Shared masks and censoring can shorten observed runs and reduce lag support"])
    (args.out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    plot_duration(durations, args.out)
    plot_correlations(correlations, args.out, args.lags, args.min_pairs)
    print(f"Saved {len(items)} performances / {protocol['works']} works; {len(correlations) * len(args.lags) * 2} lag measurements to {args.out}", flush=True)


if __name__ == "__main__":
    main()
