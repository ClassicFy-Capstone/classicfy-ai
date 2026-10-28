"""Measure change shape without an encoder; reuse a verified ASAP reference.

Run from repository root: .venv/bin/python classicfy-ai/scripts/analyze_temporal_shape.py
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from validate_embedding import CHANNELS, collect_groups
from validate_tempo import plt, setup_style, write_csv
from features.models import BeatSequence
from features.temporal_shape import fit_change_threshold, summarize_temporal_shape

REFERENCE_COMMIT = "20a4168"
ROOT = Path(__file__).resolve().parents[2]


def verified_source_files(repo=ROOT, reference=REFERENCE_COMMIT):
    """Allow added downstream code, but require all original sources unchanged."""
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", reference,
         "classicfy-ai/src/features", "classicfy-ai/src/preprocessing"], cwd=repo, text=True,
    ).splitlines()
    names = [name for name in names if name.endswith(".py")]
    if not names:
        raise ValueError("Reference source inventory is empty")
    for name in names:
        expected = subprocess.check_output(["git", "show", f"{reference}:{name}"], cwd=repo)
        if (repo / name).read_bytes() != expected:
            raise ValueError(f"Original extraction/normalization source changed: {name}")
    return [repo / name for name in names if "/features/" in name
            and Path(name).name not in {"normalization.py", "__init__.py"}] + [
                repo / name for name in names if "/preprocessing/" in name]


def restore_channels(groups, audits, split):
    """Restore original indices, retaining float64 and shared cohort masks."""
    lengths = {(a["piece"], a["grid"]): a["total_intervals"] for a in audits}
    items = []
    seen = set()
    for group in groups:
        length = lengths[group["piece"], group["grid"]]
        indices = np.asarray(group["shared_indices"])
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                or indices[0] < 0 or indices[-1] >= length or np.any(np.diff(indices) <= 0)):
            raise ValueError("Invalid original shared beat indices")
        piece = group["piece"].split("::", 1)[0]
        if split[piece] not in {"train", "validation", "test"}:
            raise ValueError("Unknown reference split")
        mask = np.zeros(length, dtype=bool)
        mask[indices] = True
        for key, source in zip(group["keys"], group["values"], strict=True):
            if key in seen:
                raise ValueError(f"Duplicate performance key: {key}")
            seen.add(key)
            if source.shape != (len(CHANNELS), len(indices)) or not np.isfinite(source).all():
                raise ValueError("Invalid normalized source values")
            channels = {}
            for channel, values in zip(CHANNELS, source, strict=True):
                full = np.full(length, np.nan)
                full[indices] = values
                channels[channel] = BeatSequence(full, mask)
            items.append(dict(piece=piece, grid=group["grid"], key=key,
                              split=split[piece], channels=channels))
    return items


def analyze(items, *, quantile=.95, tolerance=1e-6):
    thresholds, rows = [], []
    reference = [item for item in items if item["split"] == "train"]
    for channel in CHANNELS:
        threshold = fit_change_threshold((item["channels"][channel] for item in reference), quantile=quantile)
        thresholds.append(dict(channel=channel, **asdict(threshold),
                               reference_performances=len(reference),
                               reference_works=len({item["piece"] for item in reference})))
        for item in items:
            rows.append({k: item[k] for k in ("piece", "grid", "key", "split")} | dict(
                channel=channel, **summarize_temporal_shape(
                    item["channels"][channel], large_threshold=threshold.value,
                    direction_tolerance=tolerance)))
    return thresholds, rows


def plot_overview(rows, out):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(CHANNELS))
    groups = [[r for r in rows if r["channel"] == c and r["valid_pairs"]] for c in CHANNELS]
    axes[0].boxplot([[100 * r["large_ratio"] for r in rs] for rs in groups],
                    orientation="horizontal", tick_labels=CHANNELS, showfliers=False)
    axes[0].set_xlabel("큰 변화 비율 (%) · 유효한 인접 beat 쌍 기준")
    axes[0].set_title("A. 큰 변화 비율의 연주 간 분포", loc="left")
    left = np.zeros(len(CHANNELS))
    for direction, label, color in (("fall", "하강", "#2878c8"), ("flat", "유지", "#b8b8b1"),
                                    ("rise", "상승", "#dc6436")):
        values = np.array([100 * np.mean([r[f"{direction}_ratio"] for r in rs]) for rs in groups])
        axes[1].barh(y, values, left=left, label=label, color=color)
        left += values
    axes[1].set_yticks(y, CHANNELS)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("각 연주의 방향 비율을 동일 가중 평균 (%)")
    axes[1].set_title("B. 상승·유지·하강의 구성", loc="left")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False)
    fig.suptitle("인접 beat 변화의 형태: 5개 feature / 7개 채널", fontsize=21, fontweight="bold", y=.98)
    fig.text(.5, .90, "작품 공통 패턴 제거·정규화 후 측정 | 큰 변화 기준: train 작품 |Δ|의 채널별 p95", ha="center")
    fig.text(.5, .025, f"{len(rows) // len(CHANNELS)}연주 전체 · box: Q1–Q3, 중앙값, 1.5 IQR 수염 (범위 밖 점 생략) · 추천/청취 유사성 미검증", ha="center", fontsize=11)
    fig.subplots_adjust(top=.78, bottom=.23, left=.12, right=.98, wspace=.55)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(out / f"shape_overview.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root", type=Path, default=ROOT.parent / "datasets/ASAP")
    parser.add_argument("--nasap-root", type=Path, default=ROOT.parent / "datasets/nASAP")
    parser.add_argument("--cache", type=Path, default=ROOT.parent / "datasets/feature_normalization_raw.npz")
    parser.add_argument("--reference", type=Path, default=ROOT.parent / "datasets/temporal_embedding_run01/dataset.json")
    parser.add_argument("--out", type=Path, default=ROOT / "classicfy-ai/analysis/temporal_shape")
    parser.add_argument("--quantile", type=float, default=.95)
    parser.add_argument("--direction-tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    cache_hash = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if cache_hash != reference["cache_sha256"]:
        raise ValueError("Raw cache differs from reference manifest")
    files = verified_source_files()
    print("Verified original source inventory and raw cache; collecting normalized cohorts", flush=True)
    groups, audits, scales, provenance = collect_groups(SimpleNamespace(
        asap_root=args.asap_root.resolve(), nasap_root=args.nasap_root.resolve(), cache=args.cache,
        **reference["cohort_settings"]), provenance_source_files=files)
    if provenance != reference["source_provenance"] or scales != reference["scales"]:
        raise ValueError("Source provenance or normalization scales differ from reference")
    items = restore_channels(groups, audits, reference["split"])
    thresholds, rows = analyze(items, quantile=args.quantile, tolerance=args.direction_tolerance)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "performance_channels.csv", rows)
    write_csv(args.out / "thresholds.csv", thresholds)
    write_csv(args.out / "cohort_audit.csv", audits)
    code_paths = [Path(__file__), ROOT / "classicfy-ai/src/features/temporal_shape.py",
                  ROOT / "classicfy-ai/scripts/validate_embedding.py",
                  ROOT / "classicfy-ai/scripts/validate_feature_normalization.py"]
    protocol = dict(source_provenance=provenance, cache_sha256=cache_hash,
                    reference_manifest_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                    verified_source_reference=REFERENCE_COMMIT,
                    verified_source_files=[str(p.relative_to(ROOT)) for p in files],
                    analysis_code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
                    channels=CHANNELS, cohort_settings=reference["cohort_settings"],
                    works=len({item["piece"] for item in items}), performances=len(items),
                    split=reference["split"], quantile=args.quantile, direction_tolerance=args.direction_tolerance,
                    thresholds=thresholds, normalization="unchanged per-work residual and per-work MAD/SD scale",
                    validity="shared across all performances and channels in each work/grid; restored original indices",
                    weighting="threshold: pooled valid train pairs; figure: equal performance weights",
                    limits=["Exploratory numeric descriptors, not style or preference ground truth",
                            "Work-specific normalization uses the cohort, not single-performance online inference",
                            "Counts and quantiles lose longer-range ordering; duration/repetition is a separate next step",
                            "Beat length and score structure can still influence cross-work comparisons"])
    (args.out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    plot_overview(rows, args.out)
    print(f"Saved {len(items)} performances, {protocol['works']} works, {len(rows)} channel rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
