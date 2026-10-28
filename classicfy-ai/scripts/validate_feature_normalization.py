"""Collect aligned ASAP features, diagnose residual scales, and apply normalization.

Run from classicfy-ai; raw/normalized NPZ caches stay outside the repository.
--diagnose-only creates the figures used to choose the production policy.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import (
    BeatSequence, PieceFeatureInput, extract_articulation, extract_dynamics,
    extract_pedaling, separate_piece_feature,
    fit_residual_scale, standardize_piece_feature, standardize_sequence,
    DEFAULT_SCALE_METHODS,
    extract_note_articulation,
)
from preprocessing import ASAPLoader, load_match, load_midi

CHANNELS = ("dynamics", "pedal_depth", "pedal_down_ratio", "pedal_changes", "articulation")
COLORS = ("#2a78d6", "#898781", "#eb6834", "#1baf7a")
_LOADER = None


def init_worker(asap_root, nasap_root):
    global _LOADER
    _LOADER = ASAPLoader(asap_root, nasap_root)


def collect_sample(key):
    sample = _LOADER.get_sample(key)
    midi = load_midi(sample.performance_path)
    dynamics = extract_dynamics(midi, sample.performance_beats).sequence
    pedal = extract_pedaling(midi, sample.performance_beats)
    sequences = dict(zip(CHANNELS[:4], (dynamics, pedal.depth, pedal.down_ratio, pedal.changes)))
    sequences["articulation"] = (
        extract_articulation(load_match(sample.note_alignment_path), sample.performance_beats).sequence
        if sample.note_alignment_path else
        BeatSequence(np.full(len(dynamics), np.nan), np.zeros(len(dynamics), bool))
    )
    piece = PieceFeatureInput.from_asap_sample(sample, dynamics).piece_key
    piece = piece.replace(str(_LOADER.root) + "/", "")
    return {
        "key": key, "piece": piece, "composer": sample.composer,
        "robust": sample.robust_note_alignment, "match_available": sample.note_alignment_path is not None,
        "pedal_events": len(midi.pedals), "score_types": sample.score_beat_types,
        "score_beats": np.asarray(sample.score_beats),
        "beat_seconds": np.diff(sample.performance_beats),
        **{f"{name}_{field}": getattr(sequence, field)
           for name, sequence in sequences.items() for field in ("values", "mask")},
    }


def source_signature(asap_root, nasap_root, *, source_files=None):
    def revision(root):
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    src = Path(__file__).resolve().parents[1] / "src"
    paths = [asap_root / "metadata.csv", asap_root / "asap_annotations.json", nasap_root / "metadata.csv"]
    if source_files is None:
        source_files = [p for p in sorted((src / "features").glob("*.py"))
                        if p.name not in {"normalization.py", "__init__.py"}]
        source_files += sorted((src / "preprocessing").glob("*.py"))
    paths += list(source_files)
    return {
        "asap_revision": revision(asap_root), "nasap_revision": revision(nasap_root),
        "input_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest(),
        "collector_version": 1,
    }


def write_cache(path, records, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, metadata = {}, []
    for i, record in enumerate(records):
        metadata.append({k: v for k, v in record.items() if not isinstance(v, np.ndarray)})
        for name, values in record.items():
            if isinstance(values, np.ndarray):
                arrays[f"{i}/{name}"] = values
    arrays["manifest"] = np.array(json.dumps({"provenance": provenance, "records": metadata}))
    np.savez_compressed(path, **arrays)


def read_cache(path, provenance):
    with np.load(path, allow_pickle=False) as cache:
        manifest = json.loads(str(cache["manifest"]))
        if manifest["provenance"] != provenance:
            raise ValueError("Cache provenance changed; choose a new --cache path or remove the old cache")
        records = manifest["records"]
        for key in cache.files:
            if key != "manifest":
                index, name = key.split("/", 1)
                records[int(index)][name] = cache[key].copy()
    return records


def collect(args, provenance):
    if args.cache and args.cache.exists():
        print(f"Read cache: {args.cache}", flush=True)
        return read_cache(args.cache, provenance)
    loader = ASAPLoader(args.asap_root, args.nasap_root)
    keys = [sample.performance_key for sample in loader.iter_samples(aligned_only=True)]
    print(f"Extracting {len(keys)} aligned performances", flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(args.asap_root), str(args.nasap_root))) as pool:
        for i, result in enumerate(pool.map(collect_sample, keys, chunksize=4), 1):
            records.append(result)
            if i % 100 == 0:
                print(f"Extracted {i}/{len(keys)}", flush=True)
    if args.cache:
        write_cache(args.cache, records, provenance)
    return records


def separate(records, include_non_robust):
    # Grid variants are reported and grouped separately, never truncated or interpolated.
    grids = defaultdict(set)
    for r in records:
        digest = hashlib.sha256(r["score_beats"].tobytes() + json.dumps(r["score_types"]).encode()).hexdigest()[:12]
        r["grid"] = digest
        grids[r["piece"]].add(digest)
    groups = defaultdict(list)
    for r in records:
        piece = r["piece"] + (f"::grid:{r['grid']}" if len(grids[r["piece"]]) > 1 else "")
        for channel in CHANNELS:
            if channel == "articulation" and not include_non_robust and r["robust"] is not True:
                continue
            item = PieceFeatureInput(r["key"], piece, r["score_beats"],
                                     BeatSequence(r[f"{channel}_values"], r[f"{channel}_mask"]), r["score_types"])
            groups[channel, piece].append(item)
    separated, skipped = defaultdict(dict), []
    for (channel, piece), inputs in sorted(groups.items()):
        if len(inputs) < 2:
            skipped.append({"channel": channel, "piece": piece, "performances": len(inputs), "reason": "single_performance"})
        else:
            separated[channel][piece] = separate_piece_feature(inputs)
    return separated, skipped, {p: sorted(g) for p, g in grids.items() if len(g) > 1}


def valid(sequence):
    return sequence.values[sequence.mask & np.isfinite(sequence.values)]


def candidate_scales(values):
    if not len(values):
        return dict(mad=np.nan, iqr=np.nan, std=np.nan)
    return dict(mad=float(1.4826 * np.median(np.abs(values - np.median(values)))),
                iqr=float(np.diff(np.percentile(values, [25, 75]))[0] / 1.3489795003921634),
                std=float(np.std(values)))


def pooled(results, field):
    chunks = [valid(getattr(r, field)) for piece in results.values() for r in piece.values()]
    return np.concatenate(chunks) if chunks else np.zeros(0)


def diagnose(separated, out):
    rows, global_rows = [], []
    for channel, pieces in separated.items():
        for piece, results in pieces.items():
            values = np.concatenate([valid(r.relative) for r in results.values()])
            rows.append({"channel": channel, "piece": piece, "performances": len(results),
                         "valid_residuals": len(values), "zero_fraction": float(np.mean(values == 0)) if len(values) else np.nan,
                         **candidate_scales(values)})
        values = pooled(pieces, "relative")
        global_rows.append({"channel": channel, "valid_residuals": len(values),
                            "zero_fraction": float(np.mean(values == 0)) if len(values) else np.nan,
                            **candidate_scales(values)})
    scales = pd.DataFrame(rows)
    global_scales = pd.DataFrame(global_rows).set_index("channel")
    scales.to_csv(out / "piece_scale_candidates.csv", index=False)
    global_scales.to_csv(out / "global_scale_candidates.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    for channel in CHANNELS:
        pieces, subset = separated[channel], scales[scales.channel == channel]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for field, color in zip(("raw", "common", "relative"), COLORS):
            values = pooled(pieces, field)
            lo, hi = np.percentile(values, [.5, 99.5])
            if lo == hi:
                lo, hi = lo - .5, hi + .5
            axes[0, 0].hist(values, bins=np.linspace(lo, hi, 80), density=True,
                            histtype="step", label=field, color=color, linewidth=1.5)
        axes[0, 0].set_title("Raw / common / relative (central 99% shown)")
        axes[0, 0].legend()
        for method, color in zip(("mad", "iqr", "std"), COLORS):
            positive = subset[method][subset[method] > 0]
            axes[0, 1].hist(np.log10(positive), bins=25, histtype="step", linewidth=1.5,
                            color=color, label=f"{method.upper()}: zero={(subset[method] == 0).sum()}/{len(subset)}")
        axes[0, 1].set(xlabel="log10(piece residual scale)", title="Piece scales: zeros excluded from histogram")
        axes[0, 1].legend()
        residuals = pooled(pieces, "relative")
        order = np.sort(np.abs(residuals))
        axes[1, 0].plot(np.linspace(0, 1, len(order)), order, color=COLORS[2])
        axes[1, 0].set(xlabel="Cumulative fraction", ylabel="Absolute residual", title="All residuals including extremes")
        axes[1, 0].set_yscale("symlog", linthresh=.01)
        for method, color in zip(("mad", "iqr", "std"), COLORS):
            x = subset[method].to_numpy()
            axes[1, 1].scatter(subset.zero_fraction, x, s=15, alpha=.6, label=method.upper(), color=color)
        axes[1, 1].set(xlabel="Fraction of exact zero residuals", ylabel="Piece scale", title="Sparse/discrete residuals and scale failure")
        axes[1, 1].set_yscale("symlog", linthresh=.01)
        axes[1, 1].legend()
        global_row = global_scales.loc[channel]
        fig.suptitle(f"{channel} | {len(pieces)} comparable groups | pooled MAD={global_row['mad']:.4g}, "
                     f"IQR={global_row['iqr']:.4g}, SD={global_row['std']:.4g} | zeros={global_row.zero_fraction:.1%}")
        fig.tight_layout()
        fig.savefig(out / f"01_diagnose_{channel}.png", dpi=150)
        plt.close(fig)
    print(global_scales.to_string(), flush=True)
    print(scales.groupby("channel")[["mad", "iqr", "std"]].agg(lambda x: int((x == 0).sum())).to_string(), flush=True)
    return scales


def eta_squared(values, labels):
    frame = pd.DataFrame({"value": values, "label": labels}).dropna()
    if frame.empty:
        return np.nan
    total = ((frame.value - frame.value.mean()) ** 2).sum()
    if total == 0:
        return np.nan
    group_mean = frame.groupby("label").value.transform("mean")
    return float(((group_mean - frame.value.mean()) ** 2).sum() / total)


def plot_identity_means(metrics, out):
    """One readable before/after chart per label; retain all stages in the CSV."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    korean_font = next((name for name in ("Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR",
                                         "Malgun Gothic", "Arial Unicode MS") if name in installed), None)
    def text(korean, english):
        return korean if korean_font else english
    labels = text(("강약\nDynamics", "페달 깊이\nDepth", "페달 밟는 시간 비율\nDown ratio",
                   "페달 전환 횟수\nChanges", "음의 연결·분리\nArticulation"),
                  ("Dynamics", "Pedal depth", "Pedal down-time ratio", "Pedal changes", "Articulation"))
    stages = (("raw", text("처리 전 · 원본", "Before · raw"), "#7b93ac"),
              ("standardized", text("처리 후 · 공통 패턴 제거 + 단위 맞춤", "After · common removal + scaling"), "#197b6a"))
    with plt.rc_context({"font.family": [korean_font or "DejaVu Sans", "DejaVu Sans"], "font.size": 13,
                         "axes.unicode_minus": False}):
        for label, name, filename in (("piece", "작품", "04_identity_variance.png"),
                                      ("composer", "작곡가", "04_composer_identity_variance.png")):
            subset = metrics[(metrics.summary == "mean") & (metrics.label == label)]
            fig, ax = plt.subplots(figsize=(11.5, 7.8))
            fig.subplots_adjust(left=.265, right=.965, bottom=.25, top=.735)
            fig.text(.06, .95, text(f"평균 특징이 {name}별로 얼마나 달라지나?", f"How much do mean features differ by {label}?"), fontsize=24, weight="bold", color="#243730")
            fig.text(.06, .893, text(f"막대가 짧을수록 평균 특징이 {name}별로 덜 구분됩니다.", "Shorter bars mean less difference between groups in the mean features."), fontsize=15, color="#52605b")
            for i, (field, title, color) in enumerate(stages):
                rows = subset[subset.field == field].set_index("channel").reindex(CHANNELS)
                values = rows.eta_squared.to_numpy() * 100
                if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 100)):
                    raise ValueError("Identity figure requires finite eta-squared for every channel")
                positions = np.arange(len(CHANNELS)) + (i - .5) * .32
                ax.barh(positions, values, height=.27, color=color, label=title, zorder=3)
                for y, value in zip(positions, values):
                    ax.text(value + 1.2, y, f"{value:.1f}%", ha="left", va="center", fontsize=13,
                            color="#263932" if i else "#556879", weight="bold" if i else "normal")
            ax.set_yticks(np.arange(len(CHANNELS)), labels, fontsize=14)
            ax.set_ylim(len(CHANNELS)-.55, -.55)
            ax.set_xlim(0, 100)
            ax.set_xticks(np.arange(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
            ax.set_xlabel(text(f"연주 간 평균값 차이 중 {name}별 차이가 설명하는 비율", f"Share of variance in performance means explained by {label} (%)"), labelpad=16)
            ax.xaxis.grid(True, color="#e3e9e6", zorder=0)
            ax.tick_params(axis="both", length=0, pad=10)
            for spine in ax.spines.values(): spine.set_visible(False)
            fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.255, .83),
                       frameon=False, ncols=2, fontsize=13, handlelength=1.5, columnspacing=2)
            fig.text(.06, .09, text("처리 후 = 같은 작품의 공통 패턴을 뺀 뒤, feature별 숫자 범위를 맞춘 값", "After = subtract the piece's common pattern, then scale each feature"), fontsize=12, color="#52605b")
            fig.text(.06, .051, text("연주별 평균값의 분산을 비교한 비율(η²)입니다. 작품·작곡가 맞히기 정확도가 아닙니다.", "This is a descriptive variance ratio (η²), not piece/composer classification accuracy."), fontsize=11, color="#66706a")
            fig.text(.06, .019, text("평균값만 비교한 결과입니다. 연주 중 변화 폭에는 작품별 차이가 여전히 남습니다.", "Only means are compared here. Feature widths still contain piece differences."), fontsize=11, color="#66706a")
            fig.savefig(out / filename, dpi=170, facecolor="white")
            plt.close(fig)


def plot_pedal_duration(separated, records, out):
    by_key = {r["key"]: r for r in records}
    seconds, counts, residuals, standardized = [], [], [], []
    for pieces in separated["pedal_changes"].values():
        normalized = standardize_piece_feature(list(pieces.values()), feature_name="pedal_changes")
        for key, r in normalized.items():
            width = by_key[key]["beat_seconds"]
            mask = r.relative.mask & np.isfinite(width) & (width > 0)
            seconds.extend(width[mask])
            counts.extend(r.raw.values[mask])
            residuals.extend(r.relative.values[mask])
            standardized.extend(r.standardized.values[mask])
    seconds, counts, residuals, standardized = map(np.asarray, (seconds, counts, residuals, standardized))
    rates = counts / seconds
    rows, fig = [], plt.figure(figsize=(12, 8))
    for i, (label, values) in enumerate(zip(("raw changes / beat", "raw changes / second (alternative)",
                                            "relative changes / beat", "standardized changes"),
                                           (counts, rates, residuals, standardized)), 1):
        ax = fig.add_subplot(2, 2, i)
        correlation = float(np.corrcoef(np.log10(seconds), values)[0, 1])
        rows.append({"feature": label, "valid_beats": len(values), "correlation_log10_seconds": correlation,
                     "p99": float(np.percentile(values, 99)), "max": float(values.max())})
        # Log x and signed-log y reveal sparse count values and tails without removing data.
        y = np.sign(values) * np.log10(1 + np.abs(values))
        ax.hexbin(np.log10(seconds), y, gridsize=60, bins="log", mincnt=1, cmap="Blues")
        ax.set(xlabel="log10(beat interval seconds)", ylabel="signed log10(1 + |value|)",
               title=f"{label}\nPearson r with log-duration = {correlation:.3f}")
    fig.suptitle("Beat count and time rate measure different behavior; duration influence remains after scaling")
    fig.tight_layout()
    fig.savefig(out / "06_pedal_changes_duration.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "pedal_duration_comparison.csv", index=False)


def normalize_and_report(separated, records, args, provenance):
    by_key = {r["key"]: r for r in records}
    normalized, scale_rows, performance_rows, tail_rows, comparison_rows, checks = {}, [], [], [], [], []
    extreme_rows, output_records = [], []
    for channel in CHANNELS:
        pieces = separated[channel]
        normalized[channel] = {}
        candidate_values = defaultdict(list)
        global_scales = {m: fit_residual_scale([r.relative for p in pieces.values() for r in p.values()], method=m)
                         for m in ("mad", "std")}
        for piece, results in pieces.items():
            features = list(results.values())
            fitted = standardize_piece_feature(features, feature_name=channel)
            normalized[channel][piece] = fitted
            scale = next(iter(fitted.values())).scale
            scale_rows.append({"channel": channel, "piece": piece, "performances": len(fitted), **asdict(scale)})
            methods = {f"piece_{m}": fit_residual_scale([r.relative for r in features], method=m)
                       for m in ("mad", "iqr", "std")}
            methods.update({f"global_{m}": s for m, s in global_scales.items()})
            for label, denominator in methods.items():
                candidate_values[label].extend(valid(standardize_sequence(r.relative, denominator)) for r in features)
            for key, r in fitted.items():
                mask = r.relative.mask
                row = {"channel": channel, "piece": piece, "key": key, "composer": by_key[key]["composer"],
                       "robust_alignment": by_key[key]["robust"], "beats": len(mask),
                       "raw_valid": int((r.raw.mask & np.isfinite(r.raw.values)).sum()),
                       "relative_valid": int(mask.sum()), "standardized_valid": int(r.standardized.mask.sum()),
                       "scale": scale.value, "scale_method": scale.method}
                for field in ("raw", "common", "relative", "standardized"):
                    values = getattr(r, field).values[mask]
                    row[f"{field}_mean"] = float(np.mean(values)) if len(values) else np.nan
                    row[f"{field}_range"] = float(np.diff(np.percentile(values, [5, 95]))[0]) if len(values) else np.nan
                performance_rows.append(row)
                output_records.append({
                    "channel": channel, "key": key, "piece": piece, "score_types": r.score_beat_types,
                    "score_beats": r.score_beats, "minimum_support": r.minimum_support, "scale": asdict(scale),
                    "common_support": r.common_support,
                    **{f"{field}_{part}": getattr(getattr(r, field), part)
                       for field in ("raw", "common", "relative", "standardized") for part in ("values", "mask")},
                })
                indices = np.flatnonzero(mask)
                if len(indices):
                    top = indices[np.argsort(np.abs(r.standardized.values[indices]))[-3:]]
                    for beat in top:
                        extreme_rows.append({"channel": channel, "piece": piece, "key": key, "beat": int(beat),
                                             "score_time": float(r.score_beats[beat]), "beat_seconds": float(by_key[key]["beat_seconds"][beat]),
                                             "support": int(r.common_support[beat]), "robust_alignment": by_key[key]["robust"],
                                             "scale": scale.value,
                                             **{field: float(getattr(r, field).values[beat])
                                                for field in ("raw", "common", "relative", "standardized")}})
            # Compare every pair on shared valid positions, including all sparse pedal zeros.
            pair_error, scale_error, varying_pairs = 0., 0., 0
            fs = list(fitted.values())
            for i, a in enumerate(fs):
                for b in fs[i + 1:]:
                    overlap = a.relative.mask & b.relative.mask
                    if not overlap.any():
                        continue
                    raw_difference = (a.raw.values - b.raw.values)[overlap]
                    residual_difference = (a.relative.values - b.relative.values)[overlap]
                    standard_difference = (a.standardized.values - b.standardized.values)[overlap]
                    pair_error = max(pair_error, float(np.max(np.abs(raw_difference - residual_difference))))
                    scale_error = max(scale_error, float(np.max(np.abs(raw_difference / scale.value - standard_difference))))
                    varying_pairs += bool(np.any(np.abs(raw_difference) > 1e-12))
            checks.append({"channel": channel, "piece": piece, "max_pair_residual_error": pair_error,
                           "max_pair_scale_error": scale_error, "varying_pairs": varying_pairs})
        for method, chunks in candidate_values.items():
            values = np.abs(np.concatenate(chunks))
            comparison_rows.append({"channel": channel, "policy": method, "valid_beats": len(values),
                                    **{f"abs_p{q}": float(np.percentile(values, q)) for q in (95, 99, 99.9, 100)}})
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for field, color in zip(("raw", "common", "relative", "standardized"), COLORS):
            values = pooled(normalized[channel], field)
            tail_rows.append({"channel": channel, "field": field, "valid_beats": len(values),
                              **{f"p{q}": float(np.percentile(values, q)) for q in (0, 1, 5, 50, 95, 99, 100)},
                              "std": float(np.std(values))})
        for field, ax, color in (("relative", axes[0, 0], COLORS[2]), ("standardized", axes[0, 1], COLORS[3])):
            values = pooled(normalized[channel], field)
            bound = np.percentile(np.abs(values), 99.5)
            bound = max(bound, .01)
            ax.hist(values, bins=np.linspace(-bound, bound, 100), density=True, color=color, alpha=.8)
            ax.set(title=f"{field}: central 99.5% shown", xlabel=field, ylabel="Density")
        for label, chunks in candidate_values.items():
            values = np.abs(np.concatenate(chunks))
            qs = np.linspace(90, 100, 101)
            axes[1, 0].plot(qs, np.percentile(values, qs), label=label)
        axes[1, 0].set(xlabel="Absolute-value percentile", ylabel="|scaled residual|",
                       title="Candidate tails (MAD/IQR include SD fallback)", yscale="log")
        axes[1, 0].legend(fontsize=8)
        sc = pd.DataFrame([r for r in scale_rows if r["channel"] == channel])
        axes[1, 1].hist(sc.value, bins=25, color=COLORS[3])
        axes[1, 1].set(title="Selected denominator across piece groups", xlabel="Scale", ylabel="Groups")
        fig.suptitle(f"{channel}: piece {DEFAULT_SCALE_METHODS[channel].upper()} | no recentering, no clipping")
        fig.tight_layout()
        fig.savefig(args.out / f"02_normalized_{channel}.png", dpi=150)
        plt.close(fig)
        # Representative piece: enough performers, spread near the channel median.
        representatives = sc[sc.performances >= 5].copy()
        if representatives.empty:
            representatives = sc.copy()
        piece = representatives.iloc[(representatives.value - sc.value.median()).abs().argsort().iloc[0]].piece
        rs = list(normalized[channel][piece].values())
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for r in rs[:5]:
            x = np.arange(len(r.raw))
            axes[0].plot(x, np.where(r.relative.mask, r.raw.values, np.nan), lw=.8, alpha=.65,
                         label=Path(r.performance_key).stem[:35])
            axes[1].plot(x, np.where(r.relative.mask, r.relative.values, np.nan), lw=.8, alpha=.65)
            axes[2].plot(x, r.standardized.values, lw=.8, alpha=.65)
        axes[0].plot(np.arange(len(rs[0].common)), rs[0].common.values, color="black", lw=1.5, label="common median")
        for ax, label in zip(axes, ("raw / common", "relative", "standardized")):
            ax.set_ylabel(label)
        axes[0].legend(ncols=3, fontsize=7)
        axes[1].axhline(0, color="black", lw=.5)
        axes[2].axhline(0, color="black", lw=.5)
        axes[2].set_xlabel("Aligned score beat interval (no smoothing/interpolation)")
        fig.suptitle(f"{channel}: {piece}\nFirst five of {len(rs)} performances; scale={rs[0].scale.value:.5g}")
        fig.tight_layout()
        fig.savefig(args.out / f"03_curves_{channel}.png", dpi=150)
        plt.close(fig)
    table = pd.DataFrame(performance_rows)
    table.to_csv(args.out / "performance_summary.csv", index=False)
    pd.DataFrame(scale_rows).to_csv(args.out / "piece_scales.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(args.out / "distributions.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(args.out / "policy_comparison.csv", index=False)
    checks_table = pd.DataFrame(checks)
    checks_table.to_csv(args.out / "pair_preservation.csv", index=False)
    extremes = pd.DataFrame(extreme_rows)
    extremes["absolute_standardized"] = extremes.standardized.abs()
    extremes = extremes.sort_values("absolute_standardized", ascending=False).groupby("channel").head(20)
    extremes.to_csv(args.out / "extreme_beats.csv", index=False)
    # Verify the strongest observed tail for each channel against source MIDI/match.
    loader = ASAPLoader(args.asap_root, args.nasap_root)
    inspections = []
    for channel in CHANNELS:
        row = extremes[extremes.channel == channel].iloc[0]
        sample = loader.get_sample(row.key)
        midi = load_midi(sample.performance_path)
        beat = int(row.beat)
        start, end = sample.performance_beats[beat:beat + 2]
        notes = [n for n in midi.notes if start <= n.start < end]
        events = [p for p in midi.pedals if start <= p.time < end]
        if channel == "dynamics":
            recomputed = extract_dynamics(midi, sample.performance_beats).sequence.values[beat]
        elif channel == "articulation":
            recomputed = extract_articulation(load_match(sample.note_alignment_path), sample.performance_beats).sequence.values[beat]
        else:
            field = {"pedal_depth": "depth", "pedal_down_ratio": "down_ratio", "pedal_changes": "changes"}[channel]
            recomputed = getattr(extract_pedaling(midi, sample.performance_beats), field).values[beat]
        if not np.isclose(recomputed, row.raw, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"Source re-extraction mismatch: {channel}")
        inspections.append({"channel": channel, "key": row.key, "beat": beat,
                            "interval_start_seconds": start, "interval_end_seconds": end,
                            "source_raw": float(recomputed), "standardized": float(row.standardized),
                            "onset_notes": len(notes), "onset_velocities": [n.velocity for n in notes],
                            "cc64_events": [[p.time, p.value] for p in events],
                            "match_path": str(sample.note_alignment_path) if sample.note_alignment_path else None,
                            "robust_alignment": sample.robust_note_alignment,
                            "beat_types": sample.performance_beat_types[beat:beat + 2]})
        if channel == "articulation":
            alignment = load_match(sample.note_alignment_path)
            note_features = extract_note_articulation(alignment)
            inspections[-1]["aligned_notes"] = [
                {"pitch": alignment.matches[index][1].pitch,
                 "onset_seconds": alignment.matches[index][1].onset,
                 "held_seconds": alignment.matches[index][1].offset - alignment.matches[index][1].onset,
                 "score_duration_beats": alignment.matches[index][0].offset_beats - alignment.matches[index][0].onset_beats,
                 "log2_articulation": float(value)}
                for index, value in zip(note_features.match_indices, note_features.values)
                if start <= alignment.matches[index][1].onset < end
            ]
    (args.out / "extreme_source_checks.json").write_text(json.dumps(inspections, indent=2), encoding="utf-8")
    metrics = []
    rng = np.random.default_rng(8)
    for channel, group in table.groupby("channel"):
        for summary in ("mean", "range"):
            for field in ("raw", "relative", "standardized"):
                column = f"{field}_{summary}"
                for label in ("piece", "composer"):
                    eta = eta_squared(group[column], group[label])
                    shuffled = np.mean([eta_squared(group[column], rng.permutation(group[label])) for _ in range(100)])
                    metrics.append({"channel": channel, "summary": summary, "field": field, "label": label,
                                    "eta_squared": eta, "shuffled_mean": float(shuffled), "performances": len(group)})
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(args.out / "identity_variance.csv", index=False)
    plot_identity_means(metrics, args.out)
    for summary, filename in (("range", "05_identity_range_variance.png"),):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, label in zip(axes, ("piece", "composer")):
            subset = metrics[(metrics.summary == summary) & (metrics.label == label)]
            for i, field in enumerate(("raw", "relative", "standardized")):
                rows = subset[subset.field == field].set_index("channel").reindex(CHANNELS)
                ax.bar(np.arange(5) + (i - 1) * .25, rows.eta_squared, width=.25, label=field, color=COLORS[(0, 2, 3)[i]])
            ax.set(xticks=np.arange(5), xticklabels=CHANNELS, title=f"{label} eta-squared of performance {summary}", ylim=(0, 1))
            ax.tick_params(axis="x", rotation=25)
            ax.legend()
        axes[0].set_ylabel("Fraction of variance explained by label")
        fig.suptitle("Descriptive full-corpus diagnostics; not held-out ID prediction or recommendation evaluation")
        fig.tight_layout()
        fig.savefig(args.out / filename, dpi=150)
        plt.close(fig)
    plot_pedal_duration(separated, records, args.out)
    if (checks_table.max_pair_residual_error > 1e-10).any() or (checks_table.max_pair_scale_error > 1e-10).any():
        raise AssertionError("Pair differences were not preserved")
    if not (table.relative_valid == table.standardized_valid).all():
        raise AssertionError("Normalization changed valid residual coverage")
    artifact = args.normalized_cache
    write_cache(artifact, output_records, {**provenance, "include_non_robust": args.include_non_robust,
                                         "policy": dict(DEFAULT_SCALE_METHODS), "normalization_version": 1})
    stats = {"policy": dict(DEFAULT_SCALE_METHODS), "normalized_cache": str(artifact.resolve()),
             "normalized_cache_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
             "clipping": None, "recentering": False, "scope": "piece", "scale_support": "all valid residual beats",
             "channels": {c: {"groups": len(normalized[c]), "performances": int((table.channel == c).sum()),
                                "valid_beats": int(table.loc[table.channel == c, "standardized_valid"].sum()),
                                "valid_ratio": float(table.loc[table.channel == c, "standardized_valid"].sum() / table.loc[table.channel == c, "beats"].sum()),
                                "max_abs_standardized": float(extremes.loc[extremes.channel == c, "absolute_standardized"].max()),
                                "varying_pairs": int(checks_table.loc[checks_table.channel == c, "varying_pairs"].sum())} for c in CHANNELS},
             "max_pair_residual_error": float(checks_table.max_pair_residual_error.max()),
             "max_pair_scale_error": float(checks_table.max_pair_scale_error.max())}
    (args.out / "normalization.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


def main():
    datasets = Path(__file__).resolve().parents[3] / "datasets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asap-root", type=Path, default=datasets / "ASAP")
    parser.add_argument("--nasap-root", type=Path, default=datasets / "nASAP")
    parser.add_argument("--cache", type=Path, default=datasets / "feature_normalization_raw.npz")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "analysis/feature_normalization")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--include-non-robust", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--identity-figures-only", action="store_true",
                        help="Redraw mean identity charts from existing identity_variance.csv; no extraction or statistics recomputation")
    parser.add_argument("--normalized-cache", type=Path, default=datasets / "feature_normalization_standardized.npz")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.identity_figures_only:
        plot_identity_means(pd.read_csv(args.out / "identity_variance.csv"), args.out)
        return
    args.asap_root, args.nasap_root = args.asap_root.resolve(), args.nasap_root.resolve()
    provenance = source_signature(args.asap_root, args.nasap_root)
    records = collect(args, provenance)
    separated, skipped, variants = separate(records, args.include_non_robust)
    print(f"Comparable groups: { {c: len(p) for c, p in separated.items()} }", flush=True)
    pd.DataFrame(skipped).to_csv(args.out / "skipped_groups.csv", index=False)
    stats = {"provenance": provenance, "aligned_performances": len(records),
             "include_non_robust": args.include_non_robust, "grid_variants": variants,
             "minimum_support": 2, "articulation_robust_count": sum(r["robust"] is True for r in records)}
    (args.out / "collection.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    diagnose(separated, args.out)
    if not args.diagnose_only:
        normalize_and_report(separated, records, args, provenance)


if __name__ == "__main__":
    main()
