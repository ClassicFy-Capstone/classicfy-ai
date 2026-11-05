"""Render the saved Beethoven/Bach example without selecting pairs or retraining.

Run with Python plus matplotlib/numpy. Inputs are the checked-in four-performance
exports from the temporal-embedding-observations branch, not an external model run.
"""

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/classicfy-presentation-mpl")

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "capstone-2026-2-classicfy/classicfy-ai/analysis/composer_style_example"
OUT = ROOT / "presentation/figures/composer_metadata_vs_embedding"
IDS = ("B1", "J1", "B2", "J2")
BLOCKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9, 10, 11, 12, 13))
CORAL, BLUE, GREEN = "#D66B48", "#2878C8", "#148766"
GRAY, TEXT, MUTED = "#8C97A5", "#172635", "#586878"


def read_csv(name):
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_data():
    performances = {row["id"]: row for row in read_csv("selected_performances.csv")}
    raw = read_csv("selected_cnn_embeddings.csv")
    id_column = next(iter(raw[0]))
    vectors = {row[id_column]: [float(row[f"z{i}"]) for i in range(128)] for row in raw}
    if set(performances) != set(IDS) or set(vectors) != set(IDS):
        raise ValueError("Expected exactly the four saved performance IDs")
    z = np.array([vectors[code] for code in IDS])
    if not np.isfinite(z).all() or np.any(np.linalg.norm(z, axis=1) == 0):
        raise ValueError("Invalid saved CNN vectors")
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    distances = np.clip(1 - z @ z.T, 0, 2)
    centered = z - z.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ axes[:2].T
    saved_xy = np.array([[float(performances[code]["pc1"]),
                          float(performances[code]["pc2"])] for code in IDS])
    # PCA signs are arbitrary; keep the orientation of the existing presentation.
    for column in range(2):
        if np.dot(xy[:, column], saved_xy[:, column]) < 0:
            xy[:, column] *= -1
    if not np.allclose(xy, saved_xy, atol=1e-10, rtol=0):
        raise ValueError("Recomputed PCA differs from the saved four-vector PCA")
    variance = singular[:2] ** 2 / np.sum(singular ** 2)
    features = np.array([[float(performances[code][f"coordinate_{i}"])
                          for i in range(14)] for code in IDS])
    for row in read_csv("pairwise_distances.csv"):
        a, b = IDS.index(row["a"]), IDS.index(row["b"])
        expected_feature = np.sqrt(sum(np.mean((features[a, block] - features[b, block]) ** 2)
                                       for block in BLOCKS) / len(BLOCKS))
        if not np.isclose(distances[a, b], float(row["cnn_cosine_distance"]), atol=1e-10, rtol=0):
            raise ValueError("Cosine distance differs from the original export")
        if not np.isclose(expected_feature, float(row["feature_distance"]), atol=1e-10, rtol=0):
            raise ValueError("Feature distance differs from the original export")
    nearest = np.argmin(distances + np.eye(4) * 10, axis=1)
    if not np.array_equal(nearest, [1, 0, 3, 2]):
        raise ValueError("The saved example no longer has the reported nearest partners")
    return performances, distances, xy, variance


def setup_style():
    font_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        family = next((name for name in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic")
                       if any(font.name == name for font in font_manager.fontManager.ttflist)),
                      "DejaVu Sans")
    plt.rcParams.update({"font.family": [family, "DejaVu Sans"], "font.size": 11,
                         "text.color": TEXT, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.unicode_minus": False, "svg.fonttype": "path",
                         "axes.spines.top": False, "axes.spines.right": False})


def box(fig, x, y, w, h, color, edge="none", radius=.013):
    patch = FancyBboxPatch((x, y), w, h, transform=fig.transFigure,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=color, edgecolor=edge, linewidth=1, zorder=-1)
    fig.add_artist(patch)


def plot_metadata(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for y, color, title, codes in ((.52, CORAL, "베토벤", ("B1", "B2")),
                                    (.035, BLUE, "바흐", ("J1", "J2"))):
        ax.add_patch(FancyBboxPatch((.025, y), .95, .435,
                                   boxstyle="round,pad=0,rounding_size=.05",
                                   facecolor=color, alpha=.075, edgecolor="none"))
        ax.text(.5, y + .335, title, ha="center", va="center", fontsize=15,
                color=color, weight="bold")
        ax.plot([.265, .735], [y + .18, y + .18], color=color, lw=2, alpha=.22)
        for x, code in zip((.265, .735), codes):
            ax.scatter(x, y + .18, s=260, marker="o" if code[0] == "B" else "^",
                       color=color, edgecolor="white", linewidth=1.7, zorder=3)
            ax.text(x, y + .06, code, ha="center", va="center", fontsize=14,
                    color=color, weight="bold")


def plot_embedding(ax, xy, variance):
    for a, b in ((0, 2), (1, 3)):
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], ls="--", color=GRAY,
                lw=1.5, alpha=.65, zorder=1)
    for a, b in ((0, 1), (2, 3)):
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=GREEN, lw=3.1, zorder=2)
    for i, code in enumerate(IDS):
        color = CORAL if code[0] == "B" else BLUE
        ax.scatter(*xy[i], s=215, marker="o" if code[0] == "B" else "^",
                   color=color, edgecolor="white", linewidth=1.6, zorder=3)
        offset = 15 if i in (0, 3) else -20
        ax.annotate(code, xy[i], xytext=(0, offset), textcoords="offset points",
                    ha="center", va="center", fontsize=13, weight="bold", color=color)
    ax.set(xlim=(-.67, .74), ylim=(-.46, .42),
           xlabel=f"PC1 ({100 * variance[0]:.1f}%)",
           ylabel=f"PC2 ({100 * variance[1]:.1f}%)")
    # An equal physical scale preserves the geometry of this projection.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-.4, 0, .4])
    ax.set_yticks([-.4, 0, .4])
    ax.tick_params(labelsize=9, length=3)
    for spine in ax.spines.values():
        spine.set_color("#D7DEE5")
    ax.grid(color="#E8EDF1", linewidth=.8)
    ax.set_axisbelow(True)


def plot_distances(ax, distances):
    pairs = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (2, 1))
    values = [distances[a, b] for a, b in pairs]
    colors = [GREEN, GREEN, GRAY, GRAY, "#CAD2DC", "#CAD2DC"]
    ax.barh(np.arange(6), values, height=.59, color=colors)
    ax.set_yticks(np.arange(6), [f"{IDS[a]}–{IDS[b]}" for a, b in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0, .65)
    ax.set_xticks([0, .2, .4, .6])
    ax.set_xlabel("Cosine distance", fontsize=10, labelpad=8)
    ax.tick_params(axis="y", length=0, labelsize=11, pad=8)
    ax.tick_params(axis="x", labelsize=9, length=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#E8EDF1", linewidth=.8)
    ax.set_axisbelow(True)
    for y, value in enumerate(values):
        ax.text(value + .015, y, f"{value:.3f}", va="center", fontsize=11,
                color=GREEN if y < 2 else TEXT, weight="bold" if y < 2 else "normal")


def main():
    performances, distances, xy, variance = load_data()
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 9), facecolor="#F5F7FA")
    fig.text(.04, .945, "메타데이터를 넘어 연주 성향으로 연결하기", fontsize=27, weight="bold")
    fig.text(.04, .902, "동일 작품의 연주 차이를 표현한 임베딩이, 다른 작품·작곡가 사이에서도 유사한 성향을 연결하는가?",
             fontsize=12.5, color=MUTED)
    box(fig, .04, .817, .92, .053, "white")
    fig.text(.065, .844, "5개 feature로 연주 표현", va="center", fontsize=12, weight="bold")
    fig.text(.301, .844, "→", va="center", fontsize=18, color=GRAY)
    fig.text(.347, .844, "작품별 공통 패턴 제거·정규화", va="center", fontsize=12, weight="bold")
    fig.text(.648, .844, "→", va="center", fontsize=18, color=GRAY)
    fig.text(.693, .844, "연주 임베딩의 유사도 확인", va="center", fontsize=12, weight="bold")

    panels = ((.04, .275, "A  작곡가 기준의 그룹", "작곡가가 같으면 같은 범주"),
              (.337, .338, "B  연주 임베딩의 배치", "현재 CNN 128D → 네 연주의 PCA"),
              (.696, .264, "C  원래 128D 거리", "작을수록 임베딩에서 가까움"))
    for x, w, title, subtitle in panels:
        box(fig, x, .302, w, .493, "white")
        fig.text(x + .017, .754, title, fontsize=17, weight="bold")
        fig.text(x + .017, .720, subtitle, fontsize=10.5, color=MUTED)

    plot_metadata(fig.add_axes((.055, .367, .245, .325)))
    fig.text(.057, .330, "범주를 보여주는 개념도 · 수치 거리 없음", fontsize=9, color=MUTED)
    plot_embedding(fig.add_axes((.382, .398, .265, .285)), xy, variance)
    handles = [Line2D([], [], color=GREEN, lw=3, label="선정된 교차 작곡가 쌍"),
               Line2D([], [], color=GRAY, ls="--", label="같은 작곡가")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.506, .331),
               ncols=2, frameon=False, fontsize=9, columnspacing=1.4, handlelength=2.0)
    plot_distances(fig.add_axes((.772, .413, .168, .273)), distances)
    fig.text(.712, .330, "초록: feature 요약으로 먼저 선정한 쌍", fontsize=9, color=MUTED)

    box(fig, .04, .221, .92, .058, "#E6F3ED")
    fig.text(.058, .250,
             "관찰  |  선정된 네 연주 모두, 같은 작곡가의 연주보다 다른 작곡가의 선정 파트너에 더 가까움",
             va="center", fontsize=14, weight="bold", color="#126B52")
    fig.text(.04, .186,
             "추천 확장 가능성: 작곡가·작품 메타데이터에 더해, 연주 성향의 유사성을 후보 탐색 기준으로 활용",
             fontsize=12.5, weight="bold")
    labels = ((.04, .137, "B1", "베토벤 소나타 31번 1악장 · Huang01"),
              (.04, .107, "B2", "베토벤 소나타 21번 1악장 (반복 제외) · MorozovS01"),
              (.535, .137, "J1", "바흐 푸가 BWV 857 · YuP01M"),
              (.535, .107, "J2", "바흐 푸가 BWV 848 · SunY01M"))
    for x, y, code, description in labels:
        fig.text(x, y, code, fontsize=11, weight="bold", color=CORAL if code[0] == "B" else BLUE)
        fig.text(x + .029, y, description, fontsize=10.5, color=MUTED)
    fig.text(.04, .061,
             f"선정: 공통 제거·정규화 후 대표 성향 + 변화 폭의 14D 요약, 5개 feature 동일 비중. PCA 표시 분산 {100 * variance.sum():.2f}% · 거리 판정은 C 기준.",
             fontsize=9.2, color=MUTED)
    fig.text(.04, .035,
             "Feature 기준으로 고른 탐색 사례 · train/validation/test 혼합 · 청취 유사성과 추천 성능은 별도 평가 필요",
             fontsize=9.2, color=MUTED)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"composer_metadata_vs_embedding.{suffix}", dpi=200,
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    source_names = ("selected_performances.csv", "selected_cnn_embeddings.csv",
                    "pairwise_distances.csv", "stats.json")
    provenance = {
        "source_commit": "cfde116",
        "source_analysis": str(SOURCE.relative_to(ROOT)),
        "source_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                          for name in source_names},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": {"all_six_cosine_distances_match_export": True,
                   "all_six_feature_distances_match_export": True,
                   "four_vector_pca_matches_export_up_to_sign": True,
                   "each_selected_partner_is_nearest_among_these_four": True},
        "metadata_panel": "Categorical schematic only; no metadata embedding or measured distance",
        "selection": "Reuse saved feature-selected pairs; no new selection, model training, or weighting",
        "pca_variance_ratio": variance.tolist(),
        "performances": [{"id": code, "key": performances[code]["key"],
                          "split": performances[code]["split"]} for code in IDS],
        "nearest_among_four": {code: IDS[i ^ 1] for i, code in enumerate(IDS)},
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "numeric_checks": "passed",
                      "pca_explained_variance": float(variance.sum())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
