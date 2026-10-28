"""Render counts as shares of each complete dataset, preserving source units."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from _support import setup_fonts, clean_svg_outputs


def main():
    s = json.loads((OUT / "summary.json").read_text())
    setup_fonts()
    plt.rcParams.update({"axes.unicode_minus": False,
                         "svg.fonttype": "path", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), facecolor="white")
    fig.subplots_adjust(left=.09, right=.98, top=.69, bottom=.27, wspace=.32)
    fig.text(.065, .925, "ATEPP와 ASAP에서 겹치는 작품의 비중", fontsize=22, weight="bold", color="#1C2C40")
    fig.text(.065, .85, "전체 메타데이터 기준 · 작곡가·작품번호·악장 대조 · 연주 수는 같은 작품에 속한 연주 수", fontsize=11, color="#536176")
    sets = [
        ("작품 항목", "asap_total_work_keys", "atepp_total_composition_ids",
         "asap_same_unit_work_keys", "atepp_same_unit_composition_ids",
         "asap_overlapping_work_keys", "atepp_overlapping_composition_ids"),
        ("공통 작품에 속한 연주", "asap_total_performances", "atepp_total_performances",
         "asap_same_unit_performances", "atepp_same_unit_performances",
         "asap_performances_on_overlapping_works", "atepp_performances_on_overlapping_works"),
    ]
    for ax, (title, atotal, ttotal, astrict, tstrict, aexpanded, texpanded) in zip(axes, sets):
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=20)
        for y, label, total, strict, expanded in [(1, "ASAP", s[atotal], s[astrict], s[aexpanded]),
                                                  (0, "ATEPP", s[ttotal], s[tstrict], s[texpanded])]:
            p, extra = 100 * strict / total, 100 * (expanded - strict) / total
            ax.barh(y, 100, height=.34, color="#EDF1F5")
            ax.barh(y, p, height=.34, color="#2679C8")
            ax.barh(y, extra, left=p, height=.34, color="#E5A545")
            ax.text(0, y+.27, f"{strict:,} / {total:,}  ({p:.1f}%)", color="#20344F", fontsize=12, weight="bold")
            ax.text(100, y-.33, f"분할 항목 포함 시 {expanded:,} / {total:,}  ({100*expanded/total:.1f}%)",
                    ha="right", fontsize=9, color="#677589")
        ax.set(yticks=[1, 0], yticklabels=["ASAP", "ATEPP"], xlim=(0, 100), ylim=(-.6, 1.65),
               xticks=[0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color="#758094")
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.065, .12, "파랑: 같은 작품·악장 단위   ·   노랑: 한쪽에서 여러 악장을 묶은 2개 사례 추가", fontsize=10, color="#536176")
    fig.text(.065, .065, "동일 녹음·연주의 중복은 확인된 사례 0개. 공통 작품의 연주 수를 동일 연주 중복 수로 해석하지 않음.", fontsize=10, color="#536176")
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(OUT / f"overlap_summary.{ext}", dpi=220)
    plt.close(fig)
    clean_svg_outputs(OUT)


if __name__ == "__main__":
    main()
