"""out/results.csv → 발표용 차트 PNG (장표 템플릿 본문 톤: 밝은 배경 + Pretendard)."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

OUT = Path(__file__).parent / "out"
FONT_DIR = Path.home() / "AppData/Local/Microsoft/Windows/Fonts"
for f in FONT_DIR.glob("Pretendard-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Pretendard"
plt.rcParams["axes.unicode_minus"] = False

# 장표 템플릿 색 (deck/build_template.py)
PAPER, INK, GRAY, LINE = "#F4F7FB", "#0E1B2C", "#5B6B80", "#D5DEEA"
OCEAN, ORANGE = "#0E4FB8", "#FF8A3D"
MODE_COLOR = {"A": "#C3CCD8", "B": "#8C9AAE", "C": OCEAN}
MODE_NAME = {"A": "보조 없음", "B": "현재 상태 전달", "C": "Port Cast"}
OP_ORDER = ["novice", "tired", "expert"]
OP_NAME = {"novice": "초보", "tired": "피곤한 작업자", "expert": "숙련"}

# 재작업 가정: 거친 착지 → 점검 30초, 가이드 충돌 1회 → 재정렬 15초
PEN_HARD, PEN_HIT = 30.0, 15.0
DECK = os.environ.get("DECK") == "1"     # 장표용: 차트 제목 없이, 파일명 끝 _deck
SUF = "_deck" if DECK else ""


def load():
    d = pd.read_csv(OUT / "results.csv")
    d["hard"] = (d.impact >= 0.6) & (d.cycle_time < 90)
    d["eff_time"] = d.cycle_time + PEN_HARD * d.hard + PEN_HIT * d.guide_hits
    return d


def base(title, sub, ylabel):
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=125)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    if not DECK:
        fig.text(0.06, 0.92, title, fontsize=24, fontweight="bold", color=INK)
        fig.text(0.06, 0.875, sub, fontsize=13, color=GRAY)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=GRAY, labelsize=13, length=0)
    ax.yaxis.grid(True, color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=GRAY, fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86 if DECK else 0.78, bottom=0.12)
    return fig, ax


def grouped(ax, vals, fmt, ymax):
    x = np.arange(len(OP_ORDER))
    w = 0.26
    for i, m in enumerate("ABC"):
        v = [vals[(op, m)] for op in OP_ORDER]
        bars = ax.bar(x + (i - 1) * w, v, w - 0.02, color=MODE_COLOR[m], label=MODE_NAME[m], zorder=3)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.015, fmt(val), ha="center",
                    va="bottom", fontsize=12, color=INK if m == "C" else GRAY,
                    fontweight="bold" if m == "C" else "normal")
    ax.set_xticks(x, [OP_NAME[o] for o in OP_ORDER])
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=3, frameon=False, fontsize=12,
              labelcolor=INK, handlelength=1.2)


def save(fig, name):
    """장표용(PAPER 배경) + DECK일 때 Canva용 투명 배경(_canva) 한 장 더."""
    fig.savefig(OUT / f"{name}{SUF}.png", facecolor=PAPER)
    if DECK:
        fig.patch.set_alpha(0)
        for ax in fig.axes:
            ax.set_facecolor("none")
        fig.savefig(OUT / f"{name}_canva.png", transparent=True)


def main():
    d = load()
    n = d.groupby(["op", "mode"]).size().min()
    g = d.groupby(["op", "mode"])
    note = f"가상 운전자 시뮬레이션, 조건별 {n}회 · 같은 바다·바람 조건에서 비교"

    # 1) 성공률
    succ = (g.success.mean() * 100).to_dict()
    fig, ax = base(f"Port Cast로 초보의 착지 성공률이 {succ[('novice', 'A')]:.0f}% → {succ[('novice', 'C')]:.0f}%", note,
                   "정상 착지 비율 (%)")
    grouped(ax, succ, lambda v: f"{v:.0f}%", 100)
    save(fig, "chart_success")

    # 2) 위험 사건
    hard = (g.hard.mean() * 100).to_dict()
    fig, ax = base(f"거친 착지가 사라졌다: 초보 {hard[('novice', 'A')]:.0f}% → {hard[('novice', 'C')]:.0f}%", note + " · 거친 착지 = 상대 낙하속도 0.6 m/s 이상",
                   "거친 착지 비율 (%)")
    grouped(ax, hard, lambda v: f"{v:.0f}%", 80)
    save(fig, "chart_hard_landing")

    # 3) 실효 사이클 시간
    eff = g.eff_time.mean().to_dict()
    dt = eff[("novice", "A")] - eff[("novice", "C")]
    fig, ax = base(f"재작업을 포함하면 초보는 보조 없음보다 {dt:.0f}초 {'빠르다' if dt > 0 else '느리다'}",
                   note + f" · 실효 시간 = 착지까지 시간 + 거친 착지 점검 {PEN_HARD:.0f}초 + 가이드 충돌당 재정렬 {PEN_HIT:.0f}초 (가정)",
                   "실효 사이클 시간 (초)")
    grouped(ax, eff, lambda v: f"{v:.0f}초", max(eff.values()) * 1.18)
    save(fig, "chart_effective_time")

    # 4) 초보-숙련 격차
    gap = {m: (succ[("expert", m)] - succ[("novice", m)]) for m in "ABC"}
    fig, ax = base("초보와 숙련자의 격차가 사라졌다", note + " · 숙련자 성공률 − 초보 성공률", "격차 (%p)")
    ms = list("ABC")
    bars = ax.bar(range(3), [gap[m] for m in ms], 0.5, color=[MODE_COLOR[m] for m in ms], zorder=3)
    for b, m in zip(bars, ms):
        val = gap[m]
        ax.text(b.get_x() + b.get_width() / 2, max(val, 0) + 1.5, f"{val:+.0f}%p", ha="center", va="bottom",
                fontsize=15, color=INK if m == "C" else GRAY, fontweight="bold" if m == "C" else "normal")
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.set_xticks(range(3), [MODE_NAME[m] for m in ms])
    ax.set_ylim(min(0, min(gap.values())) - 10, max(gap.values()) * 1.25)
    save(fig, "chart_skill_gap")

    # 요약 표
    summ = pd.DataFrame({"성공률(%)": g.success.mean() * 100, "거친 착지(%)": g.hard.mean() * 100,
                         "가이드 충돌(회)": g.guide_hits.mean(), "진입 정렬오차(m)": g.entry_err.mean(),
                         "착지까지(초)": g.cycle_time.mean(), "실효 시간(초)": g.eff_time.mean()}).round(2)
    summ.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    print(summ)


if __name__ == "__main__":
    main()
