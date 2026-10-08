"""작업자 상태 그래프: AI가 조이스틱 입력만 보고 추정한 숙련도와, 그에 따라 정한 보조 강도."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import crane_sim as cs
from make_charts import save, DECK, GRAY, INK, LINE, OCEAN, ORANGE, PAPER, OUT, SUF  # 폰트·색 설정 공유

SEED = 10
COLORS = {"novice": ORANGE, "tired": "#8C9AAE", "expert": OCEAN}


def main():
    m = pickle.load(open(OUT / "models.pkl", "rb"))
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), dpi=125, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35})
    fig.patch.set_facecolor(PAPER)
    if not DECK:
      fig.text(0.06, 0.93, "03 주의 변화 감지: 조작 패턴만으로 작업자 상태를 읽는다", fontsize=24, fontweight="bold", color=INK)
      fig.text(0.06, 0.885, f"같은 바다 조건(시드 {SEED}), Port Cast 모드 · 상태 점수는 최근 4초 조작(방향 전환 빈도·떨림·흔들림)으로 추정",
             fontsize=13, color=GRAY)
    for op in ["novice", "tired", "expert"]:
        r = cs.run(op, "C", SEED, bc_model=m["bc"], skill=cs.SkillEstimator(m["skill_w"], m["skill_b"]), risk_model=m.get("risk"),
                   keep_log=True)
        t = np.array(r.log["t"])
        sk = np.array(r.log["skill"]) * 100
        be = np.array(r.log["beta"]) * 100
        name = cs.PROFILES[op].name
        axes[0].plot(t, sk, color=COLORS[op], lw=2)
        axes[0].text(t[-1] + 0.6, sk[-1], name, color=COLORS[op], fontsize=12, va="center", fontweight="bold")
        axes[1].plot(t, be, color=COLORS[op], lw=2)
        axes[1].text(t[-1] + 0.6, be[-1], name, color=COLORS[op], fontsize=12, va="center", fontweight="bold")
    axes[0].axhline(72, color=GRAY, lw=1, ls=(0, (4, 4)))
    axes[0].text(48, 76, "이 위로 읽히면 보조를 끈다", color=GRAY, fontsize=11)
    axes[0].set_ylabel("작업자 상태 점수 (0~100)", color=GRAY, fontsize=12)
    axes[1].set_ylabel("보조 강도 (%)", color=GRAY, fontsize=12)
    axes[1].set_xlabel("시간 (초)", color=GRAY, fontsize=12)
    for ax in axes:
        ax.set_facecolor(PAPER)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.tick_params(colors=GRAY, labelsize=12, length=0)
        ax.yaxis.grid(True, color=LINE, lw=0.8)
        ax.set_ylim(0, 100)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.95 if DECK else 0.8, bottom=0.1)
    save(fig, "chart_operator_state")
    print("saved")


if __name__ == "__main__":
    main()
