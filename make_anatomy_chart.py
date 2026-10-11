"""착지 한 판 해부: 비교 영상과 같은 판(피곤한 작업자, 가상 바다 #2)을 시간축으로 펼친다.

위에서부터
  1. 적재면까지 남은 높이 — 보조 없음 vs Port Cast
  2. Port Cast가 본 위험도 — 2.5초 앞 예측 vs 지금 위험, 진동 구간
  3. 하강 조이스틱 — 작업자가 넣은 값 vs 크레인에 실제로 들어간 값 (저항)
  4. 작업자 상태 점수 — 낮을수록 보조가 세진다

실행: python make_anatomy_chart.py [seed=2] [op=tired]
결과: out/chart_landing_anatomy.png (+ _2x.png)
"""

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import crane_sim as cs

OUT = Path(__file__).parent / "out"
for d in [Path.home() / "AppData/Local/Microsoft/Windows/Fonts", Path.home() / "Library/Fonts"]:
    for f in d.glob("Pretendard-*.otf"):
        font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Pretendard"
plt.rcParams["axes.unicode_minus"] = False

PAPER, INK, GRAY, MUTED, LINE = "#F4F7FB", "#0E1B2C", "#5B6B80", "#8C9AAE", "#D5DEEA"
OCEAN = "#0E4FB8"      # Port Cast
BASE = "#7A8799"       # 보조 없음 / 작업자 원래 입력 (배경 역할)
NOW = "#D9631E"        # 지금 위험
VIB = "#DCE6F7"        # 진동 구간 음영

T0 = 28.0              # 앞부분은 컨테이너를 옮기는 구간이라 생략


def run_pair(seed, op):
    m = pickle.load(open(OUT / "models.pkl", "rb"))
    kw = dict(bc_model=m["bc"], risk_model=m["risk"], keep_log=True)
    ra = cs.run(op, "A", seed, skill=cs.SkillEstimator(m["skill_w"], m["skill_b"]), **kw)
    rc = cs.run(op, "C", seed, skill=cs.SkillEstimator(m["skill_w"], m["skill_b"]), **kw)
    return ra, rc


def arr(r, k):
    return np.asarray(r.log[k], dtype=float)


def spans(t, mask):
    """True 구간 [(시작, 끝)]."""
    out, start = [], None
    for ti, on in zip(t, mask):
        if on and start is None:
            start = ti
        elif not on and start is not None:
            out.append((start, ti)); start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def style(ax, ylabel):
    ax.set_facecolor(PAPER)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=GRAY, labelsize=11, length=0)
    ax.yaxis.grid(True, color=LINE, lw=0.8)
    ax.set_axisbelow(True)
    ax.text(0, 1.04, ylabel, transform=ax.transAxes, fontsize=13, color=INK, fontweight="semibold",
            ha="left", va="bottom")


def end_label(ax, x, y, text, color, dy=0, weight="semibold"):
    ax.text(x + 0.25, y + dy, text, color=color, fontsize=11.5, va="center", fontweight=weight)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    op = sys.argv[2] if len(sys.argv) > 2 else "tired"
    ra, rc = run_pair(seed, op)
    ta, tc = arr(ra, "t"), arr(rc, "t")
    gap_a = arr(ra, "deck") - arr(ra, "pdepth")
    gap_c = arr(rc, "deck") - arr(rc, "pdepth")
    r_pred, r_now = arr(rc, "risk_pred"), arr(rc, "risk_now")
    vib = arr(rc, "vib") > 0
    uh_op, uh_eff = arr(rc, "u_op_h"), arr(rc, "u_eff_h")
    sk = arr(rc, "skill") * 100
    t1 = max(ta[-1], tc[-1]) + 3.6

    # 앞서 본 시간: 정렬 구간에서 예측 위험이 0.5를 넘은 시각 vs 실제 위험이 0.5를 넘은 시각
    w = tc >= T0
    i_pred = np.argmax(w & (r_pred > 0.5))
    i_now = np.argmax(w & (r_now > 0.5))
    lead = tc[i_now] - tc[i_pred]

    fig, axes = plt.subplots(4, 1, figsize=(12.8, 11.2), dpi=125, sharex=True,
                             gridspec_kw={"height_ratios": [1.5, 1.05, 1.0, 0.85], "hspace": 0.42})
    fig.patch.set_facecolor(PAPER)
    fig.text(0.06, 0.957, "피곤한 작업자 착지 한 판 해부", fontsize=24, fontweight="bold", color=INK)
    fig.text(0.06, 0.933, f"비교 영상과 같은 판 · 가상 바다 #{seed} · 같은 바다에서 보조 없음과 Port Cast를 각각 돌림 · "
             f"{T0:.0f}초 이전은 컨테이너 이동 구간이라 생략", fontsize=12.5, color=GRAY)

    # 1. 남은 높이 ---------------------------------------------------------
    ax = axes[0]
    style(ax, "① 적재면까지 남은 높이 (m)")
    ax.axhline(cs.GUIDE_H, color=MUTED, lw=1)
    ax.text(T0 + 0.15, cs.GUIDE_H + 0.18, "셀 가이드 입구 2.6 m", color=GRAY, fontsize=10.5)
    ax.plot(ta, gap_a, color=BASE, lw=2)
    ax.plot(tc, gap_c, color=OCEAN, lw=2.4)
    hard = ra.impact >= cs.IMPACT_TOL
    ax.plot(ta[-1], 0, "o", ms=9, color=NOW if hard else BASE, mec=PAPER, mew=2, zorder=5)
    ax.plot(tc[-1], 0, "o", ms=9, color=OCEAN, mec=PAPER, mew=2, zorder=5)
    for t_end, r, name, color in [(ta[-1], ra, "보조 없음", BASE), (tc[-1], rc, "Port Cast", OCEAN)]:
        rough = r.impact >= cs.IMPACT_TOL
        ax.text(t_end, -0.75, name, color=color, fontsize=11.5, ha="center", va="top", fontweight="semibold")
        ax.text(t_end, -1.6, f"{t_end:.1f}초 착지 · {r.impact:.2f} m/s · {'거친 착지' if rough else '정상 착지'}",
                color=INK, fontsize=11, ha="center", va="top")
    ax.text(t1 - 0.1, 8.4, f"착지 속도 {cs.IMPACT_TOL} m/s 이상이면 거친 착지", color=GRAY, fontsize=10.5, ha="right")
    ax.set_ylim(-2.7, 9)
    ax.set_yticks([0, 2, 4, 6, 8])

    # 2. 위험도 ------------------------------------------------------------
    ax = axes[1]
    style(ax, "② Port Cast가 본 위험도 (0 안전 ~ 1 기준 초과)  ·  음영 = 조이스틱 진동")
    for a, b in spans(tc, vib):
        if b > T0:
            for x in axes[1:]:
                x.axvspan(max(a, T0), b, color=VIB, lw=0, zorder=0)
    ax.plot(tc, r_now, color=NOW, lw=1.8)
    ax.plot(tc, r_pred, color=OCEAN, lw=2.2)
    for x in (tc[i_pred], tc[i_now]):
        ax.plot([x, x], [-0.13, 0.5], color=INK, lw=0.8)
    ax.annotate("", xy=(tc[i_pred], -0.11), xytext=(tc[i_now], -0.11),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1, shrinkA=0, shrinkB=0))
    ax.text(tc[i_pred] - 0.2, -0.11, f"예측이 {lead:.1f}초 먼저 0.5를 넘음", ha="right", va="center", fontsize=10.5,
            color=INK)
    ax.text(T0 + 0.15, 0.12, "2.5초 앞 예측", color=OCEAN, fontsize=11.5, fontweight="semibold")
    ax.text(tc[i_now] + 2.3, 0.6, "지금 위험", color=NOW, fontsize=11.5, fontweight="semibold")
    later = (tc > 40) & vib
    if later.any():
        tv = tc[np.argmax(later)]
        ax.text(tv + 0.2, 0.93, "착지 직전 예측 경보 → 진동", color=OCEAN, fontsize=10.5, va="top")
    ax.set_ylim(-0.2, 1.05)
    ax.set_yticks([0, 0.5, 1])

    # 3. 하강 조이스틱 -----------------------------------------------------
    ax = axes[2]
    style(ax, "③ 하강 조이스틱 (0 정지 ~ 1 최대)")
    ax.fill_between(tc, uh_eff, np.maximum(uh_op, uh_eff), color=OCEAN, alpha=0.14, lw=0)
    ax.plot(tc, uh_op, color=BASE, lw=1.4)
    ax.plot(tc, uh_eff, color=OCEAN, lw=2.2)
    end_label(ax, tc[-1], uh_op[-1] + 0.12, "작업자가 넣은 값", BASE)
    end_label(ax, tc[-1], uh_eff[-1] - 0.12, "크레인에 들어간 값", OCEAN)
    m = (tc > 37) & (tc < 40)
    j = np.argmax(np.where(m, uh_op - uh_eff, -1))
    ax.annotate("당김·저항으로 줄인 만큼", xy=(tc[j], (uh_op[j] + uh_eff[j]) / 2), xytext=(tc[j] + 1.2, 0.8),
                fontsize=10.5, color=OCEAN, arrowprops=dict(arrowstyle="-", color=OCEAN, lw=0.8))
    stop = w & vib & (uh_op < 0.02)
    if stop.any():
        ts = tc[np.argmax(stop)]
        ax.text(ts + 0.1, 0.62, "진동을 느낀 작업자가\n하강을 멈춤", color=GRAY, fontsize=10, va="bottom")
    ax.set_ylim(-0.05, 1.0)
    ax.set_yticks([0, 0.5, 1])

    # 4. 작업자 상태 -------------------------------------------------------
    ax = axes[3]
    style(ax, "④ 작업자 상태 점수 (조작 패턴으로 추정, 100 = 숙련자처럼 안정)")
    ax.axhline(72, color=MUTED, lw=1)
    ax.text(T0 + 0.15, 76, "72 이상이면 보조를 끈다", color=GRAY, fontsize=10.5)
    ax.plot(tc, sk, color=OCEAN, lw=2.2)
    ax.text(t1 - 0.1, 38, "낮을수록\n당김·저항이 세진다", color=GRAY, fontsize=10.5, ha="right", va="center")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_xlabel("시간 (초)", color=GRAY, fontsize=12)
    ax.set_xlim(T0, t1)

    fig.text(0.06, 0.018, "가상 작업자 시뮬레이션 결과(실제 크레인 데이터 아님). 위험도 = 가이드 진입 정렬 오차·착지 속도·흔들림 중 가장 큰 값. "
             "②~④는 Port Cast 쪽 기록.", fontsize=10, color=GRAY)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.885, bottom=0.07)
    for name, dpi in [("chart_landing_anatomy", 125), ("chart_landing_anatomy_2x", 250)]:
        fig.savefig(OUT / f"{name}.png", facecolor=PAPER, dpi=dpi)
    print(f"A {ra.cycle_time:.2f}s {ra.impact:.3f} m/s | C {rc.cycle_time:.2f}s {rc.impact:.3f} m/s | "
          f"예측 {tc[i_pred]:.2f}s → 실제 {tc[i_now]:.2f}s (lead {lead:.2f}s)")


if __name__ == "__main__":
    main()
