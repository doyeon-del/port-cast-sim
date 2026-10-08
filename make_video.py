"""같은 바다·같은 운전자: 보조 없음 vs Port Cast 나란히 비교 영상 (흰 배경·굵은 선·큰 글씨).

실행: python make_video.py [seed=10] [op=novice]
결과: out/compare_<op>_seed<seed>.mp4, 미리보기 PNG 몇 장

화면 구성
  위: 제목 / 부제
  가운데: 크레인 장면 2개 (왼쪽 보조 없음, 오른쪽 Port Cast)
     - 트롤리 위 화살표 = 지금 트롤리가 움직이는 방향·세기 (Port Cast가 당기면 파란색 'AI 보정')
     - 컨테이너 뒤 주황 꼬리 = 최근 2초 흔들린 자취
     - 위험이 예측되면 컨테이너 옆에 경고
  아래: 계기 띠 (작업자 상태 · 위험 예측 · 조이스틱 진동 · 당김)
"""

import pickle
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Ellipse, FancyArrow, FancyBboxPatch, Polygon, Rectangle, Wedge

import crane_sim as cs

OUT = Path(__file__).parent / "out"
FONT_DIR = Path.home() / "AppData/Local/Microsoft/Windows/Fonts"
for f in FONT_DIR.glob("Pretendard-*.otf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Pretendard"
plt.rcParams["axes.unicode_minus"] = False

# 흰 배경 팔레트 (Canva 장표 톤)
INK = "#111418"
GRAY = "#5B6470"
LIGHTGRAY = "#E6EAF0"
STEEL = "#26324A"       # 크레인·선박 윤곽
BLUE = "#3F63D8"        # Port Cast·AI
ORANGE = "#F07C2A"      # 컨테이너·경고
FAIL = "#D9452B"
HULL = "#C8D4E8"
STACK = "#EEF2F8"
SEA = "#DCEBFA"
WAVE = "#7FB0E3"

SUBTITLE = {
    "novice": "가상 바다 · 같은 조건  ·  왼쪽 보조 없음, 오른쪽 Port Cast",
    "tired": "가상 바다 · 같은 조건  ·  왼쪽 보조 없음, 오른쪽 Port Cast",
}
HEAD = {"novice": "초보 작업자 착지 비교", "tired": "피곤한 작업자 착지 비교"}

FPS = 25
STEP = int(1 / FPS / cs.DT)   # 시뮬 2스텝 = 1프레임
# 재생 속도: 컨테이너를 옮기기만 하는 앞부분은 빠르게, 위험·착지 장면은 거의 실제 속도로 (영상 약 35초)
FAST_UNTIL, FAST, SLOW, HOLD = 20.0, 2.0, 1.25, 3.0
FRAME_IDX = []   # 프레임 k → 시뮬 스텝 번호 (main에서 채움)


def build_timeline(t_end):
    idx, t = [], 0.0
    while t <= t_end:
        idx.append(int(round(t / cs.DT)))
        t += (FAST if t < FAST_UNTIL else SLOW) / FPS
    last = int(round(t_end / cs.DT))   # 마지막 스텝까지 확실히 보여 줘야 긴 쪽 결과(착지 판정)가 뜬다
    idx.append(last)
    idx += [last] * int(HOLD * FPS)
    return idx
TRAIL = int(2.0 * FPS)


def level(beta):
    """팀 장표의 맞춤형 보조 단계: LOW 개입 최소화 / MID 경고·속도 안내 / HIGH 조작 보정·흔들림 보조."""
    return "LOW" if beta < 0.25 else ("MID" if beta < 0.5 else "HIGH")


BUSAN_TIME = "2026-02-25 02:00:00"   # 거친 날 중 초보·피곤한 작업자 모두 '보조 없음 거친 착지 → Port Cast 성공'인 시간대


def busan_case():
    """experiment_busan과 같은 시간대 선택·시드 규칙(5000+k)으로 부산 거친 날 바다를 재현."""
    import experiment_busan as eb
    hours = eb.pick_hours(eb.fetch_sea(), 60)
    k = int(hours.index[hours.time.astype(str) == BUSAN_TIME][0])
    row = hours.loc[k]
    sc = cs.Scenario.from_sea(5000 + k, row.wave_height_m, row.wave_period_s, row.wind_gust_ms)
    return 5000 + k, sc, row


def run_pair(seed, op, scenario=None):
    m = pickle.load(open(OUT / "models.pkl", "rb"))
    kw = dict(bc_model=m["bc"], risk_model=m.get("risk"), keep_log=True, scenario=scenario)
    ra = cs.run(op, "A", seed, skill=cs.SkillEstimator(m["skill_w"], m["skill_b"]), **kw)
    rc = cs.run(op, "C", seed, skill=cs.SkillEstimator(m["skill_w"], m["skill_b"]), **kw)
    return ra, rc


SKIN = "#F2D3B8"


class Operator:
    """계기 띠 안의 원격 조종석 작업자(조이스틱을 잡고 있음). y가 위로 증가하는 좌표.
    drowsy 0(또렷)~1(졸음), alert=True면 진동에 깬 순간."""

    def __init__(self, ax, cx, label="", state_ax=None, kind="novice"):
        self.kind = kind
        self.ax = ax
        self.cx = cx
        # 책상·몸통·팔
        ax.add_patch(Rectangle((cx - 9, 1.6), 19, 1.4, fc=STEEL, ec="none", zorder=3))
        ax.add_patch(FancyBboxPatch((cx - 6.2, 3.2), 7.4, 6.6, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=BLUE, ec="none", zorder=4))
        ax.plot([cx + 0.6, cx + 5.4], [7.6, 6.0], color=SKIN, lw=7, solid_capstyle="round", zorder=6)
        # 조이스틱 (손잡이는 Hud가 움직인다)
        self.jc = (cx + 6.2, 6.0)
        ax.plot([cx + 6.2, cx + 6.2], [3.0, 5.0], color=STEEL, lw=4, zorder=5)
        self.knob = Circle(self.jc, 1.25, fc=STEEL, ec="none", zorder=7)
        ax.add_patch(self.knob)
        self.rings = [Circle(self.jc, 2.2 + j * 1.3, fc="none", ec=ORANGE, lw=3.5, alpha=0, zorder=6) for j in range(2)]
        for rg in self.rings:
            ax.add_patch(rg)
        # 머리
        self.head = Circle((0, 0), 2.9, fc=SKIN, ec=STEEL, lw=2, zorder=8)
        self.helmet = Wedge((0, 0), 3.2, 0, 180, fc="#F5C518", ec=STEEL, lw=2, zorder=9)
        self.eyes = [Ellipse((0, 0), 0.8, 1.0, fc=STEEL, ec="none", zorder=10) for _ in range(2)]
        for p_ in [self.head, self.helmet] + self.eyes:
            ax.add_patch(p_)
        self.lids = [ax.plot([], [], color=STEEL, lw=3, zorder=10)[0] for _ in range(2)]
        self.zzz = ax.text(0, 0, "", fontsize=22, fontweight="bold", color="#7A8BA6", zorder=11)
        self.bang = ax.text(0, 0, "", fontsize=34, fontweight="bold", color=ORANGE, zorder=11)
        sa = state_ax if state_ax is not None else ax
        self.state = sa.text(14.5, 26.5, "", fontsize=22, fontweight="bold", ha="center", va="center", zorder=13,
                             color="white", bbox=dict(boxstyle="round,pad=0.45", fc=BLUE, ec="none"))
        if label:
            ax.text(cx - 9.5, 18.6, label, fontsize=12, color="white", ha="left", va="center", zorder=11,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=GRAY, ec="none"))

    def draw(self, t, drowsy, alert, done, ai_note=False):
        cx = self.cx
        nod = 0.0 if alert else drowsy * 2.0 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 2.6))
        hx, hy = cx - 2.5, 12.6 - nod
        self.head.center = (hx, hy)
        self.helmet.set_center((hx, hy + 0.35))
        openness = 1.0 if alert else max(0.0, 1 - 1.15 * drowsy)
        for j, (e, ld) in enumerate(zip(self.eyes, self.lids)):
            ex = hx - 1.1 + 2.2 * j
            e.center = (ex, hy - 0.4)
            e.height = 1.4 if alert else 1.0 * openness
            e.width = 1.0 if alert else 0.8
            e.set_visible(openness > 0.15 or alert)
            ld.set_data([ex - 0.55, ex + 0.55], [hy - 0.4, hy - 0.4])
            ld.set_visible(not alert and openness <= 0.15)
        sleepy = (not alert) and drowsy > 0.55 and not done
        self.zzz.set_text("Z z z" if sleepy else "")
        self.zzz.set_position((hx + 3.2, hy + 2.2 + 0.6 * np.sin(t * 3)))
        self.bang.set_text("!" if alert and not done else "")
        self.bang.set_position((hx + 3.4, hy + 1.0))
        self.state.set_visible(not done)
        if alert and not done:
            self.state.set_text("진동으로 깨움!" if self.kind == "tired" else "진동 경고!"); self.state.get_bbox_patch().set_facecolor(ORANGE)
        elif ai_note and not done:
            self.state.set_text("AI 졸음 감지 → 보조 HIGH"); self.state.get_bbox_patch().set_facecolor(BLUE)
        elif drowsy > 0.55:
            self.state.set_text("작업자 졸음"); self.state.get_bbox_patch().set_facecolor(FAIL)
        elif drowsy > 0.25:
            self.state.set_text("주의 저하"); self.state.get_bbox_patch().set_facecolor("#E8A23A")
        else:
            self.state.set_text("집중"); self.state.get_bbox_patch().set_facecolor("#5B6470")

class Scene:
    """크레인 장면 한 칸."""

    def __init__(self, fig, rect, res: cs.RunResult, assisted, op="novice"):
        self.r = res
        self.lg = {k: np.array(v) if k != "feat" else v for k, v in res.log.items()}
        self.assisted = assisted
        ax = fig.add_axes(rect)
        ax.set_xlim(-9, 46)
        ax.set_ylim(45.5, -4)
        ax.set_aspect("equal")
        ax.axis("off")
        self.ax = ax

        # 바다(채움) + 안벽
        self.sea = Polygon([[0, 0]], closed=True, fc=SEA, ec="none", zorder=0)
        ax.add_patch(self.sea)
        self.wave, = ax.plot([], [], color=WAVE, lw=2.5, zorder=1)
        ax.add_patch(Rectangle((-9, 40), 19, 6, fc=LIGHTGRAY, ec=STEEL, lw=2.5, zorder=1))

        # 크레인 골격 (굵게)
        ax.plot([-9, 45], [0, 0], color=STEEL, lw=4, solid_capstyle="round", zorder=3)
        ax.plot([-9, 45], [1.4, 1.4], color=STEEL, lw=4, solid_capstyle="round", zorder=3)
        for xx in np.arange(-9, 45, 3.0):
            ax.plot([xx, xx + 1.5, xx + 3.0], [1.4, 0, 1.4], color=STEEL, lw=1.6, zorder=3)
        for lx in (-2, 6):
            ax.plot([lx, lx], [1.4, 40], color=STEEL, lw=5, zorder=3)
        ax.plot([-2, 6], [12, 1.4], color=STEEL, lw=2, zorder=3)
        ax.plot([-2, 6], [1.4, 12], color=STEEL, lw=2, zorder=3)

        # 선박
        self.hull = Polygon([[0, 0]], closed=True, fc=HULL, ec=STEEL, lw=3, zorder=2)
        ax.add_patch(self.hull)
        self.stacks = [Rectangle((0, 0), 2.44, 2.6, fc=STACK, ec="#7A8BA6", lw=1.6, zorder=2) for _ in range(16)]
        for s in self.stacks:
            ax.add_patch(s)
        # 목표 칸 (셀 가이드)
        self.guide_l, = ax.plot([], [], color=BLUE, lw=4, solid_capstyle="butt", zorder=4)
        self.guide_r, = ax.plot([], [], color=BLUE, lw=4, solid_capstyle="butt", zorder=4)
        self.slot = Rectangle((0, 0), 2.44, 0.35, fc=BLUE, ec="none", alpha=0.35, zorder=4)
        ax.add_patch(self.slot)
        self.target_lab = ax.text(0, 0, "← 목표 칸", fontsize=15, color=BLUE, fontweight="bold", ha="left", va="center", zorder=6)

        # 트롤리·와이어·컨테이너
        self.trolley = Rectangle((0, -1.4), 4.0, 1.4, fc=STEEL, ec="none", zorder=6)
        ax.add_patch(self.trolley)
        self.wire1, = ax.plot([], [], color=STEEL, lw=2.4, zorder=5)
        self.wire2, = ax.plot([], [], color=STEEL, lw=2.4, zorder=5)
        self.trail, = ax.plot([], [], color=ORANGE, lw=3, alpha=0.35, zorder=5, solid_capstyle="round")
        self.box = Rectangle((0, 0), 2.44, 2.6, fc=ORANGE, ec="#A9500F", lw=2.5, zorder=7)
        ax.add_patch(self.box)
        self.move_arrow = None
        self.move_lab = ax.text(0, 0, "", fontsize=16, fontweight="bold", color=BLUE, ha="center", zorder=9)
        self.warn = ax.text(0, 0, "", fontsize=17, fontweight="bold", color="white", ha="left", va="center", zorder=10,
                            bbox=dict(boxstyle="round,pad=0.35", fc=ORANGE, ec="none"))
        self.warn.set_visible(False)

        # 결과
        self.result = ax.text(18.5, 9.5, "", fontsize=46, fontweight="bold", ha="center", va="center", zorder=12)
        self.result_sub = ax.text(18.5, 16.0, "", fontsize=21, ha="center", va="center", color=INK, linespacing=1.4, zorder=12)
        self.hist = deque(maxlen=TRAIL)
        self.op = op
        self.alert_until = -1.0
        self.prev_vib = False

    def draw(self, k):
        lg = self.lg
        n = len(lg["t"])
        i = min(FRAME_IDX[k], n - 1)
        done = FRAME_IDX[k] >= n - 1
        t = lg["t"][i]
        x, L, th = lg["x"][i], lg["L"][i], lg["th"][i]
        px, pd = lg["px"][i], lg["pdepth"][i]
        deck = lg["deck"][i]
        tx = lg["tx"][i]
        heave = cs.DECK_DEPTH - deck
        sway = tx - cs.X_TARGET

        # 바다
        xs = np.linspace(-9, 46, 160)
        wy = 41.2 - heave * 0.6 + 0.4 * np.sin(xs * 0.45 + t * 1.3)
        self.wave.set_data(xs, wy)
        self.sea.set_xy(np.r_[np.c_[xs, wy], [[46, 46], [-9, 46]]])

        # 선박·스택
        top = deck + 5.2
        self.hull.set_xy([[16.5 + sway, top], [46, top], [46, 45.5 - heave], [19 + sway, 45.5 - heave]])
        k2 = 0
        for cx in [20.5, 23.0, 25.5, 28.0, 33.4, 35.9, 38.4, 40.9]:
            for lvl in range(2):
                self.stacks[k2].set_xy((cx + sway, top - 2.6 * (lvl + 1)))
                k2 += 1
        gl, gr = tx - 1.22 - 0.2, tx + 1.22 + 0.2
        self.guide_l.set_data([gl, gl], [deck - cs.GUIDE_H, top])
        self.guide_r.set_data([gr, gr], [deck - cs.GUIDE_H, top])
        self.slot.set_xy((tx - 1.22, deck - 0.35))
        self.stacks[-1].set_xy((tx - 1.22, deck))
        self.stacks[-2].set_xy((tx - 1.22, deck + 2.6))
        self.target_lab.set_position((tx + 1.9, deck - 1.4))

        # 크레인
        self.trolley.set_x(x - 2.0)
        self.wire1.set_data([x - 0.9, px - 0.9], [0.0, pd - 2.6])
        self.wire2.set_data([x + 0.9, px + 0.9], [0.0, pd - 2.6])
        self.box.set_xy((px - 1.22, pd - 2.6))
        self.hist.append((px, pd - 1.3))
        if len(self.hist) > 1:
            h = np.array(self.hist)
            self.trail.set_data(h[:, 0], h[:, 1])

        # 트롤리 이동 화살표 (조종 → 크레인)
        if self.move_arrow is not None:
            self.move_arrow.remove()
            self.move_arrow = None
        v = (lg["x"][min(i + STEP, n - 1)] - x) / (STEP * cs.DT) if not done else 0.0
        pull = lg["pull"][i] if self.assisted and not done else 0.0
        ai = abs(pull) > 0.04
        col = BLUE if ai else GRAY
        if abs(v) > 0.05:
            ln = float(np.clip(v, -3, 3)) * 2.2
            self.move_arrow = FancyArrow(x, -2.6, ln, 0, width=0.45, head_width=1.5, head_length=1.1, color=col,
                                         zorder=9, length_includes_head=True)
            self.ax.add_patch(self.move_arrow)
        self.move_lab.set_text("AI 보정" if ai else "")
        self.move_lab.set_position((x, -3.6))

        # 위험 경고 (Port Cast)
        vib = bool(lg["vib"][i]) and self.assisted and not done
        if vib:
            self.warn.set_text("충돌 위험 예측 → 진동")
            self.warn.set_position((px + 2.2, pd - 1.3))
        self.warn.set_visible(vib)

        if done:
            r = self.r
            ok = r.success
            self.result.set_text("정상 착지" if ok else ("거친 착지" if r.impact >= cs.IMPACT_TOL else "가이드 충돌"))
            self.result.set_color(BLUE if ok else FAIL)
            self.result_sub.set_text(f"착지 속도 {r.impact:.2f} m/s" + ("" if ok else "\n점검 필요"))
            for tt in (self.result, self.result_sub):
                tt.set_bbox(dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))
        # 작업자 캐릭터: 실제 피로(지연 증가)를 졸음으로, Port Cast 진동은 깨움으로
        if self.op == "tired":
            drowsy = float(np.clip((t - 6.0) / 22.0, 0, 1))
        else:
            drowsy = 0.0
        vib_now = bool(lg["vib"][i]) and self.assisted and not done
        if vib_now and not self.prev_vib:
            self.alert_until = t + 2.5
        self.prev_vib = vib_now
        alert = self.assisted and t < self.alert_until and not done
        ai_note = self.assisted and drowsy > 0.25 and lg["beta"][i] >= 0.5
        self.cur = (drowsy, alert, ai_note)
        return t, done


class Hud:
    """장면 아래 계기 띠."""

    def __init__(self, fig, rect, scene: Scene):
        self.s = scene
        ax = fig.add_axes(rect)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 20)
        ax.axis("off")
        self.ax = ax
        ax.add_patch(FancyBboxPatch((0.5, 0.5), 99, 19, boxstyle="round,pad=0,rounding_size=2.5", fc="#F3F5F8", ec="none"))
        cx = 61 if scene.assisted else 16
        self.op = Operator(ax, cx, state_ax=scene.ax, kind=scene.op)
        self.jc, self.knob, self.rings = self.op.jc, self.op.knob, self.op.rings
        if not scene.assisted:
            ax.text(34, 12.5, "보조 없음", fontsize=22, fontweight="bold", color=INK, va="center")
            ax.text(34, 6.5, "운전자 혼자 흔들림·선박 움직임을 판단", fontsize=17, color=GRAY, va="center")
            return
        # 03 상태
        ax.text(3, 16, "03 작업자 상태", fontsize=15, color=INK, fontweight="bold", va="center")
        ax.add_patch(Rectangle((3, 9.5), 22, 3.2, fc="white", ec=STEEL, lw=1.5))
        self.g_skill = Rectangle((3, 9.5), 0, 3.2, fc=STEEL, ec="none")
        ax.add_patch(self.g_skill)
        self.skill_txt = ax.text(3, 5.2, "", fontsize=17, color=GRAY, fontweight="bold", va="center")
        # 02 위험
        ax.text(29, 16, "02 2.5초 뒤 위험", fontsize=15, color=INK, fontweight="bold", va="center")
        ax.add_patch(Rectangle((29, 9.5), 22, 3.2, fc="white", ec=STEEL, lw=1.5))
        self.g_risk = Rectangle((29, 9.5), 0, 3.2, fc=ORANGE, ec="none")
        ax.add_patch(self.g_risk)
        self.risk_txt = ax.text(29, 5.2, "", fontsize=15, color=GRAY, va="center")
        # 01 당김
        ax.text(72, 16, "01 숙련자 방향 당김", fontsize=15, color=INK, fontweight="bold", va="center")
        self.pull = None
        self.pull_txt = ax.text(72, 5.2, "", fontsize=15, color=BLUE, fontweight="bold", va="center")

    def draw(self, k, done):
        s = self.s
        lg = s.lg
        n = len(lg["t"])
        i = min(FRAME_IDX[k], n - 1)
        ut, uh = lg["u_eff_t"][i], lg["u_eff_h"][i]
        self.knob.center = (self.jc[0] + 1.3 * ut, self.jc[1] - 0.8 * uh)
        vib = bool(lg["vib"][i]) and s.assisted and not done
        ph = (k % 6) / 6
        for j, rg in enumerate(self.rings):
            rg.set_alpha((1 - ph) * 0.9 if vib else 0)
            rg.set_radius(2.2 + j * 1.3 + ph * 1.2)
        self.op.draw(lg["t"][i], s.cur[0], s.cur[1], done, s.cur[2])
        if not s.assisted:
            return
        sk = lg["skill"][i]
        self.g_skill.set_width(22 * sk)
        lv = level(lg["beta"][i])
        self.skill_txt.set_text(f"보조 단계 {lv}")
        self.skill_txt.set_color({"LOW": GRAY, "MID": "#E8A23A", "HIGH": FAIL}[lv])
        rp = min(1.0, lg["risk_pred"][i]) if not done else 0.0
        self.g_risk.set_width(22 * rp)
        self.risk_txt.set_text("위험 높음" if rp > 0.5 else ("주의" if rp > 0.25 else "안전"))
        self.risk_txt.set_color(ORANGE if rp > 0.25 else GRAY)
        if self.pull is not None:
            self.pull.remove()
            self.pull = None
        p = lg["pull"][i] if not done else 0.0
        if abs(p) > 0.04:
            pc = float(np.clip(p, -1, 1))
            self.pull = FancyArrow(84 - pc * 9, 10.5, pc * 18, 0, width=1.2, head_width=3.6, head_length=2.6, color=BLUE,
                                   zorder=6, length_includes_head=True)
            self.ax.add_patch(self.pull)
            self.pull_txt.set_text("당기는 중")
        else:
            self.pull_txt.set_text("")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    op = sys.argv[2] if len(sys.argv) > 2 else "novice"
    busan = len(sys.argv) > 3 and sys.argv[3] == "busan"
    sea = None
    if busan:
        seed, sea, row = busan_case()
    ra, rc = run_pair(seed, op, sea)
    print("A:", ra.success, round(ra.cycle_time, 1), round(ra.impact, 2), "| C:", rc.success, round(rc.cycle_time, 1),
          round(rc.impact, 2))

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.025, 0.935, HEAD.get(op, "착지 비교"), color=INK, fontsize=40, fontweight="bold", va="center")
    sub = SUBTITLE.get(op, SUBTITLE["novice"])
    foot = "Port Cast PoC · 가상 운전자 시뮬레이션 · 2026 FLY ASIA 해커톤"
    if busan:
        sub = (f"부산신항 거친 날 (파고 {row.wave_height_m:.2f} m · 돌풍 {row.wind_gust_ms:.1f} m/s)"
               "  ·  왼쪽 보조 없음, 오른쪽 Port Cast")
        foot += f" · 바다: 부산신항 앞바다 {BUSAN_TIME[:13]}시 Open-Meteo 재분석(CC BY 4.0), 실측 아님"
    fig.text(0.025, 0.875, sub, color=GRAY, fontsize=20, va="center")
    fig.text(0.025, 0.012, foot, color="#9AA3AF", fontsize=12)
    # 패널 이름 칩
    for x0, lab, fc, fg in [(0.025, "보조 없음", LIGHTGRAY, INK), (0.515, "Port Cast", BLUE, "white")]:
        fig.text(x0, 0.81, f"  {lab}  ", fontsize=26, fontweight="bold", color=fg, va="center",
                 bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec="none"))
    clocks = [fig.text(0.475, 0.81, "", fontsize=20, color=GRAY, ha="right", va="center"),
              fig.text(0.975, 0.81, "", fontsize=20, color=GRAY, ha="right", va="center")]
    sa = Scene(fig, [0.015, 0.215, 0.475, 0.565], ra, False, op)
    sc = Scene(fig, [0.51, 0.215, 0.475, 0.565], rc, True, op)
    ha = Hud(fig, [0.02, 0.035, 0.465, 0.17], sa)
    hc = Hud(fig, [0.515, 0.035, 0.465, 0.17], sc)

    FRAME_IDX[:] = build_timeline(max(ra.log["t"][-1], rc.log["t"][-1]))
    n_frames = len(FRAME_IDX)
    speed_txt = fig.text(0.975, 0.935, "", fontsize=24, color=BLUE, fontweight="bold", ha="right", va="center")
    path = OUT / (f"compare_{op}_busan_rough.mp4" if busan else f"compare_{op}_seed{seed}.mp4")
    writer = FFMpegWriter(fps=FPS, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    snaps = {int(n_frames * q) for q in (0.3, 0.6, 0.99)}
    with writer.saving(fig, str(path), dpi=100):
        for k in range(n_frames):
            for sc_, hd, ck in ((sa, ha, clocks[0]), (sc, hc, clocks[1])):
                t, done = sc_.draw(k)
                hd.draw(k, done)
                ck.set_text(f"{t:4.1f}초")
            speed_txt.set_text(">> 2배속" if FRAME_IDX[k] * cs.DT < FAST_UNTIL else "")
            writer.grab_frame()
            if k in snaps:
                fig.savefig(OUT / f"frame_{op}_{seed}_{k:04d}.png", dpi=60)
    print("saved", path, n_frames, "frames")


if __name__ == "__main__":
    main()
