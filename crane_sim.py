"""STS 크레인 선박 측 착지 시뮬레이터 (2D 옆모습).

- 트롤리가 레일을 따라 움직이고, 와이어 끝의 컨테이너가 진자처럼 흔들린다.
- 선박은 상하(heave)·좌우(sway)로 출렁이고, 바람이 컨테이너를 흔든다.
- 가상 운전자(초보/숙련/피곤한 작업자)가 조이스틱 값을 낸다.
- 보조 모드
    A: 보조 없음
    B: 현재 위험만 진동으로 알림 (기존 연구 방식)
    C: 위험 예측 + 숙련자 모방 모델의 당김·저항 + 작업자 상태에 따른 강도 조절 (우리 방식)

좌표: x는 바다 쪽으로 +, 깊이(depth)는 트롤리 레일에서 아래로 +.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

G = 9.81
DT = 0.02
T_MAX = 90.0

V_TROLLEY_MAX = 3.0      # m/s
A_TROLLEY_MAX = 1.0      # m/s^2
V_HOIST_MAX = 2.0        # m/s (+ 내림)
A_HOIST_MAX = 1.5
L_START = 8.0
X_START = 0.0
X_TARGET = 32.0
DECK_DEPTH = 32.0        # 적재 위치(컨테이너 놓일 면)의 기준 깊이
GUIDE_H = 2.6            # 셀 가이드 높이: 이 높이부터는 정렬이 맞아야 들어감
GUIDE_TOL = 0.35         # 가이드에 닿지 않는 좌우 허용 오차 (m)
LAND_TOL = 0.15          # 정상 착지 좌우 허용 오차 (m)
IMPACT_TOL = 0.6         # 정상 착지 상대 낙하 속도 (m/s)


# --------------------------------------------------------------------------
# 환경
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    seed: int
    heave_amp: float
    heave_period: float
    sway_amp: float
    sway_period: float
    phase_h: float
    phase_s: float
    gust: float  # 바람 세기 (각가속도 표준편차 스케일)

    @staticmethod
    def make(seed: int) -> "Scenario":
        r = np.random.default_rng(seed)
        return Scenario(
            seed=seed,
            heave_amp=r.uniform(0.1, 0.3),
            heave_period=r.uniform(7.0, 10.0),
            sway_amp=r.uniform(0.05, 0.18),
            sway_period=r.uniform(9.0, 13.0),
            phase_h=r.uniform(0, 2 * np.pi),
            phase_s=r.uniform(0, 2 * np.pi),
            gust=r.uniform(0.05, 0.15),
        )

    @staticmethod
    def from_sea(seed: int, hs: float, tp: float, gust_ms: float) -> "Scenario":
        """실제 해상 자료(유의파고 hs m, 파주기 tp s, 돌풍 m/s) → 선석에 계류된 선박 움직임.

        가정 (부산신항은 방파제 안쪽 선석):
          - 상하 진폭 = 0.5 × hs × 항내 감쇠 0.6  (0.08~0.45 m로 제한)
          - 상하 주기 = max(7초, 1.8 × tp)        (대형선은 파주기보다 길게 반응)
          - 좌우 진폭 = 0.45 × 상하 진폭 + 0.006 × 돌풍
          - 바람 교란 세기 = 0.0125 × 돌풍       (기본 시나리오 범위 0.05~0.15 ↔ 돌풍 4~12 m/s)
        위상은 seed로 무작위.
        """
        r = np.random.default_rng(seed)
        heave = float(np.clip(0.5 * hs * 0.6, 0.08, 0.45))
        return Scenario(
            seed=seed,
            heave_amp=heave,
            heave_period=float(max(7.0, 1.8 * tp)),
            sway_amp=float(np.clip(0.45 * heave + 0.006 * gust_ms, 0.03, 0.3)),
            sway_period=float(r.uniform(9.0, 13.0)),
            phase_h=float(r.uniform(0, 2 * np.pi)),
            phase_s=float(r.uniform(0, 2 * np.pi)),
            gust=float(np.clip(0.0125 * gust_ms, 0.02, 0.3)),
        )

    def deck_depth(self, t: float) -> float:
        return DECK_DEPTH - self.heave_amp * np.sin(2 * np.pi * t / self.heave_period + self.phase_h)

    def deck_vel(self, t: float) -> float:
        """적재면 깊이 변화율 (+면 내려감)."""
        w = 2 * np.pi / self.heave_period
        return -self.heave_amp * w * np.cos(w * t + self.phase_h)

    def target_vel(self, t: float) -> float:
        w = 2 * np.pi / self.sway_period
        return self.sway_amp * w * np.cos(w * t + self.phase_s)

    def target_x(self, t: float) -> float:
        return X_TARGET + self.sway_amp * np.sin(2 * np.pi * t / self.sway_period + self.phase_s)


@dataclass
class State:
    t: float = 0.0
    x: float = X_START       # 트롤리 위치
    v: float = 0.0           # 트롤리 속도
    L: float = L_START       # 와이어 길이
    Ld: float = 0.0          # 와이어 길이 변화율 (+ 내림)
    th: float = 0.0          # 흔들림 각 (rad, +면 컨테이너가 바다 쪽)
    thd: float = 0.0
    wind: float = 0.0

    def copy(self) -> "State":
        return State(**self.__dict__)

    @property
    def px(self) -> float:
        return self.x + self.L * np.sin(self.th)

    @property
    def pdepth(self) -> float:
        return self.L * np.cos(self.th)


def step(s: State, u_t: float, u_h: float, sc: Scenario | None, rng=None) -> State:
    """조이스틱 (u_t, u_h) ∈ [-1, 1] 로 한 스텝 진행."""
    u_t = float(np.clip(u_t, -1, 1))
    u_h = float(np.clip(u_h, -1, 1))
    a = np.clip((u_t * V_TROLLEY_MAX - s.v) / 0.4, -A_TROLLEY_MAX, A_TROLLEY_MAX)
    ah = np.clip((u_h * V_HOIST_MAX - s.Ld) / 0.3, -A_HOIST_MAX, A_HOIST_MAX)

    if rng is not None and sc is not None:
        # 바람: OU 과정
        s.wind += (-s.wind / 1.5) * DT + sc.gust * np.sqrt(DT) * rng.normal()
    w = s.wind if sc is not None else 0.0

    thdd = (-G * np.sin(s.th) - a * np.cos(s.th) - 2 * s.Ld * s.thd + w) / s.L - 0.2 * s.thd   # 0.2: 크레인 기본 흔들림 억제 장치

    n = s.copy()
    n.v = s.v + a * DT
    n.x = s.x + n.v * DT
    n.Ld = s.Ld + ah * DT
    n.L = max(3.0, s.L + n.Ld * DT)
    n.thd = s.thd + thdd * DT
    n.th = s.th + n.thd * DT
    n.t = s.t + DT
    return n


# --------------------------------------------------------------------------
# 관측 / 특징
# --------------------------------------------------------------------------
def features(s: State, sc: Scenario) -> np.ndarray:
    """모방 모델·운전자가 보는 정보."""
    x_err = sc.target_x(s.t) - s.px
    gap = sc.deck_depth(s.t) - s.pdepth          # 적재면까지 남은 높이
    return np.array([x_err, s.v, s.th, s.thd, s.L, gap, s.Ld, sc.deck_vel(s.t), sc.target_vel(s.t)])


# --------------------------------------------------------------------------
# 가상 운전자
# --------------------------------------------------------------------------
@dataclass
class OperatorProfile:
    name: str
    delay: float          # 반응 지연 (s)
    k_pos: float          # 목표로 가는 공격성
    k_sway: float         # 흔들림 잡는 능력 (0이면 못 잡음)
    k_vel: float          # 속도를 미리 줄이는 정도
    noise: float          # 손 떨림·과조작
    align_tol: float      # 내리기 시작하는 정렬 기준 (m)
    sway_tol: float       # 내리기 시작하는 흔들림 기준 (rad)
    near_speed: float     # 가이드 근처 내림 속도
    land_timing: bool     # 선박 상하운동 타이밍을 보는가
    fatigue_rate: float = 0.0   # 시간에 따라 지연이 늘어나는 정도 (s/s)


PROFILES = {
    "novice": OperatorProfile("초보", delay=0.55, k_pos=0.6, k_vel=0.2, k_sway=0.2, noise=0.11,
                              align_tol=0.9, sway_tol=1.5, near_speed=0.32, land_timing=False),
    "expert": OperatorProfile("숙련", delay=0.18, k_pos=0.3, k_vel=1.0, k_sway=1.0, noise=0.04,
                              align_tol=0.28, sway_tol=0.3, near_speed=0.22, land_timing=True),
    "tired": OperatorProfile("피곤한 작업자", delay=0.35, k_pos=0.35, k_vel=0.8, k_sway=0.6, noise=0.09,
                             align_tol=0.35, sway_tol=0.5, near_speed=0.3, land_timing=True,
                             fatigue_rate=0.006),
}


class Operator:
    """지연된 관측을 보고 조이스틱을 내는 가상 운전자."""

    def __init__(self, prof: OperatorProfile, rng: np.random.Generator):
        self.p = prof
        self.rng = rng
        self.hist: deque = deque(maxlen=200)
        self.noise_t = 0.0
        self.noise_h = 0.0
        self.caution_until = -1.0
        self.alert_seen_at: float | None = None

    def delay_now(self, t: float) -> float:
        return self.p.delay + self.p.fatigue_rate * t

    def observe(self, feat: np.ndarray):
        self.hist.append(feat)

    def act(self, t: float, alert: bool) -> tuple[float, float]:
        p = self.p
        k = int(self.delay_now(t) / DT)
        f = self.hist[max(0, len(self.hist) - 1 - k)]
        x_err, v, th, thd, L, gap, Ld, dv, tv = f

        # 진동 경고에 대한 반응 (자기 반응 지연 후)
        if alert:
            if self.alert_seen_at is None:
                self.alert_seen_at = t
            if t - self.alert_seen_at >= self.delay_now(t):
                self.caution_until = t + 0.8
        else:
            self.alert_seen_at = None
        cautious = t < self.caution_until

        # 트롤리: 목표로 + 흔들림 잡기
        trolley_err = x_err + L * np.sin(th)          # 목표 - 트롤리 위치
        if p.k_vel > 0.5:   # 숙련: 남은 거리에 맞춰 미리 감속
            e = abs(trolley_err)
            v_des = np.sign(trolley_err) * min(V_TROLLEY_MAX, np.sqrt(2 * 0.6 * e), 1.0 * e) + tv  # 선박 움직임까지 따라감
        else:               # 초보: 거리 비례로만 밀다가 늦게 브레이크
            v_des = np.clip(p.k_pos * trolley_err, -V_TROLLEY_MAX, V_TROLLEY_MAX)
        u_t = (v_des + p.k_sway * (1.5 * th * L + 0.6 * thd * L)) / V_TROLLEY_MAX
        # 호이스트
        in_guide = gap < GUIDE_H + 0.3
        aligned = abs(x_err) < p.align_tol and abs(L * th) < p.sway_tol
        if gap > GUIDE_H + 4:
            u_h = 0.9 if abs(x_err) < 6 else 0.0
            if not aligned and abs(x_err) < 3:
                u_h = 0.4
        elif not in_guide:
            u_h = 0.55 if aligned else (0.2 if abs(x_err) < 2 * p.align_tol else 0.0)
        else:
            u_h = p.near_speed if aligned else (0.0 if p.align_tol < 0.5 else p.near_speed * 0.6)
            if p.land_timing and gap < 0.8 and dv < -0.15:
                u_h = 0.05   # 갑판이 올라오는 중이면 기다림
        if cautious:
            u_h = min(u_h, 0.0)   # 진동을 느끼면 일단 내리기를 멈춘다

        # 손 떨림 (저역 필터된 잡음)
        self.noise_t += (-self.noise_t / 0.3) * DT + p.noise * np.sqrt(DT) * 2.2 * self.rng.normal()
        self.noise_h += (-self.noise_h / 0.3) * DT + p.noise * 0.5 * np.sqrt(DT) * 2.2 * self.rng.normal()
        u_h_out = 0.0 if cautious else float(np.clip(u_h + self.noise_h, -1, 1))
        return float(np.clip(u_t + self.noise_t, -1, 1)), u_h_out


def expert_label(feat: np.ndarray) -> tuple[float, float]:
    """숙련자라면 이 상황에서 낼 조작 (지연·떨림 없는 숙련자 정책). 모방 학습 라벨용."""
    from dataclasses import replace
    op = Operator(replace(PROFILES["expert"], noise=0.0, delay=0.0), np.random.default_rng(0))
    op.observe(feat)
    return op.act(0.0, False)


# --------------------------------------------------------------------------
# AI: 위험 예측 / 작업자 상태 / 모방 모델
# --------------------------------------------------------------------------
def predict_risk(s: State, sc: Scenario, u_t: float, u_h: float, horizon: float = 2.5) -> float:
    """현재 조작을 유지하면 horizon초 안에 위험해지는지 (0~1). 바람 없는 모델로 앞을 굴려 봄."""
    r = s.copy()
    r.wind = 0.0
    worst = instant_risk(r, sc)
    for _ in range(int(horizon / DT)):
        r = step(r, u_t, u_h, None)
        worst = max(worst, instant_risk(r, sc))
        if sc.deck_depth(r.t) - r.pdepth <= 0:
            break
    return worst


def instant_risk(s: State, sc: Scenario) -> float:
    """지금 이 순간의 위험도 (0~1): 가이드 진입 직전 정렬 불량 / 거친 착지 / 큰 흔들림."""
    x_err = abs(sc.target_x(s.t) - s.px)
    gap = sc.deck_depth(s.t) - s.pdepth
    r = 0.0
    if GUIDE_H - 0.2 < gap < GUIDE_H + 1.5:
        r = max(r, np.clip((x_err - LAND_TOL) / (GUIDE_TOL - LAND_TOL), 0, 1))
    if gap < 1.0:
        closing = s.Ld * np.cos(s.th) - sc.deck_vel(s.t)
        r = max(r, np.clip((closing - 0.3) / (IMPACT_TOL - 0.3), 0, 1))
    if GUIDE_H < gap < GUIDE_H + 1.5:
        r = max(r, np.clip((abs(s.L * np.sin(s.th)) - 0.2) / 0.3, 0, 1))
    return float(r)


class SkillEstimator:
    """최근 4초 조작 로그로 숙련도(0~1)를 추정. 가중치는 experiment.py에서 학습해 넣는다."""

    WIN = int(4.0 / DT)

    def __init__(self, w: np.ndarray | None = None, b: float = 0.0):
        self.w = w if w is not None else np.array([-6.0, -25.0, -3.0])
        self.b = b
        self.u = deque(maxlen=self.WIN)
        self.th = deque(maxlen=self.WIN)
        self.score = 0.5

    @staticmethod
    def feats(us: np.ndarray, ths: np.ndarray) -> np.ndarray:
        sign = np.sign(np.where(np.abs(us) < 0.08, 0, us))
        nz = sign[sign != 0]
        reversals = np.sum(nz[1:] != nz[:-1]) / (len(us) * DT) if len(nz) > 1 else 0.0
        jerk = np.sqrt(np.mean(np.diff(us) ** 2)) / DT if len(us) > 1 else 0.0
        return np.array([reversals, np.mean(np.abs(ths)), jerk / 10.0])

    def update(self, u_t: float, th: float) -> float:
        self.u.append(u_t)
        self.th.append(th)
        if len(self.u) >= 25:
            f = self.feats(np.array(self.u), np.array(self.th))
            p = 1 / (1 + np.exp(-(f @ self.w + self.b)))
            self.score += (p - self.score) * 0.05   # 부드럽게
        return self.score


# --------------------------------------------------------------------------
# 한 번의 하역
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    op: str
    mode: str
    seed: int
    success: bool
    cycle_time: float
    land_err: float       # 가이드 진입 시 정렬 오차 (m)
    impact: float
    max_sway_deg: float   # 접근 구간 최대 흔들림 폭 (m)
    risk_entries: int
    guide_hits: int
    log: dict = field(default_factory=dict)


def run(op_key: str, mode: str, seed: int, bc_model=None, skill: SkillEstimator | None = None, risk_model=None,
        scenario: Scenario | None = None,
        keep_log: bool = False) -> RunResult:
    sc = scenario if scenario is not None else Scenario.make(seed)
    rng = np.random.default_rng(seed * 7 + {"novice": 1, "expert": 2, "tired": 3}[op_key])
    op = Operator(PROFILES[op_key], rng)
    s = State()
    est = skill if skill is not None else SkillEstimator()
    est.u.clear(); est.th.clear(); est.score = 0.5

    in_risk = False
    captured = False
    entry_err = float("nan")
    risk_entries = 0
    guide_hits = 0
    touching_guide = False
    max_sway = 0.0
    log = {k: [] for k in ["t", "x", "L", "th", "px", "pdepth", "deck", "tx", "u_op", "u_eff_t", "u_eff_h",
                           "risk_now", "risk_pred", "vib", "pull", "skill", "beta", "u_op_h"]} if keep_log else None

    alert = False
    u_op_t = u_op_h = 0.0
    risk_pred = 0.0
    u_prev_t = u_prev_h = 0.0
    vib_run = 0.0
    while s.t < T_MAX:
        feat = features(s, sc)
        op.observe(feat)
        u_op_t, u_op_h = op.act(s.t, alert)
        u_t, u_h = u_op_t, u_op_h
        sk = est.update(u_op_t, s.th)
        r_now = instant_risk(s, sc)
        vib = False
        pull = 0.0
        beta = 0.0

        if mode == "B":
            vib = r_now > 0.35
        elif mode == "C":
            if int(round(s.t / DT)) % 5 == 0:
                if risk_model is not None:   # 학습된 위험 예측 모델 (02 환경 보정)
                    x = np.concatenate([features(s, sc), [u_prev_t, u_prev_h]])[None, :]
                    risk_pred = float(np.clip(risk_model.predict(x)[0], 0, 1))
                else:                        # 물리 모델로 앞을 굴려 봄 (비교·검증용)
                    risk_pred = predict_risk(s, sc, u_prev_t, u_prev_h)
            assist_on = sk < 0.72                 # 숙련자로 읽히면 보조를 끈다
            vib = assist_on and risk_pred > 0.5 + 0.3 * sk
            vib_run = vib_run + DT if vib else 0.0
            if vib_run > 2.0:                     # 2초 넘게 울리면 2초 쉼 (경보 피로 방지)
                vib = False
                if vib_run > 4.0:
                    vib_run = 0.0
            beta = 0.05 + 0.65 * (1 - sk) ** 1.5  # 숙련도 낮을수록 강하게 당김
            if bc_model is not None:
                u_bc_t, u_bc_h = bc_model.predict(features(s, sc)[None, :])[0]
                pull = beta * (u_bc_t - u_op_t)
                u_t = u_op_t + pull
                u_h = u_op_h + beta * (u_bc_h - u_op_h)
            # 저항: 예측 위험이 크면 내리는 방향을 뻑뻑하게
            if assist_on and risk_pred > 0.3 and u_h > 0:
                u_h *= (1 - 0.8 * (1 - sk) * min(1.0, risk_pred))
        alert = vib
        u_prev_t, u_prev_h = u_t, u_h
        if keep_log:
            log.setdefault("feat", []).append(feat)

        s = step(s, u_t, u_h, sc, rng)

        x_err = sc.target_x(s.t) - s.px
        gap = sc.deck_depth(s.t) - s.pdepth
        if gap < GUIDE_H + 3 and not captured:
            max_sway = max(max_sway, abs(s.L * np.sin(s.th)))
        r_after = instant_risk(s, sc)
        if r_after > 0.5 and not in_risk:
            risk_entries += 1
        in_risk = r_after > 0.5

        # 셀 가이드: 정렬되어 들어오면 붙잡히고, 어긋나면 가이드 위에 걸린다
        if not captured and gap < GUIDE_H:
            if abs(x_err) <= GUIDE_TOL:
                captured = True
                entry_err = abs(x_err)
            else:
                if not touching_guide:
                    guide_hits += 1
                touching_guide = True
                s.L = (sc.deck_depth(s.t) - GUIDE_H) / max(np.cos(s.th), 0.5)
                s.Ld = min(s.Ld, 0.0)
                s.thd *= -0.3
                gap = GUIDE_H
        elif gap > GUIDE_H + 0.2:
            touching_guide = False
        if captured:
            s.th = float(np.arcsin(np.clip((sc.target_x(s.t) - s.x) / s.L, -0.3, 0.3)))
            s.thd = 0.0
            gap = sc.deck_depth(s.t) - s.pdepth

        if keep_log:
            for k, val in zip(log.keys(), [s.t, s.x, s.L, s.th, s.px, s.pdepth, sc.deck_depth(s.t), sc.target_x(s.t),
                                           u_op_t, u_t, u_h, r_now, risk_pred if mode == "C" else r_now,
                                           vib, pull, sk, beta, u_op_h]):
                log[k].append(val)

        if gap <= 0:
            closing = s.Ld * np.cos(s.th) - sc.deck_vel(s.t)
            ok = closing < IMPACT_TOL and guide_hits == 0
            return RunResult(op_key, mode, seed, ok, s.t, entry_err, closing, max_sway,
                             risk_entries, guide_hits, log or {})

    return RunResult(op_key, mode, seed, False, T_MAX, entry_err, float("nan"), max_sway,
                     risk_entries, guide_hits, log or {})
