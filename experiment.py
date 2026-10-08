"""1) 숙련자 시연으로 모방 모델 학습  2) 작업자 상태 추정기 보정  3) A/B/C 비교 실험.

실행: python experiment.py [반복 횟수, 기본 60]
결과: out/results.csv, out/models.pkl
"""

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import crane_sim as cs

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
TRAIN_SEEDS = range(1000, 1080)     # 학습용 (평가와 겹치지 않게)
EVAL_SEED0 = 0


def collect_demos(seeds=TRAIN_SEEDS):
    """숙련자 시연 + 초보·피로 운전자가 실제로 겪는 상황에 숙련자 라벨을 붙인 데이터 (DAgger 방식)."""
    X, Y = [], []
    for op in ["expert", "novice", "tired"]:
        for sd in seeds:
            lg = cs.run(op, "A", sd, keep_log=True).log
            for f in lg["feat"][::3]:
                X.append(f)
                Y.append(cs.expert_label(f))
    return np.array(X), np.array(Y)


def train_bc():
    X, Y = collect_demos(TRAIN_SEEDS[:60])
    Xt, Yt = collect_demos(TRAIN_SEEDS[60:80])
    model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                                                         random_state=0, early_stopping=True))
    model.fit(X, Y)
    r2 = model.score(Xt, Yt)
    print(f"[모방 모델] 학습 {len(X)}개 / 검증 {len(Xt)}개, 검증 R^2={r2:.3f}")
    return model, r2


def _skill_windows(seeds):
    F, y = [], []
    W = cs.SkillEstimator.WIN
    for op, label in [("novice", 0), ("expert", 1)]:
        for sd in seeds:
            lg = cs.run(op, "A", sd, keep_log=True).log
            u = np.array(lg["u_op"]); th = np.array(lg["th"])
            for i in range(W, len(u), W // 2):
                F.append(cs.SkillEstimator.feats(u[i - W:i], th[i - W:i]))
                y.append(label)
    return np.array(F), np.array(y)


def calibrate_skill():
    F, y = _skill_windows(TRAIN_SEEDS[:40])
    Ft, yt = _skill_windows(TRAIN_SEEDS[40:60])
    clf = LogisticRegression().fit(F, y)
    acc = clf.score(Ft, yt)
    print(f"[상태 추정기] 창 {len(F)}개 / 검증 {len(Ft)}개, 초보·숙련 구분 검증 정확도={acc:.3f}")
    return clf.coef_[0], float(clf.intercept_[0]), acc


HORIZON = int(2.5 / cs.DT)


def risk_dataset(seeds):
    """(현재 상태 9개 + 조이스틱 2개) → 앞으로 2.5초 동안의 최대 위험도. 실제로 일어난 미래로 라벨링."""
    X, Y = [], []
    for op in ["novice", "tired", "expert"]:
        for mode in "AB":
            for sd in seeds:
                lg = cs.run(op, mode, sd, keep_log=True).log
                F = np.array(lg["feat"]); U = np.c_[lg["u_eff_t"], lg["u_eff_h"]]; R = np.array(lg["risk_now"])
                for i in range(0, len(F), 4):
                    X.append(np.r_[F[i], U[i]])
                    Y.append(R[i:i + HORIZON].max())
    return np.array(X), np.array(Y)


def train_risk():
    Xtr, Ytr = risk_dataset(TRAIN_SEEDS[:50])
    Xte, Yte = risk_dataset(TRAIN_SEEDS[50:70])
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0).fit(Xtr, Ytr)
    p = model.predict(Xte)
    r2 = r2_score(Yte, p)
    auc = roc_auc_score(Yte > 0.5, p)
    print(f"[위험 예측 모델] 학습 {len(Xtr)}개 / 검증 {len(Xte)}개, R^2={r2:.3f}, 위험(>0.5) 판별 AUC={auc:.3f}")
    return model, r2, auc


_models = {}


def _init(path):
    global _models
    with open(path, "rb") as f:
        _models = pickle.load(f)


def _one(args):
    op, mode, sd = args
    est = cs.SkillEstimator(_models["skill_w"], _models["skill_b"])
    r = cs.run(op, mode, sd, bc_model=_models["bc"], skill=est, risk_model=_models["risk"])
    return dict(op=op, mode=mode, seed=sd, success=r.success, cycle_time=r.cycle_time, entry_err=r.land_err,
                impact=r.impact, max_swing=r.max_sway_deg, risk_entries=r.risk_entries, guide_hits=r.guide_hits)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    bc, bc_r2 = train_bc()
    w, b, sk_acc = calibrate_skill()
    risk, r2, auc = train_risk()
    path = OUT / "models.pkl"
    with open(path, "wb") as f:
        pickle.dump({"bc": bc, "skill_w": w, "skill_b": b, "risk": risk, "risk_r2": r2, "risk_auc": auc, "bc_r2": bc_r2, "skill_acc": sk_acc}, f)

    jobs = [(op, mode, sd) for op in ["novice", "tired", "expert"] for mode in "ABC"
            for sd in range(EVAL_SEED0, EVAL_SEED0 + n)]
    with ProcessPoolExecutor(initializer=_init, initargs=(str(path),)) as ex:
        rows = list(ex.map(_one, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False, encoding="utf-8-sig")
    summ = df.groupby(["op", "mode"]).agg(success=("success", "mean"), cycle=("cycle_time", "mean"),
                                          entry_err=("entry_err", "mean"), swing=("max_swing", "mean"),
                                          risk=("risk_entries", "mean"), hits=("guide_hits", "mean"),
                                          impact=("impact", "mean")).round(3)
    print(summ)


if __name__ == "__main__":
    main()
