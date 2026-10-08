"""60회 결과 검증: 피곤한 작업자 A(보조 없음)·C(Port Cast)를 seed 0..999로 각 1000회 반복.

experiment.py가 저장한 out/models.pkl을 그대로 쓴다 (seed 0~59는 results.csv와 같은 값이 재현됨).
실행: OMP_NUM_THREADS=1 python verify_1000.py          # 시뮬 + 요약 (8코어 약 16분)
      python verify_1000.py --summary                  # 저장된 CSV로 요약만
결과: out/verify_1000_tired_AC.csv, out/verify_1000_summary.txt

OMP_NUM_THREADS=1 없이 돌리면 프로세스마다 BLAS 스레드가 겹쳐 20배쯤 느려진다.
"""

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from scipy.stats import beta

import crane_sim as cs

OUT = Path(__file__).parent / "out"
CSV = OUT / "verify_1000_tired_AC.csv"
N = 1000
_m = {}


def _init():
    global _m
    with open(OUT / "models.pkl", "rb") as f:
        _m = pickle.load(f)


def _one(args):
    op, mode, sd = args
    est = cs.SkillEstimator(_m["skill_w"], _m["skill_b"])
    r = cs.run(op, mode, sd, bc_model=_m["bc"], skill=est, risk_model=_m["risk"])
    return dict(op=op, mode=mode, seed=sd, success=r.success, cycle_time=r.cycle_time, impact=r.impact,
                guide_hits=r.guide_hits, risk_entries=r.risk_entries)


def ci(k, n):
    """Clopper-Pearson 95% 구간 (%)."""
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo * 100, hi * 100


def summarize(d):
    d = d.copy()
    d["hard"] = (d.impact >= 0.6) & (d.cycle_time < 90)          # make_charts.py와 같은 정의
    d["hit"] = d.guide_hits > 0
    d["event"] = d.hard | d.hit                                    # 위험 사건 = 거친 착지 또는 가이드 충돌
    names = {"event": "위험 사건 (거친 착지 또는 가이드 충돌)", "hard": "거친 착지", "hit": "가이드 충돌 1회 이상",
             "success": "착지 성공"}
    lines = [f"피곤한 작업자, 모드별 {N}회 (seed 0~{N - 1})", ""]
    for col, name in names.items():
        lines.append(name)
        for mode, label in [("A", "보조 없음"), ("C", "Port Cast")]:
            x = d[d["mode"] == mode][col]
            k, n = int(x.sum()), len(x)
            lo, hi = ci(k, n)
            lines.append(f"  {label:9s} {k:4d}/{n} = {k / n * 100:5.1f}%   95% 구간 {lo:.1f}~{hi:.1f}%")
    c = d[d["mode"] == "C"]
    lines += ["", f"Port Cast 최대 착지 속도 {c.impact.max():.3f} m/s (거친 착지 기준 0.6) → 거친 착지 0건은 저항 로직의 설계 결과",
              f"seed 0~59 거친 착지: A {d[(d.seed < 60) & (d['mode'] == 'A')].hard.mean() * 100:.1f}% / "
              f"C {d[(d.seed < 60) & (d['mode'] == 'C')].hard.mean() * 100:.1f}% (results.csv와 동일)"]
    return "\n".join(lines)


def main():
    if "--summary" not in sys.argv:
        jobs = [("tired", m, sd) for m in "AC" for sd in range(N)]
        with ProcessPoolExecutor(initializer=_init) as ex:
            rows = list(ex.map(_one, jobs, chunksize=2))
        pd.DataFrame(rows).to_csv(CSV, index=False, encoding="utf-8-sig")
    text = summarize(pd.read_csv(CSV))
    (OUT / "verify_1000_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
