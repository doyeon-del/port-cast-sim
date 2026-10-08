"""부산신항 실제 해상 조건 검증 — Open-Meteo 재분석 자료(파고·파주기·돌풍)로 바다를 움직여 다시 비교.

  python experiment_busan.py [반복=60]
결과: data/busan_sea_hourly.csv (1년 시간별 해상 자료, 캐시)
      out/busan_results.csv, out/busan_summary.csv

- 학습된 모델(out/models.pkl)을 그대로 쓴다 (다시 학습하지 않음)
- 평상시: 1년 중 무작위 시간대 n개 / 거친 날: 파고 또는 돌풍 상위 10% 시간대 n개
- 자료 출처: Open-Meteo Marine & Historical Weather API (CC BY 4.0), 부산신항 앞바다 격자
  (관측소 실측값이 아니라 기상·파랑 모델 재분석 값)
"""

import json
import pickle
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import crane_sim as cs

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
DATA.mkdir(exist_ok=True)
SEA_CSV = DATA / "busan_sea_hourly.csv"
LAT_W, LON_W = 35.05, 128.82   # 파랑 격자 (신항 남쪽 앞바다)
LAT_A, LON_A = 35.07, 128.82   # 바람 격자
START, END = "2025-09-01", "2026-08-31"


def fetch_sea():
    if SEA_CSV.exists():
        return pd.read_csv(SEA_CSV, parse_dates=["time"])
    wave_url = (f"https://marine-api.open-meteo.com/v1/marine?latitude={LAT_W}&longitude={LON_W}"
                f"&hourly=wave_height,wave_period&start_date={START}&end_date={END}&timezone=Asia%2FSeoul")
    wind_url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT_A}&longitude={LON_A}"
                f"&start_date={START}&end_date={END}&hourly=wind_speed_10m,wind_gusts_10m"
                f"&wind_speed_unit=ms&timezone=Asia%2FSeoul")
    w = json.load(urllib.request.urlopen(wave_url, timeout=60))["hourly"]
    a = json.load(urllib.request.urlopen(wind_url, timeout=60))["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(w["time"]), "wave_height_m": w["wave_height"],
                       "wave_period_s": w["wave_period"], "wind_speed_ms": a["wind_speed_10m"],
                       "wind_gust_ms": a["wind_gusts_10m"]}).dropna()
    df.to_csv(SEA_CSV, index=False, encoding="utf-8-sig")
    return df


def pick_hours(df, n):
    rng = np.random.default_rng(2026)
    normal = df.iloc[rng.choice(len(df), n, replace=False)].assign(cond="평상시")
    rough_mask = (df.wave_height_m >= df.wave_height_m.quantile(0.9)) | (df.wind_gust_ms >= df.wind_gust_ms.quantile(0.9))
    rough_pool = df[rough_mask]
    rough = rough_pool.iloc[rng.choice(len(rough_pool), n, replace=False)].assign(cond="거친 날")
    return pd.concat([normal, rough], ignore_index=True)


_models = {}


def _init():
    global _models
    with open(OUT / "models.pkl", "rb") as f:
        _models = pickle.load(f)


def _one(args):
    op, mode, k, row = args
    seed = 5000 + k
    sc = cs.Scenario.from_sea(seed, row["wave_height_m"], row["wave_period_s"], row["wind_gust_ms"])
    est = cs.SkillEstimator(_models["skill_w"], _models["skill_b"])
    r = cs.run(op, mode, seed, bc_model=_models["bc"], skill=est, risk_model=_models["risk"], scenario=sc)
    return dict(cond=row["cond"], time=str(row["time"]), hs=row["wave_height_m"], gust=row["wind_gust_ms"],
                op=op, mode=mode, success=r.success, cycle_time=r.cycle_time, impact=r.impact,
                guide_hits=r.guide_hits)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    sea = fetch_sea()
    hours = pick_hours(sea, n)
    print(hours.groupby("cond")[["wave_height_m", "wave_period_s", "wind_gust_ms"]].describe().round(2).T)
    jobs = [(op, mode, k, row) for k, row in enumerate(hours.to_dict("records"))
            for op in ["novice", "tired", "expert"] for mode in "ABC"]
    with ProcessPoolExecutor(initializer=_init) as ex:
        rows = list(ex.map(_one, jobs, chunksize=4))
    d = pd.DataFrame(rows)
    d["hard"] = (d.impact >= cs.IMPACT_TOL) & (d.cycle_time < cs.T_MAX)
    d.to_csv(OUT / "busan_results.csv", index=False, encoding="utf-8-sig")
    s = d.groupby(["cond", "op", "mode"]).agg(success=("success", "mean"), hard=("hard", "mean"),
                                               hits=("guide_hits", "mean"), cycle=("cycle_time", "mean"))
    s[["success", "hard"]] *= 100
    s = s.round(1)
    s.to_csv(OUT / "busan_summary.csv", encoding="utf-8-sig")
    print(s)


if __name__ == "__main__":
    main()
