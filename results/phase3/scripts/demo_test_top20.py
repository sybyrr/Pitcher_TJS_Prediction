"""Demo sheet: monthly top-20 risk lists on the TEST period (2022-24).

Read-only on the frozen model (coefficients asserted vs FROZEN_MODEL.md);
produces a per-pitcher report row: rank, scores, role, top-3 risk drivers
(per-feature contribution to the standardized linear score), and the actual
outcome (surgery within 90/150d; 2024-09 H150 labels marked immature).
Names via MLB StatsAPI people endpoint. NOT canonical — demo artifact for
the proposal. Output: ../demo_test_top20.csv.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
OUT = Path(__file__).resolve().parent.parent
t0 = time.time()

FROZEN_COEF = [-0.214385, -0.083979, -0.153668, -0.011728, -0.233497,
               +0.110833, +0.133170, -0.085616, -0.010479, -0.196999]

cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v4.parquet").sort_values(["t", "pitcher"]).reset_index(drop=True)
N = len(cohort)
slim = pd.read_parquet(ROOT / "data/prospective/slim_games_v4.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                        g["pitch_count"].astype("float64").values,
                        g["mean_release_speed"].astype("float64").values,
                        g["game_year"].values.astype(np.int64))
gs = pd.read_parquet(ROOT / "data/prospective/gs_flags_v1.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
gs_by_pid = {}
for pid, g in gs.groupby("pitcher", sort=False):
    gs_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                           g["n_g"].values.astype(float), g["n_gs"].values.astype(float))
tj = pd.read_csv(ROOT / "data/prospective/tj_live_clean_20260707.csv")
tj["surg_date"] = pd.to_datetime(tj["surg_date"])
surg_by_pid = {int(p): np.sort(g["surg_date"].values) for p, g in tj.groupby("mlbamid_i")}
DAY = np.timedelta64(1, "D")

def pw_mean(v, w, mask):
    m = mask & ~np.isnan(v)
    if not m.any():
        return np.nan
    ws = w[m].sum()
    return float((v[m] * w[m]).sum() / ws) if ws > 0 else np.nan

pid_all = cohort["pitcher"].values.astype(np.int64)
t_all = cohort["t"].values.astype("datetime64[ns]")
year_all = cohort["year"].values.astype(np.int64)
month_all = cohort["month"].values.astype(np.float64)
ncg_all = cohort["n_career_games"].values.astype(np.float64)
rows = []
prior_pc_rate = np.zeros(N); vt_missing = np.zeros(N); start_share = np.zeros(N)
gs_share = np.zeros(N)
for i in range(N):
    pid = int(pid_all[i]); t = t_all[i].astype("datetime64[D]")
    gd, pc, sp, gy = by_pid[pid]
    before = gd < t
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))
    dsl = float((t - gd[before].max()) / DAY)
    vmean_30 = pw_mean(sp, pc, d30)
    yr = int(year_all[i])
    vmean_prior = pw_mean(sp, pc, before & (gy < yr))
    if np.isnan(vmean_prior):
        vmean_prior = pw_mean(sp, pc, before & (gd < t - np.timedelta64(30, "D")))
    if np.isnan(vmean_30) or np.isnan(vmean_prior):
        vel_trend = 0.0; vt_missing[i] = 1.0
    else:
        vel_trend = vmean_30 - vmean_prior
    prior_years = gy[before & (gy < yr)]
    if prior_years.size:
        m_py = before & (gy == prior_years.max())
        prior_pc_rate[i] = pc[m_py].sum() / 183.0
    rows.append((pc_90_ := pc[d90].sum() / 90.0, pc[d30].sum() / 30.0 - pc_90_, dsl, vel_trend, month_all[i]))
    rm = before & (gd >= t - np.timedelta64(365, "D"))
    ng365 = int(rm.sum())
    start_share[i] = float((pc[rm] >= 50).sum()) / ng365 if ng365 > 0 else 0.0
    sgd, sng, sngs = gs_by_pid[pid]
    sm = (sgd < t) & (sgd >= t - np.timedelta64(365, "D"))
    n_app = sng[sm].sum()
    gs_share[i] = sngs[sm].sum() / n_app if n_app > 0 else 0.0
F = pd.DataFrame(rows, columns=["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"])
F["start_share"] = start_share
F["prior_pc_rate"] = prior_pc_rate
F["ncg_log"] = np.log1p(ncg_all)
F["vt_missing"] = vt_missing
M_SA = list(F.columns)
print(f"[t={time.time()-t0:.0f}s] features built")

S_MAX = 5
fit_mask = (cohort["fold_main"].values == "train") | (cohort["fold_main"].values == "valid")
Xall = F[M_SA].values
rX, rs, ry = [], [], []
for i in np.where(fit_mask)[0]:
    surgs = surg_by_pid.get(int(pid_all[i]))
    fired = False
    for s in range(S_MAX):
        if fired:
            break
        lab = 0
        if surgs is not None:
            lo = t_all[i] + np.timedelta64(30 * s, "D"); hi = t_all[i] + np.timedelta64(30 * (s + 1), "D")
            if np.any((surgs > lo) & (surgs <= hi)):
                lab = 1; fired = True
        rX.append(Xall[i]); rs.append(s); ry.append(lab)
Xpp = np.column_stack([np.asarray(rX), np.asarray(rs, dtype=float)])
scaler = StandardScaler().fit(Xpp)
hz = LogisticRegression(max_iter=2000).fit(scaler.transform(Xpp), np.asarray(ry))
assert np.abs(hz.coef_[0] - np.array(FROZEN_COEF)).max() < 1e-5
print("frozen coefficient reproduction: OK")

te = (cohort["fold_main"].values == "test") & (year_all <= 2024)
Xt = Xall[te]
h = np.empty((Xt.shape[0], S_MAX))
for s in range(S_MAX):
    h[:, s] = hz.predict_proba(scaler.transform(np.column_stack([Xt, np.full(Xt.shape[0], float(s))])))[:, 1]
p90 = 1 - np.prod(1 - h[:, :3], axis=1)
p150 = 1 - np.prod(1 - h[:, :5], axis=1)

# per-feature contribution to the standardized linear score (s term excluded)
Z = (Xt - scaler.mean_[:9]) / scaler.scale_[:9]
contrib = Z * np.array(FROZEN_COEF[:9])
DRIVER_LABEL = {  # label when the contribution is positive (pushes risk UP)
    "pc_chronic": "최근90일 투구량 적음", "pc_acute_dev": "최근30일 급감",
    "days_since_last": "마지막 등판 최근", "vel_trend": "구속 하락",
    "month": "시즌 초반", "start_share": "선발 역할",
    "prior_pc_rate": "작년 투구량 많음", "ncg_log": "경력 짧음",
    "vt_missing": "구속추세 결측"}

sub = cohort.loc[te].reset_index(drop=True)
t_m = t_all[te]
next_surg = pd.to_datetime(sub["next_surgery_date"]).values
out_rows = []
for d in np.unique(t_m):
    sel = np.where(t_m == d)[0]
    order = sel[np.argsort(-p150[sel], kind="stable")][:20]
    for rk, r in enumerate(order, start=1):
        top3 = np.argsort(-contrib[r])[:3]
        drivers = "; ".join(f"{DRIVER_LABEL[M_SA[j]]}({contrib[r, j]:+.2f})" for j in top3 if contrib[r, j] > 0)
        ns = next_surg[r]
        has_surg_150 = pd.notna(ns) and (pd.Timestamp(ns) - pd.Timestamp(d)).days <= 150
        days_to = (pd.Timestamp(ns) - pd.Timestamp(d)).days if has_surg_150 else None
        h150_lab = int(sub.loc[r, "label_H150_B0"])
        immature = (pd.Timestamp(d) + pd.Timedelta(days=150)) > pd.Timestamp("2024-12-31")
        gsr = gs_share[np.where(te)[0][r]]
        out_rows.append(dict(
            t=pd.Timestamp(d).date(), rank=rk, pitcher=int(sub.loc[r, "pitcher"]),
            role=("SP" if gsr >= 0.5 else ("RP" if gsr <= 0.2 else "swing")),
            P90=round(float(p90[r]), 5), P150=round(float(p150[r]), 5),
            pct=round(100 * (1 - (rk - 1) / len(sel)), 1),
            drivers=drivers,
            surgery_within_90d=int(sub.loc[r, "label_H90_B0"]),
            surgery_within_150d=(h150_lab if not immature else "라벨미성숙"),
            surgery_date=(pd.Timestamp(ns).date() if has_surg_150 else ""),
            days_to_surgery=(days_to if has_surg_150 else "")))
df = pd.DataFrame(out_rows)

# pitcher names (StatsAPI batch lookup)
ids = sorted(df["pitcher"].unique())
names = {}
for k in range(0, len(ids), 100):
    chunk = ids[k:k + 100]
    try:
        js = requests.get("https://statsapi.mlb.com/api/v1/people",
                          params={"personIds": ",".join(map(str, chunk))}, timeout=30).json()
        for p in js.get("people", []):
            names[int(p["id"])] = p.get("fullName", "")
    except Exception as e:
        print(f"name lookup failed for chunk {k}: {e}")
    time.sleep(0.3)
df.insert(3, "name", df["pitcher"].map(names).fillna(""))
df.to_csv(OUT / "demo_test_top20.csv", index=False, encoding="utf-8-sig")

hits90 = df[df["surgery_within_90d"] == 1]
print(f"\nrows {len(df)} ({df['t'].nunique()} dates x top-20); named {df['name'].ne('').sum()}")
print(f"top-20 rows with surgery within 90d: {len(hits90)}  "
      f"(distinct pitchers {hits90['pitcher'].nunique()})")
print(f"[t={time.time()-t0:.0f}s] wrote demo_test_top20.csv")
