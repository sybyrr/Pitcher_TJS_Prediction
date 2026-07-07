"""Collinearity audit: correlations + VIF for the M0 workload features and the
E3 trend/variability features on cohort v2 (fitting set = train+valid)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
SCR = Path(r"C:/Users/PC/AppData/Local/Temp/claude/d--PAINS-Pitcher-TJS-Prediction/c8255b88-a70d-45ba-964b-531a2271c93f/scratchpad")

cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v2.parquet")
cohort = cohort.sort_values(["t", "pitcher"]).reset_index(drop=True)
N = len(cohort)

slim = pd.read_parquet(SCR / "slim_games.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                        g["pitch_count"].astype("float64").values,
                        g["mean_release_speed"].astype("float64").values,
                        g["game_year"].values.astype(np.int64))
DAY = np.timedelta64(1, "D")

def pw_mean(sp, pc, mask):
    m = mask & ~np.isnan(sp)
    if not m.any():
        return np.nan
    w = pc[m].sum()
    return float((sp[m] * pc[m]).sum() / w) if w > 0 else np.nan

pid_all = cohort["pitcher"].values.astype(np.int64)
t_all = cohort["t"].values.astype("datetime64[ns]")
year_all = cohort["year"].values.astype(np.int64)
rows = []
for i in range(N):
    pid = int(pid_all[i]); t = t_all[i].astype("datetime64[D]")
    gd, pc, sp, gy = by_pid[pid]
    before = gd < t
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))
    pc30 = pc[d30].sum(); pc90 = pc[d90].sum()
    acwr = (pc30 / 30.0) / (pc90 / 90.0) if pc90 > 0 else 0.0
    dsl = float((t - gd[before].max()) / DAY)
    v30 = pw_mean(sp, pc, d30)
    prior = before & (gy < int(year_all[i]))
    vp = pw_mean(sp, pc, prior)
    if np.isnan(vp):
        vp = pw_mean(sp, pc, before & (gd < t - np.timedelta64(30, "D")))
    vt = (v30 - vp) if not (np.isnan(v30) or np.isnan(vp)) else 0.0
    rows.append((pc30, pc90, float(d30.sum()), float(d90.sum()), acwr, dsl, vt, float(cohort["month"].iloc[i])))

W = pd.DataFrame(rows, columns=["pc_30", "pc_90", "ng_30", "ng_90", "acwr", "dsl", "vel_trend", "month"])
fit = (cohort["fold_main"].isin(["train", "valid"])).values
Wf = W[fit]

def vif_table(df: pd.DataFrame) -> pd.Series:
    X = (df - df.mean()) / df.std()
    out = {}
    for c in df.columns:
        y = X[c].values
        Z = X.drop(columns=c).values
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z], y, rcond=None)
        r2 = 1 - ((y - np.c_[np.ones(len(Z)), Z] @ beta) ** 2).sum() / (y ** 2).sum()
        out[c] = 1.0 / max(1 - r2, 1e-12)
    return pd.Series(out)

print("=== M0 workload (fit set, n=%d) Pearson corr ===" % fit.sum())
print(Wf.corr().round(2).to_string())
print("\n=== VIF (M0) ===")
print(vif_table(Wf).round(1).to_string())
cn = np.linalg.cond(((Wf - Wf.mean()) / Wf.std()).values)
print(f"\ncondition number (standardized): {cn:.1f}")

tv = pd.read_parquet(ROOT / "data/prospective/trendvar_features.parquet")
tv = tv.sort_values(["t", "pitcher"]).reset_index(drop=True)
assert (tv["pitcher"].values == cohort["pitcher"].values).all() and (tv["t"].values == cohort["t"].values).all()
TREND = ["ff_velo_tau15", "sl_velo_tau15", "ff_spin_tau15", "ff_velo_sen5", "fb_usage_delta", "cu_usage_delta"]
VAR = ["ff_velo_sd_recent", "ff_velo_sd_rel", "ff_relx_sd_recent", "ff_relx_sd_rel", "ff_ext_sd_rel", "ff_relx_drift"]
A = pd.concat([Wf.reset_index(drop=True), tv[fit][TREND + VAR].reset_index(drop=True)], axis=1)
C = A.corr()
print("\n=== |r|>=0.5 pairs, full 20-feature set (fit set) ===")
seen = []
for i, a in enumerate(C.columns):
    for b in C.columns[i + 1:]:
        r = C.loc[a, b]
        if abs(r) >= 0.5:
            seen.append((a, b, round(r, 2)))
for a, b, r in sorted(seen, key=lambda x: -abs(x[2])):
    print(f"  {a:18s} ~ {b:18s} r={r:+.2f}")
print("\n=== VIF (all 20) ===")
print(vif_table(A).round(1).sort_values(ascending=False).to_string())
