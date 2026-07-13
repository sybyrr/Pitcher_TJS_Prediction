"""Freeze-integrity remediation (codex 3rd audit): dump the frozen model
state at FULL float64 precision to ../frozen_model_state.json and print
SHA-256 of the state file and both score archives.

The JSON (not the rounded table in FROZEN_MODEL.md) is the authoritative
frozen state: scaler mean/scale, LR coef/intercept, interval recal a/b.
Refit here is on the identical v4 fit set (deterministic reproduction);
future scoring should LOAD this state instead of refitting.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
OUT = Path(__file__).resolve().parent.parent
t0 = time.time()

cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v4.parquet").sort_values(["t", "pitcher"]).reset_index(drop=True)
N = len(cohort)
slim = pd.read_parquet(ROOT / "data/prospective/slim_games_v4.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                        g["pitch_count"].astype("float64").values,
                        g["mean_release_speed"].astype("float64").values,
                        g["game_year"].values.astype(np.int64))
gf = pd.read_parquet(ROOT / "data/prospective/game_features_v4.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
role_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    role_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                             g["total_pitches"].astype("float64").values)
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
    rgd, rtp = role_by_pid[pid]
    rm = (rgd < t) & (rgd >= t - np.timedelta64(365, "D"))
    ng365 = int(rm.sum())
    start_share[i] = float((rtp[rm] >= 50).sum()) / ng365 if ng365 > 0 else 0.0
F = pd.DataFrame(rows, columns=["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"])
F["start_share"] = start_share
F["prior_pc_rate"] = prior_pc_rate
F["ncg_log"] = np.log1p(ncg_all)
F["vt_missing"] = vt_missing
M_SA = list(F.columns)

S_MAX = 5
fit_mask = (cohort["fold_main"].values == "train") | (cohort["fold_main"].values == "valid")
Xall = F[M_SA].values
rX, rs, ry, rw = [], [], [], []
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
        rX.append(Xall[i]); rs.append(s); ry.append(lab); rw.append(i)
Xf = np.asarray(rX); sf = np.asarray(rs, dtype=float); yf = np.asarray(ry); wf = np.asarray(rw)
Xpp = np.column_stack([Xf, sf])
scaler = StandardScaler().fit(Xpp)
hz = LogisticRegression(max_iter=2000)
hz.fit(scaler.transform(Xpp), yf)

eps = 1e-12
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
oof_h = np.full(len(yf), np.nan)
for tr, ho in GroupKFold(n_splits=5).split(Xf, yf, pid_all[wf]):
    sc_f = StandardScaler().fit(Xpp[tr])
    m_f = LogisticRegression(max_iter=2000).fit(sc_f.transform(Xpp[tr]), yf[tr])
    oof_h[ho] = m_f.predict_proba(sc_f.transform(Xpp[ho]))[:, 1]
rc = LogisticRegression(max_iter=2000).fit(logit(oof_h).reshape(-1, 1), yf)

state = {
    "frozen_at": "2026-07-13 (Asia/Seoul), codex-3rd-audit remediation dump",
    "model": "M_sa 9f + discrete-time hazard (plain LR, s=0..4), fit train 2017-20 + valid 2021 (cohort_v4)",
    "features": M_SA + ["s"],
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "lr_intercept": float(hz.intercept_[0]),
    "lr_coef": hz.coef_[0].tolist(),
    "recal_interval_logistic": {"a": float(rc.intercept_[0]), "b": float(rc.coef_[0][0]),
                                "note": "grouped-OOF interval-level; window-level P(H) NOT thereby calibrated"},
    "alert_policy": "canonical = plain top-50 by P150 (q=0); q=20 RP-reserve is a challenger only "
                    "(passed primary boundary, FAILED safety-H150 gate: 16->12)",
    "sklearn": "1.9.0", "python": "3.11.11",
}
path = OUT / "frozen_model_state.json"
path.write_text(json.dumps(state, indent=1))

for f in ("frozen_model_state.json", "frozen_scores_2025_ts20260713.csv",
          "prospective_scores_2026_ts20260713.csv"):
    p = OUT / f
    print(f"SHA256 {hashlib.sha256(p.read_bytes()).hexdigest()}  {f}")
print(f"[t={time.time()-t0:.0f}s] wrote {path.name}")
