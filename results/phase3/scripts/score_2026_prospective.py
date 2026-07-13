"""True prospective scoring: 2026 decision dates with the FROZEN model.

First timestamped scoring run after the 2026-07-13 freeze (FROZEN_MODEL.md
section 6). Raw 2026 snapshot = statcast_2026.parquet (2026-03-01..07-12,
regular season only). Decision dates 2026-04-01..07-01 (Apr-Sep grid, dates
whose features are computable strictly before t).
The hazard model is re-fit on the IDENTICAL v4 fit set and every coefficient
is ASSERTED against the frozen values recorded in FROZEN_MODEL.md (no
refit on new data, no adaptation). Labels for 2026 are unknown and unused.
Output: ../prospective_scores_2026_ts20260713.csv + md5.
"""
from __future__ import annotations
import hashlib
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

FROZEN = {  # FROZEN_MODEL.md section 3 (order: M_sa 9f + s)
    "intercept": -5.900480,
    "coef": [-0.214385, -0.083979, -0.153668, -0.011728, -0.233497,
             +0.110833, +0.133170, -0.085616, -0.010479, -0.196999],
    "recal_a": -0.164030, "recal_b": 0.966471,
}

# ---- 2026 slim/gs extensions (regular season only, spec identical to v4) ----
cols = ["pitcher", "game_date", "release_speed", "game_type",
        "game_pk", "inning_topbot", "at_bat_number"]
raw26 = pd.read_parquet(ROOT / "data/raw/statcast_2026.parquet", columns=cols)
raw26 = raw26[raw26["game_type"] == "R"].copy()
raw26["game_date"] = pd.to_datetime(raw26["game_date"])
raw26["pitcher"] = raw26["pitcher"].astype("int64")
raw26["release_speed"] = pd.to_numeric(raw26["release_speed"], errors="coerce")
slim26 = (raw26.groupby(["pitcher", "game_date"], sort=False)
               .agg(pitch_count=("release_speed", "size"),
                    mean_release_speed=("release_speed", "mean")).reset_index())
slim26["game_year"] = 2026
print(f"[t={time.time()-t0:.0f}s] 2026 R-games slim rows {len(slim26):,} "
      f"(last date {slim26['game_date'].max().date()})")

first_ab = (raw26.groupby(["game_pk", "inning_topbot"], sort=False)["at_bat_number"]
                 .transform("min"))
starters26 = raw26[raw26["at_bat_number"] == first_ab][["game_pk", "game_date", "pitcher"]].drop_duplicates()
games26 = raw26[["pitcher", "game_date", "game_pk"]].drop_duplicates()
games26 = games26.merge(starters26.assign(is_gs=1), on=["game_pk", "game_date", "pitcher"], how="left")
games26["is_gs"] = games26["is_gs"].fillna(0).astype("int8")
gs26 = (games26.groupby(["pitcher", "game_date"], sort=False)
               .agg(n_g=("game_pk", "nunique"), n_gs=("is_gs", "sum")).reset_index())

slim = pd.read_parquet(ROOT / "data/prospective/slim_games_v4.parquet")
slim = pd.concat([slim, slim26], ignore_index=True).sort_values(["pitcher", "game_date"]).reset_index(drop=True)
gs = pd.read_parquet(ROOT / "data/prospective/gs_flags_v1.parquet")
gs = pd.concat([gs, gs26], ignore_index=True).sort_values(["pitcher", "game_date"]).reset_index(drop=True)

# ---- 2026 decision cohort (frozen eligibility rule) ----
GRID = [pd.Timestamp(2026, m, 1) for m in (4, 5, 6, 7)]
rows_c = []
for t in GRID:
    before = slim[slim["game_date"] < t]
    agg = before.groupby("pitcher")["game_date"].agg(["count", "max"])
    agg["dsl"] = (t - agg["max"]).dt.days
    risk = agg[(agg["count"] >= 20) & (agg["dsl"] <= 365)]
    for pid, r in risk.iterrows():
        rows_c.append((int(pid), t, int(r["count"])))
co = pd.DataFrame(rows_c, columns=["pitcher", "t", "n_career_games"])
print(f"2026 cohort windows {len(co):,} across {co['t'].nunique()} dates "
      f"({co.groupby('t').size().to_dict()})")

# ---- M_sa features for 2026 windows (verbatim frozen definitions) ----
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                        g["pitch_count"].astype("float64").values,
                        g["mean_release_speed"].astype("float64").values,
                        g["game_year"].values.astype(np.int64))
gs_by_pid = {}
for pid, g in gs.groupby("pitcher", sort=False):
    gs_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                           g["n_g"].values.astype(float), g["n_gs"].values.astype(float))
DAY = np.timedelta64(1, "D")

def pw_mean(v, w, mask):
    m = mask & ~np.isnan(v)
    if not m.any():
        return np.nan
    ws = w[m].sum()
    return float((v[m] * w[m]).sum() / ws) if ws > 0 else np.nan

Nc = len(co)
feats = np.zeros((Nc, 9)); gs_share = np.zeros(Nc)
for i in range(Nc):
    pid = int(co.loc[i, "pitcher"]); t = np.datetime64(co.loc[i, "t"], "D")
    gd, pc, sp, gy = by_pid[pid]
    before = gd < t
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))
    dsl = float((t - gd[before].max()) / DAY)
    vmean_30 = pw_mean(sp, pc, d30)
    vmean_prior = pw_mean(sp, pc, before & (gy < 2026))
    if np.isnan(vmean_prior):
        vmean_prior = pw_mean(sp, pc, before & (gd < t - np.timedelta64(30, "D")))
    vt_missing = 0.0
    if np.isnan(vmean_30) or np.isnan(vmean_prior):
        vel_trend = 0.0; vt_missing = 1.0
    else:
        vel_trend = vmean_30 - vmean_prior
    prior_years = gy[before & (gy < 2026)]
    prior_pc_rate = pc[before & (gy == prior_years.max())].sum() / 183.0 if prior_years.size else 0.0
    rm = before & (gd >= t - np.timedelta64(365, "D"))
    ng365 = int(rm.sum())
    start_share = float((pc[rm] >= 50).sum()) / ng365 if ng365 > 0 else 0.0
    feats[i] = [pc[d90].sum() / 90.0, pc[d30].sum() / 30.0 - pc[d90].sum() / 90.0, dsl,
                vel_trend, float(co.loc[i, "t"].month), start_share, prior_pc_rate,
                np.log1p(co.loc[i, "n_career_games"]), vt_missing]
    sgd, sng, sngs = gs_by_pid[pid]
    sm = (sgd < t) & (sgd >= t - np.timedelta64(365, "D"))
    n_app = sng[sm].sum()
    gs_share[i] = sngs[sm].sum() / n_app if n_app > 0 else 0.0
print(f"[t={time.time()-t0:.0f}s] 2026 features built")

# ---- frozen model reproduction on v4 fit set + coefficient asserts ----
cohort4 = pd.read_parquet(ROOT / "data/prospective/cohort_v4.parquet").sort_values(["t", "pitcher"]).reset_index(drop=True)
slim4 = pd.read_parquet(ROOT / "data/prospective/slim_games_v4.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
by_pid4 = {}
for pid, g in slim4.groupby("pitcher", sort=False):
    by_pid4[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                         g["pitch_count"].astype("float64").values,
                         g["mean_release_speed"].astype("float64").values,
                         g["game_year"].values.astype(np.int64))
tj = pd.read_csv(ROOT / "data/prospective/tj_live_clean_20260707.csv")
tj["surg_date"] = pd.to_datetime(tj["surg_date"])
surg_by_pid = {int(p): np.sort(g["surg_date"].values) for p, g in tj.groupby("mlbamid_i")}

pid4 = cohort4["pitcher"].values.astype(np.int64)
t4 = cohort4["t"].values.astype("datetime64[ns]")
y4 = cohort4["year"].values.astype(np.int64)
m4 = cohort4["month"].values.astype(np.float64)
ncg4 = cohort4["n_career_games"].values.astype(np.float64)
N4 = len(cohort4)
X4 = np.zeros((N4, 9))
for i in range(N4):
    pid = int(pid4[i]); t = t4[i].astype("datetime64[D]")
    gd, pc, sp, gy = by_pid4[pid]
    before = gd < t
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))
    dsl = float((t - gd[before].max()) / DAY)
    vmean_30 = pw_mean(sp, pc, d30)
    yr = int(y4[i])
    vmean_prior = pw_mean(sp, pc, before & (gy < yr))
    if np.isnan(vmean_prior):
        vmean_prior = pw_mean(sp, pc, before & (gd < t - np.timedelta64(30, "D")))
    vt_missing = 0.0
    if np.isnan(vmean_30) or np.isnan(vmean_prior):
        vel_trend = 0.0; vt_missing = 1.0
    else:
        vel_trend = vmean_30 - vmean_prior
    prior_years = gy[before & (gy < yr)]
    prior_pc_rate = pc[before & (gy == prior_years.max())].sum() / 183.0 if prior_years.size else 0.0
    rm = before & (gd >= t - np.timedelta64(365, "D"))
    ng365 = int(rm.sum())
    start_share4 = float((pc[rm] >= 50).sum()) / ng365 if ng365 > 0 else 0.0
    X4[i] = [pc[d90].sum() / 90.0, pc[d30].sum() / 30.0 - pc[d90].sum() / 90.0, dsl,
             vel_trend, m4[i], start_share4, prior_pc_rate, np.log1p(ncg4[i]), vt_missing]

S_MAX = 5
fit_mask = (cohort4["fold_main"].values == "train") | (cohort4["fold_main"].values == "valid")
rX, rs, ry, rw = [], [], [], []
for i in np.where(fit_mask)[0]:
    surgs = surg_by_pid.get(int(pid4[i]))
    fired = False
    for s in range(S_MAX):
        if fired:
            break
        lab = 0
        if surgs is not None:
            lo = t4[i] + np.timedelta64(30 * s, "D"); hi = t4[i] + np.timedelta64(30 * (s + 1), "D")
            if np.any((surgs > lo) & (surgs <= hi)):
                lab = 1; fired = True
        rX.append(X4[i]); rs.append(s); ry.append(lab); rw.append(i)
Xf = np.asarray(rX); sf = np.asarray(rs, dtype=float); yf = np.asarray(ry); wf = np.asarray(rw)
Xpp = np.column_stack([Xf, sf])
scaler = StandardScaler().fit(Xpp)
hz = LogisticRegression(max_iter=2000)
hz.fit(scaler.transform(Xpp), yf)
assert abs(hz.intercept_[0] - FROZEN["intercept"]) < 1e-5, hz.intercept_[0]
assert np.abs(hz.coef_[0] - np.array(FROZEN["coef"])).max() < 1e-5, hz.coef_[0]
print("FROZEN coefficient reproduction: OK (all within 1e-5 of FROZEN_MODEL.md)")

eps = 1e-12
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
oof_h = np.full(len(yf), np.nan)
for tr, ho in GroupKFold(n_splits=5).split(Xf, yf, pid4[wf]):
    sc_f = StandardScaler().fit(Xpp[tr])
    m_f = LogisticRegression(max_iter=2000).fit(sc_f.transform(Xpp[tr]), yf[tr])
    oof_h[ho] = m_f.predict_proba(sc_f.transform(Xpp[ho]))[:, 1]
rc = LogisticRegression(max_iter=2000).fit(logit(oof_h).reshape(-1, 1), yf)
a_rc, b_rc = float(rc.intercept_[0]), float(rc.coef_[0][0])
assert abs(a_rc - FROZEN["recal_a"]) < 1e-5 and abs(b_rc - FROZEN["recal_b"]) < 1e-5
print(f"FROZEN recal reproduction: OK (a={a_rc:+.6f}, b={b_rc:.6f})")

# ---- score 2026 windows ----
h = np.empty((Nc, S_MAX))
for s in range(S_MAX):
    h[:, s] = hz.predict_proba(scaler.transform(np.column_stack([feats, np.full(Nc, float(s))])))[:, 1]
h_rc = 1 / (1 + np.exp(-(a_rc + b_rc * logit(h))))
sc_out = co[["pitcher", "t"]].copy()
sc_out["P90_raw"] = 1 - np.prod(1 - h[:, :3], axis=1)
sc_out["P150_raw"] = 1 - np.prod(1 - h[:, :5], axis=1)
sc_out["P90_recal"] = 1 - np.prod(1 - h_rc[:, :3], axis=1)
sc_out["P150_recal"] = 1 - np.prod(1 - h_rc[:, :5], axis=1)
sc_out["gs_share"] = gs_share
sc_out["rp_flag"] = (gs_share <= 0.2).astype(int)
sc_out["rank_H150"] = sc_out.groupby("t")["P150_raw"].rank(ascending=False, method="first").astype(int)
alert = np.zeros(len(sc_out), dtype=int)
for d, g in sc_out.groupby("t"):
    order = g.sort_values("P150_raw", ascending=False)
    rp_top = order[order["rp_flag"] == 1].head(20).index
    rest = order.drop(rp_top).head(50 - len(rp_top)).index
    alert[sc_out.index.get_indexer(rp_top)] = 1
    alert[sc_out.index.get_indexer(rest)] = 1
sc_out["alert_q20"] = alert
path = OUT / "prospective_scores_2026_ts20260713.csv"
sc_out.to_csv(path, index=False, float_format="%.8f")
md5 = hashlib.md5(path.read_bytes()).hexdigest()
print(f"\n2026 prospective scores: {len(sc_out):,} windows, "
      f"alerts/date {sc_out.groupby('t')['alert_q20'].sum().to_dict()}")
print(f"WROTE {path.name}  md5 {md5}")
print(f"[t={time.time()-t0:.0f}s] done")
