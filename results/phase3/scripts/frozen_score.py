"""Freeze step: fit the canonical model, print frozen coefficients, and
timestamp-archive frozen scores for all 2025 decision dates (spec v2).

Frozen model = M_sa 9f + discrete-time hazard (plain LR, s=0..4, fit on
train 2017-20 + valid 2021 of cohort_v4) + grouped-OOF interval recal
(reported; near-identity) + alert policy top-50 with RP-reserved q=20
(RP = trailing-365d GS share <= 0.2, gs_flags_v1).
2025 windows are the future label-refresh robustness set — archiving their
scores now (before label maturity) makes that evaluation tamper-proof.
Output: ../frozen_scores_2025_ts20260713.csv + md5, coefficients printout.
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
    pc_30 = pc[d30].sum(); pc_90 = pc[d90].sum()
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
    rows.append((pc_90 / 90.0, pc_30 / 30.0 - pc_90 / 90.0, dsl, vel_trend, month_all[i]))
    rgd, rtp = role_by_pid[pid]
    rm = (rgd < t) & (rgd >= t - np.timedelta64(365, "D"))
    ng365 = int(rm.sum())
    start_share[i] = float((rtp[rm] >= 50).sum()) / ng365 if ng365 > 0 else 0.0
    sgd, sng, sngs = gs_by_pid[pid]
    sm = (sgd < t) & (sgd >= t - np.timedelta64(365, "D"))
    n_app = sng[sm].sum()
    gs_share[i] = sngs[sm].sum() / n_app if n_app > 0 else 0.0

F = pd.DataFrame(rows, columns=["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"])
F["start_share"] = start_share
F["prior_pc_rate"] = prior_pc_rate
F["ncg_log"] = np.log1p(ncg_all)
F["vt_missing"] = vt_missing
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6
M_SA = list(F.columns)
print(f"[t={time.time()-t0:.0f}s] features built")

S_MAX = 5
int_label = np.zeros((N, S_MAX), dtype=np.int8)
at_risk = np.ones((N, S_MAX), dtype=bool)
for i in range(N):
    surgs = surg_by_pid.get(int(pid_all[i]))
    if surgs is None:
        continue
    t = t_all[i]
    fired = False
    for s in range(S_MAX):
        if fired:
            at_risk[i, s] = False
            continue
        lo = t + np.timedelta64(30 * s, "D"); hi = t + np.timedelta64(30 * (s + 1), "D")
        if np.any((surgs > lo) & (surgs <= hi)):
            int_label[i, s] = 1; fired = True

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
Xall = F[M_SA].values
rX, rs, ry, rw = [], [], [], []
for i in np.where(fit_mask)[0]:
    for s in range(S_MAX):
        if not at_risk[i, s]:
            break
        rX.append(Xall[i]); rs.append(s); ry.append(int(int_label[i, s])); rw.append(i)
Xf = np.asarray(rX); sf = np.asarray(rs, dtype=float); yf = np.asarray(ry); wf = np.asarray(rw)
Xpp = np.column_stack([Xf, sf])
scaler = StandardScaler().fit(Xpp)
hz = LogisticRegression(max_iter=2000)
hz.fit(scaler.transform(Xpp), yf)

print("\nFROZEN coefficients (hazard LR on standardized [M_sa, s]):")
print(f"  intercept: {hz.intercept_[0]:+.6f}")
for name, c, mu, sd in zip(M_SA + ["s"], hz.coef_[0], scaler.mean_, scaler.scale_):
    print(f"  {name:16s} coef {c:+.6f}   (scaler mean {mu:.6f} scale {sd:.6f})")

# grouped-OOF interval recalibration (as in a0_recal.py)
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
a_rc, b_rc = float(rc.intercept_[0]), float(rc.coef_[0][0])
print(f"  recal (grouped OOF): a={a_rc:+.6f} b={b_rc:.6f}")

# frozen scores for 2025 decision dates (label-refresh robustness set)
m25 = year_all == 2025
Xt = Xall[m25]
h = np.empty((Xt.shape[0], S_MAX))
for s in range(S_MAX):
    h[:, s] = hz.predict_proba(scaler.transform(np.column_stack([Xt, np.full(Xt.shape[0], float(s))])))[:, 1]
h_rc = 1 / (1 + np.exp(-(a_rc + b_rc * logit(h))))
p90 = 1 - np.prod(1 - h[:, :3], axis=1)
p150 = 1 - np.prod(1 - h[:, :5], axis=1)
p90_rc = 1 - np.prod(1 - h_rc[:, :3], axis=1)
p150_rc = 1 - np.prod(1 - h_rc[:, :5], axis=1)

sc_out = pd.DataFrame(dict(pitcher=pid_all[m25], t=pd.to_datetime(t_all[m25]),
                           P90_raw=p90, P150_raw=p150, P90_recal=p90_rc, P150_recal=p150_rc,
                           gs_share=gs_share[m25]))
sc_out["rp_flag"] = (sc_out["gs_share"] <= 0.2).astype(int)
sc_out["rank_H150"] = sc_out.groupby("t")["P150_raw"].rank(ascending=False, method="first").astype(int)
alert = np.zeros(len(sc_out), dtype=int)
for d, g in sc_out.groupby("t"):
    order = g.sort_values("P150_raw", ascending=False)
    rp_top = order[order["rp_flag"] == 1].head(20).index
    rest = order.drop(rp_top).head(50 - len(rp_top)).index
    alert[sc_out.index.get_indexer(rp_top)] = 1
    alert[sc_out.index.get_indexer(rest)] = 1
sc_out["alert_q20"] = alert
path = OUT / "frozen_scores_2025_ts20260713.csv"
sc_out.to_csv(path, index=False, float_format="%.8f")
md5 = hashlib.md5(path.read_bytes()).hexdigest()
print(f"\n2025 windows scored: {len(sc_out):,} across {sc_out['t'].nunique()} dates; "
      f"alerts/date = {int(sc_out.groupby('t')['alert_q20'].sum().iloc[0])}")
print(f"WROTE {path.name}  md5 {md5}")
print(f"[t={time.time()-t0:.0f}s] done")
