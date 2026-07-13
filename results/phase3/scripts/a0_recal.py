"""A0-2: hazard-level joint recalibration via pitcher-grouped OOF (spec v2).

Primary: 5-fold pitcher-grouped cross-fit on the fit set (train 2017-20 +
valid 2021) -> OOF person-period hazards -> logistic recalibration
a + b*logit(h) at the INTERVAL level -> apply to the canonical full-fit
hazard on test: h~ = sigmoid(a + b*logit(h)), P~(H) = 1 - prod(1 - h~).
Preserves P90 <= P150 by construction (product over nested interval sets).
Sensitivities: (a) temporal leave-one-year-out cross-fit (same recal form);
(b) marginal per-H Platt on window-level OOF P(H) — reference only, may
break P90<=P150 (violation count reported).
Report on mature test (t+H<=2024-12-31) AND safety boundary (<=2024-06-30):
calibration-in-the-large, recal slope, Brier, decile reliability, ROC
before/after. Report-only; ranking metrics stay canonical.
Feature build identical to v_codex.py (v4 data). Output: ../a0_recal.csv.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score

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
fit_idx = np.where(fit_mask)[0]

def pp_rows(idx):
    """Person-period expansion for window indices idx -> (X, s, y, win_row)."""
    rX, rs, ry, rw = [], [], [], []
    Xall = F[M_SA].values
    for i in idx:
        for s in range(S_MAX):
            if not at_risk[i, s]:
                break
            rX.append(Xall[i]); rs.append(s); ry.append(int(int_label[i, s])); rw.append(i)
    return (np.asarray(rX), np.asarray(rs, dtype=float),
            np.asarray(ry), np.asarray(rw))

def fit_hz(X, s, y):
    Xpp = np.column_stack([X, s])
    sc = StandardScaler().fit(Xpp)
    hz = LogisticRegression(max_iter=2000)
    hz.fit(sc.transform(Xpp), y)
    return sc, hz

def hz_h(model, X, s):
    sc, hz = model
    return hz.predict_proba(sc.transform(np.column_stack([X, s])))[:, 1]

eps = 1e-12
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# ---- primary: pitcher-grouped 5-fold OOF at interval level ----
Xf, sf, yf, wf = pp_rows(fit_idx)
groups = pid_all[wf]
oof_h = np.full(len(yf), np.nan)
fold_models = []
gkf = GroupKFold(n_splits=5)
for tr, ho in gkf.split(Xf, yf, groups):
    mdl = fit_hz(Xf[tr], sf[tr], yf[tr])
    oof_h[ho] = hz_h(mdl, Xf[ho], sf[ho])
    fold_models.append((mdl, np.unique(wf[ho])))
assert not np.isnan(oof_h).any()
recal_grp = LogisticRegression(max_iter=2000).fit(logit(oof_h).reshape(-1, 1), yf)
a_grp, b_grp = float(recal_grp.intercept_[0]), float(recal_grp.coef_[0][0])
print(f"[t={time.time()-t0:.0f}s] grouped-OOF recal: a={a_grp:+.4f} b={b_grp:.4f}")

# ---- sensitivity (a): temporal leave-one-year-out OOF ----
oof_h_ty = np.full(len(yf), np.nan)
yrs_pp = year_all[wf]
for yr in np.unique(yrs_pp):
    tr = yrs_pp != yr; ho = ~tr
    mdl = fit_hz(Xf[tr], sf[tr], yf[tr])
    oof_h_ty[ho] = hz_h(mdl, Xf[ho], sf[ho])
recal_ty = LogisticRegression(max_iter=2000).fit(logit(oof_h_ty).reshape(-1, 1), yf)
a_ty, b_ty = float(recal_ty.intercept_[0]), float(recal_ty.coef_[0][0])
print(f"[t={time.time()-t0:.0f}s] temporal-LOYO recal: a={a_ty:+.4f} b={b_ty:.4f}")

# ---- canonical full-fit hazard, test predictions ----
canon = fit_hz(Xf, sf, yf)
def window_p(mask, recal=None):
    Xt = F.loc[mask, M_SA].values
    h = np.empty((Xt.shape[0], S_MAX))
    for s in range(S_MAX):
        h[:, s] = hz_h(canon, Xt, np.full(Xt.shape[0], float(s)))
    if recal is not None:
        a, b = recal
        h = 1 / (1 + np.exp(-(a + b * logit(h))))
    return {90: 1 - np.prod(1 - h[:, :3], axis=1), 150: 1 - np.prod(1 - h[:, :5], axis=1)}

# marginal per-H Platt on window-level OOF P (reference only). Window OOF P
# uses the fold model to predict ALL intervals s=0..4 (prediction time does
# not know censoring), matching the deployed window_p() computation.
oof_P = {90: np.full(N, np.nan), 150: np.full(N, np.nan)}
for mdl, ho_wins in fold_models:
    Xw = F[M_SA].values[ho_wins]
    h = np.empty((len(ho_wins), S_MAX))
    for s in range(S_MAX):
        h[:, s] = hz_h(mdl, Xw, np.full(len(ho_wins), float(s)))
    oof_P[90][ho_wins] = 1 - np.prod(1 - h[:, :3], axis=1)
    oof_P[150][ho_wins] = 1 - np.prod(1 - h[:, :5], axis=1)
platt = {}
for H in (90, 150):
    pw = oof_P[H][fit_idx]
    yw = cohort[f"label_H{H}_B0"].values.astype(int)[fit_idx]
    platt[H] = LogisticRegression(max_iter=2000).fit(logit(pw).reshape(-1, 1), yw)
    print(f"  marginal Platt H={H}: a={float(platt[H].intercept_[0]):+.4f} b={float(platt[H].coef_[0][0]):.4f}")

BOUNDS = {"mature_20241231": np.datetime64("2024-12-31"),
          "safety_20240630": np.datetime64("2024-06-30")}
te_base = (fold == "test") & (year_all <= 2024)

def calib_metrics(y, p):
    citl = p.mean() - y.mean()
    sl = LogisticRegression(max_iter=2000).fit(logit(p).reshape(-1, 1), y)
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    rel = pd.DataFrame({"d": dec, "p": p, "y": y}).groupby("d").agg(
        n=("y", "size"), mean_pred=("p", "mean"), obs=("y", "mean"))
    return dict(prev=y.mean(), mean_pred=p.mean(), citl=citl,
                slope=float(sl.coef_[0][0]), brier=brier_score_loss(y, p),
                roc=roc_auc_score(y, p)), rel

out = []
for bname, bend in BOUNDS.items():
    for H in (90, 150):
        m = te_base & ((t_all + np.timedelta64(H, "D")) <= bend)
        y = cohort[f"label_H{H}_B0"].values.astype(int)[m]
        p_raw = window_p(m)[H]
        p_grp = window_p(m, recal=(a_grp, b_grp))[H]
        p_ty = window_p(m, recal=(a_ty, b_ty))[H]
        for vname, p in (("raw", p_raw), ("recal_grouped", p_grp), ("recal_temporal", p_ty)):
            met, rel = calib_metrics(y, p)
            out.append(dict(boundary=bname, H=H, variant=vname, **{k: round(v, 6) for k, v in met.items()}))
            if vname != "raw" or bname == "mature_20241231":
                print(f"{bname} H={H} {vname:15s} prev {met['prev']:.4%} mean-pred {met['mean_pred']:.4%} "
                      f"slope {met['slope']:.3f} Brier {met['brier']:.6f} ROC {met['roc']:.4f}")
        if bname == "mature_20241231":
            _, rel = calib_metrics(y, p_grp)
            print(f"  reliability (grouped recal), deciles:\n{rel.round(5).to_string()}")
    # joint-consistency check on the H90-boundary window set
    m90 = te_base & ((t_all + np.timedelta64(90, "D")) <= bend)
    pj = window_p(m90, recal=(a_grp, b_grp))
    n_viol = int((pj[90] > pj[150] + 1e-12).sum())
    pj_platt90 = platt[90].predict_proba(logit(window_p(m90)[90]).reshape(-1, 1))[:, 1]
    pj_platt150 = platt[150].predict_proba(logit(window_p(m90)[150]).reshape(-1, 1))[:, 1]
    n_viol_platt = int((pj_platt90 > pj_platt150 + 1e-12).sum())
    print(f"{bname}: P90<=P150 violations — joint hazard recal {n_viol} / marginal Platt {n_viol_platt} "
          f"(of {int(m90.sum())})")
    out.append(dict(boundary=bname, H=0, variant="viol_joint", prev=np.nan, mean_pred=np.nan,
                    citl=np.nan, slope=np.nan, brier=np.nan, roc=n_viol))
    out.append(dict(boundary=bname, H=0, variant="viol_platt", prev=np.nan, mean_pred=np.nan,
                    citl=np.nan, slope=np.nan, brier=np.nan, roc=n_viol_platt))

pd.DataFrame(out).to_csv(OUT / "a0_recal.csv", index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote {OUT / 'a0_recal.csv'}")
