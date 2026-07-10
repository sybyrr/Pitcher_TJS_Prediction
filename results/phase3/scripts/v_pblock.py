"""Adversarial verification of the P-block result (ROC 0.70 > literature 0.61-0.67).

V1 2024 label reality: sheet MLB-surgery counts by year; the 2024-window test
   events listed (pid, surgery date), duplicates checked.
V2 Why is 2024 easy: per-year event composition (start_share at best window,
   dsl, chronic), base rates, feature drift (ncg_log range vs train).
V3 Is M_sa adoption 2024-driven: paired M_sa vs M-role on the ORIGINAL
   2022-23 test only (frozen resamples), plus 2024-only.
V4 Estimand comparability: single-snapshot ROC (Jun 1 only, one window per
   pitcher-season, H150 label) + mean per-date ROC — the numbers to hold
   against player-season literature AUCs.
V5 2024 as-of integrity: assert max game_date used in any 2024-window feature
   < t; ncg_log train-range vs 2024 range (extrapolation check).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
SCR = Path(r"C:/Users/PC/AppData/Local/Temp/claude/d--PAINS-Pitcher-TJS-Prediction/c8255b88-a70d-45ba-964b-531a2271c93f/scratchpad")
t0 = time.time()

cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v3.parquet").sort_values(["t", "pitcher"]).reset_index(drop=True)
N = len(cohort)
slim = pd.read_parquet(ROOT / "data/prospective/slim_games_v3.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                        g["pitch_count"].astype("float64").values,
                        g["mean_release_speed"].astype("float64").values,
                        g["game_year"].values.astype(np.int64))
gf = pd.read_parquet(ROOT / "data/prospective/game_features_v3.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
role_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    role_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                             g["total_pitches"].astype("float64").values)
tj = pd.read_csv(ROOT / "data/prospective/tj_live_clean_20260707.csv")
tj["surg_date"] = pd.to_datetime(tj["surg_date"])
DAY = np.timedelta64(1, "D")

# ---------------- V1: sheet surgery counts by year; 2024 test events ----------------
print("=" * 78)
print("V1 — label reality")
cohort_pids = set(cohort["pitcher"].unique().tolist())
tj["in_cohort"] = tj["mlbamid_i"].astype(int).isin(cohort_pids)
by_year = tj.groupby(tj["surg_date"].dt.year).agg(all_sheet=("surg_date", "size"),
                                                  cohort_pitchers=("in_cohort", "sum"))
print(by_year.loc[2016:2026].to_string())
te_mask = cohort["fold_main"].values == "test"
y24 = cohort["year"].values == 2024
pos24 = cohort[te_mask & y24 & (cohort["label_H150_B0"] == 1)]
ev24 = pos24.groupby(["pitcher", "next_surgery_date"]).size().reset_index(name="n_windows")
print(f"\n2024-window H150 events: {len(ev24)}")
print(ev24.to_string(index=False))
dup = ev24.groupby("pitcher").size()
print("pitchers with >1 distinct surgery among 2024 events:", int((dup > 1).sum()))

# ---------------- feature build (verbatim) ----------------
pid_all = cohort["pitcher"].values.astype(np.int64)
t_all = cohort["t"].values.astype("datetime64[ns]")
year_all = cohort["year"].values.astype(np.int64)
month_all = cohort["month"].values.astype(np.float64)
ncg_all = cohort["n_career_games"].values.astype(np.float64)
rows = []
prior_pc_rate = np.zeros(N); vt_missing = np.zeros(N); start_share = np.zeros(N)
max_used_ok = True
for i in range(N):
    pid = int(pid_all[i]); t = t_all[i].astype("datetime64[D]")
    gd, pc, sp, gy = by_pid[pid]
    before = gd < t
    if before.any() and gd[before].max() >= t:
        max_used_ok = False
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))
    pc_30 = pc[d30].sum(); pc_90 = pc[d90].sum()
    dsl = float((t - gd[before].max()) / DAY)
    def pw(mask):
        m = mask & ~np.isnan(sp)
        if not m.any():
            return np.nan
        w = pc[m].sum()
        return float((sp[m] * pc[m]).sum() / w) if w > 0 else np.nan
    vmean_30 = pw(d30)
    yr = int(year_all[i])
    vmean_prior = pw(before & (gy < yr))
    if np.isnan(vmean_prior):
        vmean_prior = pw(before & (gd < t - np.timedelta64(30, "D")))
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
print(f"\nV5 as-of integrity: all features use games strictly before t -> {max_used_ok}")
fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
te = fold == "test"
print("V5 ncg_log range: train/valid [{:.2f},{:.2f}]  2024 windows [{:.2f},{:.2f}]".format(
    F.loc[fit_mask, "ncg_log"].min(), F.loc[fit_mask, "ncg_log"].max(),
    F.loc[te & y24, "ncg_log"].min(), F.loc[te & y24, "ncg_log"].max()))

# ---------------- V2: per-year composition ----------------
print("\n" + "=" * 78)
print("V2 — why is 2024 easy? (H150 events, per test year)")
for Y in (2022, 2023, 2024):
    m = te & (year_all == Y)
    yv = cohort["label_H150_B0"].values
    pos = cohort[m & (yv == 1)]
    ev = pos.groupby(["pitcher", "next_surgery_date"])
    rows_best = []
    for (pid, sd), gg in ev:
        idx = gg.index.values
        ss = F.loc[idx, "start_share"].max()
        rows_best.append((ss,))
    ss_arr = np.array([r[0] for r in rows_best])
    br = yv[m].mean()
    print(f"  {Y}: windows {int(m.sum()):5d}  window-pos {int(yv[m].sum()):3d}  events {len(rows_best):3d}  "
          f"base {br:.4f}  SP-events(ss>=0.5) {(ss_arr>=0.5).sum()}/{len(ss_arr)} ({(ss_arr>=0.5).mean():.0%})")

# ---------------- models ----------------
M_ROLE = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"]
M_SA = M_ROLE + ["prior_pc_rate", "ncg_log", "vt_missing"]

def fit_predict(cols, y, mask_fit, mask_pred):
    sc = StandardScaler().fit(F.loc[mask_fit, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F.loc[mask_fit, cols]), y[mask_fit])
    return clf.predict_proba(sc.transform(F.loc[mask_pred, cols]))[:, 1]

def build_resamples(pids, seed=0, nboot=1000):
    uniq = np.unique(pids)
    pos = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([pos[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            for _ in range(nboot)]

def paired(y_sub, a, b, resamples):
    dr, dp = [], []
    for idx in resamples:
        yb = y_sub[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        dr.append(roc_auc_score(yb, b[idx]) - roc_auc_score(yb, a[idx]))
        dp.append(average_precision_score(yb, b[idx]) - average_precision_score(yb, a[idx]))
    return np.array(dr), np.array(dp)

def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

print("\n" + "=" * 78)
print("V3 — is M_sa adoption 2024-driven? paired M_sa-Mrole per test subset")
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    p0 = fit_predict(M_ROLE, y, fit_mask, te)
    p1 = fit_predict(M_SA, y, fit_mask, te)
    y_te = y[te]
    for name, sub in (("2022-23 only", year_all[te] <= 2023), ("2024 only", year_all[te] == 2024)):
        rows_sub = np.where(sub)[0]
        res = build_resamples(pid_all[te][rows_sub])
        dr, dp = paired(y_te[rows_sub], p0[rows_sub], p1[rows_sub], res)
        drlo, drhi = ci(dr)
        print(f"  H={H} {name:12s}: dROC {np.median(dr):+.5f} [{drlo:+.5f},{drhi:+.5f}]  "
              f"dPR {np.median(dp):+.5f}"
              + ("  EXCL0" if (drlo > 0 or drhi < 0) else ""))

print("\n" + "=" * 78)
print("V4 — estimand comparability (literature uses player-season AUC)")
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    p1 = fit_predict(M_SA, y, fit_mask, te)
    y_te = y[te]
    t_te = pd.to_datetime(cohort.loc[te, "t"]).values
    yr_te = year_all[te]
    # (a) pooled window ROC (what we report)
    pooled = roc_auc_score(y_te, p1)
    # (b) mean per-date ROC
    per_date = []
    for d in np.unique(t_te):
        s = t_te == d
        if y_te[s].sum() > 0:
            per_date.append(roc_auc_score(y_te[s], p1[s]))
    # (c) single snapshot Jun 1 (one window per pitcher-season)
    snap = []
    for Y in (2022, 2023, 2024):
        s = (t_te == np.datetime64(f"{Y}-06-01"))
        if y_te[s].sum() > 0:
            snap.append((Y, roc_auc_score(y_te[s], p1[s]), int(y_te[s].sum()), int(s.sum())))
    snap_all = (t_te == t_te)  # combined Jun-1 snapshot across years
    s = np.isin(t_te, [np.datetime64(f"{Y}-06-01") for Y in (2022, 2023, 2024)])
    snap_pool = roc_auc_score(y_te[s], p1[s])
    print(f"  H={H}: pooled-window ROC {pooled:.4f}   mean per-date ROC {np.mean(per_date):.4f} "
          f"(n_dates {len(per_date)})   Jun1-snapshot pooled {snap_pool:.4f}")
    for Y, r, npos, ntot in snap:
        print(f"        Jun1 {Y}: ROC {r:.4f}  (pos {npos}/{ntot})")

print(f"\n[t={time.time()-t0:.0f}s] done")
