"""A0-3: hazard cluster/form sensitivities (spec v2, report-only).

vs canonical plain hazard (M_sa + scalar s, fit train+valid), mature test
(t+H<=2024-12-31), point ROC deltas:
  (i)   event-weighted fit: positive person-period rows weighted
        1 / (#fit landmarks in which that same surgery fires); negatives 1.
  (ii)  categorical-s baseline: scalar s -> 4 dummies (s=1..4, base s=0).
  (iii) cluster-level full-refit bootstrap: B=200 resamples of FIT pitchers
        (seed 0), refit hazard each time, ROC on the fixed mature test ->
        percentile CI reflecting fit-side cluster uncertainty.
  (iv)  one-landmark-per-surgery: keep each fit surgery's positive interval
        row only at the landmark closest to surgery (min surg-t); other
        landmarks keep their negative intervals only.
Feature build identical to v_codex.py (v4 data). Output: ../a0_sens.csv.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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
Xall = F[M_SA].values
print(f"[t={time.time()-t0:.0f}s] features built")

S_MAX = 5
int_label = np.zeros((N, S_MAX), dtype=np.int8)
at_risk = np.ones((N, S_MAX), dtype=bool)
surg_of = {}  # (window i, interval s) with int_label 1 -> surgery date
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
        hit = (surgs > lo) & (surgs <= hi)
        if np.any(hit):
            int_label[i, s] = 1; fired = True
            surg_of[(i, s)] = surgs[hit][0]

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
fit_idx = np.where(fit_mask)[0]

# person-period rows for the fit set
rX, rs, ry, rw = [], [], [], []
for i in fit_idx:
    for s in range(S_MAX):
        if not at_risk[i, s]:
            break
        rX.append(Xall[i]); rs.append(s); ry.append(int(int_label[i, s])); rw.append(i)
Xf = np.asarray(rX); sf = np.asarray(rs, dtype=float)
yf = np.asarray(ry); wf = np.asarray(rw)

def fit_hz(X, s, y, weight=None, cat_s=False):
    scol = pd.get_dummies(s.astype(int)).values[:, 1:].astype(float) if cat_s else s.reshape(-1, 1)
    Xpp = np.hstack([X, scol])
    sc = StandardScaler().fit(Xpp)
    hz = LogisticRegression(max_iter=2000)
    hz.fit(sc.transform(Xpp), y, sample_weight=weight)
    return sc, hz

REL_END = np.datetime64("2024-12-31")
te_base = (fold == "test") & (year_all <= 2024)
test_masks = {H: te_base & ((t_all + np.timedelta64(H, "D")) <= REL_END) for H in (90, 150)}
test_X = {H: Xall[test_masks[H]] for H in (90, 150)}
test_y = {H: cohort[f"label_H{H}_B0"].values.astype(int)[test_masks[H]] for H in (90, 150)}

def test_roc(model, cat_s=False):
    sc, hz = model
    out = {}
    for H, smax in ((90, 3), (150, 5)):
        Xt = test_X[H]
        h = np.empty((Xt.shape[0], S_MAX))
        for s in range(S_MAX):
            scol = (np.eye(S_MAX)[s][1:] * np.ones((Xt.shape[0], S_MAX - 1))) if cat_s else np.full((Xt.shape[0], 1), float(s))
            h[:, s] = hz.predict_proba(sc.transform(np.hstack([Xt, scol])))[:, 1]
        p = 1 - np.prod(1 - h[:, :smax], axis=1)
        out[H] = roc_auc_score(test_y[H], p)
    return out

canon = fit_hz(Xf, sf, yf)
roc_c = test_roc(canon)
print(f"canonical: H90 {roc_c[90]:.4f}  H150 {roc_c[150]:.4f}")
out = [dict(variant="canonical", H90=round(roc_c[90], 4), H150=round(roc_c[150], 4),
            d90=0.0, d150=0.0, note="")]

# (i) event-weighted
n_lm = {}
for (i, s), sd in surg_of.items():
    if fit_mask[i]:
        n_lm[(int(pid_all[i]), sd)] = n_lm.get((int(pid_all[i]), sd), 0) + 1
weights = np.ones(len(yf))
pos_rows = np.where(yf == 1)[0]
for r in pos_rows:
    i, s = int(wf[r]), int(sf[r])
    weights[r] = 1.0 / n_lm[(int(pid_all[i]), surg_of[(i, s)])]
mdl = fit_hz(Xf, sf, yf, weight=weights)
r = test_roc(mdl)
out.append(dict(variant="event_weighted", H90=round(r[90], 4), H150=round(r[150], 4),
                d90=round(r[90] - roc_c[90], 4), d150=round(r[150] - roc_c[150], 4),
                note=f"pos rows {len(pos_rows)}, distinct fit surgeries {len(n_lm)}"))
print(f"event_weighted: H90 {r[90]:.4f} ({r[90]-roc_c[90]:+.4f})  H150 {r[150]:.4f} ({r[150]-roc_c[150]:+.4f})")

# (ii) categorical s
mdl = fit_hz(Xf, sf, yf, cat_s=True)
r = test_roc(mdl, cat_s=True)
out.append(dict(variant="categorical_s", H90=round(r[90], 4), H150=round(r[150], 4),
                d90=round(r[90] - roc_c[90], 4), d150=round(r[150] - roc_c[150], 4), note=""))
print(f"categorical_s: H90 {r[90]:.4f} ({r[90]-roc_c[90]:+.4f})  H150 {r[150]:.4f} ({r[150]-roc_c[150]:+.4f})")

# (iv) one landmark per surgery (closest to surgery)
best_lm = {}
for (i, s), sd in surg_of.items():
    if not fit_mask[i]:
        continue
    key = (int(pid_all[i]), sd)
    gap = (sd - t_all[i]) / np.timedelta64(1, "D")
    if key not in best_lm or gap < best_lm[key][1]:
        best_lm[key] = (i, gap)
keep_win = {v[0] for v in best_lm.values()}
drop = np.zeros(len(yf), dtype=bool)
for r_ in pos_rows:
    i, s = int(wf[r_]), int(sf[r_])
    if i not in keep_win:
        drop[r_] = True
mdl = fit_hz(Xf[~drop], sf[~drop], yf[~drop])
r = test_roc(mdl)
out.append(dict(variant="one_landmark_per_surgery", H90=round(r[90], 4), H150=round(r[150], 4),
                d90=round(r[90] - roc_c[90], 4), d150=round(r[150] - roc_c[150], 4),
                note=f"dropped {int(drop.sum())} of {len(pos_rows)} pos rows"))
print(f"one_landmark: H90 {r[90]:.4f} ({r[90]-roc_c[90]:+.4f})  H150 {r[150]:.4f} ({r[150]-roc_c[150]:+.4f})  "
      f"dropped {int(drop.sum())}/{len(pos_rows)} pos rows")

# (iii) cluster-level full-refit bootstrap, B=200, seed 0
pp_by_pid = {}
for r_, i in enumerate(wf):
    pp_by_pid.setdefault(int(pid_all[i]), []).append(r_)
pp_by_pid = {p: np.asarray(v) for p, v in pp_by_pid.items()}
fit_pids = np.unique(pid_all[fit_idx])
rng = np.random.default_rng(0)
B = 200
boot = {90: [], 150: []}
for b in range(B):
    draw = rng.choice(fit_pids, size=len(fit_pids), replace=True)
    rows_b = np.concatenate([pp_by_pid[p] for p in draw if p in pp_by_pid])
    if yf[rows_b].sum() == 0:
        continue
    mdl = fit_hz(Xf[rows_b], sf[rows_b], yf[rows_b])
    r = test_roc(mdl)
    boot[90].append(r[90]); boot[150].append(r[150])
    if (b + 1) % 50 == 0:
        print(f"  [t={time.time()-t0:.0f}s] refit bootstrap {b+1}/{B}")
for H in (90, 150):
    arr = np.asarray(boot[H])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    out.append(dict(variant=f"refit_boot_H{H}", H90=np.nan, H150=np.nan,
                    d90=np.nan, d150=np.nan,
                    note=f"mean {arr.mean():.4f} CI [{lo:.4f},{hi:.4f}] B={len(arr)}"))
    print(f"refit bootstrap H={H}: mean {arr.mean():.4f}  CI [{lo:.4f},{hi:.4f}]  (test-side canonical CI: "
          f"{'0.643-0.759' if H==90 else '0.645-0.746'})")

pd.DataFrame(out).to_csv(OUT / "a0_sens.csv", index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote {OUT / 'a0_sens.csv'}")
