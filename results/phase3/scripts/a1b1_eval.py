"""A1+B1 evaluation on cohort_v4 (test 2022-25).

GATES: M-role on 2022-23 subset == frozen anchors; M_sa on 2022-24 subset ==
p0b anchors (0.69203/0.70398). Then:
  (1) new baseline: M-role / M_sa / hazard(M_sa) on test 2022-25
  (2) pre-registered paired tiers on M_sa:
      B1 vdecay (vdecay_30 pitch-weighted level [fit-median impute],
                 vdecay_dev = 30d - prior-season [0 impute])
      prior_tjs re-test (1f)
  (3) rolling-origin (Y in 2022..2025, fit years < Y) for any tier whose
      paired dROC or dPR CI excludes 0 upward.
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
vd = pd.read_parquet(ROOT / "data/prospective/vdecay_games_v4.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
vd_by_pid = {}
for pid, g in vd.groupby("pitcher", sort=False):
    vd_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                           g["n_fb"].astype("float64").values,
                           g["decay"].astype("float64").values,
                           pd.to_datetime(g["game_date"]).dt.year.values.astype(np.int64))
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
prior_tjs = np.zeros(N)
vdecay_30 = np.full(N, np.nan); vdecay_dev = np.full(N, np.nan)
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
    s = surg_by_pid.get(pid)
    if s is not None and (s < t.astype("datetime64[ns]")).any():
        prior_tjs[i] = 1.0
    V = vd_by_pid.get(pid)
    if V is not None:
        vgd, vn, vdc, vgy = V
        vb = vgd < t
        v30 = vb & (vgd >= t - np.timedelta64(30, "D"))
        m30 = pw_mean(vdc, vn, v30)
        base = pw_mean(vdc, vn, vb & (vgy < yr))
        if np.isnan(base):
            base = pw_mean(vdc, vn, vb & (vgd < t - np.timedelta64(30, "D")))
        vdecay_30[i] = m30
        if not (np.isnan(m30) or np.isnan(base)):
            vdecay_dev[i] = m30 - base

F = pd.DataFrame(rows, columns=["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"])
F["start_share"] = start_share
F["prior_pc_rate"] = prior_pc_rate
F["ncg_log"] = np.log1p(ncg_all)
F["vt_missing"] = vt_missing
F["prior_tjs"] = prior_tjs
F["vdecay_30"] = vdecay_30
F["vdecay_dev"] = vdecay_dev
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
te = fold == "test"
nan30 = np.isnan(F["vdecay_30"].values); nandev = np.isnan(F["vdecay_dev"].values)
print(f"[t={time.time()-t0:.0f}s] features built  vdecay_30 NaN {nan30.mean():.3f}  vdecay_dev NaN {nandev.mean():.3f}")
F["vdecay_30"] = F["vdecay_30"].fillna(float(np.nanmedian(F.loc[fit_mask, "vdecay_30"])))
F["vdecay_dev"] = F["vdecay_dev"].fillna(0.0)

pid_test = pid_all[te]
year_test = year_all[te]
t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
next_surg = pd.to_datetime(cohort["next_surgery_date"]).values

def build_resamples(pids, seed=0, nboot=1000):
    uniq = np.unique(pids)
    pos = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([pos[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            for _ in range(nboot)]

RES_FULL = build_resamples(pid_test)

def paired(y_sub, a, b, resamples):
    ra, rb, pa, pb = [], [], [], []
    for idx in resamples:
        yb = y_sub[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ra.append(roc_auc_score(yb, a[idx])); rb.append(roc_auc_score(yb, b[idx]))
        pa.append(average_precision_score(yb, a[idx])); pb.append(average_precision_score(yb, b[idx]))
    return np.array(ra), np.array(rb), np.array(pa), np.array(pb), len(ra)

def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

def excl0(lo, hi):
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)

def topk(score, k):
    flag = np.zeros(len(score), dtype=bool)
    for d in test_dates:
        sel = np.where(t_test == d)[0]
        kk = min(k, len(sel))
        flag[sel[np.argsort(-score[sel], kind="stable")[:kk]]] = True
    return flag

def evrec(y_sub, score, ks):
    pr = np.where(y_sub == 1)[0]
    keys = list(zip(pid_test[pr].tolist(),
                    pd.to_datetime(next_surg[te][pr]).astype("datetime64[ns]").tolist()))
    groups = {}
    for r, k in zip(pr, keys):
        groups.setdefault(k, []).append(r)
    out = {}
    for k in ks:
        fl = topk(score, k)
        out[k] = (sum(1 for rs in groups.values() if fl[rs].any()), len(groups))
    return out

def fit_predict(cols, y, mask_fit, mask_pred, balanced=True):
    sc = StandardScaler().fit(F.loc[mask_fit, cols])
    clf = LogisticRegression(class_weight="balanced" if balanced else None, max_iter=2000)
    clf.fit(sc.transform(F.loc[mask_fit, cols]), y[mask_fit])
    return clf.predict_proba(sc.transform(F.loc[mask_pred, cols]))[:, 1], clf

M_ROLE = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"]
M_SA = M_ROLE + ["prior_pc_rate", "ncg_log", "vt_missing"]
ANCHOR_ROLE = {90: (0.643680, 0.035973), 150: (0.643775, 0.046995)}
ANCHOR_SA = {90: 0.69203, 150: 0.70398}
KS = [10, 20, 50]
out = []

# interval labels for hazard (30d bins s=0..4)
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
assert ((int_label[:, :3].sum(axis=1) > 0).astype(int) == cohort["label_H90_B0"].values).all()
assert ((int_label[:, :5].sum(axis=1) > 0).astype(int) == cohort["label_H150_B0"].values).all()

def hazard_probs(cols):
    Xf = F.loc[fit_mask, cols].values
    fit_idx = np.where(fit_mask)[0]
    rX, rs, ry = [], [], []
    for j, i in enumerate(fit_idx):
        for s in range(S_MAX):
            if not at_risk[i, s]:
                break
            rX.append(Xf[j]); rs.append(s); ry.append(int(int_label[i, s]))
    Xpp = np.column_stack([np.asarray(rX), np.asarray(rs, dtype=float)])
    scpp = StandardScaler().fit(Xpp)
    hz = LogisticRegression(max_iter=2000)
    hz.fit(scpp.transform(Xpp), np.asarray(ry))
    Xt = F.loc[te, cols].values
    h = np.empty((Xt.shape[0], S_MAX))
    for s in range(S_MAX):
        h[:, s] = hz.predict_proba(scpp.transform(np.column_stack([Xt, np.full(Xt.shape[0], float(s))])))[:, 1]
    return {90: 1 - np.prod(1 - h[:, :3], axis=1), 150: 1 - np.prod(1 - h[:, :5], axis=1)}

probs_sa = {}
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]
    p_role, _ = fit_predict(M_ROLE, y, fit_mask, te)
    sub23 = year_test <= 2023
    g_roc = roc_auc_score(y_te[sub23], p_role[sub23]); g_pr = average_precision_score(y_te[sub23], p_role[sub23])
    assert abs(g_roc - ANCHOR_ROLE[H][0]) < 1e-4 and abs(g_pr - ANCHOR_ROLE[H][1]) < 1e-4, "M-role GATE FAIL"
    p_sa, _ = fit_predict(M_SA, y, fit_mask, te)
    sub24 = year_test <= 2024
    g2 = roc_auc_score(y_te[sub24], p_sa[sub24])
    assert abs(g2 - ANCHOR_SA[H]) < 1e-4, f"M_sa GATE FAIL {g2}"
    probs_sa[H] = p_sa
    print(f"H={H} GATES OK (M-role 22-23, M_sa 22-24)")

hz_sa = hazard_probs(M_SA)
print(f"\n{'='*78}\n(1) NEW BASELINE — test 2022-25")
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]
    for name, p in (("M_sa", probs_sa[H]), ("hazard", hz_sa[H])):
        roc = roc_auc_score(y_te, p); pr = average_precision_score(y_te, p)
        ra, _, pa, _, nv = paired(y_te, p, p, RES_FULL)
        er = evrec(y_te, p, KS)
        ys = {yr: roc_auc_score(y_te[year_test == yr], p[year_test == yr]) for yr in (2022, 2023, 2024, 2025)}
        out.append(dict(H=H, model=name, roc=roc, roc_lo=ci(ra)[0], roc_hi=ci(ra)[1], pr=pr,
                        ev50=er[50][0], ev_total=er[50][1],
                        y22=ys[2022], y23=ys[2023], y24=ys[2024], y25=ys[2025]))
        print(f"  H={H} {name:7s} ROC {roc:.5f} [{ci(ra)[0]:.4f},{ci(ra)[1]:.4f}]  PR {pr:.5f}  "
              f"evrec@50 {er[50][0]}/{er[50][1]}  years {ys[2022]:.3f}/{ys[2023]:.3f}/{ys[2024]:.3f}/{ys[2025]:.3f}")

print(f"\n(2) PRE-REGISTERED TIERS on M_sa (paired, test 2022-25)")
TIERS = [("B1_vdecay", M_SA + ["vdecay_30", "vdecay_dev"]),
         ("prior_tjs", M_SA + ["prior_tjs"]),
         ("both", M_SA + ["vdecay_30", "vdecay_dev", "prior_tjs"])]
winners = []
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]
    for name, cols in TIERS:
        p1, clf = fit_predict(cols, y, fit_mask, te)
        ra, rb, pa, pb, nv = paired(y_te, probs_sa[H], p1, RES_FULL)
        dr = rb - ra; dp = pb - pa
        drlo, drhi = ci(dr); dplo, dphi = ci(dp)
        er = evrec(y_te, p1, KS)
        fl = ("  dROC-EXCL0" if excl0(drlo, drhi) else "") + ("  dPR-EXCL0" if excl0(dplo, dphi) else "")
        if (drlo > 0) or (dplo > 0):
            winners.append((H, name, cols))
        newc = {c: round(float(v), 3) for c, v in zip(cols, clf.coef_[0]) if c not in M_SA}
        out.append(dict(H=H, model=name, roc=roc_auc_score(y_te, p1),
                        droc=float(np.median(dr)), droc_lo=drlo, droc_hi=drhi,
                        dpr=float(np.median(dp)), dpr_lo=dplo, dpr_hi=dphi,
                        ev50=er[50][0], ev_total=er[50][1]))
        print(f"  H={H} {name:10s} dROC {np.median(dr):+.5f} [{drlo:+.5f},{drhi:+.5f}]  "
              f"dPR {np.median(dp):+.5f} [{dplo:+.5f},{dphi:+.5f}]  evrec@50 {er[50][0]}  coef {newc}{fl}")

print(f"\n(3) ROLLING-ORIGIN for upward-EXCL0 tiers: {[(h, n) for h, n, _ in winners]}")
for H, name, cols in winners:
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    for Y in (2022, 2023, 2024, 2025):
        fit_m = year_all < Y
        te_m = year_all == Y
        pa0, _ = fit_predict(M_SA, y, fit_m, te_m)
        pa1, _ = fit_predict(cols, y, fit_m, te_m)
        yy = y[te_m]
        res_y = build_resamples(pid_all[te_m])
        ra, rb, pA, pB, nv = paired(yy, pa0, pa1, res_y)
        dr = rb - ra; dp = pB - pA
        print(f"  H={H} {name} Y={Y}: dROC {np.median(dr):+.4f} [{ci(dr)[0]:+.4f},{ci(dr)[1]:+.4f}]  "
              f"dPR {np.median(dp):+.4f}")

pd.DataFrame(out).to_csv(SCR / "a1b1_results.csv", index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote a1b1_results.csv")
