"""P2: (1) discrete-time hazard landmark supermodel, (2) role-stratified stitch.

(1) HAZARD: expand each window into 30d person-period intervals s=0..4
    (t+30s, t+30(s+1)]; interval label from the live surgery dates; a pitcher
    leaves the risk set after an event (later intervals dropped). One PLAIN
    (unweighted) LR on {M_sa features + s}; per-window cumulative risk
    P(H=30(k+1)) = 1 - prod_{s<=k}(1 - h_s). Guarantees P_H90 <= P_H150.
    Pre-registration: adopt the FORM as canonical if paired deltas vs the
    M_sa binary models are not significantly negative (benefits: coherent
    multi-horizon output, one model, no double-counting of shared events).
(2) ROLE-STRATIFIED: separate plain LRs for SP (start_share>=0.5) and RP on
    fit set, stitch predicted probabilities; paired vs M_sa. One pre-registered
    re-test of the error-analysis probe (+0.029/+0.020, CI incl 0 on old base).

Evaluation: expanded test 2022-24, clustered bootstrap seed 0, event recall@k.
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
surg_by_pid = {int(p): np.sort(g["surg_date"].values) for p, g in tj.groupby("mlbamid_i")}
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

# interval labels s=0..4 (30d bins) from surgery dates; at-risk censoring
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
        lo = t + np.timedelta64(30 * s, "D")
        hi = t + np.timedelta64(30 * (s + 1), "D")
        if np.any((surgs > lo) & (surgs <= hi)):
            int_label[i, s] = 1
            fired = True
# consistency: cumulative(3 bins) == label_H90_B0 ; cumulative(5) == label_H150_B0
cum90 = (int_label[:, :3].sum(axis=1) > 0).astype(int)
cum150 = (int_label[:, :5].sum(axis=1) > 0).astype(int)
assert (cum90 == cohort["label_H90_B0"].values).all(), "interval labels != H90 label"
assert (cum150 == cohort["label_H150_B0"].values).all(), "interval labels != H150 label"
print("interval-label consistency vs label_H90/H150_B0 : OK")

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
te = fold == "test"
pid_test = pid_all[te]
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

# ---------------- baseline M_sa binary (balanced LR, frozen protocol) ----------------
KS = [10, 20, 50]
M_SA_ANCHOR = {90: 0.69203, 150: 0.70398}
base_probs = {}
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    sc = StandardScaler().fit(F.loc[fit_mask, M_SA])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F.loc[fit_mask, M_SA]), y[fit_mask])
    p = clf.predict_proba(sc.transform(F.loc[te, M_SA]))[:, 1]
    roc = roc_auc_score(y[te], p)
    assert abs(roc - M_SA_ANCHOR[H]) < 1e-4, f"M_sa gate fail H{H} {roc}"
    base_probs[H] = p
print("M_sa gates OK")

# ---------------- (1) hazard supermodel ----------------
# person-period expansion on the FIT set
Xf = F.loc[fit_mask, M_SA].values
fit_idx = np.where(fit_mask)[0]
rows_X, rows_s, rows_y = [], [], []
for j, i in enumerate(fit_idx):
    for s in range(S_MAX):
        if not at_risk[i, s]:
            break
        rows_X.append(Xf[j]); rows_s.append(s); rows_y.append(int(int_label[i, s]))
Xpp = np.column_stack([np.asarray(rows_X), np.asarray(rows_s, dtype=float)])
ypp = np.asarray(rows_y)
print(f"person-period fit rows: {len(ypp):,}  events {int(ypp.sum())}  (windows {len(fit_idx):,})")
scpp = StandardScaler().fit(Xpp)
haz = LogisticRegression(max_iter=2000)  # PLAIN: calibrated hazards for the product
haz.fit(scpp.transform(Xpp), ypp)

# per test window: h_s for s=0..4 with the SAME as-of-t features
Xt = F.loc[te, M_SA].values
h = np.empty((Xt.shape[0], S_MAX))
for s in range(S_MAX):
    Xs = np.column_stack([Xt, np.full(Xt.shape[0], float(s))])
    h[:, s] = haz.predict_proba(scpp.transform(Xs))[:, 1]
p_h = {90: 1.0 - np.prod(1.0 - h[:, :3], axis=1),
       150: 1.0 - np.prod(1.0 - h[:, :5], axis=1)}
coef = dict(zip(M_SA + ["s"], haz.coef_[0].round(3)))
print(f"hazard coefs: {coef}  intercept {haz.intercept_[0]:.3f}")

out = []
for H in (90, 150):
    y_te = cohort[f"label_H{H}_B0"].values.astype(int)[te]
    p1 = p_h[H]
    roc1 = roc_auc_score(y_te, p1)
    ra, rb, pa, pb, nv = paired(y_te, base_probs[H], p1, RES_FULL)
    dr = rb - ra; dp = pb - pa
    er = evrec(y_te, p1, KS)
    fl = ("  dROC-EXCL0" if excl0(*ci(dr)) else "") + ("  dPR-EXCL0" if excl0(*ci(dp)) else "")
    out.append(dict(H=H, model="hazard", roc=roc1, droc=float(np.median(dr)),
                    droc_lo=ci(dr)[0], droc_hi=ci(dr)[1],
                    dpr=float(np.median(dp)), dpr_lo=ci(dp)[0], dpr_hi=ci(dp)[1],
                    ev50=er[50][0]))
    print(f"H={H} HAZARD ROC {roc1:.5f} (base {roc_auc_score(y_te, base_probs[H]):.5f})  "
          f"dROC {np.median(dr):+.5f} [{ci(dr)[0]:+.5f},{ci(dr)[1]:+.5f}]  "
          f"dPR {np.median(dp):+.5f} [{ci(dp)[0]:+.5f},{ci(dp)[1]:+.5f}]  evrec@50 {er[50][0]}{fl}")
# coherence check
assert (p_h[90] <= p_h[150] + 1e-12).all()
print("coherence P_H90 <= P_H150 : OK  (binary models violate this on "
      f"{int((base_probs[90] > base_probs[150]).sum())}/{len(base_probs[90])} test windows)")

# ---------------- (2) role-stratified stitch ----------------
role_sp = (F["start_share"].values >= 0.5)
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]
    p1 = np.empty(int(te.sum()))
    for r in (True, False):
        m_fit = fit_mask & (role_sp == r)
        m_te_rows = np.where(role_sp[te] == r)[0]
        sc = StandardScaler().fit(F.loc[m_fit, M_SA])
        clf = LogisticRegression(max_iter=2000)  # plain: stitched probabilities comparable
        clf.fit(sc.transform(F.loc[m_fit, M_SA]), y[m_fit])
        p1[m_te_rows] = clf.predict_proba(sc.transform(F.loc[te, M_SA].iloc[m_te_rows]))[:, 1]
    roc1 = roc_auc_score(y_te, p1)
    ra, rb, pa, pb, nv = paired(y_te, base_probs[H], p1, RES_FULL)
    dr = rb - ra; dp = pb - pa
    er = evrec(y_te, p1, KS)
    fl = ("  dROC-EXCL0" if excl0(*ci(dr)) else "") + ("  dPR-EXCL0" if excl0(*ci(dp)) else "")
    out.append(dict(H=H, model="role_strat", roc=roc1, droc=float(np.median(dr)),
                    droc_lo=ci(dr)[0], droc_hi=ci(dr)[1],
                    dpr=float(np.median(dp)), dpr_lo=ci(dp)[0], dpr_hi=ci(dp)[1],
                    ev50=er[50][0]))
    print(f"H={H} ROLE-STRAT ROC {roc1:.5f}  dROC {np.median(dr):+.5f} "
          f"[{ci(dr)[0]:+.5f},{ci(dr)[1]:+.5f}]  dPR {np.median(dp):+.5f} "
          f"[{ci(dp)[0]:+.5f},{ci(dp)[1]:+.5f}]  evrec@50 {er[50][0]}{fl}")

pd.DataFrame(out).to_csv(SCR / "p2_hazard.csv", index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote p2_hazard.csv")
