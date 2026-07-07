"""TASK B1 — regularized LR baseline on cohort v2 + E4 evaluation suite.

Features per (pitcher, t) from slim_games strictly before t (X2 recipe + month).
Models per H in {90,150}, B=0, main folds: null / dsl_only LR / workload LR.
E4: PR-AUC, ROC-AUC with pitcher-clustered bootstrap 95% CI; precision@k per date;
event-level recall; calibration slope/intercept (workload). Blackout sweep over B.

Design decisions (documented):
- Fitting set = train+valid (fold_main in {train,valid}); no hyperparameter search.
  StandardScaler AND LogisticRegression both fit on that combined set (consistency).
- acwr = 0 when pc_90==0 (B1 spec; differs from X2's 1.0 fill).
- null = random-ranking baseline (uniform scores, seed 0) for ranking metrics;
  analytic PR-AUC = base rate, ROC-AUC = 0.5.
- Paired pitcher-clustered bootstrap: 1000 pitcher resamples (seed 0) reused across
  all models/metrics so CIs are comparable.
- Embargo: at H<=150 it removes zero windows (verified in cohort build), so main
  folds are used as-is.
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

# ---------------------------------------------------------------- load
cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v2.parquet")
cohort = cohort.sort_values(["t", "pitcher"]).reset_index(drop=True)
N = len(cohort)

slim = pd.read_parquet(SCR / "slim_games.parquet")
slim = slim.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (
        g["game_date"].values.astype("datetime64[D]"),
        g["pitch_count"].astype("float64").values,
        g["mean_release_speed"].astype("float64").values,
        g["game_year"].values.astype(np.int64),
    )

DAY = np.timedelta64(1, "D")

def pw_mean(mean_sp, pc, mask):
    m = mask & ~np.isnan(mean_sp)
    if not m.any():
        return np.nan
    wsum = pc[m].sum()
    if wsum <= 0:
        return np.nan
    return float((mean_sp[m] * pc[m]).sum() / wsum)

# ---------------------------------------------------------------- features
pid_all = cohort["pitcher"].values.astype(np.int64)
t_all = cohort["t"].values.astype("datetime64[ns]")
year_all = cohort["year"].values.astype(np.int64)
month_all = cohort["month"].values.astype(np.float64)

feat_rows = []
for i in range(N):
    pid = int(pid_all[i])
    t = t_all[i].astype("datetime64[D]")
    gd, pc, sp, gy = by_pid[pid]
    before = gd < t
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))

    pc_30 = pc[d30].sum(); pc_90 = pc[d90].sum()
    ng_30 = float(d30.sum()); ng_90 = float(d90.sum())
    acwr = (pc_30 / 30.0) / (pc_90 / 90.0) if pc_90 > 0 else 0.0

    last_gd = gd[before].max()
    days_since_last = float((t - last_gd) / DAY)

    vmean_30 = pw_mean(sp, pc, d30)
    yr = int(year_all[i])
    prior_season = before & (gy < yr)
    vmean_prior = pw_mean(sp, pc, prior_season)
    if np.isnan(vmean_prior):
        older = before & (gd < t - np.timedelta64(30, "D"))
        vmean_prior = pw_mean(sp, pc, older)
    vel_trend = (vmean_30 - vmean_prior) if not (np.isnan(vmean_30) or np.isnan(vmean_prior)) else 0.0

    feat_rows.append((pc_30, pc_90, ng_30, ng_90, acwr, days_since_last, vel_trend, month_all[i]))

FEATS = ["pc_30", "pc_90", "ng_30", "ng_90", "acwr", "days_since_last", "vel_trend", "month"]
F = pd.DataFrame(feat_rows, columns=FEATS)
print(f"[t={time.time()-t0:.1f}s] features built  shape {F.shape}")

# verify days_since_last == cohort dsl
dsl_diff = np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float))
assert dsl_diff.max() < 1e-6, f"days_since_last != dsl, max diff {dsl_diff.max()}"
print("VERIFY days_since_last == cohort.dsl : OK (max diff %.2e)" % dsl_diff.max())
print("\nfeature summary:")
print(F.describe().T[["mean", "std", "min", "50%", "max"]].to_string())

# ---------------------------------------------------------------- folds
fold = cohort["fold_main"].values
tr = fold == "train"
va = fold == "valid"
te = fold == "test"
fit_mask = tr | va                      # fitting set = train+valid
print(f"\nfold_main sizes: train {tr.sum()}  valid {va.sum()}  test {te.sum()}  fit(tr+va) {fit_mask.sum()}")

t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
print(f"test decision dates ({len(test_dates)}): {[str(np.datetime64(d,'D')) for d in test_dates]}")

pid_test = pid_all[te]
uniq_test_pid = np.unique(pid_test)
print(f"test windows {te.sum()}  unique test pitchers {len(uniq_test_pid)}")

# ---------------------------------------------------------------- bootstrap resamples (pitcher-clustered, seed 0)
rng = np.random.default_rng(0)
# map pitcher -> row positions within the test block (0..n_test-1)
test_idx = np.where(te)[0]
pos_in_test = {pid: np.where(pid_test == pid)[0] for pid in uniq_test_pid}
n_pit = len(uniq_test_pid)
BOOT = 1000
boot_index_sets = []
for _ in range(BOOT):
    drawn = rng.choice(uniq_test_pid, size=n_pit, replace=True)
    idx = np.concatenate([pos_in_test[p] for p in drawn])
    boot_index_sets.append(idx)
print(f"[t={time.time()-t0:.1f}s] built {BOOT} pitcher-clustered resamples")

# null random scores (seed 0) over the test block
null_rng = np.random.default_rng(0)
null_scores = null_rng.random(te.sum())

# ---------------------------------------------------------------- helpers
def boot_ci(y_te, score_te):
    """paired PR/ROC cluster-bootstrap CIs over precomputed pitcher resamples."""
    prs, rocs = [], []
    for idx in boot_index_sets:
        yb = y_te[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        prs.append(average_precision_score(yb, score_te[idx]))
        rocs.append(roc_auc_score(yb, score_te[idx]))
    prs = np.asarray(prs); rocs = np.asarray(rocs)
    return (float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5)),
            float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5)),
            len(prs))

def prec_at_k_per_date(y_te, score_te, t_te_dates, k):
    """mean over decision dates of precision among top-k scores at that date."""
    vals = []
    for d in test_dates:
        sel = t_te_dates == d
        ys = y_te[sel]; ss = score_te[sel]
        kk = min(k, len(ys))
        order = np.argsort(-ss, kind="stable")[:kk]
        vals.append(float(ys[order].sum()) / kk)
    return float(np.mean(vals))

def topk_flags(score_te, t_te_dates, k):
    """boolean per test window: is it in top-k of its own decision date."""
    flag = np.zeros(len(score_te), dtype=bool)
    for d in test_dates:
        sel = np.where(t_te_dates == d)[0]
        ss = score_te[sel]
        kk = min(k, len(sel))
        top = sel[np.argsort(-ss, kind="stable")[:kk]]
        flag[top] = True
    return flag

def calibration(y_te, logit_te):
    """logistic fit of y on model logit score; return (slope, intercept)."""
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(logit_te.reshape(-1, 1), y_te)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])

def fit_lr(cols, y_fit):
    sc = StandardScaler().fit(F.loc[fit_mask, cols])
    Xfit = sc.transform(F.loc[fit_mask, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(Xfit, y_fit)
    Xte = sc.transform(F.loc[te, cols])
    prob = clf.predict_proba(Xte)[:, 1]
    logit = clf.decision_function(Xte)
    return clf, prob, logit

# event-level recall setup: map positive test windows -> (pitcher, next_surgery_date)
next_surg = pd.to_datetime(cohort["next_surgery_date"]).values
t_test_dates = t_test.astype("datetime64[D]")

def event_recall(y_te, score_te, ks):
    """distinct test surgeries caught at budget k / total. B=0: surgery=next_surgery_date."""
    pos_rows = np.where(y_te == 1)[0]
    pid_pos = pid_test[pos_rows]
    surg_pos = next_surg[te][pos_rows]
    # group key (pitcher, surgery_date)
    keys = list(zip(pid_pos.tolist(), pd.to_datetime(surg_pos).astype("datetime64[ns]").tolist()))
    groups = {}
    for r, key in zip(pos_rows, keys):
        groups.setdefault(key, []).append(r)
    total = len(groups)
    out = {}
    for k in ks:
        flag = topk_flags(score_te, t_test_dates, k)
        caught = sum(1 for rows in groups.values() if flag[rows].any())
        out[k] = (caught, total)
    return out

# ---------------------------------------------------------------- MAIN: H in {90,150}, B=0
KS = [10, 20, 50]
results = []
exp_pos = {(90, 0): (138, 35, 118), (150, 0): (205, 47, 162)}

for H in [90, 150]:
    B = 0
    col = f"label_H{H}_B{B}"
    y = cohort[col].values.astype(int)
    y_te = y[te]; y_fit = y[fit_mask]
    # verify positive counts
    p_tr = int(y[tr].sum()); p_va = int(y[va].sum()); p_te = int(y_te.sum())
    et, ev, ee = exp_pos[(H, B)]
    ok = (p_tr, p_va, p_te) == (et, ev, ee)
    print(f"\n{'='*72}\nH={H} B={B}  positives train {p_tr} valid {p_va} test {p_te}  "
          f"expected {et}/{ev}/{ee}  {'OK' if ok else 'MISMATCH!!'}")
    assert ok, f"positive count mismatch for {col}"
    base_rate = float(y_te.mean())
    print(f"test base rate {base_rate:.5f}  ({p_te}/{te.sum()})")

    # ----- null -----
    ev_null = event_recall(y_te, null_scores, KS)
    row_null = dict(section="main", H=H, B=B, model="null",
                    n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
                    pr_auc=base_rate, pr_lo=np.nan, pr_hi=np.nan,
                    roc_auc=0.5, roc_lo=np.nan, roc_hi=np.nan,
                    prec_at_10=prec_at_k_per_date(y_te, null_scores, t_test_dates, 10),
                    prec_at_20=prec_at_k_per_date(y_te, null_scores, t_test_dates, 20),
                    prec_at_50=prec_at_k_per_date(y_te, null_scores, t_test_dates, 50),
                    evrec_total=ev_null[10][1],
                    evrec_10=ev_null[10][0], evrec_20=ev_null[20][0], evrec_50=ev_null[50][0],
                    cal_slope=np.nan, cal_intercept=np.nan)
    results.append(row_null)

    # ----- dsl_only + workload -----
    for name, cols in [("dsl_only", ["days_since_last"]), ("workload", FEATS)]:
        clf, prob, logit = fit_lr(cols, y_fit)
        pr = average_precision_score(y_te, prob)
        roc = roc_auc_score(y_te, prob)
        pr_lo, pr_hi, roc_lo, roc_hi, nvalid = boot_ci(y_te, prob)
        p10 = prec_at_k_per_date(y_te, prob, t_test_dates, 10)
        p20 = prec_at_k_per_date(y_te, prob, t_test_dates, 20)
        p50 = prec_at_k_per_date(y_te, prob, t_test_dates, 50)
        evr = event_recall(y_te, prob, KS)
        cslope, cint = calibration(y_te, logit)
        results.append(dict(section="main", H=H, B=B, model=name,
                            n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
                            pr_auc=pr, pr_lo=pr_lo, pr_hi=pr_hi,
                            roc_auc=roc, roc_lo=roc_lo, roc_hi=roc_hi,
                            prec_at_10=p10, prec_at_20=p20, prec_at_50=p50,
                            evrec_total=evr[10][1], evrec_10=evr[10][0],
                            evrec_20=evr[20][0], evrec_50=evr[50][0],
                            cal_slope=cslope, cal_intercept=cint))
        coef = dict(zip(cols, np.round(clf.coef_[0], 3)))
        print(f"  [{name:8s}] PR {pr:.4f} [{pr_lo:.4f}-{pr_hi:.4f}]  ROC {roc:.4f} [{roc_lo:.4f}-{roc_hi:.4f}]"
              f"  p@10/20/50 {p10:.3f}/{p20:.3f}/{p50:.3f}  evrec@50 {evr[50][0]}/{evr[50][1]}"
              f"  cal(slope,int) ({cslope:.3f},{cint:.3f})  nboot {nvalid}")
        if name == "workload":
            print(f"           coef: {coef}")

# ---------------------------------------------------------------- BLACKOUT SWEEP (workload)
print(f"\n{'='*72}\nBLACKOUT SWEEP (workload LR)")
blackout_cells = [(90, 0), (90, 30), (90, 60),
                  (150, 0), (150, 30), (150, 60), (150, 90)]
for H, B in blackout_cells:
    col = f"label_H{H}_B{B}"
    y = cohort[col].values.astype(int)
    y_te = y[te]; y_fit = y[fit_mask]
    p_te = int(y_te.sum()); base_rate = float(y_te.mean())
    clf, prob, logit = fit_lr(FEATS, y_fit)
    pr = average_precision_score(y_te, prob)
    roc = roc_auc_score(y_te, prob)
    pr_lo, pr_hi, roc_lo, roc_hi, nvalid = boot_ci(y_te, prob)
    lift = pr / base_rate if base_rate > 0 else np.nan
    results.append(dict(section="blackout", H=H, B=B, model="workload",
                        n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
                        pr_auc=pr, pr_lo=pr_lo, pr_hi=pr_hi,
                        roc_auc=roc, roc_lo=roc_lo, roc_hi=roc_hi,
                        prec_at_10=np.nan, prec_at_20=np.nan, prec_at_50=np.nan,
                        evrec_total=np.nan, evrec_10=np.nan, evrec_20=np.nan, evrec_50=np.nan,
                        cal_slope=np.nan, cal_intercept=np.nan))
    print(f"  H={H} B={B:2d}  pos {p_te:3d}  base {base_rate:.5f}  "
          f"PR {pr:.4f} [{pr_lo:.4f}-{pr_hi:.4f}] (lift {lift:.2f}x)  "
          f"ROC {roc:.4f} [{roc_lo:.4f}-{roc_hi:.4f}]")

# ---------------------------------------------------------------- save
res = pd.DataFrame(results)
out_csv = SCR / "baseline_v2_results.csv"
res.to_csv(out_csv, index=False)
print(f"\n[t={time.time()-t0:.1f}s] saved {out_csv}  ({len(res)} rows)")
print(res.to_string())
