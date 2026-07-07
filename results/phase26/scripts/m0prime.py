"""M0' curated 5-feature model vs M0 8-feature reference.

Adapts results/phase26/scripts/b1_baseline_v2.py (frozen protocol) — feature
build + fit_lr + bootstrap + event recall + calibration are byte-identical so
M0 (the b1 'workload' model) reproduces the reference exactly.

M0  = 8 feats  : pc_30, pc_90, ng_30, ng_90, acwr, days_since_last, vel_trend, month
M0' = 5 feats  : pc_30, pc_90, days_since_last, vel_trend, month
     (drops ng_30, ng_90, acwr per the collinearity decision)

Paired deltas (M0' - M0) use the SAME 1000 pitcher resamples (seed 0). Since the
degenerate-resample skip depends only on y (shared labels), the surviving
resample set is identical for both models, so per-resample deltas are aligned.
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

# ---------------------------------------------------------------- load (identical to b1)
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

# ---------------------------------------------------------------- features (identical to b1)
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
FEATS_M0P = ["pc_30", "pc_90", "days_since_last", "vel_trend", "month"]
F = pd.DataFrame(feat_rows, columns=FEATS)
print(f"[t={time.time()-t0:.1f}s] features built  shape {F.shape}")

dsl_diff = np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float))
assert dsl_diff.max() < 1e-6, f"days_since_last != dsl, max diff {dsl_diff.max()}"
print("VERIFY days_since_last == cohort.dsl : OK (max diff %.2e)" % dsl_diff.max())

# ---------------------------------------------------------------- folds (identical to b1)
fold = cohort["fold_main"].values
tr = fold == "train"; va = fold == "valid"; te = fold == "test"
fit_mask = tr | va
print(f"fold_main sizes: train {tr.sum()}  valid {va.sum()}  test {te.sum()}  fit(tr+va) {fit_mask.sum()}")

t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
pid_test = pid_all[te]
uniq_test_pid = np.unique(pid_test)

# ---------------------------------------------------------------- bootstrap resamples (identical to b1)
rng = np.random.default_rng(0)
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

null_rng = np.random.default_rng(0)
null_scores = null_rng.random(te.sum())

# ---------------------------------------------------------------- helpers (b1 + paired)
def paired_boot(y_te, score_a, score_b):
    """Paired PR/ROC over shared resamples. a=M0, b=M0'. Returns arrays + count.
    Skip depends only on y so surviving set is identical for both models."""
    pr_a, roc_a, pr_b, roc_b = [], [], [], []
    for idx in boot_index_sets:
        yb = y_te[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        pr_a.append(average_precision_score(yb, score_a[idx]))
        roc_a.append(roc_auc_score(yb, score_a[idx]))
        pr_b.append(average_precision_score(yb, score_b[idx]))
        roc_b.append(roc_auc_score(yb, score_b[idx]))
    return (np.asarray(pr_a), np.asarray(roc_a),
            np.asarray(pr_b), np.asarray(roc_b), len(pr_a))

def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def prec_at_k_per_date(y_te, score_te, t_te_dates, k):
    vals = []
    for d in test_dates:
        sel = t_te_dates == d
        ys = y_te[sel]; ss = score_te[sel]
        kk = min(k, len(ys))
        order = np.argsort(-ss, kind="stable")[:kk]
        vals.append(float(ys[order].sum()) / kk)
    return float(np.mean(vals))

def topk_flags(score_te, t_te_dates, k):
    flag = np.zeros(len(score_te), dtype=bool)
    for d in test_dates:
        sel = np.where(t_te_dates == d)[0]
        ss = score_te[sel]
        kk = min(k, len(sel))
        top = sel[np.argsort(-ss, kind="stable")[:kk]]
        flag[top] = True
    return flag

def calibration(y_te, logit_te):
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

next_surg = pd.to_datetime(cohort["next_surgery_date"]).values
t_test_dates = t_test.astype("datetime64[D]")

def event_recall(y_te, score_te, ks):
    pos_rows = np.where(y_te == 1)[0]
    pid_pos = pid_test[pos_rows]
    surg_pos = next_surg[te][pos_rows]
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

# ---------------------------------------------------------------- MAIN cells
KS = [10, 20, 50]
results = []
REF = {(90): (0.023421, 0.639691), (150): (0.032400, 0.634193)}
exp_pos = {90: (138, 35, 118), 150: (205, 47, 162)}
repro_ok = {}
coef_store = {}

for H in [90, 150]:
    B = 0
    col = f"label_H{H}_B{B}"
    y = cohort[col].values.astype(int)
    y_te = y[te]; y_fit = y[fit_mask]
    p_tr, p_va, p_te = int(y[tr].sum()), int(y[va].sum()), int(y_te.sum())
    et, ev, ee = exp_pos[H]
    assert (p_tr, p_va, p_te) == (et, ev, ee), f"pos mismatch {col}"
    base_rate = float(y_te.mean())
    print(f"\n{'='*76}\nH={H} B={B}  positives {p_tr}/{p_va}/{p_te}  base_rate {base_rate:.5f}")

    # fit both models
    clf0, prob0, logit0 = fit_lr(FEATS, y_fit)           # M0
    clf1, prob1, logit1 = fit_lr(FEATS_M0P, y_fit)        # M0'
    pr0 = average_precision_score(y_te, prob0); roc0 = roc_auc_score(y_te, prob0)
    pr1 = average_precision_score(y_te, prob1); roc1 = roc_auc_score(y_te, prob1)

    # reproduction check
    ref_pr, ref_roc = REF[H]
    ok = abs(pr0 - ref_pr) < 5e-6 and abs(roc0 - ref_roc) < 5e-6
    repro_ok[H] = (ok, pr0, roc0, ref_pr, ref_roc)
    print(f"  M0 REPRO: PR {pr0:.6f} (ref {ref_pr:.6f})  ROC {roc0:.6f} (ref {ref_roc:.6f})  "
          f"{'MATCH' if ok else 'MISMATCH'}")

    # paired bootstrap
    a_pr, a_roc, b_pr, b_roc, nvalid = paired_boot(y_te, prob0, prob1)
    pr0_lo, pr0_hi = ci(a_pr); roc0_lo, roc0_hi = ci(a_roc)
    pr1_lo, pr1_hi = ci(b_pr); roc1_lo, roc1_hi = ci(b_roc)
    d_pr = b_pr - a_pr; d_roc = b_roc - a_roc
    dpr_lo, dpr_hi = ci(d_pr); droc_lo, droc_hi = ci(d_roc)
    dpr_mean = float(d_pr.mean()); droc_mean = float(d_roc.mean())
    print(f"  M0   PR {pr0:.6f} [{pr0_lo:.4f},{pr0_hi:.4f}]  ROC {roc0:.6f} [{roc0_lo:.4f},{roc0_hi:.4f}]")
    print(f"  M0'  PR {pr1:.6f} [{pr1_lo:.4f},{pr1_hi:.4f}]  ROC {roc1:.6f} [{roc1_lo:.4f},{roc1_hi:.4f}]")
    print(f"  dPR (M0'-M0)  point {pr1-pr0:+.6f}  boot-mean {dpr_mean:+.6f}  CI [{dpr_lo:+.5f},{dpr_hi:+.5f}]")
    print(f"  dROC(M0'-M0)  point {roc1-roc0:+.6f}  boot-mean {droc_mean:+.6f}  CI [{droc_lo:+.5f},{droc_hi:+.5f}]")
    print(f"  nboot valid {nvalid}")

    # event recall + prec@k + calibration
    ev0 = event_recall(y_te, prob0, KS); ev1 = event_recall(y_te, prob1, KS)
    cs0, cin0 = calibration(y_te, logit0); cs1, cin1 = calibration(y_te, logit1)
    print(f"  event recall M0 : {ev0[10][0]}/{ev0[10][1]} @10  {ev0[20][0]} @20  {ev0[50][0]} @50")
    print(f"  event recall M0': {ev1[10][0]}/{ev1[10][1]} @10  {ev1[20][0]} @20  {ev1[50][0]} @50")
    print(f"  calibration M0' : slope {cs1:.4f}  intercept {cin1:.4f}")

    coef_store[H] = dict(zip(FEATS_M0P, clf1.coef_[0]))
    print(f"  M0' std coef: " + "  ".join(f"{k}={v:+.4f}" for k, v in coef_store[H].items()))

    for tag, model, pr, roc, prlo, prhi, roclo, rochi, evd, cs, cin in [
        ("M0", "workload8", pr0, roc0, pr0_lo, pr0_hi, roc0_lo, roc0_hi, ev0, cs0, cin0),
        ("M0prime", "curated5", pr1, roc1, pr1_lo, pr1_hi, roc1_lo, roc1_hi, ev1, cs1, cin1),
    ]:
        results.append(dict(section="main", H=H, B=B, model=tag, featset=model,
            n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
            pr_auc=pr, pr_lo=prlo, pr_hi=prhi, roc_auc=roc, roc_lo=roclo, roc_hi=rochi,
            prec_at_10=prec_at_k_per_date(y_te, prob0 if tag=="M0" else prob1, t_test_dates, 10),
            prec_at_20=prec_at_k_per_date(y_te, prob0 if tag=="M0" else prob1, t_test_dates, 20),
            prec_at_50=prec_at_k_per_date(y_te, prob0 if tag=="M0" else prob1, t_test_dates, 50),
            evrec_total=evd[10][1], evrec_10=evd[10][0], evrec_20=evd[20][0], evrec_50=evd[50][0],
            cal_slope=cs, cal_intercept=cin,
            dpr_point=np.nan, dpr_lo=np.nan, dpr_hi=np.nan,
            droc_point=np.nan, droc_lo=np.nan, droc_hi=np.nan, nboot=nvalid))
    # paired-delta row
    results.append(dict(section="paired_delta", H=H, B=B, model="M0prime_minus_M0", featset="",
        n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
        pr_auc=np.nan, pr_lo=np.nan, pr_hi=np.nan, roc_auc=np.nan, roc_lo=np.nan, roc_hi=np.nan,
        prec_at_10=np.nan, prec_at_20=np.nan, prec_at_50=np.nan,
        evrec_total=np.nan, evrec_10=np.nan, evrec_20=np.nan, evrec_50=np.nan,
        cal_slope=np.nan, cal_intercept=np.nan,
        dpr_point=pr1-pr0, dpr_lo=dpr_lo, dpr_hi=dpr_hi,
        droc_point=roc1-roc0, droc_lo=droc_lo, droc_hi=droc_hi, nboot=nvalid))

# ---------------------------------------------------------------- VIF of the 5-feature set
print(f"\n{'='*76}\nVIF — M0' 5-feature set (fit set = train+valid, n={fit_mask.sum()})")
Wf5 = F.loc[fit_mask, FEATS_M0P]
vif5 = vif_table(Wf5)
print(vif5.round(2).sort_values(ascending=False).to_string())
cn5 = np.linalg.cond(((Wf5 - Wf5.mean()) / Wf5.std()).values)
print(f"condition number (standardized): {cn5:.2f}")
print("Pearson corr (M0' 5 feats):")
print(Wf5.corr().round(2).to_string())
for f in FEATS_M0P:
    results.append(dict(section="vif_m0prime", H=np.nan, B=np.nan, model=f, featset="curated5",
        n_test_windows=np.nan, n_test_pos=np.nan, base_rate=np.nan,
        pr_auc=float(vif5[f]), pr_lo=np.nan, pr_hi=np.nan, roc_auc=np.nan, roc_lo=np.nan, roc_hi=np.nan,
        prec_at_10=np.nan, prec_at_20=np.nan, prec_at_50=np.nan,
        evrec_total=np.nan, evrec_10=np.nan, evrec_20=np.nan, evrec_50=np.nan,
        cal_slope=np.nan, cal_intercept=np.nan,
        dpr_point=np.nan, dpr_lo=np.nan, dpr_hi=np.nan,
        droc_point=np.nan, droc_lo=np.nan, droc_hi=np.nan, nboot=np.nan))

# ---------------------------------------------------------------- BLACKOUT sweep (M0')
print(f"\n{'='*76}\nBLACKOUT SWEEP (M0' curated5)  H=150, B in {{0,30,60,90}}")
for H, B in [(150, 0), (150, 30), (150, 60), (150, 90)]:
    col = f"label_H{H}_B{B}"
    y = cohort[col].values.astype(int)
    y_te = y[te]; y_fit = y[fit_mask]
    p_te = int(y_te.sum()); base_rate = float(y_te.mean())
    clf, prob, logit = fit_lr(FEATS_M0P, y_fit)
    pr = average_precision_score(y_te, prob); roc = roc_auc_score(y_te, prob)
    a_pr, a_roc, b_pr, b_roc, nvalid = paired_boot(y_te, prob, prob)  # single model CI
    pr_lo, pr_hi = ci(a_pr); roc_lo, roc_hi = ci(a_roc)
    lift = pr / base_rate if base_rate > 0 else np.nan
    print(f"  H={H} B={B:2d}  pos {p_te:3d}  base {base_rate:.5f}  "
          f"PR {pr:.5f} [{pr_lo:.4f},{pr_hi:.4f}] (lift {lift:.2f}x)  "
          f"ROC {roc:.5f} [{roc_lo:.4f},{roc_hi:.4f}]")
    results.append(dict(section="blackout_m0prime", H=H, B=B, model="M0prime", featset="curated5",
        n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
        pr_auc=pr, pr_lo=pr_lo, pr_hi=pr_hi, roc_auc=roc, roc_lo=roc_lo, roc_hi=roc_hi,
        prec_at_10=np.nan, prec_at_20=np.nan, prec_at_50=np.nan,
        evrec_total=np.nan, evrec_10=np.nan, evrec_20=np.nan, evrec_50=np.nan,
        cal_slope=np.nan, cal_intercept=np.nan,
        dpr_point=np.nan, dpr_lo=np.nan, dpr_hi=np.nan,
        droc_point=np.nan, droc_lo=np.nan, droc_hi=np.nan, nboot=nvalid))

# ---------------------------------------------------------------- success criteria
print(f"\n{'='*76}\nSUCCESS CRITERIA")
crit_a = all(repro_ok[H][0] for H in [90, 150])
print(f"(a) M0 reproduction exact           : {'PASS' if crit_a else 'FAIL'}")
# criterion (b): paired |dROC| CI includes 0, both H
droc_rows = [r for r in results if r["section"] == "paired_delta"]
crit_b = all(r["droc_lo"] <= 0 <= r["droc_hi"] for r in droc_rows)
print(f"(b) paired dROC CI includes 0 (both H): {'PASS' if crit_b else 'FAIL'}")
for r in droc_rows:
    print(f"      H={r['H']}: dROC CI [{r['droc_lo']:+.5f},{r['droc_hi']:+.5f}]  "
          f"{'includes 0' if r['droc_lo']<=0<=r['droc_hi'] else 'EXCLUDES 0'}")
crit_c = float(vif5.max()) < 3.5
print(f"(c) max VIF(M0') < 3.5              : {'PASS' if crit_c else 'FAIL'}  (max {vif5.max():.3f})")

# ---------------------------------------------------------------- save
res = pd.DataFrame(results)
out_csv = SCR / "m0prime_results.csv"
res.to_csv(out_csv, index=False)
print(f"\n[t={time.time()-t0:.1f}s] saved {out_csv}  ({len(res)} rows)")
