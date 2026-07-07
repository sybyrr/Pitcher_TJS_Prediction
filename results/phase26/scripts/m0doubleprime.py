"""M0'' curated-orthogonal 5-feature model vs M0' and M0.

Adapts m0prime.py (which itself is byte-identical to results/phase26/scripts/
b1_baseline_v2.py on feature build + fit_lr + bootstrap + event recall). Same
frozen protocol: StandardScaler + LogisticRegression(class_weight='balanced',
max_iter=2000), fit on fold_main train+valid, test 2022-23, 1000 pitcher-
clustered bootstrap resamples (seed 0, shared across models for paired CIs).

M0   = 8 feats : pc_30, pc_90, ng_30, ng_90, acwr, days_since_last, vel_trend, month
M0'  = 5 feats : pc_30, pc_90, days_since_last, vel_trend, month
M0'' = 5 feats : pc_chronic, pc_acute_dev, days_since_last, vel_trend, month
   pc_chronic   = pc_90 / 90.0                 (trailing-90d daily pitch rate)
   pc_acute_dev = pc_30/30.0 - pc_90/90.0      (recent-30d rate minus chronic rate)

M0'' is a LINEAR reparametrization of the M0' workload pair: the map
(pc_30, pc_90) -> (pc_chronic, pc_acute_dev) has matrix [[0, 1/90],[1/30,-1/90]],
det = -1/2700 != 0, hence invertible. The two 5-feature sets span the same affine
subspace, so an *unpenalized* LR would give identical predictions. sklearn LR has
an L2 penalty (C=1) applied to standardized coefficients, so the penalty geometry
differs slightly between parametrizations -> deltas vs M0' expected ~0 (not exact).

Paired deltas vs M0' AND vs M0 use the SAME 1000 resamples (seed 0). Appends a new
`m0doubleprime*` section to m0prime_results.csv (does not overwrite prior rows).
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

# ---------------------------------------------------------------- load (identical to b1/m0prime)
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

# ---------------------------------------------------------------- features (identical to b1/m0prime)
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

# ---- M0'' reparametrized workload pair (same information, decorrelated) ----------
F["pc_chronic"] = F["pc_90"] / 90.0
F["pc_acute_dev"] = F["pc_30"] / 30.0 - F["pc_90"] / 90.0
FEATS_M0PP = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"]
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

# ---------------------------------------------------------------- helpers (b1 + paired)
def paired_boot(y_te, score_a, score_b):
    """Paired PR/ROC over shared resamples. delta reported downstream as b - a.
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

# ---------------------------------------------------------------- append-safe CSV setup
CSV = SCR / "m0prime_results.csv"
df_old = pd.read_csv(CSV)
COLS = list(df_old.columns)
def mkrow(**kw):
    r = {c: np.nan for c in COLS}
    r.update(kw)
    return r

# ---------------------------------------------------------------- MAIN cells
KS = [10, 20, 50]
new_rows = []
REF = {90: (0.023421, 0.639691), 150: (0.032400, 0.634193)}   # M0 reproduction anchors
exp_pos = {90: (138, 35, 118), 150: (205, 47, 162)}
repro_ok = {}
coef_store = {}
vif_pp = None
pearson_r = None

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

    # fit all three models
    clf0, prob0, logit0 = fit_lr(FEATS, y_fit)            # M0   (8)
    clf1, prob1, logit1 = fit_lr(FEATS_M0P, y_fit)        # M0'  (5)
    clf2, prob2, logit2 = fit_lr(FEATS_M0PP, y_fit)       # M0'' (5, reparam)
    pr0 = average_precision_score(y_te, prob0); roc0 = roc_auc_score(y_te, prob0)
    pr1 = average_precision_score(y_te, prob1); roc1 = roc_auc_score(y_te, prob1)
    pr2 = average_precision_score(y_te, prob2); roc2 = roc_auc_score(y_te, prob2)

    # M0 reproduction sanity (pipeline unchanged)
    ref_pr, ref_roc = REF[H]
    ok = abs(pr0 - ref_pr) < 5e-6 and abs(roc0 - ref_roc) < 5e-6
    repro_ok[H] = (ok, pr0, roc0)
    print(f"  M0 REPRO: PR {pr0:.6f} (ref {ref_pr:.6f})  ROC {roc0:.6f} (ref {ref_roc:.6f})  "
          f"{'MATCH' if ok else 'MISMATCH'}")

    # M0'' single-model CI (paired_boot vs itself gives the M0'' distribution in slot a)
    a_pr, a_roc, _b_pr, _b_roc, nvalid = paired_boot(y_te, prob2, prob2)
    pr2_lo, pr2_hi = ci(a_pr); roc2_lo, roc2_hi = ci(a_roc)

    # paired deltas vs M0' (b=M0'', a=M0')  -> delta = b - a = M0'' - M0'
    ap_pr, ap_roc, bp_pr, bp_roc, nvp = paired_boot(y_te, prob1, prob2)
    d_pr_vs_p = bp_pr - ap_pr; d_roc_vs_p = bp_roc - ap_roc
    dpr_p_lo, dpr_p_hi = ci(d_pr_vs_p); droc_p_lo, droc_p_hi = ci(d_roc_vs_p)

    # paired deltas vs M0 (b=M0'', a=M0)   -> delta = b - a = M0'' - M0
    a0_pr, a0_roc, b0_pr, b0_roc, nv0 = paired_boot(y_te, prob0, prob2)
    d_pr_vs_0 = b0_pr - a0_pr; d_roc_vs_0 = b0_roc - a0_roc
    dpr_0_lo, dpr_0_hi = ci(d_pr_vs_0); droc_0_lo, droc_0_hi = ci(d_roc_vs_0)

    assert nvalid == nvp == nv0, "resample survivor counts must match across paired comparisons"

    print(f"  M0'' PR {pr2:.6f} [{pr2_lo:.4f},{pr2_hi:.4f}]  ROC {roc2:.6f} [{roc2_lo:.4f},{roc2_hi:.4f}]")
    print(f"  vs M0'  dPR point {pr2-pr1:+.6f} CI [{dpr_p_lo:+.5f},{dpr_p_hi:+.5f}]  "
          f"dROC point {roc2-roc1:+.6f} CI [{droc_p_lo:+.5f},{droc_p_hi:+.5f}]"
          f"{'  <-- |dROC|>0.01 FLAG' if abs(roc2-roc1) > 0.01 else ''}")
    print(f"  vs M0   dPR point {pr2-pr0:+.6f} CI [{dpr_0_lo:+.5f},{dpr_0_hi:+.5f}]  "
          f"dROC point {roc2-roc0:+.6f} CI [{droc_0_lo:+.5f},{droc_0_hi:+.5f}]")
    print(f"  nboot valid {nvalid}")

    # event recall + prec@k + calibration for M0''
    ev2 = event_recall(y_te, prob2, KS)
    cs2, cin2 = calibration(y_te, logit2)
    print(f"  event recall M0'': {ev2[10][0]}/{ev2[10][1]} @10  {ev2[20][0]} @20  {ev2[50][0]} @50")
    print(f"  calibration M0'' : slope {cs2:.4f}  intercept {cin2:.4f}")

    coef_store[H] = dict(zip(FEATS_M0PP, clf2.coef_[0]))
    print(f"  M0'' std coef: " + "  ".join(f"{k}={v:+.4f}" for k, v in coef_store[H].items()))

    # --- CSV rows: main M0'' ---
    new_rows.append(mkrow(section="m0doubleprime", H=H, B=B, model="M0dprime", featset="curated_orth5",
        n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
        pr_auc=pr2, pr_lo=pr2_lo, pr_hi=pr2_hi, roc_auc=roc2, roc_lo=roc2_lo, roc_hi=roc2_hi,
        prec_at_10=prec_at_k_per_date(y_te, prob2, t_test_dates, 10),
        prec_at_20=prec_at_k_per_date(y_te, prob2, t_test_dates, 20),
        prec_at_50=prec_at_k_per_date(y_te, prob2, t_test_dates, 50),
        evrec_total=ev2[10][1], evrec_10=ev2[10][0], evrec_20=ev2[20][0], evrec_50=ev2[50][0],
        cal_slope=cs2, cal_intercept=cin2, nboot=nvalid))
    # --- paired delta vs M0' ---
    new_rows.append(mkrow(section="m0doubleprime_paired_delta", H=H, B=B,
        model="M0dprime_minus_M0prime", featset="vs_curated5",
        n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
        dpr_point=pr2-pr1, dpr_lo=dpr_p_lo, dpr_hi=dpr_p_hi,
        droc_point=roc2-roc1, droc_lo=droc_p_lo, droc_hi=droc_p_hi, nboot=nvalid))
    # --- paired delta vs M0 ---
    new_rows.append(mkrow(section="m0doubleprime_paired_delta", H=H, B=B,
        model="M0dprime_minus_M0", featset="vs_workload8",
        n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
        dpr_point=pr2-pr0, dpr_lo=dpr_0_lo, dpr_hi=dpr_0_hi,
        droc_point=roc2-roc0, droc_lo=droc_0_lo, droc_hi=droc_0_hi, nboot=nvalid))

# ---------------------------------------------------------------- VIF + condition number + pearson
print(f"\n{'='*76}\nVIF — M0'' 5-feature set (fit set = train+valid, n={fit_mask.sum()})")
Wpp = F.loc[fit_mask, FEATS_M0PP]
vif_pp = vif_table(Wpp)
print(vif_pp.round(3).sort_values(ascending=False).to_string())
cnpp = np.linalg.cond(((Wpp - Wpp.mean()) / Wpp.std()).values)
print(f"condition number (standardized): {cnpp:.3f}")
corr_pp = Wpp.corr()
pearson_r = float(corr_pp.loc["pc_chronic", "pc_acute_dev"])
print("Pearson corr (M0'' 5 feats):")
print(corr_pp.round(3).to_string())
print(f"Pearson r(pc_chronic, pc_acute_dev) on fit set: {pearson_r:+.4f}")

for f in FEATS_M0PP:
    new_rows.append(mkrow(section="m0doubleprime_vif", model=f, featset="curated_orth5",
        pr_auc=float(vif_pp[f])))
new_rows.append(mkrow(section="m0doubleprime_vif", model="cond_number", featset="curated_orth5",
    pr_auc=float(cnpp)))
new_rows.append(mkrow(section="m0doubleprime_pearson", model="pc_chronic__x__pc_acute_dev",
    featset="curated_orth5", pr_auc=pearson_r))

# ---------------------------------------------------------------- success criteria
print(f"\n{'='*76}\nSUCCESS CRITERIA")
crit_a = all(repro_ok[H][0] for H in [90, 150])
print(f"(a) M0 reproduction exact (pipeline sane) : {'PASS' if crit_a else 'FAIL'}")
crit_c = float(vif_pp.max()) < 3.5
print(f"(c) max VIF(M0'') < 3.5                    : {'PASS' if crit_c else 'FAIL'}  (max {vif_pp.max():.3f})")
# criterion: not detectably different from M0' -> both paired dROC & dPR CIs include 0
pd_rows = [r for r in new_rows if r["section"] == "m0doubleprime_paired_delta"
           and r["model"] == "M0dprime_minus_M0prime"]
crit_b = all((r["droc_lo"] <= 0 <= r["droc_hi"]) and (r["dpr_lo"] <= 0 <= r["dpr_hi"]) for r in pd_rows)
print(f"(b) M0'' not detectably != M0' (paired CIs incl 0): {'PASS' if crit_b else 'FAIL'}")
for r in pd_rows:
    print(f"      H={int(r['H'])}: dPR CI [{r['dpr_lo']:+.5f},{r['dpr_hi']:+.5f}]  "
          f"dROC CI [{r['droc_lo']:+.5f},{r['droc_hi']:+.5f}]  "
          f"point dROC {r['droc_point']:+.6f}"
          f"{'  |dROC|>0.01 FLAG' if abs(r['droc_point']) > 0.01 else ''}")

# ---------------------------------------------------------------- append + save
df_new = pd.DataFrame(new_rows, columns=COLS)
df_out = pd.concat([df_old, df_new], ignore_index=True)
df_out.to_csv(CSV, index=False)
print(f"\n[t={time.time()-t0:.1f}s] appended {len(df_new)} rows -> {CSV}  (total {len(df_out)})")
