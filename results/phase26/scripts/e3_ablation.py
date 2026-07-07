"""TASK E3 — additive feature ablation: within-pitcher TREND / VARIABILITY families.

Protocol frozen to b1_baseline_v2.py: same cohort ordering, fit set (train+valid),
StandardScaler+LR(class_weight='balanced', max_iter=2000), pitcher-clustered
bootstrap (1000 reps, seed 0), event recall at k in {10,20,50}, cells H in {90,150} B=0,
fold_main, test = 2022-23.

M0 workload (8 slim feats)  ->  reproduce reference exactly (H90 ROC .640/PR .0234;
H150 ROC .634/PR .0324) as a self-check.
M1 = M0 + TREND(6)+trend_missing ; M2 = M0 + VARIABILITY(6)+var_missing ; M3 = all.
GBM check: HistGradientBoostingClassifier on M3 features (one config, no tuning).

Features built at each (pitcher,t) from game_features_v2 using ONLY games game_date < t.
Cached to scratchpad/trendvar_features.parquet (keyed pitcher+t).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
SCR = Path(r"C:/Users/PC/AppData/Local/Temp/claude/d--PAINS-Pitcher-TJS-Prediction/c8255b88-a70d-45ba-964b-531a2271c93f/scratchpad")
t0 = time.time()

# ================================================================ load
cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v2.parquet")
cohort = cohort.sort_values(["t", "pitcher"]).reset_index(drop=True)   # SAME order as b1
N = len(cohort)

slim = pd.read_parquet(SCR / "slim_games.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                        g["pitch_count"].astype("float64").values,
                        g["mean_release_speed"].astype("float64").values,
                        g["game_year"].values.astype(np.int64))
DAY = np.timedelta64(1, "D")

def pw_mean(mean_sp, pc, mask):
    m = mask & ~np.isnan(mean_sp)
    if not m.any():
        return np.nan
    wsum = pc[m].sum()
    return float((mean_sp[m] * pc[m]).sum() / wsum) if wsum > 0 else np.nan

pid_all = cohort["pitcher"].values.astype(np.int64)
t_all = cohort["t"].values.astype("datetime64[ns]")
year_all = cohort["year"].values.astype(np.int64)
month_all = cohort["month"].values.astype(np.float64)

# ================================================================ M0 workload (exact b1 recipe)
feat_rows = []
for i in range(N):
    pid = int(pid_all[i]); t = t_all[i].astype("datetime64[D]")
    gd, pc, sp, gy = by_pid[pid]
    before = gd < t
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))
    pc_30 = pc[d30].sum(); pc_90 = pc[d90].sum()
    ng_30 = float(d30.sum()); ng_90 = float(d90.sum())
    acwr = (pc_30 / 30.0) / (pc_90 / 90.0) if pc_90 > 0 else 0.0
    days_since_last = float((t - gd[before].max()) / DAY)
    vmean_30 = pw_mean(sp, pc, d30)
    yr = int(year_all[i])
    vmean_prior = pw_mean(sp, pc, before & (gy < yr))
    if np.isnan(vmean_prior):
        vmean_prior = pw_mean(sp, pc, before & (gd < t - np.timedelta64(30, "D")))
    vel_trend = (vmean_30 - vmean_prior) if not (np.isnan(vmean_30) or np.isnan(vmean_prior)) else 0.0
    feat_rows.append((pc_30, pc_90, ng_30, ng_90, acwr, days_since_last, vel_trend, month_all[i]))

WL = ["pc_30", "pc_90", "ng_30", "ng_90", "acwr", "days_since_last", "vel_trend", "month"]
F = pd.DataFrame(feat_rows, columns=WL)
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6
print(f"[t={time.time()-t0:.1f}s] workload M0 features built {F.shape}")

# ================================================================ TREND / VARIABILITY per (pitcher,t)
gf = pd.read_parquet(ROOT / "data/prospective/game_features_v2.parquet")
gf = gf.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
GCOLS = ["total_pitches", "n_FF", "n_SI", "n_CU", "n_SL",
         "FF_velo_mean", "FF_spin_mean", "SL_velo_mean",
         "FF_velo_sd", "FF_relx_sd", "FF_ext_sd", "FF_relx_mean"]
gf_by = {}
for pid, g in gf.groupby("pitcher", sort=False):
    d = {c: g[c].values.astype("float64") for c in GCOLS}
    d["date"] = g["game_date"].values.astype("datetime64[D]")
    gf_by[int(pid)] = d

TREND = ["ff_velo_tau15", "sl_velo_tau15", "ff_spin_tau15", "ff_velo_sen5",
         "fb_usage_delta", "cu_usage_delta"]
VAR = ["ff_velo_sd_recent", "ff_velo_sd_rel", "ff_relx_sd_recent", "ff_relx_sd_rel",
       "ff_ext_sd_rel", "ff_relx_drift"]
TREND_IND14 = ["ff_velo_tau15", "sl_velo_tau15", "ff_spin_tau15", "ff_velo_sen5"]  # trend_missing = any of 1-4

def _tau(order, y):
    """kendall tau over non-nan pairs; (value, imputed_flag)."""
    m = ~np.isnan(y)
    if m.sum() < 2:
        return 0.0, True
    tau, _ = kendalltau(order[m], y[m])
    if np.isnan(tau):
        return 0.0, True
    return float(tau), False

def _mean(y):
    m = ~np.isnan(y)
    return float(y[m].mean()) if m.any() else np.nan

rows = []
for i in range(N):
    pid = int(pid_all[i]); t = t_all[i].astype("datetime64[D]")
    d = gf_by[pid]
    idx = np.searchsorted(d["date"], t, side="left")   # games strictly before t
    # ---- prefix slices (chronological) ----
    tp = d["total_pitches"][:idx]; nff = d["n_FF"][:idx]; nsi = d["n_SI"][:idx]
    ncu = d["n_CU"][:idx]; nsl = d["n_SL"][:idx]
    ff_v = d["FF_velo_mean"][:idx]; ff_sp = d["FF_spin_mean"][:idx]; sl_v = d["SL_velo_mean"][:idx]
    ff_vsd = d["FF_velo_sd"][:idx]; ff_xsd = d["FF_relx_sd"][:idx]
    ff_esd = d["FF_ext_sd"][:idx]; ff_xm = d["FF_relx_mean"][:idx]

    imp = {}   # per-feature imputed flag
    val = {}

    # ---- FF-qualifying (n_FF>=3) chronological subset ----
    qff = np.where(nff >= 3)[0]
    nq = len(qff)
    # 1 ff_velo_tau15  (>=8 qualifying, over last<=15)
    if nq >= 8:
        r15 = qff[-15:]
        val["ff_velo_tau15"], imp["ff_velo_tau15"] = _tau(np.arange(len(r15), dtype=float), ff_v[r15])
    else:
        val["ff_velo_tau15"], imp["ff_velo_tau15"] = 0.0, True
    # 3 ff_spin_tau15  (same FF recent15, >=8)
    if nq >= 8:
        r15 = qff[-15:]
        val["ff_spin_tau15"], imp["ff_spin_tau15"] = _tau(np.arange(len(r15), dtype=float), ff_sp[r15])
    else:
        val["ff_spin_tau15"], imp["ff_spin_tau15"] = 0.0, True
    # 4 ff_velo_sen5  (>=4 qualifying, Theil-Sen over last<=5, mph/game)
    if nq >= 4:
        r5 = qff[-5:]
        yv = ff_v[r5]; mm = ~np.isnan(yv)
        if mm.sum() >= 4:
            slope = theilslopes(yv[mm], np.arange(len(yv))[mm])[0]
            val["ff_velo_sen5"], imp["ff_velo_sen5"] = (0.0, True) if np.isnan(slope) else (float(slope), False)
        else:
            val["ff_velo_sen5"], imp["ff_velo_sen5"] = 0.0, True
    else:
        val["ff_velo_sen5"], imp["ff_velo_sen5"] = 0.0, True
    # 2 sl_velo_tau15  (SL qualifying n_SL>=3, >=8)
    qsl = np.where(nsl >= 3)[0]
    if len(qsl) >= 8:
        s15 = qsl[-15:]
        val["sl_velo_tau15"], imp["sl_velo_tau15"] = _tau(np.arange(len(s15), dtype=float), sl_v[s15])
    else:
        val["sl_velo_tau15"], imp["sl_velo_tau15"] = 0.0, True

    # ---- usage deltas over ALL prefix games; last15 vs prior (>=10 prior) ----
    n = idx
    if n >= 15:
        prior = slice(0, n - 15); rec = slice(n - 15, n)
        if (n - 15) >= 10:
            with np.errstate(invalid="ignore", divide="ignore"):
                fb_share = (nff + nsi) / tp
                cu_share = ncu / tp
            val["fb_usage_delta"] = float(np.nanmean(fb_share[rec]) - np.nanmean(fb_share[prior])); imp["fb_usage_delta"] = False
            val["cu_usage_delta"] = float(np.nanmean(cu_share[rec]) - np.nanmean(cu_share[prior])); imp["cu_usage_delta"] = False
        else:
            val["fb_usage_delta"], imp["fb_usage_delta"] = 0.0, True
            val["cu_usage_delta"], imp["cu_usage_delta"] = 0.0, True
    else:
        val["fb_usage_delta"], imp["fb_usage_delta"] = 0.0, True
        val["cu_usage_delta"], imp["cu_usage_delta"] = 0.0, True

    # ---- VARIABILITY (FF-qualifying window; baseline = qualifying before recent5, >=10) ----
    if nq >= 1:
        k5 = min(5, nq); rec5 = qff[nq - k5:]; base = qff[:nq - k5]
        val["ff_velo_sd_recent"] = _mean(ff_vsd[rec5]); imp["ff_velo_sd_recent"] = np.isnan(val["ff_velo_sd_recent"])
        val["ff_relx_sd_recent"] = _mean(ff_xsd[rec5]); imp["ff_relx_sd_recent"] = np.isnan(val["ff_relx_sd_recent"])
        # relative vs self-baseline (>=10 baseline games)
        if len(base) >= 10:
            b_v = _mean(ff_vsd[base]); b_x = _mean(ff_xsd[base]); b_e = _mean(ff_esd[base])
            rec_e = _mean(ff_esd[rec5])
            val["ff_velo_sd_rel"] = (val["ff_velo_sd_recent"] - b_v) if not (imp["ff_velo_sd_recent"] or np.isnan(b_v)) else 0.0
            imp["ff_velo_sd_rel"] = imp["ff_velo_sd_recent"] or np.isnan(b_v)
            val["ff_relx_sd_rel"] = (val["ff_relx_sd_recent"] - b_x) if not (imp["ff_relx_sd_recent"] or np.isnan(b_x)) else 0.0
            imp["ff_relx_sd_rel"] = imp["ff_relx_sd_recent"] or np.isnan(b_x)
            val["ff_ext_sd_rel"] = (rec_e - b_e) if not (np.isnan(rec_e) or np.isnan(b_e)) else 0.0
            imp["ff_ext_sd_rel"] = np.isnan(rec_e) or np.isnan(b_e)
        else:
            for k in ("ff_velo_sd_rel", "ff_relx_sd_rel", "ff_ext_sd_rel"):
                val[k], imp[k] = 0.0, True
        # relx drift: recent15 mean minus career-prior mean (>=10 prior)
        k15 = min(15, nq); rec15 = qff[nq - k15:]; cprior = qff[:nq - k15]
        if len(cprior) >= 10:
            rm = _mean(ff_xm[rec15]); pm = _mean(ff_xm[cprior])
            val["ff_relx_drift"] = (rm - pm) if not (np.isnan(rm) or np.isnan(pm)) else 0.0
            imp["ff_relx_drift"] = np.isnan(rm) or np.isnan(pm)
        else:
            val["ff_relx_drift"], imp["ff_relx_drift"] = 0.0, True
        # clean recent-only imputes to 0
        for k in ("ff_velo_sd_recent", "ff_relx_sd_recent"):
            if imp[k]:
                val[k] = 0.0
    else:
        for k in VAR:
            val[k], imp[k] = 0.0, True

    trend_missing = int(any(imp[k] for k in TREND_IND14))
    var_missing = int(any(imp[k] for k in VAR))
    row = {"pitcher": pid, "t": t_all[i]}
    for k in TREND + VAR:
        row[k] = val[k]
        row[k + "_imp"] = int(imp[k])
    row["trend_missing"] = trend_missing
    row["var_missing"] = var_missing
    rows.append(row)
    if (i + 1) % 5000 == 0:
        print(f"[t={time.time()-t0:.1f}s]  {i+1}/{N} windows")

TV = pd.DataFrame(rows)
# sanity: aligned to cohort order
assert (TV["pitcher"].values == pid_all).all()
assert (TV["t"].values.astype("datetime64[ns]") == t_all).all()
cache = SCR / "trendvar_features.parquet"
TV.to_parquet(cache)
print(f"[t={time.time()-t0:.1f}s] TREND/VAR built + cached -> {cache}  ({TV.shape})")

# attach to F
for c in TREND + VAR + ["trend_missing", "var_missing"]:
    F[c] = TV[c].values

# ================================================================ folds / bootstrap (exact b1)
fold = cohort["fold_main"].values
tr = fold == "train"; va = fold == "valid"; te = fold == "test"
fit_mask = tr | va
pid_test = pid_all[te]
uniq_test_pid = np.unique(pid_test)
pos_in_test = {pid: np.where(pid_test == pid)[0] for pid in uniq_test_pid}
n_pit = len(uniq_test_pid)
rng = np.random.default_rng(0)
BOOT = 1000
boot_index_sets = []
for _ in range(BOOT):
    drawn = rng.choice(uniq_test_pid, size=n_pit, replace=True)
    boot_index_sets.append(np.concatenate([pos_in_test[p] for p in drawn]))
print(f"[t={time.time()-t0:.1f}s] {BOOT} pitcher-clustered resamples; test windows {te.sum()} pitchers {n_pit}")

t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
t_test_dates = t_test.astype("datetime64[D]")
next_surg = pd.to_datetime(cohort["next_surgery_date"]).values

def boot_ci(y_te, score):
    prs, rocs = [], []
    for idx in boot_index_sets:
        yb = y_te[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        prs.append(average_precision_score(yb, score[idx]))
        rocs.append(roc_auc_score(yb, score[idx]))
    prs = np.asarray(prs); rocs = np.asarray(rocs)
    return (float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5)),
            float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5)))

def paired_delta(y_te, score_a, score_b):
    """bootstrap DIFFERENCE score_a - score_b over same pitcher resamples (a=model, b=M0)."""
    dpr, droc = [], []
    for idx in boot_index_sets:
        yb = y_te[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        dpr.append(average_precision_score(yb, score_a[idx]) - average_precision_score(yb, score_b[idx]))
        droc.append(roc_auc_score(yb, score_a[idx]) - roc_auc_score(yb, score_b[idx]))
    dpr = np.asarray(dpr); droc = np.asarray(droc)
    return (float(dpr.mean()), float(np.percentile(dpr, 2.5)), float(np.percentile(dpr, 97.5)),
            float(droc.mean()), float(np.percentile(droc, 2.5)), float(np.percentile(droc, 97.5)))

def topk_flags(score, k):
    flag = np.zeros(len(score), dtype=bool)
    for dte in test_dates:
        sel = np.where(t_test_dates == np.datetime64(dte, "D"))[0]
        kk = min(k, len(sel))
        top = sel[np.argsort(-score[sel], kind="stable")[:kk]]
        flag[top] = True
    return flag

def event_recall(y_te, score, ks):
    pos_rows = np.where(y_te == 1)[0]
    keys = list(zip(pid_test[pos_rows].tolist(),
                    pd.to_datetime(next_surg[te][pos_rows]).astype("datetime64[ns]").tolist()))
    groups = {}
    for r, key in zip(pos_rows, keys):
        groups.setdefault(key, []).append(r)
    out = {}
    for k in ks:
        flag = topk_flags(score, k)
        out[k] = (sum(1 for rr in groups.values() if flag[rr].any()), len(groups))
    return out

def fit_lr(cols, y_fit, y_te_unused):
    sc = StandardScaler().fit(F.loc[fit_mask, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F.loc[fit_mask, cols]), y_fit)
    return clf.predict_proba(sc.transform(F.loc[te, cols]))[:, 1]

def fit_gbm(cols, y_fit):
    w = compute_sample_weight("balanced", y_fit)
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1,
                                         early_stopping=False, random_state=0)
    clf.fit(F.loc[fit_mask, cols].values, y_fit, sample_weight=w)
    return clf.predict_proba(F.loc[te, cols].values)[:, 1]

# ================================================================ model feature sets
M = {
    "M0_workload": WL,
    "M1_trend": WL + TREND + ["trend_missing"],
    "M2_variability": WL + VAR + ["var_missing"],
    "M3_all": WL + TREND + ["trend_missing"] + VAR + ["var_missing"],
}
KS = [10, 20, 50]
REF = {90: (0.023420736514012663, 0.6396909297446568), 150: (0.032400299960994694, 0.6341933941415829)}

results = []
recall_rows = []
delta_rows = []
for H in [90, 150]:
    col = f"label_H{H}_B0"
    y = cohort[col].values.astype(int)
    y_te = y[te]; y_fit = y[fit_mask]
    base_rate = float(y_te.mean())
    print(f"\n{'='*74}\nH={H} B=0  test pos {int(y_te.sum())}/{te.sum()}  base {base_rate:.5f}")
    scores = {}
    # LR models
    for name, cols in M.items():
        prob = fit_lr(cols, y_fit, y_te)
        scores[name] = prob
        pr = average_precision_score(y_te, prob); roc = roc_auc_score(y_te, prob)
        pr_lo, pr_hi, roc_lo, roc_hi = boot_ci(y_te, prob)
        evr = event_recall(y_te, prob, KS)
        results.append(dict(H=H, model=name, n_feat=len(cols), pr_auc=pr, pr_lo=pr_lo, pr_hi=pr_hi,
                            roc_auc=roc, roc_lo=roc_lo, roc_hi=roc_hi, base_rate=base_rate,
                            evrec_total=evr[10][1], evrec_10=evr[10][0], evrec_20=evr[20][0], evrec_50=evr[50][0]))
        for k in KS:
            recall_rows.append(dict(H=H, model=name, k=k, caught=evr[k][0], total=evr[k][1]))
        print(f"  {name:16s} nf={len(cols):2d}  PR {pr:.4f} [{pr_lo:.4f}-{pr_hi:.4f}]  ROC {roc:.4f} [{roc_lo:.4f}-{roc_hi:.4f}]  evrec {evr[10][0]}/{evr[20][0]}/{evr[50][0]} of {evr[10][1]}")
    # M0 self-check
    ref_pr, ref_roc = REF[H]
    m0_pr = average_precision_score(y_te, scores["M0_workload"]); m0_roc = roc_auc_score(y_te, scores["M0_workload"])
    ok = abs(m0_pr - ref_pr) < 1e-9 and abs(m0_roc - ref_roc) < 1e-9
    print(f"  [self-check M0 vs reference] PR {m0_pr:.6f} (ref {ref_pr:.6f})  ROC {m0_roc:.6f} (ref {ref_roc:.6f})  {'MATCH' if ok else 'MISMATCH'}")
    assert ok, "M0 does not reproduce reference"
    # GBM on M3
    gprob = fit_gbm(M["M3_all"], y_fit)
    scores["GBM_M3"] = gprob
    pr = average_precision_score(y_te, gprob); roc = roc_auc_score(y_te, gprob)
    pr_lo, pr_hi, roc_lo, roc_hi = boot_ci(y_te, gprob)
    evr = event_recall(y_te, gprob, KS)
    results.append(dict(H=H, model="GBM_M3", n_feat=len(M["M3_all"]), pr_auc=pr, pr_lo=pr_lo, pr_hi=pr_hi,
                        roc_auc=roc, roc_lo=roc_lo, roc_hi=roc_hi, base_rate=base_rate,
                        evrec_total=evr[10][1], evrec_10=evr[10][0], evrec_20=evr[20][0], evrec_50=evr[50][0]))
    for k in KS:
        recall_rows.append(dict(H=H, model="GBM_M3", k=k, caught=evr[k][0], total=evr[k][1]))
    print(f"  {'GBM_M3':16s} nf={len(M['M3_all']):2d}  PR {pr:.4f} [{pr_lo:.4f}-{pr_hi:.4f}]  ROC {roc:.4f} [{roc_lo:.4f}-{roc_hi:.4f}]  evrec {evr[10][0]}/{evr[20][0]}/{evr[50][0]} of {evr[10][1]}")
    # ---- PAIRED deltas vs M0 ----
    print("  paired deltas vs M0_workload:")
    for name in ["M1_trend", "M2_variability", "M3_all", "GBM_M3"]:
        dpr, dpr_lo, dpr_hi, droc, droc_lo, droc_hi = paired_delta(y_te, scores[name], scores["M0_workload"])
        delta_rows.append(dict(H=H, model=name, d_pr=dpr, d_pr_lo=dpr_lo, d_pr_hi=dpr_hi,
                               d_roc=droc, d_roc_lo=droc_lo, d_roc_hi=droc_hi))
        sig_pr = "" if (dpr_lo <= 0 <= dpr_hi) else " *"
        sig_roc = "" if (droc_lo <= 0 <= droc_hi) else " *"
        print(f"    {name:16s}  dPR {dpr:+.4f} [{dpr_lo:+.4f},{dpr_hi:+.4f}]{sig_pr}   dROC {droc:+.4f} [{droc_lo:+.4f},{droc_hi:+.4f}]{sig_roc}")

# ================================================================ coverage (train vs test)
cov_rows = []
for f in TREND + VAR:
    impc = TV[f + "_imp"].values
    cov_rows.append(dict(feature=f,
                         cov_train=float(1 - impc[tr].mean()), cov_valid=float(1 - impc[va].mean()),
                         cov_test=float(1 - impc[te].mean()), cov_all=float(1 - impc.mean())))
for ind in ["trend_missing", "var_missing"]:
    m = TV[ind].values
    cov_rows.append(dict(feature=ind + " (=1 share)",
                         cov_train=float(m[tr].mean()), cov_valid=float(m[va].mean()),
                         cov_test=float(m[te].mean()), cov_all=float(m.mean())))
cov = pd.DataFrame(cov_rows)
print("\nCOVERAGE (% non-imputed; last two rows = indicator =1 share):")
print(cov.to_string(index=False))

# ================================================================ save
res = pd.DataFrame(results); dres = pd.DataFrame(delta_rows); rres = pd.DataFrame(recall_rows)
res.to_csv(SCR / "e3_results.csv", index=False)
dres.to_csv(SCR / "e3_paired_deltas.csv", index=False)
rres.to_csv(SCR / "e3_event_recall.csv", index=False)
cov.to_csv(SCR / "e3_coverage.csv", index=False)
print(f"\n[t={time.time()-t0:.1f}s] saved e3_results.csv / e3_paired_deltas.csv / e3_event_recall.csv / e3_coverage.csv")
print("\n=== RESULTS ===")
print(res.to_string(index=False))
print("\n=== PAIRED DELTAS vs M0 ===")
print(dres.to_string(index=False))
