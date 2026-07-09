"""Role-aware models vs the canonical M0dp (M0'') baseline — frozen protocol.

Adapts results/phase26/scripts/m0doubleprime.py VERBATIM on the parts that define
the evaluation (feature build for the workload block, fit_lr, pitcher-clustered
bootstrap, event recall). Adds a leakage-free role proxy and three role forms.

Protocol (unchanged): StandardScaler + LogisticRegression(class_weight='balanced',
max_iter=2000), fit on fold_main train+valid, evaluate test 2022-23; 1000 pitcher-
clustered bootstrap resamples (seed 0) SHARED across models for paired CIs; event-
level recall@k via next_surgery_date grouping.

MODELS (per H in {90,150}, B=0, fold_main):
  M0dp  = {pc_chronic, pc_acute_dev, days_since_last, vel_trend, month}   (canonical)
  M2    = M0dp + start_share                                              (+role additive)
  M3    = M0dp + start_share + start_share*z(pc_chronic) + start_share*z(pc_acute_dev)
            + start_share*z(days_since_last)   (interactions; z = fit-set standardize)
  M4    = {wz(pc_chronic), wz(pc_acute_dev), wz(days_since_last), start_share,
            vel_trend, month}   (wz = WITHIN-ROLE z-score, per-role fit-set mean/sd)

ROLE PROXY (per (pitcher,t), games STRICTLY before t, trailing 365 days,
game_features_v2.total_pitches): start_share = frac(total_pitches>=50) (0 if none);
role_sp = 1 if start_share>=0.5 else 0.
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

# ---------------------------------------------------------------- load (identical to m0doubleprime)
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

# role source: game_features_v2.total_pitches (verified byte-identical to slim pitch_count)
gf = pd.read_parquet(ROOT / "data/prospective/game_features_v2.parquet")
gf = gf.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
role_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    role_by_pid[int(pid)] = (
        g["game_date"].values.astype("datetime64[D]"),
        g["total_pitches"].astype("float64").values,
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
start_share = np.empty(N, dtype=np.float64)
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

    # ---- role proxy: trailing 365d, strictly before t, from game_features_v2 ----
    rgd, rtp = role_by_pid[pid]
    rmask = (rgd < t) & (rgd >= t - np.timedelta64(365, "D"))
    ng365 = int(rmask.sum())
    start_share[i] = float((rtp[rmask] >= 50).sum()) / ng365 if ng365 > 0 else 0.0

FEATS = ["pc_30", "pc_90", "ng_30", "ng_90", "acwr", "days_since_last", "vel_trend", "month"]
F = pd.DataFrame(feat_rows, columns=FEATS)
F["pc_chronic"] = F["pc_90"] / 90.0
F["pc_acute_dev"] = F["pc_30"] / 30.0 - F["pc_90"] / 90.0
F["start_share"] = start_share
F["role_sp"] = (start_share >= 0.5).astype(np.float64)
print(f"[t={time.time()-t0:.1f}s] features built  shape {F.shape}")

# sanity: days_since_last must equal cohort.dsl (pipeline integrity)
dsl_diff = np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float))
assert dsl_diff.max() < 1e-6, f"days_since_last != dsl, max diff {dsl_diff.max()}"
print("VERIFY days_since_last == cohort.dsl : OK (max diff %.2e)" % dsl_diff.max())

# ---------------------------------------------------------------- folds (identical to m0doubleprime)
fold = cohort["fold_main"].values
tr = fold == "train"; va = fold == "valid"; te = fold == "test"
fit_mask = tr | va
print(f"fold_main sizes: train {tr.sum()}  valid {va.sum()}  test {te.sum()}  fit(tr+va) {fit_mask.sum()}")

# ---- role diagnostics (sanity check vs task's known diagnostics) ----
sp_fit = F.loc[fit_mask, "role_sp"].values
print(f"\nROLE DIAGNOSTICS")
print(f"  fit-set SP share (role_sp==1): {sp_fit.mean():.4f}  (expect ~0.32)")
pcc = F["pc_chronic"].values
print(f"  pc_chronic median  SP: {np.median(pcc[fit_mask & (F['role_sp'].values==1)]):.4f}"
      f"  RP: {np.median(pcc[fit_mask & (F['role_sp'].values==0)]):.4f}  (expect ~4.73 / ~0.99)")
print(f"  start_share summary (fit): min {sp_fit.min():.2f} share>0 {float((F.loc[fit_mask,'start_share']>0).mean()):.3f}")

# ---------------------------------------------------------------- bootstrap resamples (identical)
t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
pid_test = pid_all[te]
uniq_test_pid = np.unique(pid_test)

rng = np.random.default_rng(0)
pos_in_test = {pid: np.where(pid_test == pid)[0] for pid in uniq_test_pid}
n_pit = len(uniq_test_pid)
BOOT = 1000
boot_index_sets = []
for _ in range(BOOT):
    drawn = rng.choice(uniq_test_pid, size=n_pit, replace=True)
    idx = np.concatenate([pos_in_test[p] for p in drawn])
    boot_index_sets.append(idx)
print(f"[t={time.time()-t0:.1f}s] built {BOOT} pitcher-clustered resamples  (n_pit={n_pit})")

# ---------------------------------------------------------------- helpers (m0doubleprime)
def paired_boot(y_te, score_a, score_b):
    """Paired PR/ROC over shared resamples. Skip depends only on y so the surviving
    set is identical for both models -> valid paired deltas."""
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

def topk_flags(score_te, t_te_dates, k):
    flag = np.zeros(len(score_te), dtype=bool)
    for d in test_dates:
        sel = np.where(t_te_dates == d)[0]
        ss = score_te[sel]
        kk = min(k, len(sel))
        top = sel[np.argsort(-ss, kind="stable")[:kk]]
        flag[top] = True
    return flag

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

def fit_lr(cols, y_fit):
    sc = StandardScaler().fit(F.loc[fit_mask, cols])
    Xfit = sc.transform(F.loc[fit_mask, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(Xfit, y_fit)
    Xte = sc.transform(F.loc[te, cols])
    prob = clf.predict_proba(Xte)[:, 1]
    logit = clf.decision_function(Xte)
    return clf, prob, logit

# ---------------------------------------------------------------- output CSV schema
COLS = ["section", "H", "B", "model", "featset", "n_test_windows", "n_test_pos",
        "base_rate", "pr_auc", "pr_lo", "pr_hi", "roc_auc", "roc_lo", "roc_hi",
        "evrec_total", "evrec_10", "evrec_20", "evrec_50",
        "dpr_point", "dpr_lo", "dpr_hi", "dpr_excl0",
        "droc_point", "droc_lo", "droc_hi", "droc_excl0", "nboot"]
def mkrow(**kw):
    r = {c: np.nan for c in COLS}
    r.update(kw)
    return r
def excl0(lo, hi):
    return bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))

# feature sets that DON'T depend on H (base workload / role columns are H-invariant)
FEATS_M0DP = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"]
FEATS_M2 = FEATS_M0DP + ["start_share"]

# interaction base z-scores (fit-set standardization; documented)
INT_BASE = ["pc_chronic", "pc_acute_dev", "days_since_last"]
for c in INT_BASE:
    mu = F.loc[fit_mask, c].mean(); sd = F.loc[fit_mask, c].std(ddof=0)
    zname = f"z_{c}"
    F[zname] = (F[c] - mu) / sd
    F[f"ss_x_{c}"] = F["start_share"] * F[zname]
FEATS_M3 = FEATS_M0DP + ["start_share", "ss_x_pc_chronic", "ss_x_pc_acute_dev", "ss_x_days_since_last"]

# within-role z-scores (per-role fit-set mean/sd, applied to ALL rows -> no test leakage)
WITHIN_BASE = ["pc_chronic", "pc_acute_dev", "days_since_last"]
role_arr = F["role_sp"].values
for c in WITHIN_BASE:
    wz = np.empty(N, dtype=np.float64)
    for r in (0.0, 1.0):
        sub = fit_mask & (role_arr == r)
        mu = F.loc[sub, c].mean(); sd = F.loc[sub, c].std(ddof=0)
        rows = role_arr == r
        wz[rows] = (F[c].values[rows] - mu) / sd
    F[f"wz_{c}"] = wz
FEATS_M4 = ["wz_pc_chronic", "wz_pc_acute_dev", "wz_days_since_last", "start_share", "vel_trend", "month"]

MODELS = [
    ("M0dp", "curated_orth5", FEATS_M0DP),
    ("M2_role_add", "m0dp+start_share", FEATS_M2),
    ("M3_role_inter", "m0dp+ss+ss*z(chr,acu,dsl)", FEATS_M3),
    ("M4_within_role", "wz(chr,acu,dsl)+ss+vel+month", FEATS_M4),
]

# ---------------------------------------------------------------- MAIN
KS = [10, 20, 50]
ANCHOR = {90: (0.615557, 0.022700), 150: (0.619131, 0.030203)}  # (ROC, PR) task anchors (== M0')
exp_pos = {90: (138, 35, 118), 150: (205, 47, 162)}
new_rows = []
coef2_store = {}
repro = {}

for H in [90, 150]:
    B = 0
    col = f"label_H{H}_B{B}"
    y = cohort[col].values.astype(int)
    y_te = y[te]; y_fit = y[fit_mask]
    p_tr, p_va, p_te = int(y[tr].sum()), int(y[va].sum()), int(y_te.sum())
    et, ev, ee = exp_pos[H]
    assert (p_tr, p_va, p_te) == (et, ev, ee), f"pos mismatch {col}: {(p_tr,p_va,p_te)}"
    base_rate = float(y_te.mean())
    # role-split test base rates
    br_sp = float(y_te[F.loc[te, "role_sp"].values == 1].mean())
    br_rp = float(y_te[F.loc[te, "role_sp"].values == 0].mean())
    print(f"\n{'='*78}\nH={H} B={B}  positives {p_tr}/{p_va}/{p_te}  base_rate {base_rate:.5f}"
          f"   test SP rate {br_sp:.4f}  RP rate {br_rp:.4f}  (expect ~0.0222 / ~0.0113 at H=90)")

    # fit all models, keep predictions
    fitres = {}
    for name, fs, cols in MODELS:
        clf, prob, logit = fit_lr(cols, y_fit)
        pr = average_precision_score(y_te, prob); roc = roc_auc_score(y_te, prob)
        fitres[name] = (clf, prob, pr, roc, cols)

    # M0dp reproduction vs anchor
    _, prob0, pr0, roc0, _ = fitres["M0dp"]
    a_roc, a_pr = ANCHOR[H]
    repro[H] = (pr0, roc0, a_pr, a_roc)
    print(f"  M0dp REPRO: ROC {roc0:.6f} (anchor {a_roc:.6f}, d={roc0-a_roc:+.2e})   "
          f"PR {pr0:.6f} (anchor {a_pr:.6f}, d={pr0-a_pr:+.2e})")

    # per-model: single-model CI + event recall + (if not M0dp) paired delta vs M0dp
    for name, fs, cols in MODELS:
        clf, prob, pr, roc, _ = fitres[name]
        pr_b, roc_b, _, _, nvalid = paired_boot(y_te, prob, prob)   # slot-a distribution
        pr_lo, pr_hi = ci(pr_b); roc_lo, roc_hi = ci(roc_b)
        er = event_recall(y_te, prob, KS)
        row = mkrow(section="main", H=H, B=B, model=name, featset=fs,
                    n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
                    pr_auc=pr, pr_lo=pr_lo, pr_hi=pr_hi, roc_auc=roc, roc_lo=roc_lo, roc_hi=roc_hi,
                    evrec_total=er[10][1], evrec_10=er[10][0], evrec_20=er[20][0], evrec_50=er[50][0],
                    nboot=nvalid)
        if name != "M0dp":
            # paired delta = model - M0dp over shared resamples
            ap_pr, ap_roc, bp_pr, bp_roc, nvp = paired_boot(y_te, prob0, prob)
            d_pr = bp_pr - ap_pr; d_roc = bp_roc - ap_roc
            dpr_lo, dpr_hi = ci(d_pr); droc_lo, droc_hi = ci(d_roc)
            assert nvp == nvalid
            row.update(dpr_point=pr - pr0, dpr_lo=dpr_lo, dpr_hi=dpr_hi, dpr_excl0=excl0(dpr_lo, dpr_hi),
                       droc_point=roc - roc0, droc_lo=droc_lo, droc_hi=droc_hi, droc_excl0=excl0(droc_lo, droc_hi))
            flag = ""
            if excl0(droc_lo, droc_hi): flag += "  dROC-CI-EXCL0"
            if excl0(dpr_lo, dpr_hi): flag += "  dPR-CI-EXCL0"
            print(f"  {name:16s} PR {pr:.5f} [{pr_lo:.4f},{pr_hi:.4f}]  ROC {roc:.5f} [{roc_lo:.4f},{roc_hi:.4f}]"
                  f"  evrec {er[10][0]}/{er[20][0]}/{er[50][0]}")
            print(f"  {'':16s} dPR {pr-pr0:+.5f} [{dpr_lo:+.5f},{dpr_hi:+.5f}]  "
                  f"dROC {roc-roc0:+.5f} [{droc_lo:+.5f},{droc_hi:+.5f}]{flag}")
        else:
            print(f"  {name:16s} PR {pr:.5f} [{pr_lo:.4f},{pr_hi:.4f}]  ROC {roc:.5f} [{roc_lo:.4f},{roc_hi:.4f}]"
                  f"  evrec {er[10][0]}/{er[20][0]}/{er[50][0]}  (reference)")
        new_rows.append(row)

    # standardized coefficients of model 2
    clf2, _, _, _, cols2 = fitres["M2_role_add"]
    coef2_store[H] = dict(zip(cols2, clf2.coef_[0]))
    print(f"  M2 std coef: " + "  ".join(f"{k}={v:+.4f}" for k, v in coef2_store[H].items()))
    for k, v in coef2_store[H].items():
        new_rows.append(mkrow(section="m2_coef", H=H, B=B, model=k, featset="m0dp+start_share", pr_auc=float(v)))

# ---------------------------------------------------------------- save
df_out = pd.DataFrame(new_rows, columns=COLS)
CSV = SCR / "role_models_results.csv"
df_out.to_csv(CSV, index=False)
print(f"\n[t={time.time()-t0:.1f}s] wrote {len(df_out)} rows -> {CSV}")

# ---------------------------------------------------------------- verdict summary
print(f"\n{'='*78}\nVERDICT (does any role form DETECTABLY beat M0dp — paired CI excludes 0 UPWARD?)")
for r in new_rows:
    if r["section"] == "main" and r["model"] != "M0dp":
        up_roc = (r["droc_lo"] > 0)
        up_pr = (r["dpr_lo"] > 0)
        print(f"  H={int(r['H'])} {r['model']:16s}: dROC {r['droc_point']:+.5f} "
              f"[{r['droc_lo']:+.5f},{r['droc_hi']:+.5f}] {'UP-EXCL0' if up_roc else 'incl0'}   "
              f"dPR {r['dpr_point']:+.6f} [{r['dpr_lo']:+.6f},{r['dpr_hi']:+.6f}] {'UP-EXCL0' if up_pr else 'incl0'}")
