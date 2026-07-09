"""R1 — rolling-origin robustness re-evaluation (+ M-role confirmation).

Purpose: the 2022-23 test fold has been inspected repeatedly (selection-bias
risk) and spans era shifts (COVID-shortened 2020 in train, pitch clock 2023 in
test). Check that the headline signal and the M-role gain are not artifacts of
that single fold.

DESIGN (per test year Y in {2021, 2022, 2023}):
  fit set = ALL windows with t.year < Y (no separate valid — no hyperparameter
            search, consistent with the frozen protocol's train+valid fitting)
  test    = windows with t.year == Y
  embargo = fit windows kept only if t + H (days) <= Y-04-01; report # removed
  models  = M0dp (5) and M-role (M0dp + start_share, 6), H in {90,150}, B=0
  metrics = n, positives, distinct surgeries, base rate, ROC[CI], PR[CI],
            event recall @10/20/50; PAIRED dROC/dPR (M-role - M0dp) with SHARED
            pitcher-clustered resamples (seed 0).

FROZEN PROTOCOL reused verbatim from results/phase26/scripts/role_models.py:
  StandardScaler + LogisticRegression(class_weight='balanced', max_iter=2000);
  pitcher-clustered bootstrap 1000 resamples seed 0 (shared across models);
  event-level recall@k per decision date via next_surgery_date grouping.

Anchor reproduction on the MAIN folds (fit train+valid, test 2022-23) runs FIRST
as a pipeline check (tolerance 3e-4, the documented L2-geometry whisker).
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

# ---------------------------------------------------------------- load (identical to role_models.py)
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

# ---------------------------------------------------------------- features (identical to role_models.py)
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

    rgd, rtp = role_by_pid[pid]
    rmask = (rgd < t) & (rgd >= t - np.timedelta64(365, "D"))
    ng365 = int(rmask.sum())
    start_share[i] = float((rtp[rmask] >= 50).sum()) / ng365 if ng365 > 0 else 0.0

FEATS = ["pc_30", "pc_90", "ng_30", "ng_90", "acwr", "days_since_last", "vel_trend", "month"]
F = pd.DataFrame(feat_rows, columns=FEATS)
F["pc_chronic"] = F["pc_90"] / 90.0
F["pc_acute_dev"] = F["pc_30"] / 30.0 - F["pc_90"] / 90.0
F["start_share"] = start_share
print(f"[t={time.time()-t0:.1f}s] features built  shape {F.shape}")

# pipeline integrity: days_since_last must equal cohort.dsl
dsl_diff = np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float))
assert dsl_diff.max() < 1e-6, f"days_since_last != dsl, max diff {dsl_diff.max()}"
print("VERIFY days_since_last == cohort.dsl : OK (max diff %.2e)" % dsl_diff.max())

FEATS_M0DP = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"]
FEATS_MROLE = FEATS_M0DP + ["start_share"]

year_arr = cohort["year"].values.astype(np.int64)
t_day_all = t_all.astype("datetime64[D]")
next_surg_all = pd.to_datetime(cohort["next_surgery_date"]).values

# ---------------------------------------------------------------- generic fold machinery
def fit_predict(cols, fit_mask, test_mask, y):
    """StandardScaler on fit set, class-balanced LR, predict on test. Frozen protocol."""
    sc = StandardScaler().fit(F.loc[fit_mask, cols])
    Xfit = sc.transform(F.loc[fit_mask, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(Xfit, y[fit_mask])
    prob = clf.predict_proba(sc.transform(F.loc[test_mask, cols]))[:, 1]
    return prob

def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def make_resamples(pid_test, seed=0, boot=1000):
    uniq = np.unique(pid_test)
    pos_in = {pid: np.where(pid_test == pid)[0] for pid in uniq}
    rng = np.random.default_rng(seed)
    n_pit = len(uniq)
    sets = []
    for _ in range(boot):
        drawn = rng.choice(uniq, size=n_pit, replace=True)
        sets.append(np.concatenate([pos_in[p] for p in drawn]))
    return sets, n_pit

def paired_boot(y_te, score_a, score_b, boot_sets):
    """Paired PR/ROC over shared resamples. Skip depends only on y so the surviving
    set is identical for both models -> valid paired deltas. delta = b - a."""
    pr_a, roc_a, pr_b, roc_b = [], [], [], []
    for idx in boot_sets:
        yb = y_te[idx]
        s = yb.sum()
        if s == 0 or s == len(yb):
            continue
        pr_a.append(average_precision_score(yb, score_a[idx]))
        roc_a.append(roc_auc_score(yb, score_a[idx]))
        pr_b.append(average_precision_score(yb, score_b[idx]))
        roc_b.append(roc_auc_score(yb, score_b[idx]))
    return (np.asarray(pr_a), np.asarray(roc_a),
            np.asarray(pr_b), np.asarray(roc_b), len(pr_a))

def topk_flags(score_te, t_te_dates, test_dates, k):
    flag = np.zeros(len(score_te), dtype=bool)
    for d in test_dates:
        sel = np.where(t_te_dates == d)[0]
        kk = min(k, len(sel))
        top = sel[np.argsort(-score_te[sel], kind="stable")[:kk]]
        flag[top] = True
    return flag

def event_recall(y_te, score_te, pid_te, surg_te, t_te_dates, test_dates, ks):
    pos_rows = np.where(y_te == 1)[0]
    keys = list(zip(pid_te[pos_rows].tolist(),
                    pd.to_datetime(surg_te[pos_rows]).astype("datetime64[ns]").tolist()))
    groups = {}
    for r, key in zip(pos_rows, keys):
        groups.setdefault(key, []).append(r)
    total = len(groups)
    out = {}
    for k in ks:
        flag = topk_flags(score_te, t_te_dates, test_dates, k)
        caught = sum(1 for rows in groups.values() if flag[rows].any())
        out[k] = (caught, total)
    return out

KS = [10, 20, 50]

# ================================================================ STEP 0: main-fold anchor reproduction
print(f"\n{'='*80}\nSTEP 0 — MAIN-FOLD ANCHOR REPRODUCTION (pipeline check, tol 3e-4)")
fold = cohort["fold_main"].values
fit_main = (fold == "train") | (fold == "valid")
te_main = fold == "test"
ANCHOR = {  # task anchors on main folds
    ("M0dp", 90): (0.615557, 0.022700), ("M0dp", 150): (0.619131, 0.030203),
    ("M-role", 90): (0.64368, 0.03597), ("M-role", 150): (0.64377, 0.04700),
}
anchor_ok = True
anchor_lines = []
for H in [90, 150]:
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    for mname, cols in [("M0dp", FEATS_M0DP), ("M-role", FEATS_MROLE)]:
        prob = fit_predict(cols, fit_main, te_main, y)
        roc = roc_auc_score(y[te_main], prob); pr = average_precision_score(y[te_main], prob)
        a_roc, a_pr = ANCHOR[(mname, H)]
        droc = roc - a_roc; dpr = pr - a_pr
        ok = abs(droc) < 3e-4 and abs(dpr) < 3e-4
        anchor_ok &= ok
        anchor_lines.append((mname, H, roc, a_roc, droc, pr, a_pr, dpr, ok))
        print(f"  {mname:7s} H={H}: ROC {roc:.6f} (anchor {a_roc:.6f}, d={droc:+.2e})  "
              f"PR {pr:.6f} (anchor {a_pr:.6f}, d={dpr:+.2e})  {'MATCH' if ok else 'MISMATCH'}")
print(f"  ANCHOR REPRODUCTION: {'PASS' if anchor_ok else 'FAIL'}")
assert anchor_ok, "anchor reproduction failed — pipeline changed"

# ================================================================ STEP 1: rolling-origin folds
Y0401 = {Y: np.datetime64(f"{Y}-04-01") for Y in (2021, 2022, 2023)}
new_rows = []
COLS = ["section", "test_year", "H", "B", "model", "featset",
        "n_fit", "n_fit_embargoed", "n_test", "n_test_pos", "distinct_surg",
        "base_rate", "pr_auc", "pr_lo", "pr_hi", "roc_auc", "roc_lo", "roc_hi",
        "evrec_total", "evrec_10", "evrec_20", "evrec_50",
        "dpr_point", "dpr_lo", "dpr_hi", "droc_point", "droc_lo", "droc_hi", "nboot"]
def mkrow(**kw):
    r = {c: np.nan for c in COLS}; r.update(kw); return r

print(f"\n{'='*80}\nSTEP 1 — ROLLING-ORIGIN FOLDS")
for Y in (2021, 2022, 2023):
    te_mask = year_arr == Y
    pid_te = pid_all[te_mask]
    t_te_dates = t_day_all[te_mask]
    test_dates = np.sort(np.unique(t_te_dates))
    surg_te = next_surg_all[te_mask]
    boot_sets, n_pit = make_resamples(pid_te, seed=0, boot=1000)
    print(f"\n{'-'*80}\nY={Y}  n_test={te_mask.sum()}  n_test_pitchers={n_pit}")
    for H in [90, 150]:
        y = cohort[f"label_H{H}_B0"].values.astype(int)
        # embargo: keep fit windows (t.year<Y) whose horizon t+H <= Y-04-01
        base_fit = year_arr < Y
        horizon_end = t_day_all + np.timedelta64(H, "D")
        keep = horizon_end <= Y0401[Y]
        fit_mask = base_fit & keep
        n_embargoed = int((base_fit & ~keep).sum())

        y_te = y[te_mask]
        n_test = int(te_mask.sum()); n_pos = int(y_te.sum())
        base_rate = float(y_te.mean())

        # models
        prob0 = fit_predict(FEATS_M0DP, fit_mask, te_mask, y)     # M0dp
        prob1 = fit_predict(FEATS_MROLE, fit_mask, te_mask, y)    # M-role

        roc0 = roc_auc_score(y_te, prob0); pr0 = average_precision_score(y_te, prob0)
        roc1 = roc_auc_score(y_te, prob1); pr1 = average_precision_score(y_te, prob1)

        # paired bootstrap (shared resamples); a=M0dp, b=M-role, delta = b - a
        pra, roca, prb, rocb, nboot = paired_boot(y_te, prob0, prob1, boot_sets)
        pr0_lo, pr0_hi = ci(pra); roc0_lo, roc0_hi = ci(roca)
        pr1_lo, pr1_hi = ci(prb); roc1_lo, roc1_hi = ci(rocb)
        d_pr = prb - pra; d_roc = rocb - roca
        dpr_lo, dpr_hi = ci(d_pr); droc_lo, droc_hi = ci(d_roc)

        # event recall
        er0 = event_recall(y_te, prob0, pid_te, surg_te, t_te_dates, test_dates, KS)
        er1 = event_recall(y_te, prob1, pid_te, surg_te, t_te_dates, test_dates, KS)
        distinct_surg = er0[10][1]

        print(f"  H={H} B=0  fit n={int(fit_mask.sum())} (embargoed {n_embargoed})  "
              f"test n={n_test} pos={n_pos} distinct_surg={distinct_surg} base={base_rate:.5f}")
        print(f"    M0dp   ROC {roc0:.5f} [{roc0_lo:.4f},{roc0_hi:.4f}]  PR {pr0:.5f} [{pr0_lo:.4f},{pr0_hi:.4f}]  "
              f"evrec {er0[10][0]}/{er0[20][0]}/{er0[50][0]} of {distinct_surg}")
        print(f"    M-role ROC {roc1:.5f} [{roc1_lo:.4f},{roc1_hi:.4f}]  PR {pr1:.5f} [{pr1_lo:.4f},{pr1_hi:.4f}]  "
              f"evrec {er1[10][0]}/{er1[20][0]}/{er1[50][0]} of {distinct_surg}")
        up_roc = "UP-EXCL0" if droc_lo > 0 else ("DOWN-EXCL0" if droc_hi < 0 else "incl0")
        up_pr = "UP-EXCL0" if dpr_lo > 0 else ("DOWN-EXCL0" if dpr_hi < 0 else "incl0")
        print(f"    PAIRED (M-role - M0dp)  dROC {d_roc.mean():+.5f} pt {roc1-roc0:+.5f} "
              f"[{droc_lo:+.5f},{droc_hi:+.5f}] {up_roc}   "
              f"dPR pt {pr1-pr0:+.6f} [{dpr_lo:+.6f},{dpr_hi:+.6f}] {up_pr}  nboot={nboot}")

        new_rows.append(mkrow(section="rolling", test_year=Y, H=H, B=0, model="M0dp", featset="curated_orth5",
            n_fit=int(fit_mask.sum()), n_fit_embargoed=n_embargoed, n_test=n_test, n_test_pos=n_pos,
            distinct_surg=distinct_surg, base_rate=base_rate,
            pr_auc=pr0, pr_lo=pr0_lo, pr_hi=pr0_hi, roc_auc=roc0, roc_lo=roc0_lo, roc_hi=roc0_hi,
            evrec_total=distinct_surg, evrec_10=er0[10][0], evrec_20=er0[20][0], evrec_50=er0[50][0],
            nboot=nboot))
        new_rows.append(mkrow(section="rolling", test_year=Y, H=H, B=0, model="M-role", featset="m0dp+start_share",
            n_fit=int(fit_mask.sum()), n_fit_embargoed=n_embargoed, n_test=n_test, n_test_pos=n_pos,
            distinct_surg=distinct_surg, base_rate=base_rate,
            pr_auc=pr1, pr_lo=pr1_lo, pr_hi=pr1_hi, roc_auc=roc1, roc_lo=roc1_lo, roc_hi=roc1_hi,
            evrec_total=distinct_surg, evrec_10=er1[10][0], evrec_20=er1[20][0], evrec_50=er1[50][0],
            dpr_point=pr1-pr0, dpr_lo=dpr_lo, dpr_hi=dpr_hi,
            droc_point=roc1-roc0, droc_lo=droc_lo, droc_hi=droc_hi, nboot=nboot))

# ---------------------------------------------------------------- save CSV (+ anchor rows)
for (mname, H, roc, a_roc, droc, pr, a_pr, dpr, ok) in anchor_lines:
    new_rows.append(mkrow(section="anchor_main", test_year=-1, H=H, B=0, model=mname, featset="main_fold_check",
        roc_auc=roc, pr_auc=pr, droc_point=droc, dpr_point=dpr, n_test_pos=int(ok)))
df_out = pd.DataFrame(new_rows, columns=COLS)
CSV = SCR / "r1_results.csv"
df_out.to_csv(CSV, index=False)
print(f"\n[t={time.time()-t0:.1f}s] wrote {len(df_out)} rows -> {CSV}")

# ---------------------------------------------------------------- PRE-SPECIFIED CRITERIA
print(f"\n{'='*80}\nPRE-SPECIFIED CRITERIA")
roll = [r for r in new_rows if r["section"] == "rolling"]
# (a) M0dp and M-role ROC > 0.55 in ALL folds
roc_all = [(r["model"], int(r["test_year"]), int(r["H"]), r["roc_auc"]) for r in roll]
crit_a = all(r[3] > 0.55 for r in roc_all)
print(f"(a) M0dp & M-role ROC > 0.55 in ALL folds: {'PASS' if crit_a else 'FAIL'}")
for m, Y, H, v in roc_all:
    print(f"      {m:7s} Y={Y} H={H}: ROC {v:.4f}{'' if v>0.55 else '  <-- <=0.55'}")
# (b) M-role dPR point > 0 in >=2/3 folds AND no fold with dPR CI excluding 0 downward
mrole = [r for r in roll if r["model"] == "M-role"]
by_H = {}
for H in [90, 150]:
    rows = [r for r in mrole if int(r["H"]) == H]
    npos = sum(1 for r in rows if r["dpr_point"] > 0)
    down = [r for r in rows if r["dpr_hi"] < 0]
    by_H[H] = (npos, len(rows), down, rows)
    print(f"  H={H}: dPR point>0 in {npos}/{len(rows)} folds; folds with dPR CI excl 0 downward: {len(down)}")
    for r in rows:
        excl = "UP-EXCL0" if r["dpr_lo"] > 0 else ("DOWN-EXCL0" if r["dpr_hi"] < 0 else "incl0")
        print(f"      Y={int(r['test_year'])}: dPR {r['dpr_point']:+.6f} [{r['dpr_lo']:+.6f},{r['dpr_hi']:+.6f}] {excl}")
crit_b_per_H = {H: (by_H[H][0] >= 2 and len(by_H[H][2]) == 0) for H in [90, 150]}
crit_b = all(crit_b_per_H.values())
print(f"(b) M-role confirmed (dPR pt>0 >=2/3 folds AND no downward-excl CI): "
      f"{'PASS' if crit_b else 'FAIL'}  per-H {crit_b_per_H}")
print(f"\nVERDICT: anchor={'PASS' if anchor_ok else 'FAIL'}  (a)={'PASS' if crit_a else 'FAIL'}  (b)={'PASS' if crit_b else 'FAIL'}")
print(f"[t={time.time()-t0:.1f}s] done")
