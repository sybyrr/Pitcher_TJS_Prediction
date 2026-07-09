"""R2 — age covariate on top of M-role. Frozen protocol adapted VERBATIM from
results/phase26/scripts/role_models.py (load, feature build, fit_lr, pitcher-
clustered bootstrap seed 0 shared across models, event recall via next_surgery_date).

Models (per H in {90,150}, B=0, fold_main; fit train+valid, test 2022-23):
  M-role      = M0dp + start_share                 (6 feats; == role_models.py M2)
                 M0dp = {pc_chronic, pc_acute_dev, days_since_last, vel_trend, month}
  M-role+age  = M-role + age                        (7 feats)
                 age at (pitcher,t) = (t - birthDate).days / 365.25
StatsAPI coverage = 1252/1252 (100%): NO missing indicator, NO imputation.

Reproduce M-role anchor FIRST (tol ~3e-4) before producing new numbers.
Paired deltas (M-role+age minus M-role) over the SAME 1000 resamples.
Shape sanity: standardized age coefficient + positive rate by fit-set age decile.
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

# ---------------------------------------------------------------- load (identical to role_models)
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

# ---- birthdates (StatsAPI, 100% coverage) ----
bd = pd.read_csv(SCR / "birthdates.csv")
bd["birthDate"] = pd.to_datetime(bd["birthDate"])
bd_map = dict(zip(bd["pitcher"].astype(int), bd["birthDate"]))
n_missing_bd = int(bd["birthDate"].isna().sum())
print(f"birthdate coverage: {bd['birthDate'].notna().sum()}/{len(bd)}  missing={n_missing_bd}")

DAY = np.timedelta64(1, "D")

def pw_mean(mean_sp, pc, mask):
    m = mask & ~np.isnan(mean_sp)
    if not m.any():
        return np.nan
    wsum = pc[m].sum()
    if wsum <= 0:
        return np.nan
    return float((mean_sp[m] * pc[m]).sum() / wsum)

# ---------------------------------------------------------------- features (identical to role_models)
pid_all = cohort["pitcher"].values.astype(np.int64)
t_all = cohort["t"].values.astype("datetime64[ns]")
year_all = cohort["year"].values.astype(np.int64)
month_all = cohort["month"].values.astype(np.float64)
t_pd = pd.to_datetime(cohort["t"])

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

# ---- age at (pitcher, t) = (t - birthDate).days / 365.25 ----
birth_series = cohort["pitcher"].astype(int).map(bd_map)
age = (t_pd.values - birth_series.values) / np.timedelta64(1, "D") / 365.25
F["age"] = age.astype(np.float64)
n_age_missing = int(np.isnan(F["age"].values).sum())
print(f"[t={time.time()-t0:.1f}s] features built  shape {F.shape}  age NaN rows={n_age_missing}")

# fit-set median-age imputation ONLY if any missing (none expected)
add_missing_indicator = n_age_missing > 0

# sanity: days_since_last must equal cohort.dsl
dsl_diff = np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float))
assert dsl_diff.max() < 1e-6, f"days_since_last != dsl, max diff {dsl_diff.max()}"
print("VERIFY days_since_last == cohort.dsl : OK (max diff %.2e)" % dsl_diff.max())

# ---------------------------------------------------------------- folds
fold = cohort["fold_main"].values
tr = fold == "train"; va = fold == "valid"; te = fold == "test"
fit_mask = tr | va
print(f"fold_main sizes: train {tr.sum()}  valid {va.sum()}  test {te.sum()}  fit(tr+va) {fit_mask.sum()}")

# impute (median age on fit set) if needed
if add_missing_indicator:
    med_age = float(np.nanmedian(F.loc[fit_mask, "age"].values))
    F["age_missing"] = np.isnan(F["age"].values).astype(np.float64)
    F["age"] = F["age"].fillna(med_age)
    print(f"imputed {n_age_missing} missing ages with fit-set median {med_age:.2f} + missing indicator")

# ---------------------------------------------------------------- bootstrap resamples (identical, seed 0)
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

# ---------------------------------------------------------------- helpers (verbatim)
def paired_boot(y_te, score_a, score_b):
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

# ---------------------------------------------------------------- model definitions
FEATS_M0DP = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"]
FEATS_MROLE = FEATS_M0DP + ["start_share"]                       # 6 feats  (== role_models M2)
FEATS_MROLE_AGE = FEATS_MROLE + ["age"]                          # 7 feats
if add_missing_indicator:
    FEATS_MROLE_AGE = FEATS_MROLE_AGE + ["age_missing"]

# task anchors
ANCHOR_M0DP = {90: (0.615557, 0.022700), 150: (0.619131, 0.030203)}   # (ROC, PR)
ANCHOR_MROLE = {90: (0.64368, 0.03597), 150: (0.64377, 0.04700)}      # (ROC, PR)
exp_pos = {90: (138, 35, 118), 150: (205, 47, 162)}
TOL = 3e-4

# ---------------------------------------------------------------- CSV schema
COLS = ["section", "H", "B", "model", "featset", "n_test_windows", "n_test_pos",
        "base_rate", "pr_auc", "pr_lo", "pr_hi", "roc_auc", "roc_lo", "roc_hi",
        "evrec_total", "evrec_10", "evrec_20", "evrec_50",
        "dpr_point", "dpr_lo", "dpr_hi", "dpr_excl0",
        "droc_point", "droc_lo", "droc_hi", "droc_excl0", "nboot",
        "age_coef", "age_decile", "decile_age_lo", "decile_age_hi", "decile_pos_rate", "decile_n"]
def mkrow(**kw):
    r = {c: np.nan for c in COLS}
    r.update(kw)
    return r
def excl0(lo, hi):
    return bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))

# ---------------------------------------------------------------- MAIN
KS = [10, 20, 50]
new_rows = []
repro_flags = {}
age_coef_store = {}

for H in [90, 150]:
    B = 0
    col = f"label_H{H}_B{B}"
    y = cohort[col].values.astype(int)
    y_te = y[te]; y_fit = y[fit_mask]
    p_tr, p_va, p_te = int(y[tr].sum()), int(y[va].sum()), int(y_te.sum())
    assert (p_tr, p_va, p_te) == exp_pos[H], f"pos mismatch {col}: {(p_tr,p_va,p_te)}"
    base_rate = float(y_te.mean())
    print(f"\n{'='*78}\nH={H} B={B}  positives {p_tr}/{p_va}/{p_te}  base_rate {base_rate:.5f}")

    # ---- reproduce anchors FIRST ----
    clf_m0, prob_m0, _ = fit_lr(FEATS_M0DP, y_fit)
    roc_m0 = roc_auc_score(y_te, prob_m0); pr_m0 = average_precision_score(y_te, prob_m0)
    a_roc0, a_pr0 = ANCHOR_M0DP[H]
    ok_m0 = abs(roc_m0 - a_roc0) < TOL and abs(pr_m0 - a_pr0) < TOL
    print(f"  M0dp   REPRO ROC {roc_m0:.6f} (anchor {a_roc0:.6f} d={roc_m0-a_roc0:+.2e})  "
          f"PR {pr_m0:.6f} (anchor {a_pr0:.6f} d={pr_m0-a_pr0:+.2e})  {'OK' if ok_m0 else 'FAIL'}")

    clf_role, prob_role, logit_role = fit_lr(FEATS_MROLE, y_fit)
    roc_role = roc_auc_score(y_te, prob_role); pr_role = average_precision_score(y_te, prob_role)
    a_roc, a_pr = ANCHOR_MROLE[H]
    ok_role = abs(roc_role - a_roc) < TOL and abs(pr_role - a_pr) < TOL
    print(f"  M-role REPRO ROC {roc_role:.6f} (anchor {a_roc:.6f} d={roc_role-a_roc:+.2e})  "
          f"PR {pr_role:.6f} (anchor {a_pr:.6f} d={pr_role-a_pr:+.2e})  {'OK' if ok_role else 'FAIL'}")
    repro_flags[H] = (ok_m0, ok_role)
    if not ok_role:
        raise SystemExit(f"M-role anchor reproduction FAILED at H={H} (tol {TOL}). STOP.")

    # ---- M-role + age ----
    clf_age, prob_age, logit_age = fit_lr(FEATS_MROLE_AGE, y_fit)
    roc_age = roc_auc_score(y_te, prob_age); pr_age = average_precision_score(y_te, prob_age)
    age_coef = float(clf_age.coef_[0][FEATS_MROLE_AGE.index("age")])
    age_coef_store[H] = dict(zip(FEATS_MROLE_AGE, clf_age.coef_[0]))
    print(f"  M-role+age   ROC {roc_age:.6f}  PR {pr_age:.6f}  |  standardized age coef = {age_coef:+.5f}")

    # ---- single-model CIs ----
    prb, rocb, _, _, nvalid = paired_boot(y_te, prob_role, prob_role)
    role_pr_lo, role_pr_hi = ci(prb); role_roc_lo, role_roc_hi = ci(rocb)
    prb2, rocb2, _, _, nvalid2 = paired_boot(y_te, prob_age, prob_age)
    age_pr_lo, age_pr_hi = ci(prb2); age_roc_lo, age_roc_hi = ci(rocb2)
    assert nvalid == nvalid2

    # ---- event recall ----
    er_role = event_recall(y_te, prob_role, KS)
    er_age = event_recall(y_te, prob_age, KS)

    # ---- paired delta: (M-role+age) minus (M-role), shared resamples ----
    a_pr_b, a_roc_b, b_pr_b, b_roc_b, nvp = paired_boot(y_te, prob_role, prob_age)
    d_pr = b_pr_b - a_pr_b; d_roc = b_roc_b - a_roc_b
    dpr_lo, dpr_hi = ci(d_pr); droc_lo, droc_hi = ci(d_roc)
    assert nvp == nvalid
    print(f"  M-role     PR {pr_role:.5f} [{role_pr_lo:.4f},{role_pr_hi:.4f}]  "
          f"ROC {roc_role:.5f} [{role_roc_lo:.4f},{role_roc_hi:.4f}]  evrec {er_role[10][0]}/{er_role[20][0]}/{er_role[50][0]} of {er_role[10][1]}")
    print(f"  M-role+age PR {pr_age:.5f} [{age_pr_lo:.4f},{age_pr_hi:.4f}]  "
          f"ROC {roc_age:.5f} [{age_roc_lo:.4f},{age_roc_hi:.4f}]  evrec {er_age[10][0]}/{er_age[20][0]}/{er_age[50][0]} of {er_age[10][1]}")
    print(f"  PAIRED dPR {pr_age-pr_role:+.5f} [{dpr_lo:+.5f},{dpr_hi:+.5f}] {'EXCL0' if excl0(dpr_lo,dpr_hi) else 'incl0'}   "
          f"dROC {roc_age-roc_role:+.5f} [{droc_lo:+.5f},{droc_hi:+.5f}] {'EXCL0' if excl0(droc_lo,droc_hi) else 'incl0'}")

    # ---- CSV rows: main ----
    new_rows.append(mkrow(section="main", H=H, B=B, model="M-role", featset="m0dp+start_share",
        n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
        pr_auc=pr_role, pr_lo=role_pr_lo, pr_hi=role_pr_hi, roc_auc=roc_role, roc_lo=role_roc_lo, roc_hi=role_roc_hi,
        evrec_total=er_role[10][1], evrec_10=er_role[10][0], evrec_20=er_role[20][0], evrec_50=er_role[50][0], nboot=nvalid))
    new_rows.append(mkrow(section="main", H=H, B=B, model="M-role+age", featset="m0dp+start_share+age",
        n_test_windows=int(te.sum()), n_test_pos=p_te, base_rate=base_rate,
        pr_auc=pr_age, pr_lo=age_pr_lo, pr_hi=age_pr_hi, roc_auc=roc_age, roc_lo=age_roc_lo, roc_hi=age_roc_hi,
        evrec_total=er_age[10][1], evrec_10=er_age[10][0], evrec_20=er_age[20][0], evrec_50=er_age[50][0],
        dpr_point=pr_age-pr_role, dpr_lo=dpr_lo, dpr_hi=dpr_hi, dpr_excl0=excl0(dpr_lo, dpr_hi),
        droc_point=roc_age-roc_role, droc_lo=droc_lo, droc_hi=droc_hi, droc_excl0=excl0(droc_lo, droc_hi),
        nboot=nvalid, age_coef=age_coef))

    # ---- shape sanity: positive rate by fit-set age decile ----
    age_fit = F.loc[fit_mask, "age"].values
    y_fit_arr = y_fit
    # decile bins from fit-set age quantiles
    edges = np.quantile(age_fit, np.linspace(0, 1, 11))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    dec = np.digitize(age_fit, edges[1:-1], right=False)  # 0..9
    print(f"  --- fit-set age-decile positive rate (H={H}) ---")
    for d in range(10):
        m = dec == d
        nd = int(m.sum())
        pr_d = float(y_fit_arr[m].mean()) if nd > 0 else np.nan
        lo = float(age_fit[m].min()) if nd > 0 else np.nan
        hi = float(age_fit[m].max()) if nd > 0 else np.nan
        print(f"    decile {d}: age [{lo:5.1f},{hi:5.1f}]  n={nd:5d}  pos_rate={pr_d:.5f}  pos={int(y_fit_arr[m].sum())}")
        new_rows.append(mkrow(section="age_decile", H=H, B=B, model="fit_set",
            age_decile=d, decile_age_lo=lo, decile_age_hi=hi, decile_pos_rate=pr_d, decile_n=nd))
    # Spearman-ish monotonicity: correlation of decile index vs pos_rate
    prs = np.array([float(y_fit_arr[dec == d].mean()) for d in range(10)])
    from scipy.stats import spearmanr, pearsonr
    rho, pval = spearmanr(np.arange(10), prs)
    print(f"    Spearman(decile idx, pos_rate) = {rho:+.3f}  (p={pval:.3f})")
    new_rows.append(mkrow(section="age_decile_monotonicity", H=H, B=B, model="spearman_rho",
        age_coef=float(rho), decile_pos_rate=float(pval)))
    # store age coef row
    for k, v in age_coef_store[H].items():
        new_rows.append(mkrow(section="mrole_age_coef", H=H, B=B, model=k,
            featset="m0dp+start_share+age", age_coef=float(v)))

# ---------------------------------------------------------------- save
df_out = pd.DataFrame(new_rows, columns=COLS)
CSV = SCR / "r2_results.csv"
df_out.to_csv(CSV, index=False)
print(f"\n[t={time.time()-t0:.1f}s] wrote {len(df_out)} rows -> {CSV}")

# ---------------------------------------------------------------- verdict
print(f"\n{'='*78}\nVERDICT")
for r in new_rows:
    if r["section"] == "main" and r["model"] == "M-role+age":
        up_pr = r["dpr_lo"] > 0; up_roc = r["droc_lo"] > 0
        print(f"  H={int(r['H'])} M-role+age vs M-role: dROC {r['droc_point']:+.5f} "
              f"[{r['droc_lo']:+.5f},{r['droc_hi']:+.5f}] {'UP-EXCL0' if up_roc else 'incl0'}   "
              f"dPR {r['dpr_point']:+.6f} [{r['dpr_lo']:+.6f},{r['dpr_hi']:+.6f}] {'UP-EXCL0' if up_pr else 'incl0'}   "
              f"age_coef {r['age_coef']:+.4f}")
