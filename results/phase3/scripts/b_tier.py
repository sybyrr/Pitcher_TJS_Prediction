"""B block: feature-tier x model ablation vs the canonical M-role baseline.

Frozen protocol (verbatim from role_models.py / m0doubleprime.py): fold_main,
fit on train+valid, test 2022-23; 1000 pitcher-clustered bootstrap resamples
(seed 0) SHARED across all cells for paired CIs; event-level recall@k.

CELLS (per H in {90,150}, B=0):
  featset F1 = M-role  {pc_chronic, pc_acute_dev, days_since_last, vel_trend,
                        month, start_share}                       (public tier)
  featset F2 = F1 + 12 Tier-2 tracking aggregates (spin/ext/release, below)
  featset F3 = content-only  {vel_trend} + Tier-2                 (no usage/dsl/role)
  model LR   = StandardScaler + LogisticRegression(balanced, max_iter=2000)
               on imputed matrix (trend/dev -> 0, level/sd -> fit-set median)
  model HGB  = HistGradientBoostingClassifier(balanced) on raw-NaN matrix;
               stage-1 grid {max_depth 2,3} x {lr .03,.1} x {max_iter 100,300},
               min_samples_leaf=30, l2=1.0, fit on TRAIN, select by valid ROC,
               refit train+valid, single test evaluation.

TIER-2 FEATURES per (pitcher,t), games strictly before t, pitch-weighted:
  FB pool = FF+SI, BR pool = SL+CU+FC (per-game weighted merge, NaN-safe).
  levels (90d): spin_fb_chronic, ext_fb_chronic, spin_br_chronic, br_share_90
  trends (30d - prior-season, fallback older-than-30d; mirrors vel_trend):
    spin_fb_trend, ext_fb_trend, relz_fb_trend, spin_br_trend,
    relx_fb_absdrift = |drift|, br_share_dev = share30 - share90
  variability (30d): velo_fb_sd30, relx_fb_sd30   (pw mean of per-game sd)

GATE: F1xLR must reproduce M-role anchors (H90 ROC 0.643680 / PR 0.035973,
H150 ROC 0.643775 / PR 0.046995) within 1e-4 or the run aborts.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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
gf["game_year"] = pd.to_datetime(gf["game_date"]).dt.year.astype(np.int64)

def pool(parts):
    """NaN-safe pitch-weighted merge of per-game (n, value) pairs -> (value, eff_n)."""
    num = np.zeros(len(parts[0][1]), dtype=np.float64)
    den = np.zeros_like(num)
    for n, v in parts:
        w = np.where(np.isnan(v), 0.0, n.astype(np.float64))
        num += np.where(np.isnan(v), 0.0, v) * w
        den += w
    return np.where(den > 0, num / den, np.nan), den

# per-pitcher arrays of per-game pooled stats
gf_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    nFF = g["n_FF"].values; nSI = g["n_SI"].values
    nSL = g["n_SL"].values; nCU = g["n_CU"].values; nFC = g["n_FC"].values
    fb_velo, _ = pool([(nFF, g["FF_velo_mean"].values), (nSI, g["SI_velo_mean"].values)])
    fb_spin, w_spin = pool([(nFF, g["FF_spin_mean"].values), (nSI, g["SI_spin_mean"].values)])
    fb_ext, w_ext = pool([(nFF, g["FF_ext_mean"].values), (nSI, g["SI_ext_mean"].values)])
    fb_relz, w_relz = pool([(nFF, g["FF_relz_mean"].values), (nSI, g["SI_relz_mean"].values)])
    fb_relx, w_relx = pool([(nFF, g["FF_relx_mean"].values), (nSI, g["SI_relx_mean"].values)])
    fb_velo_sd, w_vsd = pool([(nFF, g["FF_velo_sd"].values), (nSI, g["SI_velo_sd"].values)])
    fb_relx_sd, w_xsd = pool([(nFF, g["FF_relx_sd"].values), (nSI, g["SI_relx_sd"].values)])
    br_spin, w_brs = pool([(nSL, g["SL_spin_mean"].values), (nCU, g["CU_spin_mean"].values),
                           (nFC, g["FC_spin_mean"].values)])
    gf_by_pid[int(pid)] = dict(
        gd=g["game_date"].values.astype("datetime64[D]"),
        gy=g["game_year"].values,
        total=g["total_pitches"].astype("float64").values,
        br_n=(nSL + nCU + nFC).astype(np.float64),
        fb_velo=(fb_velo, None), fb_spin=(fb_spin, w_spin), fb_ext=(fb_ext, w_ext),
        fb_relz=(fb_relz, w_relz), fb_relx=(fb_relx, w_relx),
        fb_velo_sd=(fb_velo_sd, w_vsd), fb_relx_sd=(fb_relx_sd, w_xsd),
        br_spin=(br_spin, w_brs),
    )
print(f"[t={time.time()-t0:.1f}s] per-game pooled stats built for {len(gf_by_pid)} pitchers")

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

T2_TREND = ["spin_fb_trend", "ext_fb_trend", "relz_fb_trend", "spin_br_trend",
            "relx_fb_absdrift", "br_share_dev"]
T2_LEVEL = ["spin_fb_chronic", "ext_fb_chronic", "spin_br_chronic", "br_share_90",
            "velo_fb_sd30", "relx_fb_sd30"]
T2 = T2_LEVEL[:4] + T2_TREND[:4] + [T2_TREND[4], T2_TREND[5], T2_LEVEL[4], T2_LEVEL[5]]

feat_rows = []
t2_rows = []
start_share = np.empty(N, dtype=np.float64)
for i in range(N):
    pid = int(pid_all[i])
    t = t_all[i].astype("datetime64[D]")
    gd, pc, sp, gy = by_pid[pid]
    before = gd < t
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))

    pc_30 = pc[d30].sum(); pc_90 = pc[d90].sum()
    days_since_last = float((t - gd[before].max()) / DAY)

    vmean_30 = pw_mean(sp, pc, d30)
    yr = int(year_all[i])
    prior_season = before & (gy < yr)
    vmean_prior = pw_mean(sp, pc, prior_season)
    if np.isnan(vmean_prior):
        older = before & (gd < t - np.timedelta64(30, "D"))
        vmean_prior = pw_mean(sp, pc, older)
    vel_trend = (vmean_30 - vmean_prior) if not (np.isnan(vmean_30) or np.isnan(vmean_prior)) else 0.0

    feat_rows.append((pc_30 / 30.0 - pc_90 / 90.0, pc_90 / 90.0, days_since_last,
                      vel_trend, month_all[i]))

    # ---- Tier-2 aggregates from game_features_v2 ----
    G = gf_by_pid[pid]
    ggd, ggy = G["gd"], G["gy"]
    gbefore = ggd < t
    g30 = gbefore & (ggd >= t - np.timedelta64(30, "D"))
    g90 = gbefore & (ggd >= t - np.timedelta64(90, "D"))
    gprior = gbefore & (ggy < yr)
    golder = gbefore & (ggd < t - np.timedelta64(30, "D"))

    def agg(key, mask):
        v, w = G[key]
        return pw_mean(v, w, mask)

    def trend(key):
        m30 = agg(key, g30)
        base = agg(key, gprior)
        if np.isnan(base):
            base = agg(key, golder)
        return m30 - base if not (np.isnan(m30) or np.isnan(base)) else np.nan

    tot90 = G["total"][g90].sum(); tot30 = G["total"][g30].sum()
    br_share_90 = G["br_n"][g90].sum() / tot90 if tot90 > 0 else np.nan
    br_share_30 = G["br_n"][g30].sum() / tot30 if tot30 > 0 else np.nan
    relx_dr = trend("fb_relx")
    t2_rows.append((
        agg("fb_spin", g90), agg("fb_ext", g90), agg("br_spin", g90), br_share_90,
        trend("fb_spin"), trend("fb_ext"), trend("fb_relz"), trend("br_spin"),
        abs(relx_dr) if not np.isnan(relx_dr) else np.nan,
        (br_share_30 - br_share_90) if not (np.isnan(br_share_30) or np.isnan(br_share_90)) else np.nan,
        agg("fb_velo_sd", g30), agg("fb_relx_sd", g30),
    ))

    # ---- role proxy: trailing 365d (verbatim role_models) ----
    rmask = gbefore & (ggd >= t - np.timedelta64(365, "D"))
    ng365 = int(rmask.sum())
    start_share[i] = float((G["total"][rmask] >= 50).sum()) / ng365 if ng365 > 0 else 0.0

F = pd.DataFrame(feat_rows, columns=["pc_acute_dev", "pc_chronic", "days_since_last",
                                     "vel_trend", "month"])
F["start_share"] = start_share
F2df = pd.DataFrame(t2_rows, columns=T2)
F = pd.concat([F, F2df], axis=1)
print(f"[t={time.time()-t0:.1f}s] features built  shape {F.shape}")

dsl_diff = np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float))
assert dsl_diff.max() < 1e-6, f"days_since_last != dsl, max diff {dsl_diff.max()}"
print("VERIFY days_since_last == cohort.dsl : OK")

# ---------------------------------------------------------------- folds
fold = cohort["fold_main"].values
tr = fold == "train"; va = fold == "valid"; te = fold == "test"
fit_mask = tr | va
print(f"fold_main sizes: train {tr.sum()}  valid {va.sum()}  test {te.sum()}")

print("\nT2 NaN fraction (train / valid / test):")
for c in T2:
    v = F[c].values
    print(f"  {c:18s} {np.isnan(v[tr]).mean():.4f} / {np.isnan(v[va]).mean():.4f} / {np.isnan(v[te]).mean():.4f}")

# imputed copy for LR: trend/dev -> 0, level/sd -> fit-set median
F_imp = F.copy()
for c in T2:
    fill = 0.0 if c in T2_TREND else float(np.nanmedian(F.loc[fit_mask, c].values))
    F_imp[c] = F_imp[c].fillna(fill)
assert not F_imp.isna().any().any()

# ---------------------------------------------------------------- bootstrap resamples (identical)
t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
pid_test = pid_all[te]
uniq_test_pid = np.unique(pid_test)
rng = np.random.default_rng(0)
pos_in_test = {pid: np.where(pid_test == pid)[0] for pid in uniq_test_pid}
BOOT = 1000
boot_index_sets = []
for _ in range(BOOT):
    drawn = rng.choice(uniq_test_pid, size=len(uniq_test_pid), replace=True)
    boot_index_sets.append(np.concatenate([pos_in_test[p] for p in drawn]))
print(f"[t={time.time()-t0:.1f}s] built {BOOT} pitcher-clustered resamples (n_pit={len(uniq_test_pid)})")

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

def excl0(lo, hi):
    return bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))

def topk_flags(score_te, k):
    flag = np.zeros(len(score_te), dtype=bool)
    for d in test_dates:
        sel = np.where(t_test == d)[0]
        kk = min(k, len(sel))
        flag[sel[np.argsort(-score_te[sel], kind="stable")[:kk]]] = True
    return flag

next_surg = pd.to_datetime(cohort["next_surgery_date"]).values

def event_recall(y_te, score_te, ks):
    pos_rows = np.where(y_te == 1)[0]
    keys = list(zip(pid_test[pos_rows].tolist(),
                    pd.to_datetime(next_surg[te][pos_rows]).astype("datetime64[ns]").tolist()))
    groups = {}
    for r, key in zip(pos_rows, keys):
        groups.setdefault(key, []).append(r)
    out = {}
    for k in ks:
        flag = topk_flags(score_te, k)
        out[k] = (sum(1 for rows in groups.values() if flag[rows].any()), len(groups))
    return out

# ---------------------------------------------------------------- models
FEATSETS = {
    "F1_public": ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"],
    "F2_public+T2": ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"] + T2,
    "F3_content": ["vel_trend"] + T2,
}

HGB_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=mi)
            for d in (2, 3) for lr in (0.03, 0.1) for mi in (100, 300)]

def fit_lr(cols, y_fit):
    sc = StandardScaler().fit(F_imp.loc[fit_mask, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F_imp.loc[fit_mask, cols]), y_fit)
    return clf.predict_proba(sc.transform(F_imp.loc[te, cols]))[:, 1], None

def fit_hgb(cols, y):
    """Grid on TRAIN, select by valid ROC, refit train+valid, predict test."""
    Xtr = F.loc[tr, cols].values; Xva = F.loc[va, cols].values
    best = None
    for g in HGB_GRID:
        clf = HistGradientBoostingClassifier(
            class_weight="balanced", min_samples_leaf=30, l2_regularization=1.0,
            early_stopping=False, random_state=0, **g)
        clf.fit(Xtr, y[tr])
        v = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
        if best is None or v > best[0]:
            best = (v, g)
    vroc, g = best
    clf = HistGradientBoostingClassifier(
        class_weight="balanced", min_samples_leaf=30, l2_regularization=1.0,
        early_stopping=False, random_state=0, **g)
    clf.fit(F.loc[fit_mask, cols].values, y[fit_mask])
    return clf.predict_proba(F.loc[te, cols].values)[:, 1], dict(valid_roc=vroc, **g)

# ---------------------------------------------------------------- MAIN
ANCHOR = {90: (0.643680, 0.035973), 150: (0.643775, 0.046995)}  # M-role (ROC, PR)
exp_pos = {90: (138, 35, 118), 150: (205, 47, 162)}
KS = [10, 20, 50]
COLS = ["H", "cell", "featset", "model", "hgb_config", "pr_auc", "pr_lo", "pr_hi",
        "roc_auc", "roc_lo", "roc_hi", "evrec_total", "evrec_10", "evrec_20", "evrec_50"]
DCOLS = ["H", "contrast", "dpr_point", "dpr_lo", "dpr_hi", "dpr_excl0",
         "droc_point", "droc_lo", "droc_hi", "droc_excl0", "nboot"]
rows, drows = [], []

for H in (90, 150):
    col = f"label_H{H}_B0"
    y = cohort[col].values.astype(int)
    y_te = y[te]
    assert (int(y[tr].sum()), int(y[va].sum()), int(y_te.sum())) == exp_pos[H]
    print(f"\n{'='*78}\nH={H} B=0  positives {exp_pos[H]}  base_rate {y_te.mean():.5f}")

    probs = {}
    for fs, cols in FEATSETS.items():
        for mdl, fitter in (("LR", fit_lr), ("HGB", fit_hgb)):
            prob, cfg = fitter(cols, y) if mdl == "HGB" else fitter(cols, y[fit_mask])
            cell = f"{fs}|{mdl}"
            probs[cell] = prob
            pr = average_precision_score(y_te, prob); roc = roc_auc_score(y_te, prob)
            pr_b, roc_b, _, _, nv = paired_boot(y_te, prob, prob)
            er = event_recall(y_te, prob, KS)
            rows.append(dict(H=H, cell=cell, featset=fs, model=mdl,
                             hgb_config=str(cfg) if cfg else "",
                             pr_auc=pr, pr_lo=ci(pr_b)[0], pr_hi=ci(pr_b)[1],
                             roc_auc=roc, roc_lo=ci(roc_b)[0], roc_hi=ci(roc_b)[1],
                             evrec_total=er[10][1], evrec_10=er[10][0],
                             evrec_20=er[20][0], evrec_50=er[50][0]))
            cfgs = f"  cfg={cfg}" if cfg else ""
            print(f"  {cell:22s} PR {pr:.5f}  ROC {roc:.5f}  evrec {er[10][0]}/{er[20][0]}/{er[50][0]}{cfgs}")

    # gate: F1|LR must reproduce the M-role anchor
    a_roc, a_pr = ANCHOR[H]
    got = [r for r in rows if r["H"] == H and r["cell"] == "F1_public|LR"][0]
    d_roc = got["roc_auc"] - a_roc; d_pr = got["pr_auc"] - a_pr
    print(f"  GATE M-role repro: dROC {d_roc:+.2e}  dPR {d_pr:+.2e}")
    assert abs(d_roc) < 1e-4 and abs(d_pr) < 1e-4, "M-role reproduction FAILED"

    CONTRASTS = [
        ("F2_public+T2|LR", "F1_public|LR", "T2_increment_LR"),
        ("F2_public+T2|HGB", "F1_public|HGB", "T2_increment_HGB"),
        ("F1_public|HGB", "F1_public|LR", "model_effect_F1"),
        ("F2_public+T2|HGB", "F2_public+T2|LR", "model_effect_F2"),
        ("F3_content|LR", "F1_public|LR", "content_vs_Mrole_LR"),
        ("F3_content|HGB", "F1_public|LR", "content_vs_Mrole_HGB"),
        ("F2_public+T2|HGB", "F1_public|LR", "best_vs_Mrole"),
    ]
    for cb, ca, name in CONTRASTS:
        ap_pr, ap_roc, bp_pr, bp_roc, nv = paired_boot(y_te, probs[ca], probs[cb])
        d_pr = bp_pr - ap_pr; d_roc = bp_roc - ap_roc
        dpr_lo, dpr_hi = ci(d_pr); droc_lo, droc_hi = ci(d_roc)
        pt_pr = average_precision_score(y_te, probs[cb]) - average_precision_score(y_te, probs[ca])
        pt_roc = roc_auc_score(y_te, probs[cb]) - roc_auc_score(y_te, probs[ca])
        drows.append(dict(H=H, contrast=name, dpr_point=pt_pr, dpr_lo=dpr_lo, dpr_hi=dpr_hi,
                          dpr_excl0=excl0(dpr_lo, dpr_hi), droc_point=pt_roc,
                          droc_lo=droc_lo, droc_hi=droc_hi,
                          droc_excl0=excl0(droc_lo, droc_hi), nboot=nv))
        flag = ("  dROC-EXCL0" if excl0(droc_lo, droc_hi) else "") + \
               ("  dPR-EXCL0" if excl0(dpr_lo, dpr_hi) else "")
        print(f"  {name:22s} dPR {pt_pr:+.5f} [{dpr_lo:+.5f},{dpr_hi:+.5f}]  "
              f"dROC {pt_roc:+.5f} [{droc_lo:+.5f},{droc_hi:+.5f}]{flag}")

pd.DataFrame(rows, columns=COLS).to_csv(SCR / "b_tier_cells.csv", index=False)
pd.DataFrame(drows, columns=DCOLS).to_csv(SCR / "b_tier_deltas.csv", index=False)
print(f"\n[t={time.time()-t0:.1f}s] wrote b_tier_cells.csv ({len(rows)} rows) + b_tier_deltas.csv ({len(drows)} rows)")
