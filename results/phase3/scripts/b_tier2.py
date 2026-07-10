"""B block stage 2: coverage-corrected Tier-2 test.

b_tier.py's calendar-window T2 had 35-49% NaN (Apr/May decision dates look back
into the offseason), so its null could be dilution. Two pre-registered probes:

  (1) GAME-WINDOW T2 (T2g): acute = last 5 games strictly before t, chronic =
      last 15 games, trend baseline = prior seasons (fallback: games before the
      last-5 block). Coverage ~100% (cohort floor is 20 career games).
      Cells: F1|LR (gate), F1|HGB, F2g=F1+T2g and F3g=content {LR, HGB}.
  (2) OBSERVED-SUBSET contrast for the ORIGINAL calendar T2: same fitted models
      as b_tier.py, evaluation restricted to test windows with observed
      spin_fb_trend; pitcher-clustered resamples rebuilt on the subset.

Protocol otherwise identical to b_tier.py (frozen LR, HGB stage-1 grid on
train selected by valid ROC, paired bootstrap seed 0, event recall).
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
np.seterr(invalid="ignore", divide="ignore")

# ---------------------------------------------------------------- load (identical to b_tier)
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
    num = np.zeros(len(parts[0][1]), dtype=np.float64)
    den = np.zeros_like(num)
    for n, v in parts:
        w = np.where(np.isnan(v), 0.0, n.astype(np.float64))
        num += np.where(np.isnan(v), 0.0, v) * w
        den += w
    return np.where(den > 0, num / den, np.nan), den

gf_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    nFF = g["n_FF"].values; nSI = g["n_SI"].values
    nSL = g["n_SL"].values; nCU = g["n_CU"].values; nFC = g["n_FC"].values
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
        fb_spin=(fb_spin, w_spin), fb_ext=(fb_ext, w_ext),
        fb_relz=(fb_relz, w_relz), fb_relx=(fb_relx, w_relx),
        fb_velo_sd=(fb_velo_sd, w_vsd), fb_relx_sd=(fb_relx_sd, w_xsd),
        br_spin=(br_spin, w_brs),
    )
print(f"[t={time.time()-t0:.1f}s] per-game pooled stats built")

DAY = np.timedelta64(1, "D")

def pw_mean(v, w, mask):
    m = mask & ~np.isnan(v)
    if not m.any():
        return np.nan
    ws = w[m].sum()
    return float((v[m] * w[m]).sum() / ws) if ws > 0 else np.nan

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
T2G = [c + "_g" for c in T2]

feat_rows, t2cal_rows, t2g_rows = [], [], []
start_share = np.empty(N, dtype=np.float64)
for i in range(N):
    pid = int(pid_all[i])
    t = t_all[i].astype("datetime64[D]")
    gd, pc, sp, gy = by_pid[pid]
    before = gd < t
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))
    days_since_last = float((t - gd[before].max()) / DAY)

    vmean_30 = pw_mean(sp, pc, d30)
    yr = int(year_all[i])
    vmean_prior = pw_mean(sp, pc, before & (gy < yr))
    if np.isnan(vmean_prior):
        vmean_prior = pw_mean(sp, pc, before & (gd < t - np.timedelta64(30, "D")))
    vel_trend = (vmean_30 - vmean_prior) if not (np.isnan(vmean_30) or np.isnan(vmean_prior)) else 0.0
    feat_rows.append((pc[d30].sum() / 30.0 - pc[d90].sum() / 90.0, pc[d90].sum() / 90.0,
                      days_since_last, vel_trend, month_all[i]))

    G = gf_by_pid[pid]
    ggd, ggy, gtot, gbr = G["gd"], G["gy"], G["total"], G["br_n"]
    gbefore = ggd < t
    ib = np.where(gbefore)[0]

    def bmask(idx):
        m = np.zeros(len(ggd), dtype=bool); m[idx] = True; return m

    # ---- calendar windows (b_tier definition, for the observed-subset probe) ----
    g30 = gbefore & (ggd >= t - np.timedelta64(30, "D"))
    g90 = gbefore & (ggd >= t - np.timedelta64(90, "D"))
    gprior = gbefore & (ggy < yr)
    golder = gbefore & (ggd < t - np.timedelta64(30, "D"))

    def one_block(acute, chronic, base1, base2):
        def agg(key, mask):
            v, w = G[key]
            return pw_mean(v, w, mask)
        def trend(key):
            m = agg(key, acute)
            b = agg(key, base1)
            if np.isnan(b):
                b = agg(key, base2)
            return m - b if not (np.isnan(m) or np.isnan(b)) else np.nan
        tot_c = gtot[chronic].sum(); tot_a = gtot[acute].sum()
        share_c = gbr[chronic].sum() / tot_c if tot_c > 0 else np.nan
        share_a = gbr[acute].sum() / tot_a if tot_a > 0 else np.nan
        rdx = trend("fb_relx")
        return (
            agg("fb_spin", chronic), agg("fb_ext", chronic), agg("br_spin", chronic), share_c,
            trend("fb_spin"), trend("fb_ext"), trend("fb_relz"), trend("br_spin"),
            abs(rdx) if not np.isnan(rdx) else np.nan,
            (share_a - share_c) if not (np.isnan(share_a) or np.isnan(share_c)) else np.nan,
            agg("fb_velo_sd", acute), agg("fb_relx_sd", acute),
        )

    t2cal_rows.append(one_block(g30, g90, gprior, golder))
    # ---- game windows: acute last 5, chronic last 15, baseline prior seasons
    #      (fallback: games before the last-5 block) ----
    t2g_rows.append(one_block(bmask(ib[-5:]), bmask(ib[-15:]), gprior, bmask(ib[:-5])))

    rmask = gbefore & (ggd >= t - np.timedelta64(365, "D"))
    ng365 = int(rmask.sum())
    start_share[i] = float((gtot[rmask] >= 50).sum()) / ng365 if ng365 > 0 else 0.0

F = pd.DataFrame(feat_rows, columns=["pc_acute_dev", "pc_chronic", "days_since_last",
                                     "vel_trend", "month"])
F["start_share"] = start_share
F = pd.concat([F, pd.DataFrame(t2cal_rows, columns=T2),
               pd.DataFrame(t2g_rows, columns=T2G)], axis=1)
print(f"[t={time.time()-t0:.1f}s] features built  shape {F.shape}")

dsl_diff = np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float))
assert dsl_diff.max() < 1e-6

fold = cohort["fold_main"].values
tr = fold == "train"; va = fold == "valid"; te = fold == "test"
fit_mask = tr | va

print("\nT2g NaN fraction (train / valid / test):")
for c in T2G:
    v = F[c].values
    print(f"  {c:20s} {np.isnan(v[tr]).mean():.4f} / {np.isnan(v[va]).mean():.4f} / {np.isnan(v[te]).mean():.4f}")

TREND_ALL = set(T2_TREND) | {c + "_g" for c in T2_TREND}
F_imp = F.copy()
for c in T2 + T2G:
    fill = 0.0 if c in TREND_ALL else float(np.nanmedian(F.loc[fit_mask, c].values))
    F_imp[c] = F_imp[c].fillna(fill)
assert not F_imp.isna().any().any()

# ---------------------------------------------------------------- bootstrap machinery
t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
pid_test = pid_all[te]
next_surg = pd.to_datetime(cohort["next_surgery_date"]).values

def build_resamples(sub_rows: np.ndarray, seed: int = 0, nboot: int = 1000):
    """Pitcher-clustered resamples over the given test-row subset (row idx within test)."""
    pids = pid_test[sub_rows]
    uniq = np.unique(pids)
    pos = {p: sub_rows[np.where(pids == p)[0]] for p in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nboot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        out.append(np.concatenate([pos[p] for p in drawn]))
    return out

RESAMPLES_FULL = build_resamples(np.arange(int(te.sum())))
print(f"[t={time.time()-t0:.1f}s] full-test resamples built (n_pit={len(np.unique(pid_test))})")

def paired_boot(y_te, a, b, resamples):
    pr_a, roc_a, pr_b, roc_b = [], [], [], []
    for idx in resamples:
        yb = y_te[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        pr_a.append(average_precision_score(yb, a[idx])); roc_a.append(roc_auc_score(yb, a[idx]))
        pr_b.append(average_precision_score(yb, b[idx])); roc_b.append(roc_auc_score(yb, b[idx]))
    return np.asarray(pr_a), np.asarray(roc_a), np.asarray(pr_b), np.asarray(roc_b), len(pr_a)

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
F1 = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"]
FEATSETS = {
    "F1_public": F1,
    "F2cal": F1 + T2,          # calendar T2 (for observed-subset probe)
    "F2g": F1 + T2G,           # game-window T2
    "F3g_content": ["vel_trend"] + T2G,
}
HGB_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=mi)
            for d in (2, 3) for lr in (0.03, 0.1) for mi in (100, 300)]

def fit_lr(cols, y):
    sc = StandardScaler().fit(F_imp.loc[fit_mask, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F_imp.loc[fit_mask, cols]), y[fit_mask])
    return clf.predict_proba(sc.transform(F_imp.loc[te, cols]))[:, 1], None

def fit_hgb(cols, y):
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
    return clf.predict_proba(F.loc[te, cols].values)[:, 1], dict(valid_roc=round(vroc, 4), **g)

ANCHOR = {90: (0.643680, 0.035973), 150: (0.643775, 0.046995)}
exp_pos = {90: (138, 35, 118), 150: (205, 47, 162)}
KS = [10, 20, 50]
rows, drows = [], []

# observed-subset rows: calendar spin_fb_trend observed (test-row indices)
obs_sub = np.where(~np.isnan(F.loc[te, "spin_fb_trend"].values))[0]
RESAMPLES_SUB = build_resamples(obs_sub)

for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]
    assert (int(y[tr].sum()), int(y[va].sum()), int(y_te.sum())) == exp_pos[H]
    n_pos_sub = int(y_te[obs_sub].sum())
    print(f"\n{'='*78}\nH={H} B=0  test pos {int(y_te.sum())}  "
          f"observed-subset windows {len(obs_sub)}/{int(te.sum())}  subset pos {n_pos_sub}")

    probs = {}
    for fs, cols in FEATSETS.items():
        for mdl, fitter in (("LR", fit_lr), ("HGB", fit_hgb)):
            prob, cfg = fitter(cols, y)
            cell = f"{fs}|{mdl}"
            probs[cell] = prob
            pr = average_precision_score(y_te, prob); roc = roc_auc_score(y_te, prob)
            pr_b, roc_b, _, _, nv = paired_boot(y_te, prob, prob, RESAMPLES_FULL)
            er = event_recall(y_te, prob, KS)
            rows.append(dict(H=H, cell=cell, hgb_config=str(cfg) if cfg else "",
                             pr_auc=pr, pr_lo=ci(pr_b)[0], pr_hi=ci(pr_b)[1],
                             roc_auc=roc, roc_lo=ci(roc_b)[0], roc_hi=ci(roc_b)[1],
                             evrec_total=er[10][1], evrec_10=er[10][0],
                             evrec_20=er[20][0], evrec_50=er[50][0]))
            print(f"  {cell:20s} PR {pr:.5f}  ROC {roc:.5f}  evrec {er[10][0]}/{er[20][0]}/{er[50][0]}"
                  + (f"  cfg={cfg}" if cfg else ""))

    a_roc, a_pr = ANCHOR[H]
    got = [r for r in rows if r["H"] == H and r["cell"] == "F1_public|LR"][0]
    assert abs(got["roc_auc"] - a_roc) < 1e-4 and abs(got["pr_auc"] - a_pr) < 1e-4, "GATE FAILED"
    print(f"  GATE M-role repro: OK")

    CONTRASTS = [
        ("F2g|LR", "F1_public|LR", "T2g_increment_LR", RESAMPLES_FULL),
        ("F2g|HGB", "F1_public|HGB", "T2g_increment_HGB", RESAMPLES_FULL),
        ("F3g_content|LR", "F1_public|LR", "content_g_vs_Mrole_LR", RESAMPLES_FULL),
        ("F3g_content|HGB", "F1_public|LR", "content_g_vs_Mrole_HGB", RESAMPLES_FULL),
        ("F2cal|LR", "F1_public|LR", "SUBSET_T2cal_incr_LR", RESAMPLES_SUB),
        ("F2cal|HGB", "F1_public|HGB", "SUBSET_T2cal_incr_HGB", RESAMPLES_SUB),
    ]
    for cb, ca, name, rs in CONTRASTS:
        ap_pr, ap_roc, bp_pr, bp_roc, nv = paired_boot(y_te, probs[ca], probs[cb], rs)
        d_pr = bp_pr - ap_pr; d_roc = bp_roc - ap_roc
        dpr_lo, dpr_hi = ci(d_pr); droc_lo, droc_hi = ci(d_roc)
        pt_pr = float(np.median(d_pr)); pt_roc = float(np.median(d_roc))
        drows.append(dict(H=H, contrast=name, dpr_med=pt_pr, dpr_lo=dpr_lo, dpr_hi=dpr_hi,
                          dpr_excl0=excl0(dpr_lo, dpr_hi), droc_med=pt_roc,
                          droc_lo=droc_lo, droc_hi=droc_hi,
                          droc_excl0=excl0(droc_lo, droc_hi), nboot=nv))
        flag = ("  dROC-EXCL0" if excl0(droc_lo, droc_hi) else "") + \
               ("  dPR-EXCL0" if excl0(dpr_lo, dpr_hi) else "")
        print(f"  {name:24s} dPR {pt_pr:+.5f} [{dpr_lo:+.5f},{dpr_hi:+.5f}]  "
              f"dROC {pt_roc:+.5f} [{droc_lo:+.5f},{droc_hi:+.5f}]  nboot {nv}{flag}")

pd.DataFrame(rows).to_csv(SCR / "b_tier2_cells.csv", index=False)
pd.DataFrame(drows).to_csv(SCR / "b_tier2_deltas.csv", index=False)
print(f"\n[t={time.time()-t0:.1f}s] wrote b_tier2_cells.csv ({len(rows)}) + b_tier2_deltas.csv ({len(drows)})")
