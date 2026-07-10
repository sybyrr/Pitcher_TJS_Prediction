"""B block: HGB grid boundary check.

Stage-1/2 grids always selected the most conservative corner (depth 2, lr .03),
so extend the grid PAST that boundary (depth 1, lr .01, max_iter 50, stronger
leaf/l2 regularization). Selection on valid ROC only; single test evaluation
per (featset, H); point estimates vs the LR reference. If nothing approaches
LR, the tree question is closed without further bootstrap.

Feature build copied verbatim from b_tier2.py (F1 + game-window T2g).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
SCR = Path(r"C:/Users/PC/AppData/Local/Temp/claude/d--PAINS-Pitcher-TJS-Prediction/c8255b88-a70d-45ba-964b-531a2271c93f/scratchpad")
t0 = time.time()
np.seterr(invalid="ignore", divide="ignore")

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

DAY = np.timedelta64(1, "D")

def pw_mean(v, w, mask):
    m = mask & ~np.isnan(v)
    if not m.any():
        return np.nan
    ws = w[m].sum()
    return float((v[m] * w[m]).sum() / ws) if ws > 0 else np.nan

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

feat_rows, t2g_rows = [], []
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

    gprior = gbefore & (ggy < yr)
    acute = bmask(ib[-5:]); chronic = bmask(ib[-15:]); base2 = bmask(ib[:-5])

    def agg(key, mask):
        v, w = G[key]
        return pw_mean(v, w, mask)

    def trend(key):
        m = agg(key, acute)
        b = agg(key, gprior)
        if np.isnan(b):
            b = agg(key, base2)
        return m - b if not (np.isnan(m) or np.isnan(b)) else np.nan

    tot_c = gtot[chronic].sum(); tot_a = gtot[acute].sum()
    share_c = gbr[chronic].sum() / tot_c if tot_c > 0 else np.nan
    share_a = gbr[acute].sum() / tot_a if tot_a > 0 else np.nan
    rdx = trend("fb_relx")
    t2g_rows.append((
        agg("fb_spin", chronic), agg("fb_ext", chronic), agg("br_spin", chronic), share_c,
        trend("fb_spin"), trend("fb_ext"), trend("fb_relz"), trend("br_spin"),
        abs(rdx) if not np.isnan(rdx) else np.nan,
        (share_a - share_c) if not (np.isnan(share_a) or np.isnan(share_c)) else np.nan,
        agg("fb_velo_sd", acute), agg("fb_relx_sd", acute),
    ))

    rmask = gbefore & (ggd >= t - np.timedelta64(365, "D"))
    ng365 = int(rmask.sum())
    start_share[i] = float((gtot[rmask] >= 50).sum()) / ng365 if ng365 > 0 else 0.0

F = pd.DataFrame(feat_rows, columns=["pc_acute_dev", "pc_chronic", "days_since_last",
                                     "vel_trend", "month"])
F["start_share"] = start_share
F = pd.concat([F, pd.DataFrame(t2g_rows, columns=T2G)], axis=1)
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6
print(f"[t={time.time()-t0:.1f}s] features built {F.shape}")

fold = cohort["fold_main"].values
tr = fold == "train"; va = fold == "valid"; te = fold == "test"
fit_mask = tr | va

F1 = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"]
FEATSETS = {"F1_public": F1, "F2g": F1 + T2G}
LR_REF = {90: (0.643680, 0.035973), 150: (0.643775, 0.046995)}  # F1|LR (ROC, PR)

# extended grid PAST the previously-selected corner (depth 2, lr .03, iter 100)
GRID = [dict(max_depth=d, learning_rate=lr, max_iter=mi,
             min_samples_leaf=msl, l2_regularization=l2)
        for d in (1, 2) for lr in (0.01, 0.03) for mi in (50, 100, 200)
        for msl in (30, 100) for l2 in (1.0, 10.0)]
print(f"extended grid size: {len(GRID)}")

out = []
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]
    print(f"\nH={H}  LR reference ROC {LR_REF[H][0]:.5f}  PR {LR_REF[H][1]:.5f}")
    for fs, cols in FEATSETS.items():
        Xtr = F.loc[tr, cols].values; Xva = F.loc[va, cols].values
        best = None
        for g in GRID:
            clf = HistGradientBoostingClassifier(class_weight="balanced",
                                                 early_stopping=False, random_state=0, **g)
            clf.fit(Xtr, y[tr])
            v = roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1])
            if best is None or v > best[0]:
                best = (v, g)
        vroc, g = best
        clf = HistGradientBoostingClassifier(class_weight="balanced",
                                             early_stopping=False, random_state=0, **g)
        clf.fit(F.loc[fit_mask, cols].values, y[fit_mask])
        prob = clf.predict_proba(F.loc[te, cols].values)[:, 1]
        roc = roc_auc_score(y_te, prob); pr = average_precision_score(y_te, prob)
        on_bound = (g["max_depth"] == 1 or g["learning_rate"] == 0.01 or g["max_iter"] == 50
                    or g["min_samples_leaf"] == 100 or g["l2_regularization"] == 10.0)
        out.append(dict(H=H, featset=fs, valid_roc=round(vroc, 4), test_roc=roc, test_pr=pr,
                        d_vs_LR_roc=roc - LR_REF[H][0], **g))
        print(f"  {fs:10s} best cfg {g}  valid ROC {vroc:.4f}")
        print(f"  {'':10s} test ROC {roc:.5f} (LR {LR_REF[H][0]:.5f}, d {roc-LR_REF[H][0]:+.5f})  "
              f"test PR {pr:.5f}  beyond-old-boundary={on_bound}")

pd.DataFrame(out).to_csv(SCR / "b_hgb_boundary.csv", index=False)
print(f"\n[t={time.time()-t0:.1f}s] wrote b_hgb_boundary.csv")
