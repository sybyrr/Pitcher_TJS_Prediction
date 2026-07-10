"""P0-1c + P0-2: M-role on the expanded test (2022-24) + structural-fix variants.

Data: cohort_v3 / slim_games_v3 / game_features_v3 (P0-1b, gated vs v2).
Models (LR frozen protocol, fit on train 2017-20 + valid 2021):
  M-role  = {pc_chronic, pc_acute_dev, days_since_last, vel_trend, month, start_share}
  M_bf    = M-role with pc_chronic backfilled by prior-season rate when the 90d
            calendar window is empty (pc_90==0), + vt_missing indicator   (7f)
  M_sa    = M-role + {prior_pc_rate, ncg_log, vt_missing}                 (9f)
prior_pc_rate = most-recent prior-season total pitches / 183 (strictly before t).

GATES: (1) M-role predictions restricted to 2022-23 test windows reproduce the
frozen anchors (H90 ROC 0.643680 / PR 0.035973; H150 0.643775 / 0.046995).
Then all evaluation moves to the EXPANDED test 2022-24 (new resamples, seed 0).

Reports per H in {90,150}:
  - expanded-test ROC/PR + clustered-bootstrap CI, event recall@{10,20,50},
    year-split ROC (2022/2023/2024) for each model
  - paired deltas M_bf/M_sa vs M-role (shared resamples)
  - rolling-origin direction check for M_bf vs M-role: test year Y in
    {2022,2023,2024}, fit on all years < Y (r1_rolling style), paired delta.
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

cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v3.parquet").sort_values(["t", "pitcher"]).reset_index(drop=True)
N = len(cohort)
slim = pd.read_parquet(ROOT / "data/prospective/slim_games_v3.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                        g["pitch_count"].astype("float64").values,
                        g["mean_release_speed"].astype("float64").values,
                        g["game_year"].values.astype(np.int64))
gf = pd.read_parquet(ROOT / "data/prospective/game_features_v3.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
role_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    role_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                             g["total_pitches"].astype("float64").values)
DAY = np.timedelta64(1, "D")

def pw_mean(sp, pc, mask):
    m = mask & ~np.isnan(sp)
    if not m.any():
        return np.nan
    w = pc[m].sum()
    return float((sp[m] * pc[m]).sum() / w) if w > 0 else np.nan

pid_all = cohort["pitcher"].values.astype(np.int64)
t_all = cohort["t"].values.astype("datetime64[ns]")
year_all = cohort["year"].values.astype(np.int64)
month_all = cohort["month"].values.astype(np.float64)
ncg_all = cohort["n_career_games"].values.astype(np.float64)

rows = []
prior_pc_rate = np.zeros(N); vt_missing = np.zeros(N); start_share = np.zeros(N)
for i in range(N):
    pid = int(pid_all[i]); t = t_all[i].astype("datetime64[D]")
    gd, pc, sp, gy = by_pid[pid]
    before = gd < t
    d30 = before & (gd >= t - np.timedelta64(30, "D"))
    d90 = before & (gd >= t - np.timedelta64(90, "D"))
    pc_30 = pc[d30].sum(); pc_90 = pc[d90].sum()
    dsl = float((t - gd[before].max()) / DAY)
    vmean_30 = pw_mean(sp, pc, d30)
    yr = int(year_all[i])
    vmean_prior = pw_mean(sp, pc, before & (gy < yr))
    if np.isnan(vmean_prior):
        vmean_prior = pw_mean(sp, pc, before & (gd < t - np.timedelta64(30, "D")))
    if np.isnan(vmean_30) or np.isnan(vmean_prior):
        vel_trend = 0.0; vt_missing[i] = 1.0
    else:
        vel_trend = vmean_30 - vmean_prior
    prior_years = gy[before & (gy < yr)]
    if prior_years.size:
        m_py = before & (gy == prior_years.max())
        prior_pc_rate[i] = pc[m_py].sum() / 183.0
    rows.append((pc_90 / 90.0, pc_30 / 30.0 - pc_90 / 90.0, dsl, vel_trend, month_all[i]))
    rgd, rtp = role_by_pid[pid]
    rm = (rgd < t) & (rgd >= t - np.timedelta64(365, "D"))
    ng365 = int(rm.sum())
    start_share[i] = float((rtp[rm] >= 50).sum()) / ng365 if ng365 > 0 else 0.0

F = pd.DataFrame(rows, columns=["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"])
F["start_share"] = start_share
F["prior_pc_rate"] = prior_pc_rate
F["ncg_log"] = np.log1p(ncg_all)
F["vt_missing"] = vt_missing
F["pc_chronic_bf"] = np.where(F["pc_chronic"].values > 0, F["pc_chronic"].values, prior_pc_rate)
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6
print(f"[t={time.time()-t0:.0f}s] features built {F.shape}")

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
te = fold == "test"
te23 = te & (year_all <= 2023)  # gate subset

# expanded-test machinery
t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
pid_test = pid_all[te]
year_test = year_all[te]
next_surg = pd.to_datetime(cohort["next_surgery_date"]).values

def build_resamples(rows_sub, seed=0, nboot=1000):
    pids = pid_test[rows_sub]
    uniq = np.unique(pids)
    pos = {p: rows_sub[np.where(pids == p)[0]] for p in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([pos[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            for _ in range(nboot)]

RES_FULL = build_resamples(np.arange(int(te.sum())))
print(f"[t={time.time()-t0:.0f}s] expanded-test resamples (n_pit={len(np.unique(pid_test))}, "
      f"windows={int(te.sum())}, dates={len(test_dates)})")

def paired(y_te, a, b, resamples):
    ra, rb, pa, pb = [], [], [], []
    for idx in resamples:
        yb = y_te[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ra.append(roc_auc_score(yb, a[idx])); rb.append(roc_auc_score(yb, b[idx]))
        pa.append(average_precision_score(yb, a[idx])); pb.append(average_precision_score(yb, b[idx]))
    return np.array(ra), np.array(rb), np.array(pa), np.array(pb), len(ra)

def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

def excl0(lo, hi):
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)

def topk(score, k):
    flag = np.zeros(len(score), dtype=bool)
    for d in test_dates:
        sel = np.where(t_test == d)[0]
        kk = min(k, len(sel))
        flag[sel[np.argsort(-score[sel], kind="stable")[:kk]]] = True
    return flag

def evrec(y_te, score, ks):
    pr = np.where(y_te == 1)[0]
    keys = list(zip(pid_test[pr].tolist(),
                    pd.to_datetime(next_surg[te][pr]).astype("datetime64[ns]").tolist()))
    groups = {}
    for r, k in zip(pr, keys):
        groups.setdefault(k, []).append(r)
    out = {}
    for k in ks:
        fl = topk(score, k)
        out[k] = (sum(1 for rs in groups.values() if fl[rs].any()), len(groups))
    return out

def fit_on(cols, y, mask):
    sc = StandardScaler().fit(F.loc[mask, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F.loc[mask, cols]), y[mask])
    return sc, clf

def predict(sc, clf, cols, rows_mask):
    return clf.predict_proba(sc.transform(F.loc[rows_mask, cols]))[:, 1]

MROLE = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"]
M_BF = ["pc_chronic_bf", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share", "vt_missing"]
M_SA = MROLE + ["prior_pc_rate", "ncg_log", "vt_missing"]
MODELS = [("M-role", MROLE), ("M_bf", M_BF), ("M_sa", M_SA)]
ANCHOR = {90: (0.643680, 0.035973), 150: (0.643775, 0.046995)}
KS = [10, 20, 50]
out_rows, out_deltas, out_rolling = [], [], []

for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]
    print(f"\n{'='*78}\nH={H}  expanded test: windows {int(te.sum())}  pos {int(y_te.sum())}  "
          f"(2024 adds {int(y[te & (year_all==2024)].sum())} window-pos)")

    probs = {}
    for name, cols in MODELS:
        sc, clf = fit_on(cols, y, fit_mask)
        prob = predict(sc, clf, cols, te)
        probs[name] = prob
        if name == "M-role":
            # GATE on the 2022-23 subset
            sub = year_test <= 2023
            g_roc = roc_auc_score(y_te[sub], prob[sub]); g_pr = average_precision_score(y_te[sub], prob[sub])
            a_roc, a_pr = ANCHOR[H]
            assert abs(g_roc - a_roc) < 1e-4 and abs(g_pr - a_pr) < 1e-4, f"GATE FAIL {g_roc} {g_pr}"
            print(f"  GATE M-role on 2022-23 subset: ROC {g_roc:.6f} PR {g_pr:.6f} == anchors OK")
        roc = roc_auc_score(y_te, prob); pr = average_precision_score(y_te, prob)
        ra, _, pa, _, nv = paired(y_te, prob, prob, RES_FULL)
        er = evrec(y_te, prob, KS)
        ys = {yr: roc_auc_score(y_te[year_test == yr], prob[year_test == yr]) for yr in (2022, 2023, 2024)}
        out_rows.append(dict(H=H, model=name, roc=roc, roc_lo=ci(ra)[0], roc_hi=ci(ra)[1],
                             pr=pr, pr_lo=ci(pa)[0], pr_hi=ci(pa)[1],
                             evrec_total=er[10][1], ev10=er[10][0], ev20=er[20][0], ev50=er[50][0],
                             roc_2022=ys[2022], roc_2023=ys[2023], roc_2024=ys[2024]))
        print(f"  {name:7s} ROC {roc:.5f} [{ci(ra)[0]:.4f},{ci(ra)[1]:.4f}]  PR {pr:.5f}  "
              f"evrec {er[10][0]}/{er[20][0]}/{er[50][0]} of {er[10][1]}  "
              f"year ROC 22/23/24 = {ys[2022]:.3f}/{ys[2023]:.3f}/{ys[2024]:.3f}")

    for name in ("M_bf", "M_sa"):
        ra, rb, pa, pb, nv = paired(y_te, probs["M-role"], probs[name], RES_FULL)
        dr = rb - ra; dp = pb - pa
        drlo, drhi = ci(dr); dplo, dphi = ci(dp)
        fl = ("  dROC-EXCL0" if excl0(drlo, drhi) else "") + ("  dPR-EXCL0" if excl0(dplo, dphi) else "")
        out_deltas.append(dict(H=H, contrast=f"{name}-Mrole",
                               droc=float(np.median(dr)), droc_lo=drlo, droc_hi=drhi,
                               dpr=float(np.median(dp)), dpr_lo=dplo, dpr_hi=dphi, nboot=nv))
        print(f"  PAIRED {name}-Mrole: dROC {np.median(dr):+.5f} [{drlo:+.5f},{drhi:+.5f}]  "
              f"dPR {np.median(dp):+.5f} [{dplo:+.5f},{dphi:+.5f}]{fl}")

    # rolling-origin direction check M_bf vs M-role
    for Y in (2022, 2023, 2024):
        fit_m = year_all < Y
        te_m = year_all == Y
        sc0, c0 = fit_on(MROLE, y, fit_m); p0 = predict(sc0, c0, MROLE, te_m)
        sc1, c1 = fit_on(M_BF, y, fit_m); p1 = predict(sc1, c1, M_BF, te_m)
        yy = y[te_m]
        # per-year clustered resamples
        pids_y = pid_all[te_m]
        uniq = np.unique(pids_y)
        posy = {p: np.where(pids_y == p)[0] for p in uniq}
        rng = np.random.default_rng(0)
        res_y = [np.concatenate([posy[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
                 for _ in range(1000)]
        ra, rb, pa, pb, nv = paired(yy, p0, p1, res_y)
        dr = rb - ra; dp = pb - pa
        out_rolling.append(dict(H=H, year=Y, roc_base=roc_auc_score(yy, p0), roc_bf=roc_auc_score(yy, p1),
                                droc=float(np.median(dr)), droc_lo=ci(dr)[0], droc_hi=ci(dr)[1],
                                dpr=float(np.median(dp)), dpr_lo=ci(dp)[0], dpr_hi=ci(dp)[1]))
        print(f"  ROLLING Y={Y}: base ROC {roc_auc_score(yy, p0):.4f} -> bf {roc_auc_score(yy, p1):.4f}  "
              f"dROC {np.median(dr):+.4f} [{ci(dr)[0]:+.4f},{ci(dr)[1]:+.4f}]  dPR {np.median(dp):+.4f}")

pd.DataFrame(out_rows).to_csv(SCR / "p0_cells.csv", index=False)
pd.DataFrame(out_deltas).to_csv(SCR / "p0_deltas.csv", index=False)
pd.DataFrame(out_rolling).to_csv(SCR / "p0_rolling.csv", index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote p0_cells.csv / p0_deltas.csv / p0_rolling.csv")
