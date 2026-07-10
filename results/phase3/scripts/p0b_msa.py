"""P0-2b: M_sa confirmation — rolling-origin + component decomposition.

M_sa (= M-role + prior_pc_rate + ncg_log + vt_missing) showed dROC +0.017
[+0.001,+0.030] EXCL0 at H150 on the expanded test. Before adoption:
  (1) rolling-origin direction check vs M-role (test year 2022/2023/2024,
      fit on all years < Y) — criterion: dROC point > 0 in a clear majority.
  (2) component ablation on the frozen fit (train+valid, expanded test):
      +prior_pc_rate / +ncg_log / +vt_missing alone, and prior+vtmiss (7f).
  (3) ridge sensitivity C=0.1 for M_sa (default LR is already l2 C=1.0).
Feature build identical to p0_eval.py.
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
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6
print(f"[t={time.time()-t0:.0f}s] features built")

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
te = fold == "test"
pid_test = pid_all[te]
year_test = year_all[te]
t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
next_surg = pd.to_datetime(cohort["next_surgery_date"]).values

def build_resamples(pids, seed=0, nboot=1000):
    uniq = np.unique(pids)
    pos = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([pos[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            for _ in range(nboot)]

RES_FULL = build_resamples(pid_test)

def paired(y_sub, a, b, resamples):
    ra, rb, pa, pb = [], [], [], []
    for idx in resamples:
        yb = y_sub[idx]
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

def evrec(y_sub, score, ks):
    pr = np.where(y_sub == 1)[0]
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

def fit_predict(cols, y, mask_fit, mask_pred, C=1.0):
    sc = StandardScaler().fit(F.loc[mask_fit, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=C)
    clf.fit(sc.transform(F.loc[mask_fit, cols]), y[mask_fit])
    return clf.predict_proba(sc.transform(F.loc[mask_pred, cols]))[:, 1], clf

MROLE = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"]
VARIANTS = [
    ("M_sa", MROLE + ["prior_pc_rate", "ncg_log", "vt_missing"], 1.0),
    ("+prior_only", MROLE + ["prior_pc_rate"], 1.0),
    ("+ncg_only", MROLE + ["ncg_log"], 1.0),
    ("+vtmiss_only", MROLE + ["vt_missing"], 1.0),
    ("+prior+vtmiss", MROLE + ["prior_pc_rate", "vt_missing"], 1.0),
    ("+prior+ncg", MROLE + ["prior_pc_rate", "ncg_log"], 1.0),
    ("M_sa_C0.1", MROLE + ["prior_pc_rate", "ncg_log", "vt_missing"], 0.1),
]
KS = [10, 20, 50]
out = []
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]
    p0, _ = fit_predict(MROLE, y, fit_mask, te)
    roc0 = roc_auc_score(y_te, p0)
    er0 = evrec(y_te, p0, KS)
    print(f"\n{'='*78}\nH={H}  M-role expanded ROC {roc0:.5f}  evrec@50 {er0[50][0]}/{er0[50][1]}")
    for name, cols, C in VARIANTS:
        p1, clf = fit_predict(cols, y, fit_mask, te, C=C)
        roc1 = roc_auc_score(y_te, p1)
        ra, rb, pa, pb, nv = paired(y_te, p0, p1, RES_FULL)
        dr = rb - ra; dp = pb - pa
        drlo, drhi = ci(dr); dplo, dphi = ci(dp)
        er = evrec(y_te, p1, KS)
        fl = ("  dROC-EXCL0" if excl0(drlo, drhi) else "") + ("  dPR-EXCL0" if excl0(ci(dp)[0], ci(dp)[1]) else "")
        out.append(dict(H=H, model=name, roc=roc1, droc=float(np.median(dr)), droc_lo=drlo, droc_hi=drhi,
                        dpr=float(np.median(dp)), dpr_lo=dplo, dpr_hi=dphi,
                        ev50=er[50][0], ev50_total=er[50][1]))
        coefs = dict(zip(cols, clf.coef_[0].round(3)))
        print(f"  {name:14s} ROC {roc1:.5f}  dROC {np.median(dr):+.5f} [{drlo:+.5f},{drhi:+.5f}]  "
              f"evrec@50 {er[50][0]}{fl}")
        if name == "M_sa":
            print(f"    coef: {coefs}")

    # rolling-origin for M_sa vs M-role
    for Y in (2022, 2023, 2024):
        fit_m = year_all < Y
        te_m = year_all == Y
        pa0, _ = fit_predict(MROLE, y, fit_m, te_m)
        pa1, _ = fit_predict(MROLE + ["prior_pc_rate", "ncg_log", "vt_missing"], y, fit_m, te_m)
        yy = y[te_m]
        res_y = build_resamples(pid_all[te_m])
        ra, rb, pA, pB, nv = paired(yy, pa0, pa1, res_y)
        dr = rb - ra; dp = pB - pA
        print(f"  ROLLING M_sa Y={Y}: base {roc_auc_score(yy, pa0):.4f} -> sa {roc_auc_score(yy, pa1):.4f}  "
              f"dROC {np.median(dr):+.4f} [{ci(dr)[0]:+.4f},{ci(dr)[1]:+.4f}]  dPR {np.median(dp):+.4f}")
        out.append(dict(H=H, model=f"ROLL_{Y}", roc=roc_auc_score(yy, pa1),
                        droc=float(np.median(dr)), droc_lo=ci(dr)[0], droc_hi=ci(dr)[1],
                        dpr=float(np.median(dp)), dpr_lo=ci(dp)[0], dpr_hi=ci(dp)[1],
                        ev50=np.nan, ev50_total=np.nan))

pd.DataFrame(out).to_csv(SCR / "p0b_msa.csv", index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote p0b_msa.csv")
