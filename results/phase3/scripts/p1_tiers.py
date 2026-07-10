"""P1: three pre-registered mini feature tiers, each additive on M_sa (canonical
from P0-2), paired on the expanded test (2022-24), clustered bootstrap seed 0.

Baseline M_sa (9f) = M-role + {prior_pc_rate, ncg_log, vt_missing}.
Tiers:
  T_a prior_tjs (1f): 1 if the pitcher has a TJ surgery strictly before t
      (live sheet snapshot). Verified base-rate gradient 0.55x (protective).
  T_b pitch-mix (2f): br_share_l15 (SL+CU+FC share of last-15-games pitches),
      br_share_dev (last-5 share - last-15 share). Game windows (coverage ~100%).
      Plus a SWAP probe: torque-weighted chronic tw_chronic replacing pc_chronic
      (pre-registered weights FF/SI 1.00, SL/FC 0.95, CU 0.90, CH 0.85,
      unclassified 0.95 — elbow-varus-torque ordering from biomechanics lit).
  T_c rest structure (2f): short_rest_30 (# appearances in last 30d whose gap
      from the previous appearance is <=3 days), gap_std_90 (std of gaps between
      appearances in last 90d; needs >=3 games, else fit-set median impute).
Rolling-origin only for tiers whose paired dROC or dPR CI excludes 0 upward.
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
TW = {"FF": 1.00, "SI": 1.00, "SL": 0.95, "FC": 0.95, "CU": 0.90, "CH": 0.85}
gf_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    tot = g["total_pitches"].astype("float64").values
    br_n = (g["n_SL"] + g["n_CU"] + g["n_FC"]).astype("float64").values
    typed = np.zeros(len(g)); tw_sum = np.zeros(len(g))
    for pt, w in TW.items():
        n = g[f"n_{pt}"].astype("float64").values
        typed += n; tw_sum += w * n
    tw_game = tw_sum + 0.95 * (tot - typed)  # unclassified pitches at 0.95
    gf_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"), tot, br_n, tw_game)
DAY = np.timedelta64(1, "D")

tj = pd.read_csv(ROOT / "data/prospective/tj_live_clean_20260707.csv")
tj["surg_date"] = pd.to_datetime(tj["surg_date"])
surg_by_pid = {int(p): np.sort(g["surg_date"].values) for p, g in tj.groupby("mlbamid_i")}

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
prior_tjs = np.zeros(N); br_l15 = np.zeros(N); br_dev = np.zeros(N)
tw_chronic = np.zeros(N); short_rest_30 = np.zeros(N); gap_std_90 = np.full(N, np.nan)
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

    # rest structure from slim game dates (strictly before t)
    gdb = gd[before]
    if gdb.size >= 2:
        gaps = np.diff(gdb).astype("timedelta64[D]").astype(int)
        in30 = gdb[1:] >= t - np.timedelta64(30, "D")
        short_rest_30[i] = float(((gaps <= 3) & in30).sum())
        in90 = gdb[1:] >= t - np.timedelta64(90, "D")
        g90 = gaps[in90]
        if g90.size >= 2:
            gap_std_90[i] = float(np.std(g90))

    # gf-based: role + pitch mix + torque-weighted chronic
    rgd, rtot, rbr, rtw = gf_by_pid[pid]
    gbefore = rgd < t
    rm = gbefore & (rgd >= t - np.timedelta64(365, "D"))
    ng365 = int(rm.sum())
    start_share[i] = float((rtot[rm] >= 50).sum()) / ng365 if ng365 > 0 else 0.0
    ib = np.where(gbefore)[0]
    l15 = ib[-15:]; l5 = ib[-5:]
    t15 = rtot[l15].sum(); t5 = rtot[l5].sum()
    s15 = rbr[l15].sum() / t15 if t15 > 0 else 0.0
    s5 = rbr[l5].sum() / t5 if t5 > 0 else s15
    br_l15[i] = s15; br_dev[i] = s5 - s15
    g90m = gbefore & (rgd >= t - np.timedelta64(90, "D"))
    tw_chronic[i] = rtw[g90m].sum() / 90.0

    # prior TJS strictly before t
    s = surg_by_pid.get(pid)
    if s is not None and (s < t.astype("datetime64[ns]")).any():
        prior_tjs[i] = 1.0

F = pd.DataFrame(rows, columns=["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"])
F["start_share"] = start_share
F["prior_pc_rate"] = prior_pc_rate
F["ncg_log"] = np.log1p(ncg_all)
F["vt_missing"] = vt_missing
F["prior_tjs"] = prior_tjs
F["br_share_l15"] = br_l15
F["br_share_dev"] = br_dev
F["tw_chronic"] = tw_chronic
F["short_rest_30"] = short_rest_30
F["gap_std_90"] = gap_std_90
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
te = fold == "test"
F["gap_std_90"] = F["gap_std_90"].fillna(float(np.nanmedian(F.loc[fit_mask, "gap_std_90"])))
print(f"[t={time.time()-t0:.0f}s] features built {F.shape}  "
      f"corr(tw_chronic, pc_chronic)={np.corrcoef(F['tw_chronic'], F['pc_chronic'])[0,1]:.5f}  "
      f"prior_tjs share {F['prior_tjs'].mean():.3f}")

pid_test = pid_all[te]
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

def fit_predict(cols, y, mask_fit, mask_pred):
    sc = StandardScaler().fit(F.loc[mask_fit, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F.loc[mask_fit, cols]), y[mask_fit])
    return clf.predict_proba(sc.transform(F.loc[mask_pred, cols]))[:, 1], clf

M_SA = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month",
        "start_share", "prior_pc_rate", "ncg_log", "vt_missing"]
M_SA_ANCHOR = {90: 0.69203, 150: 0.70398}  # expanded-test ROC from p0b (gate)
TIERS = [
    ("Ta_prior_tjs", M_SA + ["prior_tjs"]),
    ("Tb_pitchmix", M_SA + ["br_share_l15", "br_share_dev"]),
    ("Tb_tw_swap", ["tw_chronic"] + M_SA[1:]),  # swap pc_chronic -> tw_chronic
    ("Tc_rest", M_SA + ["short_rest_30", "gap_std_90"]),
    ("T_all", M_SA + ["prior_tjs", "br_share_l15", "br_share_dev", "short_rest_30", "gap_std_90"]),
]
KS = [10, 20, 50]
out = []
for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]
    p0, _ = fit_predict(M_SA, y, fit_mask, te)
    roc0 = roc_auc_score(y_te, p0)
    assert abs(roc0 - M_SA_ANCHOR[H]) < 1e-4, f"M_sa GATE FAIL {roc0}"
    er0 = evrec(y_te, p0, KS)
    print(f"\n{'='*78}\nH={H}  M_sa GATE OK  ROC {roc0:.5f}  evrec {er0[10][0]}/{er0[20][0]}/{er0[50][0]} of {er0[10][1]}")
    for name, cols in TIERS:
        p1, clf = fit_predict(cols, y, fit_mask, te)
        roc1 = roc_auc_score(y_te, p1)
        ra, rb, pa, pb, nv = paired(y_te, p0, p1, RES_FULL)
        dr = rb - ra; dp = pb - pa
        drlo, drhi = ci(dr); dplo, dphi = ci(dp)
        er = evrec(y_te, p1, KS)
        fl = ("  dROC-EXCL0" if excl0(drlo, drhi) else "") + ("  dPR-EXCL0" if excl0(dplo, dphi) else "")
        new_coefs = {c: round(float(v), 3) for c, v in zip(cols, clf.coef_[0]) if c not in M_SA or name == "Tb_tw_swap"}
        out.append(dict(H=H, tier=name, roc=roc1, droc=float(np.median(dr)), droc_lo=drlo, droc_hi=drhi,
                        dpr=float(np.median(dp)), dpr_lo=dplo, dpr_hi=dphi,
                        ev10=er[10][0], ev20=er[20][0], ev50=er[50][0]))
        print(f"  {name:14s} ROC {roc1:.5f}  dROC {np.median(dr):+.5f} [{drlo:+.5f},{drhi:+.5f}]  "
              f"dPR {np.median(dp):+.5f} [{dplo:+.5f},{dphi:+.5f}]  evrec@50 {er[50][0]}{fl}")
        if name != "Tb_tw_swap":
            print(f"    new-feature coefs: {new_coefs}")

pd.DataFrame(out).to_csv(SCR / "p1_tiers.csv", index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote p1_tiers.csv")
