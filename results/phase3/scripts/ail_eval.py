"""A-IL step 3: elbow-IL history features, additive test + blackout (spec v2).

Features (as-of t, strictly before, episode START = transaction date):
  elbow2y  = # new-elbow episodes in [t-730d, t)
  dsle_log = log1p(min(days since last new-elbow episode, 1500)); none -> cap
  anyil2y  = # all IL episodes in [t-730d, t)  (contrast)
Variant M_il = M_sa + 3, binary LR frozen protocol, paired vs M_sa on the
mature test (t+H<=2024-12-31), shared resamples seed 0. If EXCL0 -> rolling.
Blackout {30,60,90}d: episodes starting in [t-X, t) removed from all three
features (dsle falls back to the previous episode). Promotion rule
(pre-registered): canonical candidate only if the 60d-blackout paired dROC
point estimate stays positive on BOTH H; otherwise "disclosed-diagnosis
triage signal" — documented, excluded from canonical.
New-info + lead decomposition (B'-style) at top-50. Output ../ail_results.csv.
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
OUT = Path(__file__).resolve().parent.parent
t0 = time.time()

cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v4.parquet").sort_values(["t", "pitcher"]).reset_index(drop=True)
N = len(cohort)
slim = pd.read_parquet(ROOT / "data/prospective/slim_games_v4.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                        g["pitch_count"].astype("float64").values,
                        g["mean_release_speed"].astype("float64").values,
                        g["game_year"].values.astype(np.int64))
gf = pd.read_parquet(ROOT / "data/prospective/game_features_v4.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
role_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    role_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                             g["total_pitches"].astype("float64").values)
ep = pd.read_parquet(ROOT / "data/ail/il_episodes.parquet")
elbow_by_pid = {int(p): np.sort(g.loc[g["new_elbow"], "start"].values)
                for p, g in ep.groupby("pid")}
anyil_by_pid = {int(p): np.sort(g["start"].values) for p, g in ep.groupby("pid")}
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
M_SA = list(F.columns)
print(f"[t={time.time()-t0:.0f}s] base features built")

CAP = 1500.0
def il_features(blackout_days=0):
    """elbow2y, dsle_log, anyil2y as of t, excluding episodes starting in [t-blackout, t)."""
    out = np.zeros((N, 3))
    bo = np.timedelta64(blackout_days, "D")
    for i in range(N):
        pid = int(pid_all[i]); t = t_all[i]
        cut = t - bo
        es = elbow_by_pid.get(pid)
        if es is not None:
            e_ok = es[es < cut]
            out[i, 0] = ((e_ok >= t - np.timedelta64(730, "D")) & (e_ok < cut)).sum()
            if e_ok.size:
                out[i, 1] = np.log1p(min(float((t - e_ok.max()) / DAY), CAP))
            else:
                out[i, 1] = np.log1p(CAP)
        else:
            out[i, 1] = np.log1p(CAP)
        av = anyil_by_pid.get(pid)
        if av is not None:
            a_ok = av[av < cut]
            out[i, 2] = ((a_ok >= t - np.timedelta64(730, "D")) & (a_ok < cut)).sum()
    return out

IL_COLS = ["elbow2y", "dsle_log", "anyil2y"]
base_il = il_features(0)
for j, c in enumerate(IL_COLS):
    F[c] = base_il[:, j]
cov = (F["elbow2y"] > 0).mean()
print(f"coverage: elbow2y>0 in {cov:.2%} of windows; anyil2y>0 in {(F['anyil2y']>0).mean():.2%}")

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
REL_END = np.datetime64("2024-12-31")
te_base = (fold == "test") & (year_all <= 2024)
mature = {H: te_base & ((t_all + np.timedelta64(H, "D")) <= REL_END) for H in (90, 150)}

def build_resamples(pids, seed=0, nboot=1000):
    uniq = np.unique(pids)
    pos = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([pos[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            for _ in range(nboot)]

def paired(y_sub, a, b, resamples):
    ra, rb, pa, pb = [], [], [], []
    for idx in resamples:
        yb = y_sub[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ra.append(roc_auc_score(yb, a[idx])); rb.append(roc_auc_score(yb, b[idx]))
        pa.append(average_precision_score(yb, a[idx])); pb.append(average_precision_score(yb, b[idx]))
    return np.array(ra), np.array(rb), np.array(pa), np.array(pb)

def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

def excl0(lo, hi):
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)

def fit_predict(cols, H, mask_fit, mask_pred):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    sc = StandardScaler().fit(F.loc[mask_fit, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F.loc[mask_fit, cols]), y[mask_fit])
    return clf.predict_proba(sc.transform(F.loc[mask_pred, cols]))[:, 1], clf

out = []
res_cache = {}
print("\nM_il = M_sa + {elbow2y, dsle_log, anyil2y} — paired vs M_sa (mature test):")
for H in (90, 150):
    m = mature[H]
    y_m = cohort[f"label_H{H}_B0"].values.astype(int)[m]
    res = build_resamples(pid_all[m])
    res_cache[H] = (m, y_m, res)
    p0, _ = fit_predict(M_SA, H, fit_mask, m)
    p1, clf = fit_predict(M_SA + IL_COLS, H, fit_mask, m)
    ra, rb, pa, pb = paired(y_m, p0, p1, res)
    dr = rb - ra; dp = pb - pa
    drlo, drhi = ci(dr); dplo, dphi = ci(dp)
    fl = ("  dROC-EXCL0" if excl0(drlo, drhi) else "") + ("  dPR-EXCL0" if excl0(dplo, dphi) else "")
    coefs = dict(zip(M_SA + IL_COLS, clf.coef_[0].round(3)))
    out.append(dict(block="additive", H=H, variant="M_il", roc=round(roc_auc_score(y_m, p1), 4),
                    droc=round(float(np.median(dr)), 5), droc_lo=round(drlo, 5), droc_hi=round(drhi, 5),
                    dpr=round(float(np.median(dp)), 5), dpr_lo=round(dplo, 5), dpr_hi=round(dphi, 5)))
    print(f"  H={H}: M_sa {roc_auc_score(y_m, p0):.4f} -> M_il {roc_auc_score(y_m, p1):.4f}  "
          f"dROC {np.median(dr):+.5f} [{drlo:+.5f},{drhi:+.5f}]  dPR {np.median(dp):+.5f} [{dplo:+.5f},{dphi:+.5f}]{fl}")
    print(f"    IL coefs: {{k: coefs[k] for k in IL_COLS}}" if False else f"    IL coefs: {dict((k, coefs[k]) for k in IL_COLS)}")

print("\nBlackout sensitivity (paired dROC point vs M_sa, same resamples):")
for bo in (30, 60, 90):
    ilb = il_features(bo)
    for j, c in enumerate(IL_COLS):
        F[f"{c}_bo"] = ilb[:, j]
    cols_bo = M_SA + [f"{c}_bo" for c in IL_COLS]
    for H in (90, 150):
        m, y_m, res = res_cache[H]
        p0, _ = fit_predict(M_SA, H, fit_mask, m)
        p1, _ = fit_predict(cols_bo, H, fit_mask, m)
        ra, rb, pa, pb = paired(y_m, p0, p1, res)
        dr = rb - ra
        drlo, drhi = ci(dr)
        out.append(dict(block=f"blackout{bo}", H=H, variant=f"M_il_bo{bo}",
                        roc=round(roc_auc_score(y_m, p1), 4),
                        droc=round(float(np.median(dr)), 5), droc_lo=round(drlo, 5), droc_hi=round(drhi, 5),
                        dpr=np.nan, dpr_lo=np.nan, dpr_hi=np.nan))
        print(f"  blackout {bo}d H={H}: dROC {np.median(dr):+.5f} [{drlo:+.5f},{drhi:+.5f}]")

# new-info + lead decomposition at top-50 (H both)
print("\nNew-info / lead decomposition (top-50 per date):")
next_surg = pd.to_datetime(cohort["next_surgery_date"]).values
for H in (90, 150):
    m, y_m, res = res_cache[H]
    p0, _ = fit_predict(M_SA, H, fit_mask, m)
    p1, _ = fit_predict(M_SA + IL_COLS, H, fit_mask, m)
    t_m = t_all[m]
    sub_idx = np.where(m)[0]
    def caught_events(p):
        flag = np.zeros(int(m.sum()), dtype=bool)
        for d in np.unique(t_m):
            sel = np.where(t_m == d)[0]
            flag[sel[np.argsort(-p[sel], kind="stable")[:min(50, len(sel))]]] = True
        groups = {}
        for r in np.where(y_m == 1)[0]:
            key = (int(pid_all[sub_idx[r]]), pd.Timestamp(next_surg[sub_idx[r]]))
            groups.setdefault(key, []).append(r)
        return {k: any(flag[r] for r in rws) for k, rws in groups.items()}, groups
    c0, groups = caught_events(p0)
    c1, _ = caught_events(p1)
    new_caught = [k for k in groups if c1[k] and not c0[k]]
    lost = [k for k in groups if c0[k] and not c1[k]]
    il_hist = {}
    lead = []
    for k, rws in groups.items():
        pid, sd = k
        es = elbow_by_pid.get(pid)
        has = False
        if es is not None:
            before_any = es[es < np.datetime64(sd)]
            if before_any.size:
                has = True
                lead.append(float((np.datetime64(sd) - before_any.max()) / DAY))
        il_hist[k] = has
    cw = [k for k in groups if c1[k]]
    no_hist_share = np.mean([not il_hist[k] for k in cw]) if cw else np.nan
    print(f"  H={H}: caught M_sa {sum(c0.values())} -> M_il {sum(c1.values())}  "
          f"(new {len(new_caught)}, lost {len(lost)})  "
          f"caught-without-elbow-IL-history {no_hist_share:.0%}")
    if lead:
        print(f"    lead (last elbow IL -> surgery, all events with history, n={len(lead)}): "
              f"median {np.median(lead):.0f}d  P25 {np.percentile(lead,25):.0f}d  P75 {np.percentile(lead,75):.0f}d")

pd.DataFrame(out).to_csv(OUT / "ail_results.csv", index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote {OUT / 'ail_results.csv'}")
