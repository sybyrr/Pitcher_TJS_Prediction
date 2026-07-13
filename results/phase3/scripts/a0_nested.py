"""A0-1: nested 3-candidate stability audit (spec v2, plan_progress "2026-07-13 (계속 3)").

Registered selection path per outer fold Y in {2022, 2023, 2024}:
  inner-fit = years <= Y-2, inner-valid = year Y-1.
  1) binary LR candidates {M-role(6f), M_bf(7f), M_sa(9f)} fit on inner-fit;
     pick max inner-valid ROC (mean of H90/H150); tie <0.002 -> fewer features.
  2) hazard form on the selected feature set: adopt if inner-valid mean ROC
     >= binary mean ROC - 0.005 (non-inferiority margin).
  3) refit the selected model on years <= Y-1, evaluate outer year Y
     (t+H <= 2024-12-31 label-reliability rule kept).
Optimism per fold/H = inner-valid ROC(selected, fit<=Y-2) - outer ROC(selected, fit<=Y-1).
Secondary = outer(selected) - outer(frozen M_sa+hazard, fit<=Y-1).
Report-only, canonical unchanged. Interpretation: LOWER BOUND of the full
adaptive-search optimism (candidate set is pre-fixed, not the actual search).
Feature build identical to v_codex.py (v4 data). Output: ../a0_nested.csv.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

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
tj = pd.read_csv(ROOT / "data/prospective/tj_live_clean_20260707.csv")
tj["surg_date"] = pd.to_datetime(tj["surg_date"])
surg_by_pid = {int(p): np.sort(g["surg_date"].values) for p, g in tj.groupby("mlbamid_i")}
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
F["pc_chronic_bf"] = np.where(F["pc_chronic"].values > 0, F["pc_chronic"].values, prior_pc_rate)
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6
print(f"[t={time.time()-t0:.0f}s] features built {F.shape}")

MROLE = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"]
M_BF = ["pc_chronic_bf", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share", "vt_missing"]
M_SA = MROLE + ["prior_pc_rate", "ncg_log", "vt_missing"]
CANDS = [("M-role", MROLE), ("M_bf", M_BF), ("M_sa", M_SA)]

# person-period interval labels (identical to v_codex.py)
S_MAX = 5
int_label = np.zeros((N, S_MAX), dtype=np.int8)
at_risk = np.ones((N, S_MAX), dtype=bool)
for i in range(N):
    surgs = surg_by_pid.get(int(pid_all[i]))
    if surgs is None:
        continue
    t = t_all[i]
    fired = False
    for s in range(S_MAX):
        if fired:
            at_risk[i, s] = False
            continue
        lo = t + np.timedelta64(30 * s, "D"); hi = t + np.timedelta64(30 * (s + 1), "D")
        if np.any((surgs > lo) & (surgs <= hi)):
            int_label[i, s] = 1; fired = True

Y_LAB = {H: cohort[f"label_H{H}_B0"].values.astype(int) for H in (90, 150)}
REL_END = np.datetime64("2024-12-31")

def fit_binary(cols, H, mask_fit):
    sc = StandardScaler().fit(F.loc[mask_fit, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F.loc[mask_fit, cols]), Y_LAB[H][mask_fit])
    return sc, clf

def pred_binary(model, cols, mask):
    sc, clf = model
    return clf.predict_proba(sc.transform(F.loc[mask, cols]))[:, 1]

def fit_hazard(cols, mask_fit):
    Xf = F.loc[mask_fit, cols].values
    fit_idx = np.where(mask_fit)[0]
    rX, rs, ry = [], [], []
    for j, i in enumerate(fit_idx):
        for s in range(S_MAX):
            if not at_risk[i, s]:
                break
            rX.append(Xf[j]); rs.append(s); ry.append(int(int_label[i, s]))
    Xpp = np.column_stack([np.asarray(rX), np.asarray(rs, dtype=float)])
    sc = StandardScaler().fit(Xpp)
    hz = LogisticRegression(max_iter=2000)
    hz.fit(sc.transform(Xpp), np.asarray(ry))
    return sc, hz

def pred_hazard(model, cols, mask):
    sc, hz = model
    Xt = F.loc[mask, cols].values
    h = np.empty((Xt.shape[0], S_MAX))
    for s in range(S_MAX):
        h[:, s] = hz.predict_proba(sc.transform(np.column_stack([Xt, np.full(Xt.shape[0], float(s))])))[:, 1]
    return {90: 1 - np.prod(1 - h[:, :3], axis=1), 150: 1 - np.prod(1 - h[:, :5], axis=1)}

def outer_mask(Y, H):
    return (year_all == Y) & ((t_all + np.timedelta64(H, "D")) <= REL_END)

def n_events(mask, H):
    y = Y_LAB[H]
    return cohort.loc[mask & (y == 1)].groupby(["pitcher", "next_surgery_date"]).ngroups

out = []
opt_all = {90: [], 150: []}
for Y in (2022, 2023, 2024):
    inner_fit = year_all <= Y - 2
    inner_val = year_all == Y - 1
    # step 1: binary candidate selection on inner-valid (mean ROC over H)
    cand_scores = {}
    for name, cols in CANDS:
        rocs = {}
        for H in (90, 150):
            mdl = fit_binary(cols, H, inner_fit)
            rocs[H] = roc_auc_score(Y_LAB[H][inner_val], pred_binary(mdl, cols, inner_val))
        cand_scores[name] = (np.mean([rocs[90], rocs[150]]), rocs)
    best_mean = max(v[0] for v in cand_scores.values())
    tied = [(name, cols) for name, cols in CANDS if cand_scores[name][0] >= best_mean - 0.002]
    sel_name, sel_cols = min(tied, key=lambda nc: len(nc[1]))  # tie rule: fewer features
    # step 2: hazard non-inferiority on inner-valid with selected cols
    hz_inner = fit_hazard(sel_cols, inner_fit)
    p_hz_val = pred_hazard(hz_inner, sel_cols, inner_val)
    hz_rocs = {H: roc_auc_score(Y_LAB[H][inner_val], p_hz_val[H]) for H in (90, 150)}
    hz_mean = np.mean([hz_rocs[90], hz_rocs[150]])
    form = "hazard" if hz_mean >= cand_scores[sel_name][0] - 0.005 else "binary"
    inner_roc = hz_rocs if form == "hazard" else cand_scores[sel_name][1]
    # step 3: refit selected on years <= Y-1, evaluate outer year Y
    refit_mask = year_all <= Y - 1
    if form == "hazard":
        hz_ref = fit_hazard(sel_cols, refit_mask)
    else:
        bin_ref = {H: fit_binary(sel_cols, H, refit_mask) for H in (90, 150)}
    frozen_ref = fit_hazard(M_SA, refit_mask)
    for H in (90, 150):
        om = outer_mask(Y, H)
        y_o = Y_LAB[H][om]
        if form == "hazard":
            p_o = pred_hazard(hz_ref, sel_cols, om)[H]
        else:
            p_o = pred_binary(bin_ref[H], sel_cols, om)
        p_frozen = pred_hazard(frozen_ref, M_SA, om)[H]
        roc_o = roc_auc_score(y_o, p_o)
        roc_fr = roc_auc_score(y_o, p_frozen)
        optim = inner_roc[H] - roc_o
        opt_all[H].append(optim)
        out.append(dict(Y=Y, H=H, selected=sel_name, form=form,
                        inner_roc=round(inner_roc[H], 4), outer_roc=round(roc_o, 4),
                        optimism=round(optim, 4), frozen_outer=round(roc_fr, 4),
                        sel_minus_frozen=round(roc_o - roc_fr, 4),
                        outer_events=n_events(om, H)))
    cs = {k: round(v[0], 4) for k, v in cand_scores.items()}
    print(f"Y={Y}: inner cand mean-ROC {cs}  -> selected {sel_name}  "
          f"hazard inner mean {hz_mean:.4f} vs binary {cand_scores[sel_name][0]:.4f} -> form={form}")
    for r in out[-2:]:
        print(f"    H={r['H']}: inner {r['inner_roc']:.4f}  outer {r['outer_roc']:.4f}  "
              f"optimism {r['optimism']:+.4f}  frozen-outer {r['frozen_outer']:.4f}  "
              f"sel-frozen {r['sel_minus_frozen']:+.4f}  events {r['outer_events']}")

print("\nSummary (3-fold mean optimism, lower bound of adaptive-search optimism):")
for H in (90, 150):
    print(f"  H={H}: mean optimism {np.mean(opt_all[H]):+.4f}  per-fold {[round(x,4) for x in opt_all[H]]}")

pd.DataFrame(out).to_csv(OUT / "a0_nested.csv", index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote {OUT / 'a0_nested.csv'}")
