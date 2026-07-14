"""A0-3: hazard cluster/form sensitivities (spec v2, report-only).

vs canonical plain hazard (M_sa + scalar s, fit train+valid), mature test
(t+H<=2024-12-31), point ROC deltas:
  (i)   event-weighted fit: positive person-period rows weighted
        1 / (#fit landmarks in which that same surgery fires); negatives 1.
  (ii)  categorical-s baseline: scalar s -> 4 dummies (s=1..4, base s=0).
  (iii) cluster-level full-refit bootstrap: B=200 resamples of FIT pitchers
        (seed 0), refit hazard each time, ROC on the fixed mature test ->
        percentile CI reflecting fit-side cluster uncertainty.
  (iv)  historical outcome-dependent stress: keep each fit surgery's positive
        interval only at the landmark closest to surgery; earlier negative
        rows remain. This is retained for history, not called deduplication.
  (v)   outcome-blind single-landmark sensitivity (pre-specified rule): before
        examining labels, retain the earliest eligible decision date within
        each (pitcher, calendar season). Keep all person-period rows belonging
        to those selected windows. Report fit-only and same-rule evaluation.

Feature build identical to v_codex.py (v4 data). The historical ../a0_sens.csv
is intentionally not overwritten. Corrected output: ../a0_sens_corrected.csv.
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
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6
M_SA = list(F.columns)
Xall = F[M_SA].values
print(f"[t={time.time()-t0:.0f}s] features built")

S_MAX = 5
int_label = np.zeros((N, S_MAX), dtype=np.int8)
at_risk = np.ones((N, S_MAX), dtype=bool)
surg_of = {}  # (window i, interval s) with int_label 1 -> surgery date
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
        hit = (surgs > lo) & (surgs <= hi)
        if np.any(hit):
            int_label[i, s] = 1; fired = True
            surg_of[(i, s)] = surgs[hit][0]

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")
fit_idx = np.where(fit_mask)[0]

# person-period rows for the fit set
rX, rs, ry, rw = [], [], [], []
for i in fit_idx:
    for s in range(S_MAX):
        if not at_risk[i, s]:
            break
        rX.append(Xall[i]); rs.append(s); ry.append(int(int_label[i, s])); rw.append(i)
Xf = np.asarray(rX); sf = np.asarray(rs, dtype=float)
yf = np.asarray(ry); wf = np.asarray(rw)

def fit_hz(X, s, y, weight=None, cat_s=False):
    scol = pd.get_dummies(s.astype(int)).values[:, 1:].astype(float) if cat_s else s.reshape(-1, 1)
    Xpp = np.hstack([X, scol])
    sc = StandardScaler().fit(Xpp)
    hz = LogisticRegression(max_iter=2000)
    hz.fit(sc.transform(Xpp), y, sample_weight=weight)
    return sc, hz

REL_END = np.datetime64("2024-12-31")
te_base = (fold == "test") & (year_all <= 2024)
test_masks = {H: te_base & ((t_all + np.timedelta64(H, "D")) <= REL_END) for H in (90, 150)}
def test_roc(model, cat_s=False, masks=None):
    """Evaluate final-window P(H) on supplied outcome-independent masks."""
    sc, hz = model
    masks = test_masks if masks is None else masks
    out = {}
    for H, smax in ((90, 3), (150, 5)):
        mask = masks[H]
        Xt = Xall[mask]
        y = cohort[f"label_H{H}_B0"].values.astype(int)[mask]
        h = np.empty((Xt.shape[0], S_MAX))
        for s in range(S_MAX):
            scol = (np.eye(S_MAX)[s][1:] * np.ones((Xt.shape[0], S_MAX - 1))) if cat_s else np.full((Xt.shape[0], 1), float(s))
            h[:, s] = hz.predict_proba(sc.transform(np.hstack([Xt, scol])))[:, 1]
        p = 1 - np.prod(1 - h[:, :smax], axis=1)
        out[H] = roc_auc_score(y, p)
    return out

def first_eligible_pitcher_season(mask):
    """Choose first t per pitcher-season without reading outcome columns."""
    candidate = np.where(mask)[0]
    frame = pd.DataFrame({
        "row": candidate,
        "pitcher": pid_all[candidate],
        "year": year_all[candidate],
        "t": t_all[candidate],
    }).sort_values(["pitcher", "year", "t", "row"])
    chosen = frame.groupby(["pitcher", "year"], sort=False)["row"].first().values
    selected = np.zeros(N, dtype=bool)
    selected[chosen.astype(int)] = True
    return selected

canon = fit_hz(Xf, sf, yf)
roc_c = test_roc(canon)
print(f"canonical: H90 {roc_c[90]:.4f}  H150 {roc_c[150]:.4f}")
out = [dict(variant="canonical", H90=round(roc_c[90], 4), H150=round(roc_c[150], 4),
            d90=0.0, d150=0.0, note="")]

# (i) event-weighted
n_lm = {}
for (i, s), sd in surg_of.items():
    if fit_mask[i]:
        n_lm[(int(pid_all[i]), sd)] = n_lm.get((int(pid_all[i]), sd), 0) + 1
weights = np.ones(len(yf))
pos_rows = np.where(yf == 1)[0]
for r in pos_rows:
    i, s = int(wf[r]), int(sf[r])
    weights[r] = 1.0 / n_lm[(int(pid_all[i]), surg_of[(i, s)])]
mdl = fit_hz(Xf, sf, yf, weight=weights)
r = test_roc(mdl)
out.append(dict(variant="event_weighted", H90=round(r[90], 4), H150=round(r[150], 4),
                d90=round(r[90] - roc_c[90], 4), d150=round(r[150] - roc_c[150], 4),
                note=f"pos rows {len(pos_rows)}, distinct fit surgeries {len(n_lm)}"))
print(f"event_weighted: H90 {r[90]:.4f} ({r[90]-roc_c[90]:+.4f})  H150 {r[150]:.4f} ({r[150]-roc_c[150]:+.4f})")

# (ii) categorical s
mdl = fit_hz(Xf, sf, yf, cat_s=True)
r = test_roc(mdl, cat_s=True)
out.append(dict(variant="categorical_s", H90=round(r[90], 4), H150=round(r[150], 4),
                d90=round(r[90] - roc_c[90], 4), d150=round(r[150] - roc_c[150], 4), note=""))
print(f"categorical_s: H90 {r[90]:.4f} ({r[90]-roc_c[90]:+.4f})  H150 {r[150]:.4f} ({r[150]-roc_c[150]:+.4f})")

# (v) outcome-blind single-landmark sensitivity. Selection is performed only
# from pitcher, calendar year, decision date, eligibility/fold, and maturity
# boundary. No surgery date or label column enters first_eligible_pitcher_season.
fit_first_window = first_eligible_pitcher_season(fit_mask)
keep_pp_first = fit_first_window[wf]
test_first_masks = {
    H: first_eligible_pitcher_season(test_masks[H]) for H in (90, 150)
}
assert int(keep_pp_first.sum()) > 0
assert np.all(fit_first_window <= fit_mask)
for H in (90, 150):
    assert np.all(test_first_masks[H] <= test_masks[H])

mdl_first = fit_hz(Xf[keep_pp_first], sf[keep_pp_first], yf[keep_pp_first])
r_first_fit_all_eval = test_roc(mdl_first)
r_canon_first_eval = test_roc(canon, masks=test_first_masks)
r_first_fit_first_eval = test_roc(mdl_first, masks=test_first_masks)
fit_first_n = int(fit_first_window.sum())
fit_first_pos = int(yf[keep_pp_first].sum())
eval_counts = {
    H: (
        int(test_first_masks[H].sum()),
        int(cohort[f"label_H{H}_B0"].values.astype(int)[test_first_masks[H]].sum()),
    )
    for H in (90, 150)
}
rule = "earliest eligible t per (pitcher, calendar year); labels/surgery dates unused"
out.append(dict(
    variant="outcome_blind_first_pitcher_season_fit_only",
    H90=round(r_first_fit_all_eval[90], 4), H150=round(r_first_fit_all_eval[150], 4),
    d90=round(r_first_fit_all_eval[90] - roc_c[90], 4),
    d150=round(r_first_fit_all_eval[150] - roc_c[150], 4),
    delta_reference="canonical on all mature test landmarks",
    selection_uses_outcome=False, fit_landmark_rule=rule,
    eval_landmark_rule="all mature test landmarks",
    n_fit_windows=fit_first_n, n_fit_positive_intervals=fit_first_pos,
    n_eval_H90=int(test_masks[90].sum()),
    positive_windows_eval_H90=int(cohort["label_H90_B0"].values.astype(int)[test_masks[90]].sum()),
    n_eval_H150=int(test_masks[150].sum()),
    positive_windows_eval_H150=int(cohort["label_H150_B0"].values.astype(int)[test_masks[150]].sum()),
    note="fit-side repeated-landmark sensitivity; evaluation unchanged",
))
out.append(dict(
    variant="canonical_first_pitcher_season_eval",
    H90=round(r_canon_first_eval[90], 4), H150=round(r_canon_first_eval[150], 4),
    d90=round(r_canon_first_eval[90] - roc_c[90], 4),
    d150=round(r_canon_first_eval[150] - roc_c[150], 4),
    delta_reference="canonical on all mature test landmarks (estimand changes)",
    selection_uses_outcome=False, fit_landmark_rule="all fit landmarks",
    eval_landmark_rule=rule, n_fit_windows=int(fit_idx.size),
    n_fit_positive_intervals=int(yf.sum()),
    n_eval_H90=eval_counts[90][0], positive_windows_eval_H90=eval_counts[90][1],
    n_eval_H150=eval_counts[150][0], positive_windows_eval_H150=eval_counts[150][1],
    note="evaluation-side sensitivity; delta is descriptive across estimands",
))
out.append(dict(
    variant="outcome_blind_first_pitcher_season_fit_eval",
    H90=round(r_first_fit_first_eval[90], 4), H150=round(r_first_fit_first_eval[150], 4),
    d90=round(r_first_fit_first_eval[90] - r_canon_first_eval[90], 4),
    d150=round(r_first_fit_first_eval[150] - r_canon_first_eval[150], 4),
    delta_reference="canonical fitted model on same first-landmark evaluation set",
    selection_uses_outcome=False, fit_landmark_rule=rule,
    eval_landmark_rule=rule, n_fit_windows=fit_first_n,
    n_fit_positive_intervals=fit_first_pos,
    n_eval_H90=eval_counts[90][0], positive_windows_eval_H90=eval_counts[90][1],
    n_eval_H150=eval_counts[150][0], positive_windows_eval_H150=eval_counts[150][1],
    note="fully outcome-blind one-landmark-per-pitcher-season sensitivity",
))
print(
    "outcome-blind first pitcher-season: "
    f"fit windows={fit_first_n}, positive intervals={fit_first_pos}; "
    f"fit-only/all-eval H90/H150={r_first_fit_all_eval[90]:.4f}/{r_first_fit_all_eval[150]:.4f}; "
    f"canonical first-eval={r_canon_first_eval[90]:.4f}/{r_canon_first_eval[150]:.4f}; "
    f"first-fit first-eval={r_first_fit_first_eval[90]:.4f}/{r_first_fit_first_eval[150]:.4f}"
)

# (iv) historical outcome-dependent stress (closest positive to surgery).
# It deletes only duplicate positive interval rows and leaves their earlier
# negative rows, so it is deliberately not called physical deduplication.
best_lm = {}
for (i, s), sd in surg_of.items():
    if not fit_mask[i]:
        continue
    key = (int(pid_all[i]), sd)
    gap = (sd - t_all[i]) / np.timedelta64(1, "D")
    if key not in best_lm or gap < best_lm[key][1]:
        best_lm[key] = (i, gap)
keep_win = {v[0] for v in best_lm.values()}
drop = np.zeros(len(yf), dtype=bool)
for r_ in pos_rows:
    i, s = int(wf[r_]), int(sf[r_])
    if i not in keep_win:
        drop[r_] = True
mdl = fit_hz(Xf[~drop], sf[~drop], yf[~drop])
r = test_roc(mdl)
out.append(dict(variant="historical_outcome_dependent_closest_surgery_positive_only",
                H90=round(r[90], 4), H150=round(r[150], 4),
                d90=round(r[90] - roc_c[90], 4), d150=round(r[150] - roc_c[150], 4),
                delta_reference="canonical on all mature test landmarks",
                selection_uses_outcome=True,
                fit_landmark_rule="closest positive landmark selected using surgery date; negatives retained",
                eval_landmark_rule="all mature test landmarks",
                note=f"historical stress: dropped {int(drop.sum())} of {len(pos_rows)} positive rows"))
print(f"historical outcome-dependent stress: H90 {r[90]:.4f} ({r[90]-roc_c[90]:+.4f})  H150 {r[150]:.4f} ({r[150]-roc_c[150]:+.4f})  "
      f"dropped {int(drop.sum())}/{len(pos_rows)} pos rows")

# (iii) cluster-level full-refit bootstrap, B=200, seed 0
pp_by_pid = {}
for r_, i in enumerate(wf):
    pp_by_pid.setdefault(int(pid_all[i]), []).append(r_)
pp_by_pid = {p: np.asarray(v) for p, v in pp_by_pid.items()}
fit_pids = np.unique(pid_all[fit_idx])
rng = np.random.default_rng(0)
B = 200
boot = {90: [], 150: []}
for b in range(B):
    draw = rng.choice(fit_pids, size=len(fit_pids), replace=True)
    rows_b = np.concatenate([pp_by_pid[p] for p in draw if p in pp_by_pid])
    if yf[rows_b].sum() == 0:
        continue
    mdl = fit_hz(Xf[rows_b], sf[rows_b], yf[rows_b])
    r = test_roc(mdl)
    boot[90].append(r[90]); boot[150].append(r[150])
    if (b + 1) % 50 == 0:
        print(f"  [t={time.time()-t0:.0f}s] refit bootstrap {b+1}/{B}")
for H in (90, 150):
    arr = np.asarray(boot[H])
    lo, hi = np.percentile(arr, [2.5, 97.5])
    out.append(dict(variant=f"refit_boot_H{H}", H90=np.nan, H150=np.nan,
                    d90=np.nan, d150=np.nan,
                    note=f"mean {arr.mean():.4f} CI [{lo:.4f},{hi:.4f}] B={len(arr)}"))
    print(f"refit bootstrap H={H}: mean {arr.mean():.4f}  CI [{lo:.4f},{hi:.4f}]  (test-side canonical CI: "
          f"{'0.643-0.759' if H==90 else '0.645-0.746'})")

corrected_path = OUT / "a0_sens_corrected.csv"
pd.DataFrame(out).to_csv(corrected_path, index=False)
print(f"\n[t={time.time()-t0:.0f}s] wrote {corrected_path}")
