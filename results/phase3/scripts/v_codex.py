"""Verify codex-audit claims by direct recomputation (read-only, v4 data).

C1 mature-label representative numbers: rule t+H <= 2024-12-31
   (H90: test = all 2022-24 dates; H150: drop 2024-09-01). Hazard(M_sa),
   fit 2017-21. Claim: H90 0.701 [0.643,0.759] ev75; H150 0.696 [0.645,0.746] ev80.
C2 hazard calibration on mature test: prevalence vs mean pred, top-decile
   actual/pred, recalibration slope. Claim: slope ~1.79/1.84, ~2x underpred top.
C3 within-role ROC of hazard scores + role of caught events @50.
   Claim: RP-within 0.633/0.640, RP events caught 1/49, 1/53.
C4 seen (in fit) vs novel pitcher ROC. Claim: ~0.71-0.72 vs ~0.65.
C5 Monte Carlo random top-50 event-recall lift (2000 reps).
   Claim: true lift ~1.5-1.8x, not 2-3x.
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

fold = cohort["fold_main"].values
fit_mask = (fold == "train") | (fold == "valid")

# interval labels for hazard
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

Xf = F.loc[fit_mask, M_SA].values
fit_idx = np.where(fit_mask)[0]
rX, rs, ry = [], [], []
for j, i in enumerate(fit_idx):
    for s in range(S_MAX):
        if not at_risk[i, s]:
            break
        rX.append(Xf[j]); rs.append(s); ry.append(int(int_label[i, s]))
Xpp = np.column_stack([np.asarray(rX), np.asarray(rs, dtype=float)])
scpp = StandardScaler().fit(Xpp)
hz = LogisticRegression(max_iter=2000)
hz.fit(scpp.transform(Xpp), np.asarray(ry))

def hazard_p(rows_mask):
    Xt = F.loc[rows_mask, M_SA].values
    h = np.empty((Xt.shape[0], S_MAX))
    for s in range(S_MAX):
        h[:, s] = hz.predict_proba(scpp.transform(np.column_stack([Xt, np.full(Xt.shape[0], float(s))])))[:, 1]
    return {90: 1 - np.prod(1 - h[:, :3], axis=1), 150: 1 - np.prod(1 - h[:, :5], axis=1)}

# mature test masks per rule t+H <= 2024-12-31
REL_END = np.datetime64("2024-12-31")
te_base = (fold == "test") & (year_all <= 2024)
mature = {}
for H in (90, 150):
    ok = (t_all + np.timedelta64(H, "D")) <= REL_END
    mature[H] = te_base & ok

for H in (90, 150):
    m = mature[H]
    dates = np.unique(t_all[m])
    print(f"H={H}: mature test windows {int(m.sum())}  dates {len(dates)}  "
          f"last date {pd.Timestamp(dates.max()).date()}")

def build_resamples(pids, seed=0, nboot=1000):
    uniq = np.unique(pids)
    pos = {p: np.where(pids == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([pos[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            for _ in range(nboot)]

print("\nC1 — mature-label representative numbers (hazard, fit 2017-21)")
res_cache = {}
for H in (90, 150):
    m = mature[H]
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_m = y[m]
    p = hazard_p(m)[H]
    res = build_resamples(pid_all[m])
    res_cache[H] = (m, y_m, p, res)
    rocs = []
    prs = []
    for idx in res:
        yb = y_m[idx]
        if yb.sum() in (0, len(yb)):
            continue
        rocs.append(roc_auc_score(yb, p[idx])); prs.append(average_precision_score(yb, p[idx]))
    ev = cohort.loc[m & (y == 1)].groupby(["pitcher", "next_surgery_date"]).ngroups
    print(f"  H={H}: ROC {roc_auc_score(y_m, p):.4f} [{np.percentile(rocs,2.5):.4f},{np.percentile(rocs,97.5):.4f}]  "
          f"PR {average_precision_score(y_m, p):.4f}  windows {int(m.sum())}  win-pos {int(y_m.sum())}  events {ev}")

print("\nC2 — calibration of hazard probabilities (mature test)")
for H in (90, 150):
    m, y_m, p, _ = res_cache[H]
    prev = y_m.mean(); mp = p.mean()
    q = np.quantile(p, 0.9)
    top = p >= q
    eps = 1e-12
    lg = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    rc = LogisticRegression(max_iter=2000).fit(lg.reshape(-1, 1), y_m)
    print(f"  H={H}: prevalence {prev:.4%}  mean-pred {mp:.4%}  "
          f"top-decile actual {y_m[top].mean():.4%} / pred {p[top].mean():.4%}  "
          f"recal slope {rc.coef_[0][0]:.3f}")

print("\nC3 — within-role ROC + caught-event roles @50 (mature test)")
for H in (90, 150):
    m, y_m, p, _ = res_cache[H]
    ss = F.loc[m, "start_share"].values
    sp_m = ss >= 0.5
    roc_sp = roc_auc_score(y_m[sp_m], p[sp_m])
    roc_rp = roc_auc_score(y_m[~sp_m], p[~sp_m])
    # events by role (role at event's best window)
    sub = cohort.loc[m].reset_index(drop=True)
    subF_ss = ss
    t_m = t_all[m]
    dates = np.unique(t_m)
    flag = np.zeros(int(m.sum()), dtype=bool)
    for d in dates:
        sel = np.where(t_m == d)[0]
        kk = min(50, len(sel))
        flag[sel[np.argsort(-p[sel], kind="stable")[:kk]]] = True
    pos_rows = np.where(y_m == 1)[0]
    groups = {}
    for r in pos_rows:
        key = (int(sub.loc[r, "pitcher"]), pd.Timestamp(sub.loc[r, "next_surgery_date"]))
        groups.setdefault(key, []).append(r)
    sp_ev = rp_ev = sp_caught = rp_caught = 0
    for key, rws in groups.items():
        is_sp = max(subF_ss[r] for r in rws) >= 0.5
        caught = any(flag[r] for r in rws)
        if is_sp:
            sp_ev += 1; sp_caught += caught
        else:
            rp_ev += 1; rp_caught += caught
    print(f"  H={H}: SP-within ROC {roc_sp:.4f}  RP-within ROC {roc_rp:.4f}  "
          f"caught SP {sp_caught}/{sp_ev}  RP {rp_caught}/{rp_ev}")

print("\nC4 — seen vs novel pitcher ROC (mature test)")
fit_pids = set(pid_all[fit_mask].tolist())
for H in (90, 150):
    m, y_m, p, _ = res_cache[H]
    seen = np.array([int(x) in fit_pids for x in pid_all[m]])
    print(f"  H={H}: seen ROC {roc_auc_score(y_m[seen], p[seen]):.4f} "
          f"(pos {int(y_m[seen].sum())})  novel ROC {roc_auc_score(y_m[~seen], p[~seen]):.4f} "
          f"(pos {int(y_m[~seen].sum())})  novel share {(~seen).mean():.2%}")

print("\nC5 — Monte Carlo random top-50 event-recall lift (2000 reps)")
rng = np.random.default_rng(1)
for H in (90, 150):
    m, y_m, p, _ = res_cache[H]
    sub = cohort.loc[m].reset_index(drop=True)
    t_m = t_all[m]
    dates = np.unique(t_m)
    pos_rows = np.where(y_m == 1)[0]
    groups = {}
    for r in pos_rows:
        key = (int(sub.loc[r, "pitcher"]), pd.Timestamp(sub.loc[r, "next_surgery_date"]))
        groups.setdefault(key, []).append(r)
    def recall_of(score):
        flag = np.zeros(int(m.sum()), dtype=bool)
        for d in dates:
            sel = np.where(t_m == d)[0]
            kk = min(50, len(sel))
            flag[sel[np.argsort(-score[sel], kind="stable")[:kk]]] = True
        return sum(1 for rws in groups.values() if any(flag[r] for r in rws))
    model_rec = recall_of(p)
    rand = np.array([recall_of(rng.random(int(m.sum()))) for _ in range(2000)])
    print(f"  H={H}: model {model_rec}/{len(groups)}  random mean {rand.mean():.1f} "
          f"[P2.5 {np.percentile(rand,2.5):.0f}, P97.5 {np.percentile(rand,97.5):.0f}]  "
          f"lift {model_rec/rand.mean():.2f}x  P(model<=random) {float((rand>=model_rec).mean()):.4f}")

print(f"\n[t={time.time()-t0:.0f}s] done")
