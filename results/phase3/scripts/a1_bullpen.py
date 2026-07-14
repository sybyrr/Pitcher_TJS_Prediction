"""A1 bullpen mini-block, corrected M1 audit artifact.

1. Role at t (modeling, 3-way) from corrected gs_flags_v2 trailing 365d GS share:
   SP >= 0.5, RP <= 0.2, else swing. 5-way split is DESCRIPTIVE ONLY.
2. RP-timescale features, exactly 6 (registered): pitches_7d, pitches_14d,
   appearances_14d, b2b_count_30d, three_in_four_count_30d, last_outing_spike
   (last pc - median pc per appearance over 90d; 0 if <2 games).
3. Cohen 2022 single pre-registered test: M_sa + {rp_flag, relx_slope,
   relx_slope x rp, relx_missing}; slope = 1-3yr trend of handedness-
   normalized game release_pos_x (>=10 games spanning >=180d, else missing).
4. Models vs M_sa binary baseline (paired, shared resamples, both primary
   t+H<=2024-12-31 and safety t+H<=2024-06-30): pooled(+6) /
   role-interaction(+rp+6+6xrp) / RP-vs-non-RP separate (ridge C=0.1).
5. Alert quota stable-region rule: RP-reserved q in {0,5,10,15,20} of 50,
   pre-test rolling folds Y in {2019, 2021} (fit years < Y; 2020 skipped,
   short season); pick max q with total-event loss <=1 vs q=0 in EVERY fold;
   fold disagreement -> smaller q. Apply once to mature test with canonical
   hazard scores; adopt iff RP capture up AND total recall loss <= 2 events.
   The mandatory safety H150 gate decides the final policy: q=20 remains a
   challenger and canonical alerts remain pure top-50 (q=0). q=5 is a
   post-hoc exploratory row only.
Gates: M_sa binary H90 mature ROC == 0.69203 (full 22-24 anchor; H90 mask
equals full test). Output ../a1_bullpen_corrected.csv; the original CSV is
retained unchanged as historical evidence.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent.parent
RESULT_OUT = OUT / "a1_bullpen_corrected.csv"
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
PTS = ["FF", "SI", "SL", "CH", "CU", "FC"]
relx_num = np.zeros(len(gf)); relx_den = np.zeros(len(gf))
for pt in PTS:
    n = gf[f"n_{pt}"].values.astype(float)
    rx = gf[f"{pt}_relx_mean"].values.astype(float)
    ok = ~np.isnan(rx)
    relx_num[ok] += n[ok] * rx[ok]
    relx_den[ok] += n[ok]
gf["relx_game"] = np.where(relx_den > 0, relx_num / np.maximum(relx_den, 1e-9), np.nan)
role_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    role_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                             g["total_pitches"].astype("float64").values,
                             g["relx_game"].astype("float64").values)
gs = pd.read_parquet(ROOT / "data/prospective/gs_flags_v2.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
gs_by_pid = {}
for pid, g in gs.groupby("pitcher", sort=False):
    gs_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                           g["n_g"].values.astype(float), g["n_gs"].values.astype(float))
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
rp6 = np.zeros((N, 6))  # pitches_7d, pitches_14d, appearances_14d, b2b_30d, three_in_four_30d, last_outing_spike
gs_share = np.zeros(N); mean_pc_app = np.zeros(N); mean_pc_gs = np.full(N, np.nan)
relx_slope = np.zeros(N); relx_missing = np.zeros(N)
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
    rgd, rtp, rrx = role_by_pid[pid]
    rm = (rgd < t) & (rgd >= t - np.timedelta64(365, "D"))
    ng365 = int(rm.sum())
    start_share[i] = float((rtp[rm] >= 50).sum()) / ng365 if ng365 > 0 else 0.0
    # RP-timescale 6
    d7 = before & (gd >= t - np.timedelta64(7, "D"))
    d14 = before & (gd >= t - np.timedelta64(14, "D"))
    rp6[i, 0] = pc[d7].sum()
    rp6[i, 1] = pc[d14].sum()
    rp6[i, 2] = d14.sum()
    gdd = gd[d30]
    if gdd.size >= 2:
        diffs = (gdd[1:] - gdd[:-1]) / DAY
        rp6[i, 3] = float((diffs == 1).sum())
        rp6[i, 4] = float(sum(1 for k in range(gdd.size)
                              if ((gdd >= gdd[k] - np.timedelta64(3, "D")) & (gdd <= gdd[k])).sum() >= 3))
    pc90_games = pc[d90]
    if pc90_games.size >= 2:
        rp6[i, 5] = float(pc[before][-1] - np.median(pc90_games))
    # GS-based role
    sgd, sng, sngs = gs_by_pid[pid]
    sm = (sgd < t) & (sgd >= t - np.timedelta64(365, "D"))
    n_app = sng[sm].sum(); n_start = sngs[sm].sum()
    gs_share[i] = n_start / n_app if n_app > 0 else 0.0
    mean_pc_app[i] = pc[rm_slim].mean() if (rm_slim := (before & (gd >= t - np.timedelta64(365, "D")))).any() else 0.0
    gs_dates = set(sgd[sm & (sngs > 0)].tolist())
    if gs_dates:
        gmask = np.array([d in gs_dates for d in gd], dtype=bool) & rm_slim
        if gmask.any():
            mean_pc_gs[i] = pc[gmask].mean()
    # Cohen release-side drift (1-3yr)
    cm = (rgd < t) & (rgd >= t - np.timedelta64(1095, "D")) & ~np.isnan(rrx)
    if cm.sum() >= 10 and (rgd[cm].max() - rgd[cm].min()) / DAY >= 180:
        x = (rgd[cm] - rgd[cm].min()) / DAY / 365.25
        relx_slope[i] = float(np.polyfit(x.astype(float), rrx[cm], 1)[0])
    else:
        relx_missing[i] = 1.0

F = pd.DataFrame(rows, columns=["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month"])
F["start_share"] = start_share
F["prior_pc_rate"] = prior_pc_rate
F["ncg_log"] = np.log1p(ncg_all)
F["vt_missing"] = vt_missing
RP6_COLS = ["pitches_7d", "pitches_14d", "appearances_14d", "b2b_30d", "three_in_four_30d", "last_outing_spike"]
for j, c in enumerate(RP6_COLS):
    F[c] = rp6[:, j]
role3 = np.where(gs_share >= 0.5, "SP", np.where(gs_share <= 0.2, "RP", "swing"))
F["rp_flag"] = (role3 == "RP").astype(float)
for c in RP6_COLS:
    F[f"{c}_x_rp"] = F[c] * F["rp_flag"]
F["relx_slope"] = relx_slope
F["relx_missing"] = relx_missing
F["relx_x_rp"] = relx_slope * F["rp_flag"]
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6
M_SA = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month",
        "start_share", "prior_pc_rate", "ncg_log", "vt_missing"]
print(f"[t={time.time()-t0:.0f}s] features built {F.shape}")

# descriptive 5-way (report only): starter / opener-bulk / swing / long RP / short RP
five = np.where(gs_share <= 0.2,
                np.where(mean_pc_app >= 25, "longRP", "shortRP"),
                np.where((gs_share > 0.2) & (mean_pc_gs < 30), "opener_bulk",
                         np.where(gs_share >= 0.5, "starter", "swing")))
fold = cohort["fold_main"].values
te_desc = (fold == "test") & (year_all <= 2024)
print("5-way role distribution (test 22-24, descriptive):")
print(pd.crosstab(five[te_desc], role3[te_desc]).to_string())

fit_mask = (fold == "train") | (fold == "valid")
te_base = (fold == "test") & (year_all <= 2024)
BOUNDARY_ENDS = {
    "primary": np.datetime64("2024-12-31"),
    "safety": np.datetime64("2024-06-30"),
}
mature = {
    (boundary, H): te_base & ((t_all + np.timedelta64(H, "D")) <= cutoff)
    for boundary, cutoff in BOUNDARY_ENDS.items()
    for H in (90, 150)
}

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

def fit_predict(cols, H, mask_fit, mask_pred, C=1.0):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    sc = StandardScaler().fit(F.loc[mask_fit, cols])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=C)
    clf.fit(sc.transform(F.loc[mask_fit, cols]), y[mask_fit])
    return clf.predict_proba(sc.transform(F.loc[mask_pred, cols]))[:, 1]

# Gate: the primary M_sa binary H90 mask reproduces the registered anchor.
p_gate = fit_predict(M_SA, 90, fit_mask, mature[("primary", 90)])
y_gate = cohort["label_H90_B0"].values.astype(int)[mature[("primary", 90)]]
roc_gate = roc_auc_score(y_gate, p_gate)
assert abs(roc_gate - 0.69203) < 1e-3, roc_gate
print(f"GATE M_sa binary H90 mature ROC {roc_gate:.5f} == anchor 0.69203 : OK")

VARIANTS = [
    ("A1_pooled", M_SA + RP6_COLS, 1.0),
    ("A1_interaction", M_SA + ["rp_flag"] + RP6_COLS + [f"{c}_x_rp" for c in RP6_COLS], 1.0),
    ("Cohen_relx", M_SA + ["rp_flag", "relx_slope", "relx_x_rp", "relx_missing"], 1.0),
]
out = []
for boundary, cutoff in BOUNDARY_ENDS.items():
    for H in (90, 150):
        m = mature[(boundary, H)]
        y_m = cohort[f"label_H{H}_B0"].values.astype(int)[m]
        res = build_resamples(pid_all[m])
        p0 = fit_predict(M_SA, H, fit_mask, m)
        rp_m = F.loc[m, "rp_flag"].values == 1
        base_roc = roc_auc_score(y_m, p0)
        base_pr = average_precision_score(y_m, p0)
        base_rp_roc = roc_auc_score(y_m[rp_m], p0[rp_m])
        out.append(
            dict(
                artifact_version="M1_corrected_20260714",
                block="model",
                boundary=boundary,
                cutoff=str(pd.Timestamp(cutoff).date()),
                H=H,
                variant="M_sa_binary_baseline",
                exploratory=False,
                canonical=False,
                roc=round(base_roc, 5),
                pr_auc=round(base_pr, 5),
                droc=0.0,
                droc_lo=0.0,
                droc_hi=0.0,
                dpr=0.0,
                dpr_lo=0.0,
                dpr_hi=0.0,
                rp_within=round(base_rp_roc, 5),
                n_rows=int(m.sum()),
                n_positive=int(y_m.sum()),
                anchor_gate_pass=(boundary == "primary" and H == 90),
            )
        )
        print(
            f"\n{boundary} H={H}: baseline M_sa ROC {base_roc:.4f}  "
            f"RP-within {base_rp_roc:.4f} (RP win-pos {int(y_m[rp_m].sum())})"
        )
        for name, cols, C in VARIANTS:
            p1 = fit_predict(cols, H, fit_mask, m, C=C)
            ra, rb, pa, pb = paired(y_m, p0, p1, res)
            dr = rb - ra
            dp = pb - pa
            drlo, drhi = ci(dr)
            dplo, dphi = ci(dp)
            fl = ("  dROC-EXCL0" if excl0(drlo, drhi) else "") + (
                "  dPR-EXCL0" if excl0(dplo, dphi) else ""
            )
            rocv = roc_auc_score(y_m, p1)
            prv = average_precision_score(y_m, p1)
            rp_roc = roc_auc_score(y_m[rp_m], p1[rp_m])
            out.append(
                dict(
                    artifact_version="M1_corrected_20260714",
                    block="model",
                    boundary=boundary,
                    cutoff=str(pd.Timestamp(cutoff).date()),
                    H=H,
                    variant=name,
                    exploratory=False,
                    canonical=False,
                    roc=round(rocv, 5),
                    pr_auc=round(prv, 5),
                    droc=round(float(np.median(dr)), 5),
                    droc_lo=round(drlo, 5),
                    droc_hi=round(drhi, 5),
                    dpr=round(float(np.median(dp)), 5),
                    dpr_lo=round(dplo, 5),
                    dpr_hi=round(dphi, 5),
                    rp_within=round(rp_roc, 5),
                    n_rows=int(m.sum()),
                    n_positive=int(y_m.sum()),
                    anchor_gate_pass=False,
                )
            )
            print(
                f"  {name:15s} ROC {rocv:.4f}  "
                f"dROC {np.median(dr):+.5f} [{drlo:+.5f},{drhi:+.5f}]  "
                f"dPR {np.median(dp):+.5f} [{dplo:+.5f},{dphi:+.5f}]  "
                f"RP-within {rp_roc:.4f}{fl}"
            )

        # Separate RP and non-RP ridge models; the paired PR delta is retained
        # exactly like every other variant (the historical CSV left it blank).
        p_sep = np.empty(int(m.sum()))
        rp_all = F["rp_flag"].values == 1
        for gmask_all in (rp_all, ~rp_all):
            fit_g = fit_mask & gmask_all
            pred_rows = np.where(m)[0]
            sel = gmask_all[pred_rows]
            y = cohort[f"label_H{H}_B0"].values.astype(int)
            sc = StandardScaler().fit(F.loc[fit_g, M_SA + RP6_COLS])
            clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1)
            clf.fit(sc.transform(F.loc[fit_g, M_SA + RP6_COLS]), y[fit_g])
            p_sep[sel] = clf.predict_proba(
                sc.transform(F.loc[m, M_SA + RP6_COLS].iloc[sel])
            )[:, 1]
        ra, rb, pa, pb = paired(y_m, p0, p_sep, res)
        dr = rb - ra
        dp = pb - pa
        drlo, drhi = ci(dr)
        dplo, dphi = ci(dp)
        sep_roc = roc_auc_score(y_m, p_sep)
        sep_pr = average_precision_score(y_m, p_sep)
        rp_roc = roc_auc_score(y_m[rp_m], p_sep[rp_m])
        out.append(
            dict(
                artifact_version="M1_corrected_20260714",
                block="model",
                boundary=boundary,
                cutoff=str(pd.Timestamp(cutoff).date()),
                H=H,
                variant="A1_separate",
                exploratory=False,
                canonical=False,
                roc=round(sep_roc, 5),
                pr_auc=round(sep_pr, 5),
                droc=round(float(np.median(dr)), 5),
                droc_lo=round(drlo, 5),
                droc_hi=round(drhi, 5),
                dpr=round(float(np.median(dp)), 5),
                dpr_lo=round(dplo, 5),
                dpr_hi=round(dphi, 5),
                rp_within=round(rp_roc, 5),
                n_rows=int(m.sum()),
                n_positive=int(y_m.sum()),
                anchor_gate_pass=False,
            )
        )
        print(
            f"  {'A1_separate':15s} ROC {sep_roc:.4f}  "
            f"dROC {np.median(dr):+.5f} [{drlo:+.5f},{drhi:+.5f}]  "
            f"dPR {np.median(dp):+.5f} [{dplo:+.5f},{dphi:+.5f}]  "
            f"RP-within {rp_roc:.4f}"
            + ("  dROC-EXCL0" if excl0(drlo, drhi) else "")
            + ("  dPR-EXCL0" if excl0(dplo, dphi) else "")
        )

# ---------------- quota stable-region (canonical hazard scores) ----------------
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

Xall = F[M_SA].values
def fit_hazard(mask_fit):
    rX, rs, ry = [], [], []
    for i in np.where(mask_fit)[0]:
        for s in range(S_MAX):
            if not at_risk[i, s]:
                break
            rX.append(Xall[i]); rs.append(s); ry.append(int(int_label[i, s]))
    Xpp = np.column_stack([np.asarray(rX), np.asarray(rs, dtype=float)])
    sc = StandardScaler().fit(Xpp)
    hz = LogisticRegression(max_iter=2000)
    hz.fit(sc.transform(Xpp), np.asarray(ry))
    return sc, hz

def hazard_p(model, mask):
    sc, hz = model
    Xt = Xall[mask]
    h = np.empty((Xt.shape[0], S_MAX))
    for s in range(S_MAX):
        h[:, s] = hz.predict_proba(sc.transform(np.column_stack([Xt, np.full(Xt.shape[0], float(s))])))[:, 1]
    return {90: 1 - np.prod(1 - h[:, :3], axis=1), 150: 1 - np.prod(1 - h[:, :5], axis=1)}

def quota_recall(mask, p, q, budget=50):
    """Event recall under RP-reserved quota q: per date, top q RP + top (budget-q) rest."""
    t_m = t_all[mask]; is_rp = F.loc[mask, "rp_flag"].values == 1
    flag = np.zeros(int(mask.sum()), dtype=bool)
    for d in np.unique(t_m):
        sel = np.where(t_m == d)[0]
        order = sel[np.argsort(-p[sel], kind="stable")]
        rp_top = [r for r in order if is_rp[r]][:q]
        reserved = set(rp_top)
        rest = [r for r in order if r not in reserved][:budget - len(rp_top)]
        flag[rp_top] = True; flag[rest] = True
    return flag

def event_table(mask, H):
    y = cohort[f"label_H{H}_B0"].values.astype(int)[mask]
    sub = cohort.loc[mask].reset_index(drop=True)
    groups = {}
    for r in np.where(y == 1)[0]:
        key = (int(sub.loc[r, "pitcher"]), pd.Timestamp(sub.loc[r, "next_surgery_date"]))
        groups.setdefault(key, []).append(r)
    r3 = role3[mask]
    ev_role = {}
    for key, rws in groups.items():
        ev_role[key] = "SP" if any(r3[r] == "SP" for r in rws) else ("RP" if all(r3[r] == "RP" for r in rws) else "mid")
    return groups, ev_role

print("\nQuota stable-region (pre-test rolling folds, hazard scores, H150):")
QGRID = [0, 5, 10, 15, 20]
fold_ok = {q: True for q in QGRID}
for Y in (2019, 2021):
    hz_y = fit_hazard(year_all < Y)
    my = year_all == Y
    p = hazard_p(hz_y, my)[150]
    groups, ev_role = event_table(my, 150)
    fold_results = {}
    line = f"  Y={Y} events {len(groups)}:"
    for q in QGRID:
        fl = quota_recall(my, p, q)
        caught = {k: any(fl[r] for r in rws) for k, rws in groups.items()}
        tot = sum(caught.values())
        rp_c = sum(1 for k, c in caught.items() if c and ev_role[k] == "RP")
        rp_t = sum(1 for k in groups if ev_role[k] == "RP")
        fold_results[q] = (tot, rp_c, rp_t)
        line += f"  q={q}: {tot}({rp_c}/{rp_t}RP)"
    base_caught = fold_results[0][0]
    for q in QGRID:
        tot, rp_c, rp_t = fold_results[q]
        fold_gate_pass = base_caught - tot <= 1
        fold_ok[q] = fold_ok[q] and fold_gate_pass
        out.append(
            dict(
                artifact_version="M1_corrected_20260714",
                block="quota_selection",
                boundary=f"rolling_{Y}",
                cutoff="",
                H=150,
                variant=f"q{q}",
                q=q,
                exploratory=False,
                canonical=False,
                total_caught=tot,
                rp_caught=rp_c,
                rp_events=rp_t,
                n_events=len(groups),
                gate_total_loss=base_caught - tot,
                gate_pass=fold_gate_pass,
                adopt=False,
                policy_status="pretest_selection",
            )
        )
    print(line)
q_star = max([q for q in QGRID if fold_ok[q]], default=0)
print(f"  stable-region choice q* = {q_star}")

print("\nQuota test application (canonical hazard, primary + safety):")
canon = fit_hazard(fit_mask)
quota_results = {}
for boundary, cutoff in BOUNDARY_ENDS.items():
    for H in (90, 150):
        m = mature[(boundary, H)]
        p = hazard_p(canon, m)[H]
        groups, ev_role = event_table(m, H)
        for q in sorted({0, 5, q_star}):
            fl = quota_recall(m, p, q)
            caught = {k: any(fl[r] for r in rws) for k, rws in groups.items()}
            tot = sum(caught.values())
            rp_c = sum(1 for k, c in caught.items() if c and ev_role[k] == "RP")
            rp_t = sum(1 for k in groups if ev_role[k] == "RP")
            quota_results[(boundary, H, q)] = (tot, rp_c, rp_t, len(groups))

        t0_, r0_, rt, n_events = quota_results[(boundary, H, 0)]
        tq_, rq_, _, _ = quota_results[(boundary, H, q_star)]
        local_gate = (rq_ > r0_) and (t0_ - tq_ <= 2)
        print(
            f"  {boundary} H={H}: q=0 {t0_}/{n_events} RP {r0_}/{rt}  ->  "
            f"q={q_star} {tq_} RP {rq_}; local_gate={'PASS' if local_gate else 'FAIL'}"
        )

# H150 is the pre-registered adoption endpoint. Both the primary and mandatory
# safety boundary must pass; safety H150 fails, so q=20 cannot be canonical.
candidate_adopt = all(
    (
        quota_results[(boundary, 150, q_star)][1]
        > quota_results[(boundary, 150, 0)][1]
    )
    and (
        quota_results[(boundary, 150, 0)][0]
        - quota_results[(boundary, 150, q_star)][0]
        <= 2
    )
    for boundary in BOUNDARY_ENDS
)
canonical_q = q_star if candidate_adopt else 0
assert canonical_q == 0, "M1 correction requires q=0 after safety H150 failure"

for boundary, cutoff in BOUNDARY_ENDS.items():
    for H in (90, 150):
        base_tot, base_rp, _, _ = quota_results[(boundary, H, 0)]
        for q in sorted({0, 5, q_star}):
            tot, rp_c, rp_t, n_events = quota_results[(boundary, H, q)]
            is_baseline = q == 0
            is_posthoc = q == 5 and q != q_star
            gate_rp_gain = np.nan if is_baseline else rp_c - base_rp
            gate_total_loss = np.nan if is_baseline else base_tot - tot
            gate_pass = (
                np.nan
                if is_baseline
                else bool((rp_c > base_rp) and (base_tot - tot <= 2))
            )
            if is_baseline:
                status = "canonical_q0"
            elif is_posthoc:
                status = "posthoc_exploratory"
            else:
                status = "challenger_safety_H150_failed"
            out.append(
                dict(
                    artifact_version="M1_corrected_20260714",
                    block="quota_test",
                    boundary=boundary,
                    cutoff=str(pd.Timestamp(cutoff).date()),
                    H=H,
                    variant=f"q{q}",
                    q=q,
                    exploratory=is_posthoc,
                    canonical=is_baseline,
                    total_caught=tot,
                    rp_caught=rp_c,
                    rp_events=rp_t,
                    n_events=n_events,
                    gate_rp_gain=gate_rp_gain,
                    gate_total_loss=gate_total_loss,
                    gate_pass=gate_pass,
                    adopt=bool(q == q_star and candidate_adopt),
                    policy_status=status,
                )
            )

out.append(
    dict(
        artifact_version="M1_corrected_20260714",
        block="policy_decision",
        boundary="primary+safety",
        cutoff="2024-12-31|2024-06-30",
        H=150,
        variant=f"q{q_star}_vs_q0",
        q=q_star,
        exploratory=False,
        canonical=False,
        gate_pass=candidate_adopt,
        adopt=False,
        canonical_q=canonical_q,
        policy_status="q20_challenger; canonical_q0; safety_H150_failed",
    )
)

result = pd.DataFrame(out)
result.to_csv(RESULT_OUT, index=False)
print(
    f"policy decision: q*={q_star}, adopt={candidate_adopt}, canonical=q{canonical_q}"
)
print(f"\n[t={time.time()-t0:.0f}s] wrote {RESULT_OUT}")
