"""B' (ii)+(iii): how much of M-role's catch is information the club already has?

(ii) USAGE-NORMAL SUBGROUP: restrict test windows to those with no visible
usage decline, re-rank within the subgroup, measure ROC (subset clustered
bootstrap) + event recall. This is the model's value net of "club already
sees the decline".
(iii) LEAD DECOMPOSITION: for events caught in the FULL top-k ranking, at the
earliest catching window report dsl / pc_acute_dev / usage-normal status and
lead time to surgery.

Pre-registered decline definitions (window-level, decision date t):
  PRIMARY  usage_normal  = dsl <= 15  AND  pc_acute_dev >= 0
  SENSITIV usage_normal2 = dsl <= 30  AND  pc_acute_dev/max(pc_chronic,.1) >= -0.25
Both are descriptive decompositions — no pass/fail threshold.

Model/protocol frozen (role_models.py): M-role 6-feature LR, fit train+valid,
gate vs anchors (H90 ROC 0.643680/PR 0.035973, H150 0.643775/0.046995).
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

# ---------------------------------------------------------------- load + F1 features (verbatim protocol)
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
role_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    role_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                             g["total_pitches"].astype("float64").values)

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

feat_rows = []
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
    feat_rows.append((pc[d90].sum() / 90.0, pc[d30].sum() / 30.0 - pc[d90].sum() / 90.0,
                      days_since_last, vel_trend, float(cohort["month"].values[i])))
    rgd, rtp = role_by_pid[pid]
    rmask = (rgd < t) & (rgd >= t - np.timedelta64(365, "D"))
    ng365 = int(rmask.sum())
    start_share[i] = float((rtp[rmask] >= 50).sum()) / ng365 if ng365 > 0 else 0.0

F = pd.DataFrame(feat_rows, columns=["pc_chronic", "pc_acute_dev", "days_since_last",
                                     "vel_trend", "month"])
F["start_share"] = start_share
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6
print(f"[t={time.time()-t0:.1f}s] features built")

fold = cohort["fold_main"].values
tr = fold == "train"; va = fold == "valid"; te = fold == "test"
fit_mask = tr | va
COLS = ["pc_chronic", "pc_acute_dev", "days_since_last", "vel_trend", "month", "start_share"]

# ---------------------------------------------------------------- decline flags (test windows)
dsl_te = F.loc[te, "days_since_last"].values
dev_te = F.loc[te, "pc_acute_dev"].values
chron_te = F.loc[te, "pc_chronic"].values
rel_dev = dev_te / np.maximum(chron_te, 0.1)
SUBGROUPS = {
    "usage_normal": (dsl_te <= 15) & (dev_te >= 0),
    "usage_normal2": (dsl_te <= 30) & (rel_dev >= -0.25),
}

t_test = pd.to_datetime(cohort.loc[te, "t"]).values
test_dates = np.sort(np.unique(t_test))
pid_test = pid_all[te]
next_surg_te = pd.to_datetime(cohort.loc[te, "next_surgery_date"]).values

def build_resamples(sub_rows, seed=0, nboot=1000):
    pids = pid_test[sub_rows]
    uniq = np.unique(pids)
    pos = {p: sub_rows[np.where(pids == p)[0]] for p in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([pos[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            for _ in range(nboot)]

def boot_ci(y_te, prob, resamples):
    rocs, prs = [], []
    for idx in resamples:
        yb = y_te[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        rocs.append(roc_auc_score(yb, prob[idx]))
        prs.append(average_precision_score(yb, prob[idx]))
    return (np.percentile(rocs, [2.5, 97.5]), np.percentile(prs, [2.5, 97.5]), len(rocs))

def event_groups(y_te, rows_filter=None):
    """{(pid, surg_date): [test-row indices of that event's positive windows]}"""
    pos_rows = np.where(y_te == 1)[0]
    if rows_filter is not None:
        pos_rows = pos_rows[np.isin(pos_rows, rows_filter)]
    groups = {}
    for r in pos_rows:
        key = (int(pid_test[r]), pd.Timestamp(next_surg_te[r]))
        groups.setdefault(key, []).append(r)
    return groups

def topk_flags(score, rows, k):
    """top-k per decision date within the given row subset."""
    flag = np.zeros(len(score), dtype=bool)
    for d in test_dates:
        sel = rows[t_test[rows] == d]
        if len(sel) == 0:
            continue
        kk = min(k, len(sel))
        flag[sel[np.argsort(-score[sel], kind="stable")[:kk]]] = True
    return flag

ANCHOR = {90: (0.643680, 0.035973), 150: (0.643775, 0.046995)}
KS = [10, 20, 50]
all_rows = np.arange(int(te.sum()))
RES_FULL = build_resamples(all_rows)
lead_records, sub_records = [], []

for H in (90, 150):
    y = cohort[f"label_H{H}_B0"].values.astype(int)
    y_te = y[te]

    sc = StandardScaler().fit(F.loc[fit_mask, COLS])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(F.loc[fit_mask, COLS]), y[fit_mask])
    prob = clf.predict_proba(sc.transform(F.loc[te, COLS]))[:, 1]
    roc = roc_auc_score(y_te, prob); pr = average_precision_score(y_te, prob)
    a_roc, a_pr = ANCHOR[H]
    assert abs(roc - a_roc) < 1e-4 and abs(pr - a_pr) < 1e-4, "GATE FAILED"
    print(f"\n{'='*78}\nH={H}  M-role GATE OK  ROC {roc:.5f}  PR {pr:.5f}")

    groups_all = event_groups(y_te)
    n_events = len(groups_all)

    # ---- window composition: how discriminative is decline alone? ----
    print(f"  window composition (test, n={len(y_te)}, pos {int(y_te.sum())}, events {n_events}):")
    for name, mask in SUBGROUPS.items():
        n_sub = int(mask.sum()); pos_sub = int(y_te[mask].sum())
        br_in = y_te[mask].mean(); br_out = y_te[~mask].mean()
        print(f"    {name:14s}: windows {n_sub} ({mask.mean():.1%})  pos {pos_sub} "
              f"(base {br_in:.4f} vs decline-side {br_out:.4f}, ratio {br_out/max(br_in,1e-9):.2f}x)")

    # ---- (ii) subgroup evaluation ----
    for name, mask in SUBGROUPS.items():
        rows = np.where(mask)[0]
        res_sub = build_resamples(rows)
        (roc_ci, pr_ci, nb) = boot_ci(y_te, prob, res_sub)
        roc_s = roc_auc_score(y_te[rows], prob[rows]) if y_te[rows].sum() > 0 else np.nan
        groups_sub = event_groups(y_te, rows_filter=rows)
        ev_reach = len(groups_sub)  # events with >=1 subgroup positive window
        recs = {}
        for k in KS:
            flag = topk_flags(prob, rows, k)
            recs[k] = sum(1 for rws in groups_sub.values() if flag[rws].any())
        sub_records.append(dict(H=H, subgroup=name, n_windows=len(rows),
                                n_pos=int(y_te[rows].sum()), n_events_reachable=ev_reach,
                                n_events_total=n_events, roc=roc_s,
                                roc_lo=roc_ci[0], roc_hi=roc_ci[1], nboot=nb,
                                rec10=recs[10], rec20=recs[20], rec50=recs[50]))
        print(f"  (ii) {name:14s}: ROC {roc_s:.4f} [{roc_ci[0]:.4f},{roc_ci[1]:.4f}]  "
              f"events reachable {ev_reach}/{n_events}  recall@10/20/50 "
              f"{recs[10]}/{recs[20]}/{recs[50]} of {ev_reach}")

    # ---- (iii) lead decomposition of FULL-ranking catches ----
    for k in (20, 50):
        flag = topk_flags(prob, all_rows, k)
        caught = {key: [r for r in rws if flag[r]] for key, rws in groups_all.items()
                  if any(flag[r] for r in rws)}
        n_norm = 0
        for (pid, sdate), rws in caught.items():
            r0 = min(rws, key=lambda r: t_test[r])  # earliest catching window
            un = bool(SUBGROUPS["usage_normal"][r0])
            n_norm += un
            lead_records.append(dict(
                H=H, k=k, pitcher=pid, surgery=str(pd.Timestamp(sdate).date()),
                catch_t=str(pd.Timestamp(t_test[r0]).date()),
                lead_days=int((pd.Timestamp(sdate) - pd.Timestamp(t_test[r0])).days),
                dsl=float(dsl_te[r0]), pc_acute_dev=float(dev_te[r0]),
                rel_dev=float(rel_dev[r0]), usage_normal=un))
        leads = [x["lead_days"] for x in lead_records if x["H"] == H and x["k"] == k]
        print(f"  (iii) full-ranking @k={k}: caught {len(caught)}/{n_events} events; "
              f"usage-normal at first catch {n_norm}/{len(caught)}  "
              f"lead days median {np.median(leads):.0f} (min {min(leads)}, max {max(leads)})")

pd.DataFrame(sub_records).to_csv(SCR / "bprime_subgroup.csv", index=False)
pd.DataFrame(lead_records).to_csv(SCR / "bprime_lead.csv", index=False)
print(f"\n[t={time.time()-t0:.1f}s] wrote bprime_subgroup.csv + bprime_lead.csv")
