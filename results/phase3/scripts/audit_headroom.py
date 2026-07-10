"""AUDIT (read-only, scratchpad): does prospectively-available but UNUSED info
recover score lost to the offseason-lookback zeroing? Frozen protocol verbatim
from role_models.py: StandardScaler + LR(balanced, max_iter=2000), fit train+valid,
test 2022-23, 1000 pitcher-clustered bootstrap (seed 0), paired CIs, event recall@k.

Base = M-role (6). Additions are all computed from games STRICTLY before t:
  prior_pc_rate  = most-recent prior-season total pitches / 183   (season-aware chronic;
                   nonzero even in April when the 90d calendar window is empty)
  ncg_log        = log1p(n_career_games)                          (career durability, in cohort, unused)
  vt_missing     = 1 if vel_trend fell back to 0.0 (unmeasured, not "no change")

Contrasts (paired, shared resamples):
  M_sa  = M-role + {prior_pc_rate, ncg_log, vt_missing}
  M_bf  = M-role but pc_chronic backfilled: pc_chronic_bf = pc_90/90 if pc_90>0 else prior_pc_rate
Gate: M-role must reproduce anchors (H90 ROC 0.643680/PR 0.035973; H150 0.643775/0.046995).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
SCR = Path(r"C:/Users/PC/AppData/Local/Temp/claude/d--PAINS-Pitcher-TJS-Prediction/c8255b88-a70d-45ba-964b-531a2271c93f/scratchpad")

cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v2.parquet").sort_values(["t","pitcher"]).reset_index(drop=True)
N = len(cohort)
slim = pd.read_parquet(SCR / "slim_games.parquet").sort_values(["pitcher","game_date"]).reset_index(drop=True)
by_pid = {}
for pid, g in slim.groupby("pitcher", sort=False):
    by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"),
                        g["pitch_count"].astype("float64").values,
                        g["mean_release_speed"].astype("float64").values,
                        g["game_year"].values.astype(np.int64))
gf = pd.read_parquet(ROOT / "data/prospective/game_features_v2.parquet").sort_values(["pitcher","game_date"]).reset_index(drop=True)
role_by_pid = {}
for pid, g in gf.groupby("pitcher", sort=False):
    role_by_pid[int(pid)] = (g["game_date"].values.astype("datetime64[D]"), g["total_pitches"].astype("float64").values)
DAY = np.timedelta64(1,"D")

def pw_mean(sp, pc, mask):
    m = mask & ~np.isnan(sp)
    if not m.any(): return np.nan
    w = pc[m].sum()
    return float((sp[m]*pc[m]).sum()/w) if w>0 else np.nan

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
    d30 = before & (gd >= t - np.timedelta64(30,"D"))
    d90 = before & (gd >= t - np.timedelta64(90,"D"))
    pc_30 = pc[d30].sum(); pc_90 = pc[d90].sum()
    dsl = float((t - gd[before].max())/DAY)
    vmean_30 = pw_mean(sp, pc, d30)
    yr = int(year_all[i])
    vmean_prior = pw_mean(sp, pc, before & (gy < yr))
    if np.isnan(vmean_prior):
        vmean_prior = pw_mean(sp, pc, before & (gd < t - np.timedelta64(30,"D")))
    if np.isnan(vmean_30) or np.isnan(vmean_prior):
        vel_trend = 0.0; vt_missing[i] = 1.0
    else:
        vel_trend = vmean_30 - vmean_prior
    # most-recent prior season pitch rate (pitches/day over 183)
    prior_years = gy[before & (gy < yr)]
    if prior_years.size:
        last_py = prior_years.max()
        m_py = before & (gy == last_py)
        prior_pc_rate[i] = pc[m_py].sum() / 183.0
    else:
        prior_pc_rate[i] = 0.0
    rows.append((pc_90/90.0, pc_30/30.0 - pc_90/90.0, dsl, vel_trend, month_all[i]))
    rgd, rtp = role_by_pid[pid]
    rm = (rgd < t) & (rgd >= t - np.timedelta64(365,"D"))
    ng365 = int(rm.sum())
    start_share[i] = float((rtp[rm] >= 50).sum())/ng365 if ng365>0 else 0.0

F = pd.DataFrame(rows, columns=["pc_chronic","pc_acute_dev","days_since_last","vel_trend","month"])
F["start_share"] = start_share
F["prior_pc_rate"] = prior_pc_rate
F["ncg_log"] = np.log1p(ncg_all)
F["vt_missing"] = vt_missing
F["pc_chronic_bf"] = np.where(F["pc_chronic"].values>0, F["pc_chronic"].values, prior_pc_rate)
assert np.abs(F["days_since_last"].values - cohort["dsl"].values.astype(float)).max() < 1e-6

fold = cohort["fold_main"].values
tr = fold=="train"; va = fold=="valid"; te = fold=="test"; fit_mask = tr|va

t_test = pd.to_datetime(cohort.loc[te,"t"]).values
test_dates = np.sort(np.unique(t_test))
pid_test = pid_all[te]
uniq = np.unique(pid_test)
rng = np.random.default_rng(0)
pos_in = {p: np.where(pid_test==p)[0] for p in uniq}
BOOT=1000
resamples=[np.concatenate([pos_in[p] for p in rng.choice(uniq,size=len(uniq),replace=True)]) for _ in range(BOOT)]
next_surg = pd.to_datetime(cohort["next_surgery_date"]).values

def paired(y_te, a, b):
    ra,rb=[],[]
    pa,pb=[],[]
    for idx in resamples:
        yb=y_te[idx]
        if yb.sum()==0 or yb.sum()==len(yb): continue
        ra.append(roc_auc_score(yb,a[idx])); rb.append(roc_auc_score(yb,b[idx]))
        pa.append(average_precision_score(yb,a[idx])); pb.append(average_precision_score(yb,b[idx]))
    return np.array(ra),np.array(rb),np.array(pa),np.array(pb),len(ra)
def ci(x): return float(np.percentile(x,2.5)), float(np.percentile(x,97.5))
def excl0(lo,hi): return (lo>0 and hi>0) or (lo<0 and hi<0)
def topk(score,k):
    flag=np.zeros(len(score),dtype=bool)
    for d in test_dates:
        sel=np.where(t_test==d)[0]; kk=min(k,len(sel))
        flag[sel[np.argsort(-score[sel],kind="stable")[:kk]]]=True
    return flag
def evrec(y_te,score,ks):
    pr=np.where(y_te==1)[0]
    keys=list(zip(pid_test[pr].tolist(), pd.to_datetime(next_surg[te][pr]).astype("datetime64[ns]").tolist()))
    groups={}
    for r,k in zip(pr,keys): groups.setdefault(k,[]).append(r)
    out={}
    for k in ks:
        fl=topk(score,k); out[k]=(sum(1 for rs in groups.values() if fl[rs].any()), len(groups))
    return out
def fit(cols,y):
    sc=StandardScaler().fit(F.loc[fit_mask,cols])
    clf=LogisticRegression(class_weight="balanced",max_iter=2000)
    clf.fit(sc.transform(F.loc[fit_mask,cols]), y[fit_mask])
    return clf.predict_proba(sc.transform(F.loc[te,cols]))[:,1]

MROLE=["pc_chronic","pc_acute_dev","days_since_last","vel_trend","month","start_share"]
M_SA = MROLE + ["prior_pc_rate","ncg_log","vt_missing"]
M_BF = ["pc_chronic_bf","pc_acute_dev","days_since_last","vel_trend","month","start_share","vt_missing"]
ANCHOR={90:(0.643680,0.035973),150:(0.643775,0.046995)}
KS=[10,20,50]
for H in (90,150):
    y=cohort[f"label_H{H}_B0"].values.astype(int); y_te=y[te]
    base=fit(MROLE,y); roc0=roc_auc_score(y_te,base); pr0=average_precision_score(y_te,base)
    a_roc,a_pr=ANCHOR[H]
    assert abs(roc0-a_roc)<1e-4 and abs(pr0-a_pr)<1e-4, f"GATE FAIL H{H}: {roc0} {pr0}"
    er0=evrec(y_te,base,KS)
    print("="*72); print(f"H={H}  M-role GATE OK  ROC {roc0:.5f}  PR {pr0:.5f}  evrec {er0[10][0]}/{er0[20][0]}/{er0[50][0]} of {er0[10][1]}")
    for name,cols in (("M_sa(+prior,ncg,vtmiss)",M_SA),("M_bf(backfill chronic)",M_BF)):
        p=fit(cols,y); roc=roc_auc_score(y_te,p); pr=average_precision_score(y_te,p); er=evrec(y_te,p,KS)
        ra,rb,pa,pb,nv=paired(y_te,base,p)
        dr=rb-ra; dp=pb-pa; drlo,drhi=ci(dr); dplo,dphi=ci(dp)
        fl=("  dROC-EXCL0" if excl0(drlo,drhi) else "")+("  dPR-EXCL0" if excl0(dplo,dphi) else "")
        print(f"  {name:26s} ROC {roc:.5f}  PR {pr:.5f}  evrec {er[10][0]}/{er[20][0]}/{er[50][0]}")
        print(f"  {'':26s} dROC {roc-roc0:+.5f} [{drlo:+.5f},{drhi:+.5f}]  dPR {pr-pr0:+.5f} [{dplo:+.5f},{dphi:+.5f}]{fl}")
