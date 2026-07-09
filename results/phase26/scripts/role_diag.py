"""Role-split diagnostics: does pooling SP/RP confound the workload features?
Role proxy: over trailing 365d games before t, start_share = frac(total_pitches>=50);
role SP if start_share>=0.5 else RP. Reports feature dists by role + surgery counts."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
SCR = Path(r"C:/Users/PC/AppData/Local/Temp/claude/d--PAINS-Pitcher-TJS-Prediction/c8255b88-a70d-45ba-964b-531a2271c93f/scratchpad")

cohort = pd.read_parquet(ROOT / "data/prospective/cohort_v2.parquet").sort_values(["t", "pitcher"]).reset_index(drop=True)
gf = pd.read_parquet(ROOT / "data/prospective/game_features_v2.parquet")[["pitcher", "game_date", "total_pitches"]]
gf = gf.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
slim = pd.read_parquet(SCR / "slim_games.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)

by_gf = {int(p): (g["game_date"].values.astype("datetime64[D]"), g["total_pitches"].astype(float).values)
         for p, g in gf.groupby("pitcher", sort=False)}
by_slim = {int(p): (g["game_date"].values.astype("datetime64[D]"), g["pitch_count"].astype(float).values)
           for p, g in slim.groupby("pitcher", sort=False)}
DAY = np.timedelta64(1, "D")

pid_all = cohort["pitcher"].values.astype(np.int64)
t_all = cohort["t"].values.astype("datetime64[ns]")
rows = []
for i in range(len(cohort)):
    pid = int(pid_all[i]); t = t_all[i].astype("datetime64[D]")
    gd, tp = by_gf[pid]
    win = (gd < t) & (gd >= t - np.timedelta64(365, "D"))
    tpw = tp[win]
    n = len(tpw)
    start_share = float((tpw >= 50).mean()) if n else np.nan
    # workload features from slim (same as baseline)
    sgd, spc = by_slim[pid]
    before = sgd < t
    d30 = before & (sgd >= t - np.timedelta64(30, "D"))
    d90 = before & (sgd >= t - np.timedelta64(90, "D"))
    pc30 = spc[d30].sum(); pc90 = spc[d90].sum()
    pc_chronic = pc90 / 90.0
    pc_acute_dev = pc30 / 30.0 - pc90 / 90.0
    ng30 = float(d30.sum()); ng90 = float(d90.sum())
    dsl = float((t - sgd[before].max()) / DAY)
    rows.append((start_share, pc_chronic, pc_acute_dev, ng30, ng90, dsl))

D = pd.DataFrame(rows, columns=["start_share", "pc_chronic", "pc_acute_dev", "ng_30", "ng_90", "dsl"])
D["fold"] = cohort["fold_main"].values
for H in (90, 150):
    D[f"y{H}"] = cohort[f"label_H{H}_B0"].values
D["pitcher"] = pid_all
D["next_surg"] = cohort["next_surgery_date"].values

fit = D["fold"].isin(["train", "valid"])
Df = D[fit].copy()

print("=== start_share distribution (fit set, n=%d) ===" % fit.sum())
print(Df["start_share"].describe(percentiles=[.1,.25,.5,.75,.9]).round(3).to_string())
sh = Df["start_share"].dropna()
print(f"swing band 0.4-0.6 share: {((sh>=0.4)&(sh<=0.6)).mean():.3f}")
print(f"near-0 (<0.1): {(sh<0.1).mean():.3f} | near-1 (>0.9): {(sh>0.9).mean():.3f}")

for thr in (0.4, 0.5, 0.6):
    role = np.where(Df["start_share"] >= thr, "SP", "RP")
    print(f"\n--- threshold {thr}: SP {np.mean(role=='SP'):.3f} / RP {np.mean(role=='RP'):.3f}")

# use 0.5 for the main comparison
Df["role"] = np.where(Df["start_share"] >= 0.5, "SP", "RP")
print("\n=== feature median [IQR] by role (fit set) ===")
for c in ["dsl", "pc_chronic", "pc_acute_dev", "ng_30", "ng_90"]:
    for r in ["SP", "RP"]:
        s = Df.loc[Df["role"] == r, c]
        print(f"  {c:14s} {r}: median {s.median():8.3f}  IQR [{s.quantile(.25):.3f}, {s.quantile(.75):.3f}]")

print("\n=== surgery windows / distinct surgeries by role, per fold ===")
D["role"] = np.where(D["start_share"] >= 0.5, "SP", "RP")
for H in (90, 150):
    print(f"-- H={H}")
    for f in ["train", "valid", "test"]:
        sub = D[(D["fold"] == f)]
        for r in ["SP", "RP"]:
            s = sub[sub["role"] == r]
            pos = s[f"y{H}"].sum()
            distinct = s.loc[s[f"y{H}"] == 1, "next_surg"].nunique()
            br = s[f"y{H}"].mean()
            print(f"   {f:5s} {r}: pos_windows {pos:4d}  distinct_surg {distinct:3d}  n {len(s):5d}  base {br:.4f}")
