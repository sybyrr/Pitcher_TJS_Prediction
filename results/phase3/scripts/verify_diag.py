"""Fable re-check of two load-bearing diagnosis numbers (read-only)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

SCR = Path(r"C:\Users\PC\AppData\Local\Temp\claude\d--PAINS-Pitcher-TJS-Prediction\c8255b88-a70d-45ba-964b-531a2271c93f\scratchpad")
cohort = pd.read_parquet("data/prospective/cohort_v2.parquet")
slim = pd.read_parquet(SCR / "slim_games.parquet").sort_values(["pitcher", "game_date"])
by = {int(p): (g["game_date"].values.astype("datetime64[D]"), g["pitch_count"].values.astype(float))
      for p, g in slim.groupby("pitcher")}
t = cohort["t"].values.astype("datetime64[D]")
pid = cohort["pitcher"].values

pc90 = np.empty(len(cohort))
for i in range(len(cohort)):
    gd, pc = by[int(pid[i])]
    m = (gd < t[i]) & (gd >= t[i] - np.timedelta64(90, "D"))
    pc90[i] = pc[m].sum()
z = pc90 == 0
print("pc90==0 overall: %.3f" % z.mean())
print("by month:", cohort.assign(z=z).groupby("month")["z"].mean().round(3).to_dict())

te = (cohort["fold_main"] == "test").values
d = cohort.loc[te].assign(z=z[te])
r = d[d["t"] == np.datetime64("2022-04-01")]
print("2022-04-01 test windows:", len(r), " pc90==0:", int(r["z"].sum()))
for H in (90, 150):
    pos = d[d[f"label_H{H}_B0"] == 1]
    print(f"test positives H{H} with pc90==0: {pos['z'].mean():.3f} ({int(pos['z'].sum())}/{len(pos)})")

# prior-TJS flag prevalence and test base-rate gradient
tj = pd.read_csv(SCR / "tj_live_clean.csv")
tj["surg_date"] = pd.to_datetime(tj["surg_date"])
sb = {int(p): np.sort(g["surg_date"].values) for p, g in tj.groupby("mlbamid_i")}
prior = np.zeros(len(cohort), bool)
for i in range(len(cohort)):
    s = sb.get(int(pid[i]))
    if s is not None and (s < t[i].astype("datetime64[ns]")).any():
        prior[i] = True
print("prior-TJS: window share %.3f  pitcher share %.3f" %
      (prior.mean(), pd.Series(prior).groupby(pid).max().mean()))
for H in (90, 150):
    y = (cohort[f"label_H{H}_B0"].values == 1)
    br1 = y[te & prior].mean(); br0 = y[te & ~prior].mean()
    print(f"H{H} test base rate: prior-TJS {br1:.4f} vs none {br0:.4f} "
          f"ratio {br1/br0:.2f}x  (pos split {int(y[te & prior].sum())}/{int(y[te].sum())})")
