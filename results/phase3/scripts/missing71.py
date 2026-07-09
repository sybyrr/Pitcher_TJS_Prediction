from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\PAINS\Pitcher_TJS_Prediction")
SCRATCH = Path(r"C:\Users\PC\AppData\Local\Temp\claude\d--PAINS-Pitcher-TJS-Prediction\c8255b88-a70d-45ba-964b-531a2271c93f\scratchpad")

snap = pd.read_csv(ROOT / "TJS_Prediction" / "Raw_data" / "list of TJ.csv")
live = pd.read_csv(SCRATCH / "tj_live_clean.csv")

live_year = pd.to_datetime(live["surg_date"], errors="coerce").dt.year
snap_year = pd.to_datetime(snap["TJ Surgery Date"], errors="coerce").dt.year

live2223 = live[live_year.isin([2022, 2023])].copy()
live2223["yr"] = live_year[live_year.isin([2022, 2023])]
snap2223 = snap[snap_year.isin([2022, 2023])].copy()
snap2223["yr"] = snap_year[snap_year.isin([2022, 2023])]

print("live 2022-23 surgeries:", len(live2223), " (2022:", (live2223.yr==2022).sum(), " 2023:", (live2223.yr==2023).sum(), ")")
print("snap 2022-23 surgeries:", len(snap2223), " (2022:", (snap2223.yr==2022).sum(), " 2023:", (snap2223.yr==2023).sum(), ")")
print()

# Definition A: match on (mlbamid, year)
snap_keys_my = set(zip(snap2223["mlbamid"], snap2223["yr"]))
missA = live2223[~live2223.apply(lambda r: (r["mlbamid_i"], r["yr"]) in snap_keys_my, axis=1)]
print("A) live 2022-23 (mlbamid,year) NOT in snapshot 2022-23:", len(missA))

# Definition B: pitcher (mlbamid) has NO 2022-23 surgery in snapshot at all
snap_mlb_2223 = set(snap2223["mlbamid"])
missB = live2223[~live2223["mlbamid_i"].isin(snap_mlb_2223)]
print("B) live 2022-23 pitchers with no 2022-23 snapshot surgery:", len(missB), " unique pitchers:", missB["mlbamid_i"].nunique())

# Definition C: pitcher (mlbamid) absent from snapshot ENTIRELY
snap_mlb_all = set(snap["mlbamid"])
missC = live2223[~live2223["mlbamid_i"].isin(snap_mlb_all)]
print("C) live 2022-23 pitchers absent from snapshot entirely:", len(missC), " unique pitchers:", missC["mlbamid_i"].nunique())

# Definition D: exact (mlbamid, surg_date) match
snap_sd = pd.to_datetime(snap["TJ Surgery Date"], errors="coerce")
snap_keys_date = set(zip(snap["mlbamid"], snap_sd))
live_sd = pd.to_datetime(live2223["surg_date"], errors="coerce")
missD = live2223[~pd.Series([ (m,d) in snap_keys_date for m,d in zip(live2223["mlbamid_i"], live_sd)], index=live2223.index)]
print("D) live 2022-23 (mlbamid,exact-date) NOT in snapshot:", len(missD))

# save the definition that matches ~71 for later cross-check with cohort changes
for label, mm in [("A",missA),("B",missB),("C",missC),("D",missD)]:
    if abs(len(mm)-71) <= 5:
        print(f"\n==> Definition {label} (n={len(mm)}) is closest to 71; saving its mlbamid set")
        mm[["Player","mlbamid_i","surg_date","yr"]].to_csv(SCRATCH / "missing_2223.csv", index=False)
        break
