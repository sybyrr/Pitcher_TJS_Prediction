from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\PAINS\Pitcher_TJS_Prediction")
SCRATCH = Path(r"C:\Users\PC\AppData\Local\Temp\claude\d--PAINS-Pitcher-TJS-Prediction\c8255b88-a70d-45ba-964b-531a2271c93f\scratchpad")

gained = [548384,605397,622098,640470,656814,660271,663753,664299,664353,666129,666201,672715,680686]
live = pd.read_csv(SCRATCH / "tj_live_clean.csv")
snap = pd.read_csv(ROOT / "TJS_Prediction" / "Raw_data" / "list of TJ.csv")

sub = live[live.mlbamid_i.isin(gained)][["mlbamid_i","Player","surg_date","Level"]].copy()
sub["surg_date"] = pd.to_datetime(sub["surg_date"])
sub = sub.sort_values("surg_date")
print("=== 13 gained-injured pitchers: their live surgery date(s) ===")
print(sub.to_string(index=False))
print()
yr = sub["surg_date"].dt.year
print("surgery-year distribution of the 13:", dict(yr.value_counts().sort_index()))
print("of the 13, dated 2022-23:", int(yr.isin([2022,2023]).sum()), " dated 2024+:", int((yr>=2024).sum()))
print()
# were any of these in the snapshot at all (any date)?
print("of the 13, present in SNAPSHOT list (any surgery):", int(sub.mlbamid_i.isin(snap.mlbamid).sum()))
print("of the 13, absent from snapshot entirely:", int((~sub.mlbamid_i.isin(snap.mlbamid)).sum()))
