from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(r"d:\PAINS\Pitcher_TJS_Prediction")
SCRATCH = Path(r"C:\Users\PC\AppData\Local\Temp\claude\d--PAINS-Pitcher-TJS-Prediction\c8255b88-a70d-45ba-964b-531a2271c93f\scratchpad")

live = pd.read_csv(SCRATCH / "tj_live_clean.csv")

# check multi-surgery counts (mirror snapshot cumcount behavior)
counts = live.groupby("mlbamid_i").size()
print("max surgeries per pitcher:", counts.max())
print("pitchers with >3 surgeries:", (counts > 3).sum())

# parse surg_date (YYYY-MM-DD) and reformat to snapshot style YYYY-M-D (no zero pad)
sd = pd.to_datetime(live["surg_date"], errors="coerce")
assert sd.isnull().sum() == 0, "unparseable surg_date present"
tj_date = sd.dt.year.astype(str) + "-" + sd.dt.month.astype(str) + "-" + sd.dt.day.astype(str)

out = pd.DataFrame({
    "mlbamid": live["mlbamid_i"].astype("int64"),
    "TJ Surgery Date": tj_date,
    "Year of TJ": live["Year"].astype("int64"),
})

# cross-check Year vs surg_date year
mismatch = (out["Year of TJ"] != sd.dt.year).sum()
print("Year-of-TJ vs surg_date-year mismatches:", mismatch)

out_path = ROOT / "data" / "tj_live_for_extract.csv"
out.to_csv(out_path, index=False)
print("wrote:", out_path, "rows:", len(out))
print()
print("dtypes:\n", out.dtypes)
print()
print("3-row sample:")
print(out.head(3).to_string(index=False))
print()
# also show a 2022 example to confirm YYYY-M-D formatting on a single-digit month/day
ex = out[out["TJ Surgery Date"].str.startswith("2022")].head(3)
print("2022 samples:")
print(ex.to_string(index=False))
