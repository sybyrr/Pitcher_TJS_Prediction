from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

scratch = Path(r"C:\Users\PC\AppData\Local\Temp\claude\d--PAINS-Pitcher-TJS-Prediction\c8255b88-a70d-45ba-964b-531a2271c93f\scratchpad")
repo = Path(r"d:\PAINS\Pitcher_TJS_Prediction")
out_dir = repo / "data" / "prospective"
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
sg = pd.read_parquet(scratch / "slim_games.parquet")
sg = sg[["pitcher", "game_date"]].copy()
sg["game_date"] = pd.to_datetime(sg["game_date"])

tj = pd.read_csv(scratch / "tj_live_clean.csv")
tj["surg_date"] = pd.to_datetime(tj["surg_date"])
# mlbamid == mlbamid_i (verified). Use mlbamid_i.
# surgeries per pitcher: sorted unique dates
surg_by_pitcher: dict[int, np.ndarray] = {}
for pid, grp in tj.groupby("mlbamid_i"):
    dates = np.sort(grp["surg_date"].values)  # datetime64[ns]
    surg_by_pitcher[int(pid)] = dates

# ------------------------------------------------------------------
# Decision grid: 1st of Apr..Sep, 2017-2023  (42 dates)
# ------------------------------------------------------------------
grid = [pd.Timestamp(year=y, month=m, day=1)
        for y in range(2017, 2024)
        for m in range(4, 10)]
assert len(grid) == 42, len(grid)

# ------------------------------------------------------------------
# Risk set per t:  n_career_games >= 20 (games strictly before t)
#                  AND dsl = (t - last_game_before_t).days <= 365
# ------------------------------------------------------------------
rows = []
for t in grid:
    before = sg[sg["game_date"] < t]
    if before.empty:
        continue
    agg = before.groupby("pitcher")["game_date"].agg(["count", "max"])
    agg = agg.rename(columns={"count": "n_career_games", "max": "last_game"})
    agg["dsl"] = (t - agg["last_game"]).dt.days
    risk = agg[(agg["n_career_games"] >= 20) & (agg["dsl"] <= 365)]
    for pid, r in risk.iterrows():
        rows.append((int(pid), t, int(r["dsl"]), int(r["n_career_games"])))

cohort = pd.DataFrame(rows, columns=["pitcher", "t", "dsl", "n_career_games"])
cohort["month"] = cohort["t"].dt.month.astype(int)
cohort["year"] = cohort["t"].dt.year.astype(int)

# ------------------------------------------------------------------
# Folds by t.year
# ------------------------------------------------------------------
def fold_main(y: int) -> str:
    if y in (2017, 2018, 2019, 2020):
        return "train"
    if y == 2021:
        return "valid"
    return "test"  # 2022, 2023

def fold_alt(y: int) -> str:  # H=365 only
    if y in (2017, 2018, 2019):
        return "train"
    if y in (2020, 2021):
        return "valid"
    return "test"  # 2022, 2023

cohort["fold_main"] = cohort["year"].map(fold_main)
cohort["fold_alt"] = cohort["year"].map(fold_alt)

# ------------------------------------------------------------------
# Labels: label(H,B) = 1 iff any surgery s with  t+B < s <= t+H
#         next_surgery_date = first surgery strictly after t (NaT if none)
# ------------------------------------------------------------------
HB = [(90, 0), (90, 30), (90, 60),
      (150, 0), (150, 30), (150, 60), (150, 90),
      (365, 0), (365, 30), (365, 60), (365, 90)]

label_cols = {f"label_H{H}_B{B}": np.zeros(len(cohort), dtype=np.int8) for (H, B) in HB}
next_surg = np.full(len(cohort), np.datetime64("NaT"), dtype="datetime64[ns]")

t_vals = cohort["t"].values.astype("datetime64[ns]")
pids = cohort["pitcher"].values
DAY = np.timedelta64(1, "D")

for i in range(len(cohort)):
    pid = int(pids[i])
    surgs = surg_by_pitcher.get(pid)
    if surgs is None:
        continue
    t = t_vals[i]
    after = surgs[surgs > t]
    if after.size:
        next_surg[i] = after.min()
    for (H, B) in HB:
        lo = t + B * DAY
        hi = t + H * DAY
        if np.any((surgs > lo) & (surgs <= hi)):
            label_cols[f"label_H{H}_B{B}"][i] = 1

for k, v in label_cols.items():
    cohort[k] = v
cohort["next_surgery_date"] = next_surg

# ------------------------------------------------------------------
# Column order
# ------------------------------------------------------------------
ordered = ["pitcher", "t", "dsl", "n_career_games", "month", "year",
           "fold_main", "fold_alt"] + \
          [f"label_H{H}_B{B}" for (H, B) in HB] + ["next_surgery_date"]
cohort = cohort[ordered].sort_values(["t", "pitcher"]).reset_index(drop=True)

# ------------------------------------------------------------------
# VERIFICATION
# ------------------------------------------------------------------
print("=" * 70)
print("Total windows per fold_main:")
print(cohort["fold_main"].value_counts().reindex(["train", "valid", "test"]))
print()
print("Total windows per fold_alt:")
print(cohort["fold_alt"].value_counts().reindex(["train", "valid", "test"]))
print()

expected = {
    ("label_H90_B0"):  {"train": 138, "valid": 35, "test": 118},
    ("label_H150_B0"): {"train": 205, "valid": 47, "test": 162},
}
all_ok = True
for col, exp in expected.items():
    pos = cohort[cohort[col] == 1].groupby("fold_main").size().reindex(["train", "valid", "test"]).fillna(0).astype(int)
    print(f"--- {col} positives by fold_main:")
    for f in ["train", "valid", "test"]:
        got = int(pos[f])
        e = exp[f]
        ok = got == e
        all_ok = all_ok and ok
        print(f"    {f:6s}: got {got:4d}  expected {e:4d}  {'OK' if ok else 'MISMATCH'}")
print()
print("ALL EXPECTED POSITIVE COUNTS MATCH:", all_ok)
print()

# full positive summary for all label cols (fold_main and, for H365, fold_alt)
print("All label positive counts (fold_main):")
for (H, B) in HB:
    col = f"label_H{H}_B{B}"
    pos = cohort[cohort[col] == 1].groupby("fold_main").size().reindex(["train", "valid", "test"]).fillna(0).astype(int)
    print(f"  {col:16s}  train {int(pos['train']):4d}  valid {int(pos['valid']):4d}  test {int(pos['test']):4d}  total {int(cohort[col].sum()):4d}")
print()
print("H365 label positive counts (fold_alt):")
for (H, B) in [(365,0),(365,30),(365,60),(365,90)]:
    col = f"label_H{H}_B{B}"
    pos = cohort[cohort[col] == 1].groupby("fold_alt").size().reindex(["train", "valid", "test"]).fillna(0).astype(int)
    print(f"  {col:16s}  train {int(pos['train']):4d}  valid {int(pos['valid']):4d}  test {int(pos['test']):4d}  total {int(cohort[col].sum()):4d}")

print()
print("cohort shape:", cohort.shape)
print("columns:", list(cohort.columns))
print("dtypes:\n", cohort.dtypes)
print()
print("n unique pitchers in cohort:", cohort["pitcher"].nunique())
print("dsl min/max:", cohort["dsl"].min(), cohort["dsl"].max())
print("n_career_games min/max:", cohort["n_career_games"].min(), cohort["n_career_games"].max())
print("windows per t (year,month):")
print(cohort.groupby(["year", "month"]).size())

# ------------------------------------------------------------------
# Save only if verification passes
# ------------------------------------------------------------------
out_path = out_dir / "cohort_v2.parquet"
if all_ok:
    cohort.to_parquet(out_path, index=False)
    print("\nSAVED:", out_path)
else:
    print("\nNOT SAVED — verification failed.")
