"""Build the (H, B, F) SUPPLY GRID for the TJS horizon/blackout/floor decision.

Counting only. No features, no models.

Definitions (from task spec):
  - Decision dates t: 1st of Apr..Sep, years 2017-2023  (6/yr * 7 = 42 dates).
  - Game universe: slim_games.parquet (pitcher, game_date), reg season 2016-2023.
  - Risk set at t per floor F: pitcher has >=20 games strictly before t AND
    days_since_last <= F, where days_since_last = (t - last game before t).days.
    F in {30, 365, 730, INF}.
  - Label(H,B) for (pitcher,t): positive iff any surgery s with t+B < s <= t+H.
    H in {90,150,365}, B in {0,30,60,90}, only B < H.
  - Folds by t.year. main: train 2017-2020 / valid 2021 / test 2022-2023.
                     alt : train 2017-2019 / valid 2020-2021 / test 2022-2023.
  - Strict embargo (train/valid only): main -> train t+H<=2021-04-01,
    valid t+H<=2022-04-01. alt -> train t+H<=2020-04-01, valid t+H<=2022-04-01.
  - Boundary: an evaluated window requires t+H <= label end. ends: conservative
    2023-12-31, E0a 2024-12-31.  (applied to all folds; only bites test in practice)
"""
from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
import pandas as pd

SP = Path(r"C:\Users\PC\AppData\Local\Temp\claude"
          r"\d--PAINS-Pitcher-TJS-Prediction"
          r"\c8255b88-a70d-45ba-964b-531a2271c93f\scratchpad")

# ---------------------------------------------------------------- load
games = pd.read_parquet(SP / "slim_games.parquet")[["pitcher", "game_date"]].copy()
games["game_date"] = pd.to_datetime(games["game_date"])
games = games.drop_duplicates().sort_values(["pitcher", "game_date"])

tj = pd.read_csv(SP / "tj_live_clean.csv")
tj["sd"] = pd.to_datetime(tj["surg_date"], errors="coerce")
tj = tj.dropna(subset=["sd", "mlbamid_i"])
tj["mlbamid_i"] = tj["mlbamid_i"].astype("int64")

# surgeries only matter for pitchers that appear in the game universe
game_pids = set(games["pitcher"].unique())
tj = tj[tj["mlbamid_i"].isin(game_pids)].copy()
# distinct surgery id (pitcher, date)
tj = tj.drop_duplicates(subset=["mlbamid_i", "sd"]).reset_index(drop=True)
surg_by_pid: dict[int, np.ndarray] = {
    int(p): np.sort(g["sd"].values.astype("datetime64[D]"))
    for p, g in tj.groupby("mlbamid_i")
}
print(f"[load] games rows={len(games):,} pitchers={len(game_pids):,}")
print(f"[load] live surgeries (in-universe, distinct): {len(tj):,}")

# per-pitcher sorted game arrays (datetime64[D])
games_by_pid: dict[int, np.ndarray] = {
    int(p): g["game_date"].values.astype("datetime64[D]")
    for p, g in games.groupby("pitcher")
}

# ---------------------------------------------------------------- grid params
DECISION_DATES = [pd.Timestamp(y, m, 1)
                  for y in range(2017, 2024) for m in range(4, 10)]
assert len(DECISION_DATES) == 42
HS = [90, 150, 365]
BS = [0, 30, 60, 90]
HB = [(h, b) for h in HS for b in BS if b < h]  # 11 pairs
FLOORS = {"30": 30, "365": 365, "730": 730, "INF": None}
ENDS = {"cons_2023-12-31": pd.Timestamp("2023-12-31"),
        "e0a_2024-12-31": pd.Timestamp("2024-12-31")}
MAXH = max(HS)

# ---------------------------------------------------------------- build candidates
# candidate = (pitcher, t) with >=20 games strictly before t.
rows = []
for pid, garr in games_by_pid.items():
    surg = surg_by_pid.get(pid)  # may be None
    for t in DECISION_DATES:
        tD = np.datetime64(t.date(), "D")
        idx = int(np.searchsorted(garr, tD, side="left"))  # games strictly before t
        if idx < 20:
            continue
        last = garr[idx - 1]
        dsl = int((tD - last).astype("timedelta64[D]").astype(int))
        rec = {"pitcher": pid, "t": t, "year": t.year, "days_since_last": dsl}
        rows.append(rec)
cand = pd.DataFrame(rows)
cand = cand.reset_index(drop=True)
cand["cid"] = cand.index
print(f"[cand] candidate windows (>=20 games before t): {len(cand):,}")

# ---------------------------------------------------------------- candidate-surgery long table
# for each candidate, each surgery of that pitcher with 0 < offset <= MAXH.
long_rows = []
for r in cand.itertuples(index=False):
    surg = surg_by_pid.get(r.pitcher)
    if surg is None:
        continue
    tD = np.datetime64(r.t.date(), "D")
    off = (surg - tD).astype("timedelta64[D]").astype(int)
    m = (off > 0) & (off <= MAXH)
    for s_date, o in zip(surg[m], off[m]):
        long_rows.append((r.cid, r.pitcher,
                          pd.Timestamp(s_date), int(o)))
lng = pd.DataFrame(long_rows, columns=["cid", "pitcher", "surg_date", "offset"])
# surgery id
lng["sid"] = (lng["pitcher"].astype(str) + "|"
              + lng["surg_date"].dt.strftime("%Y-%m-%d"))
print(f"[long] candidate-surgery pairs (0<off<=%d): %d" % (MAXH, len(lng)))

# attach t/year to long via cand
cid2t = cand.set_index("cid")[["t", "year", "days_since_last"]]
lng = lng.join(cid2t, on="cid")

# ---------------------------------------------------------------- fold defs
def fold_of(year: int, foldset: str) -> str:
    if foldset == "main":
        if 2017 <= year <= 2020:
            return "train"
        if year == 2021:
            return "valid"
        if year in (2022, 2023):
            return "test"
    else:  # alt
        if 2017 <= year <= 2019:
            return "train"
        if year in (2020, 2021):
            return "valid"
        if year in (2022, 2023):
            return "test"
    return "none"

# embargo cutoffs: (foldset, fold) -> t+H must be <= this
EMBARGO = {
    ("main", "train"): pd.Timestamp("2021-04-01"),
    ("main", "valid"): pd.Timestamp("2022-04-01"),
    ("alt", "train"): pd.Timestamp("2020-04-01"),
    ("alt", "valid"): pd.Timestamp("2022-04-01"),
}

def floor_mask(dsl: pd.Series, F):
    if F is None:
        return pd.Series(True, index=dsl.index)
    return dsl <= F

# precompute fold labels
for fs in ("main", "alt"):
    cand[f"fold_{fs}"] = cand["year"].map(lambda y, fs=fs: fold_of(y, fs))
    lng[f"fold_{fs}"] = lng["year"].map(lambda y, fs=fs: fold_of(y, fs))

# ---------------------------------------------------------------- core counting
def window_keep_mask(df, H, foldset, fold_col, embargo, end):
    """Boolean mask: window kept under embargo+boundary for its fold."""
    tH = df["t"] + pd.Timedelta(days=H)
    keep = tH <= end  # boundary (all folds)
    if embargo == "strict":
        folds = df[fold_col]
        for (fs2, fold_name), cutoff in EMBARGO.items():
            if fs2 != foldset:
                continue
            sel = folds == fold_name
            keep = keep & (~sel | (tH <= cutoff))
    return keep

records = []
for foldset in ("main", "alt"):
    fold_col = f"fold_{foldset}"
    for (H, B) in HB:
        # positive rows for this (H,B): B < offset <= H
        pos_long_all = lng[(lng["offset"] > B) & (lng["offset"] <= H)]
        for Fname, F in FLOORS.items():
            # floor-eligible candidate mask
            cand_f = cand[floor_mask(cand["days_since_last"], F)]
            # floor-eligible positive-long rows (same floor on days_since_last)
            pos_f = pos_long_all[floor_mask(pos_long_all["days_since_last"], F)]
            for ename, end in ENDS.items():
                for embargo in ("none", "strict"):
                    # window keep mask on candidates
                    kc = window_keep_mask(cand_f, H, foldset, fold_col, embargo, end)
                    kept_cand = cand_f[kc]
                    # window keep mask on positive-long rows
                    kp = window_keep_mask(pos_f, H, foldset, fold_col, embargo, end)
                    kept_pos = pos_f[kp]
                    for fold in ("train", "valid", "test"):
                        cf = kept_cand[kept_cand[fold_col] == fold]
                        pf = kept_pos[kept_pos[fold_col] == fold]
                        n_win = len(cf)
                        n_pos = pf["cid"].nunique()
                        n_surg = pf["sid"].nunique()
                        br = (n_pos / n_win) if n_win else float("nan")
                        records.append(dict(
                            foldset=foldset, H=H, B=B, F=Fname,
                            end=ename, embargo=embargo, fold=fold,
                            n_windows=n_win, n_pos=n_pos,
                            n_distinct_surg=n_surg, base_rate=br))

grid = pd.DataFrame.from_records(records)
grid.to_csv(SP / "supply_grid.csv", index=False)
print(f"[grid] wrote supply_grid.csv rows={len(grid):,}")

# ---------------------------------------------------------------- RAW (no embargo, no boundary)
# used for sanity + monotonicity. end = far future, embargo none.
FAR = pd.Timestamp("2100-01-01")
raw_recs = []
for (H, B) in HB:
    pos_long_all = lng[(lng["offset"] > B) & (lng["offset"] <= H)]
    for Fname, F in FLOORS.items():
        cand_f = cand[floor_mask(cand["days_since_last"], F)]
        pos_f = pos_long_all[floor_mask(pos_long_all["days_since_last"], F)]
        for fold in ("train", "valid", "test"):
            cf = cand_f[cand_f["fold_main"] == fold]
            pf = pos_f[pos_f["fold_main"] == fold]
            raw_recs.append(dict(H=H, B=B, F=Fname, fold=fold,
                                 n_windows=len(cf),
                                 n_pos=pf["cid"].nunique(),
                                 n_distinct_surg=pf["sid"].nunique()))
raw = pd.DataFrame.from_records(raw_recs)
raw.to_csv(SP / "supply_raw_noembargo_noboundary.csv", index=False)

# ---------------------------------------------------------------- SANITY 1: 102/31/84
print("\n" + "=" * 70)
print("SANITY 1 — (H=150,B=0,F=30, no embargo, no boundary) vs known 102/31/84")
s = raw[(raw.H == 150) & (raw.B == 0) & (raw.F == "30")].set_index("fold")
for fold in ("train", "valid", "test"):
    r = s.loc[fold]
    print(f"  {fold:5s}: n_windows={r.n_windows:5d}  n_pos={r.n_pos:4d}  "
          f"distinct_surg={r.n_distinct_surg:4d}")
print("  known (snapshot): train_pos=102 valid_pos=31 test_pos=84")

# ---------------------------------------------------------------- SANITY 2: monotone in H (raw)
print("\n" + "=" * 70)
print("SANITY 2 — positive windows monotone non-decreasing in H (raw, per B,F,fold)")
mono_H_ok = True
for B in BS:
    for Fname in FLOORS:
        for fold in ("train", "valid", "test"):
            vals = []
            for H in HS:
                if B >= H:
                    continue
                sub = raw[(raw.H == H) & (raw.B == B) & (raw.F == Fname)
                          & (raw.fold == fold)]
                if len(sub):
                    vals.append((H, int(sub.n_pos.iloc[0])))
            for i in range(1, len(vals)):
                if vals[i][1] < vals[i - 1][1]:
                    mono_H_ok = False
                    print(f"  VIOLATION B={B} F={Fname} {fold}: "
                          f"H{vals[i-1][0]}={vals[i-1][1]} > H{vals[i][0]}={vals[i][1]}")
print(f"  monotone-in-H (positives): {'PASS' if mono_H_ok else 'FAIL'}")

# ---------------------------------------------------------------- SANITY 3: decreasing in B (raw)
print("\n" + "=" * 70)
print("SANITY 3 — positive windows non-increasing in B (raw, per H,F,fold)")
mono_B_ok = True
for H in HS:
    for Fname in FLOORS:
        for fold in ("train", "valid", "test"):
            vals = []
            for B in BS:
                if B >= H:
                    continue
                sub = raw[(raw.H == H) & (raw.B == B) & (raw.F == Fname)
                          & (raw.fold == fold)]
                if len(sub):
                    vals.append((B, int(sub.n_pos.iloc[0])))
            for i in range(1, len(vals)):
                if vals[i][1] > vals[i - 1][1]:
                    mono_B_ok = False
                    print(f"  VIOLATION H={H} F={Fname} {fold}: "
                          f"B{vals[i-1][0]}={vals[i-1][1]} < B{vals[i][0]}={vals[i][1]}")
print(f"  non-increasing-in-B (positives): {'PASS' if mono_B_ok else 'FAIL'}")

print("\n[done]")
