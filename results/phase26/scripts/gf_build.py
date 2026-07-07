"""A1 — build extended per-game feature table game_features_v2.parquet.

From data/raw/statcast_{2016..2023}.parquet, game_type=='R' only.
- Flip release_pos_x sign for LHP (p_throws=='L') BEFORE aggregating.
- Per (pitcher, game_date): total_pitches + per-pitch-type stats for
  {FF, SI, SL, CH, CU, FC}.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"d:/PAINS/Pitcher_TJS_Prediction")
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "prospective"
OUT = OUT_DIR / "game_features_v2.parquet"

PTS = ["FF", "SI", "SL", "CH", "CU", "FC"]
COLS = ["pitcher", "game_date", "game_type", "pitch_type", "p_throws",
        "release_speed", "release_spin_rate", "release_pos_x",
        "release_pos_z", "release_extension"]

t0 = time.time()

# ---- load + concat (R only) ----
frames = []
for y in range(2016, 2024):
    df = pd.read_parquet(RAW / f"statcast_{y}.parquet", columns=COLS)
    df = df[df["game_type"] == "R"]
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
n_raw_R = len(data)

# ---- types ----
data["pitcher"] = data["pitcher"].astype("int64")
data["game_date"] = pd.to_datetime(data["game_date"])
for c in ["release_speed", "release_spin_rate", "release_pos_x",
          "release_pos_z", "release_extension"]:
    data[c] = pd.to_numeric(data[c], errors="coerce").astype("float64")

# ---- flip horizontal release for LHP (handedness-comparable) ----
lhp = data["p_throws"].eq("L")
data.loc[lhp, "release_pos_x"] = -data.loc[lhp, "release_pos_x"]
n_lhp = int(lhp.sum())

# ---- total_pitches: all R pitches per (pitcher, game_date) ----
base = (data.groupby(["pitcher", "game_date"], sort=False)
            .size().rename("total_pitches").reset_index())

# ---- per pitch-type aggregation ----
sub = data[data["pitch_type"].isin(PTS)].copy()
agg = (sub.groupby(["pitcher", "game_date", "pitch_type"], observed=True)
          .agg(n=("release_speed", "size"),
               velo_mean=("release_speed", "mean"),
               velo_sd=("release_speed", "std"),
               spin_mean=("release_spin_rate", "mean"),
               spin_sd=("release_spin_rate", "std"),
               relx_mean=("release_pos_x", "mean"),
               relx_sd=("release_pos_x", "std"),
               relz_mean=("release_pos_z", "mean"),
               ext_mean=("release_extension", "mean"),
               ext_sd=("release_extension", "std")))

# unstack pitch_type -> wide columns (stat, pitch_type)
wide = agg.unstack("pitch_type")
# flatten: n -> n_<pt> ; others -> <pt>_<stat>
new_cols = {}
flat = {}
for stat, pt in wide.columns:
    name = f"n_{pt}" if stat == "n" else f"{pt}_{stat}"
    flat[(stat, pt)] = name
wide.columns = [flat[c] for c in wide.columns]
wide = wide.reset_index()

# ---- assemble: base (all games) LEFT JOIN wide ----
out = base.merge(wide, on=["pitcher", "game_date"], how="left")

# n_<pt> should be integer count; NaN -> 0 (game had none of that type)
for pt in PTS:
    out[f"n_{pt}"] = out[f"n_{pt}"].fillna(0).astype("int64")

# ---- column order: keys, total, then per pitch-type block ----
stat_order = ["n", "velo_mean", "velo_sd", "spin_mean", "spin_sd",
              "relx_mean", "relx_sd", "relz_mean", "ext_mean", "ext_sd"]
ordered = ["pitcher", "game_date", "total_pitches"]
for pt in PTS:
    for stat in stat_order:
        ordered.append(f"n_{pt}" if stat == "n" else f"{pt}_{stat}")
out = out[ordered].sort_values(["pitcher", "game_date"]).reset_index(drop=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT)

print(f"raw R rows loaded: {n_raw_R:,}  (LHP rows flipped: {n_lhp:,})")
print(f"game_features_v2 shape: {out.shape}")
print(f"columns ({len(out.columns)}): {list(out.columns)}")
print(f"saved -> {OUT}")
print(f"build seconds: {time.time()-t0:.1f}")
