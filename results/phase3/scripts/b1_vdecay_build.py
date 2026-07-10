"""B1: per-game within-game fastball velocity decay table (2016-2025).

decay = mean(velo of LAST 5 fastballs) - mean(velo of FIRST 5 fastballs)
within a game, FB = FF/SI, requires >=10 FB pitches; order = (at_bat_number,
pitch_number). Negative = pitcher lost velocity during the outing.
Output: data/prospective/vdecay_games_v4.parquet (pitcher, game_date, n_fb, decay).
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "prospective" / "vdecay_games_v4.parquet"
t0 = time.time()

COLS = ["pitcher", "game_date", "game_type", "pitch_type", "release_speed",
        "at_bat_number", "pitch_number"]
parts = []
for y in range(2016, 2026):
    df = pd.read_parquet(RAW / f"statcast_{y}.parquet", columns=COLS)
    df = df[(df["game_type"] == "R") & df["pitch_type"].isin(["FF", "SI"])].copy()
    df["release_speed"] = pd.to_numeric(df["release_speed"], errors="coerce")
    df = df.dropna(subset=["release_speed"])
    df["pitcher"] = df["pitcher"].astype("int64")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["at_bat_number"] = pd.to_numeric(df["at_bat_number"], errors="coerce").fillna(0).astype("int64")
    df["pitch_number"] = pd.to_numeric(df["pitch_number"], errors="coerce").fillna(0).astype("int64")
    df = df.sort_values(["pitcher", "game_date", "at_bat_number", "pitch_number"])
    g = df.groupby(["pitcher", "game_date"], sort=False)["release_speed"]
    n = g.size()
    first5 = g.apply(lambda s: s.iloc[:5].mean())
    last5 = g.apply(lambda s: s.iloc[-5:].mean())
    out = pd.DataFrame({"n_fb": n, "decay": last5 - first5}).reset_index()
    out = out[out["n_fb"] >= 10]
    parts.append(out)
    print(f"{y}: games with >=10 FB: {len(out):,}  mean decay {out['decay'].mean():+.3f} mph", flush=True)

vd = pd.concat(parts, ignore_index=True).sort_values(["pitcher", "game_date"]).reset_index(drop=True)
vd.to_parquet(OUT)
print(f"\n[t={time.time()-t0:.0f}s] rows {len(vd):,}  overall mean decay {vd['decay'].mean():+.3f} "
      f"(expect negative, ~-0.3..-0.8 mph)  -> {OUT}")
