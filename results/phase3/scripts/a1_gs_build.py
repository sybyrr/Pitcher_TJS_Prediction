"""A1: build game-started (GS) flags from raw statcast (spec v2).

Starter definition = first pitcher of each (game_pk, inning_topbot) by
at_bat_number (the pitcher who throws the game's first pitch for that side).
Regular season only, 2016-2025. Output data/prospective/gs_flags_v1.parquet
with one row per (pitcher, game_date): n_g (games pitched that date),
n_gs (starts that date; doubleheaders can give 2).
Sanity: yearly GS totals ~ 2 * #games (2430*2 full seasons, ~1800 in 2020).
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
RAW = ROOT / "data" / "raw"
t0 = time.time()

cols = ["game_pk", "game_date", "pitcher", "game_type", "inning_topbot", "at_bat_number"]
frames = []
for y in range(2016, 2026):
    df = pd.read_parquet(RAW / f"statcast_{y}.parquet", columns=cols)
    frames.append(df[df["game_type"] == "R"])
data = pd.concat(frames, ignore_index=True)
data["game_date"] = pd.to_datetime(data["game_date"])
data["pitcher"] = data["pitcher"].astype("int64")
print(f"[t={time.time()-t0:.0f}s] raw rows {len(data):,}")

first_ab = (data.groupby(["game_pk", "inning_topbot"], sort=False)["at_bat_number"]
                .transform("min"))
starters = (data[data["at_bat_number"] == first_ab]
            [["game_pk", "game_date", "pitcher"]].drop_duplicates())
n_sides = data.groupby("game_pk")["inning_topbot"].nunique()
print(f"games {data['game_pk'].nunique():,}  starter rows {len(starters):,}  "
      f"games with 2 sides {(n_sides == 2).sum():,}")

games = data[["pitcher", "game_date", "game_pk"]].drop_duplicates()
games = games.merge(starters.assign(is_gs=1), on=["game_pk", "game_date", "pitcher"], how="left")
games["is_gs"] = games["is_gs"].fillna(0).astype("int8")
gs = (games.groupby(["pitcher", "game_date"], sort=False)
           .agg(n_g=("game_pk", "nunique"), n_gs=("is_gs", "sum"))
           .reset_index().sort_values(["pitcher", "game_date"]).reset_index(drop=True))
print(f"gs rows {len(gs):,}")
yearly = starters.assign(y=starters["game_date"].dt.year).groupby("y").size()
print("yearly GS totals:\n", yearly.to_string())
gs.to_parquet(ROOT / "data/prospective/gs_flags_v1.parquet")
print(f"[t={time.time()-t0:.0f}s] wrote gs_flags_v1.parquet")
