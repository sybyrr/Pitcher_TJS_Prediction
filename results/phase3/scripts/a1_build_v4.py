"""A1: extend the prospective base to 2025 (v4).

Same spec as p0_build_v3.py, +2025: decision grid Apr-Sep 2017-2025 (54 dates),
fold_main test = 2022-25. Outputs data/prospective/{slim_games_v4,
game_features_v4, cohort_v4}.parquet.
GATES: <=2024 subsets identical to the v3 builds; shared-fold positive counts
match v3 (train 138/205, valid 35/47, test22-24 169/227).
Label maturity: t=2025-09-01 + H150 => 2026-01-28; sheet covers through 2026-07.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "prospective"
t0 = time.time()

# ------------------------------------------------------------------ slim v4
cols = ["pitcher", "game_date", "release_speed", "game_type"]
frames = []
for y in range(2016, 2026):
    frames.append(pd.read_parquet(RAW / f"statcast_{y}.parquet", columns=cols))
data = pd.concat(frames, ignore_index=True)
data = data[data["game_type"] == "R"].copy()
data["game_date"] = pd.to_datetime(data["game_date"])
data["pitcher"] = data["pitcher"].astype("int64")
data["release_speed"] = pd.to_numeric(data["release_speed"], errors="coerce")
g = data.groupby(["pitcher", "game_date"], sort=False)
slim = g.agg(pitch_count=("release_speed", "size"),
             mean_release_speed=("release_speed", "mean")).reset_index()
slim["game_year"] = slim["game_date"].dt.year
slim = slim.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
print(f"[t={time.time()-t0:.0f}s] slim_v4 rows {len(slim):,}  2025 rows {int((slim['game_year']==2025).sum()):,}")
slim_v3 = pd.read_parquet(OUT_DIR / "slim_games_v3.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
sub = slim[slim["game_year"] <= 2024].reset_index(drop=True)
assert len(sub) == len(slim_v3)
assert (sub["pitcher"].values == slim_v3["pitcher"].values).all()
assert (sub["pitch_count"].values == slim_v3["pitch_count"].values).all()
print("G1a slim<=2024 == slim_v3 : OK")
slim.to_parquet(OUT_DIR / "slim_games_v4.parquet")

# ------------------------------------------------------------------ gf v4
PTS = ["FF", "SI", "SL", "CH", "CU", "FC"]
COLS = ["pitcher", "game_date", "game_type", "pitch_type", "p_throws",
        "release_speed", "release_spin_rate", "release_pos_x",
        "release_pos_z", "release_extension"]
frames = []
for y in range(2016, 2026):
    df = pd.read_parquet(RAW / f"statcast_{y}.parquet", columns=COLS)
    frames.append(df[df["game_type"] == "R"])
data = pd.concat(frames, ignore_index=True)
data["pitcher"] = data["pitcher"].astype("int64")
data["game_date"] = pd.to_datetime(data["game_date"])
for c in ["release_speed", "release_spin_rate", "release_pos_x",
          "release_pos_z", "release_extension"]:
    data[c] = pd.to_numeric(data[c], errors="coerce").astype("float64")
lhp = data["p_throws"].eq("L")
data.loc[lhp, "release_pos_x"] = -data.loc[lhp, "release_pos_x"]
base = (data.groupby(["pitcher", "game_date"], sort=False)
            .size().rename("total_pitches").reset_index())
sub2 = data[data["pitch_type"].isin(PTS)].copy()
agg = (sub2.groupby(["pitcher", "game_date", "pitch_type"], observed=True)
           .agg(n=("release_speed", "size"),
                velo_mean=("release_speed", "mean"), velo_sd=("release_speed", "std"),
                spin_mean=("release_spin_rate", "mean"), spin_sd=("release_spin_rate", "std"),
                relx_mean=("release_pos_x", "mean"), relx_sd=("release_pos_x", "std"),
                relz_mean=("release_pos_z", "mean"),
                ext_mean=("release_extension", "mean"), ext_sd=("release_extension", "std")))
wide = agg.unstack("pitch_type")
flat = {}
for stat, pt in wide.columns:
    flat[(stat, pt)] = f"n_{pt}" if stat == "n" else f"{pt}_{stat}"
wide.columns = [flat[c] for c in wide.columns]
wide = wide.reset_index()
gf = base.merge(wide, on=["pitcher", "game_date"], how="left")
for pt in PTS:
    gf[f"n_{pt}"] = gf[f"n_{pt}"].fillna(0).astype("int64")
stat_order = ["n", "velo_mean", "velo_sd", "spin_mean", "spin_sd",
              "relx_mean", "relx_sd", "relz_mean", "ext_mean", "ext_sd"]
ordered = ["pitcher", "game_date", "total_pitches"]
for pt in PTS:
    for stat in stat_order:
        ordered.append(f"n_{pt}" if stat == "n" else f"{pt}_{stat}")
gf = gf[ordered].sort_values(["pitcher", "game_date"]).reset_index(drop=True)
gf_v3 = pd.read_parquet(OUT_DIR / "game_features_v3.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
sub = gf[gf["game_date"].dt.year <= 2024].reset_index(drop=True)
assert len(sub) == len(gf_v3)
assert (sub["pitcher"].values == gf_v3["pitcher"].values).all()
assert (sub["total_pitches"].values == gf_v3["total_pitches"].values).all()
print("G1b gf<=2024 == gf_v3 : OK")
gf.to_parquet(OUT_DIR / "game_features_v4.parquet")
print(f"[t={time.time()-t0:.0f}s] gf_v4 shape {gf.shape}")

# ------------------------------------------------------------------ cohort v4
sg = slim[["pitcher", "game_date"]].copy()
tj = pd.read_csv(OUT_DIR / "tj_live_clean_20260707.csv")
tj["surg_date"] = pd.to_datetime(tj["surg_date"])
surg_by_pitcher = {int(p): np.sort(g["surg_date"].values) for p, g in tj.groupby("mlbamid_i")}
grid = [pd.Timestamp(year=y, month=m, day=1) for y in range(2017, 2026) for m in range(4, 10)]
assert len(grid) == 54
rows = []
for t in grid:
    before = sg[sg["game_date"] < t]
    agg2 = before.groupby("pitcher")["game_date"].agg(["count", "max"])
    agg2 = agg2.rename(columns={"count": "n_career_games", "max": "last_game"})
    agg2["dsl"] = (t - agg2["last_game"]).dt.days
    risk = agg2[(agg2["n_career_games"] >= 20) & (agg2["dsl"] <= 365)]
    for pid, r in risk.iterrows():
        rows.append((int(pid), t, int(r["dsl"]), int(r["n_career_games"])))
cohort = pd.DataFrame(rows, columns=["pitcher", "t", "dsl", "n_career_games"])
cohort["month"] = cohort["t"].dt.month.astype(int)
cohort["year"] = cohort["t"].dt.year.astype(int)
cohort["fold_main"] = cohort["year"].map(lambda y: "train" if y <= 2020 else ("valid" if y == 2021 else "test"))
HB = [(90, 0), (90, 30), (90, 60), (150, 0), (150, 30), (150, 60), (150, 90),
      (365, 0), (365, 30), (365, 60), (365, 90)]
label_cols = {f"label_H{H}_B{B}": np.zeros(len(cohort), dtype=np.int8) for (H, B) in HB}
next_surg = np.full(len(cohort), np.datetime64("NaT"), dtype="datetime64[ns]")
t_vals = cohort["t"].values.astype("datetime64[ns]")
pids = cohort["pitcher"].values
DAY = np.timedelta64(1, "D")
for i in range(len(cohort)):
    surgs = surg_by_pitcher.get(int(pids[i]))
    if surgs is None:
        continue
    t = t_vals[i]
    after = surgs[surgs > t]
    if after.size:
        next_surg[i] = after.min()
    for (H, B) in HB:
        if np.any((surgs > t + B * DAY) & (surgs <= t + H * DAY)):
            label_cols[f"label_H{H}_B{B}"][i] = 1
for k, v in label_cols.items():
    cohort[k] = v
cohort["next_surgery_date"] = next_surg
ordered = ["pitcher", "t", "dsl", "n_career_games", "month", "year", "fold_main"] + \
          [f"label_H{H}_B{B}" for (H, B) in HB] + ["next_surgery_date"]
cohort = cohort[ordered].sort_values(["t", "pitcher"]).reset_index(drop=True)

v3 = pd.read_parquet(OUT_DIR / "cohort_v3.parquet").sort_values(["t", "pitcher"]).reset_index(drop=True)
sub = cohort[cohort["year"] <= 2024].reset_index(drop=True)
assert len(sub) == len(v3)
for c in ["pitcher", "dsl", "n_career_games", "month", "year", "fold_main"] + [f"label_H{H}_B{B}" for (H, B) in HB]:
    assert (sub[c].values == v3[c].values).all(), c
print("G2 cohort_v4[year<=2024] == cohort_v3 : OK")

print("\nfold_main sizes:", cohort["fold_main"].value_counts().to_dict())
for col, (et, ev, e2224) in {"label_H90_B0": (138, 35, 169), "label_H150_B0": (205, 47, 227)}.items():
    p_tr = int(cohort.loc[cohort["fold_main"] == "train", col].sum())
    p_va = int(cohort.loc[cohort["fold_main"] == "valid", col].sum())
    p_te24 = int(cohort.loc[(cohort["fold_main"] == "test") & (cohort["year"] <= 2024), col].sum())
    p_25 = int(cohort.loc[cohort["year"] == 2025, col].sum())
    assert (p_tr, p_va, p_te24) == (et, ev, e2224)
    print(f"{col}: train {p_tr} valid {p_va} test22-24 {p_te24} (+2025: {p_25}) total-test {p_te24+p_25}")
te = cohort["fold_main"] == "test"
for H in (90, 150):
    pos = cohort[te & (cohort[f"label_H{H}_B0"] == 1)]
    ev = pos.groupby(["pitcher", "next_surgery_date"]).ngroups
    ev25 = pos[pos["year"] == 2025].groupby(["pitcher", "next_surgery_date"]).ngroups
    print(f"H{H}: distinct test events {ev}  (2025-window events {ev25})")
cohort.to_parquet(OUT_DIR / "cohort_v4.parquet")
print(f"[t={time.time()-t0:.0f}s] SAVED v4 (cohort {cohort.shape})")
