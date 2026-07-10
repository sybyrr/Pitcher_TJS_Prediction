"""P0-1b: extend the prospective data base to 2024.

Builds (2016-2024 raw, game_type=='R'):
  data/prospective/slim_games_v3.parquet     (build_slim.py spec, +2024; promoted
                                              from scratchpad to persistent storage)
  data/prospective/game_features_v3.parquet  (gf_build.py spec verbatim, +2024,
                                              incl. LHP release_pos_x sign flip)
  data/prospective/cohort_v3.parquet         (build_cohort_v2.py spec, decision
                                              grid extended to Apr-Sep 2017-2024;
                                              fold_main test = 2022-24)

Labels: data/prospective/tj_live_clean_20260707.csv (md5-identical to the sheet
that built cohort_v2; surgeries through 2026 -> 2024 decisions fully mature).

VERIFICATION GATES (abort on failure):
  G1 slim/gf rows for game_year<=2023 identical to the v2 builds.
  G2 cohort_v3 restricted to year<=2023 == cohort_v2 exactly (all columns).
  G3 fold positive counts for shared years match v2 expectations.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "prospective"
SCR = Path(r"C:/Users/PC/AppData/Local/Temp/claude/d--PAINS-Pitcher-TJS-Prediction/c8255b88-a70d-45ba-964b-531a2271c93f/scratchpad")
t0 = time.time()

# ------------------------------------------------------------------ slim v3
cols = ["pitcher", "game_date", "release_speed", "game_type"]
frames = []
for y in range(2016, 2025):
    df = pd.read_parquet(RAW / f"statcast_{y}.parquet", columns=cols)
    frames.append(df)
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
print(f"[t={time.time()-t0:.0f}s] slim_v3 rows {len(slim):,}  2024 rows "
      f"{int((slim['game_year']==2024).sum()):,}")

# G1a: <=2023 subset must equal the v2 slim
slim_v2 = pd.read_parquet(SCR / "slim_games.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
sub = slim[slim["game_year"] <= 2023].reset_index(drop=True)
assert len(sub) == len(slim_v2), (len(sub), len(slim_v2))
assert (sub["pitcher"].values == slim_v2["pitcher"].values).all()
assert (sub["pitch_count"].values == slim_v2["pitch_count"].values).all()
mrs_diff = np.nanmax(np.abs(sub["mean_release_speed"].values - slim_v2["mean_release_speed"].values))
assert mrs_diff < 1e-9, mrs_diff
print("G1a slim<=2023 == slim_v2 : OK")
slim.to_parquet(OUT_DIR / "slim_games_v3.parquet")

# ------------------------------------------------------------------ gf v3 (gf_build.py verbatim, +2024)
PTS = ["FF", "SI", "SL", "CH", "CU", "FC"]
COLS = ["pitcher", "game_date", "game_type", "pitch_type", "p_throws",
        "release_speed", "release_spin_rate", "release_pos_x",
        "release_pos_z", "release_extension"]
frames = []
for y in range(2016, 2025):
    df = pd.read_parquet(RAW / f"statcast_{y}.parquet", columns=COLS)
    df = df[df["game_type"] == "R"]
    frames.append(df)
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
sub = data[data["pitch_type"].isin(PTS)].copy()
agg = (sub.groupby(["pitcher", "game_date", "pitch_type"], observed=True)
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
print(f"[t={time.time()-t0:.0f}s] gf_v3 shape {gf.shape}")

# G1b: <=2023 subset must equal gf_v2
gf_v2 = pd.read_parquet(OUT_DIR / "game_features_v2.parquet").sort_values(["pitcher", "game_date"]).reset_index(drop=True)
sub = gf[gf["game_date"].dt.year <= 2023].reset_index(drop=True)
assert len(sub) == len(gf_v2), (len(sub), len(gf_v2))
assert (sub["pitcher"].values == gf_v2["pitcher"].values).all()
assert (sub["total_pitches"].values == gf_v2["total_pitches"].values).all()
for c in ["FF_velo_mean", "SL_spin_mean", "SI_relx_mean", "CU_ext_mean"]:
    a = sub[c].values; b = gf_v2[c].values
    m = ~(np.isnan(a) & np.isnan(b))
    assert np.nanmax(np.abs(a[m] - b[m])) < 1e-9, c
print("G1b gf<=2023 == gf_v2 : OK")
gf.to_parquet(OUT_DIR / "game_features_v3.parquet")

# ------------------------------------------------------------------ cohort v3 (build_cohort_v2.py spec, grid +2024)
sg = slim[["pitcher", "game_date"]].copy()
tj = pd.read_csv(OUT_DIR / "tj_live_clean_20260707.csv")
tj["surg_date"] = pd.to_datetime(tj["surg_date"])
surg_by_pitcher = {int(p): np.sort(g["surg_date"].values) for p, g in tj.groupby("mlbamid_i")}

grid = [pd.Timestamp(year=y, month=m, day=1) for y in range(2017, 2025) for m in range(4, 10)]
assert len(grid) == 48

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

def fold_main(y: int) -> str:
    if y in (2017, 2018, 2019, 2020):
        return "train"
    if y == 2021:
        return "valid"
    return "test"  # 2022, 2023, 2024

cohort["fold_main"] = cohort["year"].map(fold_main)

HB = [(90, 0), (90, 30), (90, 60),
      (150, 0), (150, 30), (150, 60), (150, 90),
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

# G2: year<=2023 subset must equal cohort_v2 (all shared columns)
v2 = pd.read_parquet(OUT_DIR / "cohort_v2.parquet").sort_values(["t", "pitcher"]).reset_index(drop=True)
sub = cohort[cohort["year"] <= 2023].reset_index(drop=True)
assert len(sub) == len(v2), (len(sub), len(v2))
for c in ["pitcher", "dsl", "n_career_games", "month", "year"] + [f"label_H{H}_B{B}" for (H, B) in HB]:
    assert (sub[c].values == v2[c].values).all(), c
assert (sub["t"].values == v2["t"].values).all()
ns_a = sub["next_surgery_date"].values.astype("datetime64[ns]")
ns_b = v2["next_surgery_date"].values.astype("datetime64[ns]")
assert ((ns_a == ns_b) | (np.isnat(ns_a) & np.isnat(ns_b))).all()
assert (sub["fold_main"].values == v2["fold_main"].values).all()
print("G2 cohort_v3[year<=2023] == cohort_v2 : OK")

# G3: shared-year positive counts + report 2024
print("\nfold_main sizes:", cohort["fold_main"].value_counts().to_dict())
exp = {"label_H90_B0": (138, 35, 118), "label_H150_B0": (205, 47, 162)}
for col, (et, ev, ee) in exp.items():
    p_tr = int(cohort.loc[cohort["fold_main"] == "train", col].sum())
    p_va = int(cohort.loc[cohort["fold_main"] == "valid", col].sum())
    p_te23 = int(cohort.loc[(cohort["fold_main"] == "test") & (cohort["year"] <= 2023), col].sum())
    p_te24 = int(cohort.loc[cohort["year"] == 2024, col].sum())
    assert (p_tr, p_va, p_te23) == (et, ev, ee), (col, p_tr, p_va, p_te23)
    print(f"{col}: train {p_tr} valid {p_va} test22-23 {p_te23} (+2024: {p_te24}) total-test {p_te23+p_te24}")

# distinct test events (pitcher, next_surgery) for H90/H150
te = cohort["fold_main"] == "test"
for H in (90, 150):
    pos = cohort[te & (cohort[f"label_H{H}_B0"] == 1)]
    ev = pos.groupby(["pitcher", "next_surgery_date"]).ngroups
    ev24 = pos[pos["year"] == 2024].groupby(["pitcher", "next_surgery_date"]).ngroups
    print(f"H{H}: distinct test events {ev}  (2024-window events {ev24})")

print("\ncohort_v3 shape:", cohort.shape, " 2024 windows:", int((cohort['year'] == 2024).sum()))
cohort.to_parquet(OUT_DIR / "cohort_v3.parquet")
print(f"[t={time.time()-t0:.0f}s] SAVED slim_games_v3 / game_features_v3 / cohort_v3")
