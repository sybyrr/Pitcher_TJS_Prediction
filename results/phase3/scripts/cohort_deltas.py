from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"d:\PAINS\Pitcher_TJS_Prediction")
SCRATCH = Path(r"C:\Users\PC\AppData\Local\Temp\claude\d--PAINS-Pitcher-TJS-Prediction\c8255b88-a70d-45ba-964b-531a2271c93f\scratchpad")


def samples(final_csv: Path) -> pd.DataFrame:
    """Distinct (pitcher, target) samples from a final_df, with player_name."""
    df = pd.read_csv(final_csv, usecols=["player_name", "pitcher", "target"])
    return (df.groupby(["pitcher", "target"])
              .agg(player_name=("player_name", "first"))
              .reset_index())


snap = samples(ROOT / "data" / "final_df.csv")
live = samples(ROOT / "data" / "final_df_live.csv")

print("=== SAMPLE COHORT SIZES  (a sample = one (pitcher,target) group) ===")
for name, c in [("SNAPSHOT", snap), ("LIVE", live)]:
    ni = int((c.target == 1).sum()); nn = int((c.target == 0).sum())
    print(f"{name}: samples={len(c)}  injured={ni}  normal={nn}  distinct_pitchers={c.pitcher.nunique()}")
print()

snap_s = set(zip(snap.pitcher, snap.target))
live_s = set(zip(live.pitcher, live.target))
added = live_s - snap_s
dropped = snap_s - live_s
print("=== SAMPLE-LEVEL DELTA (live vs snapshot) ===")
print(f"samples added in live : {len(added)}  (injured={sum(t==1 for _,t in added)}, normal={sum(t==0 for _,t in added)})")
print(f"samples dropped in live: {len(dropped)}  (injured={sum(t==1 for _,t in dropped)}, normal={sum(t==0 for _,t in dropped)})")
print()

# --- per-pitcher target-SET transition (handles dual-role) ---
def tset(c):
    return c.groupby("pitcher")["target"].apply(lambda s: frozenset(s)).to_dict()
st, lt = tset(snap), tset(live)
allp = sorted(set(st) | set(lt))
def lab(fs):
    if fs is None: return "absent"
    has0, has1 = 0 in fs, 1 in fs
    if has0 and has1: return "both"
    return "injured" if has1 else "normal"
rows = []
pname = pd.concat([snap, live]).drop_duplicates("pitcher").set_index("pitcher")["player_name"]
for p in allp:
    a, b = lab(st.get(p)), lab(lt.get(p))
    if a != b:
        rows.append({"pitcher": p, "player_name": pname.get(p, "?"), "snap": a, "live": b})
chg = pd.DataFrame(rows)
print("=== PER-PITCHER STATUS TRANSITIONS (snap -> live) ===")
print("total pitchers whose status changed:", len(chg))
print(chg.groupby(["snap", "live"]).size().reset_index(name="n").to_string(index=False))
print()

# pitchers who GAINED injured status (had no injured sample in snap, have one in live)
gained_inj = chg[chg.apply(lambda r: 1 not in (st.get(r.pitcher) or frozenset())
                                     and 1 in (lt.get(r.pitcher) or frozenset()), axis=1)]
print(f"pitchers who GAINED an injured sample (normal/absent -> injured/both): {len(gained_inj)}")
print(gained_inj[["pitcher","player_name","snap","live"]].to_string(index=False))
print()

# --- cross-check vs 71 missing 2022-23 surgeries ---
miss = pd.read_csv(SCRATCH / "missing_2223.csv")
miss_ids = set(miss["mlbamid_i"])
print("=== CROSS-CHECK vs 71 missing 2022-23 surgeries ===")
print("71-missing rows:", len(miss), " unique pitchers:", miss.mlbamid_i.nunique())
print("gained-injured pitchers that are in the 71-missing set:", len(gained_inj[gained_inj.pitcher.isin(miss_ids)]))
print("ALL status-changed pitchers in the 71-missing set:", len(chg[chg.pitcher.isin(miss_ids)]))
live_inj_ids = set(live[live.target == 1].pitcher)
snap_inj_ids = set(snap[snap.target == 1].pitcher)
print("71-missing pitchers now in LIVE injured cohort:", len(miss_ids & live_inj_ids))
print("71-missing pitchers already in SNAP injured cohort:", len(miss_ids & snap_inj_ids))
print("71-missing pitchers not in LIVE cohort at all (never pitched 2016-23 / ineligible):",
      len(miss_ids - set(live.pitcher)))
print()

# --- injured anchor-year (v9 temporal test = anchor>=2022) ---
def anchored(final_csv, meta_csv):
    c = samples(final_csv)
    m = pd.read_csv(meta_csv, parse_dates=["anchor_date"])
    key = {(int(p), int(t)): d.year for p, t, d in zip(m.pitcher, m.target, m.anchor_date)}
    c["ay"] = [key[(int(p), int(t))] for p, t in zip(c.pitcher, c.target)]
    return c
print("=== INJURED anchor-year distribution (temporal test = anchor_year>=2022) ===")
for name, fcsv, mcsv in [("SNAPSHOT", ROOT/"data"/"final_df.csv", SCRATCH/"cohort_meta_snapshot.csv"),
                         ("LIVE", ROOT/"data"/"final_df_live.csv", SCRATCH/"cohort_meta_live.csv")]:
    c = anchored(fcsv, mcsv)
    inj = c[c.target == 1]
    print(f"{name}: injured total={len(inj)}  injured TEST(anchor>=2022)={int((inj.ay>=2022).sum())}")
    print("   injured by anchor year:", dict(inj.ay.value_counts().sort_index()))
print()
print("=== FULL temporal TEST set (anchor>=2022), by class ===")
for name, fcsv, mcsv in [("SNAPSHOT", ROOT/"data"/"final_df.csv", SCRATCH/"cohort_meta_snapshot.csv"),
                         ("LIVE", ROOT/"data"/"final_df_live.csv", SCRATCH/"cohort_meta_live.csv")]:
    c = anchored(fcsv, mcsv)
    t = c[c.ay >= 2022]
    print(f"{name}: test total={len(t)}  injured={int((t.target==1).sum())}  normal={int((t.target==0).sum())}"
          f"  (train/valid pool anchor<2022 = {int((c.ay<2022).sum())})")
