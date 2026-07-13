"""A-IL step 2: parse transactions -> IL episodes + frozen gates (spec v2).

FROZEN before any performance evaluation (this script never touches labels
or model scores):
- Row filter: description matches an IL action AND toTeam.id is an MLB team.
  placement  = "placed ... on the {7,10,15,26,60}-day {disabled|injured} list"
  transfer   = "transferred ... to the {26,60}-day ... list"
  activation = "{activated|reinstated} ... from the ... list"
- Information time = transaction `date` (NEVER the retroactive effectiveDate).
- Episode merge per person: placement opens; transfers/dup placements within
  60d of the last event extend; activation closes; placement after close or
  >60d gap starts a new episode.
- Elbow keyword regex (frozen): elbow|forearm|ucl|ulnar|tommy john
  (case-insensitive). "recovering from ... tommy john" episodes are flagged
  post_tjs and EXCLUDED from new-elbow (kept in any-IL).
- Gates: yearly episode-count sanity table + 40 random episodes printed for
  manual precision check (target >=90%) — reviewed before ail_eval.py runs.
Output: data/ail/il_episodes.parquet + gate printout.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("d:/PAINS/Pitcher_TJS_Prediction")
AIL = ROOT / "data" / "ail"
t0 = time.time()

mlb_ids = set(json.loads((AIL / "mlb_team_ids.json").read_text()))
RE_PLACE = re.compile(r"placed .{1,80}? on the (7|10|15|26|60)-day (disabled|injured) list", re.I)
RE_TRANSFER = re.compile(r"transferred .{1,80}? to the (26|60)-day (disabled|injured) list", re.I)
RE_ACT = re.compile(r"(activated|reinstated) .{1,80}? from the .{0,40}?(disabled|injured) list", re.I)
RE_ELBOW = re.compile(r"elbow|forearm|ucl|ulnar|tommy john", re.I)
RE_POST_TJS = re.compile(r"recovering from .{0,60}?tommy john", re.I)

events = []
for year in range(2016, 2025):
    rows = json.loads((AIL / f"transactions_{year}.json").read_text())
    for r in rows:
        desc = r.get("description") or ""
        pid = (r.get("person") or {}).get("id")
        team = (r.get("toTeam") or {}).get("id")
        if pid is None or team not in mlb_ids:
            continue
        if RE_PLACE.search(desc):
            kind = "place"
        elif RE_TRANSFER.search(desc):
            kind = "transfer"
        elif RE_ACT.search(desc):
            kind = "activate"
        else:
            continue
        events.append(dict(pid=int(pid), date=r["date"], kind=kind, desc=desc,
                           elbow=bool(RE_ELBOW.search(desc)), post_tjs=bool(RE_POST_TJS.search(desc))))
ev = pd.DataFrame(events)
ev["date"] = pd.to_datetime(ev["date"])
ev = ev.drop_duplicates(subset=["pid", "date", "kind", "desc"]).sort_values(["pid", "date"]).reset_index(drop=True)
print(f"[t={time.time()-t0:.0f}s] IL action rows {len(ev):,} "
      f"(place {(ev['kind']=='place').sum():,} transfer {(ev['kind']=='transfer').sum():,} "
      f"activate {(ev['kind']=='activate').sum():,})")

episodes = []
GAP = pd.Timedelta(days=60)
for pid, g in ev.groupby("pid", sort=False):
    open_ep = None
    for _, r in g.iterrows():
        if r["kind"] == "place":
            if open_ep is not None and (r["date"] - open_ep["last"]) <= GAP:
                open_ep["last"] = r["date"]
                open_ep["elbow"] |= bool(r["elbow"]); open_ep["post_tjs"] |= bool(r["post_tjs"])
                open_ep["descs"].append(r["desc"])
            else:
                if open_ep is not None:
                    episodes.append(open_ep)
                open_ep = dict(pid=pid, start=r["date"], last=r["date"], end=pd.NaT,
                               elbow=bool(r["elbow"]), post_tjs=bool(r["post_tjs"]), descs=[r["desc"]])
        elif r["kind"] == "transfer":
            if open_ep is not None:
                open_ep["last"] = r["date"]
                open_ep["elbow"] |= bool(r["elbow"]); open_ep["post_tjs"] |= bool(r["post_tjs"])
                open_ep["descs"].append(r["desc"])
        else:  # activate
            if open_ep is not None:
                open_ep["end"] = r["date"]
                episodes.append(open_ep)
                open_ep = None
    if open_ep is not None:
        episodes.append(open_ep)
ep = pd.DataFrame(episodes)
ep["new_elbow"] = ep["elbow"] & ~ep["post_tjs"]
ep["desc0"] = ep["descs"].str[0]
ep = ep.drop(columns=["descs", "last"])
print(f"episodes {len(ep):,}  elbow {int(ep['elbow'].sum()):,}  "
      f"post_tjs {int(ep['post_tjs'].sum()):,}  new_elbow {int(ep['new_elbow'].sum()):,}")

print("\nGATE 1 — yearly episode counts (placement year):")
tab = (ep.assign(y=ep["start"].dt.year)
         .groupby("y").agg(episodes=("pid", "size"), elbow=("elbow", "sum"),
                           new_elbow=("new_elbow", "sum"), post_tjs=("post_tjs", "sum")))
print(tab.to_string())

print("\nGATE 2 — 40 random episodes for manual precision check (seed 7):")
rng = np.random.default_rng(7)
sample = ep.iloc[rng.choice(len(ep), size=40, replace=False)].sort_values("start")
for _, r in sample.iterrows():
    tag = "ELBOW" if r["new_elbow"] else ("POST_TJS" if r["post_tjs"] else "other")
    print(f"  [{tag:8s}] {r['start'].date()} pid {r['pid']}: {r['desc0'][:150]}")

ep.to_parquet(AIL / "il_episodes.parquet")
print(f"\n[t={time.time()-t0:.0f}s] wrote il_episodes.parquet")
