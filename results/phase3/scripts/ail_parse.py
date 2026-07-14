"""A-IL audit repair: parse public transactions into as-of IL episodes.

The legacy ``il_episodes.parquet`` is intentionally preserved.  This script
writes ``il_episodes_asof_v2.parquet`` with the first public elbow and
post-procedure disclosure dates so downstream features cannot backdate facts
learned from a later transfer or activation.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
AIL = ROOT / "data" / "ail"
OUTPUT = AIL / "il_episodes_asof_v2.parquet"
GAP_DAYS = 60

RE_PLACE = re.compile(
    r"placed .{1,100}? on the (7|10|15|26|60)-day (disabled|injured) list",
    re.I,
)
RE_TRANSFER = re.compile(
    r"transferred .{1,100}? to the (26|60)-day (disabled|injured) list",
    re.I,
)
RE_ACTIVATE = re.compile(
    r"(activated|reinstated) .{1,100}? from the .{0,50}?(disabled|injured) list",
    re.I,
)
RE_ELBOW_DIRECT = re.compile(r"\belbow\b|\bforearm\b", re.I)
RE_UCL = re.compile(r"\bu\.?c\.?l\.?\b|ulnar collateral", re.I)
RE_NON_ELBOW_UCL = re.compile(r"\bthumb\b|\bfinger\b|\bwrist\b", re.I)
RE_TOMMY_JOHN = re.compile(r"tommy john(?:\s+surger(?:y|ies))?", re.I)
RE_ELBOW_PROCEDURE = re.compile(
    r"(?:\belbow\b|\bu\.?c\.?l\.?\b|ulnar collateral).{0,45}?"
    r"(?:surger(?:y|ies)|reconstruction|repair)",
    re.I,
)
RE_PROCEDURE_ELBOW = re.compile(
    r"(?:surger(?:y|ies)|reconstruction|repair).{0,45}?"
    r"(?:\belbow\b|\bu\.?c\.?l\.?\b|ulnar collateral)",
    re.I,
)
RE_UCL_PROCEDURE = re.compile(
    r"(?:\bu\.?c\.?l\.?\b|ulnar collateral).{0,45}?"
    r"(?:surger(?:y|ies)|reconstruction|repair)|"
    r"(?:surger(?:y|ies)|reconstruction|repair).{0,45}?"
    r"(?:\bu\.?c\.?l\.?\b|ulnar collateral)",
    re.I,
)
RE_FUTURE_PROCEDURE = re.compile(
    r"(?:scheduled|plans?|expected|recommended|considering|candidate|might|could|will)"
    r".{0,35}?(?:tommy john|surger(?:y|ies)|reconstruction|repair)|"
    r"(?:may need|may undergo|to undergo|needs? to undergo|possible).{0,35}?"
    r"(?:tommy john|surger(?:y|ies)|reconstruction|repair)|"
    r"(?:tommy john|surger(?:y|ies)|reconstruction|repair).{0,25}?"
    r"(?:scheduled|planned|expected|recommended|considered|"
    r"will\s+(?:be|occur)|could\s+(?:be|occur)|may\s+be)",
    re.I,
)


def post_procedure_category(description: str) -> str | None:
    """Classify completed elbow procedures while rejecting future intent."""

    if RE_FUTURE_PROCEDURE.search(description):
        return None
    if RE_NON_ELBOW_UCL.search(description) and not RE_ELBOW_DIRECT.search(description):
        return None
    if RE_TOMMY_JOHN.search(description):
        return "tommy_john"
    if RE_UCL_PROCEDURE.search(description):
        return "ucl_procedure"
    if RE_ELBOW_PROCEDURE.search(description) or RE_PROCEDURE_ELBOW.search(description):
        return "other_elbow_procedure"
    return None


def is_post_tjs_disclosure(description: str) -> bool:
    """Identify explicit completed Tommy John/UCL procedures."""

    return post_procedure_category(description) in {"tommy_john", "ucl_procedure"}


def is_post_elbow_procedure_disclosure(description: str) -> bool:
    """Identify any completed elbow procedure, including non-UCL surgery."""

    return post_procedure_category(description) is not None


def is_elbow_disclosure(description: str) -> bool:
    """Identify elbow/forearm/UCL text while excluding non-elbow UCL uses."""

    if RE_ELBOW_DIRECT.search(description):
        return True
    if RE_UCL.search(description) and not RE_NON_ELBOW_UCL.search(description):
        return True
    return bool(RE_TOMMY_JOHN.search(description))


def classify_action(description: str) -> str | None:
    """Map a transaction description to place, transfer, or activate."""

    if RE_PLACE.search(description):
        return "place"
    if RE_TRANSFER.search(description):
        return "transfer"
    if RE_ACTIVATE.search(description):
        return "activate"
    return None


def run_ontology_self_tests() -> None:
    """Lock expected true/false examples for the public-text ontology."""

    procedure_true = [
        "Recovering from May 2015 Tommy John surgery.",
        "Recovering May 2015 Tommy John surgery.",
        "Recovery of Tommy John surgery.",
        "Tommy John surgery recovery.",
        "Rehab from Tommy John surgery.",
        "April 2016 Tommy John surgery.",
        "Right elbow UCL reconstruction.",
        "Right UCL surgery rehab.",
        "Recovery from right elbow surgery.",
    ]
    procedure_false = [
        "Right UCL sprain.",
        "High grade UCL tear in right elbow.",
        "Tommy John surgery recommended.",
        "Scheduled to undergo Tommy John surgery.",
        "May need right elbow surgery.",
        "Right elbow surgery will be performed.",
        "Possible Tommy John surgery.",
        "Right elbow inflammation.",
        "Right thumb UCL repair.",
    ]
    elbow_true = [
        "Right elbow inflammation.",
        "Left forearm strain.",
        "Right UCL sprain.",
        "Tommy John surgery recovery.",
    ]
    elbow_false = [
        "Right shoulder inflammation.",
        "Right thumb UCL repair.",
        "Left wrist UCL sprain.",
    ]
    assert all(is_post_elbow_procedure_disclosure(x) for x in procedure_true)
    assert not any(is_post_elbow_procedure_disclosure(x) for x in procedure_false)
    assert is_post_tjs_disclosure("Tommy John surgery recovery.")
    assert is_post_tjs_disclosure("Right elbow UCL reconstruction.")
    assert not is_post_tjs_disclosure("Recovery from right elbow debridement surgery.")
    assert all(is_elbow_disclosure(x) for x in elbow_true)
    assert not any(is_elbow_disclosure(x) for x in elbow_false)
    print(
        "SELF-TEST ontology: "
        f"procedure true={len(procedure_true)}, procedure false={len(procedure_false)}, "
        f"elbow true={len(elbow_true)}, elbow false={len(elbow_false)} PASS"
    )


def load_events() -> pd.DataFrame:
    """Load MLB-team IL actions using transaction date as information time."""

    mlb_ids = set(json.loads((AIL / "mlb_team_ids.json").read_text(encoding="utf-8")))
    rows: list[dict[str, Any]] = []
    raw_order = 0
    for year in range(2016, 2025):
        source = AIL / f"transactions_{year}.json"
        transactions = json.loads(source.read_text(encoding="utf-8"))
        for record in transactions:
            description = record.get("description") or ""
            pid = (record.get("person") or {}).get("id")
            team = (record.get("toTeam") or {}).get("id")
            action = classify_action(description)
            if pid is None or team not in mlb_ids or action is None:
                continue
            procedure_category = post_procedure_category(description)
            post_tjs = procedure_category in {"tommy_john", "ucl_procedure"}
            post_procedure = procedure_category is not None
            rows.append(
                {
                    "pid": int(pid),
                    "date": record["date"],
                    "kind": action,
                    "desc": description,
                    "elbow_signal": is_elbow_disclosure(description) or post_procedure,
                    "post_tjs_signal": post_tjs,
                    "post_procedure_signal": post_procedure,
                    "post_procedure_category": procedure_category,
                    "raw_order": raw_order,
                }
            )
            raw_order += 1
    events = pd.DataFrame(rows)
    events["date"] = pd.to_datetime(events["date"])
    events = events.drop_duplicates(subset=["pid", "date", "kind", "desc"])
    kind_order = events["kind"].map({"place": 0, "transfer": 1, "activate": 2})
    events = (
        events.assign(kind_order=kind_order)
        .sort_values(["pid", "date", "kind_order", "raw_order"], kind="stable")
        .drop(columns=["kind_order"])
        .reset_index(drop=True)
    )
    return events


def _new_episode(row: pd.Series, origin_kind: str) -> dict[str, Any]:
    elbow_date = row["date"] if bool(row["elbow_signal"]) else pd.NaT
    post_date = row["date"] if bool(row["post_tjs_signal"]) else pd.NaT
    procedure_date = row["date"] if bool(row["post_procedure_signal"]) else pd.NaT
    return {
        "pid": int(row["pid"]),
        "start": row["date"],
        "origin_kind": origin_kind,
        "last_event_date": row["date"],
        "end": pd.NaT,
        "closure_reason": None,
        "elbow_disclosure_date": elbow_date,
        "post_tjs_disclosure_date": post_date,
        "post_procedure_disclosure_date": procedure_date,
        "elbow_disclosure_desc": row["desc"] if pd.notna(elbow_date) else None,
        "post_tjs_disclosure_desc": row["desc"] if pd.notna(post_date) else None,
        "post_procedure_disclosure_desc": (
            row["desc"] if pd.notna(procedure_date) else None
        ),
        "post_procedure_category": (
            row["post_procedure_category"] if pd.notna(procedure_date) else None
        ),
        "desc0": row["desc"],
        "n_events": 1,
    }


def _add_event(episode: dict[str, Any], row: pd.Series) -> None:
    episode["last_event_date"] = row["date"]
    episode["n_events"] += 1
    if bool(row["elbow_signal"]) and pd.isna(episode["elbow_disclosure_date"]):
        episode["elbow_disclosure_date"] = row["date"]
        episode["elbow_disclosure_desc"] = row["desc"]
    if bool(row["post_tjs_signal"]) and pd.isna(episode["post_tjs_disclosure_date"]):
        episode["post_tjs_disclosure_date"] = row["date"]
        episode["post_tjs_disclosure_desc"] = row["desc"]
    if bool(row["post_procedure_signal"]) and pd.isna(
        episode["post_procedure_disclosure_date"]
    ):
        episode["post_procedure_disclosure_date"] = row["date"]
        episode["post_procedure_disclosure_desc"] = row["desc"]
        episode["post_procedure_category"] = row["post_procedure_category"]


def build_episodes(
    events: pd.DataFrame,
    gap_days: int = GAP_DAYS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build episodes with one gap rule applied before every action kind."""

    gap = pd.Timedelta(days=gap_days)
    episodes: list[dict[str, Any]] = []
    audit = {
        "gap_closures": 0,
        "activation_closures": 0,
        "right_censored": 0,
        "transfer_origin": 0,
        "orphan_activation": 0,
    }
    for _, group in events.groupby("pid", sort=False):
        open_episode: dict[str, Any] | None = None
        for _, row in group.iterrows():
            if (
                open_episode is not None
                and row["date"] - open_episode["last_event_date"] > gap
            ):
                open_episode["end"] = open_episode["last_event_date"] + gap
                open_episode["closure_reason"] = "gap_timeout"
                episodes.append(open_episode)
                open_episode = None
                audit["gap_closures"] += 1

            if row["kind"] in {"place", "transfer"}:
                if open_episode is None:
                    open_episode = _new_episode(row, str(row["kind"]))
                    if row["kind"] == "transfer":
                        audit["transfer_origin"] += 1
                else:
                    _add_event(open_episode, row)
            elif open_episode is None:
                audit["orphan_activation"] += 1
            else:
                _add_event(open_episode, row)
                open_episode["end"] = row["date"]
                open_episode["closure_reason"] = "activation"
                episodes.append(open_episode)
                open_episode = None
                audit["activation_closures"] += 1

        if open_episode is not None:
            open_episode["closure_reason"] = "right_censored"
            episodes.append(open_episode)
            audit["right_censored"] += 1

    frame = pd.DataFrame(episodes).sort_values(["pid", "start"], kind="stable")
    frame = frame.reset_index(drop=True)
    frame.insert(0, "episode_id", np.arange(len(frame), dtype=np.int64))
    frame["elbow"] = frame["elbow_disclosure_date"].notna()
    frame["post_tjs"] = frame["post_tjs_disclosure_date"].notna()
    frame["post_procedure"] = frame["post_procedure_disclosure_date"].notna()
    frame["new_elbow_final"] = frame["elbow"] & ~frame["post_procedure"]
    frame["elbow_disclosure_lag_days"] = (
        frame["elbow_disclosure_date"] - frame["start"]
    ).dt.days.astype("Int64")
    frame["post_tjs_disclosure_lag_days"] = (
        frame["post_tjs_disclosure_date"] - frame["start"]
    ).dt.days.astype("Int64")
    frame["post_procedure_disclosure_lag_days"] = (
        frame["post_procedure_disclosure_date"] - frame["start"]
    ).dt.days.astype("Int64")
    return frame, audit


def run_episode_self_tests() -> None:
    """Check non-retroactivity and uniform gap/closure behavior."""

    rows = [
        (1, "2020-01-01", "place", "Shoulder strain", False, False, False, None),
        (1, "2020-01-20", "transfer", "Right elbow inflammation", True, False, False, None),
        (1, "2020-01-25", "activate", "Activated from injured list", False, False, False, None),
        (2, "2020-01-01", "place", "Back strain", False, False, False, None),
        (2, "2020-04-05", "transfer", "Right UCL sprain", True, False, False, None),
        (2, "2020-04-10", "activate", "Activated from injured list", False, False, False, None),
        (3, "2020-01-01", "place", "Right elbow injury", True, False, False, None),
        (3, "2020-04-05", "activate", "Activated from injured list", False, False, False, None),
        (4, "2020-05-01", "place", "Right UCL sprain", True, False, False, None),
        (4, "2020-05-10", "transfer", "Tommy John surgery", True, True, True, "tommy_john"),
    ]
    synthetic = pd.DataFrame(
        rows,
        columns=[
            "pid",
            "date",
            "kind",
            "desc",
            "elbow_signal",
            "post_tjs_signal",
            "post_procedure_signal",
            "post_procedure_category",
        ],
    )
    synthetic["date"] = pd.to_datetime(synthetic["date"])
    frame, audit = build_episodes(synthetic)
    first = frame.loc[frame["pid"] == 1].iloc[0]
    assert first["start"] == pd.Timestamp("2020-01-01")
    assert first["elbow_disclosure_date"] == pd.Timestamp("2020-01-20")
    pid2 = frame.loc[frame["pid"] == 2]
    assert len(pid2) == 2
    assert pid2.iloc[0]["closure_reason"] == "gap_timeout"
    assert pid2.iloc[1]["origin_kind"] == "transfer"
    pid3 = frame.loc[frame["pid"] == 3].iloc[0]
    assert pid3["closure_reason"] == "gap_timeout"
    pid4 = frame.loc[frame["pid"] == 4].iloc[0]
    assert pid4["elbow_disclosure_date"] == pd.Timestamp("2020-05-01")
    assert pid4["post_tjs_disclosure_date"] == pd.Timestamp("2020-05-10")
    assert pid4["post_procedure_disclosure_date"] == pd.Timestamp("2020-05-10")
    assert audit["orphan_activation"] == 1
    print("SELF-TEST episodes: disclosure, gap, transfer-origin, activation PASS")


def main() -> None:
    """Run parser gates and write the versioned corrected episode table."""

    started = time.time()
    run_ontology_self_tests()
    run_episode_self_tests()
    events = load_events()
    print(
        f"[t={time.time() - started:.0f}s] IL action rows {len(events):,} "
        f"(place {(events['kind'] == 'place').sum():,} "
        f"transfer {(events['kind'] == 'transfer').sum():,} "
        f"activate {(events['kind'] == 'activate').sum():,})"
    )
    episodes, audit = build_episodes(events)
    delayed_elbow = episodes["elbow_disclosure_lag_days"].fillna(0) > 0
    print(
        f"episodes {len(episodes):,}  elbow {int(episodes['elbow'].sum()):,}  "
        f"post_tjs {int(episodes['post_tjs'].sum()):,}  "
        f"post_procedure {int(episodes['post_procedure'].sum()):,}  "
        f"new_elbow_final {int(episodes['new_elbow_final'].sum()):,}"
    )
    print(
        "closure audit: "
        + ", ".join(f"{key}={value:,}" for key, value in audit.items())
    )
    if delayed_elbow.any():
        lag = episodes.loc[delayed_elbow, "elbow_disclosure_lag_days"].astype(float)
        print(
            f"non-retroactive elbow disclosures n={len(lag):,}, "
            f"median={lag.median():.0f}d, max={lag.max():.0f}d"
        )

    print("\nGATE 1 — yearly episode counts (episode start year):")
    table = (
        episodes.assign(year=episodes["start"].dt.year)
        .groupby("year")
        .agg(
            episodes=("pid", "size"),
            elbow=("elbow", "sum"),
            new_elbow_final=("new_elbow_final", "sum"),
            post_tjs=("post_tjs", "sum"),
            post_procedure=("post_procedure", "sum"),
        )
    )
    print(table.to_string())

    print("\nGATE 2 — 40 random episodes for manual precision review (seed 7):")
    rng = np.random.default_rng(7)
    sample_n = min(40, len(episodes))
    sample = episodes.iloc[rng.choice(len(episodes), size=sample_n, replace=False)]
    for _, row in sample.sort_values("start").iterrows():
        if row["post_procedure"]:
            tag = "POST_PROC"
        elif row["elbow"]:
            tag = "ELBOW"
        else:
            tag = "other"
        disclosure = row["elbow_disclosure_date"]
        disclosed = disclosure.date() if pd.notna(disclosure) else "-"
        print(
            f"  [{tag:9s}] start={row['start'].date()} elbow_disclosed={disclosed} "
            f"pid={row['pid']}: {str(row['desc0'])[:140]}"
        )

    episodes.to_parquet(OUTPUT, index=False)
    print(f"\n[t={time.time() - started:.0f}s] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
