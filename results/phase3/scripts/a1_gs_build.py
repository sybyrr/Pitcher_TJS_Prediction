"""Build corrected game-started flags from the actual first pitch of each side."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent
FLAG_OUT = ROOT / "data" / "prospective" / "gs_flags_v2.parquet"
SANITY_OUT = OUT / "a1_gs_sanity_corrected.csv"
EXCEPTION_OUT = OUT / "a1_gs_exceptions_corrected.csv"
LEGACY_DIFF_OUT = OUT / "a1_gs_v1_v2_diff.csv"
YEARS = range(2016, 2026)


def _exception_rows(
    frame: pd.DataFrame,
    issue: str,
    detail: str,
) -> pd.DataFrame:
    """Return normalized rows for the GS exception audit."""

    if frame.empty:
        return pd.DataFrame(
            columns=["year", "game_pk", "game_date", "issue", "detail"]
        )
    out = frame.copy()
    out["issue"] = issue
    if "detail" not in out.columns or detail:
        out["detail"] = detail
    return out[["year", "game_pk", "game_date", "issue", "detail"]]


def main() -> None:
    """Create unique first-pitch starters, GS flags, and explicit sanity reports."""

    t0 = time.time()
    cols = [
        "game_pk",
        "game_date",
        "pitcher",
        "game_type",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
    ]
    frames: list[pd.DataFrame] = []
    source_offset = 0
    for year in YEARS:
        frame = pd.read_parquet(RAW / f"statcast_{year}.parquet", columns=cols)
        frame = frame.loc[frame["game_type"].eq("R")].copy()
        frame["source_order"] = np.arange(source_offset, source_offset + len(frame))
        source_offset += len(frame)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data["game_date"] = pd.to_datetime(data["game_date"])
    data["year"] = data["game_date"].dt.year.astype("Int64")
    print(f"[t={time.time() - t0:.0f}s] regular-season pitch rows {len(data):,}")

    exceptions: list[pd.DataFrame] = []
    required = ["game_pk", "game_date", "pitcher", "inning_topbot"]
    missing_required = data[required].isna().any(axis=1)
    if missing_required.any():
        miss = (
            data.loc[missing_required, ["year", "game_pk", "game_date"]]
            .drop_duplicates()
            .assign(
                detail=f"{int(missing_required.sum())} pitch rows missing a grouping/starter field"
            )
        )
        exceptions.append(_exception_rows(miss, "missing_required_field", ""))
    data = data.loc[~missing_required].copy()
    data["game_pk"] = data["game_pk"].astype("int64")
    data["pitcher"] = data["pitcher"].astype("int64")

    # A starter is exactly one row per game and batting side: the lexicographically
    # first (at_bat_number, pitch_number) pitch. source_order is only a deterministic
    # tie-breaker for duplicate raw rows and never changes the baseball definition.
    side_keys = ["game_pk", "inning_topbot"]
    ordered = data.sort_values(
        side_keys + ["at_bat_number", "pitch_number", "source_order"],
        kind="stable",
        na_position="last",
    )
    starters = ordered.drop_duplicates(side_keys, keep="first").copy()
    assert not starters.duplicated(side_keys).any(), "starter must be unique per game-side"

    missing_order = (
        data.assign(missing_order=data[["at_bat_number", "pitch_number"]].isna().any(axis=1))
        .groupby(side_keys, sort=False)
        .agg(
            all_order_missing=("missing_order", "all"),
            game_date=("game_date", "min"),
            year=("year", "min"),
        )
        .reset_index()
    )
    missing_order = missing_order.loc[missing_order["all_order_missing"]]
    if not missing_order.empty:
        miss = missing_order.assign(
            detail=lambda x: "all pitch-order fields missing for side=" + x["inning_topbot"].astype(str)
        )
        exceptions.append(_exception_rows(miss, "missing_pitch_order", ""))

    # Report raw ties at the selected first-pitch coordinates. The selected row is
    # still unique; a multi-pitcher tie is the material exception.
    first_coords = starters[side_keys + ["at_bat_number", "pitch_number"]]
    tied = data.merge(first_coords, on=side_keys + ["at_bat_number", "pitch_number"], how="inner")
    tie_summary = (
        tied.groupby(side_keys, sort=False)
        .agg(
            n_raw_rows=("pitcher", "size"),
            n_pitchers=("pitcher", "nunique"),
            game_date=("game_date", "min"),
            year=("year", "min"),
        )
        .reset_index()
    )
    tie_summary = tie_summary.loc[tie_summary["n_raw_rows"] > 1]
    if not tie_summary.empty:
        tie_rows = tie_summary.assign(
            detail=lambda x: (
                "side="
                + x["inning_topbot"].astype(str)
                + "; tied_rows="
                + x["n_raw_rows"].astype(str)
                + "; distinct_pitchers="
                + x["n_pitchers"].astype(str)
            )
        )
        exceptions.append(_exception_rows(tie_rows, "first_pitch_coordinate_tie", ""))

    game_meta = (
        starters.groupby("game_pk", sort=False)
        .agg(
            game_date=("game_date", "min"),
            year=("year", "min"),
            n_sides=("inning_topbot", "nunique"),
            side_values=("inning_topbot", lambda x: "|".join(sorted(map(str, set(x))))),
            n_starter_rows=("pitcher", "size"),
            n_starter_pitchers=("pitcher", "nunique"),
        )
        .reset_index()
    )
    invalid_games = game_meta.loc[
        (game_meta["n_sides"] != 2)
        | (game_meta["n_starter_rows"] != 2)
        | (game_meta["side_values"] != "Bot|Top")
    ].copy()
    if not invalid_games.empty:
        invalid_games["detail"] = (
            "sides="
            + invalid_games["side_values"]
            + "; starter_rows="
            + invalid_games["n_starter_rows"].astype(str)
            + "; distinct_starters="
            + invalid_games["n_starter_pitchers"].astype(str)
        )
        exceptions.append(_exception_rows(invalid_games, "not_exactly_two_game_sides", ""))

    regular_games = data[["game_pk", "game_date", "year"]].drop_duplicates("game_pk")
    sanity = (
        regular_games.groupby("year", sort=True)
        .agg(n_games=("game_pk", "nunique"))
        .join(
            starters.groupby("year", sort=True).size().rename("n_starter_rows"),
            how="left",
        )
        .fillna({"n_starter_rows": 0})
        .reset_index()
    )
    sanity["n_starter_rows"] = sanity["n_starter_rows"].astype(int)
    sanity["expected_two_per_game"] = 2 * sanity["n_games"]
    sanity["difference"] = sanity["n_starter_rows"] - sanity["expected_two_per_game"]
    exact_games = game_meta.assign(exact_two=lambda x: x["n_starter_rows"].eq(2))
    exact_by_year = exact_games.groupby("year")["exact_two"].sum()
    sanity["games_with_exactly_two_starters"] = (
        sanity["year"].map(exact_by_year).fillna(0).astype(int)
    )
    sanity["exception_games"] = (
        sanity["n_games"] - sanity["games_with_exactly_two_starters"]
    )
    sanity["pass_two_per_game"] = sanity["difference"].eq(0)

    starter_counts = (
        starters.groupby(["game_pk", "game_date", "pitcher"], sort=False)
        .size()
        .rename("is_gs")
        .reset_index()
    )
    games = data[["pitcher", "game_date", "game_pk"]].drop_duplicates()
    games = games.merge(starter_counts, on=["game_pk", "game_date", "pitcher"], how="left")
    games["is_gs"] = games["is_gs"].fillna(0).astype("int8")
    gs = (
        games.groupby(["pitcher", "game_date"], sort=False)
        .agg(n_g=("game_pk", "nunique"), n_gs=("is_gs", "sum"))
        .reset_index()
        .sort_values(["pitcher", "game_date"])
        .reset_index(drop=True)
    )

    exception_df = (
        pd.concat(exceptions, ignore_index=True)
        if exceptions
        else pd.DataFrame(columns=["year", "game_pk", "game_date", "issue", "detail"])
    )
    exception_df = exception_df.sort_values(
        ["year", "game_pk", "issue"], na_position="last"
    ).reset_index(drop=True)

    gs.to_parquet(FLAG_OUT, index=False)
    sanity.to_csv(SANITY_OUT, index=False)
    exception_df.to_csv(EXCEPTION_OUT, index=False)
    # Recreate the v1 min-at-bat logic only to identify its false GS rows. It
    # selected every pitcher appearing in the first at-bat, including a reliever
    # who entered mid-at-bat. The corrected first-pitch pitcher is attached.
    first_ab = data.groupby(side_keys, sort=False)["at_bat_number"].transform("min")
    legacy_starters = data.loc[
        data["at_bat_number"].eq(first_ab),
        side_keys + ["game_date", "year", "pitcher", "at_bat_number"],
    ].drop_duplicates(side_keys + ["pitcher"])
    corrected_starters = starters[
        side_keys + ["pitcher", "pitch_number"]
    ].rename(
        columns={
            "pitcher": "corrected_gs_pitcher",
            "pitch_number": "corrected_first_pitch_number",
        }
    )
    legacy_diff = legacy_starters.merge(corrected_starters, on=side_keys, how="left")
    legacy_diff = legacy_diff.loc[
        legacy_diff["pitcher"].ne(legacy_diff["corrected_gs_pitcher"])
    ].rename(columns={"pitcher": "legacy_extra_gs_pitcher"})
    legacy_diff["issue"] = "v1_mid_at_bat_false_GS"
    legacy_diff = legacy_diff[
        [
            "year",
            "game_pk",
            "game_date",
            "inning_topbot",
            "at_bat_number",
            "legacy_extra_gs_pitcher",
            "corrected_gs_pitcher",
            "corrected_first_pitch_number",
            "issue",
        ]
    ].sort_values(["game_date", "game_pk", "inning_topbot"])
    legacy_diff.to_csv(LEGACY_DIFF_OUT, index=False)
    print("yearly two-starter sanity:")
    print(sanity.to_string(index=False))
    print(f"exceptions: {len(exception_df):,} rows across "
          f"{exception_df['game_pk'].nunique(dropna=True):,} games")
    if not exception_df.empty:
        print(exception_df["issue"].value_counts().to_string())
    print(f"GS rows: {len(gs):,}")
    print(f"legacy v1 -> corrected v2 changed rows: {len(legacy_diff):,}")
    print(f"[t={time.time() - t0:.0f}s] wrote {FLAG_OUT.name}, "
          f"{SANITY_OUT.name}, {EXCEPTION_OUT.name}, {LEGACY_DIFF_OUT.name}")


if __name__ == "__main__":
    main()
