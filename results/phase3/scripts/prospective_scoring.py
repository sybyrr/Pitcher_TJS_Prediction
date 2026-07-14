"""Build as-of MLB windows and score them from the immutable frozen state."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from frozen_state import (
    WINDOW_FEATURES,
    add_q0_policy,
    load_frozen_state,
    write_snapshot_exclusive,
)


ROOT = Path(__file__).resolve().parents[3]
PHASE3 = Path(__file__).resolve().parent.parent
DAY = np.timedelta64(1, "D")


def aggregate_statcast(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Create pitcher-game workload and exact first-pitch starter tables."""

    columns = [
        "pitcher",
        "game_date",
        "release_speed",
        "game_type",
        "game_pk",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
    ]
    raw = pd.read_parquet(raw_path, columns=columns)
    raw = raw.loc[raw["game_type"] == "R"].copy()
    if raw.empty:
        raise ValueError(f"no regular-season pitches in {raw_path}")
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw["pitcher"] = pd.to_numeric(raw["pitcher"], errors="raise").astype("int64")
    raw["release_speed"] = pd.to_numeric(raw["release_speed"], errors="coerce")
    raw["at_bat_number"] = pd.to_numeric(raw["at_bat_number"], errors="coerce")
    raw["pitch_number"] = pd.to_numeric(raw["pitch_number"], errors="coerce")
    order_columns = [
        "game_pk",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
    ]
    if raw[order_columns].isna().any().any():
        raise ValueError("first-pitch ordering columns contain missing values")
    if raw.duplicated(order_columns, keep=False).any():
        raise ValueError("first-pitch ordering coordinates contain an exact tie")

    slim = (
        raw.groupby(["pitcher", "game_date"], sort=False)
        .agg(
            pitch_count=("release_speed", "size"),
            mean_release_speed=("release_speed", "mean"),
        )
        .reset_index()
    )
    slim["game_year"] = slim["game_date"].dt.year.astype("int64")

    first_pitch = (
        raw.sort_values(
            ["game_pk", "inning_topbot", "at_bat_number", "pitch_number"],
            kind="mergesort",
        )
        .drop_duplicates(["game_pk", "inning_topbot"], keep="first")
        [["game_pk", "game_date", "pitcher", "inning_topbot"]]
    )
    games = raw[["pitcher", "game_date", "game_pk"]].drop_duplicates()
    starter_keys = first_pitch[["game_pk", "game_date", "pitcher"]].assign(is_gs=1)
    games = games.merge(
        starter_keys,
        on=["game_pk", "game_date", "pitcher"],
        how="left",
        validate="one_to_one",
    )
    games["is_gs"] = games["is_gs"].fillna(0).astype("int8")
    gs = (
        games.groupby(["pitcher", "game_date"], sort=False)
        .agg(n_g=("game_pk", "nunique"), n_gs=("is_gs", "sum"))
        .reset_index()
    )

    starters_per_game = first_pitch.groupby("game_pk").size()
    sanity = {
        "regular_season_games": int(raw["game_pk"].nunique()),
        "starter_rows": int(len(first_pitch)),
        "games_with_two_starters": int((starters_per_game == 2).sum()),
        "games_with_non_two_starters": int((starters_per_game != 2).sum()),
    }
    if sanity["games_with_non_two_starters"]:
        raise ValueError(
            "regular-season first-pitch reconstruction did not yield exactly "
            "two starters per game"
        )
    return slim, gs, sanity


def build_cohort(
    slim: pd.DataFrame,
    decision_dates: Sequence[pd.Timestamp],
    *,
    min_career_games: int = 20,
    max_days_since_last: int = 365,
) -> pd.DataFrame:
    """Build the frozen risk set using only games strictly before each t."""

    rows: list[tuple[int, pd.Timestamp, int]] = []
    for decision_date in sorted(pd.Timestamp(value).normalize() for value in decision_dates):
        before = slim.loc[slim["game_date"] < decision_date]
        aggregate = before.groupby("pitcher")["game_date"].agg(["count", "max"])
        aggregate["days_since_last"] = (decision_date - aggregate["max"]).dt.days
        eligible = aggregate.loc[
            (aggregate["count"] >= min_career_games)
            & (aggregate["days_since_last"] <= max_days_since_last)
        ]
        rows.extend(
            (int(pitcher), decision_date, int(row["count"]))
            for pitcher, row in eligible.iterrows()
        )
    return pd.DataFrame(rows, columns=["pitcher", "t", "n_career_games"]).sort_values(
        ["t", "pitcher"], ignore_index=True
    )


def _pitch_weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    observed = mask & ~np.isnan(values)
    if not observed.any():
        return np.nan
    weight_sum = weights[observed].sum()
    if weight_sum <= 0:
        return np.nan
    return float((values[observed] * weights[observed]).sum() / weight_sum)


def build_window_features(
    slim: pd.DataFrame,
    gs: pd.DataFrame,
    cohort: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute the frozen nine features and the descriptive GS role share."""

    slim = slim.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    gs = gs.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    by_pitcher = {
        int(pitcher): (
            group["game_date"].values.astype("datetime64[D]"),
            group["pitch_count"].to_numpy(dtype=np.float64),
            group["mean_release_speed"].to_numpy(dtype=np.float64),
            group["game_year"].to_numpy(dtype=np.int64),
        )
        for pitcher, group in slim.groupby("pitcher", sort=False)
    }
    gs_by_pitcher = {
        int(pitcher): (
            group["game_date"].values.astype("datetime64[D]"),
            group["n_g"].to_numpy(dtype=np.float64),
            group["n_gs"].to_numpy(dtype=np.float64),
        )
        for pitcher, group in gs.groupby("pitcher", sort=False)
    }

    feature_values = np.zeros((len(cohort), len(WINDOW_FEATURES)), dtype=np.float64)
    gs_share = np.zeros(len(cohort), dtype=np.float64)
    for row_number, row in enumerate(cohort.itertuples(index=False)):
        pitcher = int(row.pitcher)
        decision_date = np.datetime64(pd.Timestamp(row.t), "D")
        game_dates, pitch_count, speed, game_year = by_pitcher[pitcher]
        before = game_dates < decision_date
        if not before.any():
            raise ValueError(f"pitcher {pitcher} has no game strictly before {row.t}")
        last_30 = before & (game_dates >= decision_date - np.timedelta64(30, "D"))
        last_90 = before & (game_dates >= decision_date - np.timedelta64(90, "D"))
        last_365 = before & (game_dates >= decision_date - np.timedelta64(365, "D"))
        days_since_last = float((decision_date - game_dates[before].max()) / DAY)

        current_year = pd.Timestamp(row.t).year
        velocity_30 = _pitch_weighted_mean(speed, pitch_count, last_30)
        velocity_prior = _pitch_weighted_mean(
            speed, pitch_count, before & (game_year < current_year)
        )
        if np.isnan(velocity_prior):
            velocity_prior = _pitch_weighted_mean(
                speed,
                pitch_count,
                before & (game_dates < decision_date - np.timedelta64(30, "D")),
            )
        if np.isnan(velocity_30) or np.isnan(velocity_prior):
            velocity_trend = 0.0
            velocity_missing = 1.0
        else:
            velocity_trend = velocity_30 - velocity_prior
            velocity_missing = 0.0

        prior_years = game_year[before & (game_year < current_year)]
        if prior_years.size:
            prior_pc_rate = pitch_count[before & (game_year == prior_years.max())].sum() / 183.0
        else:
            prior_pc_rate = 0.0
        games_365 = int(last_365.sum())
        start_share = (
            float((pitch_count[last_365] >= 50).sum()) / games_365
            if games_365
            else 0.0
        )
        chronic = pitch_count[last_90].sum() / 90.0
        acute_deviation = pitch_count[last_30].sum() / 30.0 - chronic
        feature_values[row_number] = (
            chronic,
            acute_deviation,
            days_since_last,
            velocity_trend,
            float(pd.Timestamp(row.t).month),
            start_share,
            prior_pc_rate,
            np.log1p(float(row.n_career_games)),
            velocity_missing,
        )

        role_rows = gs_by_pitcher.get(pitcher)
        if role_rows is not None:
            role_dates, role_games, role_starts = role_rows
            role_mask = (role_dates < decision_date) & (
                role_dates >= decision_date - np.timedelta64(365, "D")
            )
            appearances = role_games[role_mask].sum()
            if appearances > 0:
                gs_share[row_number] = role_starts[role_mask].sum() / appearances

    features = pd.DataFrame(feature_values, columns=WINDOW_FEATURES)
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("computed feature table contains a non-finite value")
    return features, gs_share


def score_decisions(
    state_path: Path,
    slim: pd.DataFrame,
    gs: pd.DataFrame,
    decision_dates: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Build and score decision windows with the canonical q=0 policy."""

    state = load_frozen_state(state_path)
    cohort = build_cohort(slim, decision_dates)
    features, gs_share = build_window_features(slim, gs, cohort)
    risks = state.cumulative_risk(features.to_numpy(dtype=np.float64))
    output = cohort[["pitcher", "t"]].copy()
    for name, values in risks.items():
        output[name] = values
    output["gs_share"] = gs_share
    output["rp_flag"] = (gs_share <= 0.2).astype("int8")
    output = add_q0_policy(output, budget=50)
    for name in WINDOW_FEATURES:
        output[name] = features[name].to_numpy()
    contributions = state.feature_contributions(features.to_numpy(dtype=np.float64))
    for index, name in enumerate(WINDOW_FEATURES):
        output[f"contrib_{name}"] = contributions[:, index]
    return output


def _default_gs_path() -> Path:
    corrected = ROOT / "data/prospective/gs_flags_v2.parquet"
    if corrected.exists():
        return corrected
    return ROOT / "data/prospective/gs_flags_v1.parquet"


def _parse_decision_dates(values: Iterable[str]) -> list[pd.Timestamp]:
    dates = [pd.Timestamp(value).normalize() for value in values]
    if len(set(dates)) != len(dates):
        raise ValueError("decision dates must be unique")
    if any(value.day != 1 or value.month not in range(4, 10) for value in dates):
        raise ValueError("decision dates must be the first day of April through September")
    return sorted(dates)


def main() -> None:
    """Run an immutable prospective scoring snapshot from explicit inputs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-date", action="append", required=True)
    parser.add_argument(
        "--raw-season", type=Path, default=ROOT / "data/raw/statcast_2026.parquet"
    )
    parser.add_argument(
        "--historic-slim",
        type=Path,
        default=ROOT / "data/prospective/slim_games_v4.parquet",
    )
    parser.add_argument("--historic-gs", type=Path, default=_default_gs_path())
    parser.add_argument(
        "--state", type=Path, default=PHASE3 / "frozen_model_state.json"
    )
    parser.add_argument("--output-dir", type=Path, default=PHASE3 / "prospective")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--allow-backfill", action="store_true")
    parser.add_argument("--max-staleness-days", type=int, default=7)
    arguments = parser.parse_args()

    decision_dates = _parse_decision_dates(arguments.decision_date)
    slim_season, gs_season, starter_sanity = aggregate_statcast(arguments.raw_season)
    raw_max_date = pd.Timestamp(slim_season["game_date"].max()).normalize()
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    today = pd.Timestamp(now.date())
    if not arguments.allow_backfill:
        if any(date < today for date in decision_dates):
            raise ValueError("past decision dates require --allow-backfill")
        for date in decision_dates:
            if raw_max_date >= date:
                raise ValueError(
                    f"raw data reaches {raw_max_date.date()}, not strictly before {date.date()}"
                )
            staleness = int((date - raw_max_date).days)
            if staleness > arguments.max_staleness_days:
                raise ValueError(
                    f"raw data is {staleness} days stale for {date.date()}; update it first"
                )

    historic_slim = pd.read_parquet(arguments.historic_slim)
    historic_gs = pd.read_parquet(arguments.historic_gs)
    slim = pd.concat([historic_slim, slim_season], ignore_index=True)
    gs = pd.concat([historic_gs, gs_season], ignore_index=True)
    scores = score_decisions(arguments.state, slim, gs, decision_dates)

    snapshot_id = arguments.snapshot_id or now.strftime("%Y%m%dT%H%M%S%z")
    date_part = "-".join(date.strftime("%Y%m%d") for date in decision_dates)
    output_path = arguments.output_dir / f"scores_{date_part}_ts{snapshot_id}.csv"
    metadata = {
        "generated_at": now.isoformat(),
        "decision_dates": [date.date().isoformat() for date in decision_dates],
        "raw_max_game_date": raw_max_date.date().isoformat(),
        "strict_as_of_rule": "game_date < t",
        "canonical_alert_policy": "q=0, monthly top-50 by P150_raw",
        "backfill": bool(arguments.allow_backfill),
        "starter_sanity": starter_sanity,
        "rows": int(len(scores)),
    }
    output_path, manifest_path = write_snapshot_exclusive(
        scores,
        output_path,
        state_path=arguments.state,
        input_paths=[arguments.raw_season, arguments.historic_slim, arguments.historic_gs],
        metadata=metadata,
    )
    print(f"WROTE {output_path}")
    print(f"WROTE {manifest_path}")


if __name__ == "__main__":
    main()
