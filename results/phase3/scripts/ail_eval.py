"""A-IL audit repair: evaluate disclosure-time IL features and alert lead.

This is a versioned correction of the legacy A-IL diagnostic.  It reads
``il_episodes_asof_v2.parquet`` and writes new artifacts without overwriting
the original episode or result files.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent.parent
EPISODES_PATH = ROOT / "data" / "ail" / "il_episodes_asof_v2.parquet"
RESULTS_PATH = OUT / "ail_results_asof_v2.csv"
ALERTS_PATH = OUT / "ail_alert_events_asof_v2.csv"
DAY = np.timedelta64(1, "D")
CAP = 1500.0
IL_COLS = ["elbow2y", "dsle_log", "anyil2y"]

EpisodeIndex = dict[int, dict[str, np.ndarray]]


def build_episode_index(episodes: pd.DataFrame) -> EpisodeIndex:
    """Index episode starts and disclosure dates by pitcher/player id."""

    index: EpisodeIndex = {}
    for pid, group in episodes.groupby("pid", sort=False):
        index[int(pid)] = {
            "start": group["start"].values.astype("datetime64[D]"),
            "elbow": group["elbow_disclosure_date"].values.astype("datetime64[D]"),
            "post": group["post_procedure_disclosure_date"].values.astype(
                "datetime64[D]"
            ),
        }
    return index


def elbow_disclosures_asof(
    index: EpisodeIndex,
    pid: int,
    decision_date: np.datetime64,
    blackout_days: int = 0,
) -> np.ndarray:
    """Return new-elbow disclosure dates legally available at decision time."""

    item = index.get(pid)
    if item is None:
        return np.array([], dtype="datetime64[D]")
    t = decision_date.astype("datetime64[D]")
    cut = t - np.timedelta64(blackout_days, "D")
    elbow = item["elbow"]
    post = item["post"]
    elbow_known_and_old_enough = ~np.isnat(elbow) & (elbow < cut)
    post_known_at_t = ~np.isnat(post) & (post < t)
    return np.sort(elbow[elbow_known_and_old_enough & ~post_known_at_t])


def any_il_starts_asof(
    index: EpisodeIndex,
    pid: int,
    decision_date: np.datetime64,
    blackout_days: int = 0,
) -> np.ndarray:
    """Return IL episode starts older than the optional blackout."""

    item = index.get(pid)
    if item is None:
        return np.array([], dtype="datetime64[D]")
    t = decision_date.astype("datetime64[D]")
    cut = t - np.timedelta64(blackout_days, "D")
    starts = item["start"]
    return np.sort(starts[starts < cut])


def run_asof_self_tests() -> None:
    """Check disclosure non-retroactivity, post correction, and blackout."""

    nat = np.datetime64("NaT", "D")
    index: EpisodeIndex = {
        1: {
            "start": np.array([np.datetime64("2020-01-01")]),
            "elbow": np.array([np.datetime64("2020-01-20")]),
            "post": np.array([np.datetime64("2020-03-01")]),
        },
        2: {
            "start": np.array([np.datetime64("2020-01-01")]),
            "elbow": np.array([np.datetime64("2020-01-01")]),
            "post": np.array([nat]),
        },
    }
    assert elbow_disclosures_asof(index, 1, np.datetime64("2020-01-15")).size == 0
    assert elbow_disclosures_asof(index, 1, np.datetime64("2020-02-01")).size == 1
    assert elbow_disclosures_asof(index, 1, np.datetime64("2020-03-01")).size == 1
    assert elbow_disclosures_asof(index, 1, np.datetime64("2020-03-02")).size == 0
    assert (
        elbow_disclosures_asof(index, 1, np.datetime64("2020-02-10"), 30).size
        == 0
    )
    assert elbow_disclosures_asof(index, 2, np.datetime64("2020-02-01")).size == 1
    print("SELF-TEST as-of features: disclosure, post-state, blackout PASS")


def pitch_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Return a pitch-weighted mean over nonmissing observations."""

    valid = mask & ~np.isnan(values)
    if not valid.any():
        return np.nan
    weight_sum = weights[valid].sum()
    return (
        float((values[valid] * weights[valid]).sum() / weight_sum)
        if weight_sum > 0
        else np.nan
    )


def build_resamples(
    pitcher_ids: np.ndarray,
    seed: int = 0,
    n_boot: int = 1000,
) -> list[np.ndarray]:
    """Create shared pitcher-clustered bootstrap row indices."""

    unique_ids = np.unique(pitcher_ids)
    positions = {pid: np.where(pitcher_ids == pid)[0] for pid in unique_ids}
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [positions[pid] for pid in rng.choice(unique_ids, size=len(unique_ids), replace=True)]
        )
        for _ in range(n_boot)
    ]


def paired_metrics(
    labels: np.ndarray,
    baseline: np.ndarray,
    variant: np.ndarray,
    resamples: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired bootstrap delta ROC and delta PR arrays."""

    delta_roc: list[float] = []
    delta_pr: list[float] = []
    for row_index in resamples:
        y_boot = labels[row_index]
        if y_boot.sum() == 0 or y_boot.sum() == len(y_boot):
            continue
        delta_roc.append(
            roc_auc_score(y_boot, variant[row_index])
            - roc_auc_score(y_boot, baseline[row_index])
        )
        delta_pr.append(
            average_precision_score(y_boot, variant[row_index])
            - average_precision_score(y_boot, baseline[row_index])
        )
    return np.asarray(delta_roc), np.asarray(delta_pr)


def interval(values: np.ndarray) -> tuple[float, float]:
    """Return the percentile 95% interval."""

    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def fit_predict(
    features: pd.DataFrame,
    labels: np.ndarray,
    columns: list[str],
    fit_mask: np.ndarray,
    predict_mask: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression]:
    """Fit the frozen A-IL diagnostic LR form and predict a held-out mask."""

    scaler = StandardScaler().fit(features.loc[fit_mask, columns])
    model = LogisticRegression(class_weight="balanced", max_iter=2000)
    model.fit(scaler.transform(features.loc[fit_mask, columns]), labels[fit_mask])
    predictions = model.predict_proba(
        scaler.transform(features.loc[predict_mask, columns])
    )[:, 1]
    return predictions, model


def caught_events(
    scores: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    pitcher_ids: np.ndarray,
    surgery_dates: np.ndarray,
    top_k: int = 50,
) -> tuple[
    dict[tuple[int, pd.Timestamp], bool],
    dict[tuple[int, pd.Timestamp], pd.Timestamp | pd.NaT],
    dict[tuple[int, pd.Timestamp], list[int]],
]:
    """Collapse positive windows to events and retain each first caught alert."""

    alerted = np.zeros(len(labels), dtype=bool)
    for date in np.unique(dates):
        rows = np.where(dates == date)[0]
        ranked = rows[np.argsort(-scores[rows], kind="stable")[: min(top_k, len(rows))]]
        alerted[ranked] = True

    groups: dict[tuple[int, pd.Timestamp], list[int]] = {}
    for row in np.where(labels == 1)[0]:
        key = (int(pitcher_ids[row]), pd.Timestamp(surgery_dates[row]))
        groups.setdefault(key, []).append(int(row))

    caught: dict[tuple[int, pd.Timestamp], bool] = {}
    first_alert: dict[tuple[int, pd.Timestamp], pd.Timestamp | pd.NaT] = {}
    for key, rows in groups.items():
        alert_rows = [row for row in rows if alerted[row]]
        caught[key] = bool(alert_rows)
        first_alert[key] = (
            min(pd.Timestamp(dates[row]) for row in alert_rows)
            if alert_rows
            else pd.NaT
        )
    return caught, first_alert, groups


def main() -> None:
    """Run corrected A-IL evaluation and write versioned result artifacts."""

    started = time.time()
    run_asof_self_tests()
    cohort = pd.read_parquet(ROOT / "data" / "prospective" / "cohort_v4.parquet")
    cohort = cohort.sort_values(["t", "pitcher"]).reset_index(drop=True)
    slim = pd.read_parquet(ROOT / "data" / "prospective" / "slim_games_v4.parquet")
    slim = slim.sort_values(["pitcher", "game_date"]).reset_index(drop=True)
    game_features = pd.read_parquet(
        ROOT / "data" / "prospective" / "game_features_v4.parquet"
    )
    game_features = game_features.sort_values(["pitcher", "game_date"]).reset_index(
        drop=True
    )
    episodes = pd.read_parquet(EPISODES_PATH)
    episode_index = build_episode_index(episodes)

    games_by_pitcher: dict[int, tuple[np.ndarray, ...]] = {}
    for pid, group in slim.groupby("pitcher", sort=False):
        games_by_pitcher[int(pid)] = (
            group["game_date"].values.astype("datetime64[D]"),
            group["pitch_count"].astype("float64").values,
            group["mean_release_speed"].astype("float64").values,
            group["game_year"].values.astype(np.int64),
        )
    role_by_pitcher: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid, group in game_features.groupby("pitcher", sort=False):
        role_by_pitcher[int(pid)] = (
            group["game_date"].values.astype("datetime64[D]"),
            group["total_pitches"].astype("float64").values,
        )

    n_rows = len(cohort)
    pitcher_ids = cohort["pitcher"].values.astype(np.int64)
    decision_dates = cohort["t"].values.astype("datetime64[ns]")
    years = cohort["year"].values.astype(np.int64)
    months = cohort["month"].values.astype(np.float64)
    career_games = cohort["n_career_games"].values.astype(np.float64)
    prior_pc_rate = np.zeros(n_rows)
    velocity_missing = np.zeros(n_rows)
    start_share = np.zeros(n_rows)
    base_rows: list[tuple[float, ...]] = []

    for row in range(n_rows):
        pid = int(pitcher_ids[row])
        decision_date = decision_dates[row].astype("datetime64[D]")
        game_date, pitch_count, speed, game_year = games_by_pitcher[pid]
        before = game_date < decision_date
        last_30 = before & (game_date >= decision_date - np.timedelta64(30, "D"))
        last_90 = before & (game_date >= decision_date - np.timedelta64(90, "D"))
        pc_30 = pitch_count[last_30].sum()
        pc_90 = pitch_count[last_90].sum()
        days_since_last = float((decision_date - game_date[before].max()) / DAY)
        speed_30 = pitch_weighted_mean(speed, pitch_count, last_30)
        year = int(years[row])
        speed_prior = pitch_weighted_mean(
            speed,
            pitch_count,
            before & (game_year < year),
        )
        if np.isnan(speed_prior):
            speed_prior = pitch_weighted_mean(
                speed,
                pitch_count,
                before & (game_date < decision_date - np.timedelta64(30, "D")),
            )
        if np.isnan(speed_30) or np.isnan(speed_prior):
            velocity_trend = 0.0
            velocity_missing[row] = 1.0
        else:
            velocity_trend = speed_30 - speed_prior
        prior_years = game_year[before & (game_year < year)]
        if prior_years.size:
            prior_year_mask = before & (game_year == prior_years.max())
            prior_pc_rate[row] = pitch_count[prior_year_mask].sum() / 183.0
        base_rows.append(
            (
                pc_90 / 90.0,
                pc_30 / 30.0 - pc_90 / 90.0,
                days_since_last,
                velocity_trend,
                months[row],
            )
        )
        role_date, role_pitches = role_by_pitcher[pid]
        role_mask = (role_date < decision_date) & (
            role_date >= decision_date - np.timedelta64(365, "D")
        )
        role_games = int(role_mask.sum())
        start_share[row] = (
            float((role_pitches[role_mask] >= 50).sum()) / role_games
            if role_games > 0
            else 0.0
        )

    features = pd.DataFrame(
        base_rows,
        columns=[
            "pc_chronic",
            "pc_acute_dev",
            "days_since_last",
            "vel_trend",
            "month",
        ],
    )
    features["start_share"] = start_share
    features["prior_pc_rate"] = prior_pc_rate
    features["ncg_log"] = np.log1p(career_games)
    features["vt_missing"] = velocity_missing
    assert (
        np.abs(
            features["days_since_last"].values
            - cohort["dsl"].values.astype(float)
        ).max()
        < 1e-6
    )
    base_columns = list(features.columns)
    print(f"[t={time.time() - started:.0f}s] base features built")

    def il_features(blackout_days: int = 0) -> np.ndarray:
        """Build disclosure-time elbow/any-IL features for every decision row."""

        values = np.zeros((n_rows, 3))
        for row in range(n_rows):
            pid = int(pitcher_ids[row])
            t = decision_dates[row].astype("datetime64[D]")
            elbow_dates = elbow_disclosures_asof(
                episode_index,
                pid,
                t,
                blackout_days,
            )
            if elbow_dates.size:
                values[row, 0] = (
                    elbow_dates >= t - np.timedelta64(730, "D")
                ).sum()
                values[row, 1] = np.log1p(
                    min(float((t - elbow_dates.max()) / DAY), CAP)
                )
            else:
                values[row, 1] = np.log1p(CAP)
            any_starts = any_il_starts_asof(
                episode_index,
                pid,
                t,
                blackout_days,
            )
            if any_starts.size:
                values[row, 2] = (
                    any_starts >= t - np.timedelta64(730, "D")
                ).sum()
        return values

    base_il = il_features(0)
    for column_index, column in enumerate(IL_COLS):
        features[column] = base_il[:, column_index]
    print(
        f"coverage: elbow2y>0 in {(features['elbow2y'] > 0).mean():.2%} "
        f"of windows; anyil2y>0 in {(features['anyil2y'] > 0).mean():.2%}"
    )

    fold = cohort["fold_main"].values
    fit_mask = (fold == "train") | (fold == "valid")
    reliable_end = np.datetime64("2024-12-31")
    test_base = (fold == "test") & (years <= 2024)
    mature = {
        horizon: test_base
        & ((decision_dates + np.timedelta64(horizon, "D")) <= reliable_end)
        for horizon in (90, 150)
    }

    result_rows: list[dict[str, Any]] = []
    result_cache: dict[
        int,
        tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray],
    ] = {}
    print("\nM_il = M_sa + disclosure-time IL features — paired mature test:")
    for horizon in (90, 150):
        mask = mature[horizon]
        labels_all = cohort[f"label_H{horizon}_B0"].values.astype(int)
        labels = labels_all[mask]
        resamples = build_resamples(pitcher_ids[mask])
        base_score, _ = fit_predict(
            features,
            labels_all,
            base_columns,
            fit_mask,
            mask,
        )
        variant_score, model = fit_predict(
            features,
            labels_all,
            base_columns + IL_COLS,
            fit_mask,
            mask,
        )
        delta_roc, delta_pr = paired_metrics(
            labels,
            base_score,
            variant_score,
            resamples,
        )
        roc_low, roc_high = interval(delta_roc)
        pr_low, pr_high = interval(delta_pr)
        result_cache[horizon] = (
            mask,
            labels,
            resamples,
            base_score,
            variant_score,
        )
        result_rows.append(
            {
                "block": "additive",
                "H": horizon,
                "variant": "M_il_asof",
                "n_windows": int(mask.sum()),
                "positive_windows": int(labels.sum()),
                "base_roc": roc_auc_score(labels, base_score),
                "variant_roc": roc_auc_score(labels, variant_score),
                "droc": float(np.median(delta_roc)),
                "droc_lo": roc_low,
                "droc_hi": roc_high,
                "base_pr": average_precision_score(labels, base_score),
                "variant_pr": average_precision_score(labels, variant_score),
                "dpr": float(np.median(delta_pr)),
                "dpr_lo": pr_low,
                "dpr_hi": pr_high,
            }
        )
        coefficients = dict(
            zip(base_columns + IL_COLS, np.round(model.coef_[0], 3), strict=True)
        )
        print(
            f"  H={horizon}: ROC {roc_auc_score(labels, base_score):.4f} -> "
            f"{roc_auc_score(labels, variant_score):.4f}; "
            f"dROC {np.median(delta_roc):+.5f} [{roc_low:+.5f},{roc_high:+.5f}]; "
            f"dPR {np.median(delta_pr):+.5f} [{pr_low:+.5f},{pr_high:+.5f}]"
        )
        print(f"    IL coefs: { {key: coefficients[key] for key in IL_COLS} }")

    print("\nBlackout sensitivity (post state remains as-of t):")
    for blackout in (30, 60, 90):
        blackout_values = il_features(blackout)
        blackout_columns: list[str] = []
        for column_index, column in enumerate(IL_COLS):
            blackout_column = f"{column}_bo{blackout}"
            features[blackout_column] = blackout_values[:, column_index]
            blackout_columns.append(blackout_column)
        for horizon in (90, 150):
            mask, labels, resamples, base_score, _ = result_cache[horizon]
            labels_all = cohort[f"label_H{horizon}_B0"].values.astype(int)
            variant_score, _ = fit_predict(
                features,
                labels_all,
                base_columns + blackout_columns,
                fit_mask,
                mask,
            )
            delta_roc, delta_pr = paired_metrics(
                labels,
                base_score,
                variant_score,
                resamples,
            )
            roc_low, roc_high = interval(delta_roc)
            pr_low, pr_high = interval(delta_pr)
            result_rows.append(
                {
                    "block": f"blackout{blackout}",
                    "H": horizon,
                    "variant": f"M_il_asof_bo{blackout}",
                    "n_windows": int(mask.sum()),
                    "positive_windows": int(labels.sum()),
                    "base_roc": roc_auc_score(labels, base_score),
                    "variant_roc": roc_auc_score(labels, variant_score),
                    "droc": float(np.median(delta_roc)),
                    "droc_lo": roc_low,
                    "droc_hi": roc_high,
                    "base_pr": average_precision_score(labels, base_score),
                    "variant_pr": average_precision_score(labels, variant_score),
                    "dpr": float(np.median(delta_pr)),
                    "dpr_lo": pr_low,
                    "dpr_hi": pr_high,
                }
            )
            print(
                f"  blackout {blackout}d H={horizon}: "
                f"dROC {np.median(delta_roc):+.5f} [{roc_low:+.5f},{roc_high:+.5f}], "
                f"dPR {np.median(delta_pr):+.5f} [{pr_low:+.5f},{pr_high:+.5f}]"
            )

    print("\nEvent-level top-50 and alert-time as-of lead:")
    alert_rows: list[dict[str, Any]] = []
    surgery_dates_all = pd.to_datetime(cohort["next_surgery_date"]).values
    for horizon in (90, 150):
        mask, labels, _, base_score, variant_score = result_cache[horizon]
        dates = decision_dates[mask]
        selected_pitchers = pitcher_ids[mask]
        surgery_dates = surgery_dates_all[mask]
        caught_base, first_base, groups = caught_events(
            base_score,
            labels,
            dates,
            selected_pitchers,
            surgery_dates,
        )
        caught_variant, first_variant, _ = caught_events(
            variant_score,
            labels,
            dates,
            selected_pitchers,
            surgery_dates,
        )
        for key in groups:
            pid, surgery_timestamp = key
            variant_alert = first_variant[key]
            history_dates = (
                elbow_disclosures_asof(
                    episode_index,
                    pid,
                    np.datetime64(variant_alert),
                )
                if pd.notna(variant_alert)
                else np.array([], dtype="datetime64[D]")
            )
            latest_history = history_dates.max() if history_dates.size else np.datetime64("NaT")
            alert_rows.append(
                {
                    "H": horizon,
                    "pitcher": pid,
                    "surgery_date": surgery_timestamp.date(),
                    "caught_base": caught_base[key],
                    "caught_il": caught_variant[key],
                    "new_caught": caught_variant[key] and not caught_base[key],
                    "lost": caught_base[key] and not caught_variant[key],
                    "first_base_alert": first_base[key],
                    "first_il_alert": variant_alert,
                    "elbow_history_asof_il_alert": bool(history_dates.size),
                    "last_elbow_disclosure_asof_il_alert": (
                        pd.Timestamp(latest_history) if history_dates.size else pd.NaT
                    ),
                    "history_disclosure_lead_days": (
                        (surgery_timestamp - pd.Timestamp(latest_history)).days
                        if history_dates.size
                        else np.nan
                    ),
                    "alert_lead_days": (
                        (surgery_timestamp - pd.Timestamp(variant_alert)).days
                        if pd.notna(variant_alert)
                        else np.nan
                    ),
                }
            )

        horizon_rows = [row for row in alert_rows if row["H"] == horizon]
        caught_rows = [row for row in horizon_rows if row["caught_il"]]
        history_rows = [
            row for row in caught_rows if row["elbow_history_asof_il_alert"]
        ]
        history_lead = np.asarray(
            [row["history_disclosure_lead_days"] for row in history_rows],
            dtype=float,
        )
        alert_lead = np.asarray(
            [row["alert_lead_days"] for row in caught_rows],
            dtype=float,
        )
        without_history = sum(
            not row["elbow_history_asof_il_alert"] for row in caught_rows
        )
        print(
            f"  H={horizon}: caught M_sa {sum(caught_base.values())} -> "
            f"M_il {sum(caught_variant.values())}; "
            f"new {sum(row['new_caught'] for row in horizon_rows)}, "
            f"lost {sum(row['lost'] for row in horizon_rows)}, "
            f"without history at caught alert {without_history}/{len(caught_rows)}"
        )
        if history_lead.size:
            print(
                "    history disclosure -> surgery among caught/history: "
                f"median {np.median(history_lead):.0f}d "
                f"[P25 {np.percentile(history_lead, 25):.0f}, "
                f"P75 {np.percentile(history_lead, 75):.0f}], n={len(history_lead)}"
            )
        if alert_lead.size:
            print(
                "    first caught alert -> surgery: "
                f"median {np.median(alert_lead):.0f}d "
                f"[P25 {np.percentile(alert_lead, 25):.0f}, "
                f"P75 {np.percentile(alert_lead, 75):.0f}], n={len(alert_lead)}"
            )

    pd.DataFrame(result_rows).to_csv(RESULTS_PATH, index=False, float_format="%.8f")
    pd.DataFrame(alert_rows).to_csv(ALERTS_PATH, index=False)
    print(
        f"\n[t={time.time() - started:.0f}s] wrote {RESULTS_PATH} and {ALERTS_PATH}"
    )


if __name__ == "__main__":
    main()
