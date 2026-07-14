"""Regression tests for immutable frozen-state prospective scoring."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "results/phase3/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from frozen_state import (  # noqa: E402
    FROZEN_STATE_SHA256,
    WINDOW_FEATURES,
    add_q0_policy,
    load_frozen_state,
    write_snapshot_exclusive,
)
from prospective_scoring import (  # noqa: E402
    aggregate_statcast,
    build_cohort,
    build_window_features,
    score_decisions,
)


STATE_PATH = ROOT / "results/phase3/frozen_model_state.json"


class FrozenStateUnitTests(unittest.TestCase):
    def test_canonical_hash_and_schema(self) -> None:
        state = load_frozen_state(STATE_PATH)
        self.assertEqual(state.state_sha256, FROZEN_STATE_SHA256)
        self.assertEqual(state.features[:-1], WINDOW_FEATURES)

    def test_feature_order_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mutated_path = Path(directory) / "mutated.json"
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            payload["features"][0], payload["features"][1] = (
                payload["features"][1],
                payload["features"][0],
            )
            mutated_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature order mismatch"):
                load_frozen_state(mutated_path, expected_sha256=None)

    def test_q0_is_exactly_monthly_top_50(self) -> None:
        table = pd.DataFrame(
            {
                "t": [pd.Timestamp("2026-08-01")] * 60,
                "P150_raw": np.arange(60, dtype=float),
            }
        )
        scored = add_q0_policy(table)
        self.assertEqual(int(scored["alert_q0"].sum()), 50)
        self.assertTrue((scored.loc[scored["alert_q0"] == 1, "rank_H150"] <= 50).all())

    def test_append_only_writer_rejects_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "input.txt"
            input_path.write_text("input", encoding="utf-8")
            output_path = directory_path / "scores.csv"
            table = pd.DataFrame({"pitcher": [1], "P150_raw": [0.1]})
            write_snapshot_exclusive(
                table,
                output_path,
                state_path=STATE_PATH,
                input_paths=[input_path],
                metadata={"test": True},
            )
            with self.assertRaises(FileExistsError):
                write_snapshot_exclusive(
                    table,
                    output_path,
                    state_path=STATE_PATH,
                    input_paths=[input_path],
                    metadata={"test": True},
                )

    def test_game_on_decision_date_is_excluded(self) -> None:
        decision = pd.Timestamp("2026-08-01")
        slim = pd.DataFrame(
            {
                "pitcher": [7, 7],
                "game_date": [decision - pd.Timedelta(days=1), decision],
                "pitch_count": [100, 200],
                "mean_release_speed": [95.0, 110.0],
                "game_year": [2026, 2026],
            }
        )
        gs = pd.DataFrame(
            {
                "pitcher": [7, 7],
                "game_date": [decision - pd.Timedelta(days=1), decision],
                "n_g": [1, 1],
                "n_gs": [1, 1],
            }
        )
        cohort = build_cohort(slim, [decision], min_career_games=1)
        features, gs_share = build_window_features(slim, gs, cohort)
        self.assertEqual(int(cohort.loc[0, "n_career_games"]), 1)
        self.assertAlmostEqual(features.loc[0, "pc_chronic"], 100.0 / 90.0)
        self.assertEqual(float(gs_share[0]), 1.0)

    def test_ambiguous_first_pitch_order_is_rejected(self) -> None:
        fixture = pd.DataFrame(
            {
                "pitcher": [1, 2, 3, 4],
                "game_date": [pd.Timestamp("2026-07-01")] * 4,
                "release_speed": [95.0] * 4,
                "game_type": ["R"] * 4,
                "game_pk": [10] * 4,
                "inning_topbot": ["Top", "Top", "Bot", "Bot"],
                "at_bat_number": [1, 1, 1, 2],
                "pitch_number": [1, 1, 1, 1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ambiguous.parquet"
            fixture.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "exact tie"):
                aggregate_statcast(path)

    def test_future_cli_contains_no_training_or_label_dependency(self) -> None:
        source = (SCRIPTS / "prospective_scoring.py").read_text(encoding="utf-8")
        wrapper = (SCRIPTS / "score_2026_prospective.py").read_text(encoding="utf-8")
        for forbidden in ("LogisticRegression", "StandardScaler", "GroupKFold", "tj_live", ".fit("):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, wrapper)


class FrozenArchiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw_path = ROOT / "data/raw/statcast_2026.parquet"
        historic_slim_path = ROOT / "data/prospective/slim_games_v4.parquet"
        historic_gs_path = ROOT / "data/prospective/gs_flags_v1.parquet"
        cls.slim_season, cls.gs_season, cls.starter_sanity = aggregate_statcast(raw_path)
        cls.slim = pd.concat(
            [pd.read_parquet(historic_slim_path), cls.slim_season], ignore_index=True
        )
        cls.gs = pd.concat(
            [pd.read_parquet(historic_gs_path), cls.gs_season], ignore_index=True
        )

    def test_first_pitch_starter_is_unique_per_side(self) -> None:
        self.assertEqual(self.starter_sanity["games_with_non_two_starters"], 0)
        self.assertEqual(
            self.starter_sanity["starter_rows"],
            2 * self.starter_sanity["regular_season_games"],
        )

    def test_load_only_scores_match_delayed_shadow_archive(self) -> None:
        dates = [pd.Timestamp(2026, month, 1) for month in (4, 5, 6, 7)]
        current = score_decisions(STATE_PATH, self.slim, self.gs, dates)
        archived = pd.read_csv(
            ROOT / "results/phase3/prospective_scores_2026_ts20260713.csv",
            parse_dates=["t"],
        )
        keys = ["pitcher", "t"]
        merged = current.merge(archived, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
        self.assertEqual(len(current), len(archived))
        self.assertEqual(len(merged), len(archived))
        for column in ("P90_raw", "P150_raw", "P90_recal", "P150_recal"):
            np.testing.assert_allclose(
                merged[f"{column}_new"],
                merged[f"{column}_old"],
                rtol=0.0,
                atol=5e-8,
            )
        np.testing.assert_array_equal(merged["rank_H150_new"], merged["rank_H150_old"])


if __name__ == "__main__":
    unittest.main()
