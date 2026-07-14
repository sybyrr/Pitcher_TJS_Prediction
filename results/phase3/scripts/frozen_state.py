"""Load and apply the immutable MLB hazard-model state without fitting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FROZEN_STATE_SHA256 = (
    "e14ba800227a5b65a12ca55114e106e20a4636857ef947d5997b9e496e02fac8"
)
MODEL_FEATURES = (
    "pc_chronic",
    "pc_acute_dev",
    "days_since_last",
    "vel_trend",
    "month",
    "start_share",
    "prior_pc_rate",
    "ncg_log",
    "vt_missing",
    "s",
)
WINDOW_FEATURES = MODEL_FEATURES[:-1]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(payload: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    value = np.asarray(payload.get(key), dtype=np.float64)
    if value.shape != (size,):
        raise ValueError(f"{key} must contain exactly {size} values")
    if not np.isfinite(value).all():
        raise ValueError(f"{key} contains a non-finite value")
    return value


@dataclass(frozen=True)
class FrozenState:
    """Validated numerical state for the frozen discrete-time hazard model."""

    features: tuple[str, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    lr_intercept: float
    lr_coef: np.ndarray
    recal_a: float
    recal_b: float
    state_sha256: str

    def hazards(self, feature_values: np.ndarray, intervals: int = 5) -> np.ndarray:
        """Score interval hazards from raw nine-feature window rows."""

        values = np.asarray(feature_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(WINDOW_FEATURES):
            raise ValueError(
                f"feature_values must have shape (n, {len(WINDOW_FEATURES)})"
            )
        if not np.isfinite(values).all():
            raise ValueError("feature_values contains a non-finite value")
        if intervals < 1 or intervals > 5:
            raise ValueError("intervals must be between 1 and 5")

        hazards = np.empty((len(values), intervals), dtype=np.float64)
        for interval in range(intervals):
            design = np.column_stack(
                [values, np.full(len(values), float(interval), dtype=np.float64)]
            )
            standardized = (design - self.scaler_mean) / self.scaler_scale
            linear = self.lr_intercept + standardized @ self.lr_coef
            hazards[:, interval] = _expit(linear)
        return hazards

    def cumulative_risk(self, feature_values: np.ndarray) -> dict[str, np.ndarray]:
        """Return frozen raw and interval-recalibrated P90/P150 scores."""

        hazards = self.hazards(feature_values, intervals=5)
        logits = _logit(hazards)
        recalibrated = _expit(self.recal_a + self.recal_b * logits)
        return {
            "P90_raw": 1.0 - np.prod(1.0 - hazards[:, :3], axis=1),
            "P150_raw": 1.0 - np.prod(1.0 - hazards[:, :5], axis=1),
            "P90_recal": 1.0 - np.prod(1.0 - recalibrated[:, :3], axis=1),
            "P150_recal": 1.0 - np.prod(1.0 - recalibrated[:, :5], axis=1),
        }

    def feature_contributions(self, feature_values: np.ndarray) -> np.ndarray:
        """Return standardized linear contributions for the nine window features."""

        values = np.asarray(feature_values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(WINDOW_FEATURES):
            raise ValueError(
                f"feature_values must have shape (n, {len(WINDOW_FEATURES)})"
            )
        return (
            (values - self.scaler_mean[:-1])
            / self.scaler_scale[:-1]
            * self.lr_coef[:-1]
        )


def _expit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def load_frozen_state(
    path: Path,
    expected_sha256: str | None = FROZEN_STATE_SHA256,
) -> FrozenState:
    """Load the canonical state and reject hash or schema drift."""

    path = Path(path)
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"frozen state hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen state must be a JSON object")
    features = tuple(str(value) for value in payload.get("features", ()))
    if features != MODEL_FEATURES:
        raise ValueError(
            f"feature order mismatch: expected {MODEL_FEATURES}, got {features}"
        )

    size = len(MODEL_FEATURES)
    scaler_mean = _finite_vector(payload, "scaler_mean", size)
    scaler_scale = _finite_vector(payload, "scaler_scale", size)
    if np.any(scaler_scale <= 0):
        raise ValueError("scaler_scale values must all be positive")
    lr_coef = _finite_vector(payload, "lr_coef", size)
    lr_intercept = float(payload.get("lr_intercept"))
    if not np.isfinite(lr_intercept):
        raise ValueError("lr_intercept must be finite")

    recal = payload.get("recal_interval_logistic")
    if not isinstance(recal, dict):
        raise ValueError("recal_interval_logistic must be a JSON object")
    recal_a = float(recal.get("a"))
    recal_b = float(recal.get("b"))
    if not np.isfinite([recal_a, recal_b]).all():
        raise ValueError("recalibration coefficients must be finite")

    return FrozenState(
        features=features,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        lr_intercept=lr_intercept,
        lr_coef=lr_coef,
        recal_a=recal_a,
        recal_b=recal_b,
        state_sha256=actual_sha256,
    )


def add_q0_policy(scores: pd.DataFrame, budget: int = 50) -> pd.DataFrame:
    """Add deterministic monthly P150 rank and the canonical q=0 alert."""

    required = {"t", "P150_raw"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"score table is missing columns: {sorted(missing)}")
    if budget < 1:
        raise ValueError("budget must be positive")
    output = scores.copy()
    output["rank_H150"] = (
        output.groupby("t", sort=False)["P150_raw"]
        .rank(ascending=False, method="first")
        .astype("int64")
    )
    output["alert_q0"] = (output["rank_H150"] <= budget).astype("int8")
    return output


def write_snapshot_exclusive(
    scores: pd.DataFrame,
    output_path: Path,
    *,
    state_path: Path,
    input_paths: Sequence[Path],
    metadata: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Create an immutable score CSV and hash manifest without overwriting."""

    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"append-only snapshot already exists: {output_path} or {manifest_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_bytes = scores.to_csv(index=False, float_format="%.10g").encode("utf-8")
    created_output = False
    try:
        with output_path.open("xb") as handle:
            handle.write(csv_bytes)
        created_output = True
        manifest = {
            "schema_version": 1,
            "score_file": output_path.name,
            "score_sha256": sha256_file(output_path),
            "state_file": str(Path(state_path).resolve()),
            "state_sha256": sha256_file(Path(state_path)),
            "inputs": [
                {
                    "path": str(Path(path).resolve()),
                    "size_bytes": Path(path).stat().st_size,
                    "sha256": sha256_file(Path(path)),
                }
                for path in input_paths
            ],
            "metadata": dict(metadata),
        }
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception:
        if created_output and not manifest_path.exists():
            output_path.unlink(missing_ok=True)
        raise
    return output_path, manifest_path
