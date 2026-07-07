"""Phase 2 data building — parameterized variant of prep_classification.

Builds (X, y, meta) arrays from a final_df with the preprocessing axes under
study in Phase 2. Defaults reproduce the G1 pipeline exactly; each option
isolates one finding from phase2_findings.md:
  drop_time_channel  — F=103 -> 102 (feature-count-mismatch)
  interp='linear'    — true linear interpolation within a pitcher's observed
                       span instead of step-fill; out-of-span tails stay
                       step-filled so the tail axis is isolated (A3 vs A6)
  outlier_train_pitchers — fit the 4.7-sigma thresholds on these (signed)
                       pitchers' rows only (train-only stats, P2-4)
  window=(lo, hi)    — bin filter, default (99, 1220) = upstream 100..1215

Returns per-sample metadata needed by the split/analysis stages:
  signed_id (pitcher key incl. the target-negation trick), real_id (abs),
  max_real_bin (oldest 5-day bin with a real observation — the fill tail
  starts beyond it).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AUTHORS_CSV = ROOT / "TJS_Prediction" / "Raw_data" / "final_df.csv"


@dataclass(frozen=True)
class SampleMeta:
    signed_ids: np.ndarray   # groupby key order, aligned with X rows
    real_ids: np.ndarray
    max_real_bin: np.ndarray


def _outlier_47sigma(df: pd.DataFrame, stats_rows: pd.DataFrame) -> pd.DataFrame:
    """Upstream Prepfortrain L8-19; stats optionally from a row subset.
    Keeps the upstream quirk of starting at diff_ax_CU."""
    start_index = df.columns.get_loc('diff_ax_CU')
    for col in df.columns[start_index:]:
        median = stats_rows[col].median()
        std_dev = stats_rows[col].std()
        a = 4.7
        df[col] = np.where(
            (df[col] < median - a * std_dev) | (df[col] > median + a * std_dev),
            np.nan, df[col])
    return df


def _fill_group(group: pd.DataFrame, interp: str) -> pd.DataFrame:
    """Upstream fill_missing_values with an optional true-linear mode.
    'step' = upstream (bin-mean -> reindex -> ffill/bfill); 'linear' = linear
    interpolation between observed bins within the span, step-fill outside."""
    numeric = group.select_dtypes(include=[np.number])
    non_numeric = group.select_dtypes(exclude=[np.number])

    numeric = numeric.groupby('new_before_tj_group').mean().reset_index()
    numeric = numeric.set_index('new_before_tj_group').reindex(range(0, 1311, 5))
    if interp == 'linear':
        numeric = numeric.interpolate(method='linear', limit_area='inside')
        numeric = numeric.ffill().bfill()
    else:
        numeric = numeric.ffill().bfill()
    numeric = numeric.reset_index().interpolate(method='linear')
    if 'new_before_tj_group' not in numeric.columns:
        numeric = numeric.rename(columns={'index': 'new_before_tj_group'})

    for col in non_numeric.columns:
        if col != 'new_before_tj_group':
            numeric[col] = non_numeric[col].iloc[0]
    return numeric


def build_arrays(final_csv: Path = AUTHORS_CSV, *,
                 drop_time_channel: bool = False,
                 interp: str = 'step',
                 outlier_train_pitchers: set | None = None,
                 ) -> tuple[np.ndarray, np.ndarray, SampleMeta, list[str]]:
    data = pd.read_csv(final_csv)

    if outlier_train_pitchers is None:
        stats_rows = data
    else:
        signed = np.where(data['target'] == 1, -data['pitcher'], data['pitcher'])
        stats_rows = data[pd.Series(signed).isin(outlier_train_pitchers).values]
    data = _outlier_47sigma(data, stats_rows)
    data.reset_index(drop=True, inplace=True)
    data['new_before_tj_group'] = (data['new_before_tj'] // 5) * 5

    # oldest real observed bin per sample (before any grid fill)
    max_real = (data.groupby(['pitcher', 'target'])['new_before_tj_group']
                .max().rename('max_real_bin'))

    filled = [_fill_group(g, interp)
              for _, g in data.groupby(['player_name', 'pitcher', 'target'])]
    grouped = pd.concat(filled, ignore_index=True)
    # a pitcher whose entire column was NaN'd by the outlier step cannot be
    # step-filled — fall back to the global column mean (same philosophy as
    # the extraction stage's all-NaN fallback; no-op when nothing is NaN)
    num_cols = grouped.select_dtypes(include=[np.number]).columns
    grouped[num_cols] = grouped[num_cols].fillna(grouped[num_cols].mean())

    grouped = grouped[(grouped['new_before_tj_group'] < 1220)
                      & (grouped['new_before_tj_group'] > 99)]

    to_drop = [c for c in ['new_before_tj', 'player_name', 'height', 'weight', 'bmi']
               if c in grouped.columns]
    df = grouped.drop(columns=to_drop)
    df = df.sort_values(by=['target', 'pitcher', 'new_before_tj_group'], ascending=False)
    df.reset_index(drop=True, inplace=True)
    df.loc[df['target'] == 1, 'pitcher'] *= -1

    if 'diff_estimated_ba_using_speedangle_CH' in df.columns:
        df = df.drop(
            df.loc[:, 'diff_estimated_ba_using_speedangle_CH':'diff_launch_speed_SL'].columns,
            axis=1)

    X = df.drop(columns=['target'])
    y = df[['pitcher', 'target']]
    feature_columns = [c for c in X.columns if c != 'pitcher']
    if drop_time_channel:
        feature_columns = [c for c in feature_columns if c != 'new_before_tj_group']

    X_list, y_list, id_list = [], [], []
    for pid, group in X.groupby('pitcher'):
        X_list.append(group[feature_columns].values)
        id_list.append(pid)
    for _, group in y.groupby('pitcher'):
        y_list.append(group['target'].values[0])

    signed_ids = np.array(id_list)
    real_ids = np.abs(signed_ids)
    targets = np.array(y_list)
    # max_real is keyed by (raw pitcher, target); signed id maps back to that
    raw_ids = np.where(targets == 1, -signed_ids, signed_ids)
    mrb = np.array([max_real.loc[(rp, t)] for rp, t in zip(raw_ids, targets)])

    meta = SampleMeta(signed_ids=signed_ids, real_ids=real_ids, max_real_bin=mrb)
    return np.array(X_list), targets, meta, feature_columns
