"""Classification preprocessing — faithful port of
TJS_Prediction/Classification/Prepfortrain.py::preprocess (upstream line refs
in comments). Turns data/final_df.csv into X (N, 224, F) / y (N,) arrays.

Deviations from upstream, all marked inline:
  [DEVIATION] explicit group iteration instead of deprecated groupby.apply
              patterns — identical values
  [GUARD]     column drops that upstream hardcodes for its richer private CSV
              (height/weight/bmi, hitter-outcome diffs) are skipped when the
              columns don't exist in our rebuilt final_df

Run: .venv\\Scripts\\python.exe src\\prep_classification.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FINAL_CSV = ROOT / "data" / "final_df.csv"


def replace_outliers_47sigma(df: pd.DataFrame) -> pd.DataFrame:
    """Upstream L8-19: NaN-out values beyond median +/- 4.7*std, from
    'diff_ax_CU' onward (upstream quirk: diff_ax_CH is skipped)."""
    start_index = df.columns.get_loc('diff_ax_CU')
    for col in df.columns[start_index:]:
        median = df[col].median()
        std_dev = df[col].std()
        a = 4.7
        df[col] = np.where(
            (df[col] < median - a * std_dev) | (df[col] > median + a * std_dev),
            np.nan, df[col])
    return df


def fill_missing_values(group: pd.DataFrame) -> pd.DataFrame:
    """Upstream L21-40: mean per 5-day bin, reindex to the 0..1310 grid,
    step-fill (ffill/bfill) then interpolate (a no-op after the fills)."""
    group_numeric = group.select_dtypes(include=[np.number])
    group_non_numeric = group.select_dtypes(exclude=[np.number])

    group_numeric = group_numeric.groupby('new_before_tj_group').mean().reset_index()
    group_numeric = group_numeric.set_index('new_before_tj_group').reindex(range(0, 1311, 5))
    group_numeric = group_numeric.ffill().bfill().reset_index()
    group_numeric = group_numeric.interpolate(method='linear')

    for col in group_non_numeric.columns:
        if col != 'new_before_tj_group':
            group_numeric[col] = group_non_numeric[col].iloc[0]
    return group_numeric


def preprocess() -> tuple[np.ndarray, np.ndarray]:
    """Upstream L77-135."""
    data = pd.read_csv(FINAL_CSV)
    data_removed = replace_outliers_47sigma(data)
    data_removed.reset_index(drop=True, inplace=True)
    data_removed['new_before_tj_group'] = (data_removed['new_before_tj'] // 5) * 5

    # [DEVIATION] upstream L84 relies on groupby.apply keeping the group
    # columns (deprecated); explicit iteration keeps them by construction.
    filled = [fill_missing_values(g)
              for _, g in data_removed.groupby(['player_name', 'pitcher', 'target'])]
    grouped = pd.concat(filled, ignore_index=True)

    # upstream L87-89: bins 100..1215 -> 224 timesteps
    grouped_1290 = grouped[
        (grouped['new_before_tj_group'] < 1220) & (grouped['new_before_tj_group'] > 99)
    ]

    # upstream L92-95 [GUARD]: height/weight/bmi only exist in the authors' CSV
    to_drop = [c for c in ['new_before_tj', 'player_name', 'height', 'weight', 'bmi']
               if c in grouped_1290.columns]
    df_1290 = grouped_1290.drop(columns=to_drop)

    df_1290 = df_1290.sort_values(
        by=['target', 'pitcher', 'new_before_tj_group'], ascending=False)
    df_1290.reset_index(drop=True, inplace=True)
    df_1290.loc[df_1290['target'] == 1, 'pitcher'] *= -1  # unique key per label

    # upstream L103 [GUARD]: hitter-outcome diffs absent from our rebuild
    if 'diff_estimated_ba_using_speedangle_CH' in df_1290.columns:
        df_1290 = df_1290.drop(
            df_1290.loc[:, 'diff_estimated_ba_using_speedangle_CH':'diff_launch_speed_SL'].columns,
            axis=1)

    X = df_1290.drop(columns=['target'])
    y = df_1290[['pitcher', 'target']]
    feature_columns = [c for c in X.columns.tolist() if c != 'pitcher']

    X_list, y_list = [], []
    for _, group in X.groupby('pitcher'):
        X_list.append(group[feature_columns].values)
    for _, group in y.groupby('pitcher'):
        y_list.append(group['target'].values[0])

    X_array = np.array(X_list)
    y_array = np.array(y_list)
    return X_array, y_array


def main() -> None:
    X, y = preprocess()
    n, seq_len, n_feat = X.shape
    print(f"X: {X.shape}  (samples, timesteps, features)")
    print(f"y: {y.shape}, injured={int(y.sum())}, normal={int((y == 0).sum())}")
    print(f"NaN in X: {int(np.isnan(X.astype(float)).sum())}")
    print("Kang reference: (620, 224, F) with injured=101, normal=519; "
          "F = 102 diffs + new_before_tj_group column = 103")


if __name__ == "__main__":
    main()
