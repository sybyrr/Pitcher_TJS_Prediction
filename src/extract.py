"""Rebuild final_df.csv from raw Statcast parquets — faithful port of
TJS_Prediction/Raw_data_extraction/Pybaseball_extract.py (upstream line refs
in stage comments).

The upstream file is a notebook export that cannot run as-is (jupyter magics,
placeholder paths, a lost 2021 date range). This port keeps its logic 1:1
except where marked:
  [DEVIATION] same math, different (non-deprecated / vectorized) pandas idiom
  [GAP-FILL]  logic the public script lost but the paper/design requires

Run: .venv\\Scripts\\python.exe src\\extract.py [--causal]
Input : data/raw/statcast_{2016..2023}.parquet (src/download_statcast.py)
        TJS_Prediction/Raw_data/list of TJ.csv, pitcher_hand.csv (repo snapshot)
Output: data/final_df.csv (--causal: data/final_df_causal.csv, expanding-mean
        diff baseline) + data/cohort_meta.csv (per-sample anchor dates)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
UPSTREAM_DATA = ROOT / "TJS_Prediction" / "Raw_data"

SEASONS = range(2016, 2024)  # upstream L28-37 (comma bug fixed: 2021 included)

# upstream L67-72
SELECT_COLUMNS = [
    'player_name', 'game_type', 'home_team', 'pitch_type', 'game_date',
    'pitcher', 'release_speed', 'release_pos_x', 'release_pos_z', 'game_year',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'effective_speed', 'release_spin_rate', 'release_extension', 'spin_axis',
]

# upstream L81-99 — 17 metrics averaged per pitch type
METRICS = [
    'release_speed', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
    'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'effective_speed', 'release_spin_rate', 'release_extension', 'spin_axis',
]

# upstream L338-362 / L480-504 — the fixed 17x6 = 102 metric_pitchtype columns,
# in upstream order (metrics alphabetical, types CH/CU/FC/FF/SI/SL)
PITCH_TYPES = ['CH', 'CU', 'FC', 'FF', 'SI', 'SL']
METRICS_ALPHA = [
    'ax', 'ay', 'az', 'effective_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'release_extension', 'release_pos_x', 'release_pos_z', 'release_speed',
    'release_spin_rate', 'spin_axis', 'vx0', 'vy0', 'vz0',
]
COLUMNS_102 = [f"{m}_{t}" for m in METRICS_ALPHA for t in PITCH_TYPES]


def load_raw() -> pd.DataFrame:
    """Stages 1-2 (upstream L17-76): load seasons, keep regular season, select columns."""
    frames = [pd.read_parquet(RAW_DIR / f"statcast_{y}.parquet") for y in SEASONS]
    data = pd.concat(frames, ignore_index=True)
    data['game_date'] = pd.to_datetime(data['game_date'])

    data = data[data['game_type'] == 'R']
    data = data[SELECT_COLUMNS]
    # [DEVIATION] pybaseball returns pandas nullable dtypes (Int64/Float64)
    # whose pd.NA poisons boolean masks downstream — the same problem upstream
    # worked around with astype(object). We normalize to numpy dtypes instead
    # (identical values, NaN instead of NA).
    for m in METRICS:
        data[m] = pd.to_numeric(data[m], errors='coerce').astype('float64')
    data['game_year'] = data['game_year'].astype('int64')
    data['pitcher'] = data['pitcher'].astype('int64')
    return data


def pivot_by_pitch_type(data: pd.DataFrame) -> pd.DataFrame:
    """Stage 3 (upstream L101-114): per-game mean of each metric per pitch type."""
    pivot = data.pivot_table(
        index=['player_name', 'home_team', 'game_date', 'pitcher', 'game_year'],
        columns=['pitch_type'],
        values=METRICS,
        aggfunc='mean',
    )
    pivot.reset_index(inplace=True)
    pivot.columns = [
        '_'.join(col).strip() if col[1] else col[0] for col in pivot.columns.values
    ]
    return pivot


def drop_low_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4 (upstream L120-142): drop columns >=90% missing or <=1 unique value."""
    missing = df.isnull().mean()
    nunique = df.nunique(dropna=True)
    drop_col = df.columns[(nunique <= 1) | (missing.round(4) >= 0.90)]
    print(f"quality drop: {len(drop_col)} columns removed "
          f"({[c for c in drop_col if c in COLUMNS_102] or 'none from the 102 set'})")
    return df.drop(columns=drop_col)


def merge_tj_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 5 (upstream L147-223): join TJ registry, derive closest future surgery."""
    tj_list = pd.read_csv(UPSTREAM_DATA / 'list of TJ.csv')
    tj_join = tj_list[['mlbamid', 'TJ Surgery Date', 'Year of TJ']].sort_values(
        by=['mlbamid', 'Year of TJ'])
    tj_join['count'] = tj_join.groupby('mlbamid').cumcount() + 1

    tj_pivot = tj_join.pivot_table(
        index='mlbamid', columns='count',
        values=['TJ Surgery Date', 'Year of TJ'],
        aggfunc=lambda x: ' '.join(map(str, x)),
    )
    tj_pivot.columns = ['_'.join(str(i) for i in col).strip()
                        for col in tj_pivot.columns.values]
    tj_pivot.columns = [col.replace('.0', '') for col in tj_pivot.columns]
    tj_pivot.reset_index(inplace=True)

    result = pd.merge(df, tj_pivot, left_on='pitcher', right_on='mlbamid',
                      how='left').drop(columns='mlbamid')

    for k in (1, 2, 3):
        if f'TJ Surgery Date_{k}' not in result.columns:  # safety if <k surgeries exist
            result[f'TJ Surgery Date_{k}'] = pd.NaT
        # [DEVIATION] errors='coerce' from the start; upstream re-parses with
        # coerce at L187-189 anyway, so coerce is its effective behavior.
        result[f'TJ Surgery Date_{k}'] = pd.to_datetime(
            result[f'TJ Surgery Date_{k}'], errors='coerce')
        result[f'TJ Surgery Year_{k}'] = result[f'TJ Surgery Date_{k}'].dt.year
        result[f'season_before_tj_{k}'] = result[f'TJ Surgery Year_{k}'] - result['game_year']
        result[f'diff_date_{k}'] = (result[f'TJ Surgery Date_{k}'] - result['game_date']).dt.days

    # closest surgery strictly after the game (upstream L195-211, row-apply)
    # [DEVIATION] vectorized: mask dates where diff <= 0, take row-wise min
    dates = result[[f'TJ Surgery Date_{k}' for k in (1, 2, 3)]]
    diffs = result[[f'diff_date_{k}' for k in (1, 2, 3)]]
    result['TJ Surgery Date'] = dates.where(diffs.gt(0).values).min(axis=1)
    result['TJ Surgery Year'] = result['TJ Surgery Date'].dt.year

    # min non-negative season gap (upstream L214-223; np.nan instead of pd.NA)
    sbt = result[[f'season_before_tj_{k}' for k in (1, 2, 3)]]
    result['season_before_tj'] = sbt.where(sbt >= 0).min(axis=1)
    return result


def filter_by_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 6 (upstream L229-267): cohort eligibility per (pitcher, surgery-year) group.

    [DEVIATION] upstream mutates groups inside groupby.apply (deprecated);
    computed here with groupby.transform — identical values.
    """
    df = df.copy()
    df['_sbt_is0'] = df['season_before_tj'].eq(0)
    df['_sbt_pos'] = df['season_before_tj'].where(df['season_before_tj'] > 0)
    g = df.groupby(['pitcher', 'TJ Surgery Year'], dropna=False)

    min_sbt = g['season_before_tj'].transform('min')
    exclude = min_sbt >= 2                       # NaN -> False, as upstream
    has_tj_season = g['_sbt_is0'].transform('any')
    pre_seasons = g['_sbt_pos'].transform('nunique')
    condition = (has_tj_season & (pre_seasons >= 2)) | (~has_tj_season & (pre_seasons >= 3))

    out = df[(condition | df['TJ Surgery Date'].isna()) & ~exclude]
    out = out[(out['season_before_tj'] < 4) | out['season_before_tj'].isna()]
    return out.drop(columns=['_sbt_is0', '_sbt_pos']).reset_index(drop=True)


def adjust_for_handedness(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 7 (upstream L273-291): mirror x-axis metrics and spin axis for LHP.

    [DEVIATION] vectorized column ops instead of row-apply — identical values.
    """
    hand = pd.read_csv(UPSTREAM_DATA / 'pitcher_hand.csv')
    df = pd.merge(df, hand, left_on='pitcher', right_on='player_id',
                  how='left').drop(columns=['player_id'])
    lhp = df['hand'] == 'lhp'

    flip_cols = [c for c in df.columns
                 if 'ax_' in c or 'pfx_x_' in c or 'release_pos_x_' in c or 'vx0_' in c]
    spin_cols = [c for c in df.columns if 'spin_axis_' in c]
    df.loc[lhp, flip_cols] = -df.loc[lhp, flip_cols]
    df.loc[lhp, spin_cols] = 360 - df.loc[lhp, spin_cols]  # NaN stays NaN
    return df


def add_days_before_tj(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 8 (upstream L296-331): days-to-surgery, 9999 sentinel, per-pitcher offset."""
    df = df.copy()
    days = (df['TJ Surgery Date'] - df['game_date']).dt.days
    days = days.fillna(9999)
    df['days_before_tj'] = np.where((days < 0) | (days > 1290), 9999, days)
    df['min_days_before_tj'] = df.groupby('pitcher')['days_before_tj'].transform('min')
    df['adjust_days_before_tj'] = np.where(
        df['days_before_tj'] == 9999, 9999,
        df['days_before_tj'] - df['min_days_before_tj'])

    df = df[df['player_name'] != 'Axford, John']
    return df.sort_values(by=['player_name', 'game_date']).reset_index(drop=True)


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 9 (upstream L336-383): per-pitcher linear interpolation; all-NaN
    pitcher columns take the *current* global mean.

    The global mean is computed on the progressively updated frame, so pitcher
    order matters — upstream order (player_name-sorted uniques) is preserved.
    """
    df = df.copy()
    cols = [c for c in COLUMNS_102 if c in df.columns]
    if len(cols) < len(COLUMNS_102):
        print(f"warning: {len(COLUMNS_102) - len(cols)} of the 102 columns missing")

    blocks = df.groupby('pitcher', sort=False).indices
    for pitcher in tqdm(df['pitcher'].unique(), desc="interpolate", unit="pitcher"):
        rows = df.index[blocks[pitcher]]
        for col in cols:
            block = df.loc[rows, col]
            if block.isnull().all():
                df.loc[rows, col] = df[col].mean()
            else:
                df.loc[rows, col] = block.interpolate(method='linear',
                                                      limit_direction='both')
    return df


def combine_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 10 (upstream L392-434): injured rows + 4-consecutive-season normals."""
    no_tj = df[df['TJ Surgery Year'].isnull()]

    # upstream first filters >=3 consecutive years then keeps the first
    # 4-consecutive run; pitchers without a 4-run contribute zero rows either
    # way, so a single 4-run pass is equivalent.
    def first_four_consecutive(data: pd.DataFrame) -> pd.DataFrame:
        years = sorted(data['game_year'].unique())
        for i in range(len(years) - 3):
            if years[i + 1] == years[i] + 1 and years[i + 2] == years[i] + 2 \
                    and years[i + 3] == years[i] + 3:
                return data[data['game_year'].isin(years[i:i + 4])]
        return data.iloc[0:0]

    normals = []
    for _, group in no_tj.groupby('pitcher'):
        sel = first_four_consecutive(group)
        if not sel.empty:
            normals.append(sel)
    normal_data = pd.concat(normals, ignore_index=True)

    # [GAP-FILL] the public script never builds the time axis for the non-TJ
    # group (its new_before_tj stays NaN, which would NaN-out every normal
    # sample downstream). Paper design (Fig 2): day 0 = last game, counting
    # backwards — the exact analogue of adjust_days_before_tj for the injured.
    last_game = normal_data.groupby('pitcher')['game_date'].transform('max')
    normal_data['new_before_tj'] = (last_game - normal_data['game_date']).dt.days

    injured = df[df['adjust_days_before_tj'] < 1500].copy()
    injured['new_before_tj'] = injured['adjust_days_before_tj']

    merged = pd.concat([injured, normal_data])
    merged = merged[~merged['player_name'].isin(
        ['Rogers, Tyler', 'Hudson, Dakota', 'Clase, Emmanuel'])]
    merged = merged[~merged['pitcher'].isin([458584])]

    # cohort metadata (game_date is dropped in finalize; capture anchors here)
    tgt = np.where(merged['TJ Surgery Year'].notnull(), 1, 0)
    meta = (merged.assign(_t=tgt)
            .groupby(['pitcher', '_t'])['game_date']
            .agg(anchor_date='max', first_date='min', n_rows='count')
            .reset_index().rename(columns={'_t': 'target'}))
    return merged, meta


def finalize(df: pd.DataFrame, causal: bool = False) -> pd.DataFrame:
    """Stages 11-13 (upstream L439-526): drop bookkeeping, target, diff features.

    causal=True replaces the per-(pitcher,target) full-window mean baseline
    with an expanding mean over games in chronological order (new_before_tj
    descending), so every diff uses only information available at that game
    (Phase 2 finding feature-leakage-lookahead)."""
    columns_to_drop = [
        'last_game_date', 'TJ Surgery Date_1', 'TJ Surgery Date_2', 'TJ Surgery Date_3',
        'Year of TJ_1', 'Year of TJ_2', 'Year of TJ_3',
        'TJ Surgery Year_1', 'TJ Surgery Year_2', 'TJ Surgery Year_3',
        'season_before_tj_1', 'season_before_tj_2', 'season_before_tj_3',
        'diff_date_1', 'diff_date_2', 'diff_date_3',
        'season_before_tj', 'min_days_before_tj', 'adjust_days_before_tj',
        'hand', 'days_before_tj', 'home_team', 'age', 'game_date',
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    df['target'] = np.where(df['TJ Surgery Year'].notnull(), 1, 0)
    df = df.sort_values(by=['player_name', 'target', 'new_before_tj'])

    cols_for_diff = [c for c in COLUMNS_102 if c in df.columns]
    if causal:
        # chronological order within a sample = new_before_tj descending
        df = df.sort_values(by=['pitcher', 'target', 'new_before_tj'],
                            ascending=[True, True, False])
        exp = (df.groupby(['pitcher', 'target'])[cols_for_diff]
               .expanding().mean().reset_index(level=[0, 1], drop=True))
        for c in cols_for_diff:
            df[f'diff_{c}'] = df[c] - exp[c]
    else:
        mean_df = df.groupby(['pitcher', 'target'])[cols_for_diff].mean().reset_index()
        mean_df = mean_df.rename(columns={c: f'mean_{c}' for c in cols_for_diff})
        df = pd.merge(df, mean_df, on=['pitcher', 'target'], how='left')
        for c in cols_for_diff:
            df[f'diff_{c}'] = df[c] - df[f'mean_{c}']

    df = df[~df['player_name'].isin(['Sobotka, Chad', 'Williams, Trevor'])]

    final_cols = (['player_name', 'pitcher', 'new_before_tj', 'target'] +
                  [c for c in df.columns if c.startswith('diff_')])
    return df[final_cols].sort_values(by=['player_name', 'new_before_tj'])


def main() -> None:
    causal = '--causal' in sys.argv
    out_csv = ROOT / "data" / ("final_df_causal.csv" if causal else "final_df.csv")
    data = load_raw()
    print(f"raw regular-season pitches: {len(data):,}")

    pivot = pivot_by_pitch_type(data)
    print(f"pivoted per-game rows: {len(pivot):,}, columns: {pivot.shape[1]}")
    del data

    pivot = drop_low_quality_columns(pivot)
    result = merge_tj_labels(pivot)
    result = filter_by_conditions(result)
    print(f"after eligibility filter: {len(result):,} rows, "
          f"{result['pitcher'].nunique():,} pitchers")

    result = adjust_for_handedness(result)
    result = add_days_before_tj(result)
    result = interpolate_missing(result)

    merged, meta = combine_groups(result)
    meta_csv = ROOT / "data" / "cohort_meta.csv"
    meta.to_csv(meta_csv, index=False)
    print(f"cohort meta: {len(meta)} samples -> {meta_csv}")
    final_df = finalize(merged, causal=causal)

    n_pitchers = final_df.groupby(['pitcher', 'target']).ngroups
    n_injured = final_df[final_df['target'] == 1].groupby('pitcher').ngroups
    n_diff = sum(c.startswith('diff_') for c in final_df.columns)
    print(f"final_df: {len(final_df):,} rows, samples={n_pitchers} "
          f"(injured={n_injured}, normal={n_pitchers - n_injured}), diff features={n_diff}")
    print("Kang reference: samples=620 (injured=101, normal=519), diff features=102")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_csv, index=False)
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
