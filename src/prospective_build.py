"""Phase 2.5 — build the prospective (rolling as-of-date) dataset.

For each decision date t (1st of Apr..Sep, 2017-2023) and each pitcher active
at t, emit a training window using ONLY information available at t:
  features — last 730 days of games, 5-day bins counted back from t
             (bin 0 = most recent), per-game causal diffs (value minus the
             expanding mean of that pitcher's games up to that game, 2016-),
             bin-mean then ffill/bfill on the 146-bin grid
  label    — 1 if a TJ surgery occurs in (t, t+150 days], else 0

Eligibility at t: >=1 game in the 30 days before t and >=20 games since 2016.
No Kang cohort filters (this is a different, deployment-shaped task).

Output: data/prospective/windows.npz
  X (N,146,102) float32, y (N,), pitcher (N,), t (N, 'YYYY-MM-DD')

Run: .venv\\Scripts\\python.exe src\\prospective_build.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from extract import (COLUMNS_102, UPSTREAM_DATA, adjust_for_handedness,
                     load_raw, pivot_by_pitch_type)

OUT = ROOT / "data" / "prospective" / "windows.npz"
WINDOW_DAYS = 730
N_BINS = WINDOW_DAYS // 5          # 146
HORIZON_DAYS = 150
YEARS = range(2017, 2024)
MONTHS = range(4, 10)


def build_game_table() -> pd.DataFrame:
    """Per-game 102-metric table with handedness adjustment and causal diffs."""
    data = load_raw()
    pivot = pivot_by_pitch_type(data)
    del data
    # fixed feature set (COLUMNS_102); columns missing entirely would appear
    # as absent after pivot — create as NaN so the schema is stable
    for c in COLUMNS_102:
        if c not in pivot.columns:
            pivot[c] = np.nan
    pivot = adjust_for_handedness(pivot)
    pivot = pivot.sort_values(['pitcher', 'game_date']).reset_index(drop=True)

    exp = (pivot.groupby('pitcher')[COLUMNS_102]
           .expanding().mean().reset_index(level=0, drop=True))
    for c in COLUMNS_102:
        pivot[f'diff_{c}'] = pivot[c] - exp[c]
    return pivot[['pitcher', 'game_date'] + [f'diff_{c}' for c in COLUMNS_102]]


def surgery_dates() -> dict[int, list[pd.Timestamp]]:
    tj = pd.read_csv(UPSTREAM_DATA / 'list of TJ.csv')
    tj = tj.dropna(subset=['mlbamid'])
    tj['date'] = pd.to_datetime(tj['TJ Surgery Date'], errors='coerce')
    tj = tj.dropna(subset=['date'])
    out: dict[int, list[pd.Timestamp]] = {}
    for pid, g in tj.groupby(tj['mlbamid'].astype(int)):
        out[pid] = sorted(g['date'])
    return out


def main() -> None:
    games = build_game_table()
    diff_cols = [f'diff_{c}' for c in COLUMNS_102]
    tj = surgery_dates()
    print(f"game table: {len(games):,} rows, {games['pitcher'].nunique():,} pitchers",
          flush=True)

    X_list, y_list, pid_list, t_list = [], [], [], []
    grid = pd.Index(range(0, WINDOW_DAYS, 5), name='bin')

    for pid, g in games.groupby('pitcher'):
        dates = g['game_date'].to_numpy()
        surgeries = tj.get(int(pid), [])
        for year in YEARS:
            for month in MONTHS:
                t = pd.Timestamp(year=year, month=month, day=1)
                in_prev30 = ((dates < t.to_datetime64())
                             & (dates >= (t - pd.Timedelta(days=30)).to_datetime64()))
                n_hist = (dates < t.to_datetime64()).sum()
                if not in_prev30.any() or n_hist < 20:
                    continue
                w = g[(g['game_date'] < t) &
                      (g['game_date'] >= t - pd.Timedelta(days=WINDOW_DAYS))]
                bins = ((t - w['game_date']).dt.days // 5) * 5
                mat = (w[diff_cols].groupby(bins.values).mean()
                       .reindex(grid).ffill().bfill())
                if mat.isna().all().all():
                    continue
                X_list.append(mat.to_numpy(dtype=np.float32))
                y_list.append(int(any(t < s <= t + pd.Timedelta(days=HORIZON_DAYS)
                                      for s in surgeries)))
                pid_list.append(int(pid))
                t_list.append(str(t.date()))

    X = np.stack(X_list)
    X = np.nan_to_num(X, nan=0.0)  # all-NaN feature columns (pitch never thrown)
    y = np.array(y_list)
    print(f"windows: {X.shape}, positives: {y.sum()} ({y.mean():.3%})", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y,
                        pitcher=np.array(pid_list), t=np.array(t_list))
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
