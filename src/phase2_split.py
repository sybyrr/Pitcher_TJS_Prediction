"""Phase 2 split modes. All modes keep upstream's scaling contract
(MinMax fit on train only, 60/20/20) and return float32 tensors like
Prepfortrain.data_split, plus the test-set sample indices for analysis.

  stratified — upstream data_split semantics (random, label-stratified)
  grouped    — GroupShuffleSplit by REAL pitcher id (fixes the 8 dual-role
               pitchers straddling splits and, for regression use, the
               row-level leakage)
  temporal   — fixed test = samples anchored (last game) on/after the cutoff;
               remaining samples split into train/valid per seed
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler


def _to_tensors(X, y, idx_train, idx_valid, idx_test):
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(X[idx_train].reshape(-1, X.shape[-1])).reshape(X[idx_train].shape)
    Xva = scaler.transform(X[idx_valid].reshape(-1, X.shape[-1])).reshape(X[idx_valid].shape)
    Xte = scaler.transform(X[idx_test].reshape(-1, X.shape[-1])).reshape(X[idx_test].shape)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return (t(Xtr), t(Xva), t(Xte),
            t(y[idx_train]).unsqueeze(1), t(y[idx_valid]).unsqueeze(1), t(y[idx_test]).unsqueeze(1))


def split(mode: str, X: np.ndarray, y: np.ndarray, seed: int, *,
          real_ids: np.ndarray | None = None,
          anchor_years: np.ndarray | None = None,
          temporal_cutoff_year: int = 2022):
    n = len(y)
    all_idx = np.arange(n)

    if mode == 'stratified':
        idx_train, idx_temp = train_test_split(
            all_idx, test_size=0.4, random_state=seed, stratify=y)
        idx_valid, idx_test = train_test_split(
            idx_temp, test_size=0.5, random_state=seed, stratify=y[idx_temp])

    elif mode == 'grouped':
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
        idx_train, idx_temp = next(gss1.split(all_idx, y, groups=real_ids))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        v, te = next(gss2.split(idx_temp, y[idx_temp], groups=real_ids[idx_temp]))
        idx_valid, idx_test = idx_temp[v], idx_temp[te]

    elif mode == 'temporal':
        test_mask = anchor_years >= temporal_cutoff_year
        idx_test = all_idx[test_mask]
        rest = all_idx[~test_mask]
        idx_train, idx_valid = train_test_split(
            rest, test_size=0.25, random_state=seed, stratify=y[rest])

    else:
        raise ValueError(mode)

    return _to_tensors(X, y, idx_train, idx_valid, idx_test) + (idx_test,)
