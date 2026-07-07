"""Phase 2 regression variant runner (1D-CNN, 10 seeds per variant).

Variants (each vs the next isolates one axis):
  r0std_row        upstream row-level split, seeds unified to 100..1000
                   (paired counterpart for r1; Phase 1 G1 used 102..1002)
  r1_grouped       split grouped by pitcher -> honest new-pitcher R^2 (P2-1)
  r1o_grouped_ours grouped split on our rebuilt final_df (comparator for r2)
  r2_causal        grouped split on the causal-diff final_df (P2-7)

Preprocessing is the upstream prepforreg chain (target==1, <550 days, hitter
cols out, 4.7-sigma outliers with row drop, 5/10-day target binning) with the
pitcher id retained for grouping. Training uses regression_cnn_fixed.CNN
(upstream loop; disk checkpoint already yields true best weights).

Run: .venv\\Scripts\\python.exe src\\run_phase2_reg.py [variant ...]
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "TJS_Prediction" / "Regression"))
sys.path.insert(0, str(ROOT / "src"))

from prepforreg import replace_outliers_47sigma_with_nan_median  # upstream
import regression_cnn_fixed
from regression_cnn_fixed import CNN

AUTHORS_CSV = ROOT / "TJS_Prediction" / "Raw_data" / "final_df.csv"
OURS_CSV = ROOT / "data" / "final_df.csv"
CAUSAL_CSV = ROOT / "data" / "final_df_causal.csv"
RESULTS_DIR = ROOT / "results" / "phase2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = range(100, 1001, 100)
HP = dict(batch_size=16, num_epochs=200, learning_rate=0.001,
          patience=15, l2_reg=0, l1_reg=0, dropout=0)

VARIANTS = {
    'r0std_row': dict(csv=AUTHORS_CSV, split='row'),
    'r1_grouped': dict(csv=AUTHORS_CSV, split='grouped'),
    'r1o_grouped_ours': dict(csv=OURS_CSV, split='grouped'),
    'r2_causal': dict(csv=CAUSAL_CSV, split='grouped'),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def group_new_before_tj(value: float) -> float:
    if value < 220:
        return (value // 5) * 5
    elif value < 550:
        return (value // 10) * 10
    return (value // 15) * 15


def preprocess(csv: Path) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Upstream prepforreg.preprocess with pitcher retained for grouping."""
    data = pd.read_csv(csv)
    data = data[data['target'] == 1]
    data = data[data['new_before_tj'] < 550]
    if 'diff_estimated_ba_using_speedangle_CH' in data.columns:
        data = data.drop(
            data.loc[:, 'diff_estimated_ba_using_speedangle_CH':'diff_launch_speed_SL'].columns,
            axis=1)
    data = replace_outliers_47sigma_with_nan_median(data)  # NaN + row drop
    data.reset_index(drop=True, inplace=True)
    pitchers = data['pitcher'].to_numpy()
    data = data.drop(columns=[c for c in
                              ['target', 'player_name', 'pitcher', 'height', 'weight', 'bmi']
                              if c in data.columns])
    x = data.drop(labels=['new_before_tj'], axis=1)
    y = data['new_before_tj'].apply(group_new_before_tj)
    return x.reset_index(drop=True), y.reset_index(drop=True), pitchers


def split_indices(mode: str, y: pd.Series, pitchers: np.ndarray, seed: int):
    idx = np.arange(len(y))
    if mode == 'row':  # upstream prepforreg.data_split (no stratify)
        tr, tmp = train_test_split(idx, test_size=0.4, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.5, random_state=seed)
    elif mode == 'grouped':
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
        tr, tmp = next(g1.split(idx, groups=pitchers))
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        v, t = next(g2.split(tmp, groups=pitchers[tmp]))
        va, te = tmp[v], tmp[t]
    else:
        raise ValueError(mode)
    return tr, va, te


def interval_rmses(y_true: np.ndarray, y_pred: np.ndarray):
    bins = [0, 100, 200, 300, 400, 500]
    bi = np.digitize(y_true, bins)
    rmses, counts = [], []
    for i in range(1, len(bins)):
        m = bi == i
        n = int(m.sum())
        rmses.append(float(np.sqrt(np.mean((y_pred[m] - y_true[m]) ** 2))) if n else 0.0)
        counts.append(n)
    return rmses, counts


def run_variant(name: str, cfg: dict) -> None:
    if not Path(cfg['csv']).exists():
        print(f"[{name}] SKIP — input missing: {cfg['csv']}", flush=True)
        return
    out_csv = RESULTS_DIR / f"{name}.csv"
    done = set(pd.read_csv(out_csv)['seed'].tolist()) if out_csv.exists() else set()

    X, y, pitchers = preprocess(cfg['csv'])
    ckpt_dir = ROOT / "data" / "checkpoints" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    regression_cnn_fixed.CHECKPOINT_DIR = ckpt_dir
    print(f"[{name}] X={X.shape}, pitchers={len(np.unique(pitchers))}", flush=True)

    for seed in SEEDS:
        if seed in done:
            continue
        set_seed(seed)
        tr, va, te = split_indices(cfg['split'], y, pitchers, seed)
        novel = len(set(pitchers[te]) - set(pitchers[tr]))
        scaler = MinMaxScaler()
        Xtr = scaler.fit_transform(X.iloc[tr])[:, np.newaxis, :]
        Xva = scaler.transform(X.iloc[va])[:, np.newaxis, :]
        Xte = scaler.transform(X.iloc[te])[:, np.newaxis, :]

        loaders = {}
        for key, (xs, ys, sh) in {'train': (Xtr, y.iloc[tr], True),
                                  'valid': (Xva, y.iloc[va], False),
                                  'test': (Xte, y.iloc[te], False)}.items():
            ds = TensorDataset(torch.tensor(xs, dtype=torch.float32).to(device),
                               torch.tensor(ys.values, dtype=torch.float32).to(device))
            loaders[key] = DataLoader(ds, batch_size=HP['batch_size'], shuffle=sh)

        model = CNN(HP['dropout'], device, X.shape[1]).to(device)
        start = time.time()
        loss_log, _ = model.train_model(loaders['train'], loaders['valid'],
                                        HP['num_epochs'], HP['learning_rate'],
                                        HP['patience'], HP['l2_reg'], HP['l1_reg'], seed)
        train_time = time.time() - start

        best = CNN(HP['dropout'], device, X.shape[1]).to(device)
        best.load_state_dict(torch.load(ckpt_dir / f'1dcnn_regression_{seed}.pth'))
        best.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loaders['test']:
                preds.append(best(xb.to(device)).cpu().numpy())
        y_pred = np.concatenate(preds).ravel()
        y_true = y.iloc[te].to_numpy()
        r2 = r2_score(y_true, y_pred)
        rmses, counts = interval_rmses(y_true, y_pred)

        row = {'variant': name, 'seed': seed, 'r2': r2, 'rmse_0_100': rmses[0],
               'novel_test_pitchers': novel,
               'n_test': len(te), 'n_test_pitchers': len(np.unique(pitchers[te])),
               'epochs': len(loss_log), 'train_time_s': round(train_time, 1)}
        pd.DataFrame([row]).to_csv(out_csv, mode='a', header=not out_csv.exists(), index=False)
        print(f"[{name} seed={seed}] epochs={len(loss_log)} time={train_time:.0f}s "
              f"r2={r2:.3f} rmse100={rmses[0]:.1f} novel_pitchers={novel}", flush=True)
    print(f"[{name}] DONE", flush=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        run_variant(name, VARIANTS[name])
    print("ALL REG VARIANTS DONE", flush=True)


if __name__ == "__main__":
    main()
