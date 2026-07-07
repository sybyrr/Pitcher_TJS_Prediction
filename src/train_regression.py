"""Phase 1 faithful-reproduction training — regression (days until surgery).

Port of TJS_Prediction/Regression/regression_train.py + prepforreg.preprocess
on the authors' final_df.csv, bug-compatible: row-level split (player leakage
kept as upstream), same hyperparameters, seeds 102..1002, Gradient SHAP with
the same per-sample loop. Uses upstream prepforreg.data_split verbatim and
src/regression_cnn_fixed.py (upstream CNN, checkpoint path fixed).

Evaluation reproduces visualization.plot_regression_performance numerically:
R^2 and per-100-day-interval RMSE (rmses[0] = the paper's "100-Day RMSE").

Run: .venv\\Scripts\\python.exe src\\train_regression.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "TJS_Prediction" / "Regression"))
sys.path.insert(0, str(ROOT / "src"))

from prepforreg import data_split, drop_columns, replace_outliers_47sigma_with_nan_median
from regression_cnn_fixed import CHECKPOINT_DIR, CNN

AUTHORS_CSV = ROOT / "TJS_Prediction" / "Raw_data" / "final_df.csv"  # Phase 1: authors' data
RESULTS_CSV = ROOT / "results" / "regression_results.csv"
TOP10_CSV = ROOT / "results" / "regression_top10_features.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = range(102, 1003, 100)
# --no-shap: train/evaluate only. SHAP can run later from the saved per-seed
# checkpoints (data/checkpoints/) without retraining.
# --shap-only: skip training, load the saved checkpoints, run SHAP only
# (results CSV is not re-appended; data_split(seed) reproduces the same splits).
RUN_SHAP = '--no-shap' not in sys.argv
SHAP_ONLY = '--shap-only' in sys.argv


def set_seed(seed: int) -> None:
    """Upstream regression_train.py L22-29."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def group_new_before_tj(value: float) -> float:
    """Upstream prepforreg.py L136-149 (>=550 branch unreachable by the <550 filter)."""
    if value < 220:
        return (value // 5) * 5
    elif value < 550:
        return (value // 10) * 10
    return (value // 15) * 15


def preprocess() -> tuple[pd.DataFrame, pd.Series]:
    """Port of prepforreg.preprocess (upstream L102-157). [GUARD]s as in
    prep_classification: hitter/anthropometric columns exist only in the
    authors' CSV, not in our rebuild."""
    data = pd.read_csv(AUTHORS_CSV)
    data = data[data['target'] == 1]
    data = data[data['new_before_tj'] < 550]

    if 'diff_estimated_ba_using_speedangle_CH' in data.columns:  # [GUARD]
        data = data.drop(
            data.loc[:, 'diff_estimated_ba_using_speedangle_CH':'diff_launch_speed_SL'].columns,
            axis=1)

    data_removed = replace_outliers_47sigma_with_nan_median(data)  # upstream (drops NaN rows)
    data_removed.reset_index(drop=True, inplace=True)

    data = data_removed[data_removed['target'] == 1].copy()
    data = data.drop(labels=['target'], axis=1)
    data = drop_columns(data, [c for c in ['player_name', 'pitcher', 'height', 'weight', 'bmi']
                               if c in data.columns])  # [GUARD]

    x = data.drop(labels=['new_before_tj'], axis=1)
    y = data['new_before_tj'].apply(group_new_before_tj)
    return x.reset_index(drop=True), y.reset_index(drop=True)


def evaluate(model: CNN, test_loader: DataLoader) -> tuple[float, list[float], list[int]]:
    """Numerical replica of visualization.plot_regression_performance L13-72."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            all_true.append(yb.cpu().numpy())
            all_pred.append(out.cpu().numpy())
    y_test = np.concatenate(all_true, axis=0).flatten()
    y_pred = np.concatenate(all_pred, axis=0).flatten()
    r2 = r2_score(y_test, y_pred)

    bins = [0, 100, 200, 300, 400, 500]
    bin_idx = np.digitize(y_test, bins)
    rmses, counts = [], []
    for i in range(1, len(bins)):
        in_bin = bin_idx == i
        n = int(in_bin.sum())
        rmses.append(float(np.sqrt(np.mean((y_pred[in_bin] - y_test[in_bin]) ** 2))) if n else 0.0)
        counts.append(n)
    return r2, rmses, counts


def main() -> None:
    batch_size, num_epochs, learning_rate = 16, 200, 0.001  # upstream L35-41
    patience, l2_reg, l1_reg, dropout = 15, 0, 0, 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    X, y = preprocess()
    print(f"X: {X.shape}, y: {y.shape}, y range: {y.min()}..{y.max()}", flush=True)

    all_top10 = pd.DataFrame()
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_valid, X_test, y_train, y_valid, y_test = data_split(X, y, seed)

        loaders = {}
        for name, (xs, ys, sh) in {'train': (X_train, y_train, True),
                                   'valid': (X_valid, y_valid, False),
                                   'test': (X_test, y_test, False)}.items():
            ds = TensorDataset(torch.tensor(xs, dtype=torch.float32).to(device),
                               torch.tensor(ys.values, dtype=torch.float32).to(device))
            loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=sh)

        input_dim = X_train.shape[2]
        start = time.time()
        loss_log = []
        if not SHAP_ONLY:
            model = CNN(dropout, device, input_dim).to(device)
            loss_log, valid_log = model.train_model(
                loaders['train'], loaders['valid'], num_epochs, learning_rate,
                patience, l2_reg, l1_reg, seed)
        train_time = time.time() - start

        best_model = CNN(dropout, device, input_dim).to(device)
        best_model.load_state_dict(torch.load(CHECKPOINT_DIR / f'1dcnn_regression_{seed}.pth'))

        r2, rmses, counts = evaluate(best_model, loaders['test'])
        print(f"[regression seed={seed}] epochs={len(loss_log)} time={train_time:.0f}s "
              f"r2={r2:.3f} rmse_0_100={rmses[0]:.1f}", flush=True)

        if RUN_SHAP:
            # SHAP (upstream L96-114): GradientExplainer, per-sample loop
            best_model.eval()
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            explainer = shap.GradientExplainer(best_model, X_test_tensor)
            shap_values_list = []
            for idx in tqdm(range(X_test_tensor.shape[0]), desc=f"shap seed={seed}"):
                shap_values_list.append(explainer.shap_values(X_test_tensor[idx].unsqueeze(0))[0])
            shap_values = np.concatenate(shap_values_list, axis=0)

            columns = X.columns.tolist()
            importances = np.abs(shap_values).mean(axis=0).flatten()
            top10_idx = np.argsort(importances)[-10:][::-1]
            all_top10 = pd.concat([all_top10, pd.DataFrame({
                'r^2': r2, 'RMSE': rmses[0],
                'Feature': [columns[j] for j in top10_idx],
                'Importance': importances[top10_idx], 'Iteration': seed})], ignore_index=True)
            all_top10.to_csv(TOP10_CSV, index=False)  # save incrementally

        if not SHAP_ONLY:
            row = {'seed': seed, 'r2': r2, 'rmse_0_100': rmses[0],
                   **{f'rmse_{b}_{b+100}': v for b, v in zip(range(0, 500, 100), rmses)},
                   **{f'n_{b}_{b+100}': c for b, c in zip(range(0, 500, 100), counts)},
                   'epochs': len(loss_log), 'train_time_s': round(train_time, 1)}
            pd.DataFrame([row]).to_csv(RESULTS_CSV, mode='a',
                                       header=not RESULTS_CSV.exists(), index=False)

    df = pd.read_csv(RESULTS_CSV)
    print("\n=== summary (mean over seeds) ===", flush=True)
    print(df[['r2', 'rmse_0_100', 'epochs', 'train_time_s']].mean().round(3), flush=True)
    print("paper reference: R^2 0.79 (repo README 0.78), 100-Day RMSE 95.7", flush=True)


if __name__ == "__main__":
    main()
