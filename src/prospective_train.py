"""Phase 2.5 — train/evaluate on the prospective dataset.

Temporal protocol: train t in 2017-2020, valid t in 2021 (early stopping),
test t in 2022-2023. Train negatives are downsampled to 10:1 (all positives
kept); valid/test are untouched. pos_weight is computed from the training
fold's ratio (not hardcoded 5 — finding threshold-calibration).

Models:
  lr  — logistic regression on a 204-dim summary (per-feature window mean +
        recent-third minus old-third trend), the "simple clinical baseline"
  vit — upstream BinaryClassificationViTModel on the (146,102) window padded
        to 224x224, trained with the fixed loop (deepcopy best), 3 seeds

Metrics (risk-ranking frame): PR-AUC (primary), ROC-AUC, precision@top50,
recall@top100, test base rate. Appends to results/phase2/prospective.csv.

Run: .venv\\Scripts\\python.exe src\\prospective_train.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "TJS_Prediction" / "Classification"))
sys.path.insert(0, str(ROOT / "src"))

from ViT import BinaryClassificationViTModel
from train_loop import train_variant

DATA = ROOT / "data" / "prospective" / "windows.npz"
OUT_CSV = ROOT / "results" / "phase2" / "prospective.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [100, 200, 300]
HP = dict(batch_size=16, num_epochs=200, learning_rate=0.000004,
          patience=20, l2_reg=0.00003, l1_reg=0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rank_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    order = np.argsort(-score)
    top50 = y_true[order[:50]]
    top100 = y_true[order[:100]]
    return {
        'pr_auc': average_precision_score(y_true, score),
        'roc_auc': roc_auc_score(y_true, score),
        'precision_at_50': float(top50.mean()),
        'recall_at_100': float(top100.sum() / max(y_true.sum(), 1)),
        'base_rate': float(y_true.mean()),
        'n_test': len(y_true), 'n_test_pos': int(y_true.sum()),
    }


def append(row: dict) -> None:
    pd.DataFrame([row]).to_csv(OUT_CSV, mode='a', header=not OUT_CSV.exists(), index=False)


def main() -> None:
    z = np.load(DATA, allow_pickle=False)
    X, y = z['X'], z['y']
    years = np.array([int(s[:4]) for s in z['t']])
    tr = years <= 2020
    va = years == 2021
    te = years >= 2022
    print(f"windows train/valid/test: {tr.sum()}/{va.sum()}/{te.sum()}, "
          f"positives: {y[tr].sum()}/{y[va].sum()}/{y[te].sum()}", flush=True)

    # ---- LR baseline on summary features ----
    third = X.shape[1] // 3
    summary = np.concatenate([X.mean(axis=1),
                              X[:, :third].mean(axis=1) - X[:, -third:].mean(axis=1)],
                             axis=1)
    scaler = MinMaxScaler().fit(summary[tr])
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(scaler.transform(summary[tr | va]), y[tr | va])
    score = lr.predict_proba(scaler.transform(summary[te]))[:, 1]
    m = rank_metrics(y[te], score)
    append({'model': 'lr_baseline', 'seed': 0, **m, 'train_time_s': 0})
    print(f"[lr_baseline] pr_auc={m['pr_auc']:.4f} roc_auc={m['roc_auc']:.3f} "
          f"p@50={m['precision_at_50']:.3f} base={m['base_rate']:.4f}", flush=True)

    # ---- ViT ----
    done = set()
    if OUT_CSV.exists():
        d = pd.read_csv(OUT_CSV)
        done = set(d[d['model'] == 'vit']['seed'].tolist())

    sc = MinMaxScaler().fit(X[tr].reshape(-1, X.shape[-1]))
    scale = lambda a: sc.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape)
    Xs = {k: torch.tensor(scale(X[m_]), dtype=torch.float32)
          for k, m_ in {'tr': tr, 'va': va, 'te': te}.items()}
    Xs = {k: F.pad(v, (0, 224 - v.shape[2], 0, 224 - v.shape[1]), "constant", -5)
          for k, v in Xs.items()}
    ys = {k: torch.tensor(y[m_], dtype=torch.float32).unsqueeze(1)
          for k, m_ in {'tr': tr, 'va': va, 'te': te}.items()}

    pos_idx = np.where(y[tr] == 1)[0]
    neg_idx = np.where(y[tr] == 0)[0]

    for seed in SEEDS:
        if seed in done:
            continue
        set_seed(seed)
        keep_neg = np.random.choice(neg_idx, size=min(len(neg_idx), 10 * len(pos_idx)),
                                    replace=False)
        keep = np.concatenate([pos_idx, keep_neg])
        pos_weight = torch.tensor(float(len(keep_neg)) / max(len(pos_idx), 1))

        train_loader = DataLoader(
            TensorDataset(Xs['tr'][keep].to(device), ys['tr'][keep].to(device)),
            batch_size=HP['batch_size'], shuffle=True)
        valid_loader = DataLoader(
            TensorDataset(Xs['va'].to(device), ys['va'].to(device)),
            batch_size=HP['batch_size'], shuffle=False)

        model = BinaryClassificationViTModel(0.5, pos_weight).to(device)
        start = time.time()
        train_variant(model, train_loader, valid_loader, HP['num_epochs'],
                      HP['learning_rate'], HP['patience'], HP['l2_reg'], HP['l1_reg'],
                      deepcopy_best=True)
        train_time = time.time() - start

        model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(Xs['te']), 64):
                out = model(Xs['te'][i:i + 64].to(device))
                scores.append(torch.sigmoid(out).cpu().numpy())
        score = np.concatenate(scores).ravel()
        m = rank_metrics(y[te], score)
        np.savez(ROOT / "results" / "phase2" / "preds" / f"prospective_vit_{seed}.npz",
                 y_true=y[te], y_prob=score, pitcher=z['pitcher'][te], t=z['t'][te])
        append({'model': 'vit', 'seed': seed, **m, 'train_time_s': round(train_time, 1)})
        print(f"[vit seed={seed}] time={train_time:.0f}s pr_auc={m['pr_auc']:.4f} "
              f"roc_auc={m['roc_auc']:.3f} p@50={m['precision_at_50']:.3f}", flush=True)


if __name__ == "__main__":
    main()
