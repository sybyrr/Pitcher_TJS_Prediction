"""Phase 1 faithful-reproduction training — classification.

Runs the five upstream models x 10 seeds on the authors' final_df.csv,
bug-compatible with TJS_Prediction/Classification/classification_train.py:
the model classes, data_split, and every hyperparameter/quirk are upstream's
(ViT train loader shuffle=False, int64 pos_weight, best-weights reload only
for ResNet/ViT). Upstream's five copy-paste train functions are collapsed
into one config table + loop — values identical, upstream line refs below.

Evaluation reproduces Visualization.plot_performance_with_auroc numerically
(full-batch forward -> sigmoid -> 0.5 threshold -> sklearn metrics), minus
the plots. Per-seed metrics append to results/classification_results.csv;
per-seed test probabilities go to results/preds/.

Run: .venv\\Scripts\\python.exe src\\train_classification.py
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
from sklearn.metrics import auc, classification_report, roc_curve
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "TJS_Prediction" / "Classification"))
sys.path.insert(0, str(ROOT / "src"))

from Prepfortrain import data_split          # upstream, verbatim
from ResNet import ResNet50Classifier        # upstream models, verbatim
from ViT import BinaryClassificationViTModel
from Transformer_Encoder import TimeSeriesTransformer
from CNN_LSTM import CNN_LSTM_Model
from LSTM import LSTMModel
import prep_classification as prep

AUTHORS_CSV = ROOT / "TJS_Prediction" / "Raw_data" / "final_df.csv"  # Phase 1: authors' data
RESULTS_CSV = ROOT / "results" / "classification_results.csv"
PREDS_DIR = ROOT / "results" / "preds"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = range(100, 1001, 100)
THRESHOLD = 0.5


def set_seed(seed: int) -> None:
    """Upstream classification_train.py L23-30."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_to_224(x: torch.Tensor) -> torch.Tensor:
    """Upstream L114-115 (ViT/image models only)."""
    return F.pad(x, (0, 224 - x.shape[2], 0, 224 - x.shape[1]), "constant", -5)


# Config per model — values verbatim from upstream train_* functions.
# 'shuffle' is the TRAIN loader flag (ViT False is an upstream quirk, kept).
# 'reload_best' mirrors which functions call model.load_state_dict(best) after
# training (ResNet L95, ViT L170) vs those that don't (Transformer/CNN+LSTM/LSTM).
MODELS = {
    'resnet': dict(  # upstream L41-104
        batch_size=16, num_epochs=300, lr=0.00006, patience=20,
        l2_reg=0, l1_reg=0.000001, shuffle=True, pad=False, reload_best=True,
        build=lambda input_dim: ResNet50Classifier(THRESHOLD, torch.tensor(5))),
    'vit': dict(     # upstream L109-179
        batch_size=16, num_epochs=200, lr=0.000004, patience=20,
        l2_reg=0.00003, l1_reg=0, shuffle=False, pad=True, reload_best=True,
        build=lambda input_dim: BinaryClassificationViTModel(THRESHOLD, torch.tensor(5))),
    'transformer': dict(  # upstream L184-259
        batch_size=64, num_epochs=300, lr=0.000003, patience=30,
        l2_reg=0.0000001, l1_reg=0.0000001, shuffle=True, pad=False, reload_best=False,
        build=lambda input_dim: TimeSeriesTransformer(
            input_dim, 512, 16, 12, 1, 0, device, torch.tensor(5), THRESHOLD)),
    'cnn_lstm': dict(     # upstream L264-334
        batch_size=64, num_epochs=300, lr=0.000002, patience=30,
        l2_reg=0, l1_reg=0, shuffle=True, pad=False, reload_best=False,
        build=lambda input_dim: CNN_LSTM_Model(
            device, input_dim, 512, 4, 0, torch.tensor(5), THRESHOLD)),
    'lstm': dict(         # upstream L339-409
        batch_size=128, num_epochs=500, lr=0.000008, patience=50,
        l2_reg=0.00001, l1_reg=0, shuffle=True, pad=False, reload_best=False,
        build=lambda input_dim: LSTMModel(
            device, input_dim, 512, torch.tensor(5), 0.2, 4, THRESHOLD)),
}


def evaluate(model, X_test: torch.Tensor, y_test: np.ndarray) -> dict:
    """Numerical replica of Visualization.plot_performance_with_auroc L25-40."""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        prob = torch.sigmoid(outputs).cpu().numpy()
        pred = (prob > THRESHOLD).astype(int)
    y_true, prob, pred = y_test.ravel(), prob.ravel(), pred.ravel()
    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, prob)
    return {
        'precision_normal': report['0.0']['precision'], 'recall_normal': report['0.0']['recall'],
        'f1_normal': report['0.0']['f1-score'],
        'precision_injured': report['1.0']['precision'], 'recall_injured': report['1.0']['recall'],
        'f1_injured': report['1.0']['f1-score'],
        'accuracy': report['accuracy'], 'roc_auc': auc(fpr, tpr),
        '_prob': prob, '_true': y_true,
    }


def append_result(row: dict) -> None:
    df = pd.DataFrame([row])
    header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode='a', header=header, index=False)


def train_one(name: str, cfg: dict, X: np.ndarray, y: np.ndarray) -> None:
    print(f"========== Starting {name} Training ==========", flush=True)
    for seed in SEEDS:
        set_seed(seed)
        Xtr, Xva, Xte, ytr, yva, yte = data_split(X, y, seed)
        if cfg['pad']:
            Xtr, Xva, Xte = pad_to_224(Xtr), pad_to_224(Xva), pad_to_224(Xte)

        train_loader = DataLoader(TensorDataset(Xtr.to(device), ytr.to(device)),
                                  batch_size=cfg['batch_size'], shuffle=cfg['shuffle'])
        valid_loader = DataLoader(TensorDataset(Xva.to(device), yva.to(device)),
                                  batch_size=cfg['batch_size'], shuffle=False)

        model = cfg['build'](Xtr.shape[2]).to(device)
        if device.type == 'cuda':
            torch.cuda.reset_max_memory_allocated(device)

        start = time.time()
        loss_log, best_wts, best_loss, valid_log = model.train_model(
            train_loader, valid_loader, cfg['num_epochs'], cfg['lr'],
            cfg['patience'], cfg['l2_reg'], cfg['l1_reg'])
        train_time = time.time() - start
        max_mem = (torch.cuda.max_memory_allocated(device) / 1024**2
                   if device.type == 'cuda' else 0)

        if cfg['reload_best']:
            model.load_state_dict(best_wts)

        m = evaluate(model, Xte.cpu(), yte.cpu().numpy())
        np.savez(PREDS_DIR / f"{name}_{seed}.npz", y_true=m.pop('_true'), y_prob=m.pop('_prob'))
        append_result({'model': name, 'seed': seed, **m,
                       'best_valid_loss': float(best_loss), 'epochs': len(loss_log),
                       'train_time_s': round(train_time, 1), 'max_gpu_mb': round(max_mem, 0)})
        print(f"[{name} seed={seed}] epochs={len(loss_log)} time={train_time:.0f}s "
              f"f1_injured={m['f1_injured']:.3f} roc_auc={m['roc_auc']:.3f}", flush=True)


def main() -> None:
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    prep.FINAL_CSV = AUTHORS_CSV
    X, y = prep.preprocess()
    print(f"X: {X.shape}, y: {y.shape}, injured={int(y.sum())}", flush=True)

    for name, cfg in MODELS.items():
        train_one(name, cfg, X, y)

    df = pd.read_csv(RESULTS_CSV)
    print("\n=== summary (mean over seeds) ===", flush=True)
    print(df.groupby('model')[['f1_injured', 'roc_auc', 'accuracy', 'train_time_s']]
          .mean().round(3), flush=True)
    print("paper reference: ViT F1_injured 0.73 / ROC-AUC 0.93; "
          "ResNet 0.64/0.88; Transformer 0.46/0.78; CNN+LSTM 0.44/0.75; LSTM 0.35/0.67",
          flush=True)


if __name__ == "__main__":
    main()
