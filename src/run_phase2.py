"""Phase 2 classification variant runner (ViT, 10 seeds per variant).

Each variant changes ONE axis vs the bugfix base (deepcopy best-weights +
train shuffle=True), so effects are attributable:

  v1_bugfix        P2-2  training-bug fixes only (split/data as G1)
  v2_f102          P2-4  + drop the time-index input channel (F=102)
  v3_window        P2-3  + restrict to a commonly-observed bin window
                          (fill-tail shortcut removed by construction)
  v4_lininterp     P2-4  + true linear interpolation within observed span
  v5_trainstat     P2-4  + 4.7-sigma outlier stats fit on train pitchers only
  v6_grouped       P2-1  + split grouped by REAL pitcher id (8 dual-role fix)
  v7_temporal      P2-5  + fixed test = samples anchored >= 2022
  v6r_grouped_ours P2-7  grouped split on our rebuilt final_df (comparator)
  v8_causal        P2-7  grouped split on the causal-diff final_df

Baseline for comparison: Phase 1 G1 results (results/classification_results.csv,
vit rows, same seeds 100..1000).

Run: .venv\\Scripts\\python.exe src\\run_phase2.py [variant ...]
(no args = all variants in the order above; finished (variant, seed) pairs
are skipped, so the process is resumable)
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
from sklearn.metrics import auc, average_precision_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "TJS_Prediction" / "Classification"))
sys.path.insert(0, str(ROOT / "src"))

from ViT import BinaryClassificationViTModel  # upstream, verbatim
import phase2_data
from phase2_split import split as make_split
from train_loop import train_variant

RESULTS_DIR = ROOT / "results" / "phase2"
PREDS_DIR = RESULTS_DIR / "preds"
META_CSV = ROOT / "data" / "cohort_meta.csv"
OURS_CSV = ROOT / "data" / "final_df.csv"
CAUSAL_CSV = ROOT / "data" / "final_df_causal.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = range(100, 1001, 100)
# upstream ViT hyperparameters (classification_train.py L117-124)
HP = dict(batch_size=16, num_epochs=200, learning_rate=0.000004,
          patience=20, l2_reg=0.00003, l1_reg=0)

VARIANTS = {
    'v1_bugfix': dict(),
    'v2_f102': dict(data=dict(drop_time_channel=True)),
    'v3_window': dict(data=dict(drop_time_channel=True), common_window=True),
    'v4_lininterp': dict(data=dict(interp='linear')),
    'v5_trainstat': dict(per_seed_outlier=True),
    'v6_grouped': dict(split='grouped'),
    'v7_temporal': dict(split='temporal'),
    'v6r_grouped_ours': dict(csv=OURS_CSV, split='grouped'),
    'v8_causal': dict(csv=CAUSAL_CSV, split='grouped'),
    # most honest retrospective number: artifact-free window x future-season test
    'v9_window_temporal': dict(data=dict(drop_time_channel=True),
                               common_window=True, split='temporal'),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_to_224(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x, (0, 224 - x.shape[2], 0, 224 - x.shape[1]), "constant", -5)


def evaluate(model, X_test, y_test) -> dict:
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(X_test.to(device))).cpu().numpy().ravel()
    y_true = y_test.cpu().numpy().ravel()
    pred = (prob > 0.5).astype(int)
    report = classification_report(y_true, pred, output_dict=True, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, prob)
    inj = report.get('1.0', {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0})
    return {
        'precision_injured': inj['precision'], 'recall_injured': inj['recall'],
        'f1_injured': inj['f1-score'], 'accuracy': report['accuracy'],
        'roc_auc': auc(fpr, tpr), 'pr_auc': average_precision_score(y_true, prob),
        'n_test': len(y_true), 'n_test_injured': int(y_true.sum()),
        '_prob': prob, '_true': y_true,
    }


def anchor_years_for(meta: phase2_data.SampleMeta, y: np.ndarray) -> np.ndarray:
    m = pd.read_csv(META_CSV, parse_dates=['anchor_date'])
    key = {(int(p), int(t)): d.year for p, t, d in
           zip(m['pitcher'], m['target'], m['anchor_date'])}
    raw = np.where(y == 1, -meta.signed_ids, meta.signed_ids)
    return np.array([key[(int(rp), int(t))] for rp, t in zip(raw, y)])


def run_variant(name: str, cfg: dict) -> None:
    csv = cfg.get('csv', phase2_data.AUTHORS_CSV)
    if not Path(csv).exists():
        print(f"[{name}] SKIP — input missing: {csv}", flush=True)
        return
    if cfg.get('split') == 'temporal' and not META_CSV.exists():
        print(f"[{name}] SKIP — cohort meta missing", flush=True)
        return

    out_csv = RESULTS_DIR / f"{name}.csv"
    done = set()
    if out_csv.exists():
        done = set(pd.read_csv(out_csv)['seed'].tolist())

    X, y, meta, feats = phase2_data.build_arrays(csv, **cfg.get('data', {}))
    common_max_bin = None
    if cfg.get('common_window'):
        # largest bin observed by >=90% of injured samples (normals cover more)
        common_max_bin = int(np.quantile(meta.max_real_bin[y == 1], 0.10) // 5 * 5)
        keep_from = (1215 - common_max_bin) // 5
        cov_n = (meta.max_real_bin[y == 0] >= common_max_bin).mean()
        print(f"[{name}] window 100..{common_max_bin} ({224 - keep_from} bins), "
              f"coverage injured>=90%, normal={cov_n:.0%}", flush=True)
        X = X[:, keep_from:, :]

    anchors = anchor_years_for(meta, y) if cfg.get('split') == 'temporal' else None
    print(f"[{name}] X={X.shape}, features={len(feats)}", flush=True)

    for seed in SEEDS:
        if seed in done:
            continue
        set_seed(seed)

        if cfg.get('per_seed_outlier'):
            # identify this seed's train pitchers (same first-stage split as
            # phase2_split 'stratified'), then rebuild arrays with outlier
            # stats fit on those pitchers only (sample order is unchanged)
            idx_all = np.arange(len(y))
            tr, _ = train_test_split(idx_all, test_size=0.4, random_state=seed, stratify=y)
            train_signed = set(meta.signed_ids[tr])
            Xs, ys, meta_s, _ = phase2_data.build_arrays(
                csv, outlier_train_pitchers=train_signed, **cfg.get('data', {}))
            assert (meta_s.signed_ids == meta.signed_ids).all()
            Xtr, Xva, Xte, ytr, yva, yte, idx_test = make_split(
                'stratified', Xs, ys, seed, real_ids=meta_s.real_ids)
        else:
            Xtr, Xva, Xte, ytr, yva, yte, idx_test = make_split(
                cfg.get('split', 'stratified'), X, y, seed,
                real_ids=meta.real_ids, anchor_years=anchors)

        Xtr, Xva, Xte = pad_to_224(Xtr), pad_to_224(Xva), pad_to_224(Xte)
        train_loader = DataLoader(TensorDataset(Xtr.to(device), ytr.to(device)),
                                  batch_size=HP['batch_size'], shuffle=True)
        valid_loader = DataLoader(TensorDataset(Xva.to(device), yva.to(device)),
                                  batch_size=HP['batch_size'], shuffle=False)

        model = BinaryClassificationViTModel(0.5, torch.tensor(5)).to(device)
        start = time.time()
        loss_log, _, best_loss, _ = train_variant(
            model, train_loader, valid_loader, HP['num_epochs'],
            HP['learning_rate'], HP['patience'], HP['l2_reg'], HP['l1_reg'],
            deepcopy_best=True)
        train_time = time.time() - start

        m = evaluate(model, Xte, yte)
        np.savez(PREDS_DIR / f"{name}_{seed}.npz",
                 y_true=m.pop('_true'), y_prob=m.pop('_prob'), idx_test=idx_test)
        row = {'variant': name, 'seed': seed, **m,
               'best_valid_loss': float(best_loss), 'epochs': len(loss_log),
               'train_time_s': round(train_time, 1)}
        pd.DataFrame([row]).to_csv(out_csv, mode='a', header=not out_csv.exists(), index=False)
        print(f"[{name} seed={seed}] epochs={len(loss_log)} time={train_time:.0f}s "
              f"f1_inj={m['f1_injured']:.3f} auc={m['roc_auc']:.3f} "
              f"pr_auc={m['pr_auc']:.3f} (test inj {m['n_test_injured']}/{m['n_test']})",
              flush=True)
    print(f"[{name}] DONE", flush=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        run_variant(name, VARIANTS[name])
    print("ALL VARIANTS DONE", flush=True)


if __name__ == "__main__":
    main()
