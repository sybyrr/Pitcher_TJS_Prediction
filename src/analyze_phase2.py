"""Phase 2 analysis (P2-6 + variant comparison table).

Aggregates every variant CSV under results/phase2/ plus the G1 baselines,
recomputes PR-AUC for G1 from the saved prediction files, and reports:
  - per-variant mean±sd for f1_injured / roc_auc / pr_auc (cls) and r2 /
    rmse_0_100 / novel_test_pitchers (reg)
  - paired Wilcoxon signed-rank vs the reference variant across shared seeds
  - the F1 resolution floor implied by the test positive count
  - span-only shortcut AUC (max_real_bin scalar)

Run: .venv\\Scripts\\python.exe src\\analyze_phase2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
P2 = ROOT / "results" / "phase2"
G1_CLS = ROOT / "results" / "classification_results.csv"
G1_PREDS = ROOT / "results" / "preds"


def g1_vit_with_prauc() -> pd.DataFrame:
    """G1 ViT rows + PR-AUC recomputed from saved test probabilities."""
    df = pd.read_csv(G1_CLS)
    df = df[df['model'] == 'vit'].copy()
    pr = {}
    for seed in df['seed']:
        z = np.load(G1_PREDS / f"vit_{seed}.npz")
        pr[seed] = average_precision_score(z['y_true'], z['y_prob'])
    df['pr_auc'] = df['seed'].map(pr)
    df['variant'] = 'v0_g1_vit'
    return df


def paired_wilcoxon(a: pd.DataFrame, b: pd.DataFrame, col: str) -> float:
    m = pd.merge(a[['seed', col]], b[['seed', col]], on='seed', suffixes=('_a', '_b'))
    if len(m) < 6 or (m[f'{col}_a'] == m[f'{col}_b']).all():
        return float('nan')
    return wilcoxon(m[f'{col}_a'], m[f'{col}_b']).pvalue


def main() -> None:
    # ---- classification ----
    cls_frames = [g1_vit_with_prauc()]
    for f in sorted(P2.glob('v*.csv')):
        cls_frames.append(pd.read_csv(f))
    cls = pd.concat(cls_frames, ignore_index=True)

    print("=== classification (ViT, mean±sd over seeds) ===")
    ref = cls[cls['variant'] == 'v1_bugfix']
    for v, g in cls.groupby('variant', sort=False):
        line = (f"{v:18s} f1_inj {g['f1_injured'].mean():.3f}±{g['f1_injured'].std():.3f}  "
                f"auc {g['roc_auc'].mean():.3f}  pr_auc {g['pr_auc'].mean():.3f}  "
                f"(n_seeds={len(g)}, test_inj~{g.get('n_test_injured', pd.Series([20])).median():.0f})")
        if v not in ('v0_g1_vit', 'v1_bugfix') and len(ref):
            p = paired_wilcoxon(g, ref, 'roc_auc')
            line += f"  vs_v1 p={p:.3f}" if p == p else ""
        print(line)

    if len(ref):
        n_inj = 20
        print(f"\nresolution floor: test injured n~{n_inj} -> F1 quantized ~1/{n_inj}"
              f" = 0.05; differences below ~0.05-0.08 are noise")

    # ---- regression ----
    print("\n=== regression (1D-CNN, mean±sd over seeds) ===")
    reg_frames = []
    for f in sorted(P2.glob('r*.csv')):
        reg_frames.append(pd.read_csv(f))
    if reg_frames:
        reg = pd.concat(reg_frames, ignore_index=True)
        rref = reg[reg['variant'] == 'r0std_row']
        for v, g in reg.groupby('variant', sort=False):
            line = (f"{v:18s} r2 {g['r2'].mean():.3f}±{g['r2'].std():.3f}  "
                    f"rmse100 {g['rmse_0_100'].mean():.1f}  "
                    f"novel_pitchers {g['novel_test_pitchers'].mean():.1f}/{g['n_test_pitchers'].mean():.0f}")
            if v != 'r0std_row' and len(rref):
                p = paired_wilcoxon(g, rref, 'r2')
                line += f"  vs_row p={p:.3f}" if p == p else ""
            print(line)

    # ---- span shortcut ----
    import phase2_data
    _, y, meta, _ = phase2_data.build_arrays()
    print(f"\nspan-only shortcut AUC (max_real_bin): "
          f"{roc_auc_score(y, -meta.max_real_bin):.4f}")


if __name__ == "__main__":
    main()
