"""Environment verification for the Kang 2025 TJS_Prediction reproduction.

Imports the cloned upstream repo's model code as-is, instantiates every model
with the paper's hyperparameters, and runs a dummy GPU forward pass + loss
computation. No training. Also smoke-tests pybaseball with one day of Statcast.

Run: .venv\\Scripts\\python.exe scripts\\verify_env.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    """Run one check, record and print pass/fail."""
    try:
        out = fn()
        RESULTS.append((name, "OK", out))
        print(f"[OK]   {name}: {out}")
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


# ---------- 1. versions ----------
import numpy, pandas, sklearn, matplotlib, seaborn, timm, shap, tqdm, imblearn  # noqa: E401
import torch, torchvision  # noqa: E401
import pybaseball

print("=== versions ===")
for mod in (sys, torch, torchvision, timm, numpy, pandas, sklearn, matplotlib,
            seaborn, shap, pybaseball, imblearn):
    name = getattr(mod, "__name__", "python")
    ver = getattr(mod, "__version__", None) or sys.version.split()[0]
    print(f"{name:>15}: {ver}")

# ---------- 2. CUDA ----------
def cuda_info() -> str:
    assert torch.cuda.is_available(), "CUDA not available"
    cap = torch.cuda.get_device_capability(0)
    return f"{torch.cuda.get_device_name(0)}, capability sm_{cap[0]}{cap[1]}"

check("cuda", cuda_info)

dev = torch.device("cuda")

# ---------- 3. Kang models (from the pristine upstream clone) ----------
REPO = Path(__file__).resolve().parent.parent / "TJS_Prediction"
sys.path.insert(0, str(REPO / "Classification"))
sys.path.insert(0, str(REPO / "Regression"))


def gpu_mem() -> str:
    return f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"


def fwd(model, x, loss_check: bool = True) -> str:
    model = model.to(dev)
    model.eval()
    with torch.no_grad():
        out = model(x.to(dev))
    logits = out[0] if isinstance(out, tuple) else out
    msg = f"out {tuple(logits.shape)}"
    if loss_check and hasattr(model, "BCEWithLogitsLoss"):
        target = torch.zeros(logits.shape, device=dev)
        loss = model.BCEWithLogitsLoss(logits, target)
        msg += f", weighted-BCE loss {loss.item():.4f}"
    return msg + f", {gpu_mem()}"


def test_vit() -> str:
    from ViT import BinaryClassificationViTModel
    m = BinaryClassificationViTModel(threshold=0.5, pos_weight=5)  # downloads timm weights
    return fwd(m, torch.randn(4, 224, 224))


def test_resnet() -> str:
    from ResNet import ResNet50Classifier
    # NOTE: ResNet.py alone passes pos_weight straight to BCEWithLogitsLoss without
    # torch.tensor() wrapping, so a raw int raises TypeError. classification_train.py
    # passes torch.tensor(5) — replicate that calling convention here.
    m = ResNet50Classifier(threshold=0.5, pos_weight=torch.tensor(5))
    return fwd(m, torch.randn(4, 224, 224))


def test_transformer() -> str:
    from Transformer_Encoder import TimeSeriesTransformer
    m = TimeSeriesTransformer(input_dim=102, model_dim=512, num_heads=16,
                              num_layers=12, num_classes=1, dropout=0.1,
                              device=dev, pos_weight=5, threshold=0.5)
    return fwd(m, torch.randn(4, 224, 102))


def test_cnn_lstm() -> str:
    from CNN_LSTM import CNN_LSTM_Model
    m = CNN_LSTM_Model(device=dev, input_dim=102, hidden_dim=512, num_layers=4,
                       dropout=0.1, pos_weight=5, threshold=0.5)
    return fwd(m, torch.randn(4, 224, 102))


def test_lstm() -> str:
    from LSTM import LSTMModel
    m = LSTMModel(device=dev, input_dim=102, hidden_dim=512, pos_weight=5,
                  dropout=0.1, num_layers=4, threshold=0.5)
    return fwd(m, torch.randn(4, 224, 102))


def test_regression_cnn() -> str:
    from regression_cnn import CNN
    m = CNN(dropout=0.0, device=dev, input_dim=102)
    return fwd(m, torch.randn(4, 1, 102), loss_check=False)


def test_scheduler() -> str:
    from cosmicannealing import CosineAnnealingWarmUpRestarts
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0)
    sch = CosineAnnealingWarmUpRestarts(opt, T_0=30, T_mult=1, eta_max=1e-3,
                                        T_up=10, gamma=0.9)
    lrs = []
    for _ in range(12):
        sch.step()
        lrs.append(opt.param_groups[0]["lr"])
    return f"lr after 12 steps: {lrs[-1]:.2e} (warmup peak ~1e-3)"


print("\n=== Kang model forward passes (GPU) ===")
check("ViT (timm vit_base_patch16_224 pretrained)", test_vit)
check("ResNet50 (torchvision pretrained)", test_resnet)
check("Transformer-Encoder (512d/16h/12L)", test_transformer)
check("CNN+LSTM (512d/4L)", test_cnn_lstm)
check("LSTM (512d/4L)", test_lstm)
check("Regression 1D-CNN", test_regression_cnn)
check("CosineAnnealingWarmUpRestarts scheduler", test_scheduler)

# ---------- 4. pybaseball smoke test ----------
print("\n=== pybaseball smoke test (single day) ===")


def test_statcast() -> str:
    from pybaseball import statcast, cache
    cache.enable()
    df = statcast(start_dt="2023-05-01", end_dt="2023-05-01", verbose=False)
    needed = ["player_name", "game_type", "home_team", "pitch_type", "game_date",
              "pitcher", "game_year", "release_speed", "release_pos_x",
              "release_pos_z", "pfx_x", "pfx_z", "plate_x", "plate_z",
              "vx0", "vy0", "vz0", "ax", "ay", "az", "effective_speed",
              "release_spin_rate", "release_extension", "spin_axis"]
    missing = [c for c in needed if c not in df.columns]
    assert len(df) > 1000, f"suspiciously few rows: {len(df)}"
    assert not missing, f"missing columns: {missing}"
    return f"{len(df)} pitches, {df.shape[1]} cols, all 24 Kang columns present"

check("statcast 2023-05-01 fetch", test_statcast)

print("\n=== summary ===")
fails = [r for r in RESULTS if r[1] == "FAIL"]
print(f"{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
sys.exit(1 if fails else 0)
