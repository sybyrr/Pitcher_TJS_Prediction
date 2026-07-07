"""Phase 2 training loop — faithful port of upstream ViT.train_model
(TJS_Prediction/Classification/ViT.py L81-153) with exactly one toggle:

  deepcopy_best — True stores best_model_wts as a deep copy (the fix for the
  reference-aliasing bug); False reproduces upstream behavior where the
  'best' snapshot mutates with further training and the final reload is a
  no-op (evaluation at final-epoch weights).

Everything else is kept verbatim: Adam(lr=0, weight_decay=l2), the
CosineAnnealingWarmUpRestarts(T_0=20, T_mult=1, eta_max=lr, T_up=5,
gamma=0.8) schedule, weighted-BCE + L1 over all params, the epoch>10 gate
before best tracking, and UNWEIGHTED validation BCE for early stopping
(the selection-objective mismatch is a separate, unfixed finding).
"""
from __future__ import annotations

import copy

import torch


def train_variant(model, train_loader, valid_loader, epochs, learning_rate,
                  patience, l2_reg, l1_reg, *, deepcopy_best: bool):
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent
                           / "TJS_Prediction" / "Classification"))
    from cosmicannealing import CosineAnnealingWarmUpRestarts

    optimizer = torch.optim.Adam(model.parameters(), lr=0, weight_decay=l2_reg)
    scheduler = CosineAnnealingWarmUpRestarts(
        optimizer, T_0=20, T_mult=1, eta_max=learning_rate, T_up=5, gamma=0.8)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = None
    train_loss_log, valid_loss_log = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for data, target in train_loader:
            data = data.to(model.device)
            target = target.to(model.device, dtype=torch.float32)
            optimizer.zero_grad()
            out = model(data)
            loss = model.BCEWithLogitsLoss(out, target)
            l1_val = 0
            for param in model.parameters():
                l1_val += torch.sum(torch.abs(param))
            loss += l1_reg * l1_val
            epoch_loss += loss.item() * data.size(0)
            loss.backward()
            optimizer.step()

        scheduler.step()
        epoch_loss /= len(train_loader.dataset)
        train_loss_log.append(epoch_loss)

        model.eval()
        valid_acc, valid_loss = model.predict(valid_loader)
        model.train()
        valid_loss_log.append(valid_loss)

        if valid_loss < best_loss and (epoch > 10):
            best_loss = valid_loss
            best_model_wts = (copy.deepcopy(model.state_dict()) if deepcopy_best
                              else model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    return train_loss_log, best_model_wts, best_loss, valid_loss_log
