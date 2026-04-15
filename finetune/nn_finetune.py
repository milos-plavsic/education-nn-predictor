from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from app.datasets import load_xy_for_grade_prediction
from app.nn_train import MLP


def _two_phase_train(
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Returns y_val, pred_after_phase1, pred_after_phase2, metrics."""
    phase1_epochs = int(os.getenv("FT_PHASE1_EPOCHS", os.getenv("NN_EPOCHS", "50")))
    phase2_epochs = int(os.getenv("FT_PHASE2_EPOCHS", "40"))
    lr1 = float(os.getenv("FT_LR_PRETRAIN", "1e-2"))
    lr2 = float(os.getenv("FT_LR_FINETUNE", "5e-4"))

    X, y, _ = load_xy_for_grade_prediction()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    Xv = torch.tensor(X_val, dtype=torch.float32)

    model = MLP(X_train.shape[1])
    loss_fn = nn.MSELoss()

    model.train()
    model.set_finetune_mode(False)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr1)
    for _ in range(phase1_epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p1 = model(Xv).numpy().ravel()
    mae_p1 = float(mean_absolute_error(y_val, p1))

    model.train()
    model.set_finetune_mode(True)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr2)
    for _ in range(phase2_epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p2 = model(Xv).numpy().ravel()
    mae_p2 = float(mean_absolute_error(y_val, p2))
    r2 = float(r2_score(y_val, p2))

    yv = np.asarray(y_val, dtype=float)
    metrics = {
        "val_mae_after_pretrain": mae_p1,
        "val_mae_after_head_finetune": mae_p2,
        "val_r2_final": r2,
        "phase1_epochs": float(phase1_epochs),
        "phase2_epochs": float(phase2_epochs),
    }
    return yv, p1, p2, metrics


def run_two_phase_finetune(random_state: int = 42) -> dict[str, float]:
    """Phase 1: train full MLP. Phase 2: freeze backbone, fine-tune last layer with lower LR."""
    _yv, _p1, _p2, m = _two_phase_train(random_state=random_state)
    return m


def finetune_val_predictions(random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    return _two_phase_train(random_state=random_state)


def main() -> None:
    out = run_two_phase_finetune()
    print("Two-phase NN fine-tune (freeze backbone → head only)")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
