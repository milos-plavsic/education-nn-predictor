from __future__ import annotations

import os

import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from app.datasets import load_xy_for_grade_prediction


class MLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def set_finetune_mode(self, freeze_backbone: bool) -> None:
        """If True, only the final linear layer is trainable (classic head fine-tuning)."""
        for i, layer in enumerate(self.net):
            trainable = (i == len(self.net) - 1) if freeze_backbone else True
            for p in layer.parameters():
                p.requires_grad = trainable


def train_and_evaluate(random_state: int = 42) -> dict[str, float]:
    epochs = int(os.getenv("NN_EPOCHS", "60"))
    X, y, _cols = load_xy_for_grade_prediction()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    model = MLP(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).numpy().ravel()
    mae = float(mean_absolute_error(y_val, val_pred))
    r2 = float(r2_score(y_val, val_pred))
    return {"val_mae": mae, "val_r2": r2, "epochs": float(epochs)}
