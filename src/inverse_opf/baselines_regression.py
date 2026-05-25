"""Black-box regression baselines: maps demand d -> generation g.

These do not respect physics or KKT; they exist only to bound the achievable
RMSE for a fair comparison. Used as a "no convex structure" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import Ridge


@dataclass
class RegressionResult:
    val_rmse: float
    train_rmse: float
    predict: object   # callable: d (np.ndarray, T x n) -> g_pred (T x n)


def ridge_baseline(
    d_train: np.ndarray | torch.Tensor,
    g_train: np.ndarray | torch.Tensor,
    d_val: np.ndarray | torch.Tensor,
    g_val: np.ndarray | torch.Tensor,
    alpha: float = 1.0,
) -> RegressionResult:
    if isinstance(d_train, torch.Tensor):
        d_train = d_train.detach().cpu().numpy()
        g_train = g_train.detach().cpu().numpy()
        d_val = d_val.detach().cpu().numpy()
        g_val = g_val.detach().cpu().numpy()
    model = Ridge(alpha=alpha)
    model.fit(d_train, g_train)
    g_pred_train = model.predict(d_train)
    g_pred_val = model.predict(d_val)
    return RegressionResult(
        val_rmse=float(np.sqrt(np.mean((g_pred_val - g_val) ** 2))),
        train_rmse=float(np.sqrt(np.mean((g_pred_train - g_train) ** 2))),
        predict=lambda x: model.predict(x),
    )


class _SmallMLP(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x))   # generation must be >= 0


def mlp_baseline(
    d_train: torch.Tensor,
    g_train: torch.Tensor,
    d_val: torch.Tensor,
    g_val: torch.Tensor,
    hidden: int = 64,
    steps: int = 1000,
    lr: float = 3e-3,
) -> RegressionResult:
    n_in = d_train.shape[1]; n_out = g_train.shape[1]
    model = _SmallMLP(n_in, n_out, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(d_train)
        loss = torch.mean((pred - g_train) ** 2)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_loss = torch.mean((model(d_val) - g_val) ** 2)
            if float(val_loss) < best_val:
                best_val = float(val_loss)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        train_rmse = float(torch.sqrt(torch.mean((model(d_train) - g_train) ** 2)))
        val_rmse = float(torch.sqrt(torch.mean((model(d_val) - g_val) ** 2)))

    def predict(x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return model(x).cpu().numpy()

    return RegressionResult(val_rmse=val_rmse, train_rmse=train_rmse, predict=predict)
