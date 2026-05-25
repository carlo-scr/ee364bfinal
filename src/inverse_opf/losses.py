from __future__ import annotations

import torch


def residual_loss(pred: torch.Tensor, target: torch.Tensor,
                  loss_name: str, huber_delta: float) -> torch.Tensor:
    if loss_name == "mse":
        return torch.mean((pred - target) ** 2)
    if loss_name == "huber":
        return torch.nn.functional.huber_loss(pred, target, delta=huber_delta)
    raise ValueError(f"Unknown loss: {loss_name}")


def l2_regularizer(*params: torch.Tensor) -> torch.Tensor:
    if not params:
        return torch.zeros(())
    return sum(torch.sum(p ** 2) for p in params)


def laplacian_smoothness(values: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
    """tr(V^T L V) for V in R^{S x N}."""
    return torch.trace(values.T @ laplacian @ values)


def slack_l1_penalty(g_pred: torch.Tensor, gmax: torch.Tensor,
                     p_pred: torch.Tensor, pmax: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    """Penalize residual constraint violation. Useful when the forward layer
    uses a lifted slack formulation (then this is the L1 of recovered slacks)
    and as a soft regularizer otherwise."""
    upper_g = torch.relu(g_pred - gmax + eps)
    lower_g = torch.relu(-g_pred + eps)
    upper_p = torch.relu(p_pred - pmax + eps)
    lower_p = torch.relu(-p_pred - pmax + eps)
    return torch.mean(upper_g + lower_g) + torch.mean(upper_p + lower_p)
