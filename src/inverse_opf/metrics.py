from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x).reshape(-1)


def cosine_recovery(x_hat, x_true) -> float:
    a = _to_np(x_hat); b = _to_np(x_true)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rmse(pred, target) -> float:
    a = _to_np(pred); b = _to_np(target)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def normalized_rmse(pred, target) -> float:
    a = _to_np(pred); b = _to_np(target)
    denom = float(np.sqrt(np.mean(b ** 2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def spearman_recovery(x_hat, x_true) -> float:
    a = _to_np(x_hat); b = _to_np(x_true)
    if a.size < 2:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def kendall_recovery(x_hat, x_true) -> float:
    a = _to_np(x_hat); b = _to_np(x_true)
    if a.size < 2:
        return float("nan")
    tau, _ = kendalltau(a, b)
    return float(tau)


def merit_order_accuracy(f_hat, f_true) -> float:
    a = _to_np(f_hat); b = _to_np(f_true)
    n = a.size
    if n < 2:
        return float("nan")
    sign_hat = np.sign(a[:, None] - a[None, :])
    sign_true = np.sign(b[:, None] - b[None, :])
    mask = sign_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float((sign_hat[mask] == sign_true[mask]).mean())


def identifiability_score(g_obs, gmax, tol: float = 1e-3) -> np.ndarray:
    g = g_obs.detach().cpu().numpy() if isinstance(g_obs, torch.Tensor) else np.asarray(g_obs)
    gm = gmax.detach().cpu().numpy() if isinstance(gmax, torch.Tensor) else np.asarray(gmax)
    if gm.ndim == 1:
        gm = gm[None, :]
    interior = (g > tol) & (g < gm - tol)
    return interior.mean(axis=0)


def per_generator_relative_error(f_hat, f_true) -> np.ndarray:
    a = _to_np(f_hat); b = _to_np(f_true)
    eps = 1e-9
    return np.abs(a - b) / np.maximum(np.abs(b), eps)
