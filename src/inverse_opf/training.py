from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from .dc_opf import DCOpfLayer
from .losses import laplacian_smoothness, residual_loss, slack_l1_penalty
from .metrics import rmse
from .model import InverseOPFModel, StratifiedInverseOPFModel


@dataclass
class TrainingConfig:
    steps: int = 300
    lr: float = 3e-2
    loss: str = "huber"
    huber_delta: float = 1.0
    l2_weight: float = 1e-4
    laplacian_weight: float = 0.0
    slack_l1_weight: float = 0.0
    clip_grad_norm: float = 5.0
    lr_schedule: str = "cosine"          # "none" | "cosine"
    lr_min: float = 1e-4
    warmup_frac: float = 0.0             # fraction of steps for linear LR warmup
    early_stopping_patience: int = 80    # 0 disables
    restore_best: bool = True
    val_every: int = 1
    verbose: bool = False


@dataclass
class TrainingResult:
    history: list[dict[str, float]]
    best_val_rmse: float
    best_step: int


def _expand_shared(x: torch.Tensor, n: int) -> torch.Tensor:
    return x.unsqueeze(0).repeat(n, 1)


def _post_softplus_l2(model: torch.nn.Module) -> torch.Tensor:
    """L2 penalty on the *post-softplus* (physical) parameters."""
    if isinstance(model, StratifiedInverseOPFModel):
        f_table = model.full_cost_table()
        gmax, pmax = model.shared_capacities()
        return (f_table ** 2).sum() + (gmax ** 2).sum() + (pmax ** 2).sum()
    if isinstance(model, InverseOPFModel):
        params = model.current_parameters()
        return (params.f ** 2).sum() + (params.gmax ** 2).sum() + (params.pmax ** 2).sum()
    return torch.zeros((), dtype=torch.float32)


def _cosine_lr(step: int, total: int, lr_max: float, lr_min: float) -> float:
    if total <= 1:
        return lr_max
    progress = (step - 1) / max(1, total - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def train_inverse_model(
    model: InverseOPFModel | StratifiedInverseOPFModel,
    opf_layer: DCOpfLayer,
    d_train: torch.Tensor,
    g_train_obs: torch.Tensor,
    d_val: torch.Tensor,
    g_val_obs: torch.Tensor,
    train_cfg: TrainingConfig,
    strata_train: torch.Tensor | None = None,
    strata_val: torch.Tensor | None = None,
    laplacian: torch.Tensor | None = None,
) -> TrainingResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_step = 0
    best_state: dict | None = None
    patience_left = train_cfg.early_stopping_patience or 10**9

    for step in range(1, train_cfg.steps + 1):
        # Optional linear warmup over the first warmup_frac * steps iterations.
        warmup_steps = int(train_cfg.warmup_frac * train_cfg.steps)
        if warmup_steps > 0 and step <= warmup_steps:
            lr = train_cfg.lr * (step / max(1, warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        elif train_cfg.lr_schedule == "cosine":
            # Cosine over the *post-warmup* portion.
            effective_step = step - warmup_steps
            effective_total = max(1, train_cfg.steps - warmup_steps)
            lr = _cosine_lr(effective_step, effective_total, train_cfg.lr, train_cfg.lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        if isinstance(model, StratifiedInverseOPFModel):
            if strata_train is None:
                raise ValueError("strata_train is required for stratified model")
            f_train = model.costs_for_strata(strata_train)
            gmax, pmax = model.shared_capacities()
            gmax_train = _expand_shared(gmax, d_train.shape[0])
            pmax_train = _expand_shared(pmax, d_train.shape[0])
        else:
            params = model.current_parameters()
            f_train = _expand_shared(params.f, d_train.shape[0])
            gmax_train = _expand_shared(params.gmax, d_train.shape[0])
            pmax_train = _expand_shared(params.pmax, d_train.shape[0])

        g_train_pred, p_train_pred = opf_layer.solve(d_train, f_train, gmax_train, pmax_train)

        fit = residual_loss(g_train_pred, g_train_obs, train_cfg.loss, train_cfg.huber_delta)
        reg_l2 = train_cfg.l2_weight * _post_softplus_l2(model)
        reg_sparse = train_cfg.slack_l1_weight * slack_l1_penalty(
            g_train_pred, gmax_train, p_train_pred, pmax_train,
        )
        reg_graph = torch.zeros((), dtype=torch.float32)
        if isinstance(model, StratifiedInverseOPFModel) and laplacian is not None:
            reg_graph = train_cfg.laplacian_weight * laplacian_smoothness(
                model.full_cost_table(), laplacian
            )

        loss = fit + reg_l2 + reg_sparse + reg_graph
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
        optimizer.step()

        val_rmse_val: float
        if step % train_cfg.val_every == 0:
            with torch.no_grad():
                if isinstance(model, StratifiedInverseOPFModel):
                    if strata_val is None:
                        raise ValueError("strata_val is required for stratified model")
                    f_val = model.costs_for_strata(strata_val)
                    gmax_v, pmax_v = model.shared_capacities()
                    gmax_val = _expand_shared(gmax_v, d_val.shape[0])
                    pmax_val = _expand_shared(pmax_v, d_val.shape[0])
                else:
                    params = model.current_parameters()
                    f_val = _expand_shared(params.f, d_val.shape[0])
                    gmax_val = _expand_shared(params.gmax, d_val.shape[0])
                    pmax_val = _expand_shared(params.pmax, d_val.shape[0])

                g_val_pred, _ = opf_layer.solve(d_val, f_val, gmax_val, pmax_val)
                val_rmse_val = rmse(g_val_pred, g_val_obs)

                if val_rmse_val + 1e-8 < best_val:
                    best_val = val_rmse_val
                    best_step = step
                    if train_cfg.restore_best:
                        best_state = copy.deepcopy(model.state_dict())
                    patience_left = train_cfg.early_stopping_patience or 10**9
                else:
                    patience_left -= 1
        else:
            val_rmse_val = float("nan")

        history.append({
            "step": float(step),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(loss.detach().cpu()),
            "fit": float(fit.detach().cpu()),
            "reg_l2": float(reg_l2.detach().cpu()) if torch.is_tensor(reg_l2) else float(reg_l2),
            "reg_graph": float(reg_graph.detach().cpu()) if torch.is_tensor(reg_graph) else float(reg_graph),
            "reg_sparse": float(reg_sparse.detach().cpu()) if torch.is_tensor(reg_sparse) else float(reg_sparse),
            "val_rmse": float(val_rmse_val),
        })

        if train_cfg.verbose and step % max(1, train_cfg.steps // 10) == 0:
            print(f"  step {step:4d}  loss {float(loss):.4f}  val_rmse {val_rmse_val:.4f}")

        if patience_left <= 0:
            break

    if train_cfg.restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return TrainingResult(history=history, best_val_rmse=best_val, best_step=best_step)
