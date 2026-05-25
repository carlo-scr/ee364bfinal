"""Shared utilities used by all experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from inverse_opf.baselines import FixedCapacityInverseOPFModel
from inverse_opf.baselines_kkt import kkt_residual_inverse
from inverse_opf.baselines_regression import mlp_baseline, ridge_baseline
from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
from inverse_opf.graph import cycle_laplacian
from inverse_opf.metrics import (
    cosine_recovery,
    kendall_recovery,
    merit_order_accuracy,
    normalized_rmse,
    rmse,
    spearman_recovery,
)
from inverse_opf.model import InverseOPFModel, StratifiedInverseOPFModel
from inverse_opf.synthetic import make_synthetic_dataset
from inverse_opf.training import TrainingConfig, train_inverse_model

PAPER_FIG_DIR = "paper/figures"


@dataclass
class StandardConfig:
    n_buses: int = 10
    n_lines: int = 14
    n_train: int = 200
    n_val: int = 100
    demand_mean: float = 40.0
    demand_std: float = 8.0
    obs_noise: float = 0.5
    f_min: float = 5.0
    f_max: float = 60.0
    gmin_scale: float = 1.05
    gmax_scale: float = 1.30
    pmin_scale: float = 0.8
    pmax_scale: float = 1.2
    n_strata: int = 24
    physics: str = "transport"
    diurnal_amp: float = 0.0
    diurnal_cost_amp: float = 0.20


def build_dataset(seed: int, sc: StandardConfig):
    return make_synthetic_dataset(
        n_buses=sc.n_buses, n_lines=sc.n_lines,
        n_train=sc.n_train, n_val=sc.n_val,
        demand_mean=sc.demand_mean, demand_std=sc.demand_std,
        observation_noise_std=sc.obs_noise,
        true_f_range=(sc.f_min, sc.f_max),
        gmax_scale=(sc.gmin_scale, sc.gmax_scale),
        pmax_scale=(sc.pmin_scale, sc.pmax_scale),
        n_strata=sc.n_strata, seed=seed,
        physics=sc.physics, diurnal_amp=sc.diurnal_amp,
        diurnal_cost_amp=sc.diurnal_cost_amp,
    )


def thin_training_set(ds, drop_frac: float, seed: int):
    """Drop a fraction of training rows (samples) uniformly at random.

    This is the simplest "missing data" ablation: we observe only a subset
    of the original training timesteps.  Returns a *new* dataset object
    with reduced training arrays; val arrays are untouched.
    """
    if drop_frac <= 0.0:
        return ds
    n = ds.d_train.shape[0]
    keep_n = max(1, int(round(n * (1.0 - drop_frac))))
    rng = np.random.default_rng(seed * 4079 + 31)
    idx = rng.choice(n, size=keep_n, replace=False)
    idx.sort()
    idx_t = torch.tensor(idx, dtype=torch.long)
    # Build a copy with the relevant fields sliced.  We keep everything
    # else (true_*, incidence, ...) so downstream code is unchanged.
    import dataclasses
    return dataclasses.replace(
        ds,
        d_train=ds.d_train[idx_t],
        g_train_obs=ds.g_train_obs[idx_t],
        strata_train=ds.strata_train[idx_t],
    )



def standard_training(steps: int = 800, lr: float = 5e-2, **overrides) -> TrainingConfig:
    base = dict(
        steps=steps, lr=lr, loss="huber", huber_delta=1.0,
        l2_weight=1e-5, laplacian_weight=0.05, slack_l1_weight=1e-3,
        clip_grad_norm=5.0, lr_schedule="cosine", lr_min=1e-4,
        warmup_frac=0.0,
        early_stopping_patience=120, restore_best=True,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def evaluate_recovery(f_hat, gmax_hat, pmax_hat, ds) -> dict:
    return {
        "f_cos": cosine_recovery(f_hat, ds.true_f),
        "f_spearman": spearman_recovery(f_hat, ds.true_f),
        "f_kendall": kendall_recovery(f_hat, ds.true_f),
        "f_merit_acc": merit_order_accuracy(f_hat, ds.true_f),
        "gmax_cos": cosine_recovery(gmax_hat, ds.true_gmax),
        "pmax_cos": cosine_recovery(pmax_hat, ds.true_pmax),
    }


def predict_dispatch_diff(opf: DCOpfLayer, d, f, gmax, pmax) -> torch.Tensor:
    f_b = f.unsqueeze(0).repeat(d.shape[0], 1) if f.dim() == 1 else f
    gmax_b = gmax.unsqueeze(0).repeat(d.shape[0], 1) if gmax.dim() == 1 else gmax
    pmax_b = pmax.unsqueeze(0).repeat(d.shape[0], 1) if pmax.dim() == 1 else pmax
    with torch.no_grad():
        g, _ = opf.solve(d, f_b, gmax_b, pmax_b)
    return g


def true_val_dispatch(ds, sc: StandardConfig) -> torch.Tensor:
    """Recover the *clean* (noise-free) ground-truth val dispatch by re-solving
    the forward problem with the true parameters."""
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance),
                     physics=sc.physics)
    f_val_t = ds.true_f_table[ds.strata_val]
    gmax_b = ds.true_gmax.unsqueeze(0).repeat(ds.d_val.shape[0], 1)
    pmax_b = ds.true_pmax.unsqueeze(0).repeat(ds.d_val.shape[0], 1)
    with torch.no_grad():
        g, _ = opf.solve(ds.d_val, f_val_t, gmax_b, pmax_b)
    return g


# ----- Methods (each returns a dict with f, gmax, pmax estimates and metrics).

def run_diff_full(ds, sc: StandardConfig, train_cfg: TrainingConfig | None = None) -> dict:
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance),
                     physics=sc.physics)
    model = InverseOPFModel(n_buses=sc.n_buses, n_lines=sc.n_lines)
    train_cfg = train_cfg or standard_training()
    res = train_inverse_model(
        model, opf, ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs, train_cfg,
    )
    params = model.current_parameters()
    g_pred = predict_dispatch_diff(opf, ds.d_val, params.f, params.gmax, params.pmax)
    g_true = true_val_dispatch(ds, sc)
    out = dict(method="diff_full",
               f=params.f.detach().cpu().numpy(),
               gmax=params.gmax.detach().cpu().numpy(),
               pmax=params.pmax.detach().cpu().numpy(),
               val_rmse=rmse(g_pred, ds.g_val_obs),
               val_rmse_clean=rmse(g_pred, g_true),
               val_nrmse=normalized_rmse(g_pred, ds.g_val_obs),
               val_nrmse_clean=normalized_rmse(g_pred, g_true),
               best_step=res.best_step)
    out.update(evaluate_recovery(out["f"], out["gmax"], out["pmax"], ds))
    return out


def run_diff_strat(ds, sc: StandardConfig, train_cfg: TrainingConfig | None = None) -> dict:
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance),
                     physics=sc.physics)
    model = StratifiedInverseOPFModel(n_buses=sc.n_buses, n_lines=sc.n_lines, n_strata=sc.n_strata)
    lap = torch.tensor(cycle_laplacian(sc.n_strata), dtype=torch.float32)
    train_cfg = train_cfg or standard_training()
    res = train_inverse_model(
        model, opf, ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs, train_cfg,
        strata_train=ds.strata_train, strata_val=ds.strata_val, laplacian=lap,
    )
    f_table = model.full_cost_table().detach().cpu().numpy()
    gmax_v, pmax_v = model.shared_capacities()
    f_mean = f_table.mean(axis=0)
    f_val_t = model.costs_for_strata(ds.strata_val)
    gmax_b = gmax_v.unsqueeze(0).repeat(ds.d_val.shape[0], 1)
    pmax_b = pmax_v.unsqueeze(0).repeat(ds.d_val.shape[0], 1)
    with torch.no_grad():
        g_pred, _ = opf.solve(ds.d_val, f_val_t, gmax_b, pmax_b)
    g_true = true_val_dispatch(ds, sc)
    out = dict(method="diff_strat",
               f=f_mean,
               f_table=f_table,
               gmax=gmax_v.detach().cpu().numpy(),
               pmax=pmax_v.detach().cpu().numpy(),
               val_rmse=rmse(g_pred, ds.g_val_obs),
               val_rmse_clean=rmse(g_pred, g_true),
               val_nrmse=normalized_rmse(g_pred, ds.g_val_obs),
               val_nrmse_clean=normalized_rmse(g_pred, g_true),
               best_step=res.best_step)
    out.update(evaluate_recovery(out["f"], out["gmax"], out["pmax"], ds))
    out["f_table_cos"] = cosine_recovery(f_table.reshape(-1), ds.true_f_table.reshape(-1).cpu().numpy())
    return out


def run_diff_fcap(ds, sc: StandardConfig, train_cfg: TrainingConfig | None = None) -> dict:
    """Differentiable, but with capacities frozen to truth (F&D-style baseline)."""
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance),
                     physics=sc.physics)
    model = FixedCapacityInverseOPFModel(
        n_buses=sc.n_buses, n_lines=sc.n_lines,
        gmax=ds.true_gmax, pmax=ds.true_pmax,
    )
    train_cfg = train_cfg or standard_training()
    res = train_inverse_model(
        model, opf, ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs, train_cfg,
    )
    params = model.current_parameters()
    g_pred = predict_dispatch_diff(opf, ds.d_val, params.f, params.gmax, params.pmax)
    g_true = true_val_dispatch(ds, sc)
    out = dict(method="diff_fcap",
               f=params.f.detach().cpu().numpy(),
               gmax=params.gmax.detach().cpu().numpy(),
               pmax=params.pmax.detach().cpu().numpy(),
               val_rmse=rmse(g_pred, ds.g_val_obs),
               val_rmse_clean=rmse(g_pred, g_true),
               val_nrmse=normalized_rmse(g_pred, ds.g_val_obs),
               val_nrmse_clean=normalized_rmse(g_pred, g_true),
               best_step=res.best_step)
    out.update(evaluate_recovery(out["f"], out["gmax"], out["pmax"], ds))
    return out


def run_kkt(ds, sc: StandardConfig, learn_capacities: bool = True) -> dict:
    data = DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance)
    res = kkt_residual_inverse(
        data, ds.d_train, ds.g_train_obs, learn_capacities=learn_capacities,
    )
    # Predict val dispatch using the recovered (f, gmax, pmax) through the QP.
    opf = DCOpfLayer(data, physics=sc.physics)
    f_t = torch.tensor(res.f, dtype=torch.float32)
    gmax_t = torch.tensor(res.gmax, dtype=torch.float32)
    pmax_t = torch.tensor(res.pmax, dtype=torch.float32)
    g_pred = predict_dispatch_diff(opf, ds.d_val, f_t, gmax_t, pmax_t)
    g_true = true_val_dispatch(ds, sc)
    out = dict(method="kkt",
               f=res.f, gmax=res.gmax, pmax=res.pmax,
               val_rmse=rmse(g_pred, ds.g_val_obs),
               val_rmse_clean=rmse(g_pred, g_true),
               val_nrmse=normalized_rmse(g_pred, ds.g_val_obs),
               val_nrmse_clean=normalized_rmse(g_pred, g_true),
               kkt_residual=res.residual_norm)
    out.update(evaluate_recovery(out["f"], out["gmax"], out["pmax"], ds))
    return out


def run_ridge(ds, sc: StandardConfig) -> dict:
    res = ridge_baseline(ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs, alpha=1.0)
    pred_val = res.predict(ds.d_val.detach().cpu().numpy())
    g_true = true_val_dispatch(ds, sc).cpu().numpy()
    return dict(method="ridge", f=np.full(sc.n_buses, np.nan),
                gmax=np.full(sc.n_buses, np.nan), pmax=np.full(sc.n_lines, np.nan),
                val_rmse=res.val_rmse,
                val_rmse_clean=float(np.sqrt(np.mean((pred_val - g_true) ** 2))),
                val_nrmse=normalized_rmse(pred_val, ds.g_val_obs),
                val_nrmse_clean=normalized_rmse(pred_val, g_true),
                f_cos=float("nan"), f_spearman=float("nan"),
                f_kendall=float("nan"), f_merit_acc=float("nan"),
                gmax_cos=float("nan"), pmax_cos=float("nan"))


def run_mlp(ds, sc: StandardConfig) -> dict:
    res = mlp_baseline(ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs)
    pred_val = res.predict(ds.d_val.detach().cpu().numpy())
    g_true = true_val_dispatch(ds, sc).cpu().numpy()
    return dict(method="mlp", f=np.full(sc.n_buses, np.nan),
                gmax=np.full(sc.n_buses, np.nan), pmax=np.full(sc.n_lines, np.nan),
                val_rmse=res.val_rmse,
                val_rmse_clean=float(np.sqrt(np.mean((pred_val - g_true) ** 2))),
                val_nrmse=normalized_rmse(pred_val, ds.g_val_obs),
                val_nrmse_clean=normalized_rmse(pred_val, g_true),
                f_cos=float("nan"), f_spearman=float("nan"),
                f_kendall=float("nan"), f_merit_acc=float("nan"),
                gmax_cos=float("nan"), pmax_cos=float("nan"))


METHOD_LABELS = {
    "ridge":      "Ridge",
    "mlp":        "MLP",
    "kkt":        "KKT-residual",
    "diff_fcap":  r"Diff. (caps fixed)",
    "diff_full":  r"Diff. (full)",
    "diff_strat": r"Diff. (stratified)",
}


def aggregate(rows, by_cols, value_cols):
    """Return mean and std DataFrame for the given grouping."""
    import pandas as pd
    df = pd.DataFrame(rows)
    grouped = df.groupby(by_cols)
    mean = grouped[value_cols].mean().add_suffix("_mean")
    std = grouped[value_cols].std().add_suffix("_std")
    return mean.join(std).reset_index()


# ---------------------------------------------------------------------------
# Bootstrap CIs and held-out test eval (Section 5 deliverables)
# ---------------------------------------------------------------------------


def bootstrap_ci_mean(values, n_boot: int = 2000, alpha: float = 0.05,
                     rng_seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean across seeds.

    Returns (lo, hi).  NaNs are ignored.  If fewer than 2 finite values are
    present, returns (nan, nan).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1.0 - alpha / 2))
    return (lo, hi)


def aggregate_with_ci(rows, by_cols, value_cols, n_boot: int = 2000):
    """Like ``aggregate`` but also emits ``<col>_ci_lo`` / ``<col>_ci_hi`` columns
    via percentile bootstrap over seeds.
    """
    import pandas as pd
    df = pd.DataFrame(rows)
    base = aggregate(rows, by_cols, value_cols)
    # Stable key order.
    ci_rows = []
    for key, sub in df.groupby(by_cols):
        ci_row: dict[str, object] = {}
        if isinstance(key, tuple):
            for col, val in zip(by_cols, key, strict=False):
                ci_row[col] = val
        else:
            ci_row[by_cols[0] if isinstance(by_cols, list) else by_cols] = key
        for c in value_cols:
            lo, hi = bootstrap_ci_mean(sub[c].to_numpy(), n_boot=n_boot)
            ci_row[f"{c}_ci_lo"] = lo
            ci_row[f"{c}_ci_hi"] = hi
        ci_rows.append(ci_row)
    ci_df = pd.DataFrame(ci_rows)
    return base.merge(ci_df, on=by_cols, how="left")


def build_test_demands(ds, sc: StandardConfig, n_test: int, seed: int):
    """Sample held-out test demands with the *same* true parameters and
    forward-solve to get the clean ground-truth dispatch.

    The test RNG sub-stream is keyed off ``seed`` but offset so it does not
    overlap with the training/val demands.
    """
    rng = np.random.default_rng(seed * 1009 + 17)
    strata = rng.integers(0, sc.n_strata, size=(n_test,))
    base = rng.normal(sc.demand_mean, sc.demand_std, size=(n_test, sc.n_buses))
    if sc.diurnal_amp > 0.0 and sc.n_strata > 1:
        t = strata.astype(float) / float(sc.n_strata)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(sc.n_buses,))
        shape = 1.0 + sc.diurnal_amp * np.sin(2.0 * np.pi * t[:, None] + phase[None, :])
        base = base * shape
    d_test_np = np.clip(base, 1.0, None)
    d_test = torch.tensor(d_test_np, dtype=torch.float32)

    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance),
                     physics=sc.physics)
    strata_t = torch.tensor(strata, dtype=torch.long)
    f_t = ds.true_f_table[strata_t]
    gmax_b = ds.true_gmax.unsqueeze(0).repeat(n_test, 1)
    pmax_b = ds.true_pmax.unsqueeze(0).repeat(n_test, 1)
    with torch.no_grad():
        g_clean, _ = opf.solve(d_test, f_t, gmax_b, pmax_b)
    return d_test, g_clean, strata_t


def eval_physics_on_test(ds, sc: StandardConfig, d_test, g_clean,
                          f_hat, gmax_hat, pmax_hat) -> float:
    """Forward-solve with estimated parameters at test demands; return
    normalized RMSE vs. the clean ground-truth."""
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance),
                     physics=sc.physics)
    f_t = torch.as_tensor(f_hat, dtype=torch.float32).unsqueeze(0).repeat(d_test.shape[0], 1)
    gmax_t = torch.as_tensor(gmax_hat, dtype=torch.float32).unsqueeze(0).repeat(d_test.shape[0], 1)
    pmax_t = torch.as_tensor(pmax_hat, dtype=torch.float32).unsqueeze(0).repeat(d_test.shape[0], 1)
    with torch.no_grad():
        g_pred, _ = opf.solve(d_test, f_t, gmax_t, pmax_t)
    return normalized_rmse(g_pred, g_clean)


# ---------------------------------------------------------------------------
# PJM-like single-seed runner (8 fuels, single bus, diurnal renewables)
# ---------------------------------------------------------------------------


def run_pjm_seed(seed: int, n_train: int = 600, n_val: int = 200,
                 steps: int = 700, lr: float = 5e-2) -> dict:
    """Train the stratified inverse-OPF on the PJM-like synthetic stack and
    return a metrics dict (one row per seed).  Matches the metrics reported
    in scripts/run_pjm_like.py but without the side-effecting figures.
    """
    from inverse_opf.synthetic import PJM_FUELS, make_pjm_like_dataset
    from inverse_opf.graph import cycle_laplacian
    from inverse_opf.model import StratifiedInverseOPFModel
    from inverse_opf.metrics import kendall_recovery

    ds = make_pjm_like_dataset(n_train=n_train, n_val=n_val, seed=seed)
    data = DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance)
    opf = DCOpfLayer(data, physics="transport")
    n_buses = ds.true_f.shape[0]
    n_lines = ds.true_pmax.shape[0]
    n_strata = ds.true_f_table.shape[0]

    model = StratifiedInverseOPFModel(
        n_buses=n_buses, n_lines=n_lines, n_strata=n_strata,
        f_init=30.0, gmax_init=20.0, pmax_init=200.0,
    )
    lap = torch.tensor(cycle_laplacian(n_strata), dtype=torch.float32)
    train_cfg = standard_training(steps=steps, lr=lr,
                                  laplacian_weight=0.5, slack_l1_weight=0.0)
    import time as _time
    t0 = _time.time()
    res = train_inverse_model(
        model, opf, ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs, train_cfg,
        strata_train=ds.strata_train, strata_val=ds.strata_val, laplacian=lap,
    )
    elapsed = _time.time() - t0

    with torch.no_grad():
        f_table_est = model.full_cost_table().cpu().numpy()
    f_mean_est = f_table_est.mean(axis=0)
    f_mean_true = ds.true_f.cpu().numpy()
    f_table_true = ds.true_f_table.cpu().numpy()

    n_fuel = len(PJM_FUELS)
    return dict(
        method="pjm_stratified",
        seed=seed,
        best_step=res.best_step,
        best_val_rmse=float(res.best_val_rmse),
        f_mean_cos=cosine_recovery(f_mean_est[:n_fuel], f_mean_true[:n_fuel]),
        f_mean_spearman=spearman_recovery(f_mean_est[:n_fuel], f_mean_true[:n_fuel]),
        f_mean_kendall=kendall_recovery(f_mean_est[:n_fuel], f_mean_true[:n_fuel]),
        f_mean_merit_acc=merit_order_accuracy(f_mean_est[:n_fuel], f_mean_true[:n_fuel]),
        f_table_cos=cosine_recovery(f_table_est[:, :n_fuel].reshape(-1),
                                    f_table_true[:, :n_fuel].reshape(-1)),
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Section 3: warm-started diff-OPF (init from KKT) and KMeans strata variant
# ---------------------------------------------------------------------------


def run_diff_warmstart(ds, sc: StandardConfig,
                       train_cfg: TrainingConfig | None = None) -> dict:
    """Run the differentiable inverse OPF after warm-starting (f, gmax, pmax)
    from the KKT-residual baseline.  Reports val/test NRMSE and timing so
    we can quantify the speedup over a cold start.
    """
    import time as _time
    t0 = _time.time()
    data = DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance)
    kkt = kkt_residual_inverse(data, ds.d_train, ds.g_train_obs, learn_capacities=True)
    opf = DCOpfLayer(data, physics=sc.physics)

    model = InverseOPFModel(n_buses=sc.n_buses, n_lines=sc.n_lines)
    # Inject the KKT estimates into the softplus pre-parameters.  Softplus
    # is monotone; for x>>0, softplus(x) ~ x, so we just clamp+log1p_exp.
    def _inv_softplus(y: np.ndarray) -> torch.Tensor:
        y = np.maximum(y, 1e-3)
        return torch.tensor(np.log(np.expm1(y)), dtype=torch.float32)
    with torch.no_grad():
        model.f_raw.copy_(_inv_softplus(kkt.f))
        model.gmax_raw.copy_(_inv_softplus(kkt.gmax))
        model.pmax_raw.copy_(_inv_softplus(kkt.pmax))

    train_cfg = train_cfg or standard_training()
    res = train_inverse_model(
        model, opf, ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs, train_cfg,
    )
    params = model.current_parameters()
    g_pred = predict_dispatch_diff(opf, ds.d_val, params.f, params.gmax, params.pmax)
    g_true = true_val_dispatch(ds, sc)
    out = dict(method="diff_warmstart",
               f=params.f.detach().cpu().numpy(),
               gmax=params.gmax.detach().cpu().numpy(),
               pmax=params.pmax.detach().cpu().numpy(),
               val_rmse=rmse(g_pred, ds.g_val_obs),
               val_rmse_clean=rmse(g_pred, g_true),
               val_nrmse=normalized_rmse(g_pred, ds.g_val_obs),
               val_nrmse_clean=normalized_rmse(g_pred, g_true),
               best_step=res.best_step,
               warmstart_s=_time.time() - t0)
    out.update(evaluate_recovery(out["f"], out["gmax"], out["pmax"], ds))
    return out


def run_diff_strat_kmeans(ds, sc: StandardConfig,
                          train_cfg: TrainingConfig | None = None) -> dict:
    """Stratified diff-OPF, but strata are recovered from demands via KMeans
    instead of using the ground-truth hour-of-day index.
    """
    from inverse_opf.analysis import kmeans_strata, strata_agreement
    labels_train = kmeans_strata(ds.d_train, sc.n_strata, seed=0)
    labels_val = kmeans_strata(ds.d_val, sc.n_strata, seed=0)
    agree = strata_agreement(labels_train, ds.strata_train.cpu().numpy())

    # Build a temporary dataset view with the KMeans labels.
    import dataclasses
    ds_km = dataclasses.replace(
        ds,
        strata_train=torch.tensor(labels_train, dtype=torch.long),
        strata_val=torch.tensor(labels_val, dtype=torch.long),
    )
    out = run_diff_strat(ds_km, sc, train_cfg)
    out["method"] = "diff_strat_kmeans"
    out["strata_agreement"] = agree
    return out


