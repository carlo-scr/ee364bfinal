"""Downstream analysis: MEF Jacobian, merit-order curve, counterfactual demand shift.

Trains the full stratified inverse model on synthetic data, then produces:
  - merit_order.png : sorted learned vs. true unit costs.
  - mef_heatmap.png : MEF matrix dC/dd at a sample operating point.
  - counterfactual.csv : dispatch shift under a +10% demand perturbation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
from inverse_opf.graph import cycle_laplacian
from inverse_opf.io import load_yaml
from inverse_opf.model import StratifiedInverseOPFModel
from inverse_opf.plotting import apply_paper_style
from inverse_opf.sensitivity import (
    counterfactual_dispatch,
    jacobian_g_wrt_d,
    marginal_emission_factors,
)
from inverse_opf.synthetic import make_synthetic_dataset
from inverse_opf.training import TrainingConfig, train_inverse_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = Path("outputs") / f"{cfg['run_name']}_mef"
    run_dir.mkdir(parents=True, exist_ok=True)

    net_cfg = cfg["network"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    true_cfg = cfg["true_params"]

    ds = make_synthetic_dataset(
        n_buses=net_cfg["n_buses"],
        n_lines=net_cfg["n_lines"],
        n_train=data_cfg["n_train"],
        n_val=data_cfg["n_val"],
        demand_mean=data_cfg["demand_mean"],
        demand_std=data_cfg["demand_std"],
        observation_noise_std=data_cfg["observation_noise_std"],
        true_f_range=(true_cfg["f_min"], true_cfg["f_max"]),
        gmax_scale=(true_cfg["gmin_scale"], true_cfg["gmax_scale"]),
        pmax_scale=(true_cfg["pmin_scale"], true_cfg["pmax_scale"]),
        n_strata=model_cfg["n_strata"],
        seed=seed,
    )

    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence))
    model = StratifiedInverseOPFModel(
        n_buses=net_cfg["n_buses"],
        n_lines=net_cfg["n_lines"],
        n_strata=model_cfg["n_strata"],
    )
    lap = torch.tensor(cycle_laplacian(model_cfg["n_strata"]), dtype=torch.float32)

    train_cfg = TrainingConfig(
        steps=cfg["training"]["steps"],
        lr=cfg["training"]["lr"],
        loss=cfg["training"]["loss"],
        huber_delta=cfg["training"]["huber_delta"],
        l2_weight=model_cfg["l2_weight"],
        laplacian_weight=model_cfg["laplacian_weight"],
        slack_l1_weight=model_cfg["slack_l1_weight"],
        clip_grad_norm=cfg["training"]["clip_grad_norm"],
    )

    train_inverse_model(
        model=model,
        opf_layer=opf,
        d_train=ds.d_train,
        g_train_obs=ds.g_train_obs,
        d_val=ds.d_val,
        g_val_obs=ds.g_val_obs,
        train_cfg=train_cfg,
        strata_train=ds.strata_train,
        strata_val=ds.strata_val,
        laplacian=lap,
    )

    with torch.no_grad():
        f_est = model.full_cost_table().mean(dim=0)
        gmax_est, pmax_est = model.shared_capacities()

    _plot_merit_order(f_est, ds.true_f, run_dir)

    d0 = ds.d_val[0]
    jac = jacobian_g_wrt_d(opf, d0, f_est, gmax_est, pmax_est)
    _plot_mef_heatmap(jac, run_dir)

    # Counterfactual: +10% demand uniformly
    delta = 0.1 * d0
    g_base = counterfactual_dispatch(opf, d0, f_est, gmax_est, pmax_est, torch.zeros_like(d0))
    g_cf = counterfactual_dispatch(opf, d0, f_est, gmax_est, pmax_est, delta)
    cf_df = pd.DataFrame(
        {
            "bus": np.arange(d0.shape[0]),
            "g_baseline": g_base.detach().cpu().numpy(),
            "g_plus10pct": g_cf.detach().cpu().numpy(),
            "delta_g": (g_cf - g_base).detach().cpu().numpy(),
        }
    )
    cf_df.to_csv(run_dir / "counterfactual.csv", index=False)

    c = torch.ones_like(f_est)
    mef = marginal_emission_factors(jac, c).detach().cpu().numpy()
    with open(run_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "mef_mean": float(np.mean(mef)),
                "mef_max": float(np.max(mef)),
                "counterfactual_total_shift": float(cf_df["delta_g"].sum()),
            },
            fh,
            indent=2,
        )


def _plot_merit_order(f_est: torch.Tensor, f_true: torch.Tensor, out_dir: Path) -> None:
    apply_paper_style()
    est = f_est.detach().cpu().numpy()
    true = f_true.detach().cpu().numpy()
    order = np.argsort(true)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(true[order], marker="o", label="true $f$ (sorted)")
    ax.plot(est[order], marker="s", label="learned $\\hat f$ (true order)")
    ax.set_xlabel("Generator index (sorted by true cost)")
    ax.set_ylabel("Unit cost")
    ax.set_title("Merit-order recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "merit_order.png")
    paper_dir = Path("paper") / "figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_dir / "merit_order.pdf")
    plt.close(fig)


def _plot_mef_heatmap(jac: torch.Tensor, out_dir: Path) -> None:
    apply_paper_style()
    arr = jac.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, aspect="auto", cmap="coolwarm")
    fig.colorbar(im, ax=ax, label=r"$\partial g^*/\partial d$")
    ax.set_xlabel("Demand bus")
    ax.set_ylabel("Generation bus")
    ax.set_title("MEF Jacobian at sample operating point")
    fig.tight_layout()
    fig.savefig(out_dir / "mef_heatmap.png")
    paper_dir = Path("paper") / "figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_dir / "mef_heatmap.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
