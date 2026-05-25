"""Run the ablation study described in the proposal.

Compares four configurations:
  1. Baseline: learn only f (Fuentes Valenzuela & Degleris [14] style).
  2. + extension 1: also learn g_max, p_max.
  3. + extension 3: Huber + slack L1 on top of (2).
  4. + extension 2: stratified time-varying costs with Laplacian reg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from inverse_opf.baselines import FixedCapacityInverseOPFModel
from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
from inverse_opf.graph import cycle_laplacian
from inverse_opf.io import load_yaml
from inverse_opf.metrics import cosine_recovery, rmse
from inverse_opf.model import InverseOPFModel, StratifiedInverseOPFModel
from inverse_opf.plotting import apply_paper_style
from inverse_opf.synthetic import make_synthetic_dataset
from inverse_opf.training import TrainingConfig, train_inverse_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def build_model(kind: str, dataset, cfg):
    n_buses = cfg["network"]["n_buses"]
    n_lines = cfg["network"]["n_lines"]
    if kind == "fixed_capacity":
        return FixedCapacityInverseOPFModel(
            n_buses=n_buses,
            n_lines=n_lines,
            gmax=dataset.true_gmax,
            pmax=dataset.true_pmax,
        )
    if kind == "full":
        return InverseOPFModel(n_buses=n_buses, n_lines=n_lines)
    if kind == "stratified":
        return StratifiedInverseOPFModel(
            n_buses=n_buses,
            n_lines=n_lines,
            n_strata=cfg["model"]["n_strata"],
        )
    raise ValueError(f"Unknown model kind: {kind}")


def evaluate(model, dataset):
    with torch.no_grad():
        if isinstance(model, StratifiedInverseOPFModel):
            f_table = model.full_cost_table()
            f_est = f_table.mean(dim=0)
            gmax_est, pmax_est = model.shared_capacities()
            f_table_recovery = cosine_recovery(
                f_table.reshape(-1), dataset.true_f_table.reshape(-1)
            )
        else:
            params = model.current_parameters()
            f_est = params.f
            gmax_est = params.gmax
            pmax_est = params.pmax
            f_table_recovery = None
    return {
        "f_cos": cosine_recovery(f_est, dataset.true_f),
        "gmax_cos": cosine_recovery(gmax_est, dataset.true_gmax),
        "pmax_cos": cosine_recovery(pmax_est, dataset.true_pmax),
        "f_table_cos": f_table_recovery,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = Path("outputs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg, net_cfg, model_cfg, true_cfg = (
        cfg["data"],
        cfg["network"],
        cfg["model"],
        cfg["true_params"],
    )

    dataset = make_synthetic_dataset(
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

    opf = DCOpfLayer(DCOpfProblemData(incidence=dataset.incidence))
    lap = torch.tensor(cycle_laplacian(model_cfg["n_strata"]), dtype=torch.float32)

    rows: list[dict] = []
    for ab in cfg["ablations"]:
        torch.manual_seed(seed)
        model = build_model(ab["model"], dataset, cfg)

        train_cfg = TrainingConfig(
            steps=cfg["training"]["steps"],
            lr=cfg["training"]["lr"],
            loss=ab.get("loss", cfg["training"]["loss"]),
            huber_delta=cfg["training"]["huber_delta"],
            l2_weight=model_cfg["l2_weight"],
            laplacian_weight=model_cfg["laplacian_weight"] if ab["model"] == "stratified" else 0.0,
            slack_l1_weight=model_cfg["slack_l1_weight"] if ab.get("loss") == "huber" else 0.0,
            clip_grad_norm=cfg["training"]["clip_grad_norm"],
        )

        result = train_inverse_model(
            model=model,
            opf_layer=opf,
            d_train=dataset.d_train,
            g_train_obs=dataset.g_train_obs,
            d_val=dataset.d_val,
            g_val_obs=dataset.g_val_obs,
            train_cfg=train_cfg,
            strata_train=dataset.strata_train if ab["model"] == "stratified" else None,
            strata_val=dataset.strata_val if ab["model"] == "stratified" else None,
            laplacian=lap if ab["model"] == "stratified" else None,
        )

        metrics = evaluate(model, dataset)
        rows.append(
            {
                "name": ab["name"],
                "description": ab["description"],
                "model": ab["model"],
                "loss": ab.get("loss", cfg["training"]["loss"]),
                "best_val_rmse": result.best_val_rmse,
                **metrics,
            }
        )
        print(json.dumps(rows[-1], indent=2))

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "ablation.csv", index=False)
    with open(run_dir / "ablation.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    _plot_summary(df, run_dir)


def _plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(df["name"], df["best_val_rmse"])
    axes[0].set_title("Best validation RMSE")
    axes[0].tick_params(axis="x", rotation=20)
    metrics = ["f_cos", "gmax_cos", "pmax_cos"]
    width = 0.25
    x = np.arange(len(df))
    for i, m in enumerate(metrics):
        axes[1].bar(x + i * width, df[m].astype(float), width=width, label=m)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(df["name"], rotation=20)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Parameter recovery (cosine)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_summary.png")
    paper_dir = Path("paper") / "figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_dir / "ablation_summary.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
