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
from inverse_opf.metrics import cosine_recovery
from inverse_opf.model import InverseOPFModel, StratifiedInverseOPFModel
from inverse_opf.plotting import apply_paper_style
from inverse_opf.synthetic import make_synthetic_dataset
from inverse_opf.training import TrainingConfig, train_inverse_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def save_curves(df: pd.DataFrame, out_dir: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["step"], df["train_loss"], label="train loss")
    ax.plot(df["step"], df["val_rmse"], label="val RMSE")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title("Training Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_dir = Path("outputs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    net_cfg = cfg["network"]
    model_cfg = cfg["model"]
    true_cfg = cfg["true_params"]

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

    if model_cfg["use_stratified_costs"]:
        model = StratifiedInverseOPFModel(
            n_buses=net_cfg["n_buses"],
            n_lines=net_cfg["n_lines"],
            n_strata=model_cfg["n_strata"],
        )
        lap = torch.tensor(cycle_laplacian(model_cfg["n_strata"]), dtype=torch.float32)
    else:
        model = InverseOPFModel(
            n_buses=net_cfg["n_buses"],
            n_lines=net_cfg["n_lines"],
        )
        lap = None

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

    result = train_inverse_model(
        model=model,
        opf_layer=opf,
        d_train=dataset.d_train,
        g_train_obs=dataset.g_train_obs,
        d_val=dataset.d_val,
        g_val_obs=dataset.g_val_obs,
        train_cfg=train_cfg,
        strata_train=dataset.strata_train,
        strata_val=dataset.strata_val,
        laplacian=lap,
    )

    history_df = pd.DataFrame(result.history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    save_curves(history_df, run_dir)

    with torch.no_grad():
        if isinstance(model, StratifiedInverseOPFModel):
            f_est_table = model.full_cost_table()
            f_est = f_est_table.mean(dim=0)
            gmax_est, pmax_est = model.shared_capacities()
        else:
            params = model.current_parameters()
            f_est = params.f
            gmax_est = params.gmax
            pmax_est = params.pmax

    summary = {
        "best_val_rmse": result.best_val_rmse,
        "f_cosine_recovery": cosine_recovery(f_est, dataset.true_f),
        "gmax_cosine_recovery": cosine_recovery(gmax_est, dataset.true_gmax),
        "pmax_cosine_recovery": cosine_recovery(pmax_est, dataset.true_pmax),
    }

    if isinstance(model, StratifiedInverseOPFModel):
        summary["f_table_cosine_recovery"] = cosine_recovery(
            f_est_table.reshape(-1),
            dataset.true_f_table.reshape(-1),
        )

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
