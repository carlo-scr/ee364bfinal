"""Noise-robustness sweep: cosine recovery vs. observation noise, multi-seed.

Compares the F&D-style baseline (learn only f, capacities given) against the
full extended model (learn f, gmax, pmax with Huber + slack L1) over a grid of
observation noise levels and multiple random seeds, producing mean +/- std
recovery curves.
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
from inverse_opf.io import load_yaml
from inverse_opf.metrics import cosine_recovery
from inverse_opf.model import InverseOPFModel
from inverse_opf.plotting import apply_paper_style
from inverse_opf.synthetic import make_synthetic_dataset
from inverse_opf.training import TrainingConfig, train_inverse_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--noises", type=str, default="0.5,1.0,2.0,4.0")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    return parser.parse_args()


def _make(seed: int, noise: float, cfg: dict):
    net_cfg = cfg["network"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    true_cfg = cfg["true_params"]
    return make_synthetic_dataset(
        n_buses=net_cfg["n_buses"],
        n_lines=net_cfg["n_lines"],
        n_train=data_cfg["n_train"],
        n_val=data_cfg["n_val"],
        demand_mean=data_cfg["demand_mean"],
        demand_std=data_cfg["demand_std"],
        observation_noise_std=noise,
        true_f_range=(true_cfg["f_min"], true_cfg["f_max"]),
        gmax_scale=(true_cfg["gmin_scale"], true_cfg["gmax_scale"]),
        pmax_scale=(true_cfg["pmin_scale"], true_cfg["pmax_scale"]),
        n_strata=model_cfg["n_strata"],
        seed=seed,
    )


def _train_variant(variant: str, ds, cfg: dict):
    net_cfg = cfg["network"]
    model_cfg = cfg["model"]
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence))
    if variant == "baseline":
        model = FixedCapacityInverseOPFModel(
            n_buses=net_cfg["n_buses"],
            n_lines=net_cfg["n_lines"],
            gmax=ds.true_gmax,
            pmax=ds.true_pmax,
        )
        loss = "mse"
        slack_w = 0.0
    else:
        model = InverseOPFModel(n_buses=net_cfg["n_buses"], n_lines=net_cfg["n_lines"])
        loss = "huber"
        slack_w = model_cfg["slack_l1_weight"]

    tcfg = TrainingConfig(
        steps=cfg["training"]["steps"],
        lr=cfg["training"]["lr"],
        loss=loss,
        huber_delta=cfg["training"]["huber_delta"],
        l2_weight=model_cfg["l2_weight"],
        slack_l1_weight=slack_w,
        clip_grad_norm=cfg["training"]["clip_grad_norm"],
    )
    res = train_inverse_model(
        model=model,
        opf_layer=opf,
        d_train=ds.d_train,
        g_train_obs=ds.g_train_obs,
        d_val=ds.d_val,
        g_val_obs=ds.g_val_obs,
        train_cfg=tcfg,
    )
    params = model.current_parameters()
    return {
        "best_val_rmse": res.best_val_rmse,
        "f_cos": cosine_recovery(params.f, ds.true_f),
        "gmax_cos": cosine_recovery(params.gmax, ds.true_gmax),
        "pmax_cos": cosine_recovery(params.pmax, ds.true_pmax),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    noises = [float(x) for x in args.noises.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    run_dir = Path("outputs") / f"{cfg['run_name']}_noise_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for noise in noises:
        for seed in seeds:
            ds = _make(seed, noise, cfg)
            for variant in ("baseline", "extended"):
                torch.manual_seed(seed)
                np.random.seed(seed)
                metrics = _train_variant(variant, ds, cfg)
                rows.append(
                    {"noise": noise, "seed": seed, "variant": variant, **metrics}
                )
                print(
                    json.dumps(
                        {"noise": noise, "seed": seed, "variant": variant, **metrics}
                    )
                )

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "noise_sweep.csv", index=False)

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
    for variant, color in zip(("baseline", "extended"), ("tab:gray", "tab:blue")):
        sub = (
            df[df["variant"] == variant]
            .groupby("noise")
            .agg(f_mean=("f_cos", "mean"), f_std=("f_cos", "std"),
                 rmse_mean=("best_val_rmse", "mean"), rmse_std=("best_val_rmse", "std"))
            .reset_index()
            .sort_values("noise")
        )
        axes[0].errorbar(
            sub["noise"], sub["f_mean"], yerr=sub["f_std"],
            marker="o", capsize=3, color=color, label=variant,
        )
        axes[1].errorbar(
            sub["noise"], sub["rmse_mean"], yerr=sub["rmse_std"],
            marker="o", capsize=3, color=color, label=variant,
        )
    axes[0].set_ylabel(r"$\cos\angle(\hat f, f^\star)$")
    axes[0].set_xlabel("Observation noise $\sigma$")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(title="Model")
    axes[1].set_ylabel("Best validation RMSE")
    axes[1].set_xlabel("Observation noise $\sigma$")
    axes[1].legend(title="Model")
    fig.suptitle("Noise robustness (mean $\pm$ 1 std over {} seeds)".format(len(seeds)))
    fig.tight_layout()
    fig.savefig(run_dir / "noise_sweep.png")
    paper_dir = Path("paper") / "figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_dir / "noise_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
