"""Recovery-curve experiment: cosine recovery vs. training-set size.

Sweeps the number of training samples and (optionally) random seeds, and
reports mean +/- 1 std bands of cosine recovery for the F&D-style baseline
(learn only f) versus the extended model (learn f, gmax, pmax with Huber).
"""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--sizes", type=str, default="25,50,100,200")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    return parser.parse_args()


def _train_one(variant: str, ds, cfg: dict) -> dict:
    net_cfg = cfg["network"]
    model_cfg = cfg["model"]
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence))
    if variant == "baseline_f_only":
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
    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    run_dir = Path("outputs") / f"{cfg['run_name']}_recovery"
    run_dir.mkdir(parents=True, exist_ok=True)

    net_cfg = cfg["network"]
    data_cfg = copy.deepcopy(cfg["data"])
    model_cfg = cfg["model"]
    true_cfg = cfg["true_params"]

    rows: list[dict] = []
    for n_train in sizes:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            ds = make_synthetic_dataset(
                n_buses=net_cfg["n_buses"],
                n_lines=net_cfg["n_lines"],
                n_train=n_train,
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
            for variant in ("baseline_f_only", "extensions_1_3"):
                torch.manual_seed(seed)
                metrics = _train_one(variant, ds, cfg)
                rows.append({"variant": variant, "n_train": n_train, "seed": seed, **metrics})
                print(json.dumps(rows[-1]))

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "recovery.csv", index=False)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = {"baseline_f_only": "tab:gray", "extensions_1_3": "tab:blue"}
    for variant, color in palette.items():
        agg = (
            df[df["variant"] == variant]
            .groupby("n_train")
            .agg(mean=("f_cos", "mean"), std=("f_cos", "std"))
            .reset_index()
            .sort_values("n_train")
        )
        ax.plot(agg["n_train"], agg["mean"], marker="o", color=color, label=variant)
        ax.fill_between(
            agg["n_train"],
            (agg["mean"] - agg["std"]).clip(lower=0),
            (agg["mean"] + agg["std"]).clip(upper=1),
            color=color,
            alpha=0.18,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training set size")
    ax.set_ylabel(r"$\cos\angle(\hat f, f^\star)$")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Cost recovery vs. dataset size (mean $\\pm$ 1 std, n={len(seeds)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "recovery_curve.png")
    paper_dir = Path("paper") / "figures"
    paper_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_dir / "recovery_curve.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
