"""PJM-like single-bus economic-dispatch experiment.

8 fuel buckets (nuclear, hydro, wind, solar, coal, ccgt, oil_st, ct_peaker)
with realistic-ish capacities and heat-rate-derived marginal costs. Hourly
load 80-130 GW with diurnal + weekly variation. Solar / wind availability is
modeled by raising their marginal cost when unavailable, producing a merit
order that *changes by hour*.

We fit the stratified inverse OPF and report (i) merit-order accuracy at the
mean cost, (ii) hour-resolved cost recovery.

Outputs:
  outputs/pjm_like/results.csv
  paper/figures/pjm_stack.pdf
  paper/figures/pjm_recovery.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import standard_training
from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
from inverse_opf.graph import cycle_laplacian
from inverse_opf.metrics import (
    cosine_recovery,
    kendall_recovery,
    merit_order_accuracy,
    spearman_recovery,
)
from inverse_opf.model import StratifiedInverseOPFModel
from inverse_opf.plotting import PALETTE, apply_paper_style, figsize, save_figure
from inverse_opf.synthetic import PJM_FUELS, make_pjm_like_dataset
from inverse_opf.training import train_inverse_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--n_train", type=int, default=600)
    p.add_argument("--steps", type=int, default=700)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path("outputs/pjm_like"); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    learned_table = None
    learned_mean = None
    true_mean = None
    true_table = None
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        ds = make_pjm_like_dataset(n_train=args.n_train, n_val=200, seed=seed)
        opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance),
                         physics="transport")
        n_buses = ds.true_f.shape[0]
        n_lines = ds.true_pmax.shape[0]
        n_strata = ds.true_f_table.shape[0]
        model = StratifiedInverseOPFModel(n_buses=n_buses, n_lines=n_lines,
                                          n_strata=n_strata,
                                          f_init=30.0, gmax_init=20.0, pmax_init=200.0)
        lap = torch.tensor(cycle_laplacian(n_strata), dtype=torch.float32)
        cfg = standard_training(steps=args.steps, lr=5e-2,
                                laplacian_weight=0.5, slack_l1_weight=0.0)
        t0 = time.time()
        res = train_inverse_model(model, opf, ds.d_train, ds.g_train_obs,
                                  ds.d_val, ds.g_val_obs, cfg,
                                  strata_train=ds.strata_train,
                                  strata_val=ds.strata_val, laplacian=lap)
        elapsed = time.time() - t0

        with torch.no_grad():
            f_table_est = model.full_cost_table().cpu().numpy()
            gmax_est, pmax_est = model.shared_capacities()
        f_mean_est = f_table_est.mean(axis=0)
        f_mean_true = ds.true_f.cpu().numpy()
        f_table_true = ds.true_f_table.cpu().numpy()

        # Restrict to actual fuel buckets (drop the load-bus index).
        n_fuel = len(PJM_FUELS)
        m_acc = merit_order_accuracy(f_mean_est[:n_fuel], f_mean_true[:n_fuel])
        spear = spearman_recovery(f_mean_est[:n_fuel], f_mean_true[:n_fuel])
        kend  = kendall_recovery(f_mean_est[:n_fuel], f_mean_true[:n_fuel])
        f_cos = cosine_recovery(f_mean_est[:n_fuel], f_mean_true[:n_fuel])
        f_table_cos = cosine_recovery(f_table_est[:, :n_fuel].reshape(-1),
                                      f_table_true[:, :n_fuel].reshape(-1))
        rows.append({"seed": seed, "best_step": res.best_step,
                     "best_val_rmse": res.best_val_rmse,
                     "f_mean_cos": f_cos, "f_mean_spearman": spear,
                     "f_mean_kendall": kend, "f_mean_merit_acc": m_acc,
                     "f_table_cos": f_table_cos,
                     "elapsed_s": elapsed})
        print(f"seed {seed}: rmse {res.best_val_rmse:.2f}  cos {f_cos:.3f}"
              f"  merit_acc {m_acc:.3f}  table_cos {f_table_cos:.3f}  ({elapsed:.1f}s)")
        if learned_table is None:
            learned_table = f_table_est
            learned_mean = f_mean_est
            true_mean = f_mean_true
            true_table = f_table_true

    df = pd.DataFrame(rows); df.to_csv(out_dir / "results.csv", index=False)
    np.savez(out_dir / "arrays.npz",
             f_mean_true=true_mean, f_mean_learned=learned_mean,
             f_table_true=true_table, f_table_learned=learned_table)
    Path(out_dir / "summary.json").write_text(json.dumps({
        "f_mean_cos_mean": float(df["f_mean_cos"].mean()),
        "f_mean_cos_std": float(df["f_mean_cos"].std()),
        "f_mean_spearman_mean": float(df["f_mean_spearman"].mean()),
        "f_mean_merit_acc_mean": float(df["f_mean_merit_acc"].mean()),
        "f_table_cos_mean": float(df["f_table_cos"].mean()),
    }, indent=2))

    # ---- merit-order figure ----
    apply_paper_style()
    n_fuel = len(PJM_FUELS)
    fuels = [name for name, _, _ in PJM_FUELS]
    order = np.argsort(true_mean[:n_fuel])
    fig, ax = plt.subplots(figsize=figsize(1.0, 2.4))
    ax.plot(np.arange(n_fuel), true_mean[:n_fuel][order],
            marker="o", color=PALETTE["red"], label="true mean cost")
    ax.plot(np.arange(n_fuel), learned_mean[:n_fuel][order],
            marker="s", color=PALETTE["blue"], label=r"learned $\hat f$")
    ax.set_xticks(np.arange(n_fuel))
    ax.set_xticklabels([fuels[i] for i in order], rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Marginal cost (\\$/MWh)")
    ax.set_title("PJM-like merit order (sorted by truth)")
    ax.legend()
    save_figure(fig, "pjm_stack", out_dir)

    # ---- diurnal recovery for solar / wind / ccgt ----
    fig, ax = plt.subplots(figsize=figsize(1.0, 2.4))
    fuel_idx = {name: i for i, (name, _, _) in enumerate(PJM_FUELS)}
    show = ["wind", "solar", "ccgt"]
    palette = [PALETTE["green"], PALETTE["orange"], PALETTE["blue"]]
    for k, fuel in enumerate(show):
        i = fuel_idx[fuel]
        ax.plot(np.arange(true_table.shape[0]), true_table[:, i],
                color=palette[k], linestyle="--", alpha=0.7)
        ax.plot(np.arange(learned_table.shape[0]), learned_table[:, i],
                color=palette[k], linestyle="-", label=fuel)
    ax.set_xlabel("Hour-of-day stratum"); ax.set_ylabel("Marginal cost (\\$/MWh)")
    ax.set_title("Hour-resolved recovery (dashed=true, solid=learned)")
    ax.set_yscale("symlog")
    ax.legend()
    save_figure(fig, "pjm_recovery", out_dir)


if __name__ == "__main__":
    main()
