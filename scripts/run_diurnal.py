"""Stratified vs non-stratified inverse OPF on diurnally-varying merit order.

We synthesize data where some generators are cheap during the day and
expensive at night (large diurnal cost amplitude). A static-cost model is
*structurally* unable to fit this, while the stratified model with cycle
Laplacian smoothing should recover the diurnal pattern.

Outputs:
  outputs/diurnal/diurnal.csv
  paper/figures/diurnal_heatmap.pdf
  paper/figures/diurnal_curves.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    StandardConfig,
    build_dataset,
    run_diff_full,
    run_diff_strat,
    standard_training,
)
from inverse_opf.metrics import cosine_recovery
from inverse_opf.plotting import PALETTE, apply_paper_style, figsize, save_figure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lap", type=float, default=0.02)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sc = StandardConfig(
        n_buses=8, n_lines=12,
        n_train=600, n_val=200,
        obs_noise=0.2, n_strata=24,
        diurnal_amp=0.30, diurnal_cost_amp=0.6,   # large diurnal cost variation
    )

    out_dir = Path("outputs/diurnal"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    f_table_estimates = None
    f_table_true = None
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        ds = build_dataset(seed, sc)

        out_static = run_diff_full(ds, sc, standard_training(steps=args.steps))
        out_strat = run_diff_strat(ds, sc, standard_training(
            steps=args.steps, laplacian_weight=args.lap))

        # Compare static estimate vs stratified mean against true mean.
        # Compare full f-table recovery for stratified.
        rows.append({"seed": seed, "method": "static",
                     "f_cos_mean": out_static["f_cos"],
                     "val_rmse_clean": out_static["val_rmse_clean"],
                     "val_nrmse_clean": out_static["val_nrmse_clean"]})
        rows.append({"seed": seed, "method": "stratified",
                     "f_cos_mean": out_strat["f_cos"],
                     "val_rmse_clean": out_strat["val_rmse_clean"],
                     "val_nrmse_clean": out_strat["val_nrmse_clean"],
                     "f_table_cos": out_strat["f_table_cos"]})

        if f_table_estimates is None:
            f_table_estimates = out_strat["f_table"]
            f_table_true = ds.true_f_table.cpu().numpy()
            f_static_estimate = out_static["f"]

    df = pd.DataFrame(rows); df.to_csv(out_dir / "diurnal.csv", index=False)
    print(df.to_string(index=False))

    # ---- post-hoc affine calibration (cost-level identifiability) ----
    # Fit single (alpha,beta) by OLS over all (hour, gen) entries.
    flat_hat = f_table_estimates.reshape(-1)
    flat_true = f_table_true.reshape(-1)
    A = np.vstack([flat_hat, np.ones_like(flat_hat)]).T
    alpha, beta = np.linalg.lstsq(A, flat_true, rcond=None)[0]
    f_table_strat_cal = alpha * f_table_estimates + beta
    f_static_cal = alpha * f_static_estimate + beta
    print(f"affine calibration: alpha={alpha:.3f}, beta={beta:.3f}")
    cal_rmse = float(np.sqrt(((f_table_strat_cal - f_table_true) ** 2).mean()))
    static_rmse = float(np.sqrt(
        ((np.tile(f_static_cal, (sc.n_strata, 1)) - f_table_true) ** 2).mean()))
    print(f"calibrated stratified RMSE: {cal_rmse:.3f}, static RMSE: {static_rmse:.3f}")

    # ---- save raw arrays for paper-figure regeneration ----
    np.savez(out_dir / "arrays.npz",
             f_table_true=f_table_true,
             f_table_strat=f_table_estimates,
             f_table_strat_cal=f_table_strat_cal,
             f_static=f_static_estimate,
             f_static_cal=f_static_cal,
             affine_alpha=alpha, affine_beta=beta,
             calibrated_rmse=cal_rmse, static_calibrated_rmse=static_rmse)

    # ---- heatmap: true vs learned (calibrated) f-table ----
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize(2.0, 2.0), sharey=True)
    vmin = min(f_table_true.min(), f_table_strat_cal.min())
    vmax = max(f_table_true.max(), f_table_strat_cal.max())
    im0 = axes[0].imshow(f_table_true.T, aspect="auto", cmap="viridis",
                         vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("(a) True $f^{(s)}$"); axes[0].set_xlabel("Hour-of-day stratum")
    axes[0].set_ylabel("Generator")
    im1 = axes[1].imshow(f_table_strat_cal.T, aspect="auto", cmap="viridis",
                         vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title(r"(b) Stratified $\alpha\hat f^{(s)}+\beta$")
    axes[1].set_xlabel("Hour-of-day stratum")
    fig.colorbar(im1, ax=[axes[0], axes[1]], shrink=0.85, pad=0.02, label="cost")

    # static model just reproduces a constant per generator across hours:
    static_table = np.tile(f_static_cal, (sc.n_strata, 1))
    im2 = axes[2].imshow(static_table.T, aspect="auto", cmap="viridis",
                         vmin=vmin, vmax=vmax, origin="lower")
    axes[2].set_title(r"(c) Static $\alpha\hat f+\beta$")
    axes[2].set_xlabel("Hour-of-day stratum")

    save_figure(fig, "diurnal_heatmap", out_dir)

    # ---- recovery curves: per-bus learned cost vs. hour ----
    fig, ax = plt.subplots(figsize=figsize(1.0, 2.2))
    n_show = min(4, sc.n_buses)
    palette = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["purple"]]
    for k in range(n_show):
        ax.plot(np.arange(sc.n_strata), f_table_true[:, k],
                color=palette[k], linestyle="--", alpha=0.7,
                label=f"true gen {k}" if k == 0 else None)
        ax.plot(np.arange(sc.n_strata), f_table_strat_cal[:, k],
                color=palette[k], linestyle="-",
                label=f"learned gen {k}" if k == 0 else None)
    ax.set_xlabel("Hour-of-day stratum"); ax.set_ylabel(r"Cost ($\alpha\hat f+\beta$)")
    ax.set_title("Recovered diurnal cost profile (4 generators)")
    ax.legend(ncol=2, loc="upper right", fontsize=6)
    save_figure(fig, "diurnal_curves", out_dir)


if __name__ == "__main__":
    main()
