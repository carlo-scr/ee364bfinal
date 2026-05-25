"""Headline experiment: methods comparison.

Runs all six methods (Ridge, MLP, KKT-residual, Diff-fcap, Diff-full,
Diff-stratified) across multiple seeds on a fixed synthetic configuration.
Reports parameter recovery (cosine, Spearman, merit-order accuracy) and clean
val RMSE with mean +/- std bands.

Outputs:
  outputs/methods_comparison/results.csv
  outputs/methods_comparison/agg.csv
  paper/figures/methods_comparison.pdf  (and .png mirror)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    METHOD_LABELS,
    PAPER_FIG_DIR,
    StandardConfig,
    aggregate,
    build_dataset,
    run_diff_fcap,
    run_diff_full,
    run_diff_strat,
    run_kkt,
    run_mlp,
    run_ridge,
    standard_training,
)
from inverse_opf.plotting import PALETTE, apply_paper_style, figsize, save_figure

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--n_train", type=int, default=200)
    p.add_argument("--obs_noise", type=float, default=0.5)
    p.add_argument("--n_strata", type=int, default=24)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.quick:
        seeds = seeds[:2]
    sc = StandardConfig(
        n_buses=10, n_lines=14,
        n_train=args.n_train, n_val=100,
        obs_noise=args.obs_noise, n_strata=args.n_strata,
    )

    out_dir = Path("outputs/methods_comparison"); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        ds = build_dataset(seed, sc)
        for name, fn in [
            ("ridge", run_ridge),
            ("mlp", run_mlp),
            ("kkt", run_kkt),
        ]:
            t0 = time.time()
            try:
                out = fn(ds, sc)
            except Exception as e:
                print(f"[seed {seed}] {name} FAILED: {e}")
                continue
            out = {**out, "seed": seed, "elapsed_s": time.time() - t0}
            rows.append(out)
            print(f"  seed {seed}  {name:11s}  rmse_clean={out.get('val_rmse_clean', float('nan')):.3f}"
                  f"  f_cos={out.get('f_cos', float('nan')):.3f}  ({out['elapsed_s']:.1f}s)")
        for name, fn in [
            ("diff_fcap", lambda ds, sc: run_diff_fcap(ds, sc, standard_training(steps=args.steps, lr=args.lr))),
            ("diff_full", lambda ds, sc: run_diff_full(ds, sc, standard_training(steps=args.steps, lr=args.lr))),
            ("diff_strat", lambda ds, sc: run_diff_strat(ds, sc, standard_training(steps=args.steps, lr=args.lr))),
        ]:
            t0 = time.time()
            try:
                out = fn(ds, sc)
            except Exception as e:
                print(f"[seed {seed}] {name} FAILED: {e}")
                continue
            # Drop f_table to keep CSV flat.
            out_flat = {k: v for k, v in out.items() if k != "f_table"}
            out_flat = {**out_flat, "seed": seed, "elapsed_s": time.time() - t0}
            rows.append(out_flat)
            print(f"  seed {seed}  {name:11s}  rmse_clean={out_flat['val_rmse_clean']:.3f}"
                  f"  f_cos={out_flat['f_cos']:.3f}  best_step={out_flat.get('best_step', '-')}  "
                  f"({out_flat['elapsed_s']:.1f}s)")

    # Drop array columns for CSV.
    rows_csv = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows]
    df = pd.DataFrame(rows_csv)
    df.to_csv(out_dir / "results.csv", index=False)

    value_cols = ["val_rmse", "val_rmse_clean", "val_nrmse_clean",
                  "f_cos", "f_spearman", "f_merit_acc", "gmax_cos", "pmax_cos"]
    agg = aggregate(rows_csv, ["method"], value_cols)
    agg.to_csv(out_dir / "agg.csv", index=False)
    print("\nAggregate (mean):")
    print(agg.to_string(index=False))

    _plot_main(agg, out_dir)
    _plot_recovery_bars(agg, out_dir)


def _plot_main(agg: pd.DataFrame, out_dir: Path):
    apply_paper_style()
    method_order = ["ridge", "mlp", "kkt", "diff_fcap", "diff_full", "diff_strat"]
    agg = agg.set_index("method").reindex([m for m in method_order if m in agg["method"].values
                                           or m in agg.index])
    methods = [m for m in method_order if m in agg.index]
    labels = [METHOD_LABELS[m] for m in methods]
    nrmse_mean = [agg.loc[m, "val_nrmse_clean_mean"] for m in methods]
    nrmse_std  = [agg.loc[m, "val_nrmse_clean_std"]  for m in methods]
    fcos_mean  = [agg.loc[m, "f_cos_mean"] for m in methods]
    fcos_std   = [agg.loc[m, "f_cos_std"]  for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.3))
    colors = [PALETTE["gray"], PALETTE["gray"], PALETTE["orange"],
              PALETTE["green"], PALETTE["blue"], PALETTE["purple"]]
    x = np.arange(len(methods))
    axes[0].bar(x, nrmse_mean, yerr=nrmse_std, color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylabel(r"Clean dispatch NRMSE")
    axes[0].set_title("(a) Forward-prediction error")

    fcos_mean_plot = [v if not np.isnan(v) else 0 for v in fcos_mean]
    fcos_std_plot = [v if not np.isnan(v) else 0 for v in fcos_std]
    axes[1].bar(x, fcos_mean_plot, yerr=fcos_std_plot, color=colors,
                edgecolor="black", linewidth=0.4)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylabel(r"$\cos\angle(\hat f, f^\star)$")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("(b) Cost-vector recovery")
    # Mark NaN bars (regression baselines).
    for i, v in enumerate(fcos_mean):
        if np.isnan(v):
            axes[1].text(i, 0.02, "n/a", ha="center", va="bottom", fontsize=6, color="white")

    fig.tight_layout(pad=0.4)
    save_figure(fig, "methods_comparison", out_dir)


def _plot_recovery_bars(agg: pd.DataFrame, out_dir: Path):
    apply_paper_style()
    method_order = ["kkt", "diff_fcap", "diff_full", "diff_strat"]
    agg = agg.set_index("method")
    methods = [m for m in method_order if m in agg.index]
    metrics = [("f_cos_mean", "f_cos_std", r"$\cos$"),
               ("f_spearman_mean", "f_spearman_std", "Spearman"),
               ("f_merit_acc_mean", "f_merit_acc_std", "Merit-order acc."),
               ("gmax_cos_mean", "gmax_cos_std", r"$\cos\,(g_{\max})$")]
    fig, ax = plt.subplots(figsize=figsize(2.0, 2.4))
    width = 0.18
    x = np.arange(len(metrics))
    colors = [PALETTE["orange"], PALETTE["green"], PALETTE["blue"], PALETTE["purple"]]
    for i, m in enumerate(methods):
        means = [agg.loc[m, mm] for mm, _, _ in metrics]
        stds  = [agg.loc[m, ss] for _, ss, _ in metrics]
        ax.bar(x + (i - 1.5) * width, means, width=width, yerr=stds,
               color=colors[i], edgecolor="black", linewidth=0.4,
               label=METHOD_LABELS[m])
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, _, lbl in metrics])
    ax.set_ylabel("Recovery score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Parameter recovery across methods (mean $\\pm$ 1 std)")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout(pad=0.3)
    save_figure(fig, "methods_recovery", out_dir)


if __name__ == "__main__":
    main()
