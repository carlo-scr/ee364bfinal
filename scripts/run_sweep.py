"""Recovery-curve and noise-sweep experiments (multi-seed, real baselines).

Sweeps either training-set size or observation noise, fits Diff-fcap (F&D),
Diff-full, and KKT-residual baselines, plots cosine recovery with shaded
mean +/- 1 std bands.
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
    METHOD_LABELS,
    StandardConfig,
    aggregate,
    build_dataset,
    run_diff_fcap,
    run_diff_full,
    run_kkt,
    standard_training,
)
from inverse_opf.plotting import PALETTE, apply_paper_style, figsize, save_figure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["size", "noise"], default="size")
    p.add_argument("--sizes", type=str, default="50,100,200,400")
    p.add_argument("--noises", type=str, default="0.1,0.3,0.7,1.5")
    p.add_argument("--seeds", type=str, default="0,1,2,3")
    p.add_argument("--steps", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sc_base = StandardConfig(n_buses=10, n_lines=14, n_val=100,
                             obs_noise=0.5, n_strata=24)

    out_dir = Path(f"outputs/sweep_{args.mode}"); out_dir.mkdir(parents=True, exist_ok=True)

    sweep = [int(x) for x in args.sizes.split(",")] if args.mode == "size" \
        else [float(x) for x in args.noises.split(",")]

    rows = []
    for x in sweep:
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            sc = StandardConfig(**{**sc_base.__dict__,
                                   "n_train": x if args.mode == "size" else 200,
                                   "obs_noise": x if args.mode == "noise" else sc_base.obs_noise})
            ds = build_dataset(seed, sc)
            for name, fn in [
                ("kkt", lambda d, s=sc: run_kkt(d, s)),
                ("diff_fcap", lambda d, s=sc: run_diff_fcap(d, s, standard_training(steps=args.steps))),
                ("diff_full", lambda d, s=sc: run_diff_full(d, s, standard_training(steps=args.steps))),
            ]:
                try:
                    out = fn(ds)
                except Exception as e:
                    print(f"  [{x} seed={seed}] {name} failed: {e}")
                    continue
                row = {"x": x, "seed": seed, "method": name,
                       "f_cos": out["f_cos"],
                       "f_spearman": out["f_spearman"],
                       "f_merit_acc": out["f_merit_acc"],
                       "val_nrmse_clean": out["val_nrmse_clean"]}
                rows.append(row)
                print(f"  x={x}  seed={seed}  {name:9s}  f_cos={row['f_cos']:.3f}"
                      f"  nrmse={row['val_nrmse_clean']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "rows.csv", index=False)
    agg = aggregate(rows, ["x", "method"], ["f_cos", "f_spearman", "f_merit_acc", "val_nrmse_clean"])
    agg.to_csv(out_dir / "agg.csv", index=False)

    apply_paper_style()
    color_map = {"kkt": PALETTE["orange"], "diff_fcap": PALETTE["green"],
                 "diff_full": PALETTE["blue"]}
    label_map = {"kkt": "KKT-residual", "diff_fcap": "Diff. (caps fixed)",
                 "diff_full": "Diff. (full)"}

    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.3), sharex=True)
    for method in ["kkt", "diff_fcap", "diff_full"]:
        sub = agg[agg["method"] == method].sort_values("x")
        if sub.empty: continue
        axes[0].plot(sub["x"], sub["f_cos_mean"], marker="o",
                     color=color_map[method], label=label_map[method])
        axes[0].fill_between(sub["x"],
                             (sub["f_cos_mean"] - sub["f_cos_std"]).clip(lower=0),
                             (sub["f_cos_mean"] + sub["f_cos_std"]).clip(upper=1),
                             color=color_map[method], alpha=0.18)
        axes[1].plot(sub["x"], sub["val_nrmse_clean_mean"], marker="o",
                     color=color_map[method], label=label_map[method])
        axes[1].fill_between(sub["x"],
                             sub["val_nrmse_clean_mean"] - sub["val_nrmse_clean_std"],
                             sub["val_nrmse_clean_mean"] + sub["val_nrmse_clean_std"],
                             color=color_map[method], alpha=0.18)

    xlabel = "Training-set size $T$" if args.mode == "size" else r"Observation noise $\sigma$"
    if args.mode == "size":
        axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_xlabel(xlabel); axes[0].set_ylabel(r"$\cos\angle(\hat f, f^\star)$")
    axes[0].set_ylim(0, 1.05); axes[0].set_title("(a) Cost recovery")
    axes[0].legend()
    axes[1].set_xlabel(xlabel); axes[1].set_ylabel("Clean dispatch NRMSE")
    axes[1].set_title("(b) Forward-prediction error")
    axes[1].legend()

    fig.tight_layout(pad=0.3)
    save_figure(fig, f"sweep_{args.mode}", out_dir)


if __name__ == "__main__":
    main()
