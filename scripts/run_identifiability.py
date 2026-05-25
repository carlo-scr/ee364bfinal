"""Identifiability experiment.

Hypothesis: per-generator parameter recovery error correlates inversely with
the empirical fraction of samples in which the generator is *strictly
interior* (0 < g_i < gmax_i). Generators that are always idle or always
binding are unidentifiable up to a sign by the inverse OPF.

We run on multiple seeds, fit the differentiable full model, then plot
per-generator |f_hat - f_true| / f_true vs. identifiability score.

Outputs:
  outputs/identifiability/scores.csv
  paper/figures/identifiability.pdf
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
from _common import StandardConfig, build_dataset, run_diff_full, standard_training
from inverse_opf.metrics import identifiability_score, per_generator_relative_error
from inverse_opf.plotting import PALETTE, apply_paper_style, figsize, save_figure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--n_buses", type=int, default=12)
    p.add_argument("--n_lines", type=int, default=18)
    p.add_argument("--steps", type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sc = StandardConfig(n_buses=args.n_buses, n_lines=args.n_lines,
                        n_train=300, n_val=100, obs_noise=0.5, n_strata=24)

    out_dir = Path("outputs/identifiability"); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        ds = build_dataset(seed, sc)
        # Identifiability score on TRAIN (the fitting data).
        # Use the *true* dispatch as the activity criterion (we synthesize it
        # below by re-solving the forward problem; using g_obs would be noisy).
        from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
        opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance))
        f_t = ds.true_f_table[ds.strata_train]
        gmax_b = ds.true_gmax.unsqueeze(0).repeat(ds.d_train.shape[0], 1)
        pmax_b = ds.true_pmax.unsqueeze(0).repeat(ds.d_train.shape[0], 1)
        with torch.no_grad():
            g_train_clean, _ = opf.solve(ds.d_train, f_t, gmax_b, pmax_b)
        score = identifiability_score(g_train_clean, ds.true_gmax)

        out = run_diff_full(ds, sc, standard_training(steps=args.steps))
        rel_err = per_generator_relative_error(out["f"], ds.true_f.numpy())
        gmax_err = per_generator_relative_error(out["gmax"], ds.true_gmax.numpy())
        for i in range(sc.n_buses):
            rows.append({
                "seed": seed, "bus": i,
                "id_score": float(score[i]),
                "f_true": float(ds.true_f[i].item()),
                "f_hat": float(out["f"][i]),
                "f_rel_err": float(rel_err[i]),
                "gmax_true": float(ds.true_gmax[i].item()),
                "gmax_hat": float(out["gmax"][i]),
                "gmax_rel_err": float(gmax_err[i]),
            })
        print(f"[seed {seed}] mean id_score={score.mean():.2f}, mean f_rel_err={rel_err.mean():.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scores.csv", index=False)

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.3), sharex=True)

    # f recovery
    axes[0].scatter(df["id_score"], df["f_rel_err"],
                    c=PALETTE["blue"], s=10, alpha=0.7, edgecolors="none")
    # bin-mean overlay
    bins = np.linspace(0, 1, 8)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (df["id_score"] >= lo) & (df["id_score"] < hi)
        means.append(df.loc[sel, "f_rel_err"].mean() if sel.sum() else np.nan)
    axes[0].plot(bin_centers, means, color=PALETTE["red"], marker="o",
                 markersize=4, label="bin mean")
    axes[0].set_ylabel(r"$|\hat f_i - f_i^\star| / |f_i^\star|$")
    axes[0].set_xlabel("Identifiability score $s_i$")
    axes[0].set_title("(a) Cost-vector identifiability")
    axes[0].legend()

    # gmax recovery
    axes[1].scatter(df["id_score"], df["gmax_rel_err"],
                    c=PALETTE["green"], s=10, alpha=0.7, edgecolors="none")
    means_g = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (df["id_score"] >= lo) & (df["id_score"] < hi)
        means_g.append(df.loc[sel, "gmax_rel_err"].mean() if sel.sum() else np.nan)
    axes[1].plot(bin_centers, means_g, color=PALETTE["red"], marker="o",
                 markersize=4, label="bin mean")
    axes[1].set_ylabel(r"$|\hat g_{\max,i} - g_{\max,i}^\star| / g_{\max,i}^\star$")
    axes[1].set_xlabel("Identifiability score $s_i$")
    axes[1].set_title("(b) Capacity identifiability")
    axes[1].legend()

    fig.tight_layout(pad=0.3)
    save_figure(fig, "identifiability", out_dir)

    # Spearman correlation of error vs id score.
    from scipy.stats import spearmanr
    rho_f, _ = spearmanr(df["id_score"], df["f_rel_err"])
    rho_g, _ = spearmanr(df["id_score"], df["gmax_rel_err"])
    summary = {
        "spearman_id_vs_f_err": float(rho_f),
        "spearman_id_vs_gmax_err": float(rho_g),
        "n_generators": int(df.shape[0]),
    }
    Path(out_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2))
    print("Spearman(id_score, |f err|) =", rho_f)
    print("Spearman(id_score, |gmax err|) =", rho_g)


if __name__ == "__main__":
    main()
