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
    p.add_argument("--tight_caps", action="store_true", default=True,
                   help="Sample gmax,i ~ U[0.8*dmean, 1.5*dmean] so generators bind frequently.")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sc = StandardConfig(n_buses=args.n_buses, n_lines=args.n_lines,
                        n_train=300, n_val=100, obs_noise=0.5, n_strata=24)
    # Tight per-generator capacities: U[0.8 * mean_demand, 1.5 * mean_demand]
    # where mean_demand is the per-bus mean. Total capacity ~= 1.15x total
    # demand, so many generators bind their cap in many samples and the
    # identifiability score s_i has meaningful variance.
    gmax_abs = (0.8 * sc.demand_mean, 1.5 * sc.demand_mean) if args.tight_caps else None

    out_dir = Path("outputs/identifiability"); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        from inverse_opf.synthetic import make_synthetic_dataset
        ds = make_synthetic_dataset(
            n_buses=sc.n_buses, n_lines=sc.n_lines,
            n_train=sc.n_train, n_val=sc.n_val,
            demand_mean=sc.demand_mean, demand_std=sc.demand_std,
            observation_noise_std=sc.obs_noise,
            true_f_range=(sc.f_min, sc.f_max),
            gmax_scale=(sc.gmin_scale, sc.gmax_scale),
            pmax_scale=(sc.pmin_scale, sc.pmax_scale),
            n_strata=sc.n_strata, seed=seed,
            physics=sc.physics, diurnal_amp=sc.diurnal_amp,
            diurnal_cost_amp=sc.diurnal_cost_amp,
            gmax_abs_range=gmax_abs,
        )
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

    # Spearman correlation of error vs id score (compute first so we can annotate).
    from scipy.stats import spearmanr
    rho_f, _ = spearmanr(df["id_score"], df["f_rel_err"])
    rho_g, _ = spearmanr(df["id_score"], df["gmax_rel_err"])

    apply_paper_style()

    def _joint_panel(fig, gs_cell, xs, ys, color, ylabel, title, rho):
        """Scatter with marginal histograms and a 1st-order regression line."""
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_cell,
                                     width_ratios=[4, 1], height_ratios=[1, 4],
                                     wspace=0.05, hspace=0.05)
        ax_x = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_y = fig.add_subplot(gs[1, 1])
        ax_main.scatter(xs, ys, c=color, s=12, alpha=0.7, edgecolors="none")
        # Regression line (least-squares fit).
        if len(xs) >= 2 and np.std(xs) > 1e-9:
            slope, intercept = np.polyfit(xs, ys, 1)
            xfit = np.linspace(xs.min(), xs.max(), 50)
            ax_main.plot(xfit, slope * xfit + intercept,
                         color=PALETTE["red"], linewidth=1.5,
                         label=fr"fit, $\rho={rho:.2f}$")
            ax_main.legend(loc="upper right", fontsize=7)
        ax_main.set_xlabel("Identifiability score $s_i$")
        ax_main.set_ylabel(ylabel)
        # Marginal histograms.
        ax_x.hist(xs, bins=15, color=color, alpha=0.7, edgecolor="none")
        ax_y.hist(ys, bins=15, orientation="horizontal",
                  color=color, alpha=0.7, edgecolor="none")
        ax_x.set_xticks([]); ax_x.set_yticks([])
        ax_y.set_xticks([]); ax_y.set_yticks([])
        ax_x.set_title(title, fontsize=9)
        for spine in ("top", "right"):
            ax_x.spines[spine].set_visible(False)
            ax_y.spines[spine].set_visible(False)
        return ax_main

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize(2.0, 2.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    _joint_panel(fig, gs[0, 0], df["id_score"].to_numpy(),
                 df["f_rel_err"].to_numpy(), PALETTE["blue"],
                 r"$|\hat f_i - f_i^\star| / |f_i^\star|$",
                 "(a) Cost-vector identifiability", rho_f)
    _joint_panel(fig, gs[0, 1], df["id_score"].to_numpy(),
                 df["gmax_rel_err"].to_numpy(), PALETTE["green"],
                 r"$|\hat g_{\max,i} - g_{\max,i}^\star| / g_{\max,i}^\star$",
                 "(b) Capacity identifiability", rho_g)
    fig.tight_layout(pad=0.3)
    save_figure(fig, "identifiability", out_dir)

    summary = {
        "spearman_id_vs_f_err": float(rho_f),
        "spearman_id_vs_gmax_err": float(rho_g),
        "n_generators": int(df.shape[0]),
        "frac_with_s_lt_0p7": float((df["id_score"] < 0.7).mean()),
        "frac_with_s_lt_0p5": float((df["id_score"] < 0.5).mean()),
        "id_score_mean": float(df["id_score"].mean()),
        "id_score_std": float(df["id_score"].std()),
        "tight_caps": bool(args.tight_caps),
    }
    Path(out_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2))
    print("Spearman(id_score, |f err|) =", rho_f)
    print("Spearman(id_score, |gmax err|) =", rho_g)
    print(f"frac with s_i<0.7 = {summary['frac_with_s_lt_0p7']:.2f}; "
          f"mean s_i = {summary['id_score_mean']:.2f}")


if __name__ == "__main__":
    main()
