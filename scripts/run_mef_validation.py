"""Sensitivity / MEF validation:
  1) Compare autograd Jacobian dg*/dd against central finite differences;
  2) Show MEF accuracy of the learned model vs F&D-style baseline.

Outputs:
  outputs/mef_validation/results.csv
  paper/figures/mef_validation.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import StandardConfig, build_dataset, run_diff_fcap, run_diff_full, standard_training
from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
from inverse_opf.metrics import cosine_recovery, rmse
from inverse_opf.plotting import PALETTE, apply_paper_style, figsize, save_figure
from inverse_opf.sensitivity import (
    finite_difference_jacobian,
    jacobian_g_wrt_d,
    marginal_emission_factors,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--n_points", type=int, default=12)
    p.add_argument("--steps", type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sc = StandardConfig(n_buses=8, n_lines=12, n_train=200, n_val=100,
                        obs_noise=0.5, n_strata=8)

    out_dir = Path("outputs/mef_validation"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fd_vs_ag_pairs = []   # for scatter plot
    mef_true_all = []; mef_learned_all = []; mef_baseline_all = []

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        ds = build_dataset(seed, sc)
        opf_true = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence,
                                               susceptance=ds.susceptance))

        # ---- finite difference vs autograd at the true parameters ----
        for k in range(min(args.n_points, ds.d_val.shape[0])):
            d0 = ds.d_val[k]
            f0 = ds.true_f_table[ds.strata_val[k]]
            jac_ag = jacobian_g_wrt_d(opf_true, d0, f0, ds.true_gmax, ds.true_pmax)
            jac_fd = finite_difference_jacobian(opf_true, d0, f0, ds.true_gmax, ds.true_pmax,
                                                eps=5e-3)
            err = float(torch.norm(jac_ag - jac_fd) / max(1e-9, float(torch.norm(jac_fd))))
            rows.append({"seed": seed, "point": k, "fd_vs_ag_rel_err": err})
            fd_vs_ag_pairs.append((jac_fd.cpu().numpy().reshape(-1),
                                   jac_ag.cpu().numpy().reshape(-1)))

        # ---- MEF accuracy: learned vs F&D-style baseline ----
        out_full = run_diff_full(ds, sc, standard_training(steps=args.steps))
        out_fcap = run_diff_fcap(ds, sc, standard_training(steps=args.steps))

        carbon = torch.rand(sc.n_buses) * 0.5 + 0.2  # tonCO2/MWh per bus
        for k in range(min(args.n_points, ds.d_val.shape[0])):
            d0 = ds.d_val[k]
            f_true = ds.true_f_table[ds.strata_val[k]]
            jac_true = jacobian_g_wrt_d(opf_true, d0, f_true, ds.true_gmax, ds.true_pmax)
            mef_true = marginal_emission_factors(jac_true, carbon)

            f_full = torch.tensor(out_full["f"], dtype=torch.float32)
            gmax_full = torch.tensor(out_full["gmax"], dtype=torch.float32)
            pmax_full = torch.tensor(out_full["pmax"], dtype=torch.float32)
            opf_learn = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence,
                                                    susceptance=ds.susceptance))
            jac_learn = jacobian_g_wrt_d(opf_learn, d0, f_full, gmax_full, pmax_full)
            mef_learn = marginal_emission_factors(jac_learn, carbon)

            f_fcap = torch.tensor(out_fcap["f"], dtype=torch.float32)
            jac_fcap = jacobian_g_wrt_d(opf_learn, d0, f_fcap, ds.true_gmax, ds.true_pmax)
            mef_fcap = marginal_emission_factors(jac_fcap, carbon)

            mef_true_all.append(mef_true.cpu().numpy())
            mef_learned_all.append(mef_learn.cpu().numpy())
            mef_baseline_all.append(mef_fcap.cpu().numpy())

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)

    mef_true = np.stack(mef_true_all); mef_learn = np.stack(mef_learned_all)
    mef_fcap = np.stack(mef_baseline_all)
    np.savez(out_dir / "arrays.npz",
             jac_fd=np.concatenate([p[0] for p in fd_vs_ag_pairs]),
             jac_ag=np.concatenate([p[1] for p in fd_vs_ag_pairs]),
             mef_true=mef_true, mef_learn=mef_learn, mef_fcap=mef_fcap)
    summary = {
        "fd_vs_ag_max_rel_err": float(df["fd_vs_ag_rel_err"].max()),
        "fd_vs_ag_mean_rel_err": float(df["fd_vs_ag_rel_err"].mean()),
        "mef_full_rmse": float(np.sqrt(np.mean((mef_learn - mef_true) ** 2))),
        "mef_full_cos": float(cosine_recovery(mef_learn.reshape(-1),
                                              mef_true.reshape(-1))),
        "mef_fcap_rmse": float(np.sqrt(np.mean((mef_fcap - mef_true) ** 2))),
        "mef_fcap_cos": float(cosine_recovery(mef_fcap.reshape(-1),
                                              mef_true.reshape(-1))),
    }
    Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.3))
    fd = np.concatenate([p[0] for p in fd_vs_ag_pairs])
    ag = np.concatenate([p[1] for p in fd_vs_ag_pairs])
    lo = float(min(fd.min(), ag.min()))
    hi = float(max(fd.max(), ag.max()))
    axes[0].scatter(fd, ag, s=4, color=PALETTE["blue"], alpha=0.5, edgecolors="none")
    axes[0].plot([lo, hi], [lo, hi], color=PALETTE["red"], linewidth=0.8)
    axes[0].set_xlabel(r"Finite-difference $\partial g^\star/\partial d$")
    axes[0].set_ylabel(r"Autograd $\partial g^\star/\partial d$")
    axes[0].set_title("(a) Jacobian validation")

    # MEF scatter: learned vs true (and baseline overlay).
    axes[1].scatter(mef_true.reshape(-1), mef_fcap.reshape(-1),
                    s=4, color=PALETTE["gray"], alpha=0.5, edgecolors="none",
                    label="caps fixed (F&D)")
    axes[1].scatter(mef_true.reshape(-1), mef_learn.reshape(-1),
                    s=4, color=PALETTE["blue"], alpha=0.6, edgecolors="none",
                    label="diff. full")
    lo = float(mef_true.min()); hi = float(mef_true.max())
    axes[1].plot([lo, hi], [lo, hi], color=PALETTE["red"], linewidth=0.8)
    axes[1].set_xlabel("True MEF (tCO\u2082/MWh)")
    axes[1].set_ylabel("Estimated MEF")
    axes[1].set_title("(b) Downstream MEF accuracy")
    axes[1].legend()

    fig.tight_layout(pad=0.3)
    save_figure(fig, "mef_validation", out_dir)


if __name__ == "__main__":
    main()
