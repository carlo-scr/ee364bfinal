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
    p.add_argument("--steps", type=int, default=800)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    sc = StandardConfig(n_buses=8, n_lines=12, n_train=400, n_val=200,
                        obs_noise=0.15, n_strata=8)

    out_dir = Path("outputs/mef_validation"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fd_vs_ag_pairs = []   # for scatter plot
    fd_vs_ag_is_boundary = []  # per scatter point (per Jacobian entry)
    mef_true_all = []; mef_learned_all = []; mef_baseline_all = []
    BOUNDARY_TOL_MW = 0.5

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        ds = build_dataset(seed, sc)
        opf_true = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence,
                                               susceptance=ds.susceptance))

        # Identify generators that are *active in the data* (not trivially idle).
        # A gen that is g==0 for every sample contributes a zero Jacobian row that
        # FD and autograd both reproduce trivially, so it should not trigger the
        # boundary classification.
        gmax_np_seed = ds.true_gmax.cpu().numpy()
        with torch.no_grad():
            g_train_clean, _ = opf_true.solve(
                ds.d_train, ds.true_f_table[ds.strata_train],
                ds.true_gmax.unsqueeze(0).repeat(ds.d_train.shape[0], 1),
                ds.true_pmax.unsqueeze(0).repeat(ds.d_train.shape[0], 1))
        gens_active = (g_train_clean.cpu().numpy().max(axis=0)
                       > BOUNDARY_TOL_MW)  # bool [n_buses]

        # ---- finite difference vs autograd at the true parameters ----
        for k in range(min(args.n_points, ds.d_val.shape[0])):
            d0 = ds.d_val[k]
            f0 = ds.true_f_table[ds.strata_val[k]]
            # Solve once to classify each generator-at-point as interior or
            # boundary-adjacent. We classify per *row* of the Jacobian because
            # each row is the sensitivity of one generator; only rows whose
            # generator is near a bound suffer from kinks under FD perturbation.
            with torch.no_grad():
                g_star, _ = opf_true.solve(
                    d0.unsqueeze(0), f0.unsqueeze(0),
                    ds.true_gmax.unsqueeze(0), ds.true_pmax.unsqueeze(0))
            g_np = g_star.squeeze(0).cpu().numpy()
            gmax_np = ds.true_gmax.cpu().numpy()
            row_near_lower = (g_np < BOUNDARY_TOL_MW) & gens_active
            row_near_upper = (g_np > gmax_np - BOUNDARY_TOL_MW)
            row_is_boundary = row_near_lower | row_near_upper  # [n_buses]
            # Point-level summary: all (non-trivial) gens strictly interior.
            is_boundary = bool(row_is_boundary.any())

            jac_ag = jacobian_g_wrt_d(opf_true, d0, f0, ds.true_gmax, ds.true_pmax)
            jac_fd = finite_difference_jacobian(opf_true, d0, f0, ds.true_gmax, ds.true_pmax,
                                                eps=1e-3)
            jac_ag_np = jac_ag.cpu().numpy()
            jac_fd_np = jac_fd.cpu().numpy()
            # Per-point Frobenius relative error restricted to *active* rows.
            active_idx = np.where(gens_active)[0]
            num_pt = float(np.linalg.norm(jac_ag_np[active_idx] - jac_fd_np[active_idx]))
            den_pt = float(np.linalg.norm(jac_fd_np[active_idx]))
            point_err = num_pt / max(1e-9, den_pt)
            rows.append({"seed": seed, "point": k,
                         "fd_vs_ag_rel_err": point_err,
                         "is_boundary": is_boundary})
            # Per-entry arrays for the scatter (active gens only), coloured by
            # the per-row classification so kink-prone entries are highlighted.
            for i in range(jac_fd_np.shape[0]):
                if not gens_active[i]:
                    continue
                fd_vs_ag_pairs.append((jac_fd_np[i], jac_ag_np[i]))
                fd_vs_ag_is_boundary.append(
                    np.full(jac_fd_np.shape[1], bool(row_is_boundary[i])))

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
    is_boundary_arr = np.concatenate(fd_vs_ag_is_boundary)
    np.savez(out_dir / "arrays.npz",
             jac_fd=np.concatenate([p[0] for p in fd_vs_ag_pairs]),
             jac_ag=np.concatenate([p[1] for p in fd_vs_ag_pairs]),
             jac_is_boundary=is_boundary_arr,
             mef_true=mef_true, mef_learn=mef_learn, mef_fcap=mef_fcap)

    interior_mask = ~df["is_boundary"].to_numpy()
    boundary_mask = df["is_boundary"].to_numpy()
    # Per-row split using per-entry classification: each scatter point is one
    # Jacobian entry; entries inherit the row's boundary flag.
    fd_arr = np.concatenate([p[0] for p in fd_vs_ag_pairs])
    ag_arr = np.concatenate([p[1] for p in fd_vs_ag_pairs])
    row_interior = ~is_boundary_arr
    row_boundary = is_boundary_arr
    # Per-row rel err uses |ag-fd| / max(tol, |fd|) to avoid blow-ups on tiny-row.
    abs_err = np.abs(ag_arr - fd_arr)
    denom = np.maximum(1e-3, np.abs(fd_arr))
    row_rel_err = abs_err / denom
    summary = {
        "fd_vs_ag_max_rel_err": float(df["fd_vs_ag_rel_err"].max()),
        "fd_vs_ag_mean_rel_err": float(df["fd_vs_ag_rel_err"].mean()),
        "fd_vs_ag_mean_rel_err_interior": (
            float(df.loc[interior_mask, "fd_vs_ag_rel_err"].mean())
            if interior_mask.any() else float("nan")),
        "fd_vs_ag_mean_rel_err_boundary": (
            float(df.loc[boundary_mask, "fd_vs_ag_rel_err"].mean())
            if boundary_mask.any() else float("nan")),
        "frac_boundary": float(boundary_mask.mean()),
        "n_points_total": int(len(df)),
        "n_points_interior": int(interior_mask.sum()),
        "n_points_boundary": int(boundary_mask.sum()),
        "boundary_tol_mw": BOUNDARY_TOL_MW,
        # Per-row (per-generator-at-point) split: the row of the Jacobian is the
        # natural unit because each row corresponds to one generator and only
        # rows whose generator is near a bound suffer from kinks.
        "row_rel_err_mean_interior": (
            float(row_rel_err[row_interior].mean()) if row_interior.any() else float("nan")),
        "row_rel_err_mean_boundary": (
            float(row_rel_err[row_boundary].mean()) if row_boundary.any() else float("nan")),
        "row_frac_boundary": float(row_boundary.mean()),
        "n_rows_total": int(len(row_rel_err)),
        "n_rows_interior": int(row_interior.sum()),
        "n_rows_boundary": int(row_boundary.sum()),
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
    # Boundary-adjacent (red) under interior (blue) so red kinks are visible.
    axes[0].scatter(fd[~is_boundary_arr], ag[~is_boundary_arr],
                    s=4, color=PALETTE["blue"], alpha=0.5, edgecolors="none",
                    label=f"interior (n={int((~is_boundary_arr).sum())})")
    axes[0].scatter(fd[is_boundary_arr], ag[is_boundary_arr],
                    s=4, color=PALETTE["red"], alpha=0.5, edgecolors="none",
                    label=f"boundary (n={int(is_boundary_arr.sum())})")
    axes[0].plot([lo, hi], [lo, hi], color="black", linewidth=0.6, linestyle="--")
    axes[0].set_xlabel(r"Finite-difference $\partial g^\star/\partial d$")
    axes[0].set_ylabel(r"Autograd $\partial g^\star/\partial d$")
    axes[0].set_title("(a) Jacobian validation")
    axes[0].legend(loc="upper left", fontsize=7)

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
