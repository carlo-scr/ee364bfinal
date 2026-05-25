"""Regenerate every paper figure from saved CSV / NPZ artefacts.

Run AFTER all experiment scripts have populated outputs/. This script
produces a coherent IEEE-ready figure set in paper/figures/ using the
unified navy palette defined in inverse_opf.plotting.

Usage:
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inverse_opf.plotting import (  # noqa: E402
    DIVERGING_CMAP,
    METHOD_COLOR,
    METHOD_LABEL,
    NAVY_CMAP,
    PALETTE,
    _ordered_blues,
    apply_paper_style,
    figsize,
    save_figure,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PAPER = ROOT / "paper" / "figures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bar_with_err(ax, xs, means, stds, color, label=None, width=0.8):
    bars = ax.bar(xs, means, width=width, color=color,
                  edgecolor=PALETTE["navy"], linewidth=0.4, label=label)
    ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor=PALETTE["navy"],
                elinewidth=0.6, capsize=1.6, capthick=0.6)
    return bars


def _annotate_bars(ax, xs, means, fmt="{:.2f}", dy=0.01, color=PALETTE["deep"]):
    ymax = ax.get_ylim()[1]
    for x, m in zip(xs, means):
        if not np.isfinite(m):
            continue
        ax.text(x, m + dy * ymax, fmt.format(m), ha="center", va="bottom",
                fontsize=6, color=color)


# ---------------------------------------------------------------------------
# 1. Methods comparison
# ---------------------------------------------------------------------------
def fig_methods_comparison():
    df = pd.read_csv(OUT / "methods_comparison" / "results.csv")
    order = ["ridge", "mlp", "kkt", "diff_fcap", "diff_full", "diff_strat"]
    agg = df.groupby("method").agg({
        "val_nrmse_clean": ["mean", "std"],
        "f_cos": ["mean", "std"],
        "f_merit_acc": ["mean", "std"],
    })
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reindex(order)

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize(2.0, 2.0))
    xs = np.arange(len(order))
    labels = [METHOD_LABEL[m] for m in order]
    colors = [METHOD_COLOR[m] for m in order]

    # (a) Forward-prediction error
    ax = axes[0]
    _bar_with_err(ax, xs, agg["val_nrmse_clean_mean"].values,
                  agg["val_nrmse_clean_std"].values, colors)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Clean dispatch NRMSE")
    ax.set_title("(a) Forward-prediction error")

    # (b) Cost recovery
    ax = axes[1]
    cos = np.array(agg["f_cos_mean"].values, dtype=float)
    cos_std = np.array(agg["f_cos_std"].values, dtype=float)
    cos[:2] = np.nan; cos_std[:2] = np.nan  # ridge / mlp don't define f
    _bar_with_err(ax, xs, np.where(np.isnan(cos), 0, cos),
                  np.where(np.isnan(cos_std), 0, cos_std), colors)
    for i in range(2):
        ax.text(xs[i], 0.02, "n/a", ha="center", va="bottom", fontsize=7,
                color=PALETTE["gray"], style="italic")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(r"$\cos\angle(\hat f, f^\star)$")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Cost-vector recovery")

    # (c) Merit-order accuracy
    ax = axes[2]
    merit = np.array(agg["f_merit_acc_mean"].values, dtype=float)
    merit_std = np.array(agg["f_merit_acc_std"].values, dtype=float)
    merit[:2] = np.nan; merit_std[:2] = np.nan
    _bar_with_err(ax, xs, np.where(np.isnan(merit), 0, merit),
                  np.where(np.isnan(merit_std), 0, merit_std), colors)
    for i in range(2):
        ax.text(xs[i], 0.02, "n/a", ha="center", va="bottom", fontsize=7,
                color=PALETTE["gray"], style="italic")
    ax.axhline(0.5, color=PALETTE["accent"], linewidth=0.6, linestyle="--",
               alpha=0.7, zorder=0)
    ax.text(len(order) - 0.5, 0.515, "chance", color=PALETTE["accent"],
            fontsize=6, ha="right", va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Merit-order accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Pairwise merit-order accuracy")

    fig.tight_layout(pad=0.4)
    save_figure(fig, "methods_comparison", OUT / "methods_comparison")


def fig_methods_recovery():
    df = pd.read_csv(OUT / "methods_comparison" / "results.csv")
    df = df[df["method"].isin(["kkt", "diff_fcap", "diff_full", "diff_strat"])]
    order = ["kkt", "diff_fcap", "diff_full", "diff_strat"]
    metrics = [("f_cos",      r"$\cos\angle(\hat f, f^\star)$"),
               ("f_spearman", r"Spearman $\rho$"),
               ("f_merit_acc", "Merit-order acc."),
               ("gmax_cos",   r"$\cos\angle(\hat g_{\max}, g_{\max}^\star)$")]
    agg = df.groupby("method").agg({m: ["mean", "std"] for m, _ in metrics})
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reindex(order)

    apply_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=figsize(2.0, 1.9), sharey=True)
    xs = np.arange(len(order))
    colors = [METHOD_COLOR[m] for m in order]
    for ax, (key, ylabel) in zip(axes, metrics):
        _bar_with_err(ax, xs, agg[f"{key}_mean"].values,
                      agg[f"{key}_std"].values, colors)
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABEL[m] for m in order], rotation=30, ha="right")
        ax.set_title(ylabel)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    fig.tight_layout(pad=0.3)
    save_figure(fig, "methods_recovery", OUT / "methods_comparison")


# ---------------------------------------------------------------------------
# 2. Identifiability
# ---------------------------------------------------------------------------
def fig_identifiability():
    df = pd.read_csv(OUT / "identifiability" / "scores.csv")
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.0))

    for ax, ycol, title in [
        (axes[0], "f_rel_err", r"(a) $|\hat f_i - f_i^\star|/|f_i^\star|$"),
        (axes[1], "gmax_rel_err",
         r"(b) $|\hat g_{\max,i} - g_{\max,i}^\star|/g_{\max,i}^\star$"),
    ]:
        ax.scatter(df["id_score"], df[ycol], s=10,
                   color=PALETTE["mid"], alpha=0.7, edgecolors=PALETTE["navy"],
                   linewidths=0.3)
        # binned mean
        bins = np.linspace(df["id_score"].min(), df["id_score"].max(), 7)
        idx = np.digitize(df["id_score"], bins)
        means_x, means_y = [], []
        for b in range(1, len(bins)):
            mask = idx == b
            if mask.sum() >= 2:
                means_x.append(df["id_score"][mask].mean())
                means_y.append(df[ycol][mask].mean())
        ax.plot(means_x, means_y, "o-", color=PALETTE["accent"],
                markersize=4, linewidth=1.0, label="bin mean")
        ax.set_xlabel(r"Identifiability score $s_i$")
        ax.set_title(title)
        ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    save_figure(fig, "identifiability", OUT / "identifiability")


# ---------------------------------------------------------------------------
# 3. Diurnal heatmap + curves
# ---------------------------------------------------------------------------
def fig_diurnal():
    arr = np.load(OUT / "diurnal" / "arrays.npz")
    f_true = arr["f_table_true"]            # (S, n_buses)
    f_strat = arr["f_table_strat"]
    f_static = arr["f_static"]
    n_strata, n_gen = f_true.shape
    static_table = np.tile(f_static, (n_strata, 1))

    apply_paper_style()
    # ---- heatmap (3 panels, shared color scale) ----
    fig, axes = plt.subplots(1, 3, figsize=figsize(2.0, 2.1), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    vmin = float(min(f_true.min(), f_strat.min(), static_table.min()))
    vmax = float(max(f_true.max(), f_strat.max(), static_table.max()))
    panels = [
        (axes[0], f_true.T,      r"(a) True $f^{(s)}$"),
        (axes[1], f_strat.T,     r"(b) Stratified $\hat f^{(s)}$"),
        (axes[2], static_table.T, r"(c) Static $\hat f$ (replicated)"),
    ]
    im = None
    for ax, mat, title in panels:
        im = ax.imshow(mat, aspect="auto", cmap=NAVY_CMAP,
                       vmin=vmin, vmax=vmax, origin="lower",
                       interpolation="nearest")
        ax.set_xlabel("Hour-of-day stratum")
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_ylabel("Generator index")
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Marginal cost $f$")
    cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.18, top=0.88)
    save_figure(fig, "diurnal_heatmap", OUT / "diurnal")

    # ---- per-generator diurnal curves (2 contrasted gens) ----
    # pick the two generators with biggest diurnal swing in truth
    swings = f_true.max(axis=0) - f_true.min(axis=0)
    top = np.argsort(-swings)[:3]
    fig, ax = plt.subplots(figsize=figsize(1.0, 2.0))
    cols = _ordered_blues(len(top))
    hours = np.arange(n_strata)
    for k, g in enumerate(top):
        ax.plot(hours, f_true[:, g], color=PALETTE["accent"], linestyle="--",
                linewidth=0.9, alpha=0.7)
        ax.plot(hours, f_strat[:, g], color=cols[k], linewidth=1.3,
                label=f"gen {g}")
    # legend explanatory entry
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=PALETTE["accent"], linestyle="--",
                      linewidth=0.9, label="truth")]
    handles += [Line2D([0], [0], color=cols[k], linewidth=1.3,
                       label=f"learned (gen {top[k]})") for k in range(len(top))]
    ax.legend(handles=handles, loc="best", ncol=1)
    ax.set_xlabel("Hour-of-day stratum")
    ax.set_ylabel(r"Marginal cost $f$")
    ax.set_title("Per-generator diurnal recovery")
    fig.tight_layout(pad=0.3)
    save_figure(fig, "diurnal_curves", OUT / "diurnal")


# ---------------------------------------------------------------------------
# 4. MEF validation
# ---------------------------------------------------------------------------
def fig_mef():
    arr = np.load(OUT / "mef_validation" / "arrays.npz")
    fd, ag = arr["jac_fd"], arr["jac_ag"]
    mef_true, mef_learn, mef_fcap = arr["mef_true"], arr["mef_learn"], arr["mef_fcap"]

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.2))

    # (a) Jacobian validation
    ax = axes[0]
    lo, hi = float(min(fd.min(), ag.min())), float(max(fd.max(), ag.max()))
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=PALETTE["accent"],
            linewidth=0.8, zorder=1)
    ax.scatter(fd, ag, s=5, color=PALETTE["mid"], alpha=0.55,
               edgecolors=PALETTE["navy"], linewidths=0.15, zorder=2)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Finite-difference $\partial g^\star/\partial d$")
    ax.set_ylabel(r"Autograd $\partial g^\star/\partial d$")
    ax.set_title("(a) Jacobian validation")

    # (b) MEF scatter
    ax = axes[1]
    lo = float(mef_true.min()); hi = float(mef_true.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=PALETTE["accent"],
            linewidth=0.8, zorder=1)
    ax.scatter(mef_true.reshape(-1), mef_fcap.reshape(-1), s=5,
               color=PALETTE["graylt"], alpha=0.55,
               edgecolors=PALETTE["gray"], linewidths=0.15,
               label="caps fixed (F\\&D)", zorder=2)
    ax.scatter(mef_true.reshape(-1), mef_learn.reshape(-1), s=5,
               color=PALETTE["navy"], alpha=0.7,
               edgecolors=PALETTE["deep"], linewidths=0.15,
               label="diff. (full)", zorder=3)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"True MEF (tCO$_2$/MWh)")
    ax.set_ylabel("Estimated MEF")
    ax.set_title("(b) Downstream MEF accuracy")
    ax.legend(loc="upper left")

    fig.tight_layout(pad=0.4)
    save_figure(fig, "mef_validation", OUT / "mef_validation")


# ---------------------------------------------------------------------------
# 5. Sweeps (size, noise)
# ---------------------------------------------------------------------------
def _sweep_plot(name, xlabel, xlogscale=True):
    df = pd.read_csv(OUT / name / "agg.csv")
    methods = ["kkt", "diff_fcap", "diff_full"]
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.0))
    cols = {"kkt": PALETTE["light"], "diff_fcap": PALETTE["mid"],
            "diff_full": PALETTE["navy"]}
    markers = {"kkt": "s", "diff_fcap": "o", "diff_full": "D"}
    panels = [(axes[0], "f_cos",            r"$\cos\angle(\hat f, f^\star)$",
               "(a) Cost recovery"),
              (axes[1], "val_nrmse_clean",  "Clean dispatch NRMSE",
               "(b) Forward-prediction error")]
    for ax, key, ylab, title in panels:
        for m in methods:
            sub = df[df["method"] == m].sort_values("x")
            ax.plot(sub["x"], sub[f"{key}_mean"], marker=markers[m],
                    color=cols[m], linewidth=1.3, markersize=3.6,
                    markerfacecolor=cols[m],
                    markeredgecolor=PALETTE["navy"], markeredgewidth=0.4,
                    label=METHOD_LABEL[m])
            ax.fill_between(sub["x"],
                            sub[f"{key}_mean"] - sub[f"{key}_std"],
                            sub[f"{key}_mean"] + sub[f"{key}_std"],
                            color=cols[m], alpha=0.18, linewidth=0)
        if xlogscale:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[0].legend(loc="lower right")
    fig.tight_layout(pad=0.4)
    save_figure(fig, name, OUT / name)


# ---------------------------------------------------------------------------
# 6. PJM-like
# ---------------------------------------------------------------------------
PJM_FUEL_NAMES = ["nuclear", "hydro", "wind", "solar", "coal", "ccgt",
                  "oil_st", "ct_peaker"]


def fig_pjm():
    arr = np.load(OUT / "pjm_like" / "arrays.npz")
    f_mean_true = arr["f_mean_true"]
    f_mean_learn = arr["f_mean_learned"]
    f_table_true = arr["f_table_true"]
    f_table_learn = arr["f_table_learned"]
    n_fuel = len(PJM_FUEL_NAMES)

    apply_paper_style()

    # ---- (a) merit-order bar chart ----
    order = np.argsort(f_mean_true[:n_fuel])
    fig, ax = plt.subplots(figsize=figsize(1.0, 2.4))
    xs = np.arange(n_fuel)
    width = 0.4
    ax.bar(xs - width / 2, f_mean_true[:n_fuel][order], width=width,
           color=PALETTE["accent"], edgecolor=PALETTE["accent"], linewidth=0.4,
           label="true mean cost", alpha=0.85)
    ax.bar(xs + width / 2, f_mean_learn[:n_fuel][order], width=width,
           color=PALETTE["navy"], edgecolor=PALETTE["navy"], linewidth=0.4,
           label=r"learned $\hat f$")
    ax.set_xticks(xs)
    ax.set_xticklabels([PJM_FUEL_NAMES[i] for i in order],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"Marginal cost (\$/MWh)")
    ax.set_title("PJM-like merit order (sorted by truth)")
    ax.set_yscale("symlog", linthresh=20)
    ax.legend(loc="upper left")
    fig.tight_layout(pad=0.3)
    save_figure(fig, "pjm_stack", OUT / "pjm_like")

    # ---- (b) hour-resolved recovery for renewables (split y-axes is overkill;
    #         use two panels) ----
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.1))
    fuel_idx = {n: i for i, n in enumerate(PJM_FUEL_NAMES)}
    hours = np.arange(f_table_true.shape[0])
    cols = _ordered_blues(3)

    # left panel: dispatchables (hydro, coal, ccgt) -- linear scale
    ax = axes[0]
    for k, fuel in enumerate(["hydro", "coal", "ccgt"]):
        i = fuel_idx[fuel]
        ax.plot(hours, f_table_true[:, i], color=PALETTE["accent"],
                linestyle="--", linewidth=0.9, alpha=0.7)
        ax.plot(hours, f_table_learn[:, i], color=cols[k], linewidth=1.3,
                label=fuel)
    ax.set_xlabel("Hour-of-day stratum")
    ax.set_ylabel(r"Marginal cost (\$/MWh)")
    ax.set_title("(a) Dispatchable fuels")
    ax.legend(loc="best")

    # right panel: renewables (wind, solar) with shadow -- log scale
    ax = axes[1]
    cols2 = [PALETTE["mid"], PALETTE["navy"]]
    for k, fuel in enumerate(["wind", "solar"]):
        i = fuel_idx[fuel]
        ax.plot(hours, f_table_true[:, i], color=PALETTE["accent"],
                linestyle="--", linewidth=0.9, alpha=0.7)
        ax.plot(hours, f_table_learn[:, i], color=cols2[k], linewidth=1.3,
                label=fuel)
    ax.set_yscale("log")
    ax.set_xlabel("Hour-of-day stratum")
    ax.set_ylabel(r"Marginal cost (\$/MWh, log)")
    ax.set_title("(b) Renewables (avail.\\ shadow)")
    ax.legend(loc="best")

    # legend annotation
    fig.text(0.5, 0.005, "dashed = truth, solid = learned",
             ha="center", fontsize=7, color=PALETTE["gray"])
    fig.tight_layout(pad=0.4, rect=[0, 0.04, 1, 1])
    save_figure(fig, "pjm_recovery", OUT / "pjm_like")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    fig_methods_comparison()
    fig_methods_recovery()
    fig_identifiability()
    fig_diurnal()
    fig_mef()
    _sweep_plot("sweep_size", "Training-set size $T$", xlogscale=True)
    _sweep_plot("sweep_noise", r"Observation noise $\sigma$", xlogscale=True)
    fig_pjm()
    print("Wrote figures to", PAPER)


if __name__ == "__main__":
    main()
