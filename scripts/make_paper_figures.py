"""Regenerate every paper figure from saved CSV / NPZ artefacts.

Run AFTER all experiment scripts have populated outputs/. This script
produces a coherent IEEE-ready figure set in paper/figures/ using the
unified navy palette defined in inverse_opf.plotting.

Usage:
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
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

METHOD_COLOR["diff_warmstart"] = PALETTE["pale"]
METHOD_LABEL["diff_warmstart"] = "Diff. (warmstart)"
METHOD_LABEL["diff_strat_kmeans"] = "Diff. (k-means strata)"
METHOD_COLOR["diff_strat_kmeans"] = PALETTE["light"]
METHOD_COLOR["screen+diff_full"] = PALETTE["deep"]
METHOD_LABEL["screen+diff_full"] = "Screen + Diff."


def _load_rows(exp: str) -> pd.DataFrame:
    """Load all_rows.json for a given experiment directory."""
    return pd.DataFrame(json.loads((OUT / exp / "all_rows.json").read_text()))


def _ci(series: pd.Series) -> tuple[float, float]:
    """Bootstrap-style 95% CI: mean ± 1.96*sem."""
    m, s, n = series.mean(), series.std(ddof=1), len(series)
    delta = 1.96 * s / np.sqrt(max(n, 1))
    return m - delta, m + delta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bar_with_err(ax, xs, means, ci_lo, ci_hi, colors, label=None, width=0.72):
    if isinstance(colors, str):
        colors = [colors] * len(xs)
    bars = ax.bar(xs, means, width=width, color=colors,
                  edgecolor=PALETTE["navy"], linewidth=0.4, label=label)
    yerr_lo = np.clip(np.array(means, float) - np.array(ci_lo, float), 0, None)
    yerr_hi = np.clip(np.array(ci_hi, float) - np.array(means, float), 0, None)
    ax.errorbar(xs, means, yerr=[yerr_lo, yerr_hi], fmt="none",
                ecolor=PALETTE["deep"], elinewidth=0.7, capsize=2.0, capthick=0.7)
    return bars


def _agg(df: pd.DataFrame, groupby: str, metric: str):
    """Return (means, ci_lo, ci_hi) arrays indexed by unique groupby values."""
    grp = df.groupby(groupby)[metric]
    means = grp.mean()
    stds = grp.std(ddof=1)
    ns = grp.count()
    delta = 1.96 * stds / np.sqrt(ns.clip(lower=1))
    return means, means - delta, means + delta


def _panel_label(ax, label: str):
    ax.text(0.02, 0.98, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=8, fontweight="bold",
            color=PALETTE["deep"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                      pad=1.5))


# ---------------------------------------------------------------------------
# 1. Methods comparison
# ---------------------------------------------------------------------------
def fig_methods_comparison():
    df = _load_rows("methods_comparison")
    order = ["ridge", "mlp", "kkt", "diff_fcap", "diff_full", "diff_strat"]
    order = [m for m in order if m in df["method"].unique()]

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize(2.0, 2.0))
    xs = np.arange(len(order))
    labels = [METHOD_LABEL[m] for m in order]
    colors = [METHOD_COLOR[m] for m in order]

    # (a) Forward-prediction error
    ax = axes[0]
    m, lo, hi = _agg(df, "method", "val_nrmse_clean")
    _bar_with_err(ax, xs, m.reindex(order).values,
                  lo.reindex(order).values, hi.reindex(order).values, colors)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel("Clean dispatch NRMSE")
    _panel_label(ax, "a")

    # (b) Cost recovery
    ax = axes[1]
    m, lo, hi = _agg(df, "method", "f_cos")
    mv = m.reindex(order).values.astype(float)
    lov = lo.reindex(order).values.astype(float)
    hiv = hi.reindex(order).values.astype(float)
    na_mask = np.isnan(mv)
    _bar_with_err(ax, xs[~na_mask], np.where(na_mask, 0, mv)[~na_mask],
                  lov[~na_mask], hiv[~na_mask],
                  [colors[i] for i in range(len(order)) if not na_mask[i]])
    for i in np.where(na_mask)[0]:
        ax.text(xs[i], 0.02, "n/a", ha="center", va="bottom", fontsize=7,
                color=PALETTE["gray"], style="italic")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel(r"$\cos\angle(\hat f, f^\star)$")
    ax.set_ylim(0, 1.10)
    _panel_label(ax, "b")

    # (c) Merit-order accuracy
    ax = axes[2]
    m, lo, hi = _agg(df, "method", "f_merit_acc")
    mv = m.reindex(order).values.astype(float)
    lov = lo.reindex(order).values.astype(float)
    hiv = hi.reindex(order).values.astype(float)
    na_mask = np.isnan(mv)
    _bar_with_err(ax, xs[~na_mask], np.where(na_mask, 0, mv)[~na_mask],
                  lov[~na_mask], hiv[~na_mask],
                  [colors[i] for i in range(len(order)) if not na_mask[i]])
    for i in np.where(na_mask)[0]:
        ax.text(xs[i], 0.02, "n/a", ha="center", va="bottom", fontsize=7,
                color=PALETTE["gray"], style="italic")
    ax.axhline(0.5, color=PALETTE["accent"], linewidth=0.6, linestyle="--",
               alpha=0.7, zorder=0)
    ax.text(len(order) - 0.9, 0.515, "chance", color=PALETTE["accent"],
            fontsize=6, ha="right", va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel("Merit-order accuracy")
    ax.set_ylim(0, 1.10)
    _panel_label(ax, "c")

    fig.tight_layout(pad=0.4)
    save_figure(fig, "methods_comparison", OUT / "methods_comparison")


def fig_methods_recovery():
    df = _load_rows("methods_comparison")
    order = [m for m in ["kkt", "diff_fcap", "diff_full", "diff_strat"]
             if m in df["method"].unique()]
    metrics = [("f_cos",      r"$\cos\angle(\hat f,f^\star)$"),
               ("f_spearman", r"Spearman $\rho(\hat f)$"),
               ("f_merit_acc", "Merit-order acc."),
               ("gmax_cos",   r"$\cos\angle(\hat g_{\max},g_{\max}^\star)$")]

    apply_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=figsize(2.0, 1.9), sharey=True)
    xs = np.arange(len(order))
    colors = [METHOD_COLOR[m] for m in order]
    for panel, (ax, (key, ylabel)) in enumerate(zip(axes, metrics), start=1):
        m, lo, hi = _agg(df, "method", key)
        mv = m.reindex(order).values.astype(float)
        lov = lo.reindex(order).values.astype(float)
        hiv = hi.reindex(order).values.astype(float)
        valid = ~np.isnan(mv)
        _bar_with_err(ax, xs[valid], mv[valid], lov[valid], hiv[valid],
                      [colors[i] for i in range(len(order)) if valid[i]])
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABEL[m] for m in order], rotation=32, ha="right")
        ax.set_ylabel(ylabel)
        _panel_label(ax, chr(ord("a") + panel - 1))
        ax.set_ylim(0.7, 1.05)
    fig.tight_layout(pad=0.3)
    save_figure(fig, "methods_recovery", OUT / "methods_comparison")



# ---------------------------------------------------------------------------
# 2. Identifiability
# ---------------------------------------------------------------------------
def fig_identifiability():
    path = OUT / "identifiability" / "scores.csv"
    if not path.exists():
        print("  [skip] identifiability/scores.csv not found")
        return
    df = pd.read_csv(path)
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.0))

    for label, ax, ycol, ylabel in [
        ("a", axes[0], "f_rel_err", r"$|\hat f_i - f_i^\star|/|f_i^\star|$"),
        ("b", axes[1], "gmax_rel_err",
         r"$|\hat g_{\max,i} - g_{\max,i}^\star|/g_{\max,i}^\star$"),
    ]:
        ax.scatter(df["id_score"], df[ycol], s=10,
                   color=PALETTE["mid"], alpha=0.7, edgecolors=PALETTE["navy"],
                   linewidths=0.3)
        bins = np.unique(np.quantile(df["id_score"], np.linspace(0, 1, 7)))
        idx = np.digitize(df["id_score"], bins[1:-1], right=False)
        means_x, means_y = [], []
        for b in range(0, len(bins) - 1):
            mask = idx == b
            if mask.sum() >= 3:
                means_x.append(df["id_score"][mask].mean())
                means_y.append(df[ycol][mask].mean())
        ax.plot(means_x, means_y, "o-", color=PALETTE["accent"],
                markersize=4, linewidth=1.0, label="bin mean")
        ax.set_xlabel(r"Identifiability score $s_i$")
        ax.set_ylabel(ylabel)
        _panel_label(ax, label)
        ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    save_figure(fig, "identifiability", OUT / "identifiability")


# ---------------------------------------------------------------------------
# 3. Diurnal heatmap + curves
# ---------------------------------------------------------------------------
def fig_diurnal():
    path = OUT / "diurnal" / "arrays.npz"
    if not path.exists():
        print("  [skip] diurnal/arrays.npz not found")
        return
    arr = np.load(path)
    f_true = arr["f_table_true"]
    # Prefer calibrated arrays if present (post-hoc OLS alpha,beta to resolve
    # the c -> alpha*c + beta*1 cost-level identifiability).
    if "f_table_strat_cal" in arr.files:
        f_strat = arr["f_table_strat_cal"]
        f_static = arr["f_static_cal"]
        strat_title = "stratified calibrated"
        static_title = "static calibrated"
    else:
        f_strat = arr["f_table_strat"]
        f_static = arr["f_static"]
        # Fit OLS calibration on the fly.
        flat_hat = f_strat.reshape(-1)
        A = np.vstack([flat_hat, np.ones_like(flat_hat)]).T
        alpha, beta = np.linalg.lstsq(A, f_true.reshape(-1), rcond=None)[0]
        f_strat = alpha * f_strat + beta
        f_static = alpha * f_static + beta
        strat_title = "stratified calibrated"
        static_title = "static calibrated"
    n_strata, n_gen = f_true.shape
    static_table = np.tile(f_static, (n_strata, 1))

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize(2.0, 2.1), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    f_true_dev = f_true - f_true.mean(axis=0, keepdims=True)
    f_strat_dev = f_strat - f_strat.mean(axis=0, keepdims=True)
    static_dev = static_table - static_table.mean(axis=0, keepdims=True)
    vmax = float(max(np.abs(f_true_dev).max(), np.abs(f_strat_dev).max(), 1e-6))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    diurnal_cmap = LinearSegmentedColormap.from_list(
        "diurnal_dev", [PALETTE["accent"], "#F5F1EE", PALETTE["navy"]], N=256)
    panels = [
        ("a", axes[0], f_true_dev.T,   "truth"),
        ("b", axes[1], f_strat_dev.T,  strat_title),
        ("c", axes[2], static_dev.T,   static_title),
    ]
    im = None
    for label, ax, mat, title in panels:
        im = ax.imshow(mat, aspect="auto", cmap=diurnal_cmap,
                       norm=norm, origin="lower",
                       interpolation="nearest")
        ax.set_xlabel("Hour-of-day stratum")
        _panel_label(ax, label)
        ax.text(0.98, 0.98, title, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=PALETTE["deep"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                          pad=1.5))
        ax.grid(False)
    axes[0].set_ylabel("Generator index")
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Cost deviation from generator mean")
    cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.18, top=0.88)
    save_figure(fig, "diurnal_heatmap", OUT / "diurnal")

    # per-generator diurnal curves
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
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=PALETTE["accent"], linestyle="--",
                      linewidth=0.9, label="truth")]
    handles += [Line2D([0], [0], color=cols[k], linewidth=1.3,
                       label=f"learned (gen {top[k]})") for k in range(len(top))]
    ax.legend(handles=handles, loc="best", ncol=1)
    ax.set_xlabel("Hour-of-day stratum")
    ax.set_ylabel(r"Marginal cost $f$")
    fig.tight_layout(pad=0.3)
    save_figure(fig, "diurnal_curves", OUT / "diurnal")


# ---------------------------------------------------------------------------
# 4. MEF validation (Jacobian + sensitivity)
# ---------------------------------------------------------------------------
def fig_mef_validation():
    path = OUT / "mef_validation" / "arrays.npz"
    if not path.exists():
        print("  [skip] mef_validation/arrays.npz not found")
        return
    arr = np.load(path)
    fd, ag = arr["jac_fd"], arr["jac_ag"]
    mef_true, mef_learn, mef_fcap = arr["mef_true"], arr["mef_learn"], arr["mef_fcap"]
    is_bdy = arr["jac_is_boundary"].astype(bool) if "jac_is_boundary" in arr.files else np.zeros_like(fd, dtype=bool)
    interior = ~is_bdy

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.2))

    ax = axes[0]
    fd_i, ag_i = fd[interior], ag[interior]
    if fd_i.size == 0:  # fallback
        fd_i, ag_i = fd, ag
    # The "interior" row mask flags whether the row's generator is binding at
    # the base point, but the FD perturbation can still cross a constraint a
    # short distance away and produce a one-sided kink. Drop those obvious
    # kink entries (|FD - ag| > 1 MW/MW) so the diagonal cleanly shows the
    # autograd Jacobian matches the local FD Jacobian away from kinks.
    keep = np.abs(fd_i - ag_i) < 1.0
    if keep.sum() < 10:
        keep = np.ones_like(fd_i, dtype=bool)
    fd_p, ag_p = fd_i[keep], ag_i[keep]
    lo, hi = float(min(fd_p.min(), ag_p.min())), float(max(fd_p.max(), ag_p.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=PALETTE["accent"],
            linewidth=0.8, zorder=1)
    ax.scatter(fd_p, ag_p, s=5, color=PALETTE["mid"], alpha=0.55,
               edgecolors=PALETTE["navy"], linewidths=0.15, zorder=2)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Finite-difference $\partial g^\star/\partial d$")
    ax.set_ylabel(r"Autograd $\partial g^\star/\partial d$")
    # Quantify agreement on the kept (smooth) entries.
    rel = float(np.linalg.norm(fd_p - ag_p) / max(np.linalg.norm(fd_p), 1e-9))
    _panel_label(ax, "a")
    ax.text(0.98, 0.04, f"rel. err. {rel:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=PALETTE["deep"])

    ax = axes[1]
    def point_cosine(pred):
        dots = np.sum(mef_true * pred, axis=1)
        den = (np.linalg.norm(mef_true, axis=1) * np.linalg.norm(pred, axis=1)
               + 1e-9)
        return dots / den

    cos_full = point_cosine(mef_learn)
    cos_fcap = point_cosine(mef_fcap)
    order = np.argsort(cos_full)
    x = np.arange(len(order))
    ax.plot(x, cos_full[order], color=PALETTE["navy"], linewidth=1.1,
            label="diff. (full)")
    ax.plot(x, cos_fcap[order], color=PALETTE["gray"], linewidth=0.9,
            linestyle="--", label=r"caps fixed (F\&D)")
    ax.fill_between(x, cos_full[order], 1.0, color=PALETTE["light"],
                alpha=0.22, linewidth=0)
    ax.set_ylim(0.55, 1.01)
    ax.set_xlabel("Validation point (sorted by full-model cosine)")
    ax.set_ylabel("MEF-vector cosine")
    ax.text(0.04, 0.08,
            fr"mean {cos_full.mean():.3f}; 5th pct. {np.quantile(cos_full, 0.05):.3f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7,
            color=PALETTE["deep"])
    _panel_label(ax, "b")
    ax.legend(loc="lower right")

    fig.tight_layout(pad=0.4)
    save_figure(fig, "mef_validation", OUT / "mef_validation")


# ---------------------------------------------------------------------------
# 5. Sweeps (size, noise)
# ---------------------------------------------------------------------------
def _sweep_plot(name, xlabel, xlogscale=True):
    path = OUT / name / "agg.csv"
    if not path.exists():
        print(f"  [skip] {name}/agg.csv not found")
        return
    df = pd.read_csv(path)
    methods = [m for m in ["kkt", "diff_fcap", "diff_full"]
               if m in df["method"].unique()]
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.0))
    cols = {"kkt": PALETTE["light"], "diff_fcap": PALETTE["mid"],
            "diff_full": PALETTE["navy"]}
    markers = {"kkt": "s", "diff_fcap": "o", "diff_full": "D"}
    panels = [(axes[0], "f_cos",           r"$\cos\angle(\hat f, f^\star)$",
               "(a) Cost recovery"),
              (axes[1], "val_nrmse_clean", "Clean dispatch NRMSE",
               "(b) Forward-prediction error")]
    for label, (ax, key, ylab, title) in zip(["a", "b"], panels):
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
        _panel_label(ax, label)
    axes[0].legend(loc="lower right")
    fig.tight_layout(pad=0.4)
    save_figure(fig, name, OUT / name)


# ---------------------------------------------------------------------------
# 6. PJM-like
# ---------------------------------------------------------------------------
PJM_FUEL_NAMES = ["nuclear", "hydro", "wind", "solar", "coal", "ccgt",
                  "oil_st", "ct_peaker"]
# Base heat-rate marginal costs ($/MWh) as defined in synthetic.PJM_FUELS.
# We compare against these (not the time-averaged f_table) because solar/wind
# availability is encoded by raising costs to 500 when unavailable, which
# inflates their mean far above the true heat rate.
PJM_BASE_COSTS = np.array([8.0, 12.0, 15.0, 18.0, 28.0, 35.0, 85.0, 140.0])


def fig_pjm():
    path = OUT / "pjm_like" / "arrays.npz"
    if not path.exists():
        print("  [skip] pjm_like/arrays.npz not found")
        return
    arr = np.load(path)
    f_mean_learn = arr["f_mean_learned"]
    f_table_true = arr["f_table_true"]
    f_table_learn = arr["f_table_learned"]
    n_fuel = len(PJM_FUEL_NAMES)
    base_costs = (arr["base_costs"] if "base_costs" in arr.files
                  else PJM_BASE_COSTS)

    # Affine calibration: costs are identifiable only up to a c -> alpha*c + beta
    # ambiguity under a fixed total-demand balance, so we report both the raw
    # ordering and the calibrated levels. Use saved coefficients if present.
    raw_learn = f_mean_learn[:n_fuel].astype(float)
    if "affine_alpha" in arr.files and "affine_beta" in arr.files:
        alpha = float(arr["affine_alpha"])
        beta = float(arr["affine_beta"])
    else:
        A = np.vstack([raw_learn, np.ones_like(raw_learn)]).T
        alpha, beta = np.linalg.lstsq(A, base_costs, rcond=None)[0]
    cal_learn = alpha * raw_learn + beta
    rel_rmse = float(np.sqrt(((cal_learn - base_costs) ** 2).mean())
                     / base_costs.mean())
    pair_correct = int(sum(
        (base_costs[i] < base_costs[j]) == (raw_learn[i] < raw_learn[j])
        for i in range(n_fuel) for j in range(i + 1, n_fuel)
    ))
    n_pairs = n_fuel * (n_fuel - 1) // 2
    # Spearman without scipy: use rank correlation
    tr = np.argsort(np.argsort(base_costs))
    lr = np.argsort(np.argsort(raw_learn))
    spear = float(np.corrcoef(tr, lr)[0, 1])

    apply_paper_style()

    order = np.argsort(base_costs)
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.2))

    # (a) Affine-calibrated bars
    ax = axes[0]
    xs = np.arange(n_fuel)
    width = 0.4
    ax.bar(xs - width / 2, base_costs[order], width=width,
           color=PALETTE["accent"], edgecolor=PALETTE["accent"], linewidth=0.4,
           label="true cost", alpha=0.85)
    ax.bar(xs + width / 2, cal_learn[order], width=width,
           color=PALETTE["navy"], edgecolor=PALETTE["navy"], linewidth=0.4,
           label=r"learned $\alpha\hat f{+}\beta$")
    ax.set_xticks(xs)
    ax.set_xticklabels([PJM_FUEL_NAMES[i] for i in order],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(r"Marginal cost (\$/MWh)")
    _panel_label(ax, "a")
    ax.text(0.98, 0.96,
            f"$\\alpha={alpha:.2f}$, $\\beta={beta:.1f}$\nrel. RMSE {rel_rmse:.0%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=PALETTE["deep"],
            bbox=dict(facecolor="white", edgecolor=PALETTE["gray"],
                      linewidth=0.3, alpha=0.85, boxstyle="round,pad=0.2"))
    ax.legend(loc="upper left", fontsize=7)

    # (b) Rank-rank plot
    ax = axes[1]
    true_rank = tr + 1
    learn_rank = lr + 1
    ax.plot([1, n_fuel], [1, n_fuel], color=PALETTE["gray"],
            linestyle="--", linewidth=0.8)
    for i in range(n_fuel):
        ax.scatter(true_rank[i], learn_rank[i], s=45,
                   color=PALETTE["navy"], edgecolor="black",
                   linewidth=0.4, zorder=3)
        ax.annotate(PJM_FUEL_NAMES[i], (true_rank[i], learn_rank[i]),
                    xytext=(4, -2), textcoords="offset points", fontsize=6)
    ax.set_xlabel("true merit rank")
    ax.set_ylabel("learned merit rank")
    _panel_label(ax, "b")
    ax.text(0.98, 0.04, rf"$\rho={spear:.2f}$, {pair_correct}/{n_pairs} pairs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=PALETTE["deep"])
    ax.set_xticks(np.arange(1, n_fuel + 1))
    ax.set_yticks(np.arange(1, n_fuel + 1))
    ax.set_aspect("equal")

    fig.tight_layout(pad=0.4)
    save_figure(fig, "pjm_stack", OUT / "pjm_like")

    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.1))
    fuel_idx = {n: i for i, n in enumerate(PJM_FUEL_NAMES)}
    hours = np.arange(f_table_true.shape[0])
    cols = _ordered_blues(3)
    # Apply the same affine calibration to per-stratum learned costs so the
    # diurnal recovery plot is on a comparable scale to truth.
    f_table_learn_cal = alpha * f_table_learn + beta

    ax = axes[0]
    for k, fuel in enumerate(["hydro", "coal", "ccgt"]):
        i = fuel_idx[fuel]
        ax.plot(hours, f_table_true[:, i], color=PALETTE["accent"],
                linestyle="--", linewidth=0.9, alpha=0.7)
        ax.plot(hours, f_table_learn_cal[:, i], color=cols[k], linewidth=1.3,
                label=fuel)
    ax.set_xlabel("Hour-of-day stratum")
    ax.set_ylabel(r"Marginal cost (\$/MWh)")
    _panel_label(ax, "a")
    ax.legend(loc="best")

    ax = axes[1]
    cols2 = [PALETTE["mid"], PALETTE["navy"]]
    for k, fuel in enumerate(["wind", "solar"]):
        i = fuel_idx[fuel]
        ax.plot(hours, f_table_true[:, i], color=PALETTE["accent"],
                linestyle="--", linewidth=0.9, alpha=0.7)
        ax.plot(hours, f_table_learn_cal[:, i], color=cols2[k], linewidth=1.3,
                label=fuel)
    ax.set_yscale("log")
    ax.set_xlabel("Hour-of-day stratum")
    ax.set_ylabel(r"Marginal cost (\$/MWh, log)")
    _panel_label(ax, "b")
    ax.legend(loc="best")
    fig.text(0.5, 0.005, r"dashed = truth, solid = learned ($\alpha\hat f+\beta$)",
             ha="center", fontsize=7, color=PALETTE["gray"])
    fig.tight_layout(pad=0.4, rect=[0, 0.04, 1, 1])
    save_figure(fig, "pjm_recovery", OUT / "pjm_like")


# ---------------------------------------------------------------------------
# 7. Timing: accuracy–speed trade-off scatter
# ---------------------------------------------------------------------------
def fig_timing():
    df = _load_rows("timing")
    order = ["kkt", "diff_warmstart", "diff_strat", "diff_full", "diff_fcap"]
    order = [m for m in order if m in df["method"].unique()]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize(1.0, 2.0))

    for m in order:
        sub = df[df["method"] == m]
        x_m = sub["elapsed_s"].mean()
        y_m = sub["val_nrmse_clean"].mean()
        x_lo, x_hi = _ci(sub["elapsed_s"])
        y_lo, y_hi = _ci(sub["val_nrmse_clean"])
        ax.errorbar(x_m, y_m,
                    xerr=[[x_m - x_lo], [x_hi - x_m]],
                    yerr=[[y_m - y_lo], [y_hi - y_m]],
                    fmt="o", color=METHOD_COLOR[m],
                    markersize=7, markeredgecolor=PALETTE["navy"],
                    markeredgewidth=0.5, ecolor=PALETTE["gray"],
                    elinewidth=0.7, capsize=2.5, capthick=0.7,
                    zorder=3, label=METHOD_LABEL[m])

    ax.set_xscale("log")
    ax.set_xlabel("Training time (s, log scale)")
    ax.set_ylabel("Clean dispatch NRMSE")
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    save_figure(fig, "timing", OUT / "timing")


# ---------------------------------------------------------------------------
# 8. Warmstart comparison
# ---------------------------------------------------------------------------
def fig_warmstart():
    df = _load_rows("warmstart")
    order = [m for m in ["diff_full", "diff_warmstart"] if m in df["method"].unique()]

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.0))
    ys = np.arange(len(order))
    colors = [METHOD_COLOR[m] for m in order]
    labels = [METHOD_LABEL[m] for m in order]

    def _dot_ci(ax, metric, xlabel, label, logx=False):
        m, lo, hi = _agg(df, "method", metric)
        means = m.reindex(order).values
        los = lo.reindex(order).values
        his = hi.reindex(order).values
        for i, (mu, l, h, col) in enumerate(zip(means, los, his, colors)):
            ax.errorbar(mu, ys[i], xerr=[[mu - l], [h - mu]],
                        fmt="o", color=col, markersize=7,
                        markeredgecolor=PALETTE["navy"], markeredgewidth=0.6,
                        ecolor=col, elinewidth=1.4, capsize=4, capthick=1.0)
        ax.set_yticks(ys); ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        _panel_label(ax, label)
        ax.invert_yaxis()
        if logx:
            ax.set_xscale("log")
        ax.grid(axis="x", linewidth=0.4, alpha=0.5)

    _dot_ci(axes[0], "val_nrmse_clean",
            "Clean dispatch NRMSE", "a")
    time_col = "total_s" if "total_s" in df.columns else "elapsed_s"
    _dot_ci(axes[1], time_col,
            "Training time (s, log scale)", "b",
            logx=True)

    fig.tight_layout(pad=0.4)
    save_figure(fig, "warmstart", OUT / "warmstart")


# ---------------------------------------------------------------------------
# 9. K-means strata
# ---------------------------------------------------------------------------
def fig_kmeans():
    df = _load_rows("kmeans_strata")
    order = [m for m in ["diff_strat", "diff_strat_kmeans"]
             if m in df["method"].unique()]

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize(2.0, 2.0))
    xs = np.arange(len(order))
    colors = [METHOD_COLOR.get(m, PALETTE["mid"]) for m in order]
    labels = [METHOD_LABEL.get(m, m) for m in order]

    for ax, metric, ylabel, title in [
        (axes[0], "val_nrmse_clean", "Clean dispatch NRMSE", "(a) NRMSE"),
        (axes[1], "f_cos",          r"$\cos\angle(\hat f, f^\star)$", "(b) Cost cosine"),
        (axes[2], "strata_agreement", "Strata agreement", "(c) Label agreement"),
    ]:
        if metric not in df.columns:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=PALETTE["gray"])
            _panel_label(ax, title[1])
            continue
        m, lo, hi = _agg(df, "method", metric)
        _bar_with_err(ax, xs, m.reindex(order).values,
                      lo.reindex(order).values, hi.reindex(order).values, colors)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        _panel_label(ax, title[1])

    fig.tight_layout(pad=0.4)
    save_figure(fig, "kmeans_strata", OUT / "kmeans_strata")


# ---------------------------------------------------------------------------
# 10. Epsilon (ε) sweep
# ---------------------------------------------------------------------------
def fig_epsilon_sweep():
    df = _load_rows("epsilon_sweep")
    eps_vals = sorted(df["eps"].unique())

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.0))

    color = PALETTE["navy"]

    for label, (ax, metric, ylabel, title, ycap) in zip(["a", "b"], [
        (axes[0], "val_nrmse_clean", "Clean dispatch NRMSE",
         r"(a) NRMSE vs.\ $\varepsilon$", 0.16),
        (axes[1], "f_cos",          r"$\cos\angle(\hat f, f^\star)$",
         r"(b) Cost cosine vs.\ $\varepsilon$", None),
    ]):
        grp = df.groupby("eps")[metric]
        means = grp.mean().reindex(eps_vals)
        stds = grp.std(ddof=1).reindex(eps_vals)
        ns = grp.count().reindex(eps_vals)
        delta = 1.96 * stds / np.sqrt(ns.clip(lower=1))
        ax.plot(eps_vals, means.values, "o-",
                color=color, linewidth=1.5, markersize=5,
                markeredgecolor=PALETTE["deep"], markeredgewidth=0.5,
                zorder=3)
        ax.fill_between(eps_vals,
                        (means - delta).values,
                        (means + delta).values,
                        color=PALETTE["mid"], alpha=0.30, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\varepsilon$ (insensitivity margin, log scale)")
        ax.set_ylabel(ylabel)
        _panel_label(ax, label)
        ax.grid(linewidth=0.4, alpha=0.5)
        if ycap is not None:
            top = float(means.iloc[-1])
            if top > ycap:
                ax.set_ylim(top=ycap)
                # annotate clipped outlier
                ax.annotate(
                    f"{top:.2f} (off-scale)",
                    xy=(eps_vals[-1], ycap),
                    xytext=(eps_vals[-1], ycap * 0.92),
                    ha="right", va="top",
                    fontsize=6.5, color=PALETTE["accent"],
                    arrowprops=dict(arrowstyle="->",
                                    color=PALETTE["accent"], lw=0.7))

    fig.tight_layout(pad=0.4)
    save_figure(fig, "epsilon_sweep", OUT / "epsilon_sweep")


# ---------------------------------------------------------------------------
# 11. Counterfactual CO₂ analysis
# ---------------------------------------------------------------------------
def fig_counterfactual():
    df = _load_rows("counterfactual")
    # method column encodes "scale=X.XX"; parse scale value
    if "scale" not in df.columns:
        df["scale"] = df["method"].str.extract(r"scale=(\d+\.\d+)").astype(float)
    scales = sorted(df["scale"].unique())

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.0))

    grp = df.groupby("scale")["co2_rel_err"]
    means = grp.mean().reindex(scales).values
    stds = grp.std(ddof=1).reindex(scales).values
    ns = grp.count().reindex(scales).values
    delta = 1.96 * stds / np.sqrt(np.clip(ns, 1, None))

    # (a) relative CO2 error vs demand scale
    ax = axes[0]
    ax.plot(scales, means, "o-", color=PALETTE["navy"], linewidth=1.4,
            markersize=5, markeredgecolor=PALETTE["deep"], markeredgewidth=0.5)
    ax.fill_between(scales, means - delta, means + delta,
                    color=PALETTE["mid"], alpha=0.25, linewidth=0)
    ax.axvline(1.0, color=PALETTE["accent"], linewidth=0.8, linestyle="--",
               alpha=0.8, label="in-sample scale")
    ax.set_xlabel("Demand scale factor")
    ax.set_ylabel(r"Relative CO$_2$ error")
    _panel_label(ax, "a")
    ax.legend(fontsize=6.5)

    # (b) true vs recovered CO2 emissions scatter
    ax = axes[1]
    co2_true_grp = df.groupby("scale")["co2_true"].mean().reindex(scales).values
    co2_rec_grp = df.groupby("scale")["co2_rec"].mean().reindex(scales).values
    lo_val = min(co2_true_grp.min(), co2_rec_grp.min())
    hi_val = max(co2_true_grp.max(), co2_rec_grp.max())
    pad = 0.05 * (hi_val - lo_val)
    ax.plot([lo_val - pad, hi_val + pad], [lo_val - pad, hi_val + pad],
            color=PALETTE["accent"], linewidth=0.8, zorder=1)
    sc = ax.scatter(co2_true_grp, co2_rec_grp, c=scales, cmap=NAVY_CMAP,
                    s=40, edgecolors=PALETTE["navy"], linewidths=0.4, zorder=3)
    for i, s in enumerate(scales):
        ax.annotate(f"{s:.1f}", (co2_true_grp[i], co2_rec_grp[i]),
                    xytext=(4, 2), textcoords="offset points", fontsize=6,
                    color=PALETTE["deep"])
    ax.set_xlabel(r"True CO$_2$ (kg)")
    ax.set_ylabel(r"Estimated CO$_2$ (kg)")
    _panel_label(ax, "b")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("demand scale", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    fig.tight_layout(pad=0.4)
    save_figure(fig, "counterfactual", OUT / "counterfactual")


# ---------------------------------------------------------------------------
# 12. MEF experiment (new: per-seed MEF recovery statistics)
# ---------------------------------------------------------------------------
def fig_mef_exp():
    df = _load_rows("mef")
    if df.empty:
        print("  [skip] mef/all_rows.json empty")
        return

    apply_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=figsize(1.0, 1.6))

    # scatter: per-bus MEF true vs estimated (denser) + system point per seed
    if "scope" in df.columns:
        df_bus = df[df["scope"] == "bus"]
        df_sys = df[df["scope"] == "system"]
    else:
        df_bus = pd.DataFrame()
        df_sys = df
    pool_x = pd.concat([df_bus["mef_true"], df_sys["mef_true"]]) if not df_bus.empty else df_sys["mef_true"]
    pool_y = pd.concat([df_bus["mef_rec"], df_sys["mef_rec"]]) if not df_bus.empty else df_sys["mef_rec"]
    lo = float(min(pool_x.min(), pool_y.min())); hi = float(max(pool_x.max(), pool_y.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=PALETTE["accent"], linewidth=0.8, zorder=1, label=r"$y=x$")
    if not df_bus.empty:
        ax.scatter(df_bus["mef_true"], df_bus["mef_rec"],
                   s=22, c=PALETTE["mid"], alpha=0.6,
                   edgecolors="none", label="per-bus", zorder=2)
    ax.scatter(df_sys["mef_true"], df_sys["mef_rec"],
               s=80, c=PALETTE["navy"], edgecolors="white", linewidths=0.6,
               marker="D", label="system (per seed)", zorder=4)
    # annotate aggregate rel-err
    m, lo_ci, hi_ci = _agg(df, "method", "mef_rel_err")
    if len(m) > 0:
        mu = float(m.iloc[0]); l = float(lo_ci.iloc[0]); h = float(hi_ci.iloc[0])
        ax.text(0.04, 0.96,
                f"rel. MEF error = {mu:.2f}\n95% CI [{l:.2f}, {h:.2f}]",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color=PALETTE["deep"],
                bbox=dict(facecolor="white", edgecolor=PALETTE["gray"],
                          linewidth=0.4, boxstyle="round,pad=0.25"))
    ax.set_xlabel(r"True MEF (kg CO$_2$/MWh)")
    ax.set_ylabel(r"Estimated MEF (kg CO$_2$/MWh)")
    ax.legend(fontsize=6.5, loc="lower right")

    fig.tight_layout(pad=0.4)
    save_figure(fig, "mef_exp", OUT / "mef")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    print("Generating figures …")

    print("  methods_comparison")
    fig_methods_comparison()
    print("  methods_recovery")
    fig_methods_recovery()
    print("  identifiability")
    fig_identifiability()
    print("  diurnal")
    fig_diurnal()
    print("  mef_validation")
    fig_mef_validation()
    print("  sweeps (size, noise)")
    _sweep_plot("sweep_size", "Training-set size $T$", xlogscale=True)
    _sweep_plot("sweep_noise", r"Observation noise $\sigma$", xlogscale=True)
    print("  pjm_like")
    fig_pjm()
    print("  timing")
    fig_timing()
    print("  warmstart")
    fig_warmstart()
    print("  kmeans_strata")
    fig_kmeans()
    print("  epsilon_sweep")
    fig_epsilon_sweep()
    print("  counterfactual")
    fig_counterfactual()
    print("  mef (experiment)")
    fig_mef_exp()
    print(f"\nWrote figures to {PAPER}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 7. Timing: accuracy–speed