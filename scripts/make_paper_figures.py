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
    ax.set_title("(a) Forward-prediction error")

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
    ax.set_title("(b) Cost-vector recovery")

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
    ax.text(len(order) - 0.5, 0.515, "chance", color=PALETTE["accent"],
            fontsize=6, ha="right", va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel("Merit-order accuracy")
    ax.set_ylim(0, 1.10)
    ax.set_title("(c) Pairwise merit-order accuracy")

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
    for ax, (key, ylabel) in zip(axes, metrics):
        m, lo, hi = _agg(df, "method", key)
        mv = m.reindex(order).values.astype(float)
        lov = lo.reindex(order).values.astype(float)
        hiv = hi.reindex(order).values.astype(float)
        valid = ~np.isnan(mv)
        _bar_with_err(ax, xs[valid], mv[valid], lov[valid], hiv[valid],
                      [colors[i] for i in range(len(order)) if valid[i]])
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABEL[m] for m in order], rotation=32, ha="right")
        ax.set_title(ylabel, fontsize=7)
        ax.set_ylim(0.7, 1.05)
    axes[0].set_ylabel("Score")
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

    for ax, ycol, title in [
        (axes[0], "f_rel_err", r"(a) $|\hat f_i - f_i^\star|/|f_i^\star|$"),
        (axes[1], "gmax_rel_err",
         r"(b) $|\hat g_{\max,i} - g_{\max,i}^\star|/g_{\max,i}^\star$"),
    ]:
        ax.scatter(df["id_score"], df[ycol], s=10,
                   color=PALETTE["mid"], alpha=0.7, edgecolors=PALETTE["navy"],
                   linewidths=0.3)
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
    path = OUT / "diurnal" / "arrays.npz"
    if not path.exists():
        print("  [skip] diurnal/arrays.npz not found")
        return
    arr = np.load(path)
    f_true = arr["f_table_true"]
    f_strat = arr["f_table_strat"]
    f_static = arr["f_static"]
    n_strata, n_gen = f_true.shape
    static_table = np.tile(f_static, (n_strata, 1))

    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize(2.0, 2.1), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    vmin = float(min(f_true.min(), f_strat.min(), static_table.min()))
    vmax = float(max(f_true.max(), f_strat.max(), static_table.max()))
    panels = [
        (axes[0], f_true.T,       r"(a) True $f^{(s)}$"),
        (axes[1], f_strat.T,      r"(b) Stratified $\hat f^{(s)}$"),
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
    ax.set_title("Per-generator diurnal recovery")
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

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.2))

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

    ax = axes[1]
    lo = float(mef_true.min()); hi = float(mef_true.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=PALETTE["accent"],
            linewidth=0.8, zorder=1)
    ax.scatter(mef_true.reshape(-1), mef_fcap.reshape(-1), s=5,
               color=PALETTE["graylt"], alpha=0.55,
               edgecolors=PALETTE["gray"], linewidths=0.15,
               label=r"caps fixed (F\&D)", zorder=2)
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
    path = OUT / "pjm_like" / "arrays.npz"
    if not path.exists():
        print("  [skip] pjm_like/arrays.npz not found")
        return
    arr = np.load(path)
    f_mean_true = arr["f_mean_true"]
    f_mean_learn = arr["f_mean_learned"]
    f_table_true = arr["f_table_true"]
    f_table_learn = arr["f_table_learned"]
    n_fuel = len(PJM_FUEL_NAMES)

    apply_paper_style()

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

    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.1))
    fuel_idx = {n: i for i, n in enumerate(PJM_FUEL_NAMES)}
    hours = np.arange(f_table_true.shape[0])
    cols = _ordered_blues(3)

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
    ax.set_title(r"(b) Renewables (avail.\ shadow)")
    ax.legend(loc="best")
    fig.text(0.5, 0.005, "dashed = truth, solid = learned",
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
    ax.set_title("Accuracy–speed trade-off")
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
    xs = np.arange(len(order))
    colors = [METHOD_COLOR[m] for m in order]
    labels = [METHOD_LABEL[m] for m in order]

    # (a) NRMSE
    ax = axes[0]
    m, lo, hi = _agg(df, "method", "val_nrmse_clean")
    _bar_with_err(ax, xs, m.reindex(order).values,
                  lo.reindex(order).values, hi.reindex(order).values, colors)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Clean dispatch NRMSE")
    ax.set_title("(a) Prediction accuracy")

    # (b) Training time
    ax = axes[1]
    time_col = "total_s" if "total_s" in df.columns else "elapsed_s"
    m, lo, hi = _agg(df, "method", time_col)
    _bar_with_err(ax, xs, m.reindex(order).values,
                  lo.reindex(order).values, hi.reindex(order).values, colors)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Training time (s)")
    ax.set_title("(b) Wall-clock time")

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
            ax.set_title(title)
            continue
        m, lo, hi = _agg(df, "method", metric)
        _bar_with_err(ax, xs, m.reindex(order).values,
                      lo.reindex(order).values, hi.reindex(order).values, colors)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

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

    cols = _ordered_blues(len(df["method"].unique()))
    method_list = sorted(df["method"].unique())

    for ax, metric, ylabel, title in [
        (axes[0], "val_nrmse_clean", "Clean dispatch NRMSE",
         r"(a) NRMSE vs.\ $\varepsilon$"),
        (axes[1], "f_cos",          r"$\cos\angle(\hat f, f^\star)$",
         r"(b) Cost cosine vs.\ $\varepsilon$"),
    ]:
        for k, method in enumerate(method_list):
            sub = df[df["method"] == method]
            grp = sub.groupby("eps")[metric]
            means = grp.mean().reindex(eps_vals)
            stds = grp.std(ddof=1).reindex(eps_vals)
            ns = grp.count().reindex(eps_vals)
            delta = 1.96 * stds / np.sqrt(ns.clip(lower=1))
            ax.plot(eps_vals, means.values, "o-",
                    color=cols[k], linewidth=1.3, markersize=4,
                    markeredgecolor=PALETTE["navy"], markeredgewidth=0.4,
                    label=method)
            ax.fill_between(eps_vals,
                            (means - delta).values,
                            (means + delta).values,
                            color=cols[k], alpha=0.18, linewidth=0)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\varepsilon$ (insensitivity margin, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[0].legend(loc="upper left", fontsize=6.5)
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
    ax.set_title(r"(a) CO$_2$ error vs.\ demand scale")
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
    ax.set_title(r"(b) True vs.\ estimated CO$_2$")
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
    fig, axes = plt.subplots(1, 2, figsize=figsize(2.0, 2.0))

    # (a) scatter: mef_true vs mef_rec per seed
    ax = axes[0]
    lo = df["mef_true"].min(); hi = df["mef_true"].max()
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color=PALETTE["accent"], linewidth=0.8, zorder=1)
    colors_seed = _ordered_blues(df["seed"].nunique())
    for k, (seed, sub) in enumerate(df.groupby("seed")):
        ax.scatter([sub["mef_true"].mean()], [sub["mef_rec"].mean()],
                   s=60, color=colors_seed[k],
                   edgecolors=PALETTE["navy"], linewidths=0.5,
                   zorder=3, label=f"seed {seed}")
        ax.errorbar([sub["mef_true"].mean()], [sub["mef_rec"].mean()],
                    xerr=[[sub["mef_true"].mean() - sub["mef_true"].min()],
                          [sub["mef_true"].max() - sub["mef_true"].mean()]],
                    yerr=[[sub["mef_rec"].mean() - sub["mef_rec"].min()],
                          [sub["mef_rec"].max() - sub["mef_rec"].mean()]],
                    fmt="none", ecolor=colors_seed[k],
                    elinewidth=0.6, capsize=2.0, alpha=0.7)
    ax.set_xlabel(r"True MEF (tCO$_2$/MWh)")
    ax.set_ylabel(r"Estimated MEF (tCO$_2$/MWh)")
    ax.set_title("(a) MEF point recovery")
    ax.legend(fontsize=6.5)

    # (b) relative error bar
    ax = axes[1]
    m, lo_ci, hi_ci = _agg(df, "method", "mef_rel_err")
    methods = list(m.index)
    xs = np.arange(len(methods))
    _bar_with_err(ax, xs, m.values, lo_ci.values, hi_ci.values,
                  [METHOD_COLOR.get(mt, PALETTE["mid"]) for mt in methods])
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABEL.get(mt, mt) for mt in methods],
                       rotation=28, ha="right")
    ax.set_ylabel("Relative MEF error")
    ax.set_title("(b) MEF relative error")

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