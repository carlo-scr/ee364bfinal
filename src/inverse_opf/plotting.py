"""IEEE-style matplotlib styling for the paper figures.

A single, paper-ready monochromatic navy / light-blue palette with one
warm accent for ground-truth references. All figures in the paper share
this palette so the manuscript reads as a single visual system.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# IEEE column widths.
COLWIDTH = 3.5
DOUBLE_COLWIDTH = 7.16
ROW_HEIGHT = 2.2

# Monochromatic navy / light-blue palette + a single warm accent.
PALETTE = {
    "navy":    "#0B2545",
    "deep":    "#13315C",
    "blue":    "#1F4E79",
    "mid":     "#2E75B6",
    "light":   "#6FA8DC",
    "pale":    "#BDD7EE",
    "ice":     "#E7F0FA",
    "accent":  "#C44536",   # warm rust -- truth / reference lines only
    "gray":    "#5B6770",
    "graylt":  "#A9B0B8",
    # legacy aliases (kept so older scripts don't break)
    "orange":  "#C44536",
    "green":   "#2E75B6",
    "red":     "#C44536",
    "purple":  "#13315C",
    "yellow":  "#BDD7EE",
    "cyan":    "#6FA8DC",
    "black":   "#000000",
}

# Sequential navy colormap (white -> navy).
NAVY_CMAP = LinearSegmentedColormap.from_list(
    "navy_seq",
    [PALETTE["ice"], PALETTE["pale"], PALETTE["light"], PALETTE["mid"],
     PALETTE["blue"], PALETTE["deep"], PALETTE["navy"]],
    N=256,
)

# Diverging palette for signed residuals.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "navy_div",
    [PALETTE["accent"], "#F5F1EE", PALETTE["mid"], PALETTE["navy"]],
    N=256,
)


def _ordered_blues(n: int) -> list[str]:
    base = [PALETTE["pale"], PALETTE["light"], PALETTE["mid"],
            PALETTE["blue"], PALETTE["deep"], PALETTE["navy"]]
    if n <= len(base):
        idx = [int(round(i * (len(base) - 1) / max(1, n - 1))) for i in range(n)]
        return [base[i] for i in idx]
    return [NAVY_CMAP(0.15 + 0.8 * i / max(1, n - 1)) for i in range(n)]


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "regular",
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": mpl.cycler(
                color=[
                    PALETTE["navy"],
                    PALETTE["mid"],
                    PALETTE["light"],
                    PALETTE["accent"],
                    PALETTE["gray"],
                    PALETTE["deep"],
                ]
            ),
            "grid.color": "#DEE2E6",
            "grid.alpha": 0.8,
            "grid.linewidth": 0.3,
            "grid.linestyle": "-",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.minor.size": 1.2,
            "ytick.minor.size": 1.2,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.0,
            "errorbar.capsize": 1.8,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.4,
            "legend.borderaxespad": 0.3,
            "legend.columnspacing": 1.0,
            "image.cmap": "Blues",
        }
    )


def figsize(cols: float = 1.0, height: float = ROW_HEIGHT) -> tuple[float, float]:
    width = COLWIDTH if cols <= 1.0 else DOUBLE_COLWIDTH
    return (width, height)


def save_figure(fig, name: str, out_dir, paper_dir="paper/figures") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(paper_dir / f"{name}.pdf")
    plt.close(fig)


@contextmanager
def paper_figure(name: str, out_dir, cols: float = 1.0, height: float = ROW_HEIGHT,
                 paper_dir: str = "paper/figures", **subplot_kw):
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figsize(cols, height), **subplot_kw)
    try:
        yield fig, ax
    finally:
        fig.tight_layout(pad=0.3)
        save_figure(fig, name, out_dir, paper_dir)


# Method-color convention used across the paper.
METHOD_COLOR = {
    "ridge":      PALETTE["graylt"],
    "mlp":        PALETTE["gray"],
    "kkt":        PALETTE["light"],
    "diff_fcap":  PALETTE["mid"],
    "diff_full":  PALETTE["blue"],
    "diff_strat": PALETTE["navy"],
    "static":     PALETTE["light"],
    "stratified": PALETTE["navy"],
    "truth":      PALETTE["accent"],
}

METHOD_LABEL = {
    "ridge":      "Ridge",
    "mlp":        "MLP",
    "kkt":        "KKT-residual",
    "diff_fcap":  "Diff. (caps fixed)",
    "diff_full":  "Diff. (full)",
    "diff_strat": "Diff. (stratified)",
    "static":     "Static",
    "stratified": "Stratified",
}
