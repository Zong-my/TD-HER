#!/usr/bin/env python3
"""
Shared figure style configuration for IEEE TII v2 manuscript.

Standard: Nature / IEEE top-journal / CS top-venue grade.
All figures import this module to ensure visual consistency.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Output directories ───────────────────────────────────────────
# Self-contained: every figure lands inside the project's results/ tree.
FIG_DIR = Path("results/paper_artifacts/figures")
PAPER_FIG_DIR = Path("results/paper_artifacts/figures")
SUPP_FIG_DIR = Path("results/paper_artifacts/supplementary_figures")

# ── Expert family color palette (colorblind-friendly, muted) ─────
EXPERT_COLORS = {
    # Tabular (blue family)
    "LightGBM":       "#2166AC",
    "KAN":            "#4393C3",
    "FT-Transformer": "#92C5DE",
    "FT-Trans.":      "#92C5DE",
    "TabR":           "#D1E5F0",
    # Temporal / Sequence (red family)
    "ConvLSTM":       "#D6604D",
    "PatchTST":       "#F4A582",
    "Mamba":          "#B2182B",
    # Graph (purple)
    "ST-GCN":         "#762A83",
    # Adapter (green)
    "LightGBM-Adapter": "#1B7837",
    "LGBM-Adapter":     "#1B7837",
}

# ── Protocol colors ──────────────────────────────────────────────
PROTOCOL_COLORS = {
    "L1": "#2166AC",
    "L2": "#F4A582",
    "L3": "#B2182B",
}

# ── Target colors ────────────────────────────────────────────────
TARGET_COLORS = {
    "y1": "#2166AC",
    "y2": "#D6604D",
}

# ── Functional palette ───────────────────────────────────────────
PALETTE = {
    "base":     "#4C78A8",
    "proposed":  "#1B7837",
    "warn":     "#E45756",
    "gray":     "#6B7280",
    "light_gray": "#F3F4F6",
    "window_bg": "#F5F5F5",
    "window_border": "#E0E0E0",
}

# ── Short expert labels ──────────────────────────────────────────
EXPERT_SHORT = {
    "LightGBM":       "LGBM",
    "KAN":            "KAN",
    "ConvLSTM":       "CLSTM",
    "PatchTST":       "PTST",
    "Mamba":          "Mamba",
    "ST-GCN":         "STGCN",
    "FT-Transformer": "FT-Tr.",
    "TabR":           "TabR",
    "LightGBM-Adapter": "Adapter",
}

# ── Single-column and double-column widths (inches) ──────────────
COL_WIDTH = 3.5     # IEEE single column
TEXT_WIDTH = 7.16   # IEEE double column


def apply_style():
    """Apply the global Nature/IEEE-grade figure style."""
    plt.rcParams.update({
        # Font (IEEE TII body uses Times Roman; figure text matches body)
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        # Spines and ticks
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.2,
        # Grid
        "axes.grid": False,
        # Output
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        # Font embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axes(ax, top=False, right=False):
    """Remove top/right spines by default."""
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)


def save_figure(fig, name, dirs=None):
    """Save figure in PDF + 600 DPI PNG + SVG to all specified directories.

    bbox_inches='tight' trims surrounding whitespace so the saved canvas
    matches the actual content extent. Without it, layouts that leave the
    axes off-center (e.g., spring-layout topology graphs) embed asymmetric
    padding into the PDF page; LaTeX then scales the entire page as the
    figure and the residual padding manifests as visible left/right
    whitespace imbalance after \\centering.
    """
    if dirs is None:
        dirs = [FIG_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        for fmt in ["pdf", "png", "svg"]:
            path = d / f"{name}.{fmt}"
            fig.savefig(path, format=fmt, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {name} → {', '.join(str(d) for d in dirs)}")


def get_expert_color(name):
    """Get color for an expert by name, with fallback."""
    return EXPERT_COLORS.get(name, PALETTE["gray"])
