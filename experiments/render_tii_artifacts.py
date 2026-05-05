#!/usr/bin/env python3
"""Render IEEE TII manuscript figures and LaTeX table drafts.

Inputs are the audited CSV exports in ``results/paper_tables``. This script
keeps manuscript rendering separate from experiment execution: it does not
recompute metrics or touch model artifacts.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.export_tii_tables import (  # noqa: E402
    export_ieee300,
    export_tdher,
    export_tdher_ablation,
    export_tdher_final_routing_weights,
    export_tdher_threshold_sensitivity,
)


PALETTE = {
    "base": "#4C78A8",
    "always": "#F58518",
    "tdher": "#54A24B",
    "ridge": "#B279A2",
    "gray": "#6B7280",
    "warn": "#E45756",
}
SCENARIO_SHORT = {
    "L1 same distribution": "L1",
    "L2 cross-condition few-shot": "L2",
    "L3 cross-topology few-shot": "L3",
}
TARGET_LABEL = {
    "y1": r"$\Delta f_{\max}$ MAE",
    "y2": r"$t_{\Delta f}$ MAE",
}
METHOD_LABELS = {
    "base_best_expert": "Best",
    "base_convex_blend": "Conv.",
    "base_affine_convex_blend": "+Aff.",
    "base_affine_convex_blend_nonnegative_y2": "+Phys.",
    "convex_blend": "Adapt",
    "affine_convex_blend": "Adapt+Aff.",
    "admission_convex_blend": "Admit",
    "admission_affine_convex_blend": "TD-HER",
    "admission_affine_convex_blend_nonnegative_y2": "TD-HER",
    "ridge_stack": "Ridge",
}
MODEL_LABELS = {
    "LightGBM": "LGBM",
    "KAN": "KAN",
    "ConvLSTM": "ConvLSTM",
    "PatchTST": "PatchTST",
    "Mamba": "Mamba",
    "ST-GCN": "ST-GCN",
    "FT-Transformer": "FT",
    "TabR": "TabR",
    "LightGBM-Adapter": "LGBM-Adapt",
}


def configure_style():
    tnr = "Times New Roman"
    available = {f.name for f in font_manager.fontManager.ttflist}
    if tnr not in available:
        tnr = "DejaVu Serif"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [tnr, "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def save_figure(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, xy, wh, text, fc="#FFFFFF", ec="#374151", lw=1.0,
             fontsize=8.5, weight="normal"):
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
    )
    return patch


def draw_arrow(ax, start, end, color="#374151", lw=1.0, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            lw=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def plot_problem_definition(fig_dir: Path):
    """Render the problem-definition figure for early frequency-extremum prediction."""
    fig = plt.figure(figsize=(7.2, 3.15))
    ax = fig.add_axes([0.075, 0.18, 0.575, 0.72])
    ax2 = fig.add_axes([0.700, 0.18, 0.255, 0.72])

    # Synthetic COI trajectory used only to illustrate the target definition.
    t = np.linspace(0.0, 3.5, 800)
    nadir = -0.42 * (t / 0.82) * np.exp(1.0 - t / 0.82)
    ring = 0.028 * np.exp(-t / 1.2) * np.sin(2.0 * np.pi * 1.45 * t)
    recovery = 0.045 * (1.0 - np.exp(-t / 2.2))
    df = nadir + ring + recovery
    idx = int(np.argmax(np.abs(df)))
    t_star = float(t[idx])
    y1 = float(df[idx])
    window = 0.100

    ax.plot(t, df, color=PALETTE["base"], lw=2.0)
    ax.axhline(0.0, color="#111827", lw=0.8, alpha=0.75)
    ax.axvline(0.0, color="#111827", lw=0.9, ls="--")
    ax.axvline(window, color=PALETTE["tdher"], lw=0.9, ls=":")
    ax.axvline(t_star, color=PALETTE["warn"], lw=1.0, ls="--")
    ax.axvspan(0.0, window, color="#DCEBDB", alpha=0.90)

    ax.scatter([t_star], [y1], s=28, zorder=5, color=PALETTE["warn"],
               edgecolor="white", linewidth=0.6)
    ax.annotate(
        r"$y_1=\Delta f_{\mathrm{COI}}(t^*)$",
        xy=(t_star, y1),
        xytext=(t_star + 0.18, y1 - 0.070),
        fontsize=8,
        color="#7F1D1D",
        arrowprops=dict(arrowstyle="->", color="#7F1D1D", lw=0.8),
    )
    ax.annotate(
        "",
        xy=(t_star, 0.0),
        xytext=(t_star, y1),
        arrowprops=dict(arrowstyle="<->", color="#7F1D1D", lw=0.8),
    )
    ax.annotate(
        r"$y_2=t^*-t_0$",
        xy=(t_star / 2.0, y1 - 0.045),
        ha="center",
        va="top",
        fontsize=8,
        color="#374151",
    )
    ax.annotate(
        "",
        xy=(t_star, y1 - 0.035),
        xytext=(0.0, y1 - 0.035),
        arrowprops=dict(arrowstyle="<->", color="#374151", lw=0.8),
    )
    ax.annotate(
        "ms10 input window\n(~100 ms)",
        xy=(window, 0.025),
        xytext=(0.45, 0.075),
        fontsize=8,
        color="#166534",
        arrowprops=dict(arrowstyle="->", color="#166534", lw=0.8),
    )
    ax.text(0.015, 0.93, r"$t_0$", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8)
    ax.text(t_star + 0.03, 0.93, r"$t^*$", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color="#7F1D1D")
    ax.text(2.02, 0.055, "future response is\nlabel-only",
            ha="center", va="center", fontsize=7.6, color="#4B5563")

    ax.set_xlabel(r"Post-trigger time $t-t_0$ (s)")
    ax.set_ylabel(r"COI frequency deviation $\Delta f_{\mathrm{COI}}$ (Hz)")
    ax.set_xlim(-0.03, 3.35)
    ax.set_ylim(-0.50, 0.12)
    ax.set_title("(a) Prediction targets on the COI response", loc="left",
                 fontsize=9.5, fontweight="bold")
    ax.grid(True, which="major", ls="--", lw=0.5, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("(b) Audited information boundary", loc="left",
                  fontsize=9.5, fontweight="bold")

    draw_box(
        ax2, (0.04, 0.72), (0.92, 0.20),
        "Online input\n"
        r"$\mathbf{x}(t_0:t_0+W)$" "\n"
        "PMU-aligned + model-assisted",
        fc="#E8F1FB", ec=PALETTE["base"], fontsize=7.5, weight="bold",
    )
    draw_box(
        ax2, (0.04, 0.42), (0.92, 0.18),
        "Offline labels\n"
        r"$y_1$ signed extremum" "\n"
        r"$y_2$ nonnegative arrival time",
        fc="#FEECEC", ec=PALETTE["warn"], fontsize=7.5, weight="bold",
    )
    draw_box(
        ax2, (0.04, 0.13), (0.92, 0.20),
        "No future-response or\nlabel leakage\n"
        "across train/cal/test splits",
        fc="#F3F4F6", ec="#6B7280", fontsize=7.6, weight="bold",
    )
    draw_arrow(ax2, (0.50, 0.72), (0.50, 0.60), color="#374151", lw=0.9)
    draw_arrow(ax2, (0.50, 0.42), (0.50, 0.33), color="#374151", lw=0.9)

    save_figure(fig, fig_dir, "problem_definition")


def plot_tdher_overview(fig_dir: Path):
    """Render the TD-HER method overview used in the TII manuscript."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    colors = {
        "data": "#E8F1FB",
        "expert": "#EDF7ED",
        "cal": "#FFF4E6",
        "target": "#F3E8FF",
        "out": "#EAF7F7",
        "line": "#374151",
        "accent": "#4C78A8",
        "adapt": "#F58518",
    }

    # Inputs and representations.
    draw_box(ax, (0.20, 2.40), (1.55, 0.88),
             "Early PMU\nwindow", colors["data"], fontsize=8, weight="bold")
    draw_box(ax, (2.15, 4.20), (1.35, 0.54), "Rep. A\nTabular",
             colors["data"], fontsize=7.5)
    draw_box(ax, (2.15, 2.72), (1.35, 0.54), "Rep. B\nTensor",
             colors["data"], fontsize=7.5)
    draw_box(ax, (2.15, 1.24), (1.35, 0.54), "Rep. C\nGraph",
             colors["data"], fontsize=7.5)

    draw_box(ax, (4.10, 1.55), (2.25, 2.85),
             "Audit-valid\nexpert bank\n\n"
             r"$K$ frozen experts" "\n"
             "tabular / temporal\n"
             "sequence / graph",
             colors["expert"], fontsize=6.5, weight="bold")

    draw_box(ax, (6.95, 3.40), (1.65, 0.78),
             "Calibration split\nbase predictions",
             colors["cal"], fontsize=6.7, weight="bold")
    draw_box(ax, (6.95, 2.05), (1.65, 0.78),
             "Few-shot adapter\nOOF predictions",
             "#FFF8DB", ec=colors["adapt"], fontsize=6.7, weight="bold")

    draw_box(ax, (9.25, 3.95), (1.75, 0.70),
             r"$y_1$ route" "\n" r"$\mathbf{w}_1\geq0,\sum w=1$",
             colors["target"], fontsize=7.4, weight="bold")
    draw_box(ax, (9.25, 2.95), (1.75, 0.70),
             r"$y_2$ route" "\n" r"$\mathbf{w}_2\geq0,\sum w=1$",
             colors["target"], fontsize=7.4, weight="bold")
    draw_box(ax, (9.25, 1.82), (1.75, 0.58),
             "Bounded affine\ncalibration",
             colors["target"], fontsize=7.1)
    draw_box(ax, (9.25, 0.88), (1.75, 0.58),
             r"$t_{\Delta f}\geq0$ projection",
             colors["out"], fontsize=7.1, weight="bold")
    draw_box(ax, (6.95, 0.88), (1.65, 0.58),
             r"Outputs $(\hat{y}_1,\hat{y}_2)$",
             colors["out"], fontsize=7.2, weight="bold")

    # Arrows from input to representations.
    draw_arrow(ax, (1.75, 2.84), (2.15, 4.47), colors["line"])
    draw_arrow(ax, (1.75, 2.84), (2.15, 2.99), colors["line"])
    draw_arrow(ax, (1.75, 2.84), (2.15, 1.51), colors["line"])

    # Arrows from representations to fixed expert bank.
    draw_arrow(ax, (3.50, 4.47), (4.10, 3.75), colors["line"])
    draw_arrow(ax, (3.50, 2.99), (4.10, 2.98), colors["line"])
    draw_arrow(ax, (3.50, 1.51), (4.10, 2.15), colors["line"])

    # Expert and adapter predictions into routing.
    draw_arrow(ax, (6.35, 2.98), (6.95, 3.79), colors["line"])
    draw_arrow(ax, (6.35, 2.98), (6.95, 2.44), colors["adapt"], rad=-0.10)
    draw_arrow(ax, (8.60, 3.79), (9.25, 4.30), colors["accent"])
    draw_arrow(ax, (8.60, 3.79), (9.25, 3.30), colors["accent"])
    draw_arrow(ax, (8.60, 2.44), (9.25, 4.30), colors["adapt"], rad=0.18)
    draw_arrow(ax, (8.60, 2.44), (9.25, 3.30), colors["adapt"], rad=-0.02)

    # Downstream correction and output.
    draw_arrow(ax, (10.12, 3.95), (10.12, 2.40), colors["line"])
    draw_arrow(ax, (10.12, 2.95), (10.12, 2.40), colors["line"])
    draw_arrow(ax, (10.12, 1.82), (10.12, 1.46), colors["line"])
    draw_arrow(ax, (9.25, 1.17), (8.60, 1.17), colors["line"])

    # Compact annotations.
    ax.text(5.22, 0.88, "Base experts are fixed\nbefore routing",
            ha="center", va="center", fontsize=6.4, color="#4B5563")
    ax.text(7.78, 4.65, "OOF admission uses\ncalibration evidence",
            ha="center", va="top", fontsize=6.2, color="#4B5563")
    ax.text(10.13, 5.03, "Target-dependent convex routing",
            ha="center", va="center", fontsize=7.0, color="#4B5563")

    save_figure(fig, fig_dir, "tdher_overview")


def regenerate_tables(router_dir: Path, ieee300_dir: Path, table_dir: Path):
    export_tdher(router_dir, table_dir)
    export_tdher_ablation(router_dir, table_dir)
    export_tdher_threshold_sensitivity(router_dir, table_dir)
    export_tdher_final_routing_weights(router_dir, table_dir)
    export_ieee300(ieee300_dir, table_dir)


def plot_ablation(table_dir: Path, fig_dir: Path):
    rows = read_csv(table_dir / "tdher_ablation.csv")
    method_order = [
        "base_best_expert",
        "base_convex_blend",
        "base_affine_convex_blend_nonnegative_y2",
        "convex_blend",
        "affine_convex_blend",
        "admission_affine_convex_blend_nonnegative_y2",
        "ridge_stack",
    ]
    scenarios = list(dict.fromkeys(row["scenario"] for row in rows))
    data = {(r["scenario"], r["method"]): r for r in rows}

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=False)
    for col, scenario in enumerate(scenarios):
        for row_idx, target in enumerate(["y1", "y2"]):
            ax = axes[row_idx, col]
            values = [as_float(data[(scenario, m)][f"{target}_mae"]) for m in method_order]
            colors = []
            for method in method_order:
                if method.startswith("base"):
                    colors.append(PALETTE["base"])
                elif method.startswith("admission"):
                    colors.append(PALETTE["tdher"])
                elif method == "ridge_stack":
                    colors.append(PALETTE["ridge"])
                else:
                    colors.append(PALETTE["always"])
            x = np.arange(len(method_order))
            ax.bar(x, values, color=colors, width=0.72)
            ax.set_title(f"{SCENARIO_SHORT[scenario]} {target.upper()}", fontsize=9, pad=3)
            if col == 0:
                ax.set_ylabel(TARGET_LABEL[target])
            ax.grid(axis="y", alpha=0.25, linewidth=0.6)
            ax.set_xticks(x)
            if row_idx == 1:
                labels = [METHOD_LABELS[m] for m in method_order]
                ax.set_xticklabels(labels, rotation=34, ha="right", fontsize=7)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis="x", length=0)
            if target == "y2" and scenario == "L3 cross-topology few-shot":
                ax.set_yscale("log")
                if col == 0:
                    ax.set_ylabel(r"$t_{\Delta f}$ MAE (log)")
            ax.tick_params(axis="y", labelsize=7)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["base"], label="Base"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["always"], label="Always-adapt"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["tdher"], label="TD-HER"),
        plt.Rectangle((0, 0), 1, 1, color=PALETTE["ridge"], label="Ridge stack"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=0.8, h_pad=1.0)
    save_figure(fig, fig_dir, "tdher_ablation")


def plot_threshold_sensitivity(table_dir: Path, fig_dir: Path):
    """DEPRECATED: superseded by render_all_figures_v2.render_fig3_threshold.

    The v2 renderer adds (a)/(b) panel labels, the [0.10, 0.30] stable-region
    shading, and the L2 +20.5% annotation that the manuscript caption
    references. Do not call this function for the manuscript figure.
    """
    raise RuntimeError(
        "plot_threshold_sensitivity in render_tii_artifacts is deprecated; "
        "use experiments.render_all_figures_v2.render_fig3_threshold instead."
    )


def plot_routing_weights(table_dir: Path, fig_dir: Path):
    rows = read_csv(table_dir / "tdher_final_routing_weights.csv")
    experts = list(dict.fromkeys(row["expert"] for row in rows))
    row_keys = []
    matrix = []
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        for target in ["y1", "y2"]:
            subset = [r for r in rows if r["scenario"] == scenario and r["target"] == target]
            weights = {r["expert"]: as_float(r["weight"]) for r in subset}
            row_keys.append(f"{SCENARIO_SHORT[scenario]}-{target.upper()}")
            matrix.append([weights.get(expert, 0.0) for expert in experts])
    mat = np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(experts)))
    ax.set_xticklabels([MODEL_LABELS.get(e, e) for e in experts], rotation=30, ha="right", fontsize=7.2)
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_yticklabels(row_keys, fontsize=7.8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val >= 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.6, color="black" if val < 0.55 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Weight", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Target-wise TD-HER routing weights", fontsize=9, pad=4)
    fig.tight_layout()
    save_figure(fig, fig_dir, "tdher_routing_weights")


def plot_bootstrap(table_dir: Path, fig_dir: Path):
    rows = read_csv(table_dir / "tdher_l3_bootstrap.csv")
    targets = [r["target"] for r in rows]
    deltas = np.array([as_float(r["delta_mae_candidate_minus_reference"]) for r in rows])
    lows = np.array([as_float(r["ci95_low"]) for r in rows])
    highs = np.array([as_float(r["ci95_high"]) for r in rows])
    x = np.arange(len(targets))

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    yerr = np.vstack([deltas - lows, highs - deltas])
    ax.errorbar(x, deltas, yerr=yerr, fmt="o", color=PALETTE["tdher"],
                ecolor=PALETTE["tdher"], capsize=4, linewidth=1.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in targets])
    ax.set_ylabel(r"$\Delta$MAE (TD-HER - base)")
    ax.set_title("L3 paired bootstrap CI")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, fig_dir, "tdher_l3_bootstrap_ci")


def plot_ieee300(table_dir: Path, fig_dir: Path):
    rows = read_csv(table_dir / "ieee300_scalability.csv")
    models = [MODEL_LABELS.get(r["model"], r["model"]) for r in rows]
    y1 = np.array([as_float(r["y1_mae"]) for r in rows])
    y2 = np.array([as_float(r["y2_mae"]) for r in rows])
    x = np.arange(len(models))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(7.2, 2.85))
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, y1, width=width, color=PALETTE["base"], label=r"$\Delta f_{\max}$")
    ax2.bar(x + width / 2, y2, width=width, color=PALETTE["tdher"], label=r"$t_{\Delta f}$")
    ax1.set_ylabel(r"$\Delta f_{\max}$ MAE")
    ax2.set_ylabel(r"$t_{\Delta f}$ MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha="right", fontsize=7.4)
    ax1.grid(axis="y", alpha=0.22, linewidth=0.6)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    ax1.set_title("IEEE300 larger-system expert bank", fontsize=9, pad=4)
    ax1.tick_params(axis="y", labelsize=7.5)
    ax2.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    save_figure(fig, fig_dir, "ieee300_scalability")


def plot_ieee300_routing(table_dir: Path, fig_dir: Path):
    path = table_dir / "ieee300_tdher_routing.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    short_labels = {
        "best_expert": "Best",
        "uniform_blend": "Uniform",
        "convex_blend": "Convex",
        "affine_convex_blend": "Conv.+Aff.",
        "affine_convex_blend_nonnegative_y2": "TD-HER",
    }
    methods = [short_labels.get(r["method"], r["label"]) for r in rows]
    y1 = np.array([as_float(r["y1_mae"]) for r in rows])
    y2 = np.array([as_float(r["y2_mae"]) for r in rows])
    x = np.arange(len(methods))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(7.2, 2.85))
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, y1, width=width, color=PALETTE["base"], label=r"$\Delta f_{\max}$")
    ax2.bar(x + width / 2, y2, width=width, color=PALETTE["tdher"], label=r"$t_{\Delta f}$")
    ax1.set_ylabel(r"$\Delta f_{\max}$ MAE")
    ax2.set_ylabel(r"$t_{\Delta f}$ MAE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right", fontsize=7.4)
    ax1.grid(axis="y", alpha=0.22, linewidth=0.6)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False,
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    ax1.set_title("IEEE300 TD-HER routing", fontsize=9, pad=4)
    ax1.tick_params(axis="y", labelsize=7.5)
    ax2.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    save_figure(fig, fig_dir, "ieee300_tdher_routing")


def tex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    out = str(text)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def fmt(value: str, digits: int = 4) -> str:
    if value in ("", None):
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return tex_escape(value)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def render_latex_tables(table_dir: Path, tex_dir: Path):
    main = read_csv(table_dir / "tdher_main_results.csv")
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Main TD-HER results under three evaluation scenarios.}",
        r"\label{tab:tdher_main}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Scenario & Base $y_1$ & Base $y_2$ & TD-HER $y_1$ & TD-HER $y_2$ \\",
        r"\midrule",
    ]
    for r in main:
        lines.append(
            f"{tex_escape(r['scenario'])} & {fmt(r['base_y1_mae'])} & {fmt(r['base_y2_mae'])} "
            f"& {fmt(r['tdher_y1_mae'])} & {fmt(r['tdher_y2_mae'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text(tex_dir / "table_tdher_main.tex", "\n".join(lines))

    boot = read_csv(table_dir / "tdher_l3_bootstrap.csv")
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Paired bootstrap significance for L3 cross-topology recovery.}",
        r"\label{tab:tdher_bootstrap}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Target & Base MAE & TD-HER MAE & Rel. imp. & 95\% CI of $\Delta$MAE \\",
        r"\midrule",
    ]
    for r in boot:
        ci = f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]"
        lines.append(
            f"{tex_escape(r['target'].upper())} & {fmt(r['reference_mae'])} & {fmt(r['candidate_mae'])} "
            f"& {float(r['relative_improvement']) * 100:.1f}\\% & {ci} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text(tex_dir / "table_tdher_bootstrap.tex", "\n".join(lines))

    ablation = read_csv(table_dir / "tdher_ablation.csv")
    keep = {
        "base_best_expert",
        "base_convex_blend",
        "base_affine_convex_blend_nonnegative_y2",
        "affine_convex_blend",
        "admission_affine_convex_blend_nonnegative_y2",
        "ridge_stack",
    }
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{TD-HER ablation. Physical variants use $t_{\Delta f}=\max(t_{\Delta f},0)$.}",
        r"\label{tab:tdher_ablation}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Scenario & Variant & $y_1$ MAE & $y_2$ MAE & Routing & Adapter & Affine & Phys. \\",
        r"\midrule",
    ]
    for r in ablation:
        if r["method"] not in keep:
            continue
        lines.append(
            f"{tex_escape(SCENARIO_SHORT[r['scenario']])} & {tex_escape(r['label'])} "
            f"& {fmt(r['y1_mae'])} & {fmt(r['y2_mae'])} & {tex_escape(r['routing'])} "
            f"& {tex_escape(r['adapter_policy'])} & {tex_escape(r['affine_calibration'])} "
            f"& {tex_escape(r['nonnegative_y2'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write_text(tex_dir / "table_tdher_ablation.tex", "\n".join(lines))

    ieee300 = read_csv(table_dir / "ieee300_scalability.csv")
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Audited IEEE300 larger-system expert results.}",
        r"\label{tab:ieee300}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & $y_1$ MAE & $y_1$ RMSE & $y_2$ MAE & $y_2$ RMSE \\",
        r"\midrule",
    ]
    for r in ieee300:
        lines.append(
            f"{tex_escape(r['model'])} & {fmt(r['y1_mae'])} & {fmt(r['y1_rmse'])} "
            f"& {fmt(r['y2_mae'])} & {fmt(r['y2_rmse'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write_text(tex_dir / "table_ieee300.tex", "\n".join(lines))

    ieee300_routing_path = table_dir / "ieee300_tdher_routing.csv"
    has_ieee300_routing = ieee300_routing_path.exists()
    if has_ieee300_routing:
        ieee300_routing = read_csv(ieee300_routing_path)
        short_labels = {
            "best_expert": "Best expert",
            "uniform_blend": "Uniform average",
            "convex_blend": "Convex",
            "affine_convex_blend": "Convex+affine",
            "affine_convex_blend_nonnegative_y2": "TD-HER physical",
        }
        lines = [
            r"\begin{table}[!t]",
            r"\centering",
            r"\caption{IEEE300 same-distribution TD-HER routing scalability.}",
            r"\label{tab:ieee300_tdher}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & $y_1$ MAE & $\Delta y_1$ & $y_2$ MAE & $\Delta y_2$ \\",
            r"\midrule",
        ]
        for r in ieee300_routing:
            lines.append(
                f"{tex_escape(short_labels.get(r['method'], r['label']))} & {fmt(r['y1_mae'])} "
                f"& {fmt(r['delta_y1_mae_vs_best_expert'])} & {fmt(r['y2_mae'])} "
                f"& {fmt(r['delta_y2_mae_vs_best_expert'])} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        write_text(tex_dir / "table_ieee300_tdher.tex", "\n".join(lines))

    figure_lines = [
        r"\begin{figure*}[!t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/problem_definition.pdf}",
        r"\caption{Problem definition and audited information boundary for early post-disturbance frequency-extremum prediction.}",
        r"\label{fig:problem_definition}",
        r"\end{figure*}",
        "",
        r"\begin{figure*}[!t]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{figures/tdher_ablation.pdf}",
        r"\caption{Ablation of the proposed TD-HER components across same-distribution, cross-condition, and cross-topology scenarios.}",
        r"\label{fig:tdher_ablation}",
        r"\end{figure*}",
        "",
        r"\begin{figure}[!t]",
        r"\centering",
        r"\includegraphics[width=\columnwidth]{figures/tdher_threshold_sensitivity.pdf}",
        r"\caption{Sensitivity of target-wise adapter admission to the required OOF improvement threshold. The dashed line marks the selected threshold.}",
        r"\label{fig:tdher_threshold}",
        r"\end{figure}",
        "",
        r"\begin{figure}[!t]",
        r"\centering",
        r"\includegraphics[width=\columnwidth]{figures/tdher_routing_weights.pdf}",
        r"\caption{Target-wise TD-HER routing weights learned from calibration data.}",
        r"\label{fig:tdher_weights}",
        r"\end{figure}",
        "",
        r"\begin{figure}[!t]",
        r"\centering",
        r"\includegraphics[width=\columnwidth]{figures/tdher_l3_bootstrap_ci.pdf}",
        r"\caption{Paired bootstrap confidence intervals for the L3 cross-topology TD-HER improvement over the base affine convex route.}",
        r"\label{fig:tdher_bootstrap}",
        r"\end{figure}",
        "",
        r"\begin{figure}[!t]",
        r"\centering",
        r"\includegraphics[width=\columnwidth]{figures/ieee300_scalability.pdf}",
        r"\caption{Audited IEEE300 larger-system expert comparison.}",
        r"\label{fig:ieee300}",
        r"\end{figure}",
        "",
    ]
    if has_ieee300_routing:
        figure_lines += [
            r"\begin{figure}[!t]",
            r"\centering",
            r"\includegraphics[width=\columnwidth]{figures/ieee300_tdher_routing.pdf}",
            r"\caption{IEEE300 same-distribution TD-HER routing scalability. The routing weights are learned on the validation split and evaluated on the held-out test split.}",
            r"\label{fig:ieee300_tdher}",
            r"\end{figure}",
            "",
        ]
    write_text(tex_dir / "figure_snippets.tex", "\n".join(figure_lines))


def write_readme(out_dir: Path):
    write_text(out_dir / "README.md", """# TII Manuscript Artifacts

Generated from audited CSV exports in `results/paper_tables`.

## Figures

- `figures/tdher_ablation.{pdf,png}`: TD-HER ablation across L1/L2/L3.
- `figures/problem_definition.{pdf,png}`: Problem definition and audited
  input/label boundary for early post-disturbance prediction.
- `figures/tdher_overview.{pdf,png}`: TD-HER method overview.
- `figures/tdher_threshold_sensitivity.{pdf,png}`: adapter admission threshold sensitivity.
- `figures/tdher_routing_weights.{pdf,png}`: target-wise final routing weights.
- `figures/tdher_l3_bootstrap_ci.{pdf,png}`: paired bootstrap confidence intervals.
- `figures/ieee300_scalability.{pdf,png}`: IEEE300 larger-system expert comparison.
- `figures/ieee300_tdher_routing.{pdf,png}`: IEEE300 same-distribution routing check,
  generated when `results/paper_tables/ieee300_tdher_routing.csv` exists.

## LaTeX Tables

- `latex/table_tdher_main.tex`
- `latex/table_tdher_ablation.tex`
- `latex/table_tdher_bootstrap.tex`
- `latex/table_ieee300.tex`
- `latex/table_ieee300_tdher.tex` when the IEEE300 routing table exists.
- `latex/figure_snippets.tex`

These files are drafts for manuscript integration. They intentionally preserve
the conservative interpretation: TD-HER admission rejects L2 false-positive
adaptation while preserving L3 cross-topology recovery.

The table snippets use `booktabs` commands (`\\toprule`, `\\midrule`,
`\\bottomrule`).
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", default="results/paper_tables")
    parser.add_argument("--router-dir", default="results/ieee39/exp_tdh_router")
    parser.add_argument("--ieee300-dir", default="results/ieee300/exp7_rebuild")
    parser.add_argument("--out-dir", default="results/paper_artifacts")
    parser.add_argument("--skip-table-refresh", action="store_true")
    args = parser.parse_args()

    configure_style()
    table_dir = Path(args.table_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    tex_dir = out_dir / "latex"

    if not args.skip_table_refresh:
        regenerate_tables(Path(args.router_dir), Path(args.ieee300_dir), table_dir)

    plot_problem_definition(fig_dir)
    plot_tdher_overview(fig_dir)
    plot_ablation(table_dir, fig_dir)
    # plot_threshold_sensitivity is superseded by
    # experiments.render_all_figures_v2.render_fig3_threshold
    plot_routing_weights(table_dir, fig_dir)
    plot_bootstrap(table_dir, fig_dir)
    plot_ieee300(table_dir, fig_dir)
    plot_ieee300_routing(table_dir, fig_dir)
    render_latex_tables(table_dir, tex_dir)
    write_readme(out_dir)
    print(f"Rendered manuscript artifacts to {out_dir}")


if __name__ == "__main__":
    main()
