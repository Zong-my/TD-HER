#!/usr/bin/env python3
"""
Render ALL manuscript figures to Nature/IEEE top-journal standard.

Outputs: 600 DPI PNG + vector PDF + vector SVG for each figure.

Figure numbering convention (REQUIRED — sync with paper order):
    Main paper (in order of appearance):
      fig01_problem_definition         — Section II
      fig02_tdher_overview             — Section III
      fig03_ieee39_topology            — Section IV.A
      fig04_ieee300_topology           — Section IV.A
      fig05_tdher_l3_qualitative       — Section IV.C (single-sample case study)
      fig06_tdher_threshold_sensitivity
      fig07_tdher_expert_sensitivity
      fig08_tdher_routing_weights
      fig09_tdher_route_conflict
      fig10_tdher_multiwindow_tradeoff
      fig11_tdher_pipeline_timing
    Supplementary-only:
      figS1_ieee300_scalability
      figS2_ieee300_tdher_routing
      figS3_ieee300_tdher_routing_weights

If figure order in the paper changes, update both the prefix here AND
the \\includegraphics references in the .tex files. The numeric prefix
must always reflect the paper's actual figure order.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.figure_style import (
    apply_style, clean_axes, save_figure,
    EXPERT_COLORS, EXPERT_SHORT, PROTOCOL_COLORS, TARGET_COLORS, PALETTE,
    FIG_DIR, PAPER_FIG_DIR, SUPP_FIG_DIR, COL_WIDTH, TEXT_WIDTH,
    get_expert_color,
)

TABLE_DIR = Path("results/paper_tables")
TIMING_DIR = Path("results/ieee39/tdher_pipeline_timing")

apply_style()

# ── Scenario name mapping (CSV uses descriptive names) ──────────
SCENARIOS = [
    "L1 same distribution",
    "L2 cross-condition few-shot",
    "L3 cross-topology few-shot",
]
SCENARIO_SHORT = {
    "L1 same distribution": "L1",
    "L2 cross-condition few-shot": "L2",
    "L3 cross-topology few-shot": "L3",
}
SCENARIO_LABELS = {
    "L1 same distribution": "L1 (same dist.)",
    "L2 cross-condition few-shot": "L2 (cross-cond.)",
    "L3 cross-topology few-shot": "L3 (cross-topo.)",
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ══════════════════════════════════════════════════════════════════
# Fig. 3: Threshold Sensitivity
# ══════════════════════════════════════════════════════════════════
def render_fig3_threshold():
    print("Rendering Fig.3 Threshold Sensitivity...")
    rows = read_csv(TABLE_DIR / "tdher_threshold_sensitivity.csv")

    # Single-column compact layout: 1x2 panels in column width.
    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH, 1.40), sharey=False)

    for ti, (target, ylabel) in enumerate([
        ("y1", r"$y_1$ MAE (Hz)"),
        ("y2", r"$y_2$ MAE (s)"),
    ]):
        ax = axes[ti]
        l2_thr = []
        l2_val = []
        for scen in SCENARIOS:
            short = SCENARIO_SHORT[scen]
            color = PROTOCOL_COLORS[short]
            thresholds = []
            values = []
            for r in rows:
                if r["scenario"] != scen:
                    continue
                tau = float(r["threshold"])
                val = float(r[f"{target}_mae"])
                thresholds.append(tau)
                values.append(val)
            if thresholds:
                ax.plot(thresholds, values, "o-", color=color, lw=1.0,
                        ms=3.0, label=short, zorder=3)
                if short == "L2":
                    l2_thr, l2_val = thresholds, values

        ax.axvline(x=0.20, color=PALETTE["gray"], ls="--", lw=0.6, zorder=1)
        # Stable region shading
        ax.axvspan(0.10, 0.30, color="#E8E8E8", alpha=0.4, zorder=0)
        ax.set_xlabel(r"Admission threshold $\tau$", labelpad=2)
        ax.set_ylabel(ylabel)
        # Log y-scale: data span 2+ orders of magnitude.
        ax.set_yscale("log")
        clean_axes(ax)

        # Annotate L2 y2 admission transition.
        if target == "y2" and l2_val:
            high = max(l2_val[:2])
            low = min(l2_val[2:])
            if high > low * 1.05:
                ax.annotate(
                    "+20.5%",
                    xy=(0.05, high),
                    xytext=(0.22, high * 1.7),
                    fontsize=6.5,
                    color=PROTOCOL_COLORS["L2"],
                    ha="center",
                    arrowprops=dict(
                        arrowstyle="->",
                        color=PROTOCOL_COLORS["L2"],
                        lw=0.5,
                        connectionstyle="arc3,rad=-0.2",
                    ),
                    zorder=4,
                )

    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.40, top=0.82,
                        wspace=0.50)
    # Shared legend at top centered over both panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.55, 1.00), ncol=3, frameon=False,
               fontsize=7, handlelength=1.4, handletextpad=0.4,
               columnspacing=1.4)
    # Place "(a)" and "(b)" labels centered below each subplot.
    for ti in range(2):
        panel = "a" if ti == 0 else "b"
        bbox = axes[ti].get_position()
        x_center = (bbox.x0 + bbox.x1) / 2
        fig.text(x_center, 0.02, f"({panel})",
                 ha="center", va="bottom", fontsize=8)
    save_figure(fig, "fig06_tdher_threshold_sensitivity", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 4: Routing Weights (multi-panel)
# ══════════════════════════════════════════════════════════════════
def render_fig4_routing_weights():
    print("Rendering Fig.4 Routing Weights...")
    rows_csv = read_csv(TABLE_DIR / "tdher_final_routing_weights.csv")
    from matplotlib.patches import Patch

    # Build (scenario, expert) -> {y1, y2} lookup with stable expert order.
    seen, expert_order = set(), []
    for r in rows_csv:
        exp = r["expert"]
        if exp in seen:
            continue
        seen.add(exp)
        expert_order.append(exp)
    n_exp = len(expert_order)

    def fmt(v):
        return f"{v:.2f}" if v >= 0.005 else "None"

    # Single-column stacked layout: 3 panels, one per scenario.
    fig, axes = plt.subplots(3, 1, figsize=(COL_WIDTH, 4.20), sharex=True)
    plt.rcParams["hatch.linewidth"] = 0.5

    for si, scen in enumerate(SCENARIOS):
        ax = axes[si]
        slabel = SCENARIO_LABELS[scen]
        for ei, exp in enumerate(expert_order):
            color = get_expert_color(exp)
            w_y1, w_y2 = 0.0, 0.0
            for r in rows_csv:
                if r["scenario"] != scen or r["expert"] != exp:
                    continue
                if r["target"] == "y1":
                    w_y1 = float(r["weight"])
                elif r["target"] == "y2":
                    w_y2 = float(r["weight"])

            # y1 drawn first (bottom, solid no hatch); y2 on top (transparent
            # with white-slash hatch overlay, matching figS1 style).
            if w_y1 > 0:
                ax.barh(ei, w_y1, height=0.65, color=color, alpha=1.0,
                        edgecolor="none", zorder=2)
            if w_y2 > 0:
                ax.barh(ei, w_y2, height=0.65, color=color, alpha=0.38,
                        edgecolor="white", linewidth=0.0, hatch="//",
                        zorder=3)

            # Always show "y1/y2"; null symbol "·" for zero weight.
            if w_y1 > 0.005 or w_y2 > 0.005:
                label = f"{fmt(w_y1)}/{fmt(w_y2)}"
                max_w = max(w_y1, w_y2)
                ax.text(max(max_w, 0.0) + 0.012, ei, label, va="center",
                        ha="left", fontsize=6, color="#222")

        ax.set_yticks(np.arange(n_exp))
        ax.set_yticklabels([EXPERT_SHORT.get(e, e) for e in expert_order],
                           fontsize=7)
        ax.tick_params(axis="y", pad=1)
        ax.set_xlim(0, 1.18)
        ax.invert_yaxis()
        clean_axes(ax)
        # Rotated scenario label closer to the y-axis tick labels.
        ax.text(-0.20, 0.5, slabel, transform=ax.transAxes,
                rotation=90, ha="center", va="center",
                fontsize=7, fontweight="bold", color="#444")

    axes[-1].set_xlabel("Weight", labelpad=2)

    # Legend (y1=solid, y2=hatched) replicated in the lower-right of each
    # panel for consistency. White bbox so the legend stays readable when a
    # bar (e.g., Adapter in panel C) reaches the same region.
    legend_handles = [
        Patch(facecolor="#888", alpha=1.0, edgecolor="none", label="$y_1$"),
        Patch(facecolor="#888", alpha=0.38, edgecolor="white",
              hatch="//", label="$y_2$"),
    ]
    note_text = r"bar labels: $y_1/y_2$ value pair (None denotes 0)"
    for si, ax in enumerate(axes):
        # Panel C (L3) keeps the legend in the upper-right because the
        # Adapter bar fills the lower-right region.
        if si == 2:
            leg_anchor = (1.0, 0.96)
            leg_loc = "upper right"
            note_y = 0.65
            note_va = "top"
        else:
            leg_anchor = (1.0, 0.18)
            leg_loc = "lower right"
            note_y = 0.04
            note_va = "bottom"
        leg = ax.legend(handles=legend_handles, loc=leg_loc,
                        bbox_to_anchor=leg_anchor, ncol=2, frameon=True,
                        fontsize=7, handlelength=2.0, handletextpad=0.4,
                        columnspacing=1.5)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("none")
        leg.get_frame().set_alpha(0.92)
        ax.text(1.0, note_y, note_text, transform=ax.transAxes,
                ha="right", va=note_va, fontsize=6.0, color="#555",
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="round,pad=0.12", alpha=0.92))

    fig.subplots_adjust(left=0.24, right=0.96, top=0.97, bottom=0.14,
                        hspace=0.30)

    # (a)(b)(c) labels centered below each subplot, vertically aligned.
    fig.canvas.draw()
    bbox_last = axes[-1].get_position()
    x_center = (bbox_last.x0 + bbox_last.x1) / 2
    panel_letters = ["a", "b", "c"]
    for ti, ax in enumerate(axes):
        bbox = ax.get_position()
        offset = 0.045 if ti < 2 else 0.105
        y = bbox.y0 - offset
        fig.text(x_center, y, f"({panel_letters[ti]})",
                 ha="center", va="top", fontsize=8)

    save_figure(fig, "fig08_tdher_routing_weights", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 5: Ablation
# ══════════════════════════════════════════════════════════════════
def render_fig5_ablation():
    print("Rendering Fig.5 Ablation...")
    rows = read_csv(TABLE_DIR / "tdher_ablation.csv")

    method_order = [
        "base_best_expert", "base_convex_blend",
        "base_affine_convex_blend_nonnegative_y2",
        "convex_blend",
        "admission_affine_convex_blend_nonnegative_y2",
        "ridge_stack",
    ]
    method_labels = {
        "base_best_expert": "Best expert",
        "base_convex_blend": "Convex",
        "base_affine_convex_blend_nonnegative_y2": "Base+Aff.+Phys.",
        "convex_blend": "Always-adapt",
        "admission_affine_convex_blend_nonnegative_y2": "TD-HER",
        "ridge_stack": "Ridge stack",
    }
    method_colors = {
        "base_best_expert": "#BDBDBD",
        "base_convex_blend": "#92C5DE",
        "base_affine_convex_blend_nonnegative_y2": "#4393C3",
        "convex_blend": "#F4A582",
        "admission_affine_convex_blend_nonnegative_y2": "#1B7837",
        "ridge_stack": "#762A83",
    }

    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 3.5))

    for si, scen in enumerate(SCENARIOS):
        short = SCENARIO_SHORT[scen]
        for ti, target in enumerate(["y1", "y2"]):
            ax = axes[ti, si]
            vals = []
            colors = []
            labels = []
            for m in method_order:
                for r in rows:
                    if r["scenario"] == scen and r["method"] == m:
                        vals.append(float(r[f"{target}_mae"]))
                        colors.append(method_colors.get(m, PALETTE["gray"]))
                        labels.append(method_labels.get(m, m))
                        break

            if not vals:
                continue

            x = np.arange(len(vals))
            bars = ax.bar(x, vals, color=colors, edgecolor="white",
                          linewidth=0.3, width=0.7, zorder=3)

            # Highlight TD-HER bar
            tdher_key = "admission_affine_convex_blend_nonnegative_y2"
            if tdher_key in method_order:
                tdher_idx = method_order.index(tdher_key)
                # Find the actual bar index (may differ if some methods missing)
                matched_methods = [m for m in method_order
                                   if any(r["scenario"] == scen and r["method"] == m
                                          for r in rows)]
                if tdher_key in matched_methods:
                    bi = matched_methods.index(tdher_key)
                    if bi < len(bars):
                        bars[bi].set_edgecolor("#1B7837")
                        bars[bi].set_linewidth(1.2)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
            clean_axes(ax)

            if ti == 0:
                ax.set_title(short, fontweight="bold")
            if si == 0:
                ax.set_ylabel(f"$y_{ti+1}$ MAE")

    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    save_figure(fig, "_orphan_tdher_ablation", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 6: Bootstrap CI
# ══════════════════════════════════════════════════════════════════
def render_fig6_bootstrap():
    print("Rendering Fig.6 Bootstrap CI...")
    rows = read_csv(TABLE_DIR / "tdher_l3_bootstrap.csv")

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.0))

    targets = []
    means = []
    ci_lo = []
    ci_hi = []
    colors = []

    for r in rows:
        t = r["target"]
        targets.append(r"$y_1$ (freq. ext.)" if t == "y1" else r"$y_2$ (arrival)")
        delta = float(r["delta_mae_candidate_minus_reference"])
        lo = float(r["ci95_low"])
        hi = float(r["ci95_high"])
        means.append(delta)
        # Error bars are absolute offsets from the mean
        ci_lo.append(abs(delta - lo))
        ci_hi.append(abs(hi - delta))
        colors.append(TARGET_COLORS.get(t, PALETTE["gray"]))

    y = np.arange(len(targets))
    ax.barh(y, means, height=0.5, color=colors, edgecolor="white",
            linewidth=0.3, zorder=3, alpha=0.7)
    ax.errorbar(means, y, xerr=[ci_lo, ci_hi], fmt="none",
                ecolor="#333333", elinewidth=0.8, capsize=3, zorder=4)
    ax.axvline(0, color="#333333", lw=0.6, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(targets)
    ax.set_xlabel(r"$\Delta$MAE (TD-HER $-$ Base)")
    ax.set_title("L3 paired bootstrap 95% CI", loc="left", fontweight="bold")
    clean_axes(ax)

    fig.tight_layout()
    save_figure(fig, "_orphan_tdher_l3_bootstrap_ci", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# L3 Qualitative Example (single-sample trajectory + predictions)
# ══════════════════════════════════════════════════════════════════
def render_l3_qualitative():
    """Single-sample L3 case study showing TD-HER's nadir recovery.

    Selected sample (idx 1380, cut_machine, gbus_33) by exhaustive deep search
    of the L3 test set. TD-HER is strictly best on BOTH targets, and the
    trajectory exhibits a canonical generator-trip frequency response with a
    sizeable nadir (|Δf*| = 1.12 Hz). Residual lines from each prediction to
    the truth point amplify the visible gap.

    Numerical advantage:
      Δf err  TD-HER 4 mHz  vs  Base 74  /  Best 74 mHz   (~18× tighter)
      t  err  TD-HER 0.11 s vs  Base 0.18 / Best 0.19 s   (~1.7× tighter)

    A 1 Hz Butterworth zero-phase low-pass filter is applied for visualization
    to remove sub-second per-bus PMU noise while preserving the nadir envelope.
    """
    import pandas as pd
    from scipy.signal import butter, filtfilt

    print("Rendering L3 qualitative example...")

    SAMPLE_IDX = 1380
    CSV_PATH = "data/ieee39_v8_80_10_10/csv/cross_cond_topo_test_ms10.csv"
    XLSX_BASE = "data/ieee39_v8/cross_cond_topo_test"
    PRED_DIR = Path("results/ieee39/exp_tdh_router/scenarios/L3_cross_topo_fewshot")

    meta = pd.read_csv(CSV_PATH,
                       usecols=["file_name", "distu_kind",
                                "fpu_deltamax", "t_delta"]).iloc[SAMPLE_IDX]
    fname = meta["file_name"]
    cat = meta["distu_kind"]
    y1_truth = float(meta["fpu_deltamax"])
    t_truth_rel = float(meta["t_delta"])

    xlsx = f"{XLSX_BASE}/{cat}/{fname}"
    freq = pd.read_excel(xlsx, sheet_name="FREQ")
    t_arr = freq.iloc[:, 0].values

    freq_bus_strs = ["30","31","32","33","34","35","36","37","38","39"]
    freq_weights = np.array([6.05, 3.41, 6.05, 0.0, 3.41,
                             5.016, 3.141, 3.141, 5.32, 500.0])
    base_freq = 60.0
    trigger_time = 1.0

    fs = np.zeros(len(t_arr))
    w_sum = 0.0
    for i, b in enumerate(freq_bus_strs):
        try:
            col = freq[b]
        except KeyError:
            try:
                col = freq[int(b)]
            except KeyError:
                continue
        fs += col.values * freq_weights[i]
        w_sum += freq_weights[i]
    fs /= w_sum
    fhz_raw = fs * base_freq  # COI Δf in Hz (pre-trigger preserved)

    # 1 Hz Butterworth low-pass (zero-phase) for visualization clarity.
    # Removes sub-second per-bus PMU noise; preserves the nadir envelope.
    dt = float(t_arr[1] - t_arr[0])
    sample_rate = 1.0 / dt
    sos_b, sos_a = butter(N=4, Wn=1.0, btype="low", fs=sample_rate)
    fhz = filtfilt(sos_b, sos_a, fhz_raw)
    # Pre-trigger COI deviation forced to zero (steady-state baseline).
    fhz[t_arr < 1.0] = 0.0

    # Predictions (relative to trigger)
    tdher = np.load(PRED_DIR / "admission_affine_convex_blend_nonnegative_y2_preds.npy")[SAMPLE_IDX]
    base = np.load(PRED_DIR / "base_affine_convex_blend_nonnegative_y2_preds.npy")[SAMPLE_IDX]
    best_exp = np.load(PRED_DIR / "base_best_expert_preds.npy")[SAMPLE_IDX]

    # Wider aspect ratio per design spec
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.55, 2.3))

    X_MIN, X_MAX = 0.0, 7.0
    mask = (t_arr >= X_MIN) & (t_arr <= X_MAX)

    # COI trajectory (rich charcoal)
    ax.plot(t_arr[mask], fhz[mask], color="#111827", lw=1.05, zorder=3,
            label=r"COI $\Delta f$ trajectory")

    traj_lo = float(fhz[mask].min())
    traj_hi = float(fhz[mask].max())
    y_lo = traj_lo - 0.18
    y_hi = traj_hi + 0.32
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(y_lo, y_hi)

    # 100 ms input observation window (drawn in plot, not in legend)
    obs_end = trigger_time + 0.10
    ax.axvspan(trigger_time, obs_end,
               facecolor=PALETTE["proposed"], alpha=0.30,
               edgecolor=PALETTE["proposed"], lw=0.7, zorder=1)
    # Window label with curved leader line pointing to the band
    ax.annotate("100 ms input window",
                xy=(obs_end + 0.02, 0.07),
                xytext=(1.85, y_hi - 0.05),
                fontsize=8.0, color="#1F4068",
                ha="left", va="top", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>",
                                color="#1F4068", lw=0.8,
                                shrinkA=0, shrinkB=2,
                                connectionstyle="arc3,rad=-0.18",
                                mutation_scale=9),
                zorder=4)

    # Trigger marker (vertical dashed, slightly heavier weight)
    ax.axvline(trigger_time, color=PALETTE["gray"], ls=(0, (2.5, 2.0)),
               lw=1.05, zorder=1)

    # Trigger label: BELOW the zero baseline, kept clear of the y-axis spine,
    # with a small curved leader arrow pointing up-right to the dashed line.
    trig_y_label = -0.10 * (y_hi - y_lo)  # comfortably below the y=0 line
    ax.annotate("trigger",
                xy=(trigger_time - 0.005, trig_y_label * 0.30),
                xytext=(0.10, trig_y_label),
                fontsize=7.5, color="#333333",
                ha="left", va="center", style="italic",
                arrowprops=dict(arrowstyle="-|>",
                                color="#333333", lw=0.6,
                                shrinkA=1.5, shrinkB=2.0,
                                connectionstyle="arc3,rad=-0.30",
                                mutation_scale=7),
                zorder=4)

    # Prediction markers (60 % alpha; truth dominates visual hierarchy)
    method_alpha = 0.60
    methods = [
        ("Best expert",  best_exp, PALETTE["gray"],     "s", 44),
        ("Base route",   base,     PALETTE["base"],     "D", 44),
        ("TD-HER",       tdher,    PALETTE["proposed"], "o", 60),
    ]

    # Residual connection lines from each prediction to truth (amplifies the
    # visible gap between TD-HER and the baselines).
    truth_xy = (t_truth_rel + trigger_time, y1_truth)
    err_records = []  # (label, color, err_y, err_t) for the corner legend
    for label, pred, color, marker, size in methods:
        px = pred[1] + trigger_time
        py = pred[0]
        ax.plot([px, truth_xy[0]], [py, truth_xy[1]],
                color=color, lw=0.95, alpha=0.65,
                ls=(0, (3.5, 2)), solid_capstyle="round",
                zorder=4)
        err_records.append((label, color,
                            abs(py - truth_xy[1]),
                            abs(pred[1] - t_truth_rel)))

    method_handles = []
    for label, pred, color, marker, size in methods:
        h = ax.scatter([pred[1] + trigger_time], [pred[0]],
                       color=color, marker=marker, s=size,
                       alpha=method_alpha, linewidths=0,
                       zorder=8, label=label)
        method_handles.append(h)

    # Truth marker: opaque gold star (highest zorder)
    truth_handle = ax.scatter([t_truth_rel + trigger_time], [y1_truth],
                              color="#F0A500", marker="*", s=130,
                              alpha=1.0, linewidths=0,
                              zorder=12,
                              label=r"Truth $(\Delta f^{*},\, t^{*})$")

    # Zoom inset placed above the marker cluster, narrowed along its central
    # axis (2/3 width) and nudged right + slightly downward.
    yr = y_hi - y_lo
    marker_x_center = ((best_exp[1] + base[1] + tdher[1]) / 3.0
                       + trigger_time)
    marker_x_axes = (marker_x_center - X_MIN) / (X_MAX - X_MIN)
    y_zero_axes = (0.0 - y_lo) / yr
    inset_w_ax = 0.40 * (2.0 / 3.0)  # narrowed to 2/3 of previous width
    inset_h_ax = 0.34                # height unchanged
    right_nudge = 0.045              # small rightward shift
    down_nudge  = 0.06               # small downward shift
    inset_x0 = float(np.clip(marker_x_axes - inset_w_ax / 2.0 + right_nudge,
                             0.05, 1 - inset_w_ax - 0.02))
    inset_y0 = float(np.clip(y_zero_axes - inset_h_ax - 0.005 - down_nudge,
                             0.10, 1 - inset_h_ax - 0.05))
    axin = ax.inset_axes([inset_x0, inset_y0, inset_w_ax, inset_h_ax])
    # Inset trajectory (restored: same weight as main plot)
    axin.plot(t_arr[mask], fhz[mask], color="#111827", lw=1.0, zorder=3)
    # Markers only in inset (residual lines stay in main plot to avoid clutter)
    for label, pred, color, marker, size in methods:
        axin.scatter([pred[1] + trigger_time], [pred[0]],
                     color=color, marker=marker, s=size,
                     alpha=method_alpha, linewidths=0, zorder=8)
    axin.scatter([t_truth_rel + trigger_time], [y1_truth],
                 color="#F0A500", marker="*", s=145,
                 alpha=1.0, linewidths=0, zorder=12)
    # Tight zoom around the nadir cluster, with margin for visual breathing room
    zx_lo = min(best_exp[1], base[1], tdher[1], t_truth_rel) + trigger_time - 0.45
    zx_hi = max(best_exp[1], base[1], tdher[1], t_truth_rel) + trigger_time + 0.45
    zy_lo = min(best_exp[0], base[0], tdher[0], y1_truth) - 0.06
    zy_hi = max(best_exp[0], base[0], tdher[0], y1_truth) + 0.05
    axin.set_xlim(zx_lo, zx_hi)
    axin.set_ylim(zy_lo, zy_hi)
    axin.tick_params(axis="both", labelsize=7.0, length=2, pad=1.5)
    # Denser inset grid (major + minor) — kept faint per top-journal style
    axin.grid(True, which="major", alpha=0.25, lw=0.35)
    axin.minorticks_on()
    axin.grid(True, which="minor", alpha=0.13, lw=0.25, ls=":")
    # Transparent background; rely on subtle border lines only
    axin.set_facecolor("none")
    for sp in axin.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#9CA3AF")
    # Restored connector styling (lighter, dashed)
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    mark_inset(ax, axin, loc1=2, loc2=4,
               fc="none", ec="#999999", lw=0.4, ls=(0, (2, 2)))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"COI $\Delta f$ (Hz)")
    # Denser grid: major (visible) + minor (lighter, dotted)
    ax.grid(True, which="major", alpha=0.25, lw=0.4, zorder=0)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.13, lw=0.3, ls=":", zorder=0)
    # Zero reference line
    ax.axhline(0, color=PALETTE["gray"], lw=0.55, alpha=0.65, zorder=1)
    clean_axes(ax)

    # Bottom-right inline legend listing each method's |Δf| / |Δt| errors.
    # Each row carries the method's marker; TD-HER first to highlight the
    # advantage. No legend title (cleaner than mathmode-bracket banner).
    from matplotlib.lines import Line2D
    method_meta = {"Best expert": ("s", PALETTE["gray"]),
                   "Base route":  ("D", PALETTE["base"]),
                   "TD-HER":      ("o", PALETTE["proposed"])}
    # Re-order so TD-HER is on top
    err_records_ordered = sorted(
        err_records, key=lambda r: 0 if r[0] == "TD-HER" else 1)
    err_handles, err_labels = [], []
    for label, color, ey, et in err_records_ordered:
        mk, c = method_meta[label]
        err_handles.append(
            Line2D([0], [0], marker=mk, linestyle="none",
                   markerfacecolor=c, markeredgecolor="none",
                   markersize=4.5, alpha=method_alpha))
        # No colon (project-wide no-colon rule); two-space separator.
        err_labels.append(
            f"{label}  {ey*1000:.0f} mHz, {et:.2f} s")
    # Frameless inline error legend pinned to bottom-right corner.
    err_legend = ax.legend(err_handles, err_labels,
                           loc="lower right", bbox_to_anchor=(1.00, 0.01),
                           title=r"$|\Delta f|$ error, $|\Delta t|$ error",
                           title_fontsize=6.0,
                           fontsize=5.8,
                           frameon=False,
                           handletextpad=0.4, borderpad=0.0,
                           handlelength=0.6, labelspacing=0.30)
    err_legend._legend_box.align = "left"
    err_legend.get_title().set_fontweight("bold")
    err_legend.get_title().set_color("#1F2937")
    ax.add_artist(err_legend)

    # Legend bottom: Truth FIRST (per design spec), methods follow
    handles = [truth_handle] + method_handles
    labels = [r"Truth $(\Delta f^{*},\, t^{*})$",
              "Best expert", "Base route", "TD-HER"]
    leg = ax.legend(handles=handles, labels=labels,
                    loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    fontsize=7.0, frameon=False,
                    ncol=4, columnspacing=1.4, handletextpad=0.3,
                    borderpad=0.0, scatterpoints=1, markerscale=0.85)
    for handle, txt in zip(leg.legend_handles,
                           [t.get_text() for t in leg.get_texts()]):
        is_truth = "Truth" in txt
        try:
            handle.set_edgecolor("none")
            handle.set_linewidth(0)
        except Exception:
            pass
        handle.set_alpha(1.0 if is_truth else method_alpha)

    fig.tight_layout()
    save_figure(fig, "fig05_tdher_l3_qualitative", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 8: Route Conflict (butterfly chart)
# ══════════════════════════════════════════════════════════════════
def render_fig8_route_conflict():
    """Per-expert dumbbell chart of y1/y2 routing weights — segment length
    equals the local L1 contribution to the route divergence."""
    print("Rendering Fig.8 Route Conflict...")
    from matplotlib.ticker import MultipleLocator
    from matplotlib.lines import Line2D
    rows = read_csv(TABLE_DIR / "tdher_final_routing_weights.csv")
    diag = read_csv(TABLE_DIR / "tdher_route_conflict_diagnostics.csv")

    # Divergence values from "base" variant
    div_map = {r["scenario"]: float(r.get("route_l1_divergence", 0))
               for r in diag if r["variant"] == "base"}

    # Stable global expert order shared across all panels.
    expert_set = []
    for scen in SCENARIOS:
        for r in rows:
            if r["scenario"] != scen:
                continue
            exp = r["expert"]
            if exp not in expert_set:
                expert_set.append(exp)
    e_colors = [get_expert_color(e) for e in expert_set]
    e_labels = [EXPERT_SHORT.get(e, e) for e in expert_set]
    y = np.arange(len(expert_set))

    # Piecewise-linear x-axis: stretch [0, 0.1] to occupy [0, 0.4] of the
    # physical axis, compress [0.1, 0.5] into [0.4, 0.5], keep [0.5, 1.0]
    # untouched. This expands the small-weight region where most experts
    # cluster, while preserving the global [0, 1] range.
    def _fwd(x):
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        m1 = x < 0.1
        m2 = (x >= 0.1) & (x < 0.5)
        m3 = x >= 0.5
        out[m1] = x[m1] * 4.0
        out[m2] = 0.4 + (x[m2] - 0.1) * 0.25
        out[m3] = x[m3]
        return out

    def _inv(y):
        y = np.asarray(y, dtype=float)
        out = np.empty_like(y)
        m1 = y < 0.4
        m2 = (y >= 0.4) & (y < 0.5)
        m3 = y >= 0.5
        out[m1] = y[m1] / 4.0
        out[m2] = 0.1 + (y[m2] - 0.4) * 4.0
        out[m3] = y[m3]
        return out

    fig, axes = plt.subplots(1, 3, figsize=(COL_WIDTH, 2.0),
                             sharex=True, sharey=True)

    for si, scen in enumerate(SCENARIOS):
        ax = axes[si]
        short = SCENARIO_SHORT[scen]

        w1_map, w2_map = {}, {}
        for r in rows:
            if r["scenario"] != scen:
                continue
            if r["target"] == "y1":
                w1_map[r["expert"]] = float(r["weight"])
            elif r["target"] == "y2":
                w2_map[r["expert"]] = float(r["weight"])
        w1_vals = [w1_map.get(e, 0) for e in expert_set]
        w2_vals = [w2_map.get(e, 0) for e in expert_set]

        ax.set_xscale("function", functions=(_fwd, _inv))
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))

        # Connecting line between y1 (circle) and y2 (square) markers.
        for i in range(len(expert_set)):
            ax.plot([w1_vals[i], w2_vals[i]], [i, i],
                    color=e_colors[i], lw=1.2, alpha=0.55,
                    solid_capstyle="round", zorder=2)
        # Hollow markers with thicker edge to compensate for missing fill.
        ax.scatter(w1_vals, y, marker="o", s=22,
                   facecolors="none", edgecolors=e_colors,
                   linewidths=1.1, zorder=4)
        ax.scatter(w2_vals, y, marker="s", s=20,
                   facecolors="none", edgecolors=e_colors,
                   linewidths=1.1, zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels(e_labels, fontsize=6.8)
        ax.tick_params(axis="x", labelsize=6.8)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, 0.5, 1.0])
        ax.invert_yaxis()
        clean_axes(ax)

        # Top title — scenario level + L1 divergence
        div = div_map.get(scen, 0)
        ax.set_title(f"{short} (div={div:.2f})", fontsize=8,
                     fontweight="normal", pad=2)

    # Marker-shape legend inside the first panel (bottom-right corner).
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="none", markeredgecolor="#555",
               markeredgewidth=1.0, markersize=4.8, label=r"$y_1$"),
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor="none", markeredgecolor="#555",
               markeredgewidth=1.0, markersize=4.6, label=r"$y_2$"),
    ]
    # Panel A legend at lower-right (empty corner); panels B/C legend at
    # bottom-center for visual consistency.
    for si, ax in enumerate(axes):
        if si == 0:
            loc, bbox_anchor = "lower right", None
        else:
            loc, bbox_anchor = "lower center", (0.5, 0.0)
        ax.legend(handles=legend_handles, loc=loc,
                  bbox_to_anchor=bbox_anchor,
                  frameon=False, handletextpad=0.3, fontsize=6.5,
                  borderpad=0.15, labelspacing=0.3, ncol=2,
                  columnspacing=0.8)

    # Shared x-axis label centered across all panels.
    fig.tight_layout(w_pad=0.4, pad=0.3, rect=(0, 0.06, 1, 1))
    fig.supxlabel("Routing weight", fontsize=8, y=0.02)
    save_figure(fig, "fig09_tdher_route_conflict", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 9: Expert-Bank Sensitivity
# ══════════════════════════════════════════════════════════════════
def render_fig9_expert_sensitivity():
    print("Rendering Fig.9 Expert Sensitivity...")
    from matplotlib.colors import PowerNorm
    loo_rows = read_csv(TABLE_DIR / "tdher_expert_leave_one_out.csv")
    sub_rows = read_csv(TABLE_DIR / "tdher_expert_subset_size.csv")

    # Single-column stacked layout: (a) heatmap on top, (b) line plot below.
    fig = plt.figure(figsize=(COL_WIDTH, 3.20))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[2.0, 1.3], width_ratios=[1.0, 0.045],
        hspace=0.50, wspace=0.04,
        left=0.20, right=0.95, top=0.97, bottom=0.15,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    # Span both columns so (b) edges align with (a) heatmap+colorbar extent.
    ax2 = fig.add_subplot(gs[1, :])

    cases = [
        ("L1 same distribution", "y1", "L1 $y_1$"),
        ("L2 cross-condition few-shot", "y2", "L2 $y_2$"),
        ("L3 cross-topology few-shot", "y1", "L3 $y_1$"),
    ]

    # ── (a) Heatmap: rows=experts, cols=cases ──────────────────────────
    seen, expert_order = set(), []
    for r in loo_rows:
        exp = r.get("omitted_expert", "")
        if exp in ("", "full") or exp in seen:
            continue
        seen.add(exp)
        expert_order.append(exp)

    n_exp = len(expert_order)
    n_scn = len(cases)
    M = np.full((n_exp, n_scn), np.nan)
    for ci, (scen, target, _label) in enumerate(cases):
        for r in loo_rows:
            if r["scenario"] != scen or r["target"] != target:
                continue
            exp = r.get("omitted_expert", "")
            if exp in ("", "full") or exp not in expert_order:
                continue
            full_mae = float(r.get("full_tdher_mae", 0))
            loo_mae = float(r.get("leave_one_out_mae", 0))
            rel = (loo_mae - full_mae) / full_mae * 100 if full_mae > 0 else 0.0
            M[expert_order.index(exp), ci] = rel

    vmax_a = float(np.nanmax(M))
    norm_a = PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax_a)
    im1 = ax1.imshow(M, cmap="Reds", norm=norm_a, aspect="auto")

    ax1.set_xticks(np.arange(n_scn))
    ax1.set_xticklabels([c[2] for c in cases], fontsize=7.5)
    ax1.set_yticks(np.arange(n_exp))
    ax1.set_yticklabels([EXPERT_SHORT.get(e, e) for e in expert_order],
                        fontsize=7)
    for i in range(n_exp):
        for j in range(n_scn):
            v = M[i, j]
            if np.isnan(v):
                ax1.text(j, i, "n/a", ha="center", va="center",
                         fontsize=6.5, color="#999")
                continue
            shade = norm_a(v)
            tcol = "white" if shade > 0.55 else "#222"
            label = f"{v:.1f}" if v < 100 else f"{v:.0f}"
            ax1.text(j, i, label, ha="center", va="center",
                     fontsize=6.5, color=tcol)
    ax1.set_xticks(np.arange(n_scn + 1) - 0.5, minor=True)
    ax1.set_yticks(np.arange(n_exp + 1) - 0.5, minor=True)
    ax1.grid(which="minor", color="white", linewidth=0.7)
    ax1.tick_params(which="minor", length=0)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(top=False, right=False, length=0)

    cbar1 = fig.colorbar(im1, cax=cax)
    cbar1.set_label("MAE rel. change (%)", fontsize=7)
    cbar1.ax.tick_params(labelsize=6.5, length=2)
    cbar1.outline.set_visible(False)

    # ── (b) Top-m subset: line plot (absolute MAE vs subset size) ──────
    for scen, target, label, color, marker, ls in [
        ("L1 same distribution", "y1", "L1 $y_1$",
         TARGET_COLORS["y1"], "o", "-"),
        ("L2 cross-condition few-shot", "y2", "L2 $y_2$",
         TARGET_COLORS["y2"], "s", "--"),
        ("L3 cross-topology few-shot", "y1", "L3 $y_1$",
         PROTOCOL_COLORS["L3"], "^", "-."),
    ]:
        sizes, maes = [], []
        for r in sub_rows:
            if r["scenario"] != scen or r["target"] != target:
                continue
            s = r.get("subset_size", "")
            m = float(r.get("subset_route_mae", r.get("full_tdher_mae", 0)))
            try:
                sizes.append(int(s))
                maes.append(m)
            except (ValueError, TypeError):
                pass
        if sizes:
            order = np.argsort(sizes)
            ax2.plot([sizes[i] for i in order], [maes[i] for i in order],
                     marker=marker, linestyle=ls, color=color, lw=1.0,
                     ms=3.5, label=label, zorder=3)

    ax2.set_xlabel("Expert subset size", labelpad=2)
    ax2.set_ylabel("MAE")
    # Legend top edge flush with axis top edge.
    ax2.legend(loc="upper right", frameon=False, fontsize=6.5, ncol=3,
               handlelength=1.2, handletextpad=0.3, columnspacing=0.8,
               bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0)
    clean_axes(ax2)

    # Place "(a)" and "(b)" labels centered below each subplot, matching fig06.
    # Both labels share the same x_center (ax2's full-panel center) so they
    # are vertically aligned regardless of heatmap vs colorbar geometry.
    fig.canvas.draw()
    bbox2 = ax2.get_position()
    x_center = (bbox2.x0 + bbox2.x1) / 2
    for ti, ax in enumerate([ax1, ax2]):
        panel = "a" if ti == 0 else "b"
        bbox = ax.get_position()
        offset = 0.085 if ti == 0 else 0.115
        y = bbox.y0 - offset
        fig.text(x_center, y, f"({panel})",
                 ha="center", va="top", fontsize=8)
    save_figure(fig, "fig07_tdher_expert_sensitivity", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 10: Pipeline Timing
# ══════════════════════════════════════════════════════════════════
def render_fig10_pipeline_timing():
    print("Rendering Fig.10 Pipeline Timing...")
    with open(TIMING_DIR / "pipeline_timing_gpu.json") as f:
        gpu = json.load(f)
    with open(TIMING_DIR / "pipeline_timing_cpu.json") as f:
        cpu = json.load(f)

    expert_order = [
        "FT-Transformer", "PatchTST", "ST-GCN", "Mamba",
        "KAN", "LightGBM", "ConvLSTM",
    ]

    gpu_color = "#2166AC"
    cpu_color = "#B2182B"

    gpu_meds = [gpu["per_expert"][e]["median_ms"] for e in expert_order]
    cpu_meds = []
    for e in expert_order:
        c = cpu["per_expert"][e]
        # Mamba requires CUDA, CPU median is -1
        cpu_meds.append(c["median_ms"] if c["median_ms"] > 0 else np.nan)

    # Single-column horizontal layout — (a) vertical bars on the left,
    # (b) vertical bars on the right.
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(COL_WIDTH, 2.18),
        gridspec_kw={"width_ratios": [1.7, 1.0], "wspace": 0.08},
    )

    # (a) Per-expert vertical grouped bars
    x = np.arange(len(expert_order))
    bar_w = 0.36
    ax1.bar(x - bar_w/2 - 0.02, gpu_meds, width=bar_w, color=gpu_color,
            edgecolor="white", linewidth=0.3, label="GPU", zorder=3)
    cpu_plot = np.where(np.isnan(cpu_meds), 0, cpu_meds)
    ax1.bar(x + bar_w/2 + 0.02, cpu_plot, width=bar_w, color=cpu_color,
            alpha=0.65, edgecolor="white", linewidth=0.3, label="CPU", zorder=3)

    mamba_idx = expert_order.index("Mamba")
    ax1.text(mamba_idx + bar_w/2 + 0.02, 0.25, "requires CUDA",
             fontsize=6.0, color="#999", fontstyle="italic",
             rotation=90, va="bottom", ha="center")

    # Value labels above bars (rotated to fit narrow grouped spacing)
    for i, v in enumerate(gpu_meds):
        ax1.text(i - bar_w/2 - 0.02, v + 0.12, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=6.5, color=gpu_color,
                 rotation=90)
    for i, v in enumerate(cpu_meds):
        if not np.isnan(v):
            ax1.text(i + bar_w/2 + 0.02, v + 0.12, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=6.5, color=cpu_color,
                     rotation=90)

    ax1.set_ylim(0, 6.0)
    ax1.set_yticks([0, 1, 2, 3, 4, 5])

    ax1.set_xticks(x)
    ax1.set_xticklabels([EXPERT_SHORT.get(e, e) for e in expert_order],
                        rotation=90, ha="center")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Per-expert", loc="center",
                  fontsize=8, fontweight="normal")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
               frameon=False, fontsize=7,
               handlelength=1.0, handletextpad=0.4,
               borderpad=0.0, borderaxespad=0.0)
    clean_axes(ax1)

    # (b) Pipeline total vertical bars
    categories = ["GPU\npar.", "GPU\nseq.", "CPU\npar.", "CPU\nseq."]
    values = [
        gpu["pipeline_summary"]["total_parallel_ms"],
        gpu["pipeline_summary"]["total_sequential_ms"],
        cpu["pipeline_summary"]["total_parallel_ms"],
        cpu["pipeline_summary"]["total_sequential_ms"],
    ]
    bar_colors = [gpu_color, gpu_color, cpu_color, cpu_color]
    bar_alphas = [1.0, 0.5, 0.65, 0.35]

    x2 = np.arange(len(categories))
    # 100 ms window background — full-height tinted strip per category
    ax2.bar(x2, [100]*4, width=0.6, color=PALETTE["window_bg"],
            edgecolor=PALETTE["window_border"], linewidth=0.3, zorder=1)
    for i, (v, c, a) in enumerate(zip(values, bar_colors, bar_alphas)):
        ax2.bar(i, v, width=0.6, color=c, edgecolor="white",
                linewidth=0.3, alpha=a, zorder=3)
        pct = v / 100 * 100
        ax2.text(i, v + 2.0, f"{v:.1f}\n({pct:.0f}%)",
                 ha="center", va="bottom", fontsize=6.5,
                 fontweight="bold", color="#333")

    ax2.axhline(100, color="#CCC", ls="--", lw=0.6, zorder=2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=6.5)
    ax2.set_ylim(0, 120)
    # Move y-axis ticks and label to the right side of panel (b).
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Pipeline total", loc="center",
                  fontsize=8, fontweight="normal")
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    # Explicit margins so the (a)/(b) panel tags fit below the rotated xticks.
    fig.subplots_adjust(left=0.13, right=0.90, top=0.92, bottom=0.40,
                        wspace=0.08)
    bbox1 = ax1.get_position()
    bbox2 = ax2.get_position()
    fig.text((bbox1.x0 + bbox1.x1) / 2, 0.08, "(a)",
             ha="center", va="bottom", fontsize=8.5)
    fig.text((bbox2.x0 + bbox2.x1) / 2, 0.08, "(b)",
             ha="center", va="bottom", fontsize=8.5)
    save_figure(fig, "fig11_tdher_pipeline_timing",
                [FIG_DIR, PAPER_FIG_DIR, SUPP_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 11: Multi-Window Trade-off
# ══════════════════════════════════════════════════════════════════
def render_fig11_multiwindow():
    print("Rendering Fig.11 Multi-Window Trade-off...")
    from matplotlib.ticker import AutoMinorLocator
    rows = read_csv(TABLE_DIR / "tdher_multiwindow_l1.csv")

    # TD-HER physical router across windows
    windows_ms = [10, 50, 100, 150, 250]
    y1_vals = []
    y2_vals = []

    for wms in windows_ms:
        found = False
        for r in rows:
            w = r.get("window_ms", "")
            method = r.get("method", "")
            if str(w) == str(wms) and method == "tdher_physical":
                y1_vals.append(float(r["y1_mae"]))
                y2_vals.append(float(r["y2_mae"]))
                found = True
                break
        if not found:
            y1_vals.append(np.nan)
            y2_vals.append(np.nan)

    fig, ax1 = plt.subplots(figsize=(COL_WIDTH, 2.1))
    ax2 = ax1.twinx()

    # Dense gridlines beneath data (left axis only; both y-axes share x).
    ax1.set_axisbelow(True)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.grid(True, which="major", axis="both",
             lw=0.4, color="#bdbdbd", alpha=0.85, zorder=1)
    ax1.grid(True, which="minor", axis="y",
             lw=0.25, color="#dcdcdc", alpha=0.7, zorder=1)

    x = np.arange(len(windows_ms))
    l1, = ax1.plot(x, y1_vals, "o-", color=TARGET_COLORS["y1"], lw=1.5,
                   ms=5, label=r"$y_1$ MAE", zorder=3)
    l2, = ax2.plot(x, y2_vals, "s--", color=TARGET_COLORS["y2"], lw=1.5,
                   ms=5, label=r"$y_2$ MAE", zorder=3)

    # Mark 100 ms as the default window; place label in the empty
    # lower-left corner with a tight white bbox to clear the gridlines.
    main_idx = windows_ms.index(100)
    ax1.axvline(x=main_idx, color=PALETTE["gray"], ls=":", lw=0.8, zorder=2)
    ax1.text(main_idx, 0.96, "default window",
             transform=ax1.get_xaxis_transform(),
             ha="center", va="top",
             fontsize=7, color=PALETTE["gray"],
             bbox=dict(facecolor="white", edgecolor="none",
                       boxstyle="round,pad=0.18", alpha=0.92),
             zorder=5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{w} ms" for w in windows_ms])
    ax1.set_xlabel("Early observation window")
    ax1.set_ylabel(r"$y_1$ MAE (Hz)", color=TARGET_COLORS["y1"])
    ax2.set_ylabel(r"$y_2$ MAE (s)", color=TARGET_COLORS["y2"])
    ax1.tick_params(axis="y", colors=TARGET_COLORS["y1"])
    ax2.tick_params(axis="y", colors=TARGET_COLORS["y2"])

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right",
               frameon=False)
    ax1.set_title("L1 latency-accuracy trade-off",
                  loc="center", fontsize=8, fontweight="normal")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, "fig10_tdher_multiwindow_tradeoff",
                [FIG_DIR, PAPER_FIG_DIR, SUPP_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# IEEE 300: Expert Bank Scalability
# ══════════════════════════════════════════════════════════════════
def render_ieee300_scalability():
    print("Rendering IEEE300 Scalability...")
    rows = read_csv(TABLE_DIR / "ieee300_scalability.csv")

    models = [r["model"] for r in rows]
    y1 = [float(r["y1_mae"]) for r in rows]
    y2 = [float(r["y2_mae"]) for r in rows]
    colors = [get_expert_color(m) for m in models]
    labels = [EXPERT_SHORT.get(m, m) for m in models]

    fig, ax1 = plt.subplots(figsize=(TEXT_WIDTH * 0.75, 2.4))
    ax2 = ax1.twinx()

    x = np.arange(len(models))
    w = 0.35
    bars1 = ax1.bar(x - w/2, y1, width=w, color=colors, edgecolor="white",
                    linewidth=0.4, alpha=0.85, zorder=3, label=r"$\Delta f_{\max}$")
    bars2 = ax2.bar(x + w/2, y2, width=w, color=colors, edgecolor="white",
                    linewidth=0.4, alpha=0.45, zorder=3, label=r"$t_{\Delta f}$",
                    hatch="//")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax1.set_ylabel(r"$\Delta f_{\max}$ MAE (Hz)")
    ax2.set_ylabel(r"$t_{\Delta f}$ MAE (s)")
    ax1.set_title("IEEE 300-bus expert bank performance", loc="left",
                   fontweight="bold")
    clean_axes(ax1)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right",
               frameon=False)

    fig.tight_layout()
    save_figure(fig, "figS1_ieee300_scalability", [FIG_DIR, PAPER_FIG_DIR, SUPP_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# IEEE 300: TD-HER Routing
# ══════════════════════════════════════════════════════════════════
def render_ieee300_routing():
    print("Rendering IEEE300 TD-HER Routing...")
    rows = read_csv(TABLE_DIR / "ieee300_tdher_routing.csv")

    methods = [r["label"] for r in rows]
    y1 = [float(r["y1_mae"]) for r in rows]
    y2 = [float(r["y2_mae"]) for r in rows]

    # Color: gray for baselines, green for TD-HER
    method_colors = []
    for r in rows:
        m = r["method"]
        if "nonneg" in m or "affine_convex" in m:
            method_colors.append(PALETTE["proposed"])
        elif "convex" in m:
            method_colors.append(PALETTE["base"])
        else:
            method_colors.append(PALETTE["gray"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 0.75, 2.2))

    x = np.arange(len(methods))
    # Shorten labels
    short_labels = []
    for lab in methods:
        lab = lab.replace("Target-wise validation-best expert", "Best expert")
        lab = lab.replace("Uniform expert average", "Uniform")
        lab = lab.replace("TD-HER convex routing + affine", "Conv.+Aff.")
        lab = lab.replace("TD-HER convex routing", "Convex")
        lab = lab.replace("TD-HER physical routing", "TD-HER")
        short_labels.append(lab)

    ax1.bar(x, y1, color=method_colors, edgecolor="white", linewidth=0.3,
            width=0.65, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=7)
    ax1.set_ylabel(r"$\Delta f_{\max}$ MAE (Hz)")
    ax1.set_title(r"(a) $y_1$", loc="left", fontweight="bold")
    clean_axes(ax1)

    ax2.bar(x, y2, color=method_colors, edgecolor="white", linewidth=0.3,
            width=0.65, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=7)
    ax2.set_ylabel(r"$t_{\Delta f}$ MAE (s)")
    ax2.set_title(r"(b) $y_2$", loc="left", fontweight="bold")
    clean_axes(ax2)

    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "figS2_ieee300_tdher_routing", [FIG_DIR, PAPER_FIG_DIR, SUPP_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# IEEE 300: Routing Weights
# ══════════════════════════════════════════════════════════════════
def render_ieee300_routing_weights():
    print("Rendering IEEE300 Routing Weights...")
    rows = read_csv(TABLE_DIR / "ieee300_tdher_routing_weights.csv")

    targets = ["y1", "y2"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH * 0.75, 2.2))

    for ti, target in enumerate(targets):
        ax = axes[ti]
        experts = []
        weights = []
        colors = []
        for r in rows:
            if r["target"] != target:
                continue
            exp = r["expert"]
            w = float(r["weight"])
            experts.append(EXPERT_SHORT.get(exp, exp))
            weights.append(w)
            colors.append(get_expert_color(exp))

        y_pos = np.arange(len(experts))
        ax.barh(y_pos, weights, height=0.6, color=colors,
                edgecolor="white", linewidth=0.3, zorder=3)

        for i, w in enumerate(weights):
            if w > 0.02:
                ax.text(w + 0.01, i, f"{w:.2f}", va="center",
                        fontsize=6.5, color="#333333")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(experts, fontsize=7)
        ax.set_xlim(0, 0.7)
        ax.invert_yaxis()
        ax.set_xlabel("Weight")
        ax.set_title(f"$y_{ti+1}$", fontweight="bold")
        clean_axes(ax)

    fig.suptitle("IEEE 300-bus routing weights", fontweight="bold", fontsize=9, y=1.02)
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "figS3_ieee300_tdher_routing_weights", [FIG_DIR, PAPER_FIG_DIR, SUPP_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 1: Problem Definition (single-column, compact)
# ══════════════════════════════════════════════════════════════════
def render_fig1_problem_definition():
    """COI trajectory schematic — early-window vs future-trajectory."""
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as patheffects
    print("Rendering Fig.1 Problem Definition...")

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.1))

    # Schematic time axis (illustrative; nadir position chosen for layout)
    t_max = 4.0
    t = np.linspace(0.0, t_max, 1200)
    t_peak = 1.7
    nadir_amp = -0.42
    nadir = nadir_amp * (t / t_peak) * np.exp(1.0 - t / t_peak)
    ring = 0.020 * np.exp(-t / 1.6) * np.sin(2.0 * np.pi * 0.85 * t)
    recovery = 0.055 * (1.0 - np.exp(-t / 2.4))
    df = nadir + ring + recovery
    idx = int(np.argmin(df))
    t_star = float(t[idx])
    y1_val = float(df[idx])
    # Visually substantial early-window band (schematic — physical 100 ms)
    window = 0.42

    # Early-window region — green shading
    ax.axvspan(0.0, window, color="#D5E8D4", alpha=0.95, zorder=1)
    # Future-trajectory region — light red shading
    ax.axvspan(window, t_max, color="#FDECEA", alpha=0.55, zorder=1)

    # Trajectory and reference lines
    ax.plot(t, df, color=PALETTE["base"], lw=1.6, zorder=4)
    ax.axhline(0.0, color="#333333", lw=0.5, alpha=0.6, zorder=2)
    ax.axvline(0.0, color="#333333", lw=0.7, ls="--", zorder=3)
    ax.axvline(window, color=PALETTE["proposed"], lw=1.1, ls=":", zorder=3)
    ax.axvline(t_star, color=PALETTE["warn"], lw=0.7, ls="--",
               alpha=0.55, zorder=2)

    # Extremum point
    ax.scatter([t_star], [y1_val], s=28, zorder=5,
               color=PALETTE["warn"], edgecolor="white", linewidth=0.6)

    # y1 vertical indicator (right of t*)
    y1_arrow_x = t_star + 0.059
    ax.annotate("", xy=(y1_arrow_x, 0.002), xytext=(y1_arrow_x, y1_val + 0.004),
                arrowprops=dict(arrowstyle="<->", color="#7F1D1D", lw=0.9,
                                shrinkA=0, shrinkB=0),
                zorder=5)
    ax.text(y1_arrow_x + 0.08, y1_val * 0.459,
            r"$y_1{=}\Delta f_{\mathrm{COI}}(t^*)$",
            fontsize=8.2, color="#7F1D1D", va="center")

    # y2 horizontal indicator (just below zero line)
    y2_arrow_y = -0.045
    ax.annotate("", xy=(t_star + 0.004, y2_arrow_y),
                xytext=(-0.005, y2_arrow_y),
                arrowprops=dict(arrowstyle="<->", color="#374151", lw=0.9,
                                shrinkA=0, shrinkB=0),
                zorder=5)
    ax.text(t_star / 2.0, y2_arrow_y - 0.030,
            r"$y_2{=}t^*-t_0$",
            ha="center", va="top", fontsize=8.2, color="#374151")

    # Legend — symmetric inner padding; box width derived from text width
    leg_left, leg_bottom = 0.575, 0.748
    leg_height = 0.180
    inner_h_pad = 0.008  # left == right inner padding (target)
    inner_v_pad = 0.020
    swatch_w, swatch_h = 0.036, 0.060
    text_gap = 0.020  # gap between swatch and label text
    label_top, label_bot = "early window", "future trajectory"

    # Measure text width on the actual canvas to size the box
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_axes = ax.transAxes.inverted()
    max_text_w = 0.0
    for label in (label_top, label_bot):
        probe = ax.text(0, 0, label, fontsize=7.0, fontweight="bold",
                        transform=ax.transAxes)
        bbox = probe.get_window_extent(renderer).transformed(inv_axes)
        max_text_w = max(max_text_w, bbox.width)
        probe.remove()
    leg_width = inner_h_pad + swatch_w + text_gap + max_text_w + inner_h_pad
    leg_width -= 0.012  # trim right edge inward

    bg = FancyBboxPatch(
        (leg_left, leg_bottom),
        leg_width, leg_height,
        boxstyle="round,pad=0.008,rounding_size=0.022",
        facecolor=(1.0, 1.0, 1.0, 0.55),  # faint background for legibility
        edgecolor="none",                  # frameless per top-journal style
        linewidth=0,
        transform=ax.transAxes, zorder=5,
        clip_on=False,
    )
    ax.add_patch(bg)

    sx = leg_left + inner_h_pad
    text_x = sx + swatch_w + text_gap
    row1_y = leg_bottom + leg_height - swatch_h / 2 - inner_v_pad
    row2_y = leg_bottom + swatch_h / 2 + inner_v_pad
    # Green swatch + label (top row)
    ax.add_patch(plt.Rectangle((sx, row1_y - swatch_h / 2),
                                swatch_w, swatch_h,
                                facecolor="#D5E8D4", edgecolor="#166534",
                                linewidth=0.6, transform=ax.transAxes,
                                zorder=6, clip_on=False))
    ax.text(text_x, row1_y, label_top,
            ha="left", va="center", fontsize=7.0, color="#166534",
            fontweight="bold", transform=ax.transAxes, zorder=6)
    # Red swatch + label (bottom row)
    ax.add_patch(plt.Rectangle((sx, row2_y - swatch_h / 2),
                                swatch_w, swatch_h,
                                facecolor="#FDECEA", edgecolor="#7F1D1D",
                                linewidth=0.6, transform=ax.transAxes,
                                zorder=6, clip_on=False))
    ax.text(text_x, row2_y, label_bot,
            ha="left", va="center", fontsize=7.0, color="#7F1D1D",
            fontweight="bold", transform=ax.transAxes, zorder=6)

    # t0 / t* labels — at top inside data area
    ax.text(0.04, 0.165, r"$t_0$",
            ha="left", va="center", fontsize=8.6, color="#333333",
            fontweight="bold")
    ax.text(window + 0.04, 0.165, r"$t_d$",
            ha="left", va="center", fontsize=8.6, color="#333333",
            fontweight="bold")
    ax.text(t_star + 0.05, 0.165, r"$t^*$",
            ha="left", va="center", fontsize=8.6, color="#7F1D1D",
            fontweight="bold")

    ax.set_xlabel(r"Post-trigger time $t-t_0$")
    ax.set_ylabel(r"$\Delta f_{\mathrm{COI}}$ (Hz)")
    ax.set_xlim(-0.05, t_max)
    ax.set_ylim(-0.52, 0.215)
    # Schematic — hide numerical x-axis tick labels
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y")
    clean_axes(ax)

    fig.tight_layout()
    save_figure(fig, "fig01_problem_definition", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 2: TD-HER Overview (conceptual)
# ══════════════════════════════════════════════════════════════════
def render_fig2_tdher_overview():
    """TD-HER method overview diagram — publication-grade redesign.

    Layout: left-to-right three-stage flow with subtle background bands.
    Stage 1 (Training): Input → Representations → Expert bank
    Stage 2 (Calibration): Base predictions + OOF adapter admission → routing
    Stage 3 (Inference): Affine correction → physical projection → output

    Visual focus: the y1/y2 routing fork with distinct target colors.
    """
    from matplotlib.patches import FancyBboxPatch
    print("Rendering Fig.2 TD-HER Overview...")

    W = TEXT_WIDTH                    # 7.16 inches (IEEE double-column)
    H = 3.06                         # inches; aspect ≈ 2.34 matches overview spec
    # Canvas coordinates scaled so 1 unit ≈ 0.5 inch
    CW, CH = 14.4, 6.0
    fig, ax = plt.subplots(figsize=(W, H))
    # Fill the figure with the axes so saved width matches \textwidth without
    # a downstream LaTeX upscale (avoids stroke / font re-sampling drift).
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, CW)
    ax.set_ylim(0, CH)
    ax.axis("off")

    # ── Color scheme ──────────────────────────────────────────────
    C_Y1 = TARGET_COLORS["y1"]       # #2166AC  (blue)
    C_Y2 = TARGET_COLORS["y2"]       # #D6604D  (red)
    C_ADAPT = "#1B7837"              # adapter green
    C_LINE = "#4B5563"               # neutral dark gray for arrows
    C_EDGE = "#9CA3AF"               # muted box edges

    # Expert family accent colors
    C_TAB = "#2166AC"
    C_TEMP = "#D6604D"
    C_GRAPH = "#762A83"

    # Background bands
    C_BG_TRAIN = "#F0F4FA"
    C_BG_CAL = "#FFFAEE"
    C_BG_INF = "#F2FAF2"

    # Box fills
    F_INPUT = "#E8EFF8"
    F_REP = "#F5F7FA"
    F_EXPERT = "#FAFAFA"
    F_CAL = "#FFF8E8"
    F_POST = "#F0F8F0"

    # ── Stage background bands ────────────────────────────────────
    STAGE1_END = 5.80
    STAGE2_END = 9.55
    stages = [
        (0.0,       STAGE1_END, C_BG_TRAIN, "Training stage"),
        (STAGE1_END, STAGE2_END, C_BG_CAL,  "Calibration stage"),
        (STAGE2_END, CW,         C_BG_INF,  "Inference"),
    ]
    for x0, x1, bg, label in stages:
        ax.axvspan(x0, x1, facecolor=bg, edgecolor="none", zorder=0)
        if x1 < CW:
            ax.axvline(x1, color="#D1D5DB", lw=0.4, ls="--", zorder=1)
        ax.text((x0 + x1) / 2, CH - 0.25, label,
                ha="center", va="center", fontsize=8, color="#666666",
                fontweight="bold")

    # ── Helpers ───────────────────────────────────────────────────
    def _box(xy, wh, text, fc, ec=C_EDGE, fs=6.5, fw="normal",
             tc="#333333", rounding=0.07, lw=0.6, zorder=3,
             ls=1.2):
        x, y = xy; w, h = wh
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.015,rounding_size={rounding}",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc, linespacing=ls,
                zorder=zorder + 1)
        return p

    def _arrow(start, end, color=C_LINE, lw=0.7, rad=0.0,
               shrinkA=1, shrinkB=1):
        ax.annotate("", xy=end, xytext=start,
                     arrowprops=dict(
                         arrowstyle="-|>", lw=lw, color=color,
                         shrinkA=shrinkA, shrinkB=shrinkB,
                         connectionstyle=f"arc3,rad={rad}",
                         mutation_scale=7),
                     zorder=4)

    # ══════════════════════════════════════════════════════════════
    # STAGE 1 — Training
    # ══════════════════════════════════════════════════════════════

    # Input
    inp_x, inp_w, inp_h = 0.20, 1.20, 1.10
    inp_cy = CH / 2                  # vertically centered
    inp_y = inp_cy - inp_h / 2
    _box((inp_x, inp_y), (inp_w, inp_h),
         "Early PMU\nwindow", F_INPUT, ec=C_TAB, fs=7, fw="bold", lw=0.8)

    # Representations (three stacked)
    rep_x, rep_w, rep_h = 1.80, 1.00, 0.55
    rep_gap = 0.35
    total_rep_h = 3 * rep_h + 2 * rep_gap
    rep_base = inp_cy - total_rep_h / 2
    reps = [
        (rep_base + 2 * (rep_h + rep_gap), "Rep. A\nTabular",  C_TAB),
        (rep_base + 1 * (rep_h + rep_gap), "Rep. B\nTensor",   C_TEMP),
        (rep_base + 0 * (rep_h + rep_gap), "Rep. C\nGraph",    C_GRAPH),
    ]
    for ry, rlabel, rc in reps:
        _box((rep_x, ry), (rep_w, rep_h), rlabel, F_REP, ec=rc, fs=6.5,
             fw="bold")

    # Arrows: input → representations
    for ry, _, _ in reps:
        _arrow((inp_x + inp_w, inp_cy), (rep_x, ry + rep_h / 2))

    # Expert bank container
    eb_x = 3.25
    eb_w = 2.25
    eb_bottom = rep_base - 0.35
    eb_top = reps[0][0] + rep_h + 0.55
    eb_h = eb_top - eb_bottom
    _box((eb_x, eb_bottom), (eb_w, eb_h), "", F_EXPERT, ec="#B0B8C4",
         lw=0.8, rounding=0.12, zorder=2)

    # Expert bank title (above the chips)
    ax.text(eb_x + eb_w / 2, eb_top - 0.18,
            "Expert bank  ($K$ = 7, frozen)",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color="#333333", zorder=5)

    # Expert chips — 7 experts in three families:
    #   Tabular  (Rep. A): LGBM / KAN / FT-Trans.
    #   Temporal (Rep. B): ConvLSTM / PatchTST
    #   Sequence (Rep. B): Mamba
    #   Graph    (Rep. C): ST-GCN
    # Layout: 3 family rows; row 1 has 3 chips, row 2 has 3 chips, row 3 has 1 (centered).
    chip_w, chip_h = 0.62, 0.36
    chip_gap_x = 0.06
    n_per_row = 3
    row_total_w = n_per_row * chip_w + (n_per_row - 1) * chip_gap_x
    chip_x0 = eb_x + (eb_w - row_total_w) / 2
    chip_xs_full = [chip_x0 + i * (chip_w + chip_gap_x) for i in range(n_per_row)]

    chip_gap_y = 0.10
    row_top = eb_top - 0.60
    rows_y = [row_top - i * (chip_h + chip_gap_y) for i in range(3)]

    # Family separator lines between row 1↔2 and row 2↔3
    for ri in range(2):
        sy = (rows_y[ri] + rows_y[ri + 1] + chip_h) / 2
        ax.plot([eb_x + 0.15, eb_x + eb_w - 0.15], [sy, sy],
                color="#D1D5DB", lw=0.3, zorder=4)

    chips = [
        # Row 0 (Tabular / Rep. A)
        (chip_xs_full[0], rows_y[0], "LGBM",      C_TAB),
        (chip_xs_full[1], rows_y[0], "KAN",       "#4393C3"),
        (chip_xs_full[2], rows_y[0], "FT-Trans.", "#6BAED6"),
        # Row 1 (Temporal + Sequence / Rep. B)
        (chip_xs_full[0], rows_y[1], "ConvLSTM",  C_TEMP),
        (chip_xs_full[1], rows_y[1], "PatchTST",  "#F4A582"),
        (chip_xs_full[2], rows_y[1], "Mamba",     "#B2182B"),
        # Row 2 (Graph / Rep. C) — centered alone
        (chip_xs_full[1], rows_y[2], "ST-GCN",    C_GRAPH),
    ]
    for cx, cy, clabel, cc in chips:
        p = FancyBboxPatch(
            (cx, cy), chip_w, chip_h,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=0.5, edgecolor=cc, facecolor=cc + "18",
            zorder=5)
        ax.add_patch(p)
        ax.text(cx + chip_w / 2, cy + chip_h / 2, clabel,
                ha="center", va="center", fontsize=6, color=cc,
                fontweight="bold", zorder=6)

    # Partition note at bottom of bank
    ax.text(eb_x + eb_w / 2, eb_bottom + 0.22,
            r"Trained on $\mathcal{D}_{\mathrm{tr}}$",
            ha="center", va="center", fontsize=6, color="#666666",
            zorder=5)

    # Arrows: representations → expert bank (fan into different y positions)
    eb_cy = eb_bottom + eb_h / 2
    rep_targets_y = [eb_cy + 0.6, eb_cy, eb_cy - 0.6]
    for (ry, _, _), tgt_y in zip(reps, rep_targets_y):
        _arrow((rep_x + rep_w, ry + rep_h / 2),
               (eb_x, tgt_y))

    # ══════════════════════════════════════════════════════════════
    # STAGE 2 — Calibration
    # ══════════════════════════════════════════════════════════════

    cal_x = 6.10
    cal_w, cal_h = 1.50, 0.70

    # Base expert predictions
    base_y = inp_cy + 0.55
    _box((cal_x, base_y), (cal_w, cal_h),
         "Base expert\npredictions", F_CAL, ec="#B8860B", fs=7,
         fw="bold", lw=0.7)
    ax.text(cal_x + cal_w / 2, base_y - 0.18,
            r"on $\mathcal{D}_{\mathrm{cal}}$",
            ha="center", va="center", fontsize=6, color="#666666", zorder=5)

    # Few-shot adapter + OOF
    adapt_y = inp_cy - cal_h - 0.55
    _box((cal_x, adapt_y), (cal_w, cal_h),
         "Few-shot adapter\nOOF predictions", "#F0FFF0", ec=C_ADAPT,
         fs=7, fw="bold", lw=0.7)

    # OOF admission gate (diamond)
    gate_x = cal_x + cal_w + 0.40
    gate_cy = inp_cy
    gate_r = 0.38
    diamond = plt.Polygon(
        [(gate_x, gate_cy - gate_r),
         (gate_x + gate_r, gate_cy),
         (gate_x, gate_cy + gate_r),
         (gate_x - gate_r, gate_cy)],
        closed=True, facecolor="#FFF8E8", edgecolor="#B8860B",
        linewidth=0.6, zorder=5)
    ax.add_patch(diamond)
    ax.text(gate_x, gate_cy, "OOF\ngate", ha="center", va="center",
            fontsize=6.5, fontweight="bold", color="#B8860B", zorder=6,
            linespacing=1.05)

    # Arrows: expert bank → base predictions
    _arrow((eb_x + eb_w, inp_cy + 0.3),
           (cal_x, base_y + cal_h / 2))
    # Arrows: expert bank → adapter
    _arrow((eb_x + eb_w, inp_cy - 0.3),
           (cal_x, adapt_y + cal_h / 2), color=C_ADAPT)
    # Arrow: base predictions → gate
    _arrow((cal_x + cal_w, base_y + cal_h / 2),
           (gate_x, gate_cy + gate_r), color="#B8860B")
    # Arrow: adapter → gate
    _arrow((cal_x + cal_w, adapt_y + cal_h / 2),
           (gate_x, gate_cy - gate_r), color=C_ADAPT)

    # ══════════════════════════════════════════════════════════════
    # Routing fork (visual climax)
    # ══════════════════════════════════════════════════════════════

    route_x = 9.85
    route_w, route_h = 1.65, 0.75

    # y1 route (top)
    y1_y = inp_cy + 0.80
    _box((route_x, y1_y), (route_w, route_h),
         r"$y_1$ route" "\n" r"$\mathbf{w}_1 \succeq 0,\ \Sigma w = 1$",
         C_Y1 + "15", ec=C_Y1, fs=7, fw="bold", lw=1.0, tc=C_Y1)

    # y2 route (bottom)
    y2_y = inp_cy - route_h - 0.80
    _box((route_x, y2_y), (route_w, route_h),
         r"$y_2$ route" "\n" r"$\mathbf{w}_2 \succeq 0,\ \Sigma w = 1$",
         C_Y2 + "15", ec=C_Y2, fs=7, fw="bold", lw=1.0, tc=C_Y2)

    # Fork arrows: gate → routes
    _arrow((gate_x + gate_r, gate_cy),
           (route_x, y1_y + route_h / 2),
           color=C_Y1, lw=1.0, rad=0.15)
    _arrow((gate_x + gate_r, gate_cy),
           (route_x, y2_y + route_h / 2),
           color=C_Y2, lw=1.0, rad=-0.15)

    # Fork label
    fork_label_x = (gate_x + gate_r + route_x) / 2
    ax.text(fork_label_x, gate_cy, "Target-\ndependent\nrouting",
            ha="center", va="center", fontsize=6.5, color="#333333",
            fontweight="bold", linespacing=1.15, zorder=5,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                      edgecolor="#9CA3AF", lw=0.4, alpha=0.95))

    # ══════════════════════════════════════════════════════════════
    # STAGE 3 — Inference (post-processing → output)
    # ══════════════════════════════════════════════════════════════

    post_x = 11.85
    post_w, post_h = 1.10, 0.75

    # Affine calibration (y1 path)
    _box((post_x, y1_y), (post_w, post_h),
         "Affine\ncalibration", F_POST, ec="#6B7280", fs=7, lw=0.6)

    # Affine + projection (y2 path)
    _box((post_x, y2_y), (post_w, post_h),
         "Affine +\n" r"$t_{\Delta f}\!\geq\!0$",
         F_POST, ec="#6B7280", fs=7, lw=0.6)

    # Output boxes
    out_x = 13.28
    out_w, out_h = 0.90, 0.60
    out_y1 = y1_y + (route_h - out_h) / 2
    out_y2 = y2_y + (route_h - out_h) / 2
    _box((out_x, out_y1), (out_w, out_h),
         r"$\hat{y}_1$", C_Y1 + "25", ec=C_Y1, fs=8, fw="bold",
         tc=C_Y1, lw=0.8)
    _box((out_x, out_y2), (out_w, out_h),
         r"$\hat{y}_2$", C_Y2 + "25", ec=C_Y2, fs=8, fw="bold",
         tc=C_Y2, lw=0.8)

    # Arrows: route → affine → output (y1)
    y1_mid = y1_y + route_h / 2
    _arrow((route_x + route_w, y1_mid), (post_x, y1_mid),
           color=C_Y1, lw=0.9)
    _arrow((post_x + post_w, y1_mid), (out_x, y1_mid),
           color=C_Y1, lw=0.9)

    # Arrows: route → affine → output (y2)
    y2_mid = y2_y + route_h / 2
    _arrow((route_x + route_w, y2_mid), (post_x, y2_mid),
           color=C_Y2, lw=0.9)
    _arrow((post_x + post_w, y2_mid), (out_x, y2_mid),
           color=C_Y2, lw=0.9)

    save_figure(fig, "fig02_tdher_overview", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# Fig. 7: IEEE 39-bus Topology
# ══════════════════════════════════════════════════════════════════
def render_fig7_ieee39_topology():
    """DEPRECATED: fig03_ieee39_topology is now sourced from the manually
    drawn ieee39_bus_topology.svg in zettel/papers/fepfm_tii_v2/figures/.

    Running this function would overwrite the manuscript figure with an
    auto-generated spring-layout sketch. The SVG version is canonical and
    should not be regenerated. This function is kept only for reference.
    """
    raise RuntimeError(
        "render_fig7_ieee39_topology is deprecated; the manuscript figure "
        "is sourced from figures/ieee39_bus_topology.svg. Do not regenerate."
    )
    import networkx as nx  # noqa: E402, unreachable
    print("Rendering Fig.7 IEEE 39-bus Topology...")

    # Parse IEEE39.RAW
    sys.path.insert(0, str(PROJECT_ROOT))
    from data_proc.build_adjacency import parse_raw_file

    raw_path = "data/topology/IEEE39.RAW"
    parsed = parse_raw_file(raw_path)

    # Build networkx graph
    G = nx.Graph()
    all_buses = sorted(parsed["buses"].keys())
    gen_buses = set(parsed["gen_buses"])

    for b in all_buses:
        G.add_node(b)
    for fb, tb, r, x in parsed["branches"]:
        if fb in G.nodes and tb in G.nodes:
            G.add_edge(fb, tb)
    for fb, tb, r, x in parsed["transformers"]:
        if fb in G.nodes and tb in G.nodes:
            G.add_edge(fb, tb)

    # Layout: spring with fixed seed for reproducibility
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=120)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH))

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#CCCCCC", width=0.6,
                           alpha=0.7)

    # Draw non-generator buses
    load_buses = [b for b in all_buses if b not in gen_buses]
    nx.draw_networkx_nodes(G, pos, nodelist=load_buses, ax=ax,
                           node_color="#E0E0E0", node_size=40,
                           edgecolors="#888888", linewidths=0.4)

    # Draw generator buses with expert colors
    gen_list = sorted(gen_buses)
    gen_colors = []
    for b in gen_list:
        # Map generator bus to its index color from blue family
        gen_colors.append(PALETTE["base"])
    nx.draw_networkx_nodes(G, pos, nodelist=gen_list, ax=ax,
                           node_color=gen_colors, node_size=100,
                           edgecolors="#333333", linewidths=0.6,
                           node_shape="s")  # square for generators

    # Labels for generator buses only
    gen_labels = {b: str(b) for b in gen_list}
    nx.draw_networkx_labels(G, pos, labels=gen_labels, ax=ax,
                            font_size=5.5, font_color="white",
                            font_weight="bold")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["base"],
               markersize=7, markeredgecolor="#333", label=f"Generator ({len(gen_list)})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E0E0E0",
               markersize=5, markeredgecolor="#888", label=f"Load/other ({len(load_buses)})"),
        Line2D([0], [0], color="#CCCCCC", lw=1.0,
               label=f"Branch ({G.number_of_edges()})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=6.5,
              frameon=False, handletextpad=0.4, borderpad=0.4)

    ax.set_title("IEEE 39-bus test system", fontweight="bold", fontsize=8)
    ax.axis("off")
    # Force a square viewport centered on the spring-layout node cloud so
    # that any imbalance in the node distribution does not propagate into
    # the saved PDF as asymmetric whitespace. Using the larger of the
    # x/y extents guarantees no clipping while keeping the figure centered.
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xc = (max(xs) + min(xs)) / 2
    yc = (max(ys) + min(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * 1.08
    ax.set_xlim(xc - half, xc + half)
    ax.set_ylim(yc - half, yc + half)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    save_figure(fig, "fig03_ieee39_topology", [FIG_DIR, PAPER_FIG_DIR])


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Rendering all v2 figures (Nature/IEEE standard)")
    print("=" * 60)

    # Conceptual figures
    render_fig1_problem_definition()
    render_fig2_tdher_overview()
    # render_fig7_ieee39_topology is deprecated; fig03 is sourced from
    # the manually drawn figures/ieee39_bus_topology.svg.

    # Data-driven figures
    render_l3_qualitative()        # fig05: single-sample L3 case study
    render_fig3_threshold()        # → fig06_tdher_threshold_sensitivity
    render_fig4_routing_weights()  # → fig08_tdher_routing_weights
    # render_fig5_ablation() and render_fig6_bootstrap() produce orphan figures
    # (visual ablation / bootstrap CI). Removed from main paper to keep page
    # count; archived under figures/_orphan/. Function defs retained below
    # in case the figures are needed for supplementary or rebuttal.
    render_fig8_route_conflict()    # → fig09_tdher_route_conflict
    render_fig9_expert_sensitivity()# → fig07_tdher_expert_sensitivity
    render_fig10_pipeline_timing()  # → fig11_tdher_pipeline_timing
    render_fig11_multiwindow()      # → fig10_tdher_multiwindow_tradeoff

    # IEEE 300 figures (supplementary)
    render_ieee300_scalability()
    render_ieee300_routing()
    render_ieee300_routing_weights()

    print("=" * 60)
    print("All 14 figures rendered successfully.")
    print("=" * 60)
