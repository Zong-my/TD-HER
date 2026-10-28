#!/usr/bin/env python3
"""Label-budget study - figure and descriptive tables (no training; reads result.json files only).

Figure: test MAE against the number of labelled shifted-domain events for "always include the adapter"
(rule C), "never include" (rule D) and the margin rule (A, the method), median over subsampling seeds with
the range for C and D, and the number of seeds in which A (and, where different, the two-criterion rule B)
admits the adapter printed under each budget. Because admission is per target, the output of A equals C
where it admits and D where it rejects; its median is drawn so that the error of the method itself, and
not only that of the two reference policies, is visible (Codex, results review).

Writes into <out>/figures/ and never overwrites: an existing file name gets a numeric suffix.
Style: experiments/figure_style.py, as for the other figures of the manuscript.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_label_budget as RB  # noqa: E402  (grid constants; chdir to the project root)
from experiments.figure_style import apply_style, clean_axes, PALETTE, COL_WIDTH  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

TARGETS = [("y1", r"$y_1$ MAE (Hz)"), ("y2", r"$y_2$ MAE (s)")]


def fresh(path):
    if not path.exists():
        return path
    i = 2
    while path.with_name(f"{path.stem}_{i}{path.suffix}").exists():
        i += 1
    return path.with_name(f"{path.stem}_{i}{path.suffix}")


def collect(out):
    plan = RB.load_json(out / "run_plan.json") if (out / "run_plan.json").exists() else None
    grid = [tuple(c) for c in plan["configs"]] if plan else RB.all_configs(RB.N_SEEDS)
    cells = {}
    for scen, n, k in grid:
        p = out / "configs" / RB.config_name(scen, n, k) / "result.json"
        if p.exists():
            res = RB.load_json(p)
            for t, v in res["targets"].items():
                cells.setdefault((scen, t, n), []).append(v)
    return cells, len(grid)


def summarise(cells):
    rows = []
    for (scen, t, n), vs in sorted(cells.items()):
        c = np.array([v["test_MAE"]["C_always"] for v in vs])
        d = np.array([v["test_MAE"]["D_never"] for v in vs])
        lo = np.array([v["C_minus_D"]["ci95"][0] for v in vs])
        hi = np.array([v["C_minus_D"]["ci95"][1] for v in vs])
        a = np.array([v["A_admits"] for v in vs])
        b = np.array([v["B_admits"] for v in vs])
        helps, harms = hi < 0, lo > 0
        rows.append({
            "scenario": scen, "target": t, "n": n, "seeds": len(vs),
            "A_admits": int(a.sum()), "B_admits": int(b.sum()), "A_admits_B_rejects": int((a & ~b).sum()),
            "adapter_helps_interval_below_0": int(helps.sum()), "adapter_harms_interval_above_0": int(harms.sum()),
            "unresolved": int((~helps & ~harms).sum()),
            "A_harmful_admissions": int((a & harms).sum()), "A_missed_recoveries": int((~a & helps).sum()),
            "B_harmful_admissions": int((b & harms).sum()), "B_missed_recoveries": int((~b & helps).sum()),
            "MAE_C_median": float(np.median(c)), "MAE_C_min": float(c.min()), "MAE_C_max": float(c.max()),
            "MAE_D_median": float(np.median(d)), "MAE_D_min": float(d.min()), "MAE_D_max": float(d.max()),
            "MAE_A_median": float(np.median([v["test_MAE"]["A_margin"] for v in vs])),
            "MAE_B_median": float(np.median([v["test_MAE"]["B_margin_and_bootstrap"] for v in vs])),
            "oof_over_required_median": float(np.median([v["oof_adapter_MAE"] / v["required_MAE"] for v in vs])),
            "oof_over_required_min": float(min(v["oof_adapter_MAE"] / v["required_MAE"] for v in vs)),
            "oof_over_required_max": float(max(v["oof_adapter_MAE"] / v["required_MAE"] for v in vs)),
        })
    return rows


def draw(rows, path):
    apply_style()
    scens = [s for s in ("L3", "L2") if any(r["scenario"] == s for r in rows)]
    fig, axes = plt.subplots(len(scens), 2, figsize=(COL_WIDTH, 1.16 * len(scens) + 0.30), squeeze=False)   # compact (2026-09-22)
    col_a, col_c, col_d = "#0072B2", "#009E73", "#E69F00"  # Okabe-Ito; keep overlapping policies visible.
    for i, scen in enumerate(scens):
        for j, (t, ylabel) in enumerate(TARGETS):
            ax = axes[i][j]
            rs = sorted((r for r in rows if r["scenario"] == scen and r["target"] == t), key=lambda r: r["n"])
            n = np.array([r["n"] for r in rs], dtype=float)
            for key, color, marker, label in (("D", col_d, "s", "never include"), ("C", col_c, "o", "always include")):
                med = np.array([r[f"MAE_{key}_median"] for r in rs])
                lo = np.array([r[f"MAE_{key}_min"] for r in rs])
                hi = np.array([r[f"MAE_{key}_max"] for r in rs])
                ax.errorbar(n, med, yerr=[med - lo, hi - med], fmt=marker + "-", color=color, lw=1.65, ms=4.0 if key == "C" else 3.8,
                            markeredgewidth=0.65, elinewidth=0.75, capsize=1.7, capthick=0.75,
                            label=label, zorder=3)
            ax.plot(n, [r["MAE_A_median"] for r in rs], "D--", color=col_a, lw=1.05, ms=2.7, mfc="white",
                    mew=0.85, label=r"margin rule ($\tau=0.20$)", zorder=4)
            ax.set_xscale("log")
            ax.set_xticks(n)
            ax.set_xticklabels([f"{int(v):,}" for v in n], rotation=0)
            ax.minorticks_off()
            ax.tick_params(axis="x", labelsize=6.2, pad=1.5)
            ax.set_ylabel(ylabel, labelpad=2)
            clean_axes(ax)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - 0.24 * (ymax - ymin), ymax)
            for r in rs:
                txt = f"{r['A_admits']}/{r['seeds']}"
                if r["B_admits"] != r["A_admits"]:
                    txt += f"\n({r['B_admits']})"
                ax.annotate(txt, xy=(r["n"], 0.02), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                            fontsize=6.0, color="#333333", linespacing=0.9)
            ax.text(0.20, 0.29 if scen == "L3" else 0.235, scen, transform=ax.transAxes, va="center", ha="center", fontsize=7.3,
                    fontweight="bold", color="#333333")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=3, frameon=False, fontsize=6.6,
               handlelength=1.6, handletextpad=0.35, columnspacing=0.9)
    fig.subplots_adjust(left=0.15, right=0.985, bottom=0.145 if len(scens) > 1 else 0.27, top=0.875, wspace=0.48,
                        hspace=0.36)
    fig.text(0.565, 0.005, "Labeled events from the changed system", ha="center", va="bottom", fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=600 if ext == "pdf" else 300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RB.OUT_DEFAULT))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out)
    cells, n_grid = collect(out)
    rows = summarise(cells)
    n_cells = sum(r["seeds"] for r in rows)
    tag = a.tag or ("final" if n_cells == 2 * n_grid else f"partial_{n_cells}_of_{2 * n_grid}")
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    table = fresh(figdir / f"budget_table_{tag}.json")
    RB.dump_json(rows, table)
    fig_path = fresh(figdir / f"fig_label_budget_{tag}.pdf")
    draw(rows, fig_path)
    print(f"[plot] {n_cells} of {2 * n_grid} cells -> {fig_path.name}, {table.name}")
    hdr = f"{'scen':<4} {'tgt':<3} {'n':>5} {'seeds':>5} {'A adm':>5} {'B adm':>5} {'helps':>5} {'harms':>5} {'unres':>5} " \
          f"{'A harm':>6} {'A miss':>6} {'B harm':>6} {'B miss':>6} {'MAE C med':>10} {'MAE D med':>10} {'oof/req med':>11}"
    print(hdr)
    for r in rows:
        print(f"{r['scenario']:<4} {r['target']:<3} {r['n']:>5} {r['seeds']:>5} {r['A_admits']:>5} {r['B_admits']:>5} "
              f"{r['adapter_helps_interval_below_0']:>5} {r['adapter_harms_interval_above_0']:>5} {r['unresolved']:>5} "
              f"{r['A_harmful_admissions']:>6} {r['A_missed_recoveries']:>6} {r['B_harmful_admissions']:>6} "
              f"{r['B_missed_recoveries']:>6} {r['MAE_C_median']:>10.5f} {r['MAE_D_median']:>10.5f} "
              f"{r['oof_over_required_median']:>11.3f}")


if __name__ == "__main__":
    main()
