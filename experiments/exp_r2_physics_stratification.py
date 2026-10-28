#!/usr/bin/env python3
"""Physics-relevant stratified error audit for Reviewer 2.

The audit conditions the frozen TD-HER predictions on two operator-side
descriptors already present in Representation A: the aggregate-inertia factor
and the load-change magnitude.  It is descriptive rather than causal and does
not tune, select, or refit a model on test labels.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REP_DIR = Path(str(data_path('ieee39_v8_80_10_10')))
ROUTER_DIR = work_path('results/ieee39/exp_tdh_router/scenarios')
OUT_DIR = work_path('results/ieee39/revision_r2_physics_stratification')
PAPER_TABLE = work_path('results/paper_tables/tdher_physics_strata_revision.csv')
MS = 10

SCENARIOS = {
    "L1": ("L1_same_dist", "test"),
    "L2": ("L2_cross_cond_fewshot", "cross_cond_test"),
    "L3": ("L3_cross_topo_fewshot", "cross_cond_topo_test"),
}


def inertia_group(value: np.ndarray) -> np.ndarray:
    return np.select(
        [value <= 0.6, value < 1.4],
        ["low (<=0.6)", "nominal (0.6,1.4)"],
        default="high (>=1.4)",
    )


def load_change_group(value: np.ndarray) -> np.ndarray:
    return np.select(
        [value == 0.0, value <= 100.0, value <= 200.0],
        ["zero", "small (0,100]", "medium (100,200]"],
        default="large (>200)",
    )


def summarize(scenario: str, descriptor: str, group: str,
              indices: np.ndarray, y_true: np.ndarray,
              prediction: np.ndarray, baseline: np.ndarray) -> dict:
    y = y_true[indices]
    pred = prediction[indices]
    base = baseline[indices]
    row = {
        "scenario": scenario,
        "descriptor": descriptor,
        "group": group,
        "n": int(indices.sum()),
        "mean_abs_y1_Hz": float(np.mean(np.abs(y[:, 0]))),
        "mean_y2_s": float(np.mean(y[:, 1])),
        "tdher_y1_MAE": float(np.mean(np.abs(pred[:, 0] - y[:, 0]))),
        "tdher_y2_MAE": float(np.mean(np.abs(pred[:, 1] - y[:, 1]))),
        "base_route_y1_MAE": float(np.mean(np.abs(base[:, 0] - y[:, 0]))),
        "base_route_y2_MAE": float(np.mean(np.abs(base[:, 1] - y[:, 1]))),
        "tdher_negative_y2": int(np.sum(pred[:, 1] < 0.0)),
    }
    row["y1_gain_vs_base_pct"] = 100.0 * (
        row["base_route_y1_MAE"] - row["tdher_y1_MAE"]
    ) / max(row["base_route_y1_MAE"], 1e-12)
    row["y2_gain_vs_base_pct"] = 100.0 * (
        row["base_route_y2_MAE"] - row["tdher_y2_MAE"]
    ) / max(row["base_route_y2_MAE"], 1e-12)
    return row


def main() -> None:
    rep_a = REP_DIR / "repA" / f"ms{MS}"
    with open(rep_a / "feature_names.json") as handle:
        feature_names = json.load(handle)
    inertia_idx = feature_names.index("h_inertia")
    load_delta_idx = feature_names.index("load_delta")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = {
        "selection_on_test": False,
        "analysis_type": "descriptive stratification; no causal interpretation",
        "inertia_bins": ["<=0.6", "(0.6,1.4)", ">=1.4"],
        "load_delta_bins": ["0", "(0,100]", "(100,200]", ">200"],
        "scenarios": {},
    }

    for scenario, (directory_name, split) in SCENARIOS.items():
        x = np.load(rep_a / f"X_{split}.npy")
        y = np.load(rep_a / f"y_{split}.npy")
        directory = ROUTER_DIR / directory_name
        tdher = np.load(
            directory
            / "certified_admission_affine_convex_blend_nonnegative_y2_preds.npy"
        )
        base = np.load(
            directory / "base_affine_convex_blend_nonnegative_y2_preds.npy"
        )
        inertia = x[:, inertia_idx]
        load_delta = x[:, load_delta_idx]

        groups = {
            "aggregate-inertia factor": inertia_group(inertia),
            "load-change magnitude": load_change_group(load_delta),
        }
        scenario_meta = {
            "n_test": int(len(y)),
            "inertia_min": float(np.min(inertia)),
            "inertia_max": float(np.max(inertia)),
            "load_delta_min": float(np.min(load_delta)),
            "load_delta_max": float(np.max(load_delta)),
        }
        for descriptor, labels in groups.items():
            for group in sorted(np.unique(labels)):
                selected = labels == group
                if not np.any(selected):
                    continue
                rows.append(
                    summarize(
                        scenario, descriptor, str(group), selected,
                        y, tdher, base,
                    )
                )
        metadata["scenarios"][scenario] = scenario_meta

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "summary.csv", index=False)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PAPER_TABLE, index=False)
    with open(OUT_DIR / "protocol.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(frame.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
