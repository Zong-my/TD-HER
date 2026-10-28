#!/usr/bin/env python3
"""Matched one-component-at-a-time ablation for Reviewer 2.

The script reuses frozen test predictions from the audited TD-HER run.  Every
row differs from the full method in one named mechanism while preserving the
remaining routing, affine, and feasibility choices whenever the saved
artifacts permit.  It never refits or selects a variant on test labels.
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
OUT_DIR = work_path('results/ieee39/revision_r2_component_ablation')
PAPER_TABLE = (
    work_path('results/paper_tables/tdher_component_ablation_revision.csv')
)
MS = 10

SCENARIOS = {
    "L1": ("L1_same_dist", "test"),
    "L2": ("L2_cross_cond_fewshot", "cross_cond_test"),
    "L3": ("L3_cross_topo_fewshot", "cross_cond_topo_test"),
}


def project_arrival_time(prediction: np.ndarray) -> np.ndarray:
    projected = np.asarray(prediction, dtype=float).copy()
    projected[:, 1] = np.maximum(projected[:, 1], 0.0)
    return projected


def evaluate(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    output = {}
    for target_idx, target in enumerate(("y1", "y2")):
        error = prediction[:, target_idx] - y_true[:, target_idx]
        output[f"{target}_MAE"] = float(np.mean(np.abs(error)))
        output[f"{target}_RMSE"] = float(np.sqrt(np.mean(error ** 2)))
    output["negative_y2_count"] = int(np.sum(prediction[:, 1] < 0.0))
    return output


def load_prediction(directory: Path, stem: str) -> np.ndarray:
    path = directory / f"{stem}_preds.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def percent_change(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / max(abs(reference), 1e-12)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    audit = {
        "protocol": {
            "selection_on_test": False,
            "purpose": "matched one-component-at-a-time evidence audit",
            "interpretation": (
                "A safeguard may preserve feasibility or reject harmful "
                "adaptation without improving every test-set average."
            ),
        },
        "scenarios": {},
    }

    for scenario, (scenario_dir_name, test_split) in SCENARIOS.items():
        scenario_dir = ROUTER_DIR / scenario_dir_name
        y_true = np.load(REP_DIR / "repA" / f"ms{MS}" / f"y_{test_split}.npy")

        full = load_prediction(
            scenario_dir,
            "certified_admission_affine_convex_blend_nonnegative_y2",
        )
        practical_gate = load_prediction(
            scenario_dir, "admission_affine_convex_blend_nonnegative_y2"
        )
        shared = load_prediction(
            scenario_dir, "admission_shared_affine_convex_blend_nonnegative_y2"
        )
        no_adapter = load_prediction(
            scenario_dir, "base_affine_convex_blend_nonnegative_y2"
        )
        always_adapt = project_arrival_time(
            load_prediction(scenario_dir, "affine_convex_blend")
        )
        no_affine = project_arrival_time(
            load_prediction(scenario_dir, "certified_admission_convex_blend")
        )
        no_projection = load_prediction(
            scenario_dir, "certified_admission_affine_convex_blend"
        )

        variants = {
            "Full TD-HER": {
                "prediction": full,
                "removed_component": "none",
                "role": "reference",
            },
            "Shared route": {
                "prediction": shared,
                "removed_component": "target-wise routing",
                "role": "predictive specialization",
            },
            "No adapter candidate": {
                "prediction": no_adapter,
                "removed_component": "few-shot adapter candidate",
                "role": "cross-regime adaptation",
            },
            "No admission gate": {
                "prediction": always_adapt,
                "removed_component": "OOF practical and bootstrap gate",
                "role": "degradation prevention",
            },
            "No affine correction": {
                "prediction": no_affine,
                "removed_component": "restricted affine correction",
                "role": "low-dimensional calibration",
            },
            "No y2 projection": {
                "prediction": no_projection,
                "removed_component": "nonnegative arrival-time projection",
                "role": "physical feasibility",
            },
            "No bootstrap certificate": {
                "prediction": practical_gate,
                "removed_component": "paired-bootstrap certificate",
                "role": "finite-sample admission evidence",
            },
        }

        full_metrics = evaluate(y_true, full)
        scenario_audit = {
            "n_test": int(len(y_true)),
            "practical_and_certified_decisions_identical": bool(
                np.array_equal(full, practical_gate)
            ),
            "full_metrics": full_metrics,
            "variants": {},
        }

        for variant, specification in variants.items():
            prediction = specification["prediction"]
            if prediction.shape != y_true.shape:
                raise ValueError(
                    f"{scenario}/{variant}: {prediction.shape} != {y_true.shape}"
                )
            result = evaluate(y_true, prediction)
            row = {
                "scenario": scenario,
                "variant": variant,
                "removed_component": specification["removed_component"],
                "component_role": specification["role"],
                "n_test": len(y_true),
                **result,
                "y1_MAE_change_vs_full_pct": percent_change(
                    result["y1_MAE"], full_metrics["y1_MAE"]
                ),
                "y2_MAE_change_vs_full_pct": percent_change(
                    result["y2_MAE"], full_metrics["y2_MAE"]
                ),
            }
            rows.append(row)
            scenario_audit["variants"][variant] = row

        audit["scenarios"][scenario] = scenario_audit

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "summary.csv", index=False)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PAPER_TABLE, index=False)
    with open(OUT_DIR / "audit.json", "w") as handle:
        json.dump(audit, handle, indent=2)

    print(frame.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
