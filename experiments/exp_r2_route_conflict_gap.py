#!/usr/bin/env python3
"""Empirical audit of the target-wise versus shared routing conflict gap.

For a fixed common bank of base experts, the target-wise feasible set contains
the diagonal (shared-route) feasible set.  This script recomputes both
calibration optima and verifies the resulting nonnegative conflict gap and a
Lipschitz upper bound.  Test labels are never loaded.
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
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REP_DIR = Path(str(data_path('ieee39_v8_80_10_10')))
ROUTER_DIR = work_path('results/ieee39/exp_tdh_router')
OUT_DIR = work_path('results/ieee39/revision_r2_route_conflict')
PAPER_TABLE = work_path('results/paper_tables/tdher_conflict_gap_revision.csv')
MS = 10
EXPERTS = (
    "LightGBM", "KAN", "ConvLSTM", "PatchTST", "Mamba", "ST-GCN",
    "FT-Transformer",
)
SCENARIOS = {
    "L1": "val",
    "L2": "cross_cond_finetune",
    "L3": "cross_cond_topo_finetune",
}


def optimize_simplex(objective, n_experts: int) -> tuple[np.ndarray, float]:
    result = minimize(
        objective,
        np.full(n_experts, 1.0 / n_experts),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_experts,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weight = np.clip(result.x, 0.0, 1.0)
    weight /= weight.sum()
    return weight, float(objective(weight))


def target_risk(y: np.ndarray, predictions: np.ndarray, weight: np.ndarray) -> float:
    return float(np.mean(np.abs(y - predictions @ weight)))


def main() -> None:
    rep_a = REP_DIR / "repA" / f"ms{MS}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    details = {
        "claim": (
            "For a fixed common expert bank and positive target weights, "
            "the optimal target-wise empirical risk is no larger than the "
            "optimal shared-route empirical risk."
        ),
        "test_labels_loaded": False,
        "normalization": "per-target best single-expert calibration MAE",
        "scenarios": {},
    }

    for scenario, cal_split in SCENARIOS.items():
        y = np.load(rep_a / f"y_{cal_split}.npy")
        prediction_by_expert = []
        for expert in EXPERTS:
            prediction_by_expert.append(
                np.load(
                    ROUTER_DIR / "experts" / expert / f"{cal_split}_preds.npy"
                )
            )
        prediction = np.stack(prediction_by_expert, axis=1)
        n_experts = prediction.shape[1]

        scales = []
        target_weights = []
        target_mae = []
        target_matrices = []
        for target_idx in range(2):
            matrix = prediction[:, :, target_idx]
            target_matrices.append(matrix)
            expert_mae = np.mean(
                np.abs(matrix - y[:, target_idx, None]), axis=0
            )
            scale = float(np.min(expert_mae))
            scales.append(scale)
            weight, risk = optimize_simplex(
                lambda w, m=matrix, t=y[:, target_idx]: target_risk(t, m, w),
                n_experts,
            )
            target_weights.append(weight)
            target_mae.append(risk)

        target_wise_value = 0.5 * sum(
            risk / scale for risk, scale in zip(target_mae, scales)
        )

        def shared_objective(weight: np.ndarray) -> float:
            return 0.5 * sum(
                target_risk(y[:, idx], target_matrices[idx], weight) / scales[idx]
                for idx in range(2)
            )

        shared_weight, shared_value = optimize_simplex(shared_objective, n_experts)
        shared_target_mae = [
            target_risk(y[:, idx], target_matrices[idx], shared_weight)
            for idx in range(2)
        ]
        gap = shared_value - target_wise_value
        route_distance = float(np.sum(np.abs(target_weights[0] - target_weights[1])))

        mean_infinity_norm = [
            float(np.mean(np.max(np.abs(matrix), axis=1)))
            for matrix in target_matrices
        ]
        bound_via_y1_route = (
            0.5 * mean_infinity_norm[1] / scales[1] * route_distance
        )
        bound_via_y2_route = (
            0.5 * mean_infinity_norm[0] / scales[0] * route_distance
        )
        upper_bound = min(bound_via_y1_route, bound_via_y2_route)
        tolerance = 1e-7
        if gap < -tolerance:
            raise AssertionError(f"Negative conflict gap in {scenario}: {gap}")
        if gap > upper_bound + tolerance:
            raise AssertionError(
                f"Conflict bound failed in {scenario}: {gap} > {upper_bound}"
            )

        row = {
            "scenario": scenario,
            "n_calibration": len(y),
            "target_wise_normalized_risk": target_wise_value,
            "shared_normalized_risk": shared_value,
            "conflict_gap": gap,
            "relative_gap_pct": 100.0 * gap / max(target_wise_value, 1e-12),
            "route_l1_distance": route_distance,
            "lipschitz_upper_bound": upper_bound,
            "y1_targetwise_MAE": target_mae[0],
            "y1_shared_MAE": shared_target_mae[0],
            "y2_targetwise_MAE": target_mae[1],
            "y2_shared_MAE": shared_target_mae[1],
        }
        rows.append(row)
        details["scenarios"][scenario] = {
            **row,
            "normalization_scales": {"y1": scales[0], "y2": scales[1]},
            "targetwise_weights": {
                "y1": dict(zip(EXPERTS, map(float, target_weights[0]))),
                "y2": dict(zip(EXPERTS, map(float, target_weights[1]))),
            },
            "shared_weights": dict(zip(EXPERTS, map(float, shared_weight))),
            "bound_via_y1_route": bound_via_y1_route,
            "bound_via_y2_route": bound_via_y2_route,
        }

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "summary.csv", index=False)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PAPER_TABLE, index=False)
    with open(OUT_DIR / "audit.json", "w") as handle:
        json.dump(details, handle, indent=2)
    print(frame.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
