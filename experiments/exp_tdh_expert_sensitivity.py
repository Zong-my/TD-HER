#!/usr/bin/env python3
"""Expert-bank sensitivity analysis for TD-HER.

This script reuses the audited prediction caches produced by
``experiments/exp_tdh_router.py``. It does not retrain any base expert. The goal
is to quantify whether the heterogeneous expert bank is actually useful:

1. leave-one-expert-out routing sensitivity;
2. top-m expert subset performance;
3. prediction diversity versus routing gain.

Outputs are written to ``results/paper_tables`` and
``results/ieee39/tdher_expert_sensitivity``.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REP_DIR = Path("data/ieee39_v8_80_10_10")
ROUTER_DIR = PROJECT_ROOT / "results/ieee39/exp_tdh_router"
OUT_DIR = PROJECT_ROOT / "results/ieee39/tdher_expert_sensitivity"
TABLE_DIR = PROJECT_ROOT / "results/paper_tables"

SCENARIOS = {
    "L1_same_dist": {
        "label": "L1 same distribution",
        "cal_split": "val",
        "test_split": "test",
    },
    "L2_cross_cond_fewshot": {
        "label": "L2 cross-condition few-shot",
        "cal_split": "cross_cond_finetune",
        "test_split": "cross_cond_test",
    },
    "L3_cross_topo_fewshot": {
        "label": "L3 cross-topology few-shot",
        "cal_split": "cross_cond_topo_finetune",
        "test_split": "cross_cond_topo_test",
    },
}

TARGETS = ["y1", "y2"]
FINAL_METHOD = "admission_affine_convex_blend_nonnegative_y2"
ADAPTER_NAME = "LightGBM-Adapter"


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def load_y(split: str) -> np.ndarray:
    return np.load(REP_DIR / f"repA/ms10/y_{split}.npy")


def load_prediction(scenario: str, model: str, split_kind: str) -> np.ndarray:
    """Load cached calibration/test prediction for one expert.

    Parameters
    ----------
    scenario:
        L1/L2/L3 scenario key.
    model:
        Expert name. ``LightGBM-Adapter`` is stored separately from frozen
        experts.
    split_kind:
        Either ``cal`` or ``test``.
    """
    if model == ADAPTER_NAME:
        adapter_dir = ROUTER_DIR / "adapters" / scenario / ADAPTER_NAME
        filename = "cal_oof_preds.npy" if split_kind == "cal" else "test_preds.npy"
        return np.load(adapter_dir / filename)

    split = SCENARIOS[scenario]["cal_split" if split_kind == "cal" else "test_split"]
    return np.load(ROUTER_DIR / "experts" / model / f"{split}_preds.npy")


def optimize_convex_weights(y: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, float]:
    n_models = preds.shape[1]

    def objective(w: np.ndarray) -> float:
        return mae(y, preds @ w)

    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bounds = [(0.0, 1.0)] * n_models
    x0 = np.full(n_models, 1.0 / n_models)
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    w = np.clip(res.x if res.success else x0, 0.0, 1.0)
    w = w / max(float(np.sum(w)), 1e-12)
    return w, objective(w)


def fit_restricted_affine(y_cal: np.ndarray, pred_cal: np.ndarray) -> tuple[float, float, float]:
    """Fit y ~= a * pred + b with the same small, bounded calibration layer."""
    y = np.asarray(y_cal, dtype=float).reshape(-1)
    z = np.asarray(pred_cal, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(z)
    y = y[finite]
    z = z[finite]
    if len(y) == 0:
        return 1.0, 0.0, float("nan")
    if np.std(z) < 1e-12:
        b = float(np.median(y))
        return 0.0, b, mae(y, np.full_like(y, b))

    try:
        a0, b0 = np.linalg.lstsq(np.column_stack([z, np.ones_like(z)]), y, rcond=None)[0]
    except np.linalg.LinAlgError:
        a0, b0 = 1.0, 0.0
    a0 = float(np.clip(a0, 0.0, 3.0)) if np.isfinite(a0) else 1.0
    b0 = float(b0) if np.isfinite(b0) else float(np.median(y - a0 * z))

    scale = max(
        float(np.ptp(y)),
        float(np.std(y)),
        float(np.mean(np.abs(y))),
        float(np.mean(np.abs(z))),
        1e-6,
    )
    b_bound = 5.0 * scale

    def objective(params: np.ndarray) -> float:
        a, b = params
        return mae(y, a * z + b) + 1e-8 * ((a - 1.0) ** 2 + (b / scale) ** 2)

    res = minimize(
        objective,
        x0=np.array([a0, float(np.clip(b0, -b_bound, b_bound))]),
        method="SLSQP",
        bounds=[(0.0, 3.0), (-b_bound, b_bound)],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if res.success and np.all(np.isfinite(res.x)):
        a, b = float(res.x[0]), float(res.x[1])
    else:
        a, b = a0, float(np.clip(b0, -b_bound, b_bound))
    return a, b, mae(y, a * z + b)


def route_target(
    y_cal: np.ndarray,
    y_test: np.ndarray,
    cal_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    target_idx: int,
) -> dict:
    models = list(cal_preds)
    p_cal = np.stack([cal_preds[m][:, target_idx] for m in models], axis=1)
    p_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
    weights, blend_cal_mae = optimize_convex_weights(y_cal[:, target_idx], p_cal)
    cal_blend = p_cal @ weights
    test_blend = p_test @ weights
    slope, bias, affine_cal_mae = fit_restricted_affine(y_cal[:, target_idx], cal_blend)
    pred = slope * test_blend + bias
    if target_idx == 1:
        pred = np.maximum(pred, 0.0)
    return {
        "test_mae": mae(y_test[:, target_idx], pred),
        "blend_cal_mae": blend_cal_mae,
        "affine_cal_mae": affine_cal_mae,
        "slope": slope,
        "bias": bias,
        "weights": {m: float(w) for m, w in zip(models, weights)},
    }


def best_single_affine(
    y_cal: np.ndarray,
    y_test: np.ndarray,
    cal_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    target_idx: int,
) -> dict:
    candidates = []
    for model in cal_preds:
        pred_cal = cal_preds[model][:, target_idx]
        pred_test = test_preds[model][:, target_idx]
        slope, bias, cal_mae = fit_restricted_affine(y_cal[:, target_idx], pred_cal)
        pred = slope * pred_test + bias
        if target_idx == 1:
            pred = np.maximum(pred, 0.0)
        candidates.append({
            "model": model,
            "cal_mae": cal_mae,
            "test_mae": mae(y_test[:, target_idx], pred),
            "slope": slope,
            "bias": bias,
        })
    return min(candidates, key=lambda row: row["cal_mae"])


def prediction_diversity(y_cal: np.ndarray, cal_preds: dict[str, np.ndarray],
                         target_idx: int) -> dict:
    models = list(cal_preds)
    if len(models) < 2:
        return {
            "mean_pairwise_abs_disagreement": 0.0,
            "normalized_disagreement_by_best_cal_mae": 0.0,
            "mean_pairwise_error_corr": float("nan"),
            "n_pairs": 0,
        }
    disagreements = []
    error_corrs = []
    errors = {
        m: np.abs(y_cal[:, target_idx] - cal_preds[m][:, target_idx])
        for m in models
    }
    best_cal_mae = min(float(np.mean(e)) for e in errors.values())
    for left, right in combinations(models, 2):
        disagreements.append(
            float(np.mean(np.abs(
                cal_preds[left][:, target_idx] - cal_preds[right][:, target_idx]
            )))
        )
        if np.std(errors[left]) > 1e-12 and np.std(errors[right]) > 1e-12:
            corr = float(np.corrcoef(errors[left], errors[right])[0, 1])
            if np.isfinite(corr):
                error_corrs.append(corr)
    mean_disagreement = float(np.mean(disagreements))
    return {
        "mean_pairwise_abs_disagreement": mean_disagreement,
        "normalized_disagreement_by_best_cal_mae": (
            mean_disagreement / best_cal_mae if best_cal_mae > 0 else float("nan")
        ),
        "mean_pairwise_error_corr": (
            float(np.mean(error_corrs)) if error_corrs else float("nan")
        ),
        "n_pairs": len(disagreements),
    }


def target_models_from_summary(summary: dict, target: str) -> list[str]:
    details = summary[FINAL_METHOD]["details"]["source_details"][target]
    return list(details["weights"])


def target_weights_from_summary(summary: dict, target: str) -> dict[str, float]:
    return {
        model: float(weight)
        for model, weight in (
            summary[FINAL_METHOD]["details"]["source_details"][target]["weights"].items()
        )
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    leave_one_rows = []
    subset_rows = []
    diversity_rows = []
    detail = {}

    for scenario, spec in SCENARIOS.items():
        label = spec["label"]
        summary = read_json(ROUTER_DIR / "scenarios" / scenario / "metrics_summary.json")
        y_cal = load_y(spec["cal_split"])
        y_test = load_y(spec["test_split"])
        detail[scenario] = {}

        for target_idx, target in enumerate(TARGETS):
            allowed_models = target_models_from_summary(summary, target)
            final_mae = float(summary[FINAL_METHOD]["metrics"][f"{target}_MAE"])
            final_weights = target_weights_from_summary(summary, target)
            cal_preds = {
                model: load_prediction(scenario, model, "cal")
                for model in allowed_models
            }
            test_preds = {
                model: load_prediction(scenario, model, "test")
                for model in allowed_models
            }

            full_recomputed = route_target(
                y_cal, y_test, cal_preds, test_preds, target_idx)
            best_single = best_single_affine(
                y_cal, y_test, cal_preds, test_preds, target_idx)
            div = prediction_diversity(y_cal, cal_preds, target_idx)
            routing_gain = best_single["test_mae"] - final_mae
            diversity_rows.append({
                "scenario": label,
                "target": target,
                "n_experts": len(allowed_models),
                "best_single_model": best_single["model"],
                "best_single_test_mae": best_single["test_mae"],
                "full_route_test_mae": final_mae,
                "routing_gain_vs_best_single": routing_gain,
                "routing_gain_relative": (
                    routing_gain / best_single["test_mae"]
                    if best_single["test_mae"] > 0 else float("nan")
                ),
                **div,
            })

            for omitted in allowed_models:
                if len(allowed_models) <= 1:
                    continue
                kept = [model for model in allowed_models if model != omitted]
                result = route_target(
                    y_cal,
                    y_test,
                    {m: cal_preds[m] for m in kept},
                    {m: test_preds[m] for m in kept},
                    target_idx,
                )
                delta = result["test_mae"] - final_mae
                leave_one_rows.append({
                    "scenario": label,
                    "target": target,
                    "omitted_expert": omitted,
                    "full_tdher_mae": final_mae,
                    "leave_one_out_mae": result["test_mae"],
                    "delta_mae_vs_full": delta,
                    "relative_loss_vs_full": (
                        delta / final_mae if final_mae > 0 else float("nan")
                    ),
                    "original_route_weight": final_weights.get(omitted, 0.0),
                    "kept_experts": ";".join(kept),
                })

            rank = sorted(
                allowed_models,
                key=lambda m: mae(y_cal[:, target_idx], cal_preds[m][:, target_idx]),
            )
            subset_sizes = sorted({1, 2, 3, len(allowed_models)})
            for subset_size in subset_sizes:
                subset = rank[:subset_size]
                if subset_size == len(allowed_models):
                    subset_mae = final_mae
                else:
                    result = route_target(
                        y_cal,
                        y_test,
                        {m: cal_preds[m] for m in subset},
                        {m: test_preds[m] for m in subset},
                        target_idx,
                    )
                    subset_mae = result["test_mae"]
                delta = subset_mae - final_mae
                subset_rows.append({
                    "scenario": label,
                    "target": target,
                    "subset_size": subset_size,
                    "subset_experts": ";".join(subset),
                    "full_expert_count": len(allowed_models),
                    "full_tdher_mae": final_mae,
                    "subset_route_mae": subset_mae,
                    "delta_mae_vs_full": delta,
                    "relative_loss_vs_full": (
                        delta / final_mae if final_mae > 0 else float("nan")
                    ),
                })

            detail[scenario][target] = {
                "allowed_models": allowed_models,
                "final_saved_mae": final_mae,
                "full_recomputed": full_recomputed,
                "best_single_affine": best_single,
                "diversity": div,
            }

    write_csv(
        TABLE_DIR / "tdher_expert_leave_one_out.csv",
        [
            "scenario",
            "target",
            "omitted_expert",
            "full_tdher_mae",
            "leave_one_out_mae",
            "delta_mae_vs_full",
            "relative_loss_vs_full",
            "original_route_weight",
            "kept_experts",
        ],
        leave_one_rows,
    )
    write_csv(
        TABLE_DIR / "tdher_expert_subset_size.csv",
        [
            "scenario",
            "target",
            "subset_size",
            "subset_experts",
            "full_expert_count",
            "full_tdher_mae",
            "subset_route_mae",
            "delta_mae_vs_full",
            "relative_loss_vs_full",
        ],
        subset_rows,
    )
    write_csv(
        TABLE_DIR / "tdher_expert_diversity.csv",
        [
            "scenario",
            "target",
            "n_experts",
            "best_single_model",
            "best_single_test_mae",
            "full_route_test_mae",
            "routing_gain_vs_best_single",
            "routing_gain_relative",
            "mean_pairwise_abs_disagreement",
            "normalized_disagreement_by_best_cal_mae",
            "mean_pairwise_error_corr",
            "n_pairs",
        ],
        diversity_rows,
    )

    with (OUT_DIR / "expert_sensitivity_summary.json").open("w") as f:
        json.dump(detail, f, indent=2)

    print(f"Wrote {TABLE_DIR / 'tdher_expert_leave_one_out.csv'}")
    print(f"Wrote {TABLE_DIR / 'tdher_expert_subset_size.csv'}")
    print(f"Wrote {TABLE_DIR / 'tdher_expert_diversity.csv'}")
    print(f"Wrote {OUT_DIR / 'expert_sensitivity_summary.json'}")


if __name__ == "__main__":
    main()
