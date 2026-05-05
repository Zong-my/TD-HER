#!/usr/bin/env python3
"""Route-level robust TD-HER probe.

This experiment tests whether TD-HER can admit a robustness-oriented stable
expert under held-out calibration evidence. It keeps the audited main TD-HER
route frozen, adds two noise-augmented stable LightGBM variants as route-level
experts, and learns a target-wise convex route plus restricted affine
calibration on clean + perturbed validation data.

The script is intentionally separate from ``exp_tdh_router.py``. It is a
diagnostic robustness redesign probe, not a replacement for the current paper
mainline unless the held-out results justify it.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The imported experiment utilities only need cached mRMR feature lists here.
if "mrmr" not in sys.modules:
    mrmr_stub = types.ModuleType("mrmr")

    def _missing_mrmr_regression(*args, **kwargs):
        raise ImportError("mrmr is not required for robust-router inference")

    mrmr_stub.mrmr_regression = _missing_mrmr_regression
    sys.modules["mrmr"] = mrmr_stub

from experiments.exp4_generalization import (  # noqa: E402
    MS,
    REP_DIR,
    evaluate,
    load_train_scalers,
)
from experiments.exp_tdh_router import (  # noqa: E402
    apply_affine,
    enforce_y2_nonnegative,
    fit_restricted_affine,
    optimize_convex_weights,
)
from experiments.exp_tdh_robustness import (  # noqa: E402
    METHOD,
    SENSOR_CONDITIONS,
    apply_tdher_route,
    attach_raw_feature_stats,
    load_sensor_raw_csv_level_data,
    load_summary,
    predict_condition,
)


SCENARIO = "L1_same_dist"
STABLE_DIR = Path("results/ieee39/tdher_robustness/stable_lgbm")
OUT_DIR = Path("results/ieee39/tdher_robustness/robust_router")
PAPER_TABLE_DIR = Path("results/paper_tables")

STABLE_RUNS = {
    "stable_core_aug": "stable_core__aug_sensor_noise_low",
    "raw_trace_aug": "raw_trace__aug_sensor_noise_low",
}

DEFAULT_CAL_CONDITIONS = ["clean_rebuild", "sensor_noise_low"]
DEFAULT_TEST_CONDITIONS = [
    "clean_rebuild",
    "freq_noise_low",
    "volt_noise_low",
    "angl_noise_low",
    "powr_noise_low",
    "spd_noise_low",
    "sensor_noise_low",
    "one_step_gap_1pct",
    "timestamp_lag_1pct",
]


def load_stable_pack(alias: str, run_name: str) -> dict:
    """Load a cached stable LightGBM run without retraining it."""
    run_dir = STABLE_DIR / run_name
    meta_path = run_dir / "feature_policy.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing stable run metadata: {meta_path}. "
            "Run exp_tdh_stable_lgbm_robustness.py first."
        )
    with meta_path.open() as f:
        meta = json.load(f)

    with open(Path(REP_DIR) / "repA" / f"ms{MS}" / "feature_names.json") as f:
        all_features = json.load(f)
    feature_to_idx = {name: idx for idx, name in enumerate(all_features)}
    feat_idx = [feature_to_idx[name] for name in meta["feature_names"]]

    x_train = np.load(Path(REP_DIR) / "repA" / f"ms{MS}" / "X_train.npy")
    scaler = StandardScaler().fit(x_train[:, feat_idx])

    models = {}
    for target in ["y1", "y2"]:
        model_path = run_dir / "models" / f"{target}_stable_lgbm.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing stable model: {model_path}")
        with model_path.open("rb") as f:
            models[target] = pickle.load(f)

    return {
        "alias": alias,
        "run_name": run_name,
        "meta": meta,
        "feat_idx": feat_idx,
        "scaler": scaler,
        "models": models,
    }


def predict_stable_pack(pack: dict, x_a: np.ndarray) -> np.ndarray:
    x_n = pack["scaler"].transform(x_a[:, pack["feat_idx"]]).astype(np.float32)
    pred = np.zeros((len(x_a), 2), dtype=float)
    pred[:, 0] = pack["models"]["y1"].predict(x_n)
    pred[:, 1] = pack["models"]["y2"].predict(x_n)
    return enforce_y2_nonnegative(pred)


def condition_seed(base_seed: int, condition: str,
                   ordered_conditions: list[str]) -> int:
    return base_seed + ordered_conditions.index(condition)


def load_condition_data(split: str, condition: str, scalers: dict, *,
                        seed: int) -> dict:
    return load_sensor_raw_csv_level_data(
        split,
        scalers,
        params=SENSOR_CONDITIONS[condition],
        seed=seed,
    )


def build_route_level_predictions(
    *,
    scenario: str,
    split: str,
    condition: str,
    data: dict,
    scalers: dict,
    summary: dict,
    stable_packs: dict[str, dict],
    cache_prefix: str,
    force_predictions: bool,
) -> dict[str, np.ndarray]:
    """Return predictions from frozen TD-HER and robustness-oriented experts."""
    cache_condition = f"robust_router/{cache_prefix}/{condition}"
    expert_preds = predict_condition(
        scenario,
        cache_condition,
        data,
        scalers,
        force=force_predictions,
    )
    preds = {
        "tdher_main": apply_tdher_route(summary[scenario], expert_preds),
    }
    for alias, pack in stable_packs.items():
        preds[alias] = predict_stable_pack(pack, data["X_A"])

    pred_dir = OUT_DIR / "predictions" / cache_prefix / condition
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, pred in preds.items():
        np.save(pred_dir / f"{split}_{name}_preds.npy", pred)
    np.save(pred_dir / f"{split}_y.npy", data["y"])
    return preds


def fit_route(y_cal: np.ndarray, cal_preds: dict[str, np.ndarray]) -> dict:
    models = list(cal_preds)
    details = {
        "method": "target-wise convex route + restricted affine calibration",
        "candidate_models": models,
        "targets": {},
    }
    for target_idx, target_name in enumerate(["y1", "y2"]):
        p_cal = np.stack(
            [cal_preds[model][:, target_idx] for model in models],
            axis=1,
        )
        weights, blend_mae = optimize_convex_weights(
            y_cal[:, target_idx],
            p_cal,
        )
        cal_blend = p_cal @ weights
        slope, bias, affine_info = fit_restricted_affine(
            y_cal[:, target_idx],
            cal_blend,
        )
        details["targets"][target_name] = {
            "convex_cal_MAE": float(blend_mae),
            "affine_cal_MAE": float(affine_info["cal_MAE"]),
            "slope": float(slope),
            "bias": float(bias),
            "weights": {
                model: float(weight)
                for model, weight in zip(models, weights)
            },
            "candidate_cal_MAE": {
                model: float(np.mean(
                    np.abs(y_cal[:, target_idx] - cal_preds[model][:, target_idx])
                ))
                for model in models
            },
        }
    return details


def apply_route(route: dict, preds: dict[str, np.ndarray]) -> np.ndarray:
    models = route["candidate_models"]
    n = len(next(iter(preds.values())))
    out = np.zeros((n, 2), dtype=float)
    for target_idx, target_name in enumerate(["y1", "y2"]):
        target_route = route["targets"][target_name]
        routed = np.zeros(n, dtype=float)
        for model in models:
            routed += target_route["weights"][model] * preds[model][:, target_idx]
        out[:, target_idx] = apply_affine(
            routed,
            target_route["slope"],
            target_route["bias"],
        )
    return enforce_y2_nonnegative(out)


def flatten_metrics(metrics: dict) -> dict:
    return {
        "y1_mae": metrics["y1_MAE"],
        "y1_rmse": metrics["y1_RMSE"],
        "y2_mae": metrics["y2_MAE"],
        "y2_rmse": metrics["y2_RMSE"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "split",
        "condition",
        "method",
        "calibration_conditions",
        "channel_noise",
        "gap_stream_frac",
        "delay_stream_frac",
        "y1_mae",
        "y1_rmse",
        "y2_mae",
        "y2_rmse",
        "clean_y1_mae",
        "clean_y2_mae",
        "ratio_y1_mae",
        "ratio_y2_mae",
        "delta_y1_vs_tdher",
        "delta_y2_vs_tdher",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_reference_columns(rows: list[dict]) -> list[dict]:
    clean = {}
    tdher = {}
    for row in rows:
        if row["condition"] == "clean_rebuild":
            clean[row["method"]] = row
        if row["method"] == "tdher_main":
            tdher[row["condition"]] = row

    out = []
    for row in rows:
        r = dict(row)
        clean_row = clean.get(row["method"])
        tdher_row = tdher.get(row["condition"])
        if clean_row:
            r["clean_y1_mae"] = clean_row["y1_mae"]
            r["clean_y2_mae"] = clean_row["y2_mae"]
            r["ratio_y1_mae"] = (
                row["y1_mae"] / clean_row["y1_mae"]
                if clean_row["y1_mae"] else ""
            )
            r["ratio_y2_mae"] = (
                row["y2_mae"] / clean_row["y2_mae"]
                if clean_row["y2_mae"] else ""
            )
        else:
            r["clean_y1_mae"] = ""
            r["clean_y2_mae"] = ""
            r["ratio_y1_mae"] = ""
            r["ratio_y2_mae"] = ""

        if tdher_row:
            r["delta_y1_vs_tdher"] = row["y1_mae"] - tdher_row["y1_mae"]
            r["delta_y2_vs_tdher"] = row["y2_mae"] - tdher_row["y2_mae"]
        else:
            r["delta_y1_vs_tdher"] = ""
            r["delta_y2_vs_tdher"] = ""
        out.append(r)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=SCENARIO, choices=[SCENARIO])
    parser.add_argument(
        "--cal-conditions",
        nargs="+",
        default=DEFAULT_CAL_CONDITIONS,
        help="Validation conditions used to fit the robust route.",
    )
    parser.add_argument(
        "--test-conditions",
        nargs="+",
        default=DEFAULT_TEST_CONDITIONS,
        help="Held-out test conditions used for evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-predictions",
        action="store_true",
        help="Recompute cached base-expert predictions for perturbed data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(
        (set(args.cal_conditions) | set(args.test_conditions))
        - set(SENSOR_CONDITIONS)
    )
    if unknown:
        raise ValueError(f"Unknown sensor conditions: {unknown}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    scalers = attach_raw_feature_stats(load_train_scalers())
    summary = load_summary()
    stable_packs = {
        alias: load_stable_pack(alias, run_name)
        for alias, run_name in STABLE_RUNS.items()
    }

    # Fit the route using only validation data.
    cal_y_blocks = []
    cal_pred_blocks: dict[str, list[np.ndarray]] = {}
    for condition in args.cal_conditions:
        seed = condition_seed(args.seed, condition, args.cal_conditions)
        logger.info(f"Building calibration condition={condition}, seed={seed}")
        data = load_condition_data("val", condition, scalers, seed=seed)
        preds = build_route_level_predictions(
            scenario=args.scenario,
            split="val",
            condition=condition,
            data=data,
            scalers=scalers,
            summary=summary,
            stable_packs=stable_packs,
            cache_prefix=f"cal_seed{args.seed}",
            force_predictions=args.force_predictions,
        )
        cal_y_blocks.append(data["y"])
        for name, pred in preds.items():
            cal_pred_blocks.setdefault(name, []).append(pred)

    y_cal = np.concatenate(cal_y_blocks, axis=0)
    cal_preds = {
        name: np.concatenate(blocks, axis=0)
        for name, blocks in cal_pred_blocks.items()
    }
    route = fit_route(y_cal, cal_preds)
    route.update({
        "scenario": args.scenario,
        "calibration_conditions": args.cal_conditions,
        "projection": "y2 = max(y2, 0)",
        "main_tdher_source_method": METHOD,
        "stable_runs": STABLE_RUNS,
    })

    rows = []
    cal_label = "+".join(args.cal_conditions)
    for condition in args.test_conditions:
        seed = condition_seed(args.seed, condition, args.test_conditions)
        logger.info(f"Evaluating test condition={condition}, seed={seed}")
        data = load_condition_data("test", condition, scalers, seed=seed)
        preds = build_route_level_predictions(
            scenario=args.scenario,
            split="test",
            condition=condition,
            data=data,
            scalers=scalers,
            summary=summary,
            stable_packs=stable_packs,
            cache_prefix=f"test_seed{args.seed}",
            force_predictions=args.force_predictions,
        )
        preds["robust_route"] = apply_route(route, preds)
        np.save(
            OUT_DIR / "predictions" / f"test_seed{args.seed}" / condition
            / "test_robust_route_preds.npy",
            preds["robust_route"],
        )

        params = SENSOR_CONDITIONS[condition]
        for method, pred in preds.items():
            metrics = flatten_metrics(evaluate(data["y"], pred))
            rows.append({
                "scenario": args.scenario,
                "split": "test",
                "condition": condition,
                "method": method,
                "calibration_conditions": cal_label if method == "robust_route" else "",
                "channel_noise": json.dumps(
                    params.get("channel_noise", {}), sort_keys=True),
                "gap_stream_frac": params.get("gap_stream_frac", ""),
                "delay_stream_frac": params.get("delay_stream_frac", ""),
                **metrics,
            })
            logger.info(
                f"{condition} {method}: "
                f"y1={metrics['y1_mae']:.6f}, y2={metrics['y2_mae']:.6f}"
            )

    rows = add_reference_columns(rows)
    write_csv(OUT_DIR / "robust_router_summary.csv", rows)
    write_csv(PAPER_TABLE_DIR / "tdher_robust_router.csv", rows)
    with (OUT_DIR / "robust_router_summary.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (OUT_DIR / "robust_route_details.json").open("w") as f:
        json.dump(route, f, indent=2)
    logger.info(f"Wrote robust-router outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
