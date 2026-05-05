#!/usr/bin/env python3
"""Stable-feature LightGBM robustness probe for TD-HER.

This script tests whether the severe y2 brittleness observed under mild
raw-sensor perturbations is caused by short-window high-order/statistical
features. It trains a LightGBM expert on the audited Exp1 train/validation
split, but uses a filtered mRMR feature set that removes features expected to be
unstable under measurement noise, one-step gaps, or timestamp slips.

This is a diagnostic prototype. It does not overwrite Exp1 or TD-HER artifacts.
"""

import argparse
import csv
import json
import pickle
import sys
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# This diagnostic only reads cached mRMR lists; it never calls mRMR selection.
if "mrmr" not in sys.modules:
    mrmr_stub = types.ModuleType("mrmr")

    def _missing_mrmr_regression(*args, **kwargs):
        raise ImportError("mrmr is not required for stable robustness inference")

    mrmr_stub.mrmr_regression = _missing_mrmr_regression
    sys.modules["mrmr"] = mrmr_stub

from experiments.exp4_generalization import (  # noqa: E402
    EXP1_DIR,
    MS,
    REP_DIR,
    evaluate,
    load_train_scalers,
)
from experiments.exp_tdh_router import enforce_y2_nonnegative  # noqa: E402
from experiments.exp_tdh_robustness import (  # noqa: E402
    SENSOR_CONDITIONS,
    attach_raw_feature_stats,
    load_sensor_raw_csv_level_data,
)

OUT_DIR = Path("results/ieee39/tdher_robustness/stable_lgbm")
CHANNEL_PREFIXES = ("ANGL_", "FREQ_", "POWR_", "SPD_", "VOLT_")
RAW_STEP_TOKENS = {str(i) for i in range(MS + 1)}

POLICIES = {
    "stable_core": {
        "drop_suffixes": {
            "argmin",
            "argmax",
            "skew",
            "kurt",
            "zcr",
            "cv",
            "last_first_ratio",
            "d2_absmean",
            "d2_absmax",
        },
        "drop_contains": {"_corr"},
    },
    "basic_stats": {
        "drop_suffixes": {
            "argmin",
            "argmax",
            "skew",
            "kurt",
            "zcr",
            "cv",
            "last_first_ratio",
            "d2_absmean",
            "d2_absmax",
        },
        "drop_contains": {"_corr"},
        "drop_raw_samples": True,
    },
    "raw_trace": {
        "keep_only_raw_samples": True,
    },
}


def _is_raw_temporal_sample(name: str) -> bool:
    if not name.startswith(CHANNEL_PREFIXES):
        return False
    parts = name.split("_")
    return len(parts) == 3 and parts[2] in RAW_STEP_TOKENS


def _feature_suffix(name: str) -> str:
    parts = name.split("_")
    if len(parts) <= 2:
        return ""
    return "_".join(parts[2:])


def keep_feature(name: str, policy: dict) -> bool:
    """Return whether a feature is kept by the stability policy."""
    is_channel = name.startswith(CHANNEL_PREFIXES)
    is_raw = _is_raw_temporal_sample(name)

    if policy.get("keep_only_raw_samples"):
        return (not is_channel) or is_raw

    if policy.get("drop_raw_samples") and is_raw:
        return False

    for token in policy.get("drop_contains", set()):
        if token in name:
            return False

    suffix = _feature_suffix(name)
    if suffix in policy.get("drop_suffixes", set()):
        return False

    return True


def load_mrmr_union() -> list[str]:
    with open(Path(EXP1_DIR) / "mrmr_features.json") as f:
        return json.load(f)["union"]


def select_features(feature_names: list[str], policy_name: str) -> list[str]:
    policy = POLICIES[policy_name]
    available = set(feature_names)
    selected = [
        name for name in load_mrmr_union()
        if name in available and keep_feature(name, policy)
    ]
    if not selected:
        raise ValueError(f"No features selected for policy={policy_name}")
    return selected


def load_split_arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
    rep_a_dir = Path(REP_DIR) / "repA" / f"ms{MS}"
    X = np.load(rep_a_dir / f"X_{split}.npy")
    y = np.load(rep_a_dir / f"y_{split}.npy")
    return X, y


def load_exp1_lgb_params() -> dict:
    """Reuse audited Exp1 LightGBM hyperparameters without rerunning Optuna."""
    params = {}
    for target in ["y1", "y2"]:
        with open(Path(EXP1_DIR) / "LightGBM" / f"{target}_lgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        p = model.get_params()
        best_iter = int(getattr(model, "best_iteration_", 0) or 0)
        if best_iter > 0:
            p["n_estimators"] = best_iter
        p.update({
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "n_jobs": 8,
            "random_state": 42,
        })
        params[target] = p
    return params


def _run_name(policy_name: str, augment_conditions: list[str] | None) -> str:
    if not augment_conditions:
        return policy_name
    return f"{policy_name}__aug_{'+'.join(augment_conditions)}"


def train_stable_lgbm(policy_name: str, *, force: bool = False,
                      augment_conditions: list[str] | None = None,
                      scalers: dict | None = None,
                      seed: int = 42) -> dict:
    run_name = _run_name(policy_name, augment_conditions)
    policy_dir = OUT_DIR / run_name
    model_dir = policy_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    meta_path = policy_dir / "feature_policy.json"

    with open(Path(REP_DIR) / "repA" / f"ms{MS}" / "feature_names.json") as f:
        feature_names = json.load(f)
    selected = select_features(feature_names, policy_name)
    feat_idx = [feature_names.index(name) for name in selected]

    X_train, y_train = load_split_arrays("train")
    X_val, y_val = load_split_arrays("val")
    scaler = StandardScaler().fit(X_train[:, feat_idx])
    X_train_n = scaler.transform(X_train[:, feat_idx]).astype(np.float32)
    X_val_n = scaler.transform(X_val[:, feat_idx]).astype(np.float32)
    train_blocks = [X_train_n]
    target_blocks = [y_train]

    if augment_conditions:
        if scalers is None:
            raise ValueError("scalers are required for raw-CSV train augmentation")
        cache_dir = policy_dir / "aug_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for aug_idx, condition in enumerate(augment_conditions):
            if condition == "clean_rebuild":
                continue
            params = SENSOR_CONDITIONS[condition]
            x_cache = cache_dir / f"train_{condition}_seed{seed + aug_idx}_X.npy"
            y_cache = cache_dir / f"train_{condition}_seed{seed + aug_idx}_y.npy"
            if x_cache.exists() and y_cache.exists() and not force:
                X_aug = np.load(x_cache)
                y_aug = np.load(y_cache)
            else:
                logger.info(
                    f"Building train augmentation condition={condition}, "
                    f"policy={policy_name}"
                )
                aug_data = load_sensor_raw_csv_level_data(
                    "train",
                    scalers,
                    params=params,
                    seed=seed + aug_idx,
                )
                X_aug = scaler.transform(
                    aug_data["X_A"][:, feat_idx]
                ).astype(np.float32)
                y_aug = aug_data["y"].astype(np.float32)
                np.save(x_cache, X_aug)
                np.save(y_cache, y_aug)
            train_blocks.append(X_aug)
            target_blocks.append(y_aug)

    if len(train_blocks) > 1:
        X_fit = np.concatenate(train_blocks, axis=0)
        y_fit = np.concatenate(target_blocks, axis=0)
    else:
        X_fit = X_train_n
        y_fit = y_train

    models = {}
    preds_val = np.zeros_like(y_val, dtype=float)
    params = load_exp1_lgb_params()

    for target_idx, target in enumerate(["y1", "y2"]):
        model_path = model_dir / f"{target}_stable_lgbm.pkl"
        if model_path.exists() and not force:
            with model_path.open("rb") as f:
                model = pickle.load(f)
        else:
            logger.info(
                f"Training stable LightGBM target={target}, "
                f"policy={run_name}, n_features={len(feat_idx)}, "
                f"n_train={len(X_fit)}"
            )
            model = lgb.LGBMRegressor(**params[target])
            model.fit(
                X_fit,
                y_fit[:, target_idx],
                eval_set=[(X_val_n, y_val[:, target_idx])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            with model_path.open("wb") as f:
                pickle.dump(model, f)
        models[target] = model
        preds_val[:, target_idx] = model.predict(X_val_n)

    meta = {
        "run_name": run_name,
        "policy": policy_name,
        "augment_conditions": augment_conditions or [],
        "feature_count": len(selected),
        "feature_names": selected,
        "drop_policy": {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in POLICIES[policy_name].items()
        },
        "validation_metrics": evaluate(y_val, enforce_y2_nonnegative(preds_val)),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "run_name": run_name,
        "models": models,
        "scaler": scaler,
        "feat_idx": feat_idx,
        "feature_names": selected,
        "meta": meta,
    }


def predict_stable(model_pack: dict, X_A: np.ndarray) -> np.ndarray:
    X_n = model_pack["scaler"].transform(
        X_A[:, model_pack["feat_idx"]]
    ).astype(np.float32)
    pred = np.zeros((len(X_A), 2), dtype=float)
    pred[:, 0] = model_pack["models"]["y1"].predict(X_n)
    pred[:, 1] = model_pack["models"]["y2"].predict(X_n)
    return enforce_y2_nonnegative(pred)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy",
        "feature_count",
        "condition",
        "channel_noise",
        "gap_stream_frac",
        "delay_stream_frac",
        "y1_mae",
        "y2_mae",
        "clean_y1_mae",
        "clean_y2_mae",
        "ratio_y1_mae",
        "ratio_y2_mae",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["stable_core", "basic_stats", "raw_trace"],
        choices=sorted(POLICIES),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "clean_rebuild",
            "freq_noise_low",
            "volt_noise_low",
            "angl_noise_low",
            "powr_noise_low",
            "spd_noise_low",
            "sensor_noise_low",
            "one_step_gap_1pct",
            "timestamp_lag_1pct",
        ],
        help="Subset of sensor_raw_csv robustness conditions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--augment-conditions",
        nargs="+",
        default=None,
        help="Optional raw-CSV train augmentation conditions.",
    )
    args = parser.parse_args()

    unknown = sorted(
        (set(args.conditions) | set(args.augment_conditions or []))
        - set(SENSOR_CONDITIONS)
    )
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")

    scalers = attach_raw_feature_stats(load_train_scalers())
    rows = []
    run_names = []

    for policy_name in args.policies:
        pack = train_stable_lgbm(
            policy_name,
            force=args.force,
            augment_conditions=args.augment_conditions,
            scalers=scalers,
            seed=args.seed,
        )
        run_names.append(pack["run_name"])
        clean_metrics = None
        for idx, condition in enumerate(args.conditions):
            params = SENSOR_CONDITIONS[condition]
            logger.info(
                f"Evaluating stable LightGBM policy={policy_name}, "
                f"condition={condition}"
            )
            data = load_sensor_raw_csv_level_data(
                "test",
                scalers,
                params=params,
                seed=args.seed + idx,
            )
            pred = predict_stable(pack, data["X_A"])
            metrics = evaluate(data["y"], pred)

            pred_dir = OUT_DIR / pack["run_name"] / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            np.save(pred_dir / f"{condition}_preds.npy", pred)

            if condition == "clean_rebuild":
                clean_metrics = metrics
            if clean_metrics is None:
                raise ValueError("clean_rebuild must be evaluated first")

            rows.append({
                "policy": pack["run_name"],
                "feature_count": pack["meta"]["feature_count"],
                "condition": condition,
                "channel_noise": json.dumps(
                    params.get("channel_noise", {}), sort_keys=True),
                "gap_stream_frac": params.get("gap_stream_frac", ""),
                "delay_stream_frac": params.get("delay_stream_frac", ""),
                "y1_mae": metrics["y1_MAE"],
                "y2_mae": metrics["y2_MAE"],
                "clean_y1_mae": clean_metrics["y1_MAE"],
                "clean_y2_mae": clean_metrics["y2_MAE"],
                "ratio_y1_mae": (
                    metrics["y1_MAE"] / clean_metrics["y1_MAE"]
                    if clean_metrics["y1_MAE"] else ""
                ),
                "ratio_y2_mae": (
                    metrics["y2_MAE"] / clean_metrics["y2_MAE"]
                    if clean_metrics["y2_MAE"] else ""
                ),
            })
            logger.info(
                f"{policy_name} {condition}: "
                f"y1={metrics['y1_MAE']:.6f} "
                f"({rows[-1]['ratio_y1_mae']:.2f}x), "
                f"y2={metrics['y2_MAE']:.6f} "
                f"({rows[-1]['ratio_y2_mae']:.2f}x)"
            )

    summary_stem = (
        f"stable_lgbm_robustness_summary__{run_names[0]}"
        if len(run_names) == 1 else
        "stable_lgbm_robustness_summary__combined"
    )
    write_rows(OUT_DIR / "stable_lgbm_robustness_summary.csv", rows)
    write_rows(OUT_DIR / f"{summary_stem}.csv", rows)
    with (OUT_DIR / "stable_lgbm_robustness_summary.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (OUT_DIR / f"{summary_stem}.json").open("w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Wrote stable LightGBM robustness summary to {OUT_DIR}")


if __name__ == "__main__":
    main()
