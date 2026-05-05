#!/usr/bin/env python3
"""Quality-aware robust routing probe for TD-HER.

The static robust route in ``exp_tdh_robust_router.py`` suppresses sensor-noise
sensitivity, especially for y2, but it degrades clean y1 and timestamp-lag
cases. This script tests a conservative quality-aware admission layer between
the frozen paper-facing TD-HER route and the previously fitted static robust
route.

The gate uses only early-window observable quality scores:
  - noise_score: channel roughness from second temporal differences, normalized
    against clean validation windows;
  - lag_score: non-frequency first-step duplication rate, which is sensitive to
    one-step timestamp slips.

This is a diagnostic redesign probe. It is not a paper-facing result unless the
held-out behavior is consistently defensible.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if "mrmr" not in sys.modules:
    mrmr_stub = types.ModuleType("mrmr")

    def _missing_mrmr_regression(*args, **kwargs):
        raise ImportError("mrmr is not required for quality-router inference")

    mrmr_stub.mrmr_regression = _missing_mrmr_regression
    sys.modules["mrmr"] = mrmr_stub

from data_proc.build_representations import parse_temporal_columns  # noqa: E402
from experiments.exp4_generalization import MS, evaluate, load_train_scalers  # noqa: E402
from experiments.exp_tdh_robust_router import (  # noqa: E402
    OUT_DIR as ROBUST_ROUTER_DIR,
    STABLE_RUNS,
    apply_route,
    build_route_level_predictions,
    load_condition_data,
    load_stable_pack,
)
from experiments.exp_tdh_robustness import (  # noqa: E402
    CSV_DIR,
    SENSOR_CONDITIONS,
    apply_sensor_perturbation,
    attach_raw_feature_stats,
    load_summary,
)
from experiments.exp_tdh_router import enforce_y2_nonnegative  # noqa: E402


SCENARIO = "L1_same_dist"
OUT_DIR = Path("results/ieee39/tdher_robustness/quality_router")
PAPER_TABLE_DIR = Path("results/paper_tables")
DEFAULT_GATE_CAL_CONDITIONS = [
    "clean_rebuild",
    "sensor_noise_low",
    "one_step_gap_1pct",
    "timestamp_lag_1pct",
]
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
SHEETS = ("FREQ", "VOLT", "ANGL", "POWR", "SPD")
NON_FREQ_SHEETS = ("VOLT", "ANGL", "POWR", "SPD")


@dataclass
class GateSpec:
    target: str
    gate_mode: str
    rule: str
    noise_threshold: float | None
    lag_threshold: float | None
    cal_mae: float


def condition_seed(base_seed: int, condition: str,
                   ordered_conditions: list[str]) -> int:
    return base_seed + ordered_conditions.index(condition)


def load_condition_df(split: str, condition: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / f"{split}_ms{MS}.csv")
    return apply_sensor_perturbation(df, SENSOR_CONDITIONS[condition], seed)


def _series_columns(structure: dict, sheet: str, bus: int,
                    timesteps: list[int]) -> list[str]:
    return [
        structure["col_map"][(sheet, bus, t)]
        for t in timesteps
        if (sheet, bus, t) in structure["col_map"]
    ]


def raw_quality_metrics(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute row-level roughness and timestamp-slip indicators."""
    structure = parse_temporal_columns(df.columns.tolist())
    timesteps = [t for t in structure["timesteps"] if t <= MS]
    metrics: dict[str, np.ndarray] = {}
    n_rows = len(df)

    for sheet in SHEETS:
        d2_blocks = []
        dup_blocks = []
        flat_blocks = []
        for bus in structure["buses"]:
            cols = _series_columns(structure, sheet, bus, timesteps)
            if len(cols) < 3:
                continue
            arr = df[cols].to_numpy(dtype=np.float64)
            d1 = np.diff(arr, axis=1)
            d2 = np.diff(arr, n=2, axis=1)
            scale = np.nanstd(arr, axis=1) + 1e-12
            d2_blocks.append(np.mean(np.abs(d2) / scale[:, None], axis=1))

            tol = 1e-12 + 1e-8 * np.maximum(np.abs(arr[:, 0]), 1.0)
            dup_blocks.append((np.abs(arr[:, 1] - arr[:, 0]) <= tol).astype(float))

            flat_tol = (
                1e-12
                + 1e-8 * np.maximum(np.max(np.abs(arr), axis=1), 1.0)
            )
            flat_blocks.append((np.min(np.abs(d1), axis=1) <= flat_tol).astype(float))

        if d2_blocks:
            metrics[f"{sheet}_d2_abs"] = np.mean(np.stack(d2_blocks, axis=1), axis=1)
            metrics[f"{sheet}_dup01"] = np.mean(np.stack(dup_blocks, axis=1), axis=1)
            metrics[f"{sheet}_flat_any"] = np.mean(np.stack(flat_blocks, axis=1), axis=1)
        else:
            metrics[f"{sheet}_d2_abs"] = np.zeros(n_rows, dtype=float)
            metrics[f"{sheet}_dup01"] = np.zeros(n_rows, dtype=float)
            metrics[f"{sheet}_flat_any"] = np.zeros(n_rows, dtype=float)
    return metrics


def fit_quality_baseline(clean_metrics: dict[str, np.ndarray]) -> dict:
    baseline = {}
    for sheet in SHEETS:
        key = f"{sheet}_d2_abs"
        values = clean_metrics[key]
        baseline[sheet] = {
            "d2_mean": float(np.mean(values)),
            "d2_std": float(max(np.std(values), 1e-9)),
        }
    return baseline


def quality_scores(metrics: dict[str, np.ndarray], baseline: dict) -> dict[str, np.ndarray]:
    z_blocks = []
    for sheet in SHEETS:
        raw = metrics[f"{sheet}_d2_abs"]
        stats = baseline[sheet]
        z_blocks.append((raw - stats["d2_mean"]) / stats["d2_std"])
    noise_score = np.maximum.reduce(z_blocks)

    lag_score = np.mean(
        np.stack([metrics[f"{sheet}_dup01"] for sheet in NON_FREQ_SHEETS], axis=1),
        axis=1,
    )
    gap_score = np.mean(
        np.stack([metrics[f"{sheet}_flat_any"] for sheet in NON_FREQ_SHEETS], axis=1),
        axis=1,
    )
    return {
        "noise_score": noise_score,
        "lag_score": lag_score,
        "gap_score": gap_score,
    }


def load_static_route() -> dict:
    path = ROBUST_ROUTER_DIR / "robust_route_details.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing static robust route: {path}. Run exp_tdh_robust_router.py first."
        )
    with path.open() as f:
        return json.load(f)


def condition_bundle(
    *,
    split: str,
    condition: str,
    seed: int,
    scalers: dict,
    summary: dict,
    stable_packs: dict,
    static_route: dict,
    baseline: dict,
    cache_prefix: str,
    force_predictions: bool,
) -> dict:
    logger.info(f"Building {split} bundle condition={condition}, seed={seed}")
    data = load_condition_data(split, condition, scalers, seed=seed)
    preds = build_route_level_predictions(
        scenario=SCENARIO,
        split=split,
        condition=condition,
        data=data,
        scalers=scalers,
        summary=summary,
        stable_packs=stable_packs,
        cache_prefix=cache_prefix,
        force_predictions=force_predictions,
    )
    preds["robust_route_static"] = apply_route(static_route, preds)
    preds["robust_route_static"] = enforce_y2_nonnegative(preds["robust_route_static"])

    df = load_condition_df(split, condition, seed)
    raw_metrics = raw_quality_metrics(df)
    q = quality_scores(raw_metrics, baseline)
    return {
        "condition": condition,
        "split": split,
        "y": data["y"],
        "preds": preds,
        "quality": q,
        "quality_mean": {k: float(np.mean(v)) for k, v in q.items()},
    }


def _threshold_candidates(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    qs = np.quantile(values, [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0])
    candidates = sorted(set(float(v) for v in qs if np.isfinite(v)))
    if candidates:
        candidates = [candidates[0] - 1e-9] + candidates + [candidates[-1] + 1e-9]
    else:
        candidates = [-1e-9, 1e-9]
    return candidates


def _mask_for_rule(rule: str, noise: np.ndarray, lag: np.ndarray,
                   noise_threshold: float | None,
                   lag_threshold: float | None) -> np.ndarray:
    if rule == "always_main":
        return np.zeros_like(noise, dtype=bool)
    if rule == "always_robust":
        return np.ones_like(noise, dtype=bool)
    if rule == "anti_lag":
        return lag <= float(lag_threshold)
    if rule == "noise_and_not_lag":
        return (noise >= float(noise_threshold)) & (lag <= float(lag_threshold))
    raise ValueError(f"Unknown gate rule: {rule}")


def _apply_mask(main_pred: np.ndarray, robust_pred: np.ndarray,
                mask: np.ndarray, target_idx: int) -> np.ndarray:
    out = np.array(main_pred[:, target_idx], copy=True)
    out[mask] = robust_pred[mask, target_idx]
    return out


def fit_gate_for_target(bundles: list[dict], target_idx: int,
                        gate_mode: str) -> GateSpec:
    target = ["y1", "y2"][target_idx]
    y = np.concatenate([b["y"][:, target_idx] for b in bundles])
    main = np.concatenate([b["preds"]["tdher_main"] for b in bundles], axis=0)
    robust = np.concatenate(
        [b["preds"]["robust_route_static"] for b in bundles], axis=0)

    if gate_mode == "sample":
        noise = np.concatenate([b["quality"]["noise_score"] for b in bundles])
        lag = np.concatenate([b["quality"]["lag_score"] for b in bundles])
        bundle_slices = None
    elif gate_mode == "batch":
        noise_parts = []
        lag_parts = []
        slices = []
        start = 0
        for b in bundles:
            n = len(b["y"])
            slices.append(slice(start, start + n))
            noise_parts.append(np.full(n, b["quality_mean"]["noise_score"]))
            lag_parts.append(np.full(n, b["quality_mean"]["lag_score"]))
            start += n
        noise = np.concatenate(noise_parts)
        lag = np.concatenate(lag_parts)
        bundle_slices = slices
        del bundle_slices
    else:
        raise ValueError(f"Unknown gate mode: {gate_mode}")

    best = None
    candidates = [
        ("always_main", None, None),
        ("always_robust", None, None),
    ]
    for lag_th in _threshold_candidates(lag):
        candidates.append(("anti_lag", None, lag_th))
    for noise_th in _threshold_candidates(noise):
        for lag_th in _threshold_candidates(lag):
            candidates.append(("noise_and_not_lag", noise_th, lag_th))

    for rule, noise_th, lag_th in candidates:
        mask = _mask_for_rule(rule, noise, lag, noise_th, lag_th)
        pred = _apply_mask(main, robust, mask, target_idx)
        mae = float(np.mean(np.abs(y - pred)))
        if best is None or mae < best.cal_mae:
            best = GateSpec(
                target=target,
                gate_mode=gate_mode,
                rule=rule,
                noise_threshold=None if noise_th is None else float(noise_th),
                lag_threshold=None if lag_th is None else float(lag_th),
                cal_mae=mae,
            )

    if best is None:
        raise RuntimeError("No gate candidate evaluated")
    return best


def apply_gate(bundle: dict, specs: dict[str, GateSpec]) -> np.ndarray:
    main = bundle["preds"]["tdher_main"]
    robust = bundle["preds"]["robust_route_static"]
    out = np.array(main, copy=True)
    for target_idx, target in enumerate(["y1", "y2"]):
        spec = specs[target]
        if spec.gate_mode == "sample":
            noise = bundle["quality"]["noise_score"]
            lag = bundle["quality"]["lag_score"]
        elif spec.gate_mode == "batch":
            n = len(bundle["y"])
            noise = np.full(n, bundle["quality_mean"]["noise_score"])
            lag = np.full(n, bundle["quality_mean"]["lag_score"])
        else:
            raise ValueError(f"Unknown gate mode: {spec.gate_mode}")
        mask = _mask_for_rule(
            spec.rule,
            noise,
            lag,
            spec.noise_threshold,
            spec.lag_threshold,
        )
        out[:, target_idx] = _apply_mask(main, robust, mask, target_idx)
    return enforce_y2_nonnegative(out)


def condition_oracle(bundle: dict) -> np.ndarray:
    """Diagnostic upper bound selecting main/static robust per target condition."""
    y = bundle["y"]
    main = bundle["preds"]["tdher_main"]
    robust = bundle["preds"]["robust_route_static"]
    out = np.zeros_like(main)
    for target_idx in range(2):
        main_mae = np.mean(np.abs(y[:, target_idx] - main[:, target_idx]))
        robust_mae = np.mean(np.abs(y[:, target_idx] - robust[:, target_idx]))
        out[:, target_idx] = (
            robust[:, target_idx] if robust_mae < main_mae else main[:, target_idx]
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
        "gate_cal_conditions",
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
        "noise_score_mean",
        "lag_score_mean",
        "gap_score_mean",
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
        c = clean.get(row["method"])
        t = tdher.get(row["condition"])
        if c:
            r["clean_y1_mae"] = c["y1_mae"]
            r["clean_y2_mae"] = c["y2_mae"]
            r["ratio_y1_mae"] = row["y1_mae"] / c["y1_mae"] if c["y1_mae"] else ""
            r["ratio_y2_mae"] = row["y2_mae"] / c["y2_mae"] if c["y2_mae"] else ""
        else:
            r["clean_y1_mae"] = ""
            r["clean_y2_mae"] = ""
            r["ratio_y1_mae"] = ""
            r["ratio_y2_mae"] = ""

        if t:
            r["delta_y1_vs_tdher"] = row["y1_mae"] - t["y1_mae"]
            r["delta_y2_vs_tdher"] = row["y2_mae"] - t["y2_mae"]
        else:
            r["delta_y1_vs_tdher"] = ""
            r["delta_y2_vs_tdher"] = ""
        out.append(r)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate-cal-conditions",
        nargs="+",
        default=DEFAULT_GATE_CAL_CONDITIONS,
    )
    parser.add_argument(
        "--test-conditions",
        nargs="+",
        default=DEFAULT_TEST_CONDITIONS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown = sorted(
        (set(args.gate_cal_conditions) | set(args.test_conditions))
        - set(SENSOR_CONDITIONS)
    )
    if unknown:
        raise ValueError(f"Unknown sensor conditions: {unknown}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    clean_val_df = load_condition_df("val", "clean_rebuild", args.seed)
    quality_baseline = fit_quality_baseline(raw_quality_metrics(clean_val_df))

    scalers = attach_raw_feature_stats(load_train_scalers())
    summary = load_summary()
    stable_packs = {
        alias: load_stable_pack(alias, run_name)
        for alias, run_name in STABLE_RUNS.items()
    }
    static_route = load_static_route()

    cal_bundles = []
    for condition in args.gate_cal_conditions:
        seed = condition_seed(args.seed, condition, args.gate_cal_conditions)
        cal_bundles.append(condition_bundle(
            split="val",
            condition=condition,
            seed=seed,
            scalers=scalers,
            summary=summary,
            stable_packs=stable_packs,
            static_route=static_route,
            baseline=quality_baseline,
            cache_prefix=f"quality_gate_cal_seed{args.seed}",
            force_predictions=args.force_predictions,
        ))

    gate_specs = {
        "sample": {
            "y1": fit_gate_for_target(cal_bundles, 0, "sample"),
            "y2": fit_gate_for_target(cal_bundles, 1, "sample"),
        },
        "batch": {
            "y1": fit_gate_for_target(cal_bundles, 0, "batch"),
            "y2": fit_gate_for_target(cal_bundles, 1, "batch"),
        },
    }

    rows = []
    gate_cal_label = "+".join(args.gate_cal_conditions)
    for condition in args.test_conditions:
        seed = condition_seed(args.seed, condition, args.test_conditions)
        bundle = condition_bundle(
            split="test",
            condition=condition,
            seed=seed,
            scalers=scalers,
            summary=summary,
            stable_packs=stable_packs,
            static_route=static_route,
            baseline=quality_baseline,
            cache_prefix=f"quality_gate_test_seed{args.seed}",
            force_predictions=args.force_predictions,
        )
        predictions = {
            "tdher_main": bundle["preds"]["tdher_main"],
            "robust_route_static": bundle["preds"]["robust_route_static"],
            "quality_gate_sample": apply_gate(bundle, gate_specs["sample"]),
            "quality_gate_batch": apply_gate(bundle, gate_specs["batch"]),
            "oracle_main_vs_robust": condition_oracle(bundle),
        }
        pred_dir = OUT_DIR / "predictions" / f"test_seed{args.seed}" / condition
        pred_dir.mkdir(parents=True, exist_ok=True)
        for method, pred in predictions.items():
            np.save(pred_dir / f"{method}_preds.npy", pred)
            metrics = flatten_metrics(evaluate(bundle["y"], pred))
            rows.append({
                "scenario": SCENARIO,
                "split": "test",
                "condition": condition,
                "method": method,
                "gate_cal_conditions": (
                    gate_cal_label if method.startswith("quality_gate") else ""
                ),
                **metrics,
                "noise_score_mean": bundle["quality_mean"]["noise_score"],
                "lag_score_mean": bundle["quality_mean"]["lag_score"],
                "gap_score_mean": bundle["quality_mean"]["gap_score"],
            })
            logger.info(
                f"{condition} {method}: "
                f"y1={metrics['y1_mae']:.6f}, y2={metrics['y2_mae']:.6f}"
            )

    rows = add_reference_columns(rows)
    write_csv(OUT_DIR / "quality_router_summary.csv", rows)
    write_csv(PAPER_TABLE_DIR / "tdher_quality_router.csv", rows)

    details = {
        "scenario": SCENARIO,
        "gate_cal_conditions": args.gate_cal_conditions,
        "quality_baseline": quality_baseline,
        "gate_specs": {
            mode: {
                target: spec.__dict__
                for target, spec in specs.items()
            }
            for mode, specs in gate_specs.items()
        },
        "static_route_source": str(ROBUST_ROUTER_DIR / "robust_route_details.json"),
    }
    with (OUT_DIR / "quality_router_summary.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (OUT_DIR / "quality_gate_details.json").open("w") as f:
        json.dump(details, f, indent=2)
    logger.info(f"Wrote quality-router outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
