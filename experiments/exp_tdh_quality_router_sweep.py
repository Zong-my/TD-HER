#!/usr/bin/env python3
"""Multi-seed and multi-intensity sweep for quality-aware TD-HER routing.

This script validates whether the quality-aware gate from
``exp_tdh_quality_router.py`` is stable beyond a single perturbation instance.
It reuses frozen trained experts, the static robust route, and the fitted
quality-gate rules. No model is retrained.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if "mrmr" not in sys.modules:
    mrmr_stub = types.ModuleType("mrmr")

    def _missing_mrmr_regression(*args, **kwargs):
        raise ImportError("mrmr is not required for quality-router sweep")

    mrmr_stub.mrmr_regression = _missing_mrmr_regression
    sys.modules["mrmr"] = mrmr_stub

from experiments.exp4_generalization import MS, evaluate, load_train_scalers  # noqa: E402
from experiments.exp_tdh_quality_router import (  # noqa: E402
    OUT_DIR as QUALITY_ROUTER_DIR,
    PAPER_TABLE_DIR,
    SCENARIO,
    GateSpec,
    apply_gate,
    condition_oracle,
    fit_quality_baseline,
    load_static_route,
    quality_scores,
    raw_quality_metrics,
)
from experiments.exp_tdh_robust_router import (  # noqa: E402
    STABLE_RUNS,
    apply_route,
    build_route_level_predictions,
    load_stable_pack,
)
from experiments.exp_tdh_robustness import (  # noqa: E402
    CSV_DIR,
    SENSOR_CONDITIONS,
    apply_sensor_perturbation,
    attach_raw_feature_stats,
    load_sensor_raw_csv_level_data,
    load_summary,
)
from experiments.exp_tdh_router import enforce_y2_nonnegative  # noqa: E402


OUT_DIR = Path("results/ieee39/tdher_robustness/quality_router_sweep")
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def _fmt_factor(value: float) -> str:
    return str(value).replace(".", "p")


def scale_channel_noise(params: dict, factor: float) -> dict:
    return {
        "channel_noise": {
            channel: float(sigma) * factor
            for channel, sigma in params["channel_noise"].items()
        }
    }


def build_condition_specs(include_single_channel: bool = False) -> list[dict]:
    specs = [
        {
            "condition": "clean_rebuild",
            "family": "clean",
            "intensity": 0.0,
            "params": {},
            "seeds_are_distinct": False,
        }
    ]

    for factor in [0.5, 1.0, 2.0]:
        specs.append({
            "condition": f"sensor_noise_x{_fmt_factor(factor)}",
            "family": "sensor_noise",
            "intensity": factor,
            "params": scale_channel_noise(SENSOR_CONDITIONS["sensor_noise_low"], factor),
            "seeds_are_distinct": True,
        })

    if include_single_channel:
        for base_name in [
            "freq_noise_low",
            "volt_noise_low",
            "angl_noise_low",
            "powr_noise_low",
            "spd_noise_low",
        ]:
            specs.append({
                "condition": base_name,
                "family": "single_channel_noise",
                "intensity": 1.0,
                "params": SENSOR_CONDITIONS[base_name],
                "seeds_are_distinct": True,
            })

    for frac in [0.005, 0.01, 0.02]:
        specs.append({
            "condition": f"one_step_gap_{_fmt_factor(frac * 100)}pct",
            "family": "one_step_gap",
            "intensity": frac,
            "params": {"gap_stream_frac": frac, "gap_len": 1},
            "seeds_are_distinct": True,
        })
    for frac in [0.005, 0.01, 0.02]:
        specs.append({
            "condition": f"timestamp_lag_{_fmt_factor(frac * 100)}pct",
            "family": "timestamp_lag",
            "intensity": frac,
            "params": {"delay_stream_frac": frac, "delay_steps": 1},
            "seeds_are_distinct": True,
        })
    return specs


def load_gate_specs() -> dict[str, dict[str, GateSpec]]:
    details_path = QUALITY_ROUTER_DIR / "quality_gate_details.json"
    if not details_path.exists():
        raise FileNotFoundError(
            f"Missing quality gate details: {details_path}. "
            "Run exp_tdh_quality_router.py first."
        )
    with details_path.open() as f:
        details = json.load(f)
    return {
        mode: {
            target: GateSpec(**spec)
            for target, spec in targets.items()
        }
        for mode, targets in details["gate_specs"].items()
    }


def load_condition_df(split: str, params: dict, seed: int) -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / f"{split}_ms{MS}.csv")
    return apply_sensor_perturbation(df, params, seed)


def condition_bundle(
    *,
    split: str,
    condition: str,
    params: dict,
    seed: int,
    scalers: dict,
    summary: dict,
    stable_packs: dict,
    static_route: dict,
    quality_baseline: dict,
    force_predictions: bool,
) -> dict:
    logger.info(f"Building sweep bundle condition={condition}, seed={seed}")
    data = load_sensor_raw_csv_level_data(split, scalers, params=params, seed=seed)
    preds = build_route_level_predictions(
        scenario=SCENARIO,
        split=split,
        condition=f"{condition}/seed{seed}",
        data=data,
        scalers=scalers,
        summary=summary,
        stable_packs=stable_packs,
        cache_prefix="quality_sweep_test",
        force_predictions=force_predictions,
    )
    preds["robust_route_static"] = enforce_y2_nonnegative(
        apply_route(static_route, preds)
    )
    df = load_condition_df(split, params, seed)
    q = quality_scores(raw_quality_metrics(df), quality_baseline)
    return {
        "condition": condition,
        "split": split,
        "seed": seed,
        "y": data["y"],
        "preds": preds,
        "quality": q,
        "quality_mean": {k: float(np.mean(v)) for k, v in q.items()},
    }


def flatten_metrics(metrics: dict) -> dict:
    return {
        "y1_mae": metrics["y1_MAE"],
        "y1_rmse": metrics["y1_RMSE"],
        "y2_mae": metrics["y2_MAE"],
        "y2_rmse": metrics["y2_RMSE"],
    }


def method_predictions(bundle: dict, gate_specs: dict) -> dict[str, np.ndarray]:
    return {
        "tdher_main": bundle["preds"]["tdher_main"],
        "robust_route_static": bundle["preds"]["robust_route_static"],
        "quality_gate_sample": apply_gate(bundle, gate_specs["sample"]),
        "quality_gate_batch": apply_gate(bundle, gate_specs["batch"]),
        "oracle_main_vs_robust": condition_oracle(bundle),
    }


def add_delta_columns(rows: list[dict]) -> list[dict]:
    ref = {
        (row["condition"], row["seed"]): row
        for row in rows
        if row["method"] == "tdher_main"
    }
    out = []
    for row in rows:
        r = dict(row)
        base = ref[(row["condition"], row["seed"])]
        r["delta_y1_vs_tdher"] = row["y1_mae"] - base["y1_mae"]
        r["delta_y2_vs_tdher"] = row["y2_mae"] - base["y2_mae"]
        out.append(r)
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    metric_cols = [
        "y1_mae",
        "y2_mae",
        "delta_y1_vs_tdher",
        "delta_y2_vs_tdher",
        "noise_score_mean",
        "lag_score_mean",
        "gap_score_mean",
    ]
    grouped = df.groupby(["condition", "family", "intensity", "method"], dropna=False)
    summary = grouped[metric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col) for col in summary.columns]
    counts = grouped.size().rename("n_runs")
    out = summary.join(counts).reset_index()
    return out.to_dict(orient="records")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--include-single-channel", action="store_true")
    parser.add_argument("--force-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    clean_val_df = load_condition_df("val", {}, args.seeds[0])
    quality_baseline = fit_quality_baseline(raw_quality_metrics(clean_val_df))

    scalers = attach_raw_feature_stats(load_train_scalers())
    summary = load_summary()
    static_route = load_static_route()
    stable_packs = {
        alias: load_stable_pack(alias, run_name)
        for alias, run_name in STABLE_RUNS.items()
    }
    gate_specs = load_gate_specs()

    rows = []
    for spec in build_condition_specs(args.include_single_channel):
        seeds = args.seeds if spec["seeds_are_distinct"] else [args.seeds[0]]
        for seed in seeds:
            bundle = condition_bundle(
                split="test",
                condition=spec["condition"],
                params=spec["params"],
                seed=seed,
                scalers=scalers,
                summary=summary,
                stable_packs=stable_packs,
                static_route=static_route,
                quality_baseline=quality_baseline,
                force_predictions=args.force_predictions,
            )
            predictions = method_predictions(bundle, gate_specs)
            pred_dir = OUT_DIR / "predictions" / spec["condition"] / f"seed{seed}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            for method, pred in predictions.items():
                np.save(pred_dir / f"{method}_preds.npy", pred)
                metrics = flatten_metrics(evaluate(bundle["y"], pred))
                row = {
                    "scenario": SCENARIO,
                    "split": "test",
                    "condition": spec["condition"],
                    "family": spec["family"],
                    "intensity": spec["intensity"],
                    "seed": seed,
                    "method": method,
                    **metrics,
                    "noise_score_mean": bundle["quality_mean"]["noise_score"],
                    "lag_score_mean": bundle["quality_mean"]["lag_score"],
                    "gap_score_mean": bundle["quality_mean"]["gap_score"],
                }
                rows.append(row)
                logger.info(
                    f"{spec['condition']} seed={seed} {method}: "
                    f"y1={metrics['y1_mae']:.6f}, y2={metrics['y2_mae']:.6f}"
                )

    rows = add_delta_columns(rows)
    summary_rows = summarize(rows)

    row_fields = [
        "scenario",
        "split",
        "condition",
        "family",
        "intensity",
        "seed",
        "method",
        "y1_mae",
        "y1_rmse",
        "y2_mae",
        "y2_rmse",
        "delta_y1_vs_tdher",
        "delta_y2_vs_tdher",
        "noise_score_mean",
        "lag_score_mean",
        "gap_score_mean",
    ]
    summary_fields = list(summary_rows[0]) if summary_rows else []
    write_csv(OUT_DIR / "quality_router_sweep_events.csv", rows, row_fields)
    write_csv(OUT_DIR / "quality_router_sweep_summary.csv",
              summary_rows, summary_fields)
    write_csv(PAPER_TABLE_DIR / "tdher_quality_router_sweep.csv",
              summary_rows, summary_fields)

    with (OUT_DIR / "quality_router_sweep_events.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (OUT_DIR / "quality_router_sweep_summary.json").open("w") as f:
        json.dump(summary_rows, f, indent=2)
    logger.info(f"Wrote quality-router sweep outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
