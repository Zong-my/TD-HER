#!/usr/bin/env python3
"""Representation-level robustness stress tests for TD-HER.

This script does not retrain any expert. It perturbs the test input
representations, recomputes expert predictions, and applies the already learned
TD-HER route from ``experiments/exp_tdh_router.py``.

The perturbations are intentionally described as representation-level stress
tests. They are useful for screening measurement-noise and missing-feature
sensitivity, but they do not replace a future raw PMU re-extraction experiment.
"""

import argparse
import csv
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The imported generalization utilities only need cached mRMR feature lists in
# this script. Avoid requiring the optional ``mrmr`` package at import time.
if "mrmr" not in sys.modules:
    mrmr_stub = types.ModuleType("mrmr")

    def _missing_mrmr_regression(*args, **kwargs):
        raise ImportError("mrmr is not required for robustness inference")

    mrmr_stub.mrmr_regression = _missing_mrmr_regression
    sys.modules["mrmr"] = mrmr_stub

from experiments.exp4_generalization import (  # noqa: E402
    MS,
    REP_DIR,
    evaluate,
    load_level_data,
    load_train_scalers,
)
from experiments.exp_tdh_router import (  # noqa: E402
    ADAPTER_NAMES,
    ROUTER_MODELS,
    SCENARIOS,
    enforce_y2_nonnegative,
    predict_expert,
)
from data_proc.build_representations import (  # noqa: E402
    build_rep_a,
    build_rep_b,
    compute_statistical_features,
    parse_temporal_columns,
)

ROUTER_DIR = Path("results/ieee39/exp_tdh_router")
OUT_DIR = Path("results/ieee39/tdher_robustness")
CSV_DIR = Path("data/ieee39_v8_80_10_10/csv")
METHOD = "admission_affine_convex_blend_nonnegative_y2"
CONDITIONS = {
    "noise_0001": {"noise_std": 0.001, "missing_frac": 0.0},
    "noise_0005": {"noise_std": 0.005, "missing_frac": 0.0},
    "noise_001": {"noise_std": 0.01, "missing_frac": 0.0},
    "noise_005": {"noise_std": 0.05, "missing_frac": 0.0},
    "missing_0005": {"noise_std": 0.0, "missing_frac": 0.005},
    "missing_001": {"noise_std": 0.0, "missing_frac": 0.01},
    "missing_005": {"noise_std": 0.0, "missing_frac": 0.05},
    "noise_005_missing_001": {"noise_std": 0.05, "missing_frac": 0.01},
}

# Channel-aware raw-CSV perturbations. These are intentionally much milder than
# the legacy normalized stress suite above and keep temporal/statistical
# features coupled by rebuilding RepA/RepB/RepC after perturbation.
SENSOR_CONDITIONS = {
    "clean_rebuild": {},
    "sensor_noise_low": {
        "channel_noise": {
            "FREQ": 1e-5,
            "VOLT": 1e-4,
            "ANGL": 0.01,
            "POWR": 0.002,
            "SPD": 1e-5,
        },
    },
    "freq_noise_low": {"channel_noise": {"FREQ": 1e-5}},
    "volt_noise_low": {"channel_noise": {"VOLT": 1e-4}},
    "angl_noise_low": {"channel_noise": {"ANGL": 0.01}},
    "powr_noise_low": {"channel_noise": {"POWR": 0.002}},
    "spd_noise_low": {"channel_noise": {"SPD": 1e-5}},
    "sensor_noise_mid": {
        "channel_noise": {
            "FREQ": 5e-5,
            "VOLT": 5e-4,
            "ANGL": 0.05,
            "POWR": 0.01,
            "SPD": 5e-5,
        },
    },
    "sensor_noise_high": {
        "channel_noise": {
            "FREQ": 1e-4,
            "VOLT": 1e-3,
            "ANGL": 0.10,
            "POWR": 0.02,
            "SPD": 1e-4,
        },
    },
    "one_step_gap_1pct": {
        "gap_stream_frac": 0.01,
        "gap_len": 1,
    },
    "one_step_gap_5pct": {
        "gap_stream_frac": 0.05,
        "gap_len": 1,
    },
    "timestamp_lag_1pct": {
        "delay_stream_frac": 0.01,
        "delay_steps": 1,
    },
    "sensor_noise_mid_gap_1pct": {
        "channel_noise": {
            "FREQ": 5e-5,
            "VOLT": 5e-4,
            "ANGL": 0.05,
            "POWR": 0.01,
            "SPD": 5e-5,
        },
        "gap_stream_frac": 0.01,
        "gap_len": 1,
    },
}


def attach_raw_feature_stats(scalers: dict) -> dict:
    X_train_A = np.load(f"{REP_DIR}/repA/ms{MS}/X_train.npy")
    mean = np.mean(X_train_A, axis=0)
    std = np.std(X_train_A, axis=0)
    std[std < 1e-12] = 1.0
    scalers["X_A_train_mean"] = mean
    scalers["X_A_train_std"] = std
    return scalers


def perturb_data(data: dict, scalers: dict, *, noise_std: float,
                 missing_frac: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {
        "X_A": np.array(data["X_A"], copy=True),
        "X_n": np.array(data["X_n"], copy=True),
        "y": data["y"],
        "Xt_n": np.array(data["Xt_n"], copy=True),
        "Xs_n": np.array(data["Xs_n"], copy=True),
        "Xn_n": np.array(data["Xn_n"], copy=True),
    }

    if noise_std > 0:
        out["X_A"] += rng.normal(
            loc=0.0,
            scale=noise_std * scalers["X_A_train_std"],
            size=out["X_A"].shape,
        )
        out["Xt_n"] += rng.normal(0.0, noise_std, size=out["Xt_n"].shape)
        out["Xs_n"] += rng.normal(0.0, noise_std, size=out["Xs_n"].shape)

    if missing_frac > 0:
        mask_A = rng.random(out["X_A"].shape) < missing_frac
        out["X_A"][mask_A] = np.broadcast_to(
            scalers["X_A_train_mean"], out["X_A"].shape
        )[mask_A]

        mask_t = rng.random(out["Xt_n"].shape) < missing_frac
        out["Xt_n"][mask_t] = 0.0

        mask_s = rng.random(out["Xs_n"].shape) < missing_frac
        out["Xs_n"][mask_s] = 0.0

    out["X_n"] = scalers["scaler_A"].transform(
        out["X_A"][:, scalers["feat_idx"]]
    ).astype(np.float32)
    out["Xt_n"] = out["Xt_n"].astype(np.float32)
    out["Xs_n"] = out["Xs_n"].astype(np.float32)
    out["Xn_n"] = np.transpose(out["Xt_n"], (0, 2, 1, 3)).astype(np.float32)
    return out


def temporal_columns(df: pd.DataFrame) -> list[str]:
    skip = {
        "distu_kind", "file_name",
        "load_level", "load_zip_z", "load_zip_i", "load_zip_p",
        "reserve_ratio", "h_inertia", "load_delta",
        "fpu_deltamax", "t_delta",
    }
    return [c for c in df.columns if c not in skip]


def load_raw_csv_level_data(split_name: str, scalers: dict, *, noise_std: float,
                            missing_frac: float, seed: int) -> dict:
    """Perturb raw ms10 CSV columns and rebuild representations.

    ``noise_std`` is interpreted as a ratio of each raw temporal column's
    empirical standard deviation in the current split, not as normalized feature
    units. This keeps temporal/statistical features physically coupled because
    RepA/RepB/RepC are rebuilt after perturbation.
    """
    csv_path = CSV_DIR / f"{split_name}_ms{MS}.csv"
    logger.info(f"Loading raw CSV for robustness: {csv_path}")
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    tcols = temporal_columns(df)

    if noise_std > 0:
        col_std = df[tcols].std(axis=0).to_numpy(dtype=np.float32)
        col_std[col_std < 1e-12] = 1.0
        noise = rng.normal(
            loc=0.0,
            scale=noise_std * col_std.reshape(1, -1),
            size=(len(df), len(tcols)),
        )
        df.loc[:, tcols] = df[tcols].to_numpy(dtype=np.float32) + noise

    if missing_frac > 0:
        values = df[tcols].to_numpy(dtype=np.float32)
        mask = rng.random(values.shape) < missing_frac
        col_mean = np.nanmean(values, axis=0)
        values[mask] = np.broadcast_to(col_mean, values.shape)[mask]
        df.loc[:, tcols] = values

    X_a, y_a, fnames = build_rep_a(df, MS)
    X_t, X_s, y_b, meta = build_rep_b(df, MS)
    if not np.allclose(y_a, y_b):
        raise ValueError("RepA/RepB targets are not aligned after raw CSV rebuild")

    X_stats, stat_names, _ = compute_statistical_features(
        X_t, meta["sheets"], meta["buses"])
    X_a_full = np.concatenate([X_a, X_stats], axis=1).astype(np.float32)
    X_s_extended = np.concatenate([X_s, X_stats], axis=1).astype(np.float32)
    fnames_full = fnames + stat_names
    if fnames_full != scalers["feature_names"]:
        raise ValueError("rebuilt feature names do not match training features")

    X_n = scalers["scaler_A"].transform(
        X_a_full[:, scalers["feat_idx"]]
    ).astype(np.float32)
    B, T, N, C = X_t.shape
    Xt_n = scalers["sc_t"].transform(
        X_t.reshape(-1, N * C)
    ).reshape(B, T, N, C).astype(np.float32)
    Xs_n = scalers["sc_s"].transform(
        X_s_extended[:, scalers["static_idx"]]
    ).astype(np.float32)
    Xn_n = np.transpose(Xt_n, (0, 2, 1, 3)).astype(np.float32)

    return {
        "X_A": X_a_full,
        "X_n": X_n,
        "y": y_a,
        "Xt_n": Xt_n,
        "Xs_n": Xs_n,
        "Xn_n": Xn_n,
    }


def rebuild_representations_from_df(df: pd.DataFrame, scalers: dict) -> dict:
    """Rebuild normalized TD-HER representations from a perturbed raw CSV."""
    X_a, y_a, fnames = build_rep_a(df, MS)
    X_t, X_s, y_b, meta = build_rep_b(df, MS)
    if not np.allclose(y_a, y_b):
        raise ValueError("RepA/RepB targets are not aligned after rebuild")

    X_stats, stat_names, _ = compute_statistical_features(
        X_t, meta["sheets"], meta["buses"])
    X_a_full = np.concatenate([X_a, X_stats], axis=1).astype(np.float32)
    X_s_extended = np.concatenate([X_s, X_stats], axis=1).astype(np.float32)
    fnames_full = fnames + stat_names
    if fnames_full != scalers["feature_names"]:
        raise ValueError("rebuilt feature names do not match training features")

    X_n = scalers["scaler_A"].transform(
        X_a_full[:, scalers["feat_idx"]]
    ).astype(np.float32)
    B, T, N, C = X_t.shape
    Xt_n = scalers["sc_t"].transform(
        X_t.reshape(-1, N * C)
    ).reshape(B, T, N, C).astype(np.float32)
    Xs_n = scalers["sc_s"].transform(
        X_s_extended[:, scalers["static_idx"]]
    ).astype(np.float32)
    Xn_n = np.transpose(Xt_n, (0, 2, 1, 3)).astype(np.float32)

    return {
        "X_A": X_a_full,
        "X_n": X_n,
        "y": y_a,
        "Xt_n": Xt_n,
        "Xs_n": Xs_n,
        "Xn_n": Xn_n,
    }


def _series_columns(structure: dict, sheet: str, bus: int,
                    timesteps: list[int]) -> list[str]:
    return [
        structure["col_map"][(sheet, bus, t)]
        for t in timesteps
        if (sheet, bus, t) in structure["col_map"]
    ]


def _fill_gap(series: np.ndarray, start: int, length: int) -> None:
    end = min(start + length, len(series))
    if start == 0 and end < len(series):
        series[start:end] = series[end]
    elif start > 0 and end < len(series):
        before = series[start - 1]
        after = series[end]
        steps = end - start + 1
        interp = np.linspace(before, after, steps + 1, dtype=series.dtype)[1:-1]
        series[start:end] = interp
    elif start > 0:
        series[start:end] = series[start - 1]


def apply_sensor_perturbation(df: pd.DataFrame, params: dict,
                              seed: int) -> pd.DataFrame:
    """Apply channel-level sensor perturbations to raw temporal columns.

    The perturbation is stream-aware: each stream is one
    (sample, sheet, generator-bus) trajectory across the early window.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    structure = parse_temporal_columns(out.columns.tolist())
    sheets = structure["sheets"]
    buses = structure["buses"]
    timesteps = [t for t in structure["timesteps"] if t <= MS]

    for sheet, sigma in params.get("channel_noise", {}).items():
        if sheet not in sheets or sigma <= 0:
            continue
        cols = [
            col for (s, _, t), col in structure["col_map"].items()
            if s == sheet and t <= MS
        ]
        if cols:
            out.loc[:, cols] = (
                out[cols].to_numpy(dtype=np.float32)
                + rng.normal(0.0, sigma, size=(len(out), len(cols)))
            )

    gap_frac = float(params.get("gap_stream_frac", 0.0) or 0.0)
    gap_len = int(params.get("gap_len", 1) or 1)
    delay_frac = float(params.get("delay_stream_frac", 0.0) or 0.0)
    delay_steps = int(params.get("delay_steps", 1) or 1)

    if gap_frac <= 0 and delay_frac <= 0:
        return out

    n_rows = len(out)
    for sheet in sheets:
        for bus in buses:
            cols = _series_columns(structure, sheet, bus, timesteps)
            if len(cols) <= 1:
                continue
            values = out[cols].to_numpy(dtype=np.float32)

            if delay_frac > 0:
                delayed = rng.random(n_rows) < delay_frac
                if delayed.any():
                    delayed_values = values[delayed].copy()
                    delayed_values[:, delay_steps:] = delayed_values[:, :-delay_steps]
                    values[delayed] = delayed_values

            if gap_frac > 0:
                gapped_rows = np.flatnonzero(rng.random(n_rows) < gap_frac)
                for row_idx in gapped_rows:
                    start = int(rng.integers(0, len(cols)))
                    _fill_gap(values[row_idx], start, gap_len)

            out.loc[:, cols] = values

    return out


def load_sensor_raw_csv_level_data(split_name: str, scalers: dict,
                                   params: dict, seed: int) -> dict:
    csv_path = CSV_DIR / f"{split_name}_ms{MS}.csv"
    logger.info(f"Loading sensor raw CSV for robustness: {csv_path}")
    df = pd.read_csv(csv_path)
    perturbed = apply_sensor_perturbation(df, params, seed)
    return rebuild_representations_from_df(perturbed, scalers)


def predict_adapter(scenario: str, data: dict) -> np.ndarray:
    adapter_dir = ROUTER_DIR / "adapters" / scenario / "LightGBM-Adapter"
    preds = np.zeros((len(data["y"]), 2), dtype=float)
    for target_idx, target in enumerate(["y1", "y2"]):
        with open(adapter_dir / f"{target}_final_model.pkl", "rb") as f:
            model = pickle.load(f)
        preds[:, target_idx] = model.predict(data["X_n"])
    return preds


def load_summary() -> dict:
    with (ROUTER_DIR / "metrics_summary.json").open() as f:
        return json.load(f)


def apply_tdher_route(scenario_summary: dict, expert_preds: dict) -> np.ndarray:
    details = scenario_summary[METHOD]["details"]["source_details"]
    n = len(next(iter(expert_preds.values())))
    pred = np.zeros((n, 2), dtype=float)
    for target_idx, target in enumerate(["y1", "y2"]):
        target_details = details[target]
        routed = np.zeros(n, dtype=float)
        for expert, weight in target_details["weights"].items():
            routed += float(weight) * expert_preds[expert][:, target_idx]
        pred[:, target_idx] = (
            float(target_details["slope"]) * routed
            + float(target_details["bias"])
        )
    return enforce_y2_nonnegative(pred)


def predict_condition(scenario: str, condition: str, data: dict,
                      scalers: dict, *, force: bool) -> dict:
    pred_dir = OUT_DIR / "predictions" / scenario / condition
    pred_dir.mkdir(parents=True, exist_ok=True)
    expert_preds = {}

    for model_name in ROUTER_MODELS:
        path = pred_dir / f"{model_name}_preds.npy"
        if path.exists() and not force:
            pred = np.load(path)
        else:
            logger.info(f"{scenario} {condition}: predicting {model_name}")
            pred = predict_expert(model_name, data, scalers)
            np.save(path, pred)
        expert_preds[model_name] = pred

    adapter_path = pred_dir / "LightGBM-Adapter_preds.npy"
    if adapter_path.exists() and not force:
        adapter_pred = np.load(adapter_path)
    else:
        logger.info(f"{scenario} {condition}: predicting LightGBM-Adapter")
        adapter_pred = predict_adapter(scenario, data)
        np.save(adapter_path, adapter_pred)
    expert_preds["LightGBM-Adapter"] = adapter_pred

    return expert_preds


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["L1_same_dist"],
        choices=list(SCENARIOS),
        help="Scenarios to stress-test. Default keeps the first run lightweight.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["representation", "raw_csv", "sensor_raw_csv"],
        default="raw_csv",
        help="raw_csv rebuilds RepA/RepB/RepC after raw temporal perturbation.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Optional subset of robustness conditions for the selected mode.",
    )
    args = parser.parse_args()

    out_dir = OUT_DIR / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    scalers = attach_raw_feature_stats(load_train_scalers())
    summary = load_summary()
    rows = []

    for scenario in args.scenarios:
        cal_split, test_split = SCENARIOS[scenario]
        del cal_split
        clean_metrics = summary[scenario][METHOD]["metrics"]
        clean_y1 = clean_metrics["y1_MAE"]
        clean_y2 = clean_metrics["y2_MAE"]
        base_data = load_level_data(test_split, scalers) if args.mode == "representation" else None

        suite = SENSOR_CONDITIONS if args.mode == "sensor_raw_csv" else CONDITIONS
        if args.conditions:
            unknown = sorted(set(args.conditions) - set(suite))
            if unknown:
                raise ValueError(f"Unknown conditions for {args.mode}: {unknown}")
            condition_items = [(c, suite[c]) for c in args.conditions]
        else:
            condition_items = list(suite.items())

        for idx, (condition, params) in enumerate(condition_items):
            logger.info(f"{scenario}: condition={condition}, params={params}")
            if args.mode == "sensor_raw_csv":
                perturbed = load_sensor_raw_csv_level_data(
                    test_split,
                    scalers,
                    params=params,
                    seed=args.seed + idx,
                )
            elif args.mode == "raw_csv":
                perturbed = load_raw_csv_level_data(
                    test_split,
                    scalers,
                    noise_std=params["noise_std"],
                    missing_frac=params["missing_frac"],
                    seed=args.seed + idx,
                )
            else:
                perturbed = perturb_data(
                    base_data,
                    scalers,
                    noise_std=params["noise_std"],
                    missing_frac=params["missing_frac"],
                    seed=args.seed + idx,
                )
            expert_preds = predict_condition(
                scenario, f"{args.mode}/{condition}", perturbed, scalers,
                force=args.force)
            final_pred = apply_tdher_route(summary[scenario], expert_preds)
            final_metrics = evaluate(perturbed["y"], final_pred)
            np.save(
                OUT_DIR / "predictions" / scenario / args.mode / condition / f"{METHOD}_preds.npy",
                final_pred,
            )

            rows.append({
                "scenario": scenario,
                "test_split": test_split,
                "mode": args.mode,
                "condition": condition,
                "noise_std_normalized": params.get("noise_std", ""),
                "missing_frac_mean_imputed": params.get("missing_frac", ""),
                "channel_noise": json.dumps(params.get("channel_noise", {}), sort_keys=True),
                "gap_stream_frac": params.get("gap_stream_frac", ""),
                "delay_stream_frac": params.get("delay_stream_frac", ""),
                "clean_y1_mae": clean_y1,
                "condition_y1_mae": final_metrics["y1_MAE"],
                "delta_y1_mae": final_metrics["y1_MAE"] - clean_y1,
                "ratio_y1_mae": final_metrics["y1_MAE"] / clean_y1 if clean_y1 else "",
                "clean_y2_mae": clean_y2,
                "condition_y2_mae": final_metrics["y2_MAE"],
                "delta_y2_mae": final_metrics["y2_MAE"] - clean_y2,
                "ratio_y2_mae": final_metrics["y2_MAE"] / clean_y2 if clean_y2 else "",
            })
            logger.info(
                f"{scenario} {condition}: y1={final_metrics['y1_MAE']:.6f} "
                f"({final_metrics['y1_MAE'] / clean_y1:.2f}x), "
                f"y2={final_metrics['y2_MAE']:.6f} "
                f"({final_metrics['y2_MAE'] / clean_y2:.2f}x)"
            )

    fieldnames = [
        "scenario",
        "test_split",
        "mode",
        "condition",
        "noise_std_normalized",
        "missing_frac_mean_imputed",
        "channel_noise",
        "gap_stream_frac",
        "delay_stream_frac",
        "clean_y1_mae",
        "condition_y1_mae",
        "delta_y1_mae",
        "ratio_y1_mae",
        "clean_y2_mae",
        "condition_y2_mae",
        "delta_y2_mae",
        "ratio_y2_mae",
    ]
    write_csv(out_dir / "robustness_summary.csv", fieldnames, rows)
    with (out_dir / "robustness_summary.json").open("w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Wrote robustness summary to {out_dir}")


if __name__ == "__main__":
    main()
