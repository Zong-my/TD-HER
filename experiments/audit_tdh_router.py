#!/usr/bin/env python3
"""Audit TD-HER router outputs before using them as manuscript evidence."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import mae_loss, mape_loss, rmse_loss, smape_loss


TARGET_NAMES = ("y1", "y2")
METRIC_FNS = {
    "MAE": mae_loss,
    "RMSE": rmse_loss,
    "MAPE": mape_loss,
    "SMAPE": smape_loss,
}
SCENARIO_TEST_SPLITS = {
    "L1_same_dist": "test",
    "L2_cross_cond_fewshot": "cross_cond_test",
    "L3_cross_topo_fewshot": "cross_cond_topo_test",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def compare_float(expected, actual, atol=1e-8, rtol=1e-6) -> bool:
    if expected is None or actual is None:
        return False
    if isinstance(expected, float) and math.isnan(expected):
        return isinstance(actual, float) and math.isnan(actual)
    if isinstance(actual, float) and math.isnan(actual):
        return False
    return abs(float(expected) - float(actual)) <= atol + rtol * abs(float(expected))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {}
    for i, target in enumerate(TARGET_NAMES):
        for suffix, fn in METRIC_FNS.items():
            metrics[f"{target}_{suffix}"] = float(fn(y_true[:, i], y_pred[:, i]))
    return metrics


def summarize_targets(y: np.ndarray) -> dict:
    return {
        "shape": list(y.shape),
        "finite": bool(np.isfinite(y).all()),
        "min": [float(np.nanmin(y[:, i])) for i in range(y.shape[1])],
        "max": [float(np.nanmax(y[:, i])) for i in range(y.shape[1])],
        "mean": [float(np.nanmean(y[:, i])) for i in range(y.shape[1])],
    }


def target_constraint_report(y_true: np.ndarray, pred: np.ndarray) -> dict:
    y2_true = y_true[:, 1]
    y2_pred = pred[:, 1]
    neg = y2_pred < -1e-9
    clipped = np.maximum(y2_pred, 0.0)
    report = {
        "target": "y2",
        "constraint": "t_delta >= 0",
        "negative_count": int(np.sum(neg)),
        "negative_fraction": float(np.mean(neg)),
        "raw_mae": float(mae_loss(y2_true, y2_pred)),
        "raw_rmse": float(rmse_loss(y2_true, y2_pred)),
        "clipped0_mae": float(mae_loss(y2_true, clipped)),
        "clipped0_rmse": float(rmse_loss(y2_true, clipped)),
    }
    if np.any(neg):
        report.update({
            "pred_negative_min": float(np.min(y2_pred[neg])),
            "pred_negative_mean": float(np.mean(y2_pred[neg])),
            "true_at_negative_min": float(np.min(y2_true[neg])),
            "true_at_negative_median": float(np.median(y2_true[neg])),
            "true_at_negative_max": float(np.max(y2_true[neg])),
        })
    return report


def audit_scenario(rep_dir: Path, router_dir: Path, ms: int,
                   scenario: str, split: str) -> dict:
    errors = []
    warnings = []
    scenario_dir = router_dir / "scenarios" / scenario
    summary_path = scenario_dir / "metrics_summary.json"
    label_path = rep_dir / "repA" / f"ms{ms}" / f"y_{split}.npy"
    report = {
        "scenario": scenario,
        "test_split": split,
        "errors": errors,
        "warnings": warnings,
        "models": [],
    }

    if not label_path.exists():
        errors.append(f"missing label file: {label_path}")
        return report
    if not summary_path.exists():
        errors.append(f"missing scenario metrics summary: {summary_path}")
        return report

    y_test = np.load(label_path)[:, :2]
    report["label_report"] = summarize_targets(y_test)
    if not np.isfinite(y_test).all():
        errors.append(f"{scenario}: labels contain non-finite values")
    if np.nanmin(y_test[:, 1]) < -1e-9:
        errors.append(f"{scenario}: ground-truth y2 contains negative values")

    summary = load_json(summary_path)
    pred_paths = sorted(scenario_dir.glob("*_preds.npy"))
    if not pred_paths:
        errors.append(f"{scenario}: no saved prediction files in {scenario_dir}")
        return report

    for pred_path in pred_paths:
        method = pred_path.stem.removesuffix("_preds")
        model_report = {
            "method": method,
            "pred_file": str(pred_path),
            "errors": [],
            "warnings": [],
            "metrics_match": {},
        }
        pred = np.load(pred_path)
        if pred.shape != y_test.shape:
            model_report["errors"].append(
                f"prediction shape {pred.shape} differs from labels {y_test.shape}"
            )
            report["models"].append(model_report)
            continue
        if not np.isfinite(pred).all():
            model_report["errors"].append("prediction contains non-finite values")

        constraint = target_constraint_report(y_test, pred)
        model_report["target_constraint_report"] = constraint
        if constraint["negative_count"] > 0:
            model_report["warnings"].append("predicted y2 contains negative values")

        recomputed = compute_metrics(y_test, pred)
        model_report["recomputed_metrics"] = recomputed
        stored = summary.get(method, {}).get("metrics")
        if stored is None:
            model_report["warnings"].append("method missing from scenario metrics summary")
        else:
            for key, value in recomputed.items():
                ok = compare_float(value, stored.get(key))
                model_report["metrics_match"][key] = ok
                if not ok:
                    model_report["errors"].append(
                        f"{key} mismatch: stored={stored.get(key)}, recomputed={value}"
                    )

        report["models"].append(model_report)

    for model_report in report["models"]:
        errors.extend(
            f"{model_report['method']}: {msg}"
            for msg in model_report["errors"]
        )
        warnings.extend(
            f"{model_report['method']}: {msg}"
            for msg in model_report["warnings"]
        )

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-dir", default="data/ieee39_v8_80_10_10")
    parser.add_argument("--router-dir", default="results/ieee39/exp_tdh_router")
    parser.add_argument("--ms", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    rep_dir = Path(args.rep_dir)
    router_dir = Path(args.router_dir)
    out_path = Path(args.out) if args.out else router_dir / "audit_report.json"

    scenario_reports = [
        audit_scenario(rep_dir, router_dir, args.ms, scenario, split)
        for scenario, split in SCENARIO_TEST_SPLITS.items()
    ]
    errors = [
        f"{r['scenario']}: {msg}"
        for r in scenario_reports
        for msg in r["errors"]
    ]
    warnings = [
        f"{r['scenario']}: {msg}"
        for r in scenario_reports
        for msg in r["warnings"]
    ]

    audit = {
        "status": "PASS" if not errors else "FAIL",
        "rep_dir": str(rep_dir),
        "router_dir": str(router_dir),
        "ms": args.ms,
        "errors": errors,
        "warnings": warnings,
        "scenarios": scenario_reports,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(audit, f, indent=2)

    print(f"TD-HER audit status: {audit['status']}")
    print(f"report: {out_path}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    for msg in errors[:20]:
        print(f"ERROR: {msg}")
    if len(errors) > 20:
        print(f"... {len(errors) - 20} more errors")
    for msg in warnings[:20]:
        print(f"WARNING: {msg}")
    if len(warnings) > 20:
        print(f"... {len(warnings) - 20} more warnings")

    if args.fail_on_error and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
