#!/usr/bin/env python3
"""Audit IEEE300 TD-HER routing outputs before manuscript use."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


TARGETS = ("y1", "y2")
ROUTER_METHODS = [
    "best_expert",
    "uniform_blend",
    "convex_blend",
    "affine_convex_blend",
    "affine_convex_blend_nonnegative_y2",
]


def load_json(path: Path) -> dict:
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


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for idx, target in enumerate(TARGETS):
        err = y_pred[:, idx] - y_true[:, idx]
        out[f"{target}_MAE"] = float(np.mean(np.abs(err)))
        out[f"{target}_RMSE"] = float(np.sqrt(np.mean(err ** 2)))
    return out


def target_report(y: np.ndarray) -> dict:
    return {
        "shape": list(y.shape),
        "finite": bool(np.isfinite(y).all()),
        "min": [float(np.nanmin(y[:, i])) for i in range(y.shape[1])],
        "max": [float(np.nanmax(y[:, i])) for i in range(y.shape[1])],
        "mean": [float(np.nanmean(y[:, i])) for i in range(y.shape[1])],
    }


def y2_constraint_report(y_true: np.ndarray, pred: np.ndarray) -> dict:
    y2_true = y_true[:, 1]
    y2_pred = pred[:, 1]
    neg = y2_pred < -1e-9
    clipped = np.maximum(y2_pred, 0.0)
    return {
        "constraint": "t_delta >= 0",
        "negative_count": int(np.sum(neg)),
        "negative_fraction": float(np.mean(neg)),
        "raw_y2_mae": float(np.mean(np.abs(y2_true - y2_pred))),
        "clipped0_y2_mae": float(np.mean(np.abs(y2_true - clipped))),
        "pred_negative_min": float(np.min(y2_pred[neg])) if np.any(neg) else None,
    }


def read_router_csv(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def audit_router(rep_dir: Path, router_dir: Path, paper_table_dir: Path,
                 ms: int) -> dict:
    errors = []
    warnings = []
    y_path = rep_dir / "repA" / f"ms{ms}" / "y_test.npy"
    summary_path = router_dir / "metrics_summary.json"
    csv_path = paper_table_dir / "ieee300_tdher_routing.csv"

    report = {
        "rep_dir": str(rep_dir),
        "router_dir": str(router_dir),
        "paper_table_dir": str(paper_table_dir),
        "ms": ms,
        "errors": errors,
        "warnings": warnings,
        "methods": [],
    }
    if not y_path.exists():
        errors.append(f"missing labels: {y_path}")
        return report
    if not summary_path.exists():
        errors.append(f"missing summary: {summary_path}")
        return report

    y_test = np.load(y_path)[:, :2]
    report["label_report"] = target_report(y_test)
    if not np.isfinite(y_test).all():
        errors.append("test labels contain non-finite values")
    if np.nanmin(y_test[:, 1]) < -1e-9:
        errors.append("test labels contain negative y2")

    summary = load_json(summary_path)
    csv_rows = read_router_csv(csv_path)
    if not csv_rows:
        warnings.append(f"missing paper table CSV: {csv_path}")

    for method in ROUTER_METHODS:
        pred_path = router_dir / f"{method}_preds.npy"
        method_report = {
            "method": method,
            "pred_file": str(pred_path),
            "errors": [],
            "warnings": [],
            "metrics_match_summary": {},
            "metrics_match_csv": {},
        }
        if not pred_path.exists():
            method_report["errors"].append("missing prediction file")
            report["methods"].append(method_report)
            continue

        pred = np.load(pred_path)
        if pred.shape != y_test.shape:
            method_report["errors"].append(
                f"prediction shape {pred.shape} differs from labels {y_test.shape}"
            )
            report["methods"].append(method_report)
            continue
        if not np.isfinite(pred).all():
            method_report["errors"].append("prediction contains non-finite values")

        constraint = y2_constraint_report(y_test, pred)
        method_report["target_constraint_report"] = constraint
        if constraint["negative_count"] > 0 and not method.endswith("nonnegative_y2"):
            method_report["warnings"].append("raw prediction contains negative y2")
        if constraint["negative_count"] > 0 and method.endswith("nonnegative_y2"):
            method_report["errors"].append("nonnegative-y2 variant contains negative y2")

        recomputed = evaluate(y_test, pred)
        method_report["recomputed_metrics"] = recomputed
        stored = summary.get(method, {}).get("metrics")
        if stored is None:
            method_report["errors"].append("method missing from metrics_summary.json")
        else:
            for key, value in recomputed.items():
                ok = compare_float(value, stored.get(key))
                method_report["metrics_match_summary"][key] = ok
                if not ok:
                    method_report["errors"].append(
                        f"{key} summary mismatch: stored={stored.get(key)}, recomputed={value}"
                    )

        row = csv_rows.get(method)
        if row:
            csv_metric_map = {
                "y1_MAE": "y1_mae",
                "y1_RMSE": "y1_rmse",
                "y2_MAE": "y2_mae",
                "y2_RMSE": "y2_rmse",
            }
            for metric_key, csv_key in csv_metric_map.items():
                ok = compare_float(recomputed[metric_key], row.get(csv_key))
                method_report["metrics_match_csv"][metric_key] = ok
                if not ok:
                    method_report["errors"].append(
                        f"{metric_key} CSV mismatch: stored={row.get(csv_key)}, "
                        f"recomputed={recomputed[metric_key]}"
                    )

        report["methods"].append(method_report)

    for method_report in report["methods"]:
        errors.extend(
            f"{method_report['method']}: {msg}"
            for msg in method_report["errors"]
        )
        warnings.extend(
            f"{method_report['method']}: {msg}"
            for msg in method_report["warnings"]
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-dir", default="data/ieee300_v2_posttrigger")
    parser.add_argument("--router-dir", default="results/ieee300/tdher_router")
    parser.add_argument("--paper-table-dir", default="results/paper_tables")
    parser.add_argument("--ms", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    router_dir = Path(args.router_dir)
    out_path = Path(args.out) if args.out else router_dir / "audit_report.json"
    audit = audit_router(
        Path(args.rep_dir),
        router_dir,
        Path(args.paper_table_dir),
        args.ms,
    )
    audit["status"] = "PASS" if not audit["errors"] else "FAIL"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(audit, f, indent=2)

    print(f"IEEE300 TD-HER router audit status: {audit['status']}")
    print(f"report: {out_path}")
    print(f"errors: {len(audit['errors'])}")
    print(f"warnings: {len(audit['warnings'])}")
    for msg in audit["errors"][:20]:
        print(f"ERROR: {msg}")
    if len(audit["errors"]) > 20:
        print(f"... {len(audit['errors']) - 20} more errors")
    for msg in audit["warnings"][:20]:
        print(f"WARNING: {msg}")
    if len(audit["warnings"]) > 20:
        print(f"... {len(audit['warnings']) - 20} more warnings")

    if args.fail_on_error and audit["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
