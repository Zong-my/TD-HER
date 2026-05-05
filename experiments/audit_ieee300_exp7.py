#!/usr/bin/env python3
"""Audit legacy IEEE300 Exp7 outputs before using them as paper evidence.

The script is intentionally read-only for existing experiment artifacts. It
recomputes metrics from ``preds.npy`` and the current representation labels,
checks representation metadata, and writes an audit report next to Exp7.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import mae_loss, mape_loss, rmse_loss, smape_loss


METRIC_FNS = {
    "MAE": mae_loss,
    "RMSE": rmse_loss,
    "MAPE": mape_loss,
    "SMAPE": smape_loss,
}
TARGET_NAMES = ("y1", "y2")
GENERATOR_META_KEYS = ("generator_ids", "generators", "node_ids", "buses")


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def compare_float(expected, actual, atol=1e-8, rtol=1e-6):
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


def summarize_array(values: np.ndarray) -> dict:
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "min": [float(np.nanmin(values[:, i])) for i in range(values.shape[1])],
        "max": [float(np.nanmax(values[:, i])) for i in range(values.shape[1])],
        "mean": [float(np.nanmean(values[:, i])) for i in range(values.shape[1])],
    }


def audit_labels(rep_dir: Path, ms: int) -> tuple[np.ndarray, list, list, dict]:
    errors = []
    warnings = []
    label_paths = {
        "repA": rep_dir / "repA" / f"ms{ms}" / "y_test.npy",
        "repB": rep_dir / "repB" / f"ms{ms}" / "y_test.npy",
        "repC": rep_dir / "repC" / f"ms{ms}" / "y_test.npy",
    }
    labels = {}
    for rep, path in label_paths.items():
        if not path.exists():
            errors.append(f"missing label file: {path}")
            continue
        labels[rep] = np.load(path)

    if "repA" not in labels:
        return np.empty((0, 2)), errors, warnings, {"labels": {}}

    y_test = labels["repA"]
    for rep, y in labels.items():
        if y.shape != y_test.shape:
            errors.append(f"{rep} y_test shape {y.shape} differs from repA {y_test.shape}")
        elif not np.allclose(y, y_test):
            errors.append(f"{rep} y_test values differ from repA")

    if y_test.ndim != 2 or y_test.shape[1] < 2:
        errors.append(f"repA y_test must be 2D with at least 2 targets, got {y_test.shape}")
    if not np.isfinite(y_test).all():
        errors.append("repA y_test contains non-finite values")
    if y_test.ndim == 2 and y_test.shape[1] >= 2 and np.nanmin(y_test[:, 1]) < -1e-9:
        errors.append("ground-truth y2 contains negative values")

    return y_test[:, :2], errors, warnings, {"labels": summarize_array(y_test[:, :2])}


def audit_metadata(rep_dir: Path, ms: int) -> tuple[list, dict]:
    warnings = []
    metadata = {}

    feature_path = rep_dir / "repA" / f"ms{ms}" / "feature_names.json"
    if feature_path.exists():
        names = load_json(feature_path)
        metadata["repA_feature_count"] = len(names)
    else:
        warnings.append(f"missing RepA feature_names.json: {feature_path}")

    meta_path = rep_dir / "repB" / f"ms{ms}" / "meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        metadata["repB_meta_keys"] = sorted(meta.keys())
        if "static_names" not in meta:
            warnings.append("RepB meta.json lacks static_names")
        identity_key = next((key for key in GENERATOR_META_KEYS if key in meta), None)
        if identity_key is None:
            warnings.append("RepB meta.json lacks generator identity metadata")
        else:
            metadata["repB_generator_identity_key"] = identity_key
            metadata["repB_generator_count"] = len(meta[identity_key])
    else:
        warnings.append(f"missing RepB meta.json: {meta_path}")

    return warnings, metadata


def target_constraint_report(y_true: np.ndarray, pred: np.ndarray) -> dict:
    """Summarize the physically nonnegative t_delta prediction constraint.

    Existing metrics remain audited against the raw saved predictions. The
    clipped metrics are reported only as a diagnostic for deciding whether a
    future model should enforce nonnegative arrival-time outputs explicitly.
    """
    y2_true = y_true[:, 1]
    y2_pred = pred[:, 1]
    neg_mask = y2_pred < -1e-9
    y2_clipped = np.maximum(y2_pred, 0.0)

    report = {
        "target": "y2",
        "constraint": "t_delta >= 0",
        "negative_count": int(np.sum(neg_mask)),
        "negative_fraction": float(np.mean(neg_mask)),
        "raw_mae": float(mae_loss(y2_true, y2_pred)),
        "raw_rmse": float(rmse_loss(y2_true, y2_pred)),
        "clipped0_mae": float(mae_loss(y2_true, y2_clipped)),
        "clipped0_rmse": float(rmse_loss(y2_true, y2_clipped)),
    }
    if np.any(neg_mask):
        report.update({
            "pred_negative_min": float(np.min(y2_pred[neg_mask])),
            "pred_negative_mean": float(np.mean(y2_pred[neg_mask])),
            "true_at_negative_min": float(np.min(y2_true[neg_mask])),
            "true_at_negative_median": float(np.median(y2_true[neg_mask])),
            "true_at_negative_max": float(np.max(y2_true[neg_mask])),
        })
    return report


def audit_model(model_dir: Path, y_test: np.ndarray) -> dict:
    report = {
        "model": model_dir.name,
        "errors": [],
        "warnings": [],
        "metrics_match": {},
    }
    pred_path = model_dir / "preds.npy"
    metrics_path = model_dir / "metrics.json"
    if not pred_path.exists():
        report["errors"].append("missing preds.npy")
        return report
    if not metrics_path.exists():
        report["errors"].append("missing metrics.json")
        return report

    pred = np.load(pred_path)
    report["pred_summary"] = summarize_array(pred[:, :2]) if pred.ndim == 2 and pred.shape[1] >= 2 else {
        "shape": list(pred.shape)
    }
    if pred.shape != y_test.shape:
        report["errors"].append(f"prediction shape {pred.shape} differs from y_test {y_test.shape}")
        return report
    if not np.isfinite(pred).all():
        report["errors"].append("preds.npy contains non-finite values")
    if np.nanmin(pred[:, 1]) < -1e-9:
        report["warnings"].append("predicted y2 contains negative values")
    report["target_constraint_report"] = target_constraint_report(y_test, pred)

    stored = load_json(metrics_path)
    recomputed = compute_metrics(y_test, pred)
    report["recomputed_metrics"] = recomputed

    for key, value in recomputed.items():
        stored_value = stored.get(key)
        ok = compare_float(value, stored_value)
        report["metrics_match"][key] = ok
        if not ok:
            report["errors"].append(
                f"{key} mismatch: stored={stored_value}, recomputed={value}"
            )

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-dir", default="data/ieee300_v2")
    parser.add_argument("--result-dir", default="results/ieee300/exp7")
    parser.add_argument("--ms", type=int, default=10)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    rep_dir = Path(args.rep_dir)
    result_dir = Path(args.result_dir)
    out_path = Path(args.out) if args.out else result_dir / "audit_report.json"

    y_test, label_errors, label_warnings, label_report = audit_labels(rep_dir, args.ms)
    meta_warnings, meta_report = audit_metadata(rep_dir, args.ms)

    model_reports = []
    if result_dir.exists() and y_test.size:
        for child in sorted(result_dir.iterdir()):
            if child.is_dir():
                model_reports.append(audit_model(child, y_test))
    elif not result_dir.exists():
        label_errors.append(f"missing result directory: {result_dir}")

    errors = list(label_errors)
    warnings = list(label_warnings) + list(meta_warnings)
    for report in model_reports:
        errors.extend(f"{report['model']}: {msg}" for msg in report["errors"])
        warnings.extend(f"{report['model']}: {msg}" for msg in report["warnings"])

    status = "PASS" if not errors else "FAIL"
    audit = {
        "status": status,
        "rep_dir": str(rep_dir),
        "result_dir": str(result_dir),
        "ms": args.ms,
        "errors": errors,
        "warnings": warnings,
        "label_report": label_report,
        "metadata_report": meta_report,
        "models": model_reports,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(audit, f, indent=2)

    print(f"IEEE300 Exp7 audit status: {status}")
    print(f"report: {out_path}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    if errors:
        for msg in errors[:20]:
            print(f"ERROR: {msg}")
        if len(errors) > 20:
            print(f"... {len(errors) - 20} more errors")
    if warnings:
        for msg in warnings[:20]:
            print(f"WARNING: {msg}")
        if len(warnings) > 20:
            print(f"... {len(warnings) - 20} more warnings")

    if args.fail_on_error and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
