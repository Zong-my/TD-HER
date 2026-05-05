#!/usr/bin/env python3
"""Audit representation files before expensive training runs.

This script checks split-level consistency across RepA/RepB/RepC, target
validity, feature metadata, and optional CSV-to-representation target equality.
It is intended to run before IEEE300 rebuilds and before any result is promoted
into manuscript evidence.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


STATIC_COLS = [
    "load_level", "load_zip_z", "load_zip_i", "load_zip_p",
    "reserve_ratio", "h_inertia", "load_delta",
]
TARGET_COLS = ["fpu_deltamax", "t_delta"]
NODE_META_KEYS = ["generator_ids", "generators", "node_ids", "buses"]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def array_summary(arr: np.ndarray, column_limit: int = 8) -> dict:
    summary = {
        "shape": list(arr.shape),
        "finite": bool(np.isfinite(arr).all()),
    }
    if arr.ndim == 2 and arr.shape[1] > 0:
        summary["global_min"] = float(np.nanmin(arr))
        summary["global_max"] = float(np.nanmax(arr))
        summary["global_mean"] = float(np.nanmean(arr))
        n_cols = arr.shape[1]
        stat_cols = n_cols if n_cols <= column_limit else column_limit
        suffix = "" if n_cols <= column_limit else f"_first{column_limit}"
        summary[f"min{suffix}"] = [float(np.nanmin(arr[:, i])) for i in range(stat_cols)]
        summary[f"max{suffix}"] = [float(np.nanmax(arr[:, i])) for i in range(stat_cols)]
        summary[f"mean{suffix}"] = [float(np.nanmean(arr[:, i])) for i in range(stat_cols)]
    return summary


def load_if_exists(path: Path, errors: list):
    if not path.exists():
        errors.append(f"missing file: {path}")
        return None
    return np.load(path)


def audit_split(rep_dir: Path, csv_dir: Path | None, ms: int, split: str) -> dict:
    errors = []
    warnings = []
    report = {"split": split, "errors": errors, "warnings": warnings}

    rep_a_dir = rep_dir / "repA" / f"ms{ms}"
    rep_b_dir = rep_dir / "repB" / f"ms{ms}"
    rep_c_dir = rep_dir / "repC" / f"ms{ms}"

    X_a = load_if_exists(rep_a_dir / f"X_{split}.npy", errors)
    y_a = load_if_exists(rep_a_dir / f"y_{split}.npy", errors)
    X_t = load_if_exists(rep_b_dir / f"X_temporal_{split}.npy", errors)
    X_s_b = load_if_exists(rep_b_dir / f"X_static_{split}.npy", errors)
    y_b = load_if_exists(rep_b_dir / f"y_{split}.npy", errors)
    X_n = load_if_exists(rep_c_dir / f"X_node_{split}.npy", errors)
    X_s_c = load_if_exists(rep_c_dir / f"X_static_{split}.npy", errors)
    y_c = load_if_exists(rep_c_dir / f"y_{split}.npy", errors)

    arrays = {
        "X_repA": X_a,
        "y_repA": y_a,
        "X_temporal_repB": X_t,
        "X_static_repB": X_s_b,
        "y_repB": y_b,
        "X_node_repC": X_n,
        "X_static_repC": X_s_c,
        "y_repC": y_c,
    }
    report["arrays"] = {
        name: array_summary(arr) for name, arr in arrays.items() if arr is not None
    }

    if y_a is not None:
        if y_a.ndim != 2 or y_a.shape[1] < 2:
            errors.append(f"RepA {split} targets must be 2D with 2 columns, got {y_a.shape}")
        elif np.nanmin(y_a[:, 1]) < -1e-9:
            neg_count = int(np.sum(y_a[:, 1] < -1e-9))
            errors.append(
                f"RepA {split} t_delta has {neg_count} negative values "
                f"(min={float(np.nanmin(y_a[:, 1]))})"
            )
        if not np.isfinite(y_a).all():
            errors.append(f"RepA {split} targets contain non-finite values")

    for rep_name, y in [("RepB", y_b), ("RepC", y_c)]:
        if y_a is None or y is None:
            continue
        if y.shape != y_a.shape:
            errors.append(f"{rep_name} {split} y shape {y.shape} differs from RepA {y_a.shape}")
        elif not np.allclose(y, y_a):
            errors.append(f"{rep_name} {split} y values differ from RepA")

    if X_t is not None and X_n is not None:
        expected = (X_t.shape[0], X_t.shape[2], X_t.shape[1], X_t.shape[3])
        if X_n.shape != expected:
            errors.append(f"RepC {split} X_node shape {X_n.shape} should be {expected}")

    if X_s_b is not None and X_s_c is not None:
        if X_s_b.shape != X_s_c.shape:
            errors.append(f"RepB/RepC {split} static shape mismatch: {X_s_b.shape} vs {X_s_c.shape}")
        elif not np.allclose(X_s_b, X_s_c):
            errors.append(f"RepB/RepC {split} static values differ")

    feature_path = rep_a_dir / "feature_names.json"
    if feature_path.exists() and X_a is not None:
        feature_names = load_json(feature_path)
        report["repA_feature_count"] = len(feature_names)
        if len(feature_names) != X_a.shape[1]:
            errors.append(
                f"RepA feature count {len(feature_names)} differs from X columns {X_a.shape[1]}"
            )
    elif X_a is not None:
        errors.append(f"missing RepA feature_names.json: {feature_path}")

    meta_path = rep_b_dir / "meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        report["repB_meta_keys"] = sorted(meta.keys())
        if "static_names" not in meta:
            errors.append("RepB meta.json lacks static_names")
        elif X_s_b is not None and len(meta["static_names"]) != X_s_b.shape[1]:
            errors.append(
                f"RepB static_names count {len(meta['static_names'])} "
                f"differs from static columns {X_s_b.shape[1]}"
            )
        if "buses" in meta and X_t is not None and len(meta["buses"]) != X_t.shape[2]:
            errors.append(f"RepB buses count {len(meta['buses'])} differs from X nodes {X_t.shape[2]}")
        identity_key = next((key for key in NODE_META_KEYS if key in meta), None)
        if identity_key is None:
            errors.append("RepB meta.json lacks node/generator identity metadata")
        elif X_t is not None and len(meta[identity_key]) != X_t.shape[2]:
            errors.append(
                f"RepB {identity_key} count {len(meta[identity_key])} differs from X nodes {X_t.shape[2]}"
            )
        else:
            report["repB_node_identity_key"] = identity_key
        if "sheets" in meta and X_t is not None and len(meta["sheets"]) != X_t.shape[3]:
            errors.append(f"RepB sheets count {len(meta['sheets'])} differs from X channels {X_t.shape[3]}")
        if "timesteps" in meta and X_t is not None and len(meta["timesteps"]) != X_t.shape[1]:
            errors.append(
                f"RepB timesteps count {len(meta['timesteps'])} differs from X timesteps {X_t.shape[1]}"
            )
    else:
        errors.append(f"missing RepB meta.json: {meta_path}")

    if csv_dir is not None:
        csv_path = csv_dir / f"{split}_ms{ms}.csv"
        if csv_path.exists() and y_a is not None:
            df = pd.read_csv(csv_path, usecols=TARGET_COLS)
            y_csv = df[TARGET_COLS].to_numpy(dtype=np.float32)
            if y_csv.shape != y_a.shape:
                errors.append(f"CSV {split} target shape {y_csv.shape} differs from RepA {y_a.shape}")
            elif not np.allclose(y_csv, y_a):
                errors.append(f"CSV {split} targets differ from RepA y")
            report["csv_targets"] = array_summary(y_csv)
        else:
            warnings.append(f"CSV target check skipped for {split}: {csv_path}")

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-dir", required=True)
    parser.add_argument("--csv-dir", default=None)
    parser.add_argument("--ms", type=int, default=10)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    rep_dir = Path(args.rep_dir)
    csv_dir = Path(args.csv_dir) if args.csv_dir else None
    out_path = Path(args.out) if args.out else Path("results/dataset_audits") / (
        f"{rep_dir.name}_ms{args.ms}_audit.json"
    )

    split_reports = [audit_split(rep_dir, csv_dir, args.ms, split) for split in args.splits]
    errors = [
        f"{r['split']}: {msg}"
        for r in split_reports
        for msg in r["errors"]
    ]
    warnings = [
        f"{r['split']}: {msg}"
        for r in split_reports
        for msg in r["warnings"]
    ]
    status = "PASS" if not errors else "FAIL"
    audit = {
        "status": status,
        "rep_dir": str(rep_dir),
        "csv_dir": str(csv_dir) if csv_dir else None,
        "ms": args.ms,
        "splits": args.splits,
        "errors": errors,
        "warnings": warnings,
        "split_reports": split_reports,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(audit, f, indent=2)

    print(f"Dataset representation audit status: {status}")
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
