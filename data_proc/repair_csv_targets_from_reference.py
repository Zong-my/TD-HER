#!/usr/bin/env python3
"""Repair target columns in a CSV from a trusted reference CSV.

The script is intended for small, auditable label repairs after the extraction
logic has been fixed. Rows are aligned by a stable key such as ``file_name``;
feature columns are left untouched.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--key", default="file_name")
    parser.add_argument("--columns", nargs="+", default=["fpu_deltamax", "t_delta"])
    parser.add_argument("--backup-suffix", default=".pre_posttrigger_repair.bak")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    target_path = Path(args.target_csv)
    reference_path = Path(args.reference_csv)
    out_report = Path(args.out_report) if args.out_report else target_path.with_suffix(
        target_path.suffix + ".repair_report.json"
    )

    target = pd.read_csv(target_path)
    reference = pd.read_csv(reference_path, usecols=[args.key] + args.columns)

    if target[args.key].duplicated().any():
        raise ValueError(f"target key has duplicates: {args.key}")
    if reference[args.key].duplicated().any():
        raise ValueError(f"reference key has duplicates: {args.key}")
    if set(target[args.key]) != set(reference[args.key]):
        missing = sorted(set(target[args.key]) - set(reference[args.key]))[:10]
        extra = sorted(set(reference[args.key]) - set(target[args.key]))[:10]
        raise ValueError(f"key mismatch: missing_in_ref={missing}, extra_in_ref={extra}")

    ref_aligned = reference.set_index(args.key).loc[target[args.key], args.columns]
    changed = {}
    for col in args.columns:
        old = target[col].to_numpy()
        new = ref_aligned[col].to_numpy()
        mask = ~np.isclose(old, new, rtol=0.0, atol=1e-12)
        changed[col] = {
            "count": int(np.sum(mask)),
            "old_values": [float(v) for v in old[mask][:20]],
            "new_values": [float(v) for v in new[mask][:20]],
        }
        target[col] = new

    backup_path = target_path.with_name(target_path.name + args.backup_suffix)
    if not backup_path.exists():
        shutil.copy2(target_path, backup_path)

    target.to_csv(target_path, index=False)

    report = {
        "target_csv": str(target_path),
        "reference_csv": str(reference_path),
        "backup_csv": str(backup_path),
        "key": args.key,
        "columns": args.columns,
        "rows": int(len(target)),
        "changed": changed,
    }
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"repaired: {target_path}")
    print(f"backup: {backup_path}")
    print(f"report: {out_report}")
    for col, info in changed.items():
        print(f"{col}: changed {info['count']} rows")


if __name__ == "__main__":
    main()
