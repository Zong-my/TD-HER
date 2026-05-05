#!/usr/bin/env python3
"""Summarize raw vs nonnegative-projected y2 metrics from an audit report."""

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "scenario",
    "model",
    "negative_count",
    "negative_fraction",
    "pred_negative_min",
    "pred_negative_mean",
    "raw_y2_mae",
    "clipped0_y2_mae",
    "clipped0_minus_raw_y2_mae",
    "raw_y2_rmse",
    "clipped0_y2_rmse",
    "clipped0_minus_raw_y2_rmse",
]


def iter_model_reports(audit: dict):
    """Yield (scenario, model_name, report) triples from supported audits."""
    if "models" in audit:
        for model in audit.get("models", []):
            yield "", model.get("model"), model
        return

    for scenario in audit.get("scenarios", []):
        scenario_name = scenario.get("scenario", "")
        for model in scenario.get("models", []):
            yield scenario_name, model.get("method") or model.get("model"), model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, help="Path to audit_report.json")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    audit_path = Path(args.audit)
    out_path = Path(args.out)
    with audit_path.open() as f:
        audit = json.load(f)

    rows = []
    for scenario, model_name, model in iter_model_reports(audit):
        report = model.get("target_constraint_report") or {}
        raw_mae = report.get("raw_mae")
        clipped_mae = report.get("clipped0_mae")
        raw_rmse = report.get("raw_rmse")
        clipped_rmse = report.get("clipped0_rmse")
        rows.append({
            "scenario": scenario,
            "model": model_name,
            "negative_count": report.get("negative_count"),
            "negative_fraction": report.get("negative_fraction"),
            "pred_negative_min": report.get("pred_negative_min", ""),
            "pred_negative_mean": report.get("pred_negative_mean", ""),
            "raw_y2_mae": raw_mae,
            "clipped0_y2_mae": clipped_mae,
            "clipped0_minus_raw_y2_mae": (
                clipped_mae - raw_mae if raw_mae is not None and clipped_mae is not None else ""
            ),
            "raw_y2_rmse": raw_rmse,
            "clipped0_y2_rmse": clipped_rmse,
            "clipped0_minus_raw_y2_rmse": (
                clipped_rmse - raw_rmse if raw_rmse is not None and clipped_rmse is not None else ""
            ),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
