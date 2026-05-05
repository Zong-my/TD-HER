#!/usr/bin/env python3
"""Export manuscript-ready result tables from audited experiment artifacts."""

import argparse
import csv
import json
from pathlib import Path


SCENARIOS = [
    ("L1_same_dist", "L1 same distribution"),
    ("L2_cross_cond_fewshot", "L2 cross-condition few-shot"),
    ("L3_cross_topo_fewshot", "L3 cross-topology few-shot"),
]
TDHER_BASE = "base_affine_convex_blend_nonnegative_y2"
TDHER_METHOD = "admission_affine_convex_blend_nonnegative_y2"
TDHER_CERT_METHOD = "certified_admission_affine_convex_blend_nonnegative_y2"
TDHER_SHARED_BASE = "base_shared_affine_convex_blend_nonnegative_y2"
TDHER_SHARED_METHOD = "admission_shared_affine_convex_blend_nonnegative_y2"
TDHER_RAW_METHOD = "admission_affine_convex_blend"
TDHER_CERT_RAW_METHOD = "certified_admission_affine_convex_blend"
TDHER_ABLATIONS = [
    {
        "method": "base_best_expert",
        "label": "Base target-wise best expert",
        "expert_set": "base",
        "routing": "best_expert",
        "adapter_policy": "none",
        "affine_calibration": "no",
        "nonnegative_y2": "no",
    },
    {
        "method": "base_convex_blend",
        "label": "Base convex routing",
        "expert_set": "base",
        "routing": "convex",
        "adapter_policy": "none",
        "affine_calibration": "no",
        "nonnegative_y2": "no",
    },
    {
        "method": "base_affine_convex_blend",
        "label": "Base convex routing + affine",
        "expert_set": "base",
        "routing": "convex",
        "adapter_policy": "none",
        "affine_calibration": "yes",
        "nonnegative_y2": "no",
    },
    {
        "method": "base_affine_convex_blend_nonnegative_y2",
        "label": "Base convex routing + affine + physical y2",
        "expert_set": "base",
        "routing": "convex",
        "adapter_policy": "none",
        "affine_calibration": "yes",
        "nonnegative_y2": "yes",
    },
    {
        "method": "convex_blend",
        "label": "Always-adapt convex routing",
        "expert_set": "base+adapter",
        "routing": "convex",
        "adapter_policy": "always",
        "affine_calibration": "no",
        "nonnegative_y2": "no",
    },
    {
        "method": "affine_convex_blend",
        "label": "Always-adapt convex routing + affine",
        "expert_set": "base+adapter",
        "routing": "convex",
        "adapter_policy": "always",
        "affine_calibration": "yes",
        "nonnegative_y2": "no",
    },
    {
        "method": "admission_convex_blend",
        "label": "Admitted convex routing",
        "expert_set": "admitted",
        "routing": "convex",
        "adapter_policy": "OOF admission",
        "affine_calibration": "no",
        "nonnegative_y2": "no",
    },
    {
        "method": "admission_affine_convex_blend",
        "label": "TD-HER raw",
        "expert_set": "admitted",
        "routing": "convex",
        "adapter_policy": "OOF admission",
        "affine_calibration": "yes",
        "nonnegative_y2": "no",
    },
    {
        "method": "admission_affine_convex_blend_nonnegative_y2",
        "label": "TD-HER physical",
        "expert_set": "admitted",
        "routing": "convex",
        "adapter_policy": "OOF admission",
        "affine_calibration": "yes",
        "nonnegative_y2": "yes",
    },
    {
        "method": "ridge_stack",
        "label": "Unconstrained ridge stacking",
        "expert_set": "base+adapter",
        "routing": "unconstrained",
        "adapter_policy": "always",
        "affine_calibration": "implicit",
        "nonnegative_y2": "no",
    },
]
IEEE300_ORDER = [
    "LightGBM",
    "KAN",
    "ConvLSTM",
    "PatchTST",
    "Mamba",
    "ST-GCN",
    "FT-Transformer",
    "TabR",
]


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_tdher(router_dir: Path, out_dir: Path):
    rows = []
    for scenario, label in SCENARIOS:
        summary = load_json(router_dir / "scenarios" / scenario / "metrics_summary.json")
        base = summary[TDHER_BASE]["metrics"]
        method = summary[TDHER_METHOD]["metrics"]
        rows.append({
            "scenario": label,
            "base_y1_mae": base["y1_MAE"],
            "base_y2_mae": base["y2_MAE"],
            "tdher_y1_mae": method["y1_MAE"],
            "tdher_y2_mae": method["y2_MAE"],
            "y1_relative_improvement": (
                (base["y1_MAE"] - method["y1_MAE"]) / base["y1_MAE"]
                if base["y1_MAE"] else ""
            ),
            "y2_relative_improvement": (
                (base["y2_MAE"] - method["y2_MAE"]) / base["y2_MAE"]
                if base["y2_MAE"] else ""
            ),
        })

    write_csv(
        out_dir / "tdher_main_results.csv",
        [
            "scenario",
            "base_y1_mae",
            "base_y2_mae",
            "tdher_y1_mae",
            "tdher_y2_mae",
            "y1_relative_improvement",
            "y2_relative_improvement",
        ],
        rows,
    )

    l3 = load_json(router_dir / "scenarios" / "L3_cross_topo_fewshot" / "metrics_summary.json")
    bootstrap = l3["paired_bootstrap"]["admission_affine_nonnegative_vs_base_affine_nonnegative"]
    boot_rows = []
    for target in ["y1", "y2"]:
        item = bootstrap[target]
        boot_rows.append({
            "scenario": "L3 cross-topology few-shot",
            "target": target,
            "reference_mae": item["reference_MAE"],
            "candidate_mae": item["candidate_MAE"],
            "delta_mae_candidate_minus_reference": item["delta_MAE_candidate_minus_reference"],
            "relative_improvement": item["relative_improvement"],
            "ci95_low": item["ci95"][0],
            "ci95_high": item["ci95"][1],
            "p_two_sided_bootstrap": item["p_two_sided_bootstrap"],
            "n_rounds": item["n_rounds"],
        })

    write_csv(
        out_dir / "tdher_l3_bootstrap.csv",
        [
            "scenario",
            "target",
            "reference_mae",
            "candidate_mae",
            "delta_mae_candidate_minus_reference",
            "relative_improvement",
            "ci95_low",
            "ci95_high",
            "p_two_sided_bootstrap",
            "n_rounds",
        ],
        boot_rows,
    )


def export_tdher_ablation(router_dir: Path, out_dir: Path):
    rows = []
    for scenario, label in SCENARIOS:
        summary = load_json(router_dir / "scenarios" / scenario / "metrics_summary.json")
        final = summary[TDHER_METHOD]["metrics"]
        for order, spec in enumerate(TDHER_ABLATIONS, start=1):
            method = spec["method"]
            metrics = summary[method]["metrics"]
            rows.append({
                "scenario": label,
                "order": order,
                "method": method,
                "label": spec["label"],
                "expert_set": spec["expert_set"],
                "routing": spec["routing"],
                "adapter_policy": spec["adapter_policy"],
                "affine_calibration": spec["affine_calibration"],
                "nonnegative_y2": spec["nonnegative_y2"],
                "y1_mae": metrics["y1_MAE"],
                "y2_mae": metrics["y2_MAE"],
                "delta_y1_mae_vs_final": metrics["y1_MAE"] - final["y1_MAE"],
                "delta_y2_mae_vs_final": metrics["y2_MAE"] - final["y2_MAE"],
            })

    write_csv(
        out_dir / "tdher_ablation.csv",
        [
            "scenario",
            "order",
            "method",
            "label",
            "expert_set",
            "routing",
            "adapter_policy",
            "affine_calibration",
            "nonnegative_y2",
            "y1_mae",
            "y2_mae",
            "delta_y1_mae_vs_final",
            "delta_y2_mae_vs_final",
        ],
        rows,
    )


def export_tdher_threshold_sensitivity(router_dir: Path, out_dir: Path):
    rows = []
    for scenario, label in SCENARIOS:
        summary = load_json(router_dir / "scenarios" / scenario / "metrics_summary.json")
        sensitivity = summary["adapter_admission_threshold_sensitivity"]
        for threshold, item in sorted(sensitivity.items(), key=lambda x: float(x[0])):
            metrics = item["metrics"]
            details = item["details"]
            y1_admission = details["y1"]["admission"]["adapter_details"]["LightGBM-Adapter"]
            y2_admission = details["y2"]["admission"]["adapter_details"]["LightGBM-Adapter"]
            rows.append({
                "scenario": label,
                "threshold": threshold,
                "y1_mae": metrics["y1_MAE"],
                "y2_mae": metrics["y2_MAE"],
                "y1_adapter_admitted": y1_admission["admitted"],
                "y2_adapter_admitted": y2_admission["admitted"],
                "y1_adapter_cal_mae": y1_admission["cal_MAE"],
                "y1_required_mae": y1_admission["required_MAE"],
                "y2_adapter_cal_mae": y2_admission["cal_MAE"],
                "y2_required_mae": y2_admission["required_MAE"],
            })

    write_csv(
        out_dir / "tdher_threshold_sensitivity.csv",
        [
            "scenario",
            "threshold",
            "y1_mae",
            "y2_mae",
            "y1_adapter_admitted",
            "y2_adapter_admitted",
            "y1_adapter_cal_mae",
            "y1_required_mae",
            "y2_adapter_cal_mae",
            "y2_required_mae",
        ],
        rows,
    )


def export_tdher_final_routing_weights(router_dir: Path, out_dir: Path):
    rows = []
    for scenario, label in SCENARIOS:
        summary = load_json(router_dir / "scenarios" / scenario / "metrics_summary.json")
        details = summary[TDHER_RAW_METHOD]["details"]
        for target in ["y1", "y2"]:
            target_details = details[target]
            admission = target_details["admission"]
            adapter_details = admission["adapter_details"].get("LightGBM-Adapter", {})
            admitted_models = set(admission["admitted_models"])
            for expert, weight in target_details["weights"].items():
                rows.append({
                    "scenario": label,
                    "target": target,
                    "expert": expert,
                    "weight": weight,
                    "is_adapter": expert == "LightGBM-Adapter",
                    "admitted": expert in admitted_models,
                    "adapter_cal_mae": adapter_details.get("cal_MAE", "") if expert == "LightGBM-Adapter" else "",
                    "adapter_required_mae": adapter_details.get("required_MAE", "") if expert == "LightGBM-Adapter" else "",
                    "base_best_model": admission["base_best_model"],
                    "base_best_cal_mae": admission["base_best_cal_MAE"],
                    "affine_slope": target_details["slope"],
                    "affine_bias": target_details["bias"],
                })

    write_csv(
        out_dir / "tdher_final_routing_weights.csv",
        [
            "scenario",
            "target",
            "expert",
            "weight",
            "is_adapter",
            "admitted",
            "adapter_cal_mae",
            "adapter_required_mae",
            "base_best_model",
            "base_best_cal_mae",
            "affine_slope",
            "affine_bias",
        ],
        rows,
    )


def export_tdher_shared_routing_ablation(router_dir: Path, out_dir: Path):
    rows = []
    variants = [
        {
            "variant": "base",
            "target_wise_method": TDHER_BASE,
            "shared_method": TDHER_SHARED_BASE,
            "description": "frozen experts only",
        },
        {
            "variant": "oof_admitted",
            "target_wise_method": TDHER_METHOD,
            "shared_method": TDHER_SHARED_METHOD,
            "description": "OOF-admitted expert pool",
        },
    ]
    for scenario, label in SCENARIOS:
        summary = load_json(router_dir / "scenarios" / scenario / "metrics_summary.json")
        for spec in variants:
            tw = summary[spec["target_wise_method"]]["metrics"]
            sh = summary[spec["shared_method"]]["metrics"]
            rows.append({
                "scenario": label,
                "variant": spec["variant"],
                "description": spec["description"],
                "target_wise_method": spec["target_wise_method"],
                "shared_method": spec["shared_method"],
                "target_wise_y1_mae": tw["y1_MAE"],
                "shared_y1_mae": sh["y1_MAE"],
                "shared_minus_target_wise_y1_mae": sh["y1_MAE"] - tw["y1_MAE"],
                "shared_relative_loss_y1": (
                    (sh["y1_MAE"] - tw["y1_MAE"]) / tw["y1_MAE"]
                    if tw["y1_MAE"] else ""
                ),
                "target_wise_y2_mae": tw["y2_MAE"],
                "shared_y2_mae": sh["y2_MAE"],
                "shared_minus_target_wise_y2_mae": sh["y2_MAE"] - tw["y2_MAE"],
                "shared_relative_loss_y2": (
                    (sh["y2_MAE"] - tw["y2_MAE"]) / tw["y2_MAE"]
                    if tw["y2_MAE"] else ""
                ),
            })

    write_csv(
        out_dir / "tdher_shared_routing_ablation.csv",
        [
            "scenario",
            "variant",
            "description",
            "target_wise_method",
            "shared_method",
            "target_wise_y1_mae",
            "shared_y1_mae",
            "shared_minus_target_wise_y1_mae",
            "shared_relative_loss_y1",
            "target_wise_y2_mae",
            "shared_y2_mae",
            "shared_minus_target_wise_y2_mae",
            "shared_relative_loss_y2",
        ],
        rows,
    )


def route_l1_divergence(weights_a: dict, weights_b: dict) -> float:
    experts = set(weights_a) | set(weights_b)
    return 0.5 * sum(
        abs(float(weights_a.get(expert, 0.0)) - float(weights_b.get(expert, 0.0)))
        for expert in experts
    )


def export_tdher_route_conflict_diagnostics(router_dir: Path, out_dir: Path):
    rows = []
    variants = [
        {
            "variant": "base",
            "description": "frozen experts only",
            "target_wise_raw": "base_affine_convex_blend",
            "target_wise_physical": TDHER_BASE,
            "shared_physical": TDHER_SHARED_BASE,
        },
        {
            "variant": "oof_admitted",
            "description": "threshold OOF-admitted expert pool",
            "target_wise_raw": TDHER_RAW_METHOD,
            "target_wise_physical": TDHER_METHOD,
            "shared_physical": TDHER_SHARED_METHOD,
        },
        {
            "variant": "certified_admitted",
            "description": "bootstrap-certified OOF-admitted expert pool",
            "target_wise_raw": TDHER_CERT_RAW_METHOD,
            "target_wise_physical": TDHER_CERT_METHOD,
            "shared_physical": "",
        },
    ]

    for scenario, label in SCENARIOS:
        summary = load_json(router_dir / "scenarios" / scenario / "metrics_summary.json")
        for spec in variants:
            if spec["target_wise_raw"] not in summary:
                continue
            details = summary[spec["target_wise_raw"]]["details"]
            y1_weights = details["y1"]["weights"]
            y2_weights = details["y2"]["weights"]
            physical = summary[spec["target_wise_physical"]]["metrics"]
            shared_metrics = (
                summary[spec["shared_physical"]]["metrics"]
                if spec["shared_physical"] and spec["shared_physical"] in summary
                else {}
            )

            rows.append({
                "scenario": label,
                "variant": spec["variant"],
                "description": spec["description"],
                "route_l1_divergence": route_l1_divergence(y1_weights, y2_weights),
                "target_wise_y1_mae": physical["y1_MAE"],
                "target_wise_y2_mae": physical["y2_MAE"],
                "shared_y1_mae": shared_metrics.get("y1_MAE", ""),
                "shared_y2_mae": shared_metrics.get("y2_MAE", ""),
                "shared_minus_target_wise_y1_mae": (
                    shared_metrics.get("y1_MAE", "") - physical["y1_MAE"]
                    if shared_metrics else ""
                ),
                "shared_minus_target_wise_y2_mae": (
                    shared_metrics.get("y2_MAE", "") - physical["y2_MAE"]
                    if shared_metrics else ""
                ),
            })

    write_csv(
        out_dir / "tdher_route_conflict_diagnostics.csv",
        [
            "scenario",
            "variant",
            "description",
            "route_l1_divergence",
            "target_wise_y1_mae",
            "target_wise_y2_mae",
            "shared_y1_mae",
            "shared_y2_mae",
            "shared_minus_target_wise_y1_mae",
            "shared_minus_target_wise_y2_mae",
        ],
        rows,
    )


def export_tdher_certified_admission(router_dir: Path, out_dir: Path):
    rows = []
    for scenario, label in SCENARIOS:
        summary = load_json(router_dir / "scenarios" / scenario / "metrics_summary.json")
        if TDHER_CERT_RAW_METHOD not in summary:
            continue
        cert_details = summary[TDHER_CERT_RAW_METHOD]["details"]
        threshold_details = summary[TDHER_RAW_METHOD]["details"]
        cert_metrics = summary[TDHER_CERT_METHOD]["metrics"]
        threshold_metrics = summary[TDHER_METHOD]["metrics"]
        for target in ["y1", "y2"]:
            cert_admission = cert_details[target]["admission"]
            threshold_admission = threshold_details[target]["admission"]
            adapter_name = "LightGBM-Adapter"
            cert_adapter = cert_admission["adapter_details"].get(adapter_name, {})
            threshold_adapter = threshold_admission["adapter_details"].get(
                adapter_name, {})
            cert = cert_adapter.get("certificate", {})
            ci = cert.get("ci95", ["", ""])
            rows.append({
                "scenario": label,
                "target": target,
                "base_best_model": cert_admission["base_best_model"],
                "base_best_cal_mae": cert_admission["base_best_cal_MAE"],
                "adapter_cal_mae": cert_adapter.get("cal_MAE", ""),
                "required_mae": cert_adapter.get("required_MAE", ""),
                "threshold_admitted": threshold_adapter.get("admitted", ""),
                "certified_admitted": cert_adapter.get("admitted", ""),
                "practical_gate": cert_adapter.get("practical_gate", ""),
                "certificate_gate": cert_adapter.get("certificate_gate", ""),
                "mean_delta_adapter_minus_base": cert.get("mean_delta", ""),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "p_two_sided_bootstrap": cert.get("p_two_sided_bootstrap", ""),
                "n_rounds": cert.get("n_rounds", ""),
                "threshold_tdher_target_mae": threshold_metrics[f"{target}_MAE"],
                "certified_tdher_target_mae": cert_metrics[f"{target}_MAE"],
                "certified_minus_threshold_mae": (
                    cert_metrics[f"{target}_MAE"] - threshold_metrics[f"{target}_MAE"]
                ),
            })

    write_csv(
        out_dir / "tdher_certified_admission.csv",
        [
            "scenario",
            "target",
            "base_best_model",
            "base_best_cal_mae",
            "adapter_cal_mae",
            "required_mae",
            "threshold_admitted",
            "certified_admitted",
            "practical_gate",
            "certificate_gate",
            "mean_delta_adapter_minus_base",
            "ci95_low",
            "ci95_high",
            "p_two_sided_bootstrap",
            "n_rounds",
            "threshold_tdher_target_mae",
            "certified_tdher_target_mae",
            "certified_minus_threshold_mae",
        ],
        rows,
    )


def export_ieee300(exp7_dir: Path, out_dir: Path):
    in_path = exp7_dir / "exp1_results.csv"
    with in_path.open() as f:
        rows_by_model = {
            row[""]: row
            for row in csv.DictReader(f)
        }

    rows = []
    for model in IEEE300_ORDER:
        row = rows_by_model.get(model)
        if not row:
            continue
        rows.append({
            "model": model,
            "y1_mae": row["y1_MAE"],
            "y1_rmse": row["y1_RMSE"],
            "y2_mae": row["y2_MAE"],
            "y2_rmse": row["y2_RMSE"],
            "timing_median_ms": row.get("timing_median_ms", ""),
            "train_time_s": row.get("train_time_s", ""),
        })

    write_csv(
        out_dir / "ieee300_scalability.csv",
        [
            "model",
            "y1_mae",
            "y1_rmse",
            "y2_mae",
            "y2_rmse",
            "timing_median_ms",
            "train_time_s",
        ],
        rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-dir", default="results/ieee39/exp_tdh_router")
    parser.add_argument("--ieee300-dir", default="results/ieee300/exp7_rebuild")
    parser.add_argument("--out-dir", default="results/paper_tables")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    export_tdher(Path(args.router_dir), out_dir)
    export_tdher_ablation(Path(args.router_dir), out_dir)
    export_tdher_threshold_sensitivity(Path(args.router_dir), out_dir)
    export_tdher_final_routing_weights(Path(args.router_dir), out_dir)
    export_tdher_shared_routing_ablation(Path(args.router_dir), out_dir)
    export_tdher_route_conflict_diagnostics(Path(args.router_dir), out_dir)
    export_tdher_certified_admission(Path(args.router_dir), out_dir)
    export_ieee300(Path(args.ieee300_dir), out_dir)
    print(f"Exported manuscript tables to {out_dir}")


if __name__ == "__main__":
    main()
