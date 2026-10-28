#!/usr/bin/env python3
"""Fresh, scope-explicit deployment timing audit for Reviewer 2.

This reuses the frozen model loaders from ``exp_tdh_pipeline_timing`` but writes
only to a new revision directory and labels the optimistic parallel aggregate
as a critical-path estimate rather than a measured concurrent workflow.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import argparse
import csv
import json
import platform
import sys
import types
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ``exp4_generalization`` imports mrmr for unrelated training functions.  The
# timing path only reads cached feature lists, so avoid adding a new dependency.
if "mrmr" not in sys.modules:
    mrmr_stub = types.ModuleType("mrmr")

    def _unused_mrmr(*args, **kwargs):
        raise RuntimeError("mrmr fitting is outside the frozen timing path")

    mrmr_stub.mrmr_regression = _unused_mrmr
    sys.modules["mrmr"] = mrmr_stub

from data_proc.datasets import adjacency_to_edge_index
from experiments import exp_tdh_pipeline_timing as base


DEFAULT_OUT = work_path('results/ieee39/revision_r2_timing')
OBSERVATION_WINDOW_MS = 100.0


def fixed_time_dl_single(model, model_name, sample_data, device):
    """Avoid per-call adjacency disk I/O while retaining input-transfer cost."""
    edge_index = edge_weight = None
    if model_name == "ST-GCN":
        adjacency = np.load(base.ADJ_PATH)
        edge_index, edge_weight = adjacency_to_edge_index(adjacency)
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=device)
        edge_weight = torch.as_tensor(
            edge_weight, dtype=torch.float32, device=device
        )

    def fn():
        with torch.no_grad():
            if model_name in ("ConvLSTM", "PatchTST", "Mamba"):
                temporal = torch.as_tensor(
                    sample_data["Xt_n"][:1], dtype=torch.float32, device=device
                )
                static = torch.as_tensor(
                    sample_data["Xs_n"][:1], dtype=torch.float32, device=device
                )
                output = model(temporal, static)
            elif model_name == "ST-GCN":
                node = torch.as_tensor(
                    sample_data["Xn_n"][:1], dtype=torch.float32, device=device
                )
                static = torch.as_tensor(
                    sample_data["Xs_n"][:1], dtype=torch.float32, device=device
                )
                output = model(node, static, edge_index, edge_weight)
            elif model_name == "FT-Transformer":
                tabular = torch.as_tensor(
                    sample_data["X_n"][:1], dtype=torch.float32, device=device
                )
                output = model(tabular)
            else:
                raise ValueError(model_name)
            if device != "cpu":
                torch.cuda.synchronize()
            return output.cpu().numpy()

    return fn


def annotate(mode: str, result: dict) -> dict:
    summary = result["pipeline_summary"]
    valid_experts = [
        timing for timing in result.get("per_expert", {}).values()
        if isinstance(timing.get("median_ms"), (int, float))
        and timing["median_ms"] > 0
    ]
    expert_sequential_p95 = float(sum(
        timing.get("p95_ms", timing["median_ms"])
        for timing in valid_experts
    ))
    expert_critical_p95 = float(max(
        (timing.get("p95_ms", timing["median_ms"])
         for timing in valid_experts),
        default=0.0,
    ))
    rep_p95 = float(result["rep_construction"].get(
        "p95_ms", summary["rep_construction_ms"]
    ))
    route_p95 = float(result.get("routing_stage", {}).get(
        "p95_ms", summary["routing_ms"]
    ))
    summary.update({
        "expert_sequential_p95_ms": expert_sequential_p95,
        "expert_critical_path_p95_ms": expert_critical_p95,
        "total_sequential_p95_ms": (
            rep_p95 + expert_sequential_p95 + route_p95
        ),
        "total_critical_path_p95_ms": (
            rep_p95 + expert_critical_p95 + route_p95
        ),
    })
    sequential = float(summary["total_sequential_ms"])
    critical = float(summary["total_parallel_ms"])
    mamba_included = bool(
        result.get("per_expert", {}).get("Mamba", {}).get("median_ms", -1) > 0
    )
    result["deployment_interpretation"] = {
        "observation_window_ms": OBSERVATION_WINDOW_MS,
        "compute_starts_after_observation_window": True,
        "model_compute_sequential_median_ms": sequential,
        "model_compute_sequential_p95_ms": float(
            summary["total_sequential_p95_ms"]
        ),
        "idealized_parallel_critical_path_median_ms": critical,
        "idealized_parallel_critical_path_p95_ms": float(
            summary["total_critical_path_p95_ms"]
        ),
        "idealized_parallel_is_measured_concurrently": False,
        "observation_plus_model_sequential_median_ms": (
            OBSERVATION_WINDOW_MS + sequential
        ),
        "observation_plus_model_idealized_critical_path_ms": (
            OBSERVATION_WINDOW_MS + critical
        ),
        "mamba_included": mamba_included,
        "mode_label": (
            "seven-expert GPU model-side inference"
            if mamba_included else (
                "six-expert CPU timing path (Mamba omitted; accuracy not evaluated)"
            )
        ),
    }
    result["timing_scope"] = {
        "included": [
            "cached feature selection and RepA scaling",
            "single-sample tensor creation and host/device transfer",
            "individual expert forward passes",
            "target-wise routing, affine correction, and y2 projection",
        ],
        "excluded": [
            "disturbance/event detection",
            "PMU communication and phasor data acquisition",
            "raw waveform-to-feature extraction before cached representations",
            "process scheduling, network contention, and control actuation",
        ],
    }
    result["environment"] = {
        "mode": mode,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gpu", "cpu", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base.N_WARMUP = args.warmup
    base.N_RUNS = args.runs
    base.time_dl_single = fixed_time_dl_single

    modes = []
    if args.mode in ("gpu", "both"):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU timing requested but CUDA is unavailable")
        modes.append(("gpu", "cuda"))
    if args.mode in ("cpu", "both"):
        modes.append(("cpu", "cpu"))

    results = {}
    rows = []
    expert_rows = []
    for label, device in modes:
        measured = annotate(label, base.run_timing(device))
        results[label] = measured
        with open(args.out_dir / f"pipeline_timing_{label}.json", "w") as handle:
            json.dump(measured, handle, indent=2, default=str)
        interpretation = measured["deployment_interpretation"]
        summary = measured["pipeline_summary"]
        rows.append({
            "mode": label,
            "mode_label": interpretation["mode_label"],
            "n_experts": summary["num_experts_timed"],
            "representation_median_ms": summary["rep_construction_ms"],
            "experts_sequential_median_ms": summary["expert_sequential_ms"],
            "idealized_expert_critical_path_median_ms": summary["expert_parallel_ms"],
            "routing_median_ms": summary["routing_ms"],
            "model_compute_sequential_median_ms": interpretation[
                "model_compute_sequential_median_ms"
            ],
            "model_compute_sequential_p95_ms": interpretation[
                "model_compute_sequential_p95_ms"
            ],
            "idealized_model_critical_path_median_ms": interpretation[
                "idealized_parallel_critical_path_median_ms"
            ],
            "idealized_model_critical_path_p95_ms": interpretation[
                "idealized_parallel_critical_path_p95_ms"
            ],
            "observation_plus_model_sequential_median_ms": interpretation[
                "observation_plus_model_sequential_median_ms"
            ],
        })
        for expert, timing in measured.get("per_expert", {}).items():
            expert_rows.append({
                "mode": label,
                "expert": expert,
                "median_ms": timing.get("median_ms"),
                "p95_ms": timing.get("p95_ms"),
                "note": timing.get("note", timing.get("error", "")),
            })

    with open(args.out_dir / "protocol.json", "w") as handle:
        json.dump({
            "warmup": args.warmup,
            "runs": args.runs,
            "observation_window_ms": OBSERVATION_WINDOW_MS,
            "test_selection": False,
            "parallel_value": "idealized critical-path estimate, not concurrency",
        }, handle, indent=2)
    for filename, field_rows in (
        ("pipeline_timing_summary.csv", rows),
        ("per_expert_timing.csv", expert_rows),
    ):
        with open(args.out_dir / filename, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_rows[0].keys())
            writer.writeheader()
            writer.writerows(field_rows)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
