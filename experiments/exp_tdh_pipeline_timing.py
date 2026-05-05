#!/usr/bin/env python3
"""
End-to-end TD-HER pipeline inference timing.

Measures the actual deployment latency of the full TD-HER pipeline,
including representation construction, all expert forward passes,
and the routing/calibration/projection stage.

Reports both GPU and CPU-only modes, with sequential and parallel
expert execution breakdowns.

This script does NOT modify any existing results or model artifacts.
All outputs are saved to a new dedicated directory.
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.preprocessing import StandardScaler

from experiments.exp4_generalization import (
    MS,
    MODELS,
    REP_DIR,
    EXP1_DIR,
    ADJ_PATH,
    evaluate,
    load_level_data,
    load_train_scalers,
)

from data_proc.datasets import TabularDataset, TensorDataset, GraphDataset
from data_proc.datasets import adjacency_to_edge_index
from models.stgcn_model import normalize_adjacency
from models.convlstm_model import ConvLSTMFreqModel
from models.patchtst_model import PatchTSTFreqModel
from models.mamba_model import MambaFreqModel
from models.stgcn_model import STGCNBatchedModel
from models.ft_transformer_model import FTTransformerModel
from models.tabr_model import TabRModel
from models.kan_model import build_kan_from_trial

# ── Configuration ────────────────────────────────────────────────
OUT_DIR = Path("results/ieee39/tdher_pipeline_timing")
N_WARMUP = 20
N_RUNS = 100
# Exclude TabR from router models (same as exp_tdh_router.py)
ROUTER_MODELS = [m for m in MODELS if m != "TabR"]

# L1 routing weights from audited results
ROUTING_WEIGHTS_PATH = Path(
    "results/paper_tables/tdher_final_routing_weights.csv"
)


# ── Model loading helpers ────────────────────────────────────────

def load_lightgbm_models():
    """Load LightGBM per-target models."""
    models = {}
    for target in ["y1", "y2"]:
        with open(f"{EXP1_DIR}/LightGBM/{target}_lgb_model.pkl", "rb") as f:
            models[target] = pickle.load(f)
    return models


def load_kan_models(scalers):
    """Load KAN per-target models."""
    KAN_K = 100
    mdir = f"{EXP1_DIR}/KAN"
    mrmr = scalers["mrmr"]
    feature_names = scalers["feature_names"]
    X_train_A = np.load(f"{REP_DIR}/repA/ms{MS}/X_train.npy")
    models = {}
    for tname in ["y1", "y2"]:
        kan_feat_names = mrmr[tname][:KAN_K]
        kan_feat_idx = [
            feature_names.index(f) for f in kan_feat_names if f in feature_names
        ]
        kan_scaler = StandardScaler().fit(X_train_A[:, kan_feat_idx])
        with open(f"{mdir}/{tname}_best_params.json") as f:
            bp = json.load(f)

        class FixedTrial:
            def __init__(self, p):
                self._params = dict(p)
                self.params = dict(p)
            def suggest_int(self, name, low, high, **kw):
                return self._params[name]
            def suggest_float(self, name, low, high, **kw):
                return self._params[name]
            def suggest_categorical(self, name, choices, **kw):
                return self._params[name]

        kan_dim = len(kan_feat_idx)
        model, _ = build_kan_from_trial(FixedTrial(bp), kan_dim)
        ckpt = torch.load(
            f"{mdir}/{tname}_model.pth", map_location="cpu", weights_only=False
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models[tname] = {
            "model": model,
            "feat_idx": kan_feat_idx,
            "scaler": kan_scaler,
            "bs": bp["bs"],
        }
    return models


def load_dl_model(model_name, device="cpu"):
    """Load a DL model checkpoint."""
    mdir = f"{EXP1_DIR}/{model_name}"
    ckpt = torch.load(
        f"{mdir}/model.pth", map_location=device, weights_only=False
    )
    model_kwargs = ckpt.get("model_kwargs", {})
    model_state = ckpt["model_state_dict"]

    model_classes = {
        "ConvLSTM": ConvLSTMFreqModel,
        "PatchTST": PatchTSTFreqModel,
        "Mamba": MambaFreqModel,
        "ST-GCN": STGCNBatchedModel,
        "FT-Transformer": FTTransformerModel,
        "TabR": TabRModel,
    }
    cls = model_classes[model_name]
    model = cls(**model_kwargs)
    if model_name == "ST-GCN":
        adj = np.load(ADJ_PATH)
        adj_norm = torch.FloatTensor(normalize_adjacency(adj))
        model.set_adj(adj_norm)
    model.load_state_dict(model_state)
    model.eval()
    return model


# ── Single-sample inference helpers ──────────────────────────────

def time_lightgbm_single(lgb_models, x_n_single):
    """Time LightGBM prediction on a single sample (CPU)."""
    def fn():
        y1 = lgb_models["y1"].predict(x_n_single)
        y2 = lgb_models["y2"].predict(x_n_single)
        return np.array([[y1[0], y2[0]]])
    return fn


def time_kan_single(kan_models, x_A_single, device):
    """Time KAN prediction on a single sample."""
    def fn():
        preds = np.zeros((1, 2))
        for ti, tname in enumerate(["y1", "y2"]):
            m = kan_models[tname]
            x_kan = m["scaler"].transform(
                x_A_single[:, m["feat_idx"]]
            ).astype(np.float32)
            x_t = torch.FloatTensor(x_kan).to(device)
            with torch.no_grad():
                out = m["model"](x_t).cpu().numpy()
            preds[:, ti] = out.flatten()
        return preds
    return fn


def time_dl_single(model, model_name, sample_data, device):
    """Time a DL model prediction on a single sample."""
    def fn():
        with torch.no_grad():
            if model_name in ("ConvLSTM", "PatchTST", "Mamba"):
                xt = torch.FloatTensor(sample_data["Xt_n"][:1]).to(device)
                xs = torch.FloatTensor(sample_data["Xs_n"][:1]).to(device)
                out = model(xt, xs)
            elif model_name == "ST-GCN":
                xn = torch.FloatTensor(sample_data["Xn_n"][:1]).to(device)
                xs = torch.FloatTensor(sample_data["Xs_n"][:1]).to(device)
                adj = np.load(ADJ_PATH)
                ei, ew = adjacency_to_edge_index(adj)
                ei_t = torch.LongTensor(ei).to(device)
                ew_t = torch.FloatTensor(ew).to(device)
                out = model(xn, xs, ei_t, ew_t)
            elif model_name == "FT-Transformer":
                xn = torch.FloatTensor(sample_data["X_n"][:1]).to(device)
                out = model(xn)
            if device != "cpu":
                torch.cuda.synchronize()
            return out.cpu().numpy()
    return fn


def measure_fn(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Measure function execution time."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        fn()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        times.append(elapsed_ms)
    return {
        "median_ms": float(np.median(times)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p5_ms": float(np.percentile(times, 5)),
        "p95_ms": float(np.percentile(times, 95)),
    }


# ── Routing stage timing ────────────────────────────────────────

def load_l1_routing_params():
    """Load L1 routing weights and affine params from audited CSV."""
    import csv
    weights = {"y1": {}, "y2": {}}
    affine = {"y1": {}, "y2": {}}
    with open(ROUTING_WEIGHTS_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["scenario"] != "L1_same_dist":
                continue
            target = row["target"]
            expert = row["expert"]
            w = float(row["weight"])
            weights[target][expert] = w
            if "affine_slope" in row and row["affine_slope"]:
                affine[target] = {
                    "slope": float(row["affine_slope"]),
                    "bias": float(row["affine_bias"]),
                }
    return weights, affine


def time_routing_stage(expert_preds, weights, affine):
    """Time only the routing + affine + projection stage."""
    expert_names = list(expert_preds.keys())

    def fn():
        results = {}
        for target in ["y1", "y2"]:
            w_arr = np.array(
                [weights[target].get(e, 0.0) for e in expert_names]
            )
            pred_arr = np.array(
                [expert_preds[e][0, 0 if target == "y1" else 1]
                 for e in expert_names]
            )
            routed = np.dot(w_arr, pred_arr)
            a = affine[target].get("slope", 1.0)
            b = affine[target].get("bias", 0.0)
            calibrated = a * routed + b
            if target == "y2":
                calibrated = max(calibrated, 0.0)
            results[target] = calibrated
        return results

    return fn


# ── Representation construction timing ───────────────────────────

def time_rep_construction(raw_sample_A, scalers):
    """Time the feature scaling and representation prep for one sample."""
    feat_idx = scalers["feat_idx"]

    def fn():
        # RepA: feature selection + scaling
        x_sel = raw_sample_A[:, feat_idx]
        x_n = scalers["scaler_A"].transform(x_sel).astype(np.float32)
        return x_n

    return fn


# ── Main ─────────────────────────────────────────────────────────

def run_timing(device: str):
    """Run full pipeline timing on a given device."""
    logger.info(f"Running pipeline timing on device={device}")

    # Load data and scalers
    scalers = load_train_scalers()
    test_data = load_level_data("test", scalers)

    # Prepare single sample data
    sample_data = {
        "X_n": test_data["X_n"][:1],
        "X_A": test_data["X_A"][:1],
        "Xt_n": test_data["Xt_n"][:1],
        "Xs_n": test_data["Xs_n"][:1],
        "Xn_n": test_data["Xn_n"][:1],
        "y": test_data["y"][:1],
    }
    raw_sample_A = test_data["X_A"][:1]

    results = {"device": device, "n_warmup": N_WARMUP, "n_runs": N_RUNS}

    # ── Stage 1: Representation construction ──
    logger.info("Timing representation construction...")
    rep_fn = time_rep_construction(raw_sample_A, scalers)
    results["rep_construction"] = measure_fn(rep_fn)

    # ── Stage 2: Per-expert inference ──
    logger.info("Timing per-expert inference...")
    expert_timings = {}
    expert_single_preds = {}

    # LightGBM (always CPU)
    lgb_models = load_lightgbm_models()
    lgb_fn = time_lightgbm_single(lgb_models, sample_data["X_n"])
    expert_timings["LightGBM"] = measure_fn(lgb_fn)
    expert_single_preds["LightGBM"] = lgb_fn()
    logger.info(f"  LightGBM: {expert_timings['LightGBM']['median_ms']:.3f} ms")

    # KAN
    kan_device = device
    kan_models = load_kan_models(scalers)
    for tname in ["y1", "y2"]:
        kan_models[tname]["model"] = kan_models[tname]["model"].to(kan_device)
    kan_fn = time_kan_single(kan_models, sample_data["X_A"], kan_device)
    expert_timings["KAN"] = measure_fn(kan_fn)
    expert_single_preds["KAN"] = kan_fn()
    logger.info(f"  KAN: {expert_timings['KAN']['median_ms']:.3f} ms")
    # Cleanup KAN GPU memory
    for tname in ["y1", "y2"]:
        kan_models[tname]["model"] = kan_models[tname]["model"].cpu()
    if device != "cpu":
        torch.cuda.empty_cache()

    # DL models
    dl_models_to_time = [
        m for m in ROUTER_MODELS if m not in ("LightGBM", "KAN")
    ]
    mamba_skipped = False
    for model_name in dl_models_to_time:
        if model_name == "Mamba" and device == "cpu":
            logger.warning(f"  Mamba requires CUDA, skipping on CPU")
            expert_timings["Mamba"] = {"median_ms": -1, "note": "requires CUDA"}
            mamba_skipped = True
            continue
        try:
            model = load_dl_model(model_name, device=device)
            model = model.to(device)
            dl_fn = time_dl_single(model, model_name, sample_data, device)
            expert_timings[model_name] = measure_fn(dl_fn)
            expert_single_preds[model_name] = dl_fn()
            logger.info(
                f"  {model_name}: "
                f"{expert_timings[model_name]['median_ms']:.3f} ms"
            )
        except Exception as e:
            logger.error(f"  {model_name} failed: {e}")
            expert_timings[model_name] = {"median_ms": -1, "error": str(e)}
        finally:
            if "model" in dir():
                del model
            if device != "cpu":
                torch.cuda.empty_cache()

    results["per_expert"] = expert_timings

    # ── Stage 3: Routing + affine + projection ──
    logger.info("Timing routing stage...")
    try:
        weights, affine = load_l1_routing_params()
        route_fn = time_routing_stage(expert_single_preds, weights, affine)
        results["routing_stage"] = measure_fn(route_fn)
        logger.info(
            f"  Routing: {results['routing_stage']['median_ms']:.4f} ms"
        )
    except Exception as e:
        logger.warning(f"  Routing timing failed: {e}")
        results["routing_stage"] = {"median_ms": 0.0, "note": str(e)}

    # ── Aggregate pipeline latency ──
    valid_expert_times = [
        v["median_ms"]
        for k, v in expert_timings.items()
        if isinstance(v.get("median_ms"), (int, float)) and v["median_ms"] > 0
    ]
    rep_time = results["rep_construction"]["median_ms"]
    route_time = results.get("routing_stage", {}).get("median_ms", 0.0)

    results["pipeline_summary"] = {
        "rep_construction_ms": rep_time,
        "expert_sequential_ms": sum(valid_expert_times),
        "expert_parallel_ms": max(valid_expert_times) if valid_expert_times else 0,
        "routing_ms": route_time,
        "total_sequential_ms": rep_time + sum(valid_expert_times) + route_time,
        "total_parallel_ms": rep_time + (
            max(valid_expert_times) if valid_expert_times else 0
        ) + route_time,
        "num_experts_timed": len(valid_expert_times),
        "assessment_window_ms": 100.0,
    }

    summary = results["pipeline_summary"]
    logger.info("=" * 60)
    logger.info(f"Pipeline timing summary ({device})")
    logger.info(f"  Rep construction:     {summary['rep_construction_ms']:.3f} ms")
    logger.info(f"  Experts (sequential): {summary['expert_sequential_ms']:.3f} ms")
    logger.info(f"  Experts (parallel):   {summary['expert_parallel_ms']:.3f} ms")
    logger.info(f"  Routing stage:        {summary['routing_ms']:.4f} ms")
    logger.info(f"  ── Total sequential:  {summary['total_sequential_ms']:.3f} ms")
    logger.info(f"  ── Total parallel:    {summary['total_parallel_ms']:.3f} ms")
    logger.info(f"  Assessment window:    {summary['assessment_window_ms']:.0f} ms")
    logger.info("=" * 60)

    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # GPU timing
    if torch.cuda.is_available():
        gpu_results = run_timing("cuda")
        all_results["gpu"] = gpu_results
        with open(OUT_DIR / "pipeline_timing_gpu.json", "w") as f:
            json.dump(gpu_results, f, indent=2, default=str)
        torch.cuda.empty_cache()
    else:
        logger.warning("No CUDA device, skipping GPU timing")

    # CPU timing
    cpu_results = run_timing("cpu")
    all_results["cpu"] = cpu_results
    with open(OUT_DIR / "pipeline_timing_cpu.json", "w") as f:
        json.dump(cpu_results, f, indent=2, default=str)

    # Combined summary CSV
    rows = []
    for mode, res in all_results.items():
        s = res["pipeline_summary"]
        rows.append({
            "mode": mode,
            "rep_ms": f"{s['rep_construction_ms']:.3f}",
            "experts_seq_ms": f"{s['expert_sequential_ms']:.3f}",
            "experts_par_ms": f"{s['expert_parallel_ms']:.3f}",
            "routing_ms": f"{s['routing_ms']:.4f}",
            "total_seq_ms": f"{s['total_sequential_ms']:.3f}",
            "total_par_ms": f"{s['total_parallel_ms']:.3f}",
            "n_experts": s["num_experts_timed"],
        })

    import csv
    csv_path = OUT_DIR / "pipeline_timing_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Summary CSV saved to {csv_path}")

    # Per-expert comparison table
    expert_rows = []
    for mode, res in all_results.items():
        for expert, timing in res.get("per_expert", {}).items():
            med = timing.get("median_ms", -1)
            expert_rows.append({
                "mode": mode,
                "expert": expert,
                "median_ms": f"{med:.3f}" if med > 0 else "N/A",
                "p95_ms": f"{timing.get('p95_ms', -1):.3f}"
                if timing.get("p95_ms", -1) > 0 else "N/A",
            })

    expert_csv = OUT_DIR / "per_expert_timing.csv"
    if expert_rows:
        with open(expert_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=expert_rows[0].keys())
            writer.writeheader()
            writer.writerows(expert_rows)
        logger.info(f"Per-expert CSV saved to {expert_csv}")


if __name__ == "__main__":
    main()
