#!/usr/bin/env python3
"""IEEE300 same-distribution TD-HER routing scalability check.

The rebuilt IEEE300 dataset currently provides train, validation, and test
splits for one operating distribution. This experiment therefore focuses on
large-system same-distribution routing scalability: whether the TD-HER routing
mechanism remains usable and beneficial at IEEE300 scale.

Protocol:
  1. Load the audited IEEE300 expert bank from ``results/ieee300/exp7_rebuild``.
  2. Recreate validation predictions from the saved expert checkpoints.
  3. Use validation labels only to learn target-wise routing weights.
  4. Evaluate the selected route on the held-out IEEE300 test predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_proc.datasets import GraphDataset, TabularDataset, TensorDataset, adjacency_to_edge_index
from models.base_nn import NNTrainer
from models.convlstm_model import ConvLSTMFreqModel
from models.ft_transformer_model import FTTransformerModel
from models.kan_model import KANRegressor
from models.mamba_model import MambaFreqModel
from models.patchtst_model import PatchTSTFreqModel
from models.stgcn_model import STGCNBatchedModel, normalize_adjacency
from models.tabr_model import TabRModel


SEED = 42
MODEL_ORDER = [
    "LightGBM",
    "KAN",
    "ConvLSTM",
    "PatchTST",
    "Mamba",
    "ST-GCN",
    "FT-Transformer",
    "TabR",
]
DL_MODEL_CLASSES = {
    "ConvLSTM": ConvLSTMFreqModel,
    "PatchTST": PatchTSTFreqModel,
    "Mamba": MambaFreqModel,
    "ST-GCN": STGCNBatchedModel,
    "FT-Transformer": FTTransformerModel,
    "TabR": TabRModel,
}
BOOTSTRAP_ROUNDS = 2000


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {}
    for idx, target in enumerate(["y1", "y2"]):
        err = y_pred[:, idx] - y_true[:, idx]
        out[f"{target}_MAE"] = float(np.mean(np.abs(err)))
        out[f"{target}_RMSE"] = float(np.sqrt(np.mean(err ** 2)))
    return out


def target_mae(y_true: np.ndarray, pred: np.ndarray, target_idx: int) -> float:
    return float(np.mean(np.abs(y_true[:, target_idx] - pred[:, target_idx])))


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def load_checkpoint(path: Path, device: str) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def infer_best_batch_size(model_dir: Path, default: int = 256) -> int:
    path = model_dir / "best_params.json"
    if not path.exists():
        return default
    params = load_json(path)
    return int(params.get("bs", default))


def load_data(rep_dir: Path, exp7_dir: Path, ms: int) -> dict:
    logger.info("Loading IEEE300 representations and normalizers")
    split_dir_a = rep_dir / "repA" / f"ms{ms}"
    split_dir_b = rep_dir / "repB" / f"ms{ms}"

    X_train_A = np.load(split_dir_a / "X_train.npy")
    X_val_A = np.load(split_dir_a / "X_val.npy")
    X_test_A = np.load(split_dir_a / "X_test.npy")
    y_train = np.load(split_dir_a / "y_train.npy")
    y_val = np.load(split_dir_a / "y_val.npy")
    y_test = np.load(split_dir_a / "y_test.npy")
    feature_names = load_json(split_dir_a / "feature_names.json")

    mrmr = load_json(exp7_dir / "mrmr_features.json")
    feat_idx = [feature_names.index(f) for f in mrmr["union"]]
    scaler_A = StandardScaler().fit(X_train_A[:, feat_idx])
    X_train_n = scaler_A.transform(X_train_A[:, feat_idx]).astype(np.float32)
    X_val_n = scaler_A.transform(X_val_A[:, feat_idx]).astype(np.float32)
    X_test_n = scaler_A.transform(X_test_A[:, feat_idx]).astype(np.float32)

    Xt_train = np.load(split_dir_b / "X_temporal_train.npy")
    Xt_val = np.load(split_dir_b / "X_temporal_val.npy")
    Xt_test = np.load(split_dir_b / "X_temporal_test.npy")
    Xs_train = np.load(split_dir_b / "X_static_train.npy")
    Xs_val = np.load(split_dir_b / "X_static_val.npy")
    Xs_test = np.load(split_dir_b / "X_static_test.npy")

    b, t, n, c = Xt_train.shape
    temporal_scaler = StandardScaler().fit(Xt_train.reshape(-1, n * c))
    Xt_train_n = temporal_scaler.transform(
        Xt_train.reshape(-1, n * c)
    ).reshape(Xt_train.shape).astype(np.float32)
    Xt_val_n = temporal_scaler.transform(
        Xt_val.reshape(-1, n * c)
    ).reshape(Xt_val.shape).astype(np.float32)
    Xt_test_n = temporal_scaler.transform(
        Xt_test.reshape(-1, n * c)
    ).reshape(Xt_test.shape).astype(np.float32)

    static_scaler = StandardScaler().fit(Xs_train)
    Xs_train_n = static_scaler.transform(Xs_train).astype(np.float32)
    Xs_val_n = static_scaler.transform(Xs_val).astype(np.float32)
    Xs_test_n = static_scaler.transform(Xs_test).astype(np.float32)

    target_scaler = StandardScaler().fit(y_train)
    y_train_n = target_scaler.transform(y_train).astype(np.float32)
    y_val_n = target_scaler.transform(y_val).astype(np.float32)

    return {
        "X_train_A": X_train_A,
        "X_val_A": X_val_A,
        "X_test_A": X_test_A,
        "feature_names": feature_names,
        "mrmr": mrmr,
        "X_train_n": X_train_n,
        "X_val_n": X_val_n,
        "X_test_n": X_test_n,
        "Xt_train_n": Xt_train_n,
        "Xt_val_n": Xt_val_n,
        "Xt_test_n": Xt_test_n,
        "Xs_train_n": Xs_train_n,
        "Xs_val_n": Xs_val_n,
        "Xs_test_n": Xs_test_n,
        "Xn_val_n": np.transpose(Xt_val_n, (0, 2, 1, 3)),
        "Xn_test_n": np.transpose(Xt_test_n, (0, 2, 1, 3)),
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_n": y_train_n,
        "y_val_n": y_val_n,
        "target_scaler": target_scaler,
    }


def make_datasets(data: dict, adj_path: Path) -> dict:
    adj = np.load(adj_path)
    edge_index, edge_weight = adjacency_to_edge_index(adj)

    train_A = TabularDataset(data["X_train_n"], data["y_train_n"])
    val_A = TabularDataset(data["X_val_n"], data["y_val_n"])
    val_B = TensorDataset(data["Xt_val_n"], data["Xs_val_n"], data["y_val_n"])
    val_C = GraphDataset(
        data["Xn_val_n"],
        data["Xs_val_n"],
        data["y_val_n"],
        edge_index,
        edge_weight,
    )
    return {
        "train_A": train_A,
        "val_A": val_A,
        "val_B": val_B,
        "val_C": val_C,
        "adj_norm": normalize_adjacency(adj),
    }


def inverse_one_target(scaler: StandardScaler, pred_scaled: np.ndarray,
                       target_idx: int) -> np.ndarray:
    dummy = np.zeros((len(pred_scaled), 2), dtype=np.float32)
    dummy[:, target_idx] = pred_scaled.reshape(-1)
    return scaler.inverse_transform(dummy)[:, target_idx]


def predict_lightgbm_val(model_dir: Path, data: dict) -> np.ndarray:
    with (model_dir / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    return model.predict(data["X_val_n"])


def predict_kan_val(model_dir: Path, data: dict, device: str) -> np.ndarray:
    preds = np.zeros_like(data["y_val"], dtype=np.float32)
    for target_idx, target in enumerate(["y1", "y2"]):
        ckpt = load_checkpoint(model_dir / f"{target}_model.pth", device)
        model = KANRegressor(**ckpt["model_kwargs"])
        trainer = NNTrainer(
            model,
            device=device,
            batch_size=1024,
            num_workers=4,
            use_amp=device != "cpu",
        )
        trainer.load(str(model_dir / f"{target}_model.pth"))

        feat_names = data["mrmr"][target][:100]
        feat_idx = [
            data["feature_names"].index(f)
            for f in feat_names
            if f in data["feature_names"]
        ]
        scaler = StandardScaler().fit(data["X_train_A"][:, feat_idx])
        X_val = scaler.transform(data["X_val_A"][:, feat_idx]).astype(np.float32)
        val_ds = TabularDataset(X_val, np.zeros((len(X_val), 1), dtype=np.float32))
        pred_scaled = trainer.predict(val_ds)
        preds[:, target_idx] = inverse_one_target(
            data["target_scaler"], pred_scaled, target_idx
        )
    return preds


def build_dl_model(model_name: str, model_dir: Path, device: str,
                   datasets: dict, data: dict):
    ckpt = load_checkpoint(model_dir / "model.pth", device)
    kwargs = dict(ckpt["model_kwargs"])
    model = DL_MODEL_CLASSES[model_name](**kwargs)
    if model_name == "ST-GCN":
        model.set_adj(datasets["adj_norm"])
    trainer = NNTrainer(
        model,
        device=device,
        batch_size=infer_best_batch_size(model_dir),
        num_workers=4,
        use_amp=device != "cpu",
    )
    trainer.load(str(model_dir / "model.pth"))
    if model_name == "TabR":
        trainer.model.build_training_cache(datasets["train_A"].X, datasets["train_A"].y)
        trainer.model.eval()
    return trainer


def predict_dl_val(model_name: str, model_dir: Path, data: dict,
                   datasets: dict, device: str) -> np.ndarray:
    trainer = build_dl_model(model_name, model_dir, device, datasets, data)
    if model_name in {"FT-Transformer", "TabR"}:
        val_ds = datasets["val_A"]
    elif model_name == "ST-GCN":
        val_ds = datasets["val_C"]
    else:
        val_ds = datasets["val_B"]
    pred_scaled = trainer.predict(val_ds)
    return data["target_scaler"].inverse_transform(pred_scaled)


def get_val_prediction(model_name: str, exp7_dir: Path, out_dir: Path, data: dict,
                       datasets: dict, device: str) -> np.ndarray:
    cache_dir = out_dir / "experts" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "val_preds.npy"
    if cache_path.exists():
        return np.load(cache_path)

    model_dir = exp7_dir / model_name
    logger.info(f"Generating IEEE300 validation predictions for {model_name}")
    if model_name == "LightGBM":
        pred = predict_lightgbm_val(model_dir, data)
    elif model_name == "KAN":
        pred = predict_kan_val(model_dir, data, device)
    else:
        pred = predict_dl_val(model_name, model_dir, data, datasets, device)

    np.save(cache_path, pred)
    with (cache_dir / "val_metrics.json").open("w") as f:
        json.dump(evaluate(data["y_val"], pred), f, indent=2)
    return pred


def get_test_prediction(model_name: str, exp7_dir: Path) -> np.ndarray:
    return np.load(exp7_dir / model_name / "preds.npy")


def fit_restricted_affine(y_cal: np.ndarray,
                          pred_cal: np.ndarray) -> tuple[float, float, dict]:
    y = np.asarray(y_cal, dtype=float).reshape(-1)
    z = np.asarray(pred_cal, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(z)
    y = y[finite]
    z = z[finite]
    if len(y) == 0:
        return 1.0, 0.0, {"success": False, "cal_MAE": float("nan")}
    if np.std(z) < 1e-12:
        b = float(np.median(y))
        return 0.0, b, {"success": True, "cal_MAE": _mae(y, np.full_like(y, b))}

    x = np.column_stack([z, np.ones_like(z)])
    try:
        a0, b0 = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        a0, b0 = 1.0, 0.0
    a0 = float(np.clip(a0, 0.0, 3.0)) if np.isfinite(a0) else 1.0
    b0 = float(b0) if np.isfinite(b0) else float(np.median(y - a0 * z))

    scale = max(float(np.ptp(y)), float(np.std(y)), float(np.mean(np.abs(y))), 1e-6)
    b_bound = 5.0 * scale

    def objective(params):
        a, b = params
        return _mae(y, a * z + b) + 1e-8 * ((a - 1.0) ** 2 + (b / scale) ** 2)

    res = minimize(
        objective,
        x0=np.array([a0, float(np.clip(b0, -b_bound, b_bound))]),
        method="SLSQP",
        bounds=[(0.0, 3.0), (-b_bound, b_bound)],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if res.success and np.all(np.isfinite(res.x)):
        a, b = float(res.x[0]), float(res.x[1])
        success = True
    else:
        a, b = a0, float(np.clip(b0, -b_bound, b_bound))
        success = False
    return a, b, {"success": success, "cal_MAE": _mae(y, a * z + b)}


def optimize_convex_weights(y: np.ndarray, pred_val: np.ndarray) -> tuple[np.ndarray, float]:
    n_models = pred_val.shape[1]

    def objective(w):
        return _mae(y, pred_val @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_models
    x0 = np.full(n_models, 1.0 / n_models)
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not res.success:
        logger.warning(f"Convex routing optimizer warning: {res.message}")
    weights = np.clip(res.x, 0.0, 1.0)
    weights = weights / max(np.sum(weights), 1e-12)
    return weights, objective(weights)


def best_expert_router(y_val: np.ndarray, val_preds: dict[str, np.ndarray],
                       test_preds: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}
    for target_idx, target in enumerate(["y1", "y2"]):
        scores = {
            model: target_mae(y_val, pred, target_idx)
            for model, pred in val_preds.items()
        }
        best_model = min(scores, key=scores.get)
        routed[:, target_idx] = test_preds[best_model][:, target_idx]
        details[target] = {
            "expert": best_model,
            "val_MAE": scores[best_model],
            "all_val_MAE": scores,
        }
    return routed, details


def uniform_blend_router(test_preds: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    models = list(test_preds)
    pred = np.mean(np.stack([test_preds[m] for m in models], axis=0), axis=0)
    return pred, {"models": models, "weights": {m: 1.0 / len(models) for m in models}}


def convex_blend_router(y_val: np.ndarray, val_preds: dict[str, np.ndarray],
                        test_preds: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    models = list(val_preds)
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}
    for target_idx, target in enumerate(["y1", "y2"]):
        p_val = np.stack([val_preds[m][:, target_idx] for m in models], axis=1)
        p_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        weights, val_mae = optimize_convex_weights(y_val[:, target_idx], p_val)
        routed[:, target_idx] = p_test @ weights
        details[target] = {
            "val_MAE": val_mae,
            "weights": {m: float(w) for m, w in zip(models, weights)},
        }
    return routed, details


def affine_convex_blend_router(y_val: np.ndarray, val_preds: dict[str, np.ndarray],
                               test_preds: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    models = list(val_preds)
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}
    for target_idx, target in enumerate(["y1", "y2"]):
        p_val = np.stack([val_preds[m][:, target_idx] for m in models], axis=1)
        p_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        weights, blend_val_mae = optimize_convex_weights(y_val[:, target_idx], p_val)
        val_blend = p_val @ weights
        test_blend = p_test @ weights
        slope, bias, info = fit_restricted_affine(y_val[:, target_idx], val_blend)
        routed[:, target_idx] = slope * test_blend + bias
        details[target] = {
            "blend_val_MAE": blend_val_mae,
            "affine_val_MAE": info["cal_MAE"],
            "slope": slope,
            "bias": bias,
            "success": info["success"],
            "weights": {m: float(w) for m, w in zip(models, weights)},
        }
    return routed, details


def enforce_y2_nonnegative(pred: np.ndarray) -> np.ndarray:
    out = np.array(pred, copy=True)
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    return out


def paired_bootstrap_delta(y_true: np.ndarray, candidate_pred: np.ndarray,
                           reference_pred: np.ndarray,
                           n_rounds: int = BOOTSTRAP_ROUNDS,
                           seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = {}
    for target_idx, target in enumerate(["y1", "y2"]):
        candidate_err = np.abs(y_true[:, target_idx] - candidate_pred[:, target_idx])
        reference_err = np.abs(y_true[:, target_idx] - reference_pred[:, target_idx])
        diff = candidate_err - reference_err
        observed = float(np.mean(diff))
        boot = np.empty(n_rounds, dtype=float)
        for i in range(n_rounds):
            idx = rng.integers(0, n, size=n)
            boot[i] = float(np.mean(diff[idx]))
        low, high = np.percentile(boot, [2.5, 97.5])
        p_two_sided = 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))
        ref_mae = float(np.mean(reference_err))
        out[target] = {
            "delta_MAE_candidate_minus_reference": observed,
            "reference_MAE": ref_mae,
            "candidate_MAE": float(np.mean(candidate_err)),
            "relative_improvement": float(-observed / ref_mae) if ref_mae > 0 else float("nan"),
            "ci95": [float(low), float(high)],
            "p_two_sided_bootstrap": float(min(p_two_sided, 1.0)),
            "n_rounds": n_rounds,
        }
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(data: dict, val_preds: dict[str, np.ndarray],
                  test_preds: dict[str, np.ndarray]) -> tuple[dict, dict[str, np.ndarray]]:
    y_val = data["y_val"]
    y_test = data["y_test"]

    expert_metrics = {}
    for model in MODEL_ORDER:
        expert_metrics[model] = {
            "validation": evaluate(y_val, val_preds[model]),
            "test": evaluate(y_test, test_preds[model]),
        }

    best_pred, best_details = best_expert_router(y_val, val_preds, test_preds)
    uniform_pred, uniform_details = uniform_blend_router(test_preds)
    convex_pred, convex_details = convex_blend_router(y_val, val_preds, test_preds)
    affine_pred, affine_details = affine_convex_blend_router(y_val, val_preds, test_preds)
    affine_nonnegative_pred = enforce_y2_nonnegative(affine_pred)
    route_preds = {
        "best_expert": best_pred,
        "uniform_blend": uniform_pred,
        "convex_blend": convex_pred,
        "affine_convex_blend": affine_pred,
        "affine_convex_blend_nonnegative_y2": affine_nonnegative_pred,
    }

    results = {
        "protocol": {
            "name": "IEEE300_same_distribution_routing_scalability",
            "calibration_split": "validation",
            "test_split": "test",
            "claims_allowed": [
                "large-system same-distribution routing scalability",
                "non-degrading target-wise heterogeneous expert routing",
                "engineering relevance of early post-disturbance frequency extremum and arrival-time prediction",
            ],
            "extension_scope": [
                "larger IEEE300 datasets can strengthen this scalability evidence",
                "future data expansion can further expose target-dependent expert complementarity",
            ],
        },
        "experts": expert_metrics,
        "best_expert": {
            "metrics": evaluate(y_test, best_pred),
            "details": best_details,
        },
        "uniform_blend": {
            "metrics": evaluate(y_test, uniform_pred),
            "details": uniform_details,
        },
        "convex_blend": {
            "metrics": evaluate(y_test, convex_pred),
            "details": convex_details,
        },
        "affine_convex_blend": {
            "metrics": evaluate(y_test, affine_pred),
            "details": affine_details,
        },
        "affine_convex_blend_nonnegative_y2": {
            "metrics": evaluate(y_test, affine_nonnegative_pred),
            "details": {
                "source": "affine_convex_blend",
                "projection": "y2 = max(y2, 0)",
                "source_details": affine_details,
            },
        },
        "paired_bootstrap": {
            "affine_nonnegative_vs_best_expert": paired_bootstrap_delta(
                y_test, affine_nonnegative_pred, best_pred
            ),
            "affine_nonnegative_vs_convex": paired_bootstrap_delta(
                y_test, affine_nonnegative_pred, convex_pred
            ),
        },
    }
    return results, route_preds


def export_tables(summary: dict, out_dir: Path, paper_table_dir: Path) -> None:
    rows = []
    method_labels = {
        "best_expert": "Target-wise validation-best expert",
        "uniform_blend": "Uniform expert average",
        "convex_blend": "TD-HER convex routing",
        "affine_convex_blend": "TD-HER convex routing + affine",
        "affine_convex_blend_nonnegative_y2": "TD-HER physical routing",
    }
    best_metrics = summary["best_expert"]["metrics"]
    for order, (method, label) in enumerate(method_labels.items(), start=1):
        metrics = summary[method]["metrics"]
        rows.append({
            "order": order,
            "method": method,
            "label": label,
            "y1_mae": metrics["y1_MAE"],
            "y1_rmse": metrics["y1_RMSE"],
            "y2_mae": metrics["y2_MAE"],
            "y2_rmse": metrics["y2_RMSE"],
            "delta_y1_mae_vs_best_expert": metrics["y1_MAE"] - best_metrics["y1_MAE"],
            "delta_y2_mae_vs_best_expert": metrics["y2_MAE"] - best_metrics["y2_MAE"],
        })

    fieldnames = [
        "order",
        "method",
        "label",
        "y1_mae",
        "y1_rmse",
        "y2_mae",
        "y2_rmse",
        "delta_y1_mae_vs_best_expert",
        "delta_y2_mae_vs_best_expert",
    ]
    write_csv(out_dir / "ieee300_tdher_routing.csv", rows, fieldnames)
    write_csv(paper_table_dir / "ieee300_tdher_routing.csv", rows, fieldnames)

    weight_rows = []
    details = summary["affine_convex_blend"]["details"]
    for target in ["y1", "y2"]:
        target_details = details[target]
        for expert, weight in target_details["weights"].items():
            weight_rows.append({
                "target": target,
                "expert": expert,
                "weight": weight,
                "affine_slope": target_details["slope"],
                "affine_bias": target_details["bias"],
                "blend_val_mae": target_details["blend_val_MAE"],
                "affine_val_mae": target_details["affine_val_MAE"],
            })
    weight_fields = [
        "target",
        "expert",
        "weight",
        "affine_slope",
        "affine_bias",
        "blend_val_mae",
        "affine_val_mae",
    ]
    write_csv(out_dir / "ieee300_tdher_routing_weights.csv", weight_rows, weight_fields)
    write_csv(
        paper_table_dir / "ieee300_tdher_routing_weights.csv",
        weight_rows,
        weight_fields,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rep-dir", default="data/ieee300_v2_posttrigger")
    parser.add_argument("--adj-path", default="data/ieee300_v2/adjacency/adjacency.npy")
    parser.add_argument("--exp7-dir", default="results/ieee300/exp7_rebuild")
    parser.add_argument("--out-dir", default="results/ieee300/tdher_router")
    parser.add_argument("--paper-table-dir", default="results/paper_tables")
    parser.add_argument("--ms", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rep_dir = Path(args.rep_dir)
    exp7_dir = Path(args.exp7_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(rep_dir, exp7_dir, args.ms)
    datasets = make_datasets(data, Path(args.adj_path))

    val_preds = {}
    test_preds = {}
    for model in MODEL_ORDER:
        val_preds[model] = get_val_prediction(
            model, exp7_dir, out_dir, data, datasets, args.device
        )
        test_preds[model] = get_test_prediction(model, exp7_dir)
        logger.info(
            f"{model}: val y1={target_mae(data['y_val'], val_preds[model], 0):.6f}, "
            f"val y2={target_mae(data['y_val'], val_preds[model], 1):.6f}, "
            f"test y1={target_mae(data['y_test'], test_preds[model], 0):.6f}, "
            f"test y2={target_mae(data['y_test'], test_preds[model], 1):.6f}"
        )

    summary, route_preds = build_summary(data, val_preds, test_preds)
    with (out_dir / "metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    for method, pred in route_preds.items():
        np.save(out_dir / f"{method}_preds.npy", pred)

    export_tables(summary, out_dir, Path(args.paper_table_dir))

    logger.info("IEEE300 TD-HER routing results:")
    for method in [
        "best_expert",
        "uniform_blend",
        "convex_blend",
        "affine_convex_blend",
        "affine_convex_blend_nonnegative_y2",
    ]:
        m = summary[method]["metrics"]
        logger.info(f"{method}: y1_MAE={m['y1_MAE']:.6f}, y2_MAE={m['y2_MAE']:.6f}")
    logger.info(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
