#!/usr/bin/env python3
"""Multi-window TD-HER routing on the IEEE 39-bus L1 protocol.

The original window-sensitivity experiment reports single-expert test metrics
for several early observation windows, but it does not save validation
predictions for all neural experts. TD-HER is a calibration-stage router, so
this script builds the missing validation/test prediction cache and then fits
target-wise routes independently at each window length.

Outputs are written under:
  - results/ieee39/exp_tdh_multi_window/
  - results/paper_tables/tdher_multiwindow_l1.csv
  - results/paper_tables/tdher_multiwindow_weights.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_proc.datasets import GraphDataset, TensorDataset, adjacency_to_edge_index  # noqa: E402
from models.base_nn import NNTrainer  # noqa: E402


SEED = 42
REP_DIR = "data/ieee39_v8_80_10_10"
ADJ_PATH = "data/ieee39_v8/adjacency/adjacency.npy"
EXP1_DIR = "results/ieee39/exp1"
EXP2_DIR = "results/ieee39/exp2"
MS_VALUES = [1, 5, 10, 15, 25]
MODELS = ["LightGBM", "ConvLSTM", "PatchTST", "Mamba", "ST-GCN"]
OUT_DIR = Path("results/ieee39/exp_tdh_multi_window")
PAPER_TABLE_DIR = Path("results/paper_tables")
PAPER_FIG_DIR = Path("results/paper_artifacts/figures")
STEP_MS = 10
ROUTER_ORDER = [
    "best_expert",
    "convex_blend",
    "shared_convex_blend",
    "affine_convex_blend",
    "shared_affine_convex_blend",
    "tdher_physical",
]
CACHE_PROTOCOL = "multiwindow_v2_reset_seed_per_model"


class FixedTrial:
    """Minimal Optuna-trial shim for model builders with fixed parameters."""

    def __init__(self, fixed_params: dict):
        self._params = dict(fixed_params)
        self.params = dict(fixed_params)

    def suggest_int(self, name, low, high, **kwargs):
        del low, high, kwargs
        return self._params[name]

    def suggest_float(self, name, low, high, **kwargs):
        del low, high, kwargs
        return self._params[name]

    def suggest_categorical(self, name, choices, **kwargs):
        del choices, kwargs
        return self._params[name]


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _safe_dump_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def seed_everything() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def clear_stale_files(directory: Path, names: list[str]) -> None:
    for name in names:
        path = directory / name
        if path.exists():
            path.unlink()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the same metrics used by the audited Exp1/Exp2 scripts."""
    metrics = {}
    for i, name in enumerate(["y1", "y2"]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        abs_err = np.abs(yt - yp)
        metrics[f"{name}_MAE"] = float(np.mean(abs_err))
        metrics[f"{name}_RMSE"] = float(np.sqrt(np.mean((yt - yp) ** 2)))
        denom = np.maximum(np.abs(yt), 1e-8)
        metrics[f"{name}_MAPE"] = float(np.mean(abs_err / denom) * 100.0)
        smape_denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, 1e-8)
        metrics[f"{name}_SMAPE"] = float(np.mean(abs_err / smape_denom) * 100.0)
    return metrics


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def target_mae(y_true: np.ndarray, pred: np.ndarray, target_idx: int) -> float:
    return _mae(y_true[:, target_idx], pred[:, target_idx])


def optimize_convex_weights(y: np.ndarray, p_val: np.ndarray) -> tuple[np.ndarray, float]:
    from scipy.optimize import minimize

    n_models = p_val.shape[1]

    def objective(w):
        return _mae(y, p_val @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_models
    x0 = np.full(n_models, 1.0 / n_models)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success:
        logger.warning(f"convex routing optimization warning: {res.message}")
    w = np.clip(res.x, 0.0, 1.0)
    w = w / max(np.sum(w), 1e-12)
    return w, objective(w)


def fit_restricted_affine(y_cal: np.ndarray,
                          pred_cal: np.ndarray) -> tuple[float, float, dict]:
    from scipy.optimize import minimize

    y = np.asarray(y_cal, dtype=float).reshape(-1)
    z = np.asarray(pred_cal, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(z)
    y = y[finite]
    z = z[finite]

    if len(y) == 0:
        return 1.0, 0.0, {"success": False, "message": "no finite samples",
                          "cal_MAE": float("nan")}
    if np.std(z) < 1e-12:
        b = float(np.median(y))
        return 0.0, b, {"success": True, "message": "constant predictor fallback",
                        "cal_MAE": _mae(y, np.full_like(y, b))}

    X = np.column_stack([z, np.ones_like(z)])
    try:
        a0, b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        a0, b0 = 1.0, 0.0

    a0 = float(np.clip(a0, 0.0, 3.0)) if np.isfinite(a0) else 1.0
    if not np.isfinite(b0):
        b0 = float(np.median(y - a0 * z))

    scale = max(
        float(np.ptp(y)),
        float(np.std(y)),
        float(np.mean(np.abs(y))),
        float(np.mean(np.abs(z))),
        1e-6,
    )
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
        message = str(res.message)
    else:
        a, b = a0, float(np.clip(b0, -b_bound, b_bound))
        success = False
        message = str(res.message)

    return a, b, {
        "success": success,
        "message": message,
        "cal_MAE": _mae(y, a * z + b),
        "slope": a,
        "bias": b,
    }


def apply_affine(pred: np.ndarray, slope: float, bias: float) -> np.ndarray:
    return slope * pred + bias


def enforce_y2_nonnegative(pred: np.ndarray) -> np.ndarray:
    out = np.array(pred, copy=True)
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    return out


def paired_bootstrap_delta(y_true: np.ndarray, candidate_pred: np.ndarray,
                           reference_pred: np.ndarray,
                           n_rounds: int = 2000,
                           seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    out = {}
    for target_idx, target_name in enumerate(["y1", "y2"]):
        cand_err = np.abs(y_true[:, target_idx] - candidate_pred[:, target_idx])
        ref_err = np.abs(y_true[:, target_idx] - reference_pred[:, target_idx])
        diff = cand_err - ref_err
        observed = float(np.mean(diff))
        boot = np.empty(n_rounds, dtype=float)
        for i in range(n_rounds):
            idx = rng.integers(0, n, size=n)
            boot[i] = float(np.mean(diff[idx]))
        lower, upper = np.percentile(boot, [2.5, 97.5])
        p_two_sided = 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))
        ref_mae = float(np.mean(ref_err))
        out[target_name] = {
            "delta_MAE_candidate_minus_reference": observed,
            "reference_MAE": ref_mae,
            "candidate_MAE": float(np.mean(cand_err)),
            "relative_improvement": float(-observed / ref_mae) if ref_mae > 0 else float("nan"),
            "ci95": [float(lower), float(upper)],
            "p_two_sided_bootstrap": float(min(p_two_sided, 1.0)),
            "n_rounds": n_rounds,
        }
    return out


def best_expert_router(y_val: np.ndarray, val_preds: dict,
                       test_preds: dict) -> tuple[np.ndarray, dict]:
    selected = {}
    routed = np.zeros_like(next(iter(test_preds.values())))
    for target_idx, target_name in enumerate(["y1", "y2"]):
        scores = {
            model: target_mae(y_val, pred, target_idx)
            for model, pred in val_preds.items()
        }
        best_model = min(scores, key=scores.get)
        selected[target_name] = {
            "expert": best_model,
            "val_MAE": scores[best_model],
            "all_val_MAE": scores,
        }
        routed[:, target_idx] = test_preds[best_model][:, target_idx]
    return routed, selected


def convex_blend_router(y_val: np.ndarray, val_preds: dict,
                        test_preds: dict) -> tuple[np.ndarray, dict]:
    models = list(val_preds)
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}
    for target_idx, target_name in enumerate(["y1", "y2"]):
        p_val = np.stack([val_preds[m][:, target_idx] for m in models], axis=1)
        p_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        w, val_mae = optimize_convex_weights(y_val[:, target_idx], p_val)
        routed[:, target_idx] = p_test @ w
        details[target_name] = {
            "val_MAE": val_mae,
            "weights": {m: float(wi) for m, wi in zip(models, w)},
        }
    return routed, details


def _target_balance_scales(y_cal: np.ndarray, cal_preds: dict,
                           models: list[str]) -> np.ndarray:
    scales = []
    for target_idx in range(2):
        y = y_cal[:, target_idx]
        best = min(_mae(y, cal_preds[m][:, target_idx]) for m in models)
        fallback = max(float(np.std(y)), float(np.mean(np.abs(y))), 1e-9)
        scales.append(best if best > 1e-12 else fallback)
    return np.asarray(scales, dtype=float)


def optimize_shared_convex_weights(y_cal: np.ndarray,
                                   p_cal: np.ndarray,
                                   scales: np.ndarray) -> tuple[np.ndarray, float, dict]:
    from scipy.optimize import minimize

    n_models = p_cal.shape[1]

    def per_target_mae(w):
        return np.asarray([
            _mae(y_cal[:, target_idx], p_cal[:, :, target_idx] @ w)
            for target_idx in range(2)
        ])

    def objective(w):
        return float(np.mean(per_target_mae(w) / scales))

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_models
    x0 = np.full(n_models, 1.0 / n_models)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success:
        logger.warning(f"shared routing optimization warning: {res.message}")
    w = np.clip(res.x, 0.0, 1.0)
    w = w / max(np.sum(w), 1e-12)
    maes = per_target_mae(w)
    return w, objective(w), {
        "balanced_objective": objective(w),
        "target_balance_scales": {"y1": float(scales[0]), "y2": float(scales[1])},
        "cal_MAE": {"y1": float(maes[0]), "y2": float(maes[1])},
    }


def shared_convex_blend_outputs(y_cal: np.ndarray, cal_preds: dict,
                                test_preds: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    models = list(cal_preds)
    p_cal = np.stack([cal_preds[m] for m in models], axis=1)
    p_test = np.stack([test_preds[m] for m in models], axis=1)
    scales = _target_balance_scales(y_cal, cal_preds, models)
    w, objective_value, info = optimize_shared_convex_weights(y_cal, p_cal, scales)
    cal_blend = np.zeros_like(next(iter(cal_preds.values())))
    test_blend = np.zeros_like(next(iter(test_preds.values())))
    for target_idx in range(2):
        cal_blend[:, target_idx] = p_cal[:, :, target_idx] @ w
        test_blend[:, target_idx] = p_test[:, :, target_idx] @ w
    return cal_blend, test_blend, {
        "routing": "shared_nonnegative_convex",
        "objective": "mean target-balanced calibration MAE",
        "balanced_objective": float(objective_value),
        "weights": {m: float(wi) for m, wi in zip(models, w)},
        **info,
    }


def shared_convex_blend_router(y_cal: np.ndarray, cal_preds: dict,
                               test_preds: dict) -> tuple[np.ndarray, dict]:
    _, test_blend, details = shared_convex_blend_outputs(y_cal, cal_preds, test_preds)
    return test_blend, details


def affine_convex_blend_router(y_cal: np.ndarray, cal_preds: dict,
                               test_preds: dict) -> tuple[np.ndarray, dict]:
    models = list(cal_preds)
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}
    for target_idx, target_name in enumerate(["y1", "y2"]):
        p_cal = np.stack([cal_preds[m][:, target_idx] for m in models], axis=1)
        p_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        w, blend_cal_mae = optimize_convex_weights(y_cal[:, target_idx], p_cal)
        cal_blend = p_cal @ w
        test_blend = p_test @ w
        a, b, info = fit_restricted_affine(y_cal[:, target_idx], cal_blend)
        routed[:, target_idx] = apply_affine(test_blend, a, b)
        details[target_name] = {
            "blend_cal_MAE": blend_cal_mae,
            "affine_cal_MAE": info["cal_MAE"],
            "slope": a,
            "bias": b,
            "success": info["success"],
            "weights": {m: float(wi) for m, wi in zip(models, w)},
        }
    return routed, details


def shared_affine_convex_blend_router(y_cal: np.ndarray, cal_preds: dict,
                                      test_preds: dict) -> tuple[np.ndarray, dict]:
    cal_blend, test_blend, shared_details = shared_convex_blend_outputs(
        y_cal, cal_preds, test_preds)
    routed = np.zeros_like(test_blend)
    details = {"shared_route": shared_details, "y1": {}, "y2": {}}
    for target_idx, target_name in enumerate(["y1", "y2"]):
        a, b, info = fit_restricted_affine(
            y_cal[:, target_idx], cal_blend[:, target_idx])
        routed[:, target_idx] = apply_affine(test_blend[:, target_idx], a, b)
        details[target_name] = {
            "affine_cal_MAE": info["cal_MAE"],
            "slope": a,
            "bias": b,
            "success": info["success"],
        }
    return routed, details


def load_exp1_params() -> dict:
    """Load fixed hyperparameters from audited Exp1 artifacts."""
    import optuna

    params = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for target in ["y1", "y2"]:
        db = Path(EXP1_DIR) / "LightGBM" / f"optuna_{target}.db"
        studies = optuna.get_all_study_names(f"sqlite:///{db}")
        study = optuna.load_study(study_name=studies[0],
                                  storage=f"sqlite:///{db}")
        params[f"LightGBM_{target}"] = study.best_params

    for model_name in ["ConvLSTM", "PatchTST", "Mamba", "ST-GCN"]:
        with open(Path(EXP1_DIR) / model_name / "best_params.json") as f:
            params[model_name] = json.load(f)
    return params


def _load_feature_union(ms: int, kind: str) -> list[str]:
    if kind == "repA":
        candidates = [
            Path(EXP2_DIR) / "mrmr_cache" / f"mrmr_repA_ms{ms}.json",
        ]
        if ms == 10:
            candidates.append(Path(EXP1_DIR) / "mrmr_features.json")
    elif kind == "static":
        candidates = [
            Path(EXP2_DIR) / "mrmr_cache" / f"mrmr_static_ms{ms}.json",
        ]
        if ms == 10:
            candidates.append(Path(EXP1_DIR) / "mrmr_static_features.json")
    else:
        raise ValueError(kind)

    for path in candidates:
        if path.exists():
            with open(path) as f:
                return json.load(f)["union"]
    raise FileNotFoundError(
        f"Missing cached mRMR feature file for ms={ms}, kind={kind}. "
        "Run the original Exp2 mRMR step or provide the cache before running "
        "multi-window TD-HER."
    )


def load_data_for_ms(ms: int) -> dict:
    """Load normalized IEEE 39 representations for one observation window."""
    logger.info(f"Loading data for ms={ms}...")

    rep_a = Path(REP_DIR) / "repA" / f"ms{ms}"
    rep_b = Path(REP_DIR) / "repB" / f"ms{ms}"
    rep_c = Path(REP_DIR) / "repC" / f"ms{ms}"

    X_train_A = np.load(rep_a / "X_train.npy")
    X_val_A = np.load(rep_a / "X_val.npy")
    X_test_A = np.load(rep_a / "X_test.npy")
    y_train = np.load(rep_a / "y_train.npy")
    y_val = np.load(rep_a / "y_val.npy")
    y_test = np.load(rep_a / "y_test.npy")
    with open(rep_a / "feature_names.json") as f:
        feature_names = json.load(f)

    Xt_train = np.load(rep_b / "X_temporal_train.npy")
    Xs_train = np.load(rep_b / "X_static_train.npy")
    Xt_val = np.load(rep_b / "X_temporal_val.npy")
    Xs_val = np.load(rep_b / "X_static_val.npy")
    Xt_test = np.load(rep_b / "X_temporal_test.npy")
    Xs_test = np.load(rep_b / "X_static_test.npy")
    with open(rep_b / "meta.json") as f:
        meta = json.load(f)
    static_names = meta["static_names"]

    selected_features = _load_feature_union(ms, "repA")
    feat_idx = [feature_names.index(name) for name in selected_features]
    scaler_A = StandardScaler().fit(X_train_A[:, feat_idx])
    X_train_n = scaler_A.transform(X_train_A[:, feat_idx]).astype(np.float32)
    X_val_n = scaler_A.transform(X_val_A[:, feat_idx]).astype(np.float32)
    X_test_n = scaler_A.transform(X_test_A[:, feat_idx]).astype(np.float32)

    static_selected = _load_feature_union(ms, "static")
    static_idx = [static_names.index(name) for name in static_selected]
    Xs_train_sel = Xs_train[:, static_idx]
    Xs_val_sel = Xs_val[:, static_idx]
    Xs_test_sel = Xs_test[:, static_idx]

    B, T, N, C = Xt_train.shape
    sc_t = StandardScaler().fit(Xt_train.reshape(-1, N * C))
    Xt_train_n = sc_t.transform(Xt_train.reshape(-1, N * C)).reshape(B, T, N, C).astype(np.float32)
    Xt_val_n = sc_t.transform(Xt_val.reshape(-1, N * C)).reshape(Xt_val.shape).astype(np.float32)
    Xt_test_n = sc_t.transform(Xt_test.reshape(-1, N * C)).reshape(Xt_test.shape).astype(np.float32)

    sc_s = StandardScaler().fit(Xs_train_sel)
    Xs_train_n = sc_s.transform(Xs_train_sel).astype(np.float32)
    Xs_val_n = sc_s.transform(Xs_val_sel).astype(np.float32)
    Xs_test_n = sc_s.transform(Xs_test_sel).astype(np.float32)

    Xn_train_n = np.transpose(Xt_train_n, (0, 2, 1, 3))
    Xn_val_n = np.transpose(Xt_val_n, (0, 2, 1, 3))
    Xn_test_n = np.transpose(Xt_test_n, (0, 2, 1, 3))

    scaler_y = StandardScaler().fit(y_train)
    y_train_n = scaler_y.transform(y_train).astype(np.float32)
    y_val_n = scaler_y.transform(y_val).astype(np.float32)

    return {
        "X_train_n": X_train_n,
        "X_val_n": X_val_n,
        "X_test_n": X_test_n,
        "Xt_train_n": Xt_train_n,
        "Xt_val_n": Xt_val_n,
        "Xt_test_n": Xt_test_n,
        "Xs_train_n": Xs_train_n,
        "Xs_val_n": Xs_val_n,
        "Xs_test_n": Xs_test_n,
        "Xn_train_n": Xn_train_n,
        "Xn_val_n": Xn_val_n,
        "Xn_test_n": Xn_test_n,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_n": y_train_n,
        "y_val_n": y_val_n,
        "scaler_y": scaler_y,
        "n_timesteps": T,
        "n_generators": N,
        "n_features": C,
        "n_static": len(static_idx),
    }


def build_model(name: str, data: dict, params: dict):
    p = dict(params[name])
    if name == "PatchTST":
        patch_len = p.get("patch_len", 3)
        if data["n_timesteps"] < patch_len:
            p["patch_len"] = max(1, data["n_timesteps"])
        if p.get("stride", 1) > p["patch_len"]:
            p["stride"] = max(1, p["patch_len"])

    trial = FixedTrial(p)
    if name == "ConvLSTM":
        from models.convlstm_model import build_convlstm_from_trial

        return build_convlstm_from_trial(
            trial,
            n_features=data["n_features"],
            n_generators=data["n_generators"],
            n_static=data["n_static"],
        )
    if name == "PatchTST":
        from models.patchtst_model import build_patchtst_from_trial

        return build_patchtst_from_trial(
            trial,
            n_features=data["n_features"],
            n_generators=data["n_generators"],
            n_static=data["n_static"],
            n_timesteps=data["n_timesteps"],
        )
    if name == "Mamba":
        from models.mamba_model import build_mamba_from_trial

        return build_mamba_from_trial(
            trial,
            n_features=data["n_features"],
            n_generators=data["n_generators"],
            n_static=data["n_static"],
            n_timesteps=data["n_timesteps"],
        )
    if name == "ST-GCN":
        from models.stgcn_model import build_stgcn_from_trial, normalize_adjacency

        adj = np.load(ADJ_PATH)
        adj_norm = normalize_adjacency(adj)
        return build_stgcn_from_trial(
            trial,
            n_features=data["n_features"],
            n_nodes=data["n_generators"],
            n_static=data["n_static"],
            adj_norm=adj_norm,
        )
    raise ValueError(f"Unsupported neural model: {name}")


def make_datasets(name: str, data: dict):
    if name in {"ConvLSTM", "PatchTST", "Mamba"}:
        train_ds = TensorDataset(
            data["Xt_train_n"], data["Xs_train_n"], data["y_train_n"])
        val_ds = TensorDataset(
            data["Xt_val_n"], data["Xs_val_n"], data["y_val_n"])
        test_ds = TensorDataset(
            data["Xt_test_n"], data["Xs_test_n"],
            np.zeros_like(data["y_test"], dtype=np.float32))
        return train_ds, val_ds, test_ds

    if name == "ST-GCN":
        adj = np.load(ADJ_PATH)
        edge_index, edge_weight = adjacency_to_edge_index(adj)
        train_ds = GraphDataset(
            data["Xn_train_n"], data["Xs_train_n"], data["y_train_n"],
            edge_index, edge_weight)
        val_ds = GraphDataset(
            data["Xn_val_n"], data["Xs_val_n"], data["y_val_n"],
            edge_index, edge_weight)
        test_ds = GraphDataset(
            data["Xn_test_n"], data["Xs_test_n"],
            np.zeros_like(data["y_test"], dtype=np.float32),
            edge_index, edge_weight)
        return train_ds, val_ds, test_ds

    raise ValueError(f"Unsupported neural model: {name}")


def predict_lightgbm_for_window(ms: int, data: dict, params: dict,
                                out_dir: Path, force_train: bool) -> dict:
    pred_dir = out_dir / f"ms{ms}" / "LightGBM"
    val_path = pred_dir / "val_preds.npy"
    test_path = pred_dir / "test_preds.npy"
    metrics_path = pred_dir / "metrics.json"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not force_train and val_path.exists() and test_path.exists() and metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        if metrics.get("cache_protocol") == CACHE_PROTOCOL:
            return {
                "val": np.load(val_path),
                "test": np.load(test_path),
                "metrics": metrics,
            }

    from models.lightgbm_model import LightGBMModel

    clear_stale_files(
        pred_dir,
        [
            "val_preds.npy",
            "test_preds.npy",
            "metrics.json",
            "y1_lgb_model.pkl",
            "y2_lgb_model.pkl",
        ],
    )
    seed_everything()
    logger.info(f"Training LightGBM for ms={ms} to build TD-HER cache...")
    model = LightGBMModel(n_trials=0, seed=SEED)
    model.best_params = {
        "y1": params["LightGBM_y1"],
        "y2": params["LightGBM_y2"],
    }
    t0 = time.time()
    model.fit(
        data["X_train_n"],
        data["y_train"],
        data["X_val_n"],
        data["y_val"],
        checkpoint_dir=str(pred_dir),
        skip_optuna=True,
    )
    train_time = time.time() - t0

    val_pred = model.predict(data["X_val_n"])
    test_pred = model.predict(data["X_test_n"])
    metrics = {
        "validation": evaluate(data["y_val"], val_pred),
        "test": evaluate(data["y_test"], test_pred),
        "train_time_s": train_time,
        "source": "trained_for_multiwindow",
        "seed": SEED,
        "cache_protocol": CACHE_PROTOCOL,
    }

    np.save(val_path, val_pred)
    np.save(test_path, test_pred)
    _safe_dump_json(metrics_path, metrics)
    return {"val": val_pred, "test": test_pred, "metrics": metrics}


def predict_from_ms10_tdher_cache(model_name: str, data: dict,
                                  out_dir: Path) -> dict | None:
    """Reuse audited ms10 expert predictions from the main TD-HER cache."""
    source_dir = Path("results/ieee39/exp_tdh_router/experts") / model_name
    val_src = source_dir / "val_preds.npy"
    test_src = source_dir / "test_preds.npy"
    if not (val_src.exists() and test_src.exists()):
        return None

    val_pred = np.load(val_src)
    test_pred = np.load(test_src)
    pred_dir = out_dir / "ms10" / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "val_preds.npy", val_pred)
    np.save(pred_dir / "test_preds.npy", test_pred)
    metrics = {
        "validation": evaluate(data["y_val"], val_pred),
        "test": evaluate(data["y_test"], test_pred),
        "source": str(source_dir),
        "audit_note": "reused audited main TD-HER ms10 expert cache",
        "cache_protocol": "audited_main_tdher_ms10_cache",
    }
    _safe_dump_json(pred_dir / "metrics.json", metrics)
    return {"val": val_pred, "test": test_pred, "metrics": metrics}


def _load_ms10_exp1_state(name: str):
    model_path = Path(EXP1_DIR) / name / "model.pth"
    if not model_path.exists():
        return None
    return torch.load(model_path, map_location="cpu", weights_only=False)


def _load_existing_neural_state(name: str, ms: int, pred_dir: Path):
    local_path = pred_dir / "model_state.pth"
    if local_path.exists():
        return torch.load(local_path, map_location="cpu", weights_only=False)
    if ms == 10:
        return _load_ms10_exp1_state(name)
    return None


def predict_neural_for_window(name: str, ms: int, data: dict, params: dict,
                              out_dir: Path, device: str,
                              use_amp: bool, force_train: bool) -> dict:
    pred_dir = out_dir / f"ms{ms}" / name
    val_path = pred_dir / "val_preds.npy"
    test_path = pred_dir / "test_preds.npy"
    metrics_path = pred_dir / "metrics.json"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if (not force_train and val_path.exists() and test_path.exists()
            and metrics_path.exists()):
        with open(metrics_path) as f:
            metrics = json.load(f)
        if metrics.get("cache_protocol") == CACHE_PROTOCOL:
            return {
                "val": np.load(val_path),
                "test": np.load(test_path),
                "metrics": metrics,
            }

    clear_stale_files(
        pred_dir,
        [
            "val_preds.npy",
            "test_preds.npy",
            "metrics.json",
            "history.json",
            "model_state.pth",
            "checkpoint.pth",
            "best_model.pth",
        ],
    )
    seed_everything()
    model, model_kwargs = build_model(name, data, params)
    state_obj = None if ms != 10 or force_train else _load_existing_neural_state(name, ms, pred_dir)
    train_ds, val_ds, test_ds = make_datasets(name, data)
    history = None
    train_time = 0.0
    source = "trained_for_multiwindow"

    if state_obj is not None:
        state_dict = state_obj.get("model_state_dict", state_obj)
        model.load_state_dict(state_dict)
        source = f"loaded:{Path(EXP1_DIR) / name / 'model.pth'}" if ms == 10 else f"loaded:{pred_dir / 'model_state.pth'}"
        trainer = NNTrainer(
            model, device=device,
            lr=params[name]["lr"],
            weight_decay=params[name]["weight_decay"],
            batch_size=params[name]["bs"],
            y1_weight=params[name]["y1_weight"],
            y2_weight=params[name]["y2_weight"],
            max_epochs=1,
            patience=1,
            use_amp=use_amp,
            warmup_epochs=1,
        )
    else:
        logger.info(f"Training {name} for ms={ms} to build TD-HER cache...")
        trainer = NNTrainer(
            model, device=device,
            lr=params[name]["lr"],
            weight_decay=params[name]["weight_decay"],
            batch_size=params[name]["bs"],
            y1_weight=params[name]["y1_weight"],
            y2_weight=params[name]["y2_weight"],
            max_epochs=200,
            patience=20,
            use_amp=use_amp,
            warmup_epochs=5,
        )
        t0 = time.time()
        _, history = trainer.fit(train_ds, val_ds, checkpoint_dir=str(pred_dir))
        train_time = time.time() - t0
        torch.save({
            "model_state_dict": trainer.model.state_dict(),
            "model_name": name,
            "model_kwargs": model_kwargs,
            "ms": ms,
            "seed": SEED,
        }, pred_dir / "model_state.pth")

    val_scaled = trainer.predict(val_ds)
    test_scaled = trainer.predict(test_ds)
    val_pred = data["scaler_y"].inverse_transform(val_scaled)
    test_pred = data["scaler_y"].inverse_transform(test_scaled)
    metrics = {
        "validation": evaluate(data["y_val"], val_pred),
        "test": evaluate(data["y_test"], test_pred),
        "train_time_s": train_time,
        "source": source,
        "seed": SEED,
        "cache_protocol": CACHE_PROTOCOL if ms != 10 else "loaded_checkpoint",
    }
    if history is not None:
        metrics["best_epoch"] = int(np.argmin(history["val_loss"]) + 1)
        _safe_dump_json(
            pred_dir / "history.json",
            {k: [float(v) for v in values] for k, values in history.items()},
        )
        NNTrainer.plot_loss_curves(
            history, f"{name} ms={ms}",
            save_path=str(pred_dir / "loss_curves.png"),
        )

    np.save(val_path, val_pred)
    np.save(test_path, test_pred)
    _safe_dump_json(metrics_path, metrics)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"val": val_pred, "test": test_pred, "metrics": metrics}


def build_predictions_for_window(ms: int, data: dict, params: dict,
                                 models: list[str], device: str,
                                 force_train: bool) -> tuple[dict, dict, dict]:
    val_preds = {}
    test_preds = {}
    expert_metrics = {}
    for model_name in models:
        logger.info(f"Preparing predictions: ms={ms}, model={model_name}")
        if ms == 10 and not force_train:
            result = predict_from_ms10_tdher_cache(model_name, data, OUT_DIR)
            if result is not None:
                val_preds[model_name] = result["val"]
                test_preds[model_name] = result["test"]
                expert_metrics[model_name] = result["metrics"]
                tm = result["metrics"]["test"]
                logger.info(
                    f"ms={ms} {model_name}: test y1={tm['y1_MAE']:.6f}, "
                    f"test y2={tm['y2_MAE']:.6f} (audited cache)"
                )
                continue

        if model_name == "LightGBM":
            result = predict_lightgbm_for_window(
                ms, data, params, OUT_DIR, force_train=force_train)
        else:
            result = predict_neural_for_window(
                model_name,
                ms,
                data,
                params,
                OUT_DIR,
                device=device,
                use_amp=(device != "cpu"),
                force_train=force_train and ms != 10,
            )
        val_preds[model_name] = result["val"]
        test_preds[model_name] = result["test"]
        expert_metrics[model_name] = result["metrics"]
        tm = result["metrics"]["test"]
        logger.info(
            f"ms={ms} {model_name}: test y1={tm['y1_MAE']:.6f}, "
            f"test y2={tm['y2_MAE']:.6f}"
        )
    return val_preds, test_preds, expert_metrics


def route_window(ms: int, y_val: np.ndarray, y_test: np.ndarray,
                 val_preds: dict, test_preds: dict) -> dict:
    routed = {}
    best_pred, best_details = best_expert_router(y_val, val_preds, test_preds)
    convex_pred, convex_details = convex_blend_router(y_val, val_preds, test_preds)
    shared_pred, shared_details = shared_convex_blend_router(y_val, val_preds, test_preds)
    affine_pred, affine_details = affine_convex_blend_router(y_val, val_preds, test_preds)
    shared_affine_pred, shared_affine_details = shared_affine_convex_blend_router(
        y_val, val_preds, test_preds)
    tdher_pred = enforce_y2_nonnegative(affine_pred)

    raw = {
        "best_expert": (best_pred, best_details),
        "convex_blend": (convex_pred, convex_details),
        "shared_convex_blend": (shared_pred, shared_details),
        "affine_convex_blend": (affine_pred, affine_details),
        "shared_affine_convex_blend": (shared_affine_pred, shared_affine_details),
        "tdher_physical": (
            tdher_pred,
            {
                "source": "affine_convex_blend",
                "projection": "y2 = max(y2, 0)",
                "source_details": affine_details,
            },
        ),
    }

    best_expert_pred = best_pred
    for name, (pred, details) in raw.items():
        routed[name] = {
            "metrics": evaluate(y_test, pred),
            "details": details,
            "paired_bootstrap_vs_best_expert": paired_bootstrap_delta(
                y_test, pred, best_expert_pred),
        }
        np.save(OUT_DIR / f"ms{ms}" / f"{name}_preds.npy", pred)
    return routed


def write_result_tables(all_results: dict, models: list[str]) -> None:
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    weight_rows = []

    for ms in sorted(all_results, key=int):
        payload = all_results[ms]
        window_ms = int(ms) * STEP_MS
        for model_name in models:
            metrics = payload["experts"][model_name]["test"]
            rows.append({
                "window_steps": int(ms),
                "window_ms": window_ms,
                "kind": "expert",
                "method": model_name,
                "y1_mae": metrics["y1_MAE"],
                "y2_mae": metrics["y2_MAE"],
                "y1_rmse": metrics["y1_RMSE"],
                "y2_rmse": metrics["y2_RMSE"],
                "y1_mape": metrics["y1_MAPE"],
                "y2_mape": metrics["y2_MAPE"],
            })

        for router_name in ROUTER_ORDER:
            metrics = payload["routers"][router_name]["metrics"]
            boot = payload["routers"][router_name]["paired_bootstrap_vs_best_expert"]
            rows.append({
                "window_steps": int(ms),
                "window_ms": window_ms,
                "kind": "router",
                "method": router_name,
                "y1_mae": metrics["y1_MAE"],
                "y2_mae": metrics["y2_MAE"],
                "y1_rmse": metrics["y1_RMSE"],
                "y2_rmse": metrics["y2_RMSE"],
                "y1_mape": metrics["y1_MAPE"],
                "y2_mape": metrics["y2_MAPE"],
                "y1_delta_vs_best": boot["y1"]["delta_MAE_candidate_minus_reference"],
                "y2_delta_vs_best": boot["y2"]["delta_MAE_candidate_minus_reference"],
                "y1_bootstrap_p": boot["y1"]["p_two_sided_bootstrap"],
                "y2_bootstrap_p": boot["y2"]["p_two_sided_bootstrap"],
            })

        for router_name in [
            "convex_blend",
            "shared_convex_blend",
            "affine_convex_blend",
            "shared_affine_convex_blend",
            "tdher_physical",
        ]:
            details = payload["routers"][router_name]["details"]
            source_details = details.get("source_details", details)
            if router_name.startswith("shared"):
                weights = source_details.get("weights")
                if weights is None and "shared_route" in source_details:
                    weights = source_details["shared_route"].get("weights", {})
                for model_name, weight in (weights or {}).items():
                    weight_rows.append({
                        "window_steps": int(ms),
                        "window_ms": window_ms,
                        "router": router_name,
                        "target": "shared",
                        "expert": model_name,
                        "weight": weight,
                    })
            else:
                for target in ["y1", "y2"]:
                    weights = source_details.get(target, {}).get("weights", {})
                    for model_name, weight in weights.items():
                        weight_rows.append({
                            "window_steps": int(ms),
                            "window_ms": window_ms,
                            "router": router_name,
                            "target": target,
                            "expert": model_name,
                            "weight": weight,
                        })

    main_path = PAPER_TABLE_DIR / "tdher_multiwindow_l1.csv"
    with open(main_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)

    weights_path = PAPER_TABLE_DIR / "tdher_multiwindow_weights.csv"
    with open(weights_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_steps", "window_ms", "router", "target", "expert", "weight",
            ],
        )
        writer.writeheader()
        writer.writerows(weight_rows)

    logger.info(f"Wrote {main_path}")
    logger.info(f"Wrote {weights_path}")


def plot_tradeoff(all_results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"Skipping plot generation: {exc}")
        return

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    windows = sorted(all_results, key=int)
    x = [int(ms) * STEP_MS for ms in windows]
    best_y1 = [
        all_results[ms]["routers"]["best_expert"]["metrics"]["y1_MAE"]
        for ms in windows
    ]
    best_y2 = [
        all_results[ms]["routers"]["best_expert"]["metrics"]["y2_MAE"]
        for ms in windows
    ]
    tdher_y1 = [
        all_results[ms]["routers"]["tdher_physical"]["metrics"]["y1_MAE"]
        for ms in windows
    ]
    tdher_y2 = [
        all_results[ms]["routers"]["tdher_physical"]["metrics"]["y2_MAE"]
        for ms in windows
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    axes[0].plot(x, best_y1, "o--", color="#767676", label="Best expert")
    axes[0].plot(x, tdher_y1, "s-", color="#1f77b4", label="TD-HER")
    axes[0].set_xlabel("Observation window (ms)")
    axes[0].set_ylabel(r"$y_1$ MAE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(x, best_y2, "o--", color="#767676", label="Best expert")
    axes[1].plot(x, tdher_y2, "s-", color="#d62728", label="TD-HER")
    axes[1].set_xlabel("Observation window (ms)")
    axes[1].set_ylabel(r"$y_2$ MAE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        fig.savefig(PAPER_FIG_DIR / f"tdher_multiwindow_tradeoff.{suffix}",
                    bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", type=int, nargs="*", default=None,
                        help="Window-step values to run. Default: all Exp2 windows.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Expert models to include. Default: Exp2 model set.")
    parser.add_argument("--device", default="cuda",
                        help="Training/inference device. Uses CPU if CUDA is unavailable.")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain non-ms10 neural experts even if cached.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = load_exp1_params()
    ms_list = args.ms if args.ms else MS_VALUES
    model_list = args.models if args.models else MODELS
    for model_name in model_list:
        if model_name not in MODELS:
            raise ValueError(f"Unsupported model {model_name}. Choose from {MODELS}.")

    all_results = {}
    for ms in ms_list:
        logger.info(f"\n{'=' * 80}\nMulti-window TD-HER: ms={ms}, window={ms * STEP_MS} ms")
        data = load_data_for_ms(ms)
        val_preds, test_preds, expert_metrics = build_predictions_for_window(
            ms, data, params, model_list, device=device,
            force_train=args.force_train)
        routers = route_window(
            ms,
            data["y_val"],
            data["y_test"],
            val_preds,
            test_preds,
        )
        payload = {
            "window_steps": ms,
            "window_ms": ms * STEP_MS,
            "models": model_list,
            "experts": expert_metrics,
            "routers": routers,
        }
        _safe_dump_json(OUT_DIR / f"ms{ms}" / "metrics_summary.json", payload)
        all_results[str(ms)] = payload

        tdher = routers["tdher_physical"]["metrics"]
        best = routers["best_expert"]["metrics"]
        logger.info(
            f"ms={ms}: best y1={best['y1_MAE']:.6f}, y2={best['y2_MAE']:.6f}; "
            f"TD-HER y1={tdher['y1_MAE']:.6f}, y2={tdher['y2_MAE']:.6f}"
        )

    _safe_dump_json(OUT_DIR / "metrics_summary.json", all_results)
    write_result_tables(all_results, model_list)
    plot_tradeoff(all_results)
    logger.info("Multi-window TD-HER complete.")


if __name__ == "__main__":
    main()
