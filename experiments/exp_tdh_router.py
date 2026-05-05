#!/usr/bin/env python3
"""Target-dependent heterogeneous expert routing.

This diagnostic experiment uses validation data only to learn target-wise
expert routing over already trained Exp1 models. It does not touch existing
Exp1 artifacts and writes all outputs to ``results/ieee39/exp_tdh_router``.

Routers evaluated:
  1. best_expert: choose one validation-best expert per target.
  2. convex_blend: learn nonnegative per-target expert weights on validation.
  3. affine_best_expert: validation-best expert after restricted affine calibration.
  4. affine_convex_blend: convex blend followed by restricted affine calibration.
"""

import json
import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from loguru import logger
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp4_generalization import (  # noqa: E402
    MS,
    MODELS,
    REP_DIR,
    evaluate,
    load_level_data,
    load_train_scalers,
    predict_dl_model,
    predict_kan,
    predict_lightgbm,
)

# TabR is excluded until its legacy retrieval-cache outputs are rerun under the
# repaired final-cache protocol. See coding_exp_plan.md for the audit note.
ROUTER_MODELS = [m for m in MODELS if m != "TabR"]

OUT_DIR = Path("results/ieee39/exp_tdh_router")
EXPERT_DIR = OUT_DIR / "experts"
ADAPTER_DIR = OUT_DIR / "adapters"
ADAPTER_N_FOLDS = 3
ADAPTER_MAX_TREES = 800
ADAPTER_N_JOBS = 8
ADAPTER_NAMES = {"LightGBM-Adapter"}
ADAPTER_ADMISSION_REL_IMPROV = 0.20
ADAPTER_ADMISSION_SWEEP = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
BOOTSTRAP_ROUNDS = 2000
CERT_ADMISSION_ALPHA = 0.05

SCENARIOS = {
    "L1_same_dist": ("val", "test"),
    "L2_cross_cond_fewshot": ("cross_cond_finetune", "cross_cond_test"),
    "L3_cross_topo_fewshot": (
        "cross_cond_topo_finetune",
        "cross_cond_topo_test",
    ),
}


def attach_base_training_data(scalers: dict) -> dict:
    X_train_A = np.load(f"{REP_DIR}/repA/ms{MS}/X_train.npy")
    y_train = np.load(f"{REP_DIR}/repA/ms{MS}/y_train.npy")
    scalers["X_train_n"] = scalers["scaler_A"].transform(
        X_train_A[:, scalers["feat_idx"]]
    ).astype(np.float32)
    scalers["y_train"] = y_train
    return scalers


def predict_expert(model_name: str, data: dict, scalers: dict) -> np.ndarray:
    if model_name == "LightGBM":
        return predict_lightgbm(data, scalers)
    if model_name == "KAN":
        return predict_kan(data, scalers)
    return predict_dl_model(model_name, data, scalers)


def get_cached_prediction(model_name: str, split: str, data: dict,
                          scalers: dict) -> np.ndarray:
    mdir = EXPERT_DIR / model_name
    mdir.mkdir(parents=True, exist_ok=True)
    pred_path = mdir / f"{split}_preds.npy"
    if pred_path.exists():
        return np.load(pred_path)
    logger.info(f"Predicting {model_name} on {split}...")
    pred = predict_expert(model_name, data, scalers)
    np.save(pred_path, pred)
    with open(mdir / f"{split}_metrics.json", "w") as f:
        json.dump(evaluate(data["y"], pred), f, indent=2)
    return pred


def load_exp1_lgb_specs() -> dict:
    specs = {}
    for target in ["y1", "y2"]:
        with open(Path("results/ieee39/exp1/LightGBM") / f"{target}_lgb_model.pkl",
                  "rb") as f:
            model = pickle.load(f)
        params = model.get_params()
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None and best_iter > 0:
            params["n_estimators"] = int(best_iter)
        params["n_estimators"] = min(int(params.get("n_estimators", 4000)),
                                     ADAPTER_MAX_TREES)
        params.update({
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "n_jobs": ADAPTER_N_JOBS,
            "random_state": 42,
        })
        specs[target] = params
    return specs


def train_lgb_adapter_oof(scenario_name: str, cal_data: dict,
                          test_data: dict, scalers: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a few-shot tabular adapter with OOF calibration predictions.

    The final test adapter trains on original train + full calibration data.
    The calibration prediction used for router selection is out-of-fold: each
    calibration sample is predicted by a model that did not train on that sample.
    """
    save_dir = ADAPTER_DIR / scenario_name / "LightGBM-Adapter"
    save_dir.mkdir(parents=True, exist_ok=True)
    cal_path = save_dir / "cal_oof_preds.npy"
    test_path = save_dir / "test_preds.npy"
    metrics_path = save_dir / "metrics.json"

    if cal_path.exists() and test_path.exists() and metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        return np.load(cal_path), np.load(test_path), metrics

    specs = load_exp1_lgb_specs()
    X_base = scalers["X_train_n"]
    y_base = scalers["y_train"]
    X_cal = cal_data["X_n"]
    y_cal = cal_data["y"]
    X_test = test_data["X_n"]

    n_cal = len(y_cal)
    n_splits = min(ADAPTER_N_FOLDS, n_cal)
    if n_splits < 2:
        raise ValueError("LightGBM adapter needs at least two calibration samples")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cal_oof = np.zeros_like(y_cal, dtype=float)
    test_pred = np.zeros_like(test_data["y"], dtype=float)

    for target_idx, target in enumerate(["y1", "y2"]):
        params = dict(specs[target])
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_cal), start=1):
            X_fold = np.concatenate([X_base, X_cal[train_idx]], axis=0)
            y_fold = np.concatenate([y_base[:, target_idx], y_cal[train_idx, target_idx]])
            model = lgb.LGBMRegressor(**params)
            model.fit(X_fold, y_fold)
            cal_oof[val_idx, target_idx] = model.predict(X_cal[val_idx])
            logger.info(
                f"{scenario_name} LightGBM-Adapter {target} OOF fold "
                f"{fold_idx}/{n_splits} done"
            )

        X_final = np.concatenate([X_base, X_cal], axis=0)
        y_final = np.concatenate([y_base[:, target_idx], y_cal[:, target_idx]])
        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X_final, y_final)
        test_pred[:, target_idx] = final_model.predict(X_test)

        with open(save_dir / f"{target}_final_model.pkl", "wb") as f:
            pickle.dump(final_model, f)

    metrics = {
        "calibration_oof": evaluate(y_cal, cal_oof),
        "test": evaluate(test_data["y"], test_pred),
        "protocol": {
            "n_folds": n_splits,
            "max_trees": ADAPTER_MAX_TREES,
            "n_jobs": ADAPTER_N_JOBS,
        },
    }
    np.save(cal_path, cal_oof)
    np.save(test_path, test_pred)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return cal_oof, test_pred, metrics


def target_mae(y_true: np.ndarray, pred: np.ndarray, target_idx: int) -> float:
    return float(np.mean(np.abs(y_true[:, target_idx] - pred[:, target_idx])))


def _mae(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - pred)))


def fit_restricted_affine(y_cal: np.ndarray,
                          pred_cal: np.ndarray) -> tuple[float, float, dict]:
    """Fit y ~= a * pred + b with small degrees of freedom.

    The slope is constrained to be nonnegative and bounded. This captures
    systematic scale/bias shift in few-shot transfer without the freedom of a
    full stacked regressor over all expert outputs.
    """
    y = np.asarray(y_cal, dtype=float).reshape(-1)
    z = np.asarray(pred_cal, dtype=float).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(z)
    y = y[finite]
    z = z[finite]

    if len(y) == 0:
        return 1.0, 0.0, {
            "success": False,
            "message": "no finite calibration samples",
            "cal_MAE": float("nan"),
        }

    if np.std(z) < 1e-12:
        b = float(np.median(y))
        return 0.0, b, {
            "success": True,
            "message": "constant predictor fallback",
            "cal_MAE": _mae(y, np.full_like(y, b)),
        }

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
        pred = a * z + b
        # A tiny prior keeps the affine correction from drifting when several
        # parameter pairs have indistinguishable MAE on a small calibration set.
        return _mae(y, pred) + 1e-8 * ((a - 1.0) ** 2 + (b / scale) ** 2)

    res = minimize(
        objective,
        x0=np.array([a0, float(np.clip(b0, -b_bound, b_bound))]),
        method="SLSQP",
        bounds=[(0.0, 3.0), (-b_bound, b_bound)],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if res.success and np.all(np.isfinite(res.x)):
        a, b = float(res.x[0]), float(res.x[1])
        message = str(res.message)
        success = True
    else:
        a, b = a0, float(np.clip(b0, -b_bound, b_bound))
        message = str(res.message) if "res" in locals() else "optimizer failed"
        success = False

    cal_pred = a * z + b
    return a, b, {
        "success": success,
        "message": message,
        "cal_MAE": _mae(y, cal_pred),
        "slope": a,
        "bias": b,
    }


def apply_affine(pred: np.ndarray, slope: float, bias: float) -> np.ndarray:
    return slope * pred + bias


def enforce_y2_nonnegative(pred: np.ndarray) -> np.ndarray:
    """Apply the known physical output constraint t_delta >= 0."""
    out = np.array(pred, copy=True)
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    return out


def paired_bootstrap_delta(y_true: np.ndarray, candidate_pred: np.ndarray,
                           reference_pred: np.ndarray,
                           n_rounds: int = BOOTSTRAP_ROUNDS,
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


def bootstrap_error_delta(candidate_error: np.ndarray,
                          reference_error: np.ndarray,
                          n_rounds: int = BOOTSTRAP_ROUNDS,
                          seed: int = 42) -> dict:
    """Bootstrap mean(candidate_error - reference_error)."""
    cand = np.asarray(candidate_error, dtype=float).reshape(-1)
    ref = np.asarray(reference_error, dtype=float).reshape(-1)
    finite = np.isfinite(cand) & np.isfinite(ref)
    diff = cand[finite] - ref[finite]
    if len(diff) == 0:
        return {
            "mean_delta": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "p_two_sided_bootstrap": float("nan"),
            "n_rounds": n_rounds,
        }

    rng = np.random.default_rng(seed)
    observed = float(np.mean(diff))
    boot = np.empty(n_rounds, dtype=float)
    n = len(diff)
    for i in range(n_rounds):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(np.mean(diff[idx]))
    lower, upper = np.percentile(boot, [2.5, 97.5])
    p_two_sided = 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))
    return {
        "mean_delta": observed,
        "ci95": [float(lower), float(upper)],
        "p_two_sided_bootstrap": float(min(p_two_sided, 1.0)),
        "n_rounds": n_rounds,
    }


def optimize_convex_weights(y: np.ndarray, P_val: np.ndarray) -> tuple[np.ndarray, float]:
    n = P_val.shape[1]

    def objective(w):
        return _mae(y, P_val @ w)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                   constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-10})
    if not res.success:
        logger.warning(f"blend optimization warning: {res.message}")
    w = np.clip(res.x, 0.0, 1.0)
    w = w / max(np.sum(w), 1e-12)
    return w, objective(w)


def _target_balance_scales(y_cal: np.ndarray, cal_preds: dict,
                           models: list[str]) -> np.ndarray:
    """Use best single-expert calibration MAE to balance y1/y2 units.

    A shared route is otherwise dominated by the arrival-time target because it
    is measured on a larger numerical scale than signed frequency extremum.
    The resulting shared-routing baseline is deliberately strong: it gives both
    targets equal normalized weight during route fitting.
    """
    scales = []
    for target_idx in range(2):
        y = y_cal[:, target_idx]
        best = min(
            _mae(y, cal_preds[m][:, target_idx])
            for m in models
        )
        fallback = max(
            float(np.std(y)),
            float(np.mean(np.abs(y))),
            1e-9,
        )
        scales.append(best if best > 1e-12 else fallback)
    return np.asarray(scales, dtype=float)


def optimize_shared_convex_weights(y_cal: np.ndarray,
                                   P_cal: np.ndarray,
                                   scales: np.ndarray
                                   ) -> tuple[np.ndarray, float, dict]:
    """Fit one simplex-constrained expert route shared by both targets.

    Parameters
    ----------
    y_cal:
        Calibration targets with shape (n_samples, 2).
    P_cal:
        Expert predictions with shape (n_samples, n_models, 2).
    scales:
        Per-target positive normalization constants.
    """
    n_models = P_cal.shape[1]

    def per_target_mae(w):
        return np.asarray([
            _mae(y_cal[:, target_idx], P_cal[:, :, target_idx] @ w)
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
        logger.warning(f"shared blend optimization warning: {res.message}")
    w = np.clip(res.x, 0.0, 1.0)
    w = w / max(np.sum(w), 1e-12)
    maes = per_target_mae(w)
    return w, objective(w), {
        "balanced_objective": objective(w),
        "target_balance_scales": {
            "y1": float(scales[0]),
            "y2": float(scales[1]),
        },
        "cal_MAE": {
            "y1": float(maes[0]),
            "y2": float(maes[1]),
        },
    }


def shared_convex_blend_outputs(y_cal: np.ndarray, cal_preds: dict,
                                test_preds: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    models = list(cal_preds)
    P_cal = np.stack([cal_preds[m] for m in models], axis=1)
    P_test = np.stack([test_preds[m] for m in models], axis=1)
    scales = _target_balance_scales(y_cal, cal_preds, models)
    w, objective_value, info = optimize_shared_convex_weights(y_cal, P_cal, scales)

    cal_blend = np.zeros_like(next(iter(cal_preds.values())))
    test_blend = np.zeros_like(next(iter(test_preds.values())))
    for target_idx in range(2):
        cal_blend[:, target_idx] = P_cal[:, :, target_idx] @ w
        test_blend[:, target_idx] = P_test[:, :, target_idx] @ w

    details = {
        "routing": "shared_nonnegative_convex",
        "objective": "mean target-balanced calibration MAE",
        "balanced_objective": float(objective_value),
        "weights": {m: float(wi) for m, wi in zip(models, w)},
        **info,
    }
    return cal_blend, test_blend, details


def shared_convex_blend_router(y_cal: np.ndarray, cal_preds: dict,
                               test_preds: dict) -> tuple[np.ndarray, dict]:
    _, test_blend, details = shared_convex_blend_outputs(
        y_cal, cal_preds, test_preds)
    return test_blend, details


def shared_affine_convex_blend_router(y_cal: np.ndarray, cal_preds: dict,
                                      test_preds: dict) -> tuple[np.ndarray, dict]:
    cal_blend, test_blend, shared_details = shared_convex_blend_outputs(
        y_cal, cal_preds, test_preds)
    routed = np.zeros_like(test_blend)
    details = {
        "shared_route": shared_details,
        "y1": {},
        "y2": {},
    }

    for target_idx, target_name in enumerate(["y1", "y2"]):
        a, b, info = fit_restricted_affine(
            y_cal[:, target_idx],
            cal_blend[:, target_idx],
        )
        routed[:, target_idx] = apply_affine(test_blend[:, target_idx], a, b)
        details[target_name] = {
            "affine_cal_MAE": info["cal_MAE"],
            "slope": a,
            "bias": b,
            "success": info["success"],
        }

    return routed, details


def admitted_models_for_shared_route(y_cal: np.ndarray, cal_preds: dict,
                                     rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV
                                     ) -> tuple[list[str], dict]:
    """Build a common admitted expert pool from target-wise OOF admission.

    The union policy gives the shared-routing baseline access to any adapter
    that is justified for at least one target, so the comparison does not
    artificially weaken the shared route.
    """
    per_target = {}
    admitted_union = set()
    for target_idx, target_name in enumerate(["y1", "y2"]):
        target_models, admission = admitted_models_for_target(
            y_cal, cal_preds, target_idx, rel_improv)
        per_target[target_name] = admission
        admitted_union.update(target_models)

    models = [m for m in cal_preds if m in admitted_union]
    return models, {
        "policy": "union_of_target_wise_oof_admitted_models",
        "relative_improvement_required": rel_improv,
        "per_target_admission": per_target,
        "admitted_models": models,
    }


def admission_shared_convex_blend_router(y_cal: np.ndarray, cal_preds: dict,
                                         test_preds: dict,
                                         rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV
                                         ) -> tuple[np.ndarray, dict]:
    models, admission = admitted_models_for_shared_route(
        y_cal, cal_preds, rel_improv)
    cal_sub = {m: cal_preds[m] for m in models}
    test_sub = {m: test_preds[m] for m in models}
    routed, details = shared_convex_blend_router(y_cal, cal_sub, test_sub)
    return routed, {
        "shared_route": details,
        "admission": admission,
    }


def admission_shared_affine_convex_blend_router(
    y_cal: np.ndarray,
    cal_preds: dict,
    test_preds: dict,
    rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV,
) -> tuple[np.ndarray, dict]:
    models, admission = admitted_models_for_shared_route(
        y_cal, cal_preds, rel_improv)
    cal_sub = {m: cal_preds[m] for m in models}
    test_sub = {m: test_preds[m] for m in models}
    routed, details = shared_affine_convex_blend_router(y_cal, cal_sub, test_sub)
    return routed, {
        **details,
        "admission": admission,
    }


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


def admitted_models_for_target(y_cal: np.ndarray, cal_preds: dict,
                               target_idx: int,
                               rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV
                               ) -> tuple[list[str], dict]:
    target_name = ["y1", "y2"][target_idx]
    base_models = [m for m in cal_preds if m not in ADAPTER_NAMES]
    adapter_models = [m for m in cal_preds if m in ADAPTER_NAMES]
    y = y_cal[:, target_idx]
    base_scores = {
        model: _mae(y, cal_preds[model][:, target_idx])
        for model in base_models
    }
    base_best_model = min(base_scores, key=base_scores.get)
    base_best = base_scores[base_best_model]

    admitted = list(base_models)
    adapter_details = {}
    for model in adapter_models:
        score = _mae(y, cal_preds[model][:, target_idx])
        required = base_best * (1.0 - rel_improv)
        is_admitted = score <= required
        if is_admitted:
            admitted.append(model)
        adapter_details[model] = {
            "cal_MAE": score,
            "required_MAE": required,
            "admitted": is_admitted,
        }

    return admitted, {
        "target": target_name,
        "base_best_model": base_best_model,
        "base_best_cal_MAE": base_best,
        "relative_improvement_required": rel_improv,
        "adapter_details": adapter_details,
        "admitted_models": admitted,
    }


def certified_admitted_models_for_target(
    y_cal: np.ndarray,
    cal_preds: dict,
    target_idx: int,
    rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV,
    alpha: float = CERT_ADMISSION_ALPHA,
) -> tuple[list[str], dict]:
    """OOF adapter admission with an event-level bootstrap certificate.

    The adapter must satisfy both gates:
    1. practical improvement over the best base expert on OOF calibration MAE;
    2. paired bootstrap CI upper bound below zero for event-wise error deltas.

    This preserves the existing practical-effect margin while requiring the
    improvement to be consistent across calibration events.
    """
    del alpha  # Currently fixed to a 95% percentile interval.
    target_name = ["y1", "y2"][target_idx]
    base_models = [m for m in cal_preds if m not in ADAPTER_NAMES]
    adapter_models = [m for m in cal_preds if m in ADAPTER_NAMES]
    y = y_cal[:, target_idx]
    base_scores = {
        model: _mae(y, cal_preds[model][:, target_idx])
        for model in base_models
    }
    base_best_model = min(base_scores, key=base_scores.get)
    base_best = base_scores[base_best_model]
    base_error = np.abs(y - cal_preds[base_best_model][:, target_idx])

    admitted = list(base_models)
    adapter_details = {}
    for model in adapter_models:
        adapter_error = np.abs(y - cal_preds[model][:, target_idx])
        score = float(np.mean(adapter_error))
        required = base_best * (1.0 - rel_improv)
        practical_gate = score <= required
        cert = bootstrap_error_delta(
            adapter_error,
            base_error,
            n_rounds=BOOTSTRAP_ROUNDS,
            seed=42 + target_idx,
        )
        ci_high = cert["ci95"][1]
        certificate_gate = bool(np.isfinite(ci_high) and ci_high < 0.0)
        is_admitted = practical_gate and certificate_gate
        if is_admitted:
            admitted.append(model)
        adapter_details[model] = {
            "cal_MAE": score,
            "required_MAE": required,
            "practical_gate": practical_gate,
            "certificate_gate": certificate_gate,
            "admitted": is_admitted,
            "certificate": cert,
            "reference_base_model": base_best_model,
        }

    return admitted, {
        "target": target_name,
        "policy": "practical_oof_margin_and_paired_bootstrap_ci",
        "base_best_model": base_best_model,
        "base_best_cal_MAE": base_best,
        "relative_improvement_required": rel_improv,
        "ci_requirement": "95% CI upper bound of adapter_error - base_error < 0",
        "adapter_details": adapter_details,
        "admitted_models": admitted,
    }


def admission_best_expert_router(y_cal: np.ndarray, cal_preds: dict,
                                 test_preds: dict,
                                 rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV
                                 ) -> tuple[np.ndarray, dict]:
    routed = np.zeros_like(next(iter(test_preds.values())))
    selected = {}
    for target_idx, target_name in enumerate(["y1", "y2"]):
        models, admission = admitted_models_for_target(
            y_cal, cal_preds, target_idx, rel_improv)
        scores = {
            model: _mae(y_cal[:, target_idx], cal_preds[model][:, target_idx])
            for model in models
        }
        best_model = min(scores, key=scores.get)
        routed[:, target_idx] = test_preds[best_model][:, target_idx]
        selected[target_name] = {
            "expert": best_model,
            "val_MAE": scores[best_model],
            "all_admitted_val_MAE": scores,
            "admission": admission,
        }
    return routed, selected


def convex_blend_router(y_val: np.ndarray, val_preds: dict,
                        test_preds: dict) -> tuple[np.ndarray, dict]:
    models = list(val_preds)
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}

    for target_idx, target_name in enumerate(["y1", "y2"]):
        P_val = np.stack([val_preds[m][:, target_idx] for m in models], axis=1)
        P_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        y = y_val[:, target_idx]
        w, val_mae = optimize_convex_weights(y, P_val)

        routed[:, target_idx] = P_test @ w
        details[target_name] = {
            "val_MAE": val_mae,
            "weights": {m: float(wi) for m, wi in zip(models, w)},
        }

    return routed, details


def admission_convex_blend_router(y_cal: np.ndarray, cal_preds: dict,
                                  test_preds: dict,
                                  rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV
                                  ) -> tuple[np.ndarray, dict]:
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}

    for target_idx, target_name in enumerate(["y1", "y2"]):
        models, admission = admitted_models_for_target(
            y_cal, cal_preds, target_idx, rel_improv)
        P_cal = np.stack([cal_preds[m][:, target_idx] for m in models], axis=1)
        P_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        y = y_cal[:, target_idx]
        w, val_mae = optimize_convex_weights(y, P_cal)

        routed[:, target_idx] = P_test @ w
        details[target_name] = {
            "val_MAE": val_mae,
            "weights": {m: float(wi) for m, wi in zip(models, w)},
            "admission": admission,
        }

    return routed, details


def affine_best_expert_router(y_cal: np.ndarray, cal_preds: dict,
                              test_preds: dict) -> tuple[np.ndarray, dict]:
    models = list(cal_preds)
    routed = np.zeros_like(next(iter(test_preds.values())))
    selected = {}

    for target_idx, target_name in enumerate(["y1", "y2"]):
        candidates = {}
        for model in models:
            a, b, info = fit_restricted_affine(
                y_cal[:, target_idx],
                cal_preds[model][:, target_idx],
            )
            base_cal_mae = _mae(
                y_cal[:, target_idx],
                cal_preds[model][:, target_idx],
            )
            candidates[model] = {
                "base_cal_MAE": base_cal_mae,
                "affine_cal_MAE": info["cal_MAE"],
                "slope": a,
                "bias": b,
                "success": info["success"],
            }

        best_model = min(candidates, key=lambda m: candidates[m]["affine_cal_MAE"])
        params = candidates[best_model]
        routed[:, target_idx] = apply_affine(
            test_preds[best_model][:, target_idx],
            params["slope"],
            params["bias"],
        )
        selected[target_name] = {
            "expert": best_model,
            **params,
            "all_candidates": candidates,
        }

    return routed, selected


def affine_convex_blend_router(y_cal: np.ndarray, cal_preds: dict,
                               test_preds: dict) -> tuple[np.ndarray, dict]:
    models = list(cal_preds)
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}

    for target_idx, target_name in enumerate(["y1", "y2"]):
        P_cal = np.stack([cal_preds[m][:, target_idx] for m in models], axis=1)
        P_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        y = y_cal[:, target_idx]
        w, blend_cal_mae = optimize_convex_weights(y, P_cal)
        cal_blend = P_cal @ w
        test_blend = P_test @ w
        a, b, info = fit_restricted_affine(y, cal_blend)

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


def admission_affine_convex_blend_router(y_cal: np.ndarray, cal_preds: dict,
                                         test_preds: dict,
                                         rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV
                                         ) -> tuple[np.ndarray, dict]:
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}

    for target_idx, target_name in enumerate(["y1", "y2"]):
        models, admission = admitted_models_for_target(
            y_cal, cal_preds, target_idx, rel_improv)
        P_cal = np.stack([cal_preds[m][:, target_idx] for m in models], axis=1)
        P_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        y = y_cal[:, target_idx]
        w, blend_cal_mae = optimize_convex_weights(y, P_cal)
        cal_blend = P_cal @ w
        test_blend = P_test @ w
        a, b, info = fit_restricted_affine(y, cal_blend)

        routed[:, target_idx] = apply_affine(test_blend, a, b)
        details[target_name] = {
            "blend_cal_MAE": blend_cal_mae,
            "affine_cal_MAE": info["cal_MAE"],
            "slope": a,
            "bias": b,
            "success": info["success"],
            "weights": {m: float(wi) for m, wi in zip(models, w)},
            "admission": admission,
        }

    return routed, details


def certified_admission_convex_blend_router(
    y_cal: np.ndarray,
    cal_preds: dict,
    test_preds: dict,
    rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV,
) -> tuple[np.ndarray, dict]:
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}

    for target_idx, target_name in enumerate(["y1", "y2"]):
        models, admission = certified_admitted_models_for_target(
            y_cal, cal_preds, target_idx, rel_improv)
        P_cal = np.stack([cal_preds[m][:, target_idx] for m in models], axis=1)
        P_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        y = y_cal[:, target_idx]
        w, val_mae = optimize_convex_weights(y, P_cal)

        routed[:, target_idx] = P_test @ w
        details[target_name] = {
            "val_MAE": val_mae,
            "weights": {m: float(wi) for m, wi in zip(models, w)},
            "admission": admission,
        }

    return routed, details


def certified_admission_affine_convex_blend_router(
    y_cal: np.ndarray,
    cal_preds: dict,
    test_preds: dict,
    rel_improv: float = ADAPTER_ADMISSION_REL_IMPROV,
) -> tuple[np.ndarray, dict]:
    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}

    for target_idx, target_name in enumerate(["y1", "y2"]):
        models, admission = certified_admitted_models_for_target(
            y_cal, cal_preds, target_idx, rel_improv)
        P_cal = np.stack([cal_preds[m][:, target_idx] for m in models], axis=1)
        P_test = np.stack([test_preds[m][:, target_idx] for m in models], axis=1)
        y = y_cal[:, target_idx]
        w, blend_cal_mae = optimize_convex_weights(y, P_cal)
        cal_blend = P_cal @ w
        test_blend = P_test @ w
        a, b, info = fit_restricted_affine(y, cal_blend)

        routed[:, target_idx] = apply_affine(test_blend, a, b)
        details[target_name] = {
            "blend_cal_MAE": blend_cal_mae,
            "affine_cal_MAE": info["cal_MAE"],
            "slope": a,
            "bias": b,
            "success": info["success"],
            "weights": {m: float(wi) for m, wi in zip(models, w)},
            "admission": admission,
        }

    return routed, details


def ridge_stack_router(y_cal: np.ndarray, cal_preds: dict,
                       test_preds: dict) -> tuple[np.ndarray, dict]:
    """Target-wise stacked calibration over expert predictions.

    The stacker is intentionally small: standardized expert predictions followed
    by RidgeCV. It can learn bias/residual corrections from a scenario-specific
    calibration split without retraining the base experts.
    """
    models = list(cal_preds)
    X_cal = np.concatenate([cal_preds[m] for m in models], axis=1)
    X_test = np.concatenate([test_preds[m] for m in models], axis=1)
    scaler = StandardScaler().fit(X_cal)
    X_cal_n = scaler.transform(X_cal)
    X_test_n = scaler.transform(X_test)

    routed = np.zeros_like(next(iter(test_preds.values())))
    details = {}
    alphas = np.logspace(-4, 4, 17)

    for target_idx, target_name in enumerate(["y1", "y2"]):
        model = RidgeCV(alphas=alphas)
        model.fit(X_cal_n, y_cal[:, target_idx])
        routed[:, target_idx] = model.predict(X_test_n)
        details[target_name] = {
            "alpha": float(model.alpha_),
            "cal_MAE": float(np.mean(np.abs(
                y_cal[:, target_idx] - model.predict(X_cal_n)
            ))),
        }
        coefs = {}
        for expert_idx, expert in enumerate(models):
            coefs[f"{expert}_y1_pred"] = float(model.coef_[2 * expert_idx])
            coefs[f"{expert}_y2_pred"] = float(model.coef_[2 * expert_idx + 1])
        details[target_name]["coef_by_feature"] = coefs

    return routed, details


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scalers = attach_base_training_data(load_train_scalers())
    all_results = {}

    for scenario_name, (cal_split, test_split) in SCENARIOS.items():
        logger.info(f"\n{'=' * 80}\n{scenario_name}: calibrate={cal_split}, test={test_split}")
        scenario_dir = OUT_DIR / "scenarios" / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        cal_data = load_level_data(cal_split, scalers)
        test_data = load_level_data(test_split, scalers)

        cal_preds = {}
        test_preds = {}
        expert_metrics = {}

        for model_name in ROUTER_MODELS:
            cal_pred = get_cached_prediction(model_name, cal_split, cal_data, scalers)
            test_pred = get_cached_prediction(model_name, test_split, test_data, scalers)
            cal_preds[model_name] = cal_pred
            test_preds[model_name] = test_pred
            expert_metrics[model_name] = {
                "calibration": evaluate(cal_data["y"], cal_pred),
                "test": evaluate(test_data["y"], test_pred),
            }
            logger.info(
                f"{model_name}: cal y1={expert_metrics[model_name]['calibration']['y1_MAE']:.6f}, "
                f"cal y2={expert_metrics[model_name]['calibration']['y2_MAE']:.6f}, "
                f"test y1={expert_metrics[model_name]['test']['y1_MAE']:.6f}, "
                f"test y2={expert_metrics[model_name]['test']['y2_MAE']:.6f}"
            )

        adapter_cal_pred, adapter_test_pred, adapter_metrics = train_lgb_adapter_oof(
            scenario_name, cal_data, test_data, scalers)
        cal_preds["LightGBM-Adapter"] = adapter_cal_pred
        test_preds["LightGBM-Adapter"] = adapter_test_pred
        expert_metrics["LightGBM-Adapter"] = adapter_metrics
        logger.info(
            f"LightGBM-Adapter: cal-oof y1={adapter_metrics['calibration_oof']['y1_MAE']:.6f}, "
            f"cal-oof y2={adapter_metrics['calibration_oof']['y2_MAE']:.6f}, "
            f"test y1={adapter_metrics['test']['y1_MAE']:.6f}, "
            f"test y2={adapter_metrics['test']['y2_MAE']:.6f}"
        )

        base_cal_preds = {
            model: pred for model, pred in cal_preds.items()
            if model not in ADAPTER_NAMES
        }
        base_test_preds = {
            model: pred for model, pred in test_preds.items()
            if model not in ADAPTER_NAMES
        }

        base_best_pred, base_best_details = best_expert_router(
            cal_data["y"], base_cal_preds, base_test_preds)
        base_blend_pred, base_blend_details = convex_blend_router(
            cal_data["y"], base_cal_preds, base_test_preds)
        base_shared_blend_pred, base_shared_blend_details = shared_convex_blend_router(
            cal_data["y"], base_cal_preds, base_test_preds)
        base_affine_blend_pred, base_affine_blend_details = (
            affine_convex_blend_router(cal_data["y"], base_cal_preds,
                                       base_test_preds)
        )
        base_shared_affine_blend_pred, base_shared_affine_blend_details = (
            shared_affine_convex_blend_router(cal_data["y"], base_cal_preds,
                                              base_test_preds)
        )
        best_pred, best_details = best_expert_router(
            cal_data["y"], cal_preds, test_preds)
        blend_pred, blend_details = convex_blend_router(
            cal_data["y"], cal_preds, test_preds)
        shared_blend_pred, shared_blend_details = shared_convex_blend_router(
            cal_data["y"], cal_preds, test_preds)
        admit_best_pred, admit_best_details = admission_best_expert_router(
            cal_data["y"], cal_preds, test_preds)
        admit_blend_pred, admit_blend_details = admission_convex_blend_router(
            cal_data["y"], cal_preds, test_preds)
        cert_blend_pred, cert_blend_details = certified_admission_convex_blend_router(
            cal_data["y"], cal_preds, test_preds)
        admit_shared_blend_pred, admit_shared_blend_details = (
            admission_shared_convex_blend_router(cal_data["y"], cal_preds,
                                                 test_preds)
        )
        affine_best_pred, affine_best_details = affine_best_expert_router(
            cal_data["y"], cal_preds, test_preds)
        affine_blend_pred, affine_blend_details = affine_convex_blend_router(
            cal_data["y"], cal_preds, test_preds)
        shared_affine_blend_pred, shared_affine_blend_details = (
            shared_affine_convex_blend_router(cal_data["y"], cal_preds,
                                              test_preds)
        )
        admit_affine_blend_pred, admit_affine_blend_details = (
            admission_affine_convex_blend_router(cal_data["y"], cal_preds,
                                                 test_preds)
        )
        cert_affine_blend_pred, cert_affine_blend_details = (
            certified_admission_affine_convex_blend_router(cal_data["y"],
                                                           cal_preds,
                                                           test_preds)
        )
        admit_shared_affine_blend_pred, admit_shared_affine_blend_details = (
            admission_shared_affine_convex_blend_router(cal_data["y"], cal_preds,
                                                        test_preds)
        )
        base_affine_blend_nn_pred = enforce_y2_nonnegative(base_affine_blend_pred)
        base_shared_affine_blend_nn_pred = enforce_y2_nonnegative(
            base_shared_affine_blend_pred)
        admit_affine_blend_nn_pred = enforce_y2_nonnegative(admit_affine_blend_pred)
        cert_affine_blend_nn_pred = enforce_y2_nonnegative(cert_affine_blend_pred)
        admit_shared_affine_blend_nn_pred = enforce_y2_nonnegative(
            admit_shared_affine_blend_pred)
        ridge_pred, ridge_details = ridge_stack_router(
            cal_data["y"], cal_preds, test_preds)

        scenario_results = {
            "calibration_split": cal_split,
            "test_split": test_split,
            "base_best_expert": {
                "metrics": evaluate(test_data["y"], base_best_pred),
                "details": base_best_details,
            },
            "base_convex_blend": {
                "metrics": evaluate(test_data["y"], base_blend_pred),
                "details": base_blend_details,
            },
            "base_shared_convex_blend": {
                "metrics": evaluate(test_data["y"], base_shared_blend_pred),
                "details": base_shared_blend_details,
            },
            "base_affine_convex_blend": {
                "metrics": evaluate(test_data["y"], base_affine_blend_pred),
                "details": base_affine_blend_details,
            },
            "base_shared_affine_convex_blend": {
                "metrics": evaluate(test_data["y"], base_shared_affine_blend_pred),
                "details": base_shared_affine_blend_details,
            },
            "base_affine_convex_blend_nonnegative_y2": {
                "metrics": evaluate(test_data["y"], base_affine_blend_nn_pred),
                "details": {
                    "source": "base_affine_convex_blend",
                    "projection": "y2 = max(y2, 0)",
                    "source_details": base_affine_blend_details,
                },
            },
            "base_shared_affine_convex_blend_nonnegative_y2": {
                "metrics": evaluate(test_data["y"], base_shared_affine_blend_nn_pred),
                "details": {
                    "source": "base_shared_affine_convex_blend",
                    "projection": "y2 = max(y2, 0)",
                    "source_details": base_shared_affine_blend_details,
                },
            },
            "best_expert": {
                "metrics": evaluate(test_data["y"], best_pred),
                "details": best_details,
            },
            "convex_blend": {
                "metrics": evaluate(test_data["y"], blend_pred),
                "details": blend_details,
            },
            "shared_convex_blend": {
                "metrics": evaluate(test_data["y"], shared_blend_pred),
                "details": shared_blend_details,
            },
            "admission_best_expert": {
                "metrics": evaluate(test_data["y"], admit_best_pred),
                "details": admit_best_details,
            },
            "admission_convex_blend": {
                "metrics": evaluate(test_data["y"], admit_blend_pred),
                "details": admit_blend_details,
            },
            "certified_admission_convex_blend": {
                "metrics": evaluate(test_data["y"], cert_blend_pred),
                "details": cert_blend_details,
            },
            "admission_shared_convex_blend": {
                "metrics": evaluate(test_data["y"], admit_shared_blend_pred),
                "details": admit_shared_blend_details,
            },
            "affine_best_expert": {
                "metrics": evaluate(test_data["y"], affine_best_pred),
                "details": affine_best_details,
            },
            "affine_convex_blend": {
                "metrics": evaluate(test_data["y"], affine_blend_pred),
                "details": affine_blend_details,
            },
            "shared_affine_convex_blend": {
                "metrics": evaluate(test_data["y"], shared_affine_blend_pred),
                "details": shared_affine_blend_details,
            },
            "admission_affine_convex_blend": {
                "metrics": evaluate(test_data["y"], admit_affine_blend_pred),
                "details": admit_affine_blend_details,
            },
            "certified_admission_affine_convex_blend": {
                "metrics": evaluate(test_data["y"], cert_affine_blend_pred),
                "details": cert_affine_blend_details,
            },
            "admission_shared_affine_convex_blend": {
                "metrics": evaluate(test_data["y"], admit_shared_affine_blend_pred),
                "details": admit_shared_affine_blend_details,
            },
            "admission_affine_convex_blend_nonnegative_y2": {
                "metrics": evaluate(test_data["y"], admit_affine_blend_nn_pred),
                "details": {
                    "source": "admission_affine_convex_blend",
                    "projection": "y2 = max(y2, 0)",
                    "source_details": admit_affine_blend_details,
                },
            },
            "certified_admission_affine_convex_blend_nonnegative_y2": {
                "metrics": evaluate(test_data["y"], cert_affine_blend_nn_pred),
                "details": {
                    "source": "certified_admission_affine_convex_blend",
                    "projection": "y2 = max(y2, 0)",
                    "source_details": cert_affine_blend_details,
                },
            },
            "admission_shared_affine_convex_blend_nonnegative_y2": {
                "metrics": evaluate(test_data["y"], admit_shared_affine_blend_nn_pred),
                "details": {
                    "source": "admission_shared_affine_convex_blend",
                    "projection": "y2 = max(y2, 0)",
                    "source_details": admit_shared_affine_blend_details,
                },
            },
            "ridge_stack": {
                "metrics": evaluate(test_data["y"], ridge_pred),
                "details": ridge_details,
            },
            "experts": expert_metrics,
        }
        scenario_results["paired_bootstrap"] = {
            "admission_affine_vs_base_affine": paired_bootstrap_delta(
                test_data["y"], admit_affine_blend_pred, base_affine_blend_pred),
            "admission_affine_vs_base_best": paired_bootstrap_delta(
                test_data["y"], admit_affine_blend_pred, base_best_pred),
            "admission_affine_vs_admission_convex": paired_bootstrap_delta(
                test_data["y"], admit_affine_blend_pred, admit_blend_pred),
            "admission_affine_nonnegative_vs_base_affine_nonnegative": paired_bootstrap_delta(
                test_data["y"], admit_affine_blend_nn_pred, base_affine_blend_nn_pred),
            "certified_admission_affine_nonnegative_vs_base_affine_nonnegative": paired_bootstrap_delta(
                test_data["y"], cert_affine_blend_nn_pred, base_affine_blend_nn_pred),
            "certified_vs_threshold_admission_affine_nonnegative": paired_bootstrap_delta(
                test_data["y"], cert_affine_blend_nn_pred, admit_affine_blend_nn_pred),
            "targetwise_vs_shared_base_affine_nonnegative": paired_bootstrap_delta(
                test_data["y"], base_affine_blend_nn_pred, base_shared_affine_blend_nn_pred),
            "targetwise_vs_shared_admission_affine_nonnegative": paired_bootstrap_delta(
                test_data["y"], admit_affine_blend_nn_pred,
                admit_shared_affine_blend_nn_pred),
        }
        sensitivity = {}
        for threshold in ADAPTER_ADMISSION_SWEEP:
            pred, details = admission_affine_convex_blend_router(
                cal_data["y"], cal_preds, test_preds, rel_improv=threshold)
            sensitivity[f"{threshold:.2f}"] = {
                "metrics": evaluate(test_data["y"], pred),
                "details": details,
            }
        scenario_results["adapter_admission_threshold_sensitivity"] = sensitivity

        np.save(scenario_dir / "base_best_expert_preds.npy", base_best_pred)
        np.save(scenario_dir / "base_convex_blend_preds.npy", base_blend_pred)
        np.save(scenario_dir / "base_shared_convex_blend_preds.npy",
                base_shared_blend_pred)
        np.save(scenario_dir / "base_affine_convex_blend_preds.npy",
                base_affine_blend_pred)
        np.save(scenario_dir / "base_shared_affine_convex_blend_preds.npy",
                base_shared_affine_blend_pred)
        np.save(scenario_dir / "base_affine_convex_blend_nonnegative_y2_preds.npy",
                base_affine_blend_nn_pred)
        np.save(scenario_dir / "base_shared_affine_convex_blend_nonnegative_y2_preds.npy",
                base_shared_affine_blend_nn_pred)
        np.save(scenario_dir / "best_expert_preds.npy", best_pred)
        np.save(scenario_dir / "convex_blend_preds.npy", blend_pred)
        np.save(scenario_dir / "shared_convex_blend_preds.npy", shared_blend_pred)
        np.save(scenario_dir / "admission_best_expert_preds.npy", admit_best_pred)
        np.save(scenario_dir / "admission_convex_blend_preds.npy", admit_blend_pred)
        np.save(scenario_dir / "certified_admission_convex_blend_preds.npy",
                cert_blend_pred)
        np.save(scenario_dir / "admission_shared_convex_blend_preds.npy",
                admit_shared_blend_pred)
        np.save(scenario_dir / "affine_best_expert_preds.npy", affine_best_pred)
        np.save(scenario_dir / "affine_convex_blend_preds.npy", affine_blend_pred)
        np.save(scenario_dir / "shared_affine_convex_blend_preds.npy",
                shared_affine_blend_pred)
        np.save(scenario_dir / "admission_affine_convex_blend_preds.npy",
                admit_affine_blend_pred)
        np.save(scenario_dir / "certified_admission_affine_convex_blend_preds.npy",
                cert_affine_blend_pred)
        np.save(scenario_dir / "admission_shared_affine_convex_blend_preds.npy",
                admit_shared_affine_blend_pred)
        np.save(scenario_dir / "admission_affine_convex_blend_nonnegative_y2_preds.npy",
                admit_affine_blend_nn_pred)
        np.save(scenario_dir / "certified_admission_affine_convex_blend_nonnegative_y2_preds.npy",
                cert_affine_blend_nn_pred)
        np.save(scenario_dir / "admission_shared_affine_convex_blend_nonnegative_y2_preds.npy",
                admit_shared_affine_blend_nn_pred)
        np.save(scenario_dir / "ridge_stack_preds.npy", ridge_pred)
        with open(scenario_dir / "metrics_summary.json", "w") as f:
            json.dump(scenario_results, f, indent=2)

        all_results[scenario_name] = scenario_results

        logger.info("Target-dependent router results:")
        for router_name in [
            "base_best_expert",
            "base_convex_blend",
            "base_shared_convex_blend",
            "base_affine_convex_blend",
            "base_shared_affine_convex_blend",
            "base_affine_convex_blend_nonnegative_y2",
            "base_shared_affine_convex_blend_nonnegative_y2",
            "best_expert",
            "convex_blend",
            "shared_convex_blend",
            "admission_best_expert",
            "admission_convex_blend",
            "certified_admission_convex_blend",
            "admission_shared_convex_blend",
            "affine_best_expert",
            "affine_convex_blend",
            "shared_affine_convex_blend",
            "admission_affine_convex_blend",
            "certified_admission_affine_convex_blend",
            "admission_shared_affine_convex_blend",
            "admission_affine_convex_blend_nonnegative_y2",
            "certified_admission_affine_convex_blend_nonnegative_y2",
            "admission_shared_affine_convex_blend_nonnegative_y2",
            "ridge_stack",
        ]:
            m = scenario_results[router_name]["metrics"]
            logger.info(
                f"{scenario_name} {router_name}: y1_MAE={m['y1_MAE']:.6f}, "
                f"y2_MAE={m['y2_MAE']:.6f}"
            )
            logger.info(json.dumps(scenario_results[router_name]["details"], indent=2))
        logger.info("Paired bootstrap deltas:")
        logger.info(json.dumps(scenario_results["paired_bootstrap"], indent=2))
        logger.info("Adapter admission threshold sensitivity:")
        logger.info(json.dumps(scenario_results["adapter_admission_threshold_sensitivity"],
                               indent=2))

    with open(OUT_DIR / "metrics_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
