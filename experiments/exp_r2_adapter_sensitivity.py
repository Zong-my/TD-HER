#!/usr/bin/env python3
"""Reviewer-2 adapter sensitivity under the frozen TD-HER protocol.

LightGBM, Ridge, and a small target-specific MLP use the same RepA features,
base training data, three OOF calibration folds, practical admission margin,
paired bootstrap certificate, target-wise simplex route, bounded affine
correction, and nonnegative arrival-time projection. Test labels are used only
for final reporting.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REP_DIR = Path(str(data_path('ieee39_v8_80_10_10')))
EXP1_DIR = work_path('results/ieee39/exp1')
ROUTER_DIR = work_path('results/ieee39/exp_tdh_router')
OUT_DIR = work_path('results/ieee39/revision_r2_adapter')
PAPER_TABLE = work_path('results/paper_tables/tdher_adapter_sensitivity_revision.csv')
MS = 10
BASE_MODELS = (
    "LightGBM", "KAN", "ConvLSTM", "PatchTST", "Mamba", "ST-GCN",
    "FT-Transformer",
)
SCENARIOS = {
    "L1": ("L1_same_dist", "val", "test"),
    "L2": ("L2_cross_cond_fewshot", "cross_cond_finetune", "cross_cond_test"),
    "L3": (
        "L3_cross_topo_fewshot", "cross_cond_topo_finetune",
        "cross_cond_topo_test",
    ),
}
N_FOLDS = 3
RELATIVE_MARGIN = 0.20
BOOTSTRAP_ROUNDS = 2000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    result = {}
    for target_idx, target in enumerate(("y1", "y2")):
        error = prediction[:, target_idx] - y_true[:, target_idx]
        result[f"{target}_MAE"] = float(np.mean(np.abs(error)))
        result[f"{target}_RMSE"] = float(np.sqrt(np.mean(error ** 2)))
    return result


def load_repa() -> dict:
    rep_a = REP_DIR / "repA" / f"ms{MS}"
    with open(rep_a / "feature_names.json") as handle:
        feature_names = json.load(handle)
    with open(EXP1_DIR / "mrmr_features.json") as handle:
        selected = json.load(handle)["union"]
    indices = [feature_names.index(name) for name in selected]
    x_train_raw = np.load(rep_a / "X_train.npy")[:, indices]
    y_train = np.load(rep_a / "y_train.npy")
    x_scaler = StandardScaler().fit(x_train_raw)
    y_scaler = StandardScaler().fit(y_train)
    return {
        "rep_a": rep_a,
        "indices": indices,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "x_train": x_scaler.transform(x_train_raw).astype(np.float32),
        "y_train": y_train,
        "y_train_n": y_scaler.transform(y_train).astype(np.float32),
        "n_features": len(indices),
    }


def load_split(split: str, data: dict) -> dict:
    rep_a = data["rep_a"]
    x_raw = np.load(rep_a / f"X_{split}.npy")[:, data["indices"]]
    y = np.load(rep_a / f"y_{split}.npy")
    return {
        "x": data["x_scaler"].transform(x_raw).astype(np.float32),
        "y": y,
        "y_n": data["y_scaler"].transform(y).astype(np.float32),
    }


def load_base_predictions(cal_split: str, test_split: str) -> tuple[dict, dict]:
    cal_predictions = {}
    test_predictions = {}
    for model in BASE_MODELS:
        model_dir = ROUTER_DIR / "experts" / model
        cal_predictions[model] = np.load(model_dir / f"{cal_split}_preds.npy")
        test_predictions[model] = np.load(model_dir / f"{test_split}_preds.npy")
    return cal_predictions, test_predictions


def select_ridge_alphas(data: dict) -> dict:
    path = OUT_DIR / "ridge_base_alphas.json"
    if path.exists():
        with open(path) as handle:
            return json.load(handle)
    alphas = np.logspace(-4, 4, 17)
    selected = {}
    for target_idx, target in enumerate(("y1", "y2")):
        model = RidgeCV(alphas=alphas)
        model.fit(data["x_train"], data["y_train"][:, target_idx])
        selected[target] = float(model.alpha_)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({
            **selected,
            "selection_data": "L1 training split only",
            "candidate_alphas": alphas.tolist(),
        }, handle, indent=2)
    return selected


def train_ridge_oof(scenario_dir: Path, cal: dict, test: dict,
                    data: dict, force: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    out_dir = scenario_dir / "Ridge-Adapter"
    cal_path = out_dir / "cal_oof_preds.npy"
    test_path = out_dir / "test_preds.npy"
    details_path = out_dir / "training.json"
    if cal_path.exists() and test_path.exists() and details_path.exists() and not force:
        with open(details_path) as handle:
            details = json.load(handle)
        return np.load(cal_path), np.load(test_path), details

    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_ridge_alphas(data)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    cal_oof = np.zeros_like(cal["y"], dtype=float)
    test_prediction = np.zeros_like(test["y"], dtype=float)
    details = {"alphas": selected, "n_folds": N_FOLDS}

    for target_idx, target in enumerate(("y1", "y2")):
        alpha = selected[target]
        for train_idx, val_idx in folds.split(cal["x"]):
            x_fit = np.concatenate([data["x_train"], cal["x"][train_idx]])
            y_fit = np.concatenate([
                data["y_train"][:, target_idx], cal["y"][train_idx, target_idx]
            ])
            model = Ridge(alpha=alpha)
            model.fit(x_fit, y_fit)
            cal_oof[val_idx, target_idx] = model.predict(cal["x"][val_idx])
        final_model = Ridge(alpha=alpha)
        final_model.fit(
            np.concatenate([data["x_train"], cal["x"]]),
            np.concatenate([data["y_train"][:, target_idx], cal["y"][:, target_idx]]),
        )
        test_prediction[:, target_idx] = final_model.predict(test["x"])
        with open(out_dir / f"{target}_final_model.pkl", "wb") as handle:
            pickle.dump(final_model, handle)

    np.save(cal_path, cal_oof)
    np.save(test_path, test_prediction)
    with open(details_path, "w") as handle:
        json.dump(details, handle, indent=2)
    return cal_oof, test_prediction, details


class SmallAdapterMLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def fit_mlp(x: np.ndarray, y: np.ndarray, n_features: int, device: str,
            seed: int, epochs: int, batch_size: int) -> tuple[SmallAdapterMLP, float]:
    set_seed(seed)
    model = SmallAdapterMLP(n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    start = time.time()
    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_batch.to(device, non_blocking=True))
            loss = criterion(prediction, y_batch.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model, time.time() - start


@torch.no_grad()
def predict_mlp(model: SmallAdapterMLP, x: np.ndarray, device: str,
                batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)), batch_size=batch_size * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    values = []
    for (x_batch,) in loader:
        values.append(model(x_batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(values)


def train_mlp_oof(scenario_dir: Path, cal: dict, test: dict,
                  data: dict, args) -> tuple[np.ndarray, np.ndarray, dict]:
    out_dir = scenario_dir / "MLP-Adapter"
    cal_path = out_dir / "cal_oof_preds.npy"
    test_path = out_dir / "test_preds.npy"
    details_path = out_dir / "training.json"
    if cal_path.exists() and test_path.exists() and details_path.exists() and not args.force:
        with open(details_path) as handle:
            details = json.load(handle)
        return np.load(cal_path), np.load(test_path), details

    out_dir.mkdir(parents=True, exist_ok=True)
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    cal_oof_n = np.zeros_like(cal["y_n"], dtype=np.float32)
    test_prediction_n = np.zeros_like(test["y_n"], dtype=np.float32)
    times = []

    for target_idx, target in enumerate(("y1", "y2")):
        for fold_idx, (train_idx, val_idx) in enumerate(folds.split(cal["x"])):
            x_fit = np.concatenate([data["x_train"], cal["x"][train_idx]])
            y_fit = np.concatenate([
                data["y_train_n"][:, target_idx],
                cal["y_n"][train_idx, target_idx],
            ]).astype(np.float32)
            model, elapsed = fit_mlp(
                x_fit, y_fit, data["n_features"], args.device,
                seed=42 + 100 * target_idx + fold_idx,
                epochs=args.mlp_epochs, batch_size=args.mlp_batch_size,
            )
            cal_oof_n[val_idx, target_idx] = predict_mlp(
                model, cal["x"][val_idx], args.device, args.mlp_batch_size
            )
            times.append(elapsed)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        x_final = np.concatenate([data["x_train"], cal["x"]])
        y_final = np.concatenate([
            data["y_train_n"][:, target_idx], cal["y_n"][:, target_idx]
        ]).astype(np.float32)
        final_model, elapsed = fit_mlp(
            x_final, y_final, data["n_features"], args.device,
            seed=42 + 100 * target_idx + 99,
            epochs=args.mlp_epochs, batch_size=args.mlp_batch_size,
        )
        test_prediction_n[:, target_idx] = predict_mlp(
            final_model, test["x"], args.device, args.mlp_batch_size
        )
        torch.save(final_model.state_dict(), out_dir / f"{target}_final_model.pth")
        times.append(elapsed)
        del final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cal_oof = data["y_scaler"].inverse_transform(cal_oof_n)
    test_prediction = data["y_scaler"].inverse_transform(test_prediction_n)
    details = {
        "n_folds": N_FOLDS,
        "architecture": [data["n_features"], 128, 64, 1],
        "target_specific": True,
        "epochs": args.mlp_epochs,
        "batch_size": args.mlp_batch_size,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "elapsed_s": float(sum(times)),
        "selection": "fixed pre-registered budget; no test-based selection",
    }
    np.save(cal_path, cal_oof)
    np.save(test_path, test_prediction)
    with open(details_path, "w") as handle:
        json.dump(details, handle, indent=2)
    return cal_oof, test_prediction, details


def load_lightgbm_adapter(router_scenario: str) -> tuple[np.ndarray, np.ndarray, dict]:
    adapter_dir = ROUTER_DIR / "adapters" / router_scenario / "LightGBM-Adapter"
    with open(adapter_dir / "metrics.json") as handle:
        details = json.load(handle)
    return (
        np.load(adapter_dir / "cal_oof_preds.npy"),
        np.load(adapter_dir / "test_preds.npy"),
        details,
    )


def bootstrap_delta(candidate_error: np.ndarray, reference_error: np.ndarray,
                    seed: int) -> dict:
    difference = np.asarray(candidate_error) - np.asarray(reference_error)
    rng = np.random.default_rng(seed)
    boot = np.empty(BOOTSTRAP_ROUNDS, dtype=float)
    for round_idx in range(BOOTSTRAP_ROUNDS):
        indices = rng.integers(0, len(difference), size=len(difference))
        boot[round_idx] = float(np.mean(difference[indices]))
    lower, upper = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_delta": float(np.mean(difference)),
        "ci95": [float(lower), float(upper)],
        "p_two_sided": float(min(
            1.0, 2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0))
        )),
    }


def optimize_simplex(y: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, float]:
    n_experts = predictions.shape[1]
    objective = lambda weight: float(np.mean(np.abs(y - predictions @ weight)))
    result = minimize(
        objective, np.full(n_experts, 1.0 / n_experts), method="SLSQP",
        bounds=[(0.0, 1.0)] * n_experts,
        constraints=({"type": "eq", "fun": lambda weight: weight.sum() - 1.0},),
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    weight = np.clip(result.x, 0.0, 1.0)
    weight /= max(weight.sum(), 1e-12)
    return weight, objective(weight)


def fit_affine(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([prediction, np.ones_like(prediction)])
    try:
        slope0, bias0 = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        slope0, bias0 = 1.0, 0.0
    slope0 = float(np.clip(slope0, 0.0, 3.0))
    scale = max(
        float(np.ptp(y)), float(np.std(y)), float(np.mean(np.abs(y))),
        float(np.mean(np.abs(prediction))), 1e-6,
    )
    bias_bound = 5.0 * scale
    objective = lambda params: float(
        np.mean(np.abs(y - (params[0] * prediction + params[1])))
        + 1e-8 * ((params[0] - 1.0) ** 2 + (params[1] / scale) ** 2)
    )
    result = minimize(
        objective, [slope0, np.clip(bias0, -bias_bound, bias_bound)],
        method="SLSQP", bounds=[(0.0, 3.0), (-bias_bound, bias_bound)],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    slope, bias = result.x if result.success else (slope0, bias0)
    return float(slope), float(bias), objective([slope, bias])


def route_candidate(y_cal: np.ndarray, y_test: np.ndarray,
                    base_cal: dict, base_test: dict, adapter_name: str,
                    adapter_cal: np.ndarray, adapter_test: np.ndarray) -> tuple[np.ndarray, dict]:
    final = np.zeros_like(y_test, dtype=float)
    details = {}
    for target_idx, target in enumerate(("y1", "y2")):
        base_scores = {
            name: float(np.mean(np.abs(
                y_cal[:, target_idx] - pred[:, target_idx]
            ))) for name, pred in base_cal.items()
        }
        reference_name = min(base_scores, key=base_scores.get)
        reference_error = np.abs(
            y_cal[:, target_idx] - base_cal[reference_name][:, target_idx]
        )
        adapter_error = np.abs(
            y_cal[:, target_idx] - adapter_cal[:, target_idx]
        )
        certificate = bootstrap_delta(
            adapter_error, reference_error, seed=42 + target_idx
        )
        adapter_score = float(adapter_error.mean())
        required = (1.0 - RELATIVE_MARGIN) * base_scores[reference_name]
        admitted = bool(
            adapter_score <= required and certificate["ci95"][1] < 0.0
        )
        names = list(BASE_MODELS) + ([adapter_name] if admitted else [])
        cal_lookup = {**base_cal, adapter_name: adapter_cal}
        test_lookup = {**base_test, adapter_name: adapter_test}
        cal_matrix = np.column_stack([
            cal_lookup[name][:, target_idx] for name in names
        ])
        test_matrix = np.column_stack([
            test_lookup[name][:, target_idx] for name in names
        ])
        weight, route_cal_mae = optimize_simplex(
            y_cal[:, target_idx], cal_matrix
        )
        cal_route = cal_matrix @ weight
        test_route = test_matrix @ weight
        slope, bias, affine_cal_mae = fit_affine(
            y_cal[:, target_idx], cal_route
        )
        final[:, target_idx] = slope * test_route + bias
        details[target] = {
            "adapter_cal_MAE": adapter_score,
            "reference_model": reference_name,
            "reference_cal_MAE": base_scores[reference_name],
            "required_MAE": required,
            "certificate": certificate,
            "admitted": admitted,
            "route_cal_MAE": route_cal_mae,
            "affine_cal_MAE": affine_cal_mae,
            "slope": slope,
            "bias": bias,
            "weights": {name: float(value) for name, value in zip(names, weight)},
        }
    final[:, 1] = np.maximum(final[:, 1], 0.0)
    return final, details


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mlp-epochs", type=int, default=25)
    parser.add_argument("--mlp-batch-size", type=int, default=4096)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_repa()
    rows = []
    all_details = {}

    for scenario, (router_scenario, cal_split, test_split) in SCENARIOS.items():
        logger.info(f"Adapter sensitivity: {scenario}")
        scenario_dir = OUT_DIR / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        cal = load_split(cal_split, data)
        test = load_split(test_split, data)
        base_cal, base_test = load_base_predictions(cal_split, test_split)

        candidates = {
            "LightGBM-Adapter": load_lightgbm_adapter(router_scenario),
            "Ridge-Adapter": train_ridge_oof(
                scenario_dir, cal, test, data, args.force
            ),
        }
        if not args.skip_mlp:
            candidates["MLP-Adapter"] = train_mlp_oof(
                scenario_dir, cal, test, data, args
            )

        scenario_details = {}
        for adapter_name, (cal_pred, test_pred, training) in candidates.items():
            final_pred, route_details = route_candidate(
                cal["y"], test["y"], base_cal, base_test,
                adapter_name, cal_pred, test_pred,
            )
            np.save(scenario_dir / f"{adapter_name}_final_route_preds.npy", final_pred)
            row = {
                "scenario": scenario,
                "adapter": adapter_name,
                **{f"adapter_oof_{key}": value
                   for key, value in metrics(cal["y"], cal_pred).items()},
                **{f"adapter_test_{key}": value
                   for key, value in metrics(test["y"], test_pred).items()},
                **{f"final_{key}": value
                   for key, value in metrics(test["y"], final_pred).items()},
                "y1_admitted": route_details["y1"]["admitted"],
                "y2_admitted": route_details["y2"]["admitted"],
                "n_calibration": len(cal["y"]),
                "n_test": len(test["y"]),
            }
            rows.append(row)
            scenario_details[adapter_name] = {
                "training": training,
                "route": route_details,
                "row": row,
            }
        all_details[scenario] = scenario_details
        with open(scenario_dir / "details.json", "w") as handle:
            json.dump(scenario_details, handle, indent=2)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "summary.csv", index=False)
    PAPER_TABLE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PAPER_TABLE, index=False)
    with open(OUT_DIR / "all_details.json", "w") as handle:
        json.dump(all_details, handle, indent=2)
    with open(OUT_DIR / "protocol.json", "w") as handle:
        json.dump({
            "representation": "RepA with the same 1243 mRMR-selected features",
            "base_training_samples": len(data["y_train"]),
            "n_folds": N_FOLDS,
            "relative_improvement_margin": RELATIVE_MARGIN,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "test_selection": False,
        }, handle, indent=2)
    logger.info(f"Results written to {OUT_DIR}")


if __name__ == "__main__":
    main()
