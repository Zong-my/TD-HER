#!/usr/bin/env python3
"""Matched post-processing for NEW results only (protocol second_backbone_convlstm_v2, section matched_postprocessing).

For every completed cell of the second-backbone block, and for the tuned widened control's final seeds (via --infer),
fit the bounded affine correction per target on calibration predictions (L1: validation split; L2/L3: full calibration
set of the refit model) with the frozen function text of experiments/exp_tdh_router.py (fit_restricted_affine,
apply_affine, enforce_y2_nonnegative, _mae), extracted verbatim from the hash-verified file with `ast` because that
module has heavy top-level imports; then report raw and corrected metrics for every cell. Nothing is selected by test
performance; both presentations are always written. Old checkpoints are never read.
"""
from __future__ import annotations

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for p in (PROJECT_ROOT, PROJECT_ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import exp_r2_mtl_baselines as R  # frozen runner  # noqa: E402
from data_proc.datasets import TensorDataset  # noqa: E402

ROUTER = PROJECT_ROOT / "experiments/exp_tdh_router.py"
FUNCS = ("_mae", "fit_restricted_affine", "apply_affine", "enforce_y2_nonnegative")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_affine(expected_sha256: str | None = None) -> dict:
    src = ROUTER.read_text()
    if expected_sha256 and sha256_file(ROUTER) != expected_sha256:
        raise RuntimeError("exp_tdh_router.py hash differs from the protocol")
    tree = ast.parse(src)
    wanted = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCS}
    missing = [f for f in FUNCS if f not in wanted]
    if missing:
        raise RuntimeError(f"frozen functions not found: {missing}")
    module = ast.Module(body=[wanted[f] for f in FUNCS], type_ignores=[])
    ns = {"np": np, "minimize": minimize}
    exec(compile(module, str(ROUTER), "exec"), ns)
    return {f: ns[f] for f in FUNCS} | {"source_sha256": sha256_file(ROUTER), "function_text_sha256": hashlib.sha256("\n".join(ast.get_source_segment(src, wanted[f]) for f in FUNCS).encode()).hexdigest()}


def correct_cell(fns: dict, preds_test: np.ndarray, preds_cal: np.ndarray, y_cal: np.ndarray) -> tuple[np.ndarray, dict]:
    out = np.array(preds_test, dtype=float, copy=True); info = {}
    for t, name in enumerate(("y1", "y2")):
        a, b, fit = fns["fit_restricted_affine"](y_cal[:, t], preds_cal[:, t])
        out[:, t] = fns["apply_affine"](preds_test[:, t].astype(float), a, b)
        info[name] = {"a": a, "b": b, "cal_MAE": fit["cal_MAE"], "success": fit["success"], "message": fit["message"], "n_cal": int(len(y_cal))}
    out = fns["enforce_y2_nonnegative"](out)
    return out, info


def cells_of_block(root: Path, strategies, seeds, scenarios=("L1", "L2", "L3"), expected_cells: int | None = None) -> list[tuple[str, int, str, Path]]:
    """Enumerate ONLY the declared strategy x seed x exact scenario paths; each must hold a valid COMPLETE.json whose identity
    matches its path (scenario_status == complete, full manifest and hashes verified). Archived or stray directories
    (e.g. L1_incomplete_<ts>) are ignored and reported; duplicate identities are refused; the expected grid is required."""
    import exp_v4_second_backbone_convlstm as D
    cells, seen, problems = [], set(), []
    for strategy in strategies:
        for seed in seeds:
            for scenario in scenarios:
                d = root / strategy / f"seed_{seed}" / scenario
                if not (d / "COMPLETE.json").exists():
                    problems.append(f"missing cell {strategy}/seed_{seed}/{scenario}"); continue
                try:
                    ident = json.loads((d / "COMPLETE.json").read_text())["identity"]
                except Exception as exc:
                    problems.append(f"unreadable marker {d}: {exc!r}"); continue
                if (ident.get("strategy"), int(ident.get("seed", -1)), ident.get("scenario")) != (strategy, seed, scenario):
                    problems.append(f"marker identity does not match its path: {d}"); continue
                status, info = D.scenario_status(d, ident)
                if status != "complete":
                    problems.append(f"cell not complete ({status}): {d} {info}"); continue
                key = (strategy, seed, scenario)
                if key in seen:
                    raise RuntimeError(f"duplicate cell identity {key}")
                seen.add(key); cells.append((strategy, seed, scenario, d))
    stray = [str(p.parent) for p in root.glob("*/seed_*/*_incomplete_*/COMPLETE.json")] + [str(p.parent) for p in root.glob("*/seed_*/*/COMPLETE.json") if p.parent.name not in scenarios]
    if stray:
        print(f"[postprocess] ignored {len(stray)} archived/stray directories holding a COMPLETE.json: {stray[:5]}")
    if expected_cells is not None and len(cells) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} valid block cells, found {len(cells)}; problems: {problems[:10]}")
    return cells


def scenario_trainer_batch(config_batch: int, scenario: str, n_cal: int) -> int:
    """Batch of the trainer that produced the saved predictions: the configured batch at L1; the frozen run_strategy refit rule
    min(batch, max(16, n_full_calibration)) at L2/L3. The frozen predict() uses twice this value."""
    return int(config_batch) if scenario == "L1" else int(min(int(config_batch), max(16, int(n_cal))))


def predict_with_frozen_trainer(model, run_config: dict, scenario: str, Xt_n, Xs_n, y_n, device: str, num_workers: int = 0):
    """Calibration-set predictions through the SAME path as the saved run's test predictions: the frozen
    ConflictAwareTrainer.predict (CUDA autocast when the strategy is vanilla and the device is CUDA; FP32 on CPU),
    batch = 2 x the scenario trainer batch, the proper calibration TensorDataset. Returns standardized predictions and
    an explicit record of device / precision / batch."""
    strategy = run_config.get("trainer_strategy", run_config.get("strategy", "vanilla"))
    if strategy not in R.STRATEGIES:
        raise RuntimeError(f"saved strategy {strategy!r} is not a frozen trainer strategy")
    bs = scenario_trainer_batch(run_config["batch_size"], scenario, len(y_n))
    trainer = R.ConflictAwareTrainer(model, strategy, device, float(run_config["lr"]), float(run_config["weight_decay"]), bs, 1, 1, num_workers)
    pred_scaled = trainer.predict(TensorDataset(np.asarray(Xt_n, dtype=np.float32), np.asarray(Xs_n, dtype=np.float32), np.asarray(y_n, dtype=np.float32)))
    info = {"path": "frozen exp_r2_mtl_baselines.ConflictAwareTrainer.predict", "strategy": strategy, "device": device, "use_amp": bool(trainer.use_amp),
            "precision": "CUDA autocast (AMP)" if trainer.use_amp else "FP32", "trainer_batch_size": bs, "predict_batch_size": 2 * bs, "num_workers": num_workers, "torch": torch.__version__}
    return pred_scaled, info


def wide_cells(wide_final_dir: Path, data: dict, out_root: Path, tiny: bool, seeds=(42, 52, 62), scenarios=("L1", "L2", "L3"), expected_cells: int = 9) -> list[tuple[str, int, str, Path]]:
    """Calibration-split predictions of the tuned widened control's saved per-scenario checkpoints (NEW tree only).
    Only the declared seeds x exact scenarios are accepted (vanilla/seed_<s>/<scenario>/model.pth + preds.npy); the expected grid is required.
    Inference matches the saved run's prediction protocol (frozen trainer.predict; CUDA/AMP in production, CPU FP32 for tiny fixtures)."""
    cells = []; missing = []
    device = "cuda" if (torch.cuda.is_available() and not tiny) else "cpu"
    for seed in seeds:
        for scenario in scenarios:
            ck_path = wide_final_dir / "vanilla" / f"seed_{seed}" / scenario / "model.pth"
            if not ck_path.exists() or not (ck_path.parent / "preds.npy").exists():
                missing.append(f"vanilla/seed_{seed}/{scenario}"); continue
            ck = torch.load(ck_path, map_location="cpu", weights_only=False); run_config = ck["run_config"]
            model = R.make_model(data, run_config); model.load_state_dict(ck["model_state_dict"])
            if scenario == "L1":
                Xt, Xs, y_cal, y_cal_n = data["Xt_val_n"], data["Xs_val_n"], data["y_val"], data["y_val_n"]
            else:
                cal_split = R.SCENARIOS[scenario][0]; sd = R.load_shift_split(cal_split, data); Xt, Xs, y_cal, y_cal_n = sd["Xt_n"], sd["Xs_n"], sd["y"], sd["y_n"]
            if tiny:
                n = 40 if scenario == "L1" else 24; Xt, Xs, y_cal, y_cal_n = Xt[:n], Xs[:n], y_cal[:n], y_cal_n[:n]
            pred_scaled, inference = predict_with_frozen_trainer(model, run_config, scenario, Xt, Xs, y_cal_n, device)
            preds_cal = data["scaler_y"].inverse_transform(pred_scaled)
            d = out_root / "postprocessing" / "wide_tuned" / f"seed_{seed}" / scenario; d.mkdir(parents=True, exist_ok=True)
            np.save(d / "preds_cal.npy", preds_cal); np.save(d / "y_cal.npy", y_cal)
            preds_test = np.load(ck_path.parent / "preds.npy"); np.save(d / "preds.npy", preds_test)
            y_test = data["y_test"][:len(preds_test)] if scenario == "L1" else R.load_shift_split(R.SCENARIOS[scenario][1], data)["y"][:len(preds_test)]
            np.save(d / "y_test.npy", y_test)
            (d / "source.json").write_text(json.dumps({"checkpoint": str(ck_path), "checkpoint_sha256": sha256_file(ck_path), "preds_sha256": sha256_file(ck_path.parent / "preds.npy"),
                                                       "calibration_inference": inference, "n_cal": int(len(y_cal)), "note": "same prediction path as the saved run; tiny fixtures run on CPU FP32"}, indent=2))
            cells.append(("wide_tuned_vanilla", seed, scenario, d))
    if len(cells) != expected_cells:
        raise RuntimeError(f"expected {expected_cells} widened cells, found {len(cells)}; missing {missing}")
    return cells


def run(out_root: Path, proto: dict, wide_final_dir: Path | None = None, tiny: bool = False, expected_cells: int = 45,
        wide_seeds=(42, 52, 62), wide_expected_cells: int = 9, strategies=("vanilla", "uncertainty", "gradnorm", "pcgrad", "mmoe"), seeds=(42, 52, 62)) -> pd.DataFrame:
    fns = load_frozen_affine(proto["frozen_sources"]["affine_and_projection"]["sha256"])
    pp = out_root / "postprocessing"; pp.mkdir(parents=True, exist_ok=True)
    cells = cells_of_block(out_root, strategies, seeds, expected_cells=expected_cells)
    data = None
    if wide_final_dir is not None:
        data = R.load_fixed_data(); cells += wide_cells(wide_final_dir, data, out_root, tiny, seeds=wide_seeds, expected_cells=wide_expected_cells)
    rows = []
    for strategy, seed, scenario, d in cells:
        preds = np.load(d / "preds.npy"); preds_cal = np.load(d / "preds_cal.npy"); y_cal = np.load(d / "y_cal.npy"); y_test = np.load(d / "y_test.npy")
        corrected, info = correct_cell(fns, preds, preds_cal, y_cal)
        raw_m = R.evaluate(y_test, preds); cor_m = R.evaluate(y_test, corrected)
        cell_dir = pp / strategy / f"seed_{seed}" / scenario; cell_dir.mkdir(parents=True, exist_ok=True)
        np.save(cell_dir / "preds_corrected.npy", corrected)
        (cell_dir / "affine.json").write_text(json.dumps({"fit": info, "raw": raw_m, "corrected": cor_m, "fitting_data": "L1: validation split predictions of the selected checkpoint; L2/L3: full calibration-set predictions of the refit model (in-sample)",
                                                          "frozen_function_source_sha256": fns["source_sha256"], "frozen_function_text_sha256": fns["function_text_sha256"]}, indent=2, default=str))
        rows.append({"strategy": strategy, "seed": seed, "scenario": scenario, "n_test": int(len(y_test)), "n_cal": int(len(y_cal)),
                     **{f"raw_{k}": v for k, v in raw_m.items() if k.endswith(("MAE", "RMSE"))}, **{f"corrected_{k}": v for k, v in cor_m.items() if k.endswith(("MAE", "RMSE"))},
                     "a_y1": info["y1"]["a"], "b_y1": info["y1"]["b"], "a_y2": info["y2"]["a"], "b_y2": info["y2"]["b"], "fit_success_y1": info["y1"]["success"], "fit_success_y2": info["y2"]["success"]})
    frame = pd.DataFrame(rows); frame.to_csv(pp / "matched_postprocessing_runs.csv", index=False)
    if len(frame):
        cols = [c for c in frame.columns if c.startswith(("raw_", "corrected_"))]
        summ = frame.groupby(["strategy", "scenario"])[cols].agg(["mean", "std", "count"]); summ.columns = ["_".join(c) for c in summ.columns]
        summ.reset_index().to_csv(pp / "summary.csv", index=False)
    (pp / "provenance.json").write_text(json.dumps({"cells": len(rows), "frozen_function_source_sha256": fns["source_sha256"], "frozen_function_text_sha256": fns["function_text_sha256"], "wide_final_dir": str(wide_final_dir) if wide_final_dir else None, "tiny": tiny}, indent=2))
    return frame
