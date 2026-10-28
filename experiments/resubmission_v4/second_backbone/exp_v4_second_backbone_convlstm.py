#!/usr/bin/env python3
"""Second-backbone multi-task control block: ConvLSTM x {vanilla, uncertainty, gradnorm, pcgrad, mmoe} x seeds.

Protocol: artifacts/r2_additive_research_20261013_190350/protocols/second_backbone_convlstm_v2.json (frozen).
The frozen runner experiments/exp_r2_mtl_baselines.py supplies, unchanged, the data pipeline (scalers fitted on the L1
training split only), the ConflictAwareTrainer with the four gradient strategies, the shifted-scenario inner-split
selection and refit, save_run and aggregate. This driver only (1) builds the ConvLSTM models, (2) runs the prescribed
grid seed-major, (3) writes preds_cal.npy / y_cal.npy / y_test.npy / inner_split_indices.npz and an atomic COMPLETE.json
per scenario, (4) resumes only genuinely incomplete cells and refuses identity mismatches, (5) records provenance
(read tracker with phase tags, fit inputs, scaler parameters, parameter counts, hashes). No hyper-parameter search.
Phases: describe (CPU, models only), train, postprocess (delegates to matched_postprocessing.py).
"""
from __future__ import annotations

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Subset

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for p in (PROJECT_ROOT, PROJECT_ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import exp_r2_mtl_baselines as R  # frozen runner  # noqa: E402
from data_proc.datasets import TensorDataset  # noqa: E402
from convlstm_mtl_model import ConvLSTMConflictMTL, ConvLSTMMMoE, count_parameters  # noqa: E402

ART = WORK_ROOT
DEFAULT_PROTOCOL = ART / "protocols/second_backbone_convlstm_v2.json"
STRATEGIES = ("vanilla", "uncertainty", "gradnorm", "pcgrad", "mmoe")
SEEDS = (42, 52, 62)
TINY = {"train": 96, "val": 40, "test": 32, "shift_cal": 24, "shift_test": 16, "batch": 32,
        "max_epochs": 1, "patience": 1, "fine_tune_epochs": 1}
REQUIRED_FILES = ("model.pth", "preds.npy", "preds_cal.npy", "y_cal.npy", "y_test.npy", "metrics.json", "fit.json", "config.json")

# ---------------------------------------------------------------- provenance helpers
_PHASE = {"tag": "init"}
_READS: list[dict] = []
_FIT_INPUTS: list[dict] = []
_ORIG_NP_LOAD = np.load


def _tracked_load(path, *args, **kwargs):
    _READS.append({"path": str(path), "tag": _PHASE["tag"], "time": time.time()})
    return _ORIG_NP_LOAD(path, *args, **kwargs)


def set_phase(tag: str) -> None:
    _PHASE["tag"] = tag


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


# ---------------------------------------------------------------- protocol / configuration
def load_protocol(path: Path) -> dict:
    proto = json.loads(path.read_text())
    proto["_sha256"] = sha256_file(path)
    proto["_path"] = str(path)
    return proto


def backbone_config(proto: dict, data: dict) -> dict:
    """Model kwargs and optimizer values per the protocol derivation rule, asserted against best_params.json."""
    bc = proto["backbone_configuration"]
    bp_path = PROJECT_ROOT / proto["frozen_sources"]["convlstm_expert"]["best_params"]["path"]
    if sha256_file(bp_path) != proto["frozen_sources"]["convlstm_expert"]["best_params"]["sha256"]:
        raise RuntimeError("best_params.json hash differs from the protocol")
    bp = json.loads(bp_path.read_text())
    derived = {"hidden_channels": [bp["hidden_1"], bp["hidden_2"]], "kernel_width": bp["kernel_width"], "mlp_hidden": bp["mlp_hidden"],
               "dropout": bp["dropout"], "tab_hidden": bp["tab_hidden"], "tab_output": bp["tab_output"], "output_dim": 2,
               "n_features": data["n_features"], "n_generators": data["n_generators"], "n_static": data["n_static"]}
    if derived != bc["model_kwargs"]:
        raise RuntimeError(f"model kwargs derived from best_params differ from the protocol: {derived} vs {bc['model_kwargs']}")
    opt = bc["optimizer"]
    if not (abs(opt["lr"] - bp["lr"]) < 1e-15 and abs(opt["weight_decay"] - bp["weight_decay"]) < 1e-15 and opt["batch_size"] == bp["bs"]
            and abs(opt["task_weights_y1_y2"][0] - bp["y1_weight"]) < 1e-15 and abs(opt["task_weights_y1_y2"][1] - bp["y2_weight"]) < 1e-15):
        raise RuntimeError("optimizer values differ between best_params.json and the protocol")
    return {"model": dict(derived), "lr": float(opt["lr"]), "weight_decay": float(opt["weight_decay"]), "batch_size": int(opt["batch_size"]),
            "y1_weight": float(opt["task_weights_y1_y2"][0]), "y2_weight": float(opt["task_weights_y1_y2"][1]),
            "source": proto["frozen_sources"]["convlstm_expert"]["best_params"]["path"], "best_params_sha256": sha256_file(bp_path),
            "task_scaling": "targets standardized on the L1 training split", "selection_metric": "mean normalized validation SmoothL1 loss",
            "scheduler": "five-epoch linear warm-up followed by cosine decay", "mmoe_head": proto["strategies"]["mmoe"]["head"]}


def make_model(strategy: str, config: dict):
    kwargs = dict(config["model"])
    if strategy == "mmoe":
        head = config["mmoe_head"]
        return ConvLSTMMMoE(num_experts=head["num_experts"], expert_hidden=head["expert_hidden"], tower_hidden=head["tower_hidden"], **kwargs)
    return ConvLSTMConflictMTL(**kwargs)


def trainer_strategy(strategy: str) -> str:
    return "vanilla" if strategy == "mmoe" else strategy


def identity_fields(proto: dict, run_config: dict, strategy: str, seed: int, scenario: str, tiny: bool) -> dict:
    return {"protocol_id": proto["protocol_id"], "protocol_sha256": proto["_sha256"], "driver_sha256": sha256_file(Path(__file__)),
            "wrapper_sha256": sha256_file(HERE / "convlstm_mtl_model.py"),
            "runner_sha256": sha256_file(PROJECT_ROOT / "experiments/exp_r2_mtl_baselines.py"),
            "backbone_sha256": sha256_file(PROJECT_ROOT / "models/convlstm_model.py"),
            "config_sha256": sha256_json(run_config), "strategy": strategy, "seed": int(seed), "scenario": scenario, "tiny": bool(tiny)}


# ---------------------------------------------------------------- completion marker / resume
def _file_desc(path: Path) -> dict:
    d = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if path.suffix == ".npy":
        arr = _ORIG_NP_LOAD(path, mmap_mode="r")
        d["shape"] = list(arr.shape)
    return d


def validate_cell(run_dir: Path, scenario: str) -> tuple[bool, dict]:
    checks = {}
    try:
        preds = _ORIG_NP_LOAD(run_dir / "preds.npy"); preds_cal = _ORIG_NP_LOAD(run_dir / "preds_cal.npy")
        y_cal = _ORIG_NP_LOAD(run_dir / "y_cal.npy"); y_test = _ORIG_NP_LOAD(run_dir / "y_test.npy")
        metrics = json.loads((run_dir / "metrics.json").read_text()); fit = json.loads((run_dir / "fit.json").read_text())
        checks["preds_shape_finite"] = preds.ndim == 2 and preds.shape[1] == 2 and preds.shape[0] == y_test.shape[0] and bool(np.isfinite(preds).all())
        checks["preds_cal_shape_finite"] = preds_cal.ndim == 2 and preds_cal.shape[1] == 2 and preds_cal.shape[0] == y_cal.shape[0] and bool(np.isfinite(preds_cal).all())
        checks["metrics_finite"] = all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))
        if scenario == "L1":
            h = fit["history"]; checks["history"] = len(h) == fit["epochs"] and 1 <= fit["best_epoch"] <= fit["epochs"]
            checks["aux_state"] = "aux_state" in fit
        else:
            sel = fit["inner_selection"]; ref = fit["full_calibration_refit"]
            checks["history"] = (len(sel["history"]) == sel["epochs"] and 1 <= sel["best_epoch"] <= sel["epochs"]
                                 and len(ref["history"]) == fit["selected_epochs"] == ref["epochs"])
            checks["aux_state"] = "aux_state" in sel and "aux_state" in ref
            checks["inner_split_file"] = (run_dir / "inner_split_indices.npz").exists()
        ck = torch.load(run_dir / "model.pth", map_location="cpu", weights_only=False)
        checks["model_pth"] = "model_state_dict" in ck and "aux_state" in ck and "run_config" in ck
    except Exception as exc:  # any missing/corrupt file -> incomplete
        checks["exception"] = repr(exc)
        return False, checks
    return all(v for k, v in checks.items() if k != "exception"), checks


def write_complete(run_dir: Path, scenario: str, identity: dict) -> dict:
    ok, checks = validate_cell(run_dir, scenario)
    if not ok:
        raise RuntimeError(f"cell validation failed for {run_dir}: {checks}")
    files = {name: _file_desc(run_dir / name) for name in REQUIRED_FILES}
    if scenario != "L1":
        files["inner_split_indices.npz"] = _file_desc(run_dir / "inner_split_indices.npz")
    marker = {"written": now(), "identity": identity, "files": files, "checks": checks, "git_head": git_head()}
    tmp = run_dir / "COMPLETE.json.tmp"
    with open(tmp, "w") as fh:
        json.dump(marker, fh, indent=2, default=str)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, run_dir / "COMPLETE.json")
    return marker


def required_manifest(scenario: str) -> set:
    files = set(REQUIRED_FILES)
    if scenario != "L1":
        files.add("inner_split_indices.npz")
    return files


def scenario_status(run_dir: Path, identity: dict) -> tuple[str, dict]:
    """'absent' | 'complete' | 'mismatch' | 'incomplete'.

    complete requires: readable marker with identity, files and checks; identity equal to the current run; the file manifest
    equal to the required set for the scenario; every check true; every listed file present with the recorded sha256.
    mismatch: a valid marker whose identity differs (refused by the driver, never retrained). Anything else: incomplete."""
    if not run_dir.exists():
        return "absent", {}
    marker_path = run_dir / "COMPLETE.json"
    if not marker_path.exists():
        return "incomplete", {"reason": "no COMPLETE.json"}
    try:
        marker = json.loads(marker_path.read_text())
    except Exception as exc:
        return "incomplete", {"reason": f"unreadable marker: {exc!r}"}
    if not all(isinstance(marker.get(k), dict) for k in ("identity", "files", "checks")):
        return "incomplete", {"reason": "marker lacks identity/files/checks"}
    diff = {k: (marker["identity"].get(k), v) for k, v in identity.items() if marker["identity"].get(k) != v}
    if diff:
        return "mismatch", {"differs": diff}
    required = required_manifest(identity["scenario"])
    if set(marker["files"]) != required:
        return "incomplete", {"reason": f"file manifest differs from the required set: missing {sorted(required - set(marker['files']))}, extra {sorted(set(marker['files']) - required)}"}
    failed = [k for k, v in marker["checks"].items() if k != "exception" and v is not True]
    if failed or "exception" in marker["checks"]:
        return "incomplete", {"reason": f"marker checks not all true: {failed or marker['checks'].get('exception')}"}
    for name, desc in marker["files"].items():
        p = run_dir / name
        if not p.exists() or sha256_file(p) != desc.get("sha256"):
            return "incomplete", {"reason": f"file missing or changed: {name}"}
    return "complete", marker


def archive_incomplete(run_dir: Path) -> Path:
    target = run_dir.parent / f"{run_dir.name}_incomplete_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.rename(target)
    logger.warning(f"incomplete cell archived (renamed, not deleted): {run_dir} -> {target}")
    return target


# ---------------------------------------------------------------- datasets
def build_l1_datasets(data: dict, tiny: bool):
    train = TensorDataset(data["Xt_train_n"], data["Xs_train_n"], data["y_train_n"])
    val = TensorDataset(data["Xt_val_n"], data["Xs_val_n"], data["y_val_n"])
    test = TensorDataset(data["Xt_test_n"], data["Xs_test_n"], data["y_test"])
    n_val, n_test = len(val), len(test)
    if tiny:
        train = Subset(train, range(TINY["train"])); val = Subset(val, range(TINY["val"])); test = Subset(test, range(TINY["test"]))
        n_val, n_test = TINY["val"], TINY["test"]
    y_cal = data["y_val"][:n_val]; y_test = data["y_test"][:n_test]
    return train, val, test, y_cal, y_test


def build_shift_datasets(split_data: dict, test_data: dict, seed: int, tiny: bool):
    """Verbatim frozen make_shift_datasets logic (RandomState(seed) permutation, 80/20) with tiny subsets when requested."""
    n_samples = len(split_data["y"])
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_end = max(1, int(0.8 * n_samples))
    train_idx, val_idx = indices[:train_end], indices[train_end:]
    if len(val_idx) == 0:
        val_idx = train_idx[-1:]
    full_idx = np.arange(n_samples)
    test_limit = len(test_data["y"])
    if tiny:
        train_idx = train_idx[:TINY["shift_cal"]]; val_idx = val_idx[:max(1, TINY["shift_cal"] // 3)]
        full_idx = np.sort(np.concatenate([train_idx, val_idx])); test_limit = min(TINY["shift_test"], test_limit)
    train = TensorDataset(split_data["Xt_n"][train_idx], split_data["Xs_n"][train_idx], split_data["y_n"][train_idx])
    val = TensorDataset(split_data["Xt_n"][val_idx], split_data["Xs_n"][val_idx], split_data["y_n"][val_idx])
    full = TensorDataset(split_data["Xt_n"][full_idx], split_data["Xs_n"][full_idx], split_data["y_n"][full_idx])
    test = TensorDataset(test_data["Xt_n"][:test_limit], test_data["Xs_n"][:test_limit], test_data["y"][:test_limit])
    idx = {"permutation": indices, "train_idx": train_idx, "val_idx": val_idx, "full_idx": full_idx, "n_samples": n_samples, "tiny": tiny}
    return train, val, full, test, test_data["y"][:test_limit], split_data["y"][full_idx], idx


# ---------------------------------------------------------------- training of one (strategy, seed)
def run_pair(strategy: str, seed: int, data: dict, config: dict, proto: dict, out_dir: Path, args) -> list[dict]:
    tstrat = trainer_strategy(strategy)
    max_epochs = TINY["max_epochs"] if args.tiny else 200
    patience = TINY["patience"] if args.tiny else 20
    fine_tune_epochs = TINY["fine_tune_epochs"] if args.tiny else 50
    batch = TINY["batch"] if args.tiny else config["batch_size"]
    run_config = {**config, "strategy": strategy, "trainer_strategy": tstrat, "seed": seed, "max_epochs": max_epochs, "patience": patience,
                  "fine_tune_epochs": fine_tune_epochs, "batch_size": batch, "tiny": bool(args.tiny), "protocol_id": proto["protocol_id"],
                  "fine_tune_protocol": "80/20 shifted-calibration inner split selects the epoch count; the L1 checkpoint is then refit for that count on the complete shifted calibration set; test labels remain unseen",
                  "precision": "FP32 for uncertainty weighting, GradNorm, and PCGrad; AMP for vanilla and MMoE (CPU: FP32)"}
    strategy_dir = out_dir / strategy / f"seed_{seed}"
    rows = []
    # ---------------- L1
    l1_dir = strategy_dir / "L1"
    ident = identity_fields(proto, run_config, strategy, seed, "L1", args.tiny)
    status, info = scenario_status(l1_dir, ident)
    if status == "mismatch":
        raise SystemExit(f"REFUSED: {l1_dir} is complete under a different identity {info}; not retraining")
    if status == "incomplete":
        archive_incomplete(l1_dir)
    R.set_seed(seed)
    l1_model = make_model(strategy, config)
    n_params = count_parameters(l1_model)
    if status == "complete":
        checkpoint = R.load_checkpoint(l1_dir / "model.pth", l1_model, args.device)
        rows.append(json.loads((l1_dir / "metrics.json").read_text())); logger.info(f"{strategy} seed={seed} L1 complete: skipped")
    else:
        set_phase("L1_prepare")
        train_ds, val_ds, test_ds, y_cal, y_test = build_l1_datasets(data, args.tiny)
        set_phase("L1_fit")
        _FIT_INPUTS.append({"stage": "L1_fit", "strategy": strategy, "seed": seed, "train_split": "train", "n_train": len(train_ds), "val_split": "val", "n_val": len(val_ds)})
        trainer = R.ConflictAwareTrainer(l1_model, tstrat, args.device, config["lr"], config["weight_decay"], batch, max_epochs, patience, args.num_workers,
                                         base_task_weights=(config["y1_weight"], config["y2_weight"]))
        fit_info = trainer.fit(train_ds, val_ds)
        set_phase("evaluate")
        predictions = data["scaler_y"].inverse_transform(trainer.predict(test_ds))
        preds_cal = data["scaler_y"].inverse_transform(trainer.predict(val_ds))
        metrics = R.evaluate(y_test, predictions)
        metrics.update({"scenario": "L1", "strategy": strategy, "seed": seed, "n_test": len(y_test), "train_time_s": fit_info["elapsed_s"], "parameter_count": n_params})
        R.save_run(l1_dir, trainer, fit_info, predictions, metrics, {**run_config, "parameter_count": n_params})
        np.save(l1_dir / "preds_cal.npy", preds_cal); np.save(l1_dir / "y_cal.npy", y_cal); np.save(l1_dir / "y_test.npy", y_test)
        write_complete(l1_dir, "L1", ident)
        checkpoint = {"model_state_dict": copy.deepcopy(trainer.model.state_dict()), "aux_state": trainer.aux_state()}
        rows.append(metrics); logger.info(f"{strategy} seed={seed} L1 done: {metrics['y1_MAE']:.5f}/{metrics['y2_MAE']:.4f}, {fit_info['elapsed_s']:.0f}s")
        del trainer
    # ---------------- L2 / L3
    for scenario, (cal_split, test_split) in R.SCENARIOS.items():
        run_dir = strategy_dir / scenario
        ident = identity_fields(proto, run_config, strategy, seed, scenario, args.tiny)
        status, info = scenario_status(run_dir, ident)
        if status == "mismatch":
            raise SystemExit(f"REFUSED: {run_dir} is complete under a different identity {info}; not retraining")
        if status == "complete":
            rows.append(json.loads((run_dir / "metrics.json").read_text())); logger.info(f"{strategy} seed={seed} {scenario} complete: skipped"); continue
        if status == "incomplete":
            archive_incomplete(run_dir)
        R.set_seed(seed)
        set_phase(f"{scenario}_cal_load"); ft_data = R.load_shift_split(cal_split, data)
        set_phase("evaluate_load"); test_data = R.load_shift_split(test_split, data)
        train_ds, val_ds, full_ds, test_ds, y_true, y_cal, idx = build_shift_datasets(ft_data, test_data, seed, args.tiny)
        set_phase(f"{scenario}_select")
        _FIT_INPUTS.append({"stage": f"{scenario}_select", "strategy": strategy, "seed": seed, "train_split": cal_split + "[inner 80%]", "n_train": len(train_ds), "val_split": cal_split + "[inner 20%]", "n_val": len(val_ds)})
        selection_model = make_model(strategy, config); selection_model.load_state_dict(checkpoint["model_state_dict"])
        selection_trainer = R.ConflictAwareTrainer(selection_model, tstrat, args.device, config["lr"] / 10.0, config["weight_decay"],
                                                   min(batch, max(16, len(train_ds))), fine_tune_epochs, min(patience, 10), args.num_workers,
                                                   base_task_weights=(config["y1_weight"], config["y2_weight"]))
        selection_trainer.load_aux_state(checkpoint.get("aux_state", {}))
        selection_info = selection_trainer.fit(train_ds, val_ds)
        selected_epochs = int(selection_info["best_epoch"])
        del selection_trainer, selection_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        R.set_seed(seed)
        set_phase(f"{scenario}_refit")
        _FIT_INPUTS.append({"stage": f"{scenario}_refit", "strategy": strategy, "seed": seed, "train_split": cal_split + "[full]", "n_train": len(full_ds), "epochs": selected_epochs})
        model = make_model(strategy, config); model.load_state_dict(checkpoint["model_state_dict"])
        trainer = R.ConflictAwareTrainer(model, tstrat, args.device, config["lr"] / 10.0, config["weight_decay"], min(batch, max(16, len(full_ds))),
                                         selected_epochs, selected_epochs, args.num_workers, base_task_weights=(config["y1_weight"], config["y2_weight"]))
        trainer.load_aux_state(checkpoint.get("aux_state", {}))
        refit_info = trainer.fit_fixed_epochs(full_ds, selected_epochs)
        fit_info = {"inner_selection": selection_info, "full_calibration_refit": refit_info, "selected_epochs": selected_epochs,
                    "n_inner_train": len(train_ds), "n_inner_validation": len(val_ds), "n_full_calibration": len(full_ds),
                    "elapsed_s": selection_info["elapsed_s"] + refit_info["elapsed_s"], "aux_state_source": "L1 checkpoint aux_state loaded into both the selection and the refit trainer"}
        set_phase("evaluate")
        predictions = data["scaler_y"].inverse_transform(trainer.predict(test_ds))
        preds_cal = data["scaler_y"].inverse_transform(trainer.predict(full_ds))
        metrics = R.evaluate(y_true, predictions)
        metrics.update({"scenario": scenario, "strategy": strategy, "seed": seed, "n_calibration": len(full_ds), "n_test": len(y_true),
                        "selected_fine_tune_epochs": selected_epochs, "train_time_s": fit_info["elapsed_s"], "parameter_count": n_params})
        R.save_run(run_dir, trainer, fit_info, predictions, metrics, {**run_config, "parameter_count": n_params})
        np.save(run_dir / "preds_cal.npy", preds_cal); np.save(run_dir / "y_cal.npy", y_cal); np.save(run_dir / "y_test.npy", y_true)
        np.savez(run_dir / "inner_split_indices.npz", **idx)
        write_complete(run_dir, scenario, ident)
        rows.append(metrics); logger.info(f"{strategy} seed={seed} {scenario} done: {metrics['y1_MAE']:.5f}/{metrics['y2_MAE']:.4f} (sel {selected_epochs} ep)")
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- phases
def phase_describe(args, proto: dict) -> None:
    rep = {"time": now(), "protocol_id": proto["protocol_id"], "protocol_sha256": proto["_sha256"]}
    set_phase("load"); data = R.load_fixed_data()
    config = backbone_config(proto, data)
    counts = {}
    for strategy in ("vanilla", "mmoe"):
        R.set_seed(42); m = make_model(strategy, config); counts[strategy] = count_parameters(m)
    rep.update({"config": config, "parameter_counts": counts, "mmoe_head_delta": counts["mmoe"] - counts["vanilla"],
                "expected": proto["backbone_configuration"]["expected_parameter_count"], "n_static": data["n_static"], "n_features": data["n_features"], "n_generators": data["n_generators"]})
    write_json(args.out_dir / "describe.json", rep)
    print(json.dumps({k: rep[k] for k in ("parameter_counts", "mmoe_head_delta", "n_static")}, indent=1))


def phase_train(args, proto: dict) -> None:
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    set_phase("load"); t0 = time.time(); data = R.load_fixed_data(); data_load_seconds = time.time() - t0
    config = backbone_config(proto, data)
    (out_dir / "provenance").mkdir(exist_ok=True)
    np.savez(out_dir / "provenance/scalers.npz", t_mean=data["sc_t"].mean_, t_scale=data["sc_t"].scale_, s_mean=data["sc_s"].mean_, s_scale=data["sc_s"].scale_,
             y_mean=data["scaler_y"].mean_, y_scale=data["scaler_y"].scale_, static_idx=np.asarray(data["static_idx"]))
    write_json(out_dir / "protocol_copy.json", {k: v for k, v in proto.items() if not k.startswith("_")})
    seeds = args.seeds or list(SEEDS); strategies = args.strategies or list(STRATEGIES)
    rows = []
    for seed in seeds:
        for strategy in strategies:
            rows.extend(run_pair(strategy, seed, data, config, proto, out_dir, args))
    counts = {}
    for strategy in STRATEGIES:
        R.set_seed(42); counts[strategy] = count_parameters(make_model(strategy, config))
    n_expected = len(SEEDS) * len(STRATEGIES) * 3
    complete = sum(1 for s in STRATEGIES for sd in SEEDS for sc in ("L1", "L2", "L3") if (out_dir / s / f"seed_{sd}" / sc / "COMPLETE.json").exists())
    prov = {"time": now(), "phase": "train", "tiny": bool(args.tiny), "tiny_sizes": TINY if args.tiny else None, "protocol_id": proto["protocol_id"], "protocol_sha256": proto["_sha256"],
            "driver_sha256": sha256_file(Path(__file__)), "wrapper_sha256": sha256_file(HERE / "convlstm_mtl_model.py"), "runner_sha256": sha256_file(PROJECT_ROOT / "experiments/exp_r2_mtl_baselines.py"),
            "backbone_sha256": sha256_file(PROJECT_ROOT / "models/convlstm_model.py"), "git_head": git_head(), "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "device": args.device, "num_workers": args.num_workers, "seeds": seeds, "strategies": strategies, "parameter_counts": counts, "data_load_seconds": data_load_seconds,
            "files_read": _READS, "fit_inputs": _FIT_INPUTS, "complete_cells": complete, "expected_cells": n_expected, "scalers": "provenance/scalers.npz",
            "files_read_sha256": {p: sha256_file(Path(p)) for p in sorted(set(r["path"] for r in _READS if r["path"].startswith(str(DATA_ROOT))))}}
    write_json(out_dir / "provenance.json", prov)
    if complete == n_expected and not (args.seeds or args.strategies):
        R.aggregate(rows, out_dir, {**config, "strategies": list(STRATEGIES), "seeds": list(SEEDS), "max_epochs": 200 if not args.tiny else TINY["max_epochs"],
                                    "patience": 20 if not args.tiny else TINY["patience"], "fine_tune_epochs": 50 if not args.tiny else TINY["fine_tune_epochs"],
                                    "test_selection": False, "protocol_id": proto["protocol_id"], "parameter_counts": counts})
        logger.info(f"grid complete: {complete}/{n_expected} cells; aggregates written")
    else:
        logger.info(f"grid not complete: {complete}/{n_expected} cells (aggregation deferred)")


def phase_postprocess(args, proto: dict) -> None:
    import matched_postprocessing as MP
    MP.run(args.out_dir, proto, wide_final_dir=args.wide_final_dir, tiny=args.tiny, expected_cells=args.expected_cells,
           wide_seeds=tuple(args.wide_seeds), wide_expected_cells=args.wide_expected_cells, strategies=STRATEGIES, seeds=SEEDS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("describe", "train", "postprocess"), required=True)
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="resume restriction only (unfinished cells of the prescribed grid)")
    ap.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=None, help="resume restriction only")
    ap.add_argument("--tiny", action="store_true", help="bounded CPU integration smoke (implementation check, not a result)")
    ap.add_argument("--wide-final-dir", type=Path, default=None, help="postprocess: tuned widened control final directory for --infer")
    ap.add_argument("--expected-cells", type=int, default=45, help="postprocess: required number of valid block cells (5 strategies x 3 seeds x 3 scenarios)")
    ap.add_argument("--wide-seeds", nargs="+", type=int, default=[42, 52, 62], help="postprocess: declared seeds of the tuned widened final phase")
    ap.add_argument("--wide-expected-cells", type=int, default=9, help="postprocess: required number of widened cells (3 seeds x 3 scenarios); the tiny synthetic fixture uses its own smaller grid")
    args = ap.parse_args()
    proto = load_protocol(args.protocol)
    if args.out_dir is None:
        args.out_dir = PROJECT_ROOT / (proto["outputs"]["smoke_root"] if args.tiny else proto["outputs"]["root"])
    if args.tiny:
        args.device = "cpu"; args.num_workers = 0
        if not str(args.out_dir).endswith("_smoke"):
            args.out_dir = Path(str(args.out_dir) + "_smoke")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for never in proto["never_write"]:
        if str(args.out_dir.resolve()).startswith(str((PROJECT_ROOT / never).resolve())):
            raise RuntimeError(f"output directory {args.out_dir} lies inside a frozen path {never}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.load = _tracked_load  # read tracker (the frozen runner shares the numpy module object)
    logger.add(args.out_dir / f"{args.phase}.log", enqueue=True)
    logger.info(f"phase={args.phase} out_dir={args.out_dir} protocol={args.protocol} tiny={args.tiny}")
    {"describe": phase_describe, "train": phase_train, "postprocess": phase_postprocess}[args.phase](args, proto)


if __name__ == "__main__":
    main()
