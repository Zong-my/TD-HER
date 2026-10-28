#!/usr/bin/env python3
"""Independent hyperparameter tuning of the widened (capacity-matched) control.

Additive research phase 2026-10-14, task 3. This driver imports
``experiments/exp_r2_mtl_baselines.py`` unchanged (same data loading rules,
model, loss, optimizer, scheduler, early stopping and L2/L3 fine-tuning
protocol) and never writes to the frozen result directory
``results/ieee39/resubmission_v3_mtl_capacity_matched``.

Phases
  describe    resolve the protocol, build the model once, print the parameter
              count and the search space (no data arrays loaded)
  search      Optuna TPE over optimizer-side hyperparameters with the widened
              architecture, batch size and warm-up fixed; objective = best
              validation mean standardized SmoothL1 (the runner's own
              checkpoint-selection metric). Loads ONLY the L1 train and
              validation splits.
  robustness  retrain selected trial configurations with additional seeds on
              train/val only and freeze the configuration with the best mean
              validation objective (tuned_config.json)
  final       run the frozen tuned configuration through the unchanged runner
              functions (M.run_strategy) for the listed seeds, including the
              L2/L3 fine-tuning protocol; test labels are read only by the
              evaluation inside the runner

Usage (from the repository root, fepfm interpreter):
  python experiments/resubmission_v4/exp_v4_wide_control_tuning.py --phase describe
  python experiments/resubmission_v4/exp_v4_wide_control_tuning.py --phase search --smoke --n-trials 2
  python experiments/resubmission_v4/exp_v4_wide_control_tuning.py --phase search --n-trials 50
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
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import experiments.exp_r2_mtl_baselines as M  # noqa: E402

ARTIFACT_ROOT = WORK_ROOT
DEFAULT_PROTOCOL = ARTIFACT_ROOT / "protocols/wide_control_tuning.json"
DEFAULT_OUT = work_path('results/ieee39/resubmission_v4_wide_control_tuned_v1')
FROZEN_CONTROL_DIR = work_path('results/ieee39/resubmission_v3_mtl_capacity_matched')
RUNNER_PATH = PROJECT_ROOT / "experiments/exp_r2_mtl_baselines.py"
DRIVER_PATH = Path(__file__).resolve()

# --------------------------------------------------------------------------
# Provenance helpers and the no-test-array guard
# --------------------------------------------------------------------------
_FILES_READ: list[str] = []
_NP_LOAD = np.load


def _tracked_np_load(file, *args, **kwargs):
    _FILES_READ.append(str(file))
    return _NP_LOAD(file, *args, **kwargs)


np.load = _tracked_np_load  # affects the runner module too (same numpy module)


def assert_no_test_reads() -> None:
    offenders = [p for p in _FILES_READ if "test" in Path(p).name]
    if offenders:
        raise RuntimeError(f"test arrays were read in a train/val-only phase: {offenders}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def runtime_provenance(protocol: dict) -> dict:
    import optuna
    import sklearn
    info = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "optuna": optuna.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "git_head": git_head(),
        "runner_path": str(RUNNER_PATH.relative_to(PROJECT_ROOT)),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "driver_path": str(DRIVER_PATH.relative_to(PROJECT_ROOT)),
        "driver_sha256": sha256_file(DRIVER_PATH),
        "protocol_id": protocol.get("protocol_id"),
    }
    expected = protocol.get("runner", {}).get("sha256")
    if expected and expected != info["runner_sha256"]:
        raise RuntimeError(
            "runner hash differs from the protocol freeze: "
            f"{info['runner_sha256']} != {expected}"
        )
    return info


def snapshot_dir(path: Path) -> dict:
    """Listing with sizes and mtimes, used to prove a frozen directory is untouched."""
    out = {}
    if path.exists():
        for p in sorted(path.rglob("*")):
            if p.is_file():
                st = p.stat()
                out[str(p.relative_to(path))] = [st.st_size, int(st.st_mtime)]
    return out


# --------------------------------------------------------------------------
# Protocol and configuration
# --------------------------------------------------------------------------
def load_protocol(path: Path) -> dict:
    with open(path) as handle:
        protocol = json.load(handle)
    for key in ("fixed", "search_space", "published_configuration", "search",
                "robustness", "final", "never_write"):
        if key not in protocol:
            raise KeyError(f"protocol lacks '{key}'")
    return protocol


def check_published_values(protocol: dict) -> None:
    """The enqueued trial must equal the optimizer-side values the frozen control used."""
    with open(work_path('results/ieee39/exp1/PatchTST/best_params.json')) as handle:
        best = json.load(handle)
    with open(FROZEN_CONTROL_DIR / "protocol.json") as handle:
        frozen = json.load(handle)
    pub = protocol["published_configuration"]
    pairs = {
        "lr": (best["lr"], frozen["lr"]),
        "weight_decay": (best["weight_decay"], frozen["weight_decay"]),
        "dropout": (best["dropout"], frozen["model"]["dropout"]),
        "y1_weight": (best["y1_weight"], frozen["y1_weight"]),
        "y2_weight": (best["y2_weight"], frozen["y2_weight"]),
    }
    for key, (a, b) in pairs.items():
        if not (abs(pub[key] - a) < 1e-12 and abs(pub[key] - b) < 1e-12):
            raise RuntimeError(f"published value mismatch for {key}: {pub[key]} vs {a} / {b}")
    fixed_model = dict(protocol["fixed"]["model"])
    frozen_model = {k: v for k, v in frozen["model"].items() if k != "dropout"}
    if fixed_model != frozen_model:
        raise RuntimeError(f"fixed architecture differs from the frozen control: {fixed_model} vs {frozen_model}")
    if protocol["fixed"]["batch_size"] != frozen["batch_size"]:
        raise RuntimeError("batch size differs from the frozen control")


def base_config(protocol: dict) -> dict:
    fixed = protocol["fixed"]
    return {
        "model": dict(fixed["model"]),
        "batch_size": int(fixed["batch_size"]),
        "warmup_epochs": int(fixed["warmup_epochs"]),
        "strategy": fixed["strategy"],
        "task_scaling": "targets standardized on the L1 training split",
        "selection_metric": "mean normalized validation SmoothL1 loss",
        "scheduler": fixed["scheduler"],
        "protocol_id": protocol["protocol_id"],
    }


def config_from_params(protocol: dict, params: dict) -> dict:
    cfg = base_config(protocol)
    cfg["model"]["dropout"] = float(params["dropout"])
    cfg["lr"] = float(params["lr"])
    cfg["weight_decay"] = float(params["weight_decay"])
    cfg["y1_weight"] = float(params["y1_weight"])
    cfg["y2_weight"] = float(params["y2_weight"])
    return cfg


def suggest_params(trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        if spec["type"] != "float":
            raise ValueError(f"unsupported search-space type for {name}: {spec['type']}")
        params[name] = trial.suggest_float(
            name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
        )
    return params


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------
# Train/validation-only data loading (mirror of M.load_fixed_data without test)
# --------------------------------------------------------------------------
def load_search_data() -> dict:
    """Same preprocessing as M.load_fixed_data, restricted to the train and val splits."""
    from sklearn.preprocessing import StandardScaler
    rep_b = Path(M.REP_DIR) / "repB" / f"ms{M.MS}"
    rep_a = Path(M.REP_DIR) / "repA" / f"ms{M.MS}"
    splits = ("train", "val")
    temporal = {s: np.load(rep_b / f"X_temporal_{s}.npy") for s in splits}
    static = {s: np.load(rep_b / f"X_static_{s}.npy") for s in splits}
    targets = {s: np.load(rep_a / f"y_{s}.npy") for s in splits}
    with open(rep_b / "meta.json") as handle:
        static_names = json.load(handle)["static_names"]
    with open(M.EXP1_DIR / "mrmr_static_features.json") as handle:
        selected = json.load(handle)["union"]
    static_idx = [static_names.index(name) for name in selected]

    train_temporal = temporal["train"]
    n_samples, n_steps, n_nodes, n_channels = train_temporal.shape
    temporal_scaler = StandardScaler().fit(train_temporal.reshape(-1, n_nodes * n_channels))
    static_scaler = StandardScaler().fit(static["train"][:, static_idx])
    target_scaler = StandardScaler().fit(targets["train"])

    result = {
        "sc_t": temporal_scaler,
        "sc_s": static_scaler,
        "scaler_y": target_scaler,
        "static_idx": static_idx,
        "n_timesteps": n_steps,
        "n_generators": n_nodes,
        "n_features": n_channels,
        "n_static": len(static_idx),
        "n_train": int(n_samples),
        "n_val": int(len(targets["val"])),
    }
    for s in splits:
        values = temporal[s]
        shape = values.shape
        result[f"Xt_{s}_n"] = temporal_scaler.transform(
            values.reshape(-1, n_nodes * n_channels)
        ).reshape(shape).astype(np.float32)
        result[f"Xs_{s}_n"] = static_scaler.transform(static[s][:, static_idx]).astype(np.float32)
        result[f"y_{s}"] = targets[s]
        result[f"y_{s}_n"] = target_scaler.transform(targets[s]).astype(np.float32)
    assert_no_test_reads()
    return result


def make_search_datasets(data: dict, subsample: int):
    train = M.TensorDataset(data["Xt_train_n"], data["Xs_train_n"], data["y_train_n"])
    val = M.TensorDataset(data["Xt_val_n"], data["Xs_val_n"], data["y_val_n"])
    if subsample > 0:
        train = M.Subset(train, range(min(subsample, len(train))))
        val = M.Subset(val, range(min(max(subsample // 4, 1), len(val))))
    return train, val


# --------------------------------------------------------------------------
# Trainer with Optuna pruning hooks (training loop identical to the runner)
# --------------------------------------------------------------------------
class PrunableTrainer(M.ConflictAwareTrainer):
    """Same loop as M.ConflictAwareTrainer.fit plus trial.report / should_prune."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list[dict] = []
        self._best_loss = float("inf")
        self._best_epoch = None
        self._start = None

    def partial_info(self) -> dict:
        return {
            "best_val_loss": None if not np.isfinite(self._best_loss) else float(self._best_loss),
            "best_epoch": self._best_epoch,
            "epochs": len(self._history),
            "elapsed_s": (time.time() - self._start) if self._start else None,
            "history": self._history,
        }

    def fit(self, train_dataset, val_dataset, trial=None) -> dict:  # noqa: D401
        import optuna
        train_loader = M.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )
        best_model = None
        best_aux = None
        stale_epochs = 0
        self._history = []
        self._best_loss = float("inf")
        self._best_epoch = None
        self._start = time.time()

        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            train_loss, train_tasks = self._train_epoch(train_loader)
            val_loss, val_tasks = self.evaluate_loss(val_dataset)
            self.scheduler.step()
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_y1": train_tasks[0],
                "train_y2": train_tasks[1],
                "val_loss": val_loss,
                "val_y1": val_tasks[0],
                "val_y2": val_tasks[1],
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.time() - epoch_start,
                **self.aux_state(),
            }
            self._history.append(row)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(
                    f"{self.strategy} epoch={epoch + 1} train={train_loss:.6f} "
                    f"val={val_loss:.6f} ({row['epoch_seconds']:.1f}s)"
                )

            if val_loss < self._best_loss - 1e-8:
                self._best_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                best_aux = copy.deepcopy(self.aux_state())
                self._best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if best_model is None:
            raise RuntimeError("Training did not produce a finite checkpoint")
        self.model.load_state_dict(best_model)
        self.load_aux_state(best_aux)
        return {
            "best_val_loss": float(self._best_loss),
            "best_epoch": self._best_epoch,
            "epochs": len(self._history),
            "elapsed_s": time.time() - self._start,
            "history": self._history,
            "aux_state": best_aux,
        }


def train_one(data: dict, cfg: dict, seed: int, args, protocol: dict,
              train_ds, val_ds, trial=None) -> tuple[dict, dict]:
    """Train one configuration on train/val; returns (fit_info, run_record)."""
    import optuna
    M.set_seed(seed)
    model = M.make_model(data, cfg)
    n_params = count_parameters(model)
    expected = int(protocol["fixed"]["expected_parameter_count"])
    if n_params != expected:
        raise RuntimeError(f"parameter count {n_params} != expected {expected}")
    trainer = PrunableTrainer(
        model, cfg["strategy"], args.device, cfg["lr"], cfg["weight_decay"],
        cfg["batch_size"], args.max_epochs, args.patience, args.num_workers,
        base_task_weights=(cfg["y1_weight"], cfg["y2_weight"]),
        warmup_epochs=cfg["warmup_epochs"],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pruned = False
    try:
        fit_info = trainer.fit(train_ds, val_ds, trial=trial)
    except optuna.TrialPruned:
        pruned = True
        fit_info = trainer.partial_info()
    record = {
        "seed": seed,
        "state": "PRUNED" if pruned else "COMPLETE",
        "parameter_count": n_params,
        "best_val_loss": fit_info.get("best_val_loss"),
        "best_epoch": fit_info.get("best_epoch"),
        "epochs": fit_info.get("epochs"),
        "elapsed_s": fit_info.get("elapsed_s"),
        "mean_epoch_seconds": (
            float(np.mean([r["epoch_seconds"] for r in fit_info["history"]]))
            if fit_info.get("history") else None
        ),
        "peak_gpu_mem_allocated_mb": (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        ),
        "peak_gpu_mem_reserved_mb": (
            torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None
        ),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fit_info, record, pruned


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------
def phase_describe(args, protocol: dict) -> None:
    check_published_values(protocol)
    rep_b = Path(M.REP_DIR) / "repB" / f"ms{M.MS}"
    shape = np.load(rep_b / "X_temporal_val.npy", mmap_mode="r").shape
    with open(M.EXP1_DIR / "mrmr_static_features.json") as handle:
        n_static = len(json.load(handle)["union"])
    data = {"n_timesteps": shape[1], "n_generators": shape[2],
            "n_features": shape[3], "n_static": n_static}
    cfg = config_from_params(protocol, protocol["published_configuration"])
    model = M.make_model(data, cfg)
    n_params = count_parameters(model)
    print(json.dumps({
        "protocol_id": protocol["protocol_id"],
        "fixed_model": cfg["model"],
        "input_shape": {"n_timesteps": shape[1], "n_generators": shape[2],
                        "n_features": shape[3], "n_static": n_static},
        "parameter_count": n_params,
        "expected_parameter_count": protocol["fixed"]["expected_parameter_count"],
        "parameter_count_ok": n_params == protocol["fixed"]["expected_parameter_count"],
        "search_space": protocol["search_space"],
        "published_configuration_ok": True,
        "files_read": _FILES_READ,
        "provenance": runtime_provenance(protocol),
    }, indent=2))


def phase_search(args, protocol: dict) -> None:
    import optuna
    import pandas as pd
    check_published_values(protocol)
    out_dir = args.out_dir / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_before = snapshot_dir(FROZEN_CONTROL_DIR)
    provenance = runtime_provenance(protocol)
    provenance.update({"phase": "search", "smoke": bool(args.smoke),
                       "max_epochs": args.max_epochs, "patience": args.patience,
                       "subsample": args.subsample, "device": args.device,
                       "num_workers": args.num_workers})
    logger.info("loading train/val arrays only")
    t0 = time.time()
    data = load_search_data()
    provenance["data_load_seconds"] = time.time() - t0
    provenance["files_read"] = list(_FILES_READ)
    provenance["files_read_sha256"] = {
        Path(p).name: sha256_file(Path(p)) for p in dict.fromkeys(_FILES_READ)
    }
    provenance["n_train"] = data["n_train"]
    provenance["n_val"] = data["n_val"]
    write_json(out_dir / "provenance.json", provenance)
    write_json(out_dir / "protocol_copy.json", protocol)
    train_ds, val_ds = make_search_datasets(data, args.subsample)

    space = protocol["search_space"]
    study_name = protocol["protocol_id"] + ("_smoke" if args.smoke else "")
    storage = f"sqlite:///{out_dir / 'study.db'}"
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True,
    )
    started = [t for t in study.trials
               if t.state in (optuna.trial.TrialState.COMPLETE,
                              optuna.trial.TrialState.PRUNED,
                              optuna.trial.TrialState.RUNNING)]
    if not started:
        pub = {k: float(protocol["published_configuration"][k]) for k in space}
        study.enqueue_trial(pub, user_attrs={"origin": "published_configuration"})
        logger.info(f"enqueued the published configuration as trial 0: {pub}")

    total = args.total_trials if args.total_trials else int(protocol["search"]["total_trials"])
    if args.smoke:
        total = min(total, args.n_trials)

    def objective(trial):
        params = suggest_params(trial, space)
        cfg = config_from_params(protocol, params)
        trial_dir = out_dir / "trials" / f"trial_{trial.number:03d}"
        write_json(trial_dir / "params.json", params)
        write_json(trial_dir / "config.json", cfg)
        fit_info, record, pruned = train_one(
            data, cfg, args.seed, args, protocol, train_ds, val_ds, trial=trial
        )
        record["trial_number"] = trial.number
        write_json(trial_dir / "fit.json", {**record, "history": fit_info.get("history", [])})
        for key in ("best_epoch", "epochs", "elapsed_s", "mean_epoch_seconds",
                    "peak_gpu_mem_allocated_mb", "peak_gpu_mem_reserved_mb",
                    "parameter_count", "state"):
            trial.set_user_attr(key, record[key])
        assert_no_test_reads()
        if pruned:
            raise optuna.TrialPruned()
        return float(fit_info["best_val_loss"])

    def stop_when_done(study_, trial_):
        finished = [t for t in study_.trials
                    if t.state in (optuna.trial.TrialState.COMPLETE,
                                   optuna.trial.TrialState.PRUNED)]
        if len(finished) >= total:
            study_.stop()

    timeout = None if args.smoke else float(protocol["search"]["safety_timeout_hours"]) * 3600.0
    t_search = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout,
                   callbacks=[stop_when_done], gc_after_trial=True)
    elapsed = time.time() - t_search

    frame = study.trials_dataframe(attrs=("number", "value", "state", "params",
                                          "user_attrs", "datetime_start",
                                          "datetime_complete", "duration"))
    frame.to_csv(out_dir / "trials.csv", index=False)
    states = frame["state"].value_counts().to_dict() if len(frame) else {}
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = min(complete, key=lambda t: t.value) if complete else None
    summary = {
        "study_name": study_name,
        "smoke": bool(args.smoke),
        "trials_total_target": total,
        "trials_by_state": states,
        "this_invocation_seconds": elapsed,
        "best_trial": None if best is None else {
            "number": best.number, "value": best.value, "params": best.params,
            "user_attrs": best.user_attrs,
        },
        "frozen_control_dir_unchanged": snapshot_dir(FROZEN_CONTROL_DIR) == frozen_before,
        "test_arrays_read": [p for p in _FILES_READ if "test" in Path(p).name],
    }
    write_json(out_dir / "search_summary.json", summary)
    if best is not None:
        write_json(out_dir / "best_trial.json", summary["best_trial"])
    assert_no_test_reads()
    if not summary["frozen_control_dir_unchanged"]:
        raise RuntimeError("frozen control directory changed during the search")
    logger.info(json.dumps(summary, indent=2, default=str))


def phase_robustness(args, protocol: dict) -> None:
    import optuna
    import pandas as pd
    check_published_values(protocol)
    search_dir = args.out_dir / "search"
    out_dir = args.out_dir / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_before = snapshot_dir(FROZEN_CONTROL_DIR)
    study = optuna.load_study(
        study_name=protocol["protocol_id"] + ("_smoke" if args.smoke else ""),
        storage=f"sqlite:///{search_dir / 'study.db'}",
    )
    complete = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )
    if args.trial_numbers:
        chosen = [t for t in complete if t.number in set(args.trial_numbers)]
    else:
        chosen = complete[: int(protocol["robustness"]["top_k"])]
    if not chosen:
        raise RuntimeError("no completed trials to check")
    data = load_search_data()
    train_ds, val_ds = make_search_datasets(data, args.subsample)
    provenance = runtime_provenance(protocol)
    provenance.update({"phase": "robustness", "trials": [t.number for t in chosen],
                       "extra_seeds": args.seeds, "files_read": list(_FILES_READ)})
    write_json(out_dir / "provenance.json", provenance)

    rows = []
    for trial in chosen:
        cfg = config_from_params(protocol, trial.params)
        rows.append({"trial_number": trial.number, "seed": args.seed,
                     "best_val_loss": trial.value, "source": "search"})
        for seed in args.seeds:
            run_dir = out_dir / f"trial_{trial.number:03d}" / f"seed_{seed}"
            fit_path = run_dir / "fit.json"
            if fit_path.exists() and not args.force:
                with open(fit_path) as handle:
                    rec = json.load(handle)
            else:
                fit_info, rec, _ = train_one(data, cfg, seed, args, protocol,
                                             train_ds, val_ds, trial=None)
                write_json(run_dir / "config.json", cfg)
                write_json(fit_path, {**rec, "history": fit_info["history"]})
            rows.append({"trial_number": trial.number, "seed": seed,
                         "best_val_loss": rec["best_val_loss"], "source": "robustness"})
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "robustness_runs.csv", index=False)
    summary = frame.groupby("trial_number")["best_val_loss"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values("mean")
    summary.to_csv(out_dir / "robustness_summary.csv", index=False)
    winner = int(summary.iloc[0]["trial_number"])
    trial = next(t for t in chosen if t.number == winner)
    tuned = config_from_params(protocol, trial.params)
    tuned.update({
        "tuned_from_trial": winner,
        "three_seed_mean_val_objective": float(summary.iloc[0]["mean"]),
        "source": "independent Optuna search (protocol wide_control_tuning_v1) + three-seed validation robustness",
        "capacity_match": (
            "same widened PatchTST architecture as the frozen capacity-matched control; "
            "only optimizer-side hyperparameters (lr, weight decay, dropout, task weights) "
            "were tuned on the L1 training/validation splits"
        ),
    })
    write_json(out_dir / "tuned_config.json", tuned)
    assert_no_test_reads()
    if snapshot_dir(FROZEN_CONTROL_DIR) != frozen_before:
        raise RuntimeError("frozen control directory changed during robustness")
    logger.info(f"tuned configuration frozen from trial {winner}: {tuned}")


def phase_final(args, protocol: dict) -> None:
    """Frozen tuned configuration through the unchanged runner (incl. L2/L3)."""
    tuned_path = args.out_dir / "robustness" / "tuned_config.json"
    with open(tuned_path) as handle:
        tuned = json.load(handle)
    out_dir = args.out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_before = snapshot_dir(FROZEN_CONTROL_DIR)
    provenance = runtime_provenance(protocol)
    provenance.update({"phase": "final", "tuned_config_sha256": sha256_file(tuned_path)})
    write_json(out_dir / "provenance.json", provenance)

    config = {k: v for k, v in tuned.items() if k not in ("strategy", "warmup_epochs")}
    data = M.load_fixed_data()  # the runner's loader; test labels are used only by evaluation
    reference = M.make_model(data, config)
    config["parameter_count"] = count_parameters(reference)
    del reference
    runner_args = argparse.Namespace(
        device=args.device, max_epochs=args.max_epochs, patience=args.patience,
        fine_tune_epochs=args.fine_tune_epochs, num_workers=args.num_workers,
        smoke=args.smoke, force=args.force,
    )
    rows = []
    for seed in args.seeds:
        logger.info(f"final run: vanilla seed={seed}")
        rows.extend(M.run_strategy("vanilla", seed, data, data, config, out_dir, runner_args))
    M.aggregate(rows, out_dir, {
        **config, "strategies": ["vanilla"], "seeds": args.seeds,
        "max_epochs": args.max_epochs, "fine_tune_epochs": args.fine_tune_epochs,
        "test_selection": False,
        "protocol_id": protocol["protocol_id"],
    })
    if snapshot_dir(FROZEN_CONTROL_DIR) != frozen_before:
        raise RuntimeError("frozen control directory changed during the final phase")
    logger.info(f"final results written to {out_dir}")


# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("describe", "search", "robustness", "final"),
                        required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="training seed per search trial")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=20,
                        help="trials to run in this invocation (worker share)")
    parser.add_argument("--total-trials", type=int, default=0,
                        help="override the protocol's total trial count (0 = protocol)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="robustness: extra seeds; final: seeds to run")
    parser.add_argument("--trial-numbers", nargs="+", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--fine-tune-epochs", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=0,
                        help="debug only: cap train samples (0 = full data)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    fixed = protocol["fixed"]
    if args.max_epochs is None:
        args.max_epochs = int(fixed["max_epochs"])
    if args.patience is None:
        args.patience = int(fixed["patience"])
    if args.fine_tune_epochs is None:
        args.fine_tune_epochs = int(fixed["fine_tune_epochs"])
    if args.seeds is None:
        args.seeds = (list(protocol["robustness"]["extra_seeds"]) if args.phase == "robustness"
                      else list(protocol["final"]["seeds"]))
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = min(args.patience, 2)
        args.fine_tune_epochs = min(args.fine_tune_epochs, 2)
        if not str(args.out_dir).endswith("_smoke"):
            args.out_dir = Path(str(args.out_dir) + "_smoke")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for never in protocol["never_write"]:
        never_path = (PROJECT_ROOT / never).resolve()
        if str(args.out_dir.resolve()).startswith(str(never_path)):
            raise RuntimeError(f"output directory {args.out_dir} lies inside a frozen path {never}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(args.out_dir / f"{args.phase}.log", enqueue=True)
    logger.info(f"phase={args.phase} out_dir={args.out_dir} protocol={args.protocol}")
    {"describe": phase_describe, "search": phase_search,
     "robustness": phase_robustness, "final": phase_final}[args.phase](args, protocol)


if __name__ == "__main__":
    main()
