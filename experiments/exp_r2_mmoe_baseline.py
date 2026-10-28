#!/usr/bin/env python3
"""Architecture-level MMoE control for Reviewer 2 on PatchTST.

The experiment retains the submitted PatchTST encoder and replaces only its
two-output linear head with a canonical four-expert, two-gate MMoE head. It
uses the same L1 training/validation protocol, three seeds, target scaling,
optimizer budget, and calibration-only L2/L3 fine-tuning protocol as the
task-conflict-aware controls in ``exp_r2_mtl_baselines.py``.
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
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from exp_r2_mtl_baselines import (
    ConflictAwareTrainer,
    SCENARIOS,
    aggregate,
    evaluate,
    load_fixed_config,
    load_fixed_data,
    load_shift_split,
    make_l1_datasets,
    make_shift_datasets,
    save_run,
    set_seed,
)
from models.patchtst_mtl_model import PatchTSTConflictMTL, PatchTSTMMoE


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = work_path('results/ieee39/revision_r2_mmoe_exact')
ARCHITECTURE = {
    "name": "MMoE-head",
    "num_experts": 4,
    "expert_hidden": 64,
    "tower_hidden": 32,
}


def make_model(data: dict, config: dict) -> PatchTSTMMoE:
    kwargs = dict(
        n_timesteps=data["n_timesteps"],
        n_generators=data["n_generators"],
        n_features=data["n_features"],
        n_static=data["n_static"],
    )
    kwargs.update(config["model"])
    return PatchTSTMMoE(
        num_experts=ARCHITECTURE["num_experts"],
        expert_hidden=ARCHITECTURE["expert_hidden"],
        tower_hidden=ARCHITECTURE["tower_hidden"],
        **kwargs,
    )


def make_shared_model(data: dict, config: dict) -> PatchTSTConflictMTL:
    kwargs = dict(
        n_timesteps=data["n_timesteps"],
        n_generators=data["n_generators"],
        n_features=data["n_features"],
        n_static=data["n_static"],
    )
    kwargs.update(config["model"])
    return PatchTSTConflictMTL(**kwargs)


def architecture_audit(data: dict, config: dict) -> dict:
    set_seed(42)
    shared = make_shared_model(data, config)
    shared_state = shared.state_dict()
    set_seed(42)
    mmoe = make_model(data, config)
    mmoe_state = mmoe.state_dict()

    encoder_keys = sorted(
        key for key in shared_state if not key.startswith("fusion_out.")
    )
    missing = [key for key in encoder_keys if key not in mmoe_state]
    unequal = [
        key for key in encoder_keys
        if key in mmoe_state
        and not torch.equal(shared_state[key], mmoe_state[key])
    ]
    shared_count = int(sum(p.numel() for p in shared.parameters()))
    mmoe_count = int(sum(p.numel() for p in mmoe.parameters()))
    return {
        "seed": 42,
        "shared_encoder_key_count": len(encoder_keys),
        "shared_encoder_missing_keys": missing,
        "shared_encoder_unequal_keys": unequal,
        "shared_encoder_bitwise_equal": not missing and not unequal,
        "shared_head_parameter_count": shared_count,
        "mmoe_parameter_count": mmoe_count,
        "additional_parameters": mmoe_count - shared_count,
        "relative_parameter_increase_pct": 100.0 * (
            mmoe_count - shared_count
        ) / shared_count,
        "architecture": ARCHITECTURE,
    }


def trainer_for(model, config: dict, args, lr: float, batch_size: int,
                max_epochs: int, patience: int) -> ConflictAwareTrainer:
    return ConflictAwareTrainer(
        model=model,
        strategy="vanilla",
        device=args.device,
        lr=lr,
        weight_decay=config["weight_decay"],
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        num_workers=args.num_workers,
        base_task_weights=(config["y1_weight"], config["y2_weight"]),
    )


def run_seed(seed: int, data: dict, config: dict,
             out_dir: Path, args) -> list[dict]:
    seed_dir = out_dir / "mmoe" / f"seed_{seed}"
    l1_dir = seed_dir / "L1"
    l1_metrics_path = l1_dir / "metrics.json"
    run_config = {
        **config,
        "strategy": "mmoe",
        "optimization": "fixed submitted PatchTST task weights",
        "architecture": ARCHITECTURE,
        "seed": seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_protocol": (
            "80/20 shifted-calibration inner split selects the epoch count; "
            "the L1 checkpoint is then refit for that count on the complete "
            "shifted calibration set; test labels remain unseen"
        ),
        "precision": "AMP, matching the original PatchTST path",
    }

    set_seed(seed)
    l1_model = make_model(data, config)
    if l1_metrics_path.exists() and not args.force:
        with open(l1_metrics_path) as handle:
            l1_metrics = json.load(handle)
        checkpoint = torch.load(
            l1_dir / "model.pth", map_location=args.device,
            weights_only=False,
        )
        l1_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        train_ds, val_ds, test_ds = make_l1_datasets(data, args.smoke)
        trainer = trainer_for(
            l1_model, config, args, config["lr"], config["batch_size"],
            args.max_epochs, args.patience,
        )
        fit_info = trainer.fit(train_ds, val_ds)
        pred_scaled = trainer.predict(test_ds)
        predictions = data["scaler_y"].inverse_transform(pred_scaled)
        y_true = data["y_test"][:len(predictions)]
        l1_metrics = evaluate(y_true, predictions)
        l1_metrics.update({
            "scenario": "L1",
            "strategy": "mmoe",
            "seed": seed,
            "n_test": len(y_true),
            "train_time_s": fit_info["elapsed_s"],
        })
        save_run(
            l1_dir, trainer, fit_info, predictions, l1_metrics, run_config
        )
        checkpoint = {
            "model_state_dict": trainer.model.state_dict(),
            "aux_state": trainer.aux_state(),
        }

    rows = [l1_metrics]
    for scenario, (cal_split, test_split) in SCENARIOS.items():
        run_dir = seed_dir / scenario
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            with open(metrics_path) as handle:
                rows.append(json.load(handle))
            continue

        set_seed(seed)
        calibration = load_shift_split(cal_split, data)
        test_data = load_shift_split(test_split, data)
        train_ds, val_ds, full_ds, test_ds, y_true = make_shift_datasets(
            calibration, test_data, seed, args.smoke
        )

        selection_model = make_model(data, config)
        selection_model.load_state_dict(checkpoint["model_state_dict"])
        selection_trainer = trainer_for(
            selection_model,
            config,
            args,
            config["lr"] / 10.0,
            min(config["batch_size"], max(16, len(train_ds))),
            args.fine_tune_epochs,
            min(args.patience, 10),
        )
        selection_info = selection_trainer.fit(train_ds, val_ds)
        selected_epochs = int(selection_info["best_epoch"])
        del selection_trainer, selection_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        set_seed(seed)
        model = make_model(data, config)
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer = trainer_for(
            model,
            config,
            args,
            config["lr"] / 10.0,
            min(config["batch_size"], max(16, len(full_ds))),
            selected_epochs,
            selected_epochs,
        )
        refit_info = trainer.fit_fixed_epochs(full_ds, selected_epochs)
        fit_info = {
            "inner_selection": selection_info,
            "full_calibration_refit": refit_info,
            "selected_epochs": selected_epochs,
            "n_inner_train": len(train_ds),
            "n_inner_validation": len(val_ds),
            "n_full_calibration": len(full_ds),
            "elapsed_s": (
                selection_info["elapsed_s"] + refit_info["elapsed_s"]
            ),
        }
        pred_scaled = trainer.predict(test_ds)
        predictions = data["scaler_y"].inverse_transform(pred_scaled)
        metrics = evaluate(y_true, predictions)
        metrics.update({
            "scenario": scenario,
            "strategy": "mmoe",
            "seed": seed,
            "n_calibration": len(calibration["y"]),
            "n_test": len(y_true),
            "selected_fine_tune_epochs": selected_epochs,
            "train_time_s": fit_info["elapsed_s"],
        })
        save_run(
            run_dir, trainer, fit_info, predictions, metrics, run_config
        )
        rows.append(metrics)
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.smoke:
        args.max_epochs = min(args.max_epochs, 2)
        args.fine_tune_epochs = min(args.fine_tune_epochs, 2)
        args.patience = min(args.patience, 2)
        args.out_dir = Path(str(args.out_dir) + "_smoke")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = load_fixed_config()
    data = load_fixed_data()
    audit = architecture_audit(data, config)
    if not audit["shared_encoder_bitwise_equal"]:
        raise RuntimeError("MMoE changes shared PatchTST initialization")
    with open(args.out_dir / "architecture_audit.json", "w") as handle:
        json.dump(audit, handle, indent=2)

    config = {
        **config,
        "parameter_count": audit["mmoe_parameter_count"],
        "shared_patchtst_parameter_count": (
            audit["shared_head_parameter_count"]
        ),
        "relative_parameter_increase_pct": (
            audit["relative_parameter_increase_pct"]
        ),
        "capacity_note": (
            "the PatchTST encoder initialization is bitwise identical; the "
            "MMoE head adds explicitly reported architecture parameters"
        ),
    }
    rows = []
    for seed in args.seeds:
        logger.info(f"Running MMoE seed={seed}")
        rows.extend(run_seed(seed, data, config, args.out_dir, args))
    aggregate(rows, args.out_dir, {
        **config,
        "architecture": ARCHITECTURE,
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "fine_tune_epochs": args.fine_tune_epochs,
        "test_selection": False,
    })
    logger.info(f"Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
