#!/usr/bin/env python3
"""Reviewer-2 task-conflict-aware MTL baselines on a fixed PatchTST backbone.

The experiment compares four optimization strategies without changing the
backbone, input representation, data split, model capacity, or test protocol:

* vanilla shared-backbone MTL with the frozen PatchTST Optuna weights;
* homoscedastic uncertainty weighting;
* GradNorm;
* PCGrad.

L1 trains on the original training split and selects epochs only on the
original validation split. For L2/L3, an 80/20 split of shifted calibration
data selects the fine-tuning epoch count; the corresponding L1 checkpoint is
then refit for that count on the complete shifted calibration set and evaluated
once on the held-out shifted test set. Test labels never affect selection.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_proc.datasets import TensorDataset
from models.base_nn import WarmupCosineScheduler
from models.patchtst_mtl_model import PatchTSTConflictMTL


REP_DIR = str(data_path('ieee39_v8_80_10_10'))
EXP1_DIR = work_path('results/ieee39/exp1')
DEFAULT_OUT = work_path('results/ieee39/revision_r2_mtl_fair')
MS = 10
STRATEGIES = ("vanilla", "uncertainty", "gradnorm", "pcgrad")
SCENARIOS = {
    "L2": ("cross_cond_finetune", "cross_cond_test"),
    "L3": ("cross_cond_topo_finetune", "cross_cond_topo_test"),
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {}
    for target_idx, target in enumerate(("y1", "y2")):
        truth = y_true[:, target_idx]
        pred = y_pred[:, target_idx]
        error = pred - truth
        denom = np.maximum(np.abs(truth), 1e-8)
        smape_denom = np.maximum(np.abs(truth) + np.abs(pred), 1e-8)
        metrics[f"{target}_MAE"] = float(np.mean(np.abs(error)))
        metrics[f"{target}_RMSE"] = float(np.sqrt(np.mean(error ** 2)))
        metrics[f"{target}_MAPE"] = float(
            100.0 * np.mean(np.abs(error) / denom)
        )
        metrics[f"{target}_SMAPE"] = float(
            200.0 * np.mean(np.abs(error) / smape_denom)
        )
    return metrics


def load_fixed_data() -> dict:
    """Load only RepB and fit all scalers on the L1 training split."""
    rep_b = Path(REP_DIR) / "repB" / f"ms{MS}"
    rep_a = Path(REP_DIR) / "repA" / f"ms{MS}"
    temporal = {
        split: np.load(rep_b / f"X_temporal_{split}.npy")
        for split in ("train", "val", "test")
    }
    static = {
        split: np.load(rep_b / f"X_static_{split}.npy")
        for split in ("train", "val", "test")
    }
    targets = {
        split: np.load(rep_a / f"y_{split}.npy")
        for split in ("train", "val", "test")
    }
    with open(rep_b / "meta.json") as handle:
        static_names = json.load(handle)["static_names"]
    with open(EXP1_DIR / "mrmr_static_features.json") as handle:
        selected = json.load(handle)["union"]
    static_idx = [static_names.index(name) for name in selected]

    train_temporal = temporal["train"]
    n_samples, n_steps, n_nodes, n_channels = train_temporal.shape
    temporal_scaler = StandardScaler().fit(
        train_temporal.reshape(-1, n_nodes * n_channels)
    )
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
    }
    for split in ("train", "val", "test"):
        values = temporal[split]
        shape = values.shape
        result[f"Xt_{split}_n"] = temporal_scaler.transform(
            values.reshape(-1, n_nodes * n_channels)
        ).reshape(shape).astype(np.float32)
        result[f"Xs_{split}_n"] = static_scaler.transform(
            static[split][:, static_idx]
        ).astype(np.float32)
        result[f"y_{split}"] = targets[split]
        if split != "test":
            result[f"y_{split}_n"] = target_scaler.transform(
                targets[split]
            ).astype(np.float32)
    return result


def load_shift_split(split: str, scalers: dict) -> dict:
    rep_b = Path(REP_DIR) / "repB" / f"ms{MS}"
    rep_a = Path(REP_DIR) / "repA" / f"ms{MS}"
    temporal = np.load(rep_b / f"X_temporal_{split}.npy")
    static = np.load(rep_b / f"X_static_{split}.npy")
    target = np.load(rep_a / f"y_{split}.npy")
    n_samples, n_steps, n_nodes, n_channels = temporal.shape
    temporal_n = scalers["sc_t"].transform(
        temporal.reshape(-1, n_nodes * n_channels)
    ).reshape(temporal.shape).astype(np.float32)
    static_n = scalers["sc_s"].transform(
        static[:, scalers["static_idx"]]
    ).astype(np.float32)
    return {
        "Xt_n": temporal_n,
        "Xs_n": static_n,
        "y": target,
        "y_n": scalers["scaler_y"].transform(target).astype(np.float32),
        "n_samples": n_samples,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_fixed_config() -> dict:
    with open(EXP1_DIR / "PatchTST/best_params.json") as handle:
        params = json.load(handle)
    model_keys = {
        "d_model", "n_heads", "n_layers", "d_ff", "patch_len", "stride",
        "dropout", "tab_hidden", "tab_output",
    }
    return {
        "model": {key: params[key] for key in model_keys},
        "lr": float(params["lr"]),
        "weight_decay": float(params["weight_decay"]),
        "batch_size": int(params["bs"]),
        "y1_weight": float(params["y1_weight"]),
        "y2_weight": float(params["y2_weight"]),
        "source": str(work_path('results/ieee39/exp1/PatchTST/best_params.json')),
        "task_scaling": "targets standardized on the L1 training split",
        "selection_metric": "mean normalized validation SmoothL1 loss",
        "scheduler": "five-epoch linear warm-up followed by cosine decay",
    }


def make_model(data: dict, config: dict) -> PatchTSTConflictMTL:
    kwargs = dict(
        n_timesteps=data["n_timesteps"],
        n_generators=data["n_generators"],
        n_features=data["n_features"],
        n_static=data["n_static"],
    )
    kwargs.update(config["model"])
    return PatchTSTConflictMTL(**kwargs)


class ConflictAwareTrainer:
    """Small trainer that changes only the two-task gradient combination."""

    def __init__(self, model: PatchTSTConflictMTL, strategy: str, device: str,
                 lr: float, weight_decay: float, batch_size: int,
                 max_epochs: int, patience: int, num_workers: int = 8,
                 gradnorm_alpha: float = 1.5,
                 base_task_weights: tuple[float, float] = (1.0, 1.0),
                 warmup_epochs: int = 5):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.model = model.to(device)
        self.strategy = strategy
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.num_workers = num_workers
        self.gradnorm_alpha = gradnorm_alpha
        self.base_task_weights = torch.as_tensor(
            base_task_weights, dtype=torch.float32, device=device
        )
        # Adaptive weighting and gradient-surgery methods stay in FP32:
        # learned inverse variances can amplify shifted losses, GradNorm
        # differentiates through gradient norms, and scaled PCGrad projections
        # can overflow. Vanilla follows the original PatchTST AMP path.
        self.use_amp = device != "cpu" and strategy == "vanilla"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        self.criterion = nn.SmoothL1Loss()

        self.log_vars = None
        self.task_weights = None
        model_params = list(self.model.parameters())
        if strategy == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(2, device=device))
            model_params.append(self.log_vars)
        self.optimizer = torch.optim.AdamW(
            model_params, lr=lr, weight_decay=weight_decay
        )
        self.weight_optimizer = None
        if strategy == "gradnorm":
            self.task_weights = nn.Parameter(torch.ones(2, device=device))
            self.weight_optimizer = torch.optim.Adam(
                [self.task_weights], lr=0.025
            )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_epochs=min(warmup_epochs, max_epochs),
            total_epochs=max_epochs, eta_min=1e-6
        )
        self.initial_losses = None

    def load_aux_state(self, state: dict) -> None:
        if self.log_vars is not None and state.get("log_vars") is not None:
            self.log_vars.data.copy_(
                torch.as_tensor(state["log_vars"], device=self.device)
            )
        if self.task_weights is not None and state.get("task_weights") is not None:
            self.task_weights.data.copy_(
                torch.as_tensor(state["task_weights"], device=self.device)
            )

    def aux_state(self) -> dict:
        return {
            "log_vars": None if self.log_vars is None
            else self.log_vars.detach().cpu().tolist(),
            "task_weights": None if self.task_weights is None
            else self.task_weights.detach().cpu().tolist(),
        }

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(
            batch["x_temporal"].to(self.device, non_blocking=True),
            batch["x_static"].to(self.device, non_blocking=True),
        )
        target = batch["y"].to(self.device, non_blocking=True)
        losses = torch.stack([
            self.criterion(prediction[:, 0], target[:, 0]),
            self.criterion(prediction[:, 1], target[:, 1]),
        ])
        return prediction, losses

    def _pcgrad_step(self, losses: torch.Tensor) -> None:
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        task_grads = []
        for task_idx in range(2):
            task_loss = (
                self.scaler.scale(losses[task_idx])
                if self.scaler is not None else losses[task_idx]
            )
            gradients = torch.autograd.grad(
                task_loss, parameters,
                retain_graph=task_idx == 0, allow_unused=True,
            )
            task_grads.append([
                torch.zeros_like(parameter) if gradient is None else gradient
                for parameter, gradient in zip(parameters, gradients)
            ])

        dot = sum(
            torch.sum(g1 * g2)
            for g1, g2 in zip(task_grads[0], task_grads[1])
        )
        norm0 = sum(torch.sum(g * g) for g in task_grads[0]).clamp_min(1e-12)
        norm1 = sum(torch.sum(g * g) for g in task_grads[1]).clamp_min(1e-12)
        if dot.item() < 0:
            projected0 = [
                g0 - dot / norm1 * g1
                for g0, g1 in zip(task_grads[0], task_grads[1])
            ]
            projected1 = [
                g1 - dot / norm0 * g0
                for g0, g1 in zip(task_grads[0], task_grads[1])
            ]
        else:
            projected0, projected1 = task_grads

        for parameter, grad0, grad1 in zip(
                parameters, projected0, projected1):
            parameter.grad = 0.5 * (grad0 + grad1)
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _standard_step(self, objective: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(objective).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            objective.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

    def _gradnorm_step(self, losses: torch.Tensor) -> None:
        if self.initial_losses is None:
            self.initial_losses = losses.detach().clamp_min(1e-8)

        positive_weights = self.task_weights.clamp_min(1e-3)
        shared = list(self.model.gradnorm_parameters())
        gradient_norms = []
        for task_idx in range(2):
            gradients = torch.autograd.grad(
                positive_weights[task_idx] * losses[task_idx], shared,
                retain_graph=True, create_graph=True,
            )
            gradient_norms.append(torch.cat([
                gradient.reshape(-1) for gradient in gradients
            ]).norm(2))
        gradient_norms = torch.stack(gradient_norms)

        loss_ratios = losses.detach() / self.initial_losses
        inverse_rates = loss_ratios / loss_ratios.mean()
        targets = gradient_norms.detach().mean() * (
            inverse_rates ** self.gradnorm_alpha
        )
        gradnorm_loss = torch.abs(gradient_norms - targets).sum()
        weight_gradient = torch.autograd.grad(
            gradnorm_loss, self.task_weights, retain_graph=True
        )[0]

        model_loss = torch.mean(positive_weights.detach() * losses)
        model_loss.backward()
        self.task_weights.grad = weight_gradient
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.weight_optimizer.step()
        with torch.no_grad():
            self.task_weights.clamp_(min=1e-3)
            self.task_weights.mul_(2.0 / self.task_weights.sum())

    def _train_epoch(self, loader: DataLoader) -> tuple[float, list[float]]:
        self.model.train()
        epoch_losses = []
        epoch_task_losses = []
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            if self.weight_optimizer is not None:
                self.weight_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, losses = self._forward(batch)

            if self.strategy == "vanilla":
                objective = torch.sum(self.base_task_weights * losses)
                self._standard_step(objective)
            elif self.strategy == "uncertainty":
                objective = 0.5 * torch.sum(
                    torch.exp(-self.log_vars) * losses + self.log_vars
                )
                self._standard_step(objective)
            elif self.strategy == "pcgrad":
                objective = losses.mean()
                self._pcgrad_step(losses)
            else:
                objective = losses.mean()
                self._gradnorm_step(losses)

            epoch_losses.append(float(objective.detach().cpu()))
            epoch_task_losses.append(losses.detach().cpu().numpy())
        return (
            float(np.mean(epoch_losses)),
            np.mean(np.asarray(epoch_task_losses), axis=0).tolist(),
        )

    @torch.no_grad()
    def evaluate_loss(self, dataset) -> tuple[float, list[float]]:
        self.model.eval()
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=max(0, self.num_workers // 2), pin_memory=True,
        )
        losses = []
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, task_losses = self._forward(batch)
            losses.append(task_losses.cpu().numpy())
        mean_tasks = np.mean(np.asarray(losses), axis=0)
        return float(mean_tasks.mean()), mean_tasks.tolist()

    def fit(self, train_dataset, val_dataset) -> dict:
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )
        best_loss = float("inf")
        best_model = None
        best_aux = None
        best_epoch = None
        stale_epochs = 0
        history = []
        start = time.time()

        for epoch in range(self.max_epochs):
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
                **self.aux_state(),
            }
            history.append(row)
            if epoch == 0 or (epoch + 1) % 10 == 0:
                logger.info(
                    f"{self.strategy} epoch={epoch + 1} "
                    f"train={train_loss:.6f} val={val_loss:.6f}"
                )

            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                best_aux = copy.deepcopy(self.aux_state())
                best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_model is None:
            raise RuntimeError("Training did not produce a finite checkpoint")
        self.model.load_state_dict(best_model)
        self.load_aux_state(best_aux)
        return {
            "best_val_loss": best_loss,
            "best_epoch": best_epoch,
            "epochs": len(history),
            "elapsed_s": time.time() - start,
            "history": history,
            "aux_state": best_aux,
        }

    def fit_fixed_epochs(self, train_dataset, epochs: int) -> dict:
        """Refit on all calibration samples for a preselected epoch count."""
        if epochs < 1:
            raise ValueError("epochs must be positive")
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
        )
        history = []
        start = time.time()
        for epoch in range(epochs):
            train_loss, train_tasks = self._train_epoch(train_loader)
            self.scheduler.step()
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_y1": train_tasks[0],
                "train_y2": train_tasks[1],
                "lr": self.optimizer.param_groups[0]["lr"],
                **self.aux_state(),
            })
        return {
            "epochs": epochs,
            "elapsed_s": time.time() - start,
            "history": history,
            "aux_state": self.aux_state(),
            "selection": "epoch count fixed by calibration-only inner split",
        }

    @torch.no_grad()
    def predict(self, dataset) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=max(0, self.num_workers // 2), pin_memory=True,
        )
        outputs = []
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                prediction = self.model(
                    batch["x_temporal"].to(self.device, non_blocking=True),
                    batch["x_static"].to(self.device, non_blocking=True),
                )
            outputs.append(prediction.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)


def make_l1_datasets(data: dict, smoke: bool):
    train = TensorDataset(
        data["Xt_train_n"], data["Xs_train_n"], data["y_train_n"]
    )
    val = TensorDataset(
        data["Xt_val_n"], data["Xs_val_n"], data["y_val_n"]
    )
    test = TensorDataset(
        data["Xt_test_n"], data["Xs_test_n"], data["y_test"]
    )
    if smoke:
        train = Subset(train, range(min(2048, len(train))))
        val = Subset(val, range(min(512, len(val))))
        test = Subset(test, range(min(512, len(test))))
    return train, val, test


def make_shift_datasets(split_data: dict, test_data: dict,
                        seed: int, smoke: bool):
    n_samples = len(split_data["y"])
    indices = np.random.RandomState(seed).permutation(n_samples)
    train_end = max(1, int(0.8 * n_samples))
    train_idx, val_idx = indices[:train_end], indices[train_end:]
    if len(val_idx) == 0:
        val_idx = train_idx[-1:]
    if smoke:
        train_idx = train_idx[:min(512, len(train_idx))]
        val_idx = val_idx[:min(128, len(val_idx))]

    train = TensorDataset(
        split_data["Xt_n"][train_idx], split_data["Xs_n"][train_idx],
        split_data["y_n"][train_idx],
    )
    val = TensorDataset(
        split_data["Xt_n"][val_idx], split_data["Xs_n"][val_idx],
        split_data["y_n"][val_idx],
    )
    test_limit = min(512, len(test_data["y"])) if smoke else len(test_data["y"])
    test = TensorDataset(
        test_data["Xt_n"][:test_limit], test_data["Xs_n"][:test_limit],
        test_data["y"][:test_limit],
    )
    full = TensorDataset(
        split_data["Xt_n"], split_data["Xs_n"], split_data["y_n"]
    )
    return train, val, full, test, test_data["y"][:test_limit]


def save_run(run_dir: Path, trainer: ConflictAwareTrainer, fit_info: dict,
             predictions: np.ndarray, metrics: dict, run_config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": trainer.model.state_dict(),
        "aux_state": trainer.aux_state(),
        "run_config": run_config,
    }, run_dir / "model.pth")
    np.save(run_dir / "preds.npy", predictions)
    with open(run_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(run_dir / "fit.json", "w") as handle:
        json.dump(fit_info, handle, indent=2)
    with open(run_dir / "config.json", "w") as handle:
        json.dump(run_config, handle, indent=2)


def load_checkpoint(path: Path, model: PatchTSTConflictMTL,
                    device: str) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run_strategy(strategy: str, seed: int, data: dict, shift_scalers: dict,
                 config: dict, out_dir: Path, args) -> list[dict]:
    strategy_dir = out_dir / strategy / f"seed_{seed}"
    l1_dir = strategy_dir / "L1"
    l1_metrics_path = l1_dir / "metrics.json"
    model_config = {
        **config,
        "strategy": strategy,
        "seed": seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_protocol": (
            "80/20 shifted-calibration inner split selects the epoch count; "
            "the L1 checkpoint is then refit for that count on the complete "
            "shifted calibration set; test labels remain unseen"
        ),
        "precision": (
            "FP32 for uncertainty weighting, GradNorm, and PCGrad; AMP for "
            "vanilla, matching the original fixed-loss PatchTST path"
        ),
    }

    set_seed(seed)
    l1_model = make_model(data, config)
    if l1_metrics_path.exists() and not args.force:
        with open(l1_metrics_path) as handle:
            l1_metrics = json.load(handle)
        checkpoint = load_checkpoint(l1_dir / "model.pth", l1_model, args.device)
    else:
        train_ds, val_ds, test_ds = make_l1_datasets(data, args.smoke)
        trainer = ConflictAwareTrainer(
            l1_model, strategy, args.device, config["lr"],
            config["weight_decay"], config["batch_size"],
            args.max_epochs, args.patience, args.num_workers,
            base_task_weights=(config["y1_weight"], config["y2_weight"]),
        )
        fit_info = trainer.fit(train_ds, val_ds)
        pred_scaled = trainer.predict(test_ds)
        predictions = data["scaler_y"].inverse_transform(pred_scaled)
        y_true = data["y_test"][:len(predictions)]
        l1_metrics = evaluate(y_true, predictions)
        l1_metrics.update({
            "scenario": "L1", "strategy": strategy, "seed": seed,
            "n_test": len(y_true), "train_time_s": fit_info["elapsed_s"],
        })
        save_run(l1_dir, trainer, fit_info, predictions, l1_metrics, model_config)
        checkpoint = {
            "model_state_dict": trainer.model.state_dict(),
            "aux_state": trainer.aux_state(),
        }

    rows = [l1_metrics]
    for scenario, (cal_split, test_split) in SCENARIOS.items():
        run_dir = strategy_dir / scenario
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            with open(metrics_path) as handle:
                rows.append(json.load(handle))
            continue

        set_seed(seed)
        ft_data = load_shift_split(cal_split, shift_scalers)
        test_data = load_shift_split(test_split, shift_scalers)
        train_ds, val_ds, full_ds, test_ds, y_true = make_shift_datasets(
            ft_data, test_data, seed, args.smoke
        )
        selection_model = make_model(data, config)
        selection_model.load_state_dict(checkpoint["model_state_dict"])
        selection_trainer = ConflictAwareTrainer(
            selection_model, strategy, args.device, config["lr"] / 10.0,
            config["weight_decay"],
            min(config["batch_size"], max(16, len(train_ds))),
            args.fine_tune_epochs, min(args.patience, 10), args.num_workers,
            base_task_weights=(config["y1_weight"], config["y2_weight"]),
        )
        selection_trainer.load_aux_state(checkpoint.get("aux_state", {}))
        selection_info = selection_trainer.fit(train_ds, val_ds)
        selected_epochs = int(selection_info["best_epoch"])
        del selection_trainer, selection_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        set_seed(seed)
        model = make_model(data, config)
        model.load_state_dict(checkpoint["model_state_dict"])
        trainer = ConflictAwareTrainer(
            model, strategy, args.device, config["lr"] / 10.0,
            config["weight_decay"],
            min(config["batch_size"], max(16, len(full_ds))),
            selected_epochs, selected_epochs, args.num_workers,
            base_task_weights=(config["y1_weight"], config["y2_weight"]),
        )
        trainer.load_aux_state(checkpoint.get("aux_state", {}))
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
        predictions = shift_scalers["scaler_y"].inverse_transform(pred_scaled)
        metrics = evaluate(y_true, predictions)
        metrics.update({
            "scenario": scenario, "strategy": strategy, "seed": seed,
            "n_calibration": len(ft_data["y"]), "n_test": len(y_true),
            "selected_fine_tune_epochs": selected_epochs,
            "train_time_s": fit_info["elapsed_s"],
        })
        save_run(run_dir, trainer, fit_info, predictions, metrics, model_config)
        rows.append(metrics)
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def aggregate(rows: list[dict], out_dir: Path, config: dict) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "all_runs.csv", index=False)
    metric_cols = ["y1_MAE", "y1_RMSE", "y2_MAE", "y2_RMSE"]
    summary = frame.groupby(["strategy", "scenario"])[metric_cols].agg(
        ["mean", "std", "count"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "protocol.json", "w") as handle:
        json.dump(config, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=list(STRATEGIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
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
    reference_model = make_model(data, config)
    config["parameter_count"] = int(sum(
        parameter.numel() for parameter in reference_model.parameters()
    ))
    config["capacity_match"] = (
        "the original PatchTST two-output head, parameterization, and "
        "initialization order are retained; only task-gradient combination changes"
    )
    del reference_model
    shift_scalers = data
    rows = []
    for strategy in args.strategies:
        for seed in args.seeds:
            logger.info(f"Running strategy={strategy}, seed={seed}")
            rows.extend(run_strategy(
                strategy, seed, data, shift_scalers, config, args.out_dir, args
            ))
    aggregate(rows, args.out_dir, {
        **config,
        "strategies": args.strategies,
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "test_selection": False,
    })
    logger.info(f"Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
