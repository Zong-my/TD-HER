#!/usr/bin/env python3
"""
Unified neural network training loop for all DL models.

Handles: training, validation, early stopping, mixed precision,
         inference, timing, Optuna integration, and training visualization.
"""

import copy
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing to eta_min."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 eta_min: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup: scale from eta_min → base_lr
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay
                    for base_lr in self.base_lrs]


class NNTrainer:
    """Unified trainer for ConvLSTM, PatchTST, Mamba, ST-GCN, KAN."""

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 max_epochs: int = 100, patience: int = 15,
                 batch_size: int = 256, loss_type: str = 'huber',
                 y1_weight: float = 1.0, y2_weight: float = 1.0,
                 num_workers: int = 16, use_amp: bool = True,
                 grad_clip: float = 1.0, warmup_epochs: int = 5,
                 scheduler_type: str = 'warmup_cosine',
                 cache_refresh_fn=None):
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.y1_weight = y1_weight
        self.y2_weight = y2_weight
        self.use_amp = use_amp and device != 'cpu'
        self.grad_clip = grad_clip
        self.cache_refresh_fn = cache_refresh_fn

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler: warmup + cosine (default) or plain cosine
        if scheduler_type == 'warmup_cosine':
            self.scheduler = WarmupCosineScheduler(
                self.optimizer, warmup_epochs=warmup_epochs,
                total_epochs=max_epochs, eta_min=1e-6
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epochs, eta_min=1e-6
            )

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        if loss_type == 'huber':
            self.criterion = nn.SmoothL1Loss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _refresh_model_cache(self):
        """Refresh optional model-side caches after weight updates.

        Retrieval models such as TabR depend on cached train-set embeddings.
        The generic trainer keeps this hook opt-in so ordinary models pay no
        extra cost and preserve the existing training path.
        """
        if self.cache_refresh_fn is not None:
            self.cache_refresh_fn(self.model)

    def _compute_loss(self, pred, target):
        """Compute weighted multi-task loss.

        Uses per-target weighting to handle scale differences between
        y1 (frequency deviation, ~[-3, 1] Hz) and y2 (time, ~[0, 30] s).
        After StandardScaler normalization both are ~N(0,1), but the
        weights allow Optuna to find the optimal balance.
        """
        if pred.dim() == 1:
            # Single-task (KAN): pred (B,), target may be (B,1) → squeeze
            return self.criterion(pred, target.squeeze(-1))
        # Multi-task: pred/target shape (B, 2)
        loss_y1 = self.criterion(pred[:, 0], target[:, 0])
        loss_y2 = self.criterion(pred[:, 1], target[:, 1])
        return self.y1_weight * loss_y1 + self.y2_weight * loss_y2

    def _forward_batch(self, batch):
        """Dispatch forward pass based on batch content."""
        if 'x_node' in batch:
            # ST-GCN (RepC) — edge_index/weight are shared across batch,
            # DataLoader stacks them to (B, 2, E) / (B, E); take [0].
            edge_index = batch['edge_index'][0].to(self.device)
            edge_weight = batch.get('edge_weight')
            if edge_weight is not None:
                edge_weight = edge_weight[0].to(self.device)
            else:
                edge_weight = torch.ones(edge_index.shape[1], device=self.device)
            return self.model(
                batch['x_node'].to(self.device),
                batch['x_static'].to(self.device),
                edge_index,
                edge_weight,
            )
        elif 'x_temporal' in batch:
            # ConvLSTM / PatchTST / Mamba (RepB)
            return self.model(
                batch['x_temporal'].to(self.device),
                batch['x_static'].to(self.device),
            )
        elif 'x' in batch:
            # KAN (RepA)
            return self.model(batch['x'].to(self.device))
        else:
            raise ValueError(f"Unknown batch keys: {batch.keys()}")

    def fit(self, train_dataset, val_dataset, verbose: bool = True,
            checkpoint_dir: str = None, trial=None):
        """Train model with early stopping and checkpoint/resume.

        Saves two files in checkpoint_dir (if set):
        - checkpoint.pth: latest epoch state (for resume after crash)
        - best_model.pth: best val_loss state (for early stopping)
        Both are cleaned up after successful training completion.

        Args:
            trial: optuna.Trial for pruning support (report val_loss each epoch).

        Returns:
            best_val_loss: float
            history: dict with train_loss, val_loss, lr lists
        """
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=self.num_workers // 2, pin_memory=True,
        )

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        start_epoch = 0

        # ── Resume from checkpoint ──
        ckpt_path = os.path.join(checkpoint_dir, 'checkpoint.pth') if checkpoint_dir else None
        best_path = os.path.join(checkpoint_dir, 'best_model.pth') if checkpoint_dir else None
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if self.scaler and ckpt.get('scaler_state_dict'):
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt['best_val_loss']
            patience_counter = ckpt['patience_counter']
            history = ckpt['history']
            for _ in range(start_epoch):
                self.scheduler.step()
            # Load best state
            if best_path and os.path.exists(best_path):
                best_state = torch.load(best_path, map_location=self.device,
                                        weights_only=False)['model_state_dict']
            logger.info(f"Resumed from epoch {start_epoch} "
                        f"(best_val={best_val_loss:.6f})")

        for epoch in range(start_epoch, self.max_epochs):
            self._refresh_model_cache()

            # ── Train ──
            self.model.train()
            train_losses = []
            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        pred = self._forward_batch(batch)
                        loss = self._compute_loss(pred, batch['y'].to(self.device))
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self._forward_batch(batch)
                    loss = self._compute_loss(pred, batch['y'].to(self.device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                train_losses.append(loss.item())

            # ── Validate ──
            self._refresh_model_cache()
            val_loss = self._evaluate(val_loader)
            self.scheduler.step()

            lr_now = self.optimizer.param_groups[0]['lr']
            train_avg = np.mean(train_losses)
            history['train_loss'].append(train_avg)
            history['val_loss'].append(val_loss)
            history['lr'].append(lr_now)

            if verbose and (epoch % 10 == 0 or epoch == self.max_epochs - 1):
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"train={train_avg:.6f} val={val_loss:.6f} lr={lr_now:.2e}")

            # ── Optuna pruning ──
            if trial is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

            # ── Early stopping ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
                # Save best model
                if best_path:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save({'model_state_dict': best_state}, best_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break

            # ── Save latest checkpoint (overwrite each epoch) ──
            if ckpt_path:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                    'history': history,
                }, ckpt_path)

            # ── Real-time loss curve (update every epoch) ──
            if checkpoint_dir and len(history['train_loss']) >= 2:
                self._update_live_loss(history, checkpoint_dir)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self._refresh_model_cache()

        # Clean up checkpoints after successful completion
        for p in [ckpt_path, best_path]:
            if p and os.path.exists(p):
                os.remove(p)

        return best_val_loss, history

    @staticmethod
    def _update_live_loss(history, save_dir):
        """Update loss curve PNG in real-time during training.

        Called every epoch. Overwrites the same file so you can
        refresh the image viewer to see live progress.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            os.makedirs(save_dir, exist_ok=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=1)
            ax1.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=1)
            best_idx = int(np.argmin(history['val_loss']))
            ax1.axvline(best_idx + 1, color='gray', linestyle='--', alpha=0.5)
            ax1.annotate(f'best={history["val_loss"][best_idx]:.6f}',
                         xy=(best_idx + 1, history['val_loss'][best_idx]),
                         fontsize=8, color='red')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss (LIVE)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            if 'lr' in history:
                ax2.plot(epochs, history['lr'], 'g-', linewidth=1)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('LR Schedule')
                ax2.set_yscale('log')
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, 'live_loss.png'), dpi=100)
            plt.close(fig)
        except Exception:
            pass  # Never let plotting crash training

    @torch.no_grad()
    def _evaluate(self, loader):
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        for batch in loader:
            bs = len(batch['y'])
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    pred = self._forward_batch(batch)
                    loss = self._compute_loss(pred, batch['y'].to(self.device))
            else:
                pred = self._forward_batch(batch)
                loss = self._compute_loss(pred, batch['y'].to(self.device))
            total_loss += loss.item() * bs
            total_samples += bs
        return total_loss / total_samples

    @torch.no_grad()
    def predict(self, dataset) -> np.ndarray:
        """Run inference and return predictions as numpy array."""
        self.model.eval()
        loader = DataLoader(
            dataset, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=self.num_workers // 2, pin_memory=True,
        )
        preds = []
        for batch in loader:
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    pred = self._forward_batch(batch)
            else:
                pred = self._forward_batch(batch)
            preds.append(pred.float().cpu().numpy())
        return np.concatenate(preds, axis=0)

    def measure_inference_time(self, dataset, n_warmup: int = 10,
                               n_runs: int = 100) -> dict:
        """Measure per-sample inference latency.

        Returns dict with median_ms, mean_ms, std_ms, p95_ms, throughput.
        """
        self.model.eval()

        # Single sample
        single_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        sample = next(iter(single_loader))

        # Move to device
        sample_gpu = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample_gpu[k] = v.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                self._forward_batch(sample_gpu)

        # Time single-sample latency
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                self._forward_batch(sample_gpu)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                times.append(elapsed_ms)

        # Batch throughput
        batch_loader = DataLoader(dataset, batch_size=256, shuffle=False)
        batch = next(iter(batch_loader))
        batch_gpu = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        batch_size = len(batch['y'])

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_warmup):
                self._forward_batch(batch_gpu)
        torch.cuda.synchronize()

        batch_times = []
        with torch.no_grad():
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                self._forward_batch(batch_gpu)
                torch.cuda.synchronize()
                batch_times.append(time.perf_counter() - start)

        throughput = batch_size / np.median(batch_times)

        return {
            'median_ms': float(np.median(times)),
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'throughput_samples_per_sec': float(throughput),
        }

    # ════════════════════════════════════════════════════════════════
    # Visualization methods
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def plot_loss_curves(history: dict, model_name: str, save_path: str):
        """Plot training and validation loss curves with LR schedule.

        Creates a two-panel figure: loss curves (left) and learning rate (right).
        Saved as PNG to ``save_path``.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history['train_loss']) + 1)

        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=1.5,
                 label='Train loss', alpha=0.8)
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=1.5,
                 label='Val loss', alpha=0.8)
        best_epoch = int(np.argmin(history['val_loss'])) + 1
        best_val = min(history['val_loss'])
        ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5,
                     label=f'Best epoch={best_epoch}')
        ax1.set_xlabel('Epoch', fontsize=13)
        ax1.set_ylabel('Loss', fontsize=13)
        ax1.set_title(f'{model_name} - Training Loss', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.annotate(f'Best val={best_val:.6f}',
                     xy=(best_epoch, best_val),
                     xytext=(best_epoch + 2, best_val * 1.1),
                     fontsize=10, color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.0))

        # Learning rate
        if 'lr' in history and history['lr']:
            ax2.plot(epochs, history['lr'], 'g-', linewidth=1.5)
            ax2.set_xlabel('Epoch', fontsize=13)
            ax2.set_ylabel('Learning Rate', fontsize=13)
            ax2.set_title(f'{model_name} - LR Schedule', fontsize=14)
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Loss curves saved to {save_path}")

    @staticmethod
    def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray,
                     model_name: str, save_path: str,
                     target_names: list = None):
        """Plot prediction vs ground truth scatter for each target.

        Creates a side-by-side scatter plot with perfect-prediction line
        and R-squared annotation.
        """
        if target_names is None:
            target_names = ['y1 (fpu_deltamax, Hz)', 'y2 (t_delta, s)']

        n_targets = y_true.shape[1] if y_true.ndim > 1 else 1
        fig, axes = plt.subplots(1, n_targets, figsize=(7 * n_targets, 6))
        if n_targets == 1:
            axes = [axes]

        for i, (ax, tname) in enumerate(zip(axes, target_names)):
            yt = y_true[:, i] if y_true.ndim > 1 else y_true
            yp = y_pred[:, i] if y_pred.ndim > 1 else y_pred

            # Subsample if too many points for readability
            n = len(yt)
            if n > 5000:
                idx = np.random.default_rng(42).choice(n, 5000, replace=False)
                yt_plot, yp_plot = yt[idx], yp[idx]
            else:
                yt_plot, yp_plot = yt, yp

            ax.scatter(yt_plot, yp_plot, alpha=0.15, s=8, c='steelblue',
                       edgecolors='none')

            # Perfect prediction line
            vmin = min(yt.min(), yp.min())
            vmax = max(yt.max(), yp.max())
            margin = (vmax - vmin) * 0.05
            ax.plot([vmin - margin, vmax + margin],
                    [vmin - margin, vmax + margin],
                    'r--', linewidth=1.5, alpha=0.7, label='y = x')

            # R-squared
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
            mae = np.mean(np.abs(yt - yp))

            ax.set_xlabel(f'Ground Truth', fontsize=13)
            ax.set_ylabel(f'Predicted', fontsize=13)
            ax.set_title(f'{model_name} - {tname}', fontsize=14)
            ax.legend(fontsize=11, loc='upper left')
            ax.grid(True, alpha=0.3)

            # Annotation box
            textstr = f'R$^2$ = {r2:.4f}\nMAE = {mae:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.95, 0.05, textstr, transform=ax.transAxes,
                    fontsize=11, verticalalignment='bottom',
                    horizontalalignment='right', bbox=props)

            ax.set_xlim(vmin - margin, vmax + margin)
            ax.set_ylim(vmin - margin, vmax + margin)
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Scatter plot saved to {save_path}")

    # ════════════════════════════════════════════════════════════════
    # Persistence
    # ════════════════════════════════════════════════════════════════

    def save(self, path: str, model_kwargs: dict = None):
        """Save model state dict and optional construction kwargs."""
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
        }
        if model_kwargs is not None:
            ckpt['model_kwargs'] = model_kwargs
        torch.save(ckpt, path)

    def load(self, path: str):
        """Load model state dict."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
