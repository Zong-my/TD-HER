#!/usr/bin/env python3
"""
Experiment P2: Gradient Conflict Analysis Between y1 and y2 Tasks.

Measures the cosine similarity between y1 and y2 task gradients during
training to quantify multi-task gradient conflict.

If gradients consistently conflict (cosine < 0), this provides optimization-
level evidence that Δf and t_nadir are fundamentally different learning tasks
that interfere with each other during joint training.

Analysis is performed on:
  1. Standard training (exp1 setting, same-distribution data)
  2. Cross-topology fine-tuning (exp4b setting)

This reveals whether gradient conflict is baseline-level or amplified
under distribution shift (topology change).

Models tested: ConvLSTM, PatchTST, Mamba

Usage:
    python experiments/exp_p2_gradient_analysis.py [--models ConvLSTM,PatchTST,Mamba]
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.convlstm_model import ConvLSTMFreqModel
from models.patchtst_model import PatchTSTFreqModel
from models.mamba_model import MambaFreqModel
from data_proc.datasets import TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

REP_DIR = str(data_path('ieee39_v8_80_10_10'))
EXP1_DIR = str(work_path('results/ieee39/exp1'))
P2_DIR = str(work_path('results/ieee39/exp_p2_gradient_analysis'))
MS = 10

MODEL_CLASSES = {
    'ConvLSTM': ConvLSTMFreqModel,
    'PatchTST': PatchTSTFreqModel,
    'Mamba': MambaFreqModel,
}


def load_data_and_model(model_name, device='cuda'):
    """Load exp1 trained model and prepare data loaders."""
    # Load data
    Xt_train = np.load(f'{REP_DIR}/repB/ms{MS}/X_temporal_train.npy')
    Xs_train = np.load(f'{REP_DIR}/repB/ms{MS}/X_static_train.npy')
    y_train = np.load(f'{REP_DIR}/repA/ms{MS}/y_train.npy')

    # mRMR static feature selection
    with open(f'{REP_DIR}/repB/ms{MS}/meta.json') as f:
        meta = json.load(f)
    static_names = meta['static_names']
    with open(f'{EXP1_DIR}/mrmr_static_features.json') as f:
        static_mrmr = json.load(f)
    static_idx = [static_names.index(f) for f in static_mrmr['union']]
    Xs_train = Xs_train[:, static_idx]

    # Normalize
    B, T, N, C = Xt_train.shape
    sc_t = StandardScaler().fit(Xt_train.reshape(-1, N * C))
    Xt_train_n = sc_t.transform(Xt_train.reshape(-1, N*C)).reshape(B, T, N, C).astype(np.float32)
    sc_s = StandardScaler().fit(Xs_train)
    Xs_train_n = sc_s.transform(Xs_train).astype(np.float32)
    scaler_y = StandardScaler().fit(y_train)
    y_train_n = scaler_y.transform(y_train).astype(np.float32)

    # Cross-topology finetune data
    Xt_ft = np.load(f'{REP_DIR}/repB/ms{MS}/X_temporal_cross_cond_topo_finetune.npy')
    Xs_ft = np.load(f'{REP_DIR}/repB/ms{MS}/X_static_cross_cond_topo_finetune.npy')
    y_ft = np.load(f'{REP_DIR}/repA/ms{MS}/y_cross_cond_topo_finetune.npy')
    Xs_ft = Xs_ft[:, static_idx]
    Bf = len(y_ft)
    Xt_ft_n = sc_t.transform(Xt_ft.reshape(-1, N*C)).reshape(Bf, T, N, C).astype(np.float32)
    Xs_ft_n = sc_s.transform(Xs_ft).astype(np.float32)
    y_ft_n = scaler_y.transform(y_ft).astype(np.float32)

    # Dataloaders — use moderate batch size for stable gradient estimates
    train_ds = TensorDataset(Xt_train_n, Xs_train_n, y_train_n)
    ft_ds = TensorDataset(Xt_ft_n, Xs_ft_n, y_ft_n)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
    ft_loader = DataLoader(ft_ds, batch_size=min(512, len(y_ft)), shuffle=True, num_workers=4, pin_memory=True)

    # Load exp1 model
    ckpt = torch.load(f'{EXP1_DIR}/{model_name}/model.pth', map_location=device, weights_only=False)
    model = MODEL_CLASSES[model_name](**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    with open(f'{EXP1_DIR}/{model_name}/best_params.json') as f:
        bp = json.load(f)

    return model, train_loader, ft_loader, bp


def compute_per_task_gradients(model, batch, criterion, device):
    """Compute separate gradients for y1 and y2 losses.

    Returns flattened gradient vectors for shared parameters.
    """
    model.zero_grad()

    # Forward pass
    if 'x_temporal' in batch:
        pred = model(batch['x_temporal'].to(device), batch['x_static'].to(device))
    else:
        raise ValueError("Unsupported batch format")

    target = batch['y'].to(device)

    # y1 gradient
    loss_y1 = criterion(pred[:, 0], target[:, 0])
    grads_y1 = torch.autograd.grad(loss_y1, model.parameters(),
                                     retain_graph=True, allow_unused=True)
    grad_y1_flat = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=device)
                               for g, p in zip(grads_y1, model.parameters())])

    # y2 gradient
    loss_y2 = criterion(pred[:, 1], target[:, 1])
    grads_y2 = torch.autograd.grad(loss_y2, model.parameters(), allow_unused=True)
    grad_y2_flat = torch.cat([g.flatten() if g is not None else torch.zeros(p.numel(), device=device)
                               for g, p in zip(grads_y2, model.parameters())])

    return grad_y1_flat, grad_y2_flat, float(loss_y1), float(loss_y2)


def cosine_similarity(g1, g2):
    """Compute cosine similarity between two gradient vectors."""
    dot = torch.dot(g1, g2)
    norm1 = torch.norm(g1)
    norm2 = torch.norm(g2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return float(dot / (norm1 * norm2))


def analyze_gradients(model, loader, criterion, device, n_batches=50, desc=""):
    """Compute gradient cosine similarity over multiple batches."""
    model.eval()  # Use eval mode for stable BN/LN statistics
    cosines = []
    y1_norms = []
    y2_norms = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        g1, g2, l1, l2 = compute_per_task_gradients(model, batch, criterion, device)
        cos = cosine_similarity(g1, g2)
        cosines.append(cos)
        y1_norms.append(float(torch.norm(g1)))
        y2_norms.append(float(torch.norm(g2)))

        if i % 10 == 0:
            logger.info(f"  {desc} batch {i}/{n_batches}: cos={cos:.4f}, "
                       f"||g_y1||={y1_norms[-1]:.4f}, ||g_y2||={y2_norms[-1]:.4f}")

    return {
        'cosines': cosines,
        'y1_norms': y1_norms,
        'y2_norms': y2_norms,
        'mean_cosine': float(np.mean(cosines)),
        'std_cosine': float(np.std(cosines)),
        'median_cosine': float(np.median(cosines)),
        'pct_negative': float(np.mean([c < 0 for c in cosines]) * 100),
        'mean_y1_norm': float(np.mean(y1_norms)),
        'mean_y2_norm': float(np.mean(y2_norms)),
        'norm_ratio': float(np.mean(y1_norms) / max(np.mean(y2_norms), 1e-12)),
    }


def plot_gradient_analysis(results, model_name, save_dir):
    """Plot gradient cosine similarity distributions and norms."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    scenarios = list(results.keys())
    colors = {'same_dist': '#2196F3', 'cross_topo': '#F44336'}
    labels = {'same_dist': 'Same-distribution', 'cross_topo': 'Cross-topology'}

    # Panel 1: Cosine similarity distribution
    ax = axes[0]
    for scenario in scenarios:
        cosines = results[scenario]['cosines']
        ax.hist(cosines, bins=25, alpha=0.6, color=colors[scenario],
                label=f"{labels[scenario]} (μ={results[scenario]['mean_cosine']:.3f})",
                density=True, edgecolor='white', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Cosine Similarity (y1 vs y2 gradients)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{model_name} — Gradient Alignment', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 2: Cosine similarity over batches
    ax = axes[1]
    for scenario in scenarios:
        cosines = results[scenario]['cosines']
        ax.plot(range(len(cosines)), cosines, alpha=0.7, color=colors[scenario],
                label=labels[scenario], linewidth=1)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title(f'{model_name} — Per-Batch Gradient Conflict', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Gradient norm ratio
    ax = axes[2]
    for scenario in scenarios:
        y1_norms = results[scenario]['y1_norms']
        y2_norms = results[scenario]['y2_norms']
        ratios = [n1/max(n2, 1e-12) for n1, n2 in zip(y1_norms, y2_norms)]
        ax.plot(range(len(ratios)), ratios, alpha=0.7, color=colors[scenario],
                label=f"{labels[scenario]} (μ={np.mean(ratios):.2f})", linewidth=1)
    ax.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('||∇y1|| / ||∇y2||', fontsize=12)
    ax.set_title(f'{model_name} — Gradient Norm Ratio', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name}_gradient_analysis.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{save_dir}/{model_name}_gradient_analysis.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Gradient analysis plot saved for {model_name}")


def main():
    parser = argparse.ArgumentParser(description='P2: Gradient Conflict Analysis')
    parser.add_argument('--models', type=str, default='ConvLSTM,PatchTST,Mamba')
    parser.add_argument('--n_batches', type=int, default=50,
                        help='Number of batches to sample for gradient analysis')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    Path(P2_DIR).mkdir(parents=True, exist_ok=True)

    criterion = nn.SmoothL1Loss()  # Same as exp1 default (Huber)
    all_results = {}

    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"P2: Gradient Analysis for {model_name}")
        logger.info(f"{'='*60}")

        save_dir = f'{P2_DIR}/{model_name}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Skip if done
        if os.path.exists(f'{save_dir}/gradient_stats.json'):
            with open(f'{save_dir}/gradient_stats.json') as f:
                results = json.load(f)
            logger.info(f"  SKIPPED (done)")
            logger.info(f"  Same-dist: mean_cos={results['same_dist']['mean_cosine']:.4f}, "
                       f"negative={results['same_dist']['pct_negative']:.1f}%")
            logger.info(f"  Cross-topo: mean_cos={results['cross_topo']['mean_cosine']:.4f}, "
                       f"negative={results['cross_topo']['pct_negative']:.1f}%")
            all_results[model_name] = results
            continue

        model, train_loader, ft_loader, bp = load_data_and_model(model_name, args.device)

        results = {}

        # Scenario 1: Same-distribution (standard training data)
        logger.info(f"\n  Analyzing same-distribution gradients...")
        results['same_dist'] = analyze_gradients(
            model, train_loader, criterion, args.device,
            n_batches=args.n_batches, desc="same_dist"
        )
        logger.info(f"  Same-dist: mean_cos={results['same_dist']['mean_cosine']:.4f}, "
                   f"negative={results['same_dist']['pct_negative']:.1f}%")

        # Scenario 2: Cross-topology (finetune data — distribution shift)
        logger.info(f"\n  Analyzing cross-topology gradients...")
        results['cross_topo'] = analyze_gradients(
            model, ft_loader, criterion, args.device,
            n_batches=args.n_batches, desc="cross_topo"
        )
        logger.info(f"  Cross-topo: mean_cos={results['cross_topo']['mean_cosine']:.4f}, "
                   f"negative={results['cross_topo']['pct_negative']:.1f}%")

        # Save stats (without raw lists for cleaner JSON)
        stats_to_save = {}
        for scenario in results:
            stats_to_save[scenario] = {k: v for k, v in results[scenario].items()
                                        if k not in ('cosines', 'y1_norms', 'y2_norms')}
        with open(f'{save_dir}/gradient_stats.json', 'w') as f:
            json.dump(stats_to_save, f, indent=2)

        # Save raw data for plotting
        np.savez(f'{save_dir}/gradient_raw.npz',
                 same_dist_cosines=results['same_dist']['cosines'],
                 same_dist_y1_norms=results['same_dist']['y1_norms'],
                 same_dist_y2_norms=results['same_dist']['y2_norms'],
                 cross_topo_cosines=results['cross_topo']['cosines'],
                 cross_topo_y1_norms=results['cross_topo']['y1_norms'],
                 cross_topo_y2_norms=results['cross_topo']['y2_norms'])

        # Plot
        plot_gradient_analysis(results, model_name, save_dir)

        all_results[model_name] = stats_to_save

        del model
        torch.cuda.empty_cache()

    # Summary table
    logger.info(f"\n{'='*80}")
    logger.info("P2: GRADIENT CONFLICT SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"\n{'Model':15s} | {'Scenario':14s} | {'Mean cos':>10s} | {'% negative':>10s} | "
                f"{'Norm ratio':>10s}")
    logger.info("-" * 75)

    for model_name in models:
        r = all_results.get(model_name, {})
        for scenario in ['same_dist', 'cross_topo']:
            s = r.get(scenario, {})
            logger.info(f"{model_name:15s} | {scenario:14s} | "
                       f"{s.get('mean_cosine', float('nan')):10.4f} | "
                       f"{s.get('pct_negative', float('nan')):10.1f} | "
                       f"{s.get('norm_ratio', float('nan')):10.2f}")
        logger.info("-" * 75)

    logger.info("\nINTERPRETATION:")
    logger.info("  cos > 0: gradients aligned (tasks cooperate)")
    logger.info("  cos ≈ 0: gradients orthogonal (tasks independent)")
    logger.info("  cos < 0: gradients conflict (tasks interfere)")
    logger.info("  If cross_topo has MORE negative cosines than same_dist →")
    logger.info("    gradient conflict is AMPLIFIED by topology change")

    logger.info("\nP2 gradient analysis complete!")


if __name__ == '__main__':
    main()
