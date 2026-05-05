#!/usr/bin/env python3
"""
Experiment 5: Interpretability Analysis (IEEE 39-bus).

5a - mRMR Feature Ranking (top-30 horizontal bar chart, color-coded by physical quantity)
5b - ALE (Accumulated Local Effects) plots for LightGBM top-10 mRMR features
5c - KAN Activation Visualization (intermediate activation distributions)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── IEEE TII figure style ──
# Try Times New Roman, fall back to serif
_TNR = 'Times New Roman'
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
if _TNR not in _available_fonts:
    _TNR = 'serif'
    logger.warning("Times New Roman not found, falling back to serif")

plt.rcParams.update({
    'font.family': _TNR if _TNR == 'serif' else 'serif',
    'font.serif': [_TNR, 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,  # TrueType for journal submission
    'ps.fonttype': 42,
})

# ── Color palette for physical quantity types ──
PHYS_COLORS = {
    'FREQ': '#1f77b4',    # blue
    'VOLT': '#2ca02c',    # green
    'POWR': '#d62728',    # red
    'SPD':  '#ff7f0e',    # orange
    'ANGL': '#9467bd',    # purple
    'GREF': '#8c564b',    # brown
    'static': '#7f7f7f',  # gray
    'cross': '#17becf',   # cyan (cross-quantity features like FREQ_VOLT_34_corr)
}

# Known static feature names
STATIC_FEATURES = {'h_inertia', 'reserve_ratio', 'load_level', 'n_gen_online',
                   'total_gen', 'total_load', 'gen_load_ratio'}

TARGET_LABELS = {
    'y1': r'$\Delta f_{\max}$ (p.u.)',
    'y2': r'$t_{\Delta f}$ (s)',
}


def classify_feature(feat_name: str) -> str:
    """Classify a feature name into its physical quantity type."""
    if feat_name in STATIC_FEATURES:
        return 'static'
    # Cross-quantity features: e.g., FREQ_VOLT_34_corr
    parts = feat_name.split('_')
    if len(parts) >= 3 and parts[0] in PHYS_COLORS and parts[1] in PHYS_COLORS:
        return 'cross'
    # Standard features: SHEET_BUS_TIMESTEP or SHEET_BUS_stat
    if parts[0] in PHYS_COLORS:
        return parts[0]
    return 'static'


def get_bar_color(feat_name: str) -> str:
    """Get color for a feature based on its physical quantity type."""
    category = classify_feature(feat_name)
    return PHYS_COLORS.get(category, PHYS_COLORS['static'])


def save_fig(fig, path_stem: str):
    """Save figure as both PNG and PDF."""
    fig.savefig(f'{path_stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{path_stem}.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {path_stem}.png/.pdf")


# ════════════════════════════════════════════════════════════════════
# 5a: mRMR Feature Ranking — Top-30 Horizontal Bar Chart
# ════════════════════════════════════════════════════════════════════

def exp5a_mrmr_ranking(exp1_dir: str, result_dir: str):
    """Plot top-30 mRMR features as color-coded horizontal bar charts."""
    logger.info("=" * 60)
    logger.info("Exp 5a: mRMR Feature Ranking (top-30 bar chart)")
    logger.info("=" * 60)

    mrmr_path = f'{exp1_dir}/mrmr_features.json'
    if not os.path.exists(mrmr_path):
        logger.error(f"mRMR features not found: {mrmr_path}")
        return
    with open(mrmr_path) as f:
        mrmr_data = json.load(f)

    for target_key, target_label in [('y1', TARGET_LABELS['y1']),
                                      ('y2', TARGET_LABELS['y2'])]:
        ranked = mrmr_data[target_key]
        top30 = ranked[:30]

        colors = [get_bar_color(f) for f in top30]
        ranks = list(range(1, 31))

        fig, ax = plt.subplots(figsize=(5.5, 6.5))

        # Horizontal bars: rank 1 at top
        y_pos = np.arange(30)
        ax.barh(y_pos, ranks[::-1], color=colors[::-1], edgecolor='white',
                linewidth=0.3, height=0.75)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top30[::-1], fontsize=8, fontfamily=_TNR)
        ax.set_xlabel('mRMR Rank', fontsize=11)
        ax.set_title(f'Top-30 mRMR Features — {target_label}', fontsize=12)
        ax.invert_xaxis()  # rank 1 = longest bar
        ax.set_xlim(31, 0)

        # Build legend from unique categories present
        seen = {}
        for feat in top30:
            cat = classify_feature(feat)
            if cat not in seen:
                seen[cat] = PHYS_COLORS[cat]
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white',
                          linewidth=0.3, label=cat.upper() if cat != 'static' else 'Static')
            for cat, color in seen.items()
        ]
        ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
                  framealpha=0.9, edgecolor='gray')

        ax.tick_params(axis='both', which='both', length=3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        save_fig(fig, f'{result_dir}/5a_mrmr_top30_{target_key}')

        # Log category distribution
        cat_counts = {}
        for feat in top30:
            cat = classify_feature(feat)
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        logger.info(f"  {target_key} top-30 category distribution: {cat_counts}")


# ════════════════════════════════════════════════════════════════════
# 5b: ALE (Accumulated Local Effects) Plots
# ════════════════════════════════════════════════════════════════════

def exp5b_ale_plots(rep_dir: str, exp1_dir: str, result_dir: str, ms: int = 10):
    """Generate ALE plots for LightGBM top-10 mRMR features per target."""
    logger.info("=" * 60)
    logger.info("Exp 5b: ALE Plots (LightGBM, top-10 mRMR features)")
    logger.info("=" * 60)

    from PyALE import ale

    # Load mRMR features
    mrmr_path = f'{exp1_dir}/mrmr_features.json'
    if not os.path.exists(mrmr_path):
        logger.error(f"mRMR features not found: {mrmr_path}")
        return
    with open(mrmr_path) as f:
        mrmr_data = json.load(f)

    # Load raw data + feature names
    data_dir = f'{rep_dir}/repA/ms{ms}'
    X_train_raw = np.load(f'{data_dir}/X_train.npy')
    X_test_raw = np.load(f'{data_dir}/X_test.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    with open(f'{data_dir}/feature_names.json') as f:
        feature_names = json.load(f)

    # Replicate exp1 pipeline: select union features, fit scaler
    union_feats = mrmr_data['union']
    feat_idx = [feature_names.index(f) for f in union_feats]
    scaler_A = StandardScaler().fit(X_train_raw[:, feat_idx])
    X_test_n = scaler_A.transform(X_test_raw[:, feat_idx]).astype(np.float32)

    for target_key, ti in [('y1', 0), ('y2', 1)]:
        target_label = TARGET_LABELS[target_key]
        lgb_path = f'{exp1_dir}/LightGBM/{target_key}_lgb_model.pkl'
        if not os.path.exists(lgb_path):
            logger.warning(f"  LightGBM model not found: {lgb_path}, skipping {target_key}")
            continue

        with open(lgb_path, 'rb') as f:
            lgb_model = pickle.load(f)

        # Top-10 mRMR features for this target
        top10_names = mrmr_data[target_key][:10]

        # Build test DataFrame with union features (same column order as training)
        X_test_df = pd.DataFrame(X_test_n, columns=union_feats)

        logger.info(f"  Generating ALE for {target_key}: {top10_names}")

        # Grid of ALE subplots: 2 rows x 5 cols
        fig, axes = plt.subplots(2, 5, figsize=(14, 5.5))
        axes_flat = axes.flatten()

        for fi, feat_name in enumerate(top10_names):
            if feat_name not in union_feats:
                logger.warning(f"    Feature {feat_name} not in union, skipping")
                axes_flat[fi].set_visible(False)
                continue

            ax = axes_flat[fi]
            try:
                ale_result = ale(
                    X=X_test_df,
                    model=lgb_model,
                    feature=[feat_name],
                    grid_size=50,
                    include_CI=True,
                    plot=False,
                )

                # ale_result is a DataFrame with index=feature bins, column='eff'
                x_vals = ale_result.index.to_numpy()
                y_vals = ale_result['eff'].to_numpy()

                color = get_bar_color(feat_name)
                ax.plot(x_vals, y_vals, color=color, linewidth=1.5)

                # Plot CI if available
                if 'lowerCI_95%' in ale_result.columns:
                    lower = ale_result['lowerCI_95%'].to_numpy()
                    upper = ale_result['upperCI_95%'].to_numpy()
                    ax.fill_between(x_vals, lower, upper, color=color, alpha=0.15)

                ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
                ax.set_xlabel(feat_name, fontsize=8)
                if fi % 5 == 0:
                    ax.set_ylabel('ALE', fontsize=9)
                ax.tick_params(labelsize=7)
                ax.set_title(f'#{fi+1}', fontsize=9, fontweight='bold')

            except Exception as e:
                logger.warning(f"    ALE failed for {feat_name}: {e}")
                ax.text(0.5, 0.5, f'Error:\n{feat_name}', ha='center', va='center',
                        fontsize=7, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(f'ALE Plots — LightGBM — {target_label}', fontsize=12, y=1.02)
        fig.tight_layout()
        save_fig(fig, f'{result_dir}/5b_ale_{target_key}')


# ════════════════════════════════════════════════════════════════════
# 5c: KAN Activation Visualization
# ════════════════════════════════════════════════════════════════════

def exp5c_kan_activations(rep_dir: str, exp1_dir: str, result_dir: str,
                          ms: int = 10, device: str = 'cuda'):
    """Extract and plot KAN intermediate activation distributions for top-5 features."""
    logger.info("=" * 60)
    logger.info("Exp 5c: KAN Activation Visualization")
    logger.info("=" * 60)

    from models.kan_model import KANRegressor, build_kan_from_trial

    # Load mRMR features
    mrmr_path = f'{exp1_dir}/mrmr_features.json'
    if not os.path.exists(mrmr_path):
        logger.error(f"mRMR features not found: {mrmr_path}")
        return
    with open(mrmr_path) as f:
        mrmr_data = json.load(f)

    # Load raw data
    data_dir = f'{rep_dir}/repA/ms{ms}'
    X_train_raw = np.load(f'{data_dir}/X_train.npy')
    X_test_raw = np.load(f'{data_dir}/X_test.npy')
    with open(f'{data_dir}/feature_names.json') as f:
        feature_names = json.load(f)

    KAN_K = 100  # Same as exp1

    target_key = 'y1'  # Visualize y1 model
    target_label = TARGET_LABELS[target_key]

    # Load best params
    params_path = f'{exp1_dir}/KAN/{target_key}_best_params.json'
    model_path = f'{exp1_dir}/KAN/{target_key}_model.pth'
    if not os.path.exists(params_path) or not os.path.exists(model_path):
        logger.error(f"KAN model or params not found for {target_key}")
        return

    with open(params_path) as f:
        best_params = json.load(f)

    # Prepare features: per-target top-100 mRMR, same as exp1
    kan_feat_names = mrmr_data[target_key][:KAN_K]
    kan_feat_idx = [feature_names.index(f) for f in kan_feat_names if f in feature_names]

    kan_scaler = StandardScaler().fit(X_train_raw[:, kan_feat_idx])
    X_test_n = kan_scaler.transform(X_test_raw[:, kan_feat_idx]).astype(np.float32)
    input_dim = X_test_n.shape[1]

    # Reconstruct model architecture from best params
    hidden_dims = [best_params['hidden_1'], best_params['hidden_2'], best_params['hidden_3']]
    grid_size = best_params['grid_size']
    spline_order = best_params['spline_order']

    model = KANRegressor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        grid_size=grid_size,
        spline_order=spline_order,
    )
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    if torch.cuda.is_available() and device == 'cuda':
        model = model.to(device)

    # ── Register forward hooks to capture intermediate activations ──
    activations = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook_fn

    # efficient-kan KAN has layers as a ModuleList inside model.kan
    # Register hooks on each KANLinear layer
    hooks = []
    for i, layer in enumerate(model.kan.layers):
        h = layer.register_forward_hook(make_hook(f'layer_{i}'))
        hooks.append(h)

    # Forward pass on test data
    X_tensor = torch.from_numpy(X_test_n)
    if torch.cuda.is_available() and device == 'cuda':
        X_tensor = X_tensor.to(device)

    with torch.no_grad():
        _ = model(X_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Clear GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Plot 1: First-layer activation distributions for top-5 input features ──
    # The first KANLinear layer maps input_dim -> hidden_dims[0]
    # Each input neuron has grid_size+spline_order basis functions
    # We look at the activation output of the first layer
    layer0_act = activations.get('layer_0')
    if layer0_act is None:
        logger.error("  Failed to capture layer_0 activations")
        return

    logger.info(f"  Layer 0 activation shape: {layer0_act.shape}")

    top5_names = kan_feat_names[:5]

    # Plot: activation output distribution of the first layer neurons
    # corresponding to top-5 features' influence
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    for fi, feat_name in enumerate(top5_names):
        ax = axes[fi]
        color = get_bar_color(feat_name)

        # Input feature value vs aggregated first-layer output
        input_vals = X_test_n[:, fi]

        # Sort by input value for a clean scatter/line plot
        sort_idx = np.argsort(input_vals)
        x_sorted = input_vals[sort_idx]

        # Each output neuron receives contribution from this input via spline
        # Show distribution of first few output neurons as a function of this input
        n_out = min(layer0_act.shape[1], 4)

        for oi in range(n_out):
            y_sorted = layer0_act[sort_idx, oi]
            # Bin and average for smoother visualization
            n_bins = 50
            bin_edges = np.linspace(x_sorted.min(), x_sorted.max(), n_bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_means = np.zeros(n_bins)
            for bi in range(n_bins):
                mask = (x_sorted >= bin_edges[bi]) & (x_sorted < bin_edges[bi + 1])
                if mask.sum() > 0:
                    bin_means[bi] = y_sorted[mask].mean()
            ax.plot(bin_centers, bin_means, linewidth=1.0, alpha=0.6,
                    label=f'neuron {oi}')

        ax.set_xlabel(feat_name, fontsize=8)
        if fi == 0:
            ax.set_ylabel('Activation', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_title(f'#{fi+1}', fontsize=9, fontweight='bold')
        if fi == 4:
            ax.legend(fontsize=6, loc='upper right', framealpha=0.7)

    fig.suptitle(f'KAN Layer-0 Activation Profiles — {target_label}', fontsize=12, y=1.05)
    fig.tight_layout()
    save_fig(fig, f'{result_dir}/5c_kan_activation_profiles_{target_key}')

    # ── Plot 2: Activation distribution histograms per layer ──
    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]

    for li in range(n_layers):
        ax = axes[li]
        act = activations[f'layer_{li}']
        flat = act.flatten()

        ax.hist(flat, bins=80, density=True, color='steelblue', alpha=0.7,
                edgecolor='white', linewidth=0.3)
        ax.set_xlabel('Activation Value', fontsize=9)
        if li == 0:
            ax.set_ylabel('Density', fontsize=9)
        ax.set_title(f'Layer {li} ({act.shape[1]} neurons)', fontsize=10)
        ax.tick_params(labelsize=8)

        # Add statistics annotation
        mu, sigma = flat.mean(), flat.std()
        ax.axvline(mu, color='red', linewidth=1, linestyle='--', alpha=0.7)
        ax.text(0.95, 0.95, f'$\\mu$={mu:.3f}\n$\\sigma$={sigma:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))

    fig.suptitle(f'KAN Activation Distributions — {target_label}', fontsize=12, y=1.03)
    fig.tight_layout()
    save_fig(fig, f'{result_dir}/5c_kan_activation_dist_{target_key}')


# ════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════

def run_experiment5(rep_dir: str, exp1_dir: str, result_dir: str,
                    ms: int = 10, device: str = 'cuda',
                    sub: str = 'all'):
    """Run Experiment 5 (interpretability analysis).

    Args:
        rep_dir: path to representation data (ieee39_v8_80_10_10)
        exp1_dir: path to exp1 results (trained models, mRMR features)
        result_dir: output directory for exp5
        ms: measurement window in cycles
        device: 'cuda' or 'cpu'
        sub: which sub-experiments to run ('all', '5a', '5b', '5c')
    """
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment 5: Interpretability Analysis")
    logger.info(f"  rep_dir:    {rep_dir}")
    logger.info(f"  exp1_dir:   {exp1_dir}")
    logger.info(f"  result_dir: {result_dir}")
    logger.info(f"  sub:        {sub}")

    if sub in ('all', '5a'):
        exp5a_mrmr_ranking(exp1_dir, result_dir)

    if sub in ('all', '5b'):
        exp5b_ale_plots(rep_dir, exp1_dir, result_dir, ms)

    if sub in ('all', '5c'):
        exp5c_kan_activations(rep_dir, exp1_dir, result_dir, ms, device)

    logger.info("Experiment 5 complete!")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Exp5: Interpretability Analysis')
    p.add_argument('--rep-dir',
                   default='data/ieee39_v8_80_10_10')
    p.add_argument('--exp1-dir',
                   default=str(PROJECT_ROOT / 'results' / 'ieee39' / 'exp1'))
    p.add_argument('--result-dir',
                   default=str(PROJECT_ROOT / 'results' / 'ieee39' / 'exp5'))
    p.add_argument('--ms', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--sub', default='all',
                   choices=['all', '5a', '5b', '5c'],
                   help='Which sub-experiment to run')
    args = p.parse_args()

    run_experiment5(args.rep_dir, args.exp1_dir, args.result_dir,
                    args.ms, args.device, args.sub)
