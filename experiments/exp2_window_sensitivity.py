#!/usr/bin/env python3
"""
Experiment 2: Observation Window Sensitivity Analysis.

Trains 5 representative models (LightGBM, ConvLSTM, PatchTST, Mamba, ST-GCN)
across 5 observation windows (ms=1,5,10,15,25) using fixed hyperparameters
from Experiment 1.

For ms=10, exp1 results are reused directly (no retraining).
For other ms values, models are retrained with exp1 best params but
input-dimension-dependent params (patch_len, stride) are auto-adjusted.

Output: results/ieee39/exp2/metrics_summary.json + sensitivity curves figure.

Usage:
    python experiments/exp2_window_sensitivity.py
    python experiments/exp2_window_sensitivity.py --ms 1  # single ms value
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
import time
import pickle
import numpy as np
import torch
import optuna
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import mae_loss, rmse_loss, mape_loss, smape_loss
from utils.mrmr_select import mrmr_select_union
from models.lightgbm_model import LightGBMModel
from models.convlstm_model import ConvLSTMFreqModel, build_convlstm_from_trial
from models.patchtst_model import PatchTSTFreqModel, build_patchtst_from_trial
from models.mamba_model import MambaFreqModel, build_mamba_from_trial
from models.stgcn_model import STGCNBatchedModel, build_stgcn_from_trial, normalize_adjacency
from models.base_nn import NNTrainer
from data_proc.datasets import (
    TabularDataset, TensorDataset, GraphDataset, adjacency_to_edge_index
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

REP_DIR = str(data_path('ieee39_v8_80_10_10'))
ADJ_PATH = str(data_path('ieee39_v8/adjacency/adjacency.npy'))
EXP1_DIR = str(work_path('results/ieee39/exp1'))
EXP2_DIR = str(work_path('results/ieee39/exp2'))
MS_VALUES = [1, 5, 10, 15, 25]
MODELS = ['LightGBM', 'ConvLSTM', 'PatchTST', 'Mamba', 'ST-GCN']


def evaluate(y_true, y_pred):
    metrics = {}
    for i, name in enumerate(['y1', 'y2']):
        yt, yp = y_true[:, i], y_pred[:, i]
        metrics[f'{name}_MAE'] = float(mae_loss(yt, yp))
        metrics[f'{name}_RMSE'] = float(rmse_loss(yt, yp))
        metrics[f'{name}_MAPE'] = float(mape_loss(yt, yp))
        metrics[f'{name}_SMAPE'] = float(smape_loss(yt, yp))
    return metrics


def load_exp1_params():
    """Load best hyperparameters from Experiment 1."""
    params = {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # LightGBM: per-target params from Optuna DB
    for tgt in ['y1', 'y2']:
        db = f'{EXP1_DIR}/LightGBM/optuna_{tgt}.db'
        studies = optuna.get_all_study_names(f'sqlite:///{db}')
        study = optuna.load_study(study_name=studies[0], storage=f'sqlite:///{db}')
        params[f'LightGBM_{tgt}'] = study.best_params

    # DL models: from best_params.json
    for model in ['ConvLSTM', 'PatchTST', 'Mamba', 'ST-GCN']:
        bp_path = f'{EXP1_DIR}/{model}/best_params.json'
        with open(bp_path) as f:
            params[model] = json.load(f)

    return params


def load_data_for_ms(ms):
    """Load and preprocess data for a specific ms value."""
    logger.info(f"Loading data for ms={ms}...")

    # RepA
    X_train_A = np.load(f'{REP_DIR}/repA/ms{ms}/X_train.npy')
    X_val_A = np.load(f'{REP_DIR}/repA/ms{ms}/X_val.npy')
    X_test_A = np.load(f'{REP_DIR}/repA/ms{ms}/X_test.npy')
    y_train = np.load(f'{REP_DIR}/repA/ms{ms}/y_train.npy')
    y_val = np.load(f'{REP_DIR}/repA/ms{ms}/y_val.npy')
    y_test = np.load(f'{REP_DIR}/repA/ms{ms}/y_test.npy')
    with open(f'{REP_DIR}/repA/ms{ms}/feature_names.json') as f:
        feature_names = json.load(f)

    # RepB
    Xt_train = np.load(f'{REP_DIR}/repB/ms{ms}/X_temporal_train.npy')
    Xs_train = np.load(f'{REP_DIR}/repB/ms{ms}/X_static_train.npy')
    Xt_val = np.load(f'{REP_DIR}/repB/ms{ms}/X_temporal_val.npy')
    Xs_val = np.load(f'{REP_DIR}/repB/ms{ms}/X_static_val.npy')
    Xt_test = np.load(f'{REP_DIR}/repB/ms{ms}/X_temporal_test.npy')
    Xs_test = np.load(f'{REP_DIR}/repB/ms{ms}/X_static_test.npy')

    # RepC
    Xn_train = np.load(f'{REP_DIR}/repC/ms{ms}/X_node_train.npy')
    Xn_val = np.load(f'{REP_DIR}/repC/ms{ms}/X_node_val.npy')
    Xn_test = np.load(f'{REP_DIR}/repC/ms{ms}/X_node_test.npy')

    with open(f'{REP_DIR}/repB/ms{ms}/meta.json') as f:
        meta = json.load(f)
    static_names = meta['static_names']

    n_timesteps = Xt_train.shape[1]
    n_generators = Xt_train.shape[2]
    n_features = Xt_train.shape[3]

    # mRMR on RepA (per-ms, cached)
    mrmr_dir = f'{EXP2_DIR}/mrmr_cache'
    Path(mrmr_dir).mkdir(parents=True, exist_ok=True)
    mrmr_path = f'{mrmr_dir}/mrmr_repA_ms{ms}.json'

    if os.path.exists(mrmr_path):
        with open(mrmr_path) as f:
            mrmr_data = json.load(f)
        selected_features = mrmr_data['union']
    else:
        logger.info(f"  Running mRMR for ms={ms}...")
        selected_features, y1_feats, y2_feats = mrmr_select_union(
            X_train_A, y_train, feature_names, adaptive=True)
        with open(mrmr_path, 'w') as f:
            json.dump({'union': selected_features, 'y1': y1_feats, 'y2': y2_feats}, f)

    feat_idx = [feature_names.index(f) for f in selected_features]

    # Normalize RepA
    scaler_A = StandardScaler().fit(X_train_A[:, feat_idx])
    X_train_n = scaler_A.transform(X_train_A[:, feat_idx]).astype(np.float32)
    X_val_n = scaler_A.transform(X_val_A[:, feat_idx]).astype(np.float32)
    X_test_n = scaler_A.transform(X_test_A[:, feat_idx]).astype(np.float32)

    # mRMR on x_static (per-ms, cached)
    mrmr_static_path = f'{mrmr_dir}/mrmr_static_ms{ms}.json'
    if os.path.exists(mrmr_static_path):
        with open(mrmr_static_path) as f:
            static_selected = json.load(f)['union']
    else:
        logger.info(f"  Running mRMR on x_static for ms={ms}...")
        static_selected, s_y1, s_y2 = mrmr_select_union(
            Xs_train, y_train, static_names, adaptive=True)
        with open(mrmr_static_path, 'w') as f:
            json.dump({'union': static_selected, 'y1': s_y1, 'y2': s_y2}, f)

    static_idx = [static_names.index(f) for f in static_selected]
    Xs_train_sel = Xs_train[:, static_idx]
    Xs_val_sel = Xs_val[:, static_idx]
    Xs_test_sel = Xs_test[:, static_idx]
    n_static = len(static_idx)

    # Normalize RepB
    B, T, N, C = Xt_train.shape
    sc_t = StandardScaler().fit(Xt_train.reshape(-1, N * C))
    Xt_train_n = sc_t.transform(Xt_train.reshape(-1, N*C)).reshape(B, T, N, C).astype(np.float32)
    Xt_val_n = sc_t.transform(Xt_val.reshape(-1, N*C)).reshape(Xt_val.shape).astype(np.float32)
    Xt_test_n = sc_t.transform(Xt_test.reshape(-1, N*C)).reshape(Xt_test.shape).astype(np.float32)

    sc_s = StandardScaler().fit(Xs_train_sel)
    Xs_train_n = sc_s.transform(Xs_train_sel).astype(np.float32)
    Xs_val_n = sc_s.transform(Xs_val_sel).astype(np.float32)
    Xs_test_n = sc_s.transform(Xs_test_sel).astype(np.float32)

    # RepC (transposed RepB)
    Xn_train_n = np.transpose(Xt_train_n, (0, 2, 1, 3))
    Xn_val_n = np.transpose(Xt_val_n, (0, 2, 1, 3))
    Xn_test_n = np.transpose(Xt_test_n, (0, 2, 1, 3))

    # Target scaler
    scaler_y = StandardScaler().fit(y_train)
    y_train_n = scaler_y.transform(y_train).astype(np.float32)
    y_val_n = scaler_y.transform(y_val).astype(np.float32)

    logger.info(f"  RepA: {X_train_n.shape}, RepB: {Xt_train_n.shape}, "
                f"n_static: {n_static}, T={n_timesteps}")

    return {
        'X_train_n': X_train_n, 'X_val_n': X_val_n, 'X_test_n': X_test_n,
        'Xt_train_n': Xt_train_n, 'Xt_val_n': Xt_val_n, 'Xt_test_n': Xt_test_n,
        'Xs_train_n': Xs_train_n, 'Xs_val_n': Xs_val_n, 'Xs_test_n': Xs_test_n,
        'Xn_train_n': Xn_train_n, 'Xn_val_n': Xn_val_n, 'Xn_test_n': Xn_test_n,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'y_train_n': y_train_n, 'y_val_n': y_val_n,
        'scaler_y': scaler_y,
        'n_timesteps': n_timesteps, 'n_generators': n_generators,
        'n_features': n_features, 'n_static': n_static,
    }


def train_lightgbm(data, params, ms, device='cpu'):
    """Train LightGBM with fixed hyperparameters from exp1."""
    mdir = f'{EXP2_DIR}/ms{ms}/LightGBM'
    Path(mdir).mkdir(parents=True, exist_ok=True)

    if os.path.exists(f'{mdir}/metrics.json'):
        with open(f'{mdir}/metrics.json') as f:
            return json.load(f)

    logger.info(f"  [LightGBM ms={ms}] Training with exp1 params...")
    lgb_model = LightGBMModel(n_trials=0, seed=SEED)

    # Set fixed params from exp1
    lgb_model.best_params = {
        'y1': params['LightGBM_y1'],
        'y2': params['LightGBM_y2'],
    }

    t0 = time.time()
    lgb_model.fit(data['X_train_n'], data['y_train'], data['X_val_n'], data['y_val'],
                  checkpoint_dir=mdir, skip_optuna=True)
    train_time = time.time() - t0

    y_pred = lgb_model.predict(data['X_test_n'])
    metrics = evaluate(data['y_test'], y_pred)
    metrics['train_time_s'] = train_time

    np.save(f'{mdir}/preds.npy', y_pred)
    with open(f'{mdir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_dl_model(name, build_fn, data, params, ms, device='cuda',
                   use_amp=True):
    """Train a DL model with fixed hyperparameters from exp1."""
    mdir = f'{EXP2_DIR}/ms{ms}/{name}'
    Path(mdir).mkdir(parents=True, exist_ok=True)

    if os.path.exists(f'{mdir}/metrics.json'):
        with open(f'{mdir}/metrics.json') as f:
            return json.load(f)

    logger.info(f"  [{name} ms={ms}] Training with exp1 params...")
    p = params[name]
    scaler_y = data['scaler_y']
    n_ts = data['n_timesteps']
    n_gen = data['n_generators']
    n_feat = data['n_features']
    n_stat = data['n_static']

    # Mock Optuna trial with fixed values
    class FixedTrial:
        """Returns fixed parameter values for build_fn compatibility."""
        def __init__(self, fixed_params):
            self._params = dict(fixed_params)
            self.params = dict(fixed_params)

        def suggest_int(self, name, low, high, **kw):
            return self._params[name]

        def suggest_float(self, name, low, high, **kw):
            return self._params[name]

        def suggest_categorical(self, name, choices, **kw):
            return self._params[name]

    # Auto-adjust sequence-dependent params
    adjusted_params = dict(p)
    if name == 'PatchTST':
        pl = adjusted_params.get('patch_len', 3)
        st = adjusted_params.get('stride', 2)
        if n_ts < pl:
            adjusted_params['patch_len'] = max(1, n_ts)
        if adjusted_params.get('stride', 1) > adjusted_params['patch_len']:
            adjusted_params['stride'] = max(1, adjusted_params['patch_len'])

    mock_trial = FixedTrial(adjusted_params)

    # Build model and datasets
    if name in ('ConvLSTM',):
        model, model_kwargs = build_fn(mock_trial, n_features=n_feat,
                                        n_generators=n_gen, n_static=n_stat)
        train_ds = TensorDataset(data['Xt_train_n'], data['Xs_train_n'], data['y_train_n'])
        val_ds = TensorDataset(data['Xt_val_n'], data['Xs_val_n'], data['y_val_n'])
        test_ds = TensorDataset(data['Xt_test_n'], data['Xs_test_n'], data['y_test'][:, :2])

    elif name in ('PatchTST', 'Mamba'):
        model, model_kwargs = build_fn(mock_trial, n_features=n_feat,
                                        n_generators=n_gen, n_static=n_stat,
                                        n_timesteps=n_ts)
        train_ds = TensorDataset(data['Xt_train_n'], data['Xs_train_n'], data['y_train_n'])
        val_ds = TensorDataset(data['Xt_val_n'], data['Xs_val_n'], data['y_val_n'])
        test_ds = TensorDataset(data['Xt_test_n'], data['Xs_test_n'], data['y_test'][:, :2])

    elif name == 'ST-GCN':
        adj = np.load(ADJ_PATH)
        adj_norm = normalize_adjacency(adj)
        edge_index, edge_weight = adjacency_to_edge_index(adj)
        model, model_kwargs = build_fn(mock_trial, n_features=n_feat,
                                        n_nodes=n_gen, n_static=n_stat,
                                        adj_norm=adj_norm)
        train_ds = GraphDataset(data['Xn_train_n'], data['Xs_train_n'], data['y_train_n'],
                                edge_index, edge_weight)
        val_ds = GraphDataset(data['Xn_val_n'], data['Xs_val_n'], data['y_val_n'],
                              edge_index, edge_weight)
        test_ds = GraphDataset(data['Xn_test_n'], data['Xs_test_n'], data['y_test'][:, :2],
                               edge_index, edge_weight)

    trainer = NNTrainer(
        model, device=device,
        lr=p['lr'], weight_decay=p['weight_decay'],
        batch_size=p['bs'],
        y1_weight=p['y1_weight'], y2_weight=p['y2_weight'],
        max_epochs=200, patience=20,
        use_amp=use_amp, warmup_epochs=5,
    )

    t0 = time.time()
    _, history = trainer.fit(train_ds, val_ds, checkpoint_dir=mdir)
    train_time = time.time() - t0

    pred_scaled = trainer.predict(test_ds)
    pred = scaler_y.inverse_transform(pred_scaled)

    metrics = evaluate(data['y_test'], pred)
    metrics['train_time_s'] = train_time

    np.save(f'{mdir}/preds.npy', pred)
    with open(f'{mdir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(f'{mdir}/history.json', 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f)
    NNTrainer.plot_loss_curves(history, f'{name} ms={ms}',
                                save_path=f'{mdir}/loss_curves.png')

    # Clean up GPU
    del trainer, model
    torch.cuda.empty_cache()

    return metrics


def copy_exp1_results(ms=10):
    """Copy ms=10 results from exp1 (no retraining needed)."""
    logger.info("Copying ms=10 results from Experiment 1...")
    for model in MODELS:
        src = f'{EXP1_DIR}/{model}/metrics.json'
        dst_dir = f'{EXP2_DIR}/ms{ms}/{model}'
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
        if os.path.exists(src):
            with open(src) as f:
                metrics = json.load(f)
            with open(f'{dst_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"  {model}: copied from exp1")


def plot_sensitivity_curves(all_results):
    """Generate IEEE TII-grade sensitivity curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    ms_vals = sorted(all_results.keys())
    x_vals = [ms * 10 for ms in ms_vals]

    model_styles = {
        'LightGBM': {'color': '#E24A33', 'marker': 's', 'ls': '-'},
        'ConvLSTM': {'color': '#348ABD', 'marker': 'o', 'ls': '-'},
        'PatchTST': {'color': '#988ED5', 'marker': '^', 'ls': '--'},
        'Mamba':    {'color': '#FBC15E', 'marker': 'D', 'ls': '--'},
        'ST-GCN':   {'color': '#8EBA42', 'marker': 'v', 'ls': '-.'},
    }

    metric_pairs = [
        ('y1_MAE', r'$y_1$ ($\Delta f$) MAE'),
        ('y1_MAPE', r'$y_1$ ($\Delta f$) MAPE (%)'),
        ('y2_MAE', r'$y_2$ ($t_{nadir}$) MAE'),
        ('y2_MAPE', r'$y_2$ ($t_{nadir}$) MAPE (%)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (metric_key, ylabel) in zip(axes.flat, metric_pairs):
        for model in MODELS:
            style = model_styles[model]
            vals = []
            for ms in ms_vals:
                m = all_results[ms].get(model, {}).get(metric_key, float('nan'))
                vals.append(m)
            ax.plot(x_vals, vals, label=model, color=style['color'],
                    marker=style['marker'], ls=style['ls'],
                    markersize=6, linewidth=1.5)
        ax.set_xlabel('Observation Window (ms)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_vals)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', framealpha=0.8)

    plt.tight_layout()
    fig.savefig(f'{EXP2_DIR}/sensitivity_curves.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{EXP2_DIR}/sensitivity_curves.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved sensitivity curves: {EXP2_DIR}/sensitivity_curves.{{png,pdf}}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', type=int, default=None,
                        help='Run only this ms value (default: all)')
    parser.add_argument('--model', type=str, default=None,
                        help='Run only this model (default: all)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    Path(EXP2_DIR).mkdir(parents=True, exist_ok=True)
    params = load_exp1_params()

    ms_list = [args.ms] if args.ms else MS_VALUES
    model_list = [args.model] if args.model else MODELS

    # Copy exp1 ms=10 results
    if 10 in ms_list:
        copy_exp1_results(ms=10)

    all_results = {}

    for ms in ms_list:
        logger.info(f"\n{'='*60}\n=== ms={ms} (window={ms*10}ms) ===")
        all_results[ms] = {}

        if ms == 10:
            for model in model_list:
                mpath = f'{EXP2_DIR}/ms{ms}/{model}/metrics.json'
                if os.path.exists(mpath):
                    with open(mpath) as f:
                        all_results[ms][model] = json.load(f)
                    logger.info(f"  {model}: loaded exp1 results")
            continue

        data = load_data_for_ms(ms)

        for model in model_list:
            try:
                if model == 'LightGBM':
                    metrics = train_lightgbm(data, params, ms)
                elif model == 'ConvLSTM':
                    metrics = train_dl_model(
                        model, build_convlstm_from_trial, data, params, ms,
                        device=args.device)
                elif model == 'PatchTST':
                    metrics = train_dl_model(
                        model, build_patchtst_from_trial, data, params, ms,
                        device=args.device, use_amp=True)
                elif model == 'Mamba':
                    metrics = train_dl_model(
                        model, build_mamba_from_trial, data, params, ms,
                        device=args.device)
                elif model == 'ST-GCN':
                    metrics = train_dl_model(
                        model, build_stgcn_from_trial, data, params, ms,
                        device=args.device)
                else:
                    continue

                all_results[ms][model] = metrics
                logger.info(f"  {model}: y1_MAE={metrics['y1_MAE']:.6f}, "
                           f"y2_MAE={metrics['y2_MAE']:.6f}")
            except Exception as e:
                logger.error(f"  {model} FAILED: {e}")
                import traceback
                traceback.print_exc()

        del data

    # Load any previously computed results
    for ms in MS_VALUES:
        if ms not in all_results:
            all_results[ms] = {}
        for model in MODELS:
            if model not in all_results[ms]:
                mpath = f'{EXP2_DIR}/ms{ms}/{model}/metrics.json'
                if os.path.exists(mpath):
                    with open(mpath) as f:
                        all_results[ms][model] = json.load(f)

    # Save summary
    with open(f'{EXP2_DIR}/metrics_summary.json', 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Print results table
    logger.info("\n=== Experiment 2 Results ===")
    for metric in ['y1_MAE', 'y1_MAPE', 'y2_MAE', 'y2_MAPE']:
        logger.info(f"\n{metric}:")
        header = f"{'Model':16s}" + "".join(f"  ms={ms:2d}" for ms in MS_VALUES)
        logger.info(header)
        for model in MODELS:
            row = f"{model:16s}"
            for ms in MS_VALUES:
                val = all_results.get(ms, {}).get(model, {}).get(metric, float('nan'))
                row += f"  {val:.5f}" if not np.isnan(val) else "     N/A"
            logger.info(row)

    # Plot
    if all(ms in all_results and len(all_results[ms]) >= 3 for ms in MS_VALUES):
        plot_sensitivity_curves(all_results)

    logger.info("\nExperiment 2 complete!")


if __name__ == '__main__':
    main()
