#!/usr/bin/env python3
"""
Train a single model independently — enables parallel training.

This script loads the same data and scalers as exp1_main_comparison.py,
trains ONE specified model, and saves results into the per-model subdirectory.
When exp1 runs later, it detects the existing preds.npy and skips that model.

Usage:
    # Train ConvLSTM while LightGBM is running in exp1
    python experiments/train_single_model.py --model ConvLSTM

    # Train multiple GPU models in parallel (separate terminals)
    python experiments/train_single_model.py --model ConvLSTM &
    python experiments/train_single_model.py --model FT-Transformer &

    # Override defaults
    python experiments/train_single_model.py --model PatchTST \
        --rep-dir data/ieee39_v8 \
        --result-dir results/ieee39/exp1 --ms 10 --n-trials 30
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
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
from models.kan_model import KANRegressor, build_kan_from_trial
from models.convlstm_model import ConvLSTMFreqModel, build_convlstm_from_trial
from models.patchtst_model import PatchTSTFreqModel, build_patchtst_from_trial
from models.mamba_model import MambaFreqModel, build_mamba_from_trial
from models.stgcn_model import STGCNBatchedModel, build_stgcn_from_trial, normalize_adjacency
from models.ft_transformer_model import FTTransformerModel, build_ft_transformer_from_trial
from models.tabr_model import TabRModel, build_tabr_from_trial
from models.tcf_model import TargetConditionedFusionModel, build_tcf_from_trial
from models.base_nn import NNTrainer
from data_proc.datasets import (
    TabularDataset, TensorDataset, GraphDataset, adjacency_to_edge_index
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Model registry ──────────────────────────────────────────────
MODEL_REGISTRY = {
    'LightGBM':       {'rep': 'A', 'type': 'gbdt'},
    'KAN':            {'rep': 'A', 'type': 'kan'},
    'ConvLSTM':       {'rep': 'B', 'type': 'multitask', 'build_fn': build_convlstm_from_trial},
    'PatchTST':       {'rep': 'B', 'type': 'multitask', 'build_fn': build_patchtst_from_trial,
                       'use_amp': True},  # AMP re-enabled: Flash SDP disabled in model module
    'Mamba':          {'rep': 'B', 'type': 'multitask', 'build_fn': build_mamba_from_trial},
    'ST-GCN':         {'rep': 'C', 'type': 'multitask', 'build_fn': build_stgcn_from_trial},
    'FT-Transformer': {'rep': 'A_mt', 'type': 'multitask', 'build_fn': build_ft_transformer_from_trial,
                       'use_amp': True},  # AMP safe: tabular input, no channel-independent batching
    'TabR':           {'rep': 'A_mt', 'type': 'tabr', 'build_fn': build_tabr_from_trial},
    'TCF-Net':        {'rep': 'B', 'type': 'multitask', 'build_fn': build_tcf_from_trial,
                       'use_amp': True},
}


def get_model_dir(result_dir: str, name: str) -> str:
    d = os.path.join(result_dir, name)
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


def evaluate(y_true, y_pred):
    metrics = {}
    for i, name in enumerate(['y1', 'y2']):
        yt, yp = y_true[:, i], y_pred[:, i]
        metrics[f'{name}_MAE'] = float(mae_loss(yt, yp))
        metrics[f'{name}_RMSE'] = float(rmse_loss(yt, yp))
        metrics[f'{name}_MAPE'] = float(mape_loss(yt, yp))
        metrics[f'{name}_SMAPE'] = float(smape_loss(yt, yp))
    return metrics


def log_metrics(model_name, metrics):
    logger.info(f"{model_name} results:")
    logger.info(f"  y1: MAE={metrics['y1_MAE']:.6f}  RMSE={metrics['y1_RMSE']:.6f}  "
                f"MAPE={metrics['y1_MAPE']:.2f}%  SMAPE={metrics['y1_SMAPE']:.2f}%")
    logger.info(f"  y2: MAE={metrics['y2_MAE']:.6f}  RMSE={metrics['y2_RMSE']:.6f}  "
                f"MAPE={metrics['y2_MAPE']:.2f}%  SMAPE={metrics['y2_SMAPE']:.2f}%")


def load_data(rep_dir, ms, result_dir):
    """Load all representations, run/load mRMR, compute scalers.

    Returns a dict with all data needed by any model.
    """
    logger.info("Loading data...")

    # RepA
    X_train_A = np.load(f'{rep_dir}/repA/ms{ms}/X_train.npy')
    X_val_A = np.load(f'{rep_dir}/repA/ms{ms}/X_val.npy')
    X_test_A = np.load(f'{rep_dir}/repA/ms{ms}/X_test.npy')
    y_train = np.load(f'{rep_dir}/repA/ms{ms}/y_train.npy')
    y_val = np.load(f'{rep_dir}/repA/ms{ms}/y_val.npy')
    y_test = np.load(f'{rep_dir}/repA/ms{ms}/y_test.npy')
    with open(f'{rep_dir}/repA/ms{ms}/feature_names.json') as f:
        feature_names = json.load(f)

    # RepB
    Xt_train = np.load(f'{rep_dir}/repB/ms{ms}/X_temporal_train.npy')
    Xs_train = np.load(f'{rep_dir}/repB/ms{ms}/X_static_train.npy')
    Xt_val = np.load(f'{rep_dir}/repB/ms{ms}/X_temporal_val.npy')
    Xs_val = np.load(f'{rep_dir}/repB/ms{ms}/X_static_val.npy')
    Xt_test = np.load(f'{rep_dir}/repB/ms{ms}/X_temporal_test.npy')
    Xs_test = np.load(f'{rep_dir}/repB/ms{ms}/X_static_test.npy')

    # RepC
    Xn_train = np.load(f'{rep_dir}/repC/ms{ms}/X_node_train.npy')
    Xn_val = np.load(f'{rep_dir}/repC/ms{ms}/X_node_val.npy')
    Xn_test = np.load(f'{rep_dir}/repC/ms{ms}/X_node_test.npy')

    n_timesteps = Xt_train.shape[1]
    n_generators = Xt_train.shape[2]
    n_features = Xt_train.shape[3]
    n_static = Xs_train.shape[1]

    logger.info(f"RepA: {X_train_A.shape}, RepB: {Xt_train.shape}, n_static: {n_static}")

    # ── mRMR (reuse if cached) ──
    mrmr_path = f'{result_dir}/mrmr_features.json'
    if os.path.exists(mrmr_path):
        logger.info("Loading cached mRMR features...")
        with open(mrmr_path) as f:
            mrmr_data = json.load(f)
        selected_features = mrmr_data['union']
        y1_feats = mrmr_data['y1']
        y2_feats = mrmr_data['y2']
    else:
        logger.info("Running adaptive mRMR (may take ~15 min)...")
        selected_features, y1_feats, y2_feats = mrmr_select_union(
            X_train_A, y_train, feature_names, adaptive=True)
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        with open(mrmr_path, 'w') as f:
            json.dump({'union': selected_features, 'y1': y1_feats, 'y2': y2_feats}, f)

    feat_idx = [feature_names.index(f) for f in selected_features]

    # ── Normalize RepA ──
    scaler_A = StandardScaler().fit(X_train_A[:, feat_idx])
    X_train_n = scaler_A.transform(X_train_A[:, feat_idx]).astype(np.float32)
    X_val_n = scaler_A.transform(X_val_A[:, feat_idx]).astype(np.float32)
    X_test_n = scaler_A.transform(X_test_A[:, feat_idx]).astype(np.float32)

    # ── mRMR on x_static for DL models ──
    # x_static has 1394 dims (7 base + 1387 per-bus temporal stats).
    # The 1387 stats overlap heavily with what the temporal branch sees.
    # Select top-K via mRMR to reduce noise and keep only complementary features.
    mrmr_static_path = f'{result_dir}/mrmr_static_features.json'
    with open(f'{rep_dir}/repB/ms{ms}/meta.json') as f:
        static_names = json.load(f)['static_names']

    if os.path.exists(mrmr_static_path):
        logger.info("Loading cached mRMR static features...")
        with open(mrmr_static_path) as f:
            static_mrmr = json.load(f)
        static_selected = static_mrmr['union']
    else:
        logger.info("Running mRMR on x_static features...")
        static_selected, s_y1, s_y2 = mrmr_select_union(
            Xs_train, y_train, static_names, adaptive=True)
        with open(mrmr_static_path, 'w') as f:
            json.dump({'union': static_selected, 'y1': s_y1, 'y2': s_y2}, f)

    static_idx = [static_names.index(f) for f in static_selected]
    Xs_train_sel = Xs_train[:, static_idx]
    Xs_val_sel = Xs_val[:, static_idx]
    Xs_test_sel = Xs_test[:, static_idx]
    n_static_sel = len(static_idx)
    logger.info(f"x_static mRMR: {n_static} -> {n_static_sel} features")

    # ── Normalize RepB ──
    B, T, N, C = Xt_train.shape
    sc_t = StandardScaler().fit(Xt_train.reshape(-1, N * C))
    Xt_train_n = sc_t.transform(Xt_train.reshape(-1, N*C)).reshape(B, T, N, C).astype(np.float32)
    Xt_val_n = sc_t.transform(Xt_val.reshape(-1, N*C)).reshape(Xt_val.shape).astype(np.float32)
    Xt_test_n = sc_t.transform(Xt_test.reshape(-1, N*C)).reshape(Xt_test.shape).astype(np.float32)

    sc_s = StandardScaler().fit(Xs_train_sel)
    Xs_train_n = sc_s.transform(Xs_train_sel).astype(np.float32)
    Xs_val_n = sc_s.transform(Xs_val_sel).astype(np.float32)
    Xs_test_n = sc_s.transform(Xs_test_sel).astype(np.float32)

    # RepC
    Xn_train_n = np.transpose(Xt_train_n, (0, 2, 1, 3))
    Xn_val_n = np.transpose(Xt_val_n, (0, 2, 1, 3))
    Xn_test_n = np.transpose(Xt_test_n, (0, 2, 1, 3))

    # Target scaler
    scaler_y = StandardScaler().fit(y_train)
    y_train_n = scaler_y.transform(y_train).astype(np.float32)
    y_val_n = scaler_y.transform(y_val).astype(np.float32)

    return {
        # Raw RepA (for KAN per-target)
        'X_train_A': X_train_A, 'X_val_A': X_val_A, 'X_test_A': X_test_A,
        'feature_names': feature_names, 'feat_idx': feat_idx,
        'y1_feats': y1_feats, 'y2_feats': y2_feats,
        # Normalized RepA (for LGB, FT-T, TabR)
        'X_train_n': X_train_n, 'X_val_n': X_val_n, 'X_test_n': X_test_n,
        # Normalized RepB (x_static is mRMR-selected)
        'Xt_train_n': Xt_train_n, 'Xt_val_n': Xt_val_n, 'Xt_test_n': Xt_test_n,
        'Xs_train_n': Xs_train_n, 'Xs_val_n': Xs_val_n, 'Xs_test_n': Xs_test_n,
        # Normalized RepC
        'Xn_train_n': Xn_train_n, 'Xn_val_n': Xn_val_n, 'Xn_test_n': Xn_test_n,
        # Targets
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'y_train_n': y_train_n, 'y_val_n': y_val_n,
        'scaler_y': scaler_y, 'scaler_A': scaler_A,
        # Dims
        'n_timesteps': n_timesteps, 'n_generators': n_generators,
        'n_features': n_features, 'n_static': n_static_sel,
    }


def train_multitask_model(name, build_fn, train_ds, val_ds, test_ds,
                          data, result_dir, device='cuda',
                          n_trials=50, timeout=7200, use_amp=True,
                          **extra_build_kwargs):
    """Train a single multi-task DL model with Optuna + final retrain."""
    mdir = get_model_dir(result_dir, name)

    if os.path.exists(f'{mdir}/preds.npy'):
        logger.info(f"[{name}] SKIPPED (already done)")
        pred = np.load(f'{mdir}/preds.npy')
        metrics = evaluate(data['y_test'], pred)
        log_metrics(name, metrics)
        return metrics

    logger.info(f"\n{'='*60}\n[{name}] Training...")
    scaler_y = data['scaler_y']
    y_test = data['y_test']

    def _refresh_tabr_cache_for_final_predict(trainer):
        if name != 'TabR':
            return
        if not hasattr(train_ds, 'X') or not hasattr(train_ds, 'y'):
            raise ValueError("TabR cache refresh requires a TabularDataset train split")
        logger.info("  TabR: rebuilding retrieval cache with final weights before prediction")
        trainer.model.build_training_cache(train_ds.X, train_ds.y)
        trainer.model.eval()

    def _make_cache_refresh_fn():
        if name != 'TabR':
            return None
        if not hasattr(train_ds, 'X') or not hasattr(train_ds, 'y'):
            raise ValueError("TabR cache refresh requires a TabularDataset train split")

        def _refresh(model):
            model.build_training_cache(train_ds.X, train_ds.y)

        return _refresh

    # Model-specific batch size choices:
    # PatchTST channel-independent processing multiplies effective batch by n_vars.
    # CUDA SDP kernels have a hard limit of 65535 on the batch dimension.
    # Effective batch = batch_size × n_vars, so cap accordingly.
    if name == 'PatchTST':
        n_vars = extra_build_kwargs.get('n_generators', 10) * extra_build_kwargs.get('n_features', 5)
        max_bs = max(32, (65535 // n_vars // 32) * 32)  # round down to multiple of 32
        bs_choices = sorted(set(bs for bs in [64, 128, 256, 512] if bs <= max_bs))
        if not bs_choices:
            bs_choices = [32]
        search_epochs, search_patience = 30, 8
    elif name == 'FT-Transformer':
        # 1243 feature tokens → O(n²) attention, very slow per epoch
        bs_choices = [512, 1024, 2048]
        search_epochs, search_patience = 20, 6
    elif name == 'TCF-Net':
        # TCF-Net is lightweight relative to ConvLSTM/PatchTST, so small
        # batches leave the GPU underused and make DataLoader overhead dominate.
        bs_choices = [1024, 2048, 4096]
        search_epochs, search_patience = 30, 8
    else:
        bs_choices = [256, 512, 1024, 2048]
        search_epochs, search_patience = 50, 10

    num_workers = 4 if name == 'TCF-Net' else 16

    def objective(trial):
        model, _ = build_fn(trial, **extra_build_kwargs)
        trainer = NNTrainer(
            model, device=device,
            lr=trial.suggest_float('lr', 5e-5, 5e-3, log=True),
            weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            batch_size=trial.suggest_categorical('bs', bs_choices),
            y1_weight=trial.suggest_float('y1_weight', 0.5, 2.0),
            y2_weight=trial.suggest_float('y2_weight', 0.5, 2.0),
                max_epochs=search_epochs, patience=search_patience,
                use_amp=use_amp, warmup_epochs=3,
                num_workers=num_workers,
                cache_refresh_fn=_make_cache_refresh_fn(),
            )
        val_loss, _ = trainer.fit(train_ds, val_ds, verbose=False, trial=trial)
        return val_loss

    study = optuna.create_study(
        study_name=f'{name}_optuna', direction='minimize',
        storage=f'sqlite:///{mdir}/optuna.db',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True,
    )
    n_done = len([t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, n_trials - n_done)
    def _optuna_live_plot(study, trial):
        """Update live Optuna progress plot every 3 trials."""
        if trial.number % 3 != 0:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            trials = [t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE]
            if len(trials) < 2:
                return
            vals = [t.value for t in trials]
            best_so_far = [min(vals[:i+1]) for i in range(len(vals))]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(1, len(vals)+1), vals, 'o-', alpha=0.4, markersize=3, label='Trial val_loss')
            ax.plot(range(1, len(best_so_far)+1), best_so_far, 'r-', linewidth=2, label='Best so far')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Val Loss')
            ax.set_title(f'{name} — Optuna Search (LIVE)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(mdir, 'live_optuna.png'), dpi=100)
            plt.close(fig)
        except Exception:
            pass

    if n_remaining > 0:
        logger.info(f"  Optuna: {n_done} done, {n_remaining} remaining")
        study.optimize(objective, n_trials=n_remaining, timeout=timeout,
                       catch=(Exception,), callbacks=[_optuna_live_plot])
    else:
        logger.info(f"  Optuna: all {n_trials} trials done (resumed)")

    best = study.best_trial
    model, model_kwargs = build_fn(best, **extra_build_kwargs)
    trainer = NNTrainer(
        model, device=device,
        lr=best.params['lr'], weight_decay=best.params['weight_decay'],
        batch_size=best.params['bs'],
        y1_weight=best.params['y1_weight'], y2_weight=best.params['y2_weight'],
        max_epochs=200, patience=20,
        use_amp=use_amp, warmup_epochs=5,
        num_workers=num_workers,
        cache_refresh_fn=_make_cache_refresh_fn(),
    )

    t0 = time.time()
    _, history = trainer.fit(train_ds, val_ds, checkpoint_dir=mdir)
    train_time = time.time() - t0

    _refresh_tabr_cache_for_final_predict(trainer)
    pred_scaled = trainer.predict(test_ds)
    pred = scaler_y.inverse_transform(pred_scaled)

    timing = trainer.measure_inference_time(test_ds)
    metrics = evaluate(y_test, pred)
    metrics['train_time_s'] = train_time
    metrics.update({f'timing_{k}': v for k, v in timing.items()})

    # Save all outputs
    trainer.save(f'{mdir}/model.pth', model_kwargs=model_kwargs)
    np.save(f'{mdir}/preds.npy', pred)
    with open(f'{mdir}/best_params.json', 'w') as f:
        json.dump(best.params, f, indent=2)
    with open(f'{mdir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(f'{mdir}/history.json', 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f)

    NNTrainer.plot_loss_curves(history, name, save_path=f'{mdir}/loss_curves.png')
    NNTrainer.plot_scatter(y_test, pred, name, save_path=f'{mdir}/scatter.png')

    log_metrics(name, metrics)
    return metrics


def run_single_model(model_name, rep_dir, adj_path, result_dir,
                     ms=10, device='cuda', n_trials=50):
    """Main entry: train one model."""
    info = MODEL_REGISTRY.get(model_name)
    if info is None:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")

    mdir = get_model_dir(result_dir, model_name)
    if os.path.exists(f'{mdir}/preds.npy'):
        logger.info(f"[{model_name}] Already done, skipping.")
        return

    data = load_data(rep_dir, ms, result_dir)

    # Build datasets
    train_B = TensorDataset(data['Xt_train_n'], data['Xs_train_n'], data['y_train_n'])
    val_B = TensorDataset(data['Xt_val_n'], data['Xs_val_n'], data['y_val_n'])
    test_B = TensorDataset(data['Xt_test_n'], data['Xs_test_n'], data['y_test'][:, :2])

    adj = np.load(adj_path)
    edge_index, edge_weight = adjacency_to_edge_index(adj)
    adj_norm = normalize_adjacency(adj)

    train_C = GraphDataset(data['Xn_train_n'], data['Xs_train_n'], data['y_train_n'],
                           edge_index, edge_weight)
    val_C = GraphDataset(data['Xn_val_n'], data['Xs_val_n'], data['y_val_n'],
                         edge_index, edge_weight)
    test_C = GraphDataset(data['Xn_test_n'], data['Xs_test_n'], data['y_test'][:, :2],
                          edge_index, edge_weight)

    train_A_mt = TabularDataset(data['X_train_n'], data['y_train_n'])
    val_A_mt = TabularDataset(data['X_val_n'], data['y_val_n'])
    test_A_mt = TabularDataset(data['X_test_n'], data['y_test'][:, :2])

    n_ts = data['n_timesteps']
    n_gen = data['n_generators']
    n_feat = data['n_features']
    n_stat = data['n_static']

    # ── Dispatch by model type ──
    if info['type'] == 'multitask':
        build_fn = info['build_fn']
        use_amp = info.get('use_amp', True)

        if info['rep'] == 'B':
            ds = (train_B, val_B, test_B)
            kwargs = dict(n_features=n_feat, n_generators=n_gen, n_static=n_stat)
            if model_name in ('PatchTST', 'Mamba', 'TCF-Net'):
                kwargs['n_timesteps'] = n_ts
        elif info['rep'] == 'C':
            ds = (train_C, val_C, test_C)
            kwargs = dict(n_features=n_feat, n_nodes=n_gen, n_static=n_stat,
                          adj_norm=adj_norm)
        elif info['rep'] == 'A_mt':
            ds = (train_A_mt, val_A_mt, test_A_mt)
            kwargs = dict(n_features=data['X_train_n'].shape[1])
        else:
            raise ValueError(f"Unknown rep type: {info['rep']}")

        train_multitask_model(
            model_name, build_fn, ds[0], ds[1], ds[2],
            data, result_dir, device=device,
            n_trials=n_trials, use_amp=use_amp, **kwargs)

    elif info['type'] == 'tabr':
        def _build_tabr_with_cache(trial, **kw):
            model, mk = build_tabr_from_trial(trial, **kw)
            model.build_training_cache(
                torch.FloatTensor(data['X_train_n']),
                torch.FloatTensor(data['y_train_n']))
            return model, mk

        train_multitask_model(
            model_name, _build_tabr_with_cache,
            train_A_mt, val_A_mt, test_A_mt,
            data, result_dir, device=device,
            n_trials=n_trials, n_features=data['X_train_n'].shape[1])

    elif info['type'] == 'gbdt':
        import pickle
        mdir = get_model_dir(result_dir, model_name)
        if os.path.exists(f'{mdir}/preds.npy'):
            logger.info(f"[{model_name}] SKIPPED (already done)")
        else:
            logger.info(f"\n{'='*60}\n[{model_name}] Training...")
            lgb_model = LightGBMModel(n_trials=100, timeout=3600, seed=SEED)
            t0 = time.time()
            lgb_model.fit(data['X_train_n'], data['y_train'], data['X_val_n'], data['y_val'],
                          checkpoint_dir=mdir)
            train_time = time.time() - t0

            y_pred = lgb_model.predict(data['X_test_n'])
            metrics = evaluate(data['y_test'], y_pred)
            timing = lgb_model.measure_inference_time(data['X_test_n'])
            metrics['train_time_s'] = train_time
            metrics.update({f'timing_{k}': v for k, v in timing.items()})

            np.save(f'{mdir}/preds.npy', y_pred)
            with open(f'{mdir}/model.pkl', 'wb') as f:
                pickle.dump(lgb_model, f)
            with open(f'{mdir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            log_metrics(model_name, metrics)

    elif info['type'] == 'kan':
        KAN_K = 100
        mdir = get_model_dir(result_dir, model_name)
        if os.path.exists(f'{mdir}/preds.npy'):
            logger.info(f"[{model_name}] SKIPPED (already done)")
        else:
            logger.info(f"\n{'='*60}\n[{model_name}] Training (per-target top-{KAN_K} mRMR)...")
            y_test = data['y_test']
            scaler_y = data['scaler_y']
            y_train_n = data['y_train_n']
            y_val_n = data['y_val_n']
            kan_preds = np.zeros((len(y_test), 2))

            per_target_feats = [data['y1_feats'][:KAN_K], data['y2_feats'][:KAN_K]]

            for ti, tname in enumerate(['y1', 'y2']):
                pred_ckpt = f'{mdir}/{tname}_pred_scaled.npy'
                if os.path.exists(pred_ckpt):
                    logger.info(f"  KAN {tname}: loading saved predictions (resumed)")
                    pred_scaled = np.load(pred_ckpt)
                    dummy = np.zeros((len(pred_scaled), 2))
                    dummy[:, ti] = pred_scaled.flatten()
                    pred_inv = scaler_y.inverse_transform(dummy)
                    kan_preds[:, ti] = pred_inv[:, ti]
                    continue

                kan_feat_names = per_target_feats[ti]
                kan_feat_idx = [data['feature_names'].index(f) for f in kan_feat_names
                                if f in data['feature_names']]

                kan_scaler = StandardScaler().fit(data['X_train_A'][:, kan_feat_idx])
                kan_X_train = kan_scaler.transform(data['X_train_A'][:, kan_feat_idx]).astype(np.float32)
                kan_X_val = kan_scaler.transform(data['X_val_A'][:, kan_feat_idx]).astype(np.float32)
                kan_X_test = kan_scaler.transform(data['X_test_A'][:, kan_feat_idx]).astype(np.float32)
                kan_dim = kan_X_train.shape[1]

                logger.info(f"  KAN {tname}: {kan_dim} features")

                def kan_objective(trial):
                    model, _ = build_kan_from_trial(trial, kan_dim)
                    trainer = NNTrainer(
                        model, device=device,
                        lr=trial.suggest_float('lr', 5e-5, 5e-3, log=True),
                        weight_decay=trial.suggest_float('wd', 1e-6, 1e-2, log=True),
                        max_epochs=100, patience=15,
                        batch_size=trial.suggest_categorical('bs', [256, 512, 1024]),
                        loss_type='huber', warmup_epochs=3,
                    )
                    tds = TabularDataset(kan_X_train, y_train_n[:, ti:ti+1])
                    vds = TabularDataset(kan_X_val, y_val_n[:, ti:ti+1])
                    val_loss, _ = trainer.fit(tds, vds, verbose=False, trial=trial)
                    return val_loss

                study = optuna.create_study(
                    study_name=f'KAN_{tname}', direction='minimize',
                    storage=f'sqlite:///{mdir}/optuna_{tname}.db',
                    sampler=optuna.samplers.TPESampler(seed=SEED),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
                    load_if_exists=True,
                )
                n_done = len([t for t in study.trials
                              if t.state == optuna.trial.TrialState.COMPLETE])
                n_remaining = max(0, n_trials - n_done)
                def _kan_optuna_cb(study, trial, _tn=tname, _md=mdir):
                    if trial.number % 3 == 0:
                        try:
                            import matplotlib; matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            ts = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                            if len(ts) < 2: return
                            vs = [t.value for t in ts]
                            bsf = [min(vs[:i+1]) for i in range(len(vs))]
                            fig, ax = plt.subplots(figsize=(8,4))
                            ax.plot(range(1,len(vs)+1), vs, 'o-', alpha=0.4, markersize=3)
                            ax.plot(range(1,len(bsf)+1), bsf, 'r-', lw=2)
                            ax.set_title(f'KAN {_tn} Optuna (LIVE)'); ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            fig.savefig(f'{_md}/live_optuna_{_tn}.png', dpi=100); plt.close(fig)
                        except Exception: pass

                if n_remaining > 0:
                    logger.info(f"  KAN {tname} Optuna: {n_done} done, {n_remaining} remaining")
                    study.optimize(kan_objective, n_trials=n_remaining, timeout=7200,
                                   catch=(Exception,), callbacks=[_kan_optuna_cb])

                best = study.best_trial
                model, kan_kwargs = build_kan_from_trial(best, kan_dim)
                trainer = NNTrainer(
                    model, device=device,
                    lr=best.params['lr'], weight_decay=best.params['wd'],
                    max_epochs=200, patience=25,
                    batch_size=best.params['bs'], loss_type='huber', warmup_epochs=5,
                )
                tds = TabularDataset(kan_X_train, y_train_n[:, ti:ti+1])
                vds = TabularDataset(kan_X_val, y_val_n[:, ti:ti+1])
                _, history = trainer.fit(tds, vds, checkpoint_dir=f'{mdir}/{tname}_ckpt')

                test_ds = TabularDataset(kan_X_test, y_test[:, ti:ti+1])
                pred_scaled = trainer.predict(test_ds)
                np.save(f'{mdir}/{tname}_pred_scaled.npy', pred_scaled)
                dummy = np.zeros((len(pred_scaled), 2))
                dummy[:, ti] = pred_scaled.flatten()
                pred_inv = scaler_y.inverse_transform(dummy)
                kan_preds[:, ti] = pred_inv[:, ti]

                trainer.save(f'{mdir}/{tname}_model.pth', model_kwargs=kan_kwargs)
                with open(f'{mdir}/{tname}_best_params.json', 'w') as f:
                    json.dump(best.params, f, indent=2)
                NNTrainer.plot_loss_curves(history, f'KAN-{tname}',
                                           save_path=f'{mdir}/{tname}_loss_curves.png')

            metrics = evaluate(y_test, kan_preds)
            np.save(f'{mdir}/preds.npy', kan_preds)
            NNTrainer.plot_scatter(y_test, kan_preds, 'KAN', save_path=f'{mdir}/scatter.png')
            with open(f'{mdir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            log_metrics(model_name, metrics)

    else:
        raise ValueError(f"Unknown model type: {info['type']}")

    logger.info(f"\n[{model_name}] Done!")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Train a single model for Experiment 1")
    p.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()),
                   help='Model name to train')
    p.add_argument('--rep-dir', default='data/ieee39_v8_80_10_10')
    p.add_argument('--adj-path', default='data/ieee39_v8/adjacency/adjacency.npy')
    p.add_argument('--result-dir', default='results/ieee39/exp1')
    p.add_argument('--ms', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--n-trials', type=int, default=50)
    args = p.parse_args()

    run_single_model(args.model, args.rep_dir, args.adj_path,
                     args.result_dir, args.ms, args.device, args.n_trials)
