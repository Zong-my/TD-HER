#!/usr/bin/env python3
"""
Experiment 1: Main method comparison on IEEE 39 (ms=10, Gen-only).

8 models × 3 disturbance types × 2 targets × 4 metrics + inference speed.
"""

import os
import sys
import json
import time
import gc
import numpy as np
import pandas as pd
import torch
import optuna
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import mae_loss, rmse_loss, mape_loss, smape_loss
from utils.mrmr_select import mrmr_select_union
from utils.timer import measure_lightgbm_time
from models.lightgbm_model import LightGBMModel
from models.kan_model import KANRegressor, build_kan_from_trial
from models.convlstm_model import ConvLSTMFreqModel, build_convlstm_from_trial
from models.patchtst_model import PatchTSTFreqModel, build_patchtst_from_trial
from models.mamba_model import MambaFreqModel, build_mamba_from_trial
from models.stgcn_model import STGCNFreqModel, build_stgcn_from_trial
from models.ft_transformer_model import FTTransformerModel, build_ft_transformer_from_trial
from models.tabr_model import TabRModel, build_tabr_from_trial
from models.base_nn import NNTrainer
from data_proc.datasets import (
    TabularDataset, TensorDataset, GraphDataset, adjacency_to_edge_index
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def evaluate(y_true, y_pred, prefix=''):
    """Compute all 4 metrics for both targets."""
    metrics = {}
    for i, name in enumerate(['y1', 'y2']):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        metrics[f'{prefix}{name}_MAE'] = mae_loss(yt, yp)
        metrics[f'{prefix}{name}_RMSE'] = rmse_loss(yt, yp)
        metrics[f'{prefix}{name}_MAPE'] = mape_loss(yt, yp)
        metrics[f'{prefix}{name}_SMAPE'] = smape_loss(yt, yp)
    return metrics


def log_metrics(model_name: str, metrics: dict):
    """Log all metrics for a model in a readable format."""
    logger.info(f"{model_name} results:")
    logger.info(f"  y1(fpu_deltamax): MAE={metrics['y1_MAE']:.6f}  RMSE={metrics['y1_RMSE']:.6f}  "
                f"MAPE={metrics['y1_MAPE']:.2f}%  SMAPE={metrics['y1_SMAPE']:.2f}%")
    logger.info(f"  y2(t_delta):      MAE={metrics['y2_MAE']:.6f}  RMSE={metrics['y2_RMSE']:.6f}  "
                f"MAPE={metrics['y2_MAPE']:.2f}%  SMAPE={metrics['y2_SMAPE']:.2f}%")


def get_model_dir(result_dir, name):
    """Create and return a per-model subdirectory under result_dir."""
    d = os.path.join(result_dir, name)
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


EXP1_MODELS = [
    'LightGBM',
    'KAN',
    'ConvLSTM',
    'PatchTST',
    'Mamba',
    'ST-GCN',
    'FT-Transformer',
    'TabR',
]


def _normalize_model_filter(models_to_run=None):
    if models_to_run is None:
        return set(EXP1_MODELS)
    requested = set(models_to_run)
    unknown = requested - set(EXP1_MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}. Available: {EXP1_MODELS}")
    return requested


def run_experiment1(rep_dir: str, adj_path: str, result_dir: str,
                    ms: int = 10, device: str = 'cuda', models_to_run=None):
    """Run full Experiment 1."""
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    models_to_run = _normalize_model_filter(models_to_run)

    def should_run(name):
        return name in models_to_run

    # ── Load data ──
    logger.info("Loading representations...")

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

    adj = np.load(adj_path)
    edge_index, edge_weight = adjacency_to_edge_index(adj)
    # Normalized adjacency for batched ST-GCN
    from models.stgcn_model import normalize_adjacency
    adj_norm = normalize_adjacency(adj)

    n_timesteps = Xt_train.shape[1]
    n_generators = Xt_train.shape[2]
    n_features = Xt_train.shape[3]
    n_static = Xs_train.shape[1]  # 7 base + all stat features

    logger.info(f"RepA: {X_train_A.shape}, RepB: {Xt_train.shape}, RepC: {Xn_train.shape}")
    logger.info(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
    logger.info(f"n_static: {n_static} (7 base + {n_static - 7} stat features)")

    # ── mRMR feature selection for RepA (reuse if already computed) ──
    mrmr_path = f'{result_dir}/mrmr_features.json'
    if os.path.exists(mrmr_path):
        logger.info("Loading cached mRMR features...")
        with open(mrmr_path) as f:
            mrmr_data = json.load(f)
        selected_features = mrmr_data['union']
        y1_feats = mrmr_data['y1']
        y2_feats = mrmr_data['y2']
    else:
        logger.info("Running adaptive mRMR feature selection...")
        selected_features, y1_feats, y2_feats = mrmr_select_union(
            X_train_A, y_train, feature_names, adaptive=True
        )
    feat_idx = [feature_names.index(f) for f in selected_features]
    X_train_sel = X_train_A[:, feat_idx]
    X_val_sel = X_val_A[:, feat_idx]
    X_test_sel = X_test_A[:, feat_idx]

    # Save feature selection results
    with open(f'{result_dir}/mrmr_features.json', 'w') as f:
        json.dump({'union': selected_features, 'y1': y1_feats, 'y2': y2_feats}, f)

    # ── Normalize ──
    scaler_A = StandardScaler().fit(X_train_sel)
    X_train_n = scaler_A.transform(X_train_sel).astype(np.float32)
    X_val_n = scaler_A.transform(X_val_sel).astype(np.float32)
    X_test_n = scaler_A.transform(X_test_sel).astype(np.float32)

    # RepB: normalize temporal per-channel, static separately
    scaler_temporal = StandardScaler()
    B, T, N, C = Xt_train.shape
    scaler_temporal.fit(Xt_train.reshape(-1, N * C))
    Xt_train_n = scaler_temporal.transform(Xt_train.reshape(-1, N*C)).reshape(B, T, N, C).astype(np.float32)
    Xt_val_n = scaler_temporal.transform(Xt_val.reshape(-1, N*C)).reshape(Xt_val.shape).astype(np.float32)
    Xt_test_n = scaler_temporal.transform(Xt_test.reshape(-1, N*C)).reshape(Xt_test.shape).astype(np.float32)

    scaler_static = StandardScaler().fit(Xs_train)
    Xs_train_n = scaler_static.transform(Xs_train).astype(np.float32)
    Xs_val_n = scaler_static.transform(Xs_val).astype(np.float32)
    Xs_test_n = scaler_static.transform(Xs_test).astype(np.float32)

    # RepC: same normalization as RepB
    Xn_train_n = np.transpose(Xt_train_n, (0, 2, 1, 3))
    Xn_val_n = np.transpose(Xt_val_n, (0, 2, 1, 3))
    Xn_test_n = np.transpose(Xt_test_n, (0, 2, 1, 3))

    # Target scaler
    scaler_y = StandardScaler().fit(y_train)
    y_train_n = scaler_y.transform(y_train).astype(np.float32)
    y_val_n = scaler_y.transform(y_val).astype(np.float32)

    all_results = {}

    # ════════════════════════════════════════════════════════════════
    # 1. LightGBM
    # ════════════════════════════════════════════════════════════════
    if should_run('LightGBM'):
        lgb_dir = get_model_dir(result_dir, 'LightGBM')
        if os.path.exists(f'{lgb_dir}/preds.npy'):
            logger.info("\n[1/8] LightGBM — SKIPPED (already done)")
            y_pred = np.load(f'{lgb_dir}/preds.npy')
            metrics = evaluate(y_test, y_pred)
            all_results['LightGBM'] = metrics
            log_metrics('LightGBM', metrics)
        else:
            logger.info("\n" + "="*60 + "\n[1/8] LightGBM")
            lgb_model = LightGBMModel(n_trials=100, timeout=3600, seed=SEED)
            t0 = time.time()
            lgb_model.fit(X_train_n, y_train, X_val_n, y_val, checkpoint_dir=lgb_dir)
            train_time = time.time() - t0

            y_pred = lgb_model.predict(X_test_n)
            metrics = evaluate(y_test, y_pred, prefix='')
            timing = lgb_model.measure_inference_time(X_test_n)
            metrics['train_time_s'] = train_time
            metrics.update({f'timing_{k}': v for k, v in timing.items()})
            all_results['LightGBM'] = metrics

            np.save(f'{lgb_dir}/preds.npy', y_pred)
            import pickle
            with open(f'{lgb_dir}/model.pkl', 'wb') as f:
                pickle.dump(lgb_model, f)
            with open(f'{lgb_dir}/metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            log_metrics('LightGBM', metrics)

    # ════════════════════════════════════════════════════════════════
    # 2. KAN (two separate models, per-target feature selection)
    # ════════════════════════════════════════════════════════════════
    KAN_K = 100  # KAN works best with low-dim input
    kan_dir = get_model_dir(result_dir, 'KAN') if should_run('KAN') else None
    if not should_run('KAN'):
        logger.info("\n[2/8] KAN — NOT REQUESTED")
    elif os.path.exists(f'{kan_dir}/preds.npy'):
        logger.info("\n[2/8] KAN — SKIPPED (already done)")
        kan_preds = np.load(f'{kan_dir}/preds.npy')
        metrics = evaluate(y_test, kan_preds)
        all_results['KAN'] = metrics
        log_metrics('KAN', metrics)
    else:
        logger.info("\n" + "="*60 + "\n[2/8] KAN (per-target top-{} features)".format(KAN_K))
        kan_preds = np.zeros((len(y_test), 2))
        kan_histories = {}

        # Per-target feature subsets from mRMR ranking
        per_target_feats = [y1_feats[:KAN_K], y2_feats[:KAN_K]]

        for ti, tname in enumerate(['y1', 'y2']):
            # Skip if this target already has saved predictions
            pred_ckpt = f'{kan_dir}/{tname}_pred_scaled.npy'
            if os.path.exists(pred_ckpt):
                logger.info(f"  KAN {tname}: loading saved predictions (resumed)")
                pred_scaled = np.load(pred_ckpt)
                dummy = np.zeros((len(pred_scaled), 2))
                dummy[:, ti] = pred_scaled.flatten()
                pred_inv = scaler_y.inverse_transform(dummy)
                kan_preds[:, ti] = pred_inv[:, ti]
                continue

            # Select per-target features
            kan_feat_names = per_target_feats[ti]
            kan_feat_idx = [feature_names.index(f) for f in kan_feat_names if f in feature_names]

            # Refit scaler on KAN's own feature subset
            kan_scaler = StandardScaler().fit(X_train_A[:, kan_feat_idx])
            kan_X_train_n = kan_scaler.transform(X_train_A[:, kan_feat_idx]).astype(np.float32)
            kan_X_val_n = kan_scaler.transform(X_val_A[:, kan_feat_idx]).astype(np.float32)
            kan_X_test_n = kan_scaler.transform(X_test_A[:, kan_feat_idx]).astype(np.float32)
            kan_input_dim = kan_X_train_n.shape[1]

            logger.info(f"  KAN {tname}: {kan_input_dim} features (top-{KAN_K} mRMR)")

            def kan_objective(trial):
                model, _ = build_kan_from_trial(trial, kan_input_dim)
                trainer = NNTrainer(
                    model, device=device,
                    lr=trial.suggest_float('lr', 5e-5, 5e-3, log=True),
                    weight_decay=trial.suggest_float('wd', 1e-6, 1e-2, log=True),
                    max_epochs=100, patience=15,
                    batch_size=trial.suggest_categorical('bs', [256, 512, 1024]),
                    loss_type='huber',
                    warmup_epochs=3,
                )
                train_ds = TabularDataset(kan_X_train_n, y_train_n[:, ti:ti+1])
                val_ds = TabularDataset(kan_X_val_n, y_val_n[:, ti:ti+1])
                val_loss, _ = trainer.fit(train_ds, val_ds, verbose=False)
                return val_loss

            study = optuna.create_study(
                study_name=f'KAN_{tname}', direction='minimize',
                storage=f'sqlite:///{kan_dir}/optuna_{tname}.db',
                sampler=optuna.samplers.TPESampler(seed=SEED),
                load_if_exists=True,
            )
            n_done = len([t for t in study.trials
                          if t.state == optuna.trial.TrialState.COMPLETE])
            n_remaining = max(0, 50 - n_done)
            if n_remaining > 0:
                logger.info(f"  KAN {tname} Optuna: {n_done} done, {n_remaining} remaining")
                study.optimize(kan_objective, n_trials=n_remaining, timeout=7200,
                               catch=(Exception,))
            else:
                logger.info(f"  KAN {tname} Optuna: all 50 trials done (resumed)")

            best = study.best_trial
            model, kan_kwargs = build_kan_from_trial(best, kan_input_dim)
            trainer = NNTrainer(
                model, device=device,
                lr=best.params['lr'], weight_decay=best.params['wd'],
                max_epochs=200, patience=25,
                batch_size=best.params['bs'], loss_type='huber',
                warmup_epochs=5,
            )
            train_ds = TabularDataset(kan_X_train_n, y_train_n[:, ti:ti+1])
            val_ds = TabularDataset(kan_X_val_n, y_val_n[:, ti:ti+1])
            _, history = trainer.fit(train_ds, val_ds,
                                     checkpoint_dir=f'{kan_dir}/{tname}_ckpt')
            kan_histories[tname] = history

            test_ds = TabularDataset(kan_X_test_n, y_test[:, ti:ti+1])
            pred_scaled = trainer.predict(test_ds)
            # Save per-target checkpoint for resume
            np.save(f'{kan_dir}/{tname}_pred_scaled.npy', pred_scaled)
            dummy = np.zeros((len(pred_scaled), 2))
            dummy[:, ti] = pred_scaled.flatten()
            pred_inv = scaler_y.inverse_transform(dummy)
            kan_preds[:, ti] = pred_inv[:, ti]

            trainer.save(f'{kan_dir}/{tname}_model.pth', model_kwargs=kan_kwargs)
            with open(f'{kan_dir}/{tname}_best_params.json', 'w') as f:
                json.dump(best.params, f, indent=2)

            NNTrainer.plot_loss_curves(
                history, f'KAN-{tname}',
                save_path=f'{kan_dir}/{tname}_loss_curves.png')

        metrics = evaluate(y_test, kan_preds)
        all_results['KAN'] = metrics
        np.save(f'{kan_dir}/preds.npy', kan_preds)

        NNTrainer.plot_scatter(
            y_test, kan_preds, 'KAN',
            save_path=f'{kan_dir}/scatter.png')

        with open(f'{kan_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        log_metrics('KAN', metrics)

    # ════════════════════════════════════════════════════════════════
    # Helper for multi-task DL models (ConvLSTM, PatchTST, Mamba, ST-GCN)
    # ════════════════════════════════════════════════════════════════

    def train_multitask_model(name, build_fn, train_ds, val_ds, test_ds,
                              n_trials=50, timeout=7200, use_amp=True,
                              **extra_build_kwargs):
        mdir = get_model_dir(result_dir, name)

        # Skip if already done
        if os.path.exists(f'{mdir}/preds.npy'):
            logger.info(f"\n[{name}] -- SKIPPED (already done)")
            pred = np.load(f'{mdir}/preds.npy')
            metrics = evaluate(y_test, pred)
            log_metrics(name, metrics)
            return metrics

        logger.info(f"\n{'='*60}\n[{name}]")

        # Model-specific search budgets. PatchTST uses channel-independent
        # processing, so its effective batch is batch_size * n_vars. The
        # generic [256, 512, 1024, 2048] search can OOM on IEEE300.
        if name == 'PatchTST':
            n_vars = (
                extra_build_kwargs.get('n_generators', 10)
                * extra_build_kwargs.get('n_features', 5)
            )
            max_bs = max(32, (65535 // max(1, n_vars) // 32) * 32)
            bs_choices = sorted(set(bs for bs in [32, 64, 128, 256, 512] if bs <= max_bs))
            if not bs_choices:
                bs_choices = [32]
            if n_vars >= 512:
                search_epochs, search_patience = 20, 6
            else:
                search_epochs, search_patience = 30, 8
        elif name == 'FT-Transformer':
            n_ft_features = extra_build_kwargs.get('n_features', 0)
            if n_ft_features >= 1024:
                bs_choices = [64, 128, 256]
                search_epochs, search_patience = 12, 4
            else:
                bs_choices = [512, 1024, 2048]
                search_epochs, search_patience = 20, 6
        elif name == 'TabR':
            n_tabr_features = extra_build_kwargs.get('n_features', 0)
            if n_tabr_features >= 1024:
                bs_choices = [128, 256, 512]
                search_epochs, search_patience = 20, 6
            else:
                bs_choices = [256, 512, 1024]
                search_epochs, search_patience = 40, 8
        else:
            bs_choices = [256, 512, 1024, 2048]
            search_epochs, search_patience = 50, 10
        logger.info(f"  Search batch sizes: {bs_choices}; epochs={search_epochs}, patience={search_patience}")

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

        def objective(trial):
            model = None
            trainer = None
            try:
                model, _ = build_fn(trial, **extra_build_kwargs)
                lr = trial.suggest_float('lr', 5e-5, 5e-3, log=True)
                wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
                bs = trial.suggest_categorical('bs', bs_choices)
                y1_w = trial.suggest_float('y1_weight', 0.5, 2.0)
                y2_w = trial.suggest_float('y2_weight', 0.5, 2.0)

                trainer = NNTrainer(
                    model, device=device,
                    lr=lr, weight_decay=wd,
                    max_epochs=search_epochs, patience=search_patience,
                    batch_size=bs,
                    y1_weight=y1_w, y2_weight=y2_w,
                    use_amp=use_amp,
                    warmup_epochs=3,
                    cache_refresh_fn=_make_cache_refresh_fn(),
                )
                val_loss, _ = trainer.fit(train_ds, val_ds, verbose=False)
                return val_loss
            finally:
                del trainer
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        study = optuna.create_study(
            study_name=f'{name}_optuna', direction='minimize',
            storage=f'sqlite:///{mdir}/optuna.db',
            sampler=optuna.samplers.TPESampler(seed=SEED),
            load_if_exists=True,
        )
        n_done = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
        n_remaining = max(0, n_trials - n_done)
        if n_remaining > 0:
            logger.info(f"  Optuna: {n_done} done, {n_remaining} remaining")
            study.optimize(objective, n_trials=n_remaining, timeout=timeout,
                           catch=(Exception,))
        else:
            logger.info(f"  Optuna: all {n_trials} trials done (resumed)")

        best = study.best_trial
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, model_kwargs = build_fn(best, **extra_build_kwargs)
        if name == 'FT-Transformer' and extra_build_kwargs.get('n_features', 0) >= 1024:
            final_epochs, final_patience = 80, 10
        elif name == 'TabR' and extra_build_kwargs.get('n_features', 0) >= 1024:
            final_epochs, final_patience = 120, 12
        else:
            final_epochs, final_patience = 200, 20

        trainer = NNTrainer(
            model, device=device,
            lr=best.params['lr'],
            weight_decay=best.params['weight_decay'],
            max_epochs=final_epochs, patience=final_patience,
            batch_size=best.params['bs'],
            y1_weight=best.params['y1_weight'],
            y2_weight=best.params['y2_weight'],
            use_amp=use_amp,
            warmup_epochs=5,
            cache_refresh_fn=_make_cache_refresh_fn(),
        )
        t0 = time.time()
        _, history = trainer.fit(train_ds, val_ds, checkpoint_dir=mdir)
        train_time = time.time() - t0

        _refresh_tabr_cache_for_final_predict(trainer)
        pred_scaled = trainer.predict(test_ds)  # (N, 2) scaled
        pred = scaler_y.inverse_transform(pred_scaled)

        timing = trainer.measure_inference_time(test_ds)
        metrics = evaluate(y_test, pred)
        metrics['train_time_s'] = train_time
        metrics.update({f'timing_{k}': v for k, v in timing.items()})

        # Save model with constructor kwargs for exp4 reuse
        trainer.save(f'{mdir}/model.pth', model_kwargs=model_kwargs)
        np.save(f'{mdir}/preds.npy', pred)

        # Save best hyperparams
        with open(f'{mdir}/best_params.json', 'w') as f:
            json.dump(best.params, f, indent=2)

        # ── Training visualization ──
        NNTrainer.plot_loss_curves(
            history, name,
            save_path=f'{mdir}/loss_curves.png')
        NNTrainer.plot_scatter(
            y_test, pred, name,
            save_path=f'{mdir}/scatter.png')

        # Save loss history as JSON for later analysis
        with open(f'{mdir}/history.json', 'w') as f:
            json.dump({k: [float(v) for v in vs]
                       for k, vs in history.items()}, f)

        # Save metrics
        with open(f'{mdir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        log_metrics(name, metrics)
        return metrics

    # ── Build datasets ──
    train_ds_B = TensorDataset(Xt_train_n, Xs_train_n, y_train_n)
    val_ds_B = TensorDataset(Xt_val_n, Xs_val_n, y_val_n)
    test_ds_B = TensorDataset(Xt_test_n, Xs_test_n, y_test[:, :2])  # unscaled y for eval

    train_ds_C = GraphDataset(Xn_train_n, Xs_train_n, y_train_n, edge_index, edge_weight)
    val_ds_C = GraphDataset(Xn_val_n, Xs_val_n, y_val_n, edge_index, edge_weight)
    test_ds_C = GraphDataset(Xn_test_n, Xs_test_n, y_test[:, :2], edge_index, edge_weight)

    # ════════════════════════════════════════════════════════════════
    # 3. ConvLSTM
    # ════════════════════════════════════════════════════════════════
    if should_run('ConvLSTM'):
        all_results['ConvLSTM'] = train_multitask_model(
            'ConvLSTM', build_convlstm_from_trial, train_ds_B, val_ds_B, test_ds_B,
            n_features=n_features, n_generators=n_generators, n_static=n_static,
        )

    # ════════════════════════════════════════════════════════════════
    # 4. PatchTST
    # ════════════════════════════════════════════════════════════════
    if should_run('PatchTST'):
        all_results['PatchTST'] = train_multitask_model(
            'PatchTST', build_patchtst_from_trial, train_ds_B, val_ds_B, test_ds_B,
            use_amp=False,  # AMP causes CUDA errors with some Transformer configs
            n_timesteps=n_timesteps, n_generators=n_generators,
            n_features=n_features, n_static=n_static,
        )

    # ════════════════════════════════════════════════════════════════
    # 5. Mamba
    # ════════════════════════════════════════════════════════════════
    if should_run('Mamba'):
        all_results['Mamba'] = train_multitask_model(
            'Mamba', build_mamba_from_trial, train_ds_B, val_ds_B, test_ds_B,
            n_timesteps=n_timesteps, n_generators=n_generators,
            n_features=n_features, n_static=n_static,
        )

    # ════════════════════════════════════════════════════════════════
    # 6. ST-GCN
    # ════════════════════════════════════════════════════════════════
    if should_run('ST-GCN'):
        all_results['ST-GCN'] = train_multitask_model(
            'ST-GCN', build_stgcn_from_trial, train_ds_C, val_ds_C, test_ds_C,
            n_features=n_features, n_nodes=n_generators, n_static=n_static, adj_norm=adj_norm,
        )

    # ════════════════════════════════════════════════════════════════
    # 7. FT-Transformer (RepA, multi-task)
    # ════════════════════════════════════════════════════════════════
    train_ds_A_mt = TabularDataset(X_train_n, y_train_n)
    val_ds_A_mt = TabularDataset(X_val_n, y_val_n)
    test_ds_A_mt = TabularDataset(X_test_n, y_test[:, :2])

    if should_run('FT-Transformer'):
        all_results['FT-Transformer'] = train_multitask_model(
            'FT-Transformer', build_ft_transformer_from_trial,
            train_ds_A_mt, val_ds_A_mt, test_ds_A_mt,
            n_trials=8, timeout=3600,
            use_amp=False,  # Pre-norm Transformer can be sensitive to AMP
            n_features=X_train_n.shape[1],
        )

    # ════════════════════════════════════════════════════════════════
    # 8. TabR (RepA, multi-task, retrieval-augmented)
    # ════════════════════════════════════════════════════════════════
    def _build_tabr_with_cache(trial, **kwargs):
        """Build TabR and attach training cache."""
        model, mk = build_tabr_from_trial(trial, **kwargs)
        # Build cache with normalized training data
        model.build_training_cache(
            torch.FloatTensor(X_train_n), torch.FloatTensor(y_train_n))
        return model, mk

    if should_run('TabR'):
        all_results['TabR'] = train_multitask_model(
            'TabR', _build_tabr_with_cache,
            train_ds_A_mt, val_ds_A_mt, test_ds_A_mt,
            n_trials=12, timeout=5400,
            n_features=X_train_n.shape[1],
        )

    # ── Save summary ──
    summary = pd.DataFrame(all_results).T
    summary.to_csv(f'{result_dir}/exp1_results.csv')
    logger.info(f"\n{'='*60}\nExperiment 1 complete! Results saved to {result_dir}")
    logger.info(f"\n{summary.to_string()}")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep-dir', default='data/ieee39_v8_80_10_10')
    parser.add_argument('--adj-path', default='data/ieee39_v8/adjacency/adjacency.npy')
    parser.add_argument('--result-dir', default='results/ieee39/exp1')
    parser.add_argument('--ms', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--models', nargs='+', choices=EXP1_MODELS, default=None)
    args = parser.parse_args()

    run_experiment1(
        args.rep_dir, args.adj_path, args.result_dir, args.ms, args.device,
        models_to_run=args.models,
    )
