#!/usr/bin/env python3
"""
Experiment 4: Three-level Generalization Verification.

Uses Experiment 1 trained models (no retraining) to predict on:
  Level 1: Same-distribution test set (from exp1)
  Level 2: Cross-condition test (new operating conditions, same topology)
  Level 3: Cross-condition + cross-topology test (new conditions + topology changes)

All 8 models evaluated. Scalers from exp1 training set applied (no refit).

Usage:
    python experiments/exp4_generalization.py
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import mae_loss, rmse_loss, mape_loss, smape_loss
from utils.mrmr_select import mrmr_select_union
from models.kan_model import KANRegressor, build_kan_from_trial
from models.convlstm_model import ConvLSTMFreqModel
from models.patchtst_model import PatchTSTFreqModel
from models.mamba_model import MambaFreqModel
from models.stgcn_model import STGCNBatchedModel, normalize_adjacency
from models.ft_transformer_model import FTTransformerModel
from models.tabr_model import TabRModel
from models.base_nn import NNTrainer
from data_proc.datasets import (
    TabularDataset, TensorDataset, GraphDataset, adjacency_to_edge_index
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

REP_DIR = 'data/ieee39_v8_80_10_10'
ADJ_PATH = 'data/ieee39_v8/adjacency/adjacency.npy'
EXP1_DIR = 'results/ieee39/exp1'
EXP4_DIR = 'results/ieee39/exp4'
MS = 10

LEVELS = {
    'L1_same_dist': 'test',
    'L2_cross_cond': 'cross_cond_test',
    'L3_cross_topo': 'cross_cond_topo_test',
}

MODELS = ['LightGBM', 'KAN', 'ConvLSTM', 'PatchTST', 'Mamba', 'ST-GCN',
          'FT-Transformer', 'TabR']


def evaluate(y_true, y_pred):
    metrics = {}
    for i, name in enumerate(['y1', 'y2']):
        yt, yp = y_true[:, i], y_pred[:, i]
        metrics[f'{name}_MAE'] = float(mae_loss(yt, yp))
        metrics[f'{name}_RMSE'] = float(rmse_loss(yt, yp))
        metrics[f'{name}_MAPE'] = float(mape_loss(yt, yp))
        metrics[f'{name}_SMAPE'] = float(smape_loss(yt, yp))
    return metrics


def load_train_scalers():
    """Fit scalers on training data (same as exp1). Must not refit on test data."""
    logger.info("Fitting scalers on training data...")

    X_train_A = np.load(f'{REP_DIR}/repA/ms{MS}/X_train.npy')
    y_train = np.load(f'{REP_DIR}/repA/ms{MS}/y_train.npy')
    with open(f'{REP_DIR}/repA/ms{MS}/feature_names.json') as f:
        feature_names = json.load(f)

    Xt_train = np.load(f'{REP_DIR}/repB/ms{MS}/X_temporal_train.npy')
    Xs_train = np.load(f'{REP_DIR}/repB/ms{MS}/X_static_train.npy')

    with open(f'{REP_DIR}/repB/ms{MS}/meta.json') as f:
        meta = json.load(f)
    static_names = meta['static_names']

    # Load cached mRMR from exp1
    with open(f'{EXP1_DIR}/mrmr_features.json') as f:
        mrmr = json.load(f)
    selected_features = mrmr['union']
    feat_idx = [feature_names.index(f) for f in selected_features]

    with open(f'{EXP1_DIR}/mrmr_static_features.json') as f:
        static_mrmr = json.load(f)
    static_selected = static_mrmr['union']
    static_idx = [static_names.index(f) for f in static_selected]

    # Fit scalers on training data only
    scaler_A = StandardScaler().fit(X_train_A[:, feat_idx])
    B, T, N, C = Xt_train.shape
    sc_t = StandardScaler().fit(Xt_train.reshape(-1, N * C))
    sc_s = StandardScaler().fit(Xs_train[:, static_idx])
    scaler_y = StandardScaler().fit(y_train)

    return {
        'scaler_A': scaler_A, 'sc_t': sc_t, 'sc_s': sc_s, 'scaler_y': scaler_y,
        'feat_idx': feat_idx, 'static_idx': static_idx,
        'feature_names': feature_names, 'mrmr': mrmr,
        'n_timesteps': T, 'n_generators': N, 'n_features': C,
        'n_static': len(static_idx),
    }


def load_level_data(split_name, scalers):
    """Load and normalize data for a generalization level."""
    X_A = np.load(f'{REP_DIR}/repA/ms{MS}/X_{split_name}.npy')
    y = np.load(f'{REP_DIR}/repA/ms{MS}/y_{split_name}.npy')
    Xt = np.load(f'{REP_DIR}/repB/ms{MS}/X_temporal_{split_name}.npy')
    Xs = np.load(f'{REP_DIR}/repB/ms{MS}/X_static_{split_name}.npy')
    Xn = np.load(f'{REP_DIR}/repC/ms{MS}/X_node_{split_name}.npy')

    # Apply train scalers (no refit!)
    X_n = scalers['scaler_A'].transform(X_A[:, scalers['feat_idx']]).astype(np.float32)

    B, T, N, C = Xt.shape
    Xt_n = scalers['sc_t'].transform(Xt.reshape(-1, N*C)).reshape(B, T, N, C).astype(np.float32)
    Xs_n = scalers['sc_s'].transform(Xs[:, scalers['static_idx']]).astype(np.float32)
    Xn_n = np.transpose(Xt_n, (0, 2, 1, 3))

    return {
        'X_A': X_A, 'X_n': X_n, 'y': y,
        'Xt_n': Xt_n, 'Xs_n': Xs_n, 'Xn_n': Xn_n,
    }


def predict_lightgbm(data, scalers):
    # Load per-target models directly (avoid Optuna DB dependency)
    import lightgbm as lgb
    mdir = f'{EXP1_DIR}/LightGBM'
    preds = np.zeros((len(data['y']), 2))
    for i, tgt in enumerate(['y1', 'y2']):
        with open(f'{mdir}/{tgt}_lgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        preds[:, i] = model.predict(data['X_n'])
    return preds


def predict_kan(data, scalers):
    """KAN: per-target prediction with top-100 mRMR features."""
    KAN_K = 100
    mdir = f'{EXP1_DIR}/KAN'
    X_A = data['X_A']
    y = data['y']
    scaler_y = scalers['scaler_y']
    feature_names = scalers['feature_names']
    mrmr = scalers['mrmr']

    X_train_A = np.load(f'{REP_DIR}/repA/ms{MS}/X_train.npy')
    preds = np.zeros((len(y), 2))

    for ti, tname in enumerate(['y1', 'y2']):
        kan_feat_names = mrmr[tname][:KAN_K]
        kan_feat_idx = [feature_names.index(f) for f in kan_feat_names
                        if f in feature_names]

        kan_scaler = StandardScaler().fit(X_train_A[:, kan_feat_idx])
        kan_X = kan_scaler.transform(X_A[:, kan_feat_idx]).astype(np.float32)
        kan_dim = kan_X.shape[1]

        with open(f'{mdir}/{tname}_best_params.json') as f:
            bp = json.load(f)

        class FixedTrial:
            def __init__(self, p):
                self._params = dict(p)
                self.params = dict(p)
            def suggest_int(self, name, low, high, **kw):
                return self._params[name]
            def suggest_float(self, name, low, high, **kw):
                return self._params[name]
            def suggest_categorical(self, name, choices, **kw):
                return self._params[name]

        model, _ = build_kan_from_trial(FixedTrial(bp), kan_dim)
        ckpt = torch.load(f'{mdir}/{tname}_model.pth', map_location='cuda',
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to('cuda').eval()

        ds = TabularDataset(kan_X, y[:, ti:ti+1])
        trainer = NNTrainer(model, device='cuda', batch_size=bp['bs'])
        pred_scaled = trainer.predict(ds)

        dummy = np.zeros((len(pred_scaled), 2))
        dummy[:, ti] = pred_scaled.flatten()
        pred_inv = scaler_y.inverse_transform(dummy)
        preds[:, ti] = pred_inv[:, ti]

        del model, trainer
        torch.cuda.empty_cache()

    return preds


def predict_dl_model(model_name, data, scalers, device='cuda'):  # noqa: PLR0912
    """Generic DL model prediction using saved checkpoint."""
    mdir = f'{EXP1_DIR}/{model_name}'
    scaler_y = scalers['scaler_y']

    ckpt = torch.load(f'{mdir}/model.pth', map_location=device, weights_only=False)
    model_kwargs = ckpt.get('model_kwargs', {})
    model_state = ckpt['model_state_dict']

    model_classes = {
        'ConvLSTM': ConvLSTMFreqModel,
        'PatchTST': PatchTSTFreqModel,
        'Mamba': MambaFreqModel,
        'ST-GCN': STGCNBatchedModel,
        'FT-Transformer': FTTransformerModel,
        'TabR': TabRModel,
    }

    cls = model_classes[model_name]
    model = cls(**model_kwargs)
    if model_name == 'ST-GCN':
        adj = np.load(ADJ_PATH)
        adj_norm_tensor = torch.FloatTensor(normalize_adjacency(adj))
        model.set_adj(adj_norm_tensor)
    model.load_state_dict(model_state)
    model.to(device).eval()

    with open(f'{mdir}/best_params.json') as f:
        bp = json.load(f)

    # Build dataset based on model type
    y_dummy = data['y'][:, :2]
    if model_name in ('ConvLSTM', 'PatchTST', 'Mamba'):
        ds = TensorDataset(data['Xt_n'], data['Xs_n'], y_dummy)
    elif model_name == 'ST-GCN':
        adj = np.load(ADJ_PATH)
        edge_index, edge_weight = adjacency_to_edge_index(adj)
        ds = GraphDataset(data['Xn_n'], data['Xs_n'], y_dummy,
                          edge_index, edge_weight)
    elif model_name in ('FT-Transformer', 'TabR'):
        ds = TabularDataset(data['X_n'], y_dummy)

    # TabR requires training cache (k-NN retrieval) — rebuild from exp1 training data
    if model_name == 'TabR':
        X_train_A = np.load(f'{REP_DIR}/repA/ms{MS}/X_train.npy')
        y_train = np.load(f'{REP_DIR}/repA/ms{MS}/y_train.npy')
        X_train_n = torch.FloatTensor(
            scalers['scaler_A'].transform(X_train_A[:, scalers['feat_idx']]).astype(np.float32)
        ).to(device)
        y_train_n = torch.FloatTensor(
            scalers['scaler_y'].transform(y_train).astype(np.float32)
        ).to(device)
        model.build_training_cache(X_train_n, y_train_n)
        model.eval()

    trainer = NNTrainer(model, device=device, batch_size=bp['bs'])
    pred_scaled = trainer.predict(ds)
    pred = scaler_y.inverse_transform(pred_scaled)

    del model, trainer
    torch.cuda.empty_cache()
    return pred


def main():
    Path(EXP4_DIR).mkdir(parents=True, exist_ok=True)

    scalers = load_train_scalers()
    all_results = {}

    for level_name, split_name in LEVELS.items():
        logger.info(f"\n{'='*60}\n{level_name} ({split_name})")
        data = load_level_data(split_name, scalers)
        logger.info(f"  Samples: {len(data['y'])}")

        level_results = {}

        for model_name in MODELS:
            ldir = f'{EXP4_DIR}/{level_name}/{model_name}'
            Path(ldir).mkdir(parents=True, exist_ok=True)

            # Skip if already done
            if os.path.exists(f'{ldir}/metrics.json'):
                with open(f'{ldir}/metrics.json') as f:
                    level_results[model_name] = json.load(f)
                logger.info(f"  {model_name}: loaded (cached)")
                continue

            try:
                if model_name == 'LightGBM':
                    pred = predict_lightgbm(data, scalers)
                elif model_name == 'KAN':
                    pred = predict_kan(data, scalers)
                else:
                    pred = predict_dl_model(model_name, data, scalers)

                metrics = evaluate(data['y'], pred)
                level_results[model_name] = metrics

                np.save(f'{ldir}/preds.npy', pred)
                with open(f'{ldir}/metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)

                logger.info(f"  {model_name}: y1_MAE={metrics['y1_MAE']:.5f}, "
                           f"y2_MAE={metrics['y2_MAE']:.5f}")
            except Exception as e:
                logger.error(f"  {model_name} FAILED: {e}")
                import traceback
                traceback.print_exc()

        all_results[level_name] = level_results

    # Save summary
    with open(f'{EXP4_DIR}/metrics_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print results table
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT 4: THREE-LEVEL GENERALIZATION RESULTS")
    logger.info(f"{'='*80}")

    for metric in ['y1_MAE', 'y1_MAPE', 'y2_MAE', 'y2_MAPE']:
        logger.info(f"\n{metric}:")
        header = f"{'Model':18s}" + "".join(f"  {ln:>14s}" for ln in LEVELS.keys())
        logger.info(header)
        for model in MODELS:
            row = f"{model:18s}"
            for ln in LEVELS.keys():
                val = all_results.get(ln, {}).get(model, {}).get(metric, float('nan'))
                row += f"  {val:14.5f}" if not np.isnan(val) else f"  {'N/A':>14s}"
            logger.info(row)

    logger.info("\nExperiment 4 complete!")


if __name__ == '__main__':
    main()
