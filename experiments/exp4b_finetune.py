#!/usr/bin/env python3
"""
Experiment 4b: Fine-tuning for Generalization.

Fine-tune Experiment 1 models on cross_cond/cross_cond_topo finetune data,
then evaluate on the corresponding test sets.

Compares zero-shot (Exp4) vs fine-tuned performance to answer:
"Can topology-change degradation be overcome with small labeled data?"

Strategy:
  - DL models: load exp1 checkpoint, fine-tune with low lr (1/10), 20 epochs
  - LightGBM: retrain from scratch on train+finetune combined data
  - KAN: fine-tune with low lr, 20 epochs per target

Usage:
    python experiments/exp4b_finetune.py
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
import pickle
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import mae_loss, rmse_loss, mape_loss, smape_loss
from models.lightgbm_model import LightGBMModel
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

REP_DIR = str(data_path('ieee39_v8_80_10_10'))
ADJ_PATH = str(data_path('ieee39_v8/adjacency/adjacency.npy'))
EXP1_DIR = str(work_path('results/ieee39/exp1'))
EXP4B_DIR = str(work_path('results/ieee39/exp4b'))
MS = 10

# Fine-tune scenarios: finetune_split -> test_split
SCENARIOS = {
    'L2_cross_cond': {
        'finetune': 'cross_cond_finetune',
        'test': 'cross_cond_test',
    },
    'L3_cross_topo': {
        'finetune': 'cross_cond_topo_finetune',
        'test': 'cross_cond_topo_test',
    },
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
    """Load scalers fitted on original training data."""
    X_train_A = np.load(f'{REP_DIR}/repA/ms{MS}/X_train.npy')
    y_train = np.load(f'{REP_DIR}/repA/ms{MS}/y_train.npy')
    with open(f'{REP_DIR}/repA/ms{MS}/feature_names.json') as f:
        feature_names = json.load(f)

    Xt_train = np.load(f'{REP_DIR}/repB/ms{MS}/X_temporal_train.npy')
    Xs_train = np.load(f'{REP_DIR}/repB/ms{MS}/X_static_train.npy')

    with open(f'{REP_DIR}/repB/ms{MS}/meta.json') as f:
        meta = json.load(f)
    static_names = meta['static_names']

    with open(f'{EXP1_DIR}/mrmr_features.json') as f:
        mrmr = json.load(f)
    feat_idx = [feature_names.index(f) for f in mrmr['union']]

    with open(f'{EXP1_DIR}/mrmr_static_features.json') as f:
        static_mrmr = json.load(f)
    static_idx = [static_names.index(f) for f in static_mrmr['union']]

    scaler_A = StandardScaler().fit(X_train_A[:, feat_idx])
    B, T, N, C = Xt_train.shape
    sc_t = StandardScaler().fit(Xt_train.reshape(-1, N * C))
    sc_s = StandardScaler().fit(Xs_train[:, static_idx])
    scaler_y = StandardScaler().fit(y_train)

    return {
        'scaler_A': scaler_A, 'sc_t': sc_t, 'sc_s': sc_s, 'scaler_y': scaler_y,
        'feat_idx': feat_idx, 'static_idx': static_idx,
        'feature_names': feature_names, 'mrmr': mrmr,
        'X_train_A': X_train_A, 'y_train': y_train,
        'Xt_train': Xt_train, 'Xs_train': Xs_train,
        'n_timesteps': T, 'n_generators': N, 'n_features': C,
        'n_static': len(static_idx),
    }


def load_split_data(split_name, scalers):
    """Load and normalize a data split using train scalers."""
    X_A = np.load(f'{REP_DIR}/repA/ms{MS}/X_{split_name}.npy')
    y = np.load(f'{REP_DIR}/repA/ms{MS}/y_{split_name}.npy')
    Xt = np.load(f'{REP_DIR}/repB/ms{MS}/X_temporal_{split_name}.npy')
    Xs = np.load(f'{REP_DIR}/repB/ms{MS}/X_static_{split_name}.npy')

    X_n = scalers['scaler_A'].transform(X_A[:, scalers['feat_idx']]).astype(np.float32)
    B, T, N, C = Xt.shape
    Xt_n = scalers['sc_t'].transform(Xt.reshape(-1, N*C)).reshape(B, T, N, C).astype(np.float32)
    Xs_n = scalers['sc_s'].transform(Xs[:, scalers['static_idx']]).astype(np.float32)
    Xn_n = np.transpose(Xt_n, (0, 2, 1, 3))
    y_n = scalers['scaler_y'].transform(y).astype(np.float32)

    return {
        'X_A': X_A, 'X_n': X_n, 'y': y, 'y_n': y_n,
        'Xt_n': Xt_n, 'Xs_n': Xs_n, 'Xn_n': Xn_n,
    }


def finetune_lightgbm(ft_data, test_data, scalers, save_dir):
    """Retrain LightGBM on original_train + finetune data."""
    if os.path.exists(f'{save_dir}/metrics.json'):
        with open(f'{save_dir}/metrics.json') as f:
            return json.load(f)

    # Combine original train + finetune for LightGBM
    X_combined = np.concatenate([
        scalers['scaler_A'].transform(scalers['X_train_A'][:, scalers['feat_idx']]).astype(np.float32),
        ft_data['X_n']
    ])
    y_combined = np.concatenate([scalers['y_train'], ft_data['y']])

    # Use finetune data as validation
    model = LightGBMModel(n_trials=0, seed=SEED)

    # Load exp1 best params
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bp = {}
    for tgt in ['y1', 'y2']:
        db = f'{EXP1_DIR}/LightGBM/optuna_{tgt}.db'
        studies = optuna.get_all_study_names(f'sqlite:///{db}')
        study = optuna.load_study(study_name=studies[0], storage=f'sqlite:///{db}')
        bp[tgt] = study.best_params

    model.best_params = {'y1': bp['y1'], 'y2': bp['y2']}
    model.fit(X_combined, y_combined, ft_data['X_n'], ft_data['y'],
              checkpoint_dir=save_dir, skip_optuna=True)

    pred = model.predict(test_data['X_n'])
    metrics = evaluate(test_data['y'], pred)

    np.save(f'{save_dir}/preds.npy', pred)
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def finetune_kan(ft_data, test_data, scalers, save_dir, device='cuda'):
    """Fine-tune KAN per-target models."""
    if os.path.exists(f'{save_dir}/metrics.json'):
        with open(f'{save_dir}/metrics.json') as f:
            return json.load(f)

    KAN_K = 100
    mdir = f'{EXP1_DIR}/KAN'
    mrmr = scalers['mrmr']
    feature_names = scalers['feature_names']
    X_train_A = scalers['X_train_A']
    scaler_y = scalers['scaler_y']
    preds = np.zeros((len(test_data['y']), 2))

    for ti, tname in enumerate(['y1', 'y2']):
        kan_feat_names = mrmr[tname][:KAN_K]
        kan_feat_idx = [feature_names.index(f) for f in kan_feat_names
                        if f in feature_names]
        kan_scaler = StandardScaler().fit(X_train_A[:, kan_feat_idx])

        ft_X = kan_scaler.transform(ft_data['X_A'][:, kan_feat_idx]).astype(np.float32)
        test_X = kan_scaler.transform(test_data['X_A'][:, kan_feat_idx]).astype(np.float32)
        ft_y_n = scaler_y.transform(ft_data['y']).astype(np.float32)
        kan_dim = ft_X.shape[1]

        with open(f'{mdir}/{tname}_best_params.json') as f:
            bp = json.load(f)

        class FixedTrial:
            def __init__(self, p):
                self._params = dict(p)
                self.params = dict(p)
            def suggest_int(self, name, low, high, **kw): return self._params[name]
            def suggest_float(self, name, low, high, **kw): return self._params[name]
            def suggest_categorical(self, name, choices, **kw): return self._params[name]

        model, _ = build_kan_from_trial(FixedTrial(bp), kan_dim)
        ckpt = torch.load(f'{mdir}/{tname}_model.pth', map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

        # Fine-tune: low lr, short epochs
        ft_ds = TabularDataset(ft_X, ft_y_n[:, ti:ti+1])
        # Use 80% finetune as train, 20% as val
        n = len(ft_X)
        idx = np.random.RandomState(SEED).permutation(n)
        n_train = int(n * 0.8)
        train_ds = TabularDataset(ft_X[idx[:n_train]], ft_y_n[idx[:n_train], ti:ti+1])
        val_ds = TabularDataset(ft_X[idx[n_train:]], ft_y_n[idx[n_train:], ti:ti+1])

        trainer = NNTrainer(
            model, device=device,
            lr=bp['lr'] / 10,
            weight_decay=bp.get('wd', 1e-4),
            max_epochs=20, patience=10,
            batch_size=min(bp['bs'], n_train),
            loss_type='huber', warmup_epochs=2,
        )
        trainer.fit(train_ds, val_ds, verbose=False)

        test_ds = TabularDataset(test_X, test_data['y'][:, ti:ti+1])
        pred_scaled = trainer.predict(test_ds)
        dummy = np.zeros((len(pred_scaled), 2))
        dummy[:, ti] = pred_scaled.flatten()
        pred_inv = scaler_y.inverse_transform(dummy)
        preds[:, ti] = pred_inv[:, ti]

        del model, trainer
        torch.cuda.empty_cache()

    metrics = evaluate(test_data['y'], preds)
    np.save(f'{save_dir}/preds.npy', preds)
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def finetune_dl_model(model_name, ft_data, test_data, scalers, save_dir,
                      device='cuda'):
    """Fine-tune a DL model from exp1 checkpoint."""
    if os.path.exists(f'{save_dir}/metrics.json'):
        with open(f'{save_dir}/metrics.json') as f:
            return json.load(f)

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
        model.set_adj(torch.FloatTensor(normalize_adjacency(adj)))
    model.load_state_dict(model_state)

    with open(f'{mdir}/best_params.json') as f:
        bp = json.load(f)

    # Build finetune datasets (80/20 split of finetune data)
    n = len(ft_data['y'])
    idx = np.random.RandomState(SEED).permutation(n)
    n_train = int(n * 0.8)
    t_idx, v_idx = idx[:n_train], idx[n_train:]

    if model_name in ('ConvLSTM', 'PatchTST', 'Mamba'):
        train_ds = TensorDataset(ft_data['Xt_n'][t_idx], ft_data['Xs_n'][t_idx], ft_data['y_n'][t_idx])
        val_ds = TensorDataset(ft_data['Xt_n'][v_idx], ft_data['Xs_n'][v_idx], ft_data['y_n'][v_idx])
        test_ds = TensorDataset(test_data['Xt_n'], test_data['Xs_n'], test_data['y'][:, :2])
    elif model_name == 'ST-GCN':
        adj = np.load(ADJ_PATH)
        ei, ew = adjacency_to_edge_index(adj)
        train_ds = GraphDataset(ft_data['Xn_n'][t_idx], ft_data['Xs_n'][t_idx], ft_data['y_n'][t_idx], ei, ew)
        val_ds = GraphDataset(ft_data['Xn_n'][v_idx], ft_data['Xs_n'][v_idx], ft_data['y_n'][v_idx], ei, ew)
        test_ds = GraphDataset(test_data['Xn_n'], test_data['Xs_n'], test_data['y'][:, :2], ei, ew)
    elif model_name in ('FT-Transformer', 'TabR'):
        train_ds = TabularDataset(ft_data['X_n'][t_idx], ft_data['y_n'][t_idx])
        val_ds = TabularDataset(ft_data['X_n'][v_idx], ft_data['y_n'][v_idx])
        test_ds = TabularDataset(test_data['X_n'], test_data['y'][:, :2])

    def _refresh_tabr_cache(active_model=None):
        if model_name != 'TabR':
            return
        target_model = active_model or model
        target_model.build_training_cache(train_ds.X, train_ds.y)
        target_model.eval()

    _refresh_tabr_cache()
    use_amp = model_name in ('PatchTST', 'FT-Transformer')

    trainer = NNTrainer(
        model, device=device,
        lr=bp['lr'] / 10,  # 1/10 of original lr
        weight_decay=bp['weight_decay'],
        batch_size=min(bp['bs'], n_train),
        y1_weight=bp['y1_weight'], y2_weight=bp['y2_weight'],
        max_epochs=20, patience=10,
        use_amp=use_amp, warmup_epochs=2,
        cache_refresh_fn=_refresh_tabr_cache if model_name == 'TabR' else None,
    )

    _, history = trainer.fit(train_ds, val_ds, verbose=False)
    _refresh_tabr_cache()
    pred_scaled = trainer.predict(test_ds)
    pred = scaler_y.inverse_transform(pred_scaled)

    metrics = evaluate(test_data['y'], pred)
    np.save(f'{save_dir}/preds.npy', pred)
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    del model, trainer
    torch.cuda.empty_cache()
    return metrics


def main():
    Path(EXP4B_DIR).mkdir(parents=True, exist_ok=True)
    scalers = load_train_scalers()
    all_results = {}

    for level_name, scenario in SCENARIOS.items():
        logger.info(f"\n{'='*60}\n{level_name}: finetune on {scenario['finetune']}, test on {scenario['test']}")

        ft_data = load_split_data(scenario['finetune'], scalers)
        test_data = load_split_data(scenario['test'], scalers)
        logger.info(f"  Finetune: {len(ft_data['y'])} samples, Test: {len(test_data['y'])} samples")

        level_results = {}

        for model_name in MODELS:
            save_dir = f'{EXP4B_DIR}/{level_name}/{model_name}'
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            try:
                if model_name == 'LightGBM':
                    metrics = finetune_lightgbm(ft_data, test_data, scalers, save_dir)
                elif model_name == 'KAN':
                    metrics = finetune_kan(ft_data, test_data, scalers, save_dir)
                else:
                    metrics = finetune_dl_model(model_name, ft_data, test_data,
                                                scalers, save_dir)

                level_results[model_name] = metrics
                logger.info(f"  {model_name}: y1_MAE={metrics['y1_MAE']:.5f}, "
                           f"y2_MAE={metrics['y2_MAE']:.5f}")
            except Exception as e:
                logger.error(f"  {model_name} FAILED: {e}")
                import traceback
                traceback.print_exc()

        all_results[level_name] = level_results

    # Load exp4 zero-shot results for comparison
    exp4_results = {}
    if os.path.exists(str(work_path('results/ieee39/exp4/metrics_summary.json'))):
        with open(str(work_path('results/ieee39/exp4/metrics_summary.json'))) as f:
            exp4_results = json.load(f)

    # Save summary
    with open(f'{EXP4B_DIR}/metrics_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    logger.info(f"\n{'='*80}")
    logger.info("ZERO-SHOT vs FINE-TUNED COMPARISON")
    logger.info(f"{'='*80}")

    for level_name in SCENARIOS:
        logger.info(f"\n--- {level_name} ---")
        logger.info(f"{'Model':18s} {'y1_MAE zs':>10s} {'y1_MAE ft':>10s} {'improv%':>8s} "
                    f"{'y2_MAE zs':>10s} {'y2_MAE ft':>10s} {'improv%':>8s}")
        for m in MODELS:
            zs1 = exp4_results.get(level_name, {}).get(m, {}).get('y1_MAE', float('nan'))
            ft1 = all_results.get(level_name, {}).get(m, {}).get('y1_MAE', float('nan'))
            zs2 = exp4_results.get(level_name, {}).get(m, {}).get('y2_MAE', float('nan'))
            ft2 = all_results.get(level_name, {}).get(m, {}).get('y2_MAE', float('nan'))
            imp1 = f"{(zs1-ft1)/zs1*100:.1f}%" if zs1 == zs1 and ft1 == ft1 and zs1 > 0 else "N/A"
            imp2 = f"{(zs2-ft2)/zs2*100:.1f}%" if zs2 == zs2 and ft2 == ft2 and zs2 > 0 else "N/A"
            zs1_s = f"{zs1:.5f}" if zs1 == zs1 else "N/A"
            ft1_s = f"{ft1:.5f}" if ft1 == ft1 else "N/A"
            zs2_s = f"{zs2:.5f}" if zs2 == zs2 else "N/A"
            ft2_s = f"{ft2:.5f}" if ft2 == ft2 else "N/A"
            logger.info(f"{m:18s} {zs1_s:>10s} {ft1_s:>10s} {imp1:>8s} "
                       f"{zs2_s:>10s} {ft2_s:>10s} {imp2:>8s}")

    logger.info("\nExperiment 4b complete!")


if __name__ == '__main__':
    main()
