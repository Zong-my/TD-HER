#!/usr/bin/env python3
"""
Experiment P1: Feature Engineering Ablation.

Tests the value of 1387-dim statistical features vs raw temporal features
for LightGBM prediction.

Configurations:
  - raw_only:   7 static + 550 raw temporal = 557 dims
  - stats_only: 7 static + 1387 statistical = 1394 dims
  - full:       7 static + 550 raw + 1387 stats = 1944 dims (Exp6 "All" baseline)

If statistical features dominate: stats_only ≈ full >> raw_only
  → 1387-dim feature engineering is the primary information source
  → Complex temporal DL models are largely learning from statistics, not raw dynamics

Usage:
    python experiments/exp_p1_feature_ablation.py
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
import re
import json
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import mae_loss, rmse_loss, mape_loss, smape_loss
from models.lightgbm_model import LightGBMModel

SEED = 42
np.random.seed(SEED)

REP_DIR = str(data_path('ieee39_v8_80_10_10'))
EXP1_DIR = str(work_path('results/ieee39/exp1'))
P1_DIR = str(work_path('results/ieee39/exp_p1_feature_ablation'))
MS = 10


def evaluate(y_true, y_pred):
    metrics = {}
    for i, name in enumerate(['y1', 'y2']):
        yt, yp = y_true[:, i], y_pred[:, i]
        metrics[f'{name}_MAE'] = float(mae_loss(yt, yp))
        metrics[f'{name}_RMSE'] = float(rmse_loss(yt, yp))
        metrics[f'{name}_MAPE'] = float(mape_loss(yt, yp))
        metrics[f'{name}_SMAPE'] = float(smape_loss(yt, yp))
    return metrics


def categorize_features(feature_names):
    """Split features into static, raw temporal, and statistical."""
    static_ops = {'load_level', 'load_zip_z', 'load_zip_i', 'load_zip_p',
                  'reserve_ratio', 'h_inertia', 'load_delta'}

    static_idx = []
    raw_idx = []
    stats_idx = []

    for i, name in enumerate(feature_names):
        if name in static_ops:
            static_idx.append(i)
        elif re.match(r'^(FREQ|VOLT|ANGL|POWR|SPD)_\d+_\d+$', name):
            raw_idx.append(i)
        else:
            stats_idx.append(i)

    return static_idx, raw_idx, stats_idx


def run_lgb_on_features(X_train, y_train, X_val, y_val, X_test, y_test,
                         save_dir, config_name):
    """Train LightGBM on a feature subset using exp1 hyperparams."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if os.path.exists(f'{save_dir}/metrics.json'):
        with open(f'{save_dir}/metrics.json') as f:
            m = json.load(f)
        logger.info(f"  [{config_name}] SKIPPED: y1={m['y1_MAE']:.5f} y2={m['y2_MAE']:.5f}")
        return m

    # Normalize
    scaler = StandardScaler().fit(X_train)
    X_train_n = scaler.transform(X_train).astype(np.float32)
    X_val_n = scaler.transform(X_val).astype(np.float32)
    X_test_n = scaler.transform(X_test).astype(np.float32)

    logger.info(f"  [{config_name}] {X_train_n.shape[1]} features, training...")

    # Use exp1 LightGBM hyperparams
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bp = {}
    for tgt in ['y1', 'y2']:
        db = f'{EXP1_DIR}/LightGBM/optuna_{tgt}.db'
        studies = optuna.get_all_study_names(f'sqlite:///{db}')
        study = optuna.load_study(study_name=studies[0], storage=f'sqlite:///{db}')
        bp[tgt] = study.best_params

    model = LightGBMModel(n_trials=0, seed=SEED)
    model.best_params = {'y1': bp['y1'], 'y2': bp['y2']}
    model.fit(X_train_n, y_train, X_val_n, y_val,
              checkpoint_dir=save_dir, skip_optuna=True)

    pred = model.predict(X_test_n)
    metrics = evaluate(y_test, pred)

    np.save(f'{save_dir}/preds.npy', pred)
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  [{config_name}] y1_MAE={metrics['y1_MAE']:.5f} y2_MAE={metrics['y2_MAE']:.5f}")
    return metrics


def main():
    Path(P1_DIR).mkdir(parents=True, exist_ok=True)

    # Load data
    X_train = np.load(f'{REP_DIR}/repA/ms{MS}/X_train.npy')
    X_val = np.load(f'{REP_DIR}/repA/ms{MS}/X_val.npy')
    X_test = np.load(f'{REP_DIR}/repA/ms{MS}/X_test.npy')
    y_train = np.load(f'{REP_DIR}/repA/ms{MS}/y_train.npy')
    y_val = np.load(f'{REP_DIR}/repA/ms{MS}/y_val.npy')
    y_test = np.load(f'{REP_DIR}/repA/ms{MS}/y_test.npy')
    with open(f'{REP_DIR}/repA/ms{MS}/feature_names.json') as f:
        feature_names = json.load(f)

    # Categorize
    static_idx, raw_idx, stats_idx = categorize_features(feature_names)
    logger.info(f"Features: {len(static_idx)} static + {len(raw_idx)} raw temporal + "
                f"{len(stats_idx)} statistical = {len(feature_names)} total")

    configs = {
        'raw_only': static_idx + raw_idx,           # 7 + 550 = 557
        'stats_only': static_idx + stats_idx,        # 7 + 1387 = 1394
        'full': list(range(len(feature_names))),     # 1944
    }

    all_results = {}

    for config_name, idx in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"P1: {config_name} ({len(idx)} features)")
        logger.info(f"{'='*60}")

        save_dir = f'{P1_DIR}/{config_name}'
        metrics = run_lgb_on_features(
            X_train[:, idx], y_train,
            X_val[:, idx], y_val,
            X_test[:, idx], y_test,
            save_dir, config_name
        )
        all_results[config_name] = metrics

    # Save summary
    with open(f'{P1_DIR}/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison
    logger.info(f"\n{'='*80}")
    logger.info("P1: FEATURE ENGINEERING ABLATION (LightGBM)")
    logger.info(f"{'='*80}")
    logger.info(f"\n{'Config':15s} | {'#Feats':>7s} | {'y1_MAE':>10s} | {'y1_MAPE':>10s} | "
                f"{'y2_MAE':>10s} | {'y2_MAPE':>10s}")
    logger.info("-" * 70)

    for config_name in ['raw_only', 'stats_only', 'full']:
        m = all_results.get(config_name, {})
        n_feats = len(configs[config_name])
        logger.info(f"{config_name:15s} | {n_feats:>7d} | {m.get('y1_MAE', float('nan')):10.5f} | "
                   f"{m.get('y1_MAPE', float('nan')):10.2f} | "
                   f"{m.get('y2_MAE', float('nan')):10.5f} | "
                   f"{m.get('y2_MAPE', float('nan')):10.2f}")

    # Interpretation
    raw = all_results.get('raw_only', {})
    stats = all_results.get('stats_only', {})
    full = all_results.get('full', {})

    if raw and stats:
        y1_gain = (raw['y1_MAE'] - stats['y1_MAE']) / raw['y1_MAE'] * 100
        y2_gain = (raw['y2_MAE'] - stats['y2_MAE']) / raw['y2_MAE'] * 100
        logger.info(f"\nStats vs Raw improvement: y1 {y1_gain:+.1f}%, y2 {y2_gain:+.1f}%")

    if stats and full:
        y1_add = (stats['y1_MAE'] - full['y1_MAE']) / stats['y1_MAE'] * 100
        y2_add = (stats['y2_MAE'] - full['y2_MAE']) / stats['y2_MAE'] * 100
        logger.info(f"Adding raw to stats:       y1 {y1_add:+.1f}%, y2 {y2_add:+.1f}%")

    logger.info("\nP1 feature ablation complete!")


if __name__ == '__main__':
    main()
