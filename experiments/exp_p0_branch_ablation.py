#!/usr/bin/env python3
"""
Experiment P0: Dual-Branch Ablation Study.

Tests whether the temporal and tabular branches serve physically distinct
information needs for Δf vs t_nadir prediction.

Three configurations per model:
  - full:        standard dual-branch (exp1 baseline, already available)
  - no_tabular:  x_static zeroed out → temporal branch only
  - no_temporal: x_temporal zeroed out → tabular branch only

If the hypothesis holds:
  - Removing tabular branch should hurt t_nadir MORE than Δf
  - Removing temporal branch should hurt Δf MORE than t_nadir

Models tested: ConvLSTM, PatchTST, Mamba (top-3 DL from exp1)

Usage:
    python experiments/exp_p0_branch_ablation.py [--models ConvLSTM,PatchTST,Mamba]
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
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation import mae_loss, rmse_loss, mape_loss, smape_loss
from models.convlstm_model import ConvLSTMFreqModel, build_convlstm_from_trial
from models.patchtst_model import PatchTSTFreqModel, build_patchtst_from_trial
from models.mamba_model import MambaFreqModel, build_mamba_from_trial
from models.base_nn import NNTrainer
from data_proc.datasets import TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

REP_DIR = str(data_path('ieee39_v8_80_10_10'))
EXP1_DIR = str(work_path('results/ieee39/exp1'))
P0_DIR = str(work_path('results/ieee39/exp_p0_branch_ablation'))
MS = 10

BUILD_FNS = {
    'ConvLSTM': build_convlstm_from_trial,
    'PatchTST': build_patchtst_from_trial,
    'Mamba': build_mamba_from_trial,
}

# AMP settings matching exp1
AMP_SETTINGS = {
    'ConvLSTM': True,
    'PatchTST': False,  # PatchTST uses AMP=False in exp1
    'Mamba': True,
}


class FixedTrial:
    """Replay Optuna best params without running a new study."""
    def __init__(self, params):
        self._params = dict(params)
        self.params = dict(params)
    def suggest_int(self, name, low, high, **kw): return self._params[name]
    def suggest_float(self, name, low, high, **kw): return self._params[name]
    def suggest_categorical(self, name, choices, **kw): return self._params[name]


def evaluate(y_true, y_pred):
    metrics = {}
    for i, name in enumerate(['y1', 'y2']):
        yt, yp = y_true[:, i], y_pred[:, i]
        metrics[f'{name}_MAE'] = float(mae_loss(yt, yp))
        metrics[f'{name}_RMSE'] = float(rmse_loss(yt, yp))
        metrics[f'{name}_MAPE'] = float(mape_loss(yt, yp))
        metrics[f'{name}_SMAPE'] = float(smape_loss(yt, yp))
    return metrics


def load_data():
    """Load and normalize all data (same pipeline as exp1)."""
    # Temporal data
    Xt_train = np.load(f'{REP_DIR}/repB/ms{MS}/X_temporal_train.npy')
    Xt_val = np.load(f'{REP_DIR}/repB/ms{MS}/X_temporal_val.npy')
    Xt_test = np.load(f'{REP_DIR}/repB/ms{MS}/X_temporal_test.npy')

    # Static data
    Xs_train = np.load(f'{REP_DIR}/repB/ms{MS}/X_static_train.npy')
    Xs_val = np.load(f'{REP_DIR}/repB/ms{MS}/X_static_val.npy')
    Xs_test = np.load(f'{REP_DIR}/repB/ms{MS}/X_static_test.npy')

    # Targets
    y_train = np.load(f'{REP_DIR}/repA/ms{MS}/y_train.npy')
    y_val = np.load(f'{REP_DIR}/repA/ms{MS}/y_val.npy')
    y_test = np.load(f'{REP_DIR}/repA/ms{MS}/y_test.npy')

    # mRMR static feature selection (same as exp1)
    with open(f'{REP_DIR}/repB/ms{MS}/meta.json') as f:
        meta = json.load(f)
    static_names = meta['static_names']
    with open(f'{EXP1_DIR}/mrmr_static_features.json') as f:
        static_mrmr = json.load(f)
    static_idx = [static_names.index(f) for f in static_mrmr['union']]

    Xs_train = Xs_train[:, static_idx]
    Xs_val = Xs_val[:, static_idx]
    Xs_test = Xs_test[:, static_idx]

    # Normalize temporal
    B, T, N, C = Xt_train.shape
    sc_t = StandardScaler().fit(Xt_train.reshape(-1, N * C))
    Xt_train_n = sc_t.transform(Xt_train.reshape(-1, N*C)).reshape(B, T, N, C).astype(np.float32)
    Xt_val_n = sc_t.transform(Xt_val.reshape(-1, N*C)).reshape(Xt_val.shape).astype(np.float32)
    Xt_test_n = sc_t.transform(Xt_test.reshape(-1, N*C)).reshape(Xt_test.shape).astype(np.float32)

    # Normalize static
    sc_s = StandardScaler().fit(Xs_train)
    Xs_train_n = sc_s.transform(Xs_train).astype(np.float32)
    Xs_val_n = sc_s.transform(Xs_val).astype(np.float32)
    Xs_test_n = sc_s.transform(Xs_test).astype(np.float32)

    # Normalize targets
    scaler_y = StandardScaler().fit(y_train)
    y_train_n = scaler_y.transform(y_train).astype(np.float32)
    y_val_n = scaler_y.transform(y_val).astype(np.float32)

    logger.info(f"Data: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    logger.info(f"Temporal: {Xt_train.shape}, Static: {Xs_train.shape}")

    return {
        'Xt_train_n': Xt_train_n, 'Xt_val_n': Xt_val_n, 'Xt_test_n': Xt_test_n,
        'Xs_train_n': Xs_train_n, 'Xs_val_n': Xs_val_n, 'Xs_test_n': Xs_test_n,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'y_train_n': y_train_n, 'y_val_n': y_val_n,
        'scaler_y': scaler_y,
        'n_timesteps': T, 'n_generators': N, 'n_features': C,
        'n_static': Xs_train.shape[1],
    }


def build_datasets(data, ablation_mode):
    """Build train/val/test datasets with ablation applied.

    ablation_mode:
      'full'        — use both branches normally
      'no_tabular'  — zero out x_static
      'no_temporal' — zero out x_temporal
    """
    Xt_train = data['Xt_train_n']
    Xt_val = data['Xt_val_n']
    Xt_test = data['Xt_test_n']
    Xs_train = data['Xs_train_n']
    Xs_val = data['Xs_val_n']
    Xs_test = data['Xs_test_n']

    if ablation_mode == 'no_tabular':
        Xs_train = np.zeros_like(Xs_train)
        Xs_val = np.zeros_like(Xs_val)
        Xs_test = np.zeros_like(Xs_test)
    elif ablation_mode == 'no_temporal':
        Xt_train = np.zeros_like(Xt_train)
        Xt_val = np.zeros_like(Xt_val)
        Xt_test = np.zeros_like(Xt_test)

    train_ds = TensorDataset(Xt_train, Xs_train, data['y_train_n'])
    val_ds = TensorDataset(Xt_val, Xs_val, data['y_val_n'])
    test_ds = TensorDataset(Xt_test, Xs_test, data['y_test'][:, :2])

    return train_ds, val_ds, test_ds


def run_ablation(model_name, ablation_mode, data, device='cuda'):
    """Train a model from scratch under the specified ablation mode."""
    save_dir = f'{P0_DIR}/{model_name}/{ablation_mode}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Skip if done
    if os.path.exists(f'{save_dir}/metrics.json'):
        with open(f'{save_dir}/metrics.json') as f:
            m = json.load(f)
        logger.info(f"  [{model_name}/{ablation_mode}] SKIPPED: "
                     f"y1={m['y1_MAE']:.5f} y2={m['y2_MAE']:.5f}")
        return m

    # Load exp1 best hyperparams
    with open(f'{EXP1_DIR}/{model_name}/best_params.json') as f:
        bp = json.load(f)

    # Build model from scratch with same architecture
    extra_kwargs = {}
    if model_name == 'ConvLSTM':
        extra_kwargs = dict(n_features=data['n_features'],
                           n_generators=data['n_generators'],
                           n_static=data['n_static'])
    elif model_name == 'PatchTST':
        extra_kwargs = dict(n_timesteps=data['n_timesteps'],
                           n_generators=data['n_generators'],
                           n_features=data['n_features'],
                           n_static=data['n_static'])
    elif model_name == 'Mamba':
        extra_kwargs = dict(n_timesteps=data['n_timesteps'],
                           n_generators=data['n_generators'],
                           n_features=data['n_features'],
                           n_static=data['n_static'])

    trial = FixedTrial(bp)
    model, model_kwargs = BUILD_FNS[model_name](trial, **extra_kwargs)

    # Build datasets with ablation
    train_ds, val_ds, test_ds = build_datasets(data, ablation_mode)

    trainer = NNTrainer(
        model, device=device,
        lr=bp['lr'],
        weight_decay=bp['weight_decay'],
        max_epochs=200, patience=20,
        batch_size=bp['bs'],
        y1_weight=bp['y1_weight'], y2_weight=bp['y2_weight'],
        use_amp=AMP_SETTINGS[model_name],
        warmup_epochs=5,
    )

    logger.info(f"  [{model_name}/{ablation_mode}] Training from scratch...")
    _, history = trainer.fit(train_ds, val_ds, checkpoint_dir=save_dir)

    pred_scaled = trainer.predict(test_ds)
    pred = data['scaler_y'].inverse_transform(pred_scaled)
    metrics = evaluate(data['y_test'], pred)

    # Save
    np.save(f'{save_dir}/preds.npy', pred)
    with open(f'{save_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(f'{save_dir}/history.json', 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f)

    NNTrainer.plot_loss_curves(history, f'{model_name} ({ablation_mode})',
                               f'{save_dir}/loss_curves.png')

    logger.info(f"  [{model_name}/{ablation_mode}] y1_MAE={metrics['y1_MAE']:.5f} "
                f"y2_MAE={metrics['y2_MAE']:.5f}")

    del model, trainer
    torch.cuda.empty_cache()
    return metrics


def load_exp1_baseline(model_name):
    """Load exp1 results as the 'full' dual-branch baseline."""
    path = f'{EXP1_DIR}/{model_name}/metrics.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def print_summary(all_results, models):
    """Print comprehensive ablation comparison table."""
    logger.info(f"\n{'='*100}")
    logger.info("P0: DUAL-BRANCH ABLATION RESULTS")
    logger.info(f"{'='*100}")

    # Table header
    logger.info(f"\n{'Model':15s} | {'Mode':14s} | {'y1_MAE':>10s} | {'y1_MAPE':>10s} | "
                f"{'y2_MAE':>10s} | {'y2_MAPE':>10s} | {'y1 vs full':>10s} | {'y2 vs full':>10s}")
    logger.info("-" * 100)

    for model_name in models:
        full = all_results.get(model_name, {}).get('full', {})
        full_y1 = full.get('y1_MAE', float('nan'))
        full_y2 = full.get('y2_MAE', float('nan'))

        for mode in ['full', 'no_tabular', 'no_temporal']:
            m = all_results.get(model_name, {}).get(mode, {})
            y1 = m.get('y1_MAE', float('nan'))
            y2 = m.get('y2_MAE', float('nan'))
            y1_mape = m.get('y1_MAPE', float('nan'))
            y2_mape = m.get('y2_MAPE', float('nan'))

            if mode == 'full':
                y1_delta = '—'
                y2_delta = '—'
            else:
                y1_pct = (y1 - full_y1) / full_y1 * 100 if full_y1 > 0 else float('nan')
                y2_pct = (y2 - full_y2) / full_y2 * 100 if full_y2 > 0 else float('nan')
                y1_delta = f"+{y1_pct:.1f}%" if y1_pct == y1_pct else "N/A"
                y2_delta = f"+{y2_pct:.1f}%" if y2_pct == y2_pct else "N/A"

            logger.info(f"{model_name:15s} | {mode:14s} | {y1:10.5f} | {y1_mape:10.2f} | "
                       f"{y2:10.5f} | {y2_mape:10.2f} | {y1_delta:>10s} | {y2_delta:>10s}")
        logger.info("-" * 100)

    # Interpretation
    logger.info(f"\n{'='*100}")
    logger.info("INTERPRETATION")
    logger.info(f"{'='*100}")
    logger.info("Hypothesis: temporal branch primarily serves Δf, tabular branch primarily serves t_nadir")
    logger.info("Evidence FOR: no_tabular hurts y2 MORE than y1; no_temporal hurts y1 MORE than y2")
    logger.info("Evidence AGAINST: both modes hurt both targets equally (no asymmetry)")

    # Compute asymmetry ratio for each model
    for model_name in models:
        full = all_results.get(model_name, {}).get('full', {})
        no_tab = all_results.get(model_name, {}).get('no_tabular', {})
        no_temp = all_results.get(model_name, {}).get('no_temporal', {})

        if not (full and no_tab and no_temp):
            continue

        full_y1, full_y2 = full['y1_MAE'], full['y2_MAE']
        # Relative degradation when removing tabular branch
        tab_deg_y1 = (no_tab['y1_MAE'] - full_y1) / full_y1
        tab_deg_y2 = (no_tab['y2_MAE'] - full_y2) / full_y2
        # Relative degradation when removing temporal branch
        temp_deg_y1 = (no_temp['y1_MAE'] - full_y1) / full_y1
        temp_deg_y2 = (no_temp['y2_MAE'] - full_y2) / full_y2

        logger.info(f"\n{model_name}:")
        logger.info(f"  Remove tabular:  y1 degrades {tab_deg_y1*100:+.1f}%, y2 degrades {tab_deg_y2*100:+.1f}%")
        logger.info(f"  Remove temporal: y1 degrades {temp_deg_y1*100:+.1f}%, y2 degrades {temp_deg_y2*100:+.1f}%")

        # Asymmetry check
        if tab_deg_y2 > tab_deg_y1 * 1.5 and temp_deg_y1 > temp_deg_y2 * 1.5:
            logger.info(f"  → STRONG asymmetry: tabular→t_nadir, temporal→Δf (hypothesis SUPPORTED)")
        elif tab_deg_y2 > tab_deg_y1 or temp_deg_y1 > temp_deg_y2:
            logger.info(f"  → WEAK asymmetry: directionally consistent but not strong")
        else:
            logger.info(f"  → NO asymmetry: hypothesis NOT supported for this model")


def main():
    parser = argparse.ArgumentParser(description='P0: Dual-Branch Ablation')
    parser.add_argument('--models', type=str, default='ConvLSTM,PatchTST,Mamba',
                        help='Comma-separated model names')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',')]
    Path(P0_DIR).mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    data = load_data()

    all_results = {}

    for model_name in models:
        all_results[model_name] = {}

        # Load exp1 baseline as 'full' mode
        full_metrics = load_exp1_baseline(model_name)
        if full_metrics:
            all_results[model_name]['full'] = full_metrics
            logger.info(f"\n[{model_name}/full] exp1 baseline: "
                       f"y1={full_metrics['y1_MAE']:.5f} y2={full_metrics['y2_MAE']:.5f}")

        # Run ablation modes
        for mode in ['no_tabular', 'no_temporal']:
            logger.info(f"\n{'='*60}")
            logger.info(f"{model_name} / {mode}")
            logger.info(f"{'='*60}")
            try:
                metrics = run_ablation(model_name, mode, data, args.device)
                all_results[model_name][mode] = metrics
            except Exception as e:
                logger.error(f"  [{model_name}/{mode}] FAILED: {e}")
                import traceback; traceback.print_exc()

    # Save all results
    with open(f'{P0_DIR}/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print_summary(all_results, models)

    logger.info("\nP0 ablation complete!")


if __name__ == '__main__':
    main()
