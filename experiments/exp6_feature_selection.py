#!/usr/bin/env python3
"""
Experiment 6: Feature Selection Ablation (IEEE 39-bus, LightGBM only).

Compares four feature selection methods under identical LightGBM configs:
  1. mRMR (adaptive)       — cached union from exp1
  2. Mutual Information     — sklearn MI regression, top-K union
  3. Random Forest importance — RF impurity-based, top-K union
  4. No selection (all)     — full ~1944 features as baseline

K is set to match the mRMR union count from exp1.
LightGBM hyperparameters are loaded from exp1 Optuna DBs (skip_optuna=True).

Usage:
    python experiments/exp6_feature_selection.py
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

# ── Paths ──
DATA_DIR = Path(str(data_path('ieee39_v8_80_10_10/repA/ms10')))
EXP1_DIR = work_path('results') / "ieee39" / "exp1"
EXP1_LGB_DIR = EXP1_DIR / "LightGBM"
MRMR_PATH = EXP1_DIR / "mrmr_features.json"
RESULT_DIR = work_path('results') / "ieee39" / "exp6"


# ─────────────────────────── helpers ────────────────────────────

def load_data():
    """Load repA ms10 data and feature names."""
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    with open(DATA_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    logger.info(f"Data loaded: X_train {X_train.shape}, features={len(feature_names)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def load_exp1_best_params():
    """Load LightGBM best hyperparameters from exp1 Optuna SQLite DBs."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_params = {}
    for target in ["y1", "y2"]:
        db_path = EXP1_LGB_DIR / f"optuna_{target}.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Optuna DB not found: {db_path}")
        storage = f"sqlite:///{db_path}"
        study = optuna.load_study(
            study_name=f"LightGBM_{target}", storage=storage
        )
        best_params[target] = study.best_params
        logger.info(f"  exp1 best {target} (RMSE={study.best_value:.6f}): {study.best_params}")
    return best_params


def evaluate(y_true, y_pred):
    """Compute MAE, RMSE, MAPE, SMAPE for both targets."""
    metrics = {}
    for i, name in enumerate(["y1", "y2"]):
        metrics[f"{name}_MAE"] = float(mae_loss(y_true[:, i], y_pred[:, i]))
        metrics[f"{name}_RMSE"] = float(rmse_loss(y_true[:, i], y_pred[:, i]))
        metrics[f"{name}_MAPE"] = float(mape_loss(y_true[:, i], y_pred[:, i]))
        metrics[f"{name}_SMAPE"] = float(smape_loss(y_true[:, i], y_pred[:, i]))
    return metrics


def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test,
                       best_params, method_name, save_dir):
    """StandardScaler + LightGBM (skip_optuna) + evaluate + save."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train).astype(np.float32)
    Xva = scaler.transform(X_val).astype(np.float32)
    Xte = scaler.transform(X_test).astype(np.float32)

    model = LightGBMModel(seed=SEED)
    model.best_params = dict(best_params)  # copy to avoid mutation
    model.fit(Xtr, y_train, Xva, y_val, checkpoint_dir=str(save_dir),
              skip_optuna=True)

    y_pred = model.predict(Xte)
    metrics = evaluate(y_test, y_pred)
    metrics["method"] = method_name
    metrics["n_features"] = int(X_train.shape[1])

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  {method_name}: n_feat={metrics['n_features']}, "
                f"y1_MAE={metrics['y1_MAE']:.4f}, y2_MAE={metrics['y2_MAE']:.4f}")
    return metrics


# ─────────────────── feature selection methods ──────────────────

def select_mrmr(feature_names):
    """Load cached mRMR union features from exp1."""
    with open(MRMR_PATH) as f:
        mrmr_data = json.load(f)
    union_feats = mrmr_data["union"]
    idx = [feature_names.index(f) for f in union_feats if f in feature_names]
    logger.info(f"  mRMR union: {len(idx)} features")
    return np.array(idx)


def select_mutual_info(X_train, y_train, feature_names, K):
    """Mutual information regression: top-K per target, union."""
    from sklearn.feature_selection import mutual_info_regression

    logger.info(f"  Computing MI for y1 ({X_train.shape[1]} features)...")
    mi_y1 = mutual_info_regression(X_train, y_train[:, 0],
                                   random_state=SEED, n_jobs=-1)
    logger.info(f"  Computing MI for y2...")
    mi_y2 = mutual_info_regression(X_train, y_train[:, 1],
                                   random_state=SEED, n_jobs=-1)

    top_y1 = set(np.argsort(mi_y1)[-K:])
    top_y2 = set(np.argsort(mi_y2)[-K:])
    union_idx = sorted(top_y1 | top_y2)
    logger.info(f"  MI union: {len(union_idx)} features (K={K} per target)")
    return np.array(union_idx)


def select_rf_importance(X_train, y_train, feature_names, K):
    """Random Forest feature importance: top-K per target, union."""
    from sklearn.ensemble import RandomForestRegressor

    logger.info(f"  Training RF for y1...")
    rf1 = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf1.fit(X_train, y_train[:, 0])
    imp_y1 = rf1.feature_importances_

    logger.info(f"  Training RF for y2...")
    rf2 = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf2.fit(X_train, y_train[:, 1])
    imp_y2 = rf2.feature_importances_

    top_y1 = set(np.argsort(imp_y1)[-K:])
    top_y2 = set(np.argsort(imp_y2)[-K:])
    union_idx = sorted(top_y1 | top_y2)
    logger.info(f"  RF union: {len(union_idx)} features (K={K} per target)")
    return np.array(union_idx)


# ─────────────────────── visualization ──────────────────────────

def plot_comparison(all_results, save_path):
    """Bar chart comparing feature selection methods (IEEE TII grade)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Use Times New Roman if available
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    methods = [r["method"] for r in all_results]
    n = len(methods)

    # Metrics to plot: y1_MAE, y1_RMSE, y2_MAE, y2_RMSE
    metric_keys = ["y1_MAE", "y1_RMSE", "y2_MAE", "y2_RMSE"]
    metric_labels = [
        r"$f_{\mathrm{pu,\Delta max}}$ MAE",
        r"$f_{\mathrm{pu,\Delta max}}$ RMSE",
        r"$t_\Delta$ MAE",
        r"$t_\Delta$ RMSE",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=False)
    colors = ["#2166ac", "#b2182b", "#1b7837", "#762a83"]
    bar_width = 0.55

    for ax, key, label, color in zip(axes, metric_keys, metric_labels, colors):
        vals = [r[key] for r in all_results]
        x = np.arange(n)
        bars = ax.bar(x, vals, width=bar_width, color=color, alpha=0.85,
                      edgecolor="black", linewidth=0.5)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0, max(vals) * 1.25)

    fig.suptitle("Feature Selection Ablation (LightGBM, IEEE 39-bus, ms=10)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {save_path}")


# ──────────────────────── main pipeline ─────────────────────────

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data()

    # Load exp1 LightGBM best params
    logger.info("Loading exp1 LightGBM best hyperparameters...")
    best_params = load_exp1_best_params()

    # Determine K from mRMR union count
    with open(MRMR_PATH) as f:
        mrmr_data = json.load(f)
    mrmr_union = mrmr_data["union"]
    # K per target: use y1 count (before union) as reference
    K_per_target = len(mrmr_data["y1"])
    logger.info(f"mRMR: union={len(mrmr_union)}, K_per_target={K_per_target}")

    all_results = []

    # ── Method 1: mRMR (adaptive) ──
    logger.info("=" * 60)
    logger.info("Method 1: mRMR (adaptive)")
    idx_mrmr = select_mrmr(feature_names)
    m1 = train_and_evaluate(
        X_train[:, idx_mrmr], X_val[:, idx_mrmr], X_test[:, idx_mrmr],
        y_train, y_val, y_test, best_params,
        "mRMR", RESULT_DIR / "mRMR"
    )
    all_results.append(m1)

    # ── Method 2: Mutual Information ──
    logger.info("=" * 60)
    logger.info("Method 2: Mutual Information")
    idx_mi = select_mutual_info(X_train, y_train, feature_names, K_per_target)
    m2 = train_and_evaluate(
        X_train[:, idx_mi], X_val[:, idx_mi], X_test[:, idx_mi],
        y_train, y_val, y_test, best_params,
        "MI", RESULT_DIR / "MI"
    )
    all_results.append(m2)

    # ── Method 3: Random Forest importance ──
    logger.info("=" * 60)
    logger.info("Method 3: Random Forest Importance")
    idx_rf = select_rf_importance(X_train, y_train, feature_names, K_per_target)
    m3 = train_and_evaluate(
        X_train[:, idx_rf], X_val[:, idx_rf], X_test[:, idx_rf],
        y_train, y_val, y_test, best_params,
        "RF", RESULT_DIR / "RF"
    )
    all_results.append(m3)

    # ── Method 4: All features (no selection) ──
    logger.info("=" * 60)
    logger.info("Method 4: All Features (no selection)")
    m4 = train_and_evaluate(
        X_train, X_val, X_test,
        y_train, y_val, y_test, best_params,
        "All", RESULT_DIR / "All"
    )
    all_results.append(m4)

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("Experiment 6 — Comparison Table:")
    header = f"{'Method':<10} {'#Feat':>6}  {'y1_MAE':>8} {'y1_RMSE':>8} {'y1_MAPE':>8} {'y1_SMAPE':>8}  {'y2_MAE':>8} {'y2_RMSE':>8} {'y2_MAPE':>8} {'y2_SMAPE':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in all_results:
        logger.info(
            f"{r['method']:<10} {r['n_features']:>6}  "
            f"{r['y1_MAE']:>8.4f} {r['y1_RMSE']:>8.4f} {r['y1_MAPE']:>8.2f} {r['y1_SMAPE']:>8.2f}  "
            f"{r['y2_MAE']:>8.4f} {r['y2_RMSE']:>8.4f} {r['y2_MAPE']:>8.2f} {r['y2_SMAPE']:>8.2f}"
        )

    # Save summary JSON
    summary_path = RESULT_DIR / "comparison.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    # Bar chart
    plot_comparison(all_results, RESULT_DIR / "exp6_comparison.png")

    logger.info("Experiment 6 complete.")


if __name__ == "__main__":
    main()
