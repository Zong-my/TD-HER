#!/usr/bin/env python3
"""LightGBM regressor with Optuna hyperparameter optimization."""

import os
import time
import numpy as np
import lightgbm as lgb
import optuna
from loguru import logger


class LightGBMModel:
    """Two independent LightGBM regressors for y1 and y2."""

    def __init__(self, n_trials: int = 100, timeout: int = 3600,
                 n_jobs: int = -1, seed: int = 42):
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.seed = seed
        self.models = {}     # {'y1': model, 'y2': model}
        self.studies = {}    # {'y1': study, 'y2': study}
        self.best_params = {}

    def _make_params(self, trial):
        """Build LightGBM param dict from Optuna trial."""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 8, 256),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': 5,
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'verbosity': -1,
            'n_jobs': self.n_jobs,
            'random_state': self.seed,
        }

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective: fast search with reduced estimators."""
        params = self._make_params(trial)
        # Use fewer estimators + aggressive early stopping during search
        params['n_estimators'] = 1000
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )
        y_pred = model.predict(X_val)
        return np.sqrt(np.mean((y_val - y_pred) ** 2))

    def fit(self, X_train, y_train, X_val, y_val, checkpoint_dir=None,
            skip_optuna=False):
        """Train two models for y1 (fpu_deltamax) and y2 (t_delta).

        Strategy: fast Optuna search (1000 trees, early_stop=20),
        then full retrain with best params (4000 trees, early_stop=50).
        Supports checkpoint/resume via checkpoint_dir.

        If skip_optuna=True and self.best_params is pre-populated,
        skips Optuna search and trains directly with those params.
        """
        import pickle as _pkl
        target_names = ['y1', 'y2']

        for i, name in enumerate(target_names):
            # Check if this target already trained (per-target checkpoint)
            if checkpoint_dir:
                ckpt_file = f'{checkpoint_dir}/{name}_lgb_model.pkl'
                if os.path.exists(ckpt_file):
                    logger.info(f"  LightGBM {name}: loading from checkpoint")
                    with open(ckpt_file, 'rb') as f:
                        self.models[name] = _pkl.load(f)
                    continue

            # Skip Optuna if pre-set params provided
            if skip_optuna and name in self.best_params:
                logger.info(f"  LightGBM {name}: using pre-set params (skip_optuna)")
                best = dict(self.best_params[name])
                best.update({
                    'objective': 'regression', 'metric': 'rmse',
                    'subsample_freq': 5,
                    'n_estimators': 4000, 'verbosity': -1,
                    'n_jobs': self.n_jobs, 'random_state': self.seed,
                })

                model = lgb.LGBMRegressor(**best)
                model.fit(
                    X_train, y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(0)],
                )
                self.models[name] = model

                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    ckpt_file = f'{checkpoint_dir}/{name}_lgb_model.pkl'
                    with open(ckpt_file, 'wb') as f:
                        _pkl.dump(model, f)
                continue

            logger.info(f"Optimizing LightGBM for {name}...")

            # Optuna with SQLite persistence
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                storage = f'sqlite:///{checkpoint_dir}/optuna_{name}.db'
                study = optuna.create_study(
                    study_name=f'LightGBM_{name}', direction='minimize',
                    storage=storage,
                    sampler=optuna.samplers.TPESampler(seed=self.seed),
                    load_if_exists=True,
                )
                n_done = len([t for t in study.trials
                              if t.state == optuna.trial.TrialState.COMPLETE])
                n_remaining = max(0, self.n_trials - n_done)
                logger.info(f"  Optuna: {n_done} done, {n_remaining} remaining")
            else:
                study = optuna.create_study(
                    direction='minimize',
                    sampler=optuna.samplers.TPESampler(seed=self.seed),
                )
                n_remaining = self.n_trials

            if n_remaining > 0:
                # Callback to update live plot every 5 trials
                def _optuna_callback(study, trial):
                    if trial.number % 5 == 0 and checkpoint_dir:
                        self._update_optuna_plot(study, name, checkpoint_dir)

                study.optimize(
                    lambda trial: self._objective(
                        trial, X_train, y_train[:, i], X_val, y_val[:, i]
                    ),
                    n_trials=n_remaining,
                    timeout=self.timeout,
                    show_progress_bar=True,
                    callbacks=[_optuna_callback],
                )

            best = study.best_params
            best.update({
                'objective': 'regression', 'metric': 'rmse',
                'subsample_freq': 5,
                'n_estimators': 4000, 'verbosity': -1,
                'n_jobs': self.n_jobs, 'random_state': self.seed,
            })

            logger.info(f"  Best {name} RMSE: {study.best_value:.6f}")
            logger.info(f"  Best params: {study.best_params}")

            # Full retrain with best params
            model = lgb.LGBMRegressor(**best)
            model.fit(
                X_train, y_train[:, i],
                eval_set=[(X_val, y_val[:, i])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )

            self.models[name] = model
            self.studies[name] = study

            # Save per-target checkpoint immediately
            if checkpoint_dir:
                ckpt_file = f'{checkpoint_dir}/{name}_lgb_model.pkl'
                with open(ckpt_file, 'wb') as f:
                    _pkl.dump(model, f)
                logger.info(f"  Saved {name} checkpoint to {ckpt_file}")

                # Live Optuna progress plot
                self._update_optuna_plot(study, name, checkpoint_dir)

            self.best_params[name] = best

    @staticmethod
    def _update_optuna_plot(study, target_name, save_dir):
        """Save Optuna optimization progress plot."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            trials = [t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE]
            if len(trials) < 2:
                return

            values = [t.value for t in trials]
            best_so_far = [min(values[:i+1]) for i in range(len(values))]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(1, len(values)+1), values, 'o-', alpha=0.4,
                    markersize=3, label='Trial RMSE')
            ax.plot(range(1, len(best_so_far)+1), best_so_far, 'r-',
                    linewidth=2, label='Best so far')
            ax.set_xlabel('Trial')
            ax.set_ylabel('Val RMSE')
            ax.set_title(f'LightGBM {target_name} — Optuna Progress (LIVE)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(os.path.join(save_dir, f'live_optuna_{target_name}.png'), dpi=100)
            plt.close(fig)
        except Exception:
            pass

    def predict(self, X) -> np.ndarray:
        """Predict both targets. Returns (N, 2)."""
        y1 = self.models['y1'].predict(X)
        y2 = self.models['y2'].predict(X)
        return np.column_stack([y1, y2])

    def measure_inference_time(self, X, n_warmup=10, n_runs=100) -> dict:
        """Measure CPU inference latency."""
        single = X[:1]

        for _ in range(n_warmup):
            self.predict(single)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter_ns()
            self.predict(single)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            times.append(elapsed_ms)

        # Batch throughput
        batch = X[:256]
        batch_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            self.predict(batch)
            batch_times.append(time.perf_counter() - start)

        return {
            'median_ms': float(np.median(times)),
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'throughput_samples_per_sec': float(len(batch) / np.median(batch_times)),
        }

    def get_feature_importance(self, target: str = 'y1') -> dict:
        """Return feature importance dict."""
        model = self.models[target]
        return dict(zip(
            [f"f{i}" for i in range(model.n_features_)],
            model.feature_importances_,
        ))
