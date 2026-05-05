#!/usr/bin/env python3
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def _as_1d_array(values):
    return np.asarray(values, dtype=float).reshape(-1)


def logloss(act, pred):
    act = _as_1d_array(act)
    pred = np.clip(_as_1d_array(pred), 1e-15, 1 - 1e-15)
    epsilon = 1e-15
    ll = sum(act * np.log(pred) + np.subtract(1, act) * np.log(np.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll

def rmse_loss(act, pred):
    return root_mean_squared_error(_as_1d_array(act), _as_1d_array(pred))

def mae_loss(act, pred):
    mae = mean_absolute_error(_as_1d_array(act), _as_1d_array(pred))
    return mae

def smape_loss(act, pred):
    y_true = np.nan_to_num(_as_1d_array(act))
    y_pred = np.nan_to_num(_as_1d_array(pred))
    smape_score = 2.0 * np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_pred) + np.abs(y_true), 1e-15)) * 100
    return smape_score

def mape_loss(act, pred):
    y_true = np.nan_to_num(_as_1d_array(act))
    y_pred = np.nan_to_num(_as_1d_array(pred))

    # Only filter on true values to avoid division by near-zero
    mask = np.abs(y_true) >= 0.001
    if mask.sum() == 0:
        return np.nan

    mape_score = (np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])).mean() * 100
    return mape_score
