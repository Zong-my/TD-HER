#!/usr/bin/env python3
"""mRMR (minimum Redundancy Maximum Relevance) feature selection.

Supports both fixed-K and adaptive selection via elbow detection on
cumulative F-statistic relevance.
"""

import numpy as np
import pandas as pd
from mrmr import mrmr_regression
from loguru import logger


def mrmr_feature_select(X_train: np.ndarray, y_train: np.ndarray,
                        feature_names: list, K: int = 200,
                        target_idx: int = 0) -> list:
    """Select top-K features using mRMR for a single target.

    Args:
        X_train: (N, D) feature matrix
        y_train: (N, 2) targets — use target_idx to select which
        feature_names: list of D column names
        K: number of features to select
        target_idx: 0 for y1 (fpu_deltamax), 1 for y2 (t_delta)

    Returns:
        sorted list of selected feature names
    """
    K = min(K, len(feature_names))
    X_df = pd.DataFrame(X_train, columns=feature_names)
    y_series = pd.Series(y_train[:, target_idx], name='target')

    selected = mrmr_regression(
        X=X_df, y=y_series, K=K,
        relevance='f', redundancy='c', denominator='mean',
    )
    logger.info(f"mRMR selected {len(selected)} features for target_idx={target_idx}")
    return selected


def find_elbow(values: np.ndarray) -> int:
    """Find elbow point using maximum perpendicular distance to the line
    connecting the first and last points.

    Args:
        values: 1D array of cumulative relevance scores (sorted descending)

    Returns:
        index of elbow point (0-based)
    """
    n = len(values)
    if n <= 2:
        return n - 1

    # Normalize to [0, 1]
    x = np.linspace(0, 1, n)
    y = (values - values[-1]) / (values[0] - values[-1] + 1e-12)

    # Line from first to last point
    # Distance from each point to this line
    # Line: from (0, 1) to (1, 0) => y = 1 - x => x + y - 1 = 0
    # But use actual endpoints for robustness
    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]
    dx, dy = x1 - x0, y1 - y0
    line_len = np.sqrt(dx**2 + dy**2)

    # Perpendicular distance
    dist = np.abs(dy * x - dx * y + x1 * y0 - y1 * x0) / (line_len + 1e-12)

    return int(np.argmax(dist))


def adaptive_mrmr_select(X_train: np.ndarray, y_train: np.ndarray,
                         feature_names: list, target_idx: int = 0,
                         min_k: int = 50, max_k: int = None,
                         relevance_ratio: float = 0.95) -> list:
    """Adaptive mRMR: rank all features, then select K via elbow detection.

    Strategy:
    1. Run mRMR to rank top max_k features.
    2. Compute cumulative F-statistic relevance.
    3. Find elbow point where marginal gain drops off.
    4. Clamp result to [min_k, max_k].

    Args:
        X_train: (N, D) feature matrix
        y_train: (N, 2) targets
        feature_names: list of D column names
        target_idx: 0 or 1
        min_k: minimum features to select
        max_k: maximum features to rank (default: min(D, 800))
        relevance_ratio: alternative cutoff — select features capturing
                         this fraction of the top feature's relevance

    Returns:
        list of selected feature names
    """
    D = len(feature_names)
    if max_k is None:
        max_k = min(D, 800)

    # Rank top max_k features
    ranked = mrmr_feature_select(X_train, y_train, feature_names,
                                 K=max_k, target_idx=target_idx)

    if len(ranked) <= min_k:
        return ranked

    # Compute individual F-statistic relevance for ranked features
    from sklearn.feature_selection import f_regression
    ranked_idx = [feature_names.index(f) for f in ranked]
    X_ranked = X_train[:, ranked_idx]
    f_scores, _ = f_regression(X_ranked, y_train[:, target_idx])
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    # Cumulative relevance (features are already mRMR-ranked)
    cum_relevance = np.cumsum(f_scores)
    total_relevance = cum_relevance[-1]

    # Method 1: Elbow detection on cumulative curve
    elbow_k = find_elbow(f_scores) + 1  # +1 for count

    # Method 2: Capture relevance_ratio of total relevance
    if total_relevance > 0:
        ratio_k = np.searchsorted(cum_relevance, total_relevance * relevance_ratio) + 1
    else:
        ratio_k = min_k

    # Take the larger of the two, clamped to [min_k, max_k]
    adaptive_k = max(min_k, min(max(elbow_k, ratio_k), max_k))

    logger.info(f"Adaptive mRMR: target_idx={target_idx}, D={D}, "
                f"elbow_k={elbow_k}, ratio_k={ratio_k}, "
                f"final K={adaptive_k}/{max_k}")

    return ranked[:adaptive_k]


def mrmr_select_union(X_train: np.ndarray, y_train: np.ndarray,
                      feature_names: list, K: int = None,
                      adaptive: bool = True) -> tuple:
    """Select features using mRMR for both targets, return union.

    Args:
        X_train: (N, D)
        y_train: (N, 2)
        feature_names: list of D names
        K: features per target (ignored if adaptive=True)
        adaptive: if True, use elbow-based adaptive selection

    Returns:
        (union_features, y1_features, y2_features)
    """
    if adaptive:
        y1_features = adaptive_mrmr_select(
            X_train, y_train, feature_names, target_idx=0)
        y2_features = adaptive_mrmr_select(
            X_train, y_train, feature_names, target_idx=1)
    else:
        if K is None:
            K = 200
        y1_features = mrmr_feature_select(X_train, y_train, feature_names, K, target_idx=0)
        y2_features = mrmr_feature_select(X_train, y_train, feature_names, K, target_idx=1)

    union = sorted(set(y1_features) | set(y2_features))
    logger.info(f"mRMR union: {len(y1_features)} + {len(y2_features)} "
                f"→ {len(union)} unique features (adaptive={adaptive})")

    return union, y1_features, y2_features
