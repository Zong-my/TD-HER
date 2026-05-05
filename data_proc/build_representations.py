#!/usr/bin/env python3
"""
Build three representations (A/B/C) from extracted CSV files.

RepA (Tabular): 1D flattened vector → LightGBM, KAN
RepB (Tensor):  (T, N, C) tensor + static → ConvLSTM, PatchTST, Mamba
RepC (Graph):   (N, T, C) node features + adjacency → ST-GCN

Usage:
    python data_proc/build_representations.py --csv-dir /path/to/csv --ms 10 --output-dir /path/to/reps
"""

import re
import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from pathlib import Path
from loguru import logger


def parse_temporal_columns(columns: list) -> dict:
    """Parse temporal column names to extract structure.

    Column format: SHEET_BUS_TIMESTEP (e.g., FREQ_30_0, FREQ_30_10)

    Returns:
        dict with keys: sheets, buses, timesteps, col_map
        col_map: {(sheet, bus, timestep): column_name}
    """
    meta_cols = {'distu_kind', 'file_name'}
    static_cols = {'load_level', 'load_zip_z', 'load_zip_i', 'load_zip_p',
                   'reserve_ratio', 'h_inertia', 'load_delta'}
    target_cols = {'fpu_deltamax', 't_delta'}
    skip = meta_cols | static_cols | target_cols

    sheets = set()
    buses = set()
    timesteps = set()
    col_map = {}

    for col in columns:
        if col in skip:
            continue
        # Parse SHEET_BUS_TIMESTEP
        parts = col.rsplit('_', 2)
        if len(parts) == 3:
            sheet, bus_str, ts_str = parts
            try:
                bus = int(bus_str)
                ts = int(ts_str)
                sheets.add(sheet)
                buses.add(bus)
                timesteps.add(ts)
                col_map[(sheet, bus, ts)] = col
            except ValueError:
                continue

    return {
        'sheets': sorted(sheets),
        'buses': sorted(buses),
        'timesteps': sorted(timesteps),
        'col_map': col_map,
    }


def compute_statistical_features(X_temporal: np.ndarray, sheets: list,
                                  buses: list) -> tuple:
    """Compute statistical features from the RepB tensor across timesteps.

    For each (sheet, bus) pair, computes temporal statistics across T timesteps.
    Also computes cross-bus spatial statistics per (sheet, timestep) and
    global system-level features.

    Args:
        X_temporal: (N, T, N_nodes, C) tensor from RepB
        sheets: list of sheet names (length C)
        buses: list of bus IDs (length N_nodes)

    Returns:
        X_stats: (N, D_stats) statistical feature array
        stat_names: list of feature names
        bcd_mask: list of bool — True for B/C/D (cross-bus/system/cross-channel)
                  features suitable for appending to x_static in RepB/RepC
    """
    N, T, N_nodes, C = X_temporal.shape
    features = []
    names = []
    bcd_mask = []  # True = B/C/D global feature, False = A per-bus temporal

    # ── 1. Per (sheet, bus) temporal statistics ──
    for c_idx, sheet in enumerate(sheets):
        for n_idx, bus in enumerate(buses):
            series = X_temporal[:, :, n_idx, c_idx]  # (N, T)
            prefix = f'{sheet}_{bus}'

            # Basic statistics
            features.append(np.mean(series, axis=1, keepdims=True))
            names.append(f'{prefix}_mean')

            features.append(np.std(series, axis=1, keepdims=True))
            names.append(f'{prefix}_std')

            features.append(np.var(series, axis=1, keepdims=True))
            names.append(f'{prefix}_var')

            features.append(np.median(series, axis=1, keepdims=True))
            names.append(f'{prefix}_median')

            features.append(np.min(series, axis=1, keepdims=True))
            names.append(f'{prefix}_min')

            features.append(np.max(series, axis=1, keepdims=True))
            names.append(f'{prefix}_max')

            features.append(np.ptp(series, axis=1, keepdims=True))  # max - min
            names.append(f'{prefix}_range')

            # RMS
            rms = np.sqrt(np.mean(series ** 2, axis=1, keepdims=True))
            features.append(rms)
            names.append(f'{prefix}_rms')

            # Argmin / Argmax (normalized to [0, 1])
            features.append(np.argmin(series, axis=1, keepdims=True).astype(np.float32) / max(T - 1, 1))
            names.append(f'{prefix}_argmin')

            features.append(np.argmax(series, axis=1, keepdims=True).astype(np.float32) / max(T - 1, 1))
            names.append(f'{prefix}_argmax')

            # First differences
            if T > 1:
                diffs = np.diff(series, axis=1)  # (N, T-1)
                features.append(np.mean(diffs, axis=1, keepdims=True))
                names.append(f'{prefix}_diff_mean')

                features.append(np.max(np.abs(diffs), axis=1, keepdims=True))
                names.append(f'{prefix}_diff_absmax')

                features.append(np.std(diffs, axis=1, keepdims=True))
                names.append(f'{prefix}_diff_std')

            # Linear slope (least-squares fit)
            if T > 1:
                t_axis = np.arange(T, dtype=np.float32)
                t_mean = t_axis.mean()
                t_var = np.sum((t_axis - t_mean) ** 2)
                s_mean = np.mean(series, axis=1, keepdims=True)
                slope = np.sum((t_axis[None, :] - t_mean) * (series - s_mean), axis=1, keepdims=True) / t_var
                features.append(slope)
                names.append(f'{prefix}_slope')

            # Skewness and Kurtosis (only if T >= 3)
            if T >= 3:
                sk = sp_stats.skew(series, axis=1)
                ku = sp_stats.kurtosis(series, axis=1)
                features.append(sk.reshape(-1, 1).astype(np.float32))
                names.append(f'{prefix}_skew')
                features.append(ku.reshape(-1, 1).astype(np.float32))
                names.append(f'{prefix}_kurt')

            # Energy (sum of squares, proportional to signal power)
            energy = np.sum(series ** 2, axis=1, keepdims=True)
            features.append(energy)
            names.append(f'{prefix}_energy')

            # Last - First value (total change)
            features.append((series[:, -1:] - series[:, :1]))
            names.append(f'{prefix}_total_change')

    # ── 2. Cross-bus spatial statistics per (sheet, timestep) ──
    # Aggregated: mean/std across buses at each timestep, then temporal stats of those
    for c_idx, sheet in enumerate(sheets):
        all_buses = X_temporal[:, :, :, c_idx]  # (N, T, N_nodes)

        # Spatial std across buses at each timestep -> (N, T)
        spatial_std = np.std(all_buses, axis=2)
        features.append(np.mean(spatial_std, axis=1, keepdims=True))
        names.append(f'{sheet}_spatial_std_mean')
        features.append(np.max(spatial_std, axis=1, keepdims=True))
        names.append(f'{sheet}_spatial_std_max')

        # Spatial range (max - min across buses) -> (N, T)
        spatial_range = np.ptp(all_buses, axis=2)
        features.append(np.mean(spatial_range, axis=1, keepdims=True))
        names.append(f'{sheet}_spatial_range_mean')
        features.append(np.max(spatial_range, axis=1, keepdims=True))
        names.append(f'{sheet}_spatial_range_max')

        # System mean (average across all buses) temporal stats
        sys_mean = np.mean(all_buses, axis=2)  # (N, T)
        features.append(np.std(sys_mean, axis=1, keepdims=True))
        names.append(f'{sheet}_sys_mean_std')
        features.append(np.ptp(sys_mean, axis=1, keepdims=True))
        names.append(f'{sheet}_sys_mean_range')

        if T > 1:
            sys_diff = np.diff(sys_mean, axis=1)
            features.append(np.max(np.abs(sys_diff), axis=1, keepdims=True))
            names.append(f'{sheet}_sys_mean_diff_absmax')

    # ── 3. Cross-channel interaction features ──
    # Pairwise correlations between key physical quantities per bus
    cross_pairs = [('FREQ', 'POWR'), ('FREQ', 'VOLT'), ('FREQ', 'SPD'),
                   ('POWR', 'SPD'), ('VOLT', 'ANGL')]
    for sheet_a, sheet_b in cross_pairs:
        if sheet_a in sheets and sheet_b in sheets:
            idx_a = sheets.index(sheet_a)
            idx_b = sheets.index(sheet_b)
            for n_idx, bus in enumerate(buses):
                sa = X_temporal[:, :, n_idx, idx_a]  # (N, T)
                sb = X_temporal[:, :, n_idx, idx_b]  # (N, T)
                if T > 1:
                    a_c = sa - sa.mean(axis=1, keepdims=True)
                    b_c = sb - sb.mean(axis=1, keepdims=True)
                    num = np.sum(a_c * b_c, axis=1, keepdims=True)
                    denom = np.sqrt(np.sum(a_c**2, axis=1, keepdims=True) *
                                    np.sum(b_c**2, axis=1, keepdims=True)) + 1e-12
                    features.append((num / denom).astype(np.float32))
                    names.append(f'{sheet_a}_{sheet_b}_{bus}_corr')

    # ── 4. Higher-order temporal features ──
    for c_idx, sheet in enumerate(sheets):
        for n_idx, bus in enumerate(buses):
            series = X_temporal[:, :, n_idx, c_idx]  # (N, T)
            prefix = f'{sheet}_{bus}'

            # Coefficient of variation (std / |mean| + eps)
            s_mean = np.mean(series, axis=1, keepdims=True)
            s_std = np.std(series, axis=1, keepdims=True)
            cv = s_std / (np.abs(s_mean) + 1e-12)
            features.append(cv.astype(np.float32))
            names.append(f'{prefix}_cv')

            # Quantiles: 25th, 75th percentile
            q25 = np.percentile(series, 25, axis=1).reshape(-1, 1)
            q75 = np.percentile(series, 75, axis=1).reshape(-1, 1)
            features.append(q25.astype(np.float32))
            names.append(f'{prefix}_q25')
            features.append(q75.astype(np.float32))
            names.append(f'{prefix}_q75')
            features.append((q75 - q25).astype(np.float32))
            names.append(f'{prefix}_iqr')

            # Zero-crossing rate (how many times signal crosses its mean)
            if T > 1:
                above = series > s_mean
                crossings = np.sum(np.abs(np.diff(above.astype(np.float32), axis=1)), axis=1, keepdims=True)
                features.append((crossings / max(T - 1, 1)).astype(np.float32))
                names.append(f'{prefix}_zcr')

            # Second differences (acceleration)
            if T > 2:
                d2 = np.diff(series, n=2, axis=1)  # (N, T-2)
                features.append(np.mean(np.abs(d2), axis=1, keepdims=True).astype(np.float32))
                names.append(f'{prefix}_d2_absmean')
                features.append(np.max(np.abs(d2), axis=1, keepdims=True).astype(np.float32))
                names.append(f'{prefix}_d2_absmax')

            # Ratio: last value / first value (relative change)
            first_val = series[:, :1]
            last_val = series[:, -1:]
            ratio = last_val / (np.abs(first_val) + 1e-12)
            features.append(ratio.astype(np.float32))
            names.append(f'{prefix}_last_first_ratio')

    # ── 5. System-level aggregate features ──
    # Weighted system frequency deviation (using all FREQ buses)
    if 'FREQ' in sheets:
        freq_idx = sheets.index('FREQ')
        all_freq = X_temporal[:, :, :, freq_idx]  # (N, T, N_nodes)
        # Max absolute deviation across all buses and timesteps
        features.append(np.max(np.abs(all_freq.reshape(N, -1)), axis=1, keepdims=True).astype(np.float32))
        names.append('FREQ_global_absmax')
        # Timestep of max system-mean deviation
        sys_freq = np.mean(all_freq, axis=2)  # (N, T)
        features.append((np.argmax(np.abs(sys_freq), axis=1, keepdims=True).astype(np.float32) / max(T-1, 1)))
        names.append('FREQ_sys_argmax_dev')

    # Concatenate all
    X_stats = np.concatenate(features, axis=1).astype(np.float32)

    # Replace NaN/Inf
    X_stats = np.nan_to_num(X_stats, nan=0.0, posinf=0.0, neginf=0.0)

    # Build BCD mask: True for cross-bus (B), system-level (C), cross-channel (D)
    bcd_mask = []
    for n in names:
        is_bcd = ('spatial_' in n or 'sys_' in n or 'global_' in n or '_corr' in n)
        bcd_mask.append(is_bcd)

    return X_stats, names, bcd_mask


def build_rep_a(df: pd.DataFrame, ms: int) -> tuple:
    """Build Representation A (tabular, 1D flattened).

    Returns:
        X: np.ndarray (N, D) — static + temporal flattened
        y: np.ndarray (N, 2) — [fpu_deltamax, t_delta]
        feature_names: list of column names
    """
    static_cols = ['load_level', 'load_zip_z', 'load_zip_i', 'load_zip_p',
                   'reserve_ratio', 'h_inertia', 'load_delta']
    target_cols = ['fpu_deltamax', 't_delta']
    meta_cols = ['distu_kind', 'file_name']

    temporal_cols = [c for c in df.columns if c not in meta_cols + static_cols + target_cols]

    feature_names = static_cols + sorted(temporal_cols)
    X = df[feature_names].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    return X, y, feature_names


def build_rep_b(df: pd.DataFrame, ms: int) -> tuple:
    """Build Representation B (tensor T×N×C + static).

    Returns:
        X_temporal: np.ndarray (N_samples, T, N_nodes, C_features)
        X_static: np.ndarray (N_samples, 7)
        y: np.ndarray (N_samples, 2)
        meta: dict with sheets, buses, timesteps ordering
    """
    static_cols = ['load_level', 'load_zip_z', 'load_zip_i', 'load_zip_p',
                   'reserve_ratio', 'h_inertia', 'load_delta']
    target_cols = ['fpu_deltamax', 't_delta']

    structure = parse_temporal_columns(df.columns.tolist())
    sheets = structure['sheets']
    buses = structure['buses']
    timesteps = [t for t in structure['timesteps'] if t <= ms]
    col_map = structure['col_map']

    N = len(df)
    T = len(timesteps)
    N_nodes = len(buses)
    C = len(sheets)

    X_temporal = np.zeros((N, T, N_nodes, C), dtype=np.float32)

    for t_idx, t in enumerate(timesteps):
        for n_idx, bus in enumerate(buses):
            for c_idx, sheet in enumerate(sheets):
                col = col_map.get((sheet, bus, t))
                if col is not None and col in df.columns:
                    X_temporal[:, t_idx, n_idx, c_idx] = df[col].values.astype(np.float32)

    X_static = df[static_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    meta = {
        'sheets': sheets,
        'buses': buses,
        'node_ids': buses,
        'generator_ids': buses,
        'timesteps': timesteps,
        'shape': f"({N}, {T}, {N_nodes}, {C})",
    }

    return X_temporal, X_static, y, meta


def build_rep_c(X_temporal_b: np.ndarray, X_static: np.ndarray,
                y: np.ndarray) -> tuple:
    """Build Representation C (graph: node-centric) from Representation B.

    Transposes temporal tensor from (N, T, Nodes, C) to (N, Nodes, T, C).

    Returns:
        X_node: np.ndarray (N, Nodes, T, C)
        X_static: same as input
        y: same as input
    """
    # (N, T, Nodes, C) -> (N, Nodes, T, C)
    X_node = np.transpose(X_temporal_b, (0, 2, 1, 3))
    return X_node, X_static, y


def save_split(output_dir: str, split_name: str, ms: int,
               rep_a: tuple = None, rep_b: tuple = None, rep_c: tuple = None,
               feature_names: list = None, meta: dict = None):
    """Save all representations for a split."""

    if rep_a is not None:
        X_a, y_a, fnames = rep_a
        d = os.path.join(output_dir, 'repA', f'ms{ms}')
        Path(d).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(d, f'X_{split_name}.npy'), X_a)
        np.save(os.path.join(d, f'y_{split_name}.npy'), y_a)
        if feature_names:
            import json
            with open(os.path.join(d, 'feature_names.json'), 'w') as f:
                json.dump(fnames, f)
        logger.info(f"  RepA {split_name}: X={X_a.shape}, y={y_a.shape}")

    if rep_b is not None:
        X_t, X_s, y_b, _ = rep_b
        d = os.path.join(output_dir, 'repB', f'ms{ms}')
        Path(d).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(d, f'X_temporal_{split_name}.npy'), X_t)
        np.save(os.path.join(d, f'X_static_{split_name}.npy'), X_s)
        np.save(os.path.join(d, f'y_{split_name}.npy'), y_b)
        if meta:
            import json
            with open(os.path.join(d, 'meta.json'), 'w') as f:
                json.dump(meta, f)
        logger.info(f"  RepB {split_name}: X_t={X_t.shape}, X_s={X_s.shape}")

    if rep_c is not None:
        X_n, X_s_c, y_c = rep_c
        d = os.path.join(output_dir, 'repC', f'ms{ms}')
        Path(d).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(d, f'X_node_{split_name}.npy'), X_n)
        np.save(os.path.join(d, f'X_static_{split_name}.npy'), X_s_c)
        np.save(os.path.join(d, f'y_{split_name}.npy'), y_c)
        logger.info(f"  RepC {split_name}: X_node={X_n.shape}")


def process_csv_to_reps(csv_dir: str, output_dir: str, ms_list: list,
                        splits: list = None):
    """Process all CSV files and build representations.

    Args:
        csv_dir: Directory containing {split}_ms{ms}.csv files
        output_dir: Where to save npy files
        ms_list: List of ms values to process
        splits: List of split names (auto-detected if None)
    """
    if splits is None:
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and '_ms' in f]
        splits = sorted(set(f.split('_ms')[0] for f in csv_files))

    logger.info(f"CSV dir: {csv_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Splits: {splits}")
    logger.info(f"MS values: {ms_list}")

    for ms in ms_list:
        logger.info(f"\n--- ms={ms} ---")

        for split in splits:
            csv_path = os.path.join(csv_dir, f"{split}_ms{ms}.csv")
            if not os.path.exists(csv_path):
                logger.warning(f"Not found: {csv_path}")
                continue

            logger.info(f"Loading {csv_path}")
            df = pd.read_csv(csv_path)
            logger.info(f"  {split}: {len(df)} samples, {df.shape[1]} columns")

            # Build representations
            X_a, y_a, fnames = build_rep_a(df, ms)

            X_t, X_s, y_b, meta = build_rep_b(df, ms)

            # Compute statistical features
            X_stats, stat_names, bcd_mask = compute_statistical_features(
                X_t, meta['sheets'], meta['buses'])

            # RepA: append ALL statistical features (A+B+C+D)
            X_a_full = np.concatenate([X_a, X_stats], axis=1)
            fnames_full = fnames + stat_names
            rep_a = (X_a_full, y_a, fnames_full)

            # RepB/RepC x_static: append ALL statistical features (A+B+C+D)
            # This gives DL models the same rich feature set as LightGBM
            X_s_extended = np.concatenate([X_s, X_stats], axis=1).astype(np.float32)

            n_stats = X_stats.shape[1]
            logger.info(f"  Stats: {n_stats} total "
                        f"(RepA: {X_a.shape[1]}+{n_stats}={X_a_full.shape[1]}) "
                        f"(x_static: 7+{n_stats}={X_s_extended.shape[1]})")

            rep_b = (X_t, X_s_extended, y_b, meta)

            X_n, X_s_c, y_c = build_rep_c(X_t, X_s_extended, y_b)
            rep_c = (X_n, X_s_c, y_c)

            # Save with extended meta
            meta['static_names'] = ['load_level', 'load_zip_z', 'load_zip_i',
                                     'load_zip_p', 'reserve_ratio', 'h_inertia',
                                     'load_delta'] + stat_names
            meta['n_static'] = X_s_extended.shape[1]

            save_split(output_dir, split, ms,
                       rep_a=rep_a, rep_b=rep_b, rep_c=rep_c,
                       feature_names=fnames_full, meta=meta)

    logger.info("\nRepresentation building complete!")


def main():
    parser = argparse.ArgumentParser(description="Build representations from CSV")
    parser.add_argument('--csv-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--ms-list', nargs='+', type=int, default=[1, 5, 10, 15, 25])
    parser.add_argument('--splits', nargs='+', default=None)
    args = parser.parse_args()

    process_csv_to_reps(args.csv_dir, args.output_dir, args.ms_list, args.splits)


if __name__ == '__main__':
    main()
