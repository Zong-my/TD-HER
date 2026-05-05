#!/usr/bin/env python3
"""
Strict cross-validation of data pipeline before training.

Checks:
1. Split ratios (80/10/10)
2. No data leakage between splits (file-level)
3. Feature dimensions consistency across RepA/B/C
4. No NaN/Inf in any array
5. Target distribution sanity
6. x_static and x_temporal shape alignment
7. mRMR feature names exist in data
8. Adjacency matrix shape matches n_generators
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from loguru import logger

PASS = 0
FAIL = 0


def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        logger.info(f"  PASS: {msg}")
    else:
        FAIL += 1
        logger.error(f"  FAIL: {msg}")


def validate_split_ratios(rep_dir, ms=10):
    """Check that splits are ~80/10/10."""
    logger.info("=== Split Ratios ===")
    counts = {}
    for split in ['train', 'val', 'test']:
        y = np.load(f'{rep_dir}/repA/ms{ms}/y_{split}.npy')
        counts[split] = len(y)
    total = sum(counts.values())
    for split, n in counts.items():
        ratio = n / total * 100
        logger.info(f"  {split}: {n} ({ratio:.1f}%)")

    check(0.78 < counts['train'] / total < 0.82,
          f"Train ratio {counts['train']/total*100:.1f}% in [78,82]%")
    check(0.08 < counts['val'] / total < 0.12,
          f"Val ratio {counts['val']/total*100:.1f}% in [8,12]%")
    check(0.08 < counts['test'] / total < 0.12,
          f"Test ratio {counts['test']/total*100:.1f}% in [8,12]%")
    return counts


def validate_no_leakage(csv_dir, ms=10):
    """Check no file overlap between splits via file_name column."""
    logger.info("=== Data Leakage Check ===")
    import pandas as pd
    sets = {}
    for split in ['train', 'val', 'test']:
        csv_path = f'{csv_dir}/{split}_ms{ms}.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=['file_name'])
            sets[split] = set(df['file_name'].tolist())
            logger.info(f"  {split}: {len(sets[split])} unique files")

    if len(sets) == 3:
        tr_val = sets['train'] & sets['val']
        tr_te = sets['train'] & sets['test']
        val_te = sets['val'] & sets['test']
        check(len(tr_val) == 0, f"train∩val overlap: {len(tr_val)}")
        check(len(tr_te) == 0, f"train∩test overlap: {len(tr_te)}")
        check(len(val_te) == 0, f"val∩test overlap: {len(val_te)}")
    else:
        logger.warning("  Could not check all splits")


def validate_shapes(rep_dir, ms=10):
    """Check dimension consistency across representations."""
    logger.info("=== Shape Consistency ===")

    for split in ['train', 'val', 'test']:
        # RepA
        Xa = np.load(f'{rep_dir}/repA/ms{ms}/X_{split}.npy')
        ya = np.load(f'{rep_dir}/repA/ms{ms}/y_{split}.npy')
        check(Xa.shape[0] == ya.shape[0],
              f"RepA {split}: X({Xa.shape[0]}) == y({ya.shape[0]})")
        check(ya.shape[1] >= 2, f"RepA {split}: y has >=2 targets")

        # RepB
        Xt = np.load(f'{rep_dir}/repB/ms{ms}/X_temporal_{split}.npy')
        Xs = np.load(f'{rep_dir}/repB/ms{ms}/X_static_{split}.npy')
        yb = np.load(f'{rep_dir}/repB/ms{ms}/y_{split}.npy')
        check(Xt.shape[0] == Xs.shape[0] == yb.shape[0],
              f"RepB {split}: B dims match ({Xt.shape[0]})")
        check(Xt.ndim == 4, f"RepB {split}: X_temporal is 4D {Xt.shape}")
        check(Xa.shape[0] == Xt.shape[0],
              f"RepA/B {split}: same N ({Xa.shape[0]} vs {Xt.shape[0]})")

        # RepC
        Xn = np.load(f'{rep_dir}/repC/ms{ms}/X_node_{split}.npy')
        check(Xn.shape[0] == Xt.shape[0],
              f"RepC {split}: same N as RepB ({Xn.shape[0]})")

    # Cross-split dimension consistency
    Xa_tr = np.load(f'{rep_dir}/repA/ms{ms}/X_train.npy')
    Xa_te = np.load(f'{rep_dir}/repA/ms{ms}/X_test.npy')
    check(Xa_tr.shape[1] == Xa_te.shape[1],
          f"RepA D consistent: train={Xa_tr.shape[1]} test={Xa_te.shape[1]}")

    Xt_tr = np.load(f'{rep_dir}/repB/ms{ms}/X_temporal_train.npy')
    Xt_te = np.load(f'{rep_dir}/repB/ms{ms}/X_temporal_test.npy')
    check(Xt_tr.shape[1:] == Xt_te.shape[1:],
          f"RepB temporal shape consistent: {Xt_tr.shape[1:]} vs {Xt_te.shape[1:]}")

    Xs_tr = np.load(f'{rep_dir}/repB/ms{ms}/X_static_train.npy')
    Xs_te = np.load(f'{rep_dir}/repB/ms{ms}/X_static_test.npy')
    check(Xs_tr.shape[1] == Xs_te.shape[1],
          f"RepB static D consistent: {Xs_tr.shape[1]} vs {Xs_te.shape[1]}")


def validate_no_nan_inf(rep_dir, ms=10):
    """Check no NaN or Inf in any array."""
    logger.info("=== NaN/Inf Check ===")
    for split in ['train', 'val', 'test']:
        arrays = {
            f'RepA X_{split}': np.load(f'{rep_dir}/repA/ms{ms}/X_{split}.npy'),
            f'RepA y_{split}': np.load(f'{rep_dir}/repA/ms{ms}/y_{split}.npy'),
            f'RepB Xt_{split}': np.load(f'{rep_dir}/repB/ms{ms}/X_temporal_{split}.npy'),
            f'RepB Xs_{split}': np.load(f'{rep_dir}/repB/ms{ms}/X_static_{split}.npy'),
            f'RepC Xn_{split}': np.load(f'{rep_dir}/repC/ms{ms}/X_node_{split}.npy'),
        }
        for name, arr in arrays.items():
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            check(not has_nan and not has_inf, f"{name}: no NaN/Inf")


def validate_targets(rep_dir, ms=10):
    """Check target distribution sanity."""
    logger.info("=== Target Distribution ===")
    for split in ['train', 'val', 'test']:
        y = np.load(f'{rep_dir}/repA/ms{ms}/y_{split}.npy')
        y1, y2 = y[:, 0], y[:, 1]
        logger.info(f"  {split} y1: [{y1.min():.4f}, {y1.max():.4f}] "
                     f"mean={y1.mean():.4f} std={y1.std():.4f}")
        logger.info(f"  {split} y2: [{y2.min():.4f}, {y2.max():.4f}] "
                     f"mean={y2.mean():.4f} std={y2.std():.4f}")
        check(y1.std() > 0.01, f"{split} y1 has variance (std={y1.std():.4f})")
        check(y2.std() > 0.01, f"{split} y2 has variance (std={y2.std():.4f})")
        check(y2.min() >= 0, f"{split} y2 (time) >= 0")


def validate_adjacency(adj_path, rep_dir, ms=10):
    """Check adjacency matrix matches generator count."""
    logger.info("=== Adjacency Matrix ===")
    if os.path.exists(adj_path):
        adj = np.load(adj_path)
        Xt = np.load(f'{rep_dir}/repB/ms{ms}/X_temporal_train.npy')
        n_gen = Xt.shape[2]
        check(adj.shape[0] == n_gen,
              f"Adjacency {adj.shape} matches n_generators={n_gen}")
        check(adj.shape[0] == adj.shape[1], "Adjacency is square")
        check(np.allclose(adj, adj.T), "Adjacency is symmetric")
    else:
        logger.warning(f"  Adjacency not found: {adj_path}")


def validate_feature_names(rep_dir, ms=10):
    """Check feature names match array dimensions."""
    logger.info("=== Feature Names ===")
    with open(f'{rep_dir}/repA/ms{ms}/feature_names.json') as f:
        fnames = json.load(f)
    Xa = np.load(f'{rep_dir}/repA/ms{ms}/X_train.npy')
    check(len(fnames) == Xa.shape[1],
          f"RepA feature_names ({len(fnames)}) == X_train D ({Xa.shape[1]})")

    with open(f'{rep_dir}/repB/ms{ms}/meta.json') as f:
        meta = json.load(f)
    Xs = np.load(f'{rep_dir}/repB/ms{ms}/X_static_train.npy')
    check(len(meta['static_names']) == Xs.shape[1],
          f"RepB static_names ({len(meta['static_names'])}) == X_static D ({Xs.shape[1]})")
    check(meta['n_static'] == Xs.shape[1],
          f"RepB meta.n_static ({meta['n_static']}) == X_static D ({Xs.shape[1]})")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rep-dir', required=True)
    p.add_argument('--adj-path', default=None)
    p.add_argument('--csv-dir', default=None)
    p.add_argument('--ms', type=int, default=10)
    args = p.parse_args()

    if args.csv_dir is None:
        args.csv_dir = os.path.join(args.rep_dir, 'csv')
    if args.adj_path is None:
        # Try parent dir for adjacency (symlinked data may not have it)
        for candidate in [
            os.path.join(args.rep_dir, 'adjacency/adjacency.npy'),
            'data/ieee39_v8/adjacency/adjacency.npy',
        ]:
            if os.path.exists(candidate):
                args.adj_path = candidate
                break

    logger.info(f"Validating: {args.rep_dir} (ms={args.ms})")

    validate_split_ratios(args.rep_dir, args.ms)
    validate_no_leakage(args.csv_dir, args.ms)
    validate_shapes(args.rep_dir, args.ms)
    validate_no_nan_inf(args.rep_dir, args.ms)
    validate_targets(args.rep_dir, args.ms)
    if args.adj_path:
        validate_adjacency(args.adj_path, args.rep_dir, args.ms)
    validate_feature_names(args.rep_dir, args.ms)

    logger.info(f"\n{'='*60}")
    logger.info(f"Results: {PASS} PASS, {FAIL} FAIL")
    if FAIL > 0:
        logger.error("VALIDATION FAILED — do NOT proceed with training!")
        sys.exit(1)
    else:
        logger.info("ALL CHECKS PASSED — safe to train.")


if __name__ == '__main__':
    main()
