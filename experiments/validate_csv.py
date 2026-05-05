#!/usr/bin/env python3
"""Validate extracted CSVs: counts, ratios, leakage, consistency."""
import argparse
import pandas as pd
import sys
from loguru import logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv-dir', required=True)
    p.add_argument('--ms', type=int, default=10)
    args = p.parse_args()

    splits = {}
    for s in ['train', 'val', 'test']:
        path = f'{args.csv_dir}/{s}_ms{args.ms}.csv'
        df = pd.read_csv(path)
        splits[s] = df
        logger.info(f"  {s}: {len(df)} samples, {df.shape[1]} cols")

    ok = True

    # NaN check
    for s, df in splits.items():
        if df.isnull().any().any():
            logger.error(f"  FAIL: {s} has NaN!")
            ok = False

    # Target columns
    for s, df in splits.items():
        for col in ['fpu_deltamax', 't_delta', 'file_name']:
            if col not in df.columns:
                logger.error(f"  FAIL: {s} missing {col}")
                ok = False

    # Ratio check
    total = sum(len(v) for v in splits.values())
    for s, lo, hi in [('train', 0.78, 0.82), ('val', 0.08, 0.12), ('test', 0.08, 0.12)]:
        ratio = len(splits[s]) / total
        if lo < ratio < hi:
            logger.info(f"  PASS: {s} ratio {ratio:.2%}")
        else:
            logger.error(f"  FAIL: {s} ratio {ratio:.2%} not in [{lo},{hi}]")
            ok = False

    # Leakage check
    tr_files = set(splits['train']['file_name'])
    val_files = set(splits['val']['file_name'])
    te_files = set(splits['test']['file_name'])
    for a, b, name in [('train', 'val', 'train/val'), ('train', 'test', 'train/test'),
                        ('val', 'test', 'val/test')]:
        overlap = set(splits[a]['file_name']) & set(splits[b]['file_name'])
        if len(overlap) == 0:
            logger.info(f"  PASS: no {name} leakage")
        else:
            logger.error(f"  FAIL: {name} overlap: {len(overlap)} files!")
            ok = False

    # Column consistency
    if splits['train'].columns.tolist() == splits['test'].columns.tolist():
        logger.info("  PASS: column consistency")
    else:
        logger.error("  FAIL: column mismatch between train and test")
        ok = False

    if ok:
        logger.info("  CSV VALIDATION PASSED")
    else:
        logger.error("  CSV VALIDATION FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
