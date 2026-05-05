#!/usr/bin/env python3
"""Experiment 7 rebuild: IEEE300 scalability verification.

The legacy ``results/ieee300/exp7`` directory failed the audit and is kept only
as a historical artifact. New runs should use the post-trigger rebuilt dataset,
write to ``exp7_rebuild`` or another fresh directory, and then pass
``experiments/audit_ieee300_exp7.py`` before any paper claim uses them.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp1_main_comparison import EXP1_MODELS, evaluate, run_experiment1

SEED = 42


def run_experiment7(rep_dir: str, adj_path: str, result_dir: str,
                    ms: int = 10, device: str = 'cuda', models_to_run=None):
    """Run Experiment 7 — same as Experiment 1 but on IEEE 300 data."""
    logger.info("="*60)
    logger.info("Experiment 7: IEEE 300 Scalability Verification")
    logger.info(f"  rep_dir: {rep_dir}")
    logger.info(f"  adj_path: {adj_path}")
    logger.info(f"  ms: {ms}")

    # Reuse Experiment 1 logic with IEEE 300 paths
    return run_experiment1(
        rep_dir, adj_path, result_dir, ms, device, models_to_run=models_to_run
    )


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rep-dir', default='data/ieee300_v2_posttrigger')
    p.add_argument('--adj-path', default='data/ieee300_v2/adjacency/adjacency.npy')
    p.add_argument('--result-dir', default='results/ieee300/exp7_rebuild')
    p.add_argument('--ms', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--models', nargs='+', choices=EXP1_MODELS, default=None)
    args = p.parse_args()
    run_experiment7(
        args.rep_dir, args.adj_path, args.result_dir, args.ms, args.device,
        models_to_run=args.models,
    )
