#!/usr/bin/env python3
"""Fair inference timing utilities for CPU and GPU models."""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import time
import numpy as np
import torch
from loguru import logger


def measure_lightgbm_time(model, X, n_warmup=10, n_runs=100) -> dict:
    """Measure LightGBM CPU inference time."""
    single = X[:1]
    batch = X[:256]

    for _ in range(n_warmup):
        model.predict(single)

    # Single sample latency
    times = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        model.predict(single)
        times.append((time.perf_counter_ns() - start) / 1e6)

    # Batch throughput
    batch_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(batch)
        batch_times.append(time.perf_counter() - start)

    return {
        'median_ms': float(np.median(times)),
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'throughput': float(len(batch) / np.median(batch_times)),
    }
