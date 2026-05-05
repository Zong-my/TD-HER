#!/usr/bin/env python3
"""Fair inference timing utilities for CPU and GPU models."""

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
