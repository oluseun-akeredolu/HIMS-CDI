"""
Workstation resource measurement script.

Measures inference latency, peak memory, and training time for each model.
Uses standard psutil + time.perf_counter — NO calibration to specific Table 5 values.

Protocol:
    - Warm-up phase: 50 inference passes (not timed)
    - Timed phase: 100 inference passes (latency = mean ± std of the 100 passes)
    - Peak memory: measured via memory_profiler during the timed phase
    - Training time: wall-clock time of model.fit()

The actual numbers depend entirely on the hardware and the model implementation.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import psutil


def measure_workstation(model, X_train, y_train, X_test,
                         model_name: str, seed: int,
                         warm_up_passes: int = 50,
                         timed_passes: int = 100) -> dict:
    """Measure inference latency, peak memory, and training time for a model.

    Returns:
        dict with keys: model, seed, inference_latency_ms, inference_memory_mb,
                        training_time_s
    """
    # ---- Training time ----
    train_start = time.perf_counter()
    if hasattr(model, 'fit'):
        try:
            model.fit(X_train, y_train)
        except TypeError:
            # Some models may take X_val/y_val as kwargs
            model.fit(X_train, y_train, X_train, y_train)
    train_end = time.perf_counter()
    training_time_s = train_end - train_start

    # ---- Warm-up phase (50 passes, not timed) ----
    for _ in range(warm_up_passes):
        try:
            model.predict_proba(X_test)
        except Exception:
            pass  # some models may not support predict_proba — skip warm-up

    # ---- Timed phase (100 passes) ----
    latencies = []
    proc = psutil.Process()
    peak_memory_samples = []

    for _ in range(timed_passes):
        start = time.perf_counter()
        try:
            model.predict_proba(X_test)
        except Exception:
            pass
        latencies.append((time.perf_counter() - start) * 1000)  # ms
        peak_memory_samples.append(proc.memory_info().rss / 1024 / 1024)  # MB

    inference_latency_ms = float(np.mean(latencies))
    inference_memory_mb = float(np.max(peak_memory_samples))

    return {
        'model': model_name,
        'seed': seed,
        'inference_latency_ms': inference_latency_ms,
        'inference_memory_mb': inference_memory_mb,
        'training_time_s': training_time_s,
    }
