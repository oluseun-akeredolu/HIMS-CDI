"""
workstation_measure.py — Workstation resource measurement (corrected: no double-fit).

Measures inference latency and maximum sampled Python-process RSS for each model.
Does NOT re-fit the model — the caller passes an already-fitted model.

Protocol:
    - The model must already be fitted before calling this function.
    - Warm-up: 50 inference passes (not timed)
    - Timed: 100 inference passes on the full 48-case batch
    - Latency: mean ± std of 100 predict_proba calls on the batch
    - Memory: maximum sampled Python-process RSS during the timed phase
    - Training time: measured separately by the caller (not here)

Note: These are Python process-level measurements of batch predict_proba calls,
NOT end-to-end pipeline measurements and NOT single-case latency.
The measured process memory exceeds the 100 MB gate threshold for most models.
This does NOT establish target-device deployment feasibility.
"""

import time
import numpy as np
import psutil


def measure_inference(model, X_test, model_name, seed,
                      warm_up_passes=50, timed_passes=100):
    """Measure inference latency and process RSS for an already-fitted model.

    Args:
        model: already-fitted model with predict_proba() method
        X_test: test features (numpy array)
        model_name: name of the model
        seed: the training seed
        warm_up_passes: warm-up inference passes (not timed)
        timed_passes: timed inference passes

    Returns:
        dict with: model, seed, inference_latency_ms, inference_latency_std_ms,
        inference_memory_mb, measurement_protocol
    """
    # Warm-up
    for _ in range(warm_up_passes):
        try:
            model.predict_proba(X_test)
        except Exception:
            pass

    # Timed phase
    latencies = []
    proc = psutil.Process()
    rss_samples = []

    for _ in range(timed_passes):
        start = time.perf_counter()
        try:
            model.predict_proba(X_test)
        except Exception:
            pass
        latencies.append((time.perf_counter() - start) * 1000)  # ms
        rss_samples.append(proc.memory_info().rss / 1024 / 1024)  # MB

    return {
        'model': model_name,
        'seed': seed,
        'inference_latency_ms': float(np.mean(latencies)),
        'inference_latency_std_ms': float(np.std(latencies)),
        'inference_memory_mb': float(np.max(rss_samples)),
        'measurement_protocol': 'batch predict_proba on 48 cases, max sampled Python-process RSS',
    }
