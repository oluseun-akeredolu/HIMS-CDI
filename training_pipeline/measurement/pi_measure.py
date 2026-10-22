"""
Raspberry Pi deployment measurement script (manuscript Section 5.3 + S14 PDF).

This is a SKELETON for the Pi measurement. Running it requires an actual
Raspberry Pi 4 Model B (4 GB RAM) with ONNX Runtime 1.15 installed.
We cannot run it on this workstation — the numbers must be measured on the Pi.

Protocol (from S14 PDF):
    - Hardware: Raspberry Pi 4 Model B (4 GB RAM, ARM Cortex-A72 @ 1.5 GHz)
    - OS: Raspberry Pi OS Lite (64-bit, Debian Bullseye)
    - Runtime: ONNX Runtime 1.15 with Python 3.11
    - Model variant: INT8 quantised via Post-Training Quantisation
    - Warm-up phase: 50 inference passes
    - Timed phase: 100 inference passes
    - Peak memory: measured via memory_profiler
"""

import time
import numpy as np


def measure_pi_single_thread(onnx_model_path, X_test, warm_up_passes=50, timed_passes=100):
    """Measure single-thread inference on the Pi.

    Run this on a Raspberry Pi 4. Requires onnxruntime==1.15.0 and memory_profiler.

    Usage on the Pi:
        pip install onnxruntime==1.15.0 memory_profiler numpy pandas
        python pi_measure.py --onnx-model HIMS_CDI_cal_INT8.onnx \\
            --archive-dir /path/to/v1.3.1_archive
    """
    try:
        import onnxruntime as ort
        import memory_profiler
    except ImportError as e:
        raise ImportError(
            f"This script must be run on a Raspberry Pi 4 with onnxruntime and memory_profiler. "
            f"Missing: {e}"
        )

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_model_path, sess_options)
    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(warm_up_passes):
        session.run(None, {input_name: X_test.astype(np.float32)})

    # Timed
    latencies = []

    def timed_run():
        for _ in range(timed_passes):
            start = time.perf_counter()
            session.run(None, {input_name: X_test.astype(np.float32)})
            latencies.append((time.perf_counter() - start) * 1000)

    peak_memory_mb = memory_profiler.memory_usage(timed_run, max_usage=True)

    return {
        'latency_ms_mean': float(np.mean(latencies)),
        'latency_ms_std': float(np.std(latencies)),
        'peak_memory_mb': float(peak_memory_mb),
        'n_passes': timed_passes,
    }
