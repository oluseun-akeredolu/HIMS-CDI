"""
streaming_engine.py

Live inference engine for shadow-mode deployment.
Reads one JSON event at a time, scores it, and writes the result to a
local log file. No network traffic leaves the machine.
"""

import sys
import json
import time
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import joblib
import psutil

# ------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------
# This file lives at: phase-1-cic-iomt/src/csra/streaming_engine.py
# We need to reach phase-1-cic-iomt/ to find artifacts/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from csra.config import ARTIFACTS_DIR
from csra.calibration import bbq_predict
from real_source_fusion import ScalarParticleFilter

# ------------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------------
FROZEN_MODEL_PATH = ARTIFACTS_DIR / "frozen_model_real.joblib"
MANIFEST_PATH = ARTIFACTS_DIR / "frozen_model_real.manifest.json"
LOG_DIR = _PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# GOVERNANCE THRESHOLDS
# ------------------------------------------------------------------
LATENCY_MS_THRESHOLD = 200
MEMORY_MB_THRESHOLD = 100

# ------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "inference.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("streaming_engine")

# ------------------------------------------------------------------
# GLOBAL STATE (created once when the engine starts)
# ------------------------------------------------------------------
_model_bundle = None
_particle_filter = None


def verify_model_integrity():
    """
    Step 1 when the engine boots: read the frozen model and check that
    its SHA-256 hash matches the manifest. If someone tampered with
    the file, this raises an error and stops everything.
    """
    logger.info("Verifying model integrity...")
    
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    with open(FROZEN_MODEL_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    if actual_hash != manifest["sha256"]:
        logger.error("SHA-256 MISMATCH!")
        logger.error("Expected: %s", manifest["sha256"])
        logger.error("Actual:   %s", actual_hash)
        raise ValueError("Frozen model integrity check failed. DO NOT DEPLOY.")

    logger.info("Model integrity verified. SHA-256: %s", actual_hash)
    return joblib.load(FROZEN_MODEL_PATH)


def initialize():
    """
    Run once at startup. Loads the model and creates the particle filter.
    The particle filter's state lives in memory and is reused for every
    event that arrives.
    """
    global _model_bundle, _particle_filter
    
    _model_bundle = verify_model_integrity()
    
    # Create particle filter with the same settings used during training
    _particle_filter = ScalarParticleFilter(
        n_particles=1000, 
        process_noise=0.1, 
        seed=7
    )
    
    # If the freeze bundle saved a particle filter state, restore it.
    if "particle_filter_state" in _model_bundle:
        state = _model_bundle["particle_filter_state"]
        _particle_filter.particles = state["particles"]
        _particle_filter.weights = state["weights"]
        logger.info("Restored particle filter state from frozen bundle.")
    else:
        logger.info("No saved PF state found; starting fresh.")

    logger.info("Engine initialized. Ready to process events.")


def process_event(event_dict):
    """
    Score a single event.
    
    event_dict is a Python dictionary. It should contain the feature
    columns from the CIC IoMT 2024 dataset. If a column is missing,
    the engine uses 0.0 as a safe default.
    """
    t0 = time.perf_counter()

    # 1. Build the feature vector in the exact order the model expects
    feature_cols = _model_bundle["feature_cols"]
    missing = [col for col in feature_cols if col not in event_dict]
    if missing:
        logger.warning("Event %s missing features: %s. Refusing to score.", event_dict.get("event_id", "unknown"), missing)
        return None
    x = np.array([[event_dict[col] for col in feature_cols]], dtype=np.float32)

    # 2. Base classifier: turns features into a raw risk score
    clf = _model_bundle["classifier"]
    raw_risk = clf.predict_proba(x)[0, 1]

    # 3. Particle filter: smooths the raw risk using past events
    smoothed_risk = _particle_filter.step(raw_risk)

    # 4. BBQ calibration: turns the smoothed score into a true probability
    bbq_model = _model_bundle["bbq_model"]
    calibrated_risk = bbq_predict(bbq_model, np.array([smoothed_risk]))[0]

    # 5. Measure speed and memory (Gate 3 checks)
    latency_ms = (time.perf_counter() - t0) * 1000
    memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    # 6. Check governance gates
    gate_status = {
        "gate3_latency_pass": latency_ms < LATENCY_MS_THRESHOLD,
        "gate3_memory_pass": memory_mb < MEMORY_MB_THRESHOLD,
    }
    all_gates_pass = all(gate_status.values())

    # 7. Build the result record
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_id": event_dict.get("event_id", "unknown"),
        "raw_risk": float(raw_risk),
        "smoothed_risk": float(smoothed_risk),
        "calibrated_risk": float(calibrated_risk),
        "latency_ms": round(latency_ms, 3),
        "memory_mb": round(memory_mb, 2),
        "gate_status": gate_status,
        "all_gates_pass": all_gates_pass,
    }

    return result


def main_loop():
    """
    The engine's heart. It reads JSON lines one by one.
    In production, this is replaced by a Kafka reader or a pcap watcher.
    For testing, it reads from your keyboard (stdin).
    """
    logger.info("=" * 60)
    logger.info("HIMS-CDI Streaming Engine Starting")
    logger.info("=" * 60)
    
    initialize()

    logger.info("Reading JSON events from stdin...")
    logger.info("Tip: Type a JSON line and press Enter. Ctrl+C to stop.")
    
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON skipped: %s", e)
                continue

            try:
                result = process_event(event)
                
                # Write to local log file ONLY — no internet, no remote server
                log_file = LOG_DIR / "scores.jsonl"
                with open(log_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                
                # Also print to the screen so you can see it
                print(json.dumps(result))
                
                # If a gate fails, shout about it
                if not result["all_gates_pass"]:
                    logger.warning("GOVERNANCE GATE BREACH: %s", result)

            except Exception as e:
                logger.error("Failed to process event: %s", e, exc_info=True)
                
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Exiting cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main_loop()
