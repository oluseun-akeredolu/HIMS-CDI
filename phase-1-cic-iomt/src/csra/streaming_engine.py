#!/usr/bin/env python3
"""
HIMS-CDI Edge Brain -- Shadow-Mode Pilot
Deployment-ready inference engine for NHS Windows 11 Edge PC.

Architecture enforced:
  classifier -> particle filter (CLEAN state) -> BBQ calibration -> governance gates

Design principles:
  - Inference ONLY. No learning, no refitting, no model mutation.
  - Crash containment: one bad event cannot kill the 30-day process.
  - Immutable model: SHA-256 verified on every startup.
  - Clean streaming state: particle filter starts fresh (no TEST contamination).
  - Honest governance: CDI/CCR are computed POST-PILOT, not fabricated per-event.
"""

import argparse
import hashlib
import json
import logging
import pathlib
import signal
import sys
import time
import traceback
from logging.handlers import RotatingFileHandler

import numpy as np
import psutil
import joblib

# Path resolution: support running standalone on Windows without PYTHONPATH
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from csra.calibration import bbq_predict
from real_source_fusion import ScalarParticleFilter

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
_model_bundle = None
_clf = None
_pf = None
_bbq = None
_config = None
_baseline_memory_mb = 0.0
_feature_cols = None
_logger = logging.getLogger("hims_cdi")
_audit_logger = logging.getLogger("hims_cdi.audit")


# -----------------------------------------------------------------------------
def load_config(config_path: pathlib.Path) -> dict:
    """Load and validate deployment configuration. Single flat schema only."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = [
        "frozen_artifact_path",
        "manifest_path",
        "log_dir",
        "log_level",
        "log_rotation_mb",
        "retention_days",
        "governance_gates",
    ]
    for key in required:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")

    gates = cfg["governance_gates"]
    gate_required = ["latency_ms_max", "memory_mb_max"]
    for gkey in gate_required:
        if gkey not in gates:
            raise KeyError(f"Missing required governance gate: {gkey}")

    # Resolve relative paths against config directory
    base = config_path.parent.resolve()
    for key in ("frozen_artifact_path", "manifest_path"):
        p = pathlib.Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(base / p)

    log_dir = pathlib.Path(cfg["log_dir"])
    if not log_dir.is_absolute():
        cfg["log_dir"] = str(base / log_dir)

    return cfg


# -----------------------------------------------------------------------------
def setup_logging(log_dir: pathlib.Path, level: str, rotation_mb: int, retention: int):
    """Rotating inference log + structured JSON Lines audit log."""
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for h in list(root.handlers):
        root.removeHandler(h)

    inf_handler = RotatingFileHandler(
        log_dir / "inference.log",
        maxBytes=rotation_mb * 1024 * 1024,
        backupCount=retention,
        encoding="utf-8",
    )
    inf_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(inf_handler)

    con_handler = logging.StreamHandler(sys.stdout)
    con_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(con_handler)

    audit_path = log_dir / "audit.jsonl"
    audit_handler = logging.FileHandler(audit_path, encoding="utf-8")
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.addHandler(audit_handler)
    _audit_logger.propagate = False

    _logger.info("Logging initialized. Audit stream: %s", audit_path)


# -----------------------------------------------------------------------------
def verify_model_integrity(model_path: pathlib.Path, manifest_path: pathlib.Path):
    """SHA-256 verification. Refuse to start if mismatch."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    expected = manifest.get("sha256", "")
    if actual != expected:
        raise ValueError(
            f"MODEL INTEGRITY FAILURE.\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  DO NOT START THE PILOT."
        )
    _logger.info("Model integrity verified (SHA-256 match).")


# -----------------------------------------------------------------------------
def validate_event(event_dict: dict, feature_cols: list) -> tuple:
    """
    Validate a single incoming event.
    Returns (is_valid: bool, reason: str, cleaned_event: dict or None)
    """
    if not isinstance(event_dict, dict):
        return False, "event_not_dict", None

    event_id = event_dict.get("event_id", "UNKNOWN")

    # Check for missing features
    missing = [col for col in feature_cols if col not in event_dict]
    if missing:
        return False, f"missing_features:{missing}", None

    # Check for extra features (warn but don't fail -- forward compatibility)
    extra = [k for k in event_dict.keys() if k not in feature_cols and k != "event_id"]
    if extra:
        _logger.debug("Event %s has extra fields (ignored): %s", event_id, extra)

    # Validate every feature is numeric and finite
    cleaned = {"event_id": event_id}
    for col in feature_cols:
        val = event_dict[col]
        if isinstance(val, bool):
            val = float(val)
        if not isinstance(val, (int, float)):
            return False, f"non_numeric_feature:{col}={val}", None
        if not np.isfinite(val):
            return False, f"non_finite_feature:{col}={val}", None
        cleaned[col] = float(val)

    return True, "", cleaned


# -----------------------------------------------------------------------------
def initialize(config_path: pathlib.Path = None):
    """Load config, verify model, warm up, record baseline memory."""
    global _model_bundle, _clf, _pf, _bbq, _config, _baseline_memory_mb, _feature_cols

    if config_path is None:
        config_path = pathlib.Path("deployment_config.json").resolve()
    else:
        config_path = pathlib.Path(config_path).resolve()

    _config = load_config(config_path)
    setup_logging(
        _config["log_dir"],
        _config["log_level"],
        _config["log_rotation_mb"],
        _config["retention_days"],
    )

    model_path = pathlib.Path(_config["frozen_artifact_path"])
    manifest_path = pathlib.Path(_config["manifest_path"])

    verify_model_integrity(model_path, manifest_path)

    _model_bundle = joblib.load(model_path)

    # Load classifier
    _clf = _model_bundle.get("classifier")
    if _clf is None:
        raise KeyError("Bundle missing 'classifier'")

    # Load BBQ calibrator
    _bbq = _model_bundle.get("bbq_model")
    if _bbq is None:
        raise KeyError("Bundle missing 'bbq_model'")

    # CRITICAL FIX: Initialize particle filter with CLEAN state.
    # We do NOT restore PF state from the bundle to avoid TEST-set contamination.
    # The Edge PC starts with a fresh, unbiased prior for the October 2026 pilot.
    _pf = ScalarParticleFilter(n_particles=1000, process_noise=0.1, seed=7)
    _logger.info("Initialized clean particle filter for deployment (no TEST contamination).")

    _feature_cols = _model_bundle["feature_cols"]
    _logger.info("Model loaded: %s", type(_clf).__name__)
    _logger.info("BBQ model: present")
    _logger.info("Feature columns: %d", len(_feature_cols))

    # Warm-up: full pipeline once to avoid cold-start latency
    dummy = np.zeros((1, len(_feature_cols)), dtype=np.float32)
    raw = float(_clf.predict_proba(dummy)[0, 1])
    smoothed = float(_pf.step(raw))
    calibrated = float(bbq_predict(_bbq, np.array([smoothed]))[0])
    calibrated = max(0.0, min(1.0, calibrated))
    _logger.info(
        "Warm-up completed: raw=%.4f, smoothed=%.4f, calibrated=%.4f",
        raw, smoothed, calibrated,
    )

    # Baseline memory after warm-up
    process = psutil.Process()
    _baseline_memory_mb = process.memory_info().rss / (1024 * 1024)
    _logger.info("Baseline memory: %.2f MB", _baseline_memory_mb)


# -----------------------------------------------------------------------------
def process_event(event_dict: dict) -> dict:
    """
    Score a single event through the full pipeline.
    Returns result dict. Raises on unrecoverable error (caller must catch).
    """
    global _clf, _pf, _bbq, _config, _baseline_memory_mb, _feature_cols

    if _clf is None:
        raise RuntimeError("Engine not initialized. Call initialize() first.")

    event_id = event_dict.get("event_id", "UNKNOWN")

    # Validate input
    is_valid, reason, cleaned = validate_event(event_dict, _feature_cols)
    if not is_valid:
        _logger.warning("Event %s rejected: %s", event_id, reason)
        return {
            "event_id": event_id,
            "raw_risk": None,
            "smoothed_risk": None,
            "calibrated_risk": None,
            "risk_score": None,
            "scored": False,
            "reason": reason,
        }

    # Build feature vector
    x = np.array([[cleaned[col] for col in _feature_cols]], dtype=np.float32)

    # Full pipeline: classifier -> particle filter -> BBQ
    t0 = time.perf_counter()
    raw_risk = float(_clf.predict_proba(x)[0, 1])
    smoothed_risk = float(_pf.step(raw_risk))
    calibrated_risk = float(bbq_predict(_bbq, np.array([smoothed_risk]))[0])
    calibrated_risk = max(0.0, min(1.0, calibrated_risk))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Delta memory
    process = psutil.Process()
    current_mb = process.memory_info().rss / (1024 * 1024)
    delta_mb = current_mb - _baseline_memory_mb

    # Governance gates (per-event: latency + memory only)
    gates = _config["governance_gates"]
    gate_latency = latency_ms <= gates["latency_ms_max"]
    gate_memory = delta_mb <= gates["memory_mb_max"]
    gate_overall = gate_latency and gate_memory

    if not gate_latency:
        _logger.warning(
            "Event %s breached latency gate: %.2f ms > %d ms",
            event_id, latency_ms, gates["latency_ms_max"],
        )
    if not gate_memory:
        _logger.warning(
            "Event %s breached memory gate: %.2f MB > %d MB",
            event_id, delta_mb, gates["memory_mb_max"],
        )

    result = {
        "event_id": event_id,
        "raw_risk": raw_risk,
        "smoothed_risk": smoothed_risk,
        "calibrated_risk": calibrated_risk,
        "risk_score": calibrated_risk,
        "scored": True,
        "latency_ms": round(latency_ms, 3),
        "memory_delta_mb": round(delta_mb, 3),
        "gate_latency_pass": gate_latency,
        "gate_memory_pass": gate_memory,
        "gate_overall_pass": gate_overall,
    }
    return result


# -----------------------------------------------------------------------------
def audit_log(result: dict):
    """Write structured JSON Lines audit record with explicit flush."""
    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record.update(result)
    _audit_logger.info(json.dumps(record, separators=(",", ":")))
    for h in _audit_logger.handlers:
        h.flush()


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HIMS-CDI Edge Brain")
    parser.add_argument(
        "--config",
        default="deployment_config.json",
        help="Path to deployment_config.json",
    )
    args = parser.parse_args()

    config_path = pathlib.Path(args.config).resolve()
    initialize(config_path)

    gates = _config["governance_gates"]
    _logger.info(
        "Edge Brain ready. Gates: latency <= %d ms, memory delta <= %d MB",
        gates["latency_ms_max"],
        gates["memory_mb_max"],
    )
    _logger.info("CDI and CCR are aggregate metrics computed AFTER the pilot.")
    _logger.info("Waiting for JSON events on stdin...")

    def _sigint_handler(signum, frame):
        _logger.info("Shutdown signal received. Exiting cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    # -------------------------------------------------------------------------
    # CRASH CONTAINMENT: The outer loop must NEVER die from one bad event.
    # -------------------------------------------------------------------------
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # 1. Parse JSON
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            _logger.error("Malformed JSON on stdin: %s", exc)
            continue

        # 2. Process with full exception boundary
        try:
            result = process_event(event)
            audit_log(result)
            print(json.dumps(result, separators=(",", ":")))
            sys.stdout.flush()
        except Exception as exc:
            event_id = event.get("event_id", "UNKNOWN") if isinstance(event, dict) else "UNKNOWN"
            _logger.error(
                "UNHANDLED EXCEPTION processing event %s: %s\n%s",
                event_id,
                exc,
                traceback.format_exc(),
            )
            # Emit a quarantine record so the event is not silently lost
            quarantine = {
                "event_id": event_id,
                "raw_risk": None,
                "smoothed_risk": None,
                "calibrated_risk": None,
                "risk_score": None,
                "scored": False,
                "reason": f"engine_exception:{type(exc).__name__}",
            }
            audit_log(quarantine)
            print(json.dumps(quarantine, separators=(",", ":")))
            sys.stdout.flush()
            # CRITICAL: Continue the loop. Do NOT re-raise. Do NOT exit.
            continue


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()