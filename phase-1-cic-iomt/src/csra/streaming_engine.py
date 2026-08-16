#!/usr/bin/env python3
"""
HIMS-CDI Streaming Engine — Shadow-Mode Pilot
Runs on Windows 11 NHS PC as a dedicated appliance.
"""

import argparse
import hashlib
import json
import logging
import pathlib
import signal
import sys
import time
from logging.handlers import RotatingFileHandler

import numpy as np
import psutil
import joblib

# -----------------------------------------------------------------------------
# Globals set during initialize()
# -----------------------------------------------------------------------------
_model_bundle = None
_clf = None
_config = None
_baseline_memory_mb = 0.0
_logger = logging.getLogger("hims_cdi")
_audit_logger = logging.getLogger("hims_cdi.audit")


# -----------------------------------------------------------------------------
def load_config(config_path: pathlib.Path) -> dict:
    """Load and validate deployment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required_top = [
        "frozen_artifact_path",
        "manifest_path",
        "log_dir",
        "log_level",
        "log_rotation_mb",
        "retention_days",
    ]
    for key in required_top:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")

    # Governance gates — provide defaults if absent so the engine never crashes
    gates = cfg.get("governance_gates", {})
    gates.setdefault("latency_ms_max", 200)
    gates.setdefault("memory_mb_max", 100)
    cfg["governance_gates"] = gates

    # Resolve relative paths against the directory containing the config file
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
    """Configure rotating file logs and console output."""
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates on re-import
    for h in list(root.handlers):
        root.removeHandler(h)

    # Rotating inference log
    inf_handler = RotatingFileHandler(
        log_dir / "inference.log",
        maxBytes=rotation_mb * 1024 * 1024,
        backupCount=retention,
        encoding="utf-8",
    )
    inf_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(inf_handler)

    # Console handler
    con_handler = logging.StreamHandler(sys.stdout)
    con_handler.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(con_handler)

    # Audit log (JSON Lines) — separate logger, events are small
    audit_path = log_dir / "audit.jsonl"
    audit_handler = logging.FileHandler(audit_path, encoding="utf-8")
    audit_handler.setFormatter(logging.Formatter("%(message)s"))
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.addHandler(audit_handler)
    _audit_logger.propagate = False

    _logger.info("Logging initialized. Audit stream: %s", audit_path)


# -----------------------------------------------------------------------------
def verify_model_integrity(model_path: pathlib.Path, manifest_path: pathlib.Path):
    """SHA-256 verification before loading."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
    expected = manifest.get("sha256", "")
    if actual != expected:
        raise ValueError(
            f"Model integrity check FAILED.\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}"
        )
    _logger.info("Model integrity verified (SHA-256 match).")


# -----------------------------------------------------------------------------
def initialize(config_path: pathlib.Path = None):
    """Load config, verify model, warm up predictor, record baseline memory."""
    global _model_bundle, _clf, _config, _baseline_memory_mb

    if config_path is None:
        config_path = pathlib.Path("phase-1-cic-iomt/deployment_config.json").resolve()
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
    _clf = _model_bundle["classifier"]
    feature_cols = _model_bundle["feature_cols"]

    _logger.info("Model loaded: %s", type(_clf).__name__)
    _logger.info("Feature columns: %d", len(feature_cols))

    # Warm-up to avoid counting numpy/sklearn thread-pool spin-up as latency
    dummy = np.zeros((1, len(feature_cols)), dtype=np.float32)
    _ = _clf.predict_proba(dummy)
    _logger.info("Warm-up predict_proba() completed.")

    # Baseline memory after model is loaded (delta measurement)
    process = psutil.Process()
    _baseline_memory_mb = process.memory_info().rss / (1024 * 1024)
    _logger.info("Baseline memory: %.2f MB", _baseline_memory_mb)


# -----------------------------------------------------------------------------
def process_event(event_dict: dict) -> dict:
    """
    Score a single event.
    Returns a dict with risk_score (or None if unscorable) and metadata.
    """
    global _clf, _model_bundle, _config, _baseline_memory_mb

    if _clf is None:
        raise RuntimeError("Engine not initialized. Call initialize() first.")

    event_id = event_dict.get("event_id", "UNKNOWN")
    feature_cols = _model_bundle["feature_cols"]

    # --- P1 Fix: Refuse to score if features are missing --------------------
    missing = [col for col in feature_cols if col not in event_dict]
    if missing:
        _logger.warning(
            "Event %s missing features: %s. Refusing to score.",
            event_id,
            missing,
        )
        return {
            "event_id": event_id,
            "risk_score": None,
            "scored": False,
            "reason": f"missing_features:{missing}",
        }

    # Build feature vector
    x = np.array([[event_dict[col] for col in feature_cols]], dtype=np.float32)

    # --- Timed inference ----------------------------------------------------
    t0 = time.perf_counter()
    raw_risk = float(_clf.predict_proba(x)[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # --- Delta memory -------------------------------------------------------
    process = psutil.Process()
    current_mb = process.memory_info().rss / (1024 * 1024)
    delta_mb = current_mb - _baseline_memory_mb

    # --- Governance gate evaluation -----------------------------------------
    gates = _config["governance_gates"]
    gate_latency = latency_ms <= gates["latency_ms_max"]
    gate_memory = delta_mb <= gates["memory_mb_max"]
    gate_overall = gate_latency and gate_memory

    if not gate_latency:
        _logger.warning(
            "Event %s breached latency gate: %.2f ms > %d ms",
            event_id,
            latency_ms,
            gates["latency_ms_max"],
        )
    if not gate_memory:
        _logger.warning(
            "Event %s breached memory gate: %.2f MB > %d MB",
            event_id,
            delta_mb,
            gates["memory_mb_max"],
        )

    result = {
        "event_id": event_id,
        "risk_score": raw_risk,
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
    """Write structured JSON Lines audit record."""
    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record.update(result)
    _audit_logger.info(json.dumps(record, separators=(",", ":")))


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HIMS-CDI Streaming Engine")
    parser.add_argument(
        "--config",
        default="phase-1-cic-iomt/deployment_config.json",
        help="Path to deployment_config.json",
    )
    args = parser.parse_args()

    config_path = pathlib.Path(args.config).resolve()
    initialize(config_path)

    gates = _config["governance_gates"]
    _logger.info(
        "Engine ready. Gates: latency <= %d ms, memory delta <= %d MB",
        gates["latency_ms_max"],
        gates["memory_mb_max"],
    )
    _logger.info("Waiting for JSON events on stdin...")

    # Graceful shutdown on Ctrl+C
    def _sigint_handler(signum, frame):
        _logger.info("Shutdown signal received. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            _logger.error("Malformed JSON on stdin: %s", exc)
            continue

        result = process_event(event)
        audit_log(result)

        # Emit compact result to stdout for downstream piping
        print(json.dumps(result, separators=(",", ":")))
        sys.stdout.flush()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()