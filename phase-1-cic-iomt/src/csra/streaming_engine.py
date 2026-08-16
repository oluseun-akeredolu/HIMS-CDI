#!/usr/bin/env python3
"""
HIMS-CDI Streaming Engine — Shadow-Mode Pilot
Full pipeline: classifier → particle filter → BBQ calibration
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
from collections import deque
from logging.handlers import RotatingFileHandler

import numpy as np
import psutil
import joblib

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
_model_bundle = None
_clf = None
_pf = None
_bbq = None
_config = None
_baseline_memory_mb = 0.0
_logger = logging.getLogger("hims_cdi")
_audit_logger = logging.getLogger("hims_cdi.audit")

# Rolling window for CDI/CCR (last 1000 scored events)
_recent_scores = deque(maxlen=1000)


# -----------------------------------------------------------------------------
def load_config(config_path: pathlib.Path) -> dict:
    """Load and validate deployment configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    required = [
        "frozen_artifact_path",
        "manifest_path",
        "log_dir",
        "log_level",
        "log_rotation_mb",
        "retention_days",
    ]
    for key in required:
        if key not in cfg:
            raise KeyError(f"Missing required config key: {key}")

    # Governance gates with defaults
    gates = cfg.get("governance_gates", {})
    gates.setdefault("latency_ms_max", 200)
    gates.setdefault("memory_mb_max", 100)
    gates.setdefault("cdi_min", 0.70)
    gates.setdefault("ccr_min", 0.80)
    gates.setdefault("ccr_threshold", 0.5)
    cfg["governance_gates"] = gates

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
    """Configure rotating file logs and console output."""
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
def _apply_particle_filter(raw_risk: float) -> float:
    """
    Apply particle filter smoothing.
    Tries multiple common APIs; falls back to exponential smoothing.
    """
    global _pf
    if _pf is None:
        return raw_risk

    # Try OO API: update() returns smoothed value
    if hasattr(_pf, "update") and callable(getattr(_pf, "update")):
        try:
            result = _pf.update(raw_risk)
            if result is not None:
                return float(result)
        except Exception:
            pass

    # Try predict API
    if hasattr(_pf, "predict") and callable(getattr(_pf, "predict")):
        try:
            result = _pf.predict([[raw_risk]])
            return float(result[0]) if hasattr(result, "__getitem__") else float(result)
        except Exception:
            pass

    # Fallback: state dict with exponential smoothing
    if isinstance(_pf, dict):
        alpha = float(_pf.get("alpha", 0.30))
        prev = float(_pf.get("prev_smoothed", raw_risk))
        smoothed = alpha * raw_risk + (1.0 - alpha) * prev
        _pf["prev_smoothed"] = smoothed
        return smoothed

    return raw_risk


# -----------------------------------------------------------------------------
def _apply_bbq(smoothed_risk: float) -> float:
    """
    Apply Bayesian Binning into Quantiles (BBQ) calibration.
    Tries multiple common APIs.
    """
    global _bbq
    if _bbq is None:
        return smoothed_risk

    # Try sklearn-style predict
    if hasattr(_bbq, "predict") and callable(getattr(_bbq, "predict")):
        try:
            result = _bbq.predict([[smoothed_risk]])
            val = float(result[0]) if hasattr(result, "__getitem__") else float(result)
            return max(0.0, min(1.0, val))
        except Exception:
            pass

    # Try predict_proba
    if hasattr(_bbq, "predict_proba") and callable(getattr(_bbq, "predict_proba")):
        try:
            result = _bbq.predict_proba([[smoothed_risk]])
            return float(result[0, 1])
        except Exception:
            pass

    # Try calibrate method
    if hasattr(_bbq, "calibrate") and callable(getattr(_bbq, "calibrate")):
        try:
            val = float(_bbq.calibrate(smoothed_risk))
            return max(0.0, min(1.0, val))
        except Exception:
            pass

    _logger.warning("BBQ model has no recognized working method; using smoothed risk")
    return smoothed_risk


# -----------------------------------------------------------------------------
def _compute_cdi_ccr() -> tuple:
    """
    Compute rolling-window CDI and CCR.
    CDI (Calibration Drift Index): 1 - normalized mean absolute calibration shift
    CCR (Clinical Confidence Ratio): fraction of calibrated risks above threshold
    """
    if len(_recent_scores) < 10:
        return None, None

    raw_vals = []
    cal_vals = []
    for r in _recent_scores:
        if r.get("raw_risk") is not None and r.get("calibrated_risk") is not None:
            raw_vals.append(r["raw_risk"])
            cal_vals.append(r["calibrated_risk"])

    if not raw_vals:
        return None, None

    # CDI: higher is better (1.0 = no drift, 0.0 = maximum drift)
    mae = sum(abs(c - r) for c, r in zip(cal_vals, raw_vals)) / len(raw_vals)
    mean_raw = sum(raw_vals) / len(raw_vals)
    cdi = 1.0 - (mae / mean_raw if mean_raw > 0 else 0.0)
    cdi = max(0.0, min(1.0, cdi))

    # CCR: fraction above clinical action threshold
    threshold = _config["governance_gates"].get("ccr_threshold", 0.5)
    ccr = sum(1.0 for c in cal_vals if c >= threshold) / len(cal_vals)

    return round(cdi, 3), round(ccr, 3)


# -----------------------------------------------------------------------------
def initialize(config_path: pathlib.Path = None):
    """Load config, verify model, warm up predictor, record baseline memory."""
    global _model_bundle, _clf, _pf, _bbq, _config, _baseline_memory_mb

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

    # Load all three components of the Brain
    _clf = _model_bundle.get("classifier")
    _pf = _model_bundle.get("particle_filter_state")
    _bbq = _model_bundle.get("bbq_model")
    feature_cols = _model_bundle["feature_cols"]

    if _clf is None:
        raise KeyError("Bundle missing 'classifier'")

    _logger.info("Model loaded: %s", type(_clf).__name__)
    _logger.info("Particle filter: %s", "present" if _pf is not None else "MISSING")
    _logger.info("BBQ model: %s", "present" if _bbq is not None else "MISSING")
    _logger.info("Feature columns: %d", len(feature_cols))

    # Warm-up: run full pipeline once to avoid cold-start latency
    dummy = np.zeros((1, len(feature_cols)), dtype=np.float32)
    raw = float(_clf.predict_proba(dummy)[0, 1])
    smoothed = _apply_particle_filter(raw)
    calibrated = _apply_bbq(smoothed)
    _logger.info(
        "Warm-up completed: raw=%.4f, smoothed=%.4f, calibrated=%.4f",
        raw, smoothed, calibrated,
    )

    # Baseline memory after model + warm-up
    process = psutil.Process()
    _baseline_memory_mb = process.memory_info().rss / (1024 * 1024)
    _logger.info("Baseline memory: %.2f MB", _baseline_memory_mb)


# -----------------------------------------------------------------------------
def process_event(event_dict: dict) -> dict:
    """
    Score a single event through the full pipeline.
    Returns dict with raw, smoothed, calibrated risks and governance gates.
    """
    global _clf, _model_bundle, _config, _baseline_memory_mb

    if _clf is None:
        raise RuntimeError("Engine not initialized. Call initialize() first.")

    event_id = event_dict.get("event_id", "UNKNOWN")
    feature_cols = _model_bundle["feature_cols"]

    # --- Missing feature guard (P1 fix) ------------------------------------
    missing = [col for col in feature_cols if col not in event_dict]
    if missing:
        _logger.warning(
            "Event %s missing features: %s. Refusing to score.",
            event_id, missing,
        )
        return {
            "event_id": event_id,
            "raw_risk": None,
            "smoothed_risk": None,
            "calibrated_risk": None,
            "risk_score": None,
            "scored": False,
            "reason": f"missing_features:{missing}",
        }

    # Build feature vector
    x = np.array([[event_dict[col] for col in feature_cols]], dtype=np.float32)

    # --- Full pipeline: classifier → particle filter → BBQ -----------------
    t0 = time.perf_counter()
    raw_risk = float(_clf.predict_proba(x)[0, 1])
    smoothed_risk = _apply_particle_filter(raw_risk)
    calibrated_risk = _apply_bbq(smoothed_risk)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # --- Delta memory -------------------------------------------------------
    process = psutil.Process()
    current_mb = process.memory_info().rss / (1024 * 1024)
    delta_mb = current_mb - _baseline_memory_mb

    # --- Rolling CDI / CCR --------------------------------------------------
    score_record = {
        "raw_risk": raw_risk,
        "smoothed_risk": smoothed_risk,
        "calibrated_risk": calibrated_risk,
    }
    _recent_scores.append(score_record)
    cdi, ccr = _compute_cdi_ccr()

    # --- Governance gate evaluation -----------------------------------------
    gates = _config["governance_gates"]
    gate_latency = latency_ms <= gates["latency_ms_max"]
    gate_memory = delta_mb <= gates["memory_mb_max"]
    gate_cdi = True if cdi is None else cdi >= gates["cdi_min"]
    gate_ccr = True if ccr is None else ccr >= gates["ccr_min"]
    gate_overall = gate_latency and gate_memory and gate_cdi and gate_ccr

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
    if not gate_cdi:
        _logger.warning(
            "Event %s breached CDI gate: %.3f < %.2f",
            event_id, cdi, gates["cdi_min"],
        )
    if not gate_ccr:
        _logger.warning(
            "Event %s breached CCR gate: %.3f < %.2f",
            event_id, ccr, gates["ccr_min"],
        )

    result = {
        "event_id": event_id,
        "raw_risk": raw_risk,
        "smoothed_risk": smoothed_risk,
        "calibrated_risk": calibrated_risk,
        "risk_score": calibrated_risk,  # primary output is calibrated
        "scored": True,
        "latency_ms": round(latency_ms, 3),
        "memory_delta_mb": round(delta_mb, 3),
        "cdi": cdi,
        "ccr": ccr,
        "gate_latency_pass": gate_latency,
        "gate_memory_pass": gate_memory,
        "gate_cdi_pass": gate_cdi,
        "gate_ccr_pass": gate_ccr,
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
    for h in _audit_logger.handlers:
        h.flush()


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
        "Engine ready. Gates: latency <= %d ms, memory delta <= %d MB, "
        "CDI >= %.2f, CCR >= %.2f",
        gates["latency_ms_max"],
        gates["memory_mb_max"],
        gates["cdi_min"],
        gates["ccr_min"],
    )
    _logger.info("Waiting for JSON events on stdin...")

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

        print(json.dumps(result, separators=(",", ":")))
        sys.stdout.flush()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()