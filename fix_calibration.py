import pathlib

p = pathlib.Path(r"phase-1-cic-iomt/src/csra/streaming_engine.py")
c = p.read_text()

# 1. Add imports after "import joblib"
old = "import numpy as np\nimport psutil\nimport joblib"
new = """import numpy as np
import psutil
import joblib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from csra.calibration import bbq_predict
from real_source_fusion import ScalarParticleFilter"""
c = c.replace(old, new)

# 2. Add _pf_obj to globals
c = c.replace(
    "_model_bundle = None\n_clf = None\n_config = None",
    "_model_bundle = None\n_clf = None\n_pf_obj = None\n_config = None"
)

# 3. Replace _apply_particle_filter function
old_pf = """def _apply_particle_filter(raw_risk: float) -> float:
    \"\"\"
    Apply particle filter smoothing.
    Tries multiple common APIs; falls back to exponential smoothing.
    \"\"\"
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

    return raw_risk"""

new_pf = """def _apply_particle_filter(raw_risk: float) -> float:
    \"\"\"Apply particle filter smoothing using ScalarParticleFilter.step().\"\"\"
    global _pf_obj
    if _pf_obj is None:
        return raw_risk
    return float(_pf_obj.step(raw_risk))"""

c = c.replace(old_pf, new_pf)

# 4. Replace _apply_bbq function
old_bbq = """def _apply_bbq(smoothed_risk: float) -> float:
    \"\"\"
    Apply Bayesian Binning into Quantiles (BBQ) calibration.
    Tries multiple common APIs.
    \"\"\"
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
    return smoothed_risk"""

new_bbq = """def _apply_bbq(smoothed_risk: float) -> float:
    \"\"\"Apply Bayesian Binning into Quantiles (BBQ) calibration.\"\"\"
    global _bbq
    if _bbq is None:
        return smoothed_risk
    calibrated = bbq_predict(_bbq, np.array([smoothed_risk]))[0]
    return max(0.0, min(1.0, float(calibrated)))"""

c = c.replace(old_bbq, new_bbq)

# 5. Update initialize() to create PF object from saved state
old_init = """    # Load all three components of the Brain
    _clf = _model_bundle.get("classifier")
    _pf = _model_bundle.get("particle_filter_state")
    _bbq = _model_bundle.get("bbq_model")
    feature_cols = _model_bundle["feature_cols"]

    if _clf is None:
        raise KeyError("Bundle missing 'classifier'")

    _logger.info("Model loaded: %s", type(_clf).__name__)
    _logger.info("Particle filter: %s", "present" if _pf is not None else "MISSING")
    _logger.info("BBQ model: %s", "present" if _bbq is not None else "MISSING")
    _logger.info("Feature columns: %d", len(feature_cols))"""

new_init = """    # Load all three components of the Brain
    _clf = _model_bundle.get("classifier")
    _pf = _model_bundle.get("particle_filter_state")
    _bbq = _model_bundle.get("bbq_model")
    feature_cols = _model_bundle["feature_cols"]

    if _clf is None:
        raise KeyError("Bundle missing 'classifier'")

    # Create particle filter object and restore state from frozen bundle
    global _pf_obj
    _pf_obj = ScalarParticleFilter(n_particles=1000, process_noise=0.1, seed=7)
    if _pf is not None:
        _pf_obj.particles = _pf["particles"]
        _pf_obj.weights = _pf["weights"]
        _logger.info("Restored particle filter state from frozen bundle.")
    else:
        _logger.info("No saved PF state found; starting fresh.")

    _logger.info("Model loaded: %s", type(_clf).__name__)
    _logger.info("BBQ model: %s", "present" if _bbq is not None else "MISSING")
    _logger.info("Feature columns: %d", len(feature_cols))"""

c = c.replace(old_init, new_init)

p.write_text(c)
print("PATCHED: calibration pipeline restored (classifier -> PF -> BBQ)")