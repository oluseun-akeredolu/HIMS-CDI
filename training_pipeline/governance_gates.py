"""
governance_gates.py — Canonical HIMS-CDI non-compensatory governance gates.

This is the SINGLE source of truth for gate evaluation. The production pipeline,
the synthetic case evaluation, and the automated test suite all call this function.
Do NOT duplicate the threshold comparisons in other files.
"""

import math


def _finite_number(value):
    """Return a finite float, or None for invalid/missing input."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# Gate thresholds (manuscript Section 3.2.3)
GATE_CDI_THRESHOLD = 0.70   # CDI >= 0.70  (passes at boundary)
GATE_CCR_THRESHOLD = 0.80   # CCR >= 0.80  (passes at boundary)
GATE_LATENCY_THRESHOLD = 200.0  # latency < 200   (FAILS at boundary — strict <)
GATE_MEMORY_THRESHOLD = 100.0   # memory < 100    (FAILS at boundary — strict <)


def evaluate_governance_gates(cdi, ccr, latency_ms, memory_mb):
    """Evaluate the four non-compensatory governance gates.

    Gate logic:
        Gate 1: CDI >= 0.70  (>= boundary: passes at exactly 0.70)
        Gate 2: CCR >= 0.80  (>= boundary: passes at exactly 0.80)
        Gate 3a: latency < 200 ms  (strict <: FAILS at exactly 200)
        Gate 3b: memory < 100 MB   (strict <: FAILS at exactly 100)

    All gates use AND logic: any single failure -> NOT_DEPLOY.
    Invalid/missing inputs -> NOT_DEPLOY (fail closed).

    Args:
        cdi: Clinical Deployment Index (float)
        ccr: Clause Coverage Rate (float)
        latency_ms: End-to-end latency in milliseconds (float)
        memory_mb: Memory consumption in megabytes (float)

    Returns:
        dict with keys: inputs_valid, CDI_pass, CCR_pass, latency_pass,
        memory_pass, all_gates_pass, decision
    """
    cdi = _finite_number(cdi)
    ccr = _finite_number(ccr)
    latency_ms = _finite_number(latency_ms)
    memory_mb = _finite_number(memory_mb)

    cdi_pass = cdi is not None and cdi >= GATE_CDI_THRESHOLD
    ccr_pass = ccr is not None and ccr >= GATE_CCR_THRESHOLD
    latency_pass = latency_ms is not None and latency_ms < GATE_LATENCY_THRESHOLD
    memory_pass = memory_mb is not None and memory_mb < GATE_MEMORY_THRESHOLD

    inputs_valid = all(v is not None for v in (cdi, ccr, latency_ms, memory_mb))
    all_gates_pass = inputs_valid and cdi_pass and ccr_pass and latency_pass and memory_pass

    return {
        "inputs_valid": inputs_valid,
        "CDI_pass": cdi_pass,
        "CCR_pass": ccr_pass,
        "latency_pass": latency_pass,
        "memory_pass": memory_pass,
        "all_gates_pass": all_gates_pass,
        "decision": "DEPLOY" if all_gates_pass else "NOT_DEPLOY",
    }
