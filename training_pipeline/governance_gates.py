"""
governance_gates.py — Canonical HIMS-CDI non-compensatory governance gates (v9.6).

v9.6 changes: Added domain-range validation. CDI and CCR must be in [0, 1]. 
Latency and memory must be >= 0. Invalid inputs fail-closed (NOT_DEPLOY).
"""

import math

GATE_CDI_THRESHOLD = 0.70
GATE_CCR_THRESHOLD = 0.80
GATE_LATENCY_THRESHOLD = 200.0
GATE_MEMORY_THRESHOLD = 100.0

CDI_MIN, CDI_MAX = 0.0, 1.0
CCR_MIN, CCR_MAX = 0.0, 1.0
LATENCY_MIN = 0.0
MEMORY_MIN = 0.0

def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _in_domain(value, lo, hi=None):
    if hi is None:
        return value >= lo
    return lo <= value <= hi

def evaluate_governance_gates(cdi, ccr, latency_ms, memory_mb):
    cdi = _finite_number(cdi)
    ccr = _finite_number(ccr)
    latency_ms = _finite_number(latency_ms)
    memory_mb = _finite_number(memory_mb)

    domain_errors = []
    if cdi is None or not _in_domain(cdi, CDI_MIN, CDI_MAX):
        domain_errors.append("CDI_out_of_domain_or_invalid")
    if ccr is None or not _in_domain(ccr, CCR_MIN, CCR_MAX):
        domain_errors.append("CCR_out_of_domain_or_invalid")
    if latency_ms is None or not _in_domain(latency_ms, LATENCY_MIN, None):
        domain_errors.append("latency_out_of_domain_or_invalid")
    if memory_mb is None or not _in_domain(memory_mb, MEMORY_MIN, None):
        domain_errors.append("memory_out_of_domain_or_invalid")

    inputs_valid = len(domain_errors) == 0

    cdi_pass = cdi is not None and _in_domain(cdi, CDI_MIN, CDI_MAX) and cdi >= GATE_CDI_THRESHOLD
    ccr_pass = ccr is not None and _in_domain(ccr, CCR_MIN, CCR_MAX) and ccr >= GATE_CCR_THRESHOLD
    latency_pass = latency_ms is not None and _in_domain(latency_ms, LATENCY_MIN, None) and latency_ms < GATE_LATENCY_THRESHOLD
    memory_pass = memory_mb is not None and _in_domain(memory_mb, MEMORY_MIN, None) and memory_mb < GATE_MEMORY_THRESHOLD

    all_gates_pass = inputs_valid and cdi_pass and ccr_pass and latency_pass and memory_pass

    return {
        "inputs_valid": inputs_valid,
        "CDI_pass": cdi_pass,
        "CCR_pass": ccr_pass,
        "latency_pass": latency_pass,
        "memory_pass": memory_pass,
        "all_gates_pass": all_gates_pass,
        "decision": "DEPLOY" if all_gates_pass else "NOT_DEPLOY",
        "domain_errors": domain_errors,
    }
