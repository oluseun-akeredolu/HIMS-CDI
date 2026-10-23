"""
test_governance_gates.py — Automated tests for the canonical governance gate function.

Tests:
  1. All gates pass -> DEPLOY
  2. CDI fails -> NOT_DEPLOY
  3. CCR fails -> NOT_DEPLOY
  4. Latency fails -> NOT_DEPLOY
  5. Memory fails -> NOT_DEPLOY
  6. CDI boundary (=0.70) -> passes (>=)
  7. CCR boundary (=0.80) -> passes (>=)
  8. Latency boundary (=200) -> FAILS (strict <)
  9. Memory boundary (=100) -> FAILS (strict <)
  10. CDI just below boundary (0.699) -> FAILS
  11. CCR just below boundary (0.799) -> FAILS
  12. Latency just above boundary (201) -> FAILS
  13. Memory just above boundary (101) -> FAILS
  14. Invalid input (None) -> NOT_DEPLOY (fail closed)
  15. Invalid input (NaN) -> NOT_DEPLOY (fail closed)

Run: python -m pytest -q training_pipeline/tests/test_governance_gates.py
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from governance_gates import evaluate_governance_gates


def test_all_gates_pass():
    """All four gates pass -> DEPLOY."""
    result = evaluate_governance_gates(0.85, 0.90, 150, 80)
    assert result["all_gates_pass"] is True
    assert result["decision"] == "DEPLOY"
    assert result["CDI_pass"] is True
    assert result["CCR_pass"] is True
    assert result["latency_pass"] is True
    assert result["memory_pass"] is True


def test_cdi_fails():
    """CDI below threshold -> NOT_DEPLOY."""
    result = evaluate_governance_gates(0.65, 0.90, 150, 80)
    assert result["all_gates_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"
    assert result["CDI_pass"] is False
    assert result["CCR_pass"] is True


def test_ccr_fails():
    """CCR below threshold -> NOT_DEPLOY."""
    result = evaluate_governance_gates(0.85, 0.75, 150, 80)
    assert result["all_gates_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"
    assert result["CCR_pass"] is False


def test_latency_fails():
    """Latency above threshold -> NOT_DEPLOY."""
    result = evaluate_governance_gates(0.85, 0.90, 250, 80)
    assert result["all_gates_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"
    assert result["latency_pass"] is False


def test_memory_fails():
    """Memory above threshold -> NOT_DEPLOY."""
    result = evaluate_governance_gates(0.85, 0.90, 150, 150)
    assert result["all_gates_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"
    assert result["memory_pass"] is False


def test_cdi_boundary_passes():
    """CDI exactly 0.70 -> passes (>= threshold)."""
    result = evaluate_governance_gates(0.70, 0.85, 150, 80)
    assert result["CDI_pass"] is True
    assert result["all_gates_pass"] is True


def test_ccr_boundary_passes():
    """CCR exactly 0.80 -> passes (>= threshold)."""
    result = evaluate_governance_gates(0.85, 0.80, 150, 80)
    assert result["CCR_pass"] is True
    assert result["all_gates_pass"] is True


def test_latency_boundary_fails():
    """Latency exactly 200 ms -> FAILS (strict < threshold)."""
    result = evaluate_governance_gates(0.85, 0.90, 200, 80)
    assert result["latency_pass"] is False
    assert result["all_gates_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"


def test_memory_boundary_fails():
    """Memory exactly 100 MB -> FAILS (strict < threshold)."""
    result = evaluate_governance_gates(0.85, 0.90, 150, 100)
    assert result["memory_pass"] is False
    assert result["all_gates_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"


def test_cdi_just_below_fails():
    """CDI 0.699 -> FAILS (just below 0.70)."""
    result = evaluate_governance_gates(0.699, 0.90, 150, 80)
    assert result["CDI_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"


def test_ccr_just_below_fails():
    """CCR 0.799 -> FAILS (just below 0.80)."""
    result = evaluate_governance_gates(0.85, 0.799, 150, 80)
    assert result["CCR_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"


def test_latency_just_above_fails():
    """Latency 201 ms -> FAILS (just above 200)."""
    result = evaluate_governance_gates(0.85, 0.90, 201, 80)
    assert result["latency_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"


def test_memory_just_above_fails():
    """Memory 101 MB -> FAILS (just above 100)."""
    result = evaluate_governance_gates(0.85, 0.90, 150, 101)
    assert result["memory_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"


def test_invalid_input_none():
    """None input -> NOT_DEPLOY (fail closed)."""
    result = evaluate_governance_gates(None, 0.90, 150, 80)
    assert result["inputs_valid"] is False
    assert result["all_gates_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"


def test_invalid_input_nan():
    """NaN input -> NOT_DEPLOY (fail closed)."""
    result = evaluate_governance_gates(float('nan'), 0.90, 150, 80)
    assert result["inputs_valid"] is False
    assert result["all_gates_pass"] is False
    assert result["decision"] == "NOT_DEPLOY"
