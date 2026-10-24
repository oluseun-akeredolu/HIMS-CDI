"""test_governance_gates.py — Automated tests for the canonical governance gate function (v9.6)."""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from governance_gates import evaluate_governance_gates

def test_all_gates_pass():
    result = evaluate_governance_gates(0.85, 0.90, 150, 80)
    assert result["all_gates_pass"] is True

def test_cdi_fails():
    result = evaluate_governance_gates(0.65, 0.90, 150, 80)
    assert result["decision"] == "NOT_DEPLOY"

def test_ccr_fails():
    result = evaluate_governance_gates(0.85, 0.75, 150, 80)
    assert result["decision"] == "NOT_DEPLOY"

def test_latency_fails():
    result = evaluate_governance_gates(0.85, 0.90, 250, 80)
    assert result["decision"] == "NOT_DEPLOY"

def test_memory_fails():
    result = evaluate_governance_gates(0.85, 0.90, 150, 150)
    assert result["decision"] == "NOT_DEPLOY"

def test_cdi_boundary_passes():
    result = evaluate_governance_gates(0.70, 0.85, 150, 80)
    assert result["CDI_pass"] is True

def test_ccr_boundary_passes():
    result = evaluate_governance_gates(0.85, 0.80, 150, 80)
    assert result["CCR_pass"] is True

def test_latency_boundary_fails():
    result = evaluate_governance_gates(0.85, 0.90, 200, 80)
    assert result["latency_pass"] is False

def test_memory_boundary_fails():
    result = evaluate_governance_gates(0.85, 0.90, 150, 100)
    assert result["memory_pass"] is False

def test_cdi_just_below_fails():
    result = evaluate_governance_gates(0.699, 0.90, 150, 80)
    assert result["CDI_pass"] is False

def test_ccr_just_below_fails():
    result = evaluate_governance_gates(0.85, 0.799, 150, 80)
    assert result["CCR_pass"] is False

def test_latency_just_above_fails():
    result = evaluate_governance_gates(0.85, 0.90, 201, 80)
    assert result["latency_pass"] is False

def test_memory_just_above_fails():
    result = evaluate_governance_gates(0.85, 0.90, 150, 101)
    assert result["memory_pass"] is False

def test_invalid_input_none():
    result = evaluate_governance_gates(None, 0.90, 150, 80)
    assert result["decision"] == "NOT_DEPLOY"

def test_invalid_input_nan():
    result = evaluate_governance_gates(float('nan'), 0.90, 150, 80)
    assert result["decision"] == "NOT_DEPLOY"

def test_cdi_above_one_fails():
    result = evaluate_governance_gates(1.5, 0.90, 150, 80)
    assert "CDI_out_of_domain_or_invalid" in result["domain_errors"]

def test_ccr_above_one_fails():
    result = evaluate_governance_gates(0.85, 1.5, 150, 80)
    assert "CCR_out_of_domain_or_invalid" in result["domain_errors"]

def test_cdi_below_zero_fails():
    result = evaluate_governance_gates(-0.10, 0.90, 150, 80)
    assert "CDI_out_of_domain_or_invalid" in result["domain_errors"]

def test_ccr_below_zero_fails():
    result = evaluate_governance_gates(0.85, -0.10, 150, 80)
    assert "CCR_out_of_domain_or_invalid" in result["domain_errors"]

def test_latency_negative_fails():
    result = evaluate_governance_gates(0.85, 0.90, -50.0, 80)
    assert "latency_out_of_domain_or_invalid" in result["domain_errors"]

def test_memory_negative_fails():
    result = evaluate_governance_gates(0.85, 0.90, 150, -20.0)
    assert "memory_out_of_domain_or_invalid" in result["domain_errors"]

def test_cdi_nonnumeric_string_fails():
    result = evaluate_governance_gates("abc", 0.90, 150, 80)
    assert "CDI_out_of_domain_or_invalid" in result["domain_errors"]

def test_ccr_nonnumeric_string_fails():
    result = evaluate_governance_gates(0.85, "abc", 150, 80)
    assert "CCR_out_of_domain_or_invalid" in result["domain_errors"]

def test_latency_positive_infinity_fails():
    result = evaluate_governance_gates(0.85, 0.90, float('inf'), 80)
    assert "latency_out_of_domain_or_invalid" in result["domain_errors"]

def test_memory_negative_infinity_fails():
    result = evaluate_governance_gates(0.85, 0.90, 150, float('-inf'))
    assert "memory_out_of_domain_or_invalid" in result["domain_errors"]

def test_cdi_zero_in_domain_but_below_threshold():
    result = evaluate_governance_gates(0.0, 0.90, 150, 80)
    assert result["inputs_valid"] is True

def test_ccr_zero_in_domain_but_below_threshold():
    result = evaluate_governance_gates(0.85, 0.0, 150, 80)
    assert result["inputs_valid"] is True

def test_latency_zero_in_domain_passes():
    result = evaluate_governance_gates(0.85, 0.90, 0.0, 80)
    assert result["latency_pass"] is True

def test_memory_zero_in_domain_passes():
    result = evaluate_governance_gates(0.85, 0.90, 150, 0.0)
    assert result["memory_pass"] is True

def test_cdi_one_in_domain_passes():
    result = evaluate_governance_gates(1.0, 1.0, 150, 80)
    assert result["CDI_pass"] is True

def test_combined_invalid_inputs():
    result = evaluate_governance_gates(None, float('nan'), float('inf'), -1.0)
    assert "CDI_out_of_domain_or_invalid" in result["domain_errors"]
    assert "CCR_out_of_domain_or_invalid" in result["domain_errors"]
    assert "latency_out_of_domain_or_invalid" in result["domain_errors"]
    assert "memory_out_of_domain_or_invalid" in result["domain_errors"]
