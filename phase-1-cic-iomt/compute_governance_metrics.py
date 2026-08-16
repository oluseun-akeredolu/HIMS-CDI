#!/usr/bin/env python3
"""
HIMS-CDI Governance Metrics Calculator
Run this AFTER the 30-day pilot completes to compute CDI and CCR.

CRITICAL: This script has NO DEFAULTS for any metric.
You must supply the ACTUAL observed values from the pilot.
It is impossible to accidentally generate a fake "PASS" report.
"""

import argparse
import json
import pathlib
import numpy as np


def compute_cdi(auroc, f1, ccr, explainability_score, edge_readiness_score, latency_score):
    """
    CDI = weighted average of six normalized rubric scores.
    Weights from manuscript Appendix G.
    """
    s_acc = min(1.0, max(0.0, auroc))
    s_f1 = min(1.0, max(0.0, f1))
    s_comp = min(1.0, max(0.0, ccr))
    s_expl = min(1.0, max(0.0, explainability_score / 5.0))
    s_edge = min(1.0, max(0.0, edge_readiness_score))
    s_lat = min(1.0, max(0.0, latency_score))

    cdi = (
        0.15 * s_acc +
        0.15 * s_f1 +
        0.20 * s_comp +
        0.15 * s_expl +
        0.20 * s_edge +
        0.15 * s_lat
    )
    return round(cdi, 3)


def compute_ccr(mapped_clauses, total_clauses=27):
    """CCR = proportion of regulatory clauses with evidence artefacts."""
    return round(mapped_clauses / total_clauses, 3)


def analyze_audit_log(audit_path: pathlib.Path):
    """Read audit.jsonl and compute aggregate statistics."""
    latencies = []
    memory_deltas = []
    latency_passes = 0
    memory_passes = 0
    total_scored = 0
    total_rejected = 0

    with open(audit_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("scored"):
                total_scored += 1
                latencies.append(record["latency_ms"])
                memory_deltas.append(record["memory_delta_mb"])
                if record.get("gate_latency_pass"):
                    latency_passes += 1
                if record.get("gate_memory_pass"):
                    memory_passes += 1
            else:
                total_rejected += 1

    if total_scored == 0:
        raise ValueError("No scored events found in audit log.")

    stats = {
        "total_events": total_scored + total_rejected,
        "total_scored": total_scored,
        "total_rejected": total_rejected,
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "max_latency_ms": round(float(max(latencies)), 2),
        "max_memory_delta_mb": round(float(max(memory_deltas)), 4),
        "latency_gate_pass_rate": round(latency_passes / total_scored, 3),
        "memory_gate_pass_rate": round(memory_passes / total_scored, 3),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute HIMS-CDI governance metrics post-pilot. NO DEFAULTS -- supply real evidence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (after pilot ends):
  python compute_governance_metrics.py ^
    --audit logs/audit.jsonl ^
    --auroc 0.87 ^
    --f1 0.81 ^
    --explainability 4.7 ^
    --mapped-clauses 25
"""
    )
    parser.add_argument("--audit", required=True, help="Path to audit.jsonl")
    parser.add_argument("--auroc", type=float, required=True, help="Observed AUROC from ground-truth validation")
    parser.add_argument("--f1", type=float, required=True, help="Observed F1 from ground-truth validation")
    parser.add_argument("--explainability", type=float, required=True, help="Analyst survey mean score (out of 5)")
    parser.add_argument("--mapped-clauses", type=int, required=True, help="Number of regulatory clauses mapped (out of 27)")
    args = parser.parse_args()

    stats = analyze_audit_log(pathlib.Path(args.audit))

    edge_score = stats["memory_gate_pass_rate"]
    latency_score = stats["latency_gate_pass_rate"]

    ccr = compute_ccr(args.mapped_clauses)
    cdi = compute_cdi(
        auroc=args.auroc,
        f1=args.f1,
        ccr=ccr,
        explainability_score=args.explainability,
        edge_readiness_score=edge_score,
        latency_score=latency_score,
    )

    print("\n" + "=" * 60)
    print("HIMS-CDI GOVERNANCE METRICS (30-Day Pilot Summary)")
    print("=" * 60)
    print(f"Total events processed:  {stats['total_events']:,}")
    print(f"Events scored:           {stats['total_scored']:,}")
    print(f"Events rejected:         {stats['total_rejected']:,}")
    print(f"P95 latency:             {stats['p95_latency_ms']} ms")
    print(f"P99 latency:             {stats['p99_latency_ms']} ms")
    print(f"Max latency:             {stats['max_latency_ms']} ms")
    print(f"Max memory delta:        {stats['max_memory_delta_mb']} MB")
    print(f"Latency gate pass rate:  {stats['latency_gate_pass_rate']}")
    print(f"Memory gate pass rate:   {stats['memory_gate_pass_rate']}")
    print("-" * 60)
    print(f"CCR (Clause Coverage):   {ccr}  (threshold: >= 0.80)")
    print(f"CDI (Clinical Deploy.):  {cdi}  (threshold: >= 0.70)")
    print("-" * 60)
    print(f"Gate 1 (CDI >= 0.70):     {'PASS' if cdi >= 0.70 else 'FAIL'}")
    print(f"Gate 2 (CCR >= 0.80):     {'PASS' if ccr >= 0.80 else 'FAIL'}")
    print(f"Gate 3 (P95 Latency <200):{'PASS' if stats['p95_latency_ms'] < 200 else 'FAIL'}")
    print(f"Gate 4 (Max Memory <100): {'PASS' if stats['max_memory_delta_mb'] < 100 else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()