#!/usr/bin/env python3
"""
threshold_sensitivity.py — Threshold-sensitivity analysis for the HIMS-CDI governance gates.

Varies the CDI, CCR, latency, and memory thresholds and reports how many cases
change gate decisions.
"""

import numpy as np
import pandas as pd
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Threshold-sensitivity analysis')
    parser.add_argument('--archive-dir', required=True,
                        help='Directory containing synthetic_hims_cdi_dataset_labelled.csv')
    parser.add_argument('--output', default='threshold_sensitivity_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    df = pd.read_csv(f'{args.archive_dir}/synthetic_hims_cdi_dataset_labelled.csv')
    n = len(df)
    print(f'Loaded {n} cases from {args.archive_dir}')

    # Baseline thresholds
    baseline = {
        'cdi': 0.70,
        'ccr': 0.80,
        'latency': 200.0,
        'memory': 100.0,
    }

    # Compute baseline gate decisions
    baseline_all = ((df['CDI'] >= 0.70) & (df['CCR'] >= 0.80) &
                    (df['latency_ms'] < 200) & (df['memory_mb'] < 100))
    baseline_pass = int(baseline_all.sum())
    baseline_labels = df['DEPLOY_SYNTH'].values

    print(f'\nBaseline thresholds: CDI>=0.70, CCR>=0.80, lat<200, mem<100')
    print(f'Baseline all-gates pass: {baseline_pass}/{n} ({baseline_pass/n*100:.1f}%)')
    print(f'Baseline DEPLOY_SYNTH labels: {baseline_labels.sum()}/{n} ({baseline_labels.sum()/n*100:.1f}%)')

    results = {
        'baseline': {
            'thresholds': baseline,
            'all_gates_pass': baseline_pass,
            'deploy_rate': float(baseline_pass / n),
        },
        'cdi_sweep': {},
        'ccr_sweep': {},
        'latency_sweep': {},
        'memory_sweep': {},
    }

    # ---- CDI threshold sweep ----
    print(f'\n=== CDI threshold sweep ===')
    for t in np.arange(0.50, 0.91, 0.05):
        all_pass = ((df['CDI'] >= t) & (df['CCR'] >= 0.80) &
                    (df['latency_ms'] < 200) & (df['memory_mb'] < 100))
        n_pass = int(all_pass.sum())
        n_changed = int((all_pass != baseline_all).sum())
        n_pass_rate = n_pass / n * 100
        print(f'  CDI >= {t:.2f}: pass={n_pass:3d} ({n_pass_rate:.1f}%)  changed={n_changed}')
        results['cdi_sweep'][f'{t:.2f}'] = {
            'threshold': float(t),
            'all_gates_pass': n_pass,
            'pass_rate': float(n_pass / n),
            'cases_changed': n_changed,
        }

    # ---- CCR threshold sweep ----
    print(f'\n=== CCR threshold sweep ===')
    for t in np.arange(0.60, 0.91, 0.05):
        all_pass = ((df['CDI'] >= 0.70) & (df['CCR'] >= t) &
                    (df['latency_ms'] < 200) & (df['memory_mb'] < 100))
        n_pass = int(all_pass.sum())
        n_changed = int((all_pass != baseline_all).sum())
        n_pass_rate = n_pass / n * 100
        print(f'  CCR >= {t:.2f}: pass={n_pass:3d} ({n_pass_rate:.1f}%)  changed={n_changed}')
        results['ccr_sweep'][f'{t:.2f}'] = {
            'threshold': float(t),
            'all_gates_pass': n_pass,
            'pass_rate': float(n_pass / n),
            'cases_changed': n_changed,
        }

    # ---- Latency threshold sweep ----
    print(f'\n=== Latency threshold sweep ===')
    for t in [100, 150, 200, 250, 300, 400, 500]:
        all_pass = ((df['CDI'] >= 0.70) & (df['CCR'] >= 0.80) &
                    (df['latency_ms'] < t) & (df['memory_mb'] < 100))
        n_pass = int(all_pass.sum())
        n_changed = int((all_pass != baseline_all).sum())
        n_pass_rate = n_pass / n * 100
        print(f'  latency < {t:3d} ms: pass={n_pass:3d} ({n_pass_rate:.1f}%)  changed={n_changed}')
        results['latency_sweep'][str(t)] = {
            'threshold': float(t),
            'all_gates_pass': n_pass,
            'pass_rate': float(n_pass / n),
            'cases_changed': n_changed,
        }

    # ---- Memory threshold sweep ----
    print(f'\n=== Memory threshold sweep ===')
    for t in [50, 75, 100, 150, 200, 300]:
        all_pass = ((df['CDI'] >= 0.70) & (df['CCR'] >= 0.80) &
                    (df['latency_ms'] < 200) & (df['memory_mb'] < t))
        n_pass = int(all_pass.sum())
        n_changed = int((all_pass != baseline_all).sum())
        n_pass_rate = n_pass / n * 100
        print(f'  memory < {t:3d} MB: pass={n_pass:3d} ({n_pass_rate:.1f}%)  changed={n_changed}')
        results['memory_sweep'][str(t)] = {
            'threshold': float(t),
            'all_gates_pass': n_pass,
            'pass_rate': float(n_pass / n),
            'cases_changed': n_changed,
        }

    # ---- Summary ----
    print(f'\n=== Threshold sensitivity summary ===')
    # Find the most sensitive threshold
    max_cdi_change = max(v['cases_changed'] for v in results['cdi_sweep'].values())
    max_ccr_change = max(v['cases_changed'] for v in results['ccr_sweep'].values())
    max_lat_change = max(v['cases_changed'] for v in results['latency_sweep'].values())
    max_mem_change = max(v['cases_changed'] for v in results['memory_sweep'].values())

    print(f'CDI sweep: max cases changed = {max_cdi_change}')
    print(f'CCR sweep: max cases changed = {max_ccr_change}')
    print(f'Latency sweep: max cases changed = {max_lat_change}')
    print(f'Memory sweep: max cases changed = {max_mem_change}')

    # Key finding: what happens at CCR = 0.75 (close to the observed mean of 0.741)?
    ccr_075 = results['ccr_sweep']['0.75']
    print(f'\nKey: at CCR threshold 0.75 (close to observed mean 0.741):')
    print(f'  All-gates pass: {ccr_075["all_gates_pass"]}/{n} ({ccr_075["pass_rate"]*100:.1f}%)')

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
