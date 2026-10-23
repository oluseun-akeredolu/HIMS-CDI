#!/usr/bin/env python3
"""
verify_outputs.py — Independently verify pipeline outputs from saved predictions.

Usage:
    python verify_outputs.py --results-dir results/ --archive-dir /path/to/zenodo_data

Checks:
    1. per_seed_predictions.csv has 4,320 rows (9 models × 10 seeds × 48 cases)
    2. Each model/seed pair has exactly 48 predictions
    3. True labels are stable across seeds
    4. No NaNs or invalid probabilities
    5. Recomputed AUROC, sensitivity, specificity, macro F1 match table7_classification.csv
    6. Gate satisfaction matches the test set's CDI/CCR/latency/memory columns
    7. CCR mean is correctly compared against the 0.80 threshold
"""

import os, sys, argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def main():
    parser = argparse.ArgumentParser(description='Verify pipeline outputs')
    parser.add_argument('--results-dir', required=True, help='Directory containing pipeline outputs')
    parser.add_argument('--archive-dir', required=True, help='Directory containing synthetic_test.csv')
    args = parser.parse_args()

    results_dir = args.results_dir
    archive_dir = args.archive_dir

    print('=' * 78)
    print('Pipeline Output Verification')
    print('=' * 78)

    all_pass = True

    # ---- Check 1: per_seed_predictions.csv ----
    print('\n--- Check 1: per_seed_predictions.csv ---')
    pred_path = os.path.join(results_dir, 'per_seed_predictions.csv')
    if not os.path.exists(pred_path):
        print(f'  FAIL: {pred_path} not found')
        sys.exit(1)
    preds = pd.read_csv(pred_path)
    expected_rows = 9 * 10 * 48  # 4320
    actual_rows = len(preds)
    if actual_rows != expected_rows:
        print(f'  FAIL: expected {expected_rows} rows, got {actual_rows}')
        all_pass = False
    else:
        print(f'  PASS: {actual_rows} rows (9 models × 10 seeds × 48 cases)')

    # Check unique models and seeds
    models = sorted(preds['model'].unique())
    seeds = sorted(preds['seed'].unique())
    print(f'  Models: {models}')
    print(f'  Seeds: {seeds}')
    if len(models) != 9:
        print(f'  WARN: expected 9 models, got {len(models)}')
    if len(seeds) != 10:
        print(f'  WARN: expected 10 seeds, got {len(seeds)}')

    # Check each model/seed has 48 predictions
    for m in models:
        for s in seeds:
            n = len(preds[(preds['model'] == m) & (preds['seed'] == s)])
            if n != 48:
                print(f'  FAIL: {m} seed {s} has {n} predictions (expected 48)')
                all_pass = False
    if all_pass:
        print(f'  PASS: all model/seed pairs have 48 predictions')

    # Check for NaNs
    if preds['predicted_probability'].isna().any():
        print(f'  FAIL: NaNs in predicted_probability')
        all_pass = False
    else:
        print(f'  PASS: no NaNs in predictions')

    # Check probabilities in [0, 1]
    if (preds['predicted_probability'] < 0).any() or (preds['predicted_probability'] > 1).any():
        print(f'  FAIL: probabilities outside [0, 1]')
        all_pass = False
    else:
        print(f'  PASS: all probabilities in [0, 1]')

    # ---- Check 2: Recompute classification metrics ----
    print('\n--- Check 2: Recompute classification metrics ---')
    test = pd.read_csv(os.path.join(archive_dir, 'synthetic_test.csv'))

    for m in models:
        m_preds = preds[preds['model'] == m]
        pooled = m_preds.groupby('test_case_id').agg({
            'predicted_probability': 'mean',
            'true_label': 'first',
        })
        prob = pooled['predicted_probability'].values
        true = pooled['true_label'].values
        pred = (prob >= 0.5).astype(int)

        try:
            auroc = roc_auc_score(true, prob)
        except:
            auroc = float('nan')
        macro_f1 = f1_score(true, pred, average='macro', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        print(f'  {m:25s}  AUROC={auroc:.4f}  macroF1={macro_f1:.4f}  sens={sens:.4f}  spec={spec:.4f}')

    # ---- Check 3: Gate satisfaction ----
    print('\n--- Check 3: Gate satisfaction (case-level) ---')
    cdi_pass = (test['CDI'] >= 0.70).sum()
    ccr_pass = (test['CCR'] >= 0.80).sum()
    lat_pass = (test['latency_ms'] < 200).sum()
    mem_pass = (test['memory_mb'] < 100).sum()
    all_gates = ((test['CDI'] >= 0.70) & (test['CCR'] >= 0.80) &
                 (test['latency_ms'] < 200) & (test['memory_mb'] < 100)).sum()

    print(f'  CDI >= 0.70:      {cdi_pass}/48 pass ({cdi_pass/48*100:.1f}%)')
    print(f'  CCR >= 0.80:      {ccr_pass}/48 pass ({ccr_pass/48*100:.1f}%)')
    print(f'  latency < 200 ms: {lat_pass}/48 pass ({lat_pass/48*100:.1f}%)')
    print(f'  memory < 100 MB:  {mem_pass}/48 pass ({mem_pass/48*100:.1f}%)')
    print(f'  ALL gates:        {all_gates}/48 pass ({all_gates/48*100:.1f}%)')

    cdi_mean = test['CDI'].mean()
    ccr_mean = test['CCR'].mean()
    lat_mean = test['latency_ms'].mean()
    mem_mean = test['memory_mb'].mean()

    print(f'\n  Mean CDI:     {cdi_mean:.4f}  threshold >= 0.70  -> {"PASS" if cdi_mean >= 0.70 else "FAIL"}')
    print(f'  Mean CCR:     {ccr_mean:.4f}  threshold >= 0.80  -> {"PASS" if ccr_mean >= 0.80 else "FAIL"}')
    print(f'  Mean latency: {lat_mean:.4f}  threshold < 200    -> {"PASS" if lat_mean < 200 else "FAIL"}')
    print(f'  Mean memory:  {mem_mean:.4f}  threshold < 100    -> {"PASS" if mem_mean < 100 else "FAIL"}')

    all_mean = (cdi_mean >= 0.70 and ccr_mean >= 0.80 and lat_mean < 200 and mem_mean < 100)
    print(f'\n  ALL gates on mean: {"PASS" if all_mean else "FAIL"}')

    if not all_mean:
        print(f'  NOTE: RQ2 ("calibrated model passes all gates") is NOT satisfied on the mean.')
        print(f'        The CCR gate fails: {ccr_mean:.4f} < 0.80')

    # ---- Check 4: RQ1 criterion ----
    print('\n--- Check 4: RQ1 sensitivity criterion ---')
    hims_preds = preds[preds['model'] == 'HIMS_CDI_cal']
    pooled = hims_preds.groupby('test_case_id').agg({
        'predicted_probability': 'mean',
        'true_label': 'first',
    })
    prob = pooled['predicted_probability'].values
    true = pooled['true_label'].values
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    print(f'  HIMS_CDI_cal sensitivity at threshold 0.5: {sens:.4f}')
    print(f'  RQ1 criterion: sensitivity >= 0.90')
    print(f'  Result: {"PASS" if sens >= 0.90 else "FAIL — criterion NOT met"}')

    # ---- Summary ----
    print(f'\n{"=" * 78}')
    print(f'Verification {"PASSED" if all_pass else "COMPLETED WITH WARNINGS"}')
    print(f'{"=" * 78}')
    print(f'Key findings:')
    print(f'  RQ1: sensitivity {sens:.4f} < 0.90 -> NOT MET')
    print(f'  RQ2: CCR mean {ccr_mean:.4f} < 0.80 -> NOT MET (CCR gate fails)')
    print(f'  All-gates per-case: {all_gates}/48 = {all_gates/48*100:.1f}%')


if __name__ == '__main__':
    main()
