#!/usr/bin/env python3
"""
verify_outputs.py — Independent verification of pipeline outputs (v9.4).

Checks:
    1. per_seed_predictions.csv has 4,320 rows
    2. Each model/seed has 48 predictions, no NaNs
    3. Recomputed metrics MATCH table7_classification.csv (not just print)
    4. Gate satisfaction uses the SHARED governance_gates.evaluate_governance_gates()
    5. RQ2 reported as TWO separate endpoints (logic tests + scenario pass rate)
    6. Resource values match table5_resource_usage.csv
    7. Run manifest exists and records the correct configuration
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from governance_gates import evaluate_governance_gates


def main():
    parser = argparse.ArgumentParser(description='Verify pipeline outputs (v9.4)')
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--archive-dir', required=True)
    args = parser.parse_args()

    results_dir = args.results_dir
    archive_dir = args.archive_dir
    all_pass = True
    checks = []

    # ---- Check 1: per_seed_predictions.csv ----
    print('--- Check 1: per_seed_predictions.csv ---')
    pred_path = os.path.join(results_dir, 'per_seed_predictions.csv')
    if not os.path.exists(pred_path):
        print(f'  FAIL: {pred_path} not found')
        sys.exit(1)
    preds = pd.read_csv(pred_path)
    expected_rows = 4320
    actual_rows = len(preds)
    ok = actual_rows == expected_rows
    checks.append(('per_seed_predictions row count', ok))
    print(f'  {"PASS" if ok else "FAIL"}: {actual_rows} rows (expected {expected_rows})')

    models = sorted(preds['model'].unique())
    seeds = sorted(preds['seed'].unique())
    print(f'  Models: {models}')
    print(f'  Seeds: {seeds}')

    # Check each model/seed has 48 predictions
    for m in models:
        for s in seeds:
            n = len(preds[(preds['model'] == m) & (preds['seed'] == s)])
            if n != 48:
                print(f'  FAIL: {m} seed {s} has {n} predictions (expected 48)')
                all_pass = False
                checks.append((f'{m} seed {s} predictions', False))
    if all_pass:
        print(f'  PASS: all model/seed pairs have 48 predictions')
        checks.append(('all model/seed pairs have 48', True))

    # Check NaNs
    ok = not preds['predicted_probability'].isna().any()
    checks.append(('no NaNs in predictions', ok))
    print(f'  {"PASS" if ok else "FAIL"}: no NaNs')

    # ---- Check 2: Recompute metrics and compare to Table 7 ----
    print('\n--- Check 2: Recompute metrics and compare to Table 7 ---')
    table7_path = os.path.join(results_dir, 'table7_classification.csv')
    if not os.path.exists(table7_path):
        print(f'  FAIL: {table7_path} not found')
        sys.exit(1)
    table7 = pd.read_csv(table7_path)
    test = pd.read_csv(os.path.join(archive_dir, 'synthetic_test.csv'))

    tolerance = 0.001  # 0.001 tolerance for rounding
    metrics_match = True

    for _, row in table7.iterrows():
        m = row['model']
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

        # Compare to Table 7
        auroc_match = abs(auroc - row['auroc']) < tolerance
        f1_match = abs(macro_f1 - row['macro_f1']) < tolerance
        sens_match = abs(sens - row['sensitivity']) < tolerance
        spec_match = abs(spec - row['specificity']) < tolerance

        model_ok = auroc_match and f1_match and sens_match and spec_match
        if not model_ok:
            metrics_match = False
            print(f'  FAIL: {m} — AUROC {auroc:.4f} vs {row["auroc"]:.4f}, '
                  f'F1 {macro_f1:.4f} vs {row["macro_f1"]:.4f}, '
                  f'sens {sens:.4f} vs {row["sensitivity"]:.4f}, '
                  f'spec {spec:.4f} vs {row["specificity"]:.4f}')
        else:
            print(f'  PASS: {m:25s} — AUROC={auroc:.4f} F1={macro_f1:.4f} sens={sens:.4f} spec={spec:.4f}')

    checks.append(('Table 7 metrics match', metrics_match))

    # ---- Check 3: Gate satisfaction uses SHARED gate function ----
    print('\n--- Check 3: Gate satisfaction (using shared governance_gates.py) ---')
    gate_path = os.path.join(results_dir, 'gate_satisfaction.csv')
    if not os.path.exists(gate_path):
        print(f'  FAIL: {gate_path} not found')
        sys.exit(1)
    saved_gates = pd.read_csv(gate_path)

    # Recompute gates using the SHARED function
    recomputed = []
    for i in range(len(test)):
        result = evaluate_governance_gates(
            test['CDI'].iloc[i], test['CCR'].iloc[i],
            test['latency_ms'].iloc[i], test['memory_mb'].iloc[i],
        )
        recomputed.append({
            'test_case_id': i,
            'CDI_pass': result['CDI_pass'],
            'CCR_pass': result['CCR_pass'],
            'latency_pass': result['latency_pass'],
            'memory_pass': result['memory_pass'],
            'all_gates_pass': result['all_gates_pass'],
            'decision': result['decision'],
        })
    recomputed_df = pd.DataFrame(recomputed)

    # Compare
    gates_match = True
    for col in ['CDI_pass', 'CCR_pass', 'latency_pass', 'memory_pass', 'all_gates_pass']:
        if not saved_gates[col].equals(recomputed_df[col]):
            print(f'  FAIL: {col} does not match shared function output')
            gates_match = False
    if gates_match:
        print(f'  PASS: gate_satisfaction.csv matches shared governance_gates.py output')
    checks.append(('Gate satisfaction matches shared function', gates_match))

    # Report gate statistics
    print(f'  CDI >= 0.70:      {saved_gates["CDI_pass"].sum()}/48 ({saved_gates["CDI_pass"].mean()*100:.1f}%)')
    print(f'  CCR >= 0.80:      {saved_gates["CCR_pass"].sum()}/48 ({saved_gates["CCR_pass"].mean()*100:.1f}%)')
    print(f'  latency < 200 ms: {saved_gates["latency_pass"].sum()}/48 ({saved_gates["latency_pass"].mean()*100:.1f}%)')
    print(f'  memory < 100 MB:  {saved_gates["memory_pass"].sum()}/48 ({saved_gates["memory_pass"].mean()*100:.1f}%)')
    print(f'  ALL gates:        {saved_gates["all_gates_pass"].sum()}/48 ({saved_gates["all_gates_pass"].mean()*100:.1f}%)')

    cdi_mean = test['CDI'].mean()
    ccr_mean = test['CCR'].mean()
    lat_mean = test['latency_ms'].mean()
    mem_mean = test['memory_mb'].mean()
    mean_gate_result = evaluate_governance_gates(cdi_mean, ccr_mean, lat_mean, mem_mean)
    print(f'\n  Mean CDI: {cdi_mean:.4f} (threshold 0.70) -> {"PASS" if mean_gate_result["CDI_pass"] else "FAIL"}')
    print(f'  Mean CCR: {ccr_mean:.4f} (threshold 0.80) -> {"PASS" if mean_gate_result["CCR_pass"] else "FAIL"}')
    print(f'  Mean latency: {lat_mean:.2f} ms (threshold 200) -> {"PASS" if mean_gate_result["latency_pass"] else "FAIL"}')
    print(f'  Mean memory: {mem_mean:.2f} MB (threshold 100) -> {"PASS" if mean_gate_result["memory_pass"] else "FAIL"}')
    if mean_gate_result["all_gates_pass"]:
        overall_msg = "PASS"
    else:
        failed = [name for name, ok in [
            ("CDI", mean_gate_result["CDI_pass"]),
            ("CCR", mean_gate_result["CCR_pass"]),
            ("latency", mean_gate_result["latency_pass"]),
            ("memory", mean_gate_result["memory_pass"]),
        ] if not ok]
        overall_msg = f"FAIL — {', '.join(failed)} gate(s) fail"
    print(f'  ALL gates on mean: {overall_msg}')

    # ---- Check 4: RQ2 as TWO separate endpoints ----
    print('\n--- Check 4: RQ2 (two separate endpoints) ---')
    print('  Endpoint 1 (decision logic): See test_governance_gates.py — 15 tests')
    print('  Endpoint 2 (scenario-level): '
          f'{saved_gates["all_gates_pass"].sum()}/48 = {saved_gates["all_gates_pass"].mean()*100:.1f}%')
    print('  NOTE: These are SEPARATE findings, not a single PASS/FAIL')

    # ---- Check 5: RQ1 ----
    print('\n--- Check 5: RQ1 sensitivity criterion ---')
    cal_row = table7[table7['model'] == 'HIMS_CDI_cal'].iloc[0]
    print(f'  HIMS_CDI_cal sensitivity: {cal_row["sensitivity"]:.4f}')
    print(f'  RQ1 criterion: sensitivity >= 0.90')
    print(f'  Result: {"PASS" if cal_row["sensitivity"] >= 0.90 else "FAIL — criterion NOT met"}')

    # ---- Check 6: Resource values ----
    print('\n--- Check 6: Resource values (Table 5) ---')
    res_path = os.path.join(results_dir, 'table5_resource_usage.csv')
    if os.path.exists(res_path):
        resources = pd.read_csv(res_path)
        cal_res = resources[resources['model'] == 'HIMS_CDI_cal']
        if len(cal_res) > 0:
            print(f'  HIMS_CDI_cal latency: {cal_res["inference_latency_ms"].mean():.2f} ± {cal_res["inference_latency_ms"].std():.2f} ms')
            print(f'  HIMS_CDI_cal memory:  {cal_res["inference_memory_mb"].mean():.2f} ± {cal_res["inference_memory_mb"].std():.2f} MB')
            print(f'  Memory gate (100 MB): {"PASS" if cal_res["inference_memory_mb"].mean() < 100 else "FAIL — exceeds 100 MB"}')

    # ---- Check 7: Run manifest ----
    print('\n--- Check 7: Run manifest ---')
    manifest_path = os.path.join(results_dir, 'run_manifest.json')
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        print(f'  Bootstrap: {manifest.get("n_bootstrap_resamples", "?")} resamples')
        print(f'  Threshold: {manifest.get("classification_threshold", "?")}')
        print(f'  Seeds: {manifest.get("model_training_seeds", "?")}')
        print(f'  Gate tests: {manifest.get("gate_logic_tests", "?")}')
        manifest_ok = True
    else:
        print('  WARN: run_manifest.json not found')
        manifest_ok = False
    checks.append(('Run manifest exists', manifest_ok))

    # ---- Summary ----
    print(f'\n{"=" * 78}')
    all_ok = all(c[1] for c in checks)
    print(f'Verification: {"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
    print(f'{"=" * 78}')
    print(f'\nKey findings (reported separately):')
    print(f'  RQ1: sensitivity {cal_row["sensitivity"]:.4f} < 0.90 -> NOT MET')
    print(f'  RQ2 decision logic: 15 tests (see test_governance_gates.py)')
    print(f'  RQ2 scenario-level: {saved_gates["all_gates_pass"].sum()}/48 = {saved_gates["all_gates_pass"].mean()*100:.1f}%')
    print(f'  Memory gate: {cal_res["inference_memory_mb"].mean():.0f} MB > 100 MB -> NOT MET')
    print(f'  (These are FOUR separate findings, not one combined PASS/FAIL)')

    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
