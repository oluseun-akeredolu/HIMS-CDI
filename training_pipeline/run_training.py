#!/usr/bin/env python3
"""
run_training.py — Main entry point for the HIMS-CDI 10-seed training pipeline (v9.3).

Key changes from v9.2:
  - Uses shared governance_gates.evaluate_governance_gates() for gate logic
  - Uses bootstrap_ensemble.bootstrap_ensemble_ci() (stratified case-bootstrap, not Rubin's rules)
  - Fixed double-fit bug: measurement uses the already-fitted model
  - Bootstrap: 10,000 stratified case-bootstrap resamples on ensemble-averaged probabilities
  - GNN labeled as "GCN-style tabular baseline" in output
  - Transformer hidden_size=128 consistently (config and code agree)
  - Classification threshold: fixed 0.50 (documented everywhere)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    GNNModel, TransformerModel,
    EdgeAwareFusionModel, EdgeAwareFusionOptModel,
    HIMS_CDIMLRBase, HIMS_CDIMLRCalibrated,
    EqualWeightsModel, RuleBasedModel, ChecklistModel,
)
from governance_gates import evaluate_governance_gates
from bootstrap.bootstrap_ensemble import bootstrap_ensemble_ci
from measurement.workstation_measure import measure_inference

# Model configs — these are the ACTUAL values used by the code
MODELS_TO_TRAIN = [
    ('Equal_Weights',         EqualWeightsModel,        {}),
    ('Rule_Based',            RuleBasedModel,           {}),
    ('GCN_style_tabular',    GNNModel,                 {'hidden_units': 64, 'learning_rate': 1e-3}),
    ('Transformer',           TransformerModel,         {'hidden_size': 128, 'learning_rate': 3e-5}),
    ('Edge_Aware_Fusion',     EdgeAwareFusionModel,    {'compression_ratio': 0.5, 'learning_rate': 1e-3}),
    ('Edge_Aware_Fusion_opt', EdgeAwareFusionOptModel, {'compression_ratio': 0.5, 'learning_rate': 1e-3}),
    ('HIMS_CDI_base',         HIMS_CDIMLRBase,          {'C': 1.0}),
    ('HIMS_CDI_cal',          HIMS_CDIMLRCalibrated,   {'C': 1.0, 'platt_cv': 5}),
    ('Checklist',             ChecklistModel,           {}),
]

CLASSIFICATION_THRESHOLD = 0.50
N_BOOTSTRAP_RESAMPLES = 10000  # stratified case-bootstrap on ensemble probabilities
BOOTSTRAP_SEED = 42


def main():
    parser = argparse.ArgumentParser(description='HIMS-CDI 10-seed training pipeline (v9.3)')
    parser.add_argument('--archive-dir', required=True)
    parser.add_argument('--output', default='results')
    parser.add_argument('--config', default='configs/seeds.yaml')
    parser.add_argument('--bootstrap-samples', type=int, default=N_BOOTSTRAP_RESAMPLES,
                        help=f'Number of stratified case-bootstrap resamples (default {N_BOOTSTRAP_RESAMPLES})')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    seeds = config['seeds']
    n_bootstrap = args.bootstrap_samples

    train = pd.read_csv(os.path.join(args.archive_dir, 'synthetic_train.csv'))
    val = pd.read_csv(os.path.join(args.archive_dir, 'synthetic_val.csv'))
    test = pd.read_csv(os.path.join(args.archive_dir, 'synthetic_test.csv'))
    features = ['CDI', 'CCR', 'latency_ms', 'memory_mb']

    X_train, y_train = train[features].values, train['DEPLOY_SYNTH'].values
    X_val, y_val = val[features].values, val['DEPLOY_SYNTH'].values
    X_test, y_test = test[features].values, test['DEPLOY_SYNTH'].values

    print(f'=== HIMS-CDI Training Pipeline (v9.3) ===')
    print(f'Archive: {args.archive_dir}')
    print(f'Train/Val/Test: {len(train)} / {len(val)} / {len(test)}')
    print(f'Seeds: {seeds}')
    print(f'Bootstrap: {n_bootstrap} stratified case-bootstrap resamples on ensemble probabilities')
    print(f'Classification threshold: {CLASSIFICATION_THRESHOLD}')
    print(f'Models: {[m[0] for m in MODELS_TO_TRAIN]}')
    print()

    os.makedirs(args.output, exist_ok=True)

    all_predictions = []
    all_resource = []
    table7_rows = []

    for model_name, model_class, model_kwargs in MODELS_TO_TRAIN:
        print(f'\n--- Training {model_name} ---')

        per_seed_probs = []
        for seed in seeds:
            if args.verbose:
                print(f'  -> {model_name} (seed={seed})...')

            kwargs = dict(model_kwargs)
            kwargs['seed'] = seed
            try:
                model = model_class(**kwargs)
            except Exception as e:
                if args.verbose:
                    print(f'    SKIP: {e}')
                continue

            # Train (time it)
            train_start = time.perf_counter()
            try:
                if model_name in ('GCN_style_tabular', 'Transformer', 'Edge_Aware_Fusion', 'Edge_Aware_Fusion_opt'):
                    model.fit(X_train, y_train, X_val, y_val, verbose=False)
                else:
                    model.fit(X_train, y_train, X_val, y_val)
            except Exception as e:
                if args.verbose:
                    print(f'    SKIP (training failed): {e}')
                continue
            train_time = time.perf_counter() - train_start

            # Predict
            try:
                probs = model.predict_proba(X_test)
            except Exception as e:
                if args.verbose:
                    print(f'    SKIP (predict failed): {e}')
                continue
            p_pos = probs[:, 1] if probs.ndim == 2 else probs

            per_seed_probs.append(p_pos)

            # Save per-seed predictions
            all_predictions.append(pd.DataFrame({
                'seed': seed,
                'model': model_name,
                'test_case_id': np.arange(len(y_test)),
                'predicted_probability': p_pos,
                'true_label': y_test,
            }))

            # Measure resources (using the ALREADY-FITTED model — no double-fit)
            try:
                resource = measure_inference(model, X_test, model_name, seed)
                resource['training_time_s'] = train_time
                all_resource.append(resource)
            except Exception as e:
                if args.verbose:
                    print(f'    measurement failed: {e}')

        if not per_seed_probs:
            print(f'  No successful seeds for {model_name}')
            continue

        # Ensemble: average probabilities across seeds
        ensemble_prob = np.mean(per_seed_probs, axis=0)

        # Bootstrap CI (stratified case-bootstrap on ensemble probabilities)
        ci_result = bootstrap_ensemble_ci(
            y_test, ensemble_prob,
            n_resamples=n_bootstrap,
            seed=BOOTSTRAP_SEED,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        table7_rows.append({
            'model': model_name,
            'macro_f1': ci_result['macro_f1'],
            'macro_f1_ci_low': ci_result['macro_f1_ci_low'],
            'macro_f1_ci_high': ci_result['macro_f1_ci_high'],
            'auroc': ci_result['auroc'],
            'auroc_ci_low': ci_result['auroc_ci_low'],
            'auroc_ci_high': ci_result['auroc_ci_high'],
            'sensitivity': ci_result['sensitivity'],
            'specificity': ci_result['specificity'],
            'threshold': CLASSIFICATION_THRESHOLD,
            'n_bootstrap': n_bootstrap,
            'bootstrap_seed': BOOTSTRAP_SEED,
        })

        print(f'  AUROC={ci_result["auroc"]:.4f} [{ci_result["auroc_ci_low"]:.4f}, {ci_result["auroc_ci_high"]:.4f}]  '
              f'macroF1={ci_result["macro_f1"]:.4f}  sens={ci_result["sensitivity"]:.4f}  spec={ci_result["specificity"]:.4f}')

    # ---- Save per-seed predictions ----
    if all_predictions:
        preds_df = pd.concat(all_predictions, ignore_index=True)
        preds_path = os.path.join(args.output, 'per_seed_predictions.csv')
        preds_df.to_csv(preds_path, index=False)
        print(f'\nSaved: {preds_path} ({len(preds_df)} rows)')

    # ---- Save resource usage ----
    if all_resource:
        resource_df = pd.DataFrame(all_resource)
        resource_path = os.path.join(args.output, 'table5_resource_usage.csv')
        resource_df.to_csv(resource_path, index=False)
        print(f'Saved: {resource_path} ({len(resource_df)} rows)')

    # ---- Save bootstrap CI ----
    ci_path = os.path.join(args.output, 'bootstrap_ci.json')
    with open(ci_path, 'w') as f:
        json.dump({r['model']: {k: v for k, v in r.items() if k != 'model'} for r in table7_rows}, f, indent=2)
    print(f'Saved: {ci_path}')

    # ---- Save Table 7 (classification metrics only) ----
    table7 = pd.DataFrame(table7_rows).round(4)
    table7_path = os.path.join(args.output, 'table7_classification.csv')
    table7.to_csv(table7_path, index=False)
    print(f'Saved: {table7_path}')

    # ---- Save gate satisfaction (case-level, using shared gate function) ----
    gate_rows = []
    for i in range(len(test)):
        result = evaluate_governance_gates(
            test['CDI'].iloc[i],
            test['CCR'].iloc[i],
            test['latency_ms'].iloc[i],
            test['memory_mb'].iloc[i],
        )
        gate_rows.append({
            'test_case_id': i,
            'CDI': test['CDI'].iloc[i],
            'CCR': test['CCR'].iloc[i],
            'latency_ms': test['latency_ms'].iloc[i],
            'memory_mb': test['memory_mb'].iloc[i],
            'CDI_pass': result['CDI_pass'],
            'CCR_pass': result['CCR_pass'],
            'latency_pass': result['latency_pass'],
            'memory_pass': result['memory_pass'],
            'all_gates_pass': result['all_gates_pass'],
            'decision': result['decision'],
        })
    gate_df = pd.DataFrame(gate_rows)
    gate_path = os.path.join(args.output, 'gate_satisfaction.csv')
    gate_df.to_csv(gate_path, index=False)
    print(f'Saved: {gate_path} ({len(gate_df)} rows)')

    # ---- Print gate results ----
    print(f'\n=== Gate satisfaction (case-level, n={len(test)}) ===')
    print(f'  CDI >= 0.70:      {gate_df["CDI_pass"].sum()}/{len(test)} ({gate_df["CDI_pass"].mean()*100:.1f}%)')
    print(f'  CCR >= 0.80:      {gate_df["CCR_pass"].sum()}/{len(test)} ({gate_df["CCR_pass"].mean()*100:.1f}%)')
    print(f'  latency < 200 ms: {gate_df["latency_pass"].sum()}/{len(test)} ({gate_df["latency_pass"].mean()*100:.1f}%)')
    print(f'  memory < 100 MB:  {gate_df["memory_pass"].sum()}/{len(test)} ({gate_df["memory_pass"].mean()*100:.1f}%)')
    print(f'  ALL gates:        {gate_df["all_gates_pass"].sum()}/{len(test)} ({gate_df["all_gates_pass"].mean()*100:.1f}%)')
    all_mean = (test['CDI'].mean() >= 0.70 and test['CCR'].mean() >= 0.80 and
                test['latency_ms'].mean() < 200 and test['memory_mb'].mean() < 100)
    print(f'  ALL gates on mean: {"PASS" if all_mean else "FAIL — CCR gate fails"}')

    # ---- RQ1 check ----
    print(f'\n=== RQ1 ===')
    for r in table7_rows:
        if r['model'] == 'HIMS_CDI_cal':
            print(f'  HIMS_CDI_cal sensitivity: {r["sensitivity"]:.4f}')
            print(f'  RQ1 criterion: sensitivity >= 0.90')
            print(f'  Result: {"PASS" if r["sensitivity"] >= 0.90 else "FAIL — criterion NOT met"}')

    print(f'\n=== Pipeline complete (v9.3) ===')


if __name__ == '__main__':
    main()
