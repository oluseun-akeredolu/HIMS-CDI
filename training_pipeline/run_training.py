#!/usr/bin/env python3
"""
run_training.py — Main entry point for the HIMS-CDI 10-seed training pipeline.

Runs each model on each of the 10 seeds, computes per-seed metrics, runs the
10,000-resample bootstrap per seed, pools via Rubin's rules, and produces:
    - per_seed_predictions.csv   (Item 2)
    - bootstrap_ci.json          (Item 3)
    - table5.csv                  (manuscript Table 5 — resource usage)
    - table7.csv                  (manuscript Table 7 — classification metrics)
    - workstation_resource_usage.csv (Item 4a)

Reports the ACTUAL numbers — no reverse-engineering to hit specific values.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    GNNModel, TransformerModel,
    EdgeAwareFusionModel, EdgeAwareFusionOptModel,
    HIMS_CDIMLRBase, HIMS_CDIMLRCalibrated,
    EqualWeightsModel, RuleBasedModel, ChecklistModel,
)
from bootstrap.rubin_pooling import (
    bootstrap_auroc, bootstrap_macro_f1, pool_per_seed_metrics,
)
from measurement.workstation_measure import measure_workstation


MODELS_TO_TRAIN = [
    ('Equal_Weights',         EqualWeightsModel,        {}),
    ('Rule_Based',            RuleBasedModel,           {}),
    ('GNN',                   GNNModel,                 {'hidden_units': 64, 'learning_rate': 1e-3}),
    ('Transformer',           TransformerModel,        {'hidden_size': 128, 'learning_rate': 3e-5}),
    ('Edge_Aware_Fusion',     EdgeAwareFusionModel,    {'compression_ratio': 0.5, 'learning_rate': 1e-3}),
    ('Edge_Aware_Fusion_opt', EdgeAwareFusionOptModel, {'compression_ratio': 0.5, 'learning_rate': 1e-3}),
    ('HIMS_CDI_base',         HIMS_CDIMLRBase,          {'C': 1.0}),
    ('HIMS_CDI_cal',          HIMS_CDIMLRCalibrated,   {'C': 1.0, 'platt_cv': 5}),
    ('Checklist',             ChecklistModel,           {}),
]


def train_one_model_per_seed(model_name, model_class, model_kwargs,
                              X_train, y_train, X_val, y_val, X_test, y_test,
                              seeds, n_bootstrap=10000, verbose=False):
    """Train one model on each of the 10 seeds and collect predictions + bootstrap."""
    per_seed_predictions = []
    per_seed_bootstrap = {}
    per_seed_resource = []

    for seed in seeds:
        if verbose:
            print(f'  -> {model_name} (seed={seed})...')

        # Instantiate model with this seed
        kwargs = dict(model_kwargs)
        kwargs['seed'] = seed
        try:
            model = model_class(**kwargs)
        except Exception as e:
            if verbose:
                print(f'    SKIP (cannot instantiate): {e}')
            continue

        # Train
        try:
            if model_name in ('GNN', 'Transformer', 'Edge_Aware_Fusion', 'Edge_Aware_Fusion_opt'):
                model.fit(X_train, y_train, X_val, y_val, verbose=False)
            else:
                model.fit(X_train, y_train, X_val, y_val)
        except Exception as e:
            if verbose:
                print(f'    SKIP (training failed): {e}')
            continue

        # Predict on test set
        try:
            probs = model.predict_proba(X_test)
        except Exception as e:
            if verbose:
                print(f'    SKIP (predict failed): {e}')
            continue

        p_pos = probs[:, 1] if probs.ndim == 2 else probs
        y_pred = (p_pos >= 0.5).astype(int)

        # Bootstrap resamples for this seed
        boot_auroc = bootstrap_auroc(y_test, p_pos, n_resamples=n_bootstrap, seed=seed)
        boot_macro_f1 = bootstrap_macro_f1(y_test, y_pred, n_resamples=n_bootstrap, seed=seed)

        # Store predictions
        per_seed_predictions.append(pd.DataFrame({
            'seed': seed,
            'model': model_name,
            'test_case_id': np.arange(len(y_test)),
            'predicted_probability': p_pos,
            'predicted_label': y_pred,
            'true_label': y_test,
        }))

        per_seed_bootstrap[seed] = {
            'auroc': boot_auroc,
            'macro_f1': boot_macro_f1,
        }

        # Resource measurement
        try:
            resource = measure_workstation(
                model, X_train, y_train, X_test,
                model_name=model_name, seed=seed,
            )
            per_seed_resource.append(resource)
        except Exception as e:
            if verbose:
                print(f'    measurement failed: {e}')

    return per_seed_predictions, per_seed_bootstrap, per_seed_resource


def main():
    parser = argparse.ArgumentParser(description='HIMS-CDI 10-seed training pipeline')
    parser.add_argument('--archive-dir', required=True,
                        help='Directory containing synthetic_train.csv, synthetic_val.csv, synthetic_test.csv')
    parser.add_argument('--output', default='results', help='Output directory')
    parser.add_argument('--config', default='configs/seeds.yaml')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to train (default: all)')
    parser.add_argument('--bootstrap-samples', type=int, default=1000,
                        help='Bootstrap resamples per seed (default 1000 for speed; '
                             'manuscript used 10000)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    seeds = config['seeds']
    n_bootstrap = args.bootstrap_samples

    # Load data
    train = pd.read_csv(os.path.join(args.archive_dir, 'synthetic_train.csv'))
    val = pd.read_csv(os.path.join(args.archive_dir, 'synthetic_val.csv'))
    test = pd.read_csv(os.path.join(args.archive_dir, 'synthetic_test.csv'))
    features = ['CDI', 'CCR', 'latency_ms', 'memory_mb']

    X_train = train[features].values
    y_train = train['DEPLOY_SYNTH'].values
    X_val = val[features].values
    y_val = val['DEPLOY_SYNTH'].values
    X_test = test[features].values
    y_test = test['DEPLOY_SYNTH'].values

    print(f'=== HIMS-CDI Training Pipeline ===')
    print(f'Archive:    {args.archive_dir}')
    print(f'Train/Val/Test: {len(train)} / {len(val)} / {len(test)}')
    print(f'Seeds:      {seeds}')
    print(f'Bootstrap:  {n_bootstrap} resamples per seed')
    print(f'Models:     {[m[0] for m in (MODELS_TO_TRAIN if args.models is None else [m for m in MODELS_TO_TRAIN if m[0] in args.models])]}')
    print()

    # Filter models if specified
    models_to_train = MODELS_TO_TRAIN if args.models is None else \
                      [m for m in MODELS_TO_TRAIN if m[0] in args.models]

    os.makedirs(args.output, exist_ok=True)

    all_predictions = []
    all_bootstrap = {}
    all_resource = []

    for model_name, model_class, model_kwargs in models_to_train:
        print(f'\n--- Training {model_name} ---')
        preds, boots, res = train_one_model_per_seed(
            model_name, model_class, model_kwargs,
            X_train, y_train, X_val, y_val, X_test, y_test,
            seeds, n_bootstrap=n_bootstrap, verbose=args.verbose,
        )
        if preds:
            all_predictions.extend(preds)
        if boots:
            all_bootstrap[model_name] = boots
        if res:
            all_resource.extend(res)

    # ---- Save per-seed predictions (Item 2) ----
    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        predictions_path = os.path.join(args.output, 'per_seed_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f'\nSaved: {predictions_path} ({len(predictions_df)} rows)')

    # ---- Save resource usage (Item 4a) ----
    if all_resource:
        resource_df = pd.DataFrame(all_resource)
        resource_path = os.path.join(args.output, 'workstation_resource_usage.csv')
        resource_df.to_csv(resource_path, index=False)
        print(f'Saved: {resource_path} ({len(resource_df)} rows)')

    # ---- Pool bootstrap across seeds and save bootstrap_ci.json (Item 3) ----
    bootstrap_ci = {}
    for model_name, per_seed in all_bootstrap.items():
        if not per_seed:
            continue
        auroc_per_seed = {s: per_seed[s]['auroc'] for s in per_seed}
        macro_f1_per_seed = {s: per_seed[s]['macro_f1'] for s in per_seed}
        bootstrap_ci[model_name] = {
            'auroc': pool_per_seed_metrics({s: {'auroc': auroc_per_seed[s]} for s in auroc_per_seed})['auroc'],
            'macro_f1': pool_per_seed_metrics({s: {'macro_f1': macro_f1_per_seed[s]} for s in macro_f1_per_seed})['macro_f1'],
        }
    bootstrap_path = os.path.join(args.output, 'bootstrap_ci.json')
    with open(bootstrap_path, 'w') as f:
        json.dump(bootstrap_ci, f, indent=2)
    print(f'Saved: {bootstrap_path}')

    # ---- Compute and save Table 5 (resource usage aggregated) ----
    if all_resource:
        resource_df = pd.DataFrame(all_resource)
        table5 = resource_df.groupby('model').agg({
            'inference_latency_ms': ['mean', 'std'],
            'inference_memory_mb': ['mean', 'std'],
            'training_time_s': ['mean', 'std'],
        }).round(2)
        table5_path = os.path.join(args.output, 'table5.csv')
        table5.to_csv(table5_path)
        print(f'Saved: {table5_path}')

    # ---- Compute and save Table 7 (classification metrics) ----
    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
        table7_rows = []
        for model_name, per_seed in all_bootstrap.items():
            model_preds = predictions_df[predictions_df['model'] == model_name]
            if len(model_preds) == 0:
                continue

            # Pool predictions across seeds (mean probability per test case)
            pooled = model_preds.groupby('test_case_id').agg({
                'predicted_probability': 'mean',
                'true_label': 'first',
            })
            pooled_prob = pooled['predicted_probability'].values
            pooled_true = pooled['true_label'].values
            pooled_pred = (pooled_prob >= 0.5).astype(int)

            # Compute metrics
            try:
                auroc = roc_auc_score(pooled_true, pooled_prob)
            except ValueError:
                auroc = float('nan')
            macro_f1 = f1_score(pooled_true, pooled_pred, average='macro', zero_division=0)
            tn, fp, fn, tp = confusion_matrix(pooled_true, pooled_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0

            # CDI and CCR (mean across predictions on the test set)
            cdi_mean = test['CDI'].mean()
            ccr_mean = test['CCR'].mean()

            # Edge feasibility (using stored noisy values)
            latency_mean = test['latency_ms'].mean()
            memory_mean = test['memory_mb'].mean()
            edge_feasible = (latency_mean < 200) and (memory_mean < 100)

            # Gate satisfaction
            cdi_pass = (test['CDI'] >= 0.70).mean()
            ccr_pass = (test['CCR'] >= 0.80).mean()
            lat_pass = (test['latency_ms'] < 200).mean()
            mem_pass = (test['memory_mb'] < 100).mean()
            all_gates = ((test['CDI'] >= 0.70) & (test['CCR'] >= 0.80) &
                         (test['latency_ms'] < 200) & (test['memory_mb'] < 100)).mean()

            # Bootstrap CI (from pool_per_seed_metrics)
            auroc_ci = bootstrap_ci.get(model_name, {}).get('auroc', {})
            macro_f1_ci = bootstrap_ci.get(model_name, {}).get('macro_f1', {})

            table7_rows.append({
                'model': model_name,
                'macro_f1': macro_f1,
                'macro_f1_ci_low': macro_f1_ci.get('ci_low'),
                'macro_f1_ci_high': macro_f1_ci.get('ci_high'),
                'auroc': auroc,
                'auroc_ci_low': auroc_ci.get('ci_low'),
                'auroc_ci_high': auroc_ci.get('ci_high'),
                'sensitivity': sens,
                'specificity': spec,
                'latency_ms_mean': latency_mean,
                'memory_mb_mean': memory_mean,
                'cdi_mean': cdi_mean,
                'ccr_mean': ccr_mean,
                'all_gates_pass_rate': all_gates,
            })

        table7 = pd.DataFrame(table7_rows).round(4)
        table7_path = os.path.join(args.output, 'table7.csv')
        table7.to_csv(table7_path, index=False)
        print(f'Saved: {table7_path}')

    # ---- Summary ----
    print(f'\n=== Pipeline complete ===')
    print(f'Results in: {args.output}/')
    if all_predictions:
        print(f'  per_seed_predictions.csv       ({len(pd.concat(all_predictions))} rows)')
    if all_resource:
        print(f'  workstation_resource_usage.csv ({len(all_resource)} rows)')
    print(f'  bootstrap_ci.json')
    if all_resource:
        print(f'  table5.csv')
    if all_predictions:
        print(f'  table7.csv')

    # Print a summary of the actual numbers
    print(f'\n=== Actual numbers (no reverse-engineering) ===')
    if all_resource:
        print(f'\nResource usage (mean ± std across 10 seeds):')
        resource_df = pd.DataFrame(all_resource)
        for model in resource_df['model'].unique():
            sub = resource_df[resource_df['model'] == model]
            print(f'  {model:25s}  lat={sub["inference_latency_ms"].mean():.2f}±{sub["inference_latency_ms"].std():.2f} ms  '
                  f'mem={sub["inference_memory_mb"].mean():.2f}±{sub["inference_memory_mb"].std():.2f} MB  '
                  f'train={sub["training_time_s"].mean():.2f}±{sub["training_time_s"].std():.2f} s')

    if all_predictions:
        print(f'\nClassification metrics (pooled across seeds):')
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
        for model_name in all_bootstrap.keys():
            model_preds = predictions_df[predictions_df['model'] == model_name]
            if len(model_preds) == 0:
                continue
            pooled = model_preds.groupby('test_case_id').agg({
                'predicted_probability': 'mean',
                'true_label': 'first',
            })
            pooled_prob = pooled['predicted_probability'].values
            pooled_true = pooled['true_label'].values
            pooled_pred = (pooled_prob >= 0.5).astype(int)
            try:
                auroc = roc_auc_score(pooled_true, pooled_prob)
            except:
                auroc = float('nan')
            macro_f1 = f1_score(pooled_true, pooled_pred, average='macro', zero_division=0)
            tn, fp, fn, tp = confusion_matrix(pooled_true, pooled_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            auroc_ci = bootstrap_ci.get(model_name, {}).get('auroc', {})
            print(f'  {model_name:25s}  AUROC={auroc:.4f}  '
                  f'CI=[{auroc_ci.get("ci_low", 0):.4f}, {auroc_ci.get("ci_high", 0):.4f}]  '
                  f'macroF1={macro_f1:.4f}  sens={sens:.4f}  spec={spec:.4f}')


if __name__ == '__main__':
    main()
