#!/usr/bin/env python3
"""
weight_sensitivity.py — Weight-sensitivity analysis for the HIMS-CDI governance gates.

Varies the CDI weights by ±0.05 and ±0.10 from the baseline, renormalises to sum to 1,
recomputes CDI for all 240 cases, and reports how many cases change gate decisions.

Also draws 100 random weight vectors from a Dirichlet distribution centred on the baseline.
"""

import numpy as np
import pandas as pd
import argparse
import json

BASELINE_WEIGHTS = np.array([0.15, 0.15, 0.20, 0.15, 0.20, 0.15])
DOMAIN_NAMES = ['Accuracy', 'F1_Score', 'Compliance', 'Explainability', 'Edge_Readiness', 'Latency']
GATE_CDI = 0.70
GATE_CCR = 0.80
GATE_LAT = 200.0
GATE_MEM = 100.0


def compute_cdi(weights, domain_scores):
    """Compute CDI = weighted sum of domain scores."""
    return np.dot(domain_scores, weights)


def compute_gates(cdi, ccr, latency, memory):
    """Compute gate satisfaction for each case."""
    return {
        'cdi_pass': cdi >= GATE_CDI,
        'ccr_pass': ccr >= GATE_CCR,
        'lat_pass': latency < GATE_LAT,
        'mem_pass': memory < GATE_MEM,
        'all_pass': (cdi >= GATE_CDI) & (ccr >= GATE_CCR) &
                    (latency < GATE_LAT) & (memory < GATE_MEM),
    }


def renormalise(weights):
    """Renormalise weights to sum to 1, clipping negatives to 0."""
    w = np.maximum(weights, 0)
    return w / w.sum()


def main():
    parser = argparse.ArgumentParser(description='Weight-sensitivity analysis')
    parser.add_argument('--archive-dir', required=True,
                        help='Directory containing synthetic_hims_cdi_dataset_labelled.csv')
    parser.add_argument('--output', default='weight_sensitivity_results.json',
                        help='Output JSON file')
    parser.add_argument('--n-dirichlet', type=int, default=100,
                        help='Number of Dirichlet random weight vectors')
    args = parser.parse_args()

    # Load the full 240-case dataset
    df = pd.read_csv(f'{args.archive_dir}/synthetic_hims_cdi_dataset_labelled.csv')
    n = len(df)
    print(f'Loaded {n} cases from {args.archive_dir}')

    # Extract the 20 indicators and compute domain scores
    # Domain indices from the generator
    domain_indices = {
        'Accuracy': [0, 1, 2, 3],
        'F1_Score': [4, 5, 6, 7],
        'Edge_Readiness': [8, 9],
        'Latency': [10, 11],
        'Compliance': [12, 13, 14, 15],
        'Explainability': [16, 17, 18, 19],
    }
    indicator_cols = [c for c in df.columns if '_ind' in c]
    indicators = df[indicator_cols].values

    domain_scores = np.zeros((n, 6))
    for i, (domain, indices) in enumerate(domain_indices.items()):
        domain_scores[:, i] = indicators[:, indices].mean(axis=1)

    # Baseline CDI
    baseline_cdi = compute_cdi(BASELINE_WEIGHTS, domain_scores)
    baseline_gates = compute_gates(baseline_cdi, df['CCR'].values,
                                   df['latency_ms'].values, df['memory_mb'].values)
    baseline_all_pass = baseline_gates['all_pass']
    baseline_deploy = baseline_all_pass.sum()
    baseline_label = df['DEPLOY_SYNTH'].values

    print(f'\nBaseline weights: {BASELINE_WEIGHTS}')
    print(f'Baseline CDI mean: {baseline_cdi.mean():.4f} ± {baseline_cdi.std():.4f}')
    print(f'Baseline all-gates pass: {baseline_deploy}/{n} ({baseline_deploy/n*100:.1f}%)')
    print(f'Baseline DEPLOY_SYNTH (clean labels): {baseline_label.sum()}/{n} ({baseline_label.sum()/n*100:.1f}%)')

    results = {
        'baseline_weights': BASELINE_WEIGHTS.tolist(),
        'baseline_cdi_mean': float(baseline_cdi.mean()),
        'baseline_cdi_std': float(baseline_cdi.std()),
        'baseline_all_gates_pass': int(baseline_deploy),
        'baseline_deploy_rate': float(baseline_deploy / n),
        'one_at_a_time': {},
        'dirichlet': {},
    }

    # ---- One-at-a-time ±0.05 and ±0.10 ----
    print(f'\n=== One-at-a-time weight perturbation ===')
    for delta in [0.05, 0.10]:
        print(f'\n  Delta = ±{delta}:')
        results['one_at_a_time'][f'pm{delta:.2f}'] = {}

        for i, domain in enumerate(DOMAIN_NAMES):
            for sign in [+1, -1]:
                w = BASELINE_WEIGHTS.copy()
                w[i] += sign * delta
                w = renormalise(w)

                cdi = compute_cdi(w, domain_scores)
                gates = compute_gates(cdi, df['CCR'].values,
                                       df['latency_ms'].values, df['memory_mb'].values)
                all_pass = gates['all_pass']
                n_changed = (all_pass != baseline_all_pass).sum()
                n_to_deploy = (all_pass & ~baseline_all_pass).sum()
                n_to_notdeploy = (~all_pass & baseline_all_pass).sum()

                label_change = (all_pass.astype(int) != baseline_label).sum()

                key = f'{domain}_{"+" if sign > 0 else "-"}{delta}'
                results['one_at_a_time'][f'pm{delta:.2f}'][key] = {
                    'weights': w.tolist(),
                    'cdi_mean': float(cdi.mean()),
                    'cdi_std': float(cdi.std()),
                    'all_gates_pass': int(all_pass.sum()),
                    'cases_changed': int(n_changed),
                    'to_deploy': int(n_to_deploy),
                    'to_not_deploy': int(n_to_notdeploy),
                    'label_disagreement': int(label_change),
                }

                if n_changed > 0:
                    print(f'    {key:30s}  changed={n_changed:3d}  to_deploy={n_to_deploy:3d}  '
                          f'to_not_deploy={n_to_notdeploy:3d}  CDI_mean={cdi.mean():.4f}')

    # ---- Dirichlet random draws ----
    print(f'\n=== Dirichlet random weight vectors (n={args.n_dirichlet}) ===')
    rng = np.random.default_rng(42)
    # Concentration parameter: lower = more spread; alpha = baseline / concentration_scale
    concentration_scale = 0.05  # small -> more spread from baseline
    alpha = BASELINE_WEIGHTS / concentration_scale

    dirichlet_results = []
    for draw in range(args.n_dirichlet):
        w = rng.dirichlet(alpha)
        cdi = compute_cdi(w, domain_scores)
        gates = compute_gates(cdi, df['CCR'].values,
                               df['latency_ms'].values, df['memory_mb'].values)
        all_pass = gates['all_pass']
        n_changed = (all_pass != baseline_all_pass).sum()

        dirichlet_results.append({
            'draw': draw,
            'weights': w.tolist(),
            'cdi_mean': float(cdi.mean()),
            'all_gates_pass': int(all_pass.sum()),
            'cases_changed': int(n_changed),
        })

    n_changed_list = [d['cases_changed'] for d in dirichlet_results]
    n_pass_list = [d['all_gates_pass'] for d in dirichlet_results]
    cdi_means = [d['cdi_mean'] for d in dirichlet_results]

    print(f'  Cases changed (mean ± std): {np.mean(n_changed_list):.1f} ± {np.std(n_changed_list):.1f}')
    print(f'  Cases changed (range): {min(n_changed_list)} to {max(n_changed_list)}')
    print(f'  All-gates pass (mean ± std): {np.mean(n_pass_list):.1f} ± {np.std(n_pass_list):.1f}')
    print(f'  CDI mean (range): {min(cdi_means):.4f} to {max(cdi_means):.4f}')
    print(f'  Max cases changed in any draw: {max(n_changed_list)} ({max(n_changed_list)/n*100:.1f}%)')

    results['dirichlet'] = {
        'n_draws': args.n_dirichlet,
        'concentration_scale': concentration_scale,
        'seed': 42,
        'mean_cases_changed': float(np.mean(n_changed_list)),
        'std_cases_changed': float(np.std(n_changed_list)),
        'max_cases_changed': int(max(n_changed_list)),
        'min_cases_changed': int(min(n_changed_list)),
        'mean_all_gates_pass': float(np.mean(n_pass_list)),
        'std_all_gates_pass': float(np.std(n_pass_list)),
        'mean_cdi_mean': float(np.mean(cdi_means)),
        'std_cdi_mean': float(np.std(cdi_means)),
        'min_cdi_mean': float(min(cdi_means)),
        'max_cdi_mean': float(max(cdi_means)),
        'draws': dirichlet_results,
    }

    # ---- Summary ----
    print(f'\n=== Weight sensitivity summary ===')
    print(f'Baseline: {baseline_deploy}/{n} cases pass all gates ({baseline_deploy/n*100:.1f}%)')
    print(f'One-at-a-time ±0.05: max cases changed = {max(results["one_at_a_time"]["pm0.05"][k]["cases_changed"] for k in results["one_at_a_time"]["pm0.05"])}')
    print(f'One-at-a-time ±0.10: max cases changed = {max(results["one_at_a_time"]["pm0.10"][k]["cases_changed"] for k in results["one_at_a_time"]["pm0.10"])}')
    print(f'Dirichlet (n={args.n_dirichlet}): max cases changed = {max(n_changed_list)} ({max(n_changed_list)/n*100:.1f}%)')
    print(f'  -> The gate decisions are {"SENSITIVE" if max(max(n_changed_list), max(results["one_at_a_time"]["pm0.05"][k]["cases_changed"] for k in results["one_at_a_time"]["pm0.05"]), max(results["one_at_a_time"]["pm0.10"][k]["cases_changed"] for k in results["one_at_a_time"]["pm0.10"])) > 0 else "INSENSITIVE"} to weight perturbations.')

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
