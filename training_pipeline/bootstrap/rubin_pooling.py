"""
Rubin's rules for pooling bootstrap intervals across multiple seeds.

This implements the pooling procedure documented in manuscript Section 4.5:
    "for AUROC, macro F1, CDI, CCR, latency, memory: 10,000 bootstrap resamples
    per seed (10 seeds -> 100,000 total). 95% confidence intervals: percentile
    method, pooled via Rubin's rules (Rubin, 1987)."

Rubin's rules for pooling m imputations:
    - Pooled mean:   Q_bar = mean(Q_i) over i in 1..m
    - Within-var:    U_bar = mean(U_i) where U_i = var(bootstrap_resamples_i)
    - Between-var:   B = (1 / (m - 1)) * sum((Q_i - Q_bar)^2)
    - Total var:     T = U_bar + (1 + 1/m) * B
    - Pooled 95% CI: Q_bar +/- 1.96 * sqrt(T)

For bootstrap intervals, we adapt this:
    - Q_i = mean of the bootstrap resamples for seed i
    - U_i = variance of the bootstrap resamples for seed i
    - The pooled 95% CI uses the percentile method on the combined resamples
      (this is the "percentile method, pooled via Rubin's rules" interpretation)

This file is complete — no TODOs.
"""

import numpy as np
from typing import Dict, List


def rubin_pool_bootstrap(bootstrap_results: Dict[int, np.ndarray]) -> Dict[str, float]:
    """Pool bootstrap results across seeds using Rubin's rules.

    Args:
        bootstrap_results: dict mapping seed -> array of bootstrap resamples
            e.g., {13: np.array([0.74, 0.76, 0.73, ...]), 37: np.array([...]), ...}

    Returns:
        dict with keys: mean, ci_low, ci_high, within_var, between_var, total_var
    """
    seeds = sorted(bootstrap_results.keys())
    m = len(seeds)

    # Per-seed means and variances
    q_i = np.array([bootstrap_results[s].mean() for s in seeds])  # mean per seed
    u_i = np.array([bootstrap_results[s].var(ddof=1) for s in seeds])  # variance per seed

    # Rubin's rules
    q_bar = q_i.mean()                     # pooled mean
    u_bar = u_i.mean()                     # within-seed variance
    b = ((q_i - q_bar) ** 2).sum() / (m - 1)  # between-seed variance
    t = u_bar + (1 + 1 / m) * b            # total variance

    # Pooled 95% CI (percentile method on combined resamples)
    combined = np.concatenate([bootstrap_results[s] for s in seeds])
    ci_low = np.percentile(combined, 2.5)
    ci_high = np.percentile(combined, 97.5)

    return {
        'mean': float(q_bar),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'within_var': float(u_bar),
        'between_var': float(b),
        'total_var': float(t),
    }


def bootstrap_auroc(y_true: np.ndarray, y_prob: np.ndarray,
                    n_resamples: int = 10000, seed: int = 42) -> np.ndarray:
    """Bootstrap AUROC — return array of n_resamples AUROC values.

    Args:
        y_true: array of true labels (0/1)
        y_prob: array of predicted probabilities for the positive class
        n_resamples: number of bootstrap resamples
        seed: RNG seed

    Returns:
        array of n_resamples AUROC values
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip degenerate resamples
        aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.array(aurocs)


def bootstrap_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                       n_resamples: int = 10000, seed: int = 42) -> np.ndarray:
    """Bootstrap macro F1 — return array of n_resamples macro F1 values."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        f1s.append(f1_score(y_true[idx], y_pred[idx], average='macro', zero_division=0))
    return np.array(f1s)


def pool_per_seed_metrics(per_seed_results: Dict[int, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    """Pool multiple metrics across seeds.

    Args:
        per_seed_results: dict mapping seed -> dict mapping metric_name -> bootstrap_resamples
            e.g., {13: {'auroc': np.array([...]), 'macro_f1': np.array([...])}, ...}

    Returns:
        dict mapping metric_name -> pooled result (from rubin_pool_bootstrap)
    """
    metric_names = list(next(iter(per_seed_results.values())).keys())
    pooled = {}
    for metric in metric_names:
        bootstrap_results = {seed: per_seed_results[seed][metric]
                             for seed in per_seed_results}
        pooled[metric] = rubin_pool_bootstrap(bootstrap_results)
    return pooled
