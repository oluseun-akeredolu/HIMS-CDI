"""
bootstrap_ensemble.py — Stratified case-bootstrap CIs for ensemble-averaged probabilities.

This replaces the old per-seed Rubin's rules approach. The primary method:
1. Average predicted probabilities across 10 seeds for each test case
2. Compute point estimates from the 48 averaged probabilities at threshold 0.50
3. Compute 95% percentile CIs using 10,000 stratified case-bootstrap resamples
   (preserving the 16 positive / 32 negative class counts per resample)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def bootstrap_ensemble_ci(y_true, y_prob, n_resamples=10000, seed=42, threshold=0.5):
    """Stratified case-bootstrap CIs for a fixed seed-averaged ensemble.

    Args:
        y_true: array of true labels (0/1), shape (n,)
        y_prob: array of predicted probabilities for positive class, shape (n,)
        n_resamples: number of bootstrap resamples (default 10,000)
        seed: RNG seed for reproducibility
        threshold: classification threshold (default 0.5)

    Returns:
        dict with keys: auroc (point estimate), auroc_ci_low, auroc_ci_high,
        macro_f1, macro_f1_ci_low, macro_f1_ci_high,
        sensitivity, specificity, n_resamples, seed, threshold
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Validate
    if len(y_true) != len(y_prob) or len(y_true) == 0:
        raise ValueError("Labels and probabilities must align.")
    if not np.isfinite(y_prob).all():
        raise ValueError("Probabilities must all be finite.")
    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must be in [0, 1].")

    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    if len(negative) == 0 or len(positive) == 0:
        raise ValueError("Both classes are required.")

    # Point estimates from the ensemble-averaged probabilities
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auroc_point = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc_point = float('nan')

    macro_f1_point = f1_score(y_true, y_pred, average='macro', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens_point = tp / (tp + fn) if (tp + fn) else 0.0
    spec_point = tn / (tn + fp) if (tn + fp) else 0.0

    # Stratified case-bootstrap
    rng = np.random.default_rng(seed)
    n_neg = len(negative)
    n_pos = len(positive)

    auroc_values = []
    f1_values = []

    for _ in range(n_resamples):
        idx = np.concatenate([
            rng.choice(negative, size=n_neg, replace=True),
            rng.choice(positive, size=n_pos, replace=True),
        ])

        # Shuffle so class order doesn't matter
        rng.shuffle(idx)

        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]

        # Check we have both classes
        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            auroc_values.append(roc_auc_score(y_true_boot, y_prob_boot))
        except ValueError:
            pass

        y_pred_boot = (y_prob_boot >= threshold).astype(int)
        if len(np.unique(y_pred_boot)) < 2 and len(np.unique(y_true_boot)) < 2:
            continue
        f1_values.append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))

    auroc_values = np.array(auroc_values)
    f1_values = np.array(f1_values)

    return {
        'auroc': float(auroc_point),
        'auroc_ci_low': float(np.percentile(auroc_values, 2.5)) if len(auroc_values) > 0 else float('nan'),
        'auroc_ci_high': float(np.percentile(auroc_values, 97.5)) if len(auroc_values) > 0 else float('nan'),
        'macro_f1': float(macro_f1_point),
        'macro_f1_ci_low': float(np.percentile(f1_values, 2.5)) if len(f1_values) > 0 else float('nan'),
        'macro_f1_ci_high': float(np.percentile(f1_values, 97.5)) if len(f1_values) > 0 else float('nan'),
        'sensitivity': float(sens_point),
        'specificity': float(spec_point),
        'n_resamples': n_resamples,
        'seed': seed,
        'threshold': threshold,
        'n_bootstrap_successful_auroc': int(len(auroc_values)),
        'n_bootstrap_successful_f1': int(len(f1_values)),
    }
