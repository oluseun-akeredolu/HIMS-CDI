"""
Platt scaling with ECE computation — provided complete (no TODOs).

This module provides:
    - apply_platt_scaling(base_model, X_train, y_train, cv=5): returns a calibrated model
    - compute_ece(y_true, y_prob, n_bins=10): computes Expected Calibration Error

Manuscript Section 4.5 reports: "Platt scaling reduced expected calibration error
(ECE) from 0.12 to 0.03."

This module is referenced by models/mlr_platt.py (HIMS_CDIMLRCalibrated class).
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def apply_platt_scaling(base_model, X_train, y_train, cv=5):
    """Apply Platt scaling (sigmoid calibration) to a base model.

    Args:
        base_model: a fitted sklearn-compatible classifier with predict_proba()
        X_train: training features
        y_train: training labels (0/1)
        cv: number of cross-validation folds for calibration

    Returns:
        A CalibratedClassifierCV instance (fitted) with method='sigmoid'.
    """
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method='sigmoid',
        cv=cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE = sum over bins of (bin_size/n_total) * |bin_accuracy - bin_confidence|

    Args:
        y_true: true labels (0/1), shape (n,)
        y_prob: predicted probability of positive class, shape (n,)
        n_bins: number of confidence bins

    Returns:
        ECE (float in [0, 1])
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    confidence = np.maximum(y_prob, 1 - y_prob)  # confidence in the predicted class
    accuracy = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1])
        else:
            mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracy[mask].mean()
        bin_conf = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def verify_calibration(base_model, X_train, y_train, X_test, y_test, cv=5, n_bins=10):
    """Verify that Platt scaling reduces ECE from 0.12 to 0.03 (manuscript Section 4.5).

    Returns a dict with pre/post ECE values.
    """
    # Pre-calibration ECE (on the base model's test predictions)
    base_proba = base_model.predict_proba(X_test)[:, 1]
    pre_ece = compute_ece(y_test, base_proba, n_bins=n_bins)

    # Apply Platt scaling
    calibrated = apply_platt_scaling(base_model, X_train, y_train, cv=cv)

    # Post-calibration ECE
    cal_proba = calibrated.predict_proba(X_test)[:, 1]
    post_ece = compute_ece(y_test, cal_proba, n_bins=n_bins)

    return {
        'pre_ece': pre_ece,
        'post_ece': post_ece,
        'ece_reduction': pre_ece - post_ece,
        'manuscript_pre_ece': 0.12,
        'manuscript_post_ece': 0.03,
        'matches_manuscript': abs(pre_ece - 0.12) < 0.05 and abs(post_ece - 0.03) < 0.03,
    }
