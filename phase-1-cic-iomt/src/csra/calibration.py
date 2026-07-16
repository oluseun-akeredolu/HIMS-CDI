"""Calibration methods: BBQ, ECE, bootstrap AUROC CI."""
import numpy as np
from sklearn.metrics import roc_auc_score

def bbq_fit(y_prob, y_true, n_bins=15, beta_alpha=2.0, beta_beta=2.0):
    bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bins[-1] = 1.0
    model = {"bins": bins, "calibration": {}}
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            n_pos = (y_true[mask]).sum()
            n_total = mask.sum()
            smoothed = (n_pos + beta_alpha - 1) / (n_total + beta_alpha + beta_beta - 2)
            model["calibration"][i] = smoothed
        else:
            model["calibration"][i] = np.nan
    return model

def bbq_predict(model, y_prob):
    bins = model["bins"]
    calibrated = np.zeros_like(y_prob)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_prob >= lo) & (y_prob < hi)
        if i in model["calibration"] and not np.isnan(model["calibration"][i]):
            calibrated[mask] = model["calibration"][i]
        else:
            calibrated[mask] = (lo + hi) / 2
    return calibrated

def expected_calibration_error(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        empirical_freq = y_true[mask].mean()
        avg_confidence = y_prob[mask].mean()
        ece += (mask.sum() / total) * abs(empirical_freq - avg_confidence)
    return ece

def bootstrap_auroc_ci(y_true, y_prob, n_boot=1000):
    aurocs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) > 1:
            aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aurocs = np.array(aurocs)
    return aurocs.mean(), np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5), len(aurocs)
