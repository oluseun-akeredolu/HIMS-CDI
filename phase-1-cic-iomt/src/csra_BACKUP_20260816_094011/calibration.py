"""Calibration methods: BBQ, ECE, bootstrap AUROC CI.

FIXES APPLIED (see accompanying CHANGELOG / verify_fixes.py for evidence):

1. Floor-collapse bug in bbq_fit/bbq_predict: the lowest bin edge was left
   as np.percentile(y_prob, 0) -- i.e. whatever TUNE's minimum happened to
   be -- rather than forced to 0.0. Any TEST point below that floor fell
   outside every bin's mask and silently kept the np.zeros_like() default
   of exactly 0.0, regardless of its actual value. Fixed by forcing
   bins[0] = 0.0 alongside the existing bins[-1] = 1.0.

2. Top-edge exclusion in bbq_fit / bbq_predict: masks used
   (y_prob >= lo) & (y_prob < hi) for every bin including the last, so
   y_prob == 1.0 matched no bin. Fixed by making the last bin's upper
   bound inclusive.

3. Same top-edge exclusion in expected_calibration_error: fixed the same
   way, and the denominator now reflects the true total sample count
   used (unchanged -- it always did -- but now no mass silently drops
   out of the numerator either).

4. bbq_predict / bbq_fit no longer crash or silently mis-bin when
   percentile-based bin edges collide (common with heavily imbalanced or
   discretised risk scores): duplicate edges are now dropped via
   np.unique before binning, and n_bins is capped to whatever remains.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def _make_bin_edges(y_prob, n_bins):
    """Equal-frequency (percentile) bin edges, floor forced to 0.0 and
    ceiling forced to 1.0, with duplicate edges collapsed so every
    resulting bin has strictly positive width."""
    raw = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    raw[0] = 0.0
    raw[-1] = 1.0
    bins = np.unique(raw)
    if len(bins) < 2:
        bins = np.array([0.0, 1.0])
    return bins


def bbq_fit(y_prob, y_true, n_bins=15, beta_alpha=2.0, beta_beta=2.0):
    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)
    bins = _make_bin_edges(y_prob, n_bins)
    n_actual_bins = len(bins) - 1

    model = {"bins": bins, "calibration": {}}
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_actual_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            n_pos = y_true[mask].sum()
            n_total = mask.sum()
            smoothed = (n_pos + beta_alpha - 1) / (n_total + beta_alpha + beta_beta - 2)
            model["calibration"][i] = smoothed
        else:
            model["calibration"][i] = np.nan
    return model


def bbq_predict(model, y_prob):
    y_prob = np.asarray(y_prob)
    bins = model["bins"]
    n_actual_bins = len(bins) - 1
    calibrated = np.full_like(y_prob, np.nan, dtype=np.float64)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_actual_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        if i in model["calibration"] and not np.isnan(model["calibration"][i]):
            calibrated[mask] = model["calibration"][i]
        else:
            calibrated[mask] = (lo + hi) / 2

    # Any point still unassigned (should not happen now that bins[0]=0.0
    # and bins[-1]=1.0 span the full valid probability range, but guarded
    # explicitly rather than silently defaulting to 0.0) falls back to
    # the raw input value -- the least-wrong assumption when no bin
    # exists to inform a correction.
    unassigned = np.isnan(calibrated)
    if unassigned.any():
        calibrated[unassigned] = y_prob[unassigned]

    return calibrated


def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    n_actual_bins = len(bins) - 1
    ece = 0.0
    total = len(y_true)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_actual_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        empirical_freq = y_true[mask].mean()
        avg_confidence = y_prob[mask].mean()
        ece += (mask.sum() / total) * abs(empirical_freq - avg_confidence)
    return ece


def bootstrap_auroc_ci(y_true, y_prob, n_boot=1000):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    aurocs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) > 1:
            aurocs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aurocs = np.array(aurocs)
    return aurocs.mean(), np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5), len(aurocs)
