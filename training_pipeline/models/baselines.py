"""
Simple baselines: equal_weights, rule_based, checklist (manuscript Table 7 rows).

These three baselines do not require neural network training. They are deterministic
and serve as lower-bound comparators.

TODO: Adjust these implementations to match your original baseline code if you used
      different definitions.
"""

import numpy as np


class EqualWeightsModel:
    """Equal-weight composite: average of the 4 features, predict 1 if average >= threshold.

    Manuscript Table 7 reports: Macro F1 = 0.28, AUROC = 0.49.
    """

    def __init__(self, threshold=0.5, seed=13):
        self.threshold = threshold
        self.seed = seed

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """No training — deterministic baseline. Just store the threshold."""
        return self

    def predict_proba(self, X):
        """Equal-weight average of the 4 features, mapped to [0, 1]."""
        # X has 4 features: CDI, CCR, latency_ms, memory_mb
        # Normalise latency and memory to [0, 1] (higher = better)
        lat_norm = 1.0 - (X[:, 2] / 320.0)  # latency max = 320 ms
        mem_norm = 1.0 - (X[:, 3] / 220.0)  # memory max = 220 MB
        composite = (X[:, 0] + X[:, 1] + lat_norm + mem_norm) / 4.0
        composite = np.clip(composite, 0, 1)
        return np.column_stack([1 - composite, composite])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


class RuleBasedModel:
    """Rule-based mapping: hard thresholds on CDI and CCR.

    Manuscript Table 7 reports: Macro F1 = 0.39, AUROC = 0.52.

    TODO: Adjust the rule definition if your original code used different thresholds.
    """

    def __init__(self, cdi_threshold=0.65, ccr_threshold=0.70, seed=13):
        self.cdi_threshold = cdi_threshold
        self.ccr_threshold = ccr_threshold
        self.seed = seed

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self

    def predict_proba(self, X):
        # Score = average of (CDI - threshold) and (CCR - threshold), normalised
        score = (X[:, 0] - self.cdi_threshold) + (X[:, 1] - self.ccr_threshold)
        # Map to [0, 1] via sigmoid
        prob = 1 / (1 + np.exp(-5 * score))
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ChecklistModel:
    """Minimal Boolean checklist: predict 1 if CDI >= 0.70 AND CCR >= 0.80.

    Manuscript Table 7 reports: Macro F1 = 0.33, AUROC = 0.55.
    The model uses hard Boolean rules and does not natively compute a CDI score;
    a post-hoc CDI is reported (mean = 0.68) for descriptive comparison only.
    """

    def __init__(self, seed=13):
        self.seed = seed

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        return self

    def predict_proba(self, X):
        cdi = X[:, 0]
        ccr = X[:, 1]
        # Boolean rule: predict 1 if CDI >= 0.70 AND CCR >= 0.80
        # For probabilities, use a soft version (close to 0 or 1)
        score = ((cdi >= 0.70).astype(float) + (ccr >= 0.80).astype(float)) / 2.0
        return np.column_stack([1 - score, score])

    def predict(self, X):
        return ((X[:, 0] >= 0.70) & (X[:, 1] >= 0.80)).astype(int)
