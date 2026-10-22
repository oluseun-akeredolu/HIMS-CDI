"""
HIMS-CDI MLR with Platt scaling — the calibrated HIMS-CDI configuration
(manuscript Table 7 row "HIMS-CDI_cal").

This file defines two classes:
    - HIMS_CDIMLRBase: the uncalibrated MLR (Table 7 row "HIMS-CDI_base")
    - HIMS_CDIMLRCalibrated: the MLR with Platt scaling (Table 7 row "HIMS-CDI_cal")

The MLR itself uses scikit-learn's LogisticRegression(C=1.0, max_iter=1000).
The Platt scaling uses scikit-learn's CalibratedClassifierCV with method='sigmoid'.

TODO: Adjust this file to match your original calibration procedure if it differs
      (e.g., different cv folds, different ECE computation method).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


class HIMS_CDIMLRBase:
    """HIMS-CDI_base — plain multinomial logistic regression without Platt scaling."""

    def __init__(self, C=1.0, max_iter=1000, seed=13):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=seed,
            multi_class='multinomial', solver='lbfgs',
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the plain MLR. X_val and y_val are ignored (no early stopping for LR)."""
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        """Return shape (n_samples, 2) — probabilities for [class_0, class_1]."""
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


class HIMS_CDIMLRCalibrated:
    """HIMS-CDI_cal — MLR with Platt scaling (sigmoid calibration).

    The calibration procedure:
        1. Train the base MLR on the training set.
        2. Wrap with CalibratedClassifierCV(method='sigmoid', cv=5).
        3. The Platt scaling reduces ECE from 0.12 to 0.03 (manuscript Section 4.5).
    """

    def __init__(self, C=1.0, max_iter=1000, seed=13, platt_cv=5):
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self.platt_cv = platt_cv
        self.base_model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=seed,
            multi_class='multinomial', solver='lbfgs',
        )
        self.calibrated_model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the base MLR, then apply Platt scaling via CalibratedClassifierCV."""
        self.base_model.fit(X_train, y_train)
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method='sigmoid',
            cv=self.platt_cv,
        )
        self.calibrated_model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        """Return shape (n_samples, 2) — calibrated probabilities."""
        return self.calibrated_model.predict_proba(X)

    def predict(self, X):
        return self.calibrated_model.predict(X)

    def compute_ece(self, X, y, n_bins=10):
        """Compute Expected Calibration Error (ECE).

        ECE = sum over bins of (bin_size/n_total) * |bin_accuracy - bin_confidence|

        The manuscript reports ECE = 0.12 before calibration and 0.03 after.
        """
        probs = self.predict_proba(X)
        # For binary classification, use the probability of the positive class
        p_pos = probs[:, 1]
        y_pred = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        accuracy = (y_pred == y).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y)
        for i in range(n_bins):
            mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = accuracy[mask].mean()
            bin_conf = confidence[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        return ece
