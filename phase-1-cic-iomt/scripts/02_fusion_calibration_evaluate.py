"""
02_fusion_calibration_evaluate.py (real-data version)

Combines Phase 1.2-1.4 into one script (the sandbox's single-CPU/4GB
constraint makes it more practical to keep the classifier and the
particle filter's carried-over state together rather than
serialize/reload between separate script invocations).

Pipeline:
  1. Systematic (order-preserving) subsample of train/tune/test --
     documented sandbox compute constraint, see subsample.py.
  2. Train the base classifier (HistGradientBoostingClassifier,
     class_weight="balanced" given ~2.7% positive... actually here label=1
     IS the attack/majority class and label=0 (benign) is the ~2.7%
     minority -- balancing matters for benign detection).
  3. Run the raw per-flow risk scores through the scalar particle filter,
     IN ORDER, with state carried continuously across train -> tune -> test
     (matches "the filter never resets" deployment behaviour).
  4. Fit true BBQ (Beta(2,2)-smoothed, 15 equal-frequency bins) on TUNE.
  5. Evaluate ONCE on TEST: ECE (calibrated + uncalibrated), bootstrap
     AUROC 95% CI, reliability diagram, and a Logistic Regression baseline
     (a genuinely different architecture from the gradient-boosted base
     classifier).
"""

import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from csra.config import DATA_DIR, ARTIFACTS_DIR
from csra.calibration import bbq_fit, bbq_predict, expected_calibration_error, bootstrap_auroc_ci
from subsample import systematic_subsample, FEATURE_COLS
from real_source_fusion import ScalarParticleFilter, smooth_risk_sequence

TARGET_ROWS = {"train": 200_000, "tune": 70_000, "test": 70_000}


def load_subsamples():
    subs = {}
    for split, target in TARGET_ROWS.items():
        path = DATA_DIR / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run 01_split_data.py first.")
        t0 = time.time()
        df = systematic_subsample(path, target_rows=target)
        subs[split] = df
        print(f"  {split}: subsampled {len(df):,} rows in {time.time()-t0:.1f}s "
              f"(label=0: {int((df['label']==0).sum()):,}, label=1: {int((df['label']==1).sum()):,})")
    return subs


def reliability_diagram(y_true, y_prob, path: Path, n_bins=15, title="Reliability diagram"):
    """
    Equal-frequency (percentile) reliability diagram, matching Appendix I.2's
    "M = 15 equal-frequency bins" and the BBQ calibration methodology itself
    -- the diagram should bin the same way the calibrator does.

    Floor/ceiling are forced to 0.0/1.0 and duplicate percentile edges are
    collapsed (np.unique) so ties in heavily-imbalanced risk scores don't
    produce zero-width bins. The last bin is inclusive of 1.0 -- probability
    mass at exactly the maximum is plotted, not silently dropped.
    """
    raw = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    raw[0] = 0.0
    raw[-1] = 1.0
    bins = np.unique(raw)
    n_actual_bins = len(bins) - 1

    accs, confs, counts = [], [], []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_actual_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        accs.append(y_true[mask].mean())
        confs.append(y_prob[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    sizes = np.array(counts) / max(counts) * 200 + 20 if counts else []
    ax.scatter(confs, accs, s=sizes, alpha=0.7, label="Observed bins")
    ax.set_xlabel("Predicted risk (confidence)")
    ax.set_ylabel("Observed attack rate")
    ax.set_title(title, fontsize=10, wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    print("Loading systematic subsamples (sandbox memory constraint -- see subsample.py docstring)...")
    subs = load_subsamples()
    train, tune, test = subs["train"], subs["tune"], subs["test"]

    X_train, y_train = train[FEATURE_COLS].values.astype(np.float32), train["label"].values
    X_tune, y_tune = tune[FEATURE_COLS].values.astype(np.float32), tune["label"].values
    X_test, y_test = test[FEATURE_COLS].values.astype(np.float32), test["label"].values

    # --- Base "source" classifier (stands in for per-source risk scoring;
    #     with only one real source, this IS the source model) ---
    print("\nTraining base classifier (HistGradientBoostingClassifier)...")
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    print(f"  fit time: {time.time()-t0:.1f}s")

    raw_train = clf.predict_proba(X_train)[:, 1]
    raw_tune = clf.predict_proba(X_tune)[:, 1]
    raw_test = clf.predict_proba(X_test)[:, 1]

    # --- Particle filter: temporal smoothing, state carried continuously
    #     across train -> tune -> test (never reset) ---
    print("\nRunning scalar particle filter (N=1000, sigma=0.1) over train -> tune -> test, state carried...")
    pf = ScalarParticleFilter(n_particles=1000, process_noise=0.1, seed=7)
    t0 = time.time()
    _ = smooth_risk_sequence(raw_train, pf)              # warms up state, not evaluated
    tune_smoothed = smooth_risk_sequence(raw_tune, pf)    # BBQ fit target
    test_smoothed = smooth_risk_sequence(raw_test, pf)    # final evaluation target
    print(f"  particle filter time: {time.time()-t0:.1f}s")

    # --- Step 1.3: BBQ calibration on TUNE only ---
    print("\nFitting BBQ (Beta(2,2), 15 bins) on TUNE...")
    bbq_model = bbq_fit(tune_smoothed, y_tune, n_bins=15, beta_alpha=2.0, beta_beta=2.0)
    tune_ece = expected_calibration_error(y_tune, bbq_predict(bbq_model, tune_smoothed))
    print(f"  BBQ ECE on TUNE (fit-set, optimistic by construction): {tune_ece:.4f}")

    # --- Step 1.4: evaluation on TEST, exactly once ---
    print("\nEvaluating on TEST (first and only time this data is touched)...")
    test_calibrated = bbq_predict(bbq_model, test_smoothed)
    ece_calibrated = expected_calibration_error(y_test, test_calibrated)
    ece_uncalibrated = expected_calibration_error(y_test, test_smoothed)
    auroc_mean, auroc_lo, auroc_hi, n_valid_boot = bootstrap_auroc_ci(y_test, test_calibrated, n_boot=1000)
    auroc_uncalibrated = roc_auc_score(y_test, test_smoothed)

    # Alternative-architecture baseline: Logistic Regression (linear, genuinely
    # different from gradient-boosted trees), scaled features, class-balanced.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(class_weight="balanced", max_iter=500)
    lr.fit(X_train_scaled, y_train)
    lr_scores = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auroc = roc_auc_score(y_test, lr_scores)
    lr_ece = expected_calibration_error(y_test, lr_scores)

    reliability_diagram(y_test, test_calibrated, ARTIFACTS_DIR / "reliability_diagram_real.png",
                         title="BBQ-calibrated single-source fusion -- real CIC IoMT 2024 TEST set")

    print("\n===== TABLE 7 -- FINAL TEST-SET RESULTS (real CIC IoMT 2024 data) =====")
    print(f"{'Model':45s} {'ECE':>8s} {'AUROC':>8s}")
    print(f"{'Proposed (BBQ-calibrated, single source)':45s} {ece_calibrated:8.4f} {auroc_mean:8.4f}  "
          f"95% CI [{auroc_lo:.4f}, {auroc_hi:.4f}] (n={n_valid_boot})")
    print(f"{'Baseline: uncalibrated particle filter':45s} {ece_uncalibrated:8.4f} {auroc_uncalibrated:8.4f}")
    print(f"{'Baseline: Logistic Regression (alt. arch.)':45s} {lr_ece:8.4f} {lr_auroc:8.4f}")

    results = {
        "dataset": "real CIC IoMT 2024 (51 files, systematic subsample due to sandbox RAM/CPU limits)",
        "subsample_sizes": {k: len(v) for k, v in subs.items()},
        "proposed_bbq_calibrated": {"ece": ece_calibrated, "auroc_mean": auroc_mean,
                                     "auroc_ci_95": [auroc_lo, auroc_hi], "n_bootstrap_valid": n_valid_boot},
        "baseline_uncalibrated_pf": {"ece": ece_uncalibrated, "auroc": auroc_uncalibrated},
        "baseline_logistic_regression": {"ece": lr_ece, "auroc": lr_auroc},
        "tune_ece_fit_set": tune_ece,
        "single_source_caveat": (
            "Real CIC IoMT 2024 provides one capture-pipeline source, not five. "
            "CorrelationAwareWeighting and the multi-dimension vector particle "
            "state were not exercised meaningfully on this data -- see "
            "real_source_fusion.py docstring."
        ),
    }
    results_path = ARTIFACTS_DIR / "test_results_real.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {results_path}")
    print(f"Reliability diagram saved to {ARTIFACTS_DIR / 'reliability_diagram_real.png'}")

    # stash objects needed for freeze step
    import joblib
    joblib.dump({
        "classifier": clf,
        "particle_filter_state": {"particles": pf.particles, "weights": pf.weights},
        "bbq_model": bbq_model,
        "feature_cols": FEATURE_COLS,
    }, ARTIFACTS_DIR / "pre_freeze_bundle_real.joblib")
    print(f"Pre-freeze bundle saved to {ARTIFACTS_DIR / 'pre_freeze_bundle_real.joblib'}")


if __name__ == "__main__":
    main()
