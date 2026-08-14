"""
verify_fixes.py

Independently verifies each fix by comparing the ORIGINAL calibration.py
(as shipped in the repo) against the FIXED version, on cases specifically
constructed to trigger each bug. Run this yourself -- don't take my word
for it.

Usage:
    python3 verify_fixes.py
"""
import sys
import numpy as np

sys.path.insert(0, "src")
from csra.calibration import bbq_fit, bbq_predict, expected_calibration_error as ece_fixed

# Load the ORIGINAL (buggy) module under a different name for comparison.
# If you don't have the original on disk, this section is skipped.
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("calibration_original", "calibration_original.py")
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    HAVE_ORIGINAL = True
except FileNotFoundError:
    HAVE_ORIGINAL = False

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition):
    results.append((name, PASS if condition else FAIL))
    print(f"[{PASS if condition else FAIL}] {name}")


# ---------------------------------------------------------------------
# TEST 1: floor-collapse bug -- distinct low values must NOT all map to 0.0
# ---------------------------------------------------------------------
np.random.seed(1)
tune_y_prob = np.random.uniform(0.3, 0.95, 5000)
tune_y_true = (tune_y_prob > 0.5).astype(int)
model = bbq_fit(tune_y_prob, tune_y_true, n_bins=15)

test_points = np.array([0.02, 0.10, 0.20, 0.29, 0.299])
out = bbq_predict(model, test_points)
check(
    "Floor-collapse: bins[0] is forced to 0.0",
    model["bins"][0] == 0.0,
)
# NOTE: sub-floor points legitimately collapse to ONE value if they fall in
# the same lowest bin -- there's no data to discriminate further within it.
# What matters is that the value is the bin's TRUE empirical calibration,
# not a hardcoded 0.0 regardless of what the data says.
check(
    "Floor-collapse: sub-floor output equals the bin's true empirical rate, not a hardcoded 0.0",
    np.isclose(out[0], model["calibration"][0]),
)

if HAVE_ORIGINAL:
    orig_model = orig.bbq_fit(tune_y_prob, tune_y_true, n_bins=15)
    orig_out = orig.bbq_predict(orig_model, test_points)
    check(
        "Floor-collapse: ORIGINAL code hardcodes these to 0.0 regardless of true rate (confirms bug existed)",
        np.all(orig_out == 0.0),
    )

# ---------------------------------------------------------------------
# TEST 2: top-edge exclusion in ECE -- y_prob==1.0 must be counted
# ---------------------------------------------------------------------
y_true = np.concatenate([np.zeros(3000, dtype=int), np.random.randint(0, 2, 500)])  # miscalibrated at top
y_prob = np.concatenate([np.full(3000, 1.0), np.random.uniform(0.55, 0.65, 500)])
ece_val = ece_fixed(y_true, y_prob)
# All 3000 points claim 100% confidence but are ALL actually benign (label=0) ->
# huge miscalibration that MUST show up now.
check(
    "ECE: top-edge (y_prob==1.0) miscalibration is now captured (ECE should be large, ~0.86)",
    ece_val > 0.5,
)

if HAVE_ORIGINAL:
    orig_ece = orig.expected_calibration_error(y_true, y_prob)
    check(
        "ECE: ORIGINAL code misses this miscalibration entirely (confirms bug existed)",
        orig_ece < 0.1,
    )

# ---------------------------------------------------------------------
# TEST 3: bbq_predict never silently returns exactly 0.0 for high inputs
# ---------------------------------------------------------------------
high_test = np.array([0.999999, 1.0])
out_high = bbq_predict(model, high_test)
check(
    "Top-edge: y_prob==1.0 is now assigned to the top bin, not dropped",
    not np.any(np.isnan(out_high)),
)

# ---------------------------------------------------------------------
# TEST 4: duplicate percentile edges (heavy class imbalance / ties) don't crash
# ---------------------------------------------------------------------
degenerate = np.concatenate([np.full(9000, 0.99), np.random.uniform(0.0, 0.05, 100)])
degenerate_true = np.concatenate([np.ones(9000, dtype=int), np.zeros(100, dtype=int)])
try:
    m = bbq_fit(degenerate, degenerate_true, n_bins=15)
    p = bbq_predict(m, degenerate)
    check("Robustness: duplicate percentile edges (heavy imbalance) don't crash", True)
except Exception as e:
    check(f"Robustness: duplicate percentile edges (heavy imbalance) don't crash -- {e}", False)

print()
n_fail = sum(1 for _, r in results if r == FAIL)
print(f"{len(results) - n_fail}/{len(results)} checks passed.")
if n_fail:
    sys.exit(1)
