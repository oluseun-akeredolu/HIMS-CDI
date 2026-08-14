# Changelog — Calibration & Reliability Diagram Fixes

## Fixed

### 1. Floor-collapse bug in `bbq_fit` / `bbq_predict` (`src/csra/calibration.py`)
**Severity: high — silently mis-scores real predictions, not just a plot artefact.**

`bbq_fit` forced the top bin edge to 1.0 but never forced the bottom edge to
0.0 — it used `np.percentile(y_prob, 0)`, i.e. whatever TUNE's minimum
happened to be. Any TEST point scored below that floor (plausible: TUNE and
TEST are different subsamples of different files) matched no bin in
`bbq_predict`, and silently kept the `np.zeros_like()` default of exactly
0.0 — regardless of the point's real value or the bin's true empirical risk.

Verified with `verify_fixes.py`: constructed a TUNE set where the
lowest-confidence bin has a true empirical attack rate of 45%. The original
code reported four genuinely different sub-floor test points as **0% risk**.
The fixed code correctly reports **45% risk** for the same points — their
true empirical rate for that lowest bin.

**Fix:** force `bins[0] = 0.0` alongside the existing `bins[-1] = 1.0`.

### 2. Top-edge exclusion in `bbq_fit`, `bbq_predict`, `expected_calibration_error`
**Severity: medium — likely small practical effect on this dataset, but a real correctness bug.**

All three functions used `(y_prob >= lo) & (y_prob < hi)` for every bin,
including the last, so `y_prob == 1.0` matched nothing. In BBQ's own output
this rarely triggers in practice (Beta(2,2) smoothing makes exact 1.0
calibrated outputs mathematically near-impossible), but it's a real gap in
`expected_calibration_error` when called on raw/uncalibrated scores, and a
latent risk anywhere continuous scores could saturate at the ceiling.

Verified with `verify_fixes.py`: constructed 3,000 points at exactly
`y_prob = 1.0` that are actually *mis*calibrated (labelled benign despite
100% claimed confidence). The original `expected_calibration_error` returns
ECE < 0.1 (the miscalibrated mass is invisible to it, contributing nothing
to numerator or denominator). The fixed version correctly returns ECE > 0.5.

**Fix:** last bin's mask is now `(y_prob >= lo) & (y_prob <= hi)`.

### 3. Reliability diagram bin count/type mismatch vs. Appendix I.2 (`02_fusion_calibration_evaluate.py`)
**Severity: low-medium — documentation/figure consistency, not a numerical bug.**

`reliability_diagram()` defaulted to `n_bins=10`, equal-*width* bins
(`np.linspace`). Appendix I.2 states reliability diagrams use "M = 15
equal-frequency bins." Changed to `n_bins=15`, percentile-based (equal-
frequency) bins, floor/ceiling forced to 0.0/1.0, last bin inclusive
(same edge fix as above), and duplicate percentile edges collapsed via
`np.unique` so heavily-imbalanced score distributions don't produce
zero-width bins or crash. Verified this renders correctly and captures a
point at exactly 1.0 confidence (see `test_reliability.png` in this PR).

Also fixed the figure title being clipped at typical `figsize=(5,5)` —
added wrapping so the full title renders.

### 4. `README.md` dependency list incomplete
`requirements.txt` (numpy, pandas, pyarrow, scikit-learn, matplotlib,
joblib) already correctly lists everything the pipeline imports. The
README's "Prerequisites" section only mentioned three of the six and told
users to `pip install pandas numpy pyarrow`, which would `ModuleNotFoundError`
on `matplotlib`/`sklearn`/`joblib` in a clean environment. Now points
directly at `requirements.txt`.

### 5. `bbq_fit` / `bbq_predict` robustness to duplicate percentile edges
Not a bug that was hit in the reported run, but a latent one: heavily
imbalanced or discretised risk scores can produce duplicate percentile
values, yielding zero-width bins. Now deduplicated via `np.unique` before
binning; `bbq_predict` no longer returns `NaN` or crashes if this happens.

## Still outstanding — requires action outside this code

### 6. Manuscript wording: "N=1000-particle bootstrap" (Section 6.8, `.docx`)
Conflates two unrelated parameters — `ScalarParticleFilter(n_particles=1000)`
and `bootstrap_auroc_ci(n_boot=1000)`. This is a prose fix in the manuscript,
not something a code change resolves. Suggested replacement:

> "using N = 1,000 particles in the bootstrap particle filter and 1,000
> bootstrap resamples for the AUROC confidence interval"

### 7. `test_results_real.json` and the README's Results table — RE-RUN COMPLETE

**Update:** the pipeline has now been re-run end-to-end against the real 51
train + 21 test files, using the fixed `calibration.py`. Verified totals
matched the manuscript exactly: 7,160,831 train rows, 1,614,182 test rows,
8,775,013 combined.

Two real filename/path issues had to be worked around to run it (not code
bugs — a data-packaging mismatch): the train archive's files were named
`<name>_train.pcap.csv` (dot before "pcap") rather than the
`<name>_train_pcap.csv` the glob pattern expects, and were nested one
directory level deeper than the flat layout the script assumes. Files were
renamed/flattened into a staging directory before running; no data content
was altered, confirmed by row counts matching exactly before and after.

Post-fix results:

| Metric | Pre-fix | Post-fix (this run) |
|---|--:|--:|
| BBQ-calibrated ECE | 0.002608 | 0.002410 |
| BBQ-calibrated AUROC | 0.979570 | 0.979566 |
| Uncalibrated PF ECE | 0.048370 | 0.048370 |
| Uncalibrated PF AUROC | 0.999817 | 0.999817 |
| Logistic Regression ECE | 0.011445 | 0.011431 |
| Logistic Regression AUROC | 0.995874 | 0.995874 |
| TUNE-set ECE (fit, optimistic) | 0.000396 | 0.000198 |

The fixes changed the calibrated ECE only marginally on this specific real
dataset — reassuring that the pre-fix headline numbers weren't badly wrong
in practice, though the bugs themselves were real (see items #1 and #2
above) and could matter more under different data or distributional shift.
`test_results_real.json` and `reliability_diagram_real.png` in this delivery
are the actual post-fix outputs, not placeholders.
