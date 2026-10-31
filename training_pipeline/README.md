# HIMS-CDI Training Pipeline — separate 10-seed reference implementation

## Scope and non-equivalence notice

This directory contains a 10-seed reference training pipeline for a separate synthetic model-comparison experiment. It is **not** the source package for the v0.2.17 manuscript's Phase I v0.2.9 four-gate simulation unless a full, version-controlled file/input/configuration/output comparison establishes that relationship.

This is a **reference implementation**, not a reproduction of the authors' original research code, which was not deposited. Its reproduced numbers differ from previously reported manuscript values. Do not cite its metrics as if they were reproduced outputs of the original manuscript or the v0.2.9 specification simulation.

The pipeline documentation describes nine candidate models, 10 seeds and 48 test cases per seed. It produces classification metrics, case-level gate decisions and model-specific resource measurements. Keep these three evidence classes separate. The README alone does not verify their statistical validity or prove that all reported metrics match the current manuscript.

## Quick start

The historical instructions below identify Zenodo DOI `10.5281/zenodo.23565580` as the data dependency. Confirm the exact DOI record, version and required files before executing or citing the pipeline; do not substitute the v0.2.9 package or another historical dataset.

```bash
pip install -r requirements.txt
python run_training.py \
    --archive-dir /path/to/verified_data_archive \
    --config configs/seeds.yaml \
    --output results/ \
    --bootstrap-samples 10000
```

Expected artefacts are `per_seed_predictions.csv`, `bootstrap_ci.json`, `table5_resource_usage.csv`, `table7_classification.csv`, `gate_satisfaction.csv` and `workstation_resource_usage.csv`. Check the actual output schema and row counts against the exact code version before reporting.

## Evidence separation

1. **Classification metrics** are computed from predicted probabilities.
2. **Case-level gate satisfaction** is a property of the synthetic cases, not of a particular model.
3. **Model-specific resource measurements** are measurements from the declared run protocol and must not be conflated with the gate status of the cases.

## Architecture notes

- The GNN implementation uses self-loop adjacency (`A = I`); it is effectively an MLP with GCN-style message passing on a four-feature tabular input. Describe it as a GCN-style tabular baseline, not as a graph model with substantive edges.
- The Transformer implementation uses `hidden_size=128`; prior manuscript text reported 512. The code's actual configuration takes precedence for describing this reference implementation, and the difference must be explicitly reconciled before any related number is cited.
- The seeds and bootstrap settings are recorded in `configs/seeds.yaml`. Re-run from a clean environment and inspect `run_manifest.json` before making reproducibility claims.

## Relationship to current manuscript

The current manuscript uses the separate package `PhaseI_Synthetic_Evaluation_v0.2.9.zip` (SHA-256 `b0052e38da33b5733dcfb4b829a5472f4d15a109a8a67d6df7b053aa8ca5bafe`). This README's 10-seed results, the Zenodo v1.5.4 release candidate, the v1.3.0 synthetic data archive and the v0.2.9 four-gate simulation are separate experiment/release lines. No cross-line numerical substitution is permitted without demonstrated equivalence.
