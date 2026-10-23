# HIMS-CDI Training Pipeline

This directory contains the 10-seed training pipeline that produces the
classification metrics, gate satisfaction data, and resource measurements
reported in the manuscript.

## Reference implementation notice

This is a **reference implementation** using standard published architectures
(Kipf & Welling 2017 for the GNN; Vaswani et al. 2017 for the Transformer).
It is NOT a reproduction of the authors' original research code, which was
not deposited. The reproduced numbers differ from the manuscript's previously-
reported values, which came from the original code.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the v1.3.1 archive from Zenodo (DOI 10.5281/zenodo.23565580)
#    or use a local copy. Required files:
#    synthetic_train.csv, synthetic_val.csv, synthetic_test.csv

# 3. Run the full 10-seed pipeline
python run_training.py \
    --archive-dir /path/to/zenodo_data \
    --config configs/seeds.yaml \
    --output results/ \
    --bootstrap-samples 10000

# 4. Outputs (in results/):
#    - per_seed_predictions.csv       (4,320 rows = 9 models x 10 seeds x 48 test cases)
#    - bootstrap_ci.json              (pooled 95% CIs via Rubin's rules)
#    - table5_resource_usage.csv      (per-model, per-seed resource measurements)
#    - table7_classification.csv      (per-model classification metrics)
#    - gate_satisfaction.csv          (case-level gate pass/fail for the test set)
#    - workstation_resource_usage.csv (per-seed latency/memory/training time)
```

## Expected outputs

| File | Rows | Description |
|------|------|-------------|
| `per_seed_predictions.csv` | 4,320 | 9 models × 10 seeds × 48 test cases |
| `bootstrap_ci.json` | — | Pooled 95% CIs for AUROC and macro F1 |
| `table7_classification.csv` | 9 | Per-model AUROC, macro F1, sensitivity, specificity |
| `gate_satisfaction.csv` | 48 | Per-case gate pass/fail (CDI, CCR, latency, memory) |
| `table5_resource_usage.csv` | — | Per-model, per-seed resource measurements |
| `workstation_resource_usage.csv` | 90 | Per-model, per-seed raw measurements |

## Separation of evidence

The pipeline produces three distinct types of evidence, reported in separate files:

1. **Classification metrics** (`table7_classification.csv`): AUROC, macro F1,
   sensitivity, specificity — computed from the model's predicted probabilities.
2. **Case-level gate satisfaction** (`gate_satisfaction.csv`): whether each of
   the 48 test cases passes each governance gate. This is a property of the
   synthetic cases, NOT of any particular model.
3. **Model-specific resource measurements** (`table5_resource_usage.csv`):
   latency, memory, and training time — measured per model, per seed.

These three types of evidence must NOT be conflated. The manuscript must report
each in its own table and draw conclusions from the appropriate source.

## Configuration

The 10 seeds, hyperparameters, and bootstrap settings are in `configs/seeds.yaml`.

## Reproducibility notes

- Seeds: 13, 37, 73, 101, 211, 421, 563, 719, 881, 997
- Bootstrap: 10,000 resamples per seed (pooled via Rubin's rules)
- The GNN implementation uses self-loop adjacency (A = I), making it effectively
  an MLP with GCN-style message passing. For a 4-feature tabular dataset without
  graph structure, this is the standard adaptation. The manuscript should describe
  it as a GCN-style tabular baseline, not as a graph model with substantive edges.
- The Transformer uses hidden_size=128 (the manuscript reports 512; 128 is used
  here because 512 was too slow on CPU for a 4-feature dataset).
