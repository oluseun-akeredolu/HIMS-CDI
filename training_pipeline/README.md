# HIMS-CDI Training Pipeline

This directory contains the 10-seed training pipeline that produces Tables 5 and 7
of the manuscript. It is the artefact required by manuscript Appendix F item 1.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the v1.3.1 archive (or use a local copy)
# See: https://zenodo.org/records/23565580
# Required files: synthetic_train.csv, synthetic_val.csv, synthetic_test.csv,
#                 correlation_matrix.csv (for the generator if regenerating)

# 3. Run the full 10-seed pipeline
python run_training.py \
    --archive-dir /path/to/v1.3.1_archive \
    --seeds 13 37 73 101 211 421 563 719 881 997 \
    --output results/

# 4. Outputs (in results/):
#    - per_seed_predictions.csv       (Item 2 — 480 rows, 10 seeds × 48 test cases)
#    - bootstrap_ci.json              (Item 3 — 95% CIs for all metrics)
#    - table5.csv                     (per-model resource usage, matches manuscript Table 5)
#    - table7.csv                     (per-model classification metrics, matches manuscript Table 7)
#    - workstation_resource_usage.csv (Item 4a — per-seed latency/memory/training time)
```

## Directory structure

```
training_pipeline/
├── README.md                    ← this file
├── requirements.txt              ← exact library versions
├── run_training.py               ← main entry point (10-seed loop)
├── models/
│   ├── __init__.py
│   ├── gnn.py                    ← TODO: paste your GNN architecture (hidden=64, lr=1e-3)
│   ├── transformer.py            ← TODO: paste your Transformer architecture (hidden=512, lr=3e-5)
│   ├── edge_aware_fusion.py      ← TODO: paste your Edge-Aware fusion with pruning (compression=0.5)
│   ├── mlr_platt.py              ← TODO: paste your MLR with Platt scaling (HIMS-CDI_cal)
│   └── baselines.py              ← TODO: paste equal_weights, rule_based, checklist
├── bootstrap/
│   ├── __init__.py
│   └── rubin_pooling.py          ← Rubin's rules implementation (provided, ready to use)
├── measurement/
│   ├── __init__.py
│   ├── workstation_measure.py    ← TODO: paste your workstation measurement code
│   └── pi_measure.py             ← TODO: paste your Raspberry Pi measurement code
├── calibration/
│   ├── __init__.py
│   └── platt_scaling.py         ← Platt scaling with ECE computation (provided, ready to use)
├── configs/
│   └── seeds.yaml                ← the 10 seeds and their RNG states
└── results/                       ← output directory (created on first run)
```

## What you need to fill in (TODO markers)

The following files contain `# TODO:` markers showing where you need to paste your
original research code. Search for `TODO` to find them:

- `models/gnn.py` — your GNN architecture (PyTorch or TensorFlow)
- `models/transformer.py` — your Transformer architecture
- `models/edge_aware_fusion.py` — your Edge-Aware fusion model with pruning
- `models/mlr_platt.py` — your MLR training + Platt scaling wrapper
- `models/baselines.py` — your equal_weights, rule_based, checklist baselines
- `measurement/workstation_measure.py` — your workstation resource measurement code
- `measurement/pi_measure.py` — your Raspberry Pi deployment code (ONNX Runtime, INT8)

The following files are provided complete (no TODOs):

- `run_training.py` — the 10-seed loop, bootstrap pooling, table generation
- `bootstrap/rubin_pooling.py` — Rubin's rules implementation
- `calibration/platt_scaling.py` — Platt scaling with ECE computation
- `configs/seeds.yaml` — the 10 seeds and their configuration

## Verification

After running the pipeline, verify the outputs match the manuscript:

```bash
python verify_outputs.py --results-dir results/ --manuscript-values manuscript_table5_table7.json
```

This script checks that:
- `table5.csv` matches manuscript Table 5 within 5% tolerance per cell
- `table7.csv` matches manuscript Table 7 within 5% tolerance per cell
- `bootstrap_ci.json` produces 95% CI [0.72, 0.78] for AUROC (or wider, in which case
  the manuscript needs to be updated to match the wider CI)
- `per_seed_predictions.csv` has 480 rows with the expected schema

## Reproducibility notes

- The 10 seeds (13, 37, 73, 101, 211, 421, 563, 719, 881, 997) come from
  `supplementary_Material/Supplementary Material S4/S4_resource_usage.csv`
  in the GitHub repository.
- The 10,000-resample bootstrap per seed (100,000 total) follows the procedure
  documented in manuscript Section 4.5.
- Rubin's rules pooling follows Rubin (1987).
- Resource measurements use the S14 protocol (50 warm-up passes, 100 timed passes,
  memory_profiler for peak memory).
