# CSRA: Cyber-Security Risk Assessment for IoMT — Phase 1

Reproducible data processing and fusion calibration pipeline for the **CIC IoMT 2024** dataset.

---

## Prerequisites

- **Python:** 3.9+
- **Required packages:** pandas, numpy, pyarrow, scikit-learn, matplotlib, joblib
  (see `requirements.txt` for pinned minimum versions)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Download the official dataset:

https://www.unb.ca/cic/datasets/iomt-dataset-2024.html

Place all CSV files in:

```text
data/raw/
```

Expected files:

- **51 training files:** `*_train_pcap.csv`
- **21 test files:** `*_test_pcap.csv`

The scripts use specific glob patterns to prevent data leakage if both training and test files are placed in the same directory.

---

## Directory Structure

```text
project-root/
├── data/
│   └── raw/
│       ├── ARP_Spoofing_train_pcap.csv
│       ├── ... (51 training files)
│       ├── ARP_Spoofing_test_pcap.csv
│       └── ... (21 test files)
├── scripts/
│   ├── 00_load_real_data.py
│   ├── 00b_load_test_shards.py
│   ├── 01_split_data.py
│   ├── 02_fusion_calibration_evaluate.py
│   └── 03_freeze.py
├── src/
│   └── csra/
│       └── config.py
└── README.md
```

---

## Alternative Data Path

If your data is stored elsewhere, override the default path by setting the environment variable:

```bash
export CSRA_DATA_DIR=/path/to/your/custom/data/folder
```

---

## Usage

Run the scripts from the project root in the following order:

```bash
# 1. Load 51 official TRAIN files, shard to parquet
python3 scripts/00_load_real_data.py

# 2. Load 21 official TEST files, shard to parquet
python3 scripts/00b_load_test_shards.py

# 3. TRAIN files -> TRAIN/TUNE (75/25); TEST files -> TEST wholesale
python3 scripts/01_split_data.py

# 4. Fusion, Calibration, and Evaluation (Phases 1.2, 1.3, 1.4 combined)
python3 scripts/02_fusion_calibration_evaluate.py

# 5. Freeze final model/artifacts (Phase 1.5) with SHA-256 integrity
python3 scripts/03_freeze.py
```

---

## Methodological Decisions & Honest Limitations

### Official Train/Test Split

Uses the official 21 test files as the held-out **TEST** partition. The remaining 51 training files are split into **TRAIN/TUNE (75/25)** while preserving capture order, yielding approximately:

- **TRAIN:** 61.2%
- **TUNE:** 20.4%
- **TEST:** 18.4%

### Pseudo-Chronological Ordering

Files are ordered alphabetically and rows retain capture order. This provides a reproducible convention rather than a true temporal sequence.

This is a documented convention, **not** a true timeline.

### Single Source Implementation

The public CIC IoMT 2024 dataset contains only one capture pipeline; therefore:

- Correlation-aware weighting is inactive.
- The particle filter operates in scalar mode.

### Systematic Subsampling

Approximately:

- **207k TRAIN rows**
- **72k TUNE rows**
- **70k TEST rows**

are processed due to hardware constraints (~1 CPU, ~3.9 GB RAM).

The identical pipeline supports the complete **8.78 million-row dataset** on higher-specification hardware.

---

## Results

The manuscript reports results from the **canonical post-fix sandbox run**
(documented in `CHANGELOG.md` item #7):

| Model | ECE | AUROC |
|-------|-----|-------|
| **Proposed (BBQ-calibrated)** | **0.0024** | **0.9796** (95% CI [0.9788, 0.9803]) |
| Uncalibrated Particle Filter | 0.0484 | 0.9998 |
| Logistic Regression | 0.0114 | 0.9959 |

These values are stored in `artifacts/test_results_real.json` (key: `proposed_bbq_calibrated`).

A subsequent local Windows replication (Python 3.11.9) produced ECE 0.0027 /
AUROC 0.9792, confirming pipeline portability across platforms; minor
deviations are expected due to OS-level floating-point and random-seed
handling differences.---

## Data Schema

Each output Parquet shard contains the original dataset columns together with the following appended fields.

| Column Name | Type | Description |
| :---------- | :--: | :---------- |
| `label` | `int8` | Binary target (0 = Benign, 1 = Attack) |
| `attack_type` | `str` | Attack name derived from filename |
| `source_file` | `str` | Original CSV filename |
| `file_order` | `int16` | Alphabetical file rank |
| `row_order_in_file` | `int32` | Original row index |
| `pseudo_sequence_index` | `int64` | Global chronological index |

---

## Integrity Verification

The `03_freeze.py` script generates a SHA-256 manifest for bit-for-bit reproducibility.

Example:

```json
{
  "artefact_path": "artifacts/frozen_model_real.joblib",
  "sha256": "[HASH]",
  "frozen_at_utc": "[TIMESTAMP]",
  "dataset": "real CIC IoMT 2024 (systematic subsample)"
}
```

---

## Citation

If you use this pipeline or the **CIC IoMT 2024** dataset in your research, please cite:

> Dadkhah, S., Pinto Neto, E. C., Ferreira, R., Molokwu, R. C., Sadeghi, S., & Ghorbani, A. A. (2024). *CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT*. **Internet of Things, 28**, 101351.

Dataset:

https://www.unb.ca/cic/datasets/iomt-dataset-2024.html

---

## License

Apache License 2.0.

---

## Contact

**Oluseun Akeredolu**

Secure Cyber Systems Research Group (SCSRG)  
WMG, University of Warwick  
Coventry CV4 7AL  
United Kingdom

**Email:** Olu.Akeredolu@warwick.ac.uk

**Telephone:** +44 (0)7459 824641

---

## Final Checklist

- [ ] CSV files placed in `data/raw/`
- [ ] `CSRA_DATA_DIR` configured (if required)
- [ ] All scripts completed successfully
- [ ] Results match `test_results_real.json`
- [ ] SHA-256 manifest verified
