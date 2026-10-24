# HIMS-CDI: Healthcare Cyber Situational Risk Assessment

This repository contains the reproducibility artefact for the manuscript:
**HIMS-CDI: A Governance Specification and Assurance Architecture for CTI-Driven Cybersecurity Risk Assessment in Healthcare**

## Canonical Source of Record
The exact code, data splits, and model artifacts used in the manuscript have been archived on Zenodo:
- **Archive DOI:** [10.5281/zenodo.23615735](https://doi.org/10.5281/zenodo.23615735) (v1.5.3)
- **Source Branch:** `v9.6-final-audit`
- **Note:** The `main` branch may contain updates or separate project lines. The Zenodo archive is the immutable canonical source for the published paper.

## Repository Structure

- **`training_pipeline/`** — The core synthetic Phase I evaluation pipeline for the HIMS-CDI manuscript.
  - Includes the shared governance gate function (`governance_gates.py`), the 31-test automated suite, and the strengthened verifier (`verify_outputs.py`).
  - See the README in this directory for instructions on running the 10-seed evaluation.

- **`phase-1-cic-iomt/`** — A separate empirical reproducibility package for the **CIC IoMT 2024** dataset. This is a companion study and is **not** the subject of the HIMS-CDI manuscript.

## Environment
The canonical run environment is documented in `run_environment.json` within the Zenodo archive (Python 3.12, PyTorch 2.14.1+cpu).

## Citation
If you use the HIMS-CDI architecture or the synthetic pipeline, please cite:
> Akeredolu, O., Epiphaniou, G., & Maple, C. (2026). HIMS-CDI: A Governance Specification and Assurance Architecture for CTI-Driven Cybersecurity Risk Assessment in Healthcare. *Computers & Security* (under review).

For the CIC IoMT 2024 dataset, please cite:
> Dadkhah, S., et al. (2024). CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoT. *Internet of Things, 28*, 101351.

## License
Apache License 2.0
