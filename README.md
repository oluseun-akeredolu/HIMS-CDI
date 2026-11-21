# HIMS-CDI: Healthcare Cyber Situational Risk Assessment

This repository contains reproducibility artefacts from multiple HIMS-CDI research stages, including an earlier CDI/CCR evaluation and a revised four-gate governance-specification manuscript under review.

## Evidence provenance and version-specific source of truth

This repository supports more than one HIMS-CDI research/evaluation stream. **Do not treat a DOI from one experiment as the reproducibility archive for another.**

- **Historical CDI/CCR experiment:** The public [Zenodo v1.5.3 archive](https://doi.org/10.5281/zenodo.23615735), associated with branch `v9.6-final-audit`, documents the earlier CDI/CCR-based synthetic pipeline. It is the historical source for that experiment, **not** a version-matched deposit of the revised four-gate study.
- **Revised four-gate governance-specification manuscript (under review):** The submission-specific **Supplement S1** identifies the corrected `v0.2.23` generator and frozen `rerun_corrected/` files as the source of its reported Phase I figures. S1 separately preserves the historical `v0.2.9` source and a *subsequent* G3 boundary-handling candidate; the candidate did not generate the reported Phase I results. **Supplement S2** documents controlled HTTPS/TAXII-style transport and replay, and **Supplement S3** documents the accompanying critical evidence review.
- **Publication status:** The revised S1/S2 four-gate release has **no verified public DOI declared here**. The version-specific supplements should be provided with the revised submission; do not cite the historical CDI/CCR archive as the revised four-gate experiment's public source.

The archived checks establish bounded internal rule-conformance and controlled transport behavior; they do not establish independent clinical validity, inter-organisational operational exchange, or independent literature screening. If a new public four-gate archive is deposited, update this section with its *actual* DOI, version, source commit and matching SHA-256 manifest.

## Repository Structure

- **`training_pipeline/`** — The core synthetic Phase I evaluation pipeline for the HIMS-CDI manuscript.
  - Includes the shared governance gate function (`governance_gates.py`), the 31-test automated suite, and the strengthened verifier (`verify_outputs.py`).
  - See the README in this directory for instructions on running the 10-seed evaluation.

- **`phase-1-cic-iomt/`** — A separate empirical reproducibility package for the **CIC IoMT 2024** dataset. This is a companion study and is **not** the subject of the HIMS-CDI manuscript.

## Environment
The *historical* Zenodo archive records its experiment environment in `run_environment.json`. The revised four-gate study has separate version-specific Supplement S1/S2 requirements and a submitted reproducibility protocol.

## Citation
For the historical HIMS-CDI programme, the manuscript citation has been listed as follows; use the exact title and evidence release associated with the version being cited:
> Akeredolu, O., Epiphaniou, G., & Maple, C. (2026). HIMS-CDI: A Governance Specification and Assurance Architecture for CTI-Driven Cybersecurity Risk Assessment in Healthcare. *Computers & Security* (under review).

For the CIC IoMT 2024 dataset, please cite:
> Dadkhah, S., et al. (2024). CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoT. *Internet of Things, 28*, 101351.

## License
Apache License 2.0
