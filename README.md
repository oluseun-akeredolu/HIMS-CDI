
# HIMS-CDI: Healthcare Cyber Situational Risk Assessment

Research artefacts for the HIMS-CDI governance-first architecture (Akeredolu et al.).

## Repository Structure

- **`phase-1-cic-iomt/`** — Phase 1 empirical reproducibility package.
  Reproducible data processing and fusion calibration pipeline for the **CIC IoMT 2024** dataset.
  Contains scripts, calibration utilities, frozen artefacts, Dockerfile, and deployment config.
  See [`phase-1-cic-iomt/README.md`](phase-1-cic-iomt/README.md) for usage instructions.

- **Root package (`src/hims_cdi/`, `setup.py`)** — Legacy synthetic-data scoring utilities
  for the earlier HIMS-CDI specification validation (multinomial logistic regression baselines).
  These are **not** the subject of the Phase 1 empirical manuscript.

## Phase 1 Reproducibility

The Phase 1 pipeline is released under **Apache License 2.0** and archived on Zenodo:

- **GitHub:** `https://github.com/oluseun-akeredolu/HIMS-CDI/tree/main/phase-1-cic-iomt`
- **Zenodo:** `https://doi.org/10.5281/zenodo.21954311`

### Citation

If you use the Phase 1 pipeline or the CIC IoMT 2024 dataset, please cite:

&gt; Dadkhah, S., Pinto Neto, E. C., Ferreira, R., Molokwu, R. C., Sadeghi, S., & Ghorbani, A. A. (2024). *CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoMT*. **Internet of Things, 28**, 101351.

## License

Apache License 2.0