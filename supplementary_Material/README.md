# Supplementary Material

This directory contains the supplementary materials for the manuscript:
**HIMS-CDI: A Governance Specification for Healthcare Cybersecurity Risk Assessment: A Synthetic Phase I Evaluation**

## Updated Archive
All supplementary code, data, and results have been packaged and published on Zenodo to ensure reproducibility. The paper-specific pipeline archive is identified in the manuscript as **HIMS-CDI Synthetic Data Archive v1.5.4** (Zenodo DOI `10.5281/zenodo.23624050`). 
The previous version (v1.5.3, DOI `10.5281/zenodo.23615735`) is superseded but remains available for provenance.
The synthetic corpus is identified separately at Zenodo DOI `10.5281/zenodo.23565580`.

## Resource Measurements
The manuscript describes a 520.5 MB Python-process RSS observation under a shared-process protocol. Isolated-subprocess benchmarks (90 measurements: 9 models × 10 seeds, each in a fresh Python process) found HIMS_CDI_cal whole-process RSS of 333.0 ± 2.3 MB — still exceeding the 100 MB gate by 3.3×. No incremental footprint was inferred by subtracting an assumed interpreter baseline. Results are in the v1.5.4 archive at `results/isolated_benchmark/`.

## Additional Sensitivity Analysis
The revised manuscript discusses additional exploratory weight, threshold, and ordinal-mapping sensitivity results. The actual analysis scripts (`weight_sensitivity.py`, `threshold_sensitivity.py`, `ordinal_mapping_sensitivity.py`) and the raw JSON output logs are included in this repository and in the new Zenodo release (v1.5.4).
