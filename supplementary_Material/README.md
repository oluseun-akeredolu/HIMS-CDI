# Supplementary Materials (S1-S22)

## About This Folder

These supplementary files support the manuscript:

**HIMS-CDI: A Governance Specification and Assurance Architecture for CTI-Driven Cybersecurity Risk Assessment in Healthcare**

- **Pre-print:** https://ssrn.com/abstract=7042047
- **DOI:** http://dx.doi.org/10.2139/ssrn.7042047

## What's Inside

This folder contains supplementary materials including:
- Data files and datasets
- Analysis scripts and code
- Protocol documents
- Supporting reference materials
- Validation and testing artifacts

All files are organized by material identifier (S1, S2, S3... S22).

## Related Resources

- **GitHub Repository:** https://github.com/oluseun-akeredolu/HIMS-CDI
- **Archival Copy (Zenodo):** https://doi.org/10.5281/zenodo.23550335
- **Open Science Framework:** https://osf.io/86bdg/

> **Note (corrected 2026-10-22):** v1.3.0 of this README pointed the "Archival Copy (Zenodo)" link to https://zenodo.org/records/16897633, which is the DOI of an earlier archive version. The current synthetic-data archive is at https://doi.org/10.5281/zenodo.23550335 (HIMS-CDI Synthetic Data Archive v1.3.0/1.3.1). The link has been updated.

## Supplementary-file corrections required

The following supplementary PDFs and CSVs contain stale values that should be updated to match the v1.3.1 archive and the v6 manuscript. See `supplementary_pdf_corrections.md` in the archive corrections pack for the full list.

- **S2_Synthetic_Data_Generator.pdf** and the S2 directory `synthetic_data_generator.py` (54-line placeholder) should be replaced with the production generator from the Zenodo archive (506 lines, beta+categorical marginals, sigma = 0.13).
- **S4_corpus_statistics.csv** describes a placeholder uniform-marginal corpus and should be replaced with the corrected `S4_corpus_statistics_corrected.csv` (recomputed from the archived production corpus).
- **S10_ROC_Curves.pdf**, **S2_Synthetic_Data_Generator.pdf**, and **S22_Comprehensive Statistical I.pdf** cite sigma = 0.10 in places; these should be updated to sigma = 0.13 to match the production generator's `NOISE_SIGMA` constant.
- **S22_Comprehensive Statistical I.pdf** sweep grid `{0.05, 0.10, 0.15, 0.20}` should be reconciled with the manuscript's stated range `0.07 to 0.20` plus `0.25`.

## Citation

If you use these materials, please cite the manuscript and include a reference to this repository.
