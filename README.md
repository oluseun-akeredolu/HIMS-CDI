# HIMS-CDI: Healthcare Cyber Situational Risk Assessment

This repository contains several historical and current artefact lines. Every manuscript result must be tied to the exact code, configuration, inputs, outputs, environment, tests and file manifest that produced it. A repository-level DOI is not, by itself, proof of that linkage.

## Current working manuscript candidate: v0.2.17 — NO-GO

The current working manuscript candidate (v0.2.17) reports results from the local package `PhaseI_Synthetic_Evaluation_v0.2.9.zip`.

The bounded rerun completed on **30 October 2026**:
- 240 synthetic environments and 7,200 simulated execution records;
- 15/240 environments (6.25%) passed all four conjunctive governance gates;
- 1,296/7,200 execution records (18.00%) had at least one resource-threshold exceedance;
- the full-input logistic-regression comparator had mean balanced accuracy 0.803556 (sample SD 0.026896) across ten repeated cross-validation runs. Because synthetic labels are generated from the declared rule and the comparator receives features derived from gate components, this is a rule-recovery diagnostic—not independent predictive validation.
- all 14 deterministic generated output/data files matched the packaged counterparts byte-for-byte; all 21 SHA-256 manifest entries and all 33 packaged verification checks passed.

The regenerated `manifest.json` differs from the archived manifest in **two environment-metadata fields**: `created_at_utc` and `runtime.platform`. The latter changed from `Linux-6.18.55-x86_64-with-glibc2.41` to `Linux-6.18.57-x86_64-with-glibc2.41`. The exact local package ZIP SHA-256 is `b0052e38da33b5733dcfb4b829a5472f4d15a109a8a67d6df7b053aa8ca5bafe`.

These checks establish bounded local reproducibility of this specific synthetic package. They do not establish clinical safety, healthcare incident prediction, legal compliance, live CTI exchange, external replication, operational effectiveness or real-world deployment readiness. The historical generator and claimed empirical dependence matrix were not recovered; v0.2.9 is a new, explicitly specified simulation, not a reconstruction of the earlier experiment.

**The exact v0.2.9 package has not yet been verified in a version-specific public archive. Do not cite an older Zenodo DOI as the source of these exact results.**

## Separate public artefact records

- **Zenodo 10.5281/zenodo.23615735 (v1.5.3):** preserve as a historical archive. The older root README mapped this record to the manuscript's canonical code/data and named `v9.6-final-audit` as its source branch. That broad linkage is not established for the current v0.2.9 package. The branch name was not present in the branch listing inspected on 29 October 2026.
- **Zenodo 10.5281/zenodo.23624050 (v1.5.4):** a separate training-pipeline release candidate; its deposited filename includes `release_candidate`. Its record description says “three” gates while listing four threshold clauses (CDI, CCR, latency and memory). Treat the predicate as four gates and correct the record description only in a new version after the archive is rechecked. Do not present this record as the v0.2.9 deposit.
- **Zenodo 10.5281/zenodo.23550335 (v1.3.0):** a separate synthetic-data/generator release, with its own correlation matrix, generator and train/validation/test data. Its description also says “three” non-compensatory gates but lists four criteria. The matching corpus size does not prove identity with v0.2.9.
- **`training_pipeline/`:** a separate 10-seed reference pipeline. Its own README states that it is not a reproduction of the authors' original research code and that its reproduced results differ from earlier manuscript values. Do not attribute these pipeline outputs to v0.2.17 without a complete file- and run-level provenance comparison.
- **`phase-1-cic-iomt/`:** a separate CIC IoMT 2024 companion-study line, not the v0.2.9 synthetic experiment.

## S4/S7 public GitHub snapshot (metadata only)

At the `main` snapshot inspected on 29 October 2026 (commit `2bdee992ec086fb021ff40727a78ba3c5d6e7d73`), the tree listed:
- `supplementary_files/File S4 - Data_Extraction_Database.zip`: 13,898 bytes; Git blob SHA-1 `58af06e3f8f89549e5555f844d4d0a0d56529999`.
- `supplementary_files/File S7 - Included_Studies_Table.zip`: 44,331 bytes; Git blob SHA-1 `d4861a2e197831632af76db57ad99bade054c815`.

These are Git blob object IDs and sizes, not SHA-256 digests of the downloaded ZIP bytes, and they do not prove equivalence to the files used in the published review. Download at a pinned commit and verify bytes, then complete the row-level, source-passage crosswalk. The 8/43 literal-mention diagnostic and historical 9/43 reported figure are both withdrawn as verified safeguard-prevalence estimates pending predeclared coding and adjudication.

## Release and registration policy

1. Freeze the precise code, configuration, dependencies, inputs, outputs, tests and manifest.
2. Obtain all-author approval and create a distinct version-specific archive without changing historical records.
3. Download the new deposit back and verify the complete manifest and SHA-256 hashes.
4. Re-run the deposited package in a documented clean environment.
5. Insert the exact DOI only after the download-back and reproduction checks pass.
6. Authenticate OSF project and registration records, dates and version history directly. Never backdate, overwrite or silently alter registration history.

## Status

The v0.2.17 manuscript and this draft pull request remain **NO-GO for submission/release**. The working revision is not a certification of journal eligibility or acceptance.

## Citation and licence

Cite each manuscript and software/data artefact according to its verified publication status and exact version. The separate CIC IoMT 2024 package must be cited using its own canonical record. Confirm the applicable licence and all contributors' approvals before a new deposit.
