# HIMS-CDI Supplementary Materials

This directory contains supplementary material for the HIMS-CDI research programme. The repository contains more than one historical and current artefact line. File names, version labels, and shared dataset sizes must not be used as substitutes for a version-to-version provenance check.

## Current manuscript candidate: v0.2.16

The current working manuscript candidate uses the synthetic evaluation package `PhaseI_Synthetic_Evaluation_v0.2.9.zip`.

The current local evidence record reports a bounded rerun on 29 October 2026 that generated 240 synthetic environments and 7,200 simulated execution records. Fifteen environments passed all four governance gates. The package verification suite reported 33/33 checks passing. An independent rerun reproduced 14 listed analysis/data files byte-for-byte; the manifest differed only in its `created_at_utc` value, and all 21 entries in the SHA-256 manifest matched.

These are local reproducibility checks for the specified synthetic package. They do not establish real-world healthcare incident validity, clinical safety, legal compliance, live CTI utility, operational effectiveness, or deployment readiness.

**The exact v0.2.9 package has not yet been deposited under a verified public archive DOI.** Do not cite any historical Zenodo DOI below as the source of these exact v0.2.9 results unless a version-to-version comparison establishes identity of the generator, configuration, dependencies, inputs, outputs, and SHA-256 manifest.

## Separate historical and reference artefact lines

### Zenodo 10.5281/zenodo.23624050 — version 1.5.4

The public Zenodo record is labelled version 1.5.4 and lists one uploaded file: `training_pipeline_v1.5.4_release_candidate.zip` (MD5 `8ec9efe975fa502800ddf850c1c20973`). The word `release_candidate` is part of the deposited filename and must not be silently removed from descriptions.

This is the separate 10-seed reference training-pipeline line. The GitHub `training_pipeline/README.md` states that this implementation is not a reproduction of the authors' original research code and that its reproduced numbers differ from previously reported manuscript values. No verified file-level comparison currently establishes that this archive is the current `PhaseI_Synthetic_Evaluation_v0.2.9.zip` package. Therefore, this DOI must not be cited as the v0.2.9 package DOI.

The public record description also says "three non-compensatory governance gates" while listing four criteria (CDI, CCR, latency, and memory). This description must be corrected in a new Zenodo version after the deposited archive and all metadata have been independently checked. The existing record must remain intact.

### Zenodo 10.5281/zenodo.23615735 — version 1.5.3

Preserve this as a historical archive. Do not overwrite, repurpose, or describe it as the canonical source of the v0.2.9 package without a complete file-level provenance comparison.

### Zenodo 10.5281/zenodo.23550335 — synthetic-data archive v1.3.0

This is a separate 240-environment, 20-indicator synthetic-data/generator release with its own generator, correlation matrix, and train/validation/test assets. Its matching row count does not establish identity with the v0.2.9 package, which also records 7,200 simulated execution records. Treat the two artefact lines separately unless complete file and checksum comparisons prove otherwise.

## Resource measurements and exploratory analyses

The 520.5 MB shared-process RSS observation, the 333.0 ± 2.3 MB isolated-process observation, and the weight-, threshold-, and ordinal-mapping sensitivity outputs belong to their specific training-pipeline or analysis runs. Do not transfer these values into the v0.2.16 manuscript merely because related scripts or outputs exist in the repository. Each reported statistic must be tied to its own raw output, code/configuration version, environment, and exact manuscript table or sentence.

The isolated-process measurement is above the 100 MB gate; it must not be presented as passing that gate. Do not infer an incremental model footprint by subtracting an assumed interpreter baseline.

## S4/S7 evidence and systematic-review screening

The S4/S7 extraction files are evidence for a separate literature-review component. Their current GitHub object identifiers and file sizes are metadata only; the ZIP bytes must be downloaded from a pinned commit and checked by size, Git blob verification, and SHA-256 before a row-level crosswalk is treated as validated.

A literal mention of GDPR, HIPAA, or another safeguard is not proof that the safeguard was implemented. The 8/43 literal-mention diagnostic and historical 9/43 reported value are not verified safeguard-prevalence estimates. Do not restore either as a prevalence claim without a predeclared coding definition, source-passage locators, and independent adjudication.

Eligibility decisions and exclusions must be supported by the original records and documented screening decisions. Do not describe independent two-person screening or inter-rater reliability as completed unless the underlying paired decisions and adjudication record are available.

## GitHub, Zenodo, and OSF release policy

1. Preserve historical GitHub commits, branches, tags, releases, and Zenodo records. Do not rewrite history or silently change a historical DOI's meaning.
2. Keep the v0.2.16 manuscript, the v0.2.9 synthetic package, and the 10-seed reference training pipeline as distinct artefact lines until file-level provenance proves a relationship.
3. Freeze the exact current generator, configuration, dependencies, inputs, outputs, tests, and manifest before creating a new public archive.
4. Deposit a new, version-specific Zenodo record only after author approval. Download the deposited file back and compare the complete file manifest and SHA-256 hashes before inserting its DOI into the manuscript.
5. Authenticate OSF project and registration identifiers, dates, version history, and registration status directly from the relevant OSF records. A public-repository search that finds no OSF URL does not prove that no OSF record exists. Never backdate, overwrite, or silently modify registration history.
6. Do not merge the draft release-lineage pull request or describe this manuscript as submission-ready until all evidence gates, author declarations, reference checks, and independent reviews are complete.

## Status

The current v0.2.16 package has a reported local reproducibility check, but its exact public deposit remains pending. S4/S7 source-level verification, independent screening, safeguard adjudication, OSF chronology authentication, and complete citation-to-claim verification remain open. These limitations must be disclosed, not hidden.

## Licence

Apache License 2.0
