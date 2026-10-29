# HIMS-CDI: Healthcare Cyber Situational Risk Assessment

This repository contains multiple historical and current project lines. A manuscript claim must be tied to a versioned package, its code and configuration, generated outputs, tests, and matching checksums—not inferred from a repository-level DOI alone.

## Current manuscript candidate: v0.2.16 (working / NO-GO for submission)

The current working manuscript uses the synthetic evaluation package `PhaseI_Synthetic_Evaluation_v0.2.9.zip`. A bounded local rerun on 29 October 2026 generated 240 synthetic environments and 7,200 simulated execution records; 15 environments passed all four governance gates; the package verification suite reported 33/33 checks passing. An independent rerun in the revision workspace reproduced 14 listed analysis/data files byte-for-byte; the manifest differed only in its `created_at_utc` value. The package SHA-256 manifest contained 21 entries, all matching.

These checks establish local reproducibility of the specified synthetic package only. They do not establish real-world healthcare cyber-risk reduction, clinical safety, legal compliance, live CTI utility, deployment effectiveness, or external replication.

**The exact v0.2.9 package supporting this manuscript candidate has not yet been deposited under a verified public archive DOI.** Do not cite the historical archive below as the source of these exact results without a version-to-version file, configuration, generator, output and checksum comparison.

## Historical archive — Zenodo v1.5.3

- **Historical DOI:** [10.5281/zenodo.23615735](https://doi.org/10.5281/zenodo.23615735)
- Preserve this DOI and its files as a historical record. Do not overwrite, repurpose or silently relabel it as the current v0.2.9 package.
- The original `main` README had described the archive as canonical for the manuscript. That claim was broader than the presently verified lineage and is the subject of the open draft pull request.
- The former source-branch label `v9.6-final-audit` was not found in the branch listing checked on 29 October 2026. A branch name is not adequate provenance unless resolved to an authenticated commit SHA or archived manifest.

## S4/S7 public GitHub snapshot (metadata only)

At the current `main` snapshot inspected on 29 October 2026 (commit `2bdee992ec086fb021ff40727a78ba3c5d6e7d73`), the repository tree listed:

- `supplementary_files/File S4 - Data_Extraction_Database.zip`: 13,898 bytes; Git blob SHA-1 `58af06e3f8f89549e5555f844d4d0a0d56529999`.
- `supplementary_files/File S7 - Included_Studies_Table.zip`: 44,331 bytes; Git blob SHA-1 `d4861a2e197831632af76db57ad99bade054c815`.

These are Git object identifiers and sizes, not SHA-256 file digests. The actual archive bytes must be downloaded at the pinned commit, recomputed and checked before a current-source crosswalk is treated as verified. A current GitHub archive is not automatically identical to the archive originally used in the published review.

## Repository structure and scope

- **`training_pipeline/`** — a separate 10-seed reference training pipeline. Its README explicitly says it is not a reproduction of the authors' original research code, and warns that the reproduced numbers differ from prior manuscript values. Do not treat it as the source of the 240-environment / 7,200-record v0.2.9 synthetic simulation unless a file-level comparison proves that relationship.
- **`phase-1-cic-iomt/`** — a separate companion-study package for the CIC IoMT 2024 dataset; it is not the current synthetic HIMS-CDI v0.2.9 study.
- **S4/S7 evidence review** — a literal GDPR/HIPAA mention is not evidence that a safeguard was implemented. The 8/43 diagnostic and historical 9/43 reported finding are not accepted as verified safeguard-prevalence estimates until the original coding definition, source passages and independent adjudication are recovered.

## Release and registration policy

Before a public release/DOI is associated with v0.2.16 or its supporting code:

1. Freeze the precise generator, configuration, dependencies, inputs, outputs and test results.
2. Create a **new, distinct, version-specific** archive; preserve all historical records.
3. Download the deposited archive back and compare its complete file manifest and SHA-256 checksums.
4. Confirm the deposited archive reproduces the reported figures using a documented, independent rerun.
5. Only then add that exact DOI to the manuscript and data-availability statement.

OSF project and registration identifiers must be treated as distinct records. Their dates, history, and registration status must be authenticated from the records themselves. Do not backdate, overwrite, or silently modify a registration.

Until these release gates pass, the current package should be described as locally reproducible with its public deposit pending. The current manuscript and this draft PR remain **not submission-ready**.

## Citation

Cite the HIMS-CDI manuscript according to its verified publication status and use a version-specific citation for each released artifact. For the separate CIC IoMT 2024 dataset, cite the dataset's own canonical publication and record.

## License

Apache License 2.0
