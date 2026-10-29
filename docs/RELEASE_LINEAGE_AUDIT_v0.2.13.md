# HIMS-CDI public release lineage audit — 29 October 2026

**Purpose:** clarify which public references can support the current v0.2.13 manuscript candidate without altering historical repository or archive records.

## Public GitHub facts inspected

- Repository: `oluseun-akeredolu/HIMS-CDI`; default branch: `main`.
- The root README cited Zenodo DOI `10.5281/zenodo.23615735` (v1.5.3) as the canonical archive for manuscript code/data and named branch `v9.6-final-audit`.
- The branch/ref lookup did not resolve `v9.6-final-audit` on 29 October 2026. A named branch reference therefore cannot currently support a provenance claim until its commit or archive identity is demonstrated.
- `training_pipeline/README.md` describes a separate 10-seed pipeline and says it is a reference implementation, not the original research code. It documents different data/configuration/outputs from the new four-gate v0.2.9 package.

## Current local synthetic candidate

- Candidate package: `PhaseI_Synthetic_Evaluation_v0.2.9.zip`.
- SHA-256 of the local package inspected: `b0052e38da33b5733dcfb4b829a5472f4d15a109a8a67d6df7b053aa8ca5bafe`.
- Fresh local rerun: 240 environments; 7,200 simulated execution records; 15 observed gate-pass environments (6.25%); 1,296/7,200 simulated execution records with at least one resource-threshold exceedance (18.00%); 33/33 packaged specification checks passed.
- 14 deterministic generated analysis files matched packaged counterparts byte-for-byte. The manifest differs because it includes run metadata.
- These facts support local reproducibility of the included synthetic package. They do not establish that this exact package is already publicly archived, reproduce the unrecovered historical run, or demonstrate real-world healthcare effectiveness.

## Action taken by this pull request

This branch proposes a correction to root `README.md` so:
- the v1.5.3 DOI remains identified as a historical archive;
- the current manuscript candidate and local v0.2.9 package are identified separately;
- the current package is not falsely claimed to be supported by the historical DOI;
- the training pipeline is not conflated with the current synthetic candidate.

No release, tag, Zenodo deposition, OSF edit, historical DOI change or main-branch merge is performed by this pull request.

## Remaining release gates

1. Freeze the approved v0.2.9 package, generator, config, dependencies and outputs.
2. Authenticate historical branch/tag mappings and compare each public archive's actual contents/manifests.
3. Publish a distinct approved archive for the current package without overwriting history.
4. Download the deposited file, recompute SHA-256, and rerun the package from a clean environment.
5. Verify OSF project/registration IDs, timestamps and file inventory using authenticated account metadata. Submitted registrations must not be edited or backdated.
6. Insert the new public archive identifier into the manuscript only after the above checks pass.

## Review limitation

The public README correction is a release-lineage safeguard, not evidence that the related systematic review's S4/S7 cohort or 8/43 versus 9/43 regulatory-coding discrepancy has been adjudicated. Those remain separate source-level research-integrity tasks.
