# HIMS-CDI: Healthcare Cyber Situational Risk Assessment

This repository contains multiple historical and current project lines. The source of record for a particular manuscript claim must be identified by a versioned package, configuration, generated outputs, and matching checksums—not inferred from a repository-level DOI alone.

## Current manuscript candidate: v0.2.13 (working / not submission-ready)

The current manuscript candidate uses the locally tested package `PhaseI_Synthetic_Evaluation_v0.2.9.zip`. A fresh local rerun on 29 October 2026 generated 240 synthetic environments and 7,200 simulated execution records; 15 environments passed all four gates, and the packaged suite passed 33/33 checks. This verifies the behaviour and repeatability of that local synthetic package only; it does not validate clinical safety, real-world cyber-risk reduction, legal compliance, or deployment effectiveness.

**The exact v0.2.9 package supporting this manuscript candidate has not yet been deposited under a verified public archive DOI.** Do not cite the historical archive below as the source of these specific results until the actual files, generator, configuration and hashes have been compared.

## Historical archive — Zenodo v1.5.3

- **Historical DOI:** [10.5281/zenodo.23615735](https://doi.org/10.5281/zenodo.23615735)
- This DOI is retained as an immutable historical record. Its scope should be read from that release's actual contents and manifest.
- The older README described this as the canonical package for the manuscript. That wording is too broad for the current v0.2.13 / v0.2.9 candidate and is corrected here so the current synthetic results are not conflated with a differently specified historical pipeline.
- The former source-branch label `v9.6-final-audit` was not resolvable in the current branch/ref lookup performed on 29 October 2026. A branch name alone is not sufficient provenance; resolve it to a commit SHA or archive manifest before relying on it.

## Repository structure and scope

- **`training_pipeline/`** — A separate 10-seed reference training pipeline. Its own README states that it is a reference implementation, not a reproduction of the original research code, and documents a different set of inputs/configuration. Do not treat it as the source of the 240-environment / 7,200-record v0.2.9 simulation unless a file-level comparison proves that relationship.
- **`phase-1-cic-iomt/`** — A separate companion-study package for the CIC IoMT 2024 dataset; it is not the current synthetic HIMS-CDI v0.2.9 study.

## Release policy for the current candidate

Before associating a public release or DOI with manuscript v0.2.13, the author team must:

1. Freeze the exact generator, configuration, dependencies, input package, outputs and tests.
2. Create a distinct version-specific archive without altering historical releases.
3. Download that archive back and verify its manifest and checksums.
4. Confirm that the deposited files reproduce the manuscript's reported results.
5. Update the manuscript data-availability statement only after those checks pass.

Until then, report the v0.2.9 result as locally reproducible and the corresponding public deposit as pending. Preserve all historical branches, tags, releases and DOI records.

## Citation

For the HIMS-CDI architecture or synthetic pipeline, please cite the manuscript only according to its verified publication status and the version-specific instructions in the release actually used.

For the separate CIC IoMT 2024 dataset, cite:
> Dadkhah, S., et al. (2024). CICIoMT2024: A benchmark dataset for multi-protocol security assessment in IoT. *Internet of Things, 28*, 101351.

## License

Apache License 2.0
