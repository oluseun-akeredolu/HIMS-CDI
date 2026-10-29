# HIMS-CDI v0.2.16 — Repository Evidence and Release Gate Audit

**Snapshot date:** 29 October 2026 (UTC)  
**Status:** Draft review evidence; not an approval to release or submit.

## Current state observed

- Repository: `oluseun-akeredolu/HIMS-CDI`; public default branch `main`.
- Observed `main` commit: `2bdee992ec086fb021ff40727a78ba3c5d6e7d73`.
- The repository's previous `main` README attributed the current manuscript's exact code/data/model artefacts to Zenodo DOI `10.5281/zenodo.23615735` (v1.5.3), and named `v9.6-final-audit` as the source branch.
- The current branch listing available at audit time did not include `v9.6-final-audit`.
- Open draft PR #2, `revision/v0.2.13-release-lineage-audit`, already proposed separating the historical v1.5.3 archive from the current locally tested synthetic package. This update carries the README/doc correction forward to the v0.2.16 candidate. It does not merge to `main`.

## Current main S4/S7 object references

| File | Size from GitHub tree | Git blob SHA-1 |
|---|---:|---|
| `supplementary_files/File S4 - Data_Extraction_Database.zip` | 13,898 bytes | `58af06e3f8f89549e5555f844d4d0a0d56529999` |
| `supplementary_files/File S7 - Included_Studies_Table.zip` | 44,331 bytes | `d4861a2e197831632af76db57ad99bade054c815` |

The SHA-1 values above are Git blob IDs, not SHA-256 digests of the ZIP bytes. The attached single-run Windows script retrieves the two ZIPs from the exact observed `main` commit, verifies the Git blob ID and size, computes SHA-256, and runs the read-only parser. If download, blob hash, size, parsing, or row-level matching fails, the result is reported as blocked/failed and must not be called a verified crosswalk.

## Evidence limits

- The source ZIPs in `main` are a current public snapshot; that alone does not prove which S4/S7 bytes were used in a past publication.
- A CSV/XLSX match establishes representation agreement, not truth of each coded field.
- The 9/43 reported value and 8/43 literal acronym-mention diagnostic are not accepted as safeguard prevalence. A concrete safeguard requires a predeclared code definition and source-passage evidence.
- S17/S18/S22 are provisional exclusion recommendations until double-screened eligibility decisions, source locators and adjudication are recorded.
- The v0.2.9 synthetic package's reproducibility checks must not be described as real-world validation.
- The existing Zenodo v1.5.3 archive and any OSF registration/project records are not modified by this PR.

## Safe release sequence

1. Review this documentation and run the one-run PowerShell automation from an authorised, networked Windows checkout.
2. Preserve generated logs and downloaded S4/S7 source snapshot; resolve row conflicts against the source passages and codebook.
3. Complete independent human screening and the claim-to-source audit.
4. Freeze the exact package and rerun it from a clean environment.
5. Create a new version-specific Zenodo deposit only after author approval and authenticated access. Download it back; compare all hashes; then insert its DOI.
6. Authenticate OSF project/registration metadata and describe chronology honestly. Do not change registration history to manufacture prospective registration.
7. Keep this pull request in draft until the evidence gates and author review are complete. Do not merge merely because the documentation builds.

## Audit decision

**NO-GO for release/submission.** The repository documentation is corrected on a draft branch to avoid attributing current synthetic outputs to an unverified historical archive. Data provenance, independent systematic-review screening, safeguard-level source adjudication, citation-to-claim verification and author declarations remain outstanding.
