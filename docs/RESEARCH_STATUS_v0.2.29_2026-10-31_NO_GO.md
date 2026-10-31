# HIMS-CDI v0.2.29 — Evidence status and repository release gates
Date: 31 October 2026. Status: **MAJOR REVISION / NO-GO — draft research record, not a release**.

This addendum updates the working evidence status of [draft PR #2](https://github.com/oluseun-akeredolu/HIMS-CDI/pull/2). It does **not** change the historical review data, reproduce clinical results, establish external validity, or authorise merging. It supersedes earlier *status summaries*, not the original frozen datasets or analyses.

## Research version separation

| Source | Status | SHA-256 |
|---|---|---|
| Historical Phase I v0.2.9 | Frozen; exactly reproduced, but implementation has a truncation defect | `b0052e38da33b5733dcfb4b829a5472f4d15a109a8a67d6df7b053aa8ca5bafe` |
| Corrected Phase I candidate v0.2.23 | Unpublished, separate exploratory corrected candidate; original unchanged | `f9d803a53169ba8e0e66ed9c7abc45451f7136aac2b5c640b6c711bcd1fa866d` |
| Historical review S4 ZIP | Frozen, pending source-level adjudication | `70047e0ddf69e527ec8af4ced3f7cf52adecd1a1b671ef3de24b53a16c523bcb` |
| Historical review S7 ZIP | Frozen, pending source-level adjudication | `088711ea7f6db0a98877bf67b933aa0eb48d125be0bffbce88808a8deb844938` |

These SHA-256 hashes identify the locally audited archive bytes; this note does not assert the files have been newly uploaded into this PR. The existing Zenodo archive records a **legacy CDI/CCR-based study** and MUST NOT be presented as the corrected four-gate Phase I version. Obtain and verify a *version-specific* public DOI only after authorised deposit and download-back checks.

## Verified local findings, carefully bounded

- The frozen Phase I run was independently re-executed and its 14 deterministic scientific outputs matched byte-for-byte. A NumPy string-width defect truncated 241 full `UNKNOWN` item-status values to `UNKN`; original full-input comparator metrics are affected.
- Separate corrected-candidate experiments retain 240 synthetic environments and three-way outcomes: 15 `ELIGIBLE`, 177 `INELIGIBLE`, 48 `UNRESOLVED`. All four new canonical-status regression checks fail on the defective original and pass on the corrected candidate.
- Independent finite-state reference evaluation checked all 81 four-gate combinations, 59,049 nested categorical combinations, 240 corrected environment decisions and 1,920 item states. These checks establish **categorical rule conformance on simulated evidence**, NOT real healthcare effectiveness.
- The corrected generator's flat CSV does NOT meet the 15 required contextual/provenance fields in the operational assessment-record schema. End-to-end interoperability has not been demonstrated.
- The 17 October 2026 *Computers & Security* decision calls for responses to 18 Reviewer 1 comments and Reviewer 2's synthetic-only concern. The replacement v0.2.29 response has been checked against the 19 original comments. A prior draft wrongly substituted login instructions for Reviewer 1 comments 1–4; it must not be submitted.
- Bibliography inventory: 62 entries. 21 have bounded identity/scope review; 16 limited source-claim scope checks exist. These are **not** a full source-passage audit or an independent human audit.
- Original submission reported `κ = 0.84` for 50 HIMS assessments and six-expert AHP numbers. Supporting authentic rater-level and elicitation records are not yet authenticated. Do NOT infer the historical kappa came from the separate 26-item citation task.
- The related systematic review's historic 43-study register is retained as reported, **not** independently certified. S17/S18/S22 protocol decisions, unmatched S4/S7 identities and 8/43 versus 9/43 coding are unresolved. Do NOT assert a validated 40-study cohort.

## Open independent verification gates

1. Full primary-source passage checks for every load-bearing manuscript claim; independent verification of the 62 bibliography entries and all citation relationships.
2. Authenticated protocol-based cohort and regulatory-coding adjudication; no silent changes to historical published review.
3. Genuine independently dual-screened review requested by Reviewer 1; the current exploratory literature map is not that review.
4. Actual two-human HIMS evidence-packet ratings before any inter-rater coefficient can be claimed.
5. Real-world healthcare/CTI testing or sharply bounded unvalidated design proposal; no fabricated live deployments.
6. Coauthor approval, independent computational source review, journal scope guidance, full manuscript/response reconciliation.
7. Authorised Zenodo/OSF updates with accurate links, immutable historical records, and hash/DOI verification.

**PR handling:** Keep this PR **draft / NO-GO**. Do not merge into `main`, retag or overwrite historical releases, claim a new DOI, or announce submission readiness. Reassess after the independent scientific checks and coauthor authorisation.

Journal major-revision deadline in the editorial letter: **16 November 2026**; request an extension if needed.
