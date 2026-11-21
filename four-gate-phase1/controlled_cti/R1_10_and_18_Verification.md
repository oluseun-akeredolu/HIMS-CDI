# Verification for Reviewer 1 Comments 10 and 18

## Threshold rationale and applicability

| Rule or input | Origin and decision effect |
| --- | --- |
| G1 scope/applicability | A categorical scenario state in Phase I. In use, the applicable versioned local registry and scoped decision determine the evidence required. |
| G2 control proxies | The author-selected cut-off of 0.05 on each of six simulated indicators generates both PASS and FAIL item states. Evidence degradation can change a passing item to UNKNOWN. |
| G3 latency and memory | The author-selected strict limits of 200 ms p95 and 100 MB observed peak are interior points in the 20-setting simulation grid. The code evaluates native units against the declared requirements. Equality or an exceedance fails the gate. A missing or invalid measurement yields UNKNOWN absent a verified breach. |
| G4 additional requirement proxies | The author-selected cut-off of 0.15 on each of two simulated indicators generates distinct item states. In use, the source of any veto is the applicable policy or requirement. |
| Post-gate preferences | Ordinal dominance applies after all gates pass. A separately elicited outranking rule could consider trade-offs between advisory dimensions while retaining mandatory admission rules. |

The 17 archived G3 cases were run again through the candidate implementation. They include latency and memory equality, missing or non-finite telemetry, and a verified breach alongside one missing value. On the stored grid, at 100 MB, moving the latency limit from 175 to 200 to 250 ms yields 12, 15 and 24 eligible synthetic environments. At 200 ms, moving the memory limit from 64 to 100 to 150 MB yields 5, 15 and 19. Those counts describe the tested simulation.

For deployment, the example workload begins when the HIMS intake receives a CTI record and ends when the evidence-request record reaches the designated review queue. The local workflow owner supplies the approved percentile and permitted delay. The platform owner supplies the tested allocation, workload and memory-measurement scope. The deployment evidence record would identify the software and hardware versions, timing endpoints, load, concurrency, number of observations, raw latency and memory observations, missing runs and the applicable boundary rule. The package contains the corresponding simulation and regression records. It contains no site-approved limits or physical deployment measurements.

## Controlled CTI exchange

The prototype implements a limited subset of TAXII 2.1 collection endpoints. It bound an authenticated HTTPS server to loopback and ran separate clients for a sender, a receiver and a role without collection access. Each authorised client used its own generated credential. The clients validated the ephemeral TLS certificate. The credentials and private key were not retained.

Two publications were accepted (HTTP 202), their status resources were checked, and both were retrieved by the receiver. The negative checks were an anonymous read (401), a read by the role outside the collection (403), a sender attempting to read (403), and a receiver attempting to publish (403). Each authorised retrieval was compared with the objects prepared by the sender. The transport logs retain the object ID and version, actual timestamps and response hashes. The programme uses a separate declared time for the freshness scenarios.

A predefined mapping associates the synthetic indicator with a demonstration patient portal. On an authorised receipt, an active and applicable indicator led to a request for local evidence. The receiver-to-gate fixtures covered missing, satisfied and failed local requirements, an expired indicator and a retrieved revoked version. The mock evidence state supplied the G2 decision. The remaining gates were fixed at PASS. The five cases matched the expected intake state, G2 state and final outcome. The earlier 11-case replay adds checks for a changed payload, a missing creator, an unverified asset link and malformed input.

This was one controlled host using synthetic objects, assigned roles, a predefined asset match and mock evidence. The run does not establish general standards conformance, automatic ATT&CK mapping, healthcare deployment performance or real organisational exchange. A live claim would need organisations’ authorised access, receipt and evidence-adjudication records.

- [OASIS TAXII 2.1, §§4.3 and 5.4–5.5](https://docs.oasis-open.org/cti/taxii/v2.1/os/taxii-v2.1-os.html)
- [OASIS STIX 2.1, Common Properties and §4.7](https://docs.oasis-open.org/cti/stix/v2.1/os/stix-v2.1-os.html)

## Reproduction

Use Python 3.12 with the packages in `requirements.txt` and OpenSSL on the PATH. In this folder run `python taxii_exchange.py` and `python replay.py`. The first command writes a new `taxii_exchange_results.json`. Its timestamps and status UUIDs vary across runs. Expected HTTP codes, object/version relationships and gate states are asserted inside the program. The second command runs the original offline fixtures, Gate 3 controls, sensitivity-summary checks and the documentary case from v0.2.66.
