# HIMS-CDI verification evidence carried forward from v0.2.66

## Simulation thresholds and deployment evidence

The Phase I model applies strict limits to p95 latency and peak memory. The nominal 200 ms and 100 MB settings are interior grid points for controlled sensitivity and exact-boundary testing. The 20-point archived grid ranges from 2 to 32 eligible environments, with 15 under the reference settings. At 100 MB, 175, 200 and 250 ms give 12, 15 and 24 eligible cases. At 200 ms, 64, 100 and 150 MB give 5, 15 and 19. These are counts from the synthetic candidate.

`replay.py` imports the supplied G3 candidate and re-executes 17 archived controlled cases including equality at either limit, absent or malformed telemetry, and a verified breach with the other measure absent. It also checks the archived 20-point sensitivity summary. This check uses the stored grid and the boundary implementation. Regeneration of the Phase I corpus is a separate workflow.

The deployment decision would need a dated latency objective and memory allocation approved by the appropriate local owners. For the proposed mammography benchmark, first define the workload as receiving a screening examination from the picture archiving and communication system, processing it on the specified version and hardware, and making the completed result available to the reading workflow. Ask the workflow owner to confirm the timing endpoints, intended concurrency and acceptable delay, and the platform owner to state the memory measurement scope and limit. Then preserve the workload manifest and raw measurements, including missing outcomes and the observation count. The verification checks supplied here do not supply those local records or measurements.

## Offline CTI replay

The replay runs a synthetic STIX 2.1 bundle for a demonstration patient portal against 11 predeclared outcomes. The input uses a reserved domain name and fictional organisations and controls. A local policy fixture models recipient grants. Another fixture models trusted receipt metadata and the expected whole-payload digest. Supplied asset and local evidence states are mock assessment inputs. The adapter retains provenance fields, classifies the intake, and hands the verified local evidence state into the existing gate conjunction. A current and relevant indicator produces an evidence request. Local evidence remains a separate input.

The cases cover expiry at the validity boundary, revocation, recipient denial, missing creator identity, changed input bytes, unverified asset link, invalid JSON, and verified, missing or failed local control evidence. All 11 intake classifications, G2 states and final dispositions matched the authored expectations. `replay_results.json` records the result, original input hash and receipt hash, object and creator identifiers where parseable, dates, marking and local asset where verified in the fixture. `replayed_payloads/` holds the exact bytes tested. The prototype reads only the fields needed for these fixtures. Its checks do not establish general STIX conformance, authenticate an external sender, execute TAXII transport, map ATT&CK techniques or test real organisational permissions. It uses no patient data or healthcare CTI. The fixture expectations and prototype are author-side work with AI assistance.

- [OASIS STIX 2.1, Common Properties and §4.7 Indicator](https://docs.oasis-open.org/cti/stix/v2.1/os/stix-v2.1-os.html)
- [OASIS TAXII 2.1](https://docs.oasis-open.org/cti/taxii/v2.1/os/taxii-v2.1-os.html)

## Documentary healthcare case

`documentary_case.json` maps the public report about Capio S:t Göran Hospital’s mammography screening workflow to the four gates. The article documents the setting and identified risks and reports proposed mitigations. For the defined admission assessment, the public packet lacks sufficient local records for each gate, so the imported gate conjunction returns UNRESOLVED. This is a documentary application of the rule with an explicit evidence boundary. It gives no finding about how the hospital itself adjudicated deployment and includes no independent case ratings. The indicated gaps are the inputs needed for such an assessment.

- [Gerigoorian et al. 2025, Methods and Results](https://pmc.ncbi.nlm.nih.gov/articles/PMC12309172/)

## Reproducing this check

From the `verification` folder on Python 3.12, install the listed package versions in an isolated environment when they are not already available. Run `python replay.py`. The command regenerates the replay outputs and checks the G3 cases, the archived sensitivity grid and the documentary-case decision. The provided Phase I candidate is imported for its decision functions. The command does not run the Phase I generator. `SHA256SUMS.txt` at the root of the bundle lists the hashes for the provided files.
