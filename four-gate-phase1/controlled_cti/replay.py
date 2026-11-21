"""Offline specification prototype. Uses simulated receipts, policy and local evidence."""
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import csv
import json
import math

from phase1_v023_G3_edge_case_candidate import conjunctive_status, resource_gate_state

HERE = Path(__file__).resolve().parent
DISPOSITION = {"PASS": "ELIGIBLE", "FAIL": "INELIGIBLE", "UNKNOWN": "UNRESOLVED"}
RULESET = "CTI-OFFLINE-DEMO-1"
ASSET = "DEMO-HOSPITAL-PATIENT-PORTAL"
CREATOR = "identity--fca39a73-4183-4c5b-a9a4-de5f80d595dd"
MARKING = "marking-definition--2f1c8c18-3db6-4867-887a-f1368aeda357"
PROVIDER = "demo-sector-sharing-org"
GRANTS = {MARKING: {"demo-hospital-security"}}

def digest(data):
    return sha256(data).hexdigest()

def encode(obj):
    return (json.dumps(obj, indent=2) + "\n").encode()

def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def classify(raw, receipt, at, asset, state):
    assert state in DISPOSITION
    audit = {"ruleset": RULESET, "provider": receipt["provider"], "recipient": receipt["recipient"],
             "received_at_utc": receipt["received"], "assessment_at_utc": at,
             "input_sha256": digest(raw), "receipt_sha256": receipt["sha256"],
             "source_object_id": None, "created_by_ref": None, "source_modified": None,
             "valid_from_utc": None, "valid_until_utc": None, "markings": [], "local_asset": None}
    if digest(raw) != receipt["sha256"]:
        reason = "integrity_mismatch"
    elif receipt["provider"] != PROVIDER:
        reason = "provider_unverified"
    else:
        reason = inspect_bundle(raw, receipt["recipient"], at, asset, audit)

    # This mock local item represents a separate, verified control review when PASS or FAIL.
    # A CTI alert can trigger a request for it. The alert alone supplies neither state.
    g2 = conjunctive_status([state])
    overall = DISPOSITION[conjunctive_status(["PASS", g2, "PASS", "PASS"])]
    return {**audit, "cti_status": reason, "local_evidence_state": state,
            "local_evidence_ref": {"PASS": "FIXTURE-LOCAL-VERIFIED", "FAIL": "FIXTURE-LOCAL-BREACH"}.get(state),
            "requested_control_evidence": reason == "request_local_control_evidence",
            "G1": "PASS", "G2": g2, "G3": "PASS", "G4": "PASS", "disposition": overall}

def inspect_bundle(raw, recipient, at, asset, audit):
    try:
        bundle = json.loads(raw)
        if bundle["type"] != "bundle":
            return "unsupported_format"
        objects = {obj["id"]: obj for obj in bundle["objects"]}
        items = [obj for obj in objects.values() if obj["type"] == "indicator" and obj["spec_version"] == "2.1"]
        if len(items) != 1:
            return "unsupported_format"
        obj = items[0]
        ref = obj.get("created_by_ref")
        marks = obj.get("object_marking_refs", [])
        audit["source_object_id"] = obj["id"]
        audit["source_modified"] = obj["modified"]
        audit["created_by_ref"] = ref
        audit["markings"] = marks
        audit["valid_from_utc"] = obj["valid_from"]
        audit["valid_until_utc"] = obj.get("valid_until")
        if ref != CREATOR or objects.get(ref, {}).get("type") != "identity":
            return "creator_unverified"
        if not marks or any(objects.get(m, {}).get("type") != "marking-definition" or
                            recipient not in GRANTS.get(m, set()) for m in marks):
            return "recipient_denied"
        assessed, start = instant(at), instant(obj["valid_from"])
        end = instant(obj["valid_until"]) if obj.get("valid_until") else None
        if end is not None and end <= start:
            return "invalid_validity_window"
        if obj.get("revoked"):
            return "revoked"
        if assessed < start or (end is not None and assessed >= end):
            return "outside_validity_window"
        if asset != ASSET:
            return "asset_unverified"
        audit["local_asset"] = asset
        return "request_local_control_evidence"
    except (KeyError, TypeError, ValueError, AttributeError):
        return "unparseable_payload"

EXPECTED_CTI = {
    "baseline": "request_local_control_evidence", "expired": "outside_validity_window", "revoked": "revoked",
    "denied": "recipient_denied", "missing_creator": "creator_unverified", "tampered": "integrity_mismatch",
    "unmapped": "asset_unverified", "bad_json": "unparseable_payload",
}

def replay_cases():
    base = json.loads((HERE / "synthetic_bundle.json").read_bytes())
    destination = HERE / "replayed_payloads"
    destination.mkdir(exist_ok=True)
    cases = json.loads((HERE / "replay_expectations.json").read_text(encoding="utf-8"))
    result = []
    for index, (name, variant, local, expected_g2, expected_disposition) in enumerate(cases, 1):
        bundle = deepcopy(base)
        indicator = next(o for o in bundle["objects"] if o["type"] == "indicator")
        if variant == "revoked":
            indicator.update(revoked=True, modified="2026-11-17T09:00:00.000Z")
        if variant == "missing_creator":
            del indicator["created_by_ref"]
        raw = b'{"type":' if variant == "bad_json" else encode(bundle)
        expect_hash = digest(raw)
        if variant == "tampered":
            raw = raw.replace(b"portal.invalid", b"changed.invalid")
            assert digest(raw) != expect_hash
        name_on_disk = f"{index:02d}_{variant}.json"
        (destination / name_on_disk).write_bytes(raw)
        receipt = {"provider": PROVIDER,
                   "recipient": "outside-recipient" if variant == "denied" else "demo-hospital-security",
                   "received": "2026-11-17T11:00:00.000Z", "sha256": expect_hash}
        at = "2026-11-17T15:00:00.000Z" if variant == "expired" else "2026-11-17T12:00:00.000Z"
        asset = None if variant == "unmapped" else ASSET
        record = classify(raw, receipt, at, asset, local)
        assert record["cti_status"] == EXPECTED_CTI[variant], record
        assert (record["G2"], record["disposition"]) == (expected_g2, expected_disposition), record
        assert record["input_sha256"] == digest((destination / name_on_disk).read_bytes())
        result.append({"case": name, "payload": f"replayed_payloads/{name_on_disk}", **record})
    (HERE / "replay_results.json").write_bytes(encode(result))
    return result

def rerun_g3():
    def read(value, *, kind):
        if not value:
            return math.nan if kind == "nan" else None
        if value == "True":
            return True
        return value if kind == "string" else float(value)
    with (HERE / "g3_controlled_cases.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 17
    for row in rows:
        kind = row["case"].split()[0]
        result = resource_gate_state(read(row["latency"], kind=kind),
                                     read(row["memory"], kind="float"), 200.0, 100.0)
        if row["case"] == "nan memory":
            result = resource_gate_state(float(row["latency"]), math.nan, 200.0, 100.0)
        assert result == row["required"], (row["case"], row["required"], result)
    with (HERE / "threshold_sensitivity.csv").open(newline="") as source:
        grid = list(csv.DictReader(source))
    near = {(int(r["latency_p95_limit_ms_strict"]), int(r["peak_memory_limit_mb_strict"])): int(r["eligible_n"]) for r in grid}
    assert len(near) == 20 and (min(near.values()), max(near.values())) == (2, 32)
    assert [near[(lat, 100)] for lat in (175, 200, 250)] == [12, 15, 24]
    assert [near[(200, mem)] for mem in (64, 100, 150)] == [5, 15, 19]
    return len(rows), len(grid)

if __name__ == "__main__":
    result = replay_cases()
    cases, settings = rerun_g3()
    documentary = json.loads((HERE / "documentary_case.json").read_text(encoding="utf-8"))
    actual = DISPOSITION[conjunctive_status(row["state"] for row in documentary["inputs"].values())]
    assert actual == documentary["expected_disposition"] == "UNRESOLVED"
    print(f"PASS. {len(result)} CTI fixtures, {cases} controlled G3 cases and {settings} archived grid settings checked.")
    print(f"Public-evidence case {documentary['assessment_id']} returned {actual}.")
