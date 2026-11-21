"""Loopback HTTPS demonstration of a limited subset of TAXII 2.1 collection operations."""
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from secrets import token_urlsafe
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4
import json
import ssl
import subprocess

from replay import ASSET, CREATOR, PROVIDER, classify, digest, encode, rerun_g3

HERE = Path(__file__).resolve().parent
API = "/demo/"
COLLECTION = "94e5fa5d-a7d5-46ea-9f04-b64195ca1dd4"
OBJECTS = f"{API}collections/{COLLECTION}/objects/"
TYPE = "application/taxii+json;version=2.1"
ROLES = {token_urlsafe(24): role for role in ("sender", "receiver", "unauthorised")}
BASIC = {role: "Bearer " + token for token, role in ROLES.items()}

def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")

class Exchange(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def answer(self, code, payload, **extra):
        data = encode(payload)
        self.send_response(code)
        self.send_header("Content-Type", TYPE)
        self.send_header("Content-Length", str(len(data)))
        for key, value in extra.items():
            self.send_header(key.replace("_", "-"), value)
        self.end_headers()
        self.wfile.write(data)
        self.server.events.append({"role": self.role or "anonymous", "path": self.path,
                                   "method": self.command, "status": code, "time_utc": now()})

    def check_role(self, required):
        self.role = ROLES.get(self.headers.get("Authorization", "").removeprefix("Bearer "))
        if not self.role:
            self.answer(401, {"title":"Authentication required"}, WWW_Authenticate="Bearer")
            return False
        if self.headers.get("Accept") != TYPE:
            self.answer(406, {"title":"Unsupported response format"})
            return False
        if self.role != required:
            self.answer(403, {"title":"Role does not permit this operation"})
            return False
        return True

    def do_POST(self):
        if not self.check_role("sender"):
            return
        if self.path != OBJECTS:
            self.answer(404, {"title":"Unknown collection"})
            return
        if self.headers.get("Content-Type") != TYPE:
            self.answer(415, {"title":"Unsupported request format"})
            return
        content = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        objects = content["objects"]
        if not objects or any(obj["type"] not in {"indicator","identity","marking-definition"} for obj in objects):
            self.answer(400, {"title":"Unsupported object set"})
            return
        self.server.received = now()
        self.server.objects = objects
        status_id = str(uuid4())
        entry = {"id": status_id, "status":"complete", "request_timestamp":self.server.received,
                 "total_count":len(objects), "success_count":len(objects), "failure_count":0, "pending_count":0,
                 "successes":[{"id":o["id"], **({"version":o["modified"]} if "modified" in o else {})} for o in objects]}
        self.server.status[status_id] = entry
        self.answer(202, entry)

    def do_GET(self):
        if self.path.startswith(API + "status/"):
            if not self.check_role("sender"):
                return
            entry = self.server.status.get(self.path.removeprefix(API+"status/").strip("/"))
            if entry is None:
                self.answer(404, {"title":"Unknown status"})
            else:
                self.answer(200, entry)
            return
        if not self.check_role("receiver"):
            return
        if self.path != OBJECTS:
            self.answer(404, {"title":"Unknown collection"})
            return
        self.answer(200, {"objects": self.server.objects, "more":False},
                    X_TAXII_Date_Added_First=self.server.received, X_TAXII_Date_Added_Last=self.server.received)

def run():
    base = json.loads((HERE/"synthetic_bundle.json").read_bytes())
    report, saved_http = [], []
    with TemporaryDirectory(prefix="hims-tls-") as temp:
        key, cert = Path(temp)/"key.pem", Path(temp)/"cert.pem"
        subprocess.run(["openssl","req","-x509","-newkey","rsa:2048","-sha256","-nodes","-days","1",
                        "-subj","/CN=localhost","-addext","subjectAltName=IP:127.0.0.1","-keyout",str(key),"-out",str(cert)],
                       check=True, capture_output=True)
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_context.load_cert_chain(cert,key)
        trusted_client = ssl.create_default_context(cafile=str(cert))
        with ThreadingHTTPServer(("127.0.0.1",0),Exchange) as server:
            server.socket = server_context.wrap_socket(server.socket,server_side=True)
            server.status, server.events, server.objects = {}, [], []
            server.received = now()
            thread = Thread(target=server.serve_forever,daemon=True)
            thread.start()
            origin = f"https://127.0.0.1:{server.server_port}"
            def request(role, method, path, status, body=None):
                headers = {"Accept":TYPE}
                if role:
                    headers["Authorization"] = BASIC[role]
                data = None if body is None else encode(body)
                if data is not None:
                    headers["Content-Type"] = TYPE
                try:
                    response = urlopen(Request(origin+path,data=data,headers=headers,method=method), context=trusted_client, timeout=5)
                except HTTPError as err:
                    response = err
                with response:
                    assert response.status == status, (role,method,path,response.status)
                    assert response.headers["Content-Type"] == TYPE
                    raw = response.read()
                    saved_http.append({"role":role or "anonymous","method":method,"endpoint":path,
                                       "status":status,"response_sha256":digest(raw),"response":json.loads(raw)})
                    return json.loads(raw), response.headers

            def transport(bundle):
                sent = {"objects":bundle["objects"]}
                accepted,_ = request("sender","POST",OBJECTS,202,sent)
                confirmed,_ = request("sender","GET",API+"status/"+accepted["id"]+"/",200)
                assert confirmed == accepted and confirmed["status"] == "complete" and confirmed["success_count"] == len(bundle["objects"])
                arrived,headers = request("receiver","GET",OBJECTS,200)
                assert arrived["objects"] == sent["objects"] and arrived["more"] is False
                assert headers["X-TAXII-Date-Added-First"] == confirmed["request_timestamp"]
                assert headers["X-TAXII-Date-Added-Last"] == confirmed["request_timestamp"]
                reconstructed = {**bundle, "objects":arrived["objects"]}
                return encode(reconstructed), headers["X-TAXII-Date-Added-First"]

            def check_case(label, payload, original, received, fixture_time, local, expectation):
                receipt = {"provider":PROVIDER,"recipient":"demo-hospital-security","received":received,"sha256":digest(encode(original))}
                result = classify(payload, receipt, fixture_time, ASSET, local)
                assert (result["cti_status"],result["G2"],result["disposition"]) == expectation, result
                assert result["input_sha256"] == result["receipt_sha256"]
                assert result["created_by_ref"] == CREATOR
                report.append({"case":label,"collection":COLLECTION,"sender":"sender","receiver":"receiver",**result})

            try:
                payload,received = transport(base)
                request(None,"GET",OBJECTS,401)
                request("unauthorised","GET",OBJECTS,403)
                request("sender","GET",OBJECTS,403)
                request("receiver","POST",OBJECTS,403,{"objects":base["objects"]})
                fixture_time = "2026-11-17T12:00:00.000Z"
                check_case("current, local evidence absent",payload,base,received,fixture_time,"UNKNOWN",
                           ("request_local_control_evidence","UNKNOWN","UNRESOLVED"))
                check_case("current, local control satisfied",payload,base,received,fixture_time,"PASS",
                           ("request_local_control_evidence","PASS","ELIGIBLE"))
                check_case("current, verified local failure",payload,base,received,fixture_time,"FAIL",
                           ("request_local_control_evidence","FAIL","INELIGIBLE"))
                check_case("expiry boundary",payload,base,received,"2026-11-17T15:00:00.000Z","UNKNOWN",
                           ("outside_validity_window","UNKNOWN","UNRESOLVED"))
                revised = deepcopy(base)
                indicator = next(o for o in revised["objects"] if o["type"]=="indicator")
                indicator.update(revoked=True,modified="2026-11-17T09:00:00.000Z")
                payload,received = transport(revised)
                check_case("revoked object version",payload,revised,received,fixture_time,"UNKNOWN",
                           ("revoked","UNKNOWN","UNRESOLVED"))
                assert report[-1]["source_object_id"] == report[0]["source_object_id"]
                assert report[-1]["source_modified"] != report[0]["source_modified"]
            finally:
                server.shutdown()
                thread.join()
        traffic = server.events
    assert len(saved_http) == len(traffic) == 10
    for idx,pair in enumerate(zip(saved_http,traffic)):
        client,service = pair
        assert (client["role"],client["method"],client["endpoint"],client["status"]) == (service["role"],service["method"],service["path"],service["status"]), idx
    out = {"transport":"authenticated loopback HTTPS; limited TAXII 2.1 collection endpoints",
           "fixture_times":"Declared scenario clock; distinct from actual receipt and HTTP log timestamps.",
           "http_requests":saved_http,"server_log":traffic,"gate_cases":report}
    (HERE/"taxii_exchange_results.json").write_bytes(encode(out))
    n,_ = rerun_g3()
    print(f"PASS. {len(traffic)} HTTPS transactions, {len(report)} receiver-to-gate outcomes and {n} controlled G3 cases.")

if __name__ == "__main__":
    run()
