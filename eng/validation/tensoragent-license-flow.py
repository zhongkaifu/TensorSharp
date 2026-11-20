#!/usr/bin/env python3
"""Walk a TensorAgent host's license-acceptance flow for one catalog entry and record what it answered.

Against a running host (eng/validation/TensorAgentHost writes its connection.json): the entry's terms are
described, its download is refused with the terms until they are accepted, a decline keeps it refused, an
acceptance of a revision the host does not have is refused, an acceptance of the current revision lets the
download run to the end, and a recorded acceptance of an older revision asks again. Nothing here accepts a
license on anyone's behalf beyond the host it is pointed at, whose data folder should be a scratch one.

    tensoragent-license-flow.py --connection <root>/connection.json --model minimax-h3-image-q4km \
        --out <evidence dir> [--acceptances <root>/data/license-acceptances.json]

Exit 0 when every step answered as described, 1 otherwise. Evidence: <out>/license-flow.json (every
request and the parts of every answer that matter) and <out>/license-text.txt (the text the host serves).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request


def call(base, cookie, method, path, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    headers = {"Cookie": cookie}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, kind, raw = response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        status, kind, raw = error.code, error.headers.get("Content-Type", ""), error.read()
    text = raw.decode("utf-8", errors="replace")
    if kind.startswith("text/event-stream"):
        frames = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
        return status, {"frames": len(frames), "last": frames[-1] if frames else None,
                        "files": sorted({f.get("file") for f in frames if f.get("file")})}
    return status, json.loads(text) if text else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--connection", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--acceptances", help="the host's license-acceptances.json, to show an older revision asks again")
    args = parser.parse_args()
    connection = json.load(open(args.connection, encoding="utf-8"))
    base, cookie = connection["baseUrl"], connection["cookie"]
    os.makedirs(args.out, exist_ok=True)
    steps, failures = [], []

    def step(name, method, path, body=None, expect=None, check=None):
        started = time.monotonic()
        status, answer = call(base, cookie, method, path, body)
        record = {"step": name, "request": f"{method} {path}", "body": body, "status": status,
                  "seconds": round(time.monotonic() - started, 3), "answer": answer}
        ok = (expect is None or status == expect) and (check is None or check(answer))
        record["ok"] = ok
        steps.append(record)
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {method} {path} -> {status}")
        if not ok:
            failures.append(name)
        return answer

    entry = f"/api/agent/catalog/{args.model}"
    described = step("describe", "GET", entry, expect=200,
                     check=lambda a: a["licenseTerms"] is not None and not a["licenseTerms"]["accepted"])
    terms = described["licenseTerms"]
    revision = terms["revision"]
    step("download refused with the terms", "POST", entry + "/download", expect=409,
         check=lambda a: a.get("licenseRequired") is True and a["license"]["revision"] == revision)
    text = step("license text", "GET", entry + "/license", expect=200, check=lambda a: bool(a.get("text")))
    with open(os.path.join(args.out, "license-text.txt"), "w", encoding="utf-8", newline="") as handle:
        handle.write(text["text"])
    served = hashlib.sha256(text["text"].encode("utf-8")).hexdigest()
    steps.append({"step": "the served text is the revision", "revision": revision, "sha256": served,
                  "ok": served == revision})
    print(f"{'ok  ' if served == revision else 'FAIL'} the served text hashes to the revision ({served[:12]})")
    if served != revision:
        failures.append("text hash")
    step("decline", "POST", entry + "/license", {"revision": revision, "accept": False}, expect=200,
         check=lambda a: a["accepted"] is False)
    step("download still refused after the decline", "POST", entry + "/download", expect=409,
         check=lambda a: a.get("licenseRequired") is True)
    step("acceptance of another revision refused", "POST", entry + "/license",
         {"revision": "0" * 64, "accept": True}, expect=409)
    step("accept", "POST", entry + "/license", {"revision": revision, "accept": True}, expect=200,
         check=lambda a: a["accepted"] is True and a["revision"] == revision)
    step("download runs", "POST", entry + "/download", expect=200,
         check=lambda a: a["last"] is not None and a["last"].get("done") is True)
    step("installed and accepted", "GET", entry, expect=200,
         check=lambda a: a["state"] == "Installed" and a["licenseTerms"]["accepted"] is True)

    if args.acceptances:
        # As if the catalog shipped a new text: the recorded acceptance is of another revision now.
        stored = json.load(open(args.acceptances, encoding="utf-8"))
        for record in stored["acceptances"]:
            if record["model"] == args.model:
                record["revision"] = "f" * 64
        json.dump(stored, open(args.acceptances, "w", encoding="utf-8"), indent=2)
        step("an older revision on record asks again", "GET", entry, expect=200,
             check=lambda a: a["licenseTerms"]["accepted"] is False and a["licenseTerms"]["changedSinceAccepted"] is True)
        step("download refused for the older revision", "POST", entry + "/download", expect=409)
        step("accept the current revision again", "POST", entry + "/license", {"revision": revision, "accept": True},
             expect=200)
        step("download allowed again", "POST", entry + "/download", expect=200,
             check=lambda a: a["last"] is not None and a["last"].get("done") is True)

    with open(os.path.join(args.out, "license-flow.json"), "w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "base": base, "terms": terms, "steps": steps, "failures": failures},
                  handle, indent=2, ensure_ascii=False)
    print(f"{len(steps) - len(failures)} of {len(steps)} steps as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
