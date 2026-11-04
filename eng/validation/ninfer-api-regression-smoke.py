#!/usr/bin/env python3
"""Fail-fast five-request regression smoke; never a benchmark or quality score.

PLAN is default and sends no HTTP. Execution reuses exact historical prompts and
settings, changing only the explicitly supplied deployed model ID. No retries,
JSON repair, output-cap changes, warmups, server launch or GPU management.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "eng/qwen38-engine-bench.py"
AUDIT = ROOT / "eng/validation/ninfer-api-no-loss-audit.py"
FIXTURE = ROOT / "InferenceWeb.Tests/TestData/Ninfer/ninfer-api-regression-smoke-r8.json"
BENCH_SHA = "4e1fa314732250648f43a06d62c02d82b9c17e8842edfeda812f969aac8c363c"
AUDIT_SHA = "54d5d6affa6edfcea58804f0eb4afc7a03fc81dc1ff4aff1d33e1cdeb92621e2"
FIXTURE_SHA = "fb1e9649e4528aa51cb8ae0f5d8daaa85c97cc9a2fe7636534260570eb50110b"
KEYS = [(1, 0, "result"), (2, 3, "result"), (2, 6, "sorted"), (1, 6, "result"), (2, 0, "result")]
MAX_RESPONSE, MAX_LINE = 2 << 20, 65536


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def regular(path):
    need(path.is_file() and not path.is_symlink(), "Regular input file required: " + str(path))
    return path.read_bytes()


def load_module(name, path, expected):
    need(sha(regular(path)) == expected, "Pinned original tool differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_fixture():
    audit = load_module("ninfer_smoke_frozen_audit", AUDIT, AUDIT_SHA)
    bench = load_module("ninfer_smoke_frozen_bench", BENCH, BENCH_SHA)
    raw = regular(FIXTURE)
    need(sha(raw) == FIXTURE_SHA, "Pinned smoke fixture differs")
    fixture = audit.parse(raw)
    need(fixture["schema"] == "ninfer-api-regression-smoke-fixtures/1"
         and fixture["source"]["auditToolSha256"] == AUDIT_SHA
         and fixture["source"]["benchmarkSha256"] == BENCH_SHA, "Fixture provenance differs")
    rows = fixture["cases"]
    need(len(rows) == 5 and [(r["repeat"], r["taskIndex"], r["selectedOldTruth"]["field"]) for r in rows] == KEYS,
         "Five fixed historical identities are required")
    for row in rows:
        body, old = row["requestBody"], row["baselineProof"]
        truth = row["selectedOldTruth"]
        need(row["scenario"] == "quality_tasks" and body["think"] is False and body["stream"] is True,
             "Original no-thinking/streamed request required")
        need(audit.sha(audit.canonical(bench.request_identity(body))) == row["requestSha256"]
             and sha(json.dumps(body).encode()) == row["wireBodySha256"], "Canonical historical request differs")
        need(sha(old["content"].encode()) == old["contentSha256"] and old["httpStatus"] == 200
             and old["finishReason"] == "stop" and old["reasoning"] == "", "Historical response proof differs")
        need(audit.sha(audit.canonical({"content": old["content"], "reasoning": old["reasoning"], "tool_calls": []}))
             == old["responseSha256"] and old["wholeTaskCheck"]["name"] == f"task{row['taskIndex']}",
             "Historical normalized response/check identity differs")
        parsed = json.loads(old["content"])
        need(truth["baselineCorrect"] is True
             and audit.strict_equal(parsed.get(truth["field"]), truth["expected"])
             and audit.strict_equal(truth["baselineValue"], truth["expected"]), "Selected field was not previously correct")
    return fixture, bench, audit


def endpoint(value):
    url = urlparse(value)
    need(url.scheme == "http" and url.hostname in ("127.0.0.1", "localhost", "::1")
         and url.port is not None and url.path in ("", "/") and not url.query and not url.fragment
         and url.username is None and url.password is None, "Explicit loopback HTTP URL required")
    return url


def fresh(path):
    need(path.is_absolute() and path == path.resolve(), "Canonical absolute output required")
    need(not path.exists() and not path.is_symlink(), "Fresh output directory required")
    need("artifacts" in path.parts or any(path.parts[i:i + 2] == ("docs", "validation") for i in range(len(path.parts) - 1)),
         "Output must be under ignored artifacts/ or docs/validation/")
    need(all(not p.is_symlink() and not p.is_file() for p in path.parents), "Aliased output parent refused")


def argument(argv, name):
    values = []
    for i, value in enumerate(argv):
        if value == name:
            need(i + 1 < len(argv), "Incomplete manifest argument: " + name)
            values.append(argv[i + 1])
        elif value.startswith(name + "="):
            values.append(value[len(name) + 1:])
    need(len(values) == 1 and values[0], "Exactly one manifest argument required: " + name)
    return values[0]


def validate_manifest(value, model, url):
    need(value.get("schema") == "ninfer-target-server/1"
         and value.get("engine") == "TensorSharp/native-cuda", "Manifest must declare the native CUDA NInfer server")
    argv = value.get("argv")
    need(isinstance(argv, list) and argv and all(type(x) is str and x for x in argv), "Manifest argv is invalid")
    need(argument(argv, "--backend") == "cuda", "Manifest backend must be cuda")
    need(argument(argv, "--port") == str(url.port), "Manifest port differs from the requested endpoint")
    need(argument(argv, "--model").lower().endswith(".ninfer"), "Manifest must load an original NInfer artifact")
    need(value.get("model") == model and model in [x.get("id") for x in value.get("modelsResponse", {}).get("data", [])],
         "Manifest model identity differs")


def consume_sse(response, timeout, set_timeout=lambda _: None):
    """Bounded transport admission; no task scoring and no timing measurements."""
    obs = {"httpStatus": response.status, "httpBodyComplete": False, "doneSeen": False,
           "finishReasons": [], "content": "", "reasoning": "", "toolCallsSeen": False,
           "multipleChoices": False, "error": None}
    raw, pending, event = bytearray(), bytearray(), []
    deadline = time.monotonic() + timeout

    def dispatch():
        if not event:
            return
        data = "\n".join(event); event.clear()
        need(not obs["doneSeen"], "Unexpected SSE data after DONE")
        if data == "[DONE]":
            obs["doneSeen"] = True
            return
        obj = json.loads(data)
        need(isinstance(obj, dict) and not obj.get("error"), "SSE server error or malformed frame")
        choices = obj.get("choices") or []
        need(isinstance(choices, list), "Invalid SSE choices")
        if len(choices) > 1 or any(c.get("index", 0) != 0 for c in choices):
            obs["multipleChoices"] = True
        for choice in choices:
            delta = choice.get("delta") or {}
            need(isinstance(delta, dict), "Invalid SSE delta")
            for name in ("content", "reasoning", "reasoning_content"):
                text = delta.get(name)
                need(text is None or isinstance(text, str), "Invalid SSE text field")
                if text:
                    obs["content" if name == "content" else "reasoning"] += text
            obs["toolCallsSeen"] |= bool(delta.get("tool_calls"))
            if choice.get("finish_reason") is not None:
                need(isinstance(choice["finish_reason"], str), "Invalid finish reason")
                obs["finishReasons"].append(choice["finish_reason"])

    def line(data):
        text = data.decode("utf-8").rstrip("\r\n")
        if text == "":
            dispatch()
        elif text.startswith("data:"):
            event.append(text[5:].lstrip(" "))

    try:
        while True:
            remaining = deadline - time.monotonic()
            need(remaining > 0, "Request deadline exceeded")
            set_timeout(remaining)
            data = response.read1(4096)
            if not data:
                # HTTPResponse.read1() can return TCP EOF without raising
                # IncompleteRead when a declared Content-Length is still short.
                need(getattr(response, "length", None) in (None, 0), "Truncated HTTP Content-Length body")
                obs["httpBodyComplete"] = True
                break
            raw.extend(data)
            need(len(raw) <= MAX_RESPONSE, "HTTP response exceeds the fixed2MiB cap")
            pending.extend(data)
            while b"\n" in pending:
                at = pending.index(10)
                need(at <= MAX_LINE, "SSE line exceeds its fixed bound")
                current = bytes(pending[:at + 1]); del pending[:at + 1]
                line(current)
            need(len(pending) <= MAX_LINE, "SSE line exceeds its fixed bound")
        if pending:
            line(bytes(pending))
        dispatch()
    except (OSError, ValueError, TypeError, AttributeError, UnicodeError, http.client.HTTPException) as error:
        obs["error"] = type(error).__name__ + ": " + str(error)
        obs["httpBodyComplete"] = False
    return obs, bytes(raw[:MAX_RESPONSE])


def query(url, wire, timeout):
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body=wire,
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        response = conn.getresponse()
        return consume_sse(response, timeout, lambda n: conn.sock.settimeout(n) if conn.sock else None)
    except (OSError, ValueError, http.client.HTTPException) as error:
        return {"httpStatus": None, "httpBodyComplete": False, "doneSeen": False, "finishReasons": [],
                "content": "", "reasoning": "", "toolCallsSeen": False, "multipleChoices": False,
                "error": type(error).__name__ + ": " + str(error)}, b""
    finally:
        conn.close()


def assess(observation, truth, audit):
    checks = {"http200": type(observation.get("httpStatus")) is int and observation["httpStatus"] == 200,
              "completeHttpBody": observation.get("httpBodyComplete") is True,
              "streamDone": observation.get("doneSeen") is True,
              "terminalStop": bool(observation.get("finishReasons")) and all(x == "stop" for x in observation["finishReasons"]),
              "noTransportError": observation.get("error") is None,
              "singleChoice": observation.get("multipleChoices") is False,
              "noToolCalls": observation.get("toolCallsSeen") is False,
              "noThinking": observation.get("reasoning") == "" and "<think>" not in observation.get("content", "").lower()
                  and "</think>" not in observation.get("content", "").lower()}
    try:
        # Preserve the original content parser, including ordinary JSON duplicate
        # key behavior. This smoke does not strip fences or repair responses.
        parsed = json.loads(observation.get("content") or "")
    except (ValueError, TypeError):
        parsed = None
    actual = parsed.get(truth["field"]) if isinstance(parsed, dict) else None
    checks["selectedOldTruth"] = audit.strict_equal(actual, truth["expected"])
    try:
        json.dumps(actual, allow_nan=False)
        diagnostic = actual
    except (ValueError, TypeError):
        # This affects evidence serialization only, never the check above.
        diagnostic = {"nonfiniteOrUnserializableJsonValue": repr(actual)}
    return {"checks": checks, "selectedField": truth["field"], "expected": truth["expected"],
            "actual": diagnostic, "passed": all(checks.values())}


def new_file(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def execute(args, fixture, bench, audit, url, manifest_bytes):
    state = {"schema": "ninfer-api-regression-smoke/1", "complete": False, "runFinished": False,
             "status": "RUNNING_NOT_PASS", "smokePassed": False, "qualityQualification": "NOT_RUN",
             "performance": "NOT_RUN", "fullSuiteStillRequired": True, "fixtureSha256": FIXTURE_SHA,
             "benchmarkSha256": BENCH_SHA, "auditToolSha256": AUDIT_SHA, "url": args.url, "model": args.model,
             "manifestSha256": sha(manifest_bytes), "nativeBackendManifestAdmitted": True,
             "liveServerIdentityVerifiedByThisTool": False, "requestsPlanned": 5, "requestsExecuted": 0,
             "scope": "Selected five historical field truths only; no complete-task score, new benchmark, quality ranking or performance claim.",
             "cases": [{"id": r["id"], "status": "NOT_RUN"} for r in fixture["cases"]]}
    args.out.mkdir(parents=True)
    new_file(args.out / "manifest.json", manifest_bytes)

    def save():
        (args.out / "result.json").write_text(json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    save()
    try:
        for index, row in enumerate(fixture["cases"]):
            need(sha(regular(args.manifest)) == state["manifestSha256"], "Manifest changed before request")
            body = copy.deepcopy(row["requestBody"]); body["model"] = args.model
            need(audit.strict_equal(bench.request_identity(body), bench.request_identity(row["requestBody"])), "Historical logical request changed")
            wire = json.dumps(body).encode()
            request_file = f"{index:02d}-{row['id']}.request.json"
            new_file(args.out / request_file, wire)
            record = state["cases"][index]
            record.update(status="REQUEST_RUNNING_NOT_PASS", historicalRequestSha256=row["requestSha256"],
                          requestFile=request_file, wireSha256=sha(wire), onlyModelIdMayDiffer=True)
            save()
            observation, raw = query(url, wire, args.timeout)
            response_file = f"{index:02d}-{row['id']}.response.sse"
            new_file(args.out / response_file, raw)
            result = assess(observation, row["selectedOldTruth"], audit)
            record.update(status="PASS" if result["passed"] else "FAIL", observation=observation,
                          selectedFieldCheck=result, responseFile=response_file, responseSha256=sha(raw))
            state["requestsExecuted"] += 1
            save()
            need(sha(regular(args.manifest)) == state["manifestSha256"], "Manifest changed after request")
            if not result["passed"]:
                for later in state["cases"][index + 1:]:
                    later["status"] = "NOT_RUN_FAIL_FAST"
                break
        need(sha(regular(FIXTURE)) == FIXTURE_SHA and sha(regular(BENCH)) == BENCH_SHA
             and sha(regular(AUDIT)) == AUDIT_SHA, "Pinned protocol input changed")
        state["smokePassed"] = state["requestsExecuted"] == 5 and all(r["status"] == "PASS" for r in state["cases"])
        state["complete"] = state["smokePassed"]
        state["status"] = "SMOKE_PASSED_FULL_SUITE_REQUIRED" if state["smokePassed"] else "REJECTED_FAIL_FAST"
    except (OSError, ValueError, TypeError, KeyError) as error:
        state["status"] = "INCOMPLETE_NOT_PASS"
        state["error"] = type(error).__name__ + ": " + str(error)
    state["runFinished"] = True
    state["allFiveRequestsExecuted"] = state["requestsExecuted"] == 5
    save()
    data = (json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    new_file(args.out / "final-result.json", data)
    new_file(args.out / "final-result.json.sha256", (sha(data) + "  final-result.json\n").encode())
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    phase = "admission"
    try:
        need(args.model.strip() == args.model and bool(args.model), "Nonempty deployed model ID required")
        need(args.manifest.is_absolute(), "Absolute manifest path required")
        need(0 < args.timeout <= 1800, "Timeout must be in(0,1800] seconds")
        url = endpoint(args.url); fresh(args.out)
        fixture, bench, audit = load_fixture()
        if not args.execute:
            print(json.dumps({"schema": "ninfer-api-regression-smoke/1", "complete": False,
                              "status": "PLAN_ONLY_NO_HTTP", "smokePassed": False, "requestsExecuted": 0,
                              "fixtureSha256": FIXTURE_SHA, "requestsPlanned": [r["id"] for r in fixture["cases"]],
                              "manifestValidation": "NOT_RUN", "qualityQualification": "NOT_RUN", "performance": "NOT_RUN"}))
            return 0
        manifest_bytes = regular(args.manifest)
        validate_manifest(audit.parse(manifest_bytes), args.model, url)
        phase = "execution"
        result = execute(args, fixture, bench, audit, url, manifest_bytes)
        print(json.dumps({"output": str(args.out / "final-result.json"), "status": result["status"],
                          "smokePassed": result["smokePassed"], "requestsExecuted": result["requestsExecuted"],
                          "qualityQualification": "NOT_RUN", "performance": "NOT_RUN"}))
        return 0 if result["smokePassed"] else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(("Smoke refused before HTTP: " if phase == "admission" else "Smoke incomplete during execution: ") + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
