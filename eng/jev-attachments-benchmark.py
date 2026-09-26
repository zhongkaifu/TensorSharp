#!/usr/bin/env python3
"""Measure uploaded/inline Jev attachments against a running real model server.

python3 eng/jev-attachments-benchmark.py --endpoint http://127.0.0.1:5188
python3 eng/jev-attachments-benchmark.py --endpoint http://127.0.0.1:5188 \
    --cases txt,pdf,docx --concurrency 1,2 --repeats 3

Standard library only. Upload time includes any processing by /api/upload;
request time includes preprocessing, inference, HTTP and queueing. Server-reported
preprocessing/inference timings are recorded separately, never inferred by
subtracting unrelated samples. Each upload-mode iteration uploads fresh bytes.
Skipped cases and unavailable models/services are not passing coverage. Exit 0
requires all selected cases, malformed-input checks, and requested budgets to pass;
exit 2 means selected coverage was incomplete. This small synthetic quality screen
does not establish general document/video/audio quality or production performance.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import mimetypes
import os
from pathlib import Path
import platform
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "InferenceWeb.Tests/Fixtures/JevAttachments/manifest.json"
SPEC = importlib.util.spec_from_file_location("jev_benchmark", ROOT / "eng/jev-benchmark.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def output_directory(value):
    path = Path(value).resolve()
    require(any(path.is_relative_to((ROOT / parent).resolve()) for parent in ("artifacts", "docs/validation")),
            "--output must be inside this checkout's ignored artifacts/ or docs/validation/")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_fixtures(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest.get("version") == 1, "Unsupported fixture version")
    for name, identity in manifest["files"].items():
        file = (path.parent / name).resolve()
        require(file.is_relative_to(path.parent.resolve()), "Fixture paths must remain under the manifest directory")
        raw = file.read_bytes()
        require(len(raw) == identity["bytes"] and hashlib.sha256(raw).hexdigest() == identity["sha256"],
                f"Fixture integrity mismatch: {name}")
    for case in manifest["cases"]:
        require(set(case["questions"]) == set(case["expected"]), "Expected question keys differ")
        for attachment in case["attachments"]:
            require(attachment["path"] in manifest["files"], "Attachment must have a recorded file identity")
    return manifest


def upload(url, path, timeout, key):
    boundary = "JevAttachment" + uuid.uuid4().hex
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    # All filenames come from the checked, local fixture manifest.
    require(not any(char in path.name for char in '\r\n"'), "Unsafe multipart filename")
    data = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f'Content-Type: {mime}\r\n\r\n').encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": "multipart/form-data; boundary=" + boundary}
    if key:
        headers["Authorization"] = "Bearer " + key
    started = time.perf_counter()
    try:
        try:
            response = urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw, status = response.read(), response.status
        body = json.loads(raw)
        return {"status": status, "body": body, "elapsed_ms": (time.perf_counter() - started) * 1000}
    except (OSError, ValueError, TimeoutError) as error:
        return {"status": None, "body": None, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "error": f"{type(error).__name__}: {error}"}


def make_payload(case, model, repetition):
    return {"model": model, "state": case.get("state", {"task": "Evaluate only the attached source material.",
                                      "request_id": f"{case['id']}-{repetition}"}),
            "questions": copy.deepcopy(case["questions"]), "samples": 1, "seed": 42}


def validate_diagnostics(case, response):
    diagnostics = response.get("diagnostics", {})
    attachments = diagnostics.get("attachments")
    require(isinstance(attachments, list) and len(attachments) == len(case["attachments"]),
            "Missing attachment diagnostics or attachments were dropped")
    for expected, actual in zip(case["attachments"], attachments):
        require(actual.get("kind") == expected["kind"], "Attachment kind differs from fixture")
        if expected["kind"] in ("text", "pdf", "document", "audio"):
            require(isinstance(actual.get("textCharacters"), int) and actual["textCharacters"] > 0,
                    "Document/audio yielded no text")
        if expected["kind"] in ("video", "image"):
            require(isinstance(actual.get("imageCount"), int) and actual["imageCount"] > 0,
                    "Image/video yielded no image spans")
    timing = diagnostics.get("timing", {})
    for key in ("preprocessing_ms", "inference_ms", "total_ms"):
        value = timing.get(key)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0,
                f"Missing or invalid diagnostics.timing.{key}")
    require(timing["total_ms"] + 1 >= timing["preprocessing_ms"] + timing["inference_ms"],
            "Server total time excludes processing stages")
    return timing


def run_case(args, case, mode, repetition, references=None):
    started = time.perf_counter()
    payload = make_payload(case, args.model, repetition)
    row = {"case": case["id"], "mode": mode, "repetition": repetition, "status": "failed",
           "upload_ms": 0, "request_ms": None, "preprocessing_ms": None, "inference_ms": None,
           "decisions": [], "uploads": []}
    try:
        for attachment in case["attachments"]:
            path = args.fixtures.parent / attachment["path"]
            if mode == "reference":
                item = {"file": references[attachment["path"]], "name": path.name}
            elif mode == "upload":
                result = upload(args.upload_url, path, args.timeout, args.api_key)
                row["uploads"].append(result)
                row["upload_ms"] += result["elapsed_ms"]
                require(result["status"] == 200 and isinstance(result["body"], dict),
                        f"Upload failed: HTTP {result['status']}")
                file = result["body"].get("file")
                require(result["body"].get("ok") is True and isinstance(file, str) and file,
                        "Upload response did not return a stored file")
                item = {"file": file, "name": path.name}
            else:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                if mode == "data-url":
                    encoded = "data:" + (mimetypes.guess_type(path.name)[0] or "application/octet-stream") + ";base64," + encoded
                item = {"name": path.name, "data": encoded}
            payload.setdefault(attachment["field"], []).append(item)
        result = BENCH.post(args.systemone_url, payload, args.timeout, args.api_key)
        row.update(response=result, request_ms=result["elapsed_ms"], request_sha256=BENCH.hash_value(payload))
        if result["status"] in (501, 503):
            row.update(status="unavailable", error=result["body"])
        else:
            row["decisions"] = BENCH.validate_response(payload, result, case["expected"])
            timing = validate_diagnostics(case, result["body"])
            row.update(preprocessing_ms=timing["preprocessing_ms"], inference_ms=timing["inference_ms"])
            require(all(item["correct"] for item in row["decisions"]), "Attachment fixture decision differs from gold")
            row["status"] = "passed"
    except (OSError, ValueError, TypeError, KeyError) as error:
        row["error"] = f"{type(error).__name__}: {error}"
    row["end_to_end_ms"] = (time.perf_counter() - started) * 1000
    return row


def invalid_cases(model):
    common = {"model": model, "state": {}, "questions": {"ok": {"type": "noul", "instructions": "Is the object red?"}}}
    return [(name, {**copy.deepcopy(common), **change}) for name, change in (
        ("malformed-base64", {"files": [{"name": "ticket.txt", "data": "%%%"}]}),
        ("ambiguous-source", {"files": [{"name": "ticket.txt", "data": "dGVzdA==", "file": "ticket.txt"}]}),
        ("path-traversal", {"files": [{"file": "../ticket.txt"}]}),
        ("unsupported-extension", {"files": [{"name": "payload.exe", "data": "dGVzdA=="}]}),
        ("missing-upload", {"documents": [{"file": "jev-benchmark-missing-" + uuid.uuid4().hex + ".txt"}]}),
        ("corrupt-document", {"documents": [{"name": "ticket.pdf", "data": "bm90IGEgcGRm"}]}),
        ("wrong-modality", {"videos": [{"name": "ticket.txt", "data": "dGVzdA=="}]}),
    )]


def run_invalid(args):
    rows = []
    for name, payload in invalid_cases(args.model):
        result = BENCH.post(args.systemone_url, payload, args.timeout, args.api_key)
        # Rejecting malformed user input as an internal error or unavailable
        # model cannot count as validation success.
        rows.append({"case": name, "status": "passed" if result["status"] in (400, 404, 413, 422) else "failed", "response": result})
    return rows


def run_reference_isolation(args, manifest):
    """Reuse stored uploads concurrently with opposing gold labels and unique keys."""
    wanted = [copy.deepcopy(case) for case in manifest["cases"] if case["id"] in ("txt", "outage")]
    if len(wanted) != 2:
        return [{"case": "reference-isolation", "status": "unavailable", "error": "Missing contrasting fixtures"}]
    references = {}
    for case in wanted:
        # Distinct response keys expose cross-request contamination even if all
        # probabilities accidentally happen to favor the same label.
        case["questions"] = {case["id"] + "_" + key: value for key, value in case["questions"].items()}
        case["expected"] = {case["id"] + "_" + key: value for key, value in case["expected"].items()}
        for attachment in case["attachments"]:
            result = upload(args.upload_url, args.fixtures.parent / attachment["path"], args.timeout, args.api_key)
            if result["status"] != 200 or not result["body"].get("file"):
                return [{"case": "reference-isolation", "status": "failed", "error": "Fixture upload failed", "response": result}]
            references[attachment["path"]] = result["body"]["file"]
    serial = [run_case(args, case, "reference", 0, references) for case in wanted]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(lambda case: run_case(args, case, "reference", 0, references), wanted))
    return [{"phase": phase, **row} for phase, rows in (("serial-reference", serial), ("concurrent-reference-reuse", concurrent)) for row in rows]


def summarize(rows, elapsed_seconds, maximum_p95=None):
    passed = [row for row in rows if row["status"] == "passed"]
    timings = {}
    for name in ("upload_ms", "request_ms", "preprocessing_ms", "inference_ms", "end_to_end_ms"):
        values = [row[name] for row in passed if row.get(name) is not None]
        timings[name] = {"p50": BENCH.percentile(values, .5), "p95": BENCH.percentile(values, .95), "samples": len(values)}
    all_decisions = [decision for row in rows for decision in row["decisions"]]
    p95 = timings["end_to_end_ms"]["p95"]
    performance = "not_evaluated" if maximum_p95 is None else (
        "passed" if len(passed) == len(rows) and p95 is not None and p95 <= maximum_p95 else "failed")
    return {"requests": len(rows), "passed": len(passed), "failed": sum(r["status"] == "failed" for r in rows),
            "unavailable": sum(r["status"] == "unavailable" for r in rows), "timings": timings,
            "successful_requests_per_second": len(passed) / elapsed_seconds if elapsed_seconds else 0,
            "correct_decisions": sum(d["correct"] for d in all_decisions), "returned_decisions": len(all_decisions),
            "performance_budget_status": performance}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--api-key-env")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--cases", help="Comma-separated fixture IDs; omitted cases are recorded as skipped")
    parser.add_argument("--modes", default="inline,upload", help="Comma-separated inline, data-url, upload")
    parser.add_argument("--concurrency", type=lambda v: BENCH.csv_ints(v, 1), default=[1])
    parser.add_argument("--repeats", type=BENCH.positive, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-p95-ms", type=float, help="Optional maximum end-to-end latency budget per mode/concurrency group")
    parser.add_argument("--description", default="", help="Record model/backend/hardware and server launch details")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/jev-attachments" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    args.systemone_url = BENCH.endpoint_url(args.endpoint)
    args.upload_url = args.systemone_url.removesuffix("/v1/systemone") + "/api/upload"
    args.api_key = os.environ[args.api_key_env] if args.api_key_env else None
    args.modes = args.modes.split(",")
    require(args.modes and set(args.modes) <= {"inline", "data-url", "upload"} and len(args.modes) == len(set(args.modes)), "Invalid or duplicate mode")
    require(args.warmup >= 0 and args.timeout > 0, "warmup must be >= 0 and timeout > 0")
    require(args.max_p95_ms is None or args.max_p95_ms > 0, "Latency budget must be positive")
    return args


def main():
    args = parse_args()
    output = output_directory(args.output)
    manifest = load_fixtures(args.fixtures)
    selected = args.cases.split(",") if args.cases else [case["id"] for case in manifest["cases"]]
    require(set(selected) <= {case["id"] for case in manifest["cases"]} and len(selected) == len(set(selected)), "Unknown or duplicate fixture ID")
    cases = [case for case in manifest["cases"] if case["id"] in selected]
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "endpoint": args.systemone_url, "model": args.model,
              "description": args.description, "platform": platform.platform(), "fixtures_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
              "configuration": {"cases": selected, "modes": args.modes, "concurrency": args.concurrency,
                                "repeats": args.repeats, "warmup": args.warmup, "timeout_seconds": args.timeout,
                                "max_p95_ms": args.max_p95_ms},
              "fixture_provenance": manifest.get("provenance", {}),
              "git_revision": BENCH.command_output(["git", "rev-parse", "HEAD"]),
              "ggml_revision": BENCH.command_output(["git", "rev-parse", "HEAD"], ROOT / "ExternalProjects/ggml"),
              "ggml_status": BENCH.command_output(["git", "status", "--short"], ROOT / "ExternalProjects/ggml"),
              "skipped": [{"case": case["id"], "reason": "not selected by --cases"} for case in manifest["cases"] if case["id"] not in selected],
              "limitations": [manifest["description"], "Model/service startup is excluded; upload processing is included in upload latency.",
                              "Reported inference time includes vision encoding; preprocessing includes document/video/audio conversion.",
                              "Repeated tiny fixtures warm process caches; latency is not a cold-content or long-media benchmark.",
                              "Sequential groups do not control thermal or cache drift; no speed claim without a supplied budget."],
              "warmup": [], "groups": [], "invalid": [], "reference_isolation": []}
    with (output / "requests.jsonl").open("w", encoding="utf-8") as evidence:
        def record(row):
            evidence.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            evidence.flush()
        for mode in args.modes:
            for repetition in range(args.warmup):
                for case in cases:
                    row = run_case(args, case, mode, -1 - repetition)
                    record({"phase": "warmup", **row})
                    report["warmup"].append({key: row[key] for key in ("case", "mode", "status")})
            for concurrency in args.concurrency:
                tasks = [(case, repetition) for repetition in range(args.repeats) for case in cases]
                started = time.perf_counter()
                rows = []
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    for row in pool.map(lambda item: run_case(args, item[0], mode, item[1]), tasks):
                        rows.append(row)
                        record({"phase": "measured", "concurrency": concurrency, **row})
                        print(f"{mode}/{concurrency} {row['case']}: {row['status']} {row['end_to_end_ms']:.1f} ms", flush=True)
                elapsed = time.perf_counter() - started
                summary = summarize(rows, elapsed, args.max_p95_ms)
                case_summaries = {}
                for case in cases:
                    selected_rows = [row for row in rows if row["case"] == case["id"]]
                    case_summary = summarize(selected_rows, elapsed)
                    # Dividing one case's count by a whole mixed workload's wall
                    # time is not isolated per-case throughput.
                    case_summary.pop("successful_requests_per_second")
                    case_summaries[case["id"]] = case_summary
                report["groups"].append({"mode": mode, "concurrency": concurrency, **summary, "cases": case_summaries})
                print(json.dumps({"mode": mode, "concurrency": concurrency, **summary}), flush=True)
        report["invalid"] = run_invalid(args)
        for row in report["invalid"]:
            record({"phase": "invalid", **row})
        report["reference_isolation"] = run_reference_isolation(args, manifest)
        for row in report["reference_isolation"]:
            record(row)
    extra = report["invalid"] + report["warmup"] + report["reference_isolation"]
    failed = any(g["failed"] or g["performance_budget_status"] == "failed" for g in report["groups"]) or any(r["status"] == "failed" for r in extra)
    unavailable = any(g["unavailable"] for g in report["groups"]) or any(r["status"] == "unavailable" for r in extra)
    report["selected_coverage_status"] = "failed" if failed else "incomplete" if unavailable else "passed"
    report["full_fixture_coverage"] = not report["skipped"] and not failed and not unavailable
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['selected_coverage_status']}: {output / 'summary.json'}", flush=True)
    return 1 if failed else 2 if unavailable else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
