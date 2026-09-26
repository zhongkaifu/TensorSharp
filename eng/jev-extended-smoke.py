#!/usr/bin/env python3
"""Additional TensorSharp Jev HTTP checks against an already running server.

  python eng/jev-extended-smoke.py --endpoint http://127.0.0.1:5000
  python eng/jev-extended-smoke.py --endpoint http://127.0.0.1:5000 --chat
  python eng/jev-extended-smoke.py --endpoint http://127.0.0.1:5000 --image

The default suite executes real inference for auto/fixed noise draws and compact
question chunking, and checks that image input that is not inline bytes is refused.
--image additionally sends a request carrying an image, which requires a server
started with the vision tower; it screens one unambiguous synthetic picture and is
not an image-understanding benchmark. --chat also runs ordinary diffusion chat after
Jev and while a Jev request is outstanding; it can take substantially longer than
structured reads.
No server, backend or model is started or reconfigured. Outputs go to the ignored
artifacts/jev/extended directory by default. This is an integration screen, not a
quality benchmark, calibration assessment or comparative performance claim.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("jev_benchmark", ROOT / "eng/jev-benchmark.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def urls(endpoint):
    systemone = BENCH.endpoint_url(endpoint)
    return systemone, systemone.removesuffix("systemone") + "chat/completions"


def evidence_directory(value):
    destination = Path(value).resolve()
    for directory in (ROOT / "artifacts", ROOT / "docs/validation"):
        if destination.is_relative_to(directory.resolve()):
            destination.mkdir(parents=True, exist_ok=True)
            return destination
    raise ValueError("--output must be inside this checkout's ignored artifacts/ or docs/validation/ directory")


def post(url, payload, timeout, api_key):
    raw = payload if isinstance(payload, bytes) else BENCH.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(url, data=raw, headers=headers)
    started = time.perf_counter()
    result = {"status": None, "body": None, "headers": {}, "error": None}
    try:
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            encoded = response.read()
            result["status"] = response.status
            result["headers"] = {name.lower(): value for name, value in response.headers.items()}
        try:
            result["body"] = json.loads(encoded)
        except (ValueError, UnicodeError):
            result["body"] = {"unparsed_response": encoded.decode("utf-8", errors="replace")}
        if result["status"] != 200:
            result["error"] = f"HTTP {result['status']}"
    except (OSError, ValueError, TimeoutError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return result


def headers_match(result):
    headers = result["headers"]
    request_id = headers.get("x-request-id")
    check(isinstance(request_id, str) and request_id, "Missing x-request-id header")
    check(headers.get("x-typesafe-request-id") == request_id, "Request ID headers differ")


def validate_samples(payload, result, expected, mode):
    BENCH.validate_response(payload, result, expected)
    headers_match(result)
    diagnostics = result["body"].get("diagnostics", {})
    check(diagnostics.get("engine") == "tensorsharp", "Expected TensorSharp diagnostics")
    check(diagnostics.get("steps") == 1, "Structured inference must use one step")
    check(diagnostics.get("probability_semantics") == "conditional_label_softmax_temperature_1",
          "Missing conditional-label probability semantics")
    check(diagnostics.get("entropy_semantics") == "conditional_label_entropy_nats",
          "Missing conditional-label entropy semantics")
    check(diagnostics.get("samples", {}).get("policy") == mode, "Incorrect sample policy")
    chunks = diagnostics.get("chunks")
    check(isinstance(chunks, list) and chunks, "Missing chunk diagnostics")
    visited = []
    reads = 0
    for chunk in chunks:
        ids = chunk.get("questions")
        check(isinstance(ids, list) and ids, "Chunk has no questions")
        visited.extend(ids)
        count = chunk.get("samples")
        check(isinstance(count, int) and not isinstance(count, bool) and count > 0,
              "Invalid chunk sample count")
        entropies = chunk.get("first_read_conditional_entropy")
        check(isinstance(entropies, list) and len(entropies) == len(ids), "Missing first-read entropies")
        check(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0
                  for v in entropies), "Invalid conditional entropy")
        if mode == "auto":
            maximum = payload.get("auto_max", 4)
            expanded = maximum > 1 and max(entropies) > payload.get("auto_threshold", 0.1)
            check(chunk.get("extended") == expanded, "Adaptive extension differs from the entropy rule")
            check(count == (maximum if expanded else 1), "Adaptive sample count differs from the entropy rule")
        else:
            check(count == payload["samples"], "Fixed sample count differs from requested count")
            check(chunk.get("extended") is False, "Fixed policy unexpectedly extended")
        for qid in ids:
            q = diagnostics.get("questions", {}).get(qid, {})
            check(q.get("samples") == count, f"{qid}: incorrect sample count")
            check(len(q.get("conditional_entropy", [])) == count, f"{qid}: missing per-read entropies")
            if count > 1:
                check(isinstance(q.get("stderr"), (int, float)) and math.isfinite(q["stderr"]) and q["stderr"] >= 0,
                      f"{qid}: missing/invalid standard error")
                BENCH.probability(q.get("agreement"), qid + ".agreement")
        reads += count
    check(visited == list(payload["questions"]), "Chunk question order or ownership differs from the request")
    check(diagnostics.get("timing", {}).get("reads") == reads, "Total read accounting differs from chunk counts")
    return chunks


def payload(model):
    return {"model": model, "seed": 42,
            "state": {"color": "red", "gate": "open", "number": 7},
            "questions": {
                "color": {"type": "choice", "instructions": "Which color is explicitly named in state?",
                          "criteria": {"red": "red", "blue": "blue", "green": "green"}},
                "open": {"type": "noul", "instructions": "Does the state explicitly say the gate is open?"},
                "magnitude": {"type": "score", "instructions": "Which number range contains the number in state?",
                              "criteria": ["0 through 3", "4 through 6", "7 through 9"]}}}


def compact_payload(model, prompt):
    body = {"model": model, "seed": 42, "samples": 1, "chunk_rows": 24, "chunk_prompt": prompt,
            "state": {"color": "red", "gate": "open", "number": 7}, "questions": {}}
    statements = ["Is the color red?", "Is the color blue?", "Is the gate open?", "Is the gate closed?",
                  "Is the number exactly 7?", "Is the number exactly 2?"]
    expected = {}
    for i in range(12):
        # Numeric IDs preserve the reference's compact `0yes 1no` token boundary.
        body["questions"][str(i)] = {"type": "noul", "instructions": statements[i % len(statements)]}
        expected[str(i)] = i % 2 == 0
    return body, expected


def image_payload(model, path):
    """The checked-in image request, re-pointed at the model alias under test.

    Its picture is a synthetic traffic light with the GREEN lamp lit, and the state
    text says only that the vehicle is approaching an intersection. Measured on this
    checkpoint, the same request without the image answers "red" and "must stop"
    with high confidence, so these two gold answers are reachable only by reading
    the pixels -- which is the failure this screen exists to catch (expanded
    placeholders with nothing behind them read as filler tokens and still produce a
    confident probability).
    """
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    body["model"] = model
    check(isinstance(body.get("images"), list) and body["images"], f"{path} carries no images")
    return body, {"signal": "green", "stop": False, "visibility": 2}


class Suite:
    def __init__(self, args, output):
        self.args, self.output = args, output
        self.systemone, self.chat = urls(args.endpoint)
        self.api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        self.records = []

    def save(self, name, body, result, validate):
        errors = []
        try:
            validate(result)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            errors.append(f"{type(error).__name__}: {error}")
        request = body if not isinstance(body, bytes) else {
            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
            "preview": body[:80].decode("utf-8", errors="replace")}
        record = {"name": name, "request": request, "response": result,
                  "passed": not errors, "validation_errors": errors}
        path = self.output / f"{len(self.records) + 1:02d}-{name}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        self.records.append({"name": name, "passed": not errors, "elapsed_ms": result["elapsed_ms"],
                             "status": result["status"], "evidence": str(path.relative_to(ROOT)), "errors": errors})
        print(f"{'PASS' if not errors else 'FAIL'} {name}: HTTP {result['status']}, {result['elapsed_ms']:.1f} ms", flush=True)
        for error in errors:
            print("  " + error, flush=True)
        return result

    def run(self, name, body, validate, chat=False):
        return self.save(name, body, post(self.chat if chat else self.systemone, body, self.args.timeout, self.api_key), validate)


def probability_distance(left, right):
    distances = []
    for qid, a in left["answers"].items():
        b = right["answers"][qid]
        if a["type"] == "noul":
            distances.append(abs(a["noul"] - b["noul"]))
        else:
            distances.extend(abs(value - b["probabilities"][key]) for key, value in a["probabilities"].items())
    return max(distances)


def validate_chat(result, expected="4"):
    check(result["status"] == 200, result["error"] or "Chat failed")
    check(isinstance(result["body"], dict), "Chat response must be an object")
    choices = result["body"].get("choices", [])
    check(len(choices) == 1, "Chat must return one choice")
    content = choices[0].get("message", {}).get("content")
    check(isinstance(content, str) and content.strip(), "Chat returned no answer text")
    # Preserve the original arithmetic oracle's equivalent spelling of four.
    matched = expected.casefold() in content.casefold() or (expected == "4" and "four" in content.casefold())
    check(matched, f"Chat answer did not contain expected substring {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="http://127.0.0.1:5000")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", default=str(ROOT / "artifacts/jev/extended"))
    parser.add_argument("--api-key-env", help="Environment variable containing the bearer token")
    parser.add_argument("--chat", action="store_true", help="Also test ordinary chat after and concurrently with Jev; potentially slow")
    parser.add_argument("--image", action="store_true", help="Also send an image request; requires a server started with the vision tower")
    parser.add_argument("--image-request", default=str(ROOT / "docs/examples/jev-traffic-light.json"),
                        help="Request JSON used by --image; must carry inline images")
    parser.add_argument("--max-body-mb", type=int, default=8,
                        help="The server's TS_JEV_MAX_BODY_MB (default 8); the oversized-body case sends one byte more")
    parser.add_argument("--chat-model", help="Default: actual model name returned by the first Jev response")
    parser.add_argument("--chat-max-tokens", type=int, default=256, help="Default 256: one full DiffusionGemma canvas budget")
    parser.add_argument("--chat-seed", type=int, default=42, help="Fixed ordinary-chat seed for repeatability checks; default 42")
    parser.add_argument("--chat-prompt", default="What is two plus two? Reply with only 4.", help="Ordinary-chat sanity-check prompt; recorded verbatim in evidence")
    parser.add_argument("--chat-expected", default="4", help="Required case-insensitive substring; default 4 also accepts its spelling four")
    parser.add_argument("--isolation-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.chat_max_tokens < 1 or not math.isfinite(args.isolation_tolerance) or args.isolation_tolerance < 0:
        parser.error("timeout/chat-max-tokens must be positive, and isolation-tolerance must be finite and nonnegative")
    if not 1 <= args.max_body_mb <= 64:
        parser.error("max-body-mb must be from 1 to 64, matching the server's TS_JEV_MAX_BODY_MB range")
    if not 0 <= args.chat_seed <= 2147483647:
        parser.error("chat-seed must be a nonnegative 32-bit integer")
    if not args.chat_prompt.strip() or not args.chat_expected.strip():
        parser.error("chat-prompt and chat-expected must be non-empty")
    try:
        output = evidence_directory(args.output)
        suite = Suite(args, output)
    except ValueError as error:
        parser.error(str(error))
    started = datetime.now(timezone.utc).isoformat()
    base = payload(args.model)
    expected = {"color": "red", "open": True, "magnitude": 2}

    initial = suite.run("default-auto", base, lambda r: validate_samples(base, r, expected, "auto"))
    fixed = dict(base, samples=4)
    suite.run("fixed-four", fixed, lambda r: validate_samples(fixed, r, expected, "fixed"))
    forced = dict(base, samples="auto", auto_max=4, auto_threshold=0)

    def forced_validator(result):
        chunks = validate_samples(forced, result, expected, "auto")
        check(all(c["samples"] == 4 and c["extended"] for c in chunks), "Threshold-zero case did not exercise adaptive expansion")

    suite.run("forced-auto-four", forced, forced_validator)
    for prompt in ("own", "shared"):
        compact, gold = compact_payload(args.model, prompt)

        def compact_validator(result, body=compact, labels=gold):
            chunks = validate_samples(body, result, labels, "fixed")
            check(len(chunks) > 1, "Compact schema did not exercise multiple chunks")
            check(all(8 <= c["canvas_width"] <= 24 for c in chunks), "Chunk exceeded requested row budget")
            # Save the predictions as part of the evidence and require the simple
            # explicit state facts; these are a narrow alignment/quality screen.
            scores = BENCH.validate_response(body, result, labels)
            check(all(score["correct"] for score in scores), "A compact question lost alignment or answered an explicit fact incorrectly")

        suite.run("compact-chunks-" + prompt, compact, compact_validator)

    def error_validator(status):
        def validate(result):
            check(result["status"] == status, f"Expected HTTP {status}, got {result['status']}")
            headers_match(result)
            check(isinstance(result["body"], dict) and isinstance(result["body"].get("error"), dict), "Expected protocol error object")
            check(isinstance(result["body"]["error"].get("message"), str), "Expected error message")
        return validate

    suite.run("malformed-json", b"{", error_validator(400))
    suite.run("oversized-body", b" " * (args.max_body_mb * 1024 * 1024 + 1), error_validator(413))

    # Refused whatever the server loaded: the request never reaches the tower, so this
    # case runs even without --image. An endpoint that fetched the URL instead would be
    # a request forger for anything its host can reach.
    remote = dict(base, samples=1, images=["https://example.com/receipt.png"])
    suite.run("image-remote-url-refused", remote, error_validator(422))

    # Named attachment arrays use the object schema, never arbitrary URL strings.
    # End-to-end supported media and upload references are exercised separately by
    # eng/jev-attachments-benchmark.py with real document/video/audio fixtures.
    for field in ("files", "documents", "videos", "audios"):
        suite.run(field + "-remote-url-refused", dict(base, **{field: ["https://example.com/media"]}), error_validator(422))
    for field in ("audio", "video", "document", "file"):
        suite.run(field + "-singular-field-refused", dict(base, **{field: "unsupported"}), error_validator(422))

    if args.image:
        image_body, image_gold = image_payload(args.model, args.image_request)

        def image_validator(result):
            validate_samples(image_body, result, image_gold, "fixed")
            scores = BENCH.validate_response(image_body, result, image_gold)
            diagnostics = result["body"]["diagnostics"]
            check(diagnostics.get("images") == len(image_body["images"]),
                  "Response did not report the images it was given")
            decisive = {score["question"]: score["correct"] for score in scores}
            # Only the two unambiguous questions gate the result; the score question's
            # outcome is recorded as evidence, not asserted.
            check(decisive.get("signal") and decisive.get("stop"),
                  "The image request answered an unambiguous visual fact incorrectly: "
                  "either the tower is not loaded or the spans did not reach the read")

        suite.run("image-request", image_body, image_validator)

    if args.chat:
        single = dict(base, samples=1)
        baseline = suite.run("jev-before-chat", single, lambda r: validate_samples(single, r, expected, "fixed"))
        chat_model = args.chat_model or (initial.get("body") or {}).get("model") or args.model
        chat_body = {"model": chat_model, "messages": [{"role": "user", "content": args.chat_prompt}],
                     "stream": False, "max_tokens": args.chat_max_tokens, "seed": args.chat_seed}
        chat_baseline = suite.run("chat-after-jev", chat_body, lambda result: validate_chat(result, args.chat_expected), chat=True)

        def isolated_validator(result):
            validate_samples(single, result, expected, "fixed")
            check(baseline["status"] == 200, "Cannot compare isolation because baseline failed")
            delta = probability_distance(baseline["body"], result["body"])
            check(delta <= args.isolation_tolerance, f"Jev probabilities changed after chat by {delta}, tolerance {args.isolation_tolerance}")

        suite.run("jev-after-chat", single, isolated_validator)
        start = threading.Barrier(2)

        def simultaneous(url, body):
            start.wait(timeout=30)
            return post(url, body, args.timeout, suite.api_key)

        with ThreadPoolExecutor(max_workers=2) as workers:
            jev_future = workers.submit(simultaneous, suite.systemone, single)
            chat_future = workers.submit(simultaneous, suite.chat, chat_body)
            jev_result, chat_result = jev_future.result(), chat_future.result()
        suite.save("jev-concurrent-chat", single, jev_result, isolated_validator)
        def repeatable_chat(result):
            validate_chat(result, args.chat_expected)
            check(chat_baseline["status"] == 200, "Cannot compare chat repeatability because baseline failed")
            baseline_message = chat_baseline["body"]["choices"][0]["message"]
            message = result["body"]["choices"][0]["message"]
            check(message.get("content") == baseline_message.get("content"), "Identical seeded chat changed while Jev was pending")
            check(message.get("reasoning_content") == baseline_message.get("reasoning_content"), "Identical seeded chat reasoning changed")
        suite.save("chat-concurrent-jev", chat_body, chat_result, repeatable_chat)

    passed = all(record["passed"] for record in suite.records)
    summary = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
               "endpoint": suite.systemone, "model_alias": args.model, "chat_enabled": args.chat,
               "image_enabled": args.image, "max_body_mb": args.max_body_mb,
               "chat_seed": args.chat_seed, "chat_max_tokens": args.chat_max_tokens,
               "chat_prompt": args.chat_prompt, "chat_expected": args.chat_expected,
               "passed": passed, "cases": suite.records,
               "limitations": ["Integration screen; no claim of general accuracy, calibration or reference performance parity.",
                               "Concurrent HTTP arrival exercises serialization; it does not imply simultaneous GPU execution.",
                               "Chat checks were not executed." if not args.chat else "Ordinary chat settings come from the running server; generation may be expensive.",
                               "Image input was not exercised." if not args.image else
                               "One synthetic image screens that pixels reach the read; it is not an image-understanding benchmark."]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'} {sum(r['passed'] for r in suite.records)}/{len(suite.records)} checks; evidence: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
