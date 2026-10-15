#!/usr/bin/env python3
"""Exercise an already-running unified-memory chat host; never starts a server.

Example (the server file logger must include RequestId scopes):
  python eng/validation/unified-server-http-checks.py --url http://127.0.0.1:18891 \
    --model model.gguf --image artifacts/multimodal-local-preflight/vision-fixtures/card-0.png \
    --server-log artifacts/server/server.jsonl --output artifacts/unified-http-v1

All HTTP calls have an absolute deadline (default/max 300 seconds). Requests,
raw responses, queue samples, and only newly appended server logs are retained.
Protocol 'stop' is NOT sufficient evidence of EOS: strict success also requires
same-request server telemetry finishReason=eos. Cancellation similarly requires
correlated server evidence, not just a closed client socket. No generated code
is evaluated. Parallel submission tests overlapping clients and isolation;
native fused-graph engagement requires the separate graph evidence checker.
Reload success is functional scope-recreation
evidence, not an independent zero-ledger/driver-memory measurement.
Use --default-output-limit-check to add a small arithmetic request that omits
both OpenAI output-limit fields and exercises the server's configured default.
"""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
MAX_BODY = 8 << 20
TIMEOUT = 300.0
HEADER = "X-TensorSharp-Request-Id"
TEXT = "What is 17 + 25? Reply with only the integer."
OTHER = "What is 100 - 19? Reply with only the integer."
OCR = "Read the four-digit code in this image. Reply with only those four digits."
CANCEL = "Write the integers from 1 through 200, one integer per line, without skipping any integer or adding commentary."


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_timeout(value):
    result = float(value)
    require(math.isfinite(result) and 0 < result <= TIMEOUT, "HTTP timeout must be finite, positive and <=300s")
    return result


def evidence_output_path(value, root=ROOT):
    # Check the requested checkout namespace before resolving a directory
    # junction. Large ignored artifact directories may legitimately live on D:.
    raw = Path(value)
    require(".." not in raw.parts, "Evidence path must not contain parent traversal")
    lexical = Path(os.path.abspath(raw))
    require(any(lexical.is_relative_to(Path(os.path.abspath(root / part)))
                for part in ("artifacts", "docs/validation")), "Output must be ignored checkout evidence")
    return lexical.resolve()


def require_model_status(body, model, require_unified=False, architecture="qwen4exp"):
    require(architecture in ("qwen4exp", "glm5-next", "glm5next"), "Unknown qualified architecture")
    require(isinstance(body, dict) and body.get("loaded") == model and body.get("architecture") == architecture
            and body.get("visionReady") is True, "Requested model architecture and vision projector must already be loaded")
    if require_unified:
        require(body.get("unifiedMemory") is True, "Server did not report unifiedMemory=true")


def queue_state(body):
    require(isinstance(body, dict), "Queue status is not an object")
    for key in ("processing", "pending_requests", "total_processed"):
        require(type(body.get(key)) is int and body[key] >= 0, "Missing/invalid queue counter: " + key)
    require(type(body.get("busy")) is bool and body["busy"] == (body["processing"] > 0), "Inconsistent queue busy state")
    return body["processing"] == 0 and body["pending_requests"] == 0


def answer_gate(row, expected):
    checks = {"http_ok": row.get("status") == 200, "transport_complete": row.get("transport_complete") is True,
              "no_transport_error": not row.get("error") and not row.get("cleanup_errors"), "request_id_echo": row.get("request_id_echo") is True}
    body = row.get("json")
    choices = body.get("choices") if isinstance(body, dict) else None
    valid = isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
    choice = choices[0] if valid else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    checks.update(single_choice=valid, exact_answer=isinstance(content, str) and content.strip() == expected,
                  protocol_stop=choice.get("finish_reason") == "stop", no_tool_calls=not message.get("tool_calls"),
                  no_error=isinstance(body, dict) and not body.get("error"))
    usage = body.get("usage") if isinstance(body, dict) else None
    checks["positive_token_usage"] = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] > 0
        for k in ("prompt_tokens", "completion_tokens"))
    # Filled from the independently preserved, same-RequestId server log.
    checks["natural_eos_telemetry"] = row.get("natural_eos_telemetry") is True
    return {"passed": all(checks.values()), "checks": checks, "expected": expected, "content": content}


def default_output_limit_gate(row):
    result = answer_gate(row, "42")
    request = row.get("request")
    body = request.get("body") if isinstance(request, dict) else None
    result["checks"]["openai_chat_request"] = isinstance(request, dict) and request.get("method") == "POST" and request.get("route") == "/v1/chat/completions"
    result["checks"]["output_limits_omitted"] = isinstance(body, dict) and "max_tokens" not in body and "max_completion_tokens" not in body
    result["checks"]["original_arithmetic_prompt"] = isinstance(body, dict) and body.get("messages") == [{"role": "user", "content": TEXT}]
    result["passed"] = all(result["checks"].values())
    result["scope"] = "Both OpenAI output-limit fields omitted; actual server default exercised, not an explicit client cap."
    return result


def answer_suite_gate(rows, default_output_limit_check=False):
    expected = ["text", "image", "concurrent-42", "concurrent-81", "after-cancel", "after-reload"]
    if default_output_limit_check:
        expected.append("default-output-limit")
    checks = {"complete_named_cases": sorted(row.get("name", "") for row in rows) == sorted(expected),
              "all_answers_qualified": all(row.get("quality", {}).get("passed") is True for row in rows)}
    if default_output_limit_check:
        defaults = [row for row in rows if row.get("name") == "default-output-limit"]
        checks["default_limit_qualified"] = len(defaults) == 1 and default_output_limit_gate(defaults[0])["passed"]
    return {"passed": all(checks.values()), "checks": checks, "expected_cases": expected}


def overlap_gate(rows):
    if len(rows) != 2 or any(not all(type(r.get(k)) in (float, int) and math.isfinite(r[k]) for k in
        ("started_monotonic", "ended_monotonic")) for r in rows):
        return {"passed": False, "reason": "Two finite request intervals required"}
    overlap = min(r["ended_monotonic"] for r in rows) - max(r["started_monotonic"] for r in rows)
    return {"passed": overlap > 0, "client_interval_overlap_seconds": max(0, overlap),
            "start_skew_seconds": abs(rows[0]["started_monotonic"] - rows[1]["started_monotonic"]),
            "scope": "Two barrier-released independent HTTP connections; GPU execution may be serial."}


def _contains_request(value, request_id):
    if isinstance(value, dict):
        return value.get("RequestId") == request_id or any(_contains_request(v, request_id) for v in value.values())
    return isinstance(value, list) and any(_contains_request(v, request_id) for v in value)


def log_evidence(text, request_id):
    eos, cancelled, evidence = False, False, []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            record = None
        if isinstance(record, dict):
            # LogEntryFormatter emits the flattened active scopes as "scope".
            # Preserve compatibility with the older plural fixture/formatter;
            # if the canonical field is present it alone is authoritative.
            # json.loads keeps the last duplicate RequestId, matching nested
            # request scopes shadowing the outer ASP.NET trace identifier.
            scope = record["scope"] if "scope" in record else record.get("scopes")
            if not _contains_request(scope, request_id):
                continue
            message = record.get("message", "")
        else:
            if not re.search(r"\bRequestId=" + re.escape(request_id) + r"(?:[,}\]\s]|$)", line):
                continue
            # Text logger emits category + ': ' before the message and scopes after it.
            message = line.split(": ", 1)[-1]
        if not isinstance(message, str):
            continue
        completed = message.startswith(("chat.complete ", "generate.complete "))
        is_eos = completed and re.search(r"\bfinishReason=eos(?:\s|$)", message) is not None
        is_cancel = message.startswith(("chat.cancelled ", "generate.cancelled ")) or (
            message.startswith("HTTP ") and re.search(r"threw (?:OperationCanceledException|TaskCanceledException)\b", message) is not None)
        if is_eos or is_cancel:
            evidence.append(line)
            eos |= is_eos
            cancelled |= is_cancel
    return {"natural_eos": eos, "cancelled": cancelled, "lines": evidence}


def cancellation_gate(row, idle_after, server_cancelled):
    checks = {"http_ok": row.get("status") == 200, "request_id_echo": row.get("request_id_echo") is True,
              "three_content_events": row.get("content_events") == 3, "client_closed": row.get("client_closed") is True,
              "no_terminal_event_before_close": row.get("terminal_before_close") is False,
              "observed_busy_before_close": row.get("busy_before_close") is True,
              "queue_drained": idle_after is True, "server_cancel_confirmed": server_cancelled is True,
              "no_transport_or_cleanup_error": not row.get("error") and not row.get("cleanup_errors")}
    return {"passed": all(checks.values()), "checks": checks,
            "scope": "Client disconnect, correlated server cancellation, queue drain and subsequent recovery are separate observations."}


class Client:
    def __init__(self, url, output, timeout=TIMEOUT):
        self.url = urlsplit(url)
        require(self.url.scheme in ("http", "https") and self.url.hostname and not self.url.username
                and not self.url.query and not self.url.fragment, "Invalid server URL")
        self.output = Path(output)
        self.timeout = finite_timeout(timeout)
        self.counter = 0
        self.lock = threading.Lock()

    def call(self, name, route, payload=None, *, cancel_after_three=False, on_third=None, barrier=None, timeout=None):
        duration = self.timeout if timeout is None else min(self.timeout, finite_timeout(timeout))
        with self.lock:
            self.counter += 1
            folder = self.output / f"{self.counter:03d}-{name}"
            folder.mkdir()
        request_id = "unified-check-" + uuid.uuid4().hex
        request = {"method": "GET" if payload is None else "POST", "route": route, "body": payload,
                   "headers": {HEADER: request_id, "Content-Type": "application/json", "Connection": "close"}}
        save(folder / "request.json", request)
        connection_type = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        conn = connection_type(self.url.hostname, self.url.port, timeout=duration)
        response = None
        expired = threading.Event()
        sock_holder = []
        result = {"name": name, "request_id": request_id, "directory": str(folder), "request": request,
                  "timeout_seconds": duration, "transport_complete": False, "cleanup_errors": [],
                  "client_closed": False, "content_events": 0, "terminal_before_close": False}
        def expire():
            expired.set()
            # Shutdown the owned socket before closing buffered HTTP objects;
            # close alone can wait behind a reader lock indefinitely.
            sock = sock_holder[0] if sock_holder else conn.sock
            if sock is not None:
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                try: sock.close()
                except OSError: pass
        timer = threading.Timer(duration, expire)
        timer.daemon = True
        try:
            if barrier is not None:
                barrier.wait(timeout=duration)
            result["started_utc"] = datetime.now(timezone.utc).isoformat()
            result["started_monotonic"] = time.monotonic()
            timer.start()
            encoded = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            conn.connect()
            sock_holder.append(conn.sock)
            remaining = duration - (time.monotonic() - result["started_monotonic"])
            require(remaining > 0 and not expired.is_set(), "HTTP deadline during connect")
            conn.sock.settimeout(remaining)
            conn.request(request["method"], self.url.path.rstrip("/") + route, body=encoded, headers=request["headers"])
            response = conn.getresponse()
            result.update(status=response.status, headers=dict(response.getheaders()),
                          request_id_echo=response.getheader(HEADER) == request_id)
            total = 0
            with (folder / "response.raw").open("wb") as output:
                if cancel_after_three and response.status == 200:
                    events = []
                    while True:
                        raw = response.readline(1 << 20)
                        require(not expired.is_set() and time.monotonic() - result["started_monotonic"] < duration, "HTTP absolute deadline")
                        if not raw: break
                        total += len(raw)
                        require(total <= MAX_BODY and len(raw) < (1 << 20), "HTTP stream exceeds evidence bound")
                        output.write(raw); output.flush()
                        event = json.loads(raw)
                        require(isinstance(event, dict) and not event.get("error"), "Ollama stream error/malformed event")
                        require(type(event.get("done")) is bool, "Ollama stream missing boolean done state")
                        events.append(event)
                        if event.get("done") is True:
                            result["terminal_before_close"] = True
                            break
                        if isinstance(event.get("response"), str) and event["response"]:
                            result["content_events"] += 1
                        if result["content_events"] == 3:
                            result["busy_before_close"] = on_third() if on_third else None
                            result["client_closed"] = True
                            # Intentional disconnect has its own marker, not a timeout.
                            sock_holder[0].shutdown(socket.SHUT_RDWR)
                            break
                    result["events"] = events
                else:
                    while True:
                        raw = response.read(65536)
                        require(not expired.is_set() and time.monotonic() - result["started_monotonic"] < duration, "HTTP absolute deadline")
                        if not raw: break
                        total += len(raw)
                        require(total <= MAX_BODY, "HTTP response exceeds evidence bound")
                        output.write(raw)
                    result["transport_complete"] = True
            if not cancel_after_three or response.status != 200:
                result["json"] = json.loads((folder / "response.raw").read_text(encoding="utf-8"))
        except BaseException as error:
            result["error"] = f"{type(error).__name__}: {error}"
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            timer.cancel()
            if timer.ident is not None:
                try:
                    timer.join(timeout=2)
                    if timer.is_alive(): result["cleanup_errors"].append("deadline thread did not stop")
                except BaseException as error:
                    result["cleanup_errors"].append(f"deadline: {type(error).__name__}: {error}")
            for label, cleanup in (("response", lambda: response.close() if response else None), ("connection", conn.close)):
                try: cleanup()
                except BaseException as error: result["cleanup_errors"].append(f"{label}: {type(error).__name__}: {error}")
            result["timed_out"] = expired.is_set()
            result["ended_monotonic"] = time.monotonic()
            result["elapsed_seconds"] = result["ended_monotonic"] - result.get("started_monotonic", result["ended_monotonic"])
            if result["timed_out"] or result["cleanup_errors"]:
                result["transport_complete"] = False
            save(folder / "result.json", result)
        return result


def expect_json(row):
    require(row.get("status") == 200 and row.get("transport_complete") is True and not row.get("error")
            and not row.get("cleanup_errors"), "HTTP infrastructure failure: " + row["name"])
    return row["json"]


def wait_idle(client, label):
    deadline = time.monotonic() + client.timeout
    samples = []
    while time.monotonic() < deadline:
        row = client.call(label, "/api/queue/status", timeout=max(.001, deadline-time.monotonic()))
        samples.append(row)
        if queue_state(expect_json(row)):
            return True, samples
        time.sleep(min(.25, max(0, deadline-time.monotonic())))
    return False, samples


def chat_payload(model, text, image=None, *, use_default_output_limit=False):
    content = text if image is None else [{"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")}}]
    payload = {"model": model, "messages": [{"role": "user", "content": content}], "stream": False,
               "temperature": 0, "top_p": 1, "presence_penalty": 0, "frequency_penalty": 0,
               "seed": 1, "think": False}
    if not use_default_output_limit:
        payload["max_tokens"] = 64
    return payload


def run(args):
    output = evidence_output_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    client = Client(args.url, output, args.timeout)
    report = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(), "completed": False, "passed": False,
              "tool_sha256": sha(__file__), "args": vars(args), "answers": [], "checks": {}, "errors": [],
              "scope": "Local HTTP integration, serialized GPU adapter with overlapping clients. No physical hard-cap certificate. No model code execution. Natural EOS and cancellation require same-request server telemetry."}
    log_path = Path(args.server_log).resolve() if args.server_log else None
    log_offset = None
    report["server_log"] = {"path": str(log_path) if log_path else None, "initial_offset": log_offset}
    try:
        log_offset = log_path.stat().st_size if log_path else None
        report["server_log"]["initial_offset"] = log_offset
        image_path = Path(args.image).resolve()
        require(image_path.stat().st_size <= MAX_BODY, "Image exceeds evidence bound")
        image = image_path.read_bytes()
        require(image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) <= MAX_BODY, "Use the bounded PNG card-0 fixture")
        report["image"] = {"path": str(image_path), "sha256": hashlib.sha256(image).hexdigest(), "expected": "4821"}
        status = client.call("initial-model", "/api/models")
        model = expect_json(status)
        require_model_status(model, args.model, args.require_unified_status, args.architecture)
        report["initial_model"] = status
        idle, report["initial_queue"] = wait_idle(client, "initial-queue")
        require(idle, "Server must be idle before validation")
        if args.default_output_limit_check:
            row = client.call("default-output-limit", "/v1/chat/completions",
                              chat_payload(args.model, TEXT, use_default_output_limit=True))
            row["expected"] = "42"
            row["advertised_server_default_max_tokens"] = model.get("defaultMaxTokens")
            report["answers"].append(row)
            expect_json(row)
            save(output / "report.json", report)
        for label, prompt, expected, pixels in (("text", TEXT, "42", None), ("image", OCR, "4821", image)):
            row = client.call(label, "/v1/chat/completions", chat_payload(args.model, prompt, pixels))
            row["expected"] = expected; report["answers"].append(row)
            expect_json(row)
            save(output / "report.json", report)
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=2) as workers:
            calls = [workers.submit(client.call, label, "/v1/chat/completions", chat_payload(args.model, prompt), barrier=barrier)
                     for label, prompt in (("concurrent-42", TEXT), ("concurrent-81", OTHER))]
            paired = [call.result() for call in calls]
        for row, expected in zip(paired, ("42", "81")):
            row["expected"] = expected; report["answers"].append(row); expect_json(row)
        report["checks"]["overlapping_clients"] = overlap_gate(paired)
        idle, report["pre_cancel_queue"] = wait_idle(client, "pre-cancel-queue")
        require(idle, "Queue did not drain before cancellation case")
        busy_sample = []
        def busy():
            sample = client.call("cancel-live-queue", "/api/queue/status", timeout=10)
            busy_sample.append(sample)
            state = expect_json(sample); queue_state(state)
            return state["processing"] > 0
        cancelled = client.call("cancel", "/api/generate", {"model": args.model, "prompt": CANCEL, "stream": True,
            "options": {"temperature": 0, "top_p": 1, "repeat_penalty": 1, "num_predict": 1024, "seed": 1}},
            cancel_after_three=True, on_third=busy)
        report["cancellation"] = cancelled; report["cancel_live_queue"] = busy_sample
        idle, report["post_cancel_queue"] = wait_idle(client, "post-cancel-queue")
        report["cancel_queue_drained"] = idle
        require(idle, "Cancelled request still owns queue; do not reload or launch another generation")
        recovered = client.call("after-cancel", "/v1/chat/completions", chat_payload(args.model, OTHER))
        recovered["expected"] = "81"; report["answers"].append(recovered); expect_json(recovered)
        idle, report["pre_reload_queue"] = wait_idle(client, "pre-reload-queue")
        require(idle, "Do not reload an active queue")
        before_counter = report["pre_reload_queue"][-1]["json"]["total_processed"]
        reload_row = client.call("reload", "/api/models/load", {"model": args.model, "backend": "ggml_cuda"})
        report["reload"] = reload_row
        loaded = expect_json(reload_row)
        require(loaded.get("ok") is True and loaded.get("model") == args.model and loaded.get("architecture") == args.architecture,
                "Reload was refused or loaded another model")
        require(isinstance(model.get("loadedMmProj"), str) and loaded.get("loadedMmProj") == model["loadedMmProj"],
                "Reload changed or lost the startup projector")
        reloaded_status = client.call("reloaded-model", "/api/models")
        report["reloaded_model"] = reloaded_status
        reloaded_model = expect_json(reloaded_status)
        require_model_status(reloaded_model, args.model, args.require_unified_status, args.architecture)
        require(reloaded_model.get("loaded") == args.model and reloaded_model.get("visionReady") is True
                and reloaded_model.get("loadedMmProj") == model["loadedMmProj"], "Reload lost the vision runtime")
        reset = client.call("reload-queue", "/api/queue/status")
        report["reload_queue"] = reset
        require(queue_state(expect_json(reset)), "Reload did not leave an idle engine")
        report["checks"]["reload_recreated_engine"] = {"passed": before_counter > 0 and reset["json"]["total_processed"] == 0,
            "before_total_processed": before_counter, "after_total_processed": reset["json"]["total_processed"],
            "scope": "Successful same-model reload and reset engine counter; scope detach/recreate inferred from exclusive loader contract, not independent ledger measurement."}
        final = client.call("after-reload", "/v1/chat/completions", chat_payload(args.model, TEXT))
        final["expected"] = "42"; report["answers"].append(final); expect_json(final)
        idle, report["final_queue"] = wait_idle(client, "final-queue")
        require(idle, "Final queue did not drain")
        report["completed"] = True
    except BaseException as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        log_text = ""
        try:
            if log_path and log_offset is not None:
                # File logging may flush asynchronously after HTTP completion.
                # Bound this grace period; missing telemetry remains unverified.
                flush_deadline = time.monotonic() + 2
                while True:
                    require(log_path.stat().st_size >= log_offset, "Server log rotated/truncated during verification")
                    with log_path.open("rb") as source:
                        source.seek(log_offset); raw = source.read(32 << 20)
                        require(not source.read(1), "New server log exceeds evidence bound")
                    pending_text = raw.decode("utf-8", errors="replace")
                    have_answers = all(log_evidence(pending_text, row["request_id"])["natural_eos"] for row in report["answers"])
                    cancel_id = report.get("cancellation", {}).get("request_id")
                    have_cancel = not cancel_id or log_evidence(pending_text, cancel_id)["cancelled"]
                    if (have_answers and have_cancel) or time.monotonic() >= flush_deadline:
                        break
                    time.sleep(.05)
                (output / "server-new.log").write_bytes(raw)
                log_text = raw.decode("utf-8", errors="replace")
                report["server_log"].update(final_offset=log_path.stat().st_size, captured_sha256=hashlib.sha256(raw).hexdigest())
            for row in report["answers"]:
                evidence = log_evidence(log_text, row["request_id"])
                row["server_evidence"] = evidence; row["natural_eos_telemetry"] = evidence["natural_eos"]
                row["quality"] = answer_gate(row, row["expected"])
                if row["name"] == "default-output-limit":
                    row["quality"] = default_output_limit_gate(row)
            cancelled = report.get("cancellation", {})
            evidence = log_evidence(log_text, cancelled.get("request_id", "missing"))
            cancelled["server_evidence"] = evidence
            report["checks"]["cancellation"] = cancellation_gate(cancelled, report.get("cancel_queue_drained"), evidence["cancelled"])
        except BaseException as error:
            report["errors"].append("Evidence finalization: " + type(error).__name__ + ": " + str(error))
        report["checks"]["answer_suite"] = answer_suite_gate(report["answers"], args.default_output_limit_check)
        report["passed"] = report["completed"] and not report["errors"] and all(
            report["checks"].get(key, {}).get("passed") is True for key in ("answer_suite", "overlapping_clients", "cancellation", "reload_recreated_engine"))
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "report.json", report)
    print(json.dumps({"report": str(output / "report.json"), "completed": report["completed"], "passed": report["passed"], "errors": report["errors"]}))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True, help="Exact currently hosted startup model filename")
    parser.add_argument("--architecture", choices=("qwen4exp", "glm5-next", "glm5next"), default="qwen4exp")
    parser.add_argument("--image", required=True, help="card-0 PNG containing 4821")
    parser.add_argument("--server-log", help="Existing file logger with RequestId scopes; absent means EOS/cancellation unverified, never passing")
    parser.add_argument("--require-unified-status", action="store_true", help="Require initial and reloaded /api/models to report unifiedMemory=true; older servers without the field fail this check")
    parser.add_argument("--default-output-limit-check", action="store_true", help="Additionally require correct natural-EOS arithmetic with max_tokens and max_completion_tokens both omitted")
    parser.add_argument("--output", required=True, help="Fresh ignored evidence directory")
    parser.add_argument("--timeout", type=finite_timeout, default=TIMEOUT)
    return run(parser.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
