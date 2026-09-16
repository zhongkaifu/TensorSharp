#!/usr/bin/env python3
"""Exercise mixed-length arrival, departure, replacement and disconnect recovery.

Every request retains its exact body, SSE frames and monotonic timeline. These
are concurrency correctness probes; qualified throughput remains a separate,
steady workload measurement.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/engine_comparison"))
import engines
from validate_inference import SAMPLING, case_spec, check_answer


def save(path, report):
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def request_body(model, scenario, tag):
    # case_spec describes engine arguments. Its optional None values are not
    # wire fields: engines.run_openai_chat omits absent tools/response_format.
    spec = case_spec(scenario, tag)
    body = {"model": model, "messages": spec["messages"], "max_tokens": spec["max_tokens"],
            **SAMPLING, **engines.thinking_body("tensorsharp", False),
            "stream": True, "stream_options": {"include_usage": True}}
    for name in ("tools", "response_format"):
        if spec.get(name):
            body[name] = spec[name]
    if scenario in ("decode", "decode_8k"):
        body["max_tokens"] = 1024
    return body


def request_case(args, scenario, tag, first_token=None):
    body = request_body(args.model, scenario, tag)
    result = {"scenario": scenario, "tag": tag, "status": "fail", "request": body,
              "events": [], "started_monotonic": time.monotonic()}
    content, usage, finish, sentinel = [], {}, None, False
    try:
        with requests.post(args.url + "/v1/chat/completions", json=body, stream=True,
                           timeout=(30, args.timeout)) as response:
            result["http_status"] = response.status_code
            if response.status_code >= 400:
                result["http_error_body"] = response.text
            response.raise_for_status()
            for raw in response.iter_lines(chunk_size=1):
                if not raw.startswith(b"data:"):
                    continue
                now = time.monotonic()
                payload = raw[5:].strip().decode("utf-8")
                result["events"].append({"monotonic": now, "data": payload})
                if payload == "[DONE]":
                    sentinel = True
                    break
                frame = json.loads(payload)
                if frame.get("error"):
                    raise ValueError(str(frame["error"]))
                if frame.get("usage"):
                    usage = frame["usage"]
                for choice in frame.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("content") or delta.get("reasoning_content"):
                        result.setdefault("first_token_monotonic", now)
                        if first_token:
                            first_token.set()
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        text = "".join(content)
        result.update(output_text=text, usage=usage, finish_reason=finish)
        if not sentinel or not usage.get("prompt_tokens") or not usage.get("completion_tokens"):
            raise ValueError("Missing final SSE sentinel or real token usage")
        if finish not in ("stop", "length") or not check_answer(scenario, text):
            raise ValueError("Mixed-arrival response failed its independent semantic check")
        result["status"] = "ok"
    except Exception as error:
        result["error"] = str(error)
    finally:
        if first_token:
            first_token.set()
    result["finished_monotonic"] = time.monotonic()
    return result


def cancelled_web_request(args, tag, first_token):
    result = {"scenario": "client-disconnect", "tag": tag, "status": "fail", "events": [],
              "started_monotonic": time.monotonic()}
    try:
        response = requests.post(args.url + "/api/sessions", timeout=30)
        response.raise_for_status()
        session = response.json()["sessionId"]
        body = {"sessionId": session, "messages": case_spec("decode", tag)["messages"],
                "maxTokens": 1024, "think": False, "temperature": 0}
        result.update(session_id=session, request=body)
        seen = 0
        with requests.post(args.url + "/api/chat", json=body, stream=True,
                           timeout=(30, args.timeout)) as response:
            result["http_status"] = response.status_code
            response.raise_for_status()
            for raw in response.iter_lines(chunk_size=1):
                if not raw.startswith(b"data:"):
                    continue
                frame = json.loads(raw[5:].decode("utf-8"))
                now = time.monotonic()
                result["events"].append({"monotonic": now, "frame": frame})
                if frame.get("token") or frame.get("thinking"):
                    seen += 1
                    result.setdefault("first_token_monotonic", now)
                    first_token.set()
                if seen >= 8:
                    result.update(client_disconnected_monotonic=time.monotonic(), status="ok")
                    break
            if seen < 8:
                raise ValueError("Generation ended before the requested in-flight disconnect")
    except Exception as error:
        result.update(status="fail", error=str(error))
    finally:
        first_token.set()
    result["finished_monotonic"] = time.monotonic()
    return result


def qualify_cancel_peer(cancelled, survivor):
    """Require the peer request to remain in flight at the actual disconnect."""
    disconnect = cancelled.get("client_disconnected_monotonic")
    start, finish = survivor.get("started_monotonic"), survivor.get("finished_monotonic")
    overlap = (all(isinstance(value, (int, float)) for value in (disconnect, start, finish))
               and start < disconnect < finish)
    cancelled["peer_active_at_disconnect"] = overlap
    if not overlap:
        cancelled["status"] = "fail"
        cancelled["peer_overlap_error"] = "No recorded peer request spanning the client disconnect"
    return overlap


def qualify_session_cleanup(cancelled, response=None, error=None):
    """A recorded rejection or transport error cannot qualify session cleanup."""
    passed = False
    if response is not None:
        cancelled["session_cleanup_http"] = response.status_code
        try:
            value = response.json()
            cancelled["session_cleanup_response"] = value
            passed = (200 <= response.status_code < 300 and isinstance(value, dict)
                      and value.get("ok") is True and value.get("sessionId") == cancelled.get("session_id"))
        except (ValueError, TypeError) as decode_error:
            error = decode_error
    if error is not None:
        cancelled["session_cleanup_error"] = str(error)
    cancelled["session_cleanup_passed"] = passed
    if not passed:
        cancelled["status"] = "fail"
    return passed


def wait_server_abort(cancelled, server_log, timeout=120, clock=time.monotonic, pause=time.sleep):
    """Require the matching asynchronous abort acknowledgment before cleanup."""
    session = cancelled.get("session_id")
    started = clock()
    deadline = started + timeout
    evidence = []
    while session:
        evidence = [line for line in server_log.read_text(errors="replace").splitlines()
                    if "Web UI chat aborted by client" in line and f"sessionId={session}," in line]
        if evidence or clock() >= deadline:
            break
        pause(min(0.25, max(0, deadline - clock())))
    cancelled["server_abort_evidence"] = evidence
    cancelled["server_abort_wait_seconds"] = clock() - started
    cancelled["server_abort_timeout_seconds"] = timeout
    if not evidence:
        cancelled["status"] = "fail"
        cancelled["server_abort_error"] = "No matching server abort log after client disconnect"
    return bool(evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "run_complete": False, "cases": [], "waves": [],
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Actual overlapping long/short arrivals, departure and replacement, client disconnect with unaffected peer and recovery; not qualified throughput."}
    save(args.output, report)
    for repeat in range(args.repeats):
        first_token = threading.Event()
        with ThreadPoolExecutor(max_workers=3) as pool:
            anchor = pool.submit(request_case, args, "decode_8k", f"arrival-anchor-{repeat}", first_token)
            if not first_token.wait(args.timeout):
                raise TimeoutError("Anchor request produced no token")
            short = pool.submit(request_case, args, "short", f"arrival-short-{repeat}")
            long_request = pool.submit(request_case, args, "long_8k", f"arrival-long-{repeat}")
            short_result = short.result()
            replacement = pool.submit(request_case, args, "short", f"replacement-{repeat}")
            cases = [anchor.result(), short_result, long_request.result(), replacement.result()]
        overlap = cases[0].get("first_token_monotonic", float("inf")) <= cases[1]["started_monotonic"] < cases[0]["finished_monotonic"]
        replaced_while_active = cases[3]["started_monotonic"] < max(cases[0]["finished_monotonic"], cases[2]["finished_monotonic"])
        report["waves"].append({"repeat": repeat, "actual_overlap": overlap, "replacement_while_peer_active": replaced_while_active})
        for case in cases:
            case["repeat"] = repeat
        report["cases"].extend(cases)
        save(args.output, report)
        first_token = threading.Event()
        with ThreadPoolExecutor(max_workers=2) as pool:
            cancelled = pool.submit(cancelled_web_request, args, f"cancel-{repeat}", first_token)
            if not first_token.wait(args.timeout):
                raise TimeoutError("Cancellation request produced no token")
            survivor = pool.submit(request_case, args, "long_8k", f"cancel-survivor-{repeat}")
            cancelled_result, survivor_result = cancelled.result(), survivor.result()
        qualify_cancel_peer(cancelled_result, survivor_result)
        wait_server_abort(cancelled_result, args.server_log)
        recovery = request_case(args, "short", f"cancel-recovery-{repeat}")
        session = cancelled_result.get("session_id")
        if session:
            try:
                response = requests.delete(args.url + "/api/sessions/" + session, timeout=30)
                qualify_session_cleanup(cancelled_result, response=response)
            except Exception as error:
                qualify_session_cleanup(cancelled_result, error=error)
        report["cases"].extend([cancelled_result, survivor_result, recovery])
        save(args.output, report)
    report["run_complete"] = True
    save(args.output, report)
    return int(any(case["status"] != "ok" for case in report["cases"]) or
               any(not wave["actual_overlap"] or not wave["replacement_while_peer_active"] for wave in report["waves"]))


if __name__ == "__main__":
    raise SystemExit(main())
