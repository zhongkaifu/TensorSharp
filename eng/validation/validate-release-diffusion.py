#!/usr/bin/env python3
"""Qualify DiffusionGemma final text, replacement previews and request isolation.

HTTP diffusion uses a server-generated sampler seed. Retain full outputs and
wall times, but do not call its final-text delivery rate autoregressive decode
speed or infer deterministic token parity from the OpenAI sampling seed.
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
from validate_inference import assistant_content, SAMPLING


def save(path, report):
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_answer(text, tag):
    expected = {"tag": tag, "answer": 42}
    actual = json.loads(text)
    if actual != expected or type(actual.get("answer")) is not int:
        raise ValueError(f"Final content does not match {expected}")


def prompt(tag):
    return f'Return only JSON with exactly two keys: "tag" must be "{tag}" and "answer" must be the integer result of 17 + 25.'


def openai_case(args, tag, stream):
    request = {"messages": [{"role": "user", "content": prompt(tag)}], "max_tokens": 256,
               "stream": stream, "extra_body": {**SAMPLING, **engines.thinking_body("tensorsharp", False)}}
    result = {"scenario": "openai-final-stream" if stream else "openai-final-blocking", "tag": tag,
              "status": "fail", "request": request}
    start = time.monotonic()
    try:
        metrics = engines.run_openai_chat(args.url, args.model, timeout_s=args.timeout, **request)
        result["metrics"] = metrics
        check_answer(assistant_content(metrics), tag)
        if metrics.get("finish_reason") != "stop" or not metrics.get("usage_present"):
            raise ValueError("Missing completed final-text response or usage")
        result["status"] = "ok"
    except Exception as error:
        result["error"] = str(error)
    result["wall_seconds"] = time.monotonic() - start
    result["timing_interpretation"] = "Whole diffusion generation and final delivery; decode_tps is not an autoregressive decode metric."
    return result


def web_case(args, tag, first_preview=None, cancel=False):
    result = {"scenario": "web-cancel" if cancel else "web-replace", "tag": tag, "status": "fail", "events": []}
    session = None
    start = time.monotonic()
    try:
        response = requests.post(args.url + "/api/sessions", timeout=30)
        response.raise_for_status()
        session = response.json()["sessionId"]
        body = {"sessionId": session, "messages": [{"role": "user", "content": prompt(tag)}],
                "maxTokens": 512 if cancel else 256, "think": False}
        result.update(session_id=session, request=body)
        with requests.post(args.url + "/api/chat", json=body, stream=True, timeout=(30, args.timeout)) as response:
            result["http_status"] = response.status_code
            response.raise_for_status()
            for raw in response.iter_lines(chunk_size=1):
                if not raw.startswith(b"data:"):
                    continue
                frame = json.loads(raw[5:].decode("utf-8"))
                result["events"].append({"elapsed_seconds": time.monotonic() - start, "frame": frame})
                if frame.get("preview") is True:
                    if first_preview:
                        first_preview.set()
                    if cancel:
                        result["client_closed_after_preview"] = True
                        break
        if cancel:
            if not result.get("client_closed_after_preview"):
                raise ValueError("No in-flight preview was observed before cancellation")
            result["status"] = "ok"
            result["qualification"] = "Client disconnect observed; require matching server abort log and recovery case to establish cancellation."
        else:
            frames = [item["frame"] for item in result["events"]]
            previews = [frame for frame in frames if frame.get("preview") is True]
            finals = [frame for frame in frames if "replace" in frame and frame.get("preview") is False]
            done = [frame for frame in frames if frame.get("done") is True]
            if not previews or len(finals) != 1 or len(done) != 1 or frames[-1] != done[0]:
                raise ValueError("Expected preview replacements, one final replacement and a terminal done frame")
            if frames.index(finals[0]) >= frames.index(done[0]):
                raise ValueError("Final replacement must precede done")
            if any("token" in frame or "tool_calls" in frame or "thinking" in frame for frame in frames):
                raise ValueError("Diffusion preview stream unexpectedly uses append/tool/reasoning frames")
            if done[0].get("error") or done[0].get("aborted") or done[0].get("tokenCount", 0) <= 0:
                raise ValueError("Diffusion request did not finish successfully with token usage")
            check_answer(finals[0]["replace"], tag)
            result.update(status="ok", output_text=finals[0]["replace"], preview_count=len(previews), final=done[0])
    except Exception as error:
        result["error"] = str(error)
    finally:
        if first_preview:
            first_preview.set()  # A failed first request must not strand the late-arrival controller.
        if session and not cancel:
            try:
                response = requests.delete(args.url + "/api/sessions/" + session, timeout=30)
                result["session_cleanup_http"] = response.status_code
            except Exception as error:
                result["session_cleanup_error"] = str(error)
    result["wall_seconds"] = time.monotonic() - start
    return result


def unsupported_tool_probe(args):
    body = {"model": args.model, "messages": [{"role": "user", "content": "Call probe now."}],
            "tools": [{"type": "function", "function": {"name": "probe", "description": "A validation fixture.",
                       "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}],
            "tool_choice": "required", "stream": False, "max_tokens": 256}
    result = {"scenario": "unsupported-tool-rejection", "request": body, "status": "fail"}
    try:
        response = requests.post(args.url + "/v1/chat/completions", json=body, timeout=(30, args.timeout))
        result.update(http_status=response.status_code, body=response.text)
        if response.status_code != 400 or "error" not in response.json():
            raise ValueError("Unsupported required tool request was not explicitly rejected")
        result["status"] = "ok"
    except Exception as error:
        result["error"] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.url = args.url.rstrip("/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "run_complete": False, "cases": [],
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Final text, replacement previews, concurrent and late-arriving requests, disconnect recovery and unsupported-tool rejection.",
              "limitations": ["HTTP sampler uses Random.Shared.Next(); the client sampling seed does not pin its diffusion seed.",
                              "HTTP previews omit the scheduler block index. Late-arrival correctness is exercised, but exact block-boundary admission requires the separate scheduler/native tests."]}
    report["warmup"] = openai_case(args, "diffusion-warmup", False)
    save(args.output, report)
    for stream in (False, True):
        for degree in (1, 4):
            for repeat in range(args.repeats):
                with ThreadPoolExecutor(max_workers=degree) as pool:
                    cases = list(pool.map(lambda index: openai_case(args, f"d-{stream}-{degree}-{repeat}-{index}", stream), range(degree)))
                for case in cases:
                    case.update(concurrency=degree, repeat=repeat)
                report["cases"].extend(cases)
                save(args.output, report)
    with ThreadPoolExecutor(max_workers=4) as pool:
        report["cases"].extend(pool.map(lambda index: web_case(args, f"preview-{index}"), range(4)))
    save(args.output, report)
    arrived = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(web_case, args, "early-request", arrived)
        if not arrived.wait(args.timeout):
            raise TimeoutError("Initial diffusion request produced no preview")
        second = pool.submit(web_case, args, "late-request")
        report["cases"].extend([first.result(), second.result()])
    save(args.output, report)
    cancelled = web_case(args, "cancelled-request", cancel=True)
    report["cases"].append(cancelled)
    report["cases"].append(web_case(args, "after-cancel-recovery"))
    if args.server_log and cancelled.get("session_id"):
        lines = args.server_log.read_text(errors="replace").splitlines()
        evidence = [line for line in lines if "diffusion chat aborted by client" in line and cancelled["session_id"] in line]
        cancelled["server_abort_evidence"] = evidence
        if not evidence:
            cancelled.update(status="fail", error="Client closed the stream but no matching server cancellation log was observed")
    else:
        cancelled.update(status="fail", error="Server log is required to verify cancellation beyond the client disconnect")
    if cancelled.get("session_id"):
        try:
            response = requests.delete(args.url + "/api/sessions/" + cancelled["session_id"], timeout=30)
            cancelled["session_cleanup_http"] = response.status_code
        except Exception as error:
            cancelled["session_cleanup_error"] = str(error)
    save(args.output, report)
    report["cases"].append(unsupported_tool_probe(args))
    report["run_complete"] = True
    save(args.output, report)
    return int(report["warmup"]["status"] != "ok" or any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
