#!/usr/bin/env python3
"""Measure request isolation and continuous admission on a running Qwen4Exp host.

Uses only Python's standard library. Reports retain complete answers, SSE events,
per-request TTFT, token usage, overlap, and serial/concurrent comparisons. Topic
checks screen relevance and obvious leakage; exact marker tasks screen request
mixups. These checks do not establish unrestricted factual quality or logit
parity. Save the served model/build/device/upstream identity with --provenance.
The optional --server-log gate requires a NEW fused-decode acceptance message
during this invocation, rather than treating a startup claim as execution proof.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import statistics
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
LONG_MARKER_ANSWER = "\n".join(f"EXACT_BETA_{index:02d}=73" for index in range(1, 25))
FIXTURES = {
    "ff7": {
        "prompt": "请详细介绍最终幻想7",
        "kind": "topic", "subject": ["最终幻想", "Final Fantasy"],
        "anchors": ["克劳德", "Cloud", "神罗", "Shinra", "米德加", "Midgar", "萨菲罗斯", "Sephiroth", "蒂法", "Tifa", "1997", "史克威尔", "Square"],
        "forbidden": ["时间简史", "霍金", "Stephen Hawking"],
    },
    "time_history": {
        "prompt": "请详细介绍时间简史",
        "kind": "topic", "subject": ["时间简史", "Brief History of Time"],
        "anchors": ["霍金", "Hawking", "黑洞", "black hole", "宇宙", "universe", "大爆炸", "Big Bang", "1988", "相对论", "relativity"],
        "forbidden": ["最终幻想", "克劳德", "萨菲罗斯", "神罗", "Final Fantasy"],
    },
    "marker_short": {
        "prompt": "这是一条独立任务。计算 17 加 25。只输出 EXACT_ALPHA=42，不要解释，不要输出其他内容。",
        "kind": "exact", "expected": "EXACT_ALPHA=42", "max_tokens": 40,
        "forbidden": ["EXACT_BETA", "EXACT_GAMMA", "时间简史", "最终幻想"],
    },
    "marker_long": {
        "prompt": "这是一条独立任务。完整地按顺序输出下面的 24 行，不要省略、不要解释、不要添加列表序号或其他内容。\n" + LONG_MARKER_ANSWER,
        "kind": "exact", "expected": LONG_MARKER_ANSWER, "max_tokens": 320,
        "forbidden": ["EXACT_ALPHA", "EXACT_GAMMA", "时间简史", "最终幻想"],
    },
    "marker_reuse": {
        "prompt": "这是一条新的独立任务。计算 100 减 19。只输出 EXACT_GAMMA=81，不要解释，不要输出其他内容。",
        "kind": "exact", "expected": "EXACT_GAMMA=81", "max_tokens": 40,
        "forbidden": ["EXACT_ALPHA", "EXACT_BETA", "时间简史", "最终幻想"],
    },
}
GROUPS = {"topics": ["ff7", "time_history"], "markers": ["marker_long", "marker_short"]}
FUSED_ACCEPTED = re.compile(r"Batched fused decode accepted\s+(\d+)\s+sequences in one graph", re.I)
FUSED_DECLINED = re.compile(r"(?:batched fused.decode|fused batched.decode|fused.decode).{0,100}(?:declin|round.robin)|serving sequences round.robin", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def evidence_path(value):
    path = Path(value).resolve()
    require(any(path.is_relative_to((ROOT / directory).resolve()) for directory in ("artifacts", "docs/validation")),
            "Output must be in this checkout's ignored artifacts/ or docs/validation/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def quality_check(fixture, answer):
    text = answer.strip()
    folded = text.casefold()
    leaked = [value for value in fixture["forbidden"] if value.casefold() in folded]
    checks = {"no_other_request_topic_or_marker": not leaked, "nonempty": bool(text)}
    detail = {"leaked_terms": leaked}
    if fixture["kind"] == "exact":
        # Whitespace around lines is immaterial, but explanations, missing lines,
        # code fences, changed markers, and incorrect arithmetic fail the gate.
        normalized = "\n".join(line.strip() for line in text.splitlines())
        checks["exact_task_answer"] = normalized == fixture["expected"]
    else:
        subject = [value for value in fixture["subject"] if value.casefold() in folded]
        anchors = [value for value in fixture["anchors"] if value.casefold() in folded]
        checks.update(subject_present=bool(subject), at_least_two_topic_anchors=len(anchors) >= 2,
                      substantive_answer=len(text) >= 80)
        detail.update(subject_terms=subject, topic_anchors=anchors, answer_characters=len(text))
    return {"passed": all(checks.values()), "checks": checks, **detail}


class Client:
    def __init__(self, url, timeout):
        self.url = urlsplit(url)
        require(self.url.scheme in ("http", "https") and self.url.hostname,
                "--url must be an HTTP(S) server URL")
        require(not self.url.username and not self.url.query and not self.url.fragment,
                "--url cannot contain credentials, query or fragment")
        self.timeout = timeout

    def connection(self):
        cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        return cls(self.url.hostname, self.url.port, timeout=self.timeout)

    def path(self, route):
        return self.url.path.rstrip("/") + route

    def models(self):
        connection = self.connection()
        try:
            connection.request("GET", self.path("/v1/models"))
            response = connection.getresponse()
            raw = response.read()
            require(response.status == 200, f"Models: HTTP {response.status}: {raw[:1024]!r}")
            return json.loads(raw)["data"]
        finally:
            connection.close()

    def chat(self, body, row, first_delta=None, cancel_after_deltas=None):
        connection = self.connection()
        row.update(answer="", reasoning="", events=[], done=False, cancelled=False,
                   ttft_ms=None, first_answer_ms=None, finish_reason=None)
        started = time.perf_counter()
        row["started_monotonic_s"] = started
        deadline = time.monotonic() + self.timeout
        delta_count = 0
        usage = None
        try:
            connection.request("POST", self.path("/v1/chat/completions"),
                               json.dumps(body, ensure_ascii=False).encode(), {"Content-Type": "application/json"})
            response = connection.getresponse()
            row["http_status"] = response.status
            if response.status != 200:
                raise ValueError(f"Chat: HTTP {response.status}: {response.read()[:2048]!r}")
            require("text/event-stream" in response.getheader("Content-Type", ""), "Response is not SSE")
            for line in response:
                require(time.monotonic() <= deadline, "Request exceeded its deadline")
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip().decode()
                if not raw:
                    continue
                elapsed = (time.perf_counter() - started) * 1000
                event = "[DONE]" if raw == "[DONE]" else json.loads(raw)
                row["events"].append({"elapsed_ms": elapsed, "data": event})
                if event == "[DONE]":
                    row["done"] = True
                    break
                require(isinstance(event, dict) and not event.get("error"), f"SSE error: {event}")
                if event.get("usage") is not None:
                    usage = event["usage"]
                emitted = False
                for choice in event.get("choices", []):
                    row["finish_reason"] = choice.get("finish_reason") or row["finish_reason"]
                    delta = choice.get("delta", {})
                    require(not delta.get("tool_calls"), "Unexpected tool call")
                    for wire_field, field in (("content", "answer"), ("reasoning_content", "reasoning")):
                        piece = delta.get(wire_field)
                        if piece:
                            require(isinstance(piece, str), f"Invalid {wire_field} delta")
                            emitted = True
                            row[field] += piece
                            if row["ttft_ms"] is None:
                                row["ttft_ms"] = elapsed
                                if first_delta:
                                    first_delta.set()
                            if field == "answer" and row["first_answer_ms"] is None:
                                row["first_answer_ms"] = elapsed
                            row["last_delta_ms"] = elapsed
                if emitted:
                    delta_count += 1
                if cancel_after_deltas is not None and delta_count >= cancel_after_deltas:
                    require(row["finish_reason"] is None, "Request reached a finish reason before intentional cancellation")
                    row["cancelled"] = True
                    break
            if cancel_after_deltas is not None:
                require(row["cancelled"] and not row["done"], "Request finished before intentional cancellation")
            else:
                require(row["done"], "SSE stream ended without [DONE]")
                require(row["answer"].strip(), "Model returned no answer")
                require(not row["answer"].lstrip().startswith("Error:"), "Server returned an error as answer text")
                require(row["finish_reason"] in ("stop", "length"), f"Unexpected finish reason: {row['finish_reason']}")
                require(isinstance(usage, dict), "Stream omitted final usage")
                for field in ("prompt_tokens", "completion_tokens"):
                    value = usage.get(field)
                    require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"Invalid {field}")
                row["usage"] = usage
                row["prompt_tokens"] = usage["prompt_tokens"]
                row["completion_tokens"] = usage["completion_tokens"]
                row["cached_tokens"] = usage.get("prompt_tokens_details", {}).get("cached_tokens")
        finally:
            # A disconnect is the actual cancellation signal, not a completed
            # request that happened to have max_tokens=1.
            connection.close()
            row["ended_monotonic_s"] = time.perf_counter()
            row["elapsed_ms"] = (row["ended_monotonic_s"] - started) * 1000
            row["answer_sha256"] = sha256(row["answer"].encode())
            row["nonempty_delta_events"] = delta_count


def body_for(args, fixture_name):
    fixture = FIXTURES[fixture_name]
    return {"model": args.model, "messages": [{"role": "user", "content": fixture["prompt"]}],
            "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": fixture.get("max_tokens", args.max_tokens), "think": False,
            "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0,
            "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0,
            "seed": args.seed, "skills": [], "skills_discovery": False, "multi_agent": False,
            "tool_choice": "none"}


def run_request(args, client, fixture_name, first_delta=None, cancel_after_deltas=None):
    row = {"fixture": fixture_name, "status": "failed", "request": body_for(args, fixture_name)}
    row["request_sha256"] = sha256(json.dumps(row["request"], ensure_ascii=False, sort_keys=True).encode())
    try:
        client.chat(row["request"], row, first_delta, cancel_after_deltas)
        if cancel_after_deltas is None:
            row["quality"] = quality_check(FIXTURES[fixture_name], row["answer"])
            require(row["quality"]["passed"], "Answer failed task/relevance/isolation checks")
        elif FIXTURES[fixture_name]["kind"] == "exact":
            normalized = "\n".join(line.strip() for line in row["answer"].strip().splitlines())
            require(normalized != FIXTURES[fixture_name]["expected"], "Complete task answer arrived before intentional cancellation")
        row["status"] = "cancelled_as_requested" if cancel_after_deltas is not None else "passed"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        # Prevent a stalled second admission if the first request failed before
        # producing any delta; its missing TTFT still fails the evidence gate.
        if first_delta:
            first_delta.set()
    return row


def group_metrics(rows, wall_ms):
    completed = [row for row in rows if row.get("done") and row.get("completion_tokens") is not None]
    tokens = sum(row["completion_tokens"] for row in completed)
    ttfts = [row["ttft_ms"] for row in completed if row.get("ttft_ms") is not None]
    spans = [(row["started_monotonic_s"], row["ended_monotonic_s"]) for row in rows
             if "started_monotonic_s" in row and "ended_monotonic_s" in row]
    overlap = any(max(a[0], b[0]) < min(a[1], b[1]) for index, a in enumerate(spans) for b in spans[index + 1:])
    return {"wall_ms": wall_ms, "completed_requests": len(completed), "completion_tokens": tokens,
            "aggregate_completion_tokens_per_s": tokens * 1000 / wall_ms if wall_ms > 0 else None,
            "ttft_median_ms": statistics.median(ttfts) if ttfts else None,
            "ttft_max_ms": max(ttfts) if ttfts else None, "client_request_spans_overlap": overlap}


def run_group(args, client, name, mode, repetition):
    fixtures = GROUPS[name]
    started = time.perf_counter()
    if mode == "serial":
        rows = [run_request(args, client, fixture) for fixture in fixtures]
    else:
        barrier = threading.Barrier(len(fixtures))
        first_delta = threading.Event()
        def work(index, fixture):
            barrier.wait(timeout=args.timeout)
            if mode == "staggered" and index:
                require(first_delta.wait(args.timeout), "First request never emitted a delta")
                if args.admission_delay_ms:
                    time.sleep(args.admission_delay_ms / 1000)
            return run_request(args, client, fixture, first_delta if index == 0 else None)
        with ThreadPoolExecutor(max_workers=len(fixtures)) as pool:
            futures = [pool.submit(work, index, fixture) for index, fixture in enumerate(fixtures)]
            rows = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000
    for row in rows:
        if "started_monotonic_s" in row:
            row["admission_ms"] = (row["started_monotonic_s"] - started) * 1000
    group = {"name": name, "mode": mode, "repetition": repetition, "requests": rows,
             "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
             "metrics": group_metrics(rows, wall_ms)}
    if mode == "parallel" and not group["metrics"]["client_request_spans_overlap"]:
        group["status"] = "failed"
        group["error"] = "Simultaneous request spans did not overlap; parallel coverage was not exercised"
    if mode == "staggered":
        first, second = rows
        group["admitted_during_generation"] = (
            first.get("ttft_ms") is not None and "started_monotonic_s" in second
            and first["started_monotonic_s"] + first["ttft_ms"] / 1000 <= second["started_monotonic_s"]
            < first["ended_monotonic_s"])
        if not group["admitted_during_generation"]:
            group["status"] = "failed"
            group["error"] = "Second request was not admitted while the first was generating"
    return group


def cancellation_and_reuse(args, client):
    """Disconnect a partial stream and admit a fresh task while its survivor is active."""
    started = time.perf_counter()
    barrier = threading.Barrier(2)
    first_delta = threading.Event()
    cancelled_closed = threading.Event()
    def cancel():
        barrier.wait(timeout=args.timeout)
        row = run_request(args, client, "marker_long", first_delta, 1)
        cancelled_closed.set()
        return row
    def survive():
        barrier.wait(timeout=args.timeout)
        return run_request(args, client, "ff7")
    with ThreadPoolExecutor(max_workers=2) as pool:
        cancelled_future, survivor_future = pool.submit(cancel), pool.submit(survive)
        require(cancelled_closed.wait(args.timeout), "Cancellation never closed its stream")
        # Submit before waiting for survivor: the next request exercises dynamic
        # admission and cleanup while another request can remain active.
        reused = run_request(args, client, "marker_reuse")
        cancelled, survivor = cancelled_future.result(), survivor_future.result()
    rows = [cancelled, survivor, reused]
    reuse_while_survivor = survivor.get("started_monotonic_s", math.inf) \
        <= reused.get("started_monotonic_s", -math.inf) < survivor.get("ended_monotonic_s", -math.inf)
    cancelled_survivor_overlap = max(cancelled.get("started_monotonic_s", math.inf), survivor.get("started_monotonic_s", math.inf)) \
        < min(cancelled.get("ended_monotonic_s", -math.inf), survivor.get("ended_monotonic_s", -math.inf))
    passed = cancelled["status"] == "cancelled_as_requested" and survivor["status"] == reused["status"] == "passed" \
        and reuse_while_survivor and cancelled_survivor_overlap
    return {"name": "cancellation_slot_reuse", "mode": "parallel", "requests": rows,
            "status": "passed" if passed else "failed",
            "error": None if passed else "Cancellation, survivor/reuse correctness, or active admission overlap was not exercised",
            "reuse_admitted_while_survivor_active": reuse_while_survivor,
            "cancelled_and_survivor_spans_overlap": cancelled_survivor_overlap,
            "metrics": group_metrics(rows, (time.perf_counter() - started) * 1000),
            "limitations": "HTTP confirms disconnect and subsequent request correctness. It does not expose which native cache slot was reused or prove the server immediately reclaimed it."}


def verify_report_integrity(report):
    """Validate retained baseline evidence before trusting its comparison fields."""
    require(isinstance(report, dict) and report.get("schema") == 1, "Baseline has an unsupported report schema")
    require(isinstance(report.get("model"), str) and report["model"], "Baseline omitted its served model ID")
    config = report.get("configuration")
    require(isinstance(config, dict), "Baseline omitted its workload configuration")
    names, modes, repeats = config.get("groups"), config.get("modes"), config.get("repeats")
    require(isinstance(names, list) and names and len(names) == len(set(names)) and all(name in GROUPS for name in names),
            "Baseline declared invalid or duplicate fixture groups")
    require(isinstance(modes, list) and modes and len(modes) == len(set(modes)) and
            all(mode in ("serial", "parallel", "staggered") for mode in modes), "Baseline declared invalid or duplicate modes")
    require(isinstance(repeats, int) and not isinstance(repeats, bool) and repeats > 0, "Baseline declared invalid repetitions")
    require(isinstance(config.get("seed"), int) and not isinstance(config["seed"], bool) and
            isinstance(config.get("max_tokens"), int) and not isinstance(config["max_tokens"], bool) and config["max_tokens"] > 0,
            "Baseline omitted valid sampling/token configuration")
    groups = report.get("groups")
    require(isinstance(groups, list) and groups, "Baseline omitted its retained groups")
    keys = [(group.get("name"), group.get("mode"), group.get("repetition")) for group in groups]
    expected = {(name, mode, repetition) for name in names for mode in modes for repetition in range(1, repeats + 1)}
    if config.get("cancellation"):
        expected.add(("cancellation_slot_reuse", "parallel", None))
    require(len(keys) == len(set(keys)), "Baseline retained duplicate group evidence")
    require(set(keys) == expected, "Baseline retained group coverage differs from its declared configuration")
    request_args = SimpleNamespace(model=report["model"], seed=config["seed"], max_tokens=config["max_tokens"])
    for group in groups:
        rows = group.get("requests")
        require(isinstance(rows, list) and rows, "Baseline omitted retained request evidence")
        fixtures = [row.get("fixture") for row in rows]
        require(len(fixtures) == len(set(fixtures)), "Baseline retained duplicate request evidence")
        expected_fixtures = ["marker_long", "ff7", "marker_reuse"] if group["name"] == "cancellation_slot_reuse" else GROUPS[group["name"]]
        require(set(fixtures) == set(expected_fixtures), "Baseline retained request coverage differs from its declared fixture group")
        for row in rows:
            body, answer = row.get("request"), row.get("answer")
            require(isinstance(body, dict) and isinstance(answer, str), "Baseline omitted retained request or answer text")
            require(sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()) == row.get("request_sha256"),
                    "Baseline retained request hash does not match its request body")
            require(sha256(answer.encode()) == row.get("answer_sha256"), "Baseline retained answer hash does not match its answer text")
            require(body == body_for(request_args, row["fixture"]), "Baseline request body differs from its declared model/fixture/sampling configuration")
            events = row.get("events")
            require(isinstance(events, list) and events, "Baseline omitted retained SSE evidence")
            decoded, finish, usage, deltas, done_count, previous = {"answer": "", "reasoning": ""}, None, None, 0, 0, -1
            for index, entry in enumerate(events):
                elapsed, event = entry.get("elapsed_ms"), entry.get("data")
                require(isinstance(elapsed, (float, int)) and math.isfinite(elapsed) and elapsed >= previous,
                        "Baseline retained invalid SSE event timing/order")
                previous = elapsed
                if event == "[DONE]":
                    done_count += 1
                    require(index == len(events) - 1, "Baseline retained SSE events after DONE")
                    continue
                require(isinstance(event, dict) and not event.get("error"), "Baseline retained invalid/error SSE data")
                emitted = False
                for choice in event.get("choices", []):
                    finish = choice.get("finish_reason") or finish
                    for wire, field in (("content", "answer"), ("reasoning_content", "reasoning")):
                        piece = choice.get("delta", {}).get(wire)
                        if piece:
                            require(isinstance(piece, str), "Baseline retained invalid SSE content")
                            decoded[field] += piece
                            emitted = True
                deltas += int(emitted)
                if event.get("usage") is not None:
                    usage = event["usage"]
            require(decoded["answer"] == answer and decoded["reasoning"] == row.get("reasoning", ""),
                    "Baseline retained answer/reasoning disagrees with SSE deltas")
            require(done_count == int(row.get("done") is True) and finish == row.get("finish_reason") and
                    deltas == row.get("nonempty_delta_events"), "Baseline retained completion/cancellation metadata disagrees with SSE events")
            if row.get("status") == "passed":
                require(row.get("done") is True and row.get("cancelled") is False and row.get("finish_reason") in ("stop", "length"),
                        "Baseline marks an unfinished/cancelled request as passed")
                actual_quality = quality_check(FIXTURES[row["fixture"]], answer)
                require(actual_quality["passed"] and row.get("quality") == actual_quality,
                        "Baseline stored quality result differs from its retained answer")
                require(isinstance(usage, dict) and usage == row.get("usage"), "Baseline final token usage disagrees with SSE events")
                for field in ("prompt_tokens", "completion_tokens"):
                    value = row.get(field)
                    require(isinstance(value, int) and not isinstance(value, bool) and value > 0 and usage.get(field) == value,
                            "Baseline retained token counts disagree with final usage")
            elif row.get("status") == "cancelled_as_requested":
                require(group["name"] == "cancellation_slot_reuse" and row["fixture"] == "marker_long" and
                        row.get("cancelled") is True and row.get("done") is False and row.get("finish_reason") is None,
                        "Baseline marks a finished stream as intentionally cancelled")
                normalized = "\n".join(line.strip() for line in answer.strip().splitlines())
                require(normalized != FIXTURES["marker_long"]["expected"], "Baseline cancellation retained a complete task answer")
            else:
                require(row.get("status") == "failed" and report.get("status") != "passed", "Baseline passed report contains invalid request status")
        metrics = group.get("metrics")
        require(isinstance(metrics, dict) and isinstance(metrics.get("wall_ms"), (float, int)) and
                math.isfinite(metrics["wall_ms"]) and metrics["wall_ms"] > 0, "Baseline retained invalid group wall time")
        measured = group_metrics(rows, metrics["wall_ms"])
        for field in ("completed_requests", "completion_tokens", "client_request_spans_overlap"):
            require(metrics.get(field) == measured[field], f"Baseline retained inconsistent {field} metrics")
        throughput = metrics.get("aggregate_completion_tokens_per_s")
        require(isinstance(throughput, (float, int)) and math.isfinite(throughput) and
                math.isclose(throughput, measured["aggregate_completion_tokens_per_s"], rel_tol=1e-10, abs_tol=1e-12),
                "Baseline retained inconsistent throughput metrics")
        if group.get("status") == "passed":
            require(all(row.get("status") == ("cancelled_as_requested" if group["name"] == "cancellation_slot_reuse" and
                       row["fixture"] == "marker_long" else "passed") for row in rows), "Baseline passed group contains failed request evidence")
            if group["name"] == "cancellation_slot_reuse":
                by_fixture = {row["fixture"]: row for row in rows}
                survivor, reuse = by_fixture["ff7"], by_fixture["marker_reuse"]
                require(survivor["started_monotonic_s"] <= reuse["started_monotonic_s"] < survivor["ended_monotonic_s"] and
                        group.get("reuse_admitted_while_survivor_active") is True, "Baseline cancellation did not exercise active survivor admission")
                cancelled = by_fixture["marker_long"]
                require(max(cancelled["started_monotonic_s"], survivor["started_monotonic_s"]) <
                        min(cancelled["ended_monotonic_s"], survivor["ended_monotonic_s"]),
                        "Baseline cancellation and survivor did not overlap")
        else:
            require(group.get("status") == "failed" and report.get("status") != "passed", "Baseline passed report contains invalid group status")
    return {"status": "verified", "retained_groups": len(groups), "retained_requests": sum(len(group["requests"]) for group in groups),
            "checks": "Retained request/answer hashes, declared workload and duplicate coverage, answer/finish/token metadata from SSE, answer quality, finite wall time and recomputed group token/throughput metrics"}


def compare_group(serial, concurrent, require_exact):
    baseline = {row["fixture"]: row for row in serial["requests"]}
    candidate_fixtures = [row["fixture"] for row in concurrent["requests"]]
    coverage_matches = len(baseline) == len(serial["requests"]) == len(candidate_fixtures) == len(set(candidate_fixtures)) \
        and set(baseline) == set(candidate_fixtures)
    pairs = []
    for row in concurrent["requests"]:
        other = baseline.get(row["fixture"])
        if other is None:
            pairs.append({"fixture": row["fixture"], "same_request": False, "both_passed": False,
                          "same_answer": False, "same_completion_tokens": False, "same_finish_reason": False})
            continue
        pairs.append({"fixture": row["fixture"], "same_request": row["request_sha256"] == other["request_sha256"],
                      "both_passed": row["status"] == other["status"] == "passed",
                      "same_answer": row.get("answer_sha256") == other.get("answer_sha256"),
                      "same_completion_tokens": row.get("completion_tokens") == other.get("completion_tokens"),
                      "same_finish_reason": row.get("finish_reason") == other.get("finish_reason")})
    groups_passed = serial["status"] == concurrent["status"] == "passed"
    qualified = groups_passed and coverage_matches and all(pair["same_request"] and pair["both_passed"] and pair["same_completion_tokens"]
                                     and pair["same_finish_reason"] for pair in pairs)
    exact = all(pair["same_answer"] for pair in pairs)
    old, new = serial["metrics"], concurrent["metrics"]
    ratio = old["wall_ms"] / new["wall_ms"] if qualified and new["wall_ms"] > 0 else None
    return {"pairs": pairs, "request_coverage_matches": coverage_matches,
            "all_answers_identical": exact and coverage_matches, "performance_qualified": qualified,
            "serial_to_concurrent_wall_speedup": ratio,
            "aggregate_throughput_ratio": new["aggregate_completion_tokens_per_s"] / old["aggregate_completion_tokens_per_s"]
            if qualified and old["aggregate_completion_tokens_per_s"] else None,
            "status": "passed" if groups_passed and coverage_matches and all(pair["same_request"] and pair["both_passed"] for pair in pairs)
            and (exact or not require_exact) else "failed"}


def log_evidence(path, offset):
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read()
    text = raw.decode("utf-8", errors="replace")
    accepted = [int(match.group(1)) for match in FUSED_ACCEPTED.finditer(text)]
    declines = [line for line in text.splitlines() if FUSED_DECLINED.search(line)]
    return {"path": str(path.resolve()), "start_byte": offset, "end_byte": offset + len(raw),
            "sha256": sha256(raw), "accepted_batch_widths": accepted, "decline_lines": declines,
            "new_acceptance_observed": any(width >= 2 for width in accepted),
            "limitations": "Runtime acceptance logging occurs once per executor. Absence in a log slice can mean an earlier batch already triggered its message. The log is process-level evidence; it does not count accepted steps or prove every concurrent step used the path."}


def compare_reports(baseline, candidate):
    key = lambda group: (group["name"], group["mode"], group.get("repetition"))
    old = {key(group): group for group in baseline.get("groups", [])}
    pairs = []
    same_model = baseline.get("model") == candidate.get("model") and bool(candidate.get("model"))
    reports_passed = baseline.get("status") == candidate.get("status") == "passed"
    for group in candidate["groups"]:
        other = old.get(key(group))
        if other is None:
            pairs.append({"key": key(group), "status": "missing_baseline", "performance_qualified": False})
            continue
        if group["name"] == "cancellation_slot_reuse":
            pairs.append({"key": key(group), "status": "control_scenario_not_benchmarked",
                          "performance_qualified": False,
                          "limitations": "Intentional cancellation omits final usage and can interrupt different amounts of native work. Correctness is tested separately; this scenario has no speedup gate."})
            continue
        comparison = compare_group(other, group, False)
        comparison.update(key=key(group), baseline_wall_ms=other["metrics"]["wall_ms"],
                          candidate_wall_ms=group["metrics"]["wall_ms"])
        comparison["performance_qualified"] = comparison["performance_qualified"] and same_model and reports_passed
        comparison["baseline_to_candidate_wall_speedup"] = comparison.pop("serial_to_concurrent_wall_speedup")
        if not comparison["performance_qualified"]:
            comparison["baseline_to_candidate_wall_speedup"] = None
            comparison["aggregate_throughput_ratio"] = None
        pairs.append(comparison)
    selected_keys = {key(group) for group in candidate["groups"]}
    coverage_matches = len(old) == len(baseline.get("groups", [])) and len(selected_keys) == len(candidate["groups"]) \
        and selected_keys <= old.keys() and all(pair.get("status") != "missing_baseline" for pair in pairs)
    baseline_only = sorted(old.keys() - selected_keys, key=repr)
    benchmark_pairs = [pair for pair in pairs if pair["key"][0] != "cancellation_slot_reuse"]
    return {"baseline_label": baseline.get("label"), "same_model_id": same_model,
            "coverage_matches": coverage_matches,
            "coverage_relation": "candidate_subset" if coverage_matches and baseline_only else "same" if coverage_matches else "unmatched_or_duplicate_groups",
            "baseline_only_groups_not_compared": baseline_only,
            "both_reports_passed": reports_passed, "pairs": pairs,
            "all_selected_answers_identical": same_model and coverage_matches and reports_passed and bool(benchmark_pairs)
            and all(pair.get("status") == "passed" and pair.get("all_answers_identical") for pair in benchmark_pairs),
            "all_performance_comparisons_qualified": same_model and coverage_matches and bool(benchmark_pairs)
            and all(pair["performance_qualified"] for pair in benchmark_pairs),
            "limitations": "Only the candidate's selected groups are compared. Extra baseline groups are explicitly omitted, not counted as candidate coverage. Model IDs alone do not bind weight bytes, build, upstream revision, device, process options, or prefix-cache state. Review the supplied provenance before attributing differences to a code change. Different output token counts or completion reasons suppress speedup ratios; answer equality is reported separately."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model")
    parser.add_argument("--groups", default="topics,markers")
    parser.add_argument("--modes", help="Comma-separated serial,parallel,staggered (default all, or parallel,staggered with --concurrent-only)")
    parser.add_argument("--concurrent-only", action="store_true",
                        help="Skip local serial controls; run only concurrent modes, optionally comparing against --compare-with")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=384, help="Topic-answer cap; marker tasks have fixed caps")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--admission-delay-ms", type=float, default=25,
                        help="Staggered requests wait for the first model delta, then delay this long")
    parser.add_argument("--cancellation", action="store_true")
    parser.add_argument("--require-exact-parity", action="store_true",
                        help="Fail any local serial/concurrent or selected external-baseline answer difference")
    parser.add_argument("--min-parallel-speedup", type=float,
                        help="Optional wall-speedup gate for every selected concurrent group; requires equal output token counts")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--require-fused", action="store_true")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--compare-with", type=Path,
                        help="Compare with a previous identical-workload report (e.g. TS_BATCHED_FUSED_DECODE=0)")
    parser.add_argument("--min-baseline-speedup", type=float,
                        help="Optional regression gate for every matched --compare-with group")
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--output", type=evidence_path, required=True)
    args = parser.parse_args(argv)
    args.groups = args.groups.split(",")
    args.modes = (args.modes or ("parallel,staggered" if args.concurrent_only else "serial,parallel,staggered")).split(",")
    require(args.groups and len(set(args.groups)) == len(args.groups) and set(args.groups) <= GROUPS.keys(), "Invalid --groups")
    require(args.modes and len(set(args.modes)) == len(args.modes) and set(args.modes) <= {"serial", "parallel", "staggered"}, "Invalid --modes")
    require(not args.concurrent_only or "serial" not in args.modes, "--concurrent-only cannot select serial mode")
    if not args.concurrent_only and "serial" not in args.modes:
        args.modes.insert(0, "serial")
    require(1 <= args.repeats <= 20 and 1 <= args.max_tokens <= 4096, "Use 1..20 repeats and 1..4096 topic max tokens")
    require(math.isfinite(args.timeout) and args.timeout > 0, "--timeout must be finite and positive")
    require(math.isfinite(args.admission_delay_ms) and 0 <= args.admission_delay_ms <= 60000, "Invalid admission delay")
    require(not args.require_fused or args.server_log, "--require-fused requires --server-log")
    require(not args.server_log or args.server_log.is_file(), "--server-log must exist")
    for value in (args.min_parallel_speedup, args.min_baseline_speedup):
        require(value is None or math.isfinite(value) and value > 0, "Speedup gates must be finite and positive")
    require(args.min_parallel_speedup is None or set(args.modes) != {"serial"}, "Parallel speedup gate needs a concurrent mode")
    require(not args.concurrent_only or args.min_parallel_speedup is None,
            "--min-parallel-speedup requires local serial controls; use --min-baseline-speedup with --compare-with")
    require(not args.concurrent_only or not args.require_exact_parity or args.compare_with,
            "--require-exact-parity in concurrent-only mode requires --compare-with")
    require(args.min_baseline_speedup is None or args.compare_with, "Baseline speedup gate needs --compare-with")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = {"schema": 1, "label": args.label, "url": args.url,
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "failures": [], "groups": [],
              "configuration": {name: getattr(args, name) for name in ("groups", "modes", "concurrent_only", "repeats", "max_tokens", "seed", "admission_delay_ms", "cancellation", "require_exact_parity", "require_fused", "min_parallel_speedup", "min_baseline_speedup")},
              "coverage": {"selected_groups": args.groups, "omitted_groups": sorted(set(GROUPS) - set(args.groups)),
                           "cancellation_selected": args.cancellation, "local_serial_controls_run": not args.concurrent_only,
                           "vision": "not covered by this text-only runner"},
              "limitations": "Loading/warmup is excluded. Default local serial and concurrent controls use the same live host, so prefix/cache warmth may differ; cached_tokens is preserved per request. Concurrent-only omits local serial controls and has no local serial text-parity or speedup claim. Greedy sampling fixes temperature/top-k/penalties and seed, but HTTP exposes text rather than token IDs/logits. Topic checks measure relevance and obvious leakage, not complete factual accuracy. Capped outputs may be truncated. Throughput is completion usage tokens divided by complete group wall time, including admission, prefill, and HTTP; it is not a pure native decode rate. Parallel speedup is qualified only for successful identical requests with equal completion token counts and finish reasons. No omitted or unavailable scenario counts as passed. Supply model/device/build/upstream provenance separately."}
    log_offset = args.server_log.stat().st_size if args.server_log else None
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        if args.provenance:
            raw = args.provenance.read_bytes()
            report["provenance"] = json.loads(raw)
            report["provenance_sha256"] = sha256(raw)
        client = Client(args.url, args.timeout)
        models = client.models()
        require(isinstance(models, list) and models, "Server has no available models")
        if args.model is None:
            require(len(models) == 1, "Specify --model when server has multiple model IDs")
            args.model = models[0]["id"]
        require(any(model.get("id") == args.model for model in models), "Requested model is absent from /v1/models")
        report.update(model=args.model, server_models=models)
        for repetition in range(1, args.repeats + 1):
            for name in args.groups:
                serial = None
                # Default runs establish local controls first. Concurrent-only
                # runs retain request/admission gates and use external controls
                # only when a comparison report was explicitly supplied.
                for mode in ([] if args.concurrent_only else ["serial"]) + [value for value in args.modes if value != "serial"]:
                    group = run_group(args, client, name, mode, repetition)
                    if mode == "serial":
                        serial = group
                    elif serial is not None:
                        group["serial_comparison"] = compare_group(serial, group, args.require_exact_parity)
                        if group["serial_comparison"]["status"] != "passed":
                            group["status"] = "failed"
                        speedup = group["serial_comparison"]["serial_to_concurrent_wall_speedup"]
                        if args.min_parallel_speedup is not None and (speedup is None or speedup < args.min_parallel_speedup):
                            group["status"] = "failed"
                            group["performance_gate_error"] = f"Qualified speedup {speedup} is below required {args.min_parallel_speedup} or unavailable"
                    else:
                        group["local_serial_control"] = "not_run_concurrent_only"
                    report["groups"].append(group)
                    if group["status"] != "passed":
                        report["failures"].append(f"{name}/{mode}/{repetition}: request, admission, or parity checks failed")
                    print(f"{name}/{mode}/{repetition}: {group['status']} "
                          f"wall={group['metrics']['wall_ms']:.1f} ms "
                          f"tokens={group['metrics']['completion_tokens']} "
                          f"aggregate={group['metrics']['aggregate_completion_tokens_per_s']:.2f} tok/s", flush=True)
                    save()
        if args.cancellation:
            group = cancellation_and_reuse(args, client)
            report["groups"].append(group)
            if group["status"] != "passed":
                report["failures"].append("Cancellation, survivor, or reuse request failed")
            save()
        if args.server_log:
            report["fused_decode_evidence"] = log_evidence(args.server_log, log_offset)
            if args.require_fused and not report["fused_decode_evidence"]["new_acceptance_observed"]:
                report["failures"].append("No new multi-sequence fused decode acceptance was observed")
            if args.require_fused and report["fused_decode_evidence"]["decline_lines"]:
                report["failures"].append("Fused-decode decline/round-robin fallback was observed")
        report["status"] = "passed" if not report["failures"] else "failed"
    except Exception as error:
        report["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        if args.compare_with:
            try:
                baseline_raw = args.compare_with.read_bytes()
                external_baseline = json.loads(baseline_raw)
                report["comparison_baseline_file"] = {"path": str(args.compare_with.resolve()), "sha256": sha256(baseline_raw)}
                report["comparison_baseline_integrity"] = verify_report_integrity(external_baseline)
                report["baseline_comparison"] = compare_reports(external_baseline, report)
                if args.require_exact_parity and not report["baseline_comparison"]["all_selected_answers_identical"]:
                    report["failures"].append("Selected external-baseline answers differ or parity coverage is unavailable")
                    report["status"] = "failed"
                    report["baseline_comparison"] = compare_reports(external_baseline, report)
                if args.min_baseline_speedup is not None:
                    comparison = report["baseline_comparison"]
                    gated = comparison["all_performance_comparisons_qualified"] and all(
                        pair["baseline_to_candidate_wall_speedup"] >= args.min_baseline_speedup for pair in comparison["pairs"]
                        if pair["key"][0] != "cancellation_slot_reuse")
                    if not gated:
                        report["failures"].append(f"Qualified baseline speedup is below {args.min_baseline_speedup} or unavailable")
                        report["status"] = "failed"
            except Exception as error:
                report["failures"].append(f"Report comparison: {type(error).__name__}: {error}")
                report["status"] = "failed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"{report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
