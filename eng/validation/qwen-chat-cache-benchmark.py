#!/usr/bin/env python3
"""Measure Qwen HTTP TTFT and continuation reuse on a running TensorSharp host.

Run identical arguments against baseline and fixed hosts; use --compare-with to
compare the reports. Web UI mode uploads the original image once and preserves
its reference across turns, reproducing the browser chat path. OpenAI mode sends
the same image bytes as a data URI each turn. Loading/startup warmup is excluded.
The first request is preserved separately from later image-encoding warm runs.
Reports and full answer text stay in ignored artifacts/ or docs/validation/.
This capped-output benchmark screens protocol/cache behavior; it does not grade
the factual accuracy of an unrestricted image description. Inspect saved answers.
"""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import mimetypes
from pathlib import Path
import statistics
import time
from urllib.parse import quote, urlsplit
import uuid


ROOT = Path(__file__).resolve().parents[2]
IMAGE_PROMPT = "请详细描述这幅图"
CONTINUATION = "请继续"
TEXT_PROMPT = ("请用中文详细介绍中国茶文化，依次说明茶叶的主要种类、冲泡方式、"
               "茶具的选择以及待客礼仪。请用自然、清晰的语言写成多个段落。")
INVALIDATION_PROMPT = "这是一个新的独立问题。不要沿用之前的话题。只输出 123 加 456 的结果，不要解释。"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def output_path(value):
    path = Path(value).resolve()
    require(any(path.is_relative_to((ROOT / folder).resolve()) for folder in ("artifacts", "docs/validation")),
            "Output must remain inside this checkout's ignored artifacts/ or docs/validation/")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class Client:
    def __init__(self, url, timeout, cookie=None):
        self.address = urlsplit(url)
        require(self.address.scheme in ("http", "https") and self.address.hostname,
                "--url must be an HTTP(S) server URL")
        require(not self.address.username and not self.address.query and not self.address.fragment,
                "--url cannot contain credentials, query or fragment")
        self.timeout = timeout
        self.cookie = cookie

    def headers(self, content_type):
        headers = {"Content-Type": content_type}
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers

    def connection(self):
        cls = http.client.HTTPSConnection if self.address.scheme == "https" else http.client.HTTPConnection
        return cls(self.address.hostname, self.address.port, timeout=self.timeout)

    def path(self, route):
        return self.address.path.rstrip("/") + route

    def json(self, method, route, body=None):
        connection = self.connection()
        try:
            raw = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            connection.request(method, self.path(route), raw, self.headers("application/json"))
            response = connection.getresponse()
            data = response.read()
            require(response.status == 200, f"{route}: HTTP {response.status}: {data[:2048]!r}")
            return json.loads(data)
        finally:
            connection.close()

    def upload(self, image):
        boundary = "TensorSharpQwenCache" + uuid.uuid4().hex
        require(not any(character in image.name for character in '\r\n"\\'), "Unsafe image filename")
        mime = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        raw = image.read_bytes()
        data = ((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="{image.name}"\r\nContent-Type: {mime}\r\n\r\n').encode()
                + raw + f"\r\n--{boundary}--\r\n".encode())
        connection = self.connection()
        started = time.perf_counter()
        try:
            connection.request("POST", self.path("/api/upload"), data,
                               self.headers("multipart/form-data; boundary=" + boundary))
            response = connection.getresponse()
            payload = response.read()
            require(response.status == 200, f"Upload: HTTP {response.status}: {payload[:2048]!r}")
            result = json.loads(payload)
            require(result.get("ok") is True and result.get("file"), "Upload returned no image reference")
            return {"elapsed_ms": (time.perf_counter() - started) * 1000,
                    "response": result, "input_bytes": len(raw), "input_sha256": digest(raw)}
        finally:
            connection.close()

    def chat(self, route, body, protocol, evidence):
        connection = self.connection()
        wire = json.dumps(body, ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        state = StreamState(protocol)
        try:
            connection.request("POST", self.path(route), wire, self.headers("application/json"))
            response = connection.getresponse()
            evidence["http_status"] = response.status
            evidence["headers_ms"] = (time.perf_counter() - started) * 1000
            if response.status != 200:
                raise ValueError(f"Chat: HTTP {response.status}: {response.read()[:2048]!r}")
            require("text/event-stream" in response.getheader("Content-Type", ""), "Chat did not return SSE")
            evidence["events"] = []
            deadline = time.monotonic() + self.timeout
            for line in response:
                require(time.monotonic() <= deadline, "Chat exceeded request deadline")
                if not line.startswith(b"data:"):
                    continue
                elapsed = (time.perf_counter() - started) * 1000
                raw = line[5:].strip().decode("utf-8")
                if not raw:
                    continue
                event = "[DONE]" if raw == "[DONE]" else json.loads(raw)
                evidence["events"].append({"elapsed_ms": elapsed, "data": event})
                state.add(event, elapsed)
                if state.done:
                    break
            evidence.update(state.result())
            evidence["elapsed_ms"] = (time.perf_counter() - started) * 1000
            state.validate()
            return state.message()
        finally:
            evidence.update(state.result())
            evidence["elapsed_ms"] = (time.perf_counter() - started) * 1000
            connection.close()


class StreamState:
    """Collect answer/reasoning deltas without mistaking headers for TTFT."""
    def __init__(self, protocol):
        self.protocol = protocol
        self.content = ""
        self.reasoning = ""
        self.ttft_ms = None
        self.first_answer_ms = None
        self.done = False
        self.final = None
        self.finish_reason = None
        self.errors = []

    def append(self, field, piece, elapsed):
        if not piece:
            return
        require(isinstance(piece, str), f"Invalid {field} delta")
        if self.ttft_ms is None:
            self.ttft_ms = elapsed
        if field == "content" and self.first_answer_ms is None:
            self.first_answer_ms = elapsed
        setattr(self, field, getattr(self, field) + piece)

    def add(self, event, elapsed):
        if event == "[DONE]":
            require(self.protocol == "openai", "Unexpected [DONE] in Web UI stream")
            self.done = True
            return
        require(isinstance(event, dict), "Invalid SSE event")
        if event.get("error"):
            self.errors.append(str(event["error"]))
        if self.protocol == "webui":
            self.append("content", event.get("token"), elapsed)
            self.append("reasoning", event.get("thinking"), elapsed)
            # The whole answer, when text the page showed proved to be reasoning (or a
            # DiffusionGemma canvas): what is sent back next turn is what remains.
            if isinstance(event.get("replace"), str):
                self.content = event["replace"]
            if event.get("tool_calls"):
                self.errors.append("Unexpected model tool call in prose benchmark")
            if event.get("done"):
                self.final = event
                self.done = True
                self.finish_reason = "length" if event.get("truncated") else "stop"
                if event.get("aborted"):
                    self.errors.append("Chat was aborted")
        else:
            if event.get("usage") is not None:
                self.final = event["usage"]
            for choice in event.get("choices", []):
                self.finish_reason = choice.get("finish_reason") or self.finish_reason
                delta = choice.get("delta", {})
                self.append("content", delta.get("content"), elapsed)
                self.append("reasoning", delta.get("reasoning_content"), elapsed)
                if delta.get("tool_calls"):
                    self.errors.append("Unexpected model tool call in prose benchmark")

    def result(self):
        final = self.final or {}
        if self.protocol == "webui":
            prompt, cached, generated = final.get("promptTokens"), final.get("kvReusedTokens"), final.get("tokenCount")
        else:
            prompt, generated = final.get("prompt_tokens"), final.get("completion_tokens")
            cached = final.get("prompt_tokens_details", {}).get("cached_tokens")
        return {"ttft_ms": self.ttft_ms, "first_answer_ms": self.first_answer_ms,
                "prompt_tokens": prompt, "cached_tokens": cached, "completion_tokens": generated,
                "prefilled_tokens": prompt - cached if isinstance(prompt, int) and isinstance(cached, int) else None,
                "finish_reason": self.finish_reason, "done": self.done, "errors": list(self.errors),
                "answer": self.content, "reasoning": self.reasoning,
                "answer_sha256": digest(self.content.encode("utf-8")), "server_final": final}

    def validate(self):
        require(self.done, "SSE stream ended without its completion marker")
        require(not self.errors, "; ".join(self.errors))
        require(self.content.strip(), "Model returned no answer text")
        require(self.ttft_ms is not None, "No model delta was observed")
        metrics = self.result()
        for name in ("prompt_tokens", "cached_tokens", "completion_tokens"):
            value = metrics[name]
            require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                    f"Missing or invalid {name}")
        require(metrics["prompt_tokens"] > 0 and metrics["completion_tokens"] > 0, "Empty token usage")
        require(metrics["cached_tokens"] <= metrics["prompt_tokens"], "Cached tokens exceed prompt tokens")
        require(self.finish_reason in ("stop", "length"), f"Unexpected finish reason {self.finish_reason}")
        require(not self.content.lstrip().startswith("Error:"), "Server returned an error as answer text")

    def message(self):
        return {"role": "assistant", "content": self.content}


def canonical_request(body, image_sha256):
    """Compare exact textual history and image bytes across generated upload names."""
    request = copy.deepcopy(body)
    request.pop("sessionId", None)
    for message in request.get("messages", []):
        if message.get("imagePaths"):
            message["imagePaths"] = ["sha256:" + image_sha256 for _ in message["imagePaths"]]
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    part["image_url"]["url"] = "sha256:" + image_sha256
    return request


def make_body(args, history, session=None, new_chat=False):
    body = {"messages": copy.deepcopy(history), "think": args.thinking,
            "temperature": 0, "repeat_penalty": 1, "skills": [], "skills_discovery": False,
            "multi_agent": False}
    if args.protocol == "webui":
        body.update({"sessionId": session, "newChat": new_chat, "maxTokens": args.max_tokens})
    else:
        body.update({"model": args.model, "stream": True, "max_tokens": args.max_tokens,
                     "stream_options": {"include_usage": True}})
    return body


def load_replay_history(args, report):
    """Read answer strings, never prior host sessions, uploads, or request bodies."""
    raw = args.replay_history_from.read_bytes()
    baseline = json.loads(raw.decode("utf-8"))
    require(baseline.get("model") == report["model"] and baseline.get("protocol") == args.protocol,
            "Replay history must use the same model and HTTP protocol")
    rows = {}
    for row in baseline.get("runs", []):
        key = (row["scenario"], row["repetition"], row["turn"])
        require(key not in rows, f"Replay history contains duplicate request {key}")
        rows[key] = row
    turns = [f"turn-{index + 1}" for index in range(args.turns)]
    turns += (["branch"] if args.branches else []) + (["invalidation"] if args.invalidation else [])
    for scenario in args.cases:
        for repetition in range(1, args.repeats + 1):
            for turn in turns:
                key = (scenario, repetition, turn)
                row = rows.get(key, {})
                require(row.get("status") == "passed" and row.get("done") and not row.get("error")
                        and isinstance(row.get("answer"), str) and row.get("request_sha256"),
                        f"Replay history lacks a completed baseline request {key}")
    args._replay_rows = rows
    report["history_replay"] = {"source": str(args.replay_history_from.resolve()),
        "sha256": digest(raw), "label": baseline.get("label"),
        "limitations": "Subsequent requests use saved baseline assistant text, with current prompts and current image uploads. A divergent candidate answer changes its live cache; replay may therefore recompute that prefix. This measures identical requested histories, not necessarily a natural continuation of the candidate answer."}


def run_workflow(args, client, scenario, repetition, image_message, report):
    session = None
    try:
        if args.protocol == "webui":
            session = client.json("POST", "/api/sessions")["sessionId"]
        first = copy.deepcopy(image_message) if scenario == "image" else {"role": "user", "content": args.text_prompt}
        history = [first]
        initial_history = None
        previous_prompt_tokens = None
        for index in range(args.turns):
            if index:
                history.append({"role": "user", "content": args.continuation_prompt})
            row = execute_turn(args, client, scenario, repetition, f"turn-{index + 1}", history, session, report,
                               previous_prompt_tokens)
            if row["status"] != "passed":
                return
            replay = getattr(args, "_replay_rows", {}).get((scenario, repetition, row["turn"]))
            history.append({"role": "assistant", "content": replay["answer"] if replay else row["answer"]})
            previous_prompt_tokens = row["prompt_tokens"]
            if index == 0:
                initial_history = copy.deepcopy(history)
        if args.branches:
            branch = initial_history + [{"role": "user", "content": "请补充说明刚才提到的细节。"}]
            execute_turn(args, client, scenario, repetition, "branch", branch, session, report)
        if args.invalidation:
            # Retain the same named session but alter its entire history. Reusing
            # conversation-private state here would return stale text or semantics.
            execute_turn(args, client, scenario, repetition, "invalidation",
                         [{"role": "user", "content": INVALIDATION_PROMPT}], session, report)
    finally:
        if session:
            try:
                client.json("DELETE", "/api/sessions/" + quote(session))
            except Exception as error:
                report["failures"].append(f"Session cleanup: {type(error).__name__}: {error}")


def execute_turn(args, client, scenario, repetition, turn, history, session, report, previous_prompt_tokens=None):
    row = {"scenario": scenario, "repetition": repetition, "turn": turn, "status": "failed",
           "benchmark_request_index": len(report["runs"]) + 1,
           "previous_prompt_tokens": previous_prompt_tokens,
           "expected_continuation": turn.startswith("turn-") and turn != "turn-1"}
    report["runs"].append(row)
    body = make_body(args, history, session)
    canonical = canonical_request(body, report.get("image", {}).get("sha256", ""))
    row["request"] = canonical
    row["request_sha256"] = digest(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    try:
        replay = getattr(args, "_replay_rows", {}).get((scenario, repetition, turn))
        if replay:
            require(row["request_sha256"] == replay["request_sha256"],
                    "Reconstructed history does not match the baseline request; check prompts, image bytes, and sampling arguments")
            row["request_history_source"] = "baseline_replay"
        route = "/api/chat" if args.protocol == "webui" else "/v1/chat/completions"
        client.chat(route, body, args.protocol, row)
        if replay:
            row["answer_matches_replay"] = row["answer"] == replay["answer"]
        if args.require_reuse and row["expected_continuation"]:
            require(row["cached_tokens"] > 0, "Continuation reused zero cached tokens")
        if args.require_full_reuse and row["expected_continuation"]:
            require(row["cached_tokens"] >= previous_prompt_tokens,
                    f"Continuation reused {row['cached_tokens']} tokens, less than its prior {previous_prompt_tokens}-token prompt")
        if turn == "invalidation":
            require(row["answer"].strip().strip("`。.! ") == "579", "Changed history produced the wrong arithmetic answer")
        row["status"] = "passed"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        report["failures"].append(f"{scenario}/{repetition}/{turn}: {row['error']}")
    print(f"{scenario}/{repetition}/{turn}: {row['status']} TTFT={row.get('ttft_ms')} ms "
          f"cached={row.get('cached_tokens')}/{row.get('prompt_tokens')} tokens "
          f"generated={row.get('completion_tokens')} finish={row.get('finish_reason')}", flush=True)
    # Preserve every completed request immediately, even if a later request fails.
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def summarize(report):
    groups = {}
    for row in report["runs"]:
        group = groups.setdefault(row["scenario"] + "/" + row["turn"], [])
        group.append(row)
    result = {}
    for name, rows in groups.items():
        passed = [row for row in rows if row["status"] == "passed"]
        result[name] = {"samples": len(rows), "passed": len(passed),
                        "ttft_median_ms": statistics.median(row["ttft_ms"] for row in passed) if passed else None,
                        "cached_tokens": [row.get("cached_tokens") for row in rows],
                        "prompt_tokens": [row.get("prompt_tokens") for row in rows],
                        "truncated": sum(row.get("finish_reason") == "length" for row in rows)}
    return result


def compare_reports(baseline, candidate):
    keys = lambda row: (row["scenario"], row["repetition"], row["turn"])
    old = {keys(row): row for row in baseline.get("runs", [])}
    pairs = []
    for row in candidate["runs"]:
        other = old.get(keys(row))
        if other is None:
            pairs.append({"key": keys(row), "status": "missing_baseline"})
            continue
        same_request = row["request_sha256"] == other.get("request_sha256")
        both_complete = all(item.get("done") and item.get("ttft_ms") is not None
                            and not item.get("error") for item in (other, row))
        pairs.append({"key": keys(row), "same_request": same_request,
                      "both_complete": both_complete,
                      "same_answer": row.get("answer_sha256") == other.get("answer_sha256"),
                      "same_generated_tokens": row.get("completion_tokens") == other.get("completion_tokens"),
                      "same_finish_reason": row.get("finish_reason") == other.get("finish_reason"),
                      "baseline_ttft_ms": other.get("ttft_ms"), "candidate_ttft_ms": row.get("ttft_ms"),
                      "baseline_cached_tokens": other.get("cached_tokens"), "candidate_cached_tokens": row.get("cached_tokens"),
                      "ttft_speedup": other["ttft_ms"] / row["ttft_ms"]
                      if same_request and both_complete and row["ttft_ms"] > 0 else None})
    same_model_protocol = baseline.get("model") == candidate.get("model") and baseline.get("protocol") == candidate.get("protocol")
    complete = (bool(pairs) and len(pairs) == len(baseline.get("runs", []))
                and len(old) == len(baseline.get("runs", []))
                and len({keys(row) for row in candidate["runs"]}) == len(candidate["runs"])
                and all(pair.get("both_complete") for pair in pairs))
    matching = same_model_protocol and complete and all(pair["same_request"] and pair["same_answer"] and pair["same_generated_tokens"]
                                and pair["same_finish_reason"] for pair in pairs)
    return {"baseline_label": baseline.get("label"), "pairs": pairs,
            "status": "identical_answers" if matching else "different_answers_or_histories" if complete else "incomplete",
            "performance_qualified": matching and all(pair["ttft_speedup"] is not None for pair in pairs),
            "same_model_and_protocol": same_model_protocol,
            "limitations": "Ratios require identical text/image-byte histories and complete streams. HTTP exposes answer text and token counts, not raw token IDs or logits. Equal text is a protocol-level parity check. Capped or changed answer histories do not establish unrestricted quality."}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--connection-file", type=Path,
                        help="TensorAgentHost connection.json for loopback authentication; --url must match its baseUrl")
    parser.add_argument("--model", help="Served model ID; discovered from the selected model endpoint when omitted")
    parser.add_argument("--model-discovery", choices=("openai", "webui"), default="openai",
                        help="Model metadata endpoint: /v1/models (default) or the Web UI host's /api/models")
    parser.add_argument("--protocol", choices=("webui", "openai"), default="webui")
    parser.add_argument("--cases", default="text,image")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--text-prompt", default=TEXT_PROMPT, help="Use a short prompt to exercise natural EOS within the token cap")
    parser.add_argument("--continuation-prompt", default=CONTINUATION)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--branches", action="store_true")
    parser.add_argument("--invalidation", action="store_true")
    parser.add_argument("--require-reuse", action="store_true", help="Fail a successful continuation with cached_tokens=0")
    parser.add_argument("--require-full-reuse", action="store_true",
                        help="Require continuation reuse to cover at least the prior turn's entire prompt, including media")
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--provenance", type=Path, help="JSON containing measured build/model/device/upstream identities")
    parser.add_argument("--compare-with", type=Path)
    parser.add_argument("--replay-history-from", type=Path,
                        help="Use completed baseline answers in later requests, preserving identical requested histories even when candidate answers differ")
    parser.add_argument("--output", type=output_path, required=True)
    args = parser.parse_args(argv)
    args.cases = args.cases.split(",")
    require(args.cases and len(args.cases) == len(set(args.cases)) and set(args.cases) <= {"text", "image"},
            "--cases must select unique text,image entries")
    require(1 <= args.turns <= 8 and 1 <= args.repeats <= 20 and 1 <= args.max_tokens <= 4096,
            "Use 1..8 turns, 1..20 repeats, 1..4096 max tokens")
    require(math.isfinite(args.timeout) and args.timeout > 0, "--timeout must be finite and positive")
    require(args.model_discovery != "webui" or args.protocol == "webui",
            "--model-discovery webui requires --protocol webui")
    if "image" in args.cases:
        require(args.image and args.image.is_file(), "The image case requires an existing --image")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = {"label": args.label, "started_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": args.protocol, "url": args.url, "status": "failed", "failures": [], "runs": [],
              "configuration": {name: getattr(args, name) for name in ("model_discovery", "cases", "turns", "repeats", "max_tokens", "text_prompt", "continuation_prompt",
                  "thinking", "branches", "invalidation", "require_reuse", "require_full_reuse")},
              "limitations": "Serial HTTP requests only. Loading and startup prefix preparation excluded. Client TTFT begins before request upload and ends at first nonempty answer/reasoning delta. First answer latency is separate. Original photo is uploaded once; later workflows may reuse vision encodings. Repetition 1 is retained separately from later samples; first benchmark request does not imply an unused or cold host. Tool delegation and skill discovery are disabled in every request. Output is capped; inspect saved text before judging description quality. No skipped scenario counts as passed. Device/build/upstream provenance must be supplied by the operator."}
    try:
        cookie = None
        if args.connection_file:
            connection = json.loads(args.connection_file.read_text(encoding="utf-8-sig"))
            require(connection["baseUrl"].rstrip("/") == args.url.rstrip("/"),
                    "--url must match the connection file's baseUrl")
            cookie = connection["cookie"]
            require(isinstance(cookie, str) and not any(c in cookie for c in "\r\n"),
                    "Invalid connection cookie")
        client = Client(args.url, args.timeout, cookie)
        if args.provenance:
            report["provenance"] = json.loads(args.provenance.read_text(encoding="utf-8"))
        if args.model_discovery == "webui":
            metadata = client.json("GET", "/api/models")
            loaded = metadata.get("loaded")
            require(isinstance(loaded, str) and loaded.strip(), "The Web UI host has no loaded model")
            available = metadata.get("models")
            require(isinstance(available, list) and loaded in available,
                    "The loaded Web UI model is not present in /api/models")
            require(args.model is None or args.model == loaded,
                    "--model must match the Web UI host's loaded model")
            args.model = loaded
            models = [{"id": loaded}]
            report["server_webui_models"] = metadata
        else:
            models = client.json("GET", "/v1/models")["data"]
            if args.model is None:
                require(len(models) == 1, "Specify --model when /v1/models returns more than one model")
                args.model = models[0]["id"]
        report["model"] = args.model
        report["server_models"] = models
        image_message = None
        if "image" in args.cases:
            raw = args.image.read_bytes()
            report["image"] = {"path": str(args.image.resolve()), "bytes": len(raw), "sha256": digest(raw)}
            if args.protocol == "webui":
                report["upload"] = client.upload(args.image)
                image_message = {"role": "user", "content": IMAGE_PROMPT,
                                 "imagePaths": [report["upload"]["response"]["file"]]}
            else:
                mime = mimetypes.guess_type(args.image.name)[0] or "image/jpeg"
                image_message = {"role": "user", "content": [{"type": "text", "text": IMAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(raw).decode()}}]}
        if args.replay_history_from:
            load_replay_history(args, report)
        for scenario in args.cases:
            for repetition in range(1, args.repeats + 1):
                run_workflow(args, client, scenario, repetition, image_message, report)
        expected = len(args.cases) * args.repeats * (args.turns + int(args.branches) + int(args.invalidation))
        require(len(report["runs"]) == expected, f"Coverage incomplete: {len(report['runs'])}/{expected} requests")
        report["status"] = "passed" if not report["failures"] else "failed"
    except Exception as error:
        report["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        report["summary"] = summarize(report)
        if args.compare_with:
            report["comparison"] = compare_reports(json.loads(args.compare_with.read_text(encoding="utf-8")), report)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
