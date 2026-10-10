#!/usr/bin/env python3
"""Check actual model-generated generic calls over the OpenAI HTTP API.

Runs streamed/nonstreamed requests with thinking on/off. Saves complete requests
and responses, checks argument types and exact string content, and feeds the
model's own call ID back for a second turn. Client tools are fixtures and never
execute code; validate-release-agent-workflows.py covers host execution.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit


PAYLOAD = 'def markup():\n    return "<tool_call></tool_call>"\n'
PLAIN_CODE = 'def answer():\n    value = 40\n    return value + 2\n'
SPECS = {
    "weather": {
        "name": "get_weather",
        "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
        "expected": {"city": "Paris", "units": "celsius"},
        "prompt": "Call get_weather once for Paris with units celsius. Wait for its result.",
    },
    "string_payload": {
        "name": "store_payload",
        "properties": {"key": {"type": "string"}, "content": {"type": "string"}},
        "expected": {"key": "123", "content": '{"answer":42}'},
        "prompt": 'Call store_payload once with key the string "123" and content the exact string {"answer":42}. Both arguments are strings. Wait for its result.',
    },
    "plain_code_payload": {
        "name": "store_payload",
        "properties": {"key": {"type": "string"}, "content": {"type": "string"}},
        "expected": {"key": "main.py", "content": PLAIN_CODE},
        "prompt": "Call store_payload once with key main.py and content exactly the decoded value of this JSON string, preserving its indentation and trailing newline: " + json.dumps(PLAIN_CODE) + ". Wait for its result.",
    },
    # Keep this stress case distinct from ordinary generated code. A model that
    # emits EOS inside the argument must fail, even if the parser can handle a
    # complete literal-marker call in deterministic regression tests.
    "code_payload": {
        "name": "store_payload",
        "properties": {"key": {"type": "string"}, "content": {"type": "string"}},
        "expected": {"key": "main.py", "content": PAYLOAD},
        "prompt": "Call store_payload once with key main.py and content exactly the decoded value of this JSON string, preserving its indentation and trailing newline: " + json.dumps(PAYLOAD) + ". Wait for its result.",
    },
}


def hosted_model(url, timeout):
    address = urlsplit(url)
    cls = http.client.HTTPSConnection if address.scheme == "https" else http.client.HTTPConnection
    connection = cls(address.hostname, address.port, timeout=timeout)
    try:
        connection.request("GET", address.path.rstrip("/") + "/v1/models")
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise ValueError(f"Model discovery returned HTTP {response.status}")
        models = json.loads(body).get("data", [])
        if len(models) != 1 or not models[0].get("id"):
            raise ValueError("Specify --model with an ID from /v1/models")
        return models[0]["id"]
    finally:
        connection.close()


def request(url, body, timeout, evidence):
    address = urlsplit(url)
    cls = http.client.HTTPSConnection if address.scheme == "https" else http.client.HTTPConnection
    connection = cls(address.hostname, address.port, timeout=timeout)
    evidence["request"] = body
    try:
        connection.request("POST", address.path.rstrip("/") + "/v1/chat/completions",
                           json.dumps(body).encode(), {"Content-Type": "application/json"})
        response = connection.getresponse()
        evidence["http_status"] = response.status
        if response.status != 200:
            evidence["body"] = response.read().decode("utf-8", "replace")
            raise ValueError(f"HTTP {response.status}: {evidence['body']}")
        if not body["stream"]:
            evidence["body"] = response.read().decode("utf-8")
            data = json.loads(evidence["body"])
            evidence["response"] = data
            return data["choices"][0]["message"], data["choices"][0]["finish_reason"], data.get("usage")

        message = {"role": "assistant", "content": ""}
        calls, finish, usage, done = {}, None, None, False
        evidence["events"] = []
        evidence["raw_events"] = []
        evidence["done"] = False
        deadline = time.monotonic() + timeout
        for line in response:
            if time.monotonic() > deadline:
                raise TimeoutError("SSE request exceeded its deadline")
            if not line.startswith(b"data:"):
                continue
            text = line[5:].strip().decode("utf-8")
            evidence["raw_events"].append(text)
            if text == "[DONE]":
                done = True
                break
            event = json.loads(text)
            evidence["events"].append(event)
            if event.get("error"):
                raise ValueError(str(event["error"]))
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta", {})
                for field in ("content", "reasoning_content"):
                    if delta.get(field):
                        message[field] = message.get(field, "") + delta[field]
                for part in delta.get("tool_calls", []):
                    call = calls.setdefault(part["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if part.get("id"):
                        call["id"] += part["id"]
                    if part.get("type"):
                        call["type"] = part["type"]
                    for field in ("name", "arguments"):
                        call["function"][field] += part.get("function", {}).get(field) or ""
        evidence["done"] = done
        if not done:
            raise ValueError("SSE stream has no [DONE] event")
        if calls:
            message["tool_calls"] = [calls[index] for index in sorted(calls)]
        evidence["assembled_message"] = message
        return message, finish, usage
    finally:
        connection.close()


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate tool argument keys")
    return result


def run_case(args, scenario, stream, thinking):
    spec = SPECS[scenario]
    result = {"scenario": scenario, "stream": stream, "thinking": thinking, "status": "fail",
              "structured_call_ok": False, "roundtrip_ok": False, "turns": []}
    started = time.monotonic()
    try:
        tool = {"type": "function", "function": {"name": spec["name"],
                "description": "Store or look up the supplied arguments and return a receipt.",
                "parameters": {"type": "object", "properties": spec["properties"], "required": list(spec["properties"])}}}
        body = {"model": args.model, "messages": [{"role": "user", "content": spec["prompt"]}],
                "tools": [tool], "tool_choice": "auto", "stream": stream, "think": thinking,
                "temperature": 0, "max_tokens": getattr(args, "max_tokens", 2048), "skills": [], "skills_discovery": False}
        if getattr(args, "reasoning_effort", None):
            body["reasoning_effort"] = args.reasoning_effort
        if stream:
            body["stream_options"] = {"include_usage": True}
        first = {}
        result["turns"].append(first)
        message, finish, usage = request(args.url, body, args.timeout, first)
        calls = message.get("tool_calls", [])
        if finish != "tool_calls" or len(calls) != 1 or not usage:
            raise ValueError(f"Expected one structured call, finish_reason=tool_calls, usage; got {finish}, {len(calls)}, {usage}")
        call = calls[0]
        if not call.get("id") or call.get("type") != "function" or call["function"]["name"] != spec["name"]:
            raise ValueError("Missing call ID or incorrect function identity")
        arguments = json.loads(call["function"]["arguments"], object_pairs_hook=unique_object)
        result["arguments"] = arguments
        if arguments != spec["expected"]:
            raise ValueError(f"Arguments differ: expected {spec['expected']!r}, received {arguments!r}")
        result["structured_call_ok"] = True
        receipt = "receipt-" + hashlib.sha256(f"{scenario}:{stream}:{thinking}".encode()).hexdigest()[:12]
        second = {}
        result["turns"].append(second)
        followup = {**body, "tool_choice": "none", "messages": [*body["messages"], message,
                    {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"receipt": receipt})},
                    {"role": "user", "content": "Return only the receipt from the tool result. Do not call another tool."}]}
        final, finish, usage = request(args.url, followup, args.timeout, second)
        if finish != "stop" or final.get("tool_calls") or final.get("content", "").strip() != receipt or not usage:
            raise ValueError(f"Tool-result round trip failed: finish={finish}, message={final}")
        result["roundtrip_ok"] = True
        result["status"] = "ok"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["wall_seconds"] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", help="Exact hosted model ID; defaults to the sole model from /v1/models")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", default=900, type=float)
    parser.add_argument("--scenarios", default=",".join(SPECS))
    parser.add_argument("--thinking", choices=("off", "on", "off,on"), default="off,on")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    args = parser.parse_args()
    scenarios = args.scenarios.split(",")
    if any(name not in SPECS for name in scenarios) or len(scenarios) != len(set(scenarios)):
        parser.error("Scenarios must be unique known names")
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")
    args.model = args.model or hosted_model(args.url, args.timeout)
    report = {"started_at_unix": time.time(), "url": args.url, "model": args.model,
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "run_complete": False, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        for thinking in (value == "on" for value in args.thinking.split(",")):
            for stream in (False, True):
                case = run_case(args, scenario, stream, thinking)
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                print(scenario, f"stream={stream}", f"thinking={thinking}", case["status"], case.get("error", ""), flush=True)
    report["run_complete"] = True
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return int(any(case["status"] != "ok" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
