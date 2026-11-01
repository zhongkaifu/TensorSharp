#!/usr/bin/env python3
"""Deterministic SDK/HTTP integration checks; no TensorSharp engine or model runs.

Build first, then run `python3 eng/tests/dotnet-local-chat.py [path/to/starter.dll]` from the repository root.
Only Python's standard library and the already-built .NET app are used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MODEL = "mock-model-Q4_K_M"
PROMPT = "Say hello from this mock test."
TEXT = "Hello from mock."
ARITHMETIC_PROMPT = "What is 2 + 2? Reply with only the single digit answer."
FOLLOW_UP = "Multiply your previous answer by 3. Reply with only the number."


def message_text(message):
    content = message.get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if part.get("type") == "text")
    return content


def reply_text(payload):
    messages = payload.get("messages", [])
    if messages and message_text(messages[-1]) == ARITHMETIC_PROMPT:
        return "4"
    if messages and message_text(messages[-1]) == FOLLOW_UP:
        expected = [("user", ARITHMETIC_PROMPT), ("assistant", "4"), ("user", FOLLOW_UP)]
        actual = [(message.get("role"), message_text(message)) for message in messages]
        return "12" if actual == expected else "History was not preserved."
    return TEXT


class MockServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *, models=None, behavior="normal", discovery_status=200):
        super().__init__(("127.0.0.1", 0), Handler)
        self.models = [MODEL] if models is None else models
        self.behavior = behavior
        self.discovery_status = discovery_status
        self.requests = []
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *unused):
        self.shutdown()
        self.server_close()
        self.thread.join(timeout=3)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server_port}/v1"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *unused):
        pass

    def json_response(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # Expected for cancellation/timeout coverage.

    def do_GET(self):
        self.server.requests.append(("GET", self.path, None))
        if self.path != "/v1/models":
            self.json_response(404, {"error": {"message": "Unexpected route"}})
            return
        if self.server.behavior == "timeout":
            time.sleep(3)
        if self.server.discovery_status != 200:
            self.json_response(
                self.server.discovery_status,
                {"error": {"message": "Mock discovery route unavailable", "type": "mock_error"}},
            )
            return
        self.json_response(
            200,
            {"object": "list", "data": [
                {"id": model, "object": "model", "owned_by": "local", "loras": []}
                for model in self.server.models
            ]},
        )

    def do_POST(self):
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except (ValueError, json.JSONDecodeError):
            self.json_response(400, {"error": {"message": "Invalid request JSON"}})
            return
        self.server.requests.append(("POST", self.path, payload))
        if self.path != "/v1/chat/completions":
            self.json_response(404, {"error": {"message": "Unexpected route"}})
            return
        if self.server.behavior == "http-error":
            self.json_response(400, {"error": {"message": "Mock rejected generation options", "type": "invalid_request_error"}})
            return
        text = reply_text(payload)
        if not payload.get("stream", False):
            if self.server.behavior == "generation-timeout":
                time.sleep(3)
            if self.server.behavior == "malformed":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "9")
                self.end_headers()
                self.wfile.write(b"{invalid}")
                return
            if self.server.behavior == "empty":
                text = ""
            if self.server.behavior == "model-error-text":
                text = "Error: Mock model load failed"
            self.json_response(200, {
                "id": "chatcmpl-deterministicmock", "object": "chat.completion", "created": 1700000000,
                "model": payload["model"] + ".gguf",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
            })
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.server.behavior == "malformed":
            self.wfile.write(b"data: {this is invalid JSON}\n\ndata: [DONE]\n\n")
            self.wfile.flush()
            return
        try:
            # TensorSharp's SseWriter frames JSON as data: ...\n\n and may emit
            # keepalive comments. A reported model ID omits the .gguf suffix;
            # TensorSharp's generated chunk model field can retain it.
            self.wfile.write(b": keep-alive\n\n")
            if self.server.behavior == "model-error-text":
                # TensorSharp can report a model load failure as HTTP 200 with
                # an ordinary content delta and finish_reason=stop. A client
                # cannot classify arbitrary assistant text as a typed error.
                self.write_chunk(payload["model"], {"content": "Error: Mock model load failed"}, "stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            pieces = () if self.server.behavior == "empty" else (("Hello", " from", " mock.") if text == TEXT else (text,))
            for piece in pieces:
                self.write_chunk(payload["model"], {"role": None, "content": piece})
                if self.server.behavior == "generation-timeout":
                    time.sleep(3)
            if self.server.behavior == "truncated":
                # EOF after valid content, without finish_reason or [DONE].
                return
            self.write_chunk(
                payload["model"], {"role": None, "content": None}, "stop",
                {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11,
                 "prompt_tokens_details": {"cached_tokens": 0}},
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def write_chunk(self, model, delta, finish_reason=None, usage=None):
        chunk = {
            "id": "chatcmpl-deterministicmock",
            "object": "chat.completion.chunk", "created": 1700000000,
            "model": model + ".gguf",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            chunk["usage"] = usage
        self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
        self.wfile.flush()


def run_app(dll, server, *arguments, base_suffix=""):
    env = os.environ.copy()
    # Keep the fixture isolated from any user endpoint or real API credential.
    for key in tuple(env):
        if key.startswith("TENSORSHARP_") or key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
            del env[key]
    return subprocess.run(
        ["dotnet", str(dll), "--base-url", server.base_url + base_suffix, *arguments],
        capture_output=True, text=True, timeout=15, env=env,
    )


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_success(result):
    require(result.returncode == 0, f"Exit {result.returncode}: {result.stderr}\n{result.stdout}")


def check_failure(result, *needles):
    require(result.returncode != 0, f"Expected failure; stdout={result.stdout!r}, stderr={result.stderr!r}")
    require(bool(result.stderr.strip()), "Failure should include a diagnostic on stderr")
    combined = (result.stderr + result.stdout).lower()
    for needle in needles:
        require(needle.lower() in combined, f"Expected {needle!r} in diagnostic: {combined!r}")


def verify_request(server, expected_model=MODEL, max_tokens=17, stream=True):
    require(server.requests[0][:2] == ("GET", "/v1/models"), f"Discovery was not first: {server.requests!r}")
    posts = [item for item in server.requests if item[0] == "POST"]
    require(len(posts) == 1, f"Expected one chat POST: {server.requests!r}")
    _, path, body = posts[0]
    require(path == "/v1/chat/completions", f"Wrong chat path: {path}")
    require(body.get("model") == expected_model, f"Wrong model: {body!r}")
    require(body.get("stream", False) is stream, f"Wrong streaming mode: {body!r}")
    # Both token-limit spellings are accepted by TensorSharp. The official SDK
    # version determines which of these fields its ChatCompletionOptions emits.
    limits = [body[key] for key in ("max_tokens", "max_completion_tokens") if key in body]
    require(limits == [max_tokens], f"Wrong/missing token limit: {body!r}")
    user_messages = [message for message in body.get("messages", []) if message.get("role") == "user"]
    require(len(user_messages) == 1, f"Expected one user message: {body!r}")
    content = message_text(user_messages[0])
    require(content == PROMPT, f"Wrong user content: {body!r}")


def no_chat(server):
    require(not any(method == "POST" for method, _, _ in server.requests), f"Unexpected generation request: {server.requests!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[2] / "examples" / "dotnet-local-chat" / "bin" / "Release" / "net10.0" / "TensorSharp.LocalChat.dll")
    args = parser.parse_args()
    dll = args.dll.resolve()
    if not dll.is_file():
        parser.error(f"Built starter DLL not found: {dll}. Build Release first or pass its path.")

    passed = 0

    def passed_case(name):
        nonlocal passed
        passed += 1
        print(f"PASS {name}")

    with MockServer() as server:
        result = run_app(dll, server, "--list-models")
        check_success(result)
        require(MODEL in result.stdout, f"Model ID missing: {result.stdout!r}")
        no_chat(server)
        passed_case("model discovery: GET /v1/models")

    for client in ("openai", "extensions"):
        with MockServer() as server:
            result = run_app(dll, server, "--client", client, "--prompt", PROMPT, "--max-tokens", "17")
            check_success(result)
            require(result.stdout == TEXT + "\n", f"Wrong streamed text: {result.stdout!r}")
            verify_request(server)
            passed_case(f"{client}: single-model discovery, payload, SSE deltas/keepalive/end")
        with MockServer(behavior="empty") as server:
            result = run_app(dll, server, "--client", client, "--prompt", PROMPT)
            check_failure(result)
            passed_case(f"{client}: empty completion fails with diagnostic")
        with MockServer(behavior="model-error-text") as server:
            result = run_app(dll, server, "--client", client, "--prompt", PROMPT)
            check_success(result)
            require(result.stdout == "Error: Mock model load failed\n", f"Wrong model failure content: {result.stdout!r}")
            passed_case(f"{client}: HTTP 200 SSE error text remains transport success (known server limitation)")

    with MockServer() as server:
        result = run_app(dll, server, "--prompt", PROMPT, "--max-tokens", "17", base_suffix="/")
        check_success(result)
        require(result.stdout == TEXT + "\n", f"Wrong output: {result.stdout!r}")
        verify_request(server)
        passed_case("base URL trailing slash")

    with MockServer(models=[MODEL, "second-model"]) as server:
        result = run_app(dll, server, "--model", "second-model", "--prompt", PROMPT, "--max-tokens", "17")
        check_success(result)
        verify_request(server, expected_model="second-model")
        passed_case("explicit model selection from multiple IDs")

    with MockServer() as server:
        result = run_app(dll, server, "--model", "missing-model", "--prompt", PROMPT)
        check_failure(result, "advertised", "--model")
        no_chat(server)
        passed_case("unknown model rejected before generation")

    with MockServer(models=[]) as server:
        result = run_app(dll, server, "--prompt", PROMPT)
        check_failure(result, "model")
        no_chat(server)
        passed_case("empty model list reports actionable failure")

    with MockServer(models=[MODEL, "second-model"]) as server:
        result = run_app(dll, server, "--prompt", PROMPT)
        check_failure(result, "--model")
        no_chat(server)
        passed_case("ambiguous model list asks for --model")

    for client in ("openai", "extensions"):
        with MockServer(behavior="http-error") as server:
            result = run_app(dll, server, "--client", client, "--prompt", PROMPT)
            check_failure(result, "400", "Mock rejected generation options")
            passed_case(f"{client}: HTTP 400 preserves server diagnostic")

    with MockServer(discovery_status=404) as server:
        result = run_app(dll, server, "--prompt", PROMPT)
        check_failure(result, "404")
        no_chat(server)
        passed_case("model-discovery HTTP failure")

    for client in ("openai", "extensions"):
        for behavior in ("malformed", "truncated"):
            with MockServer(behavior=behavior) as server:
                result = run_app(dll, server, "--client", client, "--prompt", PROMPT)
                check_failure(result)
                passed_case(f"{client}: {behavior} SSE fails with diagnostic")
        with MockServer(behavior="generation-timeout") as server:
            result = run_app(dll, server, "--client", client, "--prompt", PROMPT, "--timeout-seconds", "1")
            check_failure(result, "time")
            require(result.stdout.startswith("Hello"), "Expected first chunk before generation timeout")
            passed_case(f"{client}: generation timeout reports partial output")
            server.behavior = "normal"
            result = run_app(dll, server, "--client", client, "--prompt", PROMPT)
            check_success(result)
            require(result.stdout == TEXT + "\n", "Client did not recover after the streaming deadline")

    with MockServer(behavior="timeout") as server:
        result = run_app(dll, server, "--prompt", PROMPT, "--timeout-seconds", "1")
        check_failure(result, "time")
        no_chat(server)
        passed_case("request timeout fails with diagnostic")

    for client in ("openai", "extensions"):
        with MockServer() as server:
            result = run_app(dll, server, "--client", client, "--mode", "response", "--prompt", PROMPT, "--max-tokens", "17")
            check_success(result)
            require(result.stdout == TEXT + "\n", f"Wrong complete response: {result.stdout!r}")
            verify_request(server, stream=False)
            passed_case(f"{client}: response mode payload and complete answer")
        for behavior in ("empty", "malformed", "http-error"):
            with MockServer(behavior=behavior) as server:
                result = run_app(dll, server, "--client", client, "--mode", "response", "--prompt", PROMPT)
                check_failure(result)
                passed_case(f"{client}: response mode {behavior} fails with diagnostic")
        with MockServer(behavior="generation-timeout") as server:
            result = run_app(dll, server, "--client", client, "--mode", "response", "--prompt", PROMPT, "--timeout-seconds", "1")
            check_failure(result, "time")
            require(not result.stdout, "Response mode should not print a partial answer before completion")
            server.behavior = "normal"
            result = run_app(dll, server, "--client", client, "--mode", "response", "--prompt", PROMPT)
            check_success(result)
            require(result.stdout == TEXT + "\n", "Client did not recover after the response deadline")
            passed_case(f"{client}: response deadline and recovery")
        for mode in ("stream", "response"):
            with MockServer() as server:
                result = run_app(dll, server, "--client", client, "--mode", mode,
                                 "--prompt", ARITHMETIC_PROMPT, "--follow-up", FOLLOW_UP, "--max-tokens", "17")
                check_success(result)
                require(result.stdout == "4\n12\n", f"Wrong two-turn output: {result.stdout!r}")
                posts = [body for method, path, body in server.requests if method == "POST"]
                require(len(posts) == 2, f"Expected two chat requests: {server.requests!r}")
                expected = [("user", ARITHMETIC_PROMPT), ("assistant", "4"), ("user", FOLLOW_UP)]
                require([(message.get("role"), message_text(message)) for message in posts[1]["messages"]] == expected,
                        f"Second request lost the full conversation: {posts[1]!r}")
                require(all(body.get("stream", False) is (mode == "stream") for body in posts), "A turn used the wrong mode")
                passed_case(f"{client}: {mode} preserves user/assistant/user history (4 then 12)")

    print(f"{passed} checks passed. Mock HTTP integration only; no real model was run.")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.TimeoutExpired) as error:
        print(f"FAIL {error}", file=sys.stderr)
        sys.exit(1)
