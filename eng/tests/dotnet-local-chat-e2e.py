#!/usr/bin/env python3
"""Opt-in model-backed checks against an already-running loopback TensorSharp server.

Build examples/dotnet-local-chat first. Start TensorSharp separately with an
existing supported model and a server token cap of at least 512, then run:
  python3 eng/tests/dotnet-local-chat-e2e.py --base-url http://127.0.0.1:8080/v1

No server, model download, GPU job, or mock is started by this harness. Record
the server commit, model, backend and native-library provenance separately.
Python 3.9+; SIGINT verification requires macOS or Linux.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from urllib import error, parse, request


PROMPT = "What is 2 + 2? Reply with only the single digit answer."
LONG_PROMPT = "Count the integers from 1 through 1000, one integer per line. Start immediately with 1 and do not stop early."


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def public_text(value):
    """Keep local model/home paths out of shareable diagnostics."""
    text = str(value).replace(str(Path.home()), "[home]")
    return re.sub(r"(?<![\w:])/(?:Users|home|Volumes|private|var|tmp)/[^\s\"']+", "[local-path]", text)


def loopback_url(value):
    uri = parse.urlsplit(value)
    try:
        loopback = uri.hostname == "localhost" or ipaddress.ip_address(uri.hostname or "").is_loopback
        uri.port  # Reject malformed ports before making a request.
    except ValueError:
        loopback = False
    if (uri.scheme not in ("http", "https") or not loopback or uri.username is not None
            or uri.password is not None or uri.query or uri.fragment or not uri.path.rstrip("/").endswith("/v1")):
        raise argparse.ArgumentTypeError("Use a loopback HTTP(S) URL ending in /v1, without credentials, query or fragment.")
    return value.rstrip("/")


def child_environment():
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("TENSORSHARP_") or key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"):
            del env[key]
    return env


def command(args, model, client, prompt=PROMPT, max_tokens=64, base_url=None):
    return ["dotnet", str(args.dll), "--base-url", base_url or args.base_url,
            "--model", model, "--client", client, "--prompt", prompt,
            "--max-tokens", str(max_tokens), "--timeout-seconds", str(args.timeout_seconds)]


def run_client(args, model, client, **kwargs):
    return subprocess.run(command(args, model, client, **kwargs), env=child_environment(),
                          capture_output=True, text=True, timeout=args.timeout_seconds + 15)


def answer_text(text):
    require(bool(text.strip()), "The model produced no answer text.")
    require("error:" not in text.lower(), "The server returned error text inside a successful HTTP stream: " + public_text(text))
    require(re.search(r"\b(?:4|four)\b", text, re.IGNORECASE) is not None,
            "The model did not answer the arithmetic prompt meaningfully: " + public_text(text))


def checked_answer(result):
    require(result.returncode == 0,
            f"Client exit {result.returncode}: {public_text(result.stderr)} {public_text(result.stdout)}")
    answer_text(result.stdout)
    return result.stdout.strip()


def read_models(args, opener):
    req = request.Request(args.base_url + "/models", headers={"Authorization": "Bearer local-not-a-secret"})
    with opener.open(req, timeout=args.timeout_seconds) as response:
        require(response.status == 200, f"Discovery HTTP {response.status}")
        data = json.load(response)
    ids = [item["id"] for item in data.get("data", []) if isinstance(item, dict) and isinstance(item.get("id"), str)]
    require(bool(ids), "No model advertised by /v1/models. Configure the server's existing model first.")
    if args.model:
        require(args.model in ids, "The requested model ID is not advertised by /v1/models.")
        return args.model
    require(len(ids) == 1, "Multiple models advertised; pass --model with an exact discovery ID.")
    return ids[0]


def raw_stream(args, model, opener):
    payload = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
               "stream": True, "max_tokens": 64, "temperature": 0}
    req = request.Request(args.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json", "Authorization": "Bearer local-not-a-secret"})
    pieces, models, finish_reasons = [], set(), []
    done, usage = False, None
    deadline = time.monotonic() + args.timeout_seconds
    with opener.open(req, timeout=args.timeout_seconds) as response:
        require(response.status == 200, f"Chat HTTP {response.status}")
        require(response.headers.get_content_type() == "text/event-stream", "Expected an SSE response.")
        data_lines = []
        for raw_line in response:
            require(time.monotonic() < deadline, "SSE response exceeded the configured overall deadline.")
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
                continue
            if line or not data_lines:
                continue
            frame = "\n".join(data_lines)
            data_lines.clear()
            if frame == "[DONE]":
                done = True
                break
            chunk = json.loads(frame)
            if isinstance(chunk.get("model"), str):
                models.add(chunk["model"])
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if isinstance(content, str):
                    pieces.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reasons.append(choice["finish_reason"])
    text = "".join(pieces)
    answer_text(text)
    require(done, "SSE stream did not include [DONE].")
    require(finish_reasons and finish_reasons[-1] == "stop", "Expected a normal stop finish reason for the short answer.")
    # TensorSharp advertises an extension-free ID, while generated chunks may
    # identify the same model using its .gguf filename.
    normalized = lambda name: name[:-5] if name.lower().endswith(".gguf") else name
    require(models and all(normalized(name) == normalized(model) for name in models), "SSE model identity differs from discovery.")
    require(usage and usage.get("prompt_tokens", 0) > 0 and usage.get("completion_tokens", 0) > 0,
            "Expected positive prompt/completion token metrics from the inference stream.")
    return text.strip(), usage


def cancel_client(args, model, client):
    proc = subprocess.Popen(command(args, model, client, LONG_PROMPT, 512), env=child_environment(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, start_new_session=True)
    output, errors = [], []
    events = queue.Queue()

    def drain(pipe, sink, report_first=False):
        while True:
            chunk = pipe.read(1 if report_first else 4096)
            if not chunk:
                break
            sink.append(chunk)
            if report_first and chunk.strip():
                events.put(True)
        if report_first:
            events.put(False)

    threads = [threading.Thread(target=drain, args=(proc.stdout, output, True), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, errors), daemon=True)]
    for thread in threads:
        thread.start()
    try:
        try:
            has_output = events.get(timeout=args.timeout_seconds)
        except queue.Empty:
            raise AssertionError("No model output arrived before the cancellation-test deadline.")
        require(has_output, "Client ended before producing output for the cancellation check: "
                + public_text(b"".join(errors).decode("utf-8", errors="replace")))
        require(proc.poll() is None, "Generation finished before SIGINT; cancellation was not verified.")
        os.kill(proc.pid, signal.SIGINT)  # Only the client process started above.
        proc.wait(timeout=15)
        for thread in threads:
            thread.join(timeout=2)
        stdout = b"".join(output).decode("utf-8", errors="replace")
        stderr = b"".join(errors).decode("utf-8", errors="replace")
        require(proc.returncode == 130 and "cancelled" in stderr.lower(),
                f"SIGINT did not produce the client's cancellation result: exit={proc.returncode}; {public_text(stderr)}")
        require(bool(stdout.strip()), "Expected partial model output before cancellation.")
        return stdout.strip()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        for pipe in (proc.stdout, proc.stderr):
            pipe.close()


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, type=loopback_url)
    parser.add_argument("--model", help="Exact /v1/models ID; automatically selected only when there is one.")
    parser.add_argument("--dll", type=Path,
                        default=Path(__file__).resolve().parents[2] / "examples/dotnet-local-chat/bin/Release/net10.0/TensorSharp.LocalChat.dll")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    require(os.name == "posix", "SIGINT verification in this harness requires macOS or Linux.")
    require(0 < args.timeout_seconds <= 1800, "--timeout-seconds must be between 1 and 1800.")
    args.dll = args.dll.resolve()
    require(args.dll.is_file(), "Build the Release starter DLL first, or pass --dll.")
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    model = read_models(args, opener)
    print("PASS TensorSharp model discovery: " + public_text(Path(model).name), flush=True)
    passed = 1
    for client in ("openai", "extensions"):
        answer = checked_answer(run_client(args, model, client))
        print(f"PASS {client}: model-backed answer {public_text(answer)!r}", flush=True)
        passed += 1
    answer, usage = raw_stream(args, model, opener)
    print(f"PASS raw SSE: answer={public_text(answer)!r}; model identity; stop; [DONE]; "
          f"prompt_tokens={usage['prompt_tokens']}; completion_tokens={usage['completion_tokens']}", flush=True)
    passed += 1
    invalid = "__e2e_model_not_advertised__"
    require(invalid != model, "Unexpected discovery ID conflicts with the invalid-model fixture.")
    result = run_client(args, invalid, "extensions")
    require(result.returncode == 1 and "not advertised" in result.stderr.lower(), "Invalid model was not rejected by discovery preflight.")
    print("PASS unknown model: actionable discovery preflight error", flush=True)
    passed += 1
    uri = parse.urlsplit(args.base_url)
    missing = parse.urlunsplit((uri.scheme, uri.netloc, "/missing/v1", "", ""))
    result = run_client(args, model, "extensions", base_url=missing)
    require(result.returncode == 1 and "404" in result.stderr, "Expected an actionable HTTP 404 for the missing API base.")
    print("PASS wrong API base: actionable HTTP 404", flush=True)
    passed += 1
    for client in ("openai", "extensions"):
        partial = cancel_client(args, model, client)
        print(f"PASS {client}: SIGINT after partial output {public_text(partial[:80])!r}; exit 130", flush=True)
        passed += 1
        answer = checked_answer(run_client(args, model, client))
        print(f"PASS {client}: server recovery after cancellation; answer={public_text(answer)!r}", flush=True)
        passed += 1
    print(f"{passed} endpoint checks passed against the already-running TensorSharp server. "
          "Record its actual model/backend/build provenance; these checks are not a benchmark.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, KeyError, subprocess.TimeoutExpired, error.URLError) as exc:
        print("FAIL " + public_text(exc), file=sys.stderr)
        sys.exit(1)
