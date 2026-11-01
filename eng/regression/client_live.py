"""Bounded live HTTP SDK checks using a prebuilt client and the owned server.

This qualifies semantic two-turn conversations, not captured SDK wire history or
request cancellation. The client example intentionally omits sampler options;
its calls use the wrapper's fixed server defaults. Raw controls send seed=42.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import urllib.parse

from .processes import containment_failure_evidence, run_process
from .reports import file_hash


_ORACLE = "eng/tests/dotnet-local-chat-e2e.py"
PROMPT = "What is 2 + 2? Reply with only the single digit answer."
FOLLOW_UP = "Multiply your previous answer by 3. Reply with only the number."


def load_oracle(repo):
    spec = importlib.util.spec_from_file_location("tensorsharp_client_answer_oracle", Path(repo) / _ORACLE)
    if spec is None or spec.loader is None:
        raise ValueError("HTTP client semantic oracle is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_endpoint(base):
    parsed = urllib.parse.urlsplit(base)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username is not None or
            parsed.password is not None or parsed.path not in {"", "/"} or parsed.query or parsed.fragment or
            parsed.port is None or not 0 < parsed.port < 65536):
        raise ValueError("HTTP client target must be the owned 127.0.0.1 listener")
    return base.rstrip("/") + "/v1/"


def availability(context):
    command = context.get("http_client_command")
    if not command or isinstance(command, str):
        return False, "provide a prebuilt HTTP client with --http-client; this adapter never builds it"
    if not (Path(context["repo"]) / _ORACLE).is_file():
        return False, "repository HTTP client answer oracle is missing"
    return True, "prebuilt HTTP client and semantic oracle are present"


def validate_client_output(item, *, client, mode, model, endpoint, oracle):
    if item.get("exit_code") != 0 or item.get("timed_out") or item.get("cancelled"):
        raise ValueError("HTTP client process failed, timed out or was cancelled")
    if item.get("output_truncated"):
        raise ValueError("HTTP client output was truncated")
    if containment_failure_evidence(item) is not None or item.get("teardown", {}).get("verified") is not True:
        raise ValueError("HTTP client process containment was not verified")
    expected_header = f"TensorSharp HTTP chat via {client}; mode={mode}; model={model}; endpoint={endpoint}"
    stderr = [line.strip() for line in item.get("stderr", "").splitlines() if line.strip()]
    if stderr != [expected_header]:
        raise ValueError("HTTP client identity header differs or stderr contains additional warnings/errors")
    lines = [line for line in item.get("stdout", "").splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError("HTTP client must emit exactly two assistant-answer lines, without logs or prompts")
    try:
        for text, expected in zip(lines, ("4", "12")):
            oracle.answer_text(text, expected)
    except AssertionError as error:
        raise ValueError(str(error)) from error
    return [line.strip() for line in lines]


def _require(result, name, condition, expected=None, actual=None):
    result["assertions"].append({"name": name, "passed": bool(condition), "expected": expected, "actual": actual})
    if not condition:
        raise ValueError(name)


def run_http_client(http, model, context, result):
    # Imported only after scenarios has initialized the shared validation helpers.
    from .scenarios import validate_chat
    endpoint = local_endpoint(http.base)
    alias = result["evidence"].get("model_alias")
    if (not isinstance(alias, dict) or alias.get("loaded_filename") != model or
            alias.get("request_model") != model or alias.get("expected_response_model") != model or
            not isinstance(alias.get("advertised_id"), str) or
            not alias["advertised_id"] or alias["advertised_id"] not in {model, Path(model).stem}):
        raise ValueError("HTTP client requires a verified advertised alias for the loaded model")
    sdk_model = alias["advertised_id"]
    command = context["http_client_command"]
    oracle = load_oracle(context["repo"])
    evidence = result["evidence"]["http_client"] = {
        "endpoint": endpoint, "model": model, "sdk_model": sdk_model, "oracle": _ORACLE,
        "oracle_sha256": file_hash(Path(context["repo"]) / _ORACLE),
        "client_command": command, "client_sha256": file_hash(command[-1]), "invocations": [],
        "sampling": "SDK example omits seed/temperature; owned-server defaults are seed42 and greedy. Raw controls explicitly request them.",
        "limitations": ["Two-turn semantic results do not capture or prove complete SDK wire-history payloads.",
                        "Per-SDK terminal/usage metadata, cancellation, deadlines and native IChatClient are not qualified by this adapter."]}
    result["inputs"]["http_client_fixture"] = {"prompt": PROMPT, "follow_up": FOLLOW_UP, "expected": ["4", "12"]}

    raw = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 64,
           "temperature": 0, "seed": 42, "think": False, "stream": False}
    complete = validate_chat(http.json("/v1/chat/completions", raw), r"\A\s*4\.?\s*\Z", expected_model=model, max_tokens=64)
    streamed = http.stream("/v1/chat/completions", dict(raw, stream=True))
    try:
        oracle.answer_text(streamed.get("content", ""), "4")
    except AssertionError as error:
        raise ValueError(str(error)) from error
    _require(result, "raw HTTP stream has normal stop and terminal event", streamed.get("done") is True and streamed.get("finish_reason") == "stop", "done/stop", {key: streamed.get(key) for key in ("done", "finish_reason")})
    usage = streamed.get("usage") or {}
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    _require(result, "raw HTTP stream has bounded positive usage", type(prompt) is int and prompt > 0 and type(completion) is int and 0 < completion <= 64 and usage.get("total_tokens") == prompt + completion, "positive counts, <=64 completion tokens, consistent total", usage)
    _require(result, "raw HTTP streaming and complete answers agree semantically", complete.strip().rstrip(".") == streamed["content"].strip().rstrip("."), complete, streamed["content"])
    evidence["raw_control"] = {"complete_answer": complete, "stream_answer": streamed["content"], "usage": usage}

    for client in ("extensions", "openai"):
        for mode in ("stream", "response"):
            remaining = http.remaining()
            seconds = max(1, min(86400, math.ceil(remaining)))
            # The SDK example requires the exact /v1/models ID, while response
            # identity and raw controls use the verified loaded filename.
            argv = list(command) + ["--base-url", endpoint, "--model", sdk_model, "--client", client, "--mode", mode,
                "--prompt", PROMPT, "--follow-up", FOLLOW_UP, "--max-tokens", "64", "--timeout-seconds", str(seconds)]
            client_env = dict(context.get("env") or {})
            # The SDK client must reach only this owned loopback server even on a
            # workstation with ambient proxy configuration.
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                client_env[key] = ""
            client_env.update(NO_PROXY="127.0.0.1,localhost,::1", no_proxy="127.0.0.1,localhost,::1")
            item = run_process(argv, cwd=context["repo"], artifact_dir=context["artifact_dir"],
                               stage="http-client-" + client + "-" + mode, timeout_seconds=remaining, env=client_env)
            # Retain containment evidence even if a later assertion fails. The
            # outer runner halts subsequent work on unverified owned resources.
            result["processes"].append({key: value for key, value in item.items() if key not in {"stdout", "stderr"}})
            row = {"client": client, "mode": mode, "passed": False}
            evidence["invocations"].append(row)
            try:
                answers = validate_client_output(item, client=client, mode=mode, model=sdk_model, endpoint=endpoint, oracle=oracle)
            except ValueError as error:
                row["error"] = str(error)
                _require(result, f"HTTP {client} {mode} semantic two-turn conversation", False, ["4", "12"], str(error))
            row.update(passed=True, answers=answers)
            _require(result, f"HTTP {client} {mode} semantic two-turn conversation", True, ["4", "12"], answers)
    evidence["counts"] = {"sdk_mode_invocations": 4, "semantic_answers": 8, "raw_control_requests": 2}
