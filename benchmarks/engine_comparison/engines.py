#!/usr/bin/env python3
"""
Engine adapters: server lifecycle managers + a uniform OpenAI-HTTP runner.

Every engine is driven through the same `/v1/chat/completions` surface so the
comparison is apples-to-apples. Metrics are derived from the *streamed*
response so they are independent of any engine-specific internal timer:

  * ttft_ms      - time to first streamed token (prefill latency proxy)
  * prefill_tps  - prompt_tokens / ttft
  * decode_tps   - (completion_tokens - 1) / (t_last_token - t_first_token)
  * prompt_tokens / completion_tokens come from the final `usage` block

DiffusionGemma denoises a whole block at once (no per-token stream), so it is
run non-streaming and its throughput is wall-clock tokens/second.

A scenario may also drive a multi-turn workflow (`run_conversation`), where each
turn's request is built from the previous turn's response. Such a cell reports
the FINAL turn's metrics — the turn whose prompt is the whole conversation —
with `total_wall_ms` covering every turn and `turns` saying how many ran.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests

import config


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    engine: str
    backend: str
    model: str
    scenario: str
    status: str = "fail"            # ok | fail | skipped
    detail: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    total_wall_ms: float = 0.0
    finish_reason: str = ""
    tool_call_ok: Optional[bool] = None     # function_call scenario correctness
    output_preview: str = ""
    # Full generated text (capped) so cross-engine output quality can be
    # compared offline from the result JSONs alone (see report.py's
    # output-quality section). The preview above stays for humans skimming.
    output_text: str = ""
    # Extra benchmark axes.
    mtp: bool = False                        # MTP/NextN speculative decoding engaged
    tp: int = 1                              # tensor-parallel degree (GPUs the model is split across)
    concurrency: int = 1                     # parallel identical requests at this cell
    aggregate_decode_tps: float = 0.0        # system-wide decode tok/s across all parallel seqs
    requests_ok: int = 0                     # successful requests out of `concurrency`
    # MoE CPU offload (`--n-cpu-moe N` / `--cpu-moe-threads M`). 0 layers is the
    # baseline; -1 means `all`. 0 threads means the engine picked its own.
    cpu_moe_layers: int = 0
    cpu_moe_threads: int = 0
    # Round trips this cell drove, and how many the scenario asked for. 1/1 for
    # every single-request scenario; the client-driven workflows (agentic,
    # code_edit) report the turns they got through, and the throughput fields
    # above belong to the LAST of them (the turn that carries the whole
    # conversation as its prompt), while `total_wall_ms` covers the conversation
    # end to end. `turns < turns_expected` means the workflow stopped early, so
    # those timings describe a SHORTER conversation than the cell's name claims
    # and report.py refuses to tabulate them against a complete one.
    turns: int = 1
    turns_expected: int = 1

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ---------------------------------------------------------------------------
# Uniform OpenAI chat runner
# ---------------------------------------------------------------------------
def thinking_body(engine: str, enabled: bool) -> dict:
    """Request fields that put an engine into (or out of) reasoning mode.

    Each engine spells it differently — TensorSharp takes a top-level `think`
    boolean, llama.cpp passes `enable_thinking` through to the GGUF's chat
    template — and their DEFAULTS differ, which silently makes the two sides
    answer different questions: with reasoning left on, llama.cpp spends the
    whole token budget thinking and never reaches the final answer, so its
    output shares almost nothing with TensorSharp's direct answer and its
    `json_mode` cell returns an unfinished (invalid) object. Setting the mode
    explicitly on both sides is what keeps the output-quality comparison
    meaningful."""
    if engine == "tensorsharp":
        return {"think": bool(enabled)}
    if engine == "llamacpp":
        return {"chat_template_kwargs": {"enable_thinking": bool(enabled)}}
    return {}


def run_openai_chat(base_url: str, model_name: str, messages: list, *,
                    tools: Optional[list] = None,
                    response_format: Optional[dict] = None,
                    max_tokens: int = 128,
                    stream: bool = True,
                    extra_body: Optional[dict] = None,
                    timeout_s: float = 1200.0) -> dict:
    """Run one chat completion and return a metrics dict. Raises on transport
    error; the caller maps that onto a failed BenchResult."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if extra_body:
        body.update(extra_body)
    if tools:
        body["tools"] = tools
    if response_format:
        body["response_format"] = response_format
    if stream:
        body["stream_options"] = {"include_usage": True}
        # Ask the server to attach its own generation timer to the stream
        # (llama.cpp + TensorSharp both honor this). The client-side window
        # `t_last - t_first` over a handful of tokens is NOT a reliable decode
        # rate: some servers batch several tokens into one SSE frame and flush
        # them in a burst, compressing the measured window and inflating the
        # client rate (e.g. ~64 t/s reported for a true ~40 t/s at max_tokens=8).
        # The server timer measures actual compute and is burst-immune, so we
        # prefer it for decode_tps when present (see `_run_streaming`). Unknown
        # to an engine that ignores it; harmless.
        body["timings_per_token"] = True

    if not stream:
        return _run_blocking(url, body, timeout_s)
    return _run_streaming(url, body, timeout_s)


def _run_blocking(url: str, body: dict, timeout_s: float) -> dict:
    t0 = time.monotonic()
    resp = requests.post(url, json=body, timeout=timeout_s)
    t_end = time.monotonic()
    wall = (t_end - t0)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"completion error: {data['error']}")
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    usage = data.get("usage", {}) or {}
    completion = int(usage.get("completion_tokens", 0) or 0)
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    text = msg.get("content") or msg.get("reasoning_content") or ""
    tool_calls = msg.get("tool_calls") or []
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_ms": 0.0,
        "prefill_tps": 0.0,
        "decode_tps": (completion / wall) if wall > 0 and completion else 0.0,
        "total_wall_ms": wall * 1000.0,
        "finish_reason": choice.get("finish_reason", "") or "",
        "tool_calls": [_tc_name(t) for t in tool_calls],
        "tool_call_details": tool_calls,
        "assistant_message": msg,
        "usage_present": "prompt_tokens" in usage and "completion_tokens" in usage,
        "decode_timing_source": "request_wall",
        "reasoning_text": msg.get("reasoning_content") or "",
        "output_text": text,
        # Absolute monotonic timestamps (shared process clock) so a parallel
        # runner can stitch a system-wide throughput window. No token stream
        # here, so the whole request is the decode window.
        "t_start_abs": t0,
        "t_first_abs": t0,
        "t_last_abs": t_end,
        "t_end_abs": t_end,
    }


def _run_streaming(url: str, body: dict, timeout_s: float,
                   response_log: Optional[dict] = None) -> dict:
    t_start = time.monotonic()
    t_first = None
    t_last = None
    content_chunks = 0
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_fragments: dict[int, dict] = {}
    finish_reason = ""
    usage = {}
    srv_timings: dict = {}        # engine-reported generation timer (burst-immune)

    with requests.post(url, json=body, stream=True,
                       timeout=(30, timeout_s)) as resp:
        if response_log is not None:
            response_log["http_status"] = resp.status_code
            response_log["content_type"] = resp.headers.get("Content-Type")
            response_log["sse_lines"] = []
            if resp.status_code >= 400:
                response_log["body"] = resp.text
        resp.raise_for_status()
        # SSE is UTF-8. requests otherwise defaults text/event-stream without
        # an explicit charset to Latin-1, corrupting multilingual content and
        # tool arguments before the JSON parser receives them.
        resp.encoding = "utf-8"
        for raw in resp.iter_lines(decode_unicode=True):
            if response_log is not None:
                # Keep the wire output even if parsing or validation later
                # fails. The default benchmark path allocates no capture list.
                response_log["sse_lines"].append(raw)
            if not raw:
                continue
            if not raw.startswith("data:"):
                continue
            payload = raw[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                raise RuntimeError(f"invalid completion SSE JSON: {payload[:200]}")
            if chunk.get("error"):
                raise RuntimeError(f"completion stream error: {chunk['error']}")
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                srv_timings = chunk["timings"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {}) or {}
            now = time.monotonic()
            # A generated-token event is any visible content, reasoning content,
            # or tool-call fragment. Some engines (llama.cpp + a "thinking" chat
            # template) route the whole answer through `reasoning_content`; those
            # are still generated tokens and must count toward TTFT / decode.
            content = delta.get("content")
            reasoning = delta.get("reasoning_content")
            if content:
                if t_first is None:
                    t_first = now
                t_last = now
                content_chunks += 1
                text_parts.append(content)
            if reasoning:
                if t_first is None:
                    t_first = now
                t_last = now
                content_chunks += 1
                reasoning_parts.append(reasoning)
            for tc in (delta.get("tool_calls") or []):
                if t_first is None:
                    t_first = now
                t_last = now
                index = int(tc.get("index", 0))
                target = tool_fragments.setdefault(index, {
                    "id": "", "type": "function",
                    "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    target["id"] += tc["id"]
                if tc.get("type") is not None:
                    target["type"] = tc["type"]
                for key in ("name", "arguments"):
                    fragment = (tc.get("function") or {}).get(key)
                    if fragment:
                        target["function"][key] += fragment
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    t_end = time.monotonic()
    completion = int(usage.get("completion_tokens", 0) or 0) or content_chunks
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    ttft_ms = ((t_first - t_start) * 1000.0) if t_first else 0.0
    decode_window = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    decode_tps = ((completion - 1) / decode_window) if (decode_window > 0 and completion > 1) else 0.0
    prefill_tps = (prompt / (ttft_ms / 1000.0)) if (ttft_ms > 0 and prompt) else 0.0

    # Prefer the engine's own decode timer (burst-immune) over the streamed-chunk
    # window. `predicted_per_second` is generation-phase tok/s as measured inside
    # the server, independent of how the SSE frames were batched/flushed on the
    # wire. This removes the client-side measurement artifact that otherwise makes
    # a burst-flushing server look ~1.5-2x faster than its true compute at small
    # max_tokens. Falls back to the streamed estimate when no timer is reported.
    pps = srv_timings.get("predicted_per_second") if srv_timings else None
    if pps and pps > 0:
        decode_tps = float(pps)
    tool_calls = [tool_fragments[i] for i in sorted(tool_fragments)]
    assistant_message = {"role": "assistant", "content": "".join(text_parts) or None}
    if reasoning_parts:
        assistant_message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "total_wall_ms": (t_end - t_start) * 1000.0,
        "finish_reason": finish_reason,
        "tool_calls": [_tc_name(tc) for tc in tool_calls],
        "tool_call_details": tool_calls,
        "assistant_message": assistant_message,
        "usage_present": "prompt_tokens" in usage and "completion_tokens" in usage,
        "decode_timing_source": "server" if pps and pps > 0 else "stream_window",
        "server_timings": srv_timings,
        "reasoning_text": "".join(reasoning_parts),
        "output_text": "".join(text_parts) or "".join(reasoning_parts),
        # Absolute monotonic timestamps (shared process clock) for parallel
        # aggregation: t_first_abs..t_last_abs is this request's decode window.
        "t_start_abs": t_start,
        "t_first_abs": t_first,
        "t_last_abs": t_last,
        "t_end_abs": t_end,
    }


def _tc_name(tc: dict) -> str:
    fn = (tc or {}).get("function") or {}
    return fn.get("name") or ""


# ---------------------------------------------------------------------------
# Parallel (concurrent) request runner
# ---------------------------------------------------------------------------
def run_openai_chat_parallel(base_url: str, model_name: str, messages: list, *,
                             concurrency: int,
                             tools: Optional[list] = None,
                             response_format: Optional[dict] = None,
                             max_tokens: int = 128,
                             stream: bool = True,
                             extra_body: Optional[dict] = None,
                             timeout_s: float = 1200.0) -> dict:
    """Fire `concurrency` identical chat completions at the same server at once
    and return one aggregated metrics dict (same keys as `run_openai_chat` plus
    `aggregate_decode_tps`, `requests_ok`, `per_request`); see
    `_aggregate_parallel` for how the numbers are folded together.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = max(1, int(concurrency))
    results: list[Optional[dict]] = [None] * n
    errors: list[Optional[Exception]] = [None] * n

    def _worker(i: int):
        try:
            results[i] = run_openai_chat(
                base_url, model_name, messages,
                tools=tools, response_format=response_format,
                max_tokens=max_tokens, stream=stream,
                extra_body=extra_body, timeout_s=timeout_s)
        except Exception as ex:  # captured per-request; surfaced in aggregate
            errors[i] = ex

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(_worker, range(n)))

    ok = [r for r in results if r is not None]
    if not ok:
        first_err = next((e for e in errors if e is not None), None)
        raise RuntimeError(f"all {n} parallel requests failed: {first_err}")
    return _aggregate_parallel(ok)


def _aggregate_parallel(ok: list) -> dict:
    """Fold the successful parallel results into one cell metrics dict.

    Per-request metrics are the mean across the successful requests;
    `aggregate_decode_tps` is the *system* decode throughput — total generated
    tokens divided by the wall window during which any sequence was decoding
    (max last-token time - min first-token time, on the shared process clock)."""
    def _mean(key: str) -> float:
        return sum(float(r.get(key, 0.0) or 0.0) for r in ok) / len(ok)

    # System-wide decode window on the shared monotonic clock.
    firsts = [r["t_first_abs"] for r in ok if r.get("t_first_abs")]
    lasts = [r["t_last_abs"] for r in ok if r.get("t_last_abs")]
    decode_window = (max(lasts) - min(firsts)) if (firsts and lasts and max(lasts) > min(firsts)) else 0.0
    total_decode_tokens = sum(max(int(r.get("completion_tokens", 0) or 0) - 1, 0) for r in ok)
    aggregate_decode_tps = (total_decode_tokens / decode_window) if decode_window > 0 else 0.0

    starts = [r["t_start_abs"] for r in ok if r.get("t_start_abs")]
    ends = [r["t_end_abs"] for r in ok if r.get("t_end_abs")]
    total_wall_ms = ((max(ends) - min(starts)) * 1000.0) if (starts and ends) else _mean("total_wall_ms")

    rep = ok[0]  # representative request for non-numeric fields
    return {
        "prompt_tokens": int(rep.get("prompt_tokens", 0) or 0),
        "completion_tokens": round(_mean("completion_tokens")),
        "ttft_ms": _mean("ttft_ms"),
        "prefill_tps": _mean("prefill_tps"),
        "decode_tps": _mean("decode_tps"),
        "aggregate_decode_tps": aggregate_decode_tps,
        "total_wall_ms": total_wall_ms,
        "finish_reason": rep.get("finish_reason", "") or "",
        "tool_calls": rep.get("tool_calls") or [],
        "output_text": rep.get("output_text") or "",
        "requests_ok": len(ok),
        "per_request": ok,
    }


# ---------------------------------------------------------------------------
# Client-driven multi-turn conversation runner
# ---------------------------------------------------------------------------
def run_conversation(base_url: str, model_name: str, messages: list, *,
                     followups: list,
                     tools: Optional[list] = None,
                     response_format: Optional[dict] = None,
                     max_tokens: int = 128,
                     stream: bool = True,
                     extra_body: Optional[dict] = None,
                     timeout_s: float = 1200.0) -> dict:
    """Drive a scenario's multi-turn workflow and return ONE cell metrics dict.

    `followups` is the scenario's list of turn builders (see scenarios.py): each
    is handed the metrics of the turn that just finished plus the messages that
    produced it, and returns the request overrides for the next turn. A
    follow-up that raises ends the conversation there; the reason (including the
    exception type, so a harness bug reads differently from a model that did not
    call the tool) is returned as `conversation_detail` and the cell records it.

    The returned dict is the FINAL turn's metrics — the turn whose prompt is the
    whole conversation, which is the one an agentic workload is actually paced
    by — plus `turns`, `turn_metrics` and a `total_wall_ms` covering every turn.
    `t_start_abs` is likewise moved to the start of the conversation so a
    parallel aggregation window spans the whole workflow rather than its tail."""
    t_conv = time.monotonic()
    turn_metrics: list = []
    detail = ""
    msgs = list(messages)
    req = {"tools": tools, "response_format": response_format,
           "extra_body": dict(extra_body) if extra_body else None}
    for i in range(len(followups) + 1):
        m = run_openai_chat(base_url, model_name, msgs,
                            max_tokens=max_tokens, stream=stream,
                            timeout_s=timeout_s, **req)
        turn_metrics.append(m)
        if i >= len(followups):
            break
        try:
            nxt = followups[i](m, msgs)
        except Exception as ex:
            detail = (f"conversation stopped after turn {i + 1}/{len(followups) + 1}: "
                      f"{type(ex).__name__}: {ex}")
            break
        msgs = nxt["messages"]
        # A turn may re-declare tools / response_format, and its extra body
        # fields are merged onto the ones already in effect (so the run-wide
        # reasoning mode survives a turn that only sets `tool_choice`). Whatever
        # a turn does not mention carries forward from the turn before it, not
        # from the start of the conversation — otherwise a later turn would
        # silently undo an earlier turn's change. A scenario therefore states
        # only what it is actually changing.
        merged = dict(req["extra_body"] or {})
        merged.update(nxt.get("extra_body") or {})
        req = {"tools": nxt.get("tools", req["tools"]),
               "response_format": nxt.get("response_format", req["response_format"]),
               "extra_body": merged or None}

    t_end = time.monotonic()
    out = dict(turn_metrics[-1])
    out["turns"] = len(turn_metrics)
    out["turn_metrics"] = turn_metrics
    out["conversation_detail"] = detail
    out["final_turn_wall_ms"] = turn_metrics[-1].get("total_wall_ms", 0.0)
    out["total_wall_ms"] = (t_end - t_conv) * 1000.0
    out["t_start_abs"] = t_conv
    out["t_end_abs"] = t_end
    return out


def run_conversation_parallel(base_url: str, model_name: str, messages: list, *,
                              concurrency: int, followups: list,
                              tools: Optional[list] = None,
                              response_format: Optional[dict] = None,
                              max_tokens: int = 128,
                              stream: bool = True,
                              extra_body: Optional[dict] = None,
                              timeout_s: float = 1200.0) -> dict:
    """`concurrency` copies of the same workflow, driven at one server at once.

    Each client runs its own independent conversation (the follow-ups depend on
    that client's own responses), and the results are folded together exactly
    like `run_openai_chat_parallel` does — with the decode window spanning the
    parallel FINAL turns and the wall window spanning the whole workflows."""
    from concurrent.futures import ThreadPoolExecutor

    n = max(1, int(concurrency))
    results: list[Optional[dict]] = [None] * n
    errors: list[Optional[Exception]] = [None] * n

    def _worker(i: int):
        try:
            results[i] = run_conversation(
                base_url, model_name, messages, followups=followups,
                tools=tools, response_format=response_format,
                max_tokens=max_tokens, stream=stream,
                extra_body=extra_body, timeout_s=timeout_s)
        except Exception as ex:  # captured per-client; surfaced in aggregate
            errors[i] = ex

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(_worker, range(n)))

    ok = [r for r in results if r is not None]
    if not ok:
        first_err = next((e for e in errors if e is not None), None)
        raise RuntimeError(f"all {n} parallel conversations failed: {first_err}")

    rep = ok[0]
    agg = _aggregate_parallel(ok)
    # Correctness is judged on the representative conversation, so the checker
    # needs its structured message and turn history, not just the folded numbers.
    agg["assistant_message"] = rep.get("assistant_message")
    agg["tool_call_details"] = rep.get("tool_call_details") or []
    agg["turn_metrics"] = rep.get("turn_metrics") or []
    # `turns` is the WORST client's, not the representative one's: the cell's
    # claim is "N copies of this workflow ran", so one client that stopped early
    # makes the cell an incomplete workflow even if the representative finished.
    # Reporting the representative's count would hide that behind a full-length
    # number the aggregate timings no longer describe.
    agg["turns"] = min(int(r.get("turns", 1) or 1) for r in ok)
    details = sorted({r.get("conversation_detail") or "" for r in ok} - {""})
    agg["conversation_detail"] = "; ".join(details)
    return agg


# ---------------------------------------------------------------------------
# Server lifecycle base
# ---------------------------------------------------------------------------
def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_port_free(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _port_open(host, port):
            return
        time.sleep(0.5)


def _pid_listening(port: int):
    """PID of the process LISTENING on the port (Windows netstat), or None."""
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True,
                                      stderr=subprocess.DEVNULL)
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and "LISTENING" in parts:
            if parts[1].endswith(f":{port}"):
                try:
                    return int(parts[-1])
                except ValueError:
                    return None
    return None


def _find_free_port(start: int, tries: int = 20) -> int:
    for p in range(start, start + tries):
        if not _port_open("127.0.0.1", p):
            return p
    return start  # caller's launch will then fail with a clear bind error


class ServerHandle:
    """Common helpers for a launched OpenAI server process."""

    def __init__(self, base_url: str, port: int, log_path: Path):
        self.base_url = base_url
        self.port = port
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self.ready_hint = ""     # diagnosis when wait_ready gives up early
        self._served_name = None  # model id from /v1/models (see _note_served_name)

    def _spawn(self, cmd: list[str], cwd: Optional[Path], env: Optional[dict]):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self._log_fh.write("CMD: " + " ".join(str(c) for c in cmd) + "\n\n")
        self._log_fh.flush()
        self.proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
        )

    def _note_served_name(self, resp):
        """Remember the model id the server actually advertises, so requests
        never 404 on a name mismatch (a split GGUF, for instance, is hosted
        under one id that need not be the shard file name we launched with)."""
        try:
            served = [m.get("id") for m in (resp.json().get("data") or []) if m.get("id")]
        except Exception:
            return
        if served:
            self._served_name = served[0]

    def wait_ready(self, timeout_s: float) -> bool:
        url = self.base_url.rstrip("/") + "/v1/models"
        deadline = time.monotonic() + timeout_s
        silent_but_open = 0
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                # Process exited before becoming ready. Surface WHY: the exit
                # code alone diagnoses the common Windows failure where the
                # server binary can't even start. A silent instant exit with an
                # empty log is the signature of a DLL load failure — most often
                # an NTSTATUS in the 0xC0000xxx range, e.g. 0xC0000139
                # (ENTRYPOINT_NOT_FOUND) / 0xC0000135 (DLL_NOT_FOUND), which
                # means the binary is ABI-mismatched against the ggml/llama DLLs
                # beside it (rebuild it) or a dependency DLL is missing.
                rc = self.proc.returncode
                code = rc & 0xFFFFFFFF if rc is not None else None  # NTSTATUS is unsigned
                self.ready_hint = f"process exited early with code {rc}"
                if code is not None:
                    self.ready_hint += f" (0x{code:08X})"
                    if 0xC0000000 <= code <= 0xCFFFFFFF:
                        self.ready_hint += (
                            " — this is a Windows load-time failure (missing or "
                            "ABI-mismatched dependency DLL). Rebuild the server so it "
                            "matches the ggml/llama DLLs beside it. ")
                    else:
                        self.ready_hint += ". "
                return False
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    self._note_served_name(r)
                    return True
                silent_but_open = 0
            except requests.RequestException:
                # Distinguish "nothing listening yet" (normal while the model
                # loads) from "TCP connects but HTTP never answers". The latter
                # is the signature of a leftover server squatting the port: a
                # hung process (e.g. llama-server stuck in a GPU-driver call is
                # unkillable and keeps its socket) lets the kernel complete
                # handshakes into the listen backlog while never serving them.
                # If the port's owner is not our child process, we can never
                # become ready — bail out with a diagnosis instead of burning
                # the whole ready timeout looking stuck.
                if _port_open("127.0.0.1", self.port):
                    silent_but_open += 1
                    if silent_but_open >= 3:
                        owner = _pid_listening(self.port)
                        ours = self.proc.pid if self.proc is not None else None
                        if owner is not None and ours is not None and owner != ours:
                            self.ready_hint = (
                                f"port {self.port} is owned by PID {owner}, not our "
                                f"server (PID {ours}) — a leftover/hung server is "
                                f"squatting the port; kill it (taskkill /F /PID {owner}) "
                                f"or reboot if it will not die. ")
                            return False
                else:
                    silent_but_open = 0
            time.sleep(1.0)
        return False

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
            except Exception:
                pass
            self.proc = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        _wait_port_free("127.0.0.1", self.port, timeout_s=30.0)

    def tail_log(self, n: int = 25) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return ""


# ---------------------------------------------------------------------------
# TensorSharp.Server
# ---------------------------------------------------------------------------
class TensorSharpServer(ServerHandle):
    def __init__(self, model: config.ModelSpec, backend: str, log_path: Path,
                 max_tokens: int = config.SERVER_MAX_TOKENS, mtp: bool = False,
                 tp: int = 1, cpu_moe: Optional[config.CpuMoeSpec] = None):
        super().__init__(f"http://127.0.0.1:{config.TENSORSHARP_PORT}",
                         config.TENSORSHARP_PORT, log_path)
        self.model = model
        self.backend = backend
        self.max_tokens = max_tokens
        self.mtp = mtp
        self.tp = max(1, int(tp or 1))
        self.cpu_moe = cpu_moe or config.CpuMoeSpec()

    def start(self):
        spec = config.BACKENDS[self.backend]
        # The listen address is passed explicitly below, so the configured port
        # (BENCH_TS_PORT / paths.tensorsharp_port) is the one the health checks
        # poll, and an inherited PORT / HOST / ASPNETCORE_URLS cannot move the
        # server. A squatter is usually a leftover server that may still hold
        # GPU memory, so fail fast rather than benchmark beside it on another
        # port.
        if _port_open("127.0.0.1", self.port):
            pid = _pid_listening(self.port)
            raise RuntimeError(
                f"port {self.port} is already in use by PID {pid}. Stop that "
                f"process (taskkill /F /PID {pid}); if it will not die (stuck in "
                f"a GPU-driver call), reboot. To benchmark on another port, set "
                f"BENCH_TS_PORT or paths.tensorsharp_port.")
        # --no-multi-agent: sub-agent delegation is on by default in the server,
        # and it adds five coordination tools and a coordination prompt to every
        # request on a tool-capable model. llama.cpp and vLLM never see those, so
        # prompt tokens, TTFT and output similarity would not be like-for-like.
        cmd = ["dotnet", str(config.TENSORSHARP_SERVER_DLL),
               "--model", str(self.model.gguf),
               "--backend", spec.ts_backend,
               "--host", "127.0.0.1",
               "--port", str(self.port),
               "--no-multi-agent"]
        cmd += [str(a) for a in spec.ts_extra_args]
        cmd += ["--max-tokens", str(self.max_tokens)]
        if self.model.mmproj is not None and self.model.mmproj.exists():
            cmd += ["--mmproj", str(self.model.mmproj)]
        # MTP / NextN speculative decoding: --spec engages it; Gemma 4 also
        # needs a separate draft GGUF (Qwen 3.6 embeds NextN in the trunk).
        if self.mtp:
            cmd += ["--spec"]
            if self.model.mtp_draft is not None:
                cmd += ["--draft-model", str(self.model.mtp_draft)]
        # Tensor parallelism: split the hosted model across `tp` local GPUs.
        if self.tp > 1:
            cmd += [spec.ts_tp_arg, str(self.tp)]
        # MoE CPU offload: the routed experts of the first N layers stay in
        # system RAM and are multiplied on the host. The baseline point emits
        # nothing at all, so a run that does not use this axis launches the
        # exact command line it always did.
        if self.cpu_moe.active:
            cmd += [spec.ts_cpu_moe_arg, self.cpu_moe.layers_arg]
            if self.cpu_moe.threads > 0:
                cmd += [spec.ts_cpu_moe_threads_arg, str(self.cpu_moe.threads)]
        env = os.environ.copy()
        # Some native architectures expose explicit tensor-shard activation
        # separately from the GPU-count CLI argument. Keep it tied to this
        # matrix cell instead of accidentally benchmarking a fixed rank count.
        env.update({key: value.replace("{tp}", str(self.tp)) for key, value in spec.ts_env.items()})
        env.update(config.tp_device_env(self.backend, self.tp))
        if self.model.is_diffusion:
            env["DIFFUSION_STEPS"] = str(self.model.diffusion_steps)
        self._spawn(cmd, cwd=config.TENSORSHARP_SERVER_DLL.parent, env=env)


# ---------------------------------------------------------------------------
# llama.cpp server
# ---------------------------------------------------------------------------
class LlamaCppServer(ServerHandle):
    def __init__(self, model: config.ModelSpec, backend: str, log_path: Path,
                 tp: int = 1, cpu_moe: Optional[config.CpuMoeSpec] = None):
        super().__init__(f"http://127.0.0.1:{config.LLAMA_PORT}",
                         config.LLAMA_PORT, log_path)
        self.model = model
        self.backend = backend
        self.tp = max(1, int(tp or 1))
        self.cpu_moe = cpu_moe or config.CpuMoeSpec()

    def start(self):
        spec = config.BACKENDS[self.backend]
        exe = config.llama_server_exe_for(self.backend)
        # A leftover/hung llama-server (unkillable while stuck in a GPU-driver
        # call) may still own the configured port. llama-server's port is fully
        # under our control, so shift to a free one instead of letting the
        # health checks talk to the zombie's dead listen backlog.
        if _port_open("127.0.0.1", self.port):
            squatter = _pid_listening(self.port)
            new_port = _find_free_port(self.port + 1)
            print(f"    note: port {self.port} is already in use (PID {squatter}); "
                  f"launching llama-server on port {new_port} instead", flush=True)
            self.port = new_port
            self.base_url = f"http://127.0.0.1:{self.port}"
        cmd = [str(exe),
               "-m", str(self.model.gguf),
               "-ngl", str(spec.llama_ngl),
               "--host", "127.0.0.1",
               "--port", str(self.port),
               "-c", str(config.LLAMA_CONTEXT_SIZE)]
        cmd += [str(a) for a in config.LLAMA_EXTRA_ARGS]
        cmd += [str(a) for a in spec.llama_extra_args]
        # Tensor parallelism: the backend's `tp_extra_args` turn it on
        # (`--split-mode tensor` by default — weights and KV split across the
        # devices); the device set itself is pinned through the backend's
        # visible-devices env var below, so exactly `tp` GPUs are used.
        if self.tp > 1:
            cmd += [str(a) for a in spec.llama_tp_extra_args]
        # MoE CPU offload. llama.cpp spells a layer COUNT the same way, but not
        # `all`: it parses `-ncmoe`'s argument as an integer and has a separate
        # switch for every layer (`--cpu-moe`), so the `all` point must send
        # that instead of a value llama-server would refuse to parse. Its
        # host-thread knob is the global `--threads` (see the backend spec),
        # which is why a thread count is only sent when one was asked for.
        if self.cpu_moe.active:
            if self.cpu_moe.layers == config.CPU_MOE_ALL:
                cmd += [spec.llama_cpu_moe_all_arg]
            else:
                cmd += [spec.llama_cpu_moe_arg, self.cpu_moe.layers_arg]
            if self.cpu_moe.threads > 0:
                cmd += [spec.llama_cpu_moe_threads_arg, str(self.cpu_moe.threads)]
        if self.model.mmproj is not None and self.model.mmproj.exists():
            cmd += ["--mmproj", str(self.model.mmproj)]
        env = os.environ.copy()
        env.update(spec.llama_env)
        env.update(config.tp_device_env(self.backend, self.tp))
        self._spawn(cmd, cwd=exe.parent, env=env)


# ---------------------------------------------------------------------------
# vLLM connector (never launched here; connect-only)
# ---------------------------------------------------------------------------
class VllmConnector(ServerHandle):
    def __init__(self, model: config.ModelSpec, backend: str, log_path: Path):
        super().__init__(config.VLLM_BASE_URL, 0, log_path)
        self.model = model
        self.backend = backend
        self._served_name = None

    def start(self):
        # Nothing to launch; the endpoint is external.
        pass

    def wait_ready(self, timeout_s: float) -> bool:
        url = self.base_url.rstrip("/") + "/v1/models"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200:
                return False
            data = r.json()
            served = [m.get("id") for m in data.get("data", []) if m.get("id")]
            self._served_name = served[0] if served else None
            return True
        except requests.RequestException:
            return False

    def stop(self):
        pass


def make_server(engine: str, model: config.ModelSpec, backend: str,
                log_path: Path, max_tokens: int = config.SERVER_MAX_TOKENS,
                mtp: bool = False, tp: int = 1,
                cpu_moe: Optional[config.CpuMoeSpec] = None) -> ServerHandle:
    if engine == "tensorsharp":
        return TensorSharpServer(model, backend, log_path, max_tokens, mtp=mtp, tp=tp,
                                 cpu_moe=cpu_moe)
    if engine == "llamacpp":
        return LlamaCppServer(model, backend, log_path, tp=tp, cpu_moe=cpu_moe)
    if engine == "vllm":
        return VllmConnector(model, backend, log_path)
    raise ValueError(f"unknown engine {engine}")


def served_model_name(engine: str, server: ServerHandle, model: config.ModelSpec) -> str:
    """The model id to send in the request body: whatever the server advertises
    on `/v1/models` when it names one (vLLM's served name, and TensorSharp's
    hosted id — which for a split GGUF is not the shard file name), else the
    GGUF basename, which llama.cpp and TensorSharp both accept."""
    served = getattr(server, "_served_name", None)
    if served:
        return served
    return model.gguf.name
