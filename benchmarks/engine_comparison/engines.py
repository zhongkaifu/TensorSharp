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
"""
from __future__ import annotations

import json
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
    # Extra benchmark axes.
    mtp: bool = False                        # MTP/NextN speculative decoding engaged
    concurrency: int = 1                     # parallel identical requests at this cell
    aggregate_decode_tps: float = 0.0        # system-wide decode tok/s across all parallel seqs
    requests_ok: int = 0                     # successful requests out of `concurrency`

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ---------------------------------------------------------------------------
# Uniform OpenAI chat runner
# ---------------------------------------------------------------------------
def run_openai_chat(base_url: str, model_name: str, messages: list, *,
                    tools: Optional[list] = None,
                    response_format: Optional[dict] = None,
                    max_tokens: int = 128,
                    stream: bool = True,
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
    if tools:
        body["tools"] = tools
    if response_format:
        body["response_format"] = response_format
    if stream:
        body["stream_options"] = {"include_usage": True}

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
        "output_text": text,
        # Absolute monotonic timestamps (shared process clock) so a parallel
        # runner can stitch a system-wide throughput window. No token stream
        # here, so the whole request is the decode window.
        "t_start_abs": t0,
        "t_first_abs": t0,
        "t_last_abs": t_end,
        "t_end_abs": t_end,
    }


def _run_streaming(url: str, body: dict, timeout_s: float) -> dict:
    t_start = time.monotonic()
    t_first = None
    t_last = None
    content_chunks = 0
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_names: list[str] = []
    finish_reason = ""
    usage = {}

    with requests.post(url, json=body, stream=True,
                       timeout=(30, timeout_s)) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
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
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
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
                name = _tc_name(tc)
                if name and name not in tool_names:
                    tool_names.append(name)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    t_end = time.monotonic()
    completion = int(usage.get("completion_tokens", 0) or 0) or content_chunks
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    ttft_ms = ((t_first - t_start) * 1000.0) if t_first else 0.0
    decode_window = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    decode_tps = ((completion - 1) / decode_window) if (decode_window > 0 and completion > 1) else 0.0
    prefill_tps = (prompt / (ttft_ms / 1000.0)) if (ttft_ms > 0 and prompt) else 0.0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "ttft_ms": ttft_ms,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "total_wall_ms": (t_end - t_start) * 1000.0,
        "finish_reason": finish_reason,
        "tool_calls": tool_names,
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
                             timeout_s: float = 1200.0) -> dict:
    """Fire `concurrency` identical chat completions at the same server at once
    and return one aggregated metrics dict (same keys as `run_openai_chat` plus
    `aggregate_decode_tps`, `requests_ok`, `per_request`).

    Per-request metrics are reported as the mean across the successful requests;
    `aggregate_decode_tps` is the *system* decode throughput — total generated
    tokens divided by the wall window during which any sequence was decoding
    (max last-token time − min first-token time, on the shared process clock).
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
                max_tokens=max_tokens, stream=stream, timeout_s=timeout_s)
        except Exception as ex:  # captured per-request; surfaced in aggregate
            errors[i] = ex

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(_worker, range(n)))

    ok = [r for r in results if r is not None]
    if not ok:
        first_err = next((e for e in errors if e is not None), None)
        raise RuntimeError(f"all {n} parallel requests failed: {first_err}")

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


class ServerHandle:
    """Common helpers for a launched OpenAI server process."""

    def __init__(self, base_url: str, port: int, log_path: Path):
        self.base_url = base_url
        self.port = port
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        self._log_fh = None

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

    def wait_ready(self, timeout_s: float) -> bool:
        url = self.base_url.rstrip("/") + "/v1/models"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False  # process exited before becoming ready
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    return True
            except requests.RequestException:
                pass
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
                 max_tokens: int = config.SERVER_MAX_TOKENS, mtp: bool = False):
        super().__init__(f"http://127.0.0.1:{config.TENSORSHARP_PORT}",
                         config.TENSORSHARP_PORT, log_path)
        self.model = model
        self.backend = backend
        self.max_tokens = max_tokens
        self.mtp = mtp

    def start(self):
        ts_backend = config.TENSORSHARP_BACKEND[self.backend]
        cmd = ["dotnet", str(config.TENSORSHARP_SERVER_DLL),
               "--model", str(self.model.gguf),
               "--backend", ts_backend,
               "--max-tokens", str(self.max_tokens)]
        if self.model.mmproj is not None and self.model.mmproj.exists():
            cmd += ["--mmproj", str(self.model.mmproj)]
        # MTP / NextN speculative decoding: --mtp-spec engages it; Gemma 4 also
        # needs a separate draft GGUF (Qwen 3.6 embeds NextN in the trunk).
        if self.mtp:
            cmd += ["--mtp-spec"]
            if self.model.mtp_draft is not None:
                cmd += ["--mtp-draft-model", str(self.model.mtp_draft)]
        import os
        env = os.environ.copy()
        if self.model.is_diffusion:
            env["DIFFUSION_STEPS"] = str(self.model.diffusion_steps)
        self._spawn(cmd, cwd=config.TENSORSHARP_SERVER_DLL.parent, env=env)


# ---------------------------------------------------------------------------
# llama.cpp server
# ---------------------------------------------------------------------------
class LlamaCppServer(ServerHandle):
    def __init__(self, model: config.ModelSpec, backend: str, log_path: Path):
        super().__init__(f"http://127.0.0.1:{config.LLAMA_PORT}",
                         config.LLAMA_PORT, log_path)
        self.model = model
        self.backend = backend

    def start(self):
        ngl = config.LLAMA_NGL[self.backend]
        cmd = [str(config.LLAMA_SERVER_EXE),
               "-m", str(self.model.gguf),
               "-ngl", str(ngl),
               "--host", "127.0.0.1",
               "--port", str(config.LLAMA_PORT),
               "-c", str(config.LLAMA_CONTEXT_SIZE)]
        cmd += [str(a) for a in config.LLAMA_EXTRA_ARGS]
        if self.model.mmproj is not None and self.model.mmproj.exists():
            cmd += ["--mmproj", str(self.model.mmproj)]
        self._spawn(cmd, cwd=config.LLAMA_SERVER_EXE.parent, env=None)


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
                mtp: bool = False) -> ServerHandle:
    if engine == "tensorsharp":
        return TensorSharpServer(model, backend, log_path, max_tokens, mtp=mtp)
    if engine == "llamacpp":
        return LlamaCppServer(model, backend, log_path)
    if engine == "vllm":
        return VllmConnector(model, backend, log_path)
    raise ValueError(f"unknown engine {engine}")


def served_model_name(engine: str, server: ServerHandle, model: config.ModelSpec) -> str:
    """The model id to send in the request body. llama.cpp / TensorSharp accept
    the GGUF basename; vLLM uses whatever it advertises on /v1/models."""
    if engine == "vllm" and getattr(server, "_served_name", None):
        return server._served_name
    return model.gguf.name
