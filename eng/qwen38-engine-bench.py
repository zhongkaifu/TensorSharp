#!/usr/bin/env python3
"""Cross-engine scenarios and benchmark for Qwen3.8-27B over OpenAI-compatible servers.

Sends the SAME streamed /v1/chat/completions requests to TensorSharp, llama.cpp,
vLLM, SGLang, velox-inferer or NInfer and appends one JSON line per scenario
repeat: client-side timings from the SSE stream (time to first token, prefill
and decode rates), token counts (prompt, completion, cached), speculative-decoding
counters where the server exposes them, and a QUALITY verdict from an automatic
check. It never starts servers: launch each engine (plain, then with its DFlash2
drafter), then point --url at it.

  python3 eng/qwen38-engine-bench.py --engine vllm --url http://127.0.0.1:8000 \\
      --mode dflash2 --out artifacts/qwen38-engines/vllm-dflash2.jsonl
  python3 eng/qwen38-engine-report.py artifacts/qwen38-engines/*.jsonl

Scenarios (--scenarios, comma list; --list-scenarios prints them):
  short_chat           eight factual/arithmetic questions with checkable answers
  long_prompt_{4k,16k,32k}
                       needle in a haystack of real prose (repository docs by
                       default, --haystack FILE|DIR): a unique code word must
                       be in the answer; reports prefill rate
  json_object          response_format json_object; must parse with typed keys
  json_schema          response_format json_schema; must validate and match
  tool_call_single     one get_weather call with the right typed arguments
  tool_call_parallel   three different tools in one turn, typed arguments
  agentic_loop         read_file/list_files/calculator tools answered locally
                       until a final answer that must be correct (turn cap)
  multi_turn           three-turn conversation over a ~2.5K-token system
                       prompt: per-turn TTFT and cached tokens
  image                four small generated images and one 1344x768 screenshot
                       (eng/minimax-m3-bench-media + generated); needs vision
  long_generation      code, prose and passage-copy generations of
                       --decode-tokens: the decode-rate scenario, run against
                       servers launched with and without the DFlash2 drafter
  parallel_{2,4}       N different concurrent requests: aggregate throughput
                       and per-request correctness (server needs >= N lanes)

Method. Every request is greedy (temperature 0, top_p 1, fixed seed), streamed,
and asks for no thinking: chat_template_kwargs.enable_thinking=false (TensorSharp
reads think=false; velox takes --enable-thinking false at launch). Each
conversation opens with a system line carrying a fixed-width nonce drawn per
invocation, so no engine can reuse a previous run's or repeat's prompt beyond the
template preamble; prefix reuse is measured by multi_turn and agentic_loop. TTFT
is the first streamed content, reasoning or tool-call delta. Decode rate is
(completion_tokens - 1) / (last delta - first delta). completion_tokens comes from
the stream's usage; a server that sends none is counted with --tokenizer (an HF
tokenizer.json) and the source is recorded. Prefill rate is uncached prompt
tokens / TTFT. Speculative counters: llama.cpp timings (draft_n), the Prometheus
/metrics counters of vLLM and SGLang (diffed per request), or --server-log lines
for TensorSharp. Warm-up requests run first and are not recorded.

Each output line is JSON: one {"kind": "run"} header per invocation (engine,
mode, label, server properties, arguments), then {"kind": "case"} per scenario
repeat with "verdict" PASS/FAIL/ERROR, "checks", "metrics" and "requests".
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
import operator
import os
import random
import re
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / "eng" / "minimax-m3-bench-media"
HARNESS_VERSION = "qwen38-engine-bench/1"
ENGINES = ["tensorsharp", "llamacpp", "vllm", "sglang", "velox", "ninfer"]

# Qwen3.8's tokenizer gives about 4.3 characters per token on the repo-docs
# haystack; --chars-per-token overrides it for another corpus. Prompts are built
# from characters so every engine receives byte-identical text; the actual
# prompt_tokens from each server are recorded.
DEFAULT_CHARS_PER_TOKEN = 4.3

SCENARIOS = ["short_chat", "long_prompt_4k", "long_prompt_16k", "long_prompt_32k",
             "json_object", "json_schema", "tool_call_single", "tool_call_parallel", "agentic_loop",
             "multi_turn", "image", "long_generation", "parallel_2", "parallel_4"]


# --------------------------------------------------------------------------- HTTP

def http_json(url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_text(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def http_error_text(err):
    try:
        return err.read().decode("utf-8", "replace")[:2000]
    except Exception:
        return ""


def stream_chat(url, body, timeout):
    """POST one streamed chat completion; return the raw observation (never raises for HTTP errors)."""
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    obs = {"content": [], "reasoning": [], "tool_calls": {}, "usage": None, "timings": None,
           "finish_reason": None, "first": None, "last": None, "n_content": 0, "n_reasoning": 0,
           "n_tool_deltas": 0, "error": None, "http_status": None}
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            obs["http_status"] = resp.status
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if obj.get("error"):
                    obs["error"] = json.dumps(obj["error"])[:2000]
                    continue
                if obj.get("usage"):
                    obs["usage"] = obj["usage"]
                if obj.get("timings"):
                    obs["timings"] = obj["timings"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    now = time.perf_counter()
                    got = False
                    # vLLM and SGLang name the field "reasoning" in some versions.
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        obs["reasoning"].append(reasoning)
                        obs["n_reasoning"] += 1
                        got = True
                    if delta.get("content"):
                        obs["content"].append(delta["content"])
                        obs["n_content"] += 1
                        got = True
                    for tc in delta.get("tool_calls") or []:
                        # Streams differ: some send the name then argument fragments,
                        # others one complete call per chunk. Accumulate by index.
                        idx = tc.get("index", len(obs["tool_calls"]))
                        slot = obs["tool_calls"].setdefault(idx, {"id": None, "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = slot["id"] or tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
                        obs["n_tool_deltas"] += 1
                        got = True
                    if got:
                        obs["first"] = obs["first"] or now
                        obs["last"] = now
                    if ch.get("finish_reason"):
                        obs["finish_reason"] = ch["finish_reason"]
    except urllib.error.HTTPError as err:
        obs["http_status"] = err.code
        obs["error"] = f"HTTP {err.code}: {http_error_text(err)}"
    except Exception as err:  # connection reset, timeout, server crash
        obs["error"] = f"{type(err).__name__}: {err}"
    obs["t0"], obs["t_end"] = t0, time.perf_counter()
    return obs


# --------------------------------------------------------------------------- server counters

SPEC_METRIC = re.compile(r"spec|draft|accept", re.I)


def scrape_metrics(url):
    """Speculative-decoding series of a Prometheus /metrics page as {series: value}; {} when unavailable."""
    if not url:
        return {}
    try:
        text = http_text(url, timeout=10)
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2 or not SPEC_METRIC.search(parts[0]):
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return out


def metric_sum(series, suffix):
    """Sum of the series whose metric name (labels stripped) ends with suffix."""
    vals = [v for k, v in series.items() if k.split("{", 1)[0].endswith(suffix)]
    return sum(vals) if vals else None


def spec_from_metrics(before, after):
    """Drafted/accepted tokens and verify steps between two scrapes (vLLM and SGLang counter names)."""
    if not before or not after:
        return None
    delta = {k: after[k] - before.get(k, 0.0) for k in after if k.endswith("_total") or "_total{" in k}
    drafted = metric_sum(delta, "spec_decode_num_draft_tokens_total")
    accepted = metric_sum(delta, "spec_decode_num_accepted_tokens_total")
    steps = metric_sum(delta, "spec_decode_num_drafts_total")
    if drafted is None:  # SGLang counts verify calls only; tokens per step comes from the completion count
        steps = metric_sum(delta, "spec_verify_calls_total")
    if not drafted and not steps:
        return None
    return {"drafted": drafted, "accepted": accepted, "steps": steps,
            "gauges": {k: v for k, v in after.items() if not ("_total" in k or k.endswith("_count") or k.endswith("_sum"))
                       and "bucket" not in k}}


def server_log_mark(path):
    try:
        return os.path.getsize(path) if path else None
    except OSError:
        return None


def server_log_spec_stats(path, offset):
    """Sum TensorSharp's 'Speculative decoding stats ... drafted=N accepted=M' lines written since offset."""
    if not path or offset is None:
        return None
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    drafted = accepted = lines = 0
    for m in re.finditer(r"Speculative decoding stats for \S+: drafted=(\d+) accepted=(\d+)", text):
        drafted += int(m.group(1))
        accepted += int(m.group(2))
        lines += 1
    return {"drafted": drafted, "accepted": accepted, "requests": lines} if lines else None


# --------------------------------------------------------------------------- client

class Client:
    """Builds engine-specific request bodies and turns raw stream observations into metrics."""

    def __init__(self, args):
        self.args = args
        self.engine = args.engine
        self.url = args.url.rstrip("/")
        self.model = args.model or self.discover_model()
        self.extra = json.loads(args.extra_body) if args.extra_body else {}
        self.metrics_url = args.metrics_url if args.metrics_url is not None else \
            (self.url + "/metrics" if self.engine in ("vllm", "sglang") else "")
        self.tokenizer = load_tokenizer(args.tokenizer)

    def discover_model(self):
        try:
            return http_json(self.url + "/v1/models", timeout=30)["data"][0]["id"]
        except Exception:
            return "qwen3.8-27b"

    def body(self, messages, max_tokens, extra=None):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            # Explicit zeros: a server otherwise fills in the model's recommended non-thinking defaults
            # (presence_penalty 1.5 for Qwen3.8), which change even greedy output.
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "seed": 1234,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.engine in ("vllm", "sglang"):
            body["repetition_penalty"] = 1.0
        if self.engine == "tensorsharp":
            # Caller-declared tools only: no server skills or delegation tools; think=false is its switch.
            body.update({"skills": [], "skills_discovery": False, "multi_agent": False, "think": False,
                         "repeat_penalty": 1.0})
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.engine == "llamacpp":
            body.update({"repeat_penalty": 1.0, "cache_prompt": True})
        if extra:
            body.update(extra)
        body.update(self.extra)
        return body

    def chat(self, messages, max_tokens, extra=None):
        before = scrape_metrics(self.metrics_url)
        mark = server_log_mark(self.args.server_log)
        obs = stream_chat(self.url, self.body(messages, max_tokens, extra), self.args.timeout)
        res = self.finish(obs)
        spec = spec_from_metrics(before, scrape_metrics(self.metrics_url)) if before else None
        if spec is None:
            log = server_log_spec_stats(self.args.server_log, mark)
            if log:
                spec = {"drafted": log["drafted"], "accepted": log["accepted"], "steps": None, "source": "server_log"}
        elif spec:
            spec["source"] = "metrics"
        if spec is None and res.get("draft_n") is not None:
            spec = {"drafted": res["draft_n"], "accepted": res.get("draft_n_accepted"), "steps": None, "source": "timings"}
        res["spec"] = spec
        return res

    def count_tokens(self, text):
        if not text or self.tokenizer is None:
            return None
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def finish(self, obs):
        usage = obs["usage"] or {}
        timings = obs["timings"] or {}
        content = "".join(obs["content"])
        reasoning = "".join(obs["reasoning"])
        tool_calls = []
        for idx in sorted(obs["tool_calls"]):
            tc = obs["tool_calls"][idx]
            try:
                parsed = json.loads(tc["arguments"]) if tc["arguments"].strip() else {}
            except ValueError:
                parsed = None
            tool_calls.append({"id": tc["id"], "name": tc["name"], "arguments": tc["arguments"], "parsed": parsed})
        completion = usage.get("completion_tokens") or timings.get("predicted_n")
        completion_source = "usage" if completion else None
        if not completion:
            text = reasoning + content + "".join(t["name"] + t["arguments"] for t in tool_calls)
            completion = self.count_tokens(text)
            completion_source = "tokenizer" if completion else None
        if not completion:
            n_events = obs["n_content"] + obs["n_reasoning"] + obs["n_tool_deltas"]
            completion, completion_source = (n_events or None), "deltas"
        prompt = usage.get("prompt_tokens") or (timings.get("prompt_n", 0) + timings.get("cache_n", 0) if timings else None)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None and timings:
            cached = timings.get("cache_n")
        first, last = obs["first"], obs["last"]
        ttft = (first - obs["t0"]) if first else None
        decode = ((completion - 1) / (last - first)) if (first and last and last > first and completion and completion > 1) else None
        res = {
            "error": obs["error"], "http_status": obs["http_status"],
            "prompt_tokens": prompt, "completion_tokens": completion, "completion_tokens_source": completion_source,
            "cached_tokens": cached, "reasoning_chars": len(reasoning),
            "ttft_s": ttft, "total_s": obs["t_end"] - obs["t0"], "decode_tok_s": decode,
            "first_offset_s": (first - obs["t0"]) if first else None, "last_offset_s": (last - obs["t0"]) if last else None,
            "finish_reason": obs["finish_reason"],
            "n_content_deltas": obs["n_content"], "n_reasoning_deltas": obs["n_reasoning"],
            "n_tool_deltas": obs["n_tool_deltas"],
            "content": content, "reasoning": reasoning, "tool_calls": tool_calls,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        if prompt and ttft:
            uncached = prompt - (cached or 0)
            res["prefill_tok_s"] = uncached / ttft if uncached > 0 else None
        if timings:
            res["server_timings"] = timings
            res["server_prefill_tok_s"] = timings.get("prompt_per_second")
            res["server_decode_tok_s"] = timings.get("predicted_per_second")
            if "draft_n" in timings:
                res["draft_n"] = timings.get("draft_n")
                res["draft_n_accepted"] = timings.get("draft_n_accepted")
        return res


def load_tokenizer(path):
    if not path:
        return None
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("warning: --tokenizer needs the 'tokenizers' package; completion counts fall back to deltas", file=sys.stderr)
        return None
    return Tokenizer.from_file(str(path))


def trim(res, keep):
    """Copy of a request result with long texts cut for the JSONL (hashes keep the full identity)."""
    out = dict(res)
    for key in ("content", "reasoning"):
        text = out.get(key) or ""
        if len(text) > keep:
            out[key] = text[:keep] + f"... [{len(text) - keep} more chars]"
    return out


# --------------------------------------------------------------------------- helpers

def med(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.median(values) if values else None


def norm(text):
    return (text or "").lower().replace(",", "")


def check(checks, name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def strip_fence(text):
    """Return (json_text, fenced): some engines' grammars wrap response_format output in a ```json fence."""
    m = re.match(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", text or "", re.S)
    return (m.group(1), True) if m else ((text or "").strip(), False)


def validate_schema(value, schema, path="$"):
    """Minimal JSON Schema (draft 2020 subset used here): returns a list of violations."""
    errs = []
    types = schema.get("type")
    if types is not None:
        types = types if isinstance(types, list) else [types]

        def is_type(t):
            return {"object": isinstance(value, dict), "array": isinstance(value, list),
                    "string": isinstance(value, str), "boolean": isinstance(value, bool), "null": value is None,
                    "integer": isinstance(value, int) and not isinstance(value, bool),
                    "number": isinstance(value, (int, float)) and not isinstance(value, bool)}.get(t, False)
        if not any(is_type(t) for t in types):
            return [f"{path}: expected {types}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        errs.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                errs += validate_schema(item, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errs.append(f"{path}: missing required {key!r}")
        for key, item in value.items():
            if key in props:
                errs += validate_schema(item, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected property {key!r}")
    return errs


_SAFE_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
             ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
             ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculate(expression):
    """Arithmetic only (numbers, + - * / ** % //, parentheses); raises ValueError otherwise."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 64:
                raise ValueError("exponent too large")
            return _SAFE_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element {type(node).__name__}")
    text = str(expression).replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    value = ev(ast.parse(text, mode="eval"))
    return int(value) if isinstance(value, float) and value.is_integer() else round(value, 10)


# --------------------------------------------------------------------------- haystack (real prose)

def clean_markdown(text):
    """Prose paragraphs of a markdown file: no code blocks, tables, headings, link targets or HTML;
    paragraphs dominated by identifiers or numbers are dropped."""
    paragraphs, current, in_code = [], [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            current = []
            continue
        if in_code or not stripped or stripped.startswith(("|", "#", "<", ">", "![", "---", "===", "    ")) \
                or line.startswith(("\t", "    ")):
            if current:
                paragraphs.append(" ".join(current))
            current = []
            continue
        current.append(re.sub(r"^([-*+]|\d+\.)\s+", "", stripped))
    if current:
        paragraphs.append(" ".join(current))
    out = []
    for p in paragraphs:
        p = re.sub(r"`([^`]*)`", r"\1", p)                   # inline code keeps its text
        p = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", p)           # images
        p = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", p)        # links -> text
        p = re.sub(r"https?://\S+", "", p)
        p = re.sub(r"(?<!\w)([*_]{1,3})(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", p)  # emphasis, not snake_case
        p = re.sub(r"<[^>]+>", "", p)
        p = re.sub(r"\s+", " ", p).strip()
        words = p.split()
        alpha = sum(1 for w in words if re.fullmatch(r"[A-Za-z][A-Za-z'’-]*[.,;:!?)]?", w))
        if len(words) >= 25 and alpha >= 0.8 * len(words):
            out.append(p)
    return out


def load_haystack(spec):
    """Deterministic list of prose paragraphs: a text file, a directory of .md/.txt, or the repo docs."""
    if spec:
        path = Path(spec)
        files = sorted(p for p in path.rglob("*") if p.suffix in (".md", ".txt")) if path.is_dir() else [path]
        source = str(path)
    else:
        files = sorted(p for p in (ROOT / "docs").rglob("*.md")
                       if "validation" not in p.parts and "zh" not in p.name.lower())
        files += sorted(p for p in ROOT.glob("*.md") if "zh" not in p.name.lower())
        source = "repo docs (docs/**/*.md + top-level *.md, English, prose paragraphs)"
    paragraphs, seen = [], set()
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        chunks = clean_markdown(text) if f.suffix == ".md" else \
            [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 25]
        for p in chunks:
            if p not in seen:
                seen.add(p)
                paragraphs.append(p)
    blob = "\n\n".join(paragraphs)
    return {"paragraphs": paragraphs, "source": source, "files": len(files), "chars": len(blob),
            "words": len(blob.split()), "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()}


def take_prose(haystack, chars, start=0):
    """About `chars` characters of whole paragraphs starting at paragraph `start` (wrapping around)."""
    paras, out, total, i = haystack["paragraphs"], [], 0, start
    if not paras:
        raise RuntimeError("haystack is empty")
    while total < chars:
        p = paras[i % len(paras)]
        out.append(p)
        total += len(p) + 2
        i += 1
        if i - start > 4 * len(paras):
            break
    return out


# --------------------------------------------------------------------------- scenario context

class Ctx:
    def __init__(self, args, client, haystack):
        self.args, self.client, self.haystack = args, client, haystack
        self.seed = args.nonce_seed

    def nonce(self, scenario, repeat, index=0):
        """Fixed-width per-invocation nonce: identical token count across runs, never a cache hit."""
        h = hashlib.sha256(f"{self.seed}:{scenario}:{repeat}:{index}".encode()).hexdigest()
        return f"{int(h[:8], 16) % 900000 + 100000}"

    def messages(self, scenario, repeat, user, index=0, system="You are a helpful assistant."):
        """A system line carrying the nonce, then the user turn (a nonce in the user turn can be
        mistaken for content, e.g. an order number)."""
        return [{"role": "system", "content": f"{system} Session {self.nonce(scenario, repeat, index)}."},
                {"role": "user", "content": user}]

    def rng(self, scenario, repeat):
        return random.Random(f"{self.seed}:{scenario}:{repeat}")

    def chat(self, messages, max_tokens, extra=None):
        return self.client.chat(messages, max_tokens, extra)


def case(scenario, repeat, checks, metrics, requests, keep, notes=None):
    errors = [r.get("error") for r in requests if r.get("error")]
    passed = bool(checks) and all(c["ok"] for c in checks)
    verdict = "PASS" if passed else ("ERROR" if errors and not any(c["ok"] for c in checks) else "FAIL")
    rec = {"kind": "case", "scenario": scenario, "repeat": repeat, "verdict": verdict, "pass": passed,
           "checks": checks, "metrics": metrics, "requests": [trim(r, keep) for r in requests]}
    if errors:
        rec["errors"] = errors[:5]
    if notes:
        rec["notes"] = notes
    return rec


def spec_totals(results):
    specs = [r.get("spec") for r in results if r.get("spec")]
    if not specs:
        return {}
    drafted = sum(s.get("drafted") or 0 for s in specs)
    accepted = sum(s.get("accepted") or 0 for s in specs)
    steps = sum(s.get("steps") or 0 for s in specs)
    stepped_tokens = sum(r.get("completion_tokens") or 0 for r in results if (r.get("spec") or {}).get("steps"))
    out = {"spec_drafted": drafted or None, "spec_accepted": accepted if drafted else None,
           "spec_steps": steps or None, "spec_source": specs[0].get("source")}
    if drafted:
        out["acceptance_rate"] = accepted / drafted
    if steps:
        # Acceptance length as DFlash reports it: emitted tokens per verification step.
        out["tokens_per_step"] = stepped_tokens / steps
    return out


def single_metrics(res):
    keys = ("prompt_tokens", "completion_tokens", "completion_tokens_source", "cached_tokens", "ttft_s",
            "prefill_tok_s", "decode_tok_s", "total_s", "finish_reason", "server_prefill_tok_s", "server_decode_tok_s")
    if res.get("error"):
        return {"failed": True}  # an HTTP error's elapsed time is not a latency
    out = {k: res.get(k) for k in keys}
    out.update(spec_totals([res]))
    return out


def pooled_metrics(results):
    """Medians over the requests of one case; token totals are sums. Failed requests carry no timings."""
    n = len(results)
    results = [r for r in results if not r.get("error")]
    out = {
        "n_requests": n,
        "n_failed_requests": n - len(results),
        "ttft_s": med([r.get("ttft_s") for r in results]),
        "decode_tok_s": med([r.get("decode_tok_s") for r in results]),
        "prefill_tok_s": med([r.get("prefill_tok_s") for r in results]),
        "total_s": med([r.get("total_s") for r in results]),
        "prompt_tokens": med([r.get("prompt_tokens") for r in results]),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in results),
        "cached_tokens": med([r.get("cached_tokens") for r in results]),
    }
    out.update(spec_totals(results))
    return out


def answer_ok(res, pattern, flags=re.I):
    return bool(re.search(pattern, norm(res.get("content")) if flags & re.I else (res.get("content") or ""), flags))


def budget_note(res):
    if res.get("error"):
        return f"error: {res['error'][:200]}"
    if res.get("finish_reason") == "length" and not (res.get("content") or "").strip():
        return "max_tokens ran out before any content (raise --answer-tokens or --decode-tokens)"
    return f"answer={(res.get('content') or '')[:120]!r}"


# --------------------------------------------------------------------------- scenarios

SHORT_QUESTIONS = [
    ("What is 17 multiplied by 23? Reply with just the number.", r"\b391\b", re.I),
    ("What is the capital city of Australia? Reply with just the city name.", r"\bcanberra\b", re.I),
    ("How many days are there in a leap year? Reply with just the number.", r"\b366\b", re.I),
    ("What is the chemical symbol for gold? Reply with just the symbol.", r"\bAu\b", 0),
    ("A train travels 120 kilometres in 1.5 hours. What is its average speed in km/h? Reply with just the number.",
     r"\b80(\.0+)?\b", re.I),
    ("Which planet of our solar system is known as the Red Planet? Reply with one word.", r"\bmars\b", re.I),
    ("What is the square root of 144? Reply with just the number.", r"\b12\b", re.I),
    ("Translate the English word 'cat' into French. Reply with one word.", r"\bchat\b", re.I),
]


def run_short_chat(ctx, r):
    checks, results = [], []
    for i, (question, pattern, flags) in enumerate(SHORT_QUESTIONS):
        res = ctx.chat(ctx.messages("short_chat", r, question, i), ctx.args.answer_tokens)
        results.append(res)
        check(checks, f"q{i}", answer_ok(res, pattern, flags), budget_note(res))
    m = pooled_metrics(results)
    m["n_correct"] = sum(c["ok"] for c in checks)
    m["n_questions"] = len(checks)
    return case("short_chat", r, checks, m, results, ctx.args.keep_text)


CODE_WORDS_A = ["AMBER", "COBALT", "VERMILION", "SAFFRON", "INDIGO", "CERULEAN", "OCHRE", "VIRIDIAN", "MAGENTA", "UMBER"]
CODE_WORDS_B = ["OTTER", "HERON", "LYNX", "MARTEN", "PETREL", "IBEX", "GECKO", "BITTERN", "TAPIR", "WOMBAT"]


def run_long_prompt(ctx, r, k, name):
    rng = ctx.rng(name, r)
    code = f"{rng.choice(CODE_WORDS_A)}-{rng.choice(CODE_WORDS_B)}-{rng.randint(1000, 9999)}"
    depth = (0.5, 0.25, 0.75)[r % 3]
    target_tokens = int(k * 1024 * 0.96)  # leaves room for the template preamble and the question
    paragraphs = take_prose(ctx.haystack, int(target_tokens * ctx.args.chars_per_token), start=97 * k)
    if any(code in p for p in paragraphs):
        raise RuntimeError("needle collides with the haystack")
    needle = f"Note for the archive: the secret code word for the vault is {code}."
    at = max(1, int(len(paragraphs) * depth))
    paragraphs = paragraphs[:at] + [needle] + paragraphs[at:]
    prompt = ("Read the following document carefully.\n\n<document>\n"
              + "\n\n".join(paragraphs)
              + "\n</document>\n\nWhat is the secret code word for the vault mentioned in the document above? "
                "Reply with the code word only.")
    res = ctx.chat(ctx.messages(name, r, prompt), ctx.args.answer_tokens)
    flat = re.sub(r"[^A-Z0-9]", "", (res.get("content") or "").upper())
    checks = []
    check(checks, "needle", re.sub(r"[^A-Z0-9]", "", code) in flat, budget_note(res))
    m = single_metrics(res)
    m.update({"target_tokens": target_tokens, "prompt_chars": len(prompt), "needle_depth": depth})
    return case(name, r, checks, m, [res], ctx.args.keep_text, notes={"code_word": code})


JSON_OBJECT_PROMPT = ("Describe the city of Tokyo as a JSON object with exactly these keys: \"city\" (string), "
                      "\"country\" (string), \"population_millions\" (number) and \"landmarks\" (an array of exactly "
                      "three strings). Output only the JSON object.")

ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string"},
        "items": {"type": "array", "minItems": 1, "items": {
            "type": "object",
            "properties": {"product": {"type": "string"}, "quantity": {"type": "integer", "minimum": 1},
                           "unit_price": {"type": "number"}},
            "required": ["product", "quantity", "unit_price"], "additionalProperties": False}},
        "due_date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
    },
    "required": ["customer", "items", "due_date"],
    "additionalProperties": False,
}
ORDER_TEXT = ("Hi, this is Maria Lopez. Please send me 3 blue notebooks at $4.50 each and 2 black pens at $1.25 each. "
              "I need them by 2026-11-05.")


def parse_json_answer(res, checks):
    text, fenced = strip_fence(res.get("content") or "")
    try:
        value = json.loads(text)
        check(checks, "parses", True, "fenced" if fenced else "raw")
    except ValueError as err:
        check(checks, "parses", False, f"{err}; {budget_note(res)}")
        value = None
    return value, fenced


def run_json_object(ctx, r):
    res = ctx.chat(ctx.messages("json_object", r, JSON_OBJECT_PROMPT), ctx.args.answer_tokens,
                   {"response_format": {"type": "json_object"}})
    checks = []
    value, fenced = parse_json_answer(res, checks)
    if value is not None:
        ok = isinstance(value, dict)
        check(checks, "object", ok)
        if ok:
            check(checks, "city", isinstance(value.get("city"), str) and "tokyo" in value["city"].lower(), repr(value.get("city")))
            check(checks, "country", isinstance(value.get("country"), str) and "japan" in value["country"].lower(), repr(value.get("country")))
            pop = value.get("population_millions")
            check(checks, "population_number", isinstance(pop, (int, float)) and not isinstance(pop, bool), repr(pop))
            lm = value.get("landmarks")
            check(checks, "landmarks", isinstance(lm, list) and len(lm) == 3 and all(isinstance(x, str) for x in lm), repr(lm)[:120])
    m = single_metrics(res)
    m["fenced"] = fenced
    return case("json_object", r, checks, m, [res], ctx.args.keep_text)


def run_json_schema(ctx, r):
    rf = {"type": "json_schema", "json_schema": {"name": "order", "strict": True, "schema": ORDER_SCHEMA}}
    res = ctx.chat(ctx.messages("json_schema", r, f"Extract the order from this message as JSON.\n\n{ORDER_TEXT}"),
                   ctx.args.answer_tokens, {"response_format": rf})
    checks = []
    value, fenced = parse_json_answer(res, checks)
    if value is not None:
        errs = validate_schema(value, ORDER_SCHEMA)
        check(checks, "schema", not errs, "; ".join(errs)[:300])
        if not errs:
            check(checks, "customer", "maria" in value["customer"].lower(), value["customer"])
            got = sorted((it["quantity"], round(float(it["unit_price"]), 2)) for it in value["items"])
            check(checks, "items", got == [(2, 1.25), (3, 4.5)], repr(got))
            check(checks, "due_date", value["due_date"] == "2026-11-05", value["due_date"])
    m = single_metrics(res)
    m["fenced"] = fenced
    return case("json_schema", r, checks, m, [res], ctx.args.keep_text)


def tool(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


WEATHER_TOOL = tool("get_weather", "Get the current weather for a city.",
                    {"city": {"type": "string", "description": "City name"},
                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city", "unit"])
STOCK_TOOL = tool("get_stock_price", "Get the latest stock price for a ticker symbol.",
                  {"symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}}, ["symbol"])
CURRENCY_TOOL = tool("convert_currency", "Convert an amount of money from one currency to another.",
                     {"amount": {"type": "number", "description": "Amount to convert"},
                      "from_currency": {"type": "string", "description": "ISO 4217 code, e.g. USD"},
                      "to_currency": {"type": "string", "description": "ISO 4217 code, e.g. EUR"}},
                     ["amount", "from_currency", "to_currency"])


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def run_tool_call_single(ctx, r):
    msgs = ctx.messages("tool_call_single", r, "What is the weather like in Paris right now? Use Celsius.")
    res = ctx.chat(msgs, ctx.args.answer_tokens, {"tools": [WEATHER_TOOL], "tool_choice": "auto"})
    calls, checks = res["tool_calls"], []
    check(checks, "one_call", len(calls) == 1, f"{len(calls)} call(s); {budget_note(res) if not calls else ''}")
    if calls:
        c = calls[0]
        args = c["parsed"] if isinstance(c["parsed"], dict) else {}
        check(checks, "name", c["name"] == "get_weather", c["name"])
        check(checks, "args_json", isinstance(c["parsed"], dict), c["arguments"][:200])
        check(checks, "city", isinstance(args.get("city"), str) and "paris" in args["city"].lower(), repr(args.get("city")))
        check(checks, "unit", args.get("unit") == "celsius", repr(args.get("unit")))
    m = single_metrics(res)
    m["n_tool_calls"] = len(calls)
    return case("tool_call_single", r, checks, m, [res], ctx.args.keep_text)


def run_tool_call_parallel(ctx, r):
    msgs = ctx.messages("tool_call_parallel", r, "Please do all three of these at once by calling the tools in "
                        "parallel: (1) get the weather in Tokyo in celsius, (2) get the current stock price of NVDA, "
                        "(3) convert 250 US dollars to euros.")
    res = ctx.chat(msgs, ctx.args.answer_tokens, {"tools": [WEATHER_TOOL, STOCK_TOOL, CURRENCY_TOOL],
                                                   "tool_choice": "auto", "parallel_tool_calls": True})
    calls, checks = res["tool_calls"], []
    by_name = {c["name"]: (c["parsed"] if isinstance(c["parsed"], dict) else {}) for c in calls}
    check(checks, "three_calls", len(calls) == 3, f"{[c['name'] for c in calls]}; {budget_note(res) if not calls else ''}")
    check(checks, "names", set(by_name) == {"get_weather", "get_stock_price", "convert_currency"}, repr(sorted(by_name)))
    w, s, cv = by_name.get("get_weather", {}), by_name.get("get_stock_price", {}), by_name.get("convert_currency", {})
    check(checks, "weather_args", isinstance(w.get("city"), str) and "tokyo" in w["city"].lower() and w.get("unit") == "celsius", repr(w))
    check(checks, "stock_args", isinstance(s.get("symbol"), str) and s["symbol"].upper() == "NVDA", repr(s))
    check(checks, "amount_is_number", is_number(cv.get("amount")) and float(cv["amount"]) == 250.0, repr(cv.get("amount")))
    check(checks, "currencies", str(cv.get("from_currency", "")).upper() == "USD"
          and str(cv.get("to_currency", "")).upper() == "EUR", repr(cv))
    m = single_metrics(res)
    m["n_tool_calls"] = len(calls)
    return case("tool_call_parallel", r, checks, m, [res], ctx.args.keep_text)


AGENT_FILES = {
    "/data/README.txt": "Orders are listed in /data/orders.csv. Unit prices in US dollars are in /data/prices.json.",
    "/data/orders.csv": "order_id,product,quantity\n1001,widget,17\n1002,gadget,4\n1003,widget,26\n"
                        "1004,sprocket,11\n1005,widget,9\n",
    "/data/prices.json": json.dumps({"widget": 3.75, "gadget": 12.0, "sprocket": 0.8}),
}
AGENT_TOOLS = [
    tool("list_files", "List the files in a directory.", {"directory": {"type": "string"}}, ["directory"]),
    tool("read_file", "Read a text file and return its contents.", {"path": {"type": "string"}}, ["path"]),
    tool("calculator", "Evaluate an arithmetic expression such as '(2 + 3) * 4.5'.",
         {"expression": {"type": "string"}}, ["expression"]),
]
AGENT_TASK = ("Using the tools, work out the total cost in US dollars of all widget orders. The order file and the "
              "price list are in the /data directory: list it to find them, read both, add up the quantities of every "
              "widget order and multiply the sum by the widget unit price. Use the calculator tool for the arithmetic. "
              "End with the line 'TOTAL: <amount>'.")


def agent_tool_result(name, args):
    if name == "list_files":
        d = str(args.get("directory", "")).rstrip("/") or "/"
        names = sorted(p.rsplit("/", 1)[1] for p in AGENT_FILES if p.rsplit("/", 1)[0] == d)
        return "\n".join(names) if names else f"error: no such directory {d}"
    if name == "read_file":
        path = str(args.get("path", ""))
        path = path if path.startswith("/") else "/data/" + path
        return AGENT_FILES.get(path, f"error: no such file {path}")
    if name == "calculator":
        try:
            return str(calculate(args.get("expression", "")))
        except Exception as err:
            return f"error: {err}"
    return f"error: unknown tool {name}"


def run_agentic_loop(ctx, r):
    msgs = ctx.messages("agentic_loop", r, AGENT_TASK, system="You are a helpful assistant. You can inspect files "
                                                             "and do arithmetic with the tools provided.")
    results, trace, final = [], [], None
    for turn in range(ctx.args.agent_max_turns):
        res = ctx.chat(msgs, ctx.args.answer_tokens, {"tools": AGENT_TOOLS, "tool_choice": "auto"})
        results.append(res)
        if res.get("error"):
            break
        if not res["tool_calls"]:
            final = res
            break
        assistant = {"role": "assistant", "content": res.get("content") or "", "tool_calls": []}
        tool_msgs = []
        for i, c in enumerate(res["tool_calls"]):
            call_id = c["id"] or f"call_{turn}_{i}"
            args = c["parsed"] if isinstance(c["parsed"], dict) else {}
            out = agent_tool_result(c["name"], args) if c["parsed"] is not None else "error: arguments are not valid JSON"
            assistant["tool_calls"].append({"id": call_id, "type": "function",
                                            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}})
            tool_msgs.append({"role": "tool", "tool_call_id": call_id, "content": out})
            trace.append({"turn": turn, "tool": c["name"], "args": args, "result": out[:200]})
        msgs = msgs + [assistant] + tool_msgs
    checks = []
    read_paths = {str(t["args"].get("path", "")) for t in trace if t["tool"] == "read_file"}
    check(checks, "read_orders", any("orders.csv" in p for p in read_paths), repr(sorted(read_paths)))
    check(checks, "read_prices", any("prices.json" in p for p in read_paths), repr(sorted(read_paths)))
    if final is None:
        check(checks, "final_answer", False, "turn cap reached" if results and not results[-1].get("error")
              else budget_note(results[-1]) if results else "no response")
    else:
        check(checks, "final_answer", re.search(r"\b195(\.0+)?\b", norm(final.get("content"))) is not None,
              budget_note(final))
    m = pooled_metrics(results)
    m.update({"turns": len(results), "tool_calls": len(trace),
              "turn_ttft_s": [x.get("ttft_s") for x in results],
              "turn_cached_tokens": [x.get("cached_tokens") for x in results],
              "used_calculator": any(t["tool"] == "calculator" for t in trace),
              "end_to_end_s": sum(x.get("total_s") or 0 for x in results)})
    return case("agentic_loop", r, checks, m, results, ctx.args.keep_text, notes={"trace": trace})


TOPICS = ["how lighthouses guide ships", "why leaves change colour in autumn", "how a bicycle gear system works",
          "the water cycle", "how bees make honey", "how a refrigerator keeps food cold",
          "why the sky looks blue", "how tides are caused by the moon"]


def run_parallel(ctx, r, n, name):
    prompts, expected = [], []
    for i in range(n):
        a, b = 12 + 7 * i + r, 31 + 3 * i
        expected.append(a * b)
        prompts.append(ctx.messages(name, r, f"First compute {a} times {b} and write the line 'PRODUCT: <number>'. "
                                             f"Then explain {TOPICS[i % len(TOPICS)]} in about 200 words.", i))
    results = [None] * n
    barrier = threading.Barrier(n)

    def worker(i):
        barrier.wait()
        # Concurrent requests share the metrics counters: per-request diffs would overlap.
        obs = stream_chat(ctx.client.url, ctx.client.body(prompts[i], ctx.args.decode_tokens), ctx.args.timeout)
        results[i] = ctx.client.finish(obs)

    before = scrape_metrics(ctx.client.metrics_url)
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    spec = spec_from_metrics(before, scrape_metrics(ctx.client.metrics_url)) if before else None
    checks = []
    for i, res in enumerate(results):
        check(checks, f"req{i}", re.search(rf"\b{expected[i]}\b", norm(res.get("content"))) is not None, budget_note(res))
    total = sum(x.get("completion_tokens") or 0 for x in results)
    firsts = [x["first_offset_s"] for x in results if x.get("first_offset_s") is not None]
    lasts = [x["last_offset_s"] for x in results if x.get("last_offset_s") is not None]
    m = {"n": n, "wall_s": wall, "total_completion_tokens": total, "aggregate_tok_s": total / wall if wall > 0 else None,
         "aggregate_decode_tok_s": (total - len(firsts)) / (max(lasts) - min(firsts))
         if firsts and lasts and max(lasts) > min(firsts) else None,
         "per_request_decode_tok_s": [x.get("decode_tok_s") for x in results],
         "decode_tok_s": med([x.get("decode_tok_s") for x in results]),
         "ttft_s": med([x.get("ttft_s") for x in results]),
         "per_request_ttft_s": [x.get("ttft_s") for x in results],
         "n_correct": sum(c["ok"] for c in checks)}
    if spec and spec.get("drafted"):
        m.update({"spec_drafted": spec["drafted"], "spec_accepted": spec["accepted"],
                  "acceptance_rate": (spec["accepted"] or 0) / spec["drafted"]})
    return case(name, r, checks, m, results, ctx.args.keep_text)


KV_FACTS = [
    ("The museum's oldest ship model, the Gannet, was built in 1847.",
     "In what year was the Gannet ship model built? Reply with the year only.", r"\b1847\b"),
    ("The head curator of the museum is named Ilse Marrow.",
     "What is the name of the head curator? Reply with the name only.", r"\bmarrow\b"),
    ("The museum's lighthouse lens room is open only on Thursdays.",
     "On which day of the week is the lens room open? Reply with the day only.", r"\bthursdays?\b"),
]


def run_multi_turn(ctx, r):
    paragraphs = take_prose(ctx.haystack, int(2500 * ctx.args.chars_per_token), start=31 + r)
    for j, (fact, _, _) in enumerate(KV_FACTS):
        paragraphs.insert(int(len(paragraphs) * (j + 1) / (len(KV_FACTS) + 1)), fact)
    system = (f"Session {ctx.nonce('multi_turn', r)}. You are the archivist of the Harbor Museum. Answer questions "
              "about the museum's records below concisely.\n\n<records>\n" + "\n\n".join(paragraphs) + "\n</records>")
    msgs = [{"role": "system", "content": system}]
    results, checks = [], []
    for i, (_, question, pattern) in enumerate(KV_FACTS):
        msgs = msgs + [{"role": "user", "content": question}]
        res = ctx.chat(msgs, ctx.args.answer_tokens)
        results.append(res)
        check(checks, f"turn{i + 1}", re.search(pattern, norm(res.get("content"))) is not None, budget_note(res))
        msgs = msgs + [{"role": "assistant", "content": res.get("content") or ""}]
    t = [x.get("ttft_s") for x in results]
    later = med(t[1:])
    m = {"turn_ttft_s": t, "turn_prompt_tokens": [x.get("prompt_tokens") for x in results],
         "turn_cached_tokens": [x.get("cached_tokens") for x in results],
         "turn1_ttft_s": t[0], "later_ttft_s": later, "ttft_s": med(t),
         "ttft_drop": (t[0] / later) if (t[0] and later) else None,
         "decode_tok_s": med([x.get("decode_tok_s") for x in results]),
         "prompt_tokens": results[0].get("prompt_tokens") if results else None}
    m.update(spec_totals(results))
    return case("multi_turn", r, checks, m, results, ctx.args.keep_text)


def load_generator():
    spec = importlib.util.spec_from_file_location("bench_media_generate", MEDIA_DIR / "generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def screenshot_png():
    """A 1344x768 dashboard-like picture (42 x 24 patches of 32 px: 1008 visual tokens for Qwen3.8):
    grey panels and one white box holding the number the question asks for."""
    gen = load_generator()
    c = gen.Canvas(1344, 768, (236, 238, 242))
    c.rect(0, 0, 1344, 64, (40, 52, 74))
    c.rect(0, 64, 240, 704, (210, 214, 222))
    for i, color in enumerate([(220, 30, 30), (30, 160, 60), (30, 60, 220)]):
        c.rect(300 + i * 340, 120, 300, 200, color)
    c.rect(300, 380, 980, 320, (255, 255, 255))
    c.text("4815", 514, 456, 24, (20, 20, 20))
    return c.png()


def load_media():
    manifest = json.loads((MEDIA_DIR / "manifest.json").read_text(encoding="utf-8"))
    items = []
    for it in manifest["images"]:
        data = (MEDIA_DIR / it["file"]).read_bytes()
        items.append(dict(it, url=f"data:{it['mime']};base64,{base64.b64encode(data).decode('ascii')}", bytes=len(data)))
    data = screenshot_png()
    items.append({"file": "screenshot-1344x768-4815.png", "mime": "image/png", "bytes": len(data),
                  "question": "This screenshot shows a dashboard. What number is written in the large white box? "
                              "Reply with the digits only.",
                  "expect_regex": r"4\s*,?\s*8\s*1\s*5",
                  "url": f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"})
    return items


def run_image(ctx, r):
    checks, results, per = [], [], []
    for i, it in enumerate(load_media()):
        msgs = ctx.messages("image", r, [{"type": "image_url", "image_url": {"url": it["url"]}},
                                          {"type": "text", "text": it["question"]}], i)
        res = ctx.chat(msgs, ctx.args.answer_tokens)
        results.append(res)
        check(checks, it["file"], re.search(it["expect_regex"], norm(res.get("content"))) is not None, budget_note(res))
        per.append({"file": it["file"], "ttft_s": res.get("ttft_s"), "prompt_tokens": res.get("prompt_tokens")})
    m = pooled_metrics(results)
    m["n_correct"] = sum(c["ok"] for c in checks)
    m["n_images"] = len(checks)
    m["per_image"] = per
    m["screenshot_ttft_s"] = per[-1]["ttft_s"] if per else None
    return case("image", r, checks, m, results, ctx.args.keep_text)


GEN_PROMPTS = [
    ("code", "Write a Python module that implements a function is_prime(n) and a function primes_below(n) using a "
             "sieve of Eratosthenes. Include docstrings, type hints and a small unittest test case.",
     r"def\s+is_prime"),
    ("prose", "Write a detailed, multi-paragraph explanation of how a hash table works, including hashing, collisions, "
              "chaining, open addressing and resizing.", r"hash"),
]


def run_long_generation(ctx, r):
    prompts = list(GEN_PROMPTS)
    # Open the passage on a plain-prose paragraph (no digits) so "copy it exactly" is a fair ask.
    paras = ctx.haystack["paragraphs"]
    start = next(i for i in range(211, 211 + len(paras))
                 if not re.search(r"\d", paras[i % len(paras)]) and len(paras[i % len(paras)].split()) >= 40)
    passage = take_prose(ctx.haystack, int(1800 * ctx.args.chars_per_token), start=start)
    first_words = " ".join(passage[0].split()[:8])
    prompts.append(("copy", "Copy the first paragraph of the passage below exactly, word for word, then summarize "
                    "the whole passage in three sentences.\n\n<passage>\n" + "\n\n".join(passage) + "\n</passage>",
                    re.escape(norm(first_words))))
    checks, results, per = [], [], []
    for i, (kind, prompt, pattern) in enumerate(prompts):
        res = ctx.chat(ctx.messages("long_generation", r, prompt, i), ctx.args.decode_tokens)
        results.append(res)
        check(checks, kind, re.search(pattern, norm(res.get("content"))) is not None, budget_note(res))
        sp = spec_totals([res])
        per.append({"kind": kind, "decode_tok_s": res.get("decode_tok_s"), "completion_tokens": res.get("completion_tokens"),
                    "ttft_s": res.get("ttft_s"), "prompt_tokens": res.get("prompt_tokens"),
                    "finish_reason": res.get("finish_reason"), "content_sha256": res.get("content_sha256"),
                    "acceptance_rate": sp.get("acceptance_rate"), "tokens_per_step": sp.get("tokens_per_step")})
    m = pooled_metrics(results)
    m["per_prompt"] = per
    return case("long_generation", r, checks, m, results, ctx.args.keep_text)


def dispatch(name):
    if name == "short_chat":
        return run_short_chat
    if name.startswith("long_prompt_"):
        k = int(name[len("long_prompt_"):].rstrip("k"))
        return lambda ctx, r: run_long_prompt(ctx, r, k, name)
    if name.startswith("parallel_"):
        n = int(name.split("_")[1])
        return lambda ctx, r: run_parallel(ctx, r, n, name)
    return {"json_object": run_json_object, "json_schema": run_json_schema,
            "tool_call_single": run_tool_call_single, "tool_call_parallel": run_tool_call_parallel,
            "agentic_loop": run_agentic_loop, "multi_turn": run_multi_turn, "image": run_image,
            "long_generation": run_long_generation}.get(name)


# --------------------------------------------------------------------------- run

def server_info(client):
    info = {}
    for path in ("/v1/models", "/props", "/health"):
        try:
            data = http_json(client.url + path, timeout=30)
            if path == "/props":
                gen = data.get("default_generation_settings") or {}
                data = {k: data.get(k) for k in ("total_slots", "model_path", "build_info", "modalities")
                        if k in data} | {"n_ctx": gen.get("n_ctx")}
            info[path] = data
        except Exception as err:
            info[path] = f"unavailable: {type(err).__name__}"
    return info


def summary_line(rec):
    m, name = rec["metrics"], rec["scenario"]

    def f(v, d=2):
        return "-" if v is None else f"{v:.{d}f}"
    acc = f" acc {f(m.get('acceptance_rate'))}" if m.get("acceptance_rate") is not None else ""
    acc += f" tok/step {f(m.get('tokens_per_step'))}" if m.get("tokens_per_step") is not None else ""
    if name.startswith("parallel_"):
        core = f"wall {f(m.get('wall_s'), 1)}s aggregate {f(m.get('aggregate_tok_s'))} tok/s correct {m.get('n_correct')}/{m.get('n')}"
    elif name in ("multi_turn",):
        core = (f"ttft {[round(x, 3) if x else None for x in m.get('turn_ttft_s', [])]} "
                f"cached {m.get('turn_cached_tokens')} drop {f(m.get('ttft_drop'))}x")
    elif name == "agentic_loop":
        core = (f"turns {m.get('turns')} tool calls {m.get('tool_calls')} ttft {f(m.get('ttft_s'), 3)}s "
                f"e2e {f(m.get('end_to_end_s'), 1)}s")
    else:
        core = (f"prompt {m.get('prompt_tokens')} ttft {f(m.get('ttft_s'), 3)}s prefill {f(m.get('prefill_tok_s'), 1)} "
                f"decode {f(m.get('decode_tok_s'), 1)} tok/s out {m.get('completion_tokens')}")
    bad = [c["name"] + (f" ({c['detail'][:80]})" if c["detail"] else "") for c in rec["checks"] if not c["ok"]]
    return f"  {name} #{rec['repeat']}: {rec['verdict']} {core}{acc}" + (f" | failed: {'; '.join(bad[:3])}" if bad else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=ENGINES)
    ap.add_argument("--url")
    ap.add_argument("--out", type=Path, help="JSONL file; runs are appended (one header line per invocation)")
    ap.add_argument("--overwrite", action="store_true", help="truncate --out instead of appending")
    ap.add_argument("--mode", default="plain", help="plain or dflash2 (what the server was launched with)")
    ap.add_argument("--label", default="", help="free-form configuration tag recorded in every line")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--model", help="model id to send (default: first id of GET /v1/models)")
    ap.add_argument("--answer-tokens", type=int, default=512, help="max_tokens for answer-checked scenarios")
    ap.add_argument("--decode-tokens", type=int, default=1024, help="max_tokens for long_generation and parallel_N")
    ap.add_argument("--agent-max-turns", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3, help="unrecorded warm-up requests before the scenarios")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--haystack", help="prose source for long prompts: a .txt/.md file or a directory (default: repo docs)")
    ap.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN)
    ap.add_argument("--nonce-seed", type=int, default=None, help="default: random per invocation (recorded)")
    ap.add_argument("--extra-body", help="JSON object merged into every request body (engine-specific fields)")
    ap.add_argument("--metrics-url", help="Prometheus page with speculative counters "
                                          "(default: <url>/metrics for vllm and sglang, none otherwise)")
    ap.add_argument("--tokenizer", help="HF tokenizer.json to count completion tokens when a stream has no usage")
    ap.add_argument("--server-log", help="server stdout log; TensorSharp's speculative stats are read from it")
    ap.add_argument("--keep-text", type=int, default=1500, help="characters of content/reasoning kept per request")
    ap.add_argument("--list-scenarios", action="store_true")
    args = ap.parse_args()
    if args.list_scenarios:
        print("\n".join(SCENARIOS))
        return 0
    if not (args.engine and args.url and args.out):
        ap.error("--engine, --url and --out are required")
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [n for n in names if dispatch(n) is None]
    if unknown:
        ap.error(f"unknown scenario(s): {', '.join(unknown)} (see --list-scenarios)")
    if args.nonce_seed is None:
        args.nonce_seed = random.SystemRandom().randrange(1 << 31)

    client = Client(args)
    haystack = load_haystack(args.haystack)
    ctx = Ctx(args, client, haystack)
    header = {"kind": "run", "harness": HARNESS_VERSION, "engine": args.engine, "mode": args.mode, "url": client.url,
              "label": args.label, "model": client.model, "host": socket.gethostname(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "loadavg": list(os.getloadavg()),
              "scenarios": names, "repeats": args.repeats, "answer_tokens": args.answer_tokens,
              "decode_tokens": args.decode_tokens, "nonce_seed": args.nonce_seed,
              "chars_per_token": args.chars_per_token, "extra_body": client.extra,
              "metrics_url": client.metrics_url, "tokenizer": args.tokenizer,
              "haystack": {k: v for k, v in haystack.items() if k != "paragraphs"},
              "server": server_info(client)}
    run_id = hashlib.sha256(json.dumps(header, sort_keys=True, default=str).encode()).hexdigest()[:12]
    header["run_id"] = run_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = args.out.open("w" if args.overwrite else "a", encoding="utf-8")

    def emit(rec):
        rec["run_id"], rec["engine"], rec["mode"], rec["label"] = run_id, args.engine, args.mode, args.label
        out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        out.flush()

    emit(header)
    print(f"run {run_id}: {args.engine} ({args.mode}) {client.url} model={client.model} label={args.label!r} "
          f"haystack {haystack['words']} words", flush=True)
    for w in range(args.warmup):
        res = client.chat(ctx.messages("warmup", w, "Write two sentences about the sea."), 64)
        print(f"warmup #{w}: ttft {res.get('ttft_s') or 0:.3f}s out {res.get('completion_tokens')} "
              f"{('error ' + res['error'][:300]) if res.get('error') else ''}", flush=True)

    tally = {}
    for name in names:
        print(f"[{name}]", flush=True)
        fn = dispatch(name)
        for r in range(args.repeats):
            try:
                rec = fn(ctx, r)
            except Exception as err:  # a harness or server fault must not stop the other scenarios
                rec = {"kind": "case", "scenario": name, "repeat": r, "verdict": "ERROR", "pass": False,
                       "checks": [{"name": "harness", "ok": False, "detail": f"{type(err).__name__}: {err}"}],
                       "metrics": {}, "requests": [], "errors": [f"{type(err).__name__}: {err}"]}
            rec["loadavg"] = list(os.getloadavg())
            emit(rec)
            tally.setdefault(name, []).append(rec["verdict"])
            print(summary_line(rec), flush=True)
    out.close()
    print("\nverdicts: " + ", ".join(f"{n} {v.count('PASS')}/{len(v)}" for n, v in tally.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
