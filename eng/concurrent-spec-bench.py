#!/usr/bin/env python3
"""Concurrent speculative-decoding throughput against one running TensorSharp server.

Sends waves of N requests started together (N from --concurrency, e.g. 1,2,4) to the
server's Ollama endpoint (/api/chat/ollama, non-streamed, plain greedy with no penalties, thinking off,
num_predict --max-tokens) and appends one JSON line per wave to --out:
  - decode_tok_s_sum: the sum over the wave's requests of eval_count / eval_duration
    (each request's server-side decode rate; issue #391's "total decode tok/s"),
  - wave_tok_s: the wave's completion tokens over its wall time (first send to last
    answer, prompt time included),
  - per-request eval counts and rates, TTFT (prompt_eval_duration),
  - the load average before and after the wave,
  - with --server-log, the "Speculative decoding stats" lines the server logged for the
    wave's requests, parsed: drafted/accepted/verifySteps and the per-phase ms
    (draft/verify/commit/catchUp, totals and per verify step), and per-wave sums.

Prompts are four short code and prose tasks; request i of wave w uses prompt i % 4 behind
the line "(wave w request i)", the same in every run, so no request reuses another's
prompt beyond the chat template and two builds see the same prompts. The server is
started and stopped by the caller (one process per A/B arm), with the agent loop off:
TensorSharp.Server.Host ... --no-multi-agent --no-skills.

  python3 eng/concurrent-spec-bench.py --url http://127.0.0.1:5099 --concurrency 1,2,4 \\
      --repeats 2 --server-log server.log --label main --out artifacts/i391/bench.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string without using datetime.fromisoformat, "
    "with docstrings and a few unit tests.",
    "Write a short essay about why lighthouses were built where they were, and how they changed coastal towns.",
    "Implement a thread-safe LRU cache class in C# with Get, Put and an eviction callback, and explain the locking.",
    "Explain how a bicycle stays upright when moving, in plain language, with two everyday examples.",
]

STATS = re.compile(r"Speculative decoding stats for (?P<id>\S+): (?P<rest>.*)")
NUMBER = re.compile(r"(\w+)=([\d.]+)")


def parse_stats(line_rest):
    """The counts and phase totals of one stats line (older builds log fewer phases: missing ones stay absent), with
    each phase's ms a verify step computed from its total."""
    parts = line_rest.split("|")
    counts = dict(NUMBER.findall(parts[0]))
    out = {"drafted": float(counts.get("drafted", 0)), "accepted": float(counts.get("accepted", 0)),
           "verify_steps": float(counts.get("verifySteps", 0)), "plain_steps": float(counts.get("plainSteps", 0))}
    for part in parts[1:]:
        if part.strip().startswith("phaseMs"):
            for k, v in NUMBER.findall(part):
                out[{"catchUp": "catchup"}.get(k, k) + "_ms"] = float(v)
    steps = out["verify_steps"]
    for phase in ("draft", "verify", "commit", "catchup"):
        if f"{phase}_ms" in out and steps:
            out[f"{phase}_pv"] = out[f"{phase}_ms"] / steps
    return out


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def model_name(url, timeout):
    with urllib.request.urlopen(url + "/api/tags", timeout=timeout) as resp:
        tags = json.loads(resp.read().decode("utf-8"))
    return tags["models"][0]["name"]


def body(model, wave, i, max_tokens):
    text = f"(wave {wave} request {i})\n" + PROMPTS[i % len(PROMPTS)]
    return {"model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": text}],
            # Plain greedy: no repetition penalty either (a server's default 1.1 makes the rows sampled, which a trunk
            # without a batched logits verify cannot serve together).
            "options": {"num_predict": max_tokens, "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0,
                        "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0, "seed": 7}}


def log_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_stats(path, offset):
    if not path:
        return []
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for m in STATS.finditer(text):
        out.append({"id": m.group("id"), **parse_stats(m.group("rest"))})
    return out


def wave(args, model, w, n):
    results = [None] * n
    barrier = threading.Barrier(n)

    def worker(i):
        b = body(model, w, i, args.max_tokens)
        barrier.wait()
        t0 = time.perf_counter()
        r = post(args.url + "/api/chat/ollama", b, args.timeout)
        results[i] = {"start": t0, "end": time.perf_counter(), "eval_count": r.get("eval_count"),
                      "eval_ns": r.get("eval_duration"), "prompt_count": r.get("prompt_eval_count"),
                      "prompt_ns": r.get("prompt_eval_duration"), "chars": len((r.get("message") or {}).get("content") or "")}

    log0 = log_size(args.server_log) if args.server_log else 0
    load0 = os.getloadavg()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    load1 = os.getloadavg()
    time.sleep(args.log_settle)
    stats = read_stats(args.server_log, log0)
    first = min(r["start"] for r in results)
    last = max(r["end"] for r in results)
    rates = [r["eval_count"] / (r["eval_ns"] / 1e9) for r in results if r["eval_count"] and r["eval_ns"]]
    tokens = sum(r["eval_count"] or 0 for r in results)
    rec = {"kind": "wave", "label": args.label, "concurrency": n, "wave": w, "max_tokens": args.max_tokens,
           "decode_tok_s_sum": sum(rates), "decode_tok_s_per_request": rates,
           "wave_tok_s": tokens / (last - first), "wave_wall_s": last - first, "tokens": tokens,
           "eval_counts": [r["eval_count"] for r in results],
           "ttft_ms": [r["prompt_ns"] / 1e6 if r["prompt_ns"] else None for r in results],
           "loadavg_before": load0, "loadavg_after": load1, "spec": stats}
    if stats:
        steps = sum(s["verify_steps"] for s in stats)
        rec["spec_sum"] = {
            "drafted": sum(s["drafted"] for s in stats), "accepted": sum(s["accepted"] for s in stats),
            "verify_steps": steps,
            "accepted_per_verify_step": sum(s["accepted"] for s in stats) / steps if steps else None,
            **{f"{p}_ms_per_verify_step": sum(s.get(f"{p}_ms", 0) for s in stats) / steps if steps else None
               for p in ("draft", "verify", "commit", "catchup")},
        }
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--concurrency", default="1,2,4")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=2, help="waves per concurrency")
    ap.add_argument("--warmup", type=int, default=1, help="unrecorded single requests first")
    ap.add_argument("--server-log", help="the server's log file, for its speculative stats lines")
    ap.add_argument("--log-settle", type=float, default=1.0, help="seconds to wait for the stats lines after a wave")
    ap.add_argument("--label", default="")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = model_name(args.url, args.timeout)
    for i in range(args.warmup):
        post(args.url + "/api/chat/ollama", body(model, -1 - i, 0, 64), args.timeout)
    levels = [int(c) for c in args.concurrency.split(",")]
    w = 0
    with open(args.out, "a") as out:
        out.write(json.dumps({"kind": "run", "label": args.label, "url": args.url, "model": model,
                              "concurrency": levels, "max_tokens": args.max_tokens, "repeats": args.repeats,
                              "time": time.time()}) + "\n")
        for r in range(args.repeats):
            for n in levels:
                rec = wave(args, model, w, n)
                w += 1
                out.write(json.dumps(rec) + "\n")
                out.flush()
                s = rec.get("spec_sum") or {}
                print(f"[{args.label}] C={n} wave {w}: decode {rec['decode_tok_s_sum']:.1f} tok/s (sum), wave "
                      f"{rec['wave_tok_s']:.1f} tok/s, tokens {rec['eval_counts']}, load {rec['loadavg_before'][0]:.2f}; "
                      f"draft {s.get('draft_ms_per_verify_step')} verify {s.get('verify_ms_per_verify_step')} "
                      f"commit {s.get('commit_ms_per_verify_step')} catchUp {s.get('catchup_ms_per_verify_step')} "
                      f"acc/step {s.get('accepted_per_verify_step')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
