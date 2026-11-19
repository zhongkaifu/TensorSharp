#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Time to first token of text prompts from the client, two or more OpenAI-compatible servers interleaved.

    ttft.py --server ts=http://127.0.0.1:5738 --server llama=http://127.0.0.1:5732 \
        --lengths 16,128,1024,4096,16384,32768 --reps 5 [--tokenizer-url http://127.0.0.1:5732] [--out ttft.jsonl]

Every request opens with its own nonce line (server, length, repetition), so no prefix cache holds any of its prompt past
the chat template's first tokens; the bulk text is the same filler for every request. Streamed, greedy, max_tokens 1,
thinking off. The servers take turns in a rotating order per repetition (A B, B A, ...) so machine drift falls on all of
them. Per server and length it prints the median TTFT with the min-max spread, and the prefill rate it implies
(prompt tokens / TTFT); llama-server's own timings (prompt_n, prompt_ms) are recorded beside it, which also shows whether
it reused any prefix (prompt_n below the prompt). --tokenizer-url (a llama-server) sizes the prompts in tokens.
"""
import argparse
import http.client
import json
import statistics
import sys
import time
import urllib.parse

WORDS = ("the birch canoe slid on the smooth planks glue the sheet to the dark blue background it is easy to tell the "
         "depth of a well these days a chicken leg is a rare dish rice is often served in round bowls").split()


def post(url, path, body, timeout=1200):
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    return conn, conn.getresponse()


def count_tokens(url, text):
    conn, resp = post(url, "/tokenize", {"content": text})
    data = json.loads(resp.read())
    conn.close()
    return len(data.get("tokens", []))


def filler(words):
    return " ".join(WORDS[(i * 7) % len(WORDS)] for i in range(words))


def size_prompt(tokenizer_url, n_tok):
    """Filler of about n_tok tokens (sized with --tokenizer-url when given, else one word a token)."""
    words = max(1, n_tok)
    if not tokenizer_url:
        return filler(words)
    for _ in range(6):
        have = count_tokens(tokenizer_url, filler(words))
        if abs(have - n_tok) <= max(2, n_tok // 200):
            break
        words = max(1, int(words * n_tok / max(1, have)))
    return filler(words)


def model_id(url):
    """The server's first /v1/models id (TensorSharp refuses a model it does not host)."""
    u = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    conn.request("GET", "/v1/models")
    data = json.loads(conn.getresponse().read()).get("data") or []
    conn.close()
    return data[0]["id"] if data else "default"


MODELS = {}


def ttft(url, text):
    if url not in MODELS:
        MODELS[url] = model_id(url)
    body = {
        "model": MODELS[url], "messages": [{"role": "user", "content": text}], "max_tokens": 1, "stream": True,
        "temperature": 0, "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    conn, resp = post(url, "/v1/chat/completions", body)
    if resp.status != 200:
        raise RuntimeError(f"{url}: HTTP {resp.status}: {resp.read()[:300]!r}")
    first, usage, timings, buf, error = None, None, None, b"", None
    while True:
        chunk = resp.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            try:
                ev = json.loads(line[5:])
            except json.JSONDecodeError:
                continue
            usage = ev.get("usage") or usage
            if ev.get("error"):
                error = ev["error"]
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    first = time.perf_counter()
    end = time.perf_counter()
    conn.close()
    if error:
        raise RuntimeError(f"{url}: {error}")
    # A first token the server does not stream as text (a thinking-off reply opening with <think>) ends the stream: its
    # end is when the token was decided.
    return {"ttft_ms": ((first or end) - t0) * 1000, "streamed": first is not None, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "prompt_n": (timings or {}).get("prompt_n"), "prompt_ms": (timings or {}).get("prompt_ms")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", action="append", required=True, help="NAME=URL (repeat)")
    ap.add_argument("--lengths", default="16,128,1024,4096")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tokenizer-url", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    servers = [tuple(s.split("=", 1)) for s in args.server]
    lengths = [int(x) for x in args.lengths.split(",")]
    texts = {n: size_prompt(args.tokenizer_url, n) for n in lengths}
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    for name, url in servers:
        for w in range(args.warmup):
            ttft(url, f"Warm-up {name} {w}.\n" + texts[lengths[0]])
    rows = []
    for rep in range(args.reps):
        for n in lengths:
            order = servers[rep % len(servers):] + servers[:rep % len(servers)]
            for name, url in order:
                r = ttft(url, f"Request {name} {n} {rep} {time.time_ns()}.\n" + texts[n])
                r.update({"server": name, "length": n, "rep": rep})
                rows.append(r)
                if out:
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                print(json.dumps(r), file=sys.stderr, flush=True)
    print("| prompt | " + " | ".join(f"{name} TTFT ms (median [min-max])" for name, _ in servers)
          + " | " + " | ".join(f"{name} prefill tok/s" for name, _ in servers) + " |")
    print("|---|" + "---|" * (2 * len(servers)))
    for n in lengths:
        cells, rates = [], []
        for name, _ in servers:
            sel = [r for r in rows if r["server"] == name and r["length"] == n]
            v = [r["ttft_ms"] for r in sel]
            pt = statistics.median([r["prompt_tokens"] for r in sel if r["prompt_tokens"]] or [0])
            cells.append(f"{statistics.median(v):.1f} [{min(v):.1f}-{max(v):.1f}]")
            rates.append(f"{pt / (statistics.median(v) / 1000):.0f} ({pt:.0f} tok)" if pt else "-")
        print(f"| {n} | " + " | ".join(cells) + " | " + " | ".join(rates) + " |")


if __name__ == "__main__":
    sys.exit(main())
