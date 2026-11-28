#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Concurrent requests against one running TensorSharp server: each prompt alone, then all of them at once.

Greedy (temperature 0), thinking off, non-streamed OpenAI chat completions (standard library only). Prints one JSON
line: the burst's wall time and completion tokens, its aggregate tok/s, the lone requests' tok/s, how many concurrent
replies equal their lone replies, and any HTTP errors; then the first characters of each reply that differs.

A batched step runs other kernels than a lone one, so near-tie tokens can come out differently: compare the identical
count and the errors against the same model on one device rather than expecting all of them to match. An error (an
HTTP 500, a device out of memory) is the failure this checks for, with the aggregate rate as each placement's score.

  python3 eng/concurrent-parity.py --url http://127.0.0.1:5000 --n 8 --max-tokens 256 --out tp2.json
"""
import argparse
import http.client
import json
import threading
import time
import urllib.parse

PROMPTS = [
    "Write a short poem about the ocean at night.",
    "Explain in three sentences why the sky is blue.",
    "List five prime numbers greater than 100 and explain how you checked one of them.",
    "Translate into French: The library opens at nine and closes at six on weekdays.",
    "Describe how a binary search works, step by step.",
    "Give three tips for writing clean Python code.",
    "What is the difference between a process and a thread?",
    "Summarize the plot of a heist movie you invent, in four sentences.",
    "Name the planets of the solar system in order from the sun.",
    "Write a haiku about autumn leaves.",
    "Explain what a hash table is to a ten year old.",
    "Describe the water cycle briefly.",
    "Give a recipe for a simple tomato soup.",
    "What are the main causes of inflation? Answer briefly.",
    "Write two sentences about the history of the bicycle.",
    "Explain recursion with a short example in pseudocode.",
]


def chat(url, model, prompt, max_tokens, timeout=1800):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.monotonic()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read()
    seconds = time.monotonic() - t0
    if r.status != 200:
        return {"error": f"HTTP {r.status}: {raw[:300]!r}", "seconds": seconds}
    j = json.loads(raw)
    return {"text": j["choices"][0]["message"].get("content") or "",
            "tokens": j.get("usage", {}).get("completion_tokens", 0), "seconds": seconds,
            "finish": j["choices"][0].get("finish_reason")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--n", type=int, default=4, help="concurrent requests (prompts repeat past 16 with a variant tag)")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--out", help="write the summary and every reply here (JSON)")
    ap.add_argument("--skip-lone", action="store_true", help="only the burst (no parity check)")
    a = ap.parse_args()
    u = urllib.parse.urlparse(a.url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
    c.request("GET", "/v1/models")
    model = json.loads(c.getresponse().read())["data"][0]["id"]
    prompts = [PROMPTS[i % len(PROMPTS)] + ("" if i < len(PROMPTS) else f" (variant {i})") for i in range(a.n)]
    chat(a.url, model, "Say hi.", 8)   # warm-up
    lone = [] if a.skip_lone else [chat(a.url, model, p, a.max_tokens) for p in prompts]
    burst = [None] * a.n

    def run(i):
        burst[i] = chat(a.url, model, prompts[i], a.max_tokens)

    t0 = time.monotonic()
    threads = [threading.Thread(target=run, args=(i,)) for i in range(a.n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0
    tokens = sum(r.get("tokens", 0) for r in burst)
    same = sum(1 for i in range(a.n) if lone and "error" not in burst[i] and "error" not in lone[i]
               and burst[i]["text"] == lone[i]["text"])
    lone_rate = sum(r.get("tokens", 0) for r in lone) / sum(r["seconds"] for r in lone) if lone else 0
    summary = {"model": model, "n": a.n, "wall_s": round(wall, 2), "tokens": tokens,
               "aggregate_tok_s": round(tokens / wall, 1), "lone_tok_s": round(lone_rate, 1),
               "identical_to_lone": f"{same}/{a.n}" if lone else "n/a",
               "errors": [r["error"] for r in burst + lone if "error" in r]}
    print(json.dumps(summary))
    for i in range(a.n):
        if lone and "error" not in burst[i] and "error" not in lone[i] and burst[i]["text"] != lone[i]["text"]:
            print(f"  differs [{i}] lone:  {lone[i]['text'][:160]!r}")
            print(f"  differs [{i}] burst: {burst[i]['text'][:160]!r}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "lone": lone, "burst": burst}, f, indent=1)


if __name__ == "__main__":
    main()
