#!/usr/bin/env python3
"""Tabulate serve-vs.sh results: one markdown table per (ISL, OSL), a row per (concurrency, engine).

usage: serve-vs-summary.py DIR [DIR ...]
Each DIR is one serve-vs.sh output (a round); a cell measured in several rounds shows the median of the rounds with the
min-max in brackets. Columns: requests completed, TTFT p50/p99 (ms), TPOT p50/p99 (ms), output tok/s, GPU memory in use
after the cell (MiB, nvidia-smi; vLLM and SGLang preallocate theirs at startup), and the 1-minute load average after the
cell (min-max over the rounds; logs written before serve.log recorded it show '-').
"""
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

cells = defaultdict(list)   # (isl, osl, c, engine) -> [result dict]
mem = defaultdict(list)
load = defaultdict(list)
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "*-i*-o*-c*.json")):
        m = re.match(r"(.+)-i(\d+)-o(\d+)-c(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        e, isl, osl, c = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        with open(f) as fh:
            cells[(isl, osl, c, e)].append(json.load(fh))
    log = os.path.join(d, "serve.log")
    if os.path.exists(log):
        for line in open(log):
            m = re.match(r"(.+)-i(\d+)-o(\d+)-c(\d+) exit \d+ \S+ gpu_mem_mib (\d+)", line.strip())
            if m:
                key = (int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(1))
                mem[key].append(int(m.group(5)))
                lm = re.search(r" load1 ([0-9.]+)", line)
                if lm:
                    load[key].append(float(lm.group(1)))


def fmt(vals, digits=0):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    med = statistics.median(vals)
    s = f"{med:.{digits}f}"
    if len(vals) > 1:
        s += f" [{min(vals):.{digits}f}-{max(vals):.{digits}f}]"
    return s


def span(vals):
    return "-" if not vals else f"{min(vals):.1f}" if len(vals) == 1 else f"{min(vals):.1f}-{max(vals):.1f}"


shapes = sorted({(k[0], k[1]) for k in cells})
order = ["ts-cuda", "ts-ggml_cuda", "ts-ggml_vulkan", "vllm", "sglang", "llama-cuda", "llama-vk"]
for isl, osl in shapes:
    print(f"\n**ISL {isl}, OSL {osl}**\n")
    print("| C | engine | done | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | TPOT p99 ms | output tok/s | GPU MiB | load1 | rounds |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted({k[2] for k in cells if k[:2] == (isl, osl)}):
        engines = sorted({k[3] for k in cells if k[:3] == (isl, osl, c)}, key=lambda e: (order.index(e) if e in order else 99, e))
        for e in engines:
            rs = cells[(isl, osl, c, e)]
            g = lambda key: [r.get(key) for r in rs]
            done = "/".join(str(r.get("completed")) for r in rs)
            print(f"| {c} | {e} | {done} | {fmt(g('p50_ttft_ms'))} | {fmt(g('p99_ttft_ms'))} | {fmt(g('p50_tpot_ms'), 2)} | "
                  f"{fmt(g('p99_tpot_ms'), 2)} | {fmt(g('output_throughput'))} | {fmt(mem.get((isl, osl, c, e), []))} | "
                  f"{span(load.get((isl, osl, c, e), []))} | {len(rs)} |")
