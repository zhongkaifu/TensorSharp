#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""One model on every placement it should run: one GPU, --tp N and --layer-split N, on each backend asked.

For each (backend, placement) the CLI writes a greedy reply to a fixed prompt (temperature 0) and runs its benchmark
(--benchmark --bench-prefill/--bench-decode, best of --runs). The table reports the best prefill and decode tok/s and
whether the reply equals the one-GPU reply of the same backend (and where it first differs): a whole-layer split
should give the same text (the same kernels, one device after another), tensor parallelism sums partials in another
order and may part after a while. Logs go to --out (default artifacts/placement-matrix, ignored by git).

  python3 eng/placement-matrix.py --cli TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Qwen3.5-9B-Q8_0.gguf \
      --backends ggml_cuda,cuda --placements 1,tp2,ls2
"""
import argparse
import os
import re
import subprocess
import sys

PROMPT = ("Explain, in a short paragraph, how a refrigerator keeps food cold, then list three ways to save energy at "
          "home.")
PLACEMENTS = {"1": [], "tp2": ["--tp", "2"], "ls2": ["--layer-split", "2"], "tp4": ["--tp", "4"],
              "ls4": ["--layer-split", "4"], "tp2ls2": ["--tp", "2", "--layer-split", "2"]}


def run(cmd, log, timeout):
    with open(log, "w") as f:
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[placement-matrix] timed out after {timeout} s\n")
            return 124


def reply(log):
    text = open(log, errors="replace").read()
    m = re.search(r'cli\.inference\.complete chars=\d+ preview="(.*)"\s*$', text, re.M)
    return m.group(1) if m else None


def bench(log):
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    p = re.search(r"bestPrefillTps=([\d.]+)", text)
    d = re.search(r"bestDecodeTps=([\d.]+)", text)
    return (p.group(1) if p else "-", d.group(1) if d else "-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", required=True, help="TensorSharp.Cli.dll")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backends", default="ggml_cuda")
    ap.add_argument("--placements", default="1,tp2,ls2", help="comma list of " + ", ".join(PLACEMENTS))
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prefill", type=int, default=512)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per CLI run")
    ap.add_argument("--out", default=os.path.join("artifacts", "placement-matrix"))
    ap.add_argument("extra", nargs="*", help="further CLI arguments (after --)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.model))[0]
    prompt = os.path.join(a.out, "prompt.txt")
    with open(prompt, "w") as f:
        f.write(PROMPT)
    print(f"## {name}")
    for backend in a.backends.split(","):
        reference = None
        for placement in a.placements.split(","):
            base = os.path.join(a.out, f"{name}.{backend}.{placement}")
            common = ["dotnet", a.cli, "--model", a.model, "--backend", backend, *PLACEMENTS[placement], *a.extra]
            gen_rc = run(common + ["--input", prompt, "--max-tokens", str(a.max_tokens), "--temperature", "0"],
                         base + ".gen.log", a.timeout)
            bench_rc = run(common + ["--benchmark", "--bench-prefill", str(a.prefill), "--bench-decode", str(a.decode),
                                     "--bench-runs", str(a.runs)], base + ".bench.log", a.timeout)
            text = reply(base + ".gen.log")
            prefill, decode = bench(base + ".bench.log")
            if text is None:
                verdict = f"no reply (exit {gen_rc}; see {base}.gen.log)"
            elif placement == "1" or reference is None:
                reference = reference or text
                verdict = "reference" if placement == "1" else "no one-GPU reference"
            elif text == reference:
                verdict = "same text"
            else:
                n = next((i for i, (x, y) in enumerate(zip(text, reference)) if x != y), min(len(text), len(reference)))
                verdict = f"differs at char {n}/{len(reference)}"
            if bench_rc != 0:
                verdict += f"; benchmark exit {bench_rc}"
            print(f"  {backend:9s} {placement:7s} prefill {prefill:>9s}  decode {decode:>7s}  {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
