#!/usr/bin/env python3
"""Summarize eng/concurrent-spec-bench.py results: one row per (label, concurrency) over every wave of every arm.

  python3 eng/concurrent-spec-report.py artifacts/i391/ab/c1/mlx/bench.jsonl [--base main]

Columns: waves, median wave tok/s (completion tokens over the wave's wall time, prompts included), median sum of the
requests' decode rates, the per-request phase ms a verify step (draft, verify, commit, catch-up: each request's phase total
over its verify steps, averaged over the wave's requests), accepted tokens a verify step, and the load average before the
waves (min-max). With --base, each row's median wave tok/s against that label's at the same concurrency.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import statistics
import sys
from collections import defaultdict


def bench_module():
    spec = importlib.util.spec_from_file_location("bench", os.path.join(os.path.dirname(__file__), "concurrent-spec-bench.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reparse(records, log_dir):
    """Waves whose stats the bench could not parse (an older build's line): read again from the arm's server log, the
    arm's i-th log of its label holding one warm-up line and then each wave's lines in order."""
    bench = bench_module()
    logs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(log_dir, "server-*.log"))):
        label = os.path.basename(path)[len("server-"):].rsplit("-", 1)[0]
        logs[label].append(path)
    seen = defaultdict(int)
    lines = []
    for r in records:
        if r.get("kind") == "run":
            label = r["label"]
            paths = logs.get(label, [])
            path = paths[seen[label]] if seen[label] < len(paths) else None
            seen[label] += 1
            lines = []
            if path:
                for line in open(path, errors="replace"):
                    m = bench.STATS.search(line)
                    if m:
                        lines.append({"id": m.group("id"), **bench.parse_stats(m.group("rest"))})
                lines = lines[1:]   # the warm-up request
            continue
        if r.get("kind") != "wave":
            continue
        taken, lines = lines[:r["concurrency"]], lines[r["concurrency"]:]
        if not r.get("spec") and taken:
            r["spec"] = taken
            steps = sum(s["verify_steps"] for s in taken)
            r["spec_sum"] = {"accepted_per_verify_step": sum(s["accepted"] for s in taken) / steps if steps else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--base", help="label the others are compared with")
    ap.add_argument("--strip", default="", help="suffix removed from labels (e.g. -cuda)")
    ap.add_argument("--logs", help="the arms' server logs, to read stats the bench could not parse")
    args = ap.parse_args()
    rows = defaultdict(list)
    for path in args.files:
        records = [json.loads(line) for line in open(path)]
        if args.logs:
            reparse(records, args.logs)
        for r in records:
            if r.get("kind") != "wave":
                continue
            label = r["label"][:-len(args.strip)] if args.strip and r["label"].endswith(args.strip) else r["label"]
            rows[(label, r["concurrency"])].append(r)
    med = statistics.median
    summary = {}
    print("| build | C | waves | wave tok/s | decode sum tok/s | draft | verify | commit | catchUp | acc/step | load |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for (label, c) in sorted(rows, key=lambda k: (k[1], k[0])):
        waves = rows[(label, c)]
        wave = med(w["wave_tok_s"] for w in waves)
        summary[(label, c)] = wave

        def phase(name):
            vals = [s[name] for w in waves for s in w.get("spec", []) if s.get("verify_steps") and name in s]
            return med(vals) if vals else float("nan")

        acc = [w["spec_sum"]["accepted_per_verify_step"] for w in waves if (w.get("spec_sum") or {}).get("accepted_per_verify_step") is not None]
        loads = [w["loadavg_before"][0] for w in waves]
        print(f"| {label} | {c} | {len(waves)} | {wave:.1f} | {med(w['decode_tok_s_sum'] for w in waves):.1f} | "
              f"{phase('draft_pv'):.1f} | {phase('verify_pv'):.1f} | {phase('commit_pv'):.1f} | {phase('catchup_pv'):.1f} | "
              f"{med(acc) if acc else float('nan'):.2f} | {min(loads):.1f}-{max(loads):.1f} |")
    # Greedy speculation is lossless: every build gives each prompt the same completion (where the bench recorded digests).
    by_prompt = defaultdict(lambda: defaultdict(set))
    for (label, c), waves in rows.items():
        for w in waves:
            for i, d in enumerate(w.get("digests") or []):
                by_prompt[c][(w["wave"], i)].add(d)
    for c in sorted(by_prompt):
        prompts = by_prompt[c]
        same = sum(1 for ds in prompts.values() if len(ds) == 1)
        print(f"\nC={c}: {same} of {len(prompts)} prompts completed identically by every build")
    if args.base:
        print()
        print(f"| build | C | wave tok/s vs {args.base} |")
        print("|---|---:|---:|")
        for (label, c), v in sorted(summary.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            b = summary.get((args.base, c))
            if label != args.base and b:
                print(f"| {label} | {c} | x{v / b:.3f} |")


if __name__ == "__main__":
    sys.exit(main())
