#!/usr/bin/env python3
"""Per-op time of the graphs TensorSharp ran between two of TS_GGML_NODE_PROFILE's cumulative reports (ggml_ops_core.cpp
graph_compute_profiled prints one every TS_GGML_NODE_PROFILE_EVERY graphs): the last report minus a baseline report, so
the warm-up graphs up to the baseline drop out. The baseline is the last report before Lfm3Probe bench's "INFO warm-up
done" line when the log has one (its start-up warm-up runs a number of steps that grows with MAX_CONTEXT), else the
first report. Per graph, ms and share per op key (TS_GGML_NODE_PROFILE=types or shapes splits the products by weight
type / shape), in the layout of eng/lfm3/bench/llamacpp-op-profile.cpp.

usage: node-profile-diff.py LOG [--tag SUBSTRING] [--first N]
  --first N  subtract the report printed after N graphs (default: the baseline above)
Only the top 24 ops of each report are printed by the profiler, so ops below them in either report are missing.
"""
import argparse
import re
import sys

HEAD = re.compile(r"^\[node-profile\] (.+?): (\d+) graphs x (\d+) nodes, ([0-9.]+) ms/graph")
ROW = re.compile(r"^\s+(\S+)\s+([0-9.]+) ms/graph\s+([0-9.]+)%\s+(\d+) nodes/graph")
WARMED = re.compile(r"^INFO warm-up done")


def reports(path, tag):
    """The log's reports, and how many of them came before the warm-up's end (None: the log has no such line)."""
    out, cur, before_warmed = [], None, None
    for line in open(path, errors="replace"):
        if before_warmed is None and WARMED.match(line):
            before_warmed = len(out)
            cur = None
            continue
        m = HEAD.match(line)
        if m:
            cur = None
            if tag and tag not in m.group(1):
                continue
            cur = {"tag": m.group(1), "graphs": int(m.group(2)), "nodes": int(m.group(3)), "ms": float(m.group(4)), "ops": {}}
            out.append(cur)
            continue
        m = ROW.match(line)
        if m and cur is not None:
            cur["ops"][m.group(1)] = (float(m.group(2)), int(m.group(4)))
    return out, before_warmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--first", type=int, default=0)
    args = ap.parse_args()
    reps, before_warmed = reports(args.log, args.tag)
    if not reps:
        sys.exit("no node-profile reports")
    last = reps[-1]
    if args.first:
        first = next((r for r in reps if r["graphs"] == args.first), None)
    elif before_warmed is not None:
        first = reps[before_warmed - 1] if before_warmed > 0 else None
        if first is None:
            print("node-profile-diff: no report before the warm-up's end, so its graphs are counted", file=sys.stderr)
    else:
        first = reps[0] if len(reps) > 1 else None
    g0 = first["graphs"] if first else 0
    n = last["graphs"] - g0
    if n <= 0:
        sys.exit("no report after the baseline")
    total = (last["ms"] * last["graphs"] - (first["ms"] * g0 if first else 0)) / n
    print(f"{last['tag']}: graphs {g0 + 1}..{last['graphs']} ({n}), {last['nodes']} nodes, {total:.3f} ms/graph profiled")
    rows = []
    for k, (ms, nodes) in last["ops"].items():
        before = first["ops"].get(k, (0.0, 0))[0] * g0 if first else 0.0
        per = (ms * last["graphs"] - before) / n
        rows.append((per, k, nodes))
    for per, k, nodes in sorted(rows, reverse=True):
        print(f"    {k:48s} {per:9.3f} ms/graph {100.0 * per / total:6.1f}% {nodes:6d} nodes/graph")


if __name__ == "__main__":
    sys.exit(main())
