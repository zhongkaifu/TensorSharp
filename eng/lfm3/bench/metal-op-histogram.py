#!/usr/bin/env python3
"""Metal kernel histograms per graph from a GGML_METAL_GRAPH_DEBUG=1 log (TensorSharp or llama.cpp).

TensorSharp: GGML_METAL_GRAPH_DEBUG=1 TS_GGML_LOG_DEBUG=1 <run> 2> ts.log
llama.cpp:   GGML_METAL_GRAPH_DEBUG=1 llama-bench -v ... 2> llama.log   (llama-bench without -v drops debug logs)

Each encoded node is one dispatch; a fused group prints once (its "fuse N ops" line follows). Graphs are split where
the token embedding GET_ROWS starts a forward (--start overrides the regex). Identical graphs (same op sequence) are
grouped, so a decode run prints one histogram with its count.

usage: metal-op-histogram.py LOG [--start REGEX] [--min-nodes N] [--diff OTHER_LOG]
"""
import argparse
import collections
import re
import sys

NODE = re.compile(r"node\[\s*(\d+)\] - (\S+)\s+(.*?)\s*(\(concurrent\)|\(noop\))?\s*$")
FUSE = re.compile(r"fuse (\d+) ops")


def graphs(path, start):
    current, fused = [], []
    with open(path, errors="replace") as f:
        for line in f:
            if "ggml_metal_op_encode_impl" not in line:
                continue
            m = FUSE.search(line)
            if m:
                if current:
                    current[-1] = (current[-1][0], int(m.group(1)), current[-1][2])
                continue
            m = NODE.search(line)
            if not m or m.group(4) == "(noop)":
                continue
            op, name = m.group(2), m.group(3)
            if start.search(op + " " + name) and current:
                yield current
                current = []
            current.append((op, 1, name))
    if current:
        yield current


def label(node):
    op, fuse, _ = node
    return op if fuse == 1 else f"{op}+fuse{fuse}"


def summarize(path, start, min_nodes):
    groups = collections.OrderedDict()
    for g in graphs(path, start):
        if len(g) < min_nodes:
            continue
        key = tuple(label(n) for n in g)
        groups.setdefault(key, [0, g])[0] += 1
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--start", default=r"^GET_ROWS (node_0|inp_embd)")
    ap.add_argument("--min-nodes", type=int, default=20)
    args = ap.parse_args()
    start = re.compile(args.start)
    for key, (count, g) in summarize(args.log, start, args.min_nodes).items():
        hist = collections.Counter(key)
        fused_ops = sum(n[1] for n in g)
        print(f"== graph x{count}: {len(key)} dispatches covering {fused_ops} ops")
        for op, n in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"   {n:5d}  {op}")


if __name__ == "__main__":
    sys.exit(main())
