#!/usr/bin/env python3
"""Merge interleaved runs into one table: ab.py --json files, or ab.py --log directories (their NAME.roundN.log files).

A campaign too long for one quiet window (or one hold of a shared lock) runs as several ab.py calls, each with its own
--json; this tabulates them together, as ab.py does: per test and arm, the median of the rounds' medians, the min-max
spread of the round medians and the round count, plus each arm's machine-busy figures when ab.py --quiet recorded them.

usage: ab-merge.py [--arms A,B,...] [--ratio A/B ...] SOURCE [SOURCE ...]
  SOURCE   an ab.py --json file or an ab.py --log directory
  --arms   column order (default: the order arms first appear)
  --ratio  extra columns, each the ratio of two arms' medians (e.g. ts-g12/l-noblas-fa1-t12)
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys


def load(source, runs):
    if os.path.isdir(source):
        for name in sorted(os.listdir(source)):
            m = re.match(r"(.+)\.round(\d+)\.log$", name)
            if not m:
                continue
            res = {}
            for line in open(os.path.join(source, name), errors="replace"):
                parts = line.split()
                if len(parts) >= 5 and parts[0] == "RESULT":
                    res[parts[1]] = [float(x) for x in parts[2:5]]
            runs.append({"arm": m.group(1), "round": int(m.group(2)), "results": res, "source": source})
    else:
        data = json.load(open(source))
        for r in data["runs"]:
            r["source"] = source
            runs.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None)
    ap.add_argument("--ratio", action="append", default=[])
    ap.add_argument("sources", nargs="+")
    args = ap.parse_args()
    runs = []
    for s in args.sources:
        load(s, runs)
    order = args.arms.split(",") if args.arms else list(dict.fromkeys(r["arm"] for r in runs))
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    tests = []
    for r in runs:
        for test, vals in r["results"].items():
            table[test][r["arm"]].append(vals[0])
            if test not in tests:
                tests.append(test)
    cols = [a for a in order if any(a in table[t] for t in tests)]
    print("| test | " + " | ".join(cols) + "".join(f" | {x}" for x in args.ratio) + " |")
    print("|---|" + "---|" * (len(cols) + len(args.ratio)))
    for t in tests:
        cells, med = [], {}
        for a in cols:
            v = table[t].get(a, [])
            if v:
                med[a] = statistics.median(v)
                cells.append(f"{med[a]:.1f} [{min(v):.1f}-{max(v):.1f}] n={len(v)}")
            else:
                cells.append("-")
        ratios = []
        for x in args.ratio:
            a, _, b = x.partition("/")
            ratios.append(f"{med[a] / med[b]:.2f}" if a in med and b in med and med[b] else "-")
        print(f"| {t} | " + " | ".join(cells) + "".join(f" | {x}" for x in ratios) + " |")
    busy = collections.defaultdict(list)
    for r in runs:
        if isinstance(r.get("busy_before"), (int, float)) and r["busy_before"] == r["busy_before"]:
            busy[r["arm"]].append(r["busy_before"])
    if busy:
        print()
        print("machine busy before each run (% of all cores, one-second top sample): "
              + "; ".join(f"{a} {', '.join(f'{b:.0f}' for b in busy[a])}" for a in cols if a in busy))


if __name__ == "__main__":
    sys.exit(main())
