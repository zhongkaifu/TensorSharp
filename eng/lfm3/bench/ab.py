#!/usr/bin/env python3
"""Interleaved A/B(/C...) runs of benchmark commands that print "RESULT name median min max" lines.

The Mac is shared, so arms alternate (A B A B ..., the first arm rotating each round) and each result is reported as
the median of the rounds' medians with the min-max spread. Each arm is NAME=COMMAND (a shell command line; environment
assignments may lead it).

usage: ab.py [--rounds N] [--log DIR] NAME=COMMAND NAME=COMMAND ...
example:
  ab.py --rounds 3 \\
    'ts=MAX_CONTEXT=20480 eng/lfm3/bench/ts-bench.sh $M/LFM3-1.7B-Q4_K_M.gguf ggml_metal 512,4096 128 0,4096' \\
    'llama=eng/lfm3/bench/llama-bench-result.sh $M/LFM3-1.7B-Q4_K_M.gguf 512,4096 128 0,4096'
"""
import argparse
import collections
import os
import statistics
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--log", default=None, help="directory for each run's raw output")
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()
    arms = []
    for a in args.arms:
        name, _, cmd = a.partition("=")
        arms.append((name, cmd))
    results = collections.defaultdict(lambda: collections.defaultdict(list))   # test -> arm -> [medians]
    if args.log:
        os.makedirs(args.log, exist_ok=True)
    for r in range(args.rounds):
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        for name, cmd in order:
            t0 = time.time()
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/bash")
            out = p.stdout + p.stderr
            if args.log:
                with open(os.path.join(args.log, f"{name}.round{r}.log"), "w") as f:
                    f.write(out)
            got = 0
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "RESULT":
                    results[parts[1]][name].append(float(parts[2]))
                    got += 1
            load = os.getloadavg()[0]
            print(f"round {r} {name}: {got} results in {time.time() - t0:.0f} s (load {load:.1f})"
                  + ("" if p.returncode == 0 else f" EXIT {p.returncode}"), file=sys.stderr)
            if p.returncode != 0 and not got:
                print(out[-2000:], file=sys.stderr)
    names = [n for n, _ in arms]
    print("| test | " + " | ".join(f"{n} (median [min-max] of round medians)" for n in names)
          + (f" | {names[0]}/{names[1]}" if len(names) == 2 else "") + " |")
    print("|---|" + "---|" * (len(names) + (1 if len(names) == 2 else 0)))
    for test in results:
        cells, med = [], {}
        for n in names:
            v = results[test].get(n, [])
            if v:
                med[n] = statistics.median(v)
                cells.append(f"{med[n]:.1f} [{min(v):.1f}-{max(v):.1f}] (n={len(v)})")
            else:
                cells.append("-")
        ratio = ""
        if len(names) == 2:
            ratio = f" | {med[names[0]] / med[names[1]]:.3f}" if names[0] in med and names[1] in med else " | -"
        print(f"| {test} | " + " | ".join(cells) + ratio + " |")


if __name__ == "__main__":
    sys.exit(main())
