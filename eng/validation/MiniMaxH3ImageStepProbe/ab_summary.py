#!/usr/bin/env python3
# Summarize MiniMaxH3ImageStepProbe --engine both runs: per run, how many step pairs were bit-exact (or the largest
# difference), and each arm's step seconds (steps 2.. of every round: the first step of a round carries one-time costs
# in the first round only, so it is reported apart), with the median, min, max and the runtime/legacy median ratio.
#
#   python3 ab_summary.py <run-dir-or-result.json> [...]
import json
import os
import statistics
import sys


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "result.json")
    with open(path, encoding="utf-8") as f:
        return path, json.load(f)


def main(paths):
    for p in paths:
        path, r = load(p)
        rows = r.get("comparisons") or []
        if not rows:
            print(f"{path}: no comparisons (engine {r.get('engine')})")
            continue
        exact = sum(1 for c in rows if c["valuesDiffering"] == 0)
        worst = max(rows, key=lambda c: c["maxRelDiff"])
        legacy = [c["legacySeconds"] for c in rows if c["step"] > 1]
        runtime = [c["runtimeSeconds"] for c in rows if c["step"] > 1]
        first = [(c["round"], c["legacySeconds"], c["runtimeSeconds"]) for c in rows if c["step"] == 1]
        ml, mr = statistics.median(legacy), statistics.median(runtime)
        print(f"{path}")
        print(f"  {r['backend']} {os.path.basename(r['dit'])} {r['width']}x{r['height']} rounds {r['rounds']}: "
              f"{exact}/{len(rows)} step pairs bit-exact" +
              ("" if exact == len(rows) else f"; worst max|d| {worst['maxAbsDiff']:.3e} ({worst['maxRelDiff']:.3e} rel), "
                                              f"round {worst['round']} step {worst['step']}"))
        print(f"  steps 2..{r['steps']} (n={len(legacy)} a arm): legacy median {ml:.3f} s [{min(legacy):.3f}, {max(legacy):.3f}], "
              f"runtime median {mr:.3f} s [{min(runtime):.3f}, {max(runtime):.3f}], runtime/legacy {mr / ml:.4f}")
        print("  step 1 (round, legacy s, runtime s): " + ", ".join(f"({a}, {b:.2f}, {c:.2f})" for a, b, c in first))
        print(f"  latent {r['latentSha256']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: ab_summary.py <run-dir-or-result.json> [...]")
    main(sys.argv[1:])
