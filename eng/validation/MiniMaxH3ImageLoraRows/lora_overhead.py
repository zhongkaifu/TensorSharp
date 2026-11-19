#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""The unmerged LoRA's cost per step, from MiniMaxH3ImageAB runs (stage S7): each process's warm jobs (2..n), every step
of a run with adapters and steps 2..n of a base run (S4's convention: the first step of a picture may upload), medians per
arm across its processes, and the overhead of each arm against the base arm's step. Also the picture's total and the
speed-up of a speed adapter over the base picture.

  python3 -I lora_overhead.py <dir>... [--base-label none]
Labels are <prefix>-<arm>-<n> as the drivers write them (arm: none, pdmd, seedream, pdmd+seedream, ...).
"""
import glob
import json
import os
import statistics
import sys


def arm_of(label):
    parts = label.split("-")
    # <quant>-<arm...>-<n>: the arm may itself hold a '+'; quant and n are the first and last parts.
    return "-".join(parts[1:-1]) if len(parts) >= 3 else label


def main():
    dirs = [a for a in sys.argv[1:] if not a.startswith("--")]
    rows = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "jobs" not in r:
                continue
            arm = arm_of(r.get("label", os.path.basename(f)))
            jobs = [j for j in r["jobs"] if j.get("rc") == 0]
            if len(jobs) < 2:
                continue
            warm = jobs[1:]
            steps, totals = [], []
            for j in warm:
                s = j["step_seconds"]
                steps.extend(s if (j.get("recipe") or {}).get("steps", len(s)) < 6 else s[1:])
                totals.append(j["total_seconds"])
            first = jobs[0]["total_seconds"]
            rows.setdefault(arm, []).append({"file": f, "step": statistics.median(steps), "total": statistics.median(totals),
                                             "first": first, "steps": len(jobs[0]["step_seconds"]),
                                             "load": r.get("load_start", "")})
    if "none" not in rows:
        print("no base arm")
        return
    base = statistics.median(x["step"] for x in rows["none"])
    base_total = statistics.median(x["total"] for x in rows["none"])
    print(f"{'arm':16s} {'procs':>5s} {'step s':>8s} {'[min, max]':>18s} {'overhead':>9s} {'picture s':>10s} {'speed-up':>9s}")
    for arm, xs in sorted(rows.items(), key=lambda kv: (kv[0] != "none", kv[0])):
        st = [x["step"] for x in xs]
        tot = statistics.median(x["total"] for x in xs)
        print(f"{arm:16s} {len(xs):5d} {statistics.median(st):8.3f} [{min(st):7.3f}, {max(st):7.3f}] "
              f"{(statistics.median(st) / base - 1) * 100:8.2f}% {tot:10.2f} {base_total / tot:8.2f}x  steps {xs[0]['steps']}")


if __name__ == "__main__":
    main()
