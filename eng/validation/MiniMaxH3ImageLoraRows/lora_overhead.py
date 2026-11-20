#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""The unmerged LoRA's cost per step, from MiniMaxH3ImageAB runs (stage S7): the median, over every picture of each arm's
processes (the first, cold one included), of every step but each picture's first (that step can include uploads: on
the A40 a bf16 picture re-uploads its weights there), and the overhead of each arm against the base arm. Also the
warm pictures' total (jobs 2..n) and the speed-up of a speed adapter over the base picture.

  python3 -I lora_overhead.py <dir>...
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
            jobs = [j for j in r["jobs"] if j.get("rc", 0) == 0 and len(j.get("step_seconds", [])) > 1]
            if not jobs:
                continue
            steps = [t for j in jobs for t in j["step_seconds"][1:]]
            warm = jobs[1:] or jobs
            rows.setdefault(arm, []).append({"file": f, "steps_seen": steps, "total": statistics.median(j["total_seconds"] for j in warm),
                                             "first": jobs[0]["total_seconds"], "steps": len(jobs[0]["step_seconds"])})
    if "none" not in rows:
        print("no base arm")
        return
    def step(xs):
        return statistics.median(t for x in xs for t in x["steps_seen"])
    base = step(rows["none"])
    base_total = statistics.median(x["total"] for x in rows["none"])
    print(f"{'arm':16s} {'procs':>5s} {'n':>4s} {'step s':>8s} {'overhead':>9s} {'picture s':>10s} {'speed-up':>9s}")
    for arm, xs in sorted(rows.items(), key=lambda kv: (kv[0] != "none", kv[0])):
        tot = statistics.median(x["total"] for x in xs)
        n = sum(len(x["steps_seen"]) for x in xs)
        print(f"{arm:16s} {len(xs):5d} {n:4d} {step(xs):8.3f} {(step(xs) / base - 1) * 100:8.2f}% {tot:10.2f} "
              f"{base_total / tot:8.2f}x  steps {xs[0]['steps']}")


if __name__ == "__main__":
    main()
