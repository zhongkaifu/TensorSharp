#!/usr/bin/env python3
"""Summarize MiniMaxH3ImageAB runs: per arm (legacy-*.json / runtime-*.json in a directory) the first job and the warm
jobs' phases, the medians over the arm's processes, the runtime/legacy ratio per phase, and the PNGs bit for bit.

    python3 -I ab_summary.py <dir> [<dir> ...]

A phase's value per process is the job's seconds (text, decode, total) or the median of its steps 2..n (per step); the
ratio is the median over the runtime's processes over the median over the legacy's, with [min, max] spreads.
"""
import json, os, statistics, sys

def load(d):
    arms = {"legacy": [], "runtime": []}
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json") or "-" not in name: continue
        eng = name.split("-")[0]
        if eng not in arms: continue
        with open(os.path.join(d, name)) as f: r = json.load(f)
        r["_file"] = name
        arms[eng].append(r)
    return arms

def phases(job):
    steps = job.get("step_seconds") or []
    per = statistics.median(steps[1:]) if len(steps) > 1 else (steps[0] if steps else float("nan"))
    return {"text": job["text_seconds"], "step": per, "step1": steps[0] if steps else float("nan"),
            "denoise": job["denoise_seconds"], "decode": job["decode_seconds"], "total": job["total_seconds"]}

def fmt(v): return f"{v:.3f}"

def summarize(d):
    arms = load(d)
    print(f"## {d}")
    for eng, runs in arms.items():
        for r in runs:
            jobs = [j for j in r["jobs"] if j.get("rc") == 0]
            print(f"  {r['_file']}: {len(jobs)} job(s), load {r.get('load_start')} -> {r.get('load_end')}, "
                  f"peak device {r.get('peak_device_bytes', 0)/2**30:.2f} GiB, load {r.get('load_seconds', 0):.2f}s")
    which = {"first": lambda js: js[:1], "second": lambda js: js[1:2], "warm": lambda js: js[1:]}
    for label, pick in which.items():
        vals = {}
        for eng, runs in arms.items():
            per = []
            for r in runs:
                js = [j for j in r["jobs"] if j.get("rc") == 0]
                sel = pick(js)
                if not sel: continue
                ph = [phases(j) for j in sel]
                per.append({k: statistics.median([p[k] for p in ph]) for k in ph[0]})
            vals[eng] = per
        if not vals["legacy"] and not vals["runtime"]: continue
        print(f"  {label} job(s):")
        for k in ("text", "step1", "step", "denoise", "decode", "total"):
            row = []
            med = {}
            for eng in ("legacy", "runtime"):
                xs = [p[k] for p in vals[eng]]
                if not xs: row.append(f"{eng} -"); continue
                med[eng] = statistics.median(xs)
                row.append(f"{eng} {fmt(med[eng])} [{fmt(min(xs))}, {fmt(max(xs))}] n={len(xs)}")
            ratio = f"ratio {med['runtime']/med['legacy']:.4f}" if len(med) == 2 and med["legacy"] > 0 else ""
            print(f"    {k:8s} " + " | ".join(row) + f"  {ratio}")
    # PNGs: every (job id, size) across arms
    shas = {}
    for eng, runs in arms.items():
        for r in runs:
            for j in r["jobs"]:
                if j.get("rc") != 0: continue
                shas.setdefault((j["id"], j["width"]), {}).setdefault(eng, set()).add(j["png_sha256"])
    print("  PNGs:")
    for key, by in sorted(shas.items()):
        all_ = set().union(*by.values())
        state = "bit-exact" if len(all_) == 1 and len(by) == 2 else ("one arm" if len(by) == 1 else "DIFFER")
        print(f"    {key[0]} {key[1]}: {state} " + ", ".join(f"{e} {sorted(s)[0][:16]}" + ("" if len(s) == 1 else f" (+{len(s)-1})") for e, s in by.items()))

for d in sys.argv[1:]: summarize(d)
