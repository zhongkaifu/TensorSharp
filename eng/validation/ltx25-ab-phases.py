#!/usr/bin/env python3
"""Per-phase wall times of LTX 2.5 A/B probe runs (eng/validation/Ltx25AbProbe), from the phase lines both the retired
pipeline and the diffusion runtime log ([ltx25-timing] text-encode, image-encode, upsample, refine-image-encode, vae-decode,
audio-decode; the retired pipeline's denoise and refine, the runtime's [diffusion] step lines), and "other": the clip's
total less those phases.

  python3 eng/validation/ltx25-ab-phases.py <dir> <label>...

Reads <dir>/<label>.json and <dir>/<label>.log (the probe's stdout).
"""
import json, re, sys
d = sys.argv[1]
table = {}
for lab in sys.argv[2:]:
    log = open(f"{d}/{lab}.log").read()
    runs = {r["case"]: r["generation_ms"] / 1000 for r in json.load(open(f"{d}/{lab}.json"))["runs"]}
    for chunk in re.split(r"\[ab\] \S+ ", log)[1:]:
        case = chunk.split(" ")[0]
        if case not in runs or " mode=" not in chunk.split("\n")[0]: continue
        t = {}
        for name, ms in re.findall(r"\[ltx25-timing\] ([\w-]+): ([\d.]+)ms", chunk):
            t[name] = t.get(name, 0) + float(ms) / 1000
        steps = [float(x) for x in re.findall(r"\[diffusion\] \S+ step \d+/\d+ .*? ([\d.]+)s", chunk)]
        if steps:  # the runtime: per-step lines; the first pass is the steps before the upsample
            n1 = 8 if "upsample" in t else len(steps)
            t["denoise"] = sum(steps[:n1]); t["refine"] = sum(steps[n1:])
        ph = {k: t.get(k, 0) for k in ("text-encode", "image-encode", "denoise", "upsample", "refine-image-encode", "refine", "vae-decode", "audio-decode")}
        ph["other"] = runs[case] - sum(ph.values())
        ph["total"] = runs[case]
        table.setdefault(case, {})[lab] = ph
for case, labs in table.items():
    print(case)
    print("  " + "phase".ljust(20) + "".join(l.rjust(12) for l in labs))
    for k in next(iter(labs.values())):
        print("  " + k.ljust(20) + "".join(f"{labs[l][k]:.2f}".rjust(12) for l in labs))
