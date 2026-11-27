#!/usr/bin/env python3
"""Per-phase wall times of LTX 2.5 A/B probe runs (eng/validation/Ltx25AbProbe), from each run's progress events:
encodes (text and images, in either order), the first step (with the denoiser upload), the rest of each pass, the
decodes, the total and the process's peak working set.

  python3 eng/validation/ltx25-ab-phases.py <dir> <label>...
"""
import json, sys
d = sys.argv[1]
rows = {}
for lab in sys.argv[2:]:
    for r in json.load(open(f"{d}/{lab}.json"))["runs"]:
        ev = r["events"]
        t = lambda pred: next((e["elapsed"] for e in ev if pred(e)), None)
        steps = [e for e in ev if e["phase"] == "denoise" and e["detail"].startswith("step ")]
        starts = [e for e in ev if e["phase"] == "denoise" and not e["detail"].startswith("step ")]
        st = [s["elapsed"] for s in steps]
        first_pass_start = starts[0]["elapsed"]
        # Text and image encodes, whichever order the pipeline ran them in.
        enc0 = min(e["elapsed"] for e in ev if e["phase"] in ("text-encode", "image-encode", "reference-encode"))
        ph = {"encodes": first_pass_start - enc0}
        ph["step1"] = st[0] - first_pass_start
        if len(starts) > 1:
            k = next(i for i, s in enumerate(steps) if s["stage"] != steps[0]["stage"])
            ph["pass1 steps2+"] = st[k - 1] - st[0]
            ph["upscale"] = starts[1]["elapsed"] - st[k - 1]
            ph["pass2 step1"] = st[k] - starts[1]["elapsed"]
            ph["pass2 rest"] = st[-1] - st[k]
        else:
            ph["steps2+"] = st[-1] - st[0]
        vd = t(lambda e: e["phase"] == "vae-decode"); ad = t(lambda e: e["phase"] == "audio-decode"); dn = t(lambda e: e["phase"] == "done")
        ph["video-decode"] = (ad or dn) - vd
        if ad: ph["audio-decode"] = dn - ad
        ph["total"] = r["generation_ms"] / 1000
        ph["peakWS GB"] = r["process_peak_working_set"] / 1e9
        rows.setdefault(r["case"], {})[lab] = ph
for case, labs in rows.items():
    print(case)
    keys = list(next(iter(labs.values())).keys())
    print("  " + "phase".ljust(16) + "".join(l.rjust(13) for l in labs))
    for k in keys:
        print("  " + k.ljust(16) + "".join((f"{labs[l].get(k, float('nan')):.2f}").rjust(13) for l in labs))
