#!/usr/bin/env python3
"""Compare two LTX 2.5 A/B probe runs (eng/validation/Ltx25AbProbe): decoded RGB and PCM per case and round
(bitwise equality, max abs difference, PSNR) and the generation times (median per case and label).

  python3 eng/validation/ltx25-ab-compare.py <dir> --a legacy --b runtime [--json out.json]

Reads <dir>/<label>.json and the raw files beside it. Exit status 0 always; the report says what matched.
"""
import argparse, json, math, os, statistics, sys
from array import array

def floats(path):
    a = array('f')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    return a

def diff(x, y):
    if len(x) != len(y):
        return {"same_length": False, "a": len(x), "b": len(y)}
    if x.tobytes() == y.tobytes():
        return {"bitwise": True, "max_abs": 0.0, "psnr_db": math.inf}
    mx, se = 0.0, 0.0
    for p, q in zip(x, y):
        d = abs(p - q); mx = max(mx, d); se += d * d
    mse = se / len(x)
    return {"bitwise": False, "max_abs": mx, "rmse": math.sqrt(mse), "psnr_db": (10 * math.log10(1.0 / mse)) if mse > 0 else math.inf}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    reports = {}
    for label in (args.a, args.b):
        with open(os.path.join(args.dir, label + ".json")) as f:
            reports[label] = json.load(f)
    out = {"a": args.a, "b": args.b, "cases": {}}
    runs = {label: {(r["case"], r["round"]): r for r in reports[label]["runs"]} for label in reports}
    for key in sorted(set(runs[args.a]) & set(runs[args.b])):
        case, rnd = key
        ra, rb = runs[args.a][key], runs[args.b][key]
        entry = out["cases"].setdefault(case, {"rounds": []})
        rgb = diff(floats(os.path.join(args.dir, f"{args.a}.{case}.{rnd}.rgb.f32")),
                   floats(os.path.join(args.dir, f"{args.b}.{case}.{rnd}.rgb.f32"))) if ra["rgb_sha256"] != rb["rgb_sha256"] else {"bitwise": True}
        audio = []
        for ch in range(len(ra["audio_sha256"])):
            if ch < len(rb["audio_sha256"]) and ra["audio_sha256"][ch] == rb["audio_sha256"][ch]:
                audio.append({"bitwise": True}); continue
            audio.append(diff(floats(os.path.join(args.dir, f"{args.a}.{case}.{rnd}.audio-{ch}.f32")),
                              floats(os.path.join(args.dir, f"{args.b}.{case}.{rnd}.audio-{ch}.f32"))))
        entry["rounds"].append({"round": rnd, "rgb": rgb, "audio": audio,
                                "a_s": ra["generation_ms"] / 1000, "b_s": rb["generation_ms"] / 1000})
    for case, entry in out["cases"].items():
        a = [r["a_s"] for r in entry["rounds"]]; b = [r["b_s"] for r in entry["rounds"]]
        entry["median_a_s"] = statistics.median(a); entry["median_b_s"] = statistics.median(b)
        entry["ratio_b_over_a"] = entry["median_b_s"] / entry["median_a_s"]
        entry["all_bitwise"] = all(r["rgb"].get("bitwise") and all(x.get("bitwise") for x in r["audio"]) for r in entry["rounds"])
        print(f"{case:24s} {args.a} {entry['median_a_s']:8.2f}s  {args.b} {entry['median_b_s']:8.2f}s  "
              f"ratio {entry['ratio_b_over_a']:.3f}  bitwise={entry['all_bitwise']}"
              + ("" if entry["all_bitwise"] else f"  rgb psnr={min(r['rgb'].get('psnr_db', math.inf) for r in entry['rounds']):.1f} dB"))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)

if __name__ == "__main__":
    main()
