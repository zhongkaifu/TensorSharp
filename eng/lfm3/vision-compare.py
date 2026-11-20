#!/usr/bin/env python3
"""Compare TensorSharp's LFM3 vision dumps (eng/lfm3/Lfm3VisionProbe) with llama.cpp's (eng/lfm3/lfm3-vision-ref.cpp):
per case the chunk layout, the graph inputs (bit-exact count, max |diff|) and every embedding row (cosine per row,
max |abs| error, max relative error = max |diff| / max |ref|, RMS of the difference over the RMS of the reference), and
the encode times. Prints a Markdown table.

usage: eng/lfm3/vision-compare.py <llama.cpp dir> <TensorSharp dir> [case...]
  (case names default to every <name>.json present in both directories; a "ts:ref" pair compares TensorSharp's case
  ts with llama.cpp's case ref, e.g. mp4:frames for TensorSharp's own frames encoded by llama.cpp)
"""
import json
import os
import sys

import numpy as np


def load(d, name):
    meta = json.load(open(os.path.join(d, name + ".json")))
    px = np.fromfile(os.path.join(d, name + ".pixels.f32"), dtype="<f4")
    em = np.fromfile(os.path.join(d, name + ".embd.f32"), dtype="<f4").reshape(-1, 2048)
    return meta, px, em


def layout(meta):
    return [(c["type"], c.get("text"), c.get("rows")) for c in meta["chunks"]]


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    ref_dir, ts_dir, names = argv[1], argv[2], argv[3:]
    if not names:
        names = sorted(n[:-5] for n in os.listdir(ts_dir) if n.endswith(".json") and os.path.exists(os.path.join(ref_dir, n)))
    print("| case | rows | layout | pixels equal | pixels max diff | cos min | cos mean | max abs err | max rel err | rel RMS | TS ms | llama.cpp ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    worst = 1.0
    for spec in names:
        ts_name, ref_name = spec.split(":") if ":" in spec else (spec, spec)
        tm, tp, te = load(ts_dir, ts_name)
        rm, rp, re_ = load(ref_dir, ref_name)
        same_layout = layout(tm) == layout(rm)
        if tp.shape == rp.shape:
            eq = f"{int((tp == rp).sum())}/{tp.size}"
            pdiff = f"{float(np.abs(tp - rp).max()):.3g}" if tp.size else "-"
        else:
            eq, pdiff = f"shape {tp.size} vs {rp.size}", "-"
        if te.shape == re_.shape:
            a, r = te.astype(np.float64), re_.astype(np.float64)
            cos = (a * r).sum(1) / np.sqrt((a * a).sum(1) * (r * r).sum(1))
            d = a - r
            row = (f"{cos.min():.6f} | {cos.mean():.6f} | {np.abs(d).max():.3g} | {np.abs(d).max() / np.abs(r).max():.3g} | "
                   f"{np.sqrt((d * d).mean() / (r * r).mean()):.3g}")
            worst = min(worst, cos.min())
        else:
            row = f"shape {te.shape} vs {re_.shape} | | | | "
        print(f"| {spec} | {te.shape[0]} | {'same' if same_layout else 'DIFFERS'} | {eq} | {pdiff} | {row} | "
              f"{tm.get('encode_ms_total', 0):.1f} | {rm.get('encode_ms_total', 0):.1f} |")
        if not same_layout:
            print(f"<!-- {spec}: TS {layout(tm)} -->")
            print(f"<!-- {spec}: llama.cpp {layout(rm)} -->")
    print(f"\nworst row cosine: {worst:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
