#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""The by-step head contract against diffusers' fused heads (docs/design/diffusion-runtime.md 5.3).

diffusers' MiniMax-H3-Image transformer runs step i with its own head pdd_proj_out[i]; MiniMaxAI/MiniMax-H3-Image ships
those heads as pdd_fused_heads.safetensors (video_out.weight [steps, patch, hidden], video_out.bias [steps, patch]).
TensorSharp's by-step policy picks grid intervals [i, i + 1) and MiniMaxH3ImageSampling.CombineHead builds that head from
the checkpoint's stacked bank: block 0, plus block i for i >= 1 (the dt-weighted mean over one interval is block i
itself). This checks, row by row over the whole bank, that the two are the same heads, from the base checkpoint's own
final_layer.video_out (GGUF or the safetensors release).

  python3 -I pdd_heads.py --model <dit.gguf|.safetensors> --fused pdd_fused_heads.safetensors [--out report.json]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_rows import Gguf, Safetensors  # noqa: E402


def bank(path):
    if path.endswith(".gguf"):
        g = Gguf(path)
        name = next(n for n in g.tensors if n.endswith("final_layer.video_out.weight"))
        bias = next(n for n in g.tensors if n.endswith("final_layer.video_out.bias"))
        rows = g.tensors[name][0][1]
        w = g.rows(name, range(rows))
        b = g.rows(bias, [0]).reshape(-1)
        return w, b, name
    st = Safetensors(path)
    name = next(n for n in st.names() if n.endswith("final_layer.video_out.weight"))
    bias = next(n for n in st.names() if n.endswith("final_layer.video_out.bias"))
    return st.tensor(name), st.tensor(bias).reshape(-1), name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    w, b, name = bank(a.model)                       # [steps * patch, hidden], [steps * patch]
    fused = Safetensors(a.fused)
    fw = fused.tensor("video_out.weight")            # [steps, patch, hidden]
    fb = fused.tensor("video_out.bias")              # [steps, patch]
    steps, patch, hidden = fw.shape
    w = w.reshape(steps, patch, hidden)
    b = b.reshape(steps, patch)
    report = {"model": a.model, "bank": name, "fused": a.fused, "steps": steps, "patch": patch, "hidden": hidden, "heads": []}
    worst = 0.0
    for i in range(steps):
        cw = w[0] + (w[i] if i > 0 else 0)
        cb = b[0] + (b[i] if i > 0 else 0)
        ew = float(np.max(np.abs(cw - fw[i])) / np.max(np.abs(fw[i])))
        eb = float(np.max(np.abs(cb - fb[i])) / np.max(np.abs(fb[i])))
        # And the head a step would get from the bank's block i alone (the policy that would be wrong).
        alone = float(np.max(np.abs(w[i] - fw[i])) / np.max(np.abs(fw[i])))
        worst = max(worst, ew, eb)
        print(f"head {i}: CombineHead[{i}, {i + 1}) vs pdd_proj_out[{i}]: weight max rel {ew:.3e}, bias {eb:.3e} "
              f"(block {i} alone: {alone:.3e})")
        report["heads"].append({"index": i, "weight_rel_max": ew, "bias_rel_max": eb, "block_alone_rel_max": alone})
    report["worst"] = worst
    print(f"worst {worst:.3e}")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
