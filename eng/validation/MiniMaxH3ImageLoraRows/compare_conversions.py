#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Two files of one adapter (diffusers PEFT and its ComfyUI conversion) against each other, from two lora_rows.py
reports of the same rows and activations: NumPy's deltas (independent of TensorSharp) and TensorSharp's.

  python3 -I compare_conversions.py <parity-a.json> <parity-b.json>
"""
import json, sys
import numpy as np
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
worst_ref = worst_ts = 0.0
for ea, eb in zip(a["entries"], b["entries"]):
    assert ea["where"] == eb["where"] and ea["rows"] == eb["rows"]
    ra, rb = np.array(ea["delta_ref"]), np.array(eb["delta_ref"])
    ta, tb = np.array(ea["delta_ts"]), np.array(eb["delta_ts"])
    ref = float(np.abs(ra - rb).max() / np.abs(ra).max())
    ts = float(np.abs(ta - tb).max() / np.abs(ta).max())
    worst_ref, worst_ts = max(worst_ref, ref), max(worst_ts, ts)
    print(f"{ea['where']:30s} NumPy diffusers vs comfy {ref:.3e}; TensorSharp diffusers vs comfy {ts:.3e}")
print(f"worst: NumPy {worst_ref:.3e}, TensorSharp {worst_ts:.3e}")
