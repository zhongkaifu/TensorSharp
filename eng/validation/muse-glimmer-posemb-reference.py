#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""
The Muse-Glimmer (and Muse Spark) vision tower's learned position grid, resampled to an image's patch grid by the
reference implementation, as a test fixture for TensorSharp's port (InferenceWeb.Tests MuseGlimmerVisionPositionGridTests).

The reference is HF transformers' MuseGlimmerVisionPatchEmbedder (src/transformers/models/muse_glimmer/
modular_muse_glimmer.py at main 7394e56, PR huggingface/transformers#50847, merged 2026-11-18):

    # Patch (h, w) reads the position table at (w, h): ref code passes
    # `meshgrid(pos_h, pos_w, indexing="xy")` to `grid_sample` which takes first coord as width.
    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        grid_thw[:, [0, 2, 1]], num_grid_per_side=..., mode="bilinear", align_corners=False,
        spatial_merge_size=1, padding="zeros")

Muse Spark's embedder subclasses it with the same call. This script imports transformers' own helper
(transformers.vision_utils.get_vision_interpolation_indices_and_weights) and calls it that way, so the fixture is the
reference's arithmetic, not a re-derivation. It also emulates llama.cpp's graph (tools/mtmd/models/muse-glimmer.cpp at
d3c3e242c, PR ggml-org/llama.cpp#32433: reshape, permute(2, 0, 1), ggml_interpolate(H, W) bilinear, permute(1, 2, 0)) with
ggml's border-clamped bilinear (ggml-cpu ops.cpp GGML_SCALE_MODE_BILINEAR), and the pre-#32433 order, and prints how
far each is from the reference.

Usage: muse-glimmer-posemb-reference.py <out.json>   (a python with torch and transformers >= 5.22; run with -I)
"""
import json
import math
import sys

import numpy as np
import torch
import transformers
from transformers.vision_utils import get_vision_interpolation_indices_and_weights

# Small synthetic tables (side 4: windows of 4 x 4 patches in the synthetic towers) and the real side (32) on the
# geometries of the vision gate's images; grids are (patches across, patches down).
CASES = [
    (4, [(4, 4), (3, 5), (5, 3), (2, 8), (8, 2), (6, 4), (4, 6), (7, 9), (12, 12), (9, 3), (16, 6)]),
    (32, [(32, 16), (16, 32), (40, 14), (14, 40)]),
]
HIDDEN = 2


def table(side, hidden):
    i = np.arange(side * side * hidden, dtype=np.float64)
    return (np.sin(0.37 * i) + 0.25 * np.cos(1.3 * i)).astype(np.float32).reshape(side * side, hidden)


def reference(tab, side, gw, gh):
    """transformers main: the raster-order patch rows' position embeddings [gh * gw, hidden]."""
    grid_thw = torch.tensor([[1, gh, gw]], dtype=torch.long)
    idx, w = get_vision_interpolation_indices_and_weights(
        grid_thw[:, [0, 2, 1]], num_grid_per_side=side, mode="bilinear", align_corners=False,
        spatial_merge_size=1, padding="zeros")
    t = torch.from_numpy(tab)
    return (t[idx] * w[:, :, None].float()).sum(1).numpy()


def ggml_bilinear(src, ne0_out, ne1_out):
    """ggml_interpolate(..., GGML_SCALE_MODE_BILINEAR) of src[ne1][ne0] (ne0 fastest), ggml-cpu's arithmetic."""
    ne1_in, ne0_in = src.shape[0], src.shape[1]
    sf0, sf1 = ne0_out / ne0_in, ne1_out / ne1_in
    out = np.zeros((ne1_out, ne0_out) + src.shape[2:], dtype=np.float32)
    for i1 in range(ne1_out):
        y = (i1 + 0.5) / sf1 - 0.5
        y0 = math.floor(y)
        y1 = y0 + 1
        y0 = max(0, min(y0, ne1_in - 1))
        y1 = max(0, min(y1, ne1_in - 1))
        dy = max(0.0, min(y - y0, 1.0))
        for i0 in range(ne0_out):
            x = (i0 + 0.5) / sf0 - 0.5
            x0 = math.floor(x)
            x1 = x0 + 1
            x0 = max(0, min(x0, ne0_in - 1))
            x1 = max(0, min(x1, ne0_in - 1))
            dx = max(0.0, min(x - x0, 1.0))
            out[i1, i0] = (src[y0, x0] * (1 - dx) * (1 - dy) + src[y0, x1] * dx * (1 - dy)
                           + src[y1, x0] * (1 - dx) * dy + src[y1, x1] * dx * dy)
    return out


def llamacpp_new(tab, side, gw, gh):
    """d3c3e242c: pos [C, side(a), side(b)] -> permute(2,0,1) [a, b, C] -> interpolate to [gh, gw, C] -> permute(1,2,0)
    [C, gh, gw] -> cont_2d: token t = j * gh + i (i over the gh axis, j over the gw axis)."""
    # ggml [a, b, C] (ne0 = a, the table's column; ne1 = b, its row) is numpy [b][a][C]: the table itself. Interpolate
    # ne0 (a) to gh and ne1 (b) to gw.
    grid = tab.reshape(side, side, -1)
    res = ggml_bilinear(grid, gh, gw)             # [j over gw][i over gh][C]
    return res.reshape(gw * gh, -1)               # t = j * gh + i


def llamacpp_old(tab, side, gw, gh):
    """Before #32433 (resize_position_embeddings): table [C, x, y] interpolated to [gw, gh], token t = y * gw + x."""
    grid = tab.reshape(side, side, -1)            # [y][x][C]
    res = ggml_bilinear(grid, gw, gh)             # [y over gh][x over gw][C]
    return res.reshape(gw * gh, -1)


def main():
    out_path = sys.argv[1]
    cases = []
    for side, grids in CASES:
        tab = table(side, HIDDEN)
        for gw, gh in grids:
            ref = reference(tab, side, gw, gh)
            new = llamacpp_new(tab, side, gw, gh)
            old = llamacpp_old(tab, side, gw, gh)
            print(f"side {side:2d} grid {gw:2d}x{gh:2d}: llama.cpp d3c3e242c vs reference max |d| {np.abs(new - ref).max():.3e}; "
                  f"pre-#32433 order (border-clamped) vs reference {np.abs(old - ref).max():.3e}")
            cases.append({"side": side, "gridW": gw, "gridH": gh, "values": [float("%.9g" % v) for v in ref.reshape(-1)]})
    doc = {
        "source": "transformers " + transformers.__version__ + " vision_utils.get_vision_interpolation_indices_and_weights, "
                  "called as main 7394e56's MuseGlimmerVisionPatchEmbedder calls it (grid_thw[:, [0, 2, 1]], bilinear, "
                  "align_corners=False, padding='zeros'); torch " + torch.__version__,
        "generator": "eng/validation/muse-glimmer-posemb-reference.py",
        "table": "t[i] = sin(0.37 i) + 0.25 cos(1.3 i) over the flat [side * side, hidden] table, as float32",
        "hidden": HIDDEN,
        "layout": "values: the raster-order patch rows (row y, column x at y * gridW + x), hidden values each",
        "cases": cases,
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
