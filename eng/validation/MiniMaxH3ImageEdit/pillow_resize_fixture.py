#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Pillow's Image.resize of 8-bit RGB pictures with LANCZOS and BICUBIC, the filters MiniMax-H3-Image's references are
resized with (diffusers' VaeImageProcessor.resize and ComfyUI's Lanczos for the VAE and vision copies; the Qwen3-VL image
processor's bicubic for a vision copy whose side is not a multiple of 32). Writes
InferenceWeb.Tests/Fixtures/MiniMaxH3Image/pillow-resize.json: the source bytes and every resize's bytes, base64.

    python3 -I eng/validation/MiniMaxH3ImageEdit/pillow_resize_fixture.py InferenceWeb.Tests/Fixtures/MiniMaxH3Image/pillow-resize.json

Requires Pillow (the fixture records its version) and numpy.
"""
import base64
import json
import sys

import numpy as np
import PIL
from PIL import Image


def source(width, height, seed):
    """A picture with edges and smooth ramps: a hash pattern over a gradient."""
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack([
        (x * 255 // max(1, width - 1) + seed) % 256,
        (y * 255 // max(1, height - 1) + 3 * seed) % 256,
        ((x * 31 + y * 17 + seed) * 2654435761 >> 7) % 256,
    ], axis=-1)
    return rgb.astype(np.uint8)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "pillow-resize.json"
    cases = []
    for (w, h, seed, targets) in [
        (37, 29, 1, [(64, 48), (16, 12), (37, 64), (20, 29)]),
        (48, 80, 7, [(32, 48), (16, 16), (80, 48), (48, 112)]),
    ]:
        rgb = source(w, h, seed)
        im = Image.fromarray(rgb, "RGB")
        outs = []
        for (tw, th) in targets:
            for name, flt in (("lanczos", Image.LANCZOS), ("bicubic", Image.BICUBIC)):
                r = np.asarray(im.resize((tw, th), flt))
                outs.append({"width": tw, "height": th, "filter": name,
                             "rgb": base64.b64encode(r.tobytes()).decode("ascii")})
        cases.append({"width": w, "height": h, "rgb": base64.b64encode(rgb.tobytes()).decode("ascii"), "resizes": outs})
    with open(out_path, "w") as f:
        json.dump({"pillow": PIL.__version__, "cases": cases}, f, separators=(",", ":"))
    print(f"wrote {out_path} (Pillow {PIL.__version__})", file=sys.stderr)


if __name__ == "__main__":
    main()
