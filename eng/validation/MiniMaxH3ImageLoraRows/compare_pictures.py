#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""PSNR and mean absolute difference of pictures against a reference picture (8-bit RGB), and a side-by-side sheet.

  python3 -I compare_pictures.py <sheet.jpg> <reference.png> <label=picture.png>...
"""
import sys
import numpy as np
from PIL import Image, ImageDraw
out, ref = sys.argv[1], sys.argv[2]
r = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float64)
items = [("reference", ref)] + [tuple(a.split("=", 1)) for a in sys.argv[3:]]
for label, path in items[1:]:
    p = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    mse = ((p - r) ** 2).mean()
    print(f"{label}: PSNR {10 * np.log10(255 ** 2 / mse):.2f} dB, mean |diff| {np.abs(p - r).mean():.2f}")
cell = 640
sheet = Image.new("RGB", (cell * len(items), cell + 18), "white")
d = ImageDraw.Draw(sheet)
for x, (label, path) in enumerate(items):
    im = Image.open(path).convert("RGB"); im.thumbnail((cell, cell))
    sheet.paste(im, (x * cell, 18)); d.text((x * cell + 4, 3), label, fill="black")
sheet.save(out, quality=90)
