#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""NumPy oracle for MiniMax-H3-Image's VAE encoder (editing), written independently of TensorSharp from diffusers'
AutoencoderKLMiniMaxH3Image (main d5df05ed): the causal 3-D CNN encoder at one frame (every kernel's last temporal tap,
reflect spatial padding, per-frame GroupNorm(32, 1e-6), SiLU, the bottom/right reflect pad before a stride-2 downsample),
256-pixel tiles with overlaps of at least 64 (_split_tiles) and the moments stitched against each RAW neighbour
(_stitch_tiles). Computed in float64.

The synthetic encoder's weights are a formula of the tensor's name and the element index (no random generator), so the C#
test (InferenceWeb.Tests/MiniMaxH3ImageVaeEncoderTests.cs) writes the same file at test time; this script writes only the
expected moments, as InferenceWeb.Tests/Fixtures/MiniMaxH3Image/vae-encoder-oracle.json.

    python3 -I eng/validation/MiniMaxH3ImageEdit/vae_encoder_oracle.py InferenceWeb.Tests/Fixtures/MiniMaxH3Image/vae-encoder-oracle.json

Requires numpy only.
"""
import json
import sys
import zlib

import numpy as np

# The synthetic encoder: 4 levels, each one resnet block and a stride-2 downsample (16x in all, as the release's 2,2,2,2),
# channels 32, 64, 64, 64 (level 1's first block widens through a 1x1 shortcut); 24 latent channels (48 moments).
CHANNELS = [32, 64, 64, 64]
LATENT = 24
GROUPS = 32
EPS = 1e-6
TILE, OVERLAP, RATIO = 256, 64, 16
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CASES = [(320, 288), (64, 48)]  # (width, height)


def values(name, count, scale, offset):
    """The synthetic tensor `name`: element i is a hash of (name, i) mapped to [-1, 1), times scale, plus offset, rounded
    to float16 (the file's storage). The C# test computes the same uint32 arithmetic."""
    salt = zlib.crc32(name.encode("ascii")) & 0xFFFFFFFF
    i = np.arange(count, dtype=np.uint64)
    h = (i * np.uint64(2654435761) + np.uint64(salt) * np.uint64(40503) + np.uint64(12345)) & np.uint64(0xFFFFFFFF)
    h = h ^ (h >> np.uint64(15))
    h = (h * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
    u = (h >> np.uint64(8)).astype(np.float64) / float(1 << 24)  # [0, 1)
    v = (offset + scale * (2.0 * u - 1.0)).astype(np.float32)
    return v.astype(np.float16)


def tensors():
    """name -> (shape in torch order, float16 values), the names of Comfy-Org's minimax_h3_image_vae_fp16 encoder."""
    out = {}

    def conv(name, oc, ic, k):
        kd = 3 if k == 3 else 1
        shape = (oc, ic, kd, k, k)
        n = int(np.prod(shape))
        out[name + ".weight"] = (shape, values(name + ".weight", n, 1.0 / np.sqrt(ic * k * k), 0.0))
        out[name + ".bias"] = ((oc,), values(name + ".bias", oc, 0.05, 0.0))

    def norm(name, c):
        out[name + ".weight"] = ((c,), values(name + ".weight", c, 0.2, 1.0))
        out[name + ".bias"] = ((c,), values(name + ".bias", c, 0.1, 0.0))

    conv("encoder.conv_in", CHANNELS[0], 3, 3)
    cin = CHANNELS[0]
    for level, c in enumerate(CHANNELS):
        p = f"encoder.down.{level}."
        norm(p + "block.0.norm1", cin)
        conv(p + "block.0.conv1", c, cin, 3)
        norm(p + "block.0.norm2", c)
        conv(p + "block.0.conv2", c, c, 3)
        if cin != c:
            conv(p + "block.0.nin_shortcut", c, cin, 1)
        conv(p + "downsample.conv", c, c, 3)
        cin = c
    norm("encoder.norm_out", cin)
    conv("encoder.conv_out", 2 * LATENT, cin, 3)
    conv("quant_conv", 2 * LATENT, 2 * LATENT, 1)
    return out


def image(width, height):
    """The synthetic RGB picture's bytes [H, W, 3]."""
    y, x = np.mgrid[0:height, 0:width]
    rgb = np.stack([(x * 7 + y * 13 + c * 101 + (x * y) % 17) % 256 for c in range(3)], axis=-1)
    return rgb.astype(np.uint8)


def conv2d(x, w, b, stride):
    """x [C, H, W], w [OC, IC, k, k]: an unpadded cross-correlation, float64."""
    oc, ic, k, _ = w.shape
    c, h, wd = x.shape
    oh, ow = (h - k) // stride + 1, (wd - k) // stride + 1
    cols = np.empty((ic, k, k, oh, ow), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            cols[:, i, j] = x[:, i : i + stride * oh : stride, j : j + stride * ow : stride]
    y = np.tensordot(w.astype(np.float64), cols, axes=([1, 2, 3], [0, 1, 2]))
    return y + b.astype(np.float64)[:, None, None]


def reflect(x, top, bottom, left, right):
    return np.pad(x, ((0, 0), (top, bottom), (left, right)), mode="reflect")


def group_norm_silu(x, weight, bias):
    c, h, w = x.shape
    g = x.reshape(GROUPS, -1)
    mean = g.mean(axis=1, keepdims=True)
    var = g.var(axis=1, keepdims=True)
    n = ((g - mean) / np.sqrt(var + EPS)).reshape(c, h, w)
    n = n * weight.astype(np.float64)[:, None, None] + bias.astype(np.float64)[:, None, None]
    return n / (1.0 + np.exp(-n))


def encode_tile(t, x):
    """diffusers' MiniMaxH3ImageEncoder3d + quant_conv over one frame [3, H, W] (normalized pixels)."""

    def kernel(name):
        shape, v = t[name + ".weight"]
        w = v.astype(np.float32).reshape(shape)
        return w[:, :, -1] if len(shape) == 5 else w  # the causal kernel's last temporal tap

    def bias(name):
        return t[name + ".bias"][1].astype(np.float32)

    def conv3(name, h):
        return conv2d(reflect(h, 1, 1, 1, 1), kernel(name), bias(name), 1)

    def vec(name):
        return t[name][1].astype(np.float32)

    h = conv3("encoder.conv_in", x)
    for level in range(len(CHANNELS)):
        p = f"encoder.down.{level}."
        r = group_norm_silu(h, vec(p + "block.0.norm1.weight"), vec(p + "block.0.norm1.bias"))
        r = conv3(p + "block.0.conv1", r)
        r = group_norm_silu(r, vec(p + "block.0.norm2.weight"), vec(p + "block.0.norm2.bias"))
        r = conv3(p + "block.0.conv2", r)
        skip = h
        if p + "block.0.nin_shortcut.weight" in t:
            skip = conv2d(h, kernel(p + "block.0.nin_shortcut"), bias(p + "block.0.nin_shortcut"), 1)
        h = skip + r
        h = conv2d(reflect(h, 0, 1, 0, 1), kernel(p + "downsample.conv"), bias(p + "downsample.conv"), 2)
    h = group_norm_silu(h, vec("encoder.norm_out.weight"), vec("encoder.norm_out.bias"))
    h = conv3("encoder.conv_out", h)
    return conv2d(h, kernel("quant_conv"), bias("quant_conv"), 1)


def split_tiles(length):
    if TILE >= length:
        return [0], [length], []
    n = -(-length // TILE)
    while TILE * n - OVERLAP * (n - 1) - length < 0:
        n += 1
    overlaps = [OVERLAP] * (n - 1)
    remaining = TILE * n - sum(overlaps) - length
    for i in range(remaining // RATIO):
        overlaps[i % (n - 1)] += RATIO
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + TILE - overlaps[i])
    return starts, [TILE] * n, overlaps


def blend(a, b, extent, axis):
    extent = min(a.shape[axis], b.shape[axis], extent)
    pos = np.arange(extent, dtype=np.float64) / extent
    shape = [1] * a.ndim
    shape[axis] = extent
    wa, wb = (1 - pos).reshape(shape), pos.reshape(shape)
    sa = [slice(None)] * a.ndim
    sa[axis] = slice(-extent, None)
    sb = [slice(None)] * b.ndim
    sb[axis] = slice(0, extent)
    blended = a[tuple(sa)] * wa + b[tuple(sb)] * wb
    rest = [slice(None)] * b.ndim
    rest[axis] = slice(extent, None)
    return np.concatenate([blended, b[tuple(rest)]], axis=axis)


def stitch(tiles, h_overlaps, w_overlaps):
    rows = []
    for i, row in enumerate(tiles):
        out = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend(tiles[i - 1][j], tile, h_overlaps[i - 1], axis=-2)
            if j > 0:
                tile = blend(row[j - 1], tile, w_overlaps[j - 1], axis=-1)
            if i < len(tiles) - 1:
                tile = tile[:, : -h_overlaps[i], :]
            if j < len(row) - 1:
                tile = tile[:, :, : -w_overlaps[j]]
            out.append(tile)
        rows.append(np.concatenate(out, axis=-1))
    return np.concatenate(rows, axis=-2)


def encode(t, rgb):
    x = rgb.astype(np.float32) / np.float32(255.0)
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float64)
    _, height, width = x.shape
    ys, yl, yo = split_tiles(height)
    xs, xl, xo = split_tiles(width)
    tiles = [[encode_tile(t, x[:, y0 : y0 + h, x0 : x0 + w]) for x0, w in zip(xs, xl)] for y0, h in zip(ys, yl)]
    return stitch(tiles, [o // RATIO for o in yo], [o // RATIO for o in xo])


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "vae-encoder-oracle.json"
    t = tensors()
    cases = []
    for width, height in CASES:
        moments = encode(t, image(width, height))
        cases.append({
            "width": width,
            "height": height,
            "latentHeight": int(moments.shape[1]),
            "latentWidth": int(moments.shape[2]),
            # [48, h, w] float32: the mean (channels 0-23) and the log-variance (24-47), unclamped.
            "moments": [float(str(np.float32(v))) for v in moments.reshape(-1)],
        })
        print(f"{width}x{height}: moments {moments.shape}, |max| {np.abs(moments).max():.4f}", file=sys.stderr)
    doc = {
        "source": "eng/validation/MiniMaxH3ImageEdit/vae_encoder_oracle.py (numpy float64, diffusers d5df05ed semantics)",
        "channels": CHANNELS,
        "latentChannels": LATENT,
        "cases": cases,
    }
    with open(out_path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
