#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""Golden arrays of diffusers' MiniMax-H3-Image edit pipeline (main d5df05ed), computed by diffusers' own functions on the
CPU with no weights, for InferenceWeb.Tests/MiniMaxH3ImageEditGoldenTests.cs:

* sizes   bound_short_edge (the VAE copy at 1024-2048, the vision copy at 1024) and the Qwen2-VL processor's smart_resize;
* canvas  get_image_canvas_size at the 2048 x 2048 generation floor;
* layout  build_image_packed_sequence (positions as float32 bits, token tags, reference rows) and the step-0 row
          timestep plan of MiniMaxH3ImageSetTimestepsStep;
* stream  one CPU torch.Generator through the edit's draws in the pipeline's order: every reference's posterior sample
          (DiagonalGaussianDistribution.sample over the uncropped latent, rounded to float16, cropped to whole patches,
          normalized with the VAE config's latents_mean / latents_std), the target noise (PrepareLatents), and every
          reference's augmentation (EditPrepareLatents: scheduler.scale_noise at a float32 0.999) - as packed rows.

    python -I eng/validation/MiniMaxH3ImageEdit/diffusers_edit_golden.py --vae-config <vae_diffusers/config.json> OUT.json

Requires torch, transformers and diffusers at d5df05ed.
"""
import argparse
import json
import struct

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
    build_image_packed_sequence,
    get_image_canvas_size,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge
from diffusers.schedulers import MiniMaxH3Scheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

PATCH = (1, 2, 2)
TEXT_TAG, IMAGE_TAG = 1, 0
SAMPLE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]


def bits(t):
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in t.reshape(-1).tolist()]


def moments_for(channels, h, w, salt):
    """Fixed, smooth posterior parameters [1, 2C, h, w] (a mean and a log-variance around -6)."""
    c = torch.arange(2 * channels, dtype=torch.float64).view(-1, 1, 1)
    y = torch.arange(h, dtype=torch.float64).view(1, -1, 1)
    x = torch.arange(w, dtype=torch.float64).view(1, 1, -1)
    mean = torch.sin(0.37 * c + 0.11 * y * (salt + 1) + 0.23 * x) * 1.5
    logvar = -6.0 + torch.cos(0.29 * c + 0.07 * x * (salt + 2) + 0.31 * y)
    m = torch.where(c < channels, mean, logvar)
    return m.unsqueeze(0).to(torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-config", required=True)
    ap.add_argument("out")
    args = ap.parse_args()
    vae = json.load(open(args.vae_config))
    latents_mean = torch.tensor(vae["latents_mean"]).view(1, -1, 1, 1)
    latents_std = torch.tensor(vae["latents_std"]).view(1, -1, 1, 1)

    sizes = []
    for w, h in [(1024, 1024), (2048, 2048), (512, 512), (640, 480), (4000, 3000), (1500, 1024), (1024, 1500), (3000, 1000),
                 (777, 333), (2050, 1030), (32, 32), (33, 47), (8192, 2048), (1366, 768), (1000, 1000), (1023, 4097)]:
        vae_copy = bound_short_edge(w, h, 1024, 2048, 32)
        vis = bound_short_edge(w, h, 1024, 1024, 32)
        sh, sw = smart_resize(vis[1], vis[0], factor=32, min_pixels=65536, max_pixels=16777216)
        sizes.append({"w": w, "h": h, "vae": list(vae_copy), "vision": list(vis), "smart": [sw, sh]})

    canvas = []
    for h, w in [(2048, 2048), (1024, 1024), (512, 768), (1080, 1920), (4096, 4096), (1000, 1000), (2048, 1024), (96, 4000)]:
        ch, cw = get_image_canvas_size(h, w, 2048 * 2048, 32)
        canvas.append({"h": h, "w": w, "canvas": [cw, ch]})

    layouts = []
    for name, tags, refs, target in [
        ("t2i", [TEXT_TAG] * 7, [], (64, 96)),
        ("one", [TEXT_TAG] * 4 + [IMAGE_TAG] * 6 + [TEXT_TAG] * 5, [(8, 12)], (16, 16)),
        ("two", [TEXT_TAG] * 3 + [IMAGE_TAG] * 4 + [TEXT_TAG] * 3 + [IMAGE_TAG] * 10 + [TEXT_TAG] * 2, [(6, 10), (12, 4)], (10, 14)),
    ]:
        pos, tok, img_idx, txt_idx, nref = build_image_packed_sequence(
            torch.tensor(tags), refs, target[0], target[1], PATCH, IMAGE_TAG)
        timesteps = torch.tensor([1.0 - s for s in SAMPLE_SIGMAS[:-1]], dtype=torch.float32)
        row = torch.full((pos.shape[0],), float(timesteps[0]), dtype=torch.float32)
        row[img_idx[:nref]] = 0.999
        uniq, inv = torch.unique(row, sorted=True, return_inverse=True)
        layouts.append({
            "name": name, "tags": tags, "references": [list(r) for r in refs], "target": list(target),
            "positions": bits(pos.to(torch.float32)), "tokenTags": tok.tolist(), "referenceRows": int(nref),
            "uniqueTimesteps": bits(uniq), "timestepIndex": inv.tolist(),
        })

    # The edit's draws, in the pipeline's order, off one CPU generator.
    seed = 1234
    generator = torch.Generator("cpu").manual_seed(seed)
    ref_shapes = [(5, 7), (4, 4), (3, 6)]  # uncropped latent (h, w)
    target = (4, 6)
    refs_pre = []
    for k, (h, w) in enumerate(ref_shapes):
        posterior = DiagonalGaussianDistribution(moments_for(24, h, w, k))
        latents = posterior.sample(generator=generator).to(torch.float16).float()
        latents = latents[..., : h // 2 * 2, : w // 2 * 2]
        refs_pre.append((latents - latents_mean) / latents_std)
    target_noise = randn_tensor((1, 24, 1, target[0], target[1]), generator=generator, device=torch.device("cpu"), dtype=torch.float32)
    target_rows = patchify_video_latents(target_noise, PATCH)
    scheduler = MiniMaxH3Scheduler()
    ref_pre_rows, ref_post_rows = [], []
    for latents in refs_pre:
        noise = randn_tensor(latents.shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
        noised = scheduler.scale_noise(latents, torch.tensor(0.999), noise)
        ref_pre_rows.append(bits(patchify_video_latents(latents[:, :, None], PATCH)))
        ref_post_rows.append(bits(patchify_video_latents(noised[:, :, None], PATCH)))
    stream = {
        "seed": seed,
        "references": [{"height": h, "width": w,
                        "moments": bits(moments_for(24, h, w, k)),
                        "rowsBeforeAugmentation": ref_pre_rows[k], "rows": ref_post_rows[k]}
                       for k, (h, w) in enumerate(ref_shapes)],
        "target": list(target),
        "targetNoiseRows": bits(target_rows),
    }

    import diffusers
    doc = {"source": f"diffusers {diffusers.__version__} (d5df05ed), torch {torch.__version__}",
           "sizes": sizes, "canvas": canvas, "layouts": layouts, "stream": stream}
    with open(args.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
