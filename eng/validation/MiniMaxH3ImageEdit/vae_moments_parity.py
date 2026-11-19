#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""The VAE encoder's moments, TensorSharp (a MiniMaxH3ImageAB --vae-dump directory, any backend) against diffusers'
AutoencoderKLMiniMaxH3Image (main d5df05ed) in strict float32 (no TF32 anywhere: on the CPU, or on CUDA with TF32 off)
with the official float32 weights and with Comfy-Org's float16 weights, beside the float16 rounding of the result (the
yardstick). One or more dump directories may be given; each is reported by itself.

    python -I vae_moments_parity.py --official DIR --comfy-vae FILE [--device cpu|cuda] [--tf32-reference] --out report.json DUMP [DUMP ...]
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from diffusers import AutoencoderKLMiniMaxH3Image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def stats(want, got):
    want = np.asarray(want, np.float64).reshape(-1)
    got = np.asarray(got, np.float64).reshape(-1)
    d = got - want
    return {"max_abs_err": float(np.abs(d).max()), "rms_err": float(np.sqrt((d * d).mean())),
            "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(want)), "max_abs_ref": float(np.abs(want).max())}


def rename(name):
    if name.startswith("encoder.down."):
        level, rest = name[len("encoder.down."):].split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1).replace("downsample.", "downsamplers.0.", 1)
        return f"encoder.down_blocks.{level}.{rest}"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--comfy-vae", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tf32-reference", action="store_true", help="also run diffusers' own CUDA default (TF32 cuDNN convolutions)")
    ap.add_argument("dumps", nargs="+")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(a.device)
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(os.path.join(a.official, "vae_diffusers"), torch_dtype=torch.float32).to(dev).eval()
    mean = torch.tensor(MEAN, device=dev).view(1, -1, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, -1, 1, 1)
    comfy = {}
    with safe_open(a.comfy_vae, "pt") as f:
        for n in f.keys():
            if n.startswith(("encoder.", "quant_conv.")):
                comfy[rename(n)] = f.get_tensor(n).float()
    official_sd = {k: v.clone() for k, v in vae.state_dict().items() if k.startswith(("encoder.", "quant_conv."))}
    report = {"device": a.device, "tf32": False, "dumps": {}}
    for d in a.dumps:
        k = 0
        out = []
        while os.path.exists(os.path.join(d, f"moments-{k}.f32")):
            px = torch.from_numpy(np.array(Image.open(os.path.join(d, f"vae-copy-{k}.png")).convert("RGB"))).to(dev).permute(2, 0, 1)[None]
            x = (px.to(torch.float32).div(255.0) - mean) / std
            res = {}
            for label, sd in (("official_fp32", official_sd), ("comfy_fp16_weights", comfy)):
                vae.load_state_dict(sd, strict=False)
                with torch.no_grad():
                    m = vae.encode(x, return_dict=False)[0].parameters.float().cpu().numpy()[0]
                res[label] = m
            if a.tf32_reference and torch.cuda.is_available():
                # diffusers' own default on a GPU: cuDNN convolutions in TF32 (torch.backends.cudnn.allow_tf32 is True by default).
                vae.load_state_dict(official_sd, strict=False)
                vae_gpu = vae.to("cuda")
                torch.backends.cudnn.allow_tf32 = True
                with torch.no_grad():
                    res["official_cuda_default_tf32"] = vae_gpu.encode(x.to("cuda"), return_dict=False)[0].parameters.float().cpu().numpy()[0]
                torch.backends.cudnn.allow_tf32 = False
                vae.to(dev)
            h, w = res["official_fp32"].shape[-2:]
            ts = np.fromfile(os.path.join(d, f"moments-{k}.f32"), dtype="<f4").reshape(48, h, w)
            o, c = res["official_fp32"], res["comfy_fp16_weights"]
            out.append({
                "latent": [int(h), int(w)],
                "ts_vs_official_fp32": {"mean": stats(o[:24], ts[:24]), "logvar": stats(o[24:], ts[24:])},
                "ts_vs_same_fp16_weights_fp32": {"mean": stats(c[:24], ts[:24]), "logvar": stats(c[24:], ts[24:])},
                "fp16_weights_vs_official": {"mean": stats(o[:24], c[:24])},
                "f16_rounding_of_official_mean": stats(o[:24], o[:24].astype(np.float16).astype(np.float32)),
            })
            if "official_cuda_default_tf32" in res:
                out[-1]["diffusers_cuda_default_tf32_vs_strict_fp32"] = {"mean": stats(o[:24], res["official_cuda_default_tf32"][:24])}
            k += 1
        report["dumps"][d] = out
    json.dump(report, open(a.out, "w"), indent=2)
    for d, out in report["dumps"].items():
        for i, e in enumerate(out):
            print(d, i, e["latent"], "ts vs same weights (mean rel_l2 %.2e max %.3e)" % (e["ts_vs_same_fp16_weights_fp32"]["mean"]["rel_l2"], e["ts_vs_same_fp16_weights_fp32"]["mean"]["max_abs_err"]),
                  "ts vs official (mean rel_l2 %.2e)" % e["ts_vs_official_fp32"]["mean"]["rel_l2"],
                  "f16 rounding (rel_l2 %.2e max %.3e)" % (e["f16_rounding_of_official_mean"]["rel_l2"], e["f16_rounding_of_official_mean"]["max_abs_err"]))


if __name__ == "__main__":
    main()
