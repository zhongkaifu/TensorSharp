#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""A diffusers reference picture of a MiniMax-H3-Image speed adapter (docs/design/diffusion-runtime.md 7.3, stage S7).

The whole bf16 pipeline does not fit a 46 GB GPU and 50 GB of host memory, so diffusers' DiT is given TensorSharp's text
conditioning (MiniMaxH3ImageAB --dump-cond: the layer-50 hidden rows and their tags) and runs everything else itself:
diffusers' packed layout (build_image_packed_sequence), the initial noise off torch.Generator("cpu").manual_seed(seed)
(prepare_latents' shape and draw), MiniMaxH3Scheduler on the adapter's sigmas, the official bf16 transformer_diffusers
(MiniMaxAI/MiniMax-H3-Image) with the LoRA fused into each block's weights the way the adapter's card does it
(W += scale * B @ A in the weights' dtype), the step's own head (pdd_step_index = i: by-step), and the official float32
VAE. The 50 blocks are streamed to the GPU one at a time (read by HTTP range from the Hub with an hf:// path, or locally).

  python -I diffusers_pdmd.py --cond DIR/<id>.json --official <the release's scheduler, VAE and config dir> \
      --transformer hf://MiniMaxAI/MiniMax-H3-Image@8ced6006.../transformer_diffusers --lora <diffusers PEFT file> \
      --sigmas 1,0.9230769230769231,0 --out <dir>
"""
import argparse
import json
import os
import struct
import time

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3ImageTransformer2DModel
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import build_image_packed_sequence, patchify_video_latents
from diffusers.schedulers import MiniMaxH3Scheduler
from diffusers.utils.torch_utils import randn_tensor

PATCH = (1, 2, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, help="a MiniMaxH3ImageAB --dump-cond <id>.json (beside its .hidden.f32)")
    ap.add_argument("--official", required=True, help="the official repo's small files (vae_diffusers, scheduler)")
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--lora", help="diffusers PEFT LoRA file (fused per block); omit for the base model")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--sigmas", default="1,0.9230769230769231,0")
    ap.add_argument("--no-keep", dest="keep", action="store_false",
                    help="read the blocks again each step instead of keeping them fused in host memory (66 GB; more "
                         "than a 50 GB container holds)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda")
    meta = json.load(open(a.cond))
    hidden = np.fromfile(a.cond.replace(".json", ".hidden.f32"), dtype="<f4").reshape(meta["rows"], meta["width"])
    H, W, seed = meta["image_height"], meta["image_width"], meta["seed"]
    lh, lw = H // 16, W // 16
    tags = torch.tensor(meta["modalities"] if meta.get("modalities") else [1] * meta["rows"], dtype=torch.long)
    pos, tok, img_idx, txt_idx, nref = build_image_packed_sequence(tags, [], lh, lw, PATCH, 0)

    # The noise exactly as prepare_latents draws it (a CPU generator, float32).
    gen = torch.Generator("cpu").manual_seed(seed)
    latents = randn_tensor((1, 24, 1, lh, lw), generator=gen, device=torch.device("cpu"), dtype=torch.float32)
    rows = patchify_video_latents(latents, PATCH)             # [tokens, 96] (batch 1: the rows of its one picture)
    if rows.shape != ((lh // 2) * (lw // 2), 96):
        raise ValueError(f"patchified rows {tuple(rows.shape)}, expected {((lh // 2) * (lw // 2), 96)}")

    # ---- the transformer, streamed block by block, with the LoRA fused into each block as it loads
    tdir = a.transformer
    remote = tdir.startswith("hf://")
    if remote:
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        base = tdir[len("hf://"):].rstrip("/")
        config = json.load(fs.open(f"{base}/config.json", "r"))
        index = json.load(fs.open(f"{base}/diffusion_pytorch_model.safetensors.index.json", "r"))["weight_map"]
    else:
        config = json.load(open(os.path.join(tdir, "config.json")))
        index = json.load(open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    with init_empty_weights():
        model = MiniMaxH3ImageTransformer2DModel.from_config(config)
    model.eval()
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    handles = {}
    fetched = [0]

    def tensor(name):
        shard = index[name]
        if not remote:
            if shard not in handles:
                handles[shard] = safe_open(os.path.join(tdir, shard), "pt", device="cpu")
            return handles[shard].get_tensor(name)
        path = f"{base}/{shard}"
        if shard not in handles:
            n = struct.unpack("<Q", fs.cat_file(path, start=0, end=8))[0]
            handles[shard] = (8 + n, json.loads(fs.cat_file(path, start=8, end=8 + n)))
        start, header = handles[shard]
        info = header[name]
        s0, s1 = info["data_offsets"]
        fetched[0] += s1 - s0
        data = bytearray(fs.cat_file(path, start=start + s0, end=start + s1))
        return torch.frombuffer(data, dtype=dtypes[info["dtype"]]).reshape(info["shape"]).clone()

    lora = safe_open(a.lora, "pt", device="cpu") if a.lora else None
    lora_keys = set(lora.keys()) if lora else set()
    fused = [0]

    def fuse(key, value):
        """The card's fuse_lora: weight += scale * (B @ A), the product in float32, added in the weight's dtype."""
        if lora is None or not key.endswith(".weight"):
            return value
        stem = key[: -len(".weight")]
        for prefix in ("transformer.", ""):
            ka, kb = f"{prefix}{stem}.lora_A.weight", f"{prefix}{stem}.lora_B.weight"
            if ka in lora_keys:
                # On the GPU (the CPU takes seconds a module): the float32 product, added in the weight's dtype.
                delta = lora.get_tensor(kb).to(device).float() @ lora.get_tensor(ka).to(device).float()
                fused[0] += 1
                return value.to(device) + (a.lora_scale * delta).to(value.dtype)
        return value

    kept = {}   # each block's fused tensors in host memory after the first step reads them (66 GB for the bf16 release)

    def materialize(prefix, module):
        for n, _ in list(module.named_parameters(recurse=True)) + list(module.named_buffers(recurse=True)):
            key = prefix + n
            if key in index:
                value = kept.get(key)
                if value is None:
                    value = fuse(key, tensor(key))
                    if a.keep:
                        kept[key] = value.to("cpu")
                set_module_tensor_to_device(model, key, device, value=value, dtype=value.dtype)

    def release(prefix, module):
        for n, _ in list(module.named_parameters(recurse=True)):
            set_module_tensor_to_device(model, prefix + n, "meta")
        torch.cuda.empty_cache()

    for name, child in model.named_children():
        if name != "transformer_blocks":
            materialize(name + ".", child)
    for i, block in enumerate(model.transformer_blocks):
        prefix = f"transformer_blocks.{i}."
        block.register_forward_pre_hook(lambda m, args, p=prefix: materialize(p, m))
        block.register_forward_hook(lambda m, args, o, p=prefix: release(p, m))
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            raise RuntimeError(f"buffer {name} has no value")
        if buf.device != device:
            set_module_tensor_to_device(model, name, device, value=buf, dtype=buf.dtype)

    # ---- the loop: MiniMaxH3Scheduler on the adapter's sigmas (taken verbatim: shift_provided_sigmas false in the
    # official scheduler config), head i at step i, the velocity form x_next = x + (sigma - sigma_next) * v
    sigmas = [float(s) for s in a.sigmas.split(",")]
    scheduler = MiniMaxH3Scheduler.from_pretrained(os.path.join(a.official, "scheduler"))
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    x = rows.clone().to(device)
    enc = torch.from_numpy(hidden)[None].to(device, torch.bfloat16)
    steps = []
    for i, t in enumerate(scheduler.timesteps):
        row_t = torch.full((pos.shape[0],), float(t), dtype=torch.float32)
        uniq, inv = torch.unique(row_t, sorted=True, return_inverse=True)
        t0 = time.time()
        with torch.no_grad():
            v = model(hidden_states=x[None], encoder_hidden_states=enc, timestep=uniq.to(device),
                      timestep_indices=inv.to(device), token_tags=tok.to(device), position_ids=pos.to(device, torch.float32),
                      image_indices=img_idx.to(device), text_indices=txt_idx.to(device), pdd_step_index=i,
                      return_dict=False)[0][0].float()
            x = scheduler.step(v, t, x, return_dict=False)[0]
        steps.append({"step": i, "timestep": float(t), "seconds": time.time() - t0})
        print(f"step {i + 1}: {time.time() - t0:.1f}s, {fetched[0] / 1e9:.1f} GB read, {fused[0]} LoRA terms fused so far", flush=True)

    # ---- decode as diffusers' MiniMaxH3ImageDecodeStep does: unpatchify, denormalize, the float32 VAE, ImageNet
    # normalization reverted, 8 bits by truncation
    lat = x.detach().float().cpu()
    np.asarray(lat.numpy(), dtype="<f4").tofile(os.path.join(a.out, "final-rows.f32"))
    model = None
    torch.cuda.empty_cache()
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(os.path.join(a.official, "vae_diffusers"), torch_dtype=torch.float32).to(device)
    vae.eval()
    lat = lat.reshape(-1, lh // 2, lw // 2, 24, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 24, lh, lw).to(device)
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1)
    with torch.no_grad():
        image = vae.decode(lat * std + mean, return_dict=False)[0]
    pm = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, -1, 1, 1)
    ps = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, -1, 1, 1)
    image = (image.float() * ps + pm).clamp(0, 1)
    img = (image[0] * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(os.path.join(a.out, f"diffusers-{meta['id']}.png"))
    json.dump({"cond": a.cond, "seed": seed, "size": [W, H], "sigmas": sigmas, "lora": a.lora, "lora_terms_fused": fused[0],
               "transformer": a.transformer, "bytes_read": fetched[0], "steps": steps}, open(os.path.join(a.out, "diffusers.json"), "w"),
              indent=1)
    print("done", os.path.join(a.out, f"diffusers-{meta['id']}.png"))


if __name__ == "__main__":
    main()
