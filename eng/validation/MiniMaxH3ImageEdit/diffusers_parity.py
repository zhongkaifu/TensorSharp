#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# https://github.com/zhongkaifu/TensorSharp
#
# This file is part of TensorSharp.
#
# TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
"""MiniMax-H3-Image edit parity, TensorSharp against diffusers (main d5df05ed) with the official checkpoint
(MiniMaxAI/MiniMax-H3-Image), component by component (docs/design/diffusion-runtime.md section 7.3). Reads a
MiniMaxH3ImageAB --parity-dump directory and compares:

  resizes      each reference's VAE copy and vision copy (diffusers' MiniMaxH3ImageEditResizeStep, PIL), bytes
  presentation the token ids and tags (MiniMaxH3ImageEditTextEncoderStep with the official tokenizer and processor)
  moments      the VAE encoder's posterior (the official float32 vae_diffusers, tiled as released; and the same module
               with Comfy-Org's float16 weights, to separate the weights' rounding from the computation)
  rows         every reference's rows (posterior sample off the request's CPU generator, float16, crop, normalize,
               augmentation) and the target's noise, in diffusers' draw order, from diffusers' own moments
  layout       build_image_packed_sequence's positions (float32) and tags against TensorSharp's
  step         (--transformer DIR) the first step's velocity of the official bf16 transformer_diffusers, its blocks
               streamed to the GPU one at a time, given TensorSharp's conditioning, reference rows and target noise;
               with --pruned FILE also in float32 (TF32 off), with the official and with the pruned checkpoint's AdaLN

    python -I diffusers_parity.py --dump DIR --official /root/models/h3img/official \
        --comfy-vae minimax_h3_image_vae_fp16.safetensors [--transformer DIR/transformer_diffusers] --out report.json
"""
import argparse
import json
import os
import struct

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors import safe_open

from diffusers import AutoencoderKLMiniMaxH3Image
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import build_image_packed_sequence, patchify_video_latents
from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge
from diffusers.schedulers import MiniMaxH3Scheduler
from diffusers.utils.torch_utils import randn_tensor

PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]
PATCH = (1, 2, 2)
SAMPLE_SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]


def f32(path, shape=None):
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(shape) if shape is not None else a


def png(path):
    return np.asarray(Image.open(path).convert("RGB"))


def stats(want, got):
    want = np.asarray(want, dtype=np.float64).reshape(-1)
    got = np.asarray(got, dtype=np.float64).reshape(-1)
    d = got - want
    cos = float(np.dot(want, got) / (np.linalg.norm(want) * np.linalg.norm(got) + 1e-30))
    return {
        "n": int(want.size), "max_abs_err": float(np.abs(d).max()), "rms_err": float(np.sqrt((d * d).mean())),
        "rel_l2": float(np.linalg.norm(d) / (np.linalg.norm(want) + 1e-30)), "cosine": cos,
        "max_abs_ref": float(np.abs(want).max()),
        "bit_exact": int((want.astype(np.float32).view(np.uint32) == got.astype(np.float32).view(np.uint32)).sum()),
    }


def f16_rounding(x):
    """What rounding x to float16 alone moves it by (the yardstick 'within F16 rounding' is read against)."""
    x = np.asarray(x, dtype=np.float32)
    return stats(x, x.astype(np.float16).astype(np.float32))


def comfy_to_diffusers_vae(name):
    """diffusers' convert script's encoder renames (convert_minimax_h3_to_diffusers._rename_video_vae_key)."""
    if name.startswith("encoder.down."):
        level, rest = name[len("encoder.down."):].split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        return f"encoder.down_blocks.{level}.{rest}"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--comfy-vae")
    ap.add_argument("--transformer")
    ap.add_argument("--pruned", help="Comfy-Org's pruned bf16 DiT (its curve-table AdaLN): with --transformer, the step "
                                     "also runs in float32 with the official and with the pruned modulation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dump = args.dump
    req = json.load(open(os.path.join(dump, "request.json")))
    device = torch.device(args.device)
    report = {"request": {k: req[k] for k in ("prompt", "seed", "images", "sampling", "output") if k in req}}

    # ---- resizes (MiniMaxH3ImageEditResizeStep)
    processor_vae = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)
    images = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in req["images"]]

    def resize(image, lo, hi):
        size = bound_short_edge(*image.size, lo, hi, 32)
        return image if image.size == size else processor_vae.resize(image, height=size[1], width=size[0])

    vae_images = [resize(i, 1024, 2048) for i in images]
    vision_images = [resize(i, 1024, 1024) for i in images]
    resizes = []
    for k, (vi, wi) in enumerate(zip(vae_images, vision_images)):
        dv = np.asarray(vi)
        dv = dv[: dv.shape[0] // 16 * 16, : dv.shape[1] // 16 * 16]
        tv = png(os.path.join(dump, f"vae-copy-{k}.png"))
        dw = np.asarray(wi)
        tw = png(os.path.join(dump, f"vision-copy-{k}.png"))
        entry = {"source": list(images[k].size), "vae": {"diffusers": [dv.shape[1], dv.shape[0]], "ts": [tv.shape[1], tv.shape[0]]},
                 "vision": {"diffusers": [dw.shape[1], dw.shape[0]], "ts": [tw.shape[1], tw.shape[0]]}}
        if dv.shape == tv.shape:
            entry["vae"]["bytes_differ"] = int((dv != tv).sum())
        if dw.shape == tw.shape:
            entry["vision"]["bytes_differ"] = int((dw != tw).sum())
        resizes.append(entry)
    report["resizes"] = resizes

    # ---- presentation (MiniMaxH3ImageEditTextEncoderStep)
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
    tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.official, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.official, "processor"))
    vision = processor.image_processor(images=vision_images, return_tensors="pt")
    merge = processor.image_processor.merge_size ** 2
    ids, tags = [], []
    for index, grid in enumerate(vision["image_grid_thw"]):
        label = tokenizer(f"Picture {index + 1}: ", add_special_tokens=False)["input_ids"][:4096]
        vids = ([tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge)
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")])
        ids += label + vids
        tags += [1] * len(label) + [0] * len(vids)
    prompt_ids = tokenizer(req["prompt"], add_special_tokens=False)["input_ids"][:4096]
    ids += prompt_ids
    tags += [1] * len(prompt_ids)
    report["presentation"] = {
        "rows": {"diffusers": len(ids), "ts": len(req["cond_ids"])},
        "ids_equal": ids == req["cond_ids"], "tags_equal": tags == req["cond_tags"],
        "grids": vision["image_grid_thw"].tolist(),
    }

    # ---- moments (AutoencoderKLMiniMaxH3Image, float32, tiling on as released)
    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(os.path.join(args.official, "vae_diffusers"), torch_dtype=torch.float32).to(device).eval()
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)

    def pixels_of(image):
        px = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None]
        px = px[..., : px.shape[-2] // 16 * 16, : px.shape[-1] // 16 * 16]
        return (px.to(torch.float32).div(255.0) - mean) / std

    official_moments = []
    with torch.no_grad():
        for image in vae_images:
            official_moments.append(vae.encode(pixels_of(image), return_dict=False)[0].parameters.float().cpu())
    comfy_moments = None
    if args.comfy_vae:
        sd = {}
        with safe_open(args.comfy_vae, "pt") as f:
            for name in f.keys():
                if name.startswith("encoder.") or name.startswith("quant_conv."):
                    sd[comfy_to_diffusers_vae(name)] = f.get_tensor(name).float()
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        report["comfy_vae_load"] = {"loaded": len(sd), "unexpected": list(unexpected),
                                    "missing_encoder": [m for m in missing if m.startswith(("encoder.", "quant_conv."))]}
        comfy_moments = []
        with torch.no_grad():
            for image in vae_images:
                comfy_moments.append(vae.encode(pixels_of(image), return_dict=False)[0].parameters.float().cpu())
    moments = []
    for k, om in enumerate(official_moments):
        h, w = om.shape[-2:]
        ts = f32(os.path.join(dump, f"moments-{k}.f32"), (48, h, w))
        entry = {"shape": [48, int(h), int(w)],
                 "ts_vs_official_fp32": {"mean": stats(om[0, :24].numpy(), ts[:24]), "logvar": stats(om[0, 24:].numpy(), ts[24:])},
                 "f16_rounding_of_official_mean": f16_rounding(om[0, :24].numpy())}
        if comfy_moments is not None:
            cm = comfy_moments[k]
            entry["ts_vs_diffusers_with_comfy_fp16_weights"] = {"mean": stats(cm[0, :24].numpy(), ts[:24]), "logvar": stats(cm[0, 24:].numpy(), ts[24:])}
            entry["comfy_fp16_weights_vs_official"] = {"mean": stats(om[0, :24].numpy(), cm[0, :24].numpy())}
        moments.append(entry)
    report["moments"] = moments
    del vae
    torch.cuda.empty_cache()

    # ---- rows: the draws in diffusers' order off the request's CPU generator, from diffusers' moments
    generator = torch.Generator("cpu").manual_seed(int(req["seed"]))
    refs = []
    for om in official_moments:
        posterior = DiagonalGaussianDistribution(om)
        lat = posterior.sample(generator=generator).to(torch.float16).float()
        lat = lat[..., : lat.shape[-2] // 2 * 2, : lat.shape[-1] // 2 * 2]
        refs.append((lat - latents_mean) / latents_std)
    sw, sh = [int(v) for v in req["sampling"].split("x")]
    target = randn_tensor((1, 24, 1, sh // 16, sw // 16), generator=generator, device=torch.device("cpu"), dtype=torch.float32)
    target_rows = patchify_video_latents(target, PATCH)
    scheduler = MiniMaxH3Scheduler()
    rows = []
    ref_rows = []
    for k, lat in enumerate(refs):
        noise = randn_tensor(lat.shape, generator=generator, device=torch.device("cpu"), dtype=torch.float32)
        r = patchify_video_latents(scheduler.scale_noise(lat, torch.tensor(0.999), noise)[:, :, None], PATCH)
        ref_rows.append(r)
        ts = f32(os.path.join(dump, f"ref-rows-{k}.f32"))
        rows.append({"latent": list(lat.shape[-2:]), "ts_vs_diffusers": stats(r.numpy(), ts)})
    report["rows"] = {"references": rows,
                      "target_noise": stats(target_rows.numpy(), f32(os.path.join(dump, "latent.f32")))}

    # ---- layout (build_image_packed_sequence) for TensorSharp's tags
    grids = [tuple(lat.shape[-2:]) for lat in refs]
    pos, tok, img_idx, txt_idx, nref = build_image_packed_sequence(torch.tensor(req["cond_tags"]), grids, sh // 16, sw // 16, PATCH, 0)
    ts_pos = f32(os.path.join(dump, "positions.f32"), (-1, 3))
    report["layout"] = {"rows": {"diffusers": int(pos.shape[0]), "ts": int(ts_pos.shape[0])},
                        "reference_rows": {"diffusers": int(nref), "ts": int(req["target_start"]) - len(req["cond_tags"])}}
    if ts_pos.shape == tuple(pos.shape):
        p32 = pos.to(torch.float32).numpy()
        report["layout"]["positions_bit_exact"] = int((p32.view(np.uint32) == ts_pos.view(np.uint32)).sum())
        report["layout"]["positions"] = int(p32.size)

    # ---- the first step (official bf16 transformer_diffusers, streamed)
    if args.transformer:
        report["step"] = step_parity(args, req, dump, pos, tok, img_idx, txt_idx, nref, device)

    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2)[:6000])


def step_parity(args, req, dump, pos, tok, img_idx, txt_idx, nref, device):
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers import MiniMaxH3ImageTransformer2DModel

    tdir = args.transformer
    remote = tdir.startswith("hf://")
    if remote:
        # hf://<repo>@<revision>/<subfolder>: every tensor read by HTTP range from the hub (66 GB, no local copy).
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        base = tdir[len("hf://"):].rstrip("/")   # <owner>/<repo>@<revision>/<subfolder>
        config = json.load(fs.open(f"{base}/config.json", "r"))
        index = json.load(fs.open(f"{base}/diffusion_pytorch_model.safetensors.index.json", "r"))["weight_map"]
    else:
        config = json.load(open(os.path.join(tdir, "config.json")))
        index = json.load(open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    with init_empty_weights():
        model = MiniMaxH3ImageTransformer2DModel.from_config(config)
    model.eval()
    handles = {}
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

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
        a, b = info["data_offsets"]
        data = bytearray(fs.cat_file(path, start=start + a, end=start + b))   # one HTTP range request
        return torch.frombuffer(data, dtype=dtypes[info["dtype"]]).reshape(info["shape"]).clone()

    def materialize(prefix, module):
        for n, _ in list(module.named_parameters(recurse=True)) + list(module.named_buffers(recurse=True)):
            key = prefix + n
            if key in index:
                value = tensor(key)
                # The checkpoint's own dtypes (bf16 blocks, float32 heads and patch projection), not the meta model's.
                set_module_tensor_to_device(model, key, device, value=value, dtype=value.dtype)

    def release(prefix, module):
        for n, _ in list(module.named_parameters(recurse=True)):
            set_module_tensor_to_device(model, prefix + n, "meta")
        torch.cuda.empty_cache()

    # Everything but the 50 blocks stays on the device; each block loads before it runs and leaves after.
    for name, child in model.named_children():
        if name != "transformer_blocks":
            materialize(name + ".", child)
    if not args.pruned:
        for i, block in enumerate(model.transformer_blocks):
            prefix = f"transformer_blocks.{i}."
            block.register_forward_pre_hook(lambda m, a, p=prefix: materialize(p, m))
            block.register_forward_hook(lambda m, a, o, p=prefix: release(p, m))
    # Buffers the checkpoint does not hold (the rope's frequencies) keep the module's own values, on the device.
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            raise RuntimeError(f"buffer {name} has no value")
        if buf.device != device:
            set_module_tensor_to_device(model, name, device, value=buf, dtype=buf.dtype)

    hidden = torch.from_numpy(f32(os.path.join(dump, "cond-hidden.f32"), (-1, 5120)))[None].to(device)
    refs = [torch.from_numpy(f32(os.path.join(dump, f"ref-rows-{k}.f32"), (-1, 96))) for k in range(len(req["images"]))]
    latent = torch.from_numpy(f32(os.path.join(dump, "latent.f32"), (-1, 96)))
    image_rows = torch.cat(refs + [latent])[None].to(device)
    timesteps = torch.tensor([1.0 - s for s in SAMPLE_SIGMAS[:-1]], dtype=torch.float32)
    row = torch.full((pos.shape[0],), float(timesteps[0]), dtype=torch.float32)
    row[img_idx[:nref]] = 0.999
    uniq, inv = torch.unique(row, sorted=True, return_inverse=True)
    import time
    if args.pruned:
        return step_variants(args, dump, model, materialize, release, hidden, image_rows, uniq, inv, tok, pos, img_idx,
                             txt_idx, nref, device)
    t0 = time.time()
    with torch.no_grad():
        out = model(hidden_states=image_rows, encoder_hidden_states=hidden.to(torch.bfloat16), timestep=uniq.to(device),
                    timestep_indices=inv.to(device), token_tags=tok.to(device), position_ids=pos.to(device, torch.float32),
                    image_indices=img_idx.to(device), text_indices=txt_idx.to(device), pdd_step_index=0, return_dict=False)[0]
    seconds = time.time() - t0
    v = out[0, nref:].float().cpu().numpy()
    ts = f32(os.path.join(dump, "velocity.f32"), v.shape)
    s = stats(v, ts)
    s_neg = stats(-v, ts)
    return {"seconds": seconds, "ts_vs_diffusers": s, "ts_vs_negated_diffusers_cosine": s_neg["cosine"],
            "unique_timesteps": uniq.tolist()}


def step_variants(args, dump, model, materialize, release, hidden, image_rows, uniq, inv, tok, pos, img_idx, txt_idx, nref,
                  device):
    """The first step three ways in one pass over the streamed blocks (each block read once): as released (bf16,
    the module's own forward order), in float32 (TF32 off) with the official modulation, and in float32 with the
    pruned checkpoint's modulation (its 1025-entry curve table, interpolated as TensorSharp does, through the
    rank-8 adaln_proj). Separates the reference's own bf16 rounding and the pruned AdaLN from TensorSharp's arithmetic."""
    import copy
    import time
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    pr = safe_open(args.pruned, "pt", device="cpu")
    table = pr.get_tensor("adaln_t_table").float()                      # [1025, 8]
    grid = table.shape[0]

    def curve(t):
        p = min(max(float(t), 0.0), 1.0) * (grid - 1)
        lo = min(max(int(np.floor(p)), 0), grid - 2)
        f = p - lo
        return table[lo] * (1.0 - f) + table[lo + 1] * f

    temb8 = torch.stack([curve(t) for t in uniq.tolist()]).to(device)  # [T, 8]

    def pruned_linear(prefix):
        return (pr.get_tensor(prefix + "adaln_proj.linear.weight").float().to(device),
                pr.get_tensor(prefix + "adaln_proj.linear.bias").float().to(device))

    ts = f32(os.path.join(dump, "velocity.f32"), (-1, 96))
    dev = lambda x: x.to(device)
    ti, tags, positions, ii, xi = dev(inv), dev(tok), dev(pos.to(torch.float32)), dev(img_idx), dev(txt_idx)
    t0 = time.time()
    mods = []
    with torch.no_grad():
        rotary = model.rope(positions)
        adaln_indices = ti * 3 + tags
        bf = lambda m: next(m.parameters()).dtype

        def embed(ctx_embedder, refiner, dtype):
            image = model.proj_in(image_rows.to(bf(model.proj_in)))
            text = refiner(ctx_embedder(hidden.to(dtype)))
            h = text.new_zeros((1, positions.shape[0], text.shape[-1]))
            h = h.index_copy(1, xi, text)
            return h.index_copy(1, ii, image.to(text.dtype))

        temb = model.time_embedder(model.time_proj(dev(uniq)).to(bf(model.time_embedder)))
        ctx32, ref32 = copy.deepcopy(model.context_embedder).float(), copy.deepcopy(model.token_refiner).float()
        norm32 = copy.deepcopy(model.norm_out).float()
        h_rel = embed(model.context_embedder, model.token_refiner, bf(model.context_embedder))
        h_f32 = embed(ctx32, ref32, torch.float32)
        h_pr = h_f32.clone()
        temb32 = temb.float()
        for i, block in enumerate(model.transformer_blocks):
            prefix = f"transformer_blocks.{i}."
            materialize(prefix, block)
            h_rel = block(h_rel, temb, adaln_indices, rotary)
            block.float()
            h_f32 = block(h_f32, temb32, adaln_indices, rotary)
            w, b = pruned_linear(f"blocks.{i}.")
            official = torch.cat(block.adaln_proj(temb32), dim=-1)
            pruned = F.linear(temb8, w, b).view(-1, 6 * block.adaln_proj.hidden_size)
            pruned_chunks = pruned.chunk(6, dim=-1)
            mods.append(stats(official.cpu().numpy(), torch.cat(pruned_chunks, dim=-1).cpu().numpy())["rel_l2"])
            block.adaln_proj.forward = lambda e, c=pruned_chunks: c
            h_pr = block(h_pr, temb32, adaln_indices, rotary)
            del block.adaln_proj.forward
            release(prefix, block)
            del w, b, official, pruned, pruned_chunks
            print(f"[step] block {i} modulation rel_l2 {mods[-1]:.3e}", flush=True)

        head = model.pdd_proj_out[0]
        rel = head(model.norm_out(h_rel, temb, ti).to(bf(head))).index_select(1, ii)
        f32v = head(norm32(h_f32, temb32, ti).to(bf(head))).index_select(1, ii)
        fw, fb = (pr.get_tensor("final_layer.adaln_proj.linear.weight").float().to(device),
                  pr.get_tensor("final_layer.adaln_proj.linear.bias").float().to(device))
        shift, scale = F.linear(temb8, fw, fb).chunk(2, dim=-1)
        official_final = torch.cat(norm32.linear(F.silu(temb32)).chunk(2, dim=-1), dim=-1)
        final_mod = stats(official_final.cpu().numpy(), torch.cat([shift, scale], dim=-1).cpu().numpy())
        hp = norm32.norm(h_pr) * (1.0 + scale.index_select(0, ti)) + shift.index_select(0, ti)
        prv = head(hp.to(bf(head))).index_select(1, ii)
    seconds = time.time() - t0
    out_dir = os.path.dirname(os.path.abspath(args.out))
    vel = {}
    for name, o in (("released_bf16", rel), ("float32", f32v), ("float32_pruned_adaln", prv)):
        vel[name] = -o[0, nref:].float().cpu().numpy()      # diffusers' transformer returns the negated velocity
        np.save(os.path.join(out_dir, f"step-{name}.npy"), vel[name])
    return {
        "seconds": seconds,
        "sign": "TensorSharp's velocity is the negation of the transformer's output; compared after negating",
        "ts_vs": {k: stats(v, ts) for k, v in vel.items()},
        "released_bf16_vs_float32": stats(vel["float32"], vel["released_bf16"]),
        "float32_pruned_vs_float32": stats(vel["float32"], vel["float32_pruned_adaln"]),
        "modulation_rel_l2": {"blocks_max": max(mods), "blocks_mean": float(np.mean(mods)), "blocks": mods,
                              "final": final_mod["rel_l2"]},
        "unique_timesteps": uniq.tolist(),
    }


if __name__ == "__main__":
    main()
