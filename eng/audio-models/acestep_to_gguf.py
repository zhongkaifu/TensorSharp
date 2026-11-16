#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Convert ACE-Step 1.5 (ACE Studio and StepFun, Apache-2.0) into the "ace-step-1.5" GGUF TensorSharp loads.

Input: ComfyUI's all-in-one checkpoint (Comfy-Org/ace_step_1.5_ComfyUI_files, checkpoints/
ace_step_1.5_turbo_aio.safetensors), which holds every part of the pipeline, and a ComfyUI checkout for the Qwen
tokenizer files and the silence latent the model conditions on:

    python eng/audio-models/acestep_to_gguf.py --input ace_step_1.5_turbo_aio.safetensors --comfyui C:/ComfyUI-src \\
        --out ace-step-1.5-turbo.gguf

Tensors, renamed by part and otherwise kept in the release's precision (BF16 / F16):

    te.*     text_encoders.qwen3_06b.transformer.model.*   Qwen3-0.6B caption encoder (and lyric token embeddings)
    lm.*     text_encoders.qwen3_2b.transformer.model.*    the Qwen3 planner LM that writes 5 Hz audio codes
    dit.*    model.diffusion_model.*                       condition encoder, FSQ detokenizer and the DiT
    vae.decoder.*                                          the Oobleck decoder, weight norm folded, F32

The VAE encoder, the audio tokenizer's attention pooler (used only to encode reference audio) and logit_scale are left
out. Metadata: ace.silence_latent [64, 750] (ComfyUI's get_silence_latent), the Qwen BPE vocabulary, merges and
special tokens, and the checkpoint's sampling defaults (shift 3, 8 steps).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_writer import BF16, F16, F32, GgufWriter, read_safetensors  # noqa: E402

PREFIXES = [
    ("text_encoders.qwen3_06b.transformer.model.", "te."),
    ("text_encoders.qwen3_2b.transformer.model.", "lm."),
    ("model.diffusion_model.", "dit."),
]
SKIP = ("dit.tokenizer.attention_pooler.", "dit.tokenizer.audio_acoustic_proj.", "dit.tokenizer.quantizer.project_in.")


def tokenizer_tables(tokenizer_dir):
    vocab = json.load(open(os.path.join(tokenizer_dir, "vocab.json"), encoding="utf-8"))
    cfg = json.load(open(os.path.join(tokenizer_dir, "tokenizer_config.json"), encoding="utf-8"))
    id_to_token = {int(i): t for t, i in vocab.items()}
    special = set()
    for sid, info in cfg.get("added_tokens_decoder", {}).items():
        id_to_token[int(sid)] = info["content"]
        special.add(int(sid))
    size = max(id_to_token) + 1
    tokens = [id_to_token.get(i, f"<|unused_{i}|>") for i in range(size)]
    types = [3 if i in special else (1 if i in id_to_token else 5) for i in range(size)]
    merges = []
    with open(os.path.join(tokenizer_dir, "merges.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line and not line.startswith("#version"):
                merges.append(line)
    return tokens, types, merges


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="ace_step_1.5_turbo_aio.safetensors")
    ap.add_argument("--comfyui", required=True, help="a ComfyUI checkout (the comfy/ package's directory's parent)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    st = read_safetensors(args.input)
    w = GgufWriter("ace-step-1.5")
    w.str("general.name", os.path.splitext(os.path.basename(args.input))[0])
    w.f32("ace.shift", 3.0)
    w.u32("ace.steps", 8)
    w.u32("ace.sample_rate", 48000)
    w.u32("ace.latent_hop", 1920)

    tokens, types, merges = tokenizer_tables(os.path.join(args.comfyui, "comfy", "text_encoders", "qwen25_tokenizer"))
    w.str("tokenizer.ggml.model", "gpt2")
    w.str("tokenizer.ggml.pre", "qwen2")
    w.strs("tokenizer.ggml.tokens", tokens)
    w.i32s("tokenizer.ggml.token_type", types)
    w.strs("tokenizer.ggml.merges", merges)

    sys.path.insert(0, args.comfyui)
    import torch
    from comfy.ldm.ace.ace_step15 import get_silence_latent
    silence = get_silence_latent(750, "cpu")[0].float().numpy()   # [64, 750]
    w.tensor("ace.silence_latent", silence.astype(np.float32), F32)

    # Weight-normed VAE decoder convolutions: weight = g * v / ||v|| (norm over all axes but the first).
    vae = {}
    for name in st:
        if name.startswith("vae.decoder."):
            vae[name] = st[name]
    folded = set()
    for name in sorted(vae):
        if name.endswith(".parametrizations.weight.original1"):
            base = name[: -len(".parametrizations.weight.original1")]
            g = vae[base + ".parametrizations.weight.original0"][2]().astype(np.float64)
            v = vae[name][2]().astype(np.float64)
            norm = np.sqrt((v.reshape(v.shape[0], -1) ** 2).sum(axis=1)).reshape([-1] + [1] * (v.ndim - 1))
            w.tensor(base + ".weight", (g * v / norm).astype(np.float32), F32)
            folded.update({name, base + ".parametrizations.weight.original0"})
    for name in sorted(vae):
        if name in folded:
            continue
        w.tensor(name, vae[name][2]().astype(np.float32), F32)

    for src in sorted(st):
        dst = None
        for p, q in PREFIXES:
            if src.startswith(p):
                dst = q + src[len(p):]
        if dst is None or dst.startswith(SKIP) or dst.endswith("logit_scale"):
            continue
        dt, shape, load = st[src]
        if dt == "BF16":
            w.tensor(dst, (lambda load=load: load()), BF16, shape)
        elif dt == "F16":
            w.tensor(dst, (lambda load=load: load()), F16, shape)
        else:
            w.tensor(dst, (lambda load=load: load()), F32, shape)
    w.write(args.out, quiet=args.quiet)


if __name__ == "__main__":
    main()
