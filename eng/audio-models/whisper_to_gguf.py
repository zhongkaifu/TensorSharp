#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Convert an OpenAI Whisper checkpoint in Hugging Face layout into the "whisper" GGUF TensorSharp loads.

Input: a directory with config.json, generation_config.json, tokenizer.json and model.safetensors, for example a
snapshot of openai/whisper-large-v3-turbo (MIT). Then:

    python eng/audio-models/whisper_to_gguf.py --input DIR --out whisper-large-v3-turbo-f16.gguf

The tensors keep their Hugging Face names (model.encoder.conv1.weight, model.decoder.layers.N.encoder_attn.k_proj.weight,
...). Matrices and convolution kernels stay F16 (the release precision); biases, norms and the position tables are F32.
The file is self-contained, as whisper.cpp's are:

    whisper.config / whisper.generation_config   the two JSON files verbatim (special-token ids, language ids,
                                                 suppressed tokens, alignment heads)
    whisper.mel_filters        [n_mels, n_freq]  the Slaney-normalised mel filterbank of WhisperFeatureExtractor
                                                 (librosa's, which openai/whisper ships as mel_filters.npz)
    tokenizer.ggml.*                             the byte-level BPE vocabulary, merges and token types

Only numpy is needed.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_writer import F16, F32, GgufWriter, read_safetensors  # noqa: E402


def hertz_to_mel_slaney(freq):
    freq = np.asarray(freq, dtype=np.float64)
    mels = 3.0 * freq / 200.0
    logstep = 27.0 / np.log(6.4)
    return np.where(freq >= 1000.0, 15.0 + np.log(np.maximum(freq, 1e-10) / 1000.0) * logstep, mels)


def mel_to_hertz_slaney(mels):
    mels = np.asarray(mels, dtype=np.float64)
    freq = 200.0 * mels / 3.0
    return np.where(mels >= 15.0, 1000.0 * np.exp(np.log(6.4) * (mels - 15.0) / 27.0), freq)


def mel_filter_bank(n_freq, n_mels, f_min, f_max, sample_rate):
    """transformers.audio_utils.mel_filter_bank(norm="slaney", mel_scale="slaney"): [n_freq, n_mels]."""
    mel_freqs = np.linspace(hertz_to_mel_slaney(f_min), hertz_to_mel_slaney(f_max), n_mels + 2)
    filter_freqs = mel_to_hertz_slaney(mel_freqs)
    fft_freqs = np.linspace(0, sample_rate // 2, n_freq)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down = -slopes[:, :-2] / filter_diff[:-1]
    up = slopes[:, 2:] / filter_diff[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    enorm = 2.0 / (filter_freqs[2:n_mels + 2] - filter_freqs[:n_mels])
    return filters * enorm[None, :]


def load_vocab(tokenizer_json):
    tok = json.load(open(tokenizer_json, encoding="utf-8"))
    model = tok["model"]
    if model.get("type") != "BPE":
        sys.exit(f"error: expected a BPE tokenizer, got {model.get('type')}")
    id_to_token = {int(i): t for t, i in model["vocab"].items()}
    special = set()
    for added in tok.get("added_tokens", []):
        id_to_token[int(added["id"])] = added["content"]
        if added.get("special"):
            special.add(int(added["id"]))
    size = max(id_to_token) + 1
    missing = [i for i in range(size) if i not in id_to_token]
    if missing:
        sys.exit(f"error: the vocabulary has holes, first at id {missing[0]}")
    tokens = [id_to_token[i] for i in range(size)]
    types = [3 if i in special else 1 for i in range(size)]
    merges = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
    return tokens, types, merges


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Hugging Face Whisper snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg_path = os.path.join(args.input, "config.json")
    gen_path = os.path.join(args.input, "generation_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    gen = json.load(open(gen_path, encoding="utf-8"))
    if cfg.get("model_type") != "whisper":
        sys.exit(f"error: {cfg_path} is model_type {cfg.get('model_type')!r}, not whisper")
    pre_path = os.path.join(args.input, "preprocessor_config.json")
    pre = json.load(open(pre_path, encoding="utf-8")) if os.path.exists(pre_path) else {}
    n_fft = int(pre.get("n_fft", 400))
    sample_rate = int(pre.get("sampling_rate", 16000))
    n_mels = int(cfg["num_mel_bins"])

    w = GgufWriter("whisper")
    w.str("general.name", os.path.basename(os.path.normpath(cfg.get("_name_or_path") or args.input)))
    w.str("whisper.config", json.dumps(cfg))
    w.str("whisper.generation_config", json.dumps(gen))
    w.u32("whisper.n_fft", n_fft)
    w.u32("whisper.hop_length", int(pre.get("hop_length", 160)))
    w.u32("whisper.sample_rate", sample_rate)
    w.u32("whisper.chunk_length", int(pre.get("chunk_length", 30)))

    tokens, types, merges = load_vocab(os.path.join(args.input, "tokenizer.json"))
    if len(tokens) != int(cfg["vocab_size"]):
        sys.exit(f"error: tokenizer has {len(tokens)} tokens, config vocab_size is {cfg['vocab_size']}")
    w.str("tokenizer.ggml.model", "gpt2")
    w.str("tokenizer.ggml.pre", "whisper")
    w.strs("tokenizer.ggml.tokens", tokens)
    w.i32s("tokenizer.ggml.token_type", types)
    w.strs("tokenizer.ggml.merges", merges)
    w.u32("tokenizer.ggml.eos_token_id", int(cfg["eos_token_id"]))
    w.u32("tokenizer.ggml.bos_token_id", int(cfg["bos_token_id"]))

    filters = mel_filter_bank(1 + n_fft // 2, n_mels, 0.0, sample_rate / 2.0, sample_rate)
    w.tensor("whisper.mel_filters", filters.T.astype(np.float32), F32)  # [n_mels, n_freq]

    st = read_safetensors(os.path.join(args.input, "model.safetensors"))
    for name in sorted(st):
        dt, shape, load = st[name]
        if name.startswith("proj_out."):
            continue  # tied to model.decoder.embed_tokens
        gtype = F16 if len(shape) >= 2 and "embed_positions" not in name else F32
        w.tensor(name, (lambda load=load: load()), gtype, shape)
    w.write(args.out, quiet=args.quiet)


if __name__ == "__main__":
    main()
