#!/usr/bin/env python3
"""Inkling 2 Mini's DSpark drafter (mtp.safetensors) as a GGUF TensorSharp loads beside the target.

Layout (ggml order: ne0 is a product's input width):
  dspark.fuse_alpha            F32 [taps, layers + 1]   softmax(fuse_weight) per row (row = draft layer; last = Markov head)
  dspark.mask_embed            F32 [hidden]
  dspark.output_norm           F32 [hidden]
  dspark.blk.L.attn_norm / ffn_norm / dflare_norm      F32 [hidden]
  dspark.blk.L.attn_q_norm / attn_k_norm / attn_k_ctx_norm   F32 [head_dim]
  dspark.blk.L.attn_qkv        Q8_0 [hidden, q + k + v]
  dspark.blk.L.attn_kv_ctx     Q8_0 [hidden, k + v]
  dspark.blk.L.attn_output     Q8_0 [heads * head_dim, hidden]
  dspark.blk.L.ffn_gate_up     Q8_0 [hidden, 2 * ffn]   (gate rows, then up)
  dspark.blk.L.ffn_down        Q8_0 [ffn, hidden]
  dspark.markov.dflare_norm    F32 [hidden]
  dspark.markov.w1             F16 [rank, vocab]        Markov embedding of the previous token
  dspark.markov.rnn_state      F32 [2 rank, 3 rank]     rnn_proj's [state, w1] columns
  dspark.markov.rnn_ctx        Q8_0 [2 hidden, 3 rank]  rnn_proj's [hidden, f_rnn] columns
  dspark.markov.rnn_bias       F32 [3 rank]
  dspark.markov.fold           Q8_0 [rank, vocab]       unembed @ hidden_out.weight (fp64 then rounded)
  dspark.markov.fold_bias      F32 [vocab]              unembed @ hidden_out.bias
Run where torch and safetensors are installed (the vLLM image has both), on the official checkpoint's files:
  python3 eng/inkling2/dspark_to_gguf.py --mtp mtp.safetensors --target model.safetensors --config config.json       --out Inkling-2-Mini-DSpark.gguf
Serve it with the target: --draft-model Inkling-2-Mini-DSpark.gguf --spec-type block (TensorSharp.Server.Host or the CLI).
"""
import argparse
import json
import struct

import numpy as np
import torch
from safetensors import safe_open

F32, F16, Q8_0 = 0, 1, 8


def q8_0(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert x.shape[-1] % 32 == 0
    b = x.reshape(-1, 32)
    amax = np.abs(b).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0.0)
    q = np.clip(np.rint(b * inv[:, None]), -128, 127).astype(np.int8)
    out = np.empty((b.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).reshape(-1, 1).view(np.uint8)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


class Gguf:
    def __init__(self):
        self.kv = []
        self.tensors = []

    def add(self, key, kind, value):
        self.kv.append((key, kind, value))

    def tensor(self, name, array, kind):
        """array: numpy, row-major [rows, ne0] (torch Linear [out, in] is already that)."""
        array = np.asarray(array)
        dims = tuple(reversed(array.shape))
        if kind == F32:
            data = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        elif kind == F16:
            data = np.ascontiguousarray(array, dtype=np.float16).tobytes()
        else:
            data = q8_0(array)
        self.tensors.append((name, dims, kind, data))

    def write(self, path, alignment=32):
        def s(x):
            b = x.encode()
            return struct.pack("<Q", len(b)) + b

        with open(path, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv)))
            for key, kind, value in self.kv:
                f.write(s(key))
                if kind == "u32":
                    f.write(struct.pack("<II", 4, value))
                elif kind == "f32":
                    f.write(struct.pack("<If", 6, value))
                elif kind == "str":
                    f.write(struct.pack("<I", 8) + s(value))
                elif kind == "i32[]":
                    f.write(struct.pack("<IIQ", 9, 5, len(value)) + struct.pack("<" + "i" * len(value), *value))
                elif kind == "bool[]":
                    f.write(struct.pack("<IIQ", 9, 7, len(value)) + bytes(1 if v else 0 for v in value))
            offset = 0
            offsets = []
            for name, dims, kind, data in self.tensors:
                f.write(s(name) + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims))
                f.write(struct.pack("<IQ", kind, offset))
                offsets.append(offset)
                offset += (len(data) + alignment - 1) // alignment * alignment
            f.write(b"\0" * ((f.tell() + alignment - 1) // alignment * alignment - f.tell()))
            for (name, dims, kind, data), off in zip(self.tensors, offsets):
                f.write(data)
                f.write(b"\0" * ((len(data) + alignment - 1) // alignment * alignment - len(data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    mc = cfg["mtp_config"]
    text = cfg.get("text_config", cfg)
    m = safe_open(args.mtp, "pt")
    t = safe_open(args.target, "pt")

    def w(name):
        return m.get_tensor(name).float().numpy()

    hidden, layers, rank = mc["hidden_size"], mc["num_hidden_layers"], mc["markov_rank"]
    heads, kvh, hd, ffn = mc["num_attention_heads"], mc["num_key_value_heads"], mc["head_dim"], mc["intermediate_size"]
    g = Gguf()
    a = "inkling2_dspark"
    g.add("general.architecture", "str", a)
    g.add("general.name", "str", "Inkling 2 Mini DSpark")
    g.add(f"{a}.block_count", "u32", layers)
    g.add(f"{a}.embedding_length", "u32", hidden)
    g.add(f"{a}.attention.head_count", "u32", heads)
    g.add(f"{a}.attention.head_count_kv", "u32", kvh)
    g.add(f"{a}.attention.key_length", "u32", hd)
    g.add(f"{a}.feed_forward_length", "u32", ffn)
    g.add(f"{a}.attention.layer_norm_rms_epsilon", "f32", float(mc["rms_norm_eps"]))
    g.add(f"{a}.rope.freq_base", "f32", float(mc["rope_theta"]))
    g.add(f"{a}.attention.sliding_window", "u32", mc["sliding_window_size"])
    g.add(f"{a}.attention.sliding_window_pattern", "bool[]", [i in mc["local_layer_ids"] for i in range(layers)])
    g.add(f"{a}.block_size", "u32", mc["block_size"])
    g.add(f"{a}.markov_rank", "u32", rank)
    g.add(f"{a}.target_layers", "i32[]", list(mc["target_layer_ids"]))
    g.add(f"{a}.final_logit_softcapping", "f32", float(text.get("final_logit_softcapping", 30.0)))

    fuse = torch.softmax(m.get_tensor("fuse_weight").float(), dim=-1).numpy()   # [layers + 1, taps]
    g.tensor("dspark.fuse_alpha", fuse, F32)
    g.tensor("dspark.mask_embed", w("mask_embed"), F32)
    g.tensor("dspark.output_norm", w("norm.weight"), F32)
    for L in range(layers):
        p = f"layers.{L}."
        g.tensor(f"dspark.blk.{L}.attn_norm", w(p + "input_layernorm.weight"), F32)
        g.tensor(f"dspark.blk.{L}.ffn_norm", w(p + "post_attention_layernorm.weight"), F32)
        g.tensor(f"dspark.blk.{L}.dflare_norm", w(p + "dflare_norm.weight"), F32)
        g.tensor(f"dspark.blk.{L}.attn_q_norm", w(p + "self_attn.q_norm.weight"), F32)
        g.tensor(f"dspark.blk.{L}.attn_k_norm", w(p + "self_attn.k_norm.weight"), F32)
        g.tensor(f"dspark.blk.{L}.attn_k_ctx_norm", w(p + "self_attn.k_norm_ctx.weight"), F32)
        qkv = np.concatenate([w(p + "self_attn.q_proj.weight"), w(p + "self_attn.k_proj.weight"), w(p + "self_attn.v_proj.weight")], axis=0)
        g.tensor(f"dspark.blk.{L}.attn_qkv", qkv, Q8_0)
        kv = np.concatenate([w(p + "self_attn.k_proj_ctx.weight"), w(p + "self_attn.v_proj_ctx.weight")], axis=0)
        g.tensor(f"dspark.blk.{L}.attn_kv_ctx", kv, Q8_0)
        g.tensor(f"dspark.blk.{L}.attn_output", w(p + "self_attn.o_proj.weight"), Q8_0)
        gu = np.concatenate([w(p + "mlp.gate_proj.weight"), w(p + "mlp.up_proj.weight")], axis=0)
        g.tensor(f"dspark.blk.{L}.ffn_gate_up", gu, Q8_0)
        g.tensor(f"dspark.blk.{L}.ffn_down", w(p + "mlp.down_proj.weight"), Q8_0)
        print(f"layer {L}", flush=True)
    g.tensor("dspark.markov.dflare_norm", w("markov_head.dflare_norm.weight"), F32)
    vocab = text.get("padded_vocab_size") or t.get_slice("model.llm.unembed.weight").get_shape()[0]
    w1 = w("markov_head.markov_w1.weight")[:vocab]
    g.tensor("dspark.markov.w1", w1, F16)
    rp = w("markov_head.rnn_proj.weight")                  # [3 rank, 2 rank + 2 hidden]
    g.tensor("dspark.markov.rnn_state", rp[:, : 2 * rank], F32)
    g.tensor("dspark.markov.rnn_ctx", rp[:, 2 * rank:], Q8_0)
    g.tensor("dspark.markov.rnn_bias", w("markov_head.rnn_proj.bias"), F32)
    proj = m.get_tensor("markov_head.hidden_out.weight").double()   # [hidden, rank]
    bias = m.get_tensor("markov_head.hidden_out.bias").double()     # [hidden]
    head = t.get_tensor("model.llm.unembed.weight")                 # [vocab, hidden] bf16
    fold = np.empty((head.shape[0], rank), dtype=np.float32)
    fold_bias = np.empty(head.shape[0], dtype=np.float32)
    for s0 in range(0, head.shape[0], 8192):
        hb = head[s0:s0 + 8192].double()
        fold[s0:s0 + 8192] = (hb @ proj).float().numpy()
        fold_bias[s0:s0 + 8192] = (hb @ bias).float().numpy()
    g.tensor("dspark.markov.fold", fold, Q8_0)
    g.tensor("dspark.markov.fold_bias", fold_bias, F32)
    g.write(args.out)
    print(f"wrote {args.out}: {len(g.tensors)} tensors, vocab {head.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
