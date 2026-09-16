#!/usr/bin/env python3
"""Slow, independent float32 reference for DeepSeek V4.1 GGUF correctness checks.

Equations follow deepseek-ai/DeepSeek-V4.1-Flash/inference/{model,engram}.py.
Weights stay memory mapped; only matrix row blocks and selected expert slices
are dequantized. This checks GGUF execution, not the original FP8 checkpoint's
activation quantization or its quality. Requires torch, numpy, and gguf.
"""
import argparse
import json
import math
import re
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class GgufWeights:
    def __init__(self, path, device="cpu", row_block=512):
        from gguf import GGUFReader
        from gguf.quants import dequantize, Q2_K
        self.dequantize = dequantize
        self.q2_blocks = Q2_K.dequantize_blocks
        self.device = torch.device(device)
        self.row_block = row_block
        path = Path(path)
        match = re.fullmatch(r"(.*)-\d{5}-of-(\d{5})\.gguf", path.name)
        paths = [path] if not match else [path.with_name(f"{match[1]}-{i:05}-of-{int(match[2]):05}.gguf")
                                         for i in range(1, int(match[2]) + 1)]
        self.readers = [GGUFReader(str(shard), mode="r") for shard in paths]
        self.tensors = {tensor.name: tensor for reader in self.readers for tensor in reader.tensors}
        self.small_cache = {}

    def shape(self, name):
        return tuple(int(x) for x in self.tensors[name].shape)

    def rows(self, name, row_ids):
        tensor = self.tensors[name]
        # Some NumPy versions promote Python-int * np.uint64 to float64.
        # Keep dimensions as Python integers so reshape receives an integer.
        row_count = math.prod(int(dimension) for dimension in tensor.shape[1:]) or 1
        raw = tensor.data.reshape(row_count, -1)[row_ids]
        raw = np.ascontiguousarray(raw)
        # Whole matrix blocks avoid the generic converter's Python loop over
        # 16-row groups; selected expert projections otherwise spend most of
        # their validation time in that dispatch loop.
        if tensor.tensor_type.name == "Q2_K":
            values = self.q2_blocks(raw.reshape(-1, 84)).reshape(raw.shape[0], int(tensor.shape[0]))
        else:
            values = self.dequantize(raw, tensor.tensor_type)
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(self.device)

    def weight(self, name):
        if name not in self.small_cache:
            value = self.rows(name, slice(None))
            if value.numel() <= 1048576:
                self.small_cache[name] = value
            return value
        return self.small_cache[name]

    def mm(self, name, x, expert=None, out_features=None):
        shape = self.shape(name)
        out_features = out_features or (shape[1] if len(shape) > 1 else 1)
        offset = 0 if expert is None else int(expert) * out_features
        output = torch.empty((x.shape[0], out_features), dtype=torch.float32, device=x.device)
        for row in range(0, out_features, self.row_block):
            stop = min(row + self.row_block, out_features)
            w = self.rows(name, slice(offset + row, offset + stop))
            output[:, row:stop] = F.linear(x.float(), w)
        return output


def load_engram(path):
    with open(path, "rb") as source:
        if source.read(8) != b"TSD41E01":
            raise ValueError("Invalid Engram sidecar")
        def read(fmt):
            return struct.unpack("<" + fmt, source.read(struct.calcsize("<" + fmt)))
        vocab, compressed, pad, layers, max_ngram, heads, dim, fingerprint, candidate, topk, block, nkv, nidx = read("7IQi4I")
        kv_sources, index_sources = list(read(f"{nkv}i")), list(read(f"{nidx}i"))
        token_map = np.frombuffer(source.read(4 * vocab), dtype="<i4").copy()
        layouts = []
        for _ in range(layers):
            layer_id, rows = read("iQ")
            multipliers = np.array(read(f"{max_ngram}Q"), dtype=np.int64)
            primes = np.array(read(f"{(max_ngram - 1) * heads}I"), dtype=np.int64).reshape(max_ngram - 1, heads)
            offsets = np.array(read(f"{(max_ngram - 1) * heads}Q"), dtype=np.int64)
            layouts.append(dict(id=layer_id, rows=rows, multipliers=multipliers, primes=primes, offsets=offsets))
        return dict(token_map=token_map, pad=pad, max_ngram=max_ngram, heads=heads, dim=dim, layouts=layouts,
                    kv_sources=kv_sources, index_sources=index_sources, candidate=candidate,
                    candidate_topk=topk, candidate_block=block, fingerprint=fingerprint, compressed=compressed)


def rms(x, epsilon, weight=None):
    output = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
    return output if weight is None else output * weight


def hc_mixes(x, fn, scale, base, iterations, epsilon, rms_epsilon):
    flat = x.flatten(1)
    projection = fn(flat) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_epsilon)
    streams = x.shape[1]
    pre = torch.sigmoid(projection[:, :streams] * scale[0] + base[:streams]) + epsilon
    post = 2 * torch.sigmoid(projection[:, streams:2 * streams] * scale[1] + base[streams:2 * streams])
    comb = (projection[:, 2 * streams:] * scale[2] + base[2 * streams:]).reshape(-1, streams, streams)
    comb = comb.softmax(-1) + epsilon
    comb = comb / (comb.sum(-2, keepdim=True) + epsilon)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + epsilon)
        comb = comb / (comb.sum(-2, keepdim=True) + epsilon)
    return pre, post, comb


class Reference:
    def __init__(self, weights, config, engram, cache_type="f16", output_dir=None):
        self.w, self.c, self.engram = weights, config.get("text_config", config), engram
        self.device = weights.device
        self.eps = self.c["rms_norm_eps"]
        self.cache_type = cache_type
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        self.states = [{} for _ in range(self.c["num_hidden_layers"])]
        self.comp_source = self.index_source = None
        self.indices = self.candidates = None
        self.position = 0

    def dump(self, name, tensor):
        if self.output_dir:
            data = tensor.detach().float().cpu().numpy()
            np.save(self.output_dir / f"p{self.position:06}_{name}.npy", data)
            data.astype("<f4").tofile(self.output_dir / f"p{self.position:06}_{name}.f32")

    def cache(self, x, kind="raw"):
        if self.cache_type != "model":
            return x.half().float() if self.cache_type == "f16" else x.float()
        # Training kernels take BF16, including the rounding before block amax.
        block = 16 if kind == "compressed" else 32
        z = x.bfloat16().float().reshape(*x.shape[:-1], -1, block)
        maximum = z.abs().amax(-1, keepdim=True)
        if kind == "raw":
            scale = 2 ** torch.ceil(torch.log2(maximum.clamp_min(1e-4) / 448))
            q = (z / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        else:
            if kind == "compressed":
                scale = (maximum.clamp_min(6 * 2 ** -9) / 6).to(torch.float8_e4m3fn).float()
            else:
                scale = 2 ** torch.ceil(torch.log2(maximum.clamp_min(6 * 2 ** -126) / 6))
            normalized = (z / scale).clamp(-6, 6)
            levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=z.device)
            midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5], device=z.device)
            index = torch.bucketize(normalized.abs().contiguous(), midpoints)
            tie = (normalized.abs() == midpoints[index.clamp(max=6)]) & (index % 2 == 1)
            index += tie
            q = torch.copysign(levels[index], normalized)
        return (q * scale).reshape(x.shape).bfloat16().float()

    def rope(self, x, positions, ratio, inverse=False):
        dim = self.c["qk_rope_head_dim"]
        base = self.c["compress_rope_theta"] if ratio else self.c["rope_theta"]
        freq = 1 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        if ratio:
            yarn = self.c["rope_scaling"]
            original = yarn["original_max_position_embeddings"]
            def corr(rotations):
                return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
            lo, hi = max(math.floor(corr(yarn["beta_fast"])), 0), min(math.ceil(corr(yarn["beta_slow"])), dim - 1)
            ramp = ((torch.arange(dim // 2, device=x.device).float() - lo) / max(hi - lo, 1e-3)).clamp(0, 1)
            freq = freq * (1 - ramp) + freq / yarn["factor"] * ramp
        angles = torch.as_tensor(positions, device=x.device).float().reshape(-1, 1) * freq
        if inverse:
            angles = -angles
        while angles.ndim < x.ndim:
            angles = angles.unsqueeze(1)
        result = x.clone()
        pairs = x[..., -dim:].reshape(*x.shape[:-1], dim // 2, 2)
        result[..., -dim::2] = pairs[..., 0] * angles.cos() - pairs[..., 1] * angles.sin()
        result[..., -dim + 1::2] = pairs[..., 0] * angles.sin() + pairs[..., 1] * angles.cos()
        return result

    def eng_inject(self, layer, h, tokens):
        layout = next((x for x in self.engram["layouts"] if x["id"] == layer), None)
        if layout is None:
            return h
        hashed = []
        mapping = self.engram["token_map"]
        for offset in range(len(tokens)):
            pos = self.position + offset
            history, blocked = [], False
            for shift in range(self.engram["max_ngram"]):
                blocked = blocked or pos < shift or self.history[pos - shift] < 0
                history.append(mapping[self.engram["pad"]] if blocked else mapping[self.history[pos - shift]])
            products = np.asarray(history, dtype=np.int64) * layout["multipliers"]
            rows = [np.bitwise_xor.reduce(products[:n]) % layout["primes"][n - 2]
                    for n in range(2, self.engram["max_ngram"] + 1)]
            hashed.extend((np.concatenate(rows) + layout["offsets"]).tolist())
        prefix = f"blk.{layer}.engram_"
        embeddings = self.w.rows(prefix + "embd.weight", hashed).reshape(len(tokens), -1)
        self.dump(f"blk{layer:02}_engram_embeddings", embeddings)
        kv = self.w.mm(prefix + "wkv.weight", embeddings)
        self.dump(f"blk{layer:02}_engram_kv", kv)
        dim, streams = self.c["hidden_size"], self.c["hc_mult"]
        key, value = kv[:, :streams * dim].reshape(-1, streams, dim), kv[:, streams * dim:]
        weight = self.w.weight(prefix + "q.weight").reshape(streams, dim) * self.w.weight(prefix + "k.weight").reshape(streams, dim)
        dot = (rms(h, self.eps) * rms(key, self.eps) * weight).sum(-1) * dim ** -0.5
        self.dump(f"blk{layer:02}_engram_dot", dot)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        self.dump(f"blk{layer:02}_engram_gate", gate)
        mask = torch.tensor([t >= 0 for t in tokens], device=h.device)
        result = h + gate.unsqueeze(-1) * value.unsqueeze(1) * mask[:, None, None]
        self.dump(f"blk{layer:02}_engram_out", result)
        return result

    def attention(self, layer, x):
        c, state = self.c, self.states[layer]
        prefix = f"blk.{layer}."
        nt, ratio = x.shape[0], c["compress_ratios"][layer]
        positions = list(range(self.position, self.position + nt))
        qr = rms(self.w.mm(prefix + "attn_q_a.weight", x), self.eps, self.w.weight(prefix + "attn_q_a_norm.weight").flatten())
        q = self.w.mm(prefix + "attn_q_b.weight", qr).reshape(nt, c["num_attention_heads"], c["head_dim"])
        q = self.rope(q, positions, ratio)
        kv = rms(self.w.mm(prefix + "attn_kv.weight", x), self.eps, self.w.weight(prefix + "attn_kv_a_norm.weight").flatten())
        kv = self.cache(self.rope(kv, positions, ratio))
        state["raw"] = torch.cat([state["raw"], kv]) if "raw" in state else kv
        self.dump(f"blk{layer:02}_q", q)
        self.dump(f"blk{layer:02}_raw_k", kv)
        latent = None
        if ratio and layer in self.engram["kv_sources"]:
            self.comp_source = layer
            projected = self.w.mm(prefix + "attn_compressor_kv.weight", x)
            if ratio > 1:
                gates = self.w.mm(prefix + "attn_compressor_gate.weight", x)
                state["pool_k"] = torch.cat([state["pool_k"], projected]) if "pool_k" in state else projected
                state["pool_g"] = torch.cat([state["pool_g"], gates]) if "pool_g" in state else gates
                groups = state["pool_k"].shape[0] // ratio
                if groups:
                    values = state["pool_k"][:groups * ratio].reshape(groups, ratio, -1)
                    scores = state["pool_g"][:groups * ratio].reshape(groups, ratio, -1)
                    projected = (values * scores.softmax(1)).sum(1)
                    state["pool_k"], state["pool_g"] = state["pool_k"][groups * ratio:], state["pool_g"][groups * ratio:]
                else:
                    projected = None
            if projected is not None:
                latent = rms(projected, self.eps, self.w.weight(prefix + "attn_compressor_norm.weight").flatten())
                previous = 0 if "comp" not in state else state["comp"].shape[0]
                comp_positions = list(range(previous * ratio, (previous + latent.shape[0]) * ratio, ratio))
                compressed = self.cache(self.rope(latent, comp_positions, ratio), "compressed")
                state["comp"] = torch.cat([state["comp"], compressed]) if "comp" in state else compressed
                index_k = rms(self.w.mm(prefix + "indexer.attn_k.weight", latent), self.eps,
                              self.w.weight(prefix + "indexer.k_norm.weight").flatten())
                index_k = self.cache(self.rope(index_k, comp_positions, ratio), "index")
                state["index_k"] = torch.cat([state["index_k"], index_k]) if "index_k" in state else index_k
                self.dump(f"blk{layer:02}_compress_latent", latent)
        if ratio and layer in self.engram["index_sources"]:
            source = self.states[self.comp_source]
            index_q = self.w.mm(prefix + "indexer.attn_q_b.weight", qr).reshape(nt, c["index_n_heads"], c["index_head_dim"])
            index_q = self.cache(self.rope(index_q, positions, ratio), "index")
            weights = self.w.mm(prefix + "indexer.proj.weight", x) * (c["index_head_dim"] * c["index_n_heads"]) ** -0.5
            index_k = source.get("index_k", torch.empty((0, c["index_head_dim"]), device=x.device))
            scores = (torch.einsum("thd,kd->thk", index_q, index_k).relu() * weights.unsqueeze(-1)).sum(1)
            self.indices = []
            if layer == self.engram["candidate"]:
                self.candidates = []
            for token, position in enumerate(positions):
                visible = (position + 1) // ratio
                row = scores[token, :visible].clone()
                if layer == self.engram["candidate"]:
                    block = self.engram["candidate_block"]
                    if visible:
                        blocks = F.pad(row, (0, -visible % block), value=-torch.inf).reshape(-1, block).amax(-1)
                        blocks[-1] = torch.inf
                        chosen = blocks.argsort(descending=True, stable=True)[:self.engram["candidate_topk"]]
                        keep = torch.zeros_like(blocks, dtype=torch.bool)
                        keep[chosen] = True
                        self.candidates.append(keep.repeat_interleave(block)[:visible])
                    else:
                        self.candidates.append(torch.empty(0, dtype=torch.bool, device=x.device))
                elif self.engram["candidate"] >= 0 and layer > self.engram["candidate"]:
                    row.masked_fill_(~self.candidates[token], -torch.inf)
                # GGML resolves equal scores by the lowest position. Torch's
                # topk tie order differs across CPU/CUDA and is unspecified.
                indices = row.argsort(descending=True, stable=True)[:c["index_topk"]].sort().values
                self.indices.append(indices)
                scores[token, :visible] = row
                scores[token, visible:] = -torch.inf
            self.dump(f"blk{layer:02}_index_scores", scores)
            padded = torch.full((nt, c["index_topk"]), -1, dtype=torch.int32, device=x.device)
            for token, selected in enumerate(self.indices):
                padded[token, :selected.numel()] = selected
            self.dump(f"blk{layer:02}_index_top", padded)
        output = []
        for token, position in enumerate(positions):
            keys = state["raw"][max(0, position - c["sliding_window"] + 1):position + 1]
            if ratio and self.indices[token].numel():
                keys = torch.cat([keys, self.states[self.comp_source]["comp"][self.indices[token]]])
            scores = q[token] @ keys.T * c["head_dim"] ** -0.5
            sink = self.w.weight(prefix + "attn_sinks.weight").flatten().unsqueeze(1)
            probs = torch.cat([scores, sink], dim=-1).softmax(-1)[:, :-1]
            output.append(probs @ keys)
        output = self.rope(torch.stack(output), positions, ratio, inverse=True)
        groups = c["o_groups"]
        output = output.reshape(nt, groups, -1)
        projected = torch.cat([self.w.mm(prefix + "attn_output_a.weight", output[:, group], group, c["o_lora_rank"])
                               for group in range(groups)], dim=-1)
        return self.w.mm(prefix + "attn_output_b.weight", projected)

    def moe(self, layer, x):
        c, prefix = self.c, f"blk.{layer}."
        scores = F.softplus(self.w.mm(prefix + "ffn_gate_inp.weight", x)).sqrt()
        bias = self.w.weight(prefix + "exp_probs_b.bias").flatten()
        if self.image_mask.any():
            visual_bias = self.vision_weights.weight(f"layers.{layer}.ffn.gate.bias_vl").flatten()
            bias = torch.where(self.image_mask[:, None], visual_bias, bias)
        indices = (scores + bias).topk(c["num_experts_per_tok"], dim=-1).indices
        weights = scores.gather(1, indices)
        if c["norm_topk_prob"] and c["num_experts_per_tok"] > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights *= c["routed_scaling_factor"]
        self.dump(f"blk{layer:02}_expert_ids", indices)
        def expert(inp, suffix, index=None):
            gate = self.w.mm(prefix + f"ffn_gate_{suffix}.weight", inp, index)
            up = self.w.mm(prefix + f"ffn_up_{suffix}.weight", inp, index)
            limit = c.get("swiglu_limit", 0)
            if limit > 0:
                gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
            return F.silu(gate) * up
        result = self.w.mm(prefix + "ffn_down_shexp.weight", expert(x, "shexp"))
        for index in indices.unique().tolist():
            token, rank = torch.where(indices == index)
            activated = expert(x[token], "exps", index) * weights[token, rank, None]
            result[token] += self.w.mm(prefix + "ffn_down_exps.weight", activated, index)
        return result

    @torch.inference_mode()
    def forward(self, tokens, image_embeddings=None, image_mask=None, vision_weights=None):
        c = self.c
        self.image_mask = torch.zeros(len(tokens), dtype=torch.bool, device=self.device) if image_mask is None else torch.as_tensor(image_mask, dtype=torch.bool, device=self.device)
        if self.image_mask.shape != (len(tokens),):
            raise ValueError("Image mask does not match input tokens")
        if vision_weights is not None:
            self.vision_weights = vision_weights
        hash_tokens = [token if not visual else -1 for token, visual in zip(tokens, self.image_mask.tolist())]
        self.history.extend(hash_tokens)
        embedding = self.w.rows("token_embd.weight", tokens)
        if self.image_mask.any():
            if image_embeddings is None or not hasattr(self, "vision_weights"):
                raise ValueError("Image positions require embeddings and VL router weights")
            visual = torch.as_tensor(image_embeddings, dtype=torch.float32, device=self.device)
            if visual.shape != (int(self.image_mask.sum()), c["hidden_size"]):
                raise ValueError("Image embedding rows do not match image mask")
            embedding[self.image_mask] = visual
        h = embedding.unsqueeze(1).repeat(1, c["hc_mult"], 1)
        pre = torch.full((len(tokens), c["hc_mult"]), 1 / c["hc_mult"], device=h.device)
        self.dump("embedding", h)
        for layer in range(c["num_hidden_layers"]):
            started = time.monotonic()
            prefix = f"blk.{layer}."
            h = self.eng_inject(layer, h, hash_tokens)
            for kind, operation in (("attn", self.attention), ("ffn", self.moe)):
                next_pre, post, comb = hc_mixes(h, lambda z: self.w.mm(prefix + f"hc_{kind}_fn.weight", z),
                    self.w.weight(prefix + f"hc_{kind}_scale.weight").flatten(),
                    self.w.weight(prefix + f"hc_{kind}_base.weight").flatten(),
                    c["hc_sinkhorn_iters"], c["hc_eps"], self.eps)
                collapsed = (h * pre.unsqueeze(-1)).sum(1)
                x = rms(collapsed, self.eps, self.w.weight(prefix + f"{kind}_norm.weight").flatten())
                self.dump(f"blk{layer:02}_{kind}_input", x)
                y = operation(layer, x)
                self.dump(f"blk{layer:02}_{kind}_out", y)
                h = post.unsqueeze(-1) * y.unsqueeze(1) + torch.einsum("tij,tid->tjd", comb, h)
                pre = next_pre
            self.dump(f"blk{layer:02}_hidden", h)
            print(f"layer {layer}: {time.monotonic() - started:.2f}s, hidden RMS={h.square().mean().sqrt().item():.7g}", flush=True)
        collapsed = (h * pre.unsqueeze(-1)).sum(1)
        output = rms(collapsed, self.eps, self.w.weight("output_norm.weight").flatten())
        logits = self.w.mm("output.weight", output)
        self.dump("logits", logits)
        self.position += len(tokens)
        return logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", required=True, help="Comma-separated token ids, or a JSON file containing a token array")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--row-block", type=int, default=512)
    parser.add_argument("--cache-type", choices=("f16", "f32", "model"), default="model")
    parser.add_argument("--chunk-size", type=int, default=0, help="0 runs one prefill; smaller values check chunk/cache behavior")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    text = Path(args.tokens).read_text() if args.tokens.endswith(".json") else args.tokens
    tokens = json.loads(text) if text.strip().startswith("[") else [int(value) for value in text.split(",")]
    config = json.loads((args.model.parent / "deepseek41.config.json").read_text())["config"]
    engram = load_engram(args.model.parent / "deepseek41.engram.bin")
    weights = GgufWeights(args.model, args.device, args.row_block)
    reference = Reference(weights, config, engram, args.cache_type, args.output)
    chunk = args.chunk_size or len(tokens)
    outputs = [reference.forward(tokens[start:start + chunk]) for start in range(0, len(tokens), chunk)]
    logits = torch.cat(outputs).float().cpu().numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "logits.npy", logits)
    logits.astype("<f4").tofile(args.output / "logits.f32")
    print(json.dumps({"tokens": tokens, "top_tokens": np.argsort(logits[-1])[-10:][::-1].tolist(),
                      "last_logits_l2": float(np.linalg.norm(logits[-1])), "cache_type": args.cache_type}, indent=2))


if __name__ == "__main__":
    main()
