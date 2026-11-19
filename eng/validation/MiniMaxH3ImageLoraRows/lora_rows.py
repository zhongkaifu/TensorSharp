#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Row parity of MiniMax-H3-Image LoRA plug-ins against an independent NumPy merge (docs/design/diffusion-runtime.md 7.3).

TensorSharp side: MiniMaxH3ImageAB --lora-rows <rows.json> runs each updated projection of some blocks as the backend
applies it (a LoraLinear region over the packed factors) on activations drawn from a 32-bit LCG, and records y = W x +
update and y_base = W x for the first tokens.

This side reads the same LoRA file *independently of TensorSharp's loader* (its safetensors header and only the tensors
and rows it needs, by HTTP range requests or from a local file), maps the names to checkpoint rows by the published
conversions, not by TensorSharp's code:
  * diffusers PEFT (diffusers' convert_minimax_h3_transformer_checkpoint_to_diffusers): attn.qkv_proj is stacked
    [q; k; v], so checkpoint row j of qkv is to_q / to_k / to_v row j mod inner; mlp.fc1 is [gate; value] while
    ff.net.0.proj is [value; gate], so checkpoint row j is ff.net.0.proj row (j + ffn) mod 2 ffn; alpha from the metadata;
  * ComfyUI native / kohya: the checkpoint's names and rows as stored, alpha from each module's .alpha (else the rank),
and computes delta = scale * B[rows] (A x) and y = W[rows] x + delta in float64, W dequantized here from the local base
checkpoint (GGUF Q8_0 / F16 / BF16 / F32, or the safetensors release). It reports the relative errors of TensorSharp's
update (y - y_base) and of its y.

  python3 -I lora_rows.py --rows rows.json --lora <url|path> --model <dit.gguf|.safetensors> [--scale 1] [--out report.json]
"""
import argparse
import json
import math
import os
import struct
import sys
import urllib.request

import numpy as np

# ---- byte sources -------------------------------------------------------------------------------------------------


class Source:
    """A file or URL read by byte ranges (HTTP Range for a URL)."""

    def __init__(self, where):
        self.where = where
        self.remote = where.startswith("http://") or where.startswith("https://")
        self.fetched = 0
        if not self.remote:
            self.f = open(where, "rb")

    def read(self, start, length):
        if length <= 0:
            return b""
        self.fetched += length
        if not self.remote:
            self.f.seek(start)
            return self.f.read(length)
        req = urllib.request.Request(self.where, headers={"Range": f"bytes={start}-{start + length - 1}", "User-Agent": "range-reader"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != length:
                    raise IOError(f"short read {len(data)} of {length}")
                return data
            except Exception:
                if attempt == 4:
                    raise


DTYPES = {"F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (None, 2), "F64": (np.float64, 8), "I64": (np.int64, 8),
          "I32": (np.int32, 4)}


def decode(raw, dtype):
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64)
    t, _ = DTYPES[dtype]
    return np.frombuffer(raw, dtype=t).astype(np.float64)


class Safetensors:
    def __init__(self, where):
        self.src = Source(where)
        n = struct.unpack("<Q", self.src.read(0, 8))[0]
        self.header = json.loads(self.src.read(8, n))
        self.base = 8 + n
        self.meta = self.header.get("__metadata__", {})

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def shape(self, name):
        return self.header[name]["shape"]

    def tensor(self, name):
        h = self.header[name]
        a, b = h["data_offsets"]
        return decode(self.src.read(self.base + a, b - a), h["dtype"]).reshape(h["shape"])

    def rows(self, name, rows):
        """Rows of a 2-D tensor [out, cols] (each row one range read)."""
        h = self.header[name]
        cols = h["shape"][1]
        size = DTYPES[h["dtype"]][1]
        a = h["data_offsets"][0]
        return np.stack([decode(self.src.read(self.base + a + r * cols * size, cols * size), h["dtype"]) for r in rows])


# ---- the base checkpoint's rows -------------------------------------------------------------------------------------

GGML_TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 30: ("BF16", 1, 2)}


class Gguf:
    def __init__(self, path):
        self.f = open(path, "rb")
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", self.f.read(24))
        if magic != 0x46554747:
            raise ValueError("not a GGUF")
        alignment = 32
        for _ in range(n_kv):
            key = self._string()
            vtype = struct.unpack("<I", self.f.read(4))[0]
            value = self._value(vtype)
            if key == "general.alignment":
                alignment = int(value)
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._string()
            nd = struct.unpack("<I", self.f.read(4))[0]
            ne = struct.unpack(f"<{nd}Q", self.f.read(8 * nd))
            t, off = struct.unpack("<IQ", self.f.read(12))
            self.tensors[name] = (list(ne), t, off)
        pos = self.f.tell()
        self.data = (pos + alignment - 1) // alignment * alignment

    def _string(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8")

    def _value(self, t):
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in fmt:
            return struct.unpack(fmt[t], self.f.read(struct.calcsize(fmt[t])))[0]
        if t == 8:
            return self._string()
        if t == 9:
            et = struct.unpack("<I", self.f.read(4))[0]
            n = struct.unpack("<Q", self.f.read(8))[0]
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"GGUF value type {t}")

    def rows(self, name, rows):
        ne, t, off = self.tensors[name]
        kind, block, size = GGML_TYPES[t]
        cols = ne[0]
        row_bytes = cols // block * size
        out = []
        for r in rows:
            self.f.seek(self.data + off + r * row_bytes)
            raw = self.f.read(row_bytes)
            if kind == "Q8_0":
                blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
                d = blocks[:, :2].copy().view(np.float16).astype(np.float64)
                q = blocks[:, 2:].copy().view(np.int8).astype(np.float64)
                out.append((d * q).reshape(-1))
            else:
                out.append(decode(raw, kind))
        return np.stack(out)


def open_base(path):
    if path.endswith(".gguf"):
        return Gguf(path)
    st = Safetensors(path)

    class Rows:
        def rows(self, name, rows):
            return st.rows(name, rows)
    return Rows()


# ---- the activations TensorSharp drew --------------------------------------------------------------------------------


def activations(inputs, tokens, seed):
    """The harness's x: a 32-bit LCG mapped to [-1, 1), [tokens][inputs]."""
    n = inputs * tokens
    out = np.empty(n, dtype=np.float32)
    state = seed & 0xFFFFFFFF
    # Vectorized LCG: x_{k+1} = a x_k + c mod 2^32, done in chunks with uint64 arithmetic.
    a, c = 1664525, 1013904223
    for k in range(n):
        state = (a * state + c) & 0xFFFFFFFF
        out[k] = np.float32((state >> 8) / 16777216.0 * 2.0 - 1.0)
    return out.reshape(tokens, inputs).astype(np.float64)


# ---- the LoRA file's mapping ----------------------------------------------------------------------------------------


def grammar_of(names):
    if any(n.startswith("lora_unet_") for n in names):
        return "kohya"
    if any(n.startswith("diffusion_model.") or n.startswith("blocks.") for n in names):
        return "comfy"
    return "diffusers"


def first(st, *names):
    for n in names:
        if n in st.header:
            return n
    return None


def terms_for(st, grammar, refiner, block, projection, inner, ffn):
    """[(A name, B name, scale, row_map)] for a checkpoint projection; row_map(j) -> (B row) or None."""
    alpha_meta = st.meta.get("alpha") or st.meta.get("lora_alpha")
    if grammar == "diffusers":
        pre = "transformer." if any(n.startswith("transformer.") for n in st.names()) else ""
        scope = f"{pre}{'token_refiner.refiner_blocks' if refiner else 'transformer_blocks'}.{block}."

        def pair(module):
            a = first(st, scope + module + ".lora_A.weight", scope + module + ".lora_A.default.weight")
            b = first(st, scope + module + ".lora_B.weight", scope + module + ".lora_B.default.weight")
            if a is None:
                return None
            rank = st.shape(a)[0]
            scale = (float(alpha_meta) / rank) if alpha_meta else 1.0
            return a, b, scale
        if projection == "attn.qkv_proj":
            out = []
            for s, m in enumerate(["attn.to_q", "attn.to_k", "attn.to_v"]):
                p = pair(m)
                if p:
                    out.append((*p, (lambda s: lambda j: (j - s * inner) if s * inner <= j < (s + 1) * inner else None)(s)))
            return out
        module = {"attn.out_proj": "attn.to_out.0", "mlp.fc1": "ff.net.0.proj", "mlp.fc2": "ff.net.2"}[projection]
        p = pair(module)
        if not p:
            return []
        if projection == "mlp.fc1":
            return [(*p, lambda j: (j + ffn) % (2 * ffn))]
        return [(*p, lambda j: j)]
    if grammar == "comfy":
        stems = [f"diffusion_model.{'token_refiner.blocks' if refiner else 'blocks'}.{block}.{projection}",
                 f"{'token_refiner.blocks' if refiner else 'blocks'}.{block}.{projection}"]
        downs, ups = (".lora_A.weight", ".lora_down.weight"), (".lora_B.weight", ".lora_up.weight")
    else:
        stems = [f"lora_unet_{'token_refiner_blocks' if refiner else 'blocks'}_{block}_{projection.replace('.', '_')}"]
        downs, ups = (".lora_down.weight",), (".lora_up.weight",)
    for stem in stems:
        a = first(st, *[stem + d for d in downs])
        if a is None:
            continue
        b = first(st, *[stem + u for u in ups])
        rank = st.shape(a)[0]
        alpha = float(st.tensor(stem + ".alpha").reshape(-1)[0]) if (stem + ".alpha") in st.header else float(rank)
        return [(a, b, alpha / rank, lambda j: j)]
    return []


def b64(v):
    import base64
    return np.frombuffer(base64.b64decode(v), dtype=np.float32).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--sample", type=int, default=24, help="random rows per projection, beside the thirds' and halves' edges")
    ap.add_argument("--out")
    a = ap.parse_args()
    ts = json.load(open(a.rows))
    st = Safetensors(a.lora)
    base = open_base(a.model)
    grammar = grammar_of(st.names())
    rng = np.random.default_rng(1234)
    report = {"lora": a.lora, "grammar": grammar, "metadata": st.meta, "scale": a.scale, "rows_json": a.rows, "entries": []}
    print(f"{os.path.basename(a.lora)}: {grammar} names, {len(st.names())} tensors")
    worst_delta = worst_y = 0.0
    for e in ts["lora_rows"]:
        inp, out, tokens, kept = e["in"], e["out"], e["tokens"], e["kept_tokens"]
        inner = out // 3 if e["projection"] == "attn.qkv_proj" else None
        ffn = out // 2 if e["projection"] == "mlp.fc1" else 0
        if inner is None:
            inner = 7168
        edges = {"attn.qkv_proj": [0, 1, inner - 1, inner, inner + 1, 2 * inner - 1, 2 * inner, 3 * inner - 1],
                 "mlp.fc1": [0, 1, ffn - 1, ffn, ffn + 1, 2 * ffn - 1]}.get(e["projection"], [0, out - 1])
        rows = sorted(set(edges) | set(int(r) for r in rng.choice(out, size=a.sample, replace=False)))
        terms = terms_for(st, grammar, e["refiner"], e["block"], e["projection"], inner, ffn)
        x = activations(inp, kept, e["x_seed"])            # [kept][in]
        delta = np.zeros((kept, len(rows)))
        for an, bn, scale, row_map in terms:
            A = st.tensor(an)                              # [rank, in]
            mapped = [(i, row_map(r)) for i, r in enumerate(rows)]
            mapped = [(i, m) for i, m in mapped if m is not None]
            if not mapped:
                continue
            B = st.rows(bn, [m for _, m in mapped])        # [n, rank]
            ax = x @ A.T                                   # [kept][rank]
            d = (ax @ B.T) * scale * a.scale               # [kept][n]
            for k, (i, _) in enumerate(mapped):
                delta[:, i] += d[:, k]
        W = base.rows(e["tensor"], rows)                   # [rows][in]
        y_ref = x @ W.T + delta
        y = b64(e["y"]).reshape(kept, out)[:, rows]
        y0 = b64(e["y_base"]).reshape(kept, out)[:, rows]
        dts = y - y0
        dmax = float(np.max(np.abs(delta)))
        derr = float(np.max(np.abs(dts - delta))) / dmax if dmax > 0 else float("nan")
        dl2 = float(np.linalg.norm(dts - delta) / np.linalg.norm(delta)) if dmax > 0 else float("nan")
        yerr = float(np.max(np.abs(y - y_ref)) / np.max(np.abs(y_ref)))
        ybase = float(np.max(np.abs(y0 - x @ W.T)) / np.max(np.abs(x @ W.T)))
        worst_delta, worst_y = max(worst_delta, derr), max(worst_y, yerr)
        where = f"{'refiner ' if e['refiner'] else ''}block {e['block']} {e['projection']}"
        print(f"  {where:28s} {len(terms)} term(s), {len(rows)} rows x {kept} tokens: update rel max {derr:.3e} (L2 {dl2:.3e}), "
              f"|update|/|Wx| {dmax / float(np.max(np.abs(x @ W.T))):.3e}; y rel max {yerr:.3e} (base alone {ybase:.3e})")
        report["entries"].append({"where": where, "terms": len(terms), "rows": rows, "update_rel_max": derr, "update_rel_l2": dl2,
                                  "update_over_wx": dmax / float(np.max(np.abs(x @ W.T))), "y_rel_max": yerr, "base_rel_max": ybase,
                                  "delta_ref": delta.tolist(), "delta_ts": dts.tolist()})
    report["worst_update_rel_max"], report["worst_y_rel_max"] = worst_delta, worst_y
    report["bytes_fetched"] = st.src.fetched
    print(f"worst: update {worst_delta:.3e}, y {worst_y:.3e}; {st.src.fetched / 1e6:.1f} MB of the LoRA file read")
    if a.out:
        json.dump(report, open(a.out, "w"))


if __name__ == "__main__":
    main()
