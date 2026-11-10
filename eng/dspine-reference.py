#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""NumPy reference of the DSpine block draft as TensorSharp runs it, and the fixture generator for
TensorSharp.GGML.Native/tests/dspine_draft_test.cpp.

DSpine (Tencent, https://github.com/Tencent/DSpine, Apache-2.0; arXiv 2609.36173) is a DFlash block drafter with
two changes, both written here from a reading of dspine_mlx/{model,bridge,propose}.py and
sglang_dspine/{models,kernels}/dspine.py at commit 321b0fc (no code from there is imported or run):

  * attention is CAUSAL inside the block: block row i sees every visible context row and block rows 0..i;
  * a CODE BRIDGE writes into the residual stream after every draft layer:
        raw        = layer output                                      [b, H]
        rms        = sqrt(max(mean(raw^2), FLT_MIN))
        projected  = (raw / (rms + eps)) @ read_l^T                    [b, C]
        codes      = sqrt(C) * projected / (|projected| + eps); row 0 <- sqrt(C) * codebook[anchor]
        q, s, v    = codes @ [gate; predecessor_gate; prev]^T          [b, R] each
        message_t  = sigmoid(q_t + s_{t-1} + gate_bias) * v_{t-1}      (row 0: none)
        out_t      = raw_t + rms_t * (message_t @ write_l^T)           (row 0: raw_0)
    and the next layer's input is next_norm(out) (the last layer's next_norm is the final norm).

The proposal ("greedy" exact-sender refinement of the LAST write, propose.py RefinementTail.greedy):
  candidates : top-K of target_head @ final_rows[1..b-1] (ascending ids, ties to the lower id)
  tables     : TG[v]  = (head[v] * final_norm_w) @ write_last                    [V, R]
               TSV[v] = (sqrt(C) * codebook[v]) @ [predecessor_gate; prev]^T      [V, 2R]
  score[e][p][c] = base[e][c] + rms_e * sum_r m[e][p][r] * TG[cand[e][c]][r]
      base[e][c] = (raw_e * final_norm_w) . head[cand[e][c]]       raw/rms/q: the LAST layer's pre-write stash, row e+1
      m[e][p]    = sigmoid(q_e + TSV[sid][:R] + gate_bias) * TSV[sid][R:],  sid = anchor (e = 0) or cand[e-1][p]
  chain: j = argmax_c score[0][*][c]; then j = argmax_c score[e][j][c] - the greedy lattice walk.
TensorSharp returns score / rms_e (same argmax per row; the softmax of it is the reference's confidence
1 / sum_c exp((s_c - s_best) / max(rms, 1e-6))).

Precision (an assumption, stated in the report): TensorSharp keeps F32 activations everywhere. The reference
implementation rounds activations to BF16 at fixed points; the weights and tables can still be BF16 here. This
script computes in float64 and rounds only what TensorSharp stores rounded (bridge weights, optionally the tables).

Weights and inputs come from a 32-bit LCG that the C++ test replays bit for bit (same order, same magnitudes),
so the fixture holds only expected outputs.

    python3 eng/dspine-reference.py selftest
    python3 eng/dspine-reference.py fixture --out TensorSharp.GGML.Native/tests/data/dspine_draft_reference.bin

`real` recomputes drafts TensorSharp made on the real model from the checkpoint itself: run the server with a
BF16-backbone DSpine GGUF (eng/dspine-to-gguf.py --type bf16, so both sides hold the same weights) and
TS_DSPINE_DUMP=DIR, then

    python3 eng/dspine-reference.py real --checkpoint DSPINE_HF_DIR --target-gguf TARGET.gguf DIR/draft*.bin

It reads the draft's weights from model.safetensors, the target's LM head and token embedding from the target GGUF
(dequantized with gguf-py, as TensorSharp binds them), replays the dumped context rows, and compares the candidate
sets, the lattice and the greedy chain with TensorSharp's (needs gguf-py; ~8 GB of RAM).
"""

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

# numpy's Accelerate BLAS on Apple Silicon raises spurious divide/overflow/invalid warnings from finite matmuls;
# the draft below checks its outputs for non-finite values explicitly instead.
np.seterr(divide="ignore", over="ignore", invalid="ignore")

FLT_MIN = 1.1754943508222875e-38

# ---------------------------------------------------------------------------------------------------------------
# The LCG shared with the C++ test: state = state * 1664525 + 1013904223 (mod 2^32), value = (int(state >> 16) -
# 32768) * (mag / 32768), every operation in float32.
# ---------------------------------------------------------------------------------------------------------------
LCG_A = 1664525
LCG_C = 1013904223


class Lcg:
    def __init__(self, seed):
        self.state = int(seed) & 0xFFFFFFFF

    def take(self, n, mag):
        n = int(n)
        pw = np.cumprod(np.full(n, LCG_A, dtype=np.uint64))                    # A^1 .. A^n (mod 2^64)
        geo = np.cumsum(np.concatenate(([np.uint64(1)], pw[:-1])))             # 1 + A + .. + A^(k-1)
        with np.errstate(over="ignore"):
            st = (pw * np.uint64(self.state) + np.uint64(LCG_C) * geo) & np.uint64(0xFFFFFFFF)
        self.state = int(st[-1])
        ints = (st >> np.uint64(16)).astype(np.int64) - 32768
        scale = np.float32(mag) / np.float32(32768.0)
        return ints.astype(np.float32) * scale


def mag_in(fan_in):
    """1.75 / sqrt(fan_in) in float32 (C++: 1.75f / std::sqrt(float(fan_in)))."""
    return np.float32(1.75) / np.sqrt(np.float32(fan_in))


def bf16_round(x):
    """Round float32 to bfloat16 (nearest even), returned as float32 (ggml_fp32_to_bf16 for finite values)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + np.uint64(0x7FFF) + ((u >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)
    return (u.astype(np.uint32) << np.uint32(16)).view(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Geometry. "small" stores every stage of the first draft; "wide" (rows past 1024 floats, as the 27B's 5120 and its
# 1024-wide codes) stores the tail and per-row checksums of the layer stages.
# ---------------------------------------------------------------------------------------------------------------
CONFIGS = {
    "small": dict(H=192, NH=4, NKV=2, HD=32, FF=256, L=3, NT=2, C=64, R=32, V=256, K=16, B=16, SW=64, SEED=20261109),
    "wide": dict(H=1280, NH=4, NKV=2, HD=64, FF=384, L=2, NT=2, C=1056, R=96, V=384, K=16, B=16, SW=64, SEED=31),
}
EPS = 1e-6          # draft rms_norm_eps
BRIDGE_EPS = 1e-6   # selected_prefix_code_bridge.eps
ROPE_BASE = 10000.0
MASK_ID_OFFSET = 1  # mask token = V - 1


def ring_rows(cfg):
    return ((cfg["SW"] + cfg["B"] + 1 + 31) // 32) * 32


def make_weights(cfg):
    """The draft, the borrowed target pieces and the bridge, in the order the C++ test generates them."""
    H, NH, NKV, HD, FF, L, C, R, V = (cfg[k] for k in ("H", "NH", "NKV", "HD", "FF", "L", "C", "R", "V"))
    FEAT = cfg["NT"] * H
    g = Lcg(cfg["SEED"])
    w = {}
    w["tok_embd"] = g.take(V * H, 1.0).reshape(V, H)
    w["head"] = g.take(V * H, 3 * mag_in(H)).reshape(V, H)
    w["fc"] = g.take(H * FEAT, mag_in(FEAT)).reshape(H, FEAT)
    w["enc_norm"] = 1.0 + g.take(H, 0.25)
    w["out_norm"] = 1.0 + g.take(H, 0.25)
    w["codebook"] = bf16_round(g.take(V * C, mag_in(C)).reshape(V, C))
    w["packed"] = bf16_round(g.take(3 * R * C, mag_in(C)).reshape(3 * R, C))     # [gate; predecessor_gate; prev]
    w["gate_bias"] = g.take(R, 0.5)
    layers = []
    for _ in range(L):
        lw = {}
        lw["attn_norm"] = 1.0 + g.take(H, 0.25)
        lw["q"] = g.take(NH * HD * H, mag_in(H)).reshape(NH * HD, H)
        lw["k"] = g.take(NKV * HD * H, mag_in(H)).reshape(NKV * HD, H)
        lw["v"] = g.take(NKV * HD * H, mag_in(H)).reshape(NKV * HD, H)
        lw["q_norm"] = 1.0 + g.take(HD, 0.25)
        lw["k_norm"] = 1.0 + g.take(HD, 0.25)
        lw["o"] = g.take(H * NH * HD, mag_in(NH * HD)).reshape(H, NH * HD)
        lw["ffn_norm"] = 1.0 + g.take(H, 0.25)
        lw["gate"] = g.take(FF * H, mag_in(H)).reshape(FF, H)
        lw["up"] = g.take(FF * H, mag_in(H)).reshape(FF, H)
        lw["down"] = g.take(H * FF, mag_in(FF)).reshape(H, FF)
        lw["read"] = bf16_round(g.take(C * H, mag_in(H)).reshape(C, H))
        lw["write"] = bf16_round(g.take(H * R, 0.5 * mag_in(R)).reshape(H, R))
        layers.append(lw)
    w["layers"] = layers
    return w


def feature_rows(cfg, start, n):
    """Feature row of position p: its own LCG stream (C++: seed = 977 * p + 12345)."""
    FEAT = cfg["NT"] * cfg["H"]
    return np.stack([Lcg(977 * (start + i) + 12345).take(FEAT, 1.0) for i in range(n)])


# ---------------------------------------------------------------------------------------------------------------
# The draft (float64)
# ---------------------------------------------------------------------------------------------------------------
def f64(x):
    return np.asarray(x, dtype=np.float64)


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope_neox(x, pos, base):
    """x [n, heads, hd]; ggml's NeoX RoPE (pairs i, i + hd/2), HF rotate_half."""
    hd = x.shape[-1]
    half = hd // 2
    inv = base ** (-np.arange(half, dtype=np.float64) * 2.0 / hd)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Context:
    """Every injected position's K/V per draft layer (what the ring holds, without the ring)."""

    def __init__(self, cfg, w):
        self.cfg, self.w = cfg, w
        self.k = [dict() for _ in range(cfg["L"])]
        self.v = [dict() for _ in range(cfg["L"])]
        self.filled = 0
        self.valid_from = 0

    def inject(self, start, n):
        self.inject_rows(feature_rows(self.cfg, start, n), np.arange(start, start + n))
        if start > self.filled or start < self.valid_from:
            self.valid_from = start
        self.filled = start + n

    def inject_rows(self, feat, pos):
        """The context K/V of feature rows `feat` at positions `pos` (fc -> hidden_norm -> k/v, k-norm, RoPE)."""
        cfg, w = self.cfg, self.w
        NKV, HD, eps, base = cfg["NKV"], cfg["HD"], cfg.get("EPS", EPS), cfg.get("ROPE", ROPE_BASE)
        n = len(pos)
        g = rms_norm(f64(feat) @ f64(w["fc"]).T, f64(w["enc_norm"]), eps)
        for l, lw in enumerate(w["layers"]):
            k = (g @ f64(lw["k"]).T).reshape(n, NKV, HD)
            k = rope_neox(rms_norm(k, f64(lw["k_norm"]), eps), pos, base)
            v = (g @ f64(lw["v"]).T).reshape(n, NKV, HD)
            for i in range(n):
                self.k[l][int(pos[i])] = k[i]
                self.v[l][int(pos[i])] = v[i]

    def visible(self, anchor_pos, qp):
        """Context positions a block query at qp sees: live in the ring, before the anchor, inside the window."""
        rows = ring_rows(self.cfg)
        lo = max(max(0, self.filled - rows), self.valid_from)
        return [p for p in range(lo, self.filled) if p < anchor_pos and qp - p < self.cfg["SW"]]


class DenseTables:
    """TG / TSV held whole (the synthetic configs)."""

    def __init__(self, tables):
        self.tg_all, self.tsv_all = f64(tables[0]), f64(tables[1])

    def tg(self, ids):
        return self.tg_all[np.asarray(ids)]

    def tsv(self, ids):
        return self.tsv_all[np.asarray(ids)]


class DenseHead:
    def __init__(self, head):
        self.w = f64(head)

    def logits(self, rows):
        return rows @ self.w.T

    def rows(self, ids):
        return self.w[np.asarray(ids)]


def draft(ctx, anchor, p0, b, tables=None, ids=None, head=None):
    """One DSpine block. Returns every stage plus the proposal (candidates, scores / rms, chain)."""
    cfg, w = ctx.cfg, ctx.w
    H, NH, NKV, HD, C, R, V, K, L = (cfg[k] for k in ("H", "NH", "NKV", "HD", "C", "R", "V", "K", "L"))
    eps, base = cfg.get("EPS", EPS), cfg.get("ROPE", ROPE_BASE)
    sqrt_c = math.sqrt(C)
    if ids is None:
        ids = [anchor] + [V - MASK_ID_OFFSET] * (b - 1)
    pos = np.arange(p0, p0 + b)
    st = {"raw": [], "codes": [], "out": []}

    if tables is None or isinstance(tables, tuple):
        tables = DenseTables(tables if tables is not None else make_tables(w, None))
    head = head if head is not None else DenseHead(w["head"])
    inp = f64(w["embed_rows"](ids)) if "embed_rows" in w else f64(w["tok_embd"])[ids]
    h = rms_norm(inp, f64(w["layers"][0]["attn_norm"]), eps)
    group = NH // NKV
    anchor_code = sqrt_c * f64(w["codebook"][anchor])
    packed_w = f64(w["packed"])
    for l, lw in enumerate(w["layers"]):
        q = rope_neox(rms_norm((h @ f64(lw["q"]).T).reshape(b, NH, HD), f64(lw["q_norm"]), eps), pos, base)
        k = rope_neox(rms_norm((h @ f64(lw["k"]).T).reshape(b, NKV, HD), f64(lw["k_norm"]), eps), pos, base)
        v = (h @ f64(lw["v"]).T).reshape(b, NKV, HD)
        attn = np.zeros((b, NH, HD))
        for i in range(b):
            cps = ctx.visible(p0, p0 + i)
            ks = np.stack([ctx.k[l][p] for p in cps] + [k[j] for j in range(i + 1)])      # [n, NKV, HD]
            vs = np.stack([ctx.v[l][p] for p in cps] + [v[j] for j in range(i + 1)])
            for hh in range(NH):
                kv = hh // group
                s = ks[:, kv, :] @ q[i, hh] / math.sqrt(HD)
                pr = np.exp(s - s.max())
                pr /= pr.sum()
                attn[i, hh] = pr @ vs[:, kv, :]
        ffn_inp = attn.reshape(b, NH * HD) @ f64(lw["o"]).T + inp
        fh = rms_norm(ffn_inp, f64(lw["ffn_norm"]), eps)
        down = (silu(fh @ f64(lw["gate"]).T) * (fh @ f64(lw["up"]).T)) @ f64(lw["down"]).T
        raw = down + ffn_inp

        # the code bridge
        rms = np.sqrt(np.maximum(np.mean(raw * raw, axis=-1), FLT_MIN))                    # [b]
        proj = (raw / (rms[:, None] + BRIDGE_EPS)) @ f64(lw["read"]).T                      # [b, C]
        codes = sqrt_c * proj / (np.sqrt(np.sum(proj * proj, axis=-1, keepdims=True)) + BRIDGE_EPS)
        codes[0] = anchor_code
        packed = codes @ packed_w.T                                                          # [b, 3R]
        # The anchor row's sender halves ARE the table row TSV[anchor] (its code is sqrt(C) * codebook[anchor]):
        # TensorSharp reads them from the stored table, so they carry the table's rounding (BF16, as the
        # reference's own BF16 packed row does).
        packed[0, R:] = tables.tsv([anchor])[0]
        qq, ss, vv = packed[:, :R], packed[:, R:2 * R], packed[:, 2 * R:]
        msg = sigmoid(qq[1:] + ss[:-1] + f64(w["gate_bias"])) * vv[:-1]                    # [b-1, R]
        out = raw.copy()
        out[1:] += rms[1:, None] * (msg @ f64(lw["write"]).T)
        st["raw"].append(raw)
        st["codes"].append(codes)
        st["out"].append(out)
        if l == L - 1:
            st["stash_rms"], st["stash_q"] = rms, qq
        inp = out
        nxt = f64(w["layers"][l + 1]["attn_norm"]) if l + 1 < L else f64(w["out_norm"])
        h = rms_norm(out, nxt, eps)
    st["final"] = h
    if b < 2:
        return st

    gamma = b - 1
    logits = head.logits(h[1:])                                                               # [gamma, V]
    st["logits"] = logits
    cand = np.zeros((gamma, K), dtype=np.int64)
    for e in range(gamma):
        order = np.argsort(-logits[e], kind="stable")[:K]                                    # ties -> lower id
        cand[e] = np.sort(order)
    st["cand"] = cand
    raw_l, rms_l, q_l = st["raw"][-1], st["stash_rms"], st["stash_q"]
    u = raw_l[1:] * f64(w["out_norm"])                                                       # [gamma, H]
    base = np.stack([head.rows(cand[e]) @ u[e] for e in range(gamma)])                       # [gamma, K]
    st["base"] = base
    gb = f64(w["gate_bias"])
    scores = np.zeros((gamma, K, K))                                                          # [e][p][c]
    for e in range(gamma):
        sid = np.full(K, anchor) if e == 0 else cand[e - 1]
        tsv_sid = tables.tsv(sid)
        m = sigmoid(q_l[e + 1][None, :] + tsv_sid[:, :R] + gb) * tsv_sid[:, R:]              # [K(p), R]
        s = m @ tables.tg(cand[e]).T                                                          # [K(p), K(c)]
        scores[e] = base[e][None, :] / rms_l[e + 1] + s
    st["scores"] = scores
    chain = np.zeros(gamma, dtype=np.int64)
    conf = np.zeros(gamma)
    j = 0
    for e in range(gamma):
        row = scores[e][j if e > 0 else 0]
        j = int(np.argmax(row))                       # candidates ascending: the first max is the lowest id
        chain[e] = cand[e][j]
        ex = np.exp(row - row.max())
        conf[e] = ex[j] / ex.sum()
    st["chain"], st["conf"] = chain, conf
    for key in ("final", "base", "scores", "conf"):
        if not np.all(np.isfinite(st[key])):
            raise FloatingPointError(f"non-finite values in the reference's {key}")
    return st


def make_tables(w, table_round):
    """TG [V, R], TSV [V, 2R] in float64, rounded the way the table is stored (None = exact, 'bf16')."""
    sqrt_c = math.sqrt(w["codebook"].shape[1])
    R = w["gate_bias"].shape[0]
    tg = (f64(w["head"]) * f64(w["out_norm"])) @ f64(w["layers"][-1]["write"])
    tsv = (sqrt_c * f64(w["codebook"])) @ f64(w["packed"][R:]).T
    if table_round == "bf16":
        tg = bf16_round(tg.astype(np.float32)).astype(np.float64)
        tsv = bf16_round(tsv.astype(np.float32)).astype(np.float64)
    return tg, tsv


# ---------------------------------------------------------------------------------------------------------------
# The fixture: three drafts on one context (fresh, after an accept, after the ring wrapped past the window).
# ---------------------------------------------------------------------------------------------------------------
STEPS = [
    # (inject start, inject rows, anchor token, block width)
    (0, 40, 7, 16),
    (40, 5, 11, 16),
    (45, 75, 3, 16),
]


class FixtureWriter:
    def __init__(self):
        self.entries = []

    def add(self, name, arr):
        arr = np.asarray(arr)
        if arr.dtype.kind in "iu":
            self.entries.append((name, 1, arr.astype(np.int32)))
        else:
            self.entries.append((name, 0, arr.astype(np.float32)))

    def write(self, path):
        with open(path, "wb") as f:
            f.write(struct.pack("<4sII", b"DSPF", 1, len(self.entries)))
            for name, dt, arr in self.entries:
                nb = name.encode()
                f.write(struct.pack("<H", len(nb)))
                f.write(nb)
                f.write(struct.pack("<BB", dt, arr.ndim))
                f.write(struct.pack("<%dI" % arr.ndim, *arr.shape))
                f.write(np.ascontiguousarray(arr).tobytes())


def add_config(fw, name, cfg):
    keys = ("H", "NH", "NKV", "HD", "FF", "L", "NT", "C", "R", "V", "K", "B", "SW", "SEED")
    fw.add(f"{name}.config", np.array([cfg[k] for k in keys] + [ring_rows(cfg)], dtype=np.int64))


def fixture(path):
    fw = FixtureWriter()
    for name, cfg in CONFIGS.items():
        w = make_weights(cfg)
        add_config(fw, name, cfg)
        tables = {"f32": make_tables(w, None), "bf16": make_tables(w, "bf16")}
        if name == "small":
            # 64 table rows (the first 48 and the last 16) are enough to see the build is right everywhere
            rows = np.r_[0:48, cfg["V"] - 16:cfg["V"]]
            fw.add(f"{name}.table_rows", rows)
            fw.add(f"{name}.tg", tables["f32"][0][rows])
            fw.add(f"{name}.tsv", tables["f32"][1][rows])
        ctx = Context(cfg, w)
        for si, (start, n, anchor, b) in enumerate(STEPS):
            ctx.inject(start, n)
            p0 = start + n
            for tname, tab in tables.items():
                st = draft(ctx, anchor, p0, b, tab)
                pre = f"{name}.step{si}.{tname}"
                if tname == "f32":
                    full = name == "small" and si == 0
                    for l in range(cfg["L"]):
                        for key in ("raw", "codes", "out"):
                            a = st[key][l]
                            if full:
                                fw.add(f"{name}.step{si}.{key}{l}", a)
                            fw.add(f"{name}.step{si}.{key}{l}.rowsum", a.sum(axis=1))
                            fw.add(f"{name}.step{si}.{key}{l}.rowsq", (a * a).sum(axis=1))
                    if full:
                        fw.add(f"{name}.step{si}.final", st["final"])
                    fw.add(f"{name}.step{si}.final.rowsum", st["final"].sum(axis=1))
                    fw.add(f"{name}.step{si}.final.rowsq", (st["final"] ** 2).sum(axis=1))
                    fw.add(f"{name}.step{si}.cand", st["cand"])
                    fw.add(f"{name}.step{si}.base", st["base"])
                    # the K-th and (K+1)-th logits per row: a candidate set is only well defined above that gap
                    srt = -np.sort(-st["logits"], axis=1)
                    fw.add(f"{name}.step{si}.kgap", srt[:, cfg["K"] - 1] - srt[:, cfg["K"]])
                if tname == "f32" or (name == "small" and si == 0):
                    fw.add(f"{pre}.scores", st["scores"])
                fw.add(f"{pre}.chain", st["chain"])
                fw.add(f"{pre}.conf", st["conf"])
                # the winning margin of each chain step: the walk is only well defined above float noise
                fw.add(f"{pre}.margin", chain_margins(st))
    fw.write(path)
    print(f"wrote {path} ({len(fw.entries)} arrays)")


def chain_margins(st):
    scores, gamma = st["scores"], st["scores"].shape[0]
    out = np.zeros(gamma)
    j = 0
    for e in range(gamma):
        row = np.sort(scores[e][j if e > 0 else 0])
        out[e] = row[-1] - row[-2]
        j = int(np.argmax(scores[e][j if e > 0 else 0]))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Self-checks of the reference itself
# ---------------------------------------------------------------------------------------------------------------
def selftest():
    cfg = CONFIGS["small"]
    w = make_weights(cfg)
    ok = True

    # 1. the LCG against a scalar replay of the C++ recurrence
    g = Lcg(12345)
    vals = g.take(5, 0.5)
    s = 12345
    for i in range(5):
        s = (s * LCG_A + LCG_C) & 0xFFFFFFFF
        ref = np.float32((s >> 16) - 32768) * (np.float32(0.5) / np.float32(32768.0))
        ok &= bool(vals[i] == ref)
    print("lcg replay:", "ok" if ok else "MISMATCH")

    # 2. the table form of the refinement equals the brute-force exact-sender logits
    ctx = Context(cfg, w)
    ctx.inject(0, 40)
    st = draft(ctx, 7, 40, cfg["B"])
    R, C = cfg["R"], cfg["C"]
    raw, rms, q = st["raw"][-1], st["stash_rms"], st["stash_q"]
    head, nw, wl = f64(w["head"]), f64(w["out_norm"]), f64(w["layers"][-1]["write"])
    bad = 0
    for e in range(cfg["B"] - 1):
        sids = np.full(cfg["K"], 7) if e == 0 else st["cand"][e - 1]
        for p, sid in enumerate(sids):
            code = math.sqrt(C) * f64(w["codebook"][sid])
            gp = code @ f64(w["packed"][R:2 * R]).T
            vp = code @ f64(w["packed"][2 * R:]).T
            msg = sigmoid(q[e + 1] + gp + f64(w["gate_bias"])) * vp
            refined = raw[e + 1] + rms[e + 1] * (msg @ wl.T)
            logits = rms_norm(refined, nw, EPS) @ head[st["cand"][e]].T
            bad += int(np.argmax(logits) != np.argmax(st["scores"][e][p]))
    print(f"table refinement vs exact-sender logits: {bad} argmax mismatches over {(cfg['B'] - 1) * cfg['K']} rows")
    ok &= bad == 0

    # 3. causal block: a narrower block is the first rows of the wider one
    short = draft(ctx, 7, 40, 6)
    d = max(np.max(np.abs(short["final"] - st["final"][:6])), np.max(np.abs(short["scores"] - st["scores"][:5])))
    print(f"block 6 vs first rows of block 16: max |diff| {d:.3g}")
    ok &= d < 1e-9
    ok &= bool(np.array_equal(short["chain"], st["chain"][:5]))

    # 4. the chain is not degenerate (the bridge changes choices; candidates are spread)
    print("chain:", st["chain"].tolist(), "conf:", np.round(st["conf"], 3).tolist())
    print("min chain margin:", float(chain_margins(st).min()), "min k-gap:",
          float(np.min(-np.sort(-st["logits"], axis=1)[:, cfg["K"] - 1] + np.sort(-st["logits"], axis=1)[:, cfg["K"]])))
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------------------------
# real: recompute TensorSharp's dumped drafts (TS_DSPINE_DUMP) from the checkpoint
# ---------------------------------------------------------------------------------------------------------------
def read_dspf(path):
    out = {}
    with open(path, "rb") as f:
        magic, version, count = struct.unpack("<4sII", f.read(12))
        assert magic == b"DSPF" and version == 1, path
        for _ in range(count):
            (n,) = struct.unpack("<H", f.read(2))
            name = f.read(n).decode()
            dt, nd = struct.unpack("<BB", f.read(2))
            dims = struct.unpack("<%dI" % nd, f.read(4 * nd))
            size = int(np.prod(dims)) if dims else 1
            arr = np.frombuffer(f.read(4 * size), dtype=np.int32 if dt else np.float32).reshape(dims)
            out[name] = arr
    return out


class BF16Weights:
    """A model.safetensors of BF16 tensors, each converted to float32 when read."""

    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def __call__(self, name, index=None):
        h = self.header[name]
        a, b = h["data_offsets"]
        bits = self.mm[self.base + a:self.base + b].view(np.uint16).reshape(h["shape"])
        if index is not None:
            bits = bits[index]
        return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


class LazyLayer(dict):
    """Layer weights read from the checkpoint on access (keeps ~2 GB instead of 8 GB resident)."""

    def __init__(self, getters):
        super().__init__()
        self.getters = getters

    def __getitem__(self, key):
        return self.getters[key]()


class QuantRows:
    """Rows of a target GGUF tensor, dequantized with gguf-py as needed."""

    def __init__(self, tensor, chunk=16384):
        from gguf.quants import dequantize
        self.dq = dequantize
        self.t = tensor
        self.data = np.asarray(tensor.data)
        self.qtype = tensor.tensor_type
        self.chunk = chunk

    def rows(self, ids):
        return f64(self.dq(self.data[np.asarray(ids)], self.qtype))

    def logits(self, rows):
        out = []
        for s in range(0, self.data.shape[0], self.chunk):
            w = self.dq(self.data[s:s + self.chunk], self.qtype).astype(np.float64)
            out.append(rows @ w.T)
        return np.concatenate(out, axis=1)


class LazyTables:
    """TG / TSV rows computed when read, rounded to BF16 as TensorSharp stores them."""

    def __init__(self, head, w):
        self.head, self.w = head, w
        self.R = w["gate_bias"].shape[0]
        self.sqrt_c = math.sqrt(w["codebook"].shape[1])
        self.write_last = f64(w["layers"][-1]["write"])
        self.pv = f64(w["packed"][self.R:])

    def tg(self, ids):
        rows = (self.head.rows(ids) * f64(self.w["out_norm"])) @ self.write_last
        return bf16_round(rows.astype(np.float32)).astype(np.float64)

    def tsv(self, ids):
        rows = (self.sqrt_c * f64(self.w["codebook"][np.asarray(ids)])) @ self.pv.T
        return bf16_round(rows.astype(np.float32)).astype(np.float64)


def real(args):
    from gguf import GGUFReader
    cfg_raw = json.load(open(os.path.join(args.checkpoint, "config.json")))
    st = BF16Weights(os.path.join(args.checkpoint, "model.safetensors"))
    df = cfg_raw["dflash_config"]
    bridge = df["selected_prefix_code_bridge"]
    L = int(cfg_raw["num_hidden_layers"])
    cfg = dict(H=int(cfg_raw["hidden_size"]), NH=int(cfg_raw["num_attention_heads"]),
               NKV=int(cfg_raw["num_key_value_heads"]), HD=int(cfg_raw["head_dim"]), FF=int(cfg_raw["intermediate_size"]),
               L=L, C=int(bridge["code_size"]), R=int(bridge["rank"]), V=int(cfg_raw["vocab_size"]),
               K=int(cfg_raw["code_bridge_second_pass"]["head_topk"]), B=int(cfg_raw["block_size"]),
               SW=int(cfg_raw["sliding_window"]), EPS=float(cfg_raw["rms_norm_eps"]),
               ROPE=float(cfg_raw["rope_parameters"]["rope_theta"]))
    if abs(float(bridge["eps"]) - BRIDGE_EPS) > 0:
        sys.exit("this reference assumes bridge eps 1e-6")
    target = GGUFReader(args.target_gguf)
    tensors = {t.name: t for t in target.tensors}
    head = QuantRows(tensors["output.weight"])
    embd = QuantRows(tensors["token_embd.weight"])
    P = "selected_prefix_bridge."
    w = {
        "fc": st("fc.weight"), "enc_norm": st("hidden_norm.weight"), "out_norm": st("norm.weight"),
        "codebook": st(P + "codebook"),
        "packed": np.concatenate([st(P + "gate.weight"), st(P + "predecessor_gate.weight"), st(P + "prev.weight")]),
        "gate_bias": st(P + "gate.bias"), "embed_rows": embd.rows,
    }
    layers = []
    for l in range(L):
        q = f"layers.{l}."
        get = {
            "attn_norm": lambda q=q: st(q + "input_layernorm.weight"),
            "ffn_norm": lambda q=q: st(q + "post_attention_layernorm.weight"),
            "q": lambda q=q: st(q + "self_attn.q_proj.weight"), "k": lambda q=q: st(q + "self_attn.k_proj.weight"),
            "v": lambda q=q: st(q + "self_attn.v_proj.weight"), "o": lambda q=q: st(q + "self_attn.o_proj.weight"),
            "q_norm": lambda q=q: st(q + "self_attn.q_norm.weight"), "k_norm": lambda q=q: st(q + "self_attn.k_norm.weight"),
            "gate": lambda q=q: st(q + "mlp.gate_proj.weight"), "up": lambda q=q: st(q + "mlp.up_proj.weight"),
            "down": lambda q=q: st(q + "mlp.down_proj.weight"),
            "read": (lambda: st(P + "read_weight")) if l == L - 1 else (lambda l=l: st(P + "early_read_weights", l)),
            "write": lambda l=l: st(P + "feature_write_weights", l),
        }
        layers.append(LazyLayer(get))
    w["layers"] = layers
    tables = LazyTables(head, w)

    worst = 0.0
    checked = mismatched = 0
    for path in args.dumps:
        d = read_dspf(path)
        ids = [int(x) for x in d["ids"]]
        b, p0, K = len(ids), int(d["p0"][0]), cfg["K"]
        g = b - 1
        ctx = Context(cfg, w)
        pos = d["ctx.pos"].astype(np.int64)
        ctx.inject_rows(d["ctx.feat"], pos)
        ctx.filled, ctx.valid_from = p0, int(pos.min())
        ref = draft(ctx, ids[0], p0, b, tables=tables, ids=ids, head=head)
        # TensorSharp's lattice in id order, as the reference keeps it
        cand = d["cand"].reshape(g, K)
        scores = d["scores"]
        order = [np.argsort(cand[e], kind="stable") for e in range(g)]
        same = [bool(np.array_equal(np.sort(cand[e]), ref["cand"][e])) for e in range(g)]
        srt = -np.sort(-ref["logits"], axis=1)
        kgap = srt[:, K - 1] - srt[:, K]
        num, den = 0.0, np.abs(ref["scores"]).max()
        for e in range(g):
            if not same[e] or (e > 0 and not same[e - 1]):
                continue
            for pi in range(1 if e == 0 else K):
                for ci in range(K):
                    a = scores[order[0][ci]] if e == 0 else \
                        scores[K + (e - 1) * K * K + order[e - 1][pi] * K + order[e][ci]]
                    num = max(num, abs(a - ref["scores"][e][pi][ci]))
        err = num / den
        worst = max(worst, err)
        margins = chain_margins(ref)
        steps = 0
        for e in range(g):
            if not same[e] or margins[e] < args.margin:
                break
            steps += 1
            if int(d["chain"][e]) != int(ref["chain"][e]):
                mismatched += 1
                print(f"  chain step {e}: TensorSharp {int(d['chain'][e])} vs reference {int(ref['chain'][e])} "
                      f"(margin {margins[e]:.3g})")
                break
        checked += steps
        print(f"{os.path.basename(path)}: anchor {ids[0]} at {p0}, {len(pos)} context rows, {b} rows; "
              f"candidate sets equal {sum(same)}/{g} (rows that differ have k-gap "
              f"{[round(float(kgap[e]), 4) for e in range(g) if not same[e]]}); lattice normalized error {err:.3g}; "
              f"whole chain identical {bool(np.array_equal(d['chain'], ref['chain']))} "
              f"({[int(x) for x in d['chain']]}); {steps} steps above the margin compared")
    print(f"real: {checked} chain steps compared, {mismatched} mismatches, worst lattice error {worst:.3g}")
    return 0 if mismatched == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    fx = sub.add_parser("fixture")
    fx.add_argument("--out", required=True)
    rl = sub.add_parser("real")
    rl.add_argument("--checkpoint", required=True, help="directory with the DSpine config.json and model.safetensors")
    rl.add_argument("--target-gguf", required=True, help="the target GGUF TensorSharp ran (its head and embedding)")
    rl.add_argument("--margin", type=float, default=0.05, help="chain steps closer than this are not compared")
    rl.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()
    if args.cmd == "real":
        return real(args)
    fixture(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
