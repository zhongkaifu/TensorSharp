#!/usr/bin/env python3
"""Independent NumPy reference for the Qwen-Image-2.1 native transformer.

Runs small, deterministic full forwards (including the conditioning projections,
time embedding, segmented causal attention and final projection) on CPU or Metal.
No model downloads required. This validates operators, not image quality.
The LoRA cases attach random low-rank updates (TSGQi21Adapter) to every projection,
including stacked Q/K/V and gate/up shrinks, a factor shared by both gate/up halves,
DoRA row scales, F16 and F32 factors and a per-call output head, and compare them
with the same updates applied in NumPy.
Example: python3 eng/tests/qwen-image21-dit.py --backend metal
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import time
import numpy as np


class Weight(ct.Structure):
    _fields_ = [("data", ct.c_void_p), ("type", ct.c_int32), ("reserved", ct.c_int32),
                ("ne0", ct.c_int64), ("ne1", ct.c_int64), ("bytes", ct.c_int64)]


class Block(ct.Structure):
    _fields_ = [(n, Weight) for n in ("q", "k", "v", "out", "gate", "up", "down")] + [
        ("norm_q", ct.c_void_p), ("norm_k", ct.c_void_p)]


class Segment(ct.Structure):
    _fields_ = [(n, ct.c_int32) for n in ("start", "end", "source_start", "is_image")]


GLOBALS = ("image_in", "text_in", "text_out", "time_in", "time_out", "modulation", "norm_out", "proj_out")
PROJECTIONS = ("q", "k", "v", "out", "gate", "up", "down")
GGML_F32, GGML_F16 = 0, 1


class Lora(ct.Structure):
    _fields_ = [("down", ct.c_void_p), ("up", ct.c_void_p), ("row_scale", ct.c_void_p),
                ("type", ct.c_int32), ("rank", ct.c_int32), ("in_", ct.c_int64), ("out", ct.c_int64)]


class BlockLora(ct.Structure):
    _fields_ = [(n, Lora) for n in PROJECTIONS]


class Adapter(ct.Structure):
    _fields_ = [("struct_bytes", ct.c_int32), ("num_layers", ct.c_int32)] + [(n, Lora) for n in GLOBALS] + [
        ("blocks", ct.c_void_p), ("output_head", ct.c_void_p), ("output_head_type", ct.c_int32), ("reserved", ct.c_int32)]


class Desc(ct.Structure):
    _fields_ = [(n, ct.c_void_p) for n in ("images", "text", "time_embedding", "cos", "sin", "output")] + [
        (n, Weight) for n in GLOBALS] + [
        (n, ct.c_void_p) for n in ("text_norm", "blocks", "segments")] + [
        (n, ct.c_int32) for n in ("struct_bytes", "dim", "heads", "head_dim", "channels", "text_dim", "image_seq", "text_seq", "total_seq", "prefix_seq", "num_layers", "num_segments")] + [
        ("eps", ct.c_float), ("prefix_cache_key", ct.c_uint64), ("prefix_cache_type", ct.c_int32), ("tp_ranks", ct.c_int32),
        ("adapter", ct.c_void_p)]


# Mirrors the static_asserts in ggml_ops_qwen_image21.h.
assert ct.sizeof(Desc) == 472 and Desc.adapter.offset == 464
assert ct.sizeof(Lora) == 48 and ct.sizeof(BlockLora) == 336 and ct.sizeof(Adapter) == 416


def rms(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def norm(x):
    return rms(x - np.mean(x, axis=-1, keepdims=True))


def silu(x):
    return x / (1 + np.exp(-x))


def gelu(x):
    return .5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + .044715 * x**3)))


def linear(x, w):
    # Explicit contraction keeps the oracle independent of platform BLAS kernels.
    return np.einsum("...i,oi->...o", x, w, optimize=False)


def adapted(x, w, update):
    """x W^T with a LoRA update (A [rank, in], B [out, rank], optional per-output row scale)."""
    y = linear(x, w)
    if update is None:
        return y
    a, b, row_scale = update
    if row_scale is not None:
        y = y * row_scale
    if a is not None:
        y = y + linear(linear(x, a), b)
    return y


def reference(w, blocks, images, text, ts, cos, sin, segments, prefix, heads, lora=None):
    g = (lora or {}).get("globals", {})
    bl = (lora or {}).get("blocks", [{} for _ in blocks])
    time = silu(adapted(silu(adapted(ts, w["time_in"], g.get("time_in"))), w["time_out"], g.get("time_out")))
    mods = np.split(adapted(time, w["modulation"], g.get("modulation")), 4, axis=-1)
    text = adapted(gelu(adapted(rms(text) * (1 + w["text_norm"]), w["text_in"], g.get("text_in"))), w["text_out"], g.get("text_out"))
    images = adapted(images, w["image_in"], g.get("image_in"))
    x = np.concatenate([(images if s.is_image else text)[s.source_start:s.source_start+s.end-s.start] for s in segments])
    seq, dim = x.shape
    hd = dim // heads

    def modulate(h, mod, gate=False):
        factors = np.concatenate([np.repeat(mod[1:2], prefix, axis=0), np.repeat(mod[:1], seq-prefix, axis=0)])
        return h * (np.tanh(factors) if gate else 1 + factors)

    def rope(h):
        e, o = h[..., ::2], h[..., 1::2]
        # Interleaved reference intentionally differs from the native half-split
        # optimization: attention must remain invariant to that channel permutation.
        out = np.empty_like(h)
        out[..., ::2] = e * cos[:, None, :] - o * sin[:, None, :]
        out[..., 1::2] = o * cos[:, None, :] + e * sin[:, None, :]
        return out

    for b, u in zip(blocks, bl):
        h = modulate(norm(x), mods[0])
        q = rope(rms((adapted(h, b["q"], u.get("q"))).reshape(seq, heads, hd)) * b["norm_q"])
        k = rope(rms((adapted(h, b["k"], u.get("k"))).reshape(seq, heads, hd)) * b["norm_k"])
        v = (adapted(h, b["v"], u.get("v"))).reshape(seq, heads, hd)
        outputs = []
        for s in segments:
            scores = np.einsum("qhd,khd->hqk", q[s.start:s.end], k[:s.end]) / np.sqrt(hd)
            if not s.is_image:
                mask = np.arange(s.end)[None, :] > np.arange(s.start, s.end)[:, None]
                scores = np.where(mask[None, :, :], -np.inf, scores)
            probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            outputs.append(np.einsum("hqk,khd->qhd", probs, v[:s.end]).reshape(s.end-s.start, dim))
        x += modulate(adapted(np.concatenate(outputs), b["out"], u.get("out")), mods[1], True)
        h = modulate(norm(x), mods[2])
        if "gate_up" in u:
            # One update of the fused [gate; up] projection.
            gu = adapted(h, np.concatenate([b["gate"], b["up"]]), u["gate_up"])
            gate, up = np.split(gu, 2, axis=-1)
        else:
            gate, up = adapted(h, b["gate"], u.get("gate")), adapted(h, b["up"], u.get("up"))
        h = silu(gate) * up
        x += modulate(adapted(h, b["down"], u.get("down")), mods[3], True)
    proj_out = (lora or {}).get("output_head", w["proj_out"])
    return adapted(norm(x[prefix:]) * (1 + adapted(time[:1], w["norm_out"], g.get("norm_out"))), proj_out,
                   None if "output_head" in (lora or {}) else g.get("proj_out"))


def make_adapter(variant, rng, keep, w, blocks, dim, channels, fused):
    """Random LoRA updates for every projection, as (ctypes adapter, NumPy description).

    "f16": F16 factors; Q/K/V (and split gate/up) downs share one contiguous allocation,
    so the native graph runs one stacked shrink; DoRA row scales on q, gate and down; a
    row-scale-only update on modulation; a per-call F16 output head replacing proj_out.
    "f32": F32 factors in separate allocations; gate and up share one down factor (a
    fused LoRA seen as its halves); a row scale on out; proj_out keeps an ordinary update.
    """
    f16 = variant == "f16"
    dtype = np.float16 if f16 else np.float32
    ggml_type = GGML_F16 if f16 else GGML_F32

    def factors(rows, cols, std):
        a = (rng.standard_normal((rows, cols)) * std).astype(dtype)
        keep.append(a)
        return a

    def row_scale(out):
        r = (1 + .2 * rng.standard_normal(out)).astype(np.float32)
        keep.append(r)
        return r

    def update(weight, rank, scale_rows=False, down=None, scale_only=False):
        out, inp = weight.shape
        lora = Lora(type=ggml_type, rank=0 if scale_only else rank, in_=inp, out=out)
        described = [None, None, None]
        if not scale_only:
            a = down if down is not None else factors(rank, inp, 1 / np.sqrt(inp))
            b = factors(out, rank, .18 / np.sqrt(rank))
            lora.down, lora.up = a.ctypes.data, b.ctypes.data
            described[0], described[1] = a.astype(np.float32), b.astype(np.float32)
        if scale_rows or scale_only:
            r = row_scale(out)
            lora.row_scale = r.ctypes.data
            described[2] = r
        return lora, tuple(described)

    adapter = Adapter(struct_bytes=ct.sizeof(Adapter), num_layers=len(blocks))
    described = {"globals": {}, "blocks": []}
    for i, name in enumerate(GLOBALS):
        if name == "proj_out" and f16:
            continue  # replaced by the output head below
        lora, d = update(w[name], 8 + 8 * (i % 2), scale_only=f16 and name == "modulation")
        setattr(adapter, name, lora)
        described["globals"][name] = d
    native_blocks = (BlockLora * len(blocks))()
    keep.append(native_blocks)
    for index, b in enumerate(blocks):
        u, n = {}, native_blocks[index]
        inp = b["q"].shape[1]
        ranks = (8, 16, 8)
        if f16:
            # One allocation: rows [0,8) are q's down, [8,24) k's, [24,32) v's.
            stacked = factors(sum(ranks), inp, 1 / np.sqrt(inp))
            starts = np.cumsum((0,) + ranks)
            downs = [stacked[starts[j]:starts[j + 1]] for j in range(3)]
        else:
            downs = [None, None, None]
        for j, name in enumerate(("q", "k", "v")):
            lora, d = update(b[name], ranks[j], scale_rows=f16 and name == "q", down=downs[j])
            setattr(n, name, lora)
            u[name] = d
        n.out, u["out"] = update(b["out"], 8, scale_rows=not f16)
        if fused:
            gate_up = np.concatenate([b["gate"], b["up"]])
            n.gate, u["gate_up"] = update(gate_up, 16, scale_rows=f16)
        elif f16:
            stacked = factors(16, inp, 1 / np.sqrt(inp))
            n.gate, u["gate"] = update(b["gate"], 8, scale_rows=True, down=stacked[:8])
            n.up, u["up"] = update(b["up"], 8, down=stacked[8:])
        else:
            shared = factors(8, inp, 1 / np.sqrt(inp))
            n.gate, u["gate"] = update(b["gate"], 8, down=shared)
            n.up, u["up"] = update(b["up"], 8, down=shared)
        n.down, u["down"] = update(b["down"], 8, scale_rows=f16)
        described["blocks"].append(u)
    adapter.blocks = ct.addressof(native_blocks)
    if f16:
        head = (w["proj_out"] + rng.standard_normal(w["proj_out"].shape) * .05).astype(np.float16)
        keep.append(head)
        adapter.output_head, adapter.output_head_type = head.ctypes.data, GGML_F16
        described["output_head"] = head.astype(np.float32)
    keep.append(adapter)
    return adapter, described


def run_case(lib, backend, case, fused, flash, benchmark_tokens=0, lora_variant=None, measure=False):
    try:
        return _run_case(lib, backend, case, fused, flash, benchmark_tokens, lora_variant, measure)
    finally:
        # Device backends keep uploaded weights keyed by host pointer, and NumPy reuses
        # freed addresses: a case that fails before its own cleanup must not leave stale
        # weights (or a prefix cache) for the next case.
        lib.TSGgml_QwenImage21ReleasePrefixCaches()
        lib.TSGgml_ClearHostBufferCache()


def _run_case(lib, backend, case, fused, flash, benchmark_tokens, lora_variant, measure):
    rng = np.random.default_rng(2141)
    keep = []

    def array(shape, scale=1):
        a = (rng.standard_normal(shape) * scale).astype(np.float32)
        keep.append(a)
        return a

    def weight(a):
        return Weight(a.ctypes.data, 0, 0, a.shape[-1], a.shape[0], a.nbytes)

    dim, hd, heads, channels, td, ff = 256, 128, 2, 64, 96, 320
    if benchmark_tokens:
        specs = [(0, 64, 0, 0), (64, 64 + benchmark_tokens, 0, 1)]
        tsq, isq, prefix = 64, benchmark_tokens, 64
    elif case == "text":
        specs = [(0, 5, 0, 0), (5, 11, 0, 1)]
        tsq, isq, prefix = 5, 6, 5
    elif case == "edit":
        specs = [(0, 2, 0, 0), (2, 6, 0, 1), (6, 9, 3, 0), (9, 15, 4, 1)]
        tsq, isq, prefix = 6, 10, 9
    elif case == "multi-edit":
        specs = [(0, 2, 0, 0), (2, 6, 0, 1), (6, 8, 3, 0), (8, 12, 4, 1), (12, 15, 6, 0), (15, 21, 8, 1)]
        tsq, isq, prefix = 9, 14, 15
    else:
        # Text and image boundaries straddle flash-attention tile lengths;
        # the shape mutation below changes 512 KV/256 query rows to 511/255.
        specs = [(0, 63, 0, 0), (63, 255, 0, 1), (255, 256, 63, 0), (256, 512, 192, 1)]
        tsq, isq, prefix = 64, 448, 256
    segments = (Segment * len(specs))(*(Segment(*s) for s in specs))
    seq = specs[-1][1]
    images, text = array((isq, channels)), array((tsq, td))
    ts = array((2, 256), .5)
    angles = array((seq, hd//2))
    cos, sin = np.cos(angles), np.sin(angles)
    w = {}
    for name, shape in {
        "image_in": (dim, channels), "text_in": (dim, td), "text_out": (dim, dim),
        "time_in": (dim, 256), "time_out": (dim, dim), "modulation": (4*dim, dim),
        "norm_out": (dim, dim), "proj_out": (channels, dim),
    }.items():
        w[name] = array(shape, .6 / np.sqrt(shape[-1]))
    w["text_norm"] = array((td,), .2)
    blocks, native_blocks = [], (Block * 2)()
    for i in range(2):
        b = {n: array(shape, .6 / np.sqrt(shape[-1])) for n, shape in {
            "q": (dim, dim), "k": (dim, dim), "v": (dim, dim), "out": (dim, dim),
            "gate": (ff, dim), "up": (ff, dim), "down": (dim, ff),
        }.items()}
        b["norm_q"] = array((hd,), .2) + 1
        b["norm_k"] = array((hd,), .2) + 1
        blocks.append(b)
        for name in ("q", "k", "v", "out", "gate", "up", "down"):
            setattr(native_blocks[i], name, weight(b[name]))
        if fused:
            gu = np.concatenate([b["gate"], b["up"]])
            keep.append(gu)
            native_blocks[i].gate, native_blocks[i].up = weight(gu), Weight()
        native_blocks[i].norm_q, native_blocks[i].norm_k = b["norm_q"].ctypes.data, b["norm_k"].ctypes.data
    adapter, lora = (None, None)
    if lora_variant:
        adapter, lora = make_adapter(lora_variant, rng, keep, w, blocks, dim, channels, fused)
    expected = None if benchmark_tokens else reference(w, blocks, images, text, ts, cos, sin, segments, prefix, heads, lora)
    output = np.zeros((seq - prefix, channels), dtype=np.float32)
    d = Desc()
    for name, a in dict(images=images, text=text, time_embedding=ts, cos=cos, sin=sin, output=output).items():
        setattr(d, name, a.ctypes.data)
    for name in ("image_in", "text_in", "text_out", "time_in", "time_out", "modulation", "norm_out", "proj_out"):
        setattr(d, name, weight(w[name]))
    d.text_norm = w["text_norm"].ctypes.data
    d.blocks, d.segments = ct.addressof(native_blocks), ct.addressof(segments)
    for name, value in dict(struct_bytes=ct.sizeof(Desc), dim=dim, heads=heads, head_dim=hd, channels=channels,
                            text_dim=td, image_seq=isq, text_seq=tsq, total_seq=seq, prefix_seq=prefix,
                            num_layers=2, num_segments=len(specs), eps=1e-6).items():
        setattr(d, name, value)
    if adapter is not None:
        d.adapter = ct.addressof(adapter)
    # The LoRA text case also runs the prefix KV cache: the stored text K/V must be
    # computed with the updates active (extract), then reused (cached).
    prefix_key = 77 if lora_variant and case == "text" else 0
    os.environ["TS_QWEN21_FLASH"] = "1" if flash else "0"
    # The larger boundary fixture uses Metal's half-precision matrix kernels;
    # the unchanged baseline has ~1.4e-3 absolute error against the F32 oracle.
    # Half-precision matrix paths bound GPU agreement with the F32 oracle: Metal's half
    # tiles (~1.4e-3 on tile-boundaries), CUDA's tensor-core GEMMs (~1.8e-3 on every case,
    # with or without LoRA) and Vulkan's coopmat kernels (~1.5e-3).
    tolerance = {"metal": .002 if case == "tile-boundaries" else .001,
                 "cuda": .003 if case == "tile-boundaries" else .0025,
                 "vulkan": .003 if case == "tile-boundaries" else .0025}.get(backend, .0001)
    if lora_variant == "f16":
        # ggml's F16 matmul kernels round the activation operand to F16 (CPU vec_dot,
        # Metal half tiles), so F16 factors add ~1e-3 on top of the F32 oracle; the F32
        # variant keeps the base tolerance and pins the update math itself.
        tolerance = .004 if backend == "cpu" else tolerance * 1.5
    errors, seconds, paths = [], [], []
    first_output = None
    for repeat in range(6 if benchmark_tokens else 5):
        # Changed latent input exercises allocation/cache reuse with new data.
        if repeat == 2 and not benchmark_tokens:
            images *= 1.1
            expected = reference(w, blocks, images, text, ts, cos, sin, segments, prefix, heads, lora)
        if repeat == 1:
            d.prefix_cache_key = prefix_key
        if repeat == 3 and not benchmark_tokens:
            # Real CFG alternates prompt shapes through the shared allocator.
            # Shrink then restore the target with resident weights untouched.
            segments[-1].end -= 1
            d.total_seq -= 1
            d.image_seq -= 1
            expected = reference(w, blocks, images[:-1], text, ts, cos[:-1], sin[:-1], segments, prefix, heads, lora)
        if repeat == 4 and not benchmark_tokens:
            segments[-1].end += 1
            d.total_seq += 1
            d.image_seq += 1
            expected = reference(w, blocks, images, text, ts, cos, sin, segments, prefix, heads, lora)
        output.fill(np.nan)
        start = time.perf_counter()
        path = lib.TSGgml_QwenImage21Forward(ct.byref(d))
        if path == 0:
            raise RuntimeError(lib.TSGgml_GetLastError().decode())
        paths.append(path)
        seconds.append(time.perf_counter() - start)
        if benchmark_tokens:
            assert np.isfinite(output).all(), "Non-finite benchmark output"
            if first_output is None:
                first_output = output.copy()
            else:
                assert np.array_equal(first_output, output), "Repeated input produced a different output"
            continue
        actual = output[:len(expected)]
        error = float(np.max(np.abs(actual - expected)))
        if not measure:
            assert np.isfinite(actual).all() and error < tolerance, (case, fused, flash, repeat, error)
        assert np.isnan(output[len(expected):]).all(), "Native output wrote beyond target shape"
        errors.append(error)
    if prefix_key:
        # 1 = full, 2 = extract, 3 = cached: the cache stored LoRA K/V and served them.
        assert paths[1] == 2 and paths[2] == 3, paths
        lib.TSGgml_QwenImage21ReleasePrefixCache(ct.c_uint64(prefix_key))
        d.prefix_cache_key = 0
    if adapter is not None:
        # A LoRA that does not fit its projection is refused before any work.
        good = adapter.time_in.out
        adapter.time_in.out = good + 1
        assert lib.TSGgml_QwenImage21Forward(ct.byref(d)) == 0
        assert b"LoRA" in lib.TSGgml_GetLastError()
        adapter.time_in.out = good
    # Reject a malformed segment before any ggml assertion or output write.
    segments[0].start = 1
    assert lib.TSGgml_QwenImage21Forward(ct.byref(d)) == 0
    lib.TSGgml_ClearHostBufferCache()
    if benchmark_tokens:
        return dict(case=case, fused_mlp=fused, flash=flash, target_tokens=benchmark_tokens,
                    prefix_tokens=prefix, dim=dim, head_dim=hd, layers=len(blocks), weight_type="F32",
                    first_seconds=seconds[0], warm_seconds=seconds[1:], warm_median_seconds=float(np.median(seconds[1:])),
                    reference_validated=False,
                    limitation="Synthetic two-layer graph; excludes model loading, conditioning, scheduler and VAE. No image quality validation.")
    return dict(case=case, fused_mlp=fused, flash=flash, lora=lora_variant, tolerance=tolerance,
                max_absolute_errors=errors, paths=paths)


def run_tp_case(lib, backend, case, fused, lora_variant, ranks=2):
    """Two tensor-parallel ranks (a loopback group on one device) with every projection
    sharded like QwenImage21DiT.ShardBlocks and the LoRA sharded like QwenImage21LoraSet:
    Q/K/V/gate/up keep output rows (up rows and row scales sliced, down replicated),
    to_out/down keep input columns (down columns copied, up and row scale replicated), and
    each rank adds its partial LoRA term before the all-reduce. Compared with the
    unsharded NumPy reference."""
    try:
        return _run_tp_case(lib, backend, case, fused, lora_variant, ranks)
    finally:
        lib.TSGgml_QwenImage21ReleasePrefixCaches()
        lib.TSGgml_ClearHostBufferCache()


def _run_tp_case(lib, backend, case, fused, lora_variant, ranks):
    rng = np.random.default_rng(2141)
    keep = []

    def array(shape, scale=1):
        a = (rng.standard_normal(shape) * scale).astype(np.float32)
        keep.append(a)
        return a

    def weight(a):
        a = np.ascontiguousarray(a)
        keep.append(a)
        return Weight(a.ctypes.data, 0, 0, a.shape[-1], a.shape[0], a.nbytes)

    dim, hd, heads, channels, td, ff = 256, 128, 2, 64, 96, 320
    specs = {"text": [(0, 5, 0, 0), (5, 11, 0, 1)], "edit": [(0, 2, 0, 0), (2, 6, 0, 1), (6, 9, 3, 0), (9, 15, 4, 1)]}[case]
    tsq, isq, prefix = {"text": (5, 6, 5), "edit": (6, 10, 9)}[case]
    segments = (Segment * len(specs))(*(Segment(*sp) for sp in specs))
    seq = specs[-1][1]
    images, text = array((isq, channels)), array((tsq, td))
    ts = array((2, 256), .5)
    angles = array((seq, hd // 2))
    cos, sin = np.cos(angles), np.sin(angles)
    w = {name: array(shape, .6 / np.sqrt(shape[-1])) for name, shape in {
        "image_in": (dim, channels), "text_in": (dim, td), "text_out": (dim, dim),
        "time_in": (dim, 256), "time_out": (dim, dim), "modulation": (4 * dim, dim),
        "norm_out": (dim, dim), "proj_out": (channels, dim)}.items()}
    w["text_norm"] = array((td,), .2)
    blocks = []
    for _ in range(2):
        b = {n: array(shape, .6 / np.sqrt(shape[-1])) for n, shape in {
            "q": (dim, dim), "k": (dim, dim), "v": (dim, dim), "out": (dim, dim),
            "gate": (ff, dim), "up": (ff, dim), "down": (dim, ff)}.items()}
        b["norm_q"] = array((hd,), .2) + 1
        b["norm_k"] = array((hd,), .2) + 1
        blocks.append(b)
    _, lora = make_adapter(lora_variant, rng, keep, w, blocks, dim, channels, fused)
    expected = reference(w, blocks, images, text, ts, cos, sin, segments, prefix, heads, lora)

    f16 = lora_variant == "f16"
    dtype, ggml_type = (np.float16, GGML_F16) if f16 else (np.float32, GGML_F32)

    def lora_struct(a, b, rs, inp, out):
        l = Lora(type=ggml_type, rank=0 if a is None else a.shape[0], in_=inp, out=out)
        if a is not None:
            a, b = np.ascontiguousarray(a.astype(dtype)), np.ascontiguousarray(b.astype(dtype))
            keep.extend([a, b])
            l.down, l.up = a.ctypes.data, b.ctypes.data
        if rs is not None:
            rs = np.ascontiguousarray(rs.astype(np.float32))
            keep.append(rs)
            l.row_scale = rs.ctypes.data
        return l

    def column(u, start, count, out_full):
        a, b, rs = u
        return lora_struct(a, None if b is None else b[start:start + count], None if rs is None else rs[start:start + count],
                           out_full if a is None else a.shape[1], count) if a is not None else \
            lora_struct(None, None, None if rs is None else rs[start:start + count], 0, count)

    def row(u, start, count):
        a, b, rs = u
        return lora_struct(None if a is None else a[:, start:start + count], b, rs, count, b.shape[0] if b is not None else len(rs))

    local, ff_local = dim // ranks, ff // ranks
    descs, outputs, adapters = [], [], []
    for r in range(ranks):
        native_blocks = (Block * 2)()
        block_loras = (BlockLora * 2)()
        keep.extend([native_blocks, block_loras])
        for i, b in enumerate(blocks):
            n, u, bl = native_blocks[i], lora["blocks"][i], block_loras[i]
            q0 = r * local
            n.q, n.k, n.v = (weight(b[x][q0:q0 + local]) for x in ("q", "k", "v"))
            n.out = weight(b["out"][:, q0:q0 + local])
            g0 = r * ff_local
            n.gate, n.up = weight(b["gate"][g0:g0 + ff_local]), weight(b["up"][g0:g0 + ff_local])
            n.down = weight(b["down"][:, g0:g0 + ff_local])
            n.norm_q, n.norm_k = b["norm_q"].ctypes.data, b["norm_k"].ctypes.data
            if "gate_up" in u:
                # A fused update shards into the rank's gate rows and up rows, sharing down.
                a, bb, rs = u["gate_up"]
                gate_u = (a, bb[:ff], None if rs is None else rs[:ff])
                up_u = (a, bb[ff:], None if rs is None else rs[ff:])
            else:
                gate_u, up_u = u["gate"], u["up"]
            if f16 and all(u[x][0] is not None for x in ("q", "k", "v")):
                # Keep the stacked Q/K/V down factors contiguous on every rank.
                stacked = np.ascontiguousarray(np.concatenate([u[x][0] for x in ("q", "k", "v")]).astype(dtype))
                keep.append(stacked)
                offset = 0
                for x in ("q", "k", "v"):
                    a, bb, rs = u[x]
                    l = column((a, bb, rs), q0, local, dim)
                    l.down = stacked.ctypes.data + offset * stacked.strides[0]
                    offset += a.shape[0]
                    setattr(bl, x, l)
            else:
                for x in ("q", "k", "v"):
                    setattr(bl, x, column(u[x], q0, local, dim))
            bl.out = row(u["out"], q0, local)
            bl.gate, bl.up = column(gate_u, g0, ff_local, ff), column(up_u, g0, ff_local, ff)
            bl.down = row(u["down"], g0, ff_local)
        adapter = Adapter(struct_bytes=ct.sizeof(Adapter), num_layers=2)
        for name in GLOBALS:
            if name in lora["globals"]:
                a, bb, rs = lora["globals"][name]
                setattr(adapter, name, lora_struct(a, bb, rs, w[name].shape[1], w[name].shape[0]))
        adapter.blocks = ct.addressof(block_loras)
        if "output_head" in lora:
            head = np.ascontiguousarray(lora["output_head"].astype(np.float16))
            keep.append(head)
            adapter.output_head, adapter.output_head_type = head.ctypes.data, GGML_F16
        adapters.append(adapter)
        output = np.full((seq - prefix, channels), np.nan, dtype=np.float32)
        outputs.append(output)
        d = Desc()
        for name, a in dict(images=images, text=text, time_embedding=ts, cos=cos, sin=sin, output=output).items():
            setattr(d, name, a.ctypes.data)
        for name in GLOBALS:
            setattr(d, name, weight(w[name]))
        d.text_norm = w["text_norm"].ctypes.data
        d.blocks, d.segments = ct.addressof(native_blocks), ct.addressof(segments)
        for name, value in dict(struct_bytes=ct.sizeof(Desc), dim=dim, heads=heads // ranks, head_dim=hd, channels=channels,
                                text_dim=td, image_seq=isq, text_seq=tsq, total_seq=seq, prefix_seq=prefix,
                                num_layers=2, num_segments=len(specs), eps=1e-6, tp_ranks=ranks).items():
            setattr(d, name, value)
        d.adapter = ct.addressof(adapter)
        descs.append(d)
    pointers = (ct.POINTER(Desc) * ranks)(*(ct.pointer(d) for d in descs))
    if lib.TSGgml_QwenImage21ForwardTp(pointers, ranks) == 0:
        raise RuntimeError(lib.TSGgml_GetLastError().decode())
    error = float(np.max(np.abs(outputs[0] - expected)))
    return dict(case=case, fused_mlp=fused, lora=lora_variant, tp_ranks=ranks, max_absolute_error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "metal", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--library", type=Path, default=Path(__file__).resolve().parents[2]/"TensorSharp.GGML.Native/build/libGgmlOps.dylib")
    parser.add_argument("--tp", action="store_true",
                        help="Also run 2-rank tensor-parallel LoRA forwards (a loopback group on one device) after the other cases.")
    parser.add_argument("--measure", action="store_true",
                        help="Record every case's error instead of stopping at the first one above tolerance (exit status still reports failures).")
    parser.add_argument("--benchmark-target-tokens", type=int, default=0,
                        help="Run one synthetic F32/fused/flash timing workload with this many target tokens; skips the NumPy oracle.")
    args = parser.parse_args()
    if args.benchmark_target_tokens < 0:
        parser.error("--benchmark-target-tokens must be positive when supplied")
    lib = ct.CDLL(str(args.library.resolve()))
    lib.TSGgml_GetLastError.restype = ct.c_char_p
    lib.TSGgml_QwenImage21Forward.argtypes = [ct.POINTER(Desc)]
    lib.TSGgml_QwenImage21ReleasePrefixCache.argtypes = [ct.c_uint64]
    lib.TSGgml_QwenImage21ReleasePrefixCaches.argtypes = []
    assert lib.TSGgml_IsBackendAvailable({"cpu": 2, "metal": 1, "cuda": 3, "vulkan": 4}[args.backend]) == 1
    try:
        if args.benchmark_target_tokens:
            rows = [run_case(lib, args.backend, "text", True, True, args.benchmark_target_tokens)]
        else:
            rows = [run_case(lib, args.backend, case, fused, flash, measure=args.measure)
                    for case in ("text", "edit", "multi-edit", "tile-boundaries") for fused in (False, True) for flash in (False, True)]
            rows += [run_case(lib, args.backend, case, fused, flash, lora_variant=variant, measure=args.measure)
                     for variant in ("f16", "f32") for case in ("text", "edit", "tile-boundaries")
                     for fused in (False, True) for flash in (False, True)]
    finally:
        lib.TSGgml_ReleaseReuseComputeBuffers()
        lib.TSGgml_ClearHostBufferCache()
        if args.tp:
            lib.TSGgml_QwenImage21ForwardTp.argtypes = [ct.c_void_p, ct.c_int]
            if lib.TSGgml_TensorParallelInitLoopback({"cpu": 2, "metal": 1, "cuda": 3, "vulkan": 4}[args.backend], 2) != 1:
                raise RuntimeError(lib.TSGgml_GetLastError().decode())
            # Partial sums reduce in another order than one device's GEMM: agreement is to rounding.
            tp_tolerance = {"cpu": .0005, "metal": .003, "cuda": .004, "vulkan": .004}[args.backend]
            for variant in ("f32", "f16"):
                for case in ("text", "edit"):
                    for fused in (False, True):
                        r = run_tp_case(lib, args.backend, case, fused, variant)
                        # F16 factors add the F16-activation rounding of the unsharded f16 cases.
                        r["tolerance"] = max(tp_tolerance * 1.5, .004) if variant == "f16" else tp_tolerance
                        r["max_absolute_errors"] = [r["max_absolute_error"]]
                        rows.append(r)
    failed = [r for r in rows if "tolerance" in r and not all(np.isfinite(e) and e < r["tolerance"] for e in r["max_absolute_errors"])]
    print(json.dumps(dict(backend=args.backend, passed=0 if args.benchmark_target_tokens else len(rows) - len(failed),
                          failed=len(failed), cases=rows), indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
