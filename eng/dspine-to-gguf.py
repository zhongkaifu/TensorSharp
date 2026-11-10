#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Convert a DSpine speculative drafter (Tencent DSpine, https://github.com/Tencent/DSpine, Apache-2.0) into the
"dflash"-architecture GGUF TensorSharp loads with --draft-model.

Pinned input: tencent/Qwen3.8-27B-DSpine-b16 at revision 5f54174d301dbca65d9aa4448dc47e56732d9a12 (config.json git
blob f755743e..., model.safetensors sha256 ac90f1df...). Download those two files into one directory, then:

    python3 eng/dspine-to-gguf.py --input DIR --out Qwen3.8-27B-DSpine-b16-Q4_K_M.gguf --type q4_k_m

Every --type except bf16 needs a BUILT TensorSharp native library (TensorSharp.GGML.Native/build-macos.sh or
build-linux.sh): the quantizer is upstream ggml's, loaded from it. The script looks in the repo's build output
(TensorSharp.GGML.Native/build/) unless --ggml-lib names the file.

A DSpine draft is a DFlash block drafter (fc over the concatenated target features -> hidden_norm -> context K/V,
5 Qwen3 layers over [anchor, MASK x 15]) with block-causal attention and a code bridge written after every layer.
The file therefore uses the DFlash layout (llama.cpp's dflash export, as the DFlash2 GGUFs): dflash.* keys and the
fc / enc.output_norm / output_norm / blk.N.* tensors; the target supplies the token embedding and the LM head. The
DSpine parts are:

    dflash.attention.causal = true          block-causal draft attention (DFlash files say false)
    dflash.dspine.code_size / rank / eps    the code bridge (selected_prefix_code_bridge)
    dflash.dspine.head_topk                 K of the exact-sender last-write refinement (serial_table)
    dflash.dspine.version = 1               this export's layout
    dspine.codebook.weight      [V, C]      selected_prefix_bridge.codebook
    dspine.bridge.weight        [3R, C]     [gate.weight; predecessor_gate.weight; prev.weight] (the "packed" weight)
    dspine.gate.bias            [R]         gate.bias (F32)
    blk.N.dspine_read.weight    [C, H]      early_read_weights[N], read_weight for the last layer
    blk.N.dspine_write.weight   [H, R]      feature_write_weights[N]

dflash.target_layers is the checkpoint's target_layer_ids + 1 (the DFlash converters' convention: id L is the
residual ENTERING target layer L, i.e. the output of layer L-1; DSpine's hooks read layer OUTPUTS).

The backbone (fc and the layers' linears) is written as --type bf16 (exact), q8_0, q4_k, or q4_k_m (q4_k with q6_k
for attn_v / ffn_down on llama.cpp's use_more_bits layers, the mix of the z-lab DFlash2 Q4_K_M file). Quantized
types are produced by upstream ggml's own reference quantizer, ggml_quantize_chunk, loaded with ctypes from the
TensorSharp native library, because gguf-py has no K-quant encoder. The bridge (codebook, read/write
projections, packed gate weights) stays BF16 as the reference requires, the norms and the gate bias F32.

Refused (the reference loaders refuse them too): terminal-enabled bridges (a nonzero out.weight or
terminal_disabled_version != 1), non-independent writes, a missing feature write on any layer, non-causal blocks,
DSpark next-token rows, DFlash2 selectors / convolutions, non-default RoPE, and full-attention drafters (TensorSharp's
draft context is a sliding-window ring).
"""

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import math
import os
import struct
import sys

import numpy as np

PINNED = {
    # revision -> (config.json git blob sha1, model.safetensors sha256)
    "5f54174d301dbca65d9aa4448dc47e56732d9a12": (
        "f755743efeb3b4f3e987f39de687aa94e7c8a801",
        "ac90f1df842ea66b23e4224f3e86624add547526f689ffe9e9c6b7a5937116b1",
    ),
}
DEFAULT_REVISION = "5f54174d301dbca65d9aa4448dc47e56732d9a12"
SOURCE_REPO = "tencent/Qwen3.8-27B-DSpine-b16"

# ggml types and their (block elements, block bytes)
GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q4_K, GGML_Q6_K, GGML_BF16 = 0, 1, 8, 12, 14, 30
BLOCKS = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_BF16: (1, 2), GGML_Q8_0: (32, 34), GGML_Q4_K: (256, 144),
          GGML_Q6_K: (256, 210)}
TYPE_NAMES = {GGML_F32: "F32", GGML_BF16: "BF16", GGML_Q8_0: "Q8_0", GGML_Q4_K: "Q4_K", GGML_Q6_K: "Q6_K"}
FILE_TYPES = {"bf16": 32, "q8_0": 7, "q4_k": 14, "q4_k_m": 15}   # llama_ftype values (general.file_type)

# ---------------------------------------------------------------------------------------------------------------
# input pinning
# ---------------------------------------------------------------------------------------------------------------


def git_blob_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_pins(input_dir, revision, skip_sha256):
    if revision not in PINNED:
        sys.exit(f"error: no pinned hashes for revision {revision}; known: {sorted(PINNED)}")
    cfg_sha1, st_sha256 = PINNED[revision]
    got = git_blob_sha1(os.path.join(input_dir, "config.json"))
    if got != cfg_sha1:
        sys.exit(f"error: config.json is git blob {got}, revision {revision} pins {cfg_sha1}")
    if skip_sha256:
        print("  model.safetensors sha256 NOT verified (--skip-sha256)")
        return
    print("  hashing model.safetensors ...", flush=True)
    got = sha256_file(os.path.join(input_dir, "model.safetensors"))
    if got != st_sha256:
        sys.exit(f"error: model.safetensors sha256 {got}, revision {revision} pins {st_sha256}")
    print(f"  input matches revision {revision}")


# ---------------------------------------------------------------------------------------------------------------
# safetensors (single file, read through a memory map)
# ---------------------------------------------------------------------------------------------------------------
class SafeTensors:
    DTYPES = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16, "I64": np.int64}

    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        self.header = header
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def names(self):
        return list(self.header)

    def info(self, name):
        h = self.header[name]
        return h["dtype"], tuple(h["shape"])

    def raw(self, name):
        h = self.header[name]
        a, b = h["data_offsets"]
        dt = self.DTYPES[h["dtype"]]
        arr = self.mm[self.base + a:self.base + b].view(dt)
        return arr.reshape(h["shape"]) if h["shape"] else arr.reshape(())

    def bf16(self, name, shape=None):
        """The tensor's BF16 bits (uint16), checked to be BF16 and of `shape`."""
        dtype, got = self.info(name)
        if dtype != "BF16":
            sys.exit(f"error: {name} is {dtype}, expected BF16")
        if shape is not None and tuple(got) != tuple(shape):
            sys.exit(f"error: {name} has shape {got}, expected {shape}")
        return self.raw(name)

    def f32(self, name, shape=None):
        return bf16_to_f32(self.bf16(name, shape))

    def int_scalar(self, name):
        dtype, shape = self.info(name)
        if dtype != "I64" or shape != ():
            sys.exit(f"error: {name} is {dtype}{list(shape)}, expected an I64 scalar")
        return int(self.raw(name))


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# config validation (the DSpine inference contract, as dspine_mlx/config.py and sglang_dspine/models/dspine.py state it)
# ---------------------------------------------------------------------------------------------------------------
def require(cond, msg):
    if not cond:
        sys.exit(f"error: unsupported DSpine checkpoint: {msg}")


def parse_config(raw):
    df = raw.get("dflash_config")
    require(isinstance(df, dict), "missing dflash_config")
    for key in ("selector_rank", "selector_top_k", "projector_type", "attention_sink_bias", "conv_kernel_size",
                "conv_group_size"):
        require(not df.get(key), f"dflash_config.{key} is a DFlash2/DSpark feature, not part of a DSpine draft")
    require(df.get("attention_value_scale") is None, "attention_value_scale")
    require(df.get("selected_prefix_feature_reliability") is None, "selected_prefix_feature_reliability")
    require(not df.get("selected_prefix_feature_next_token_rows", False),
            "selected_prefix_feature_next_token_rows (DSpark row layout)")
    require(df.get("selected_prefix_terminal") is False, "selected_prefix_terminal must be false")
    require(df.get("selected_prefix_decode_mode") == "serial", "selected_prefix_decode_mode must be serial")
    require(df.get("selected_prefix_feature_writes") is True, "selected_prefix_feature_writes must be true")
    require(df.get("selected_prefix_feature_write_mode") == "independent", "writes must be independent")
    require(df.get("draft_block_causal") is True, "draft_block_causal must be true")
    require(not df.get("selected_prefix_terminal_residual_rank"), "terminal token residual")
    bridge = df.get("selected_prefix_code_bridge")
    require(isinstance(bridge, dict), "missing selected_prefix_code_bridge")

    L = int(raw["num_hidden_layers"])
    feature = [int(x) - 1 for x in df.get("selected_prefix_feature_layers") or []]
    require(feature == list(range(L)), f"every draft layer must carry a bridge write (got {feature})")

    found = []
    refine = df.get("selected_prefix_last_write_refine")
    if refine is not None:
        require(set(refine) == {"kind", "version", "head_topk", "weight"}, f"refine keys {sorted(refine)}")
        require(refine["kind"] == "exact_sender" and int(refine["version"]) == 1, "refine must be exact_sender v1")
        require(float(refine["weight"]) >= 0, "refine weight must be >= 0")
        found.append(int(refine["head_topk"]))
    second = raw.get("code_bridge_second_pass")
    if second is not None:
        require(set(second) <= {"mode", "first_exact", "head_topk"}, "code_bridge_second_pass keys")
        require(second.get("mode") == "serial_table", "code_bridge_second_pass.mode must be serial_table")
        require(int(second.get("first_exact", -1)) == L - 1, "serial_table must redo the last write only")
        found.append(int(second["head_topk"]))
    require(found and len(set(found)) == 1, f"refinement width missing or inconsistent: {found}")
    k = found[0]
    require(k & (k - 1) == 0 and 2 <= k <= 64, f"head_topk {k} must be a power of two in [2, 64]")

    archs = raw.get("architectures") or []
    require(not archs or any(a in ("DFlashDraftModel", "DSpineDraftModel") for a in archs), f"architectures {archs}")
    require(raw.get("model_type") in (None, "qwen3"), f"model_type {raw.get('model_type')}")
    require(not raw.get("attention_bias", False), "attention_bias")
    require(raw.get("hidden_act", "silu") == "silu", f"hidden_act {raw.get('hidden_act')}")
    require(not raw.get("tie_word_embeddings", False), "tie_word_embeddings (the draft borrows the target head)")
    require(float(raw.get("partial_rotary_factor") or 1.0) == 1.0, "partial_rotary_factor")
    require(raw.get("rope_scaling") is None, "rope_scaling")
    rope = raw.get("rope_parameters") or {}
    require(rope.get("rope_type", "default") == "default", f"rope_type {rope.get('rope_type')}")
    theta = raw.get("rope_theta", rope.get("rope_theta"))
    require(theta is not None, "rope_theta")
    vocab = int(raw["vocab_size"])
    require(int(raw.get("draft_vocab_size", vocab)) == vocab, "draft_vocab_size != vocab_size")

    types = raw.get("layer_types") or []
    window = raw.get("sliding_window")
    require(len(types) == L and all(t == "sliding_attention" for t in types) and isinstance(window, int)
            and window > 0, "TensorSharp's draft context is a sliding-window ring: every layer must be "
            f"sliding_attention with a window (got {types}, window {window})")
    require(raw.get("is_causal", True) is True, "is_causal must be true")

    return dict(
        L=L, H=int(raw["hidden_size"]), FF=int(raw["intermediate_size"]), NH=int(raw["num_attention_heads"]),
        NKV=int(raw["num_key_value_heads"]), HD=int(raw["head_dim"]), V=vocab, eps=float(raw["rms_norm_eps"]),
        theta=float(theta), block=int(raw["block_size"]), mask=int(df["mask_token_id"]),
        targets=[int(x) for x in df["target_layer_ids"]], n_target=int(raw["num_target_layers"]),
        window=int(window), ctx=int(raw.get("max_position_embeddings", 262144)),
        C=int(bridge["code_size"]), R=int(bridge["rank"]), beps=float(bridge["eps"]), K=k,
        eos=raw.get("eos_token_id"), pad=raw.get("pad_token_id"),
    )


# ---------------------------------------------------------------------------------------------------------------
# quantization through upstream ggml (ggml_quantize_chunk)
# ---------------------------------------------------------------------------------------------------------------
class GgmlQuant:
    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)
        f = self.lib.ggml_quantize_chunk
        f.restype = ctypes.c_size_t
        f.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                      ctypes.c_int64, ctypes.c_void_p]
        self.fn = f
        req = self.lib.ggml_quantize_requires_imatrix
        req.restype = ctypes.c_bool
        req.argtypes = [ctypes.c_int]
        for t in (GGML_Q8_0, GGML_Q4_K, GGML_Q6_K):
            if req(t):
                sys.exit(f"error: ggml says type {t} needs an importance matrix")

    def quantize(self, gtype, x, threads):
        """x float32 [rows, n_per_row] -> the quantized rows, rows in parallel chunks."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        rows, n = x.shape
        be, bb = BLOCKS[gtype]
        if n % be:
            sys.exit(f"error: row length {n} is not a multiple of the {TYPE_NAMES[gtype]} block ({be})")
        row_bytes = n // be * bb
        out = np.empty(rows * row_bytes, dtype=np.uint8)
        step = max(1, (rows + threads - 1) // threads)

        def run(r0):
            r1 = min(rows, r0 + step)
            # ggml indexes dst by `start` (elements), so both pointers are the tensor bases.
            got = self.fn(gtype, x.ctypes.data, out.ctypes.data, r0 * n, r1 - r0, n, None)
            if got != (r1 - r0) * row_bytes:
                raise RuntimeError(f"ggml_quantize_chunk wrote {got} bytes, expected {(r1 - r0) * row_bytes}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
            list(ex.map(run, range(0, rows, step)))
        return out


# ---------------------------------------------------------------------------------------------------------------
# a streaming GGUF writer (tensor payloads are produced one at a time, after the header)
# ---------------------------------------------------------------------------------------------------------------
GGUF_MAGIC, GGUF_VERSION, ALIGN = 0x46554747, 3, 32
KV_U32, KV_I32, KV_F32, KV_BOOL, KV_STR, KV_ARR = 4, 5, 6, 7, 8, 9


class GgufStream:
    def __init__(self):
        self.kv = []
        self.tensors = []   # (name, ggml dims (ne0 first), type, nbytes, producer)

    def u32(self, k, v): self.kv.append((k, KV_U32, int(v)))
    def f32(self, k, v): self.kv.append((k, KV_F32, float(v)))
    def bool(self, k, v): self.kv.append((k, KV_BOOL, bool(v)))
    def str(self, k, v): self.kv.append((k, KV_STR, str(v)))
    def i32s(self, k, v): self.kv.append((k, KV_ARR, (KV_I32, [int(x) for x in v])))
    def bools(self, k, v): self.kv.append((k, KV_ARR, (KV_BOOL, [bool(x) for x in v])))

    def tensor(self, name, dims, gtype, producer):
        be, bb = BLOCKS[gtype]
        n = 1
        for d in dims:
            n *= d
        if dims[0] % be:
            sys.exit(f"error: {name}: row {dims[0]} is not a multiple of the {TYPE_NAMES[gtype]} block")
        self.tensors.append((name, list(dims), gtype, n // be * bb, producer))

    @staticmethod
    def _s(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    def _value(self, t, v):
        if t == KV_U32: return struct.pack("<I", v)
        if t == KV_I32: return struct.pack("<i", v)
        if t == KV_F32: return struct.pack("<f", v)
        if t == KV_BOOL: return struct.pack("<B", 1 if v else 0)
        if t == KV_STR: return self._s(v)
        et, items = v
        return struct.pack("<IQ", et, len(items)) + b"".join(self._value(et, x) for x in items)

    def write(self, path):
        head = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(self.tensors), len(self.kv)))
        for k, t, v in self.kv:
            head += self._s(k) + struct.pack("<I", t) + self._value(t, v)
        offset = 0
        offsets = []
        for name, dims, gtype, nbytes, _ in self.tensors:
            offsets.append(offset)
            head += self._s(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
            head += struct.pack("<IQ", gtype, offset)
            offset += (nbytes + ALIGN - 1) // ALIGN * ALIGN
        head += b"\0" * ((-len(head)) % ALIGN)
        tmp = path + ".partial"
        with open(tmp, "wb") as f:
            f.write(head)
            for (name, dims, gtype, nbytes, producer), off in zip(self.tensors, offsets):
                payload = producer()
                data = payload.tobytes() if isinstance(payload, np.ndarray) else bytes(payload)
                if len(data) != nbytes:
                    sys.exit(f"error: {name}: produced {len(data)} bytes, expected {nbytes}")
                f.write(data)
                f.write(b"\0" * ((-nbytes) % ALIGN))
                print(f"  {name:40s} {TYPE_NAMES[gtype]:5s} {dims}", flush=True)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------


def find_native_lib():
    """The native library in this repo's build output, if it was built."""
    build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TensorSharp.GGML.Native", "build")
    for name in ("libGgmlOps.dylib", "libGgmlOps.so", os.path.join("Release", "GgmlOps.dll"), "GgmlOps.dll"):
        path = os.path.normpath(os.path.join(build, name))
        if os.path.isfile(path):
            return path
    return None


def use_more_bits(i, n):
    """llama.cpp's Q4_K_M rule for attn_v / ffn_down (layers 2 and 4 of a 5-layer drafter)."""
    return i < n // 8 or i >= 7 * n // 8 or (i - n // 8) % 3 == 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="directory with config.json and model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="q4_k_m", choices=sorted(FILE_TYPES), help="backbone weights")
    ap.add_argument("--ggml-lib", help="the built TensorSharp native library (libGgmlOps.dylib / .so / "
                    "GgmlOps.dll), whose upstream ggml quantizes every --type but bf16; default: the repo's "
                    "TensorSharp.GGML.Native/build output")
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="the pinned source revision the input must match")
    ap.add_argument("--skip-sha256", action="store_true", help="skip hashing the 4 GB safetensors (config is "
                    "still checked); for repeated local conversions of an already verified file")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    print(f"source {SOURCE_REPO}@{args.revision}")
    verify_pins(args.input, args.revision, args.skip_sha256)
    raw_cfg = json.load(open(os.path.join(args.input, "config.json")))
    c = parse_config(raw_cfg)
    st = SafeTensors(os.path.join(args.input, "model.safetensors"))
    L, H, FF, NH, NKV, HD, V, C, R = (c[k] for k in ("L", "H", "FF", "NH", "NKV", "HD", "V", "C", "R"))
    FEAT = len(c["targets"]) * H
    for t in c["targets"]:
        require(0 <= t < c["n_target"] - 1, f"target layer {t} (the drafter reads layer outputs below the last)")

    # checkpoint versions and the disabled terminal projection
    for name in ("draft_block_causal_version", "last_write_refine_version",
                 "selected_prefix_bridge.independent_write_version",
                 "selected_prefix_bridge.terminal_disabled_version"):
        require(st.int_scalar(name) == 1, f"{name} = {st.int_scalar(name)}, expected 1")
    out_w = st.bf16("selected_prefix_bridge.out.weight", (C, R))
    require(not np.any(out_w & np.uint16(0x7FFF)), "the terminal projection (out.weight) is nonzero")

    expected = {"fc.weight", "hidden_norm.weight", "norm.weight", "draft_block_causal_version",
                "last_write_refine_version"}
    expected |= {f"selected_prefix_bridge.{n}" for n in (
        "codebook", "early_read_weights", "feature_write_weights", "gate.bias", "gate.weight",
        "independent_write_version", "out.weight", "predecessor_gate.weight", "prev.weight", "read_weight",
        "terminal_disabled_version")}
    for i in range(L):
        expected |= {f"layers.{i}.{n}" for n in (
            "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight",
            "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
            "mlp.down_proj.weight")}
    got = {n.removeprefix("model.") for n in st.names()}
    require(got == expected, f"tensor set mismatch: missing {sorted(expected - got)}, extra {sorted(got - expected)}")

    quant = None
    if args.type != "bf16":
        lib = args.ggml_lib or find_native_lib()
        if lib is None:
            sys.exit(f"error: --type {args.type} needs the built TensorSharp native library (upstream ggml's "
                     "quantizer); build TensorSharp.GGML.Native or pass --ggml-lib")
        print(f"  quantizing with ggml from {lib}")
        quant = GgmlQuant(lib)

    w = GgufStream()
    w.str("general.architecture", "dflash")
    w.str("general.name", "Qwen3.8-27B-DSpine-b16 (TensorSharp conversion)")
    w.str("general.source.url", f"https://huggingface.co/{SOURCE_REPO}")
    w.str("general.license", "apache-2.0")
    w.u32("general.file_type", FILE_TYPES[args.type])
    w.u32("general.quantization_version", 2)
    w.u32("dflash.context_length", c["ctx"])
    w.u32("dflash.embedding_length", H)
    w.u32("dflash.feed_forward_length", FF)
    w.u32("dflash.block_count", L)
    w.u32("dflash.attention.head_count", NH)
    w.u32("dflash.attention.head_count_kv", NKV)
    w.u32("dflash.attention.key_length", HD)
    w.u32("dflash.attention.value_length", HD)
    w.f32("dflash.attention.layer_norm_rms_epsilon", c["eps"])
    w.bool("dflash.attention.causal", True)
    w.f32("dflash.rope.freq_base", c["theta"])
    w.u32("dflash.block_size", c["block"])
    w.i32s("dflash.target_layers", [t + 1 for t in c["targets"]])
    w.u32("dflash.attention.sliding_window", c["window"])
    w.bools("dflash.attention.sliding_window_pattern", [True] * L)
    w.u32("dflash.dspine.version", 1)
    w.u32("dflash.dspine.code_size", C)
    w.u32("dflash.dspine.rank", R)
    w.f32("dflash.dspine.eps", c["beps"])
    w.u32("dflash.dspine.head_topk", c["K"])
    w.str("dflash.dspine.source_revision", args.revision)
    w.u32("tokenizer.ggml.mask_token_id", c["mask"])
    if c["eos"] is not None:
        w.u32("tokenizer.ggml.eos_token_id", c["eos"])
    if c["pad"] is not None:
        w.u32("tokenizer.ggml.padding_token_id", c["pad"])

    def backbone_type(kind, layer):
        if args.type == "bf16":
            return GGML_BF16
        if args.type == "q8_0":
            return GGML_Q8_0
        if args.type == "q4_k_m" and kind in ("attn_v", "ffn_down") and use_more_bits(layer, L):
            return GGML_Q6_K
        return GGML_Q4_K

    def linear(gguf_name, st_name, shape, kind, layer=-1):
        gtype = backbone_type(kind, layer)

        def produce():
            bits = st.bf16(st_name, shape)
            if gtype == GGML_BF16:
                return np.ascontiguousarray(bits)
            return quant.quantize(gtype, bf16_to_f32(bits).reshape(shape[0], -1), args.threads)

        w.tensor(gguf_name, [shape[1], shape[0]], gtype, produce)

    def bf16_tensor(gguf_name, getter, shape):
        w.tensor(gguf_name, list(reversed(shape)), GGML_BF16, lambda: np.ascontiguousarray(getter()))

    def f32_tensor(gguf_name, st_name, shape):
        w.tensor(gguf_name, list(reversed(shape)), GGML_F32, lambda: st.f32(st_name, shape))

    linear("fc.weight", "fc.weight", (H, FEAT), "fc")
    f32_tensor("enc.output_norm.weight", "hidden_norm.weight", (H,))
    f32_tensor("output_norm.weight", "norm.weight", (H,))
    early = st.bf16("selected_prefix_bridge.early_read_weights", (L - 1, C, H))
    writes = st.bf16("selected_prefix_bridge.feature_write_weights", (L, H, R))
    for i in range(L):
        p = f"layers.{i}"
        f32_tensor(f"blk.{i}.attn_norm.weight", f"{p}.input_layernorm.weight", (H,))
        f32_tensor(f"blk.{i}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight", (H,))
        f32_tensor(f"blk.{i}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight", (HD,))
        f32_tensor(f"blk.{i}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight", (HD,))
        linear(f"blk.{i}.attn_q.weight", f"{p}.self_attn.q_proj.weight", (NH * HD, H), "attn_q", i)
        linear(f"blk.{i}.attn_k.weight", f"{p}.self_attn.k_proj.weight", (NKV * HD, H), "attn_k", i)
        linear(f"blk.{i}.attn_v.weight", f"{p}.self_attn.v_proj.weight", (NKV * HD, H), "attn_v", i)
        linear(f"blk.{i}.attn_output.weight", f"{p}.self_attn.o_proj.weight", (H, NH * HD), "attn_output", i)
        linear(f"blk.{i}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", (FF, H), "ffn_gate", i)
        linear(f"blk.{i}.ffn_up.weight", f"{p}.mlp.up_proj.weight", (FF, H), "ffn_up", i)
        linear(f"blk.{i}.ffn_down.weight", f"{p}.mlp.down_proj.weight", (H, FF), "ffn_down", i)
        if i < L - 1:
            bf16_tensor(f"blk.{i}.dspine_read.weight", lambda i=i: early[i], (C, H))
        else:
            bf16_tensor(f"blk.{i}.dspine_read.weight",
                        lambda: st.bf16("selected_prefix_bridge.read_weight", (C, H)), (C, H))
        bf16_tensor(f"blk.{i}.dspine_write.weight", lambda i=i: writes[i], (H, R))
    bf16_tensor("dspine.codebook.weight", lambda: st.bf16("selected_prefix_bridge.codebook", (V, C)), (V, C))
    bf16_tensor("dspine.bridge.weight", lambda: np.concatenate([
        st.bf16("selected_prefix_bridge.gate.weight", (R, C)),
        st.bf16("selected_prefix_bridge.predecessor_gate.weight", (R, C)),
        st.bf16("selected_prefix_bridge.prev.weight", (R, C))]), (3 * R, C))
    f32_tensor("dspine.gate.bias", "selected_prefix_bridge.gate.bias", (R,))

    print(f"writing {args.out} ({args.type} backbone, BF16 bridge, {len(w.tensors)} tensors)")
    w.write(args.out)
    print(f"done: {os.path.getsize(args.out) / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
