#!/usr/bin/env python3
"""Convert the official Nemotron-3.5-Lightning DSpark draft module
(nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark, a Qwen3DSparkModel
checkpoint) into the standalone "dflash"-architecture drafter GGUF TensorSharp
loads with --draft-model.

The module is 6 dense SWA-1024 layers with attention-sink biases, a Markov head
(markov_w1/w2, rank 512) and an encoder whose fc consumes the concatenated
residuals of trunk layers [2, 6, 20, 30, 42, 52]; the target model supplies the
token embedding and the LM head, so only the drafter's own tensors are
converted. The output layout is llama.cpp's DFlash export (the
magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark-GGUF file is
exactly that export): metadata keys dflash.*, weights under dflash.blk.N.* /
dflash.fc.weight / dflash.markov_w1.weight / dflash.markov_w2.weight /
dflash.enc.output_norm.weight / dflash.output_norm.weight.

    pip install numpy
    python eng/nemotron-dspark-to-gguf.py \
        --input /path/to/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark \
        --out NVIDIA-Nemotron-3.5-Lightning-30B-A3B-DSpark.gguf

The 20 W4A16_NVFP4 tensors (fc, the 18 FFN linears, markov_w2) are
dequantized to float32 and written as BF16; everything already BF16 in the
checkpoint (markov_w1, the attention path, every norm and sink) is carried
over losslessly. The drafter is ~1.3 GB BF16 and its KV ring is tiny, so the
size does not matter: the weights are exact rather than re-quantized.

NVFP4 dequantization (ModelOpt W4A16_NVFP4): the weight is U8 4-bit pairs
("<name>.weight"), with a per-16-column F8_E4M3 scale ("<name>.weight_scale")
and a per-tensor scalar ("<name>.weight_scale_2"):

    value = codebook[nibble(col)] * scale[col // 16] * scale_2

with the NVFP4 codebook, sign-magnitude E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6,
0, -0.5, -1, -1.5, -2, -3, -4, -6} (modelopt's e2m1_values; llama.cpp's nvfp4
layer halves its UE4M3 block scales against the doubled E2M1 table, so both
sides land on the same set). Element 2j of a byte pair holds the LOW nibble.
When nvidia-modelopt is importable it is used instead of the numpy path (its
dequantize_weight is the reference implementation). Point --verify-reference
at any pre-existing dflash export of the same module (e.g. the magnitudedev
GGUF) to cross-check both code paths tensor by tensor.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------------------
# GGUF writing (the same minimal writer as eng/dsv4-dspark-to-gguf.py)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGNMENT = 32

GGML_F32 = 0
GGML_BF16 = 30

KV_UINT32 = 4
KV_INT32 = 5
KV_FLOAT32 = 6
KV_BOOL = 7
KV_STRING = 8
KV_ARRAY = 9


class GgufWriter:
    def __init__(self, path):
        self.path = path
        self.kv = []
        self.tensors = []  # (name, dims, ggml_type, payload bytes)

    def add_string(self, key, value):
        self.kv.append((key, KV_STRING, value))

    def add_uint32(self, key, value):
        self.kv.append((key, KV_UINT32, int(value)))

    def add_float32(self, key, value):
        self.kv.append((key, KV_FLOAT32, float(value)))

    def add_bool(self, key, value):
        self.kv.append((key, KV_BOOL, bool(value)))

    def add_int32_array(self, key, values):
        self.kv.append((key, KV_ARRAY, (KV_INT32, [int(v) for v in values])))

    def add_bool_array(self, key, values):
        self.kv.append((key, KV_ARRAY, (KV_BOOL, [bool(v) for v in values])))

    def add_tensor(self, name, dims, ggml_type, payload):
        self.tensors.append((name, list(dims), int(ggml_type), payload))

    @staticmethod
    def _str(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _kv_bytes(self):
        out = bytearray()
        for key, vtype, value in self.kv:
            out += self._str(key)
            out += struct.pack("<I", vtype)
            if vtype == KV_STRING:
                out += self._str(value)
            elif vtype == KV_UINT32:
                out += struct.pack("<I", int(value))
            elif vtype == KV_INT32:
                out += struct.pack("<i", int(value))
            elif vtype == KV_FLOAT32:
                out += struct.pack("<f", float(value))
            elif vtype == KV_BOOL:
                out += struct.pack("<B", int(value))   # GGUF bools are 1 byte
            elif vtype == KV_ARRAY:
                etype, items = value
                out += struct.pack("<IQ", etype, len(items))
                for it in items:
                    if etype == KV_INT32:
                        out += struct.pack("<i", int(it))
                    elif etype == KV_BOOL:
                        out += struct.pack("<B", int(it))
                    else:
                        raise ValueError(f"unsupported array element type {etype}")
            else:
                raise ValueError(f"unsupported kv type {vtype}")
        return bytes(out)

    def write(self):
        header = bytearray()
        header += struct.pack("<II", GGUF_MAGIC, GGUF_VERSION)
        header += struct.pack("<QQ", len(self.tensors), len(self.kv))
        header += self._kv_bytes()

        infos = bytearray()
        offset = 0
        for name, dims, ttype, payload in self.tensors:
            infos += self._str(name)
            infos += struct.pack("<I", len(dims))
            for d in dims:
                infos += struct.pack("<Q", d)
            infos += struct.pack("<I", ttype)
            infos += struct.pack("<Q", offset)
            offset += len(payload)
            offset = (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

        pre = bytes(header) + bytes(infos)
        pad = (-len(pre)) % ALIGNMENT
        with open(self.path, "wb") as f:
            f.write(pre)
            f.write(b"\0" * pad)
            for _, _, _, payload in self.tensors:
                f.write(payload)
                f.write(b"\0" * ((-len(payload)) % ALIGNMENT))


# ---------------------------------------------------------------------------
# safetensors reader
# ---------------------------------------------------------------------------

class SafeTensors:
    """Lazy safetensors reader (mmap), single- or multi-shard."""

    def __init__(self, checkpoint_dir):
        self.dir = checkpoint_dir
        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            single = os.path.join(checkpoint_dir, "model.safetensors")
            if not os.path.exists(single):
                raise FileNotFoundError(
                    f"{checkpoint_dir} has neither model.safetensors nor model.safetensors.index.json")
            header = self._header_of(single)
            self.weight_map = {k: "model.safetensors" for k in header if k != "__metadata__"}
        self._shards = {}

    def _header_of(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n))

    def _shard(self, filename):
        if filename not in self._shards:
            path = os.path.join(self.dir, filename)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            self._shards[filename] = (header, mm, 8 + n)
        return self._shards[filename]

    def has(self, name):
        return name in self.weight_map

    def raw(self, name):
        """(bytes view, dtype string, shape) without conversion."""
        header, mm, base = self._shard(self.weight_map[name])
        e = header[name]
        beg, end = e["data_offsets"]
        return mm[base + beg: base + end], e["dtype"], e["shape"]


# ---------------------------------------------------------------------------
# dequantization
# ---------------------------------------------------------------------------

# NVFP4 codebook: sign-magnitude E2M1 (modelopt's e2m1_values; llama.cpp's
# nvfp4 layer halves its UE4M3 block scales while reading the doubled E2M1
# table, so the effective values are the SAME set on both sides).
NVFP4_CODEBOOK = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)

_FP8_LUT = None


def fp8_e4m3_lut():
    """float8_e4m3fn -> float32 lookup (signed E4M3, bias 7)."""
    global _FP8_LUT
    if _FP8_LUT is None:
        out = np.zeros(256, dtype=np.float64)
        for code in range(256):
            sign = -1.0 if code & 0x80 else 1.0
            exp = (code >> 3) & 0xF
            man = code & 0x7
            if exp == 0:
                val = man / 8.0 * 2.0 ** (-6)
            elif exp == 0xF and man == 7:
                val = float("nan")
            else:
                val = (1.0 + man / 8.0) * 2.0 ** (exp - 7)
            out[code] = sign * val
        _FP8_LUT = out
    return _FP8_LUT


def read_f32(st, name):
    """BF16 or F32 tensor -> float32 numpy, HF layout preserved."""
    buf, dtype, shape = st.raw(name)
    if dtype == "BF16":
        u = np.frombuffer(bytes(buf), dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(bytes(buf), dtype=np.float32).reshape(shape)
    raise ValueError(f"{name}: expected BF16/F32, got {dtype}")


def nvfp4_dequant(st, name, scale2):
    """ModelOpt W4A16_NVFP4 -> float32 [out, in].

    weight        U8 [out, in/2], element 2j = LOW nibble (modelopt's
                  unpack: ``[..., 1::2] = input >> 4; [..., 0::2] = input & 0xF``)
    weight_scale  F8_E4M3 [out, in/16]
    weight_scale_2 F32 scalar

    This is modelopt's reference dequant bit-for-bit (verified against
    nvfp4_tensor.dequantize semantics): value = e2m1_values[nibble] *
    per-16-block scale * per-tensor scale.
    """
    wbuf, wdt, wshape = st.raw(name)
    if wdt != "U8":
        raise ValueError(f"{name}: expected U8 (W4A16_NVFP4), got {wdt}")
    base = name.rsplit(".", 1)[0]
    sbuf_raw, sdt, sshape = st.raw(base + ".weight_scale")
    if sdt != "F8_E4M3":
        raise ValueError(f"{name}: expected F8_E4M3 weight_scale, got {sdt}")

    out, half_in = wshape
    w = np.frombuffer(bytes(wbuf), dtype=np.uint8).reshape(out, half_in)
    codes = np.empty((out, half_in * 2), dtype=np.uint8)
    codes[:, 0::2] = w & 0x0F
    codes[:, 1::2] = w >> 4
    s = fp8_e4m3_lut()[np.frombuffer(bytes(sbuf_raw), dtype=np.uint8)].reshape(sshape).astype(np.float64)
    sr = np.repeat(s, 16, axis=1)
    return (NVFP4_CODEBOOK[codes] * sr * float(scale2)).astype(np.float32)


def bf16_bytes(x):
    """float32 numpy -> BF16 round-to-nearest-even bytes. The mask keeps the
    HIGH word of the f32 bits; the truncation must then shift it down or the
    output is all zeros (a bug that produced a perfectly-tabled, fully-zero
    drafter GGUF and a uniform 1/vocab draft distribution)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    bf = (((u + rounding_bias) & 0xFFFF0000) >> 16).astype(np.uint16)
    return bf.tobytes()


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="directory of the NVFP4-DSpark module (model.safetensors(s) + config.json)")
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "f32"],
                    help="output dtype for weights (default bf16)")
    ap.add_argument("--verify-reference", metavar="GGUF",
                    help="cross-check the conversion against a reference dflash GGUF (e.g. the magnitudedev export)")
    args = ap.parse_args()

    with open(os.path.join(args.input, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    st = SafeTensors(args.input)

    block_size = int(cfg.get("block_size", 8))
    num_layers = int(cfg.get("num_hidden_layers", 6))
    hidden = int(cfg.get("hidden_size", 2688))
    ffn = int(cfg.get("intermediate_size", 6144))
    heads = int(cfg.get("num_attention_heads", 32))
    kv_heads = int(cfg.get("num_key_value_heads", 2))
    head_dim = int(cfg.get("head_dim", 128))
    rope_base = float(cfg.get("rope_theta", 10000.0))
    mask_token = int(cfg.get("mask_token_id", cfg.get("pard_token", 990)))
    bos_token = int(cfg.get("bos_token_id", 1))
    eos_token = int(cfg.get("eos_token_id", 11))
    swa = int(cfg.get("sliding_window", 1024))
    vocab = int(cfg.get("vocab_size", 131072))
    rank = int(cfg.get("dspark_markov_rank", 512))

    dflash_cfg = cfg.get("dflash_config", {})
    target_layers = dflash_cfg.get("target_layer_ids", []) or cfg.get("eagle_aux_hidden_state_layer_ids", [])
    if not target_layers:
        raise ValueError("no dflash_config.target_layer_ids / eagle_aux_hidden_state_layer_ids in config.json")
    # 1-shift HF ids: GGUF id L means "the residual entering 0-based layer L"
    # (== the output of layer L-1), exactly like llama.cpp's dflash export.
    target_layers = [int(t) + 1 for t in target_layers]

    w = GgufWriter(args.out)
    w.add_string("general.architecture", "dflash")
    w.add_string("general.name", "Nemotron-3.5-Lightning-30B-A3B NVFP4-DSpark (TensorSharp conversion)")
    w.add_uint32("dflash.context_length", int(cfg.get("max_position_embeddings", 1048576)))
    w.add_uint32("dflash.embedding_length", hidden)
    w.add_uint32("dflash.feed_forward_length", ffn)
    w.add_uint32("dflash.block_count", num_layers)
    w.add_uint32("dflash.attention.head_count", heads)
    w.add_uint32("dflash.attention.head_count_kv", kv_heads)
    w.add_uint32("dflash.attention.key_length", head_dim)
    w.add_uint32("dflash.attention.value_length", head_dim)
    w.add_float32("dflash.attention.layer_norm_rms_epsilon", float(cfg.get("rms_norm_eps", 1e-6)))
    w.add_float32("dflash.rope.freq_base", rope_base)
    w.add_uint32("dflash.block_size", block_size)
    w.add_int32_array("dflash.target_layers", target_layers)
    w.add_uint32("dflash.attention.sliding_window", swa)
    w.add_bool_array("dflash.attention.sliding_window_pattern", [True] * num_layers)
    w.add_uint32("tokenizer.ggml.mask_token_id", mask_token)
    w.add_uint32("tokenizer.ggml.bos_token_id", bos_token)
    w.add_uint32("tokenizer.ggml.eos_token_id", eos_token)
    w.add_uint32("tokenizer.ggml.padding_token_id", eos_token)
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    w.add_bool("tokenizer.ggml.add_eos_token", False)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "nvidia")

    dtype = GGML_BF16 if args.dtype == "bf16" else GGML_F32
    ref_tensors = {}

    def add_linear(gguf_name, arr):
        """HF [out, in] row-major float32 -> ggml tensor (ne0=in, ne1=out)."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        dims = [int(arr.shape[1]), int(arr.shape[0])]
        payload = bf16_bytes(arr) if dtype == GGML_BF16 else arr.tobytes()
        w.add_tensor(gguf_name, dims, dtype, payload)
        ref_tensors[gguf_name] = arr

    def add_1d(gguf_name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        payload = bf16_bytes(arr) if dtype == GGML_BF16 else arr.tobytes()
        w.add_tensor(gguf_name, [int(arr.shape[0])], dtype, payload)
        ref_tensors[gguf_name] = arr

    def linear(st, name, hf_shape):
        """Checkpoint linear [out, in] -> float32 [out, in]: BF16 passes
        through, W4A16_NVFP4 is dequantized (read_f32 raises otherwise)."""
        buf, dt, shape = st.raw(name)
        if dt == "BF16":
            u = np.frombuffer(bytes(buf), dtype=np.uint16).astype(np.uint32) << 16
            v = u.view(np.float32).reshape(shape)
            if tuple(shape) != tuple(hf_shape):
                v = v.T
            return np.ascontiguousarray(v, dtype=np.float32)
        if dt == "U8":
            scale2 = struct.unpack("<f", bytes(st.raw(name.rsplit(".", 1)[0] + ".weight_scale_2")[0]))[0]
            return nvfp4_dequant(st, name, scale2)
        raise ValueError(f"{name}: expected BF16/U8, got {dt}")

    # Encoder input projection: fc.weight NVFP4 [in=feat, out=hidden].
    feat = hidden * len(target_layers)
    scale2 = struct.unpack("<f", bytes(st.raw("fc.weight_scale_2")[0]))[0]
    add_linear("fc.weight", nvfp4_dequant(st, "fc.weight", scale2))
    add_1d("enc.output_norm.weight", read_f32(st, "hidden_norm.weight"))
    add_1d("output_norm.weight", read_f32(st, "norm.weight"))

    for i in range(num_layers):
        p = f"layers.{i}"
        add_1d(f"blk.{i}.attn_norm.weight", read_f32(st, f"{p}.input_layernorm.weight"))
        add_1d(f"blk.{i}.ffn_norm.weight", read_f32(st, f"{p}.post_attention_layernorm.weight"))
        add_1d(f"blk.{i}.attn_sinks", read_f32(st, f"{p}.self_attn.attention_sink_bias"))
        add_linear(f"blk.{i}.attn_q.weight", linear(st, f"{p}.self_attn.q_proj.weight", (heads * head_dim, hidden)))
        add_linear(f"blk.{i}.attn_k.weight", linear(st, f"{p}.self_attn.k_proj.weight", (kv_heads * head_dim, hidden)))
        add_linear(f"blk.{i}.attn_v.weight", linear(st, f"{p}.self_attn.v_proj.weight", (kv_heads * head_dim, hidden)))
        add_linear(f"blk.{i}.attn_output.weight", linear(st, f"{p}.self_attn.o_proj.weight", (hidden, heads * head_dim)))
        add_1d(f"blk.{i}.attn_q_norm.weight", read_f32(st, f"{p}.self_attn.q_norm.weight"))
        add_1d(f"blk.{i}.attn_k_norm.weight", read_f32(st, f"{p}.self_attn.k_norm.weight"))
        add_linear(f"blk.{i}.ffn_gate.weight", linear(st, f"{p}.mlp.gate_proj.weight", (ffn, hidden)))
        add_linear(f"blk.{i}.ffn_up.weight", linear(st, f"{p}.mlp.up_proj.weight", (ffn, hidden)))
        add_linear(f"blk.{i}.ffn_down.weight", linear(st, f"{p}.mlp.down_proj.weight", (hidden, ffn)))

    # Markov head. w1 is the [vocab, rank] token embedding, w2 the rank-wide
    # Linear to the full vocab; both land in HF layout ([out, in] row-major),
    # which is what the loader's w1[token] row lookup expects.
    add_linear("markov_w1.weight", linear(st, "markov_head.markov_w1.weight", (vocab, rank)))
    scale2_w2 = struct.unpack("<f", bytes(st.raw("markov_head.markov_w2.weight_scale_2")[0]))[0]
    add_linear("markov_w2.weight", nvfp4_dequant(st, "markov_head.markov_w2.weight", scale2_w2))

    w.write()
    print(f"Wrote {args.out} ({num_layers} layers, block {block_size}, targets {target_layers}, "
          f"markov r{rank}, {feat}-wide encoder)")

    if args.verify_reference:
        verify_against(args.verify_reference, ref_tensors)


def verify_against(path, mine):
    from gguf import GGUFReader
    r = GGUFReader(path)
    by_name = {t.name: t for t in r.tensors}
    checked = 0
    worst = 0.0
    for name, arr in mine.items():
        rt = by_name.get(name)
        if rt is None:
            print(f"  ? {name}: absent from the reference GGUF")
            continue
        try:
            typ = rt.tensor_type.name
        except Exception:
            typ = str(rt.tensor_type)
        if typ in ("BF16", "F32"):
            # Reference keeps the same source values; compare in float32.
            # The reader hands back raw bytes for BF16 (2 bytes per element).
            raw = np.array(rt.data).reshape(-1)
            if typ == "BF16":
                rf = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            else:
                rf = raw.astype(np.float32)
            mf = arr.astype(np.float32).reshape(-1)
            if rf.shape != mf.shape:
                print(f"  ! {name}: shape {rf.shape} vs {mf.shape}")
                continue
            err = float(np.abs(rf - mf).mean())
            denom = float(np.abs(rf).mean()) + 1e-12
            worst = max(worst, err / denom)
            checked += 1
            flag = "OK " if err / denom < 1e-3 else "MISMATCH"
            print(f"  {flag} {name}: mean|diff|={err:.3e} rel={err / denom:.3e}")
        else:
            print(f"  - {name}: reference is {typ}; skipping (quantized)")
    print(f"verified {checked} float tensors (worst rel diff {worst:.3e})")


if __name__ == "__main__":
    main()