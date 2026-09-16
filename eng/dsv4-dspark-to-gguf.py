#!/usr/bin/env python3
"""Convert DeepSeek V4/V4.1 DSpark support tensors into a standalone GGUF.

V4.1 uses a distinct architecture tag: converting its tensors does not make
an older V4-only runtime compatible with its delayed mHC computation.

The drafter is three DSV4 blocks (compress_ratio 0) plus a Markov head and a
confidence head; the target model supplies the token embedding and the LM head,
so only the mtp.* tensors are converted. Only the three checkpoint shards that
hold them are needed (see model.safetensors.index.json), not the whole model:

    python eng/dsv4-dspark-to-gguf.py \\
        --checkpoint /path/to/DeepSeek-V4-Flash-0731 \\
        --out DeepSeek-V4-Flash-0731-DSpark.gguf

Quantization: the routed experts are already stored as FP4 (E2M1) with per-32
E8M0 scales, which is exactly GGUF's MXFP4 block layout up to the nibble order,
so `--expert-type mxfp4` repacks them losslessly. `--expert-type q2_k` trades
accuracy for ~40% of the size when the drafter has to share a GPU with the
target's layers. Everything else (FP8 E4M3 with checkpoint-defined block scales) is
dequantized to F32 and stored as Q8_0, except norms/gates which stay F32.
"""

import argparse
import json
import os
import re
import struct

import numpy as np

# ---------------------------------------------------------------------------
# GGUF writing
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGNMENT = 32

GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_BF16 = 30
GGML_MXFP4 = 39

KV_UINT32 = 4
KV_INT32 = 5
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

    def add_int32_array(self, key, values):
        self.kv.append((key, KV_ARRAY, (KV_INT32, [int(v) for v in values])))

    def add_tensor(self, name, dims, ggml_type, payload):
        block, size = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_BF16: (1, 2),
                       GGML_Q8_0: (32, 34), GGML_Q2_K: (256, 84), GGML_MXFP4: (32, 17)}[ggml_type]
        if not dims or any(type(d) is not int or d <= 0 for d in dims) or dims[0] % block:
            raise ValueError(f"{name}: invalid dimensions {dims} for GGML type {ggml_type}")
        expected = int(np.prod(dims)) // block * size
        if len(payload) != expected:
            raise ValueError(f"{name}: payload has {len(payload)} bytes, expected {expected}")
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
                out += struct.pack("<I", value)
            elif vtype == KV_ARRAY:
                etype, items = value
                out += struct.pack("<IQ", etype, len(items))
                for it in items:
                    out += struct.pack("<i", it)
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
# safetensors reading
# ---------------------------------------------------------------------------

class SafeTensors:
    """Lazy multi-shard safetensors reader (mmap per shard)."""

    def __init__(self, checkpoint_dir, fp8_block_shape=(128, 128)):
        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            self.weight_map = json.load(f)["weight_map"]
        self.dir = checkpoint_dir
        self.fp8_block_shape = tuple(fp8_block_shape)
        if len(self.fp8_block_shape) != 2 or any(type(n) is not int or n <= 0 for n in self.fp8_block_shape):
            raise ValueError("FP8 weight_block_size must contain two positive integers")
        self._shards = {}

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
        """(bytes view, dtype string, shape) without any conversion."""
        header, mm, base = self._shard(self.weight_map[name])
        e = header[name]
        beg, end = e["data_offsets"]
        return mm[base + beg: base + end], e["dtype"], e["shape"]


FP8_E4M3_LUT = None


def fp8_e4m3_lut():
    """256-entry lookup for float8_e4m3fn -> float32."""
    global FP8_E4M3_LUT
    if FP8_E4M3_LUT is None:
        out = np.zeros(256, dtype=np.float32)
        for code in range(256):
            sign = -1.0 if code & 0x80 else 1.0
            exp = (code >> 3) & 0xF
            man = code & 0x7
            if exp == 0:
                val = man / 8.0 * 2.0 ** (-6)
            elif exp == 0xF and man == 0x7:
                val = float("nan")
            else:
                val = (1.0 + man / 8.0) * 2.0 ** (exp - 7)
            out[code] = sign * val
        FP8_E4M3_LUT = out
    return FP8_E4M3_LUT


def dequant_block_scaled(st, name):
    """Dequantize `name` to float32, applying its `.scale` sibling if present.

    FP8 E4M3 weights carry per-block E8M0 (or F32) scales; BF16/F32
    tensors are returned as-is.
    """
    buf, dtype, shape = st.raw(name)
    if dtype == "BF16":
        v = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return v.view(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()
    if dtype != "F8_E4M3":
        raise ValueError(f"{name}: unexpected dtype {dtype}")

    w = fp8_e4m3_lut()[np.frombuffer(buf, dtype=np.uint8)].reshape(shape)
    scale_name = name.rsplit(".", 1)[0] + ".scale"
    if not st.has(scale_name):
        raise ValueError(f"{name}: FP8 tensor has no block scales")

    sbuf, sdtype, sshape = st.raw(scale_name)
    if sdtype == "F8_E8M0":
        s = np.exp2(np.frombuffer(sbuf, dtype=np.uint8).astype(np.float32) - 127.0)
    elif sdtype == "F32":
        s = np.frombuffer(sbuf, dtype=np.float32).astype(np.float32)
    else:
        raise ValueError(f"{scale_name}: unexpected scale dtype {sdtype}")
    s = s.reshape(sshape)

    # Use the checkpoint's block size: inferring it from the number of scales
    # shifts every tile boundary when the final tile is partial.
    rows, cols = shape
    br, bc = st.fp8_block_shape
    if list(sshape) != [(rows + br - 1) // br, (cols + bc - 1) // bc]:
        raise ValueError(f"{scale_name}: scale shape {sshape} does not cover {shape} with blocks {[br, bc]}")
    s_full = np.repeat(np.repeat(s, br, axis=0), bc, axis=1)[:rows, :cols]
    return (w * s_full).astype(np.float32)


# ---------------------------------------------------------------------------
# quantization
# ---------------------------------------------------------------------------

def quantize_q8_0(x):
    """[..., n] float32 -> Q8_0 blocks (half d + int8[32])."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1)
    d = (amax / 127.0).astype(np.float32)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    q = np.rint(x * inv[:, None]).clip(-127, 127).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def _make_qkx2(x, nmax=3, nstep=15, rmin=-0.5, rdelta=0.1):
    """llama.cpp's make_qkx2_quants (unweighted): fit x ~ scale*q - min with
    q in [0, nmax], returning (scale, min, q) per row."""
    n = x.shape[1]
    vmin = np.minimum(x.min(axis=1), 0.0)
    vmax = x.max(axis=1)
    span = vmax - vmin
    flat = span <= 0
    span_safe = np.where(flat, 1.0, span)

    iscale = nmax / span_safe
    q = np.rint(iscale[:, None] * (x - vmin[:, None])).clip(0, nmax)
    scale = 1.0 / iscale
    best = (((scale[:, None] * q + vmin[:, None]) - x) ** 2).sum(axis=1)
    best_scale = scale.copy()
    best_min = -vmin.copy()
    best_q = q.copy()

    for step in range(nstep + 1):
        iscale_t = (rmin + rdelta * step + nmax) / span_safe
        qt = np.rint(iscale_t[:, None] * (x - vmin[:, None])).clip(0, nmax)
        sum_l = qt.sum(axis=1)
        sum_l2 = (qt * qt).sum(axis=1)
        sum_xl = (qt * x).sum(axis=1)
        sum_x = x.sum(axis=1)
        D = n * sum_l2 - sum_l * sum_l
        ok = D > 0
        Dsafe = np.where(ok, D, 1.0)
        this_scale = (n * sum_xl - sum_x * sum_l) / Dsafe
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / Dsafe
        neg = this_min > 0
        this_min = np.where(neg, 0.0, this_min)
        this_scale = np.where(neg, np.where(sum_l2 > 0, sum_xl / np.where(sum_l2 > 0, sum_l2, 1.0), 0.0), this_scale)
        mad = (((this_scale[:, None] * qt + this_min[:, None]) - x) ** 2).sum(axis=1)
        take = ok & (mad < best) & (this_scale > 0)
        best = np.where(take, mad, best)
        best_scale = np.where(take, this_scale, best_scale)
        best_min = np.where(take, -this_min, best_min)
        best_q = np.where(take[:, None], qt, best_q)

    best_scale = np.where(flat, 0.0, best_scale)
    best_min = np.where(flat, -vmin, best_min)
    best_q = np.where(flat[:, None], 0.0, best_q)
    return best_scale.astype(np.float32), np.maximum(best_min, 0.0).astype(np.float32), best_q.astype(np.uint8)


def quantize_q2_k(x):
    """[..., n] float32 (n % 256 == 0) -> Q2_K super-blocks (84 bytes each)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 256)
    nsb = x.shape[0]
    sub = x.reshape(nsb * 16, 16)
    scale, mn, q = _make_qkx2(sub)
    scale = scale.reshape(nsb, 16)
    mn = mn.reshape(nsb, 16)
    q = q.reshape(nsb, 16, 16)

    d = scale.max(axis=1) / 15.0
    dmin = mn.max(axis=1) / 15.0
    ls = np.where(d[:, None] > 0, np.rint(scale / np.where(d[:, None] > 0, d[:, None], 1.0)), 0).clip(0, 15).astype(np.uint8)
    lm = np.where(dmin[:, None] > 0, np.rint(mn / np.where(dmin[:, None] > 0, dmin[:, None], 1.0)), 0).clip(0, 15).astype(np.uint8)

    out = np.zeros((nsb, 84), dtype=np.uint8)
    out[:, 0:16] = ls | (lm << 4)

    # qs[gh*32 + pos] holds element gh*128 + j*32 + pos at bit 2*j.
    flat_q = q.reshape(nsb, 256)
    qs = np.zeros((nsb, 64), dtype=np.uint8)
    for gh in range(2):
        for j in range(4):
            beg = gh * 128 + j * 32
            qs[:, gh * 32: gh * 32 + 32] |= (flat_q[:, beg: beg + 32] & 3) << (2 * j)
    out[:, 16:80] = qs
    out[:, 80:82] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 82:84] = dmin.astype(np.float16).view(np.uint8).reshape(-1, 2)
    return out.tobytes()


def repack_mxfp4(packed, scales, rows, cols):
    """DeepSeek FP4 (E2M1, 2 per byte, little nibble first) + per-32 E8M0
    scales -> GGUF MXFP4 blocks (uint8 e + qs[16], low nibble = element j,
    high nibble = element j+16). Pure nibble permutation: lossless."""
    nblk = cols // 32
    p = packed.reshape(rows, nblk, 16)
    lo = p & 0x0F           # elements 0,2,4,...,30 of the block
    hi = (p >> 4) & 0x0F    # elements 1,3,5,...,31
    elems = np.empty((rows, nblk, 32), dtype=np.uint8)
    elems[:, :, 0::2] = lo
    elems[:, :, 1::2] = hi

    out = np.empty((rows, nblk, 17), dtype=np.uint8)
    out[:, :, 0] = scales.reshape(rows, nblk)
    out[:, :, 1:] = elems[:, :, 0:16] | (elems[:, :, 16:32] << 4)
    return out.tobytes()


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def resolve_dspark_config(raw_config, weight_map):
    """Validate draft metadata/index before reading any model tensor payload."""
    cfg = dict(raw_config.get("text_config", raw_config))
    is_v41 = raw_config.get("model_type") == "deepseek_v41" or cfg.get("model_type") == "deepseek_v41_text"
    cfg["dspark_target_architecture"] = "deepseek41" if is_v41 else "deepseek4"
    for key in ("dspark_block_size", "dspark_markov_rank", "dspark_noise_token_id"):
        value = cfg.get(key)
        minimum = 0 if key == "dspark_noise_token_id" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"config has no valid {key}: this checkpoint has no usable DSpark module")
    layers = cfg.get("dspark_target_layer_ids")
    if (not isinstance(layers, list) or not layers or
            any(type(n) is not int or n < 0 for n in layers) or layers != sorted(set(layers))):
        raise ValueError("dspark_target_layer_ids must be nonempty, unique and increasing")
    if "num_hidden_layers" in cfg and layers[-1] >= cfg["num_hidden_layers"]:
        raise ValueError("DSpark target layer is outside the target model")
    if "vocab_size" in cfg and cfg["dspark_noise_token_id"] >= cfg["vocab_size"]:
        raise ValueError("DSpark noise token is outside the target vocabulary")
    stages = sorted(int(m.group(1)) for k in weight_map
                    if (m := re.fullmatch(r"mtp\.(\d+)\.attn_norm\.weight", k)))
    if not stages or stages != list(range(len(stages))):
        raise ValueError("checkpoint must contain contiguous mtp stages starting at zero")
    if cfg.get("num_nextn_predict_layers", len(stages)) != len(stages):
        raise ValueError("DSpark stage count differs from num_nextn_predict_layers")
    count = cfg.get("dspark_n_routed_experts", cfg.get("n_routed_experts"))
    if type(count) is not int or count <= 0:
        raise ValueError("DSpark routed expert count must be a positive integer")
    active = cfg.get("dspark_num_experts_per_tok", cfg.get("num_experts_per_tok"))
    if active is not None and (type(active) is not int or not 0 < active <= count):
        raise ValueError("DSpark active expert count is outside the routed expert count")
    cfg["n_routed_experts"] = count
    cfg["num_experts_per_tok"] = active
    for stage in stages:
        for projection in ("w1", "w2", "w3"):
            pattern = rf"mtp\.{stage}\.ffn\.experts\.(\d+)\.{projection}\.weight"
            actual = sorted(int(m.group(1)) for k in weight_map if (m := re.fullmatch(pattern, k)))
            if actual != list(range(count)):
                raise ValueError(f"mtp.{stage}.{projection}: expert IDs differ from configured count {count}")
    return cfg, len(stages)


def stage_tensors(st, writer, stage, cfg, expert_type, log):
    src = f"mtp.{stage}."
    dst = f"mtp.{stage}."

    def f32(name_src, name_dst):
        v = dequant_block_scaled(st, src + name_src).astype(np.float32)
        writer.add_tensor(dst + name_dst, list(reversed(v.shape)) if v.ndim > 1 else [v.shape[0]],
                          GGML_F32, np.ascontiguousarray(v).tobytes())

    def q8(name_src, name_dst):
        v = dequant_block_scaled(st, src + name_src)
        rows, cols = v.shape
        writer.add_tensor(dst + name_dst, [cols, rows], GGML_Q8_0, quantize_q8_0(v))

    q8("attn.wq_a.weight", "attn_q_a.weight")
    q8("attn.wq_b.weight", "attn_q_b.weight")
    q8("attn.wkv.weight", "attn_kv.weight")
    q8("attn.wo_a.weight", "attn_output_a.weight")
    q8("attn.wo_b.weight", "attn_output_b.weight")
    f32("attn.q_norm.weight", "attn_q_a_norm.weight")
    f32("attn.kv_norm.weight", "attn_kv_a_norm.weight")
    f32("attn.attn_sink", "attn_sinks.weight")
    f32("attn_norm.weight", "attn_norm.weight")
    f32("ffn_norm.weight", "ffn_norm.weight")
    f32("hc_attn_fn", "hc_attn_fn.weight")
    f32("hc_attn_scale", "hc_attn_scale.weight")
    f32("hc_attn_base", "hc_attn_base.weight")
    f32("hc_ffn_fn", "hc_ffn_fn.weight")
    f32("hc_ffn_scale", "hc_ffn_scale.weight")
    f32("hc_ffn_base", "hc_ffn_base.weight")
    f32("ffn.gate.weight", "ffn_gate_inp.weight")
    f32("ffn.gate.bias", "exp_probs_b.bias")
    if st.has(src + "ffn.gate.bias_vl"):
        f32("ffn.gate.bias_vl", "exp_probs_b_vl.bias")
    q8("ffn.shared_experts.w1.weight", "ffn_gate_shexp.weight")
    q8("ffn.shared_experts.w2.weight", "ffn_down_shexp.weight")
    q8("ffn.shared_experts.w3.weight", "ffn_up_shexp.weight")

    n_experts = cfg["n_routed_experts"]
    for w_src, w_dst in (("w1", "ffn_gate_exps.weight"), ("w3", "ffn_up_exps.weight"), ("w2", "ffn_down_exps.weight")):
        chunks = []
        rows = cols = None
        stored_type = stored_shape = None
        for e in range(n_experts):
            name = f"{src}ffn.experts.{e}.{w_src}.weight"
            buf, dtype, shape = st.raw(name)
            if dtype == "I8":
                # FP4: [out, in/2] packed nibbles + [out, in/32] E8M0 scales.
                rows, half = shape
                cols = half * 2
                packed = np.frombuffer(buf, dtype=np.uint8).reshape(rows, half)
                sbuf, sdtype, sshape = st.raw(f"{src}ffn.experts.{e}.{w_src}.scale")
                if sdtype != "F8_E8M0" or cols % 32 or list(sshape) != [rows, cols // 32]:
                    raise ValueError(f"{name}: invalid FP4 scales {sdtype} {sshape}")
                scales = np.frombuffer(sbuf, dtype=np.uint8).reshape(sshape)
                if expert_type == "mxfp4":
                    chunks.append(repack_mxfp4(packed, scales, rows, cols))
                    ttype = GGML_MXFP4
                else:
                    lut = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
                    lo = lut[packed & 0x0F]
                    hi = lut[(packed >> 4) & 0x0F]
                    vals = np.empty((rows, cols), dtype=np.float32)
                    vals[:, 0::2] = lo
                    vals[:, 1::2] = hi
                    s = np.exp2(scales.astype(np.float32) - 127.0)
                    vals *= np.repeat(s, 32, axis=1)
                    chunks.append(quantize_q2_k(vals))
                    ttype = GGML_Q2_K
            else:
                v = dequant_block_scaled(st, name)
                rows, cols = v.shape
                chunks.append(quantize_q2_k(v) if expert_type == "q2_k" else quantize_q8_0(v))
                ttype = GGML_Q2_K if expert_type == "q2_k" else GGML_Q8_0
            if stored_type is not None and (ttype != stored_type or (rows, cols) != stored_shape):
                raise ValueError(f"{name}: experts in one GGUF tensor must have the same shape and encoding")
            stored_type, stored_shape = ttype, (rows, cols)
        writer.add_tensor(dst + w_dst, [cols, rows, n_experts], stored_type, b"".join(chunks))
        log(f"  {dst}{w_dst}: [{cols}, {rows}, {n_experts}] GGML type {stored_type}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="DeepSeek-V4-Flash checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expert-type", choices=["mxfp4", "q2_k"], default="mxfp4")
    args = ap.parse_args()

    with open(os.path.join(args.checkpoint, "config.json"), "r", encoding="utf-8") as f:
        raw_config = json.load(f)
    block_shape = raw_config.get("quantization_config", {}).get("weight_block_size", [128, 128])
    st = SafeTensors(args.checkpoint, block_shape)
    cfg, n_stages = resolve_dspark_config(raw_config, st.weight_map)

    def log(msg):
        print(msg, flush=True)

    log(f"DSpark: {n_stages} stage(s), block_size={cfg['dspark_block_size']}, "
        f"markov_rank={cfg['dspark_markov_rank']}, target_layers={cfg['dspark_target_layer_ids']}, "
        f"experts={args.expert_type}")

    w = GgufWriter(args.out)
    w.add_string("general.architecture", cfg["dspark_target_architecture"] + "-dspark")
    w.add_string("dspark.target_architecture", cfg["dspark_target_architecture"])
    w.add_uint32("dspark.expert_count", cfg["n_routed_experts"])
    if cfg["num_experts_per_tok"] is not None:
        w.add_uint32("dspark.expert_used_count", cfg["num_experts_per_tok"])
    w.add_string("general.name", os.path.basename(os.path.normpath(args.checkpoint)) + " DSpark")
    w.add_uint32("general.alignment", ALIGNMENT)
    w.add_uint32("dspark.n_layers", n_stages)
    w.add_uint32("dspark.stage_count", n_stages)
    w.add_uint32("dspark.block_size", cfg["dspark_block_size"])
    w.add_uint32("dspark.markov_rank", cfg["dspark_markov_rank"])
    w.add_uint32("dspark.noise_token_id", cfg["dspark_noise_token_id"])
    w.add_int32_array("dspark.target_layer_ids", cfg["dspark_target_layer_ids"])

    for stage in range(n_stages):
        log(f"stage {stage}")
        stage_tensors(st, w, stage, cfg, args.expert_type, log)

    last = n_stages - 1
    main_norm = dequant_block_scaled(st, "mtp.0.main_norm.weight").astype(np.float32)
    w.add_tensor("mtp.0.main_norm.weight", [main_norm.shape[0]], GGML_F32, main_norm.tobytes())
    mp = dequant_block_scaled(st, "mtp.0.main_proj.weight")
    w.add_tensor("mtp.0.main_proj.weight", [mp.shape[1], mp.shape[0]], GGML_Q8_0, quantize_q8_0(mp))

    head_tensors = [
        (f"mtp.{last}.norm.weight", f"mtp.{last}.norm.weight", GGML_F32),
        (f"mtp.{last}.confidence_head.proj.weight", f"mtp.{last}.confidence_head.proj.weight", GGML_F32),
    ]
    if cfg["dspark_target_architecture"] == "deepseek4":
        head_tensors.extend([
        (f"mtp.{last}.hc_head_fn", f"mtp.{last}.hc_head_fn.weight", GGML_F32),
        (f"mtp.{last}.hc_head_scale", f"mtp.{last}.hc_head_scale.weight", GGML_F32),
        (f"mtp.{last}.hc_head_base", f"mtp.{last}.hc_head_base.weight", GGML_F32),
        ])
    for src, dst, ttype in head_tensors:
        v = dequant_block_scaled(st, src).astype(np.float32)
        dims = list(reversed(v.shape)) if v.ndim > 1 else [v.shape[0]]
        w.add_tensor(dst, dims, ttype, np.ascontiguousarray(v).tobytes())

    # The Markov head is gathered per token (w1) and used as a matmul (w2); both
    # are vocab-sized, so w1 stays BF16 (cheap to dequantize once at load) and
    # w2 is quantized like the other projections.
    v41 = cfg["dspark_target_architecture"] == "deepseek41"
    w1_name = "embed" if v41 else "markov_w1"
    w2_name = "head" if v41 else "markov_w2"
    w1 = dequant_block_scaled(st, f"mtp.{last}.markov_head.{w1_name}.weight").astype(np.float32)
    w1_bf16 = (w1.view(np.uint32) >> 16).astype(np.uint16)
    w.add_tensor(f"mtp.{last}.markov_head.markov_w1.weight", [w1.shape[1], w1.shape[0]], GGML_BF16, w1_bf16.tobytes())
    w2 = dequant_block_scaled(st, f"mtp.{last}.markov_head.{w2_name}.weight")
    w.add_tensor(f"mtp.{last}.markov_head.markov_w2.weight", [w2.shape[1], w2.shape[0]], GGML_Q8_0, quantize_q8_0(w2))

    log(f"writing {args.out} ({len(w.tensors)} tensors)")
    w.write()
    log(f"done: {os.path.getsize(args.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
