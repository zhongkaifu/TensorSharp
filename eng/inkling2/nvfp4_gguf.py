#!/usr/bin/env python3
"""Inkling 2 Mini GGUF with the official checkpoint's NVFP4 experts.

Copies a ggml-org Inkling-2-Mini GGUF (IQ4_NL) tensor for tensor, except blk.{l}.ffn_gate_up_exps.weight and
blk.{l}.ffn_down_exps.weight of the NVFP4 layers, which are taken from the official checkpoint's experts.w13 / w2
(U8 e2m1 pairs + F8_E4M3 block scales of 16) as ggml NVFP4 blocks (64 elements: four UE4M3 scales, then 32 bytes where
qs[s*8 + j] holds element s*16 + j in its low nibble and s*16 + j + 8 in its high one), and adds
blk.{l}.ffn_gate_up_exps.scale / ffn_down_exps.scale: F32 [experts], each expert's outer scale amax / denominator.
The MXFP8 layers keep the GGUF's tensors. --check compares expert 0 of each converted layer with a Q8_0 GGUF.

Run where torch and safetensors are installed (the vLLM image has both):
  python3 eng/inkling2/nvfp4_gguf.py --src Inkling-2-Mini-IQ4_NL.gguf --ckpt model.safetensors       --out Inkling-2-Mini-NVFP4.gguf [--check Inkling-2-Mini-Q8_0.gguf]
On ggml_cuda its NVFP4 experts take FP4 tensor cores for prompt chunks (tsg_cuda_inkling_fp4.cuh).
"""
import argparse
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

GGML_NVFP4 = 40
GGML_F32 = 0
TYPE_SIZE = {0: (1, 4), 1: (1, 2), 8: (32, 34), 20: (32, 18), 40: (64, 36), 30: (1, 2)}


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n)


def skip_value(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes:
        f.seek(sizes[t], 1)
    elif t == 8:
        read_str(f)
    elif t == 9:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if et in sizes:
            f.seek(sizes[et] * n, 1)
        else:
            for _ in range(n):
                skip_value(f, et)
    else:
        raise ValueError(t)


def read_value(f, t):
    fmt = {4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
    if t in fmt:
        return struct.unpack(fmt[t], f.read(struct.calcsize(fmt[t])))[0]
    pos = f.tell()
    skip_value(f, t)
    return None


def parse(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    version, nt, nkv = struct.unpack("<IQQ", f.read(20))
    kv_start = f.tell()
    alignment = 32
    for _ in range(nkv):
        key = read_str(f)
        (t,) = struct.unpack("<I", f.read(4))
        if key == b"general.alignment":
            alignment = read_value(f, t)
        else:
            skip_value(f, t)
    kv_end = f.tell()
    f.seek(kv_start)
    kv_bytes = f.read(kv_end - kv_start)
    tensors = []
    for _ in range(nt):
        name = read_str(f).decode()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        t, off = struct.unpack("<IQ", f.read(12))
        tensors.append([name, dims, t, off])
    info_end = f.tell()
    data_start = (info_end + alignment - 1) // alignment * alignment
    for x in tensors:
        n = 1
        for d in x[1]:
            n *= d
        be, bb = TYPE_SIZE[x[2]]
        x.append(n // be * bb)
    return f, version, nkv, kv_bytes, tensors, data_start, alignment


def gate_up_rows(x):
    """The checkpoint's w13 rows interleave gate and up (g0, u0, g1, u1, ...); the GGUF stacks gate, then up."""
    return np.concatenate([x[:, 0::2], x[:, 1::2]], axis=1)


def nvfp4_blocks(weight_u8, scales_u8):
    """[E, R, K/2] e2m1 pairs (low nibble first) and [E, R, K/16] UE4M3 scales -> ggml NVFP4 [E, R, K/64 * 36] bytes."""
    e, r, half = weight_u8.shape
    k = half * 2
    codes = np.empty((e, r, k), dtype=np.uint8)
    codes[..., 0::2] = weight_u8 & 0x0F
    codes[..., 1::2] = weight_u8 >> 4
    codes = codes.reshape(e, r, k // 64, 4, 16)
    qs = (codes[..., :8] | (codes[..., 8:] << 4)).reshape(e, r, k // 64, 32)
    d = scales_u8.reshape(e, r, k // 64, 4)
    return np.concatenate([d, qs], axis=-1).reshape(e, r, k // 64 * 36)


E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def ue4m3(x):
    x = x.astype(np.int32)
    exp = (x >> 3) & 0xF
    man = x & 0x7
    v = np.where(exp == 0, man / 512.0, (1 + man / 8.0) * np.exp2(exp - 7.0))
    return np.where((x == 0) | (x == 0x7F), 0.0, v).astype(np.float32)


def dequant_nvfp4_row(row_bytes):
    b = row_bytes.reshape(-1, 36)
    d = ue4m3(b[:, :4])                        # [blocks, 4]
    qs = b[:, 4:].reshape(-1, 4, 8)
    lo = E2M1[qs & 0x0F]
    hi = E2M1[qs >> 4]
    vals = np.concatenate([lo, hi], axis=-1)   # [blocks, 4, 16]
    return (vals * d[..., None]).reshape(-1)


def dequant_q8_0_row(row_bytes):
    b = row_bytes.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    q = b[:, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check")
    ap.add_argument("--prefix", default="model.llm.layers.")
    args = ap.parse_args()

    src, version, nkv, kv_bytes, tensors, data_start, alignment = parse(args.src)
    ck = safe_open(args.ckpt, "pt")
    keys = set(ck.keys())

    def ckpt(name):
        t = ck.get_tensor(name)
        return t.view(torch.uint8).numpy() if t.dtype in (torch.uint8, torch.float8_e4m3fn) else t.float().numpy()

    replace = {}     # gguf name -> (layer, proj)
    added = []       # (name, layer, proj)
    for name, dims, t, off, size in tensors:
        for proj, suffix in (("w13", "ffn_gate_up_exps.weight"), ("w2", "ffn_down_exps.weight")):
            if name.startswith("blk.") and name.endswith(suffix):
                layer = int(name.split(".")[1])
                base = f"{args.prefix}{layer}.mlp.experts.{proj}_weight"
                if base + "_amax" in keys:
                    replace[name] = (layer, proj)
                    added.append((name[: -len("weight")] + "scale", layer, proj))
    print(f"converting {len(replace)} expert tensors to NVFP4, adding {len(added)} scale vectors", flush=True)

    out_tensors = []
    for name, dims, t, off, size in tensors:
        if name in replace:
            n = 1
            for d in dims:
                n *= d
            out_tensors.append((name, dims, GGML_NVFP4, n // 64 * 36))
        else:
            out_tensors.append((name, dims, t, size))
    for name, layer, proj in added:
        experts = [x for x in tensors if x[0] in replace and replace[x[0]] == (layer, proj)][0][1][2]
        out_tensors.append((name, (experts,), GGML_F32, experts * 4))

    with open(args.out, "wb") as o:
        o.write(b"GGUF")
        o.write(struct.pack("<IQQ", version, len(out_tensors), nkv))
        o.write(kv_bytes)
        offset = 0
        offsets = []
        for name, dims, t, size in out_tensors:
            nb = name.encode()
            o.write(struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims))
            o.write(struct.pack("<IQ", t, offset))
            offsets.append(offset)
            offset += (size + alignment - 1) // alignment * alignment
        pad = (o.tell() + alignment - 1) // alignment * alignment - o.tell()
        o.write(b"\0" * pad)
        base = o.tell()
        by_name = {x[0]: x for x in tensors}
        for (name, dims, t, size), rel in zip(out_tensors, offsets):
            assert o.tell() == base + rel
            if name in replace:
                layer, proj = replace[name]
                stem = f"{args.prefix}{layer}.mlp.experts.{proj}_weight"
                w = ck.get_tensor(stem)
                s = ck.get_tensor(stem + "_scale")
                experts = w.shape[0]
                for e0 in range(0, experts, 16):
                    wu = w[e0:e0 + 16].view(torch.uint8).numpy()
                    su = s[e0:e0 + 16].view(torch.uint8).numpy()
                    if proj == "w13":
                        wu, su = gate_up_rows(wu), gate_up_rows(su)
                    blk = nvfp4_blocks(wu, su)
                    o.write(blk.tobytes())
                print(f"  {name}: NVFP4 from {stem}", flush=True)
            elif name.endswith(".scale") and name not in by_name:
                layer, proj = [(l, p) for (n, l, p) in added if n == name][0]
                stem = f"{args.prefix}{layer}.mlp.experts.{proj}_weight"
                amax = ck.get_tensor(stem + "_amax").float().numpy()
                den = ck.get_tensor(stem + "_outer_scale_denominator").float().numpy()
                o.write((amax / den[0]).astype(np.float32).tobytes())
            else:
                src.seek(data_start + by_name[name][3])
                left = size
                while left > 0:
                    chunk = src.read(min(left, 1 << 26))
                    o.write(chunk)
                    left -= len(chunk)
            o.write(b"\0" * ((size + alignment - 1) // alignment * alignment - size))
    print(f"wrote {args.out}", flush=True)

    if args.check:
        q8, _, _, _, q8_tensors, q8_start, _ = parse(args.check)
        q8_by = {x[0]: x for x in q8_tensors}
        nv, _, _, _, nv_tensors, nv_start, _ = parse(args.out)
        nv_by = {x[0]: x for x in nv_tensors}
        for name in sorted(replace, key=lambda n: (int(n.split(".")[1]), n))[:6]:
            dims = nv_by[name][1]
            k, rows = dims[0], dims[1]
            scale_name = name[: -len("weight")] + "scale"
            nv.seek(nv_start + nv_by[scale_name][3])
            scale0 = np.frombuffer(nv.read(4), dtype=np.float32)[0]
            for row in (0, rows // 2 - 1, rows // 2, rows - 1):
                nv.seek(nv_start + nv_by[name][3] + row * (k // 64 * 36))
                a = dequant_nvfp4_row(np.frombuffer(nv.read(k // 64 * 36), dtype=np.uint8)) * scale0
                q8.seek(q8_start + q8_by[name][3] + row * (k // 32 * 34))
                b = dequant_q8_0_row(np.frombuffer(q8.read(k // 32 * 34), dtype=np.uint8))
                corr = float(np.corrcoef(a, b)[0, 1])
                ratio = float(np.linalg.norm(a) / max(np.linalg.norm(b), 1e-30))
                print(f"check {name} expert 0 row {row}: corr {corr:.4f}, norm ratio {ratio:.4f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
