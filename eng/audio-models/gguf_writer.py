# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""A small streaming GGUF v3 writer shared by the audio-model converters in this folder.

Tensors are numpy arrays (or callables that return one) written in their numpy layout: the GGUF dims are the numpy
shape reversed (ne0 = the last, fastest axis), which is what TensorSharp's GgufFile reports as Shape. Only the unquantized
types are written (F32, F16, BF16); the audio models are small enough that their converters keep the release precision.
"""

import os
import struct
import sys

import numpy as np

GGUF_MAGIC, GGUF_VERSION, ALIGN = 0x46554747, 3, 32
KV_U8, KV_I8, KV_U16, KV_I16, KV_U32, KV_I32, KV_F32, KV_BOOL, KV_STR, KV_ARR, KV_U64, KV_I64, KV_F64 = range(13)

F32, F16, BF16 = 0, 1, 30
TYPE_NAMES = {F32: "F32", F16: "F16", BF16: "BF16"}
TYPE_BYTES = {F32: 4, F16: 2, BF16: 2}


def to_bf16_bits(a):
    """float32 array -> uint16 BF16 bit patterns, rounded to nearest even (as torch does)."""
    bits = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> 16) & 1) + 0x7FFF
    return ((bits + rounding) >> 16).astype(np.uint16)


class GgufWriter:
    def __init__(self, architecture):
        self.kv = []
        self.tensors = []  # (name, dims ne0-first, type, nbytes, producer)
        self.names = set()
        self.str("general.architecture", architecture)

    # ---- metadata ----
    def u32(self, k, v): self.kv.append((k, KV_U32, int(v)))
    def i32(self, k, v): self.kv.append((k, KV_I32, int(v)))
    def f32(self, k, v): self.kv.append((k, KV_F32, float(v)))
    def bool(self, k, v): self.kv.append((k, KV_BOOL, bool(v)))
    def str(self, k, v): self.kv.append((k, KV_STR, str(v)))
    def strs(self, k, v): self.kv.append((k, KV_ARR, (KV_STR, [str(x) for x in v])))
    def i32s(self, k, v): self.kv.append((k, KV_ARR, (KV_I32, [int(x) for x in v])))
    def f32s(self, k, v): self.kv.append((k, KV_ARR, (KV_F32, [float(x) for x in v])))

    # ---- tensors ----
    def tensor(self, name, array_or_producer, gtype, shape=None):
        """Queue a tensor. shape (numpy order) is required when a producer is passed."""
        if name in self.names:
            sys.exit(f"error: duplicate tensor {name}")
        self.names.add(name)
        if callable(array_or_producer):
            if shape is None:
                sys.exit(f"error: {name}: a producer needs its shape")
            producer = array_or_producer
        else:
            arr = array_or_producer
            shape = arr.shape
            producer = lambda arr=arr: arr
        n = 1
        for d in shape:
            n *= int(d)
        dims = [int(d) for d in reversed(shape)] or [1]
        self.tensors.append((name, dims, gtype, n * TYPE_BYTES[gtype], producer))

    @staticmethod
    def _s(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _value(self, t, v):
        if t == KV_U32: return struct.pack("<I", v)
        if t == KV_I32: return struct.pack("<i", v)
        if t == KV_F32: return struct.pack("<f", v)
        if t == KV_BOOL: return struct.pack("<B", 1 if v else 0)
        if t == KV_STR: return self._s(v)
        et, items = v
        return struct.pack("<IQ", et, len(items)) + b"".join(self._value(et, x) for x in items)

    @staticmethod
    def _encode(name, payload, gtype):
        a = np.asarray(payload)
        if gtype == F32:
            a = a.astype(np.float32, copy=False)
        elif gtype == F16:
            a = a.astype(np.float16, copy=False)
        elif gtype == BF16:
            a = a if a.dtype == np.uint16 else to_bf16_bits(a.astype(np.float32, copy=False))
        else:
            sys.exit(f"error: {name}: unsupported type {gtype}")
        return np.ascontiguousarray(a).tobytes()

    def write(self, path, quiet=False):
        head = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(self.tensors), len(self.kv)))
        for k, t, v in self.kv:
            head += self._s(k) + struct.pack("<I", t) + self._value(t, v)
        offset, offsets = 0, []
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
                data = self._encode(name, producer(), gtype)
                if len(data) != nbytes:
                    sys.exit(f"error: {name}: produced {len(data)} bytes, expected {nbytes}")
                f.write(data)
                f.write(b"\0" * ((-nbytes) % ALIGN))
                if not quiet:
                    print(f"  {name:64s} {TYPE_NAMES[gtype]:4s} {dims}", flush=True)
        os.replace(tmp, path)
        print(f"wrote {path}: {len(self.tensors)} tensors, {os.path.getsize(path) / 2**20:.1f} MiB")


def read_safetensors(path):
    """name -> (dtype string, shape, memmap-backed loader). BF16 loads as float32."""
    import json
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    header.pop("__metadata__", None)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    np_types = {"F16": np.float16, "F32": np.float32, "BF16": np.uint16, "I64": np.int64, "I32": np.int32}
    out = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        dt = info["dtype"]
        shape = tuple(info["shape"])

        def load(start=start, end=end, dt=dt, shape=shape):
            raw = mm[base + start: base + end].view(np_types[dt]).reshape(shape)
            if dt == "BF16":
                return (raw.astype(np.uint32) << 16).view(np.float32)
            return np.array(raw)

        out[name] = (dt, shape, load)
    return out
