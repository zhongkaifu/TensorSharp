"""Print a GGUF file's metadata and tensor table, standard library only.

    python -I eng/inkling2/gguf_dump.py model.gguf [--tensors PATTERN]
"""

from __future__ import annotations

import argparse
import re
import struct
import sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
         11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
         19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
         27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4"}


class Reader:
    def __init__(self, handle):
        self.f = handle

    def read(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, self.f.read(size))

    def string(self):
        (n,) = self.read("Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, kind):
        scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind in scalars:
            return self.read(scalars[kind])[0]
        if kind == 8:
            return self.string()
        if kind == 9:
            (inner,) = self.read("I")
            (n,) = self.read("Q")
            return [self.value(inner) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {kind}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--tensors", help="regex filter for the tensor table; default prints layer 0, 6 and globals")
    args = parser.parse_args()
    with open(args.path, "rb") as handle:
        r = Reader(handle)
        magic, version = handle.read(4), r.read("I")[0]
        (n_tensors,) = r.read("Q")
        (n_kv,) = r.read("Q")
        print(f"{magic!r} v{version}: {n_tensors} tensors, {n_kv} keys")
        for _ in range(n_kv):
            key = r.string()
            (kind,) = r.read("I")
            value = r.value(kind)
            if isinstance(value, list) and len(value) > 24:
                value = f"[{len(value)} items: {value[:6]} ...]"
            print(f"  {key} = {value}")
        pattern = re.compile(args.tensors) if args.tensors else re.compile(r"^(?!blk\.)|^blk\.(0|5|6|20)\.")
        counts = {}
        for _ in range(n_tensors):
            name = r.string()
            (n_dims,) = r.read("I")
            dims = r.read("Q" * n_dims)
            (kind,) = r.read("I")
            r.read("Q")
            counts[TYPES.get(kind, kind)] = counts.get(TYPES.get(kind, kind), 0) + 1
            if pattern.search(name):
                print(f"  {name} {TYPES.get(kind, kind)} {list(dims)}")
        print("types:", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
