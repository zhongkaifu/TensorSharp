#!/usr/bin/env python3
"""Compare TS_Q4E_NODE_DUMP span graphs row by row.

nodediff.py <dir> pairs A:B[,rowsA-offset] ...
Each dump file is <call>_T<T>_p<pos>_r<rows>_kv<nkv>.bin. For a pair of calls,
rows are matched by absolute token position (pos + row). Nodes are aligned by
op sequence; for every aligned node the overlapping token rows are compared bit
for bit, and the report names the first node whose output differs while all of
its in-graph sources agreed (the first diverging op), plus a per-op summary.
"""
import difflib, glob, os, re, struct, sys
import numpy as np

GGML_H = "/workspace/ts-q4x/repo/ExternalProjects/ggml/include/ggml.h"
MAX_SRC = 10


def op_names():
    text = open(GGML_H).read()
    body = text[text.index("enum ggml_op {"):]
    body = body[:body.index("};")]
    names = re.findall(r"\b(GGML_OP_[A-Z0-9_]+)\b", body)
    return [n[len("GGML_OP_"):] for n in names]


OPS = op_names()
TYPES = {0: "f32", 1: "f16", 24: "i8", 25: "i16", 26: "i32", 27: "i64"}


def load(path):
    nodes = []
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    while off < len(data):
        magic, idx, op, typ = struct.unpack_from("<4i", data, off); off += 16
        assert magic == 0x444E3451, path
        ne = struct.unpack_from("<4q", data, off); off += 32
        src = struct.unpack_from("<%di" % MAX_SRC, data, off); off += 4 * MAX_SRC
        name = data[off:off + 64].split(b"\0")[0].decode(); off += 64
        (count,) = struct.unpack_from("<q", data, off); off += 8
        values = np.frombuffer(data, dtype=np.float32, count=count, offset=off) if count else None
        off += 4 * count
        nodes.append(dict(idx=idx, op=OPS[op] if op < len(OPS) else str(op), type=TYPES.get(typ, str(typ)),
                          ne=list(ne), src=[s for s in src], name=name, values=values))
    return nodes


def meta(path):
    m = re.search(r"(\d+)_T(\d+)_p(\d+)_r(\d+)_kv(\d+)\.bin$", path)
    return dict(call=int(m.group(1)), T=int(m.group(2)), pos=int(m.group(3)), rows=int(m.group(4)), kv=int(m.group(5)))


def find(directory, call):
    return glob.glob(os.path.join(directory, "%05d_*.bin" % call))[0]


def row_slices(a, b, ma, mb):
    """Return list of (abs_pos, slice_a, slice_b) over overlapping token rows, or
    None when the node has no identifiable token axis."""
    ne_a, ne_b = a["ne"], b["ne"]
    Ta, Tb = ma["T"], mb["T"]
    axis = None
    if Ta != Tb:
        cands = [d for d in range(4) if ne_a[d] == Ta and ne_b[d] == Tb]
        if len(cands) == 1:
            axis = cands[0]
    else:
        cands = [d for d in range(4) if ne_a[d] == Ta]
        if len(cands) == 1 and ne_a == ne_b:
            axis = cands[0]
    return axis


def compare(directory, ca, cb, verbose=False, limit=12):
    pa, pb = find(directory, ca), find(directory, cb)
    ma, mb = meta(pa), meta(pb)
    A, B = load(pa), load(pb)
    sm = difflib.SequenceMatcher(a=[n["op"] for n in A], b=[n["op"] for n in B], autojunk=False)
    pairs = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs[i1 + k] = j1 + k
    lo = max(ma["pos"], mb["pos"]); hi = min(ma["pos"] + ma["T"], mb["pos"] + mb["T"])
    print(f"== {os.path.basename(pa)} vs {os.path.basename(pb)}  aligned {len(pairs)}/{len(A)},{len(B)}  rows {lo}..{hi-1}")
    status = {}  # index in A -> True (equal on overlap) / False (differs) / None (unknown)
    firsts = []
    per_op = {}
    for i, a in enumerate(A):
        if i not in pairs:
            status[i] = None
            continue
        b = B[pairs[i]]
        if a["values"] is None or b["values"] is None:
            status[i] = None
            continue
        axis = row_slices(a, b, ma, mb)
        va = a["values"].reshape(a["ne"][::-1])  # [ne3, ne2, ne1, ne0]
        vb = b["values"].reshape(b["ne"][::-1])
        if axis is None:
            # no token axis: compare the common leading region only when shapes agree
            if a["ne"] != b["ne"]:
                status[i] = None
                continue
            sa, sb = va, vb
            if ma["pos"] + ma["T"] != mb["pos"] + mb["T"]:
                status[i] = None
                continue
        else:
            npaxis = 3 - axis
            ra = slice(lo - ma["pos"], hi - ma["pos"])
            rb = slice(lo - mb["pos"], hi - mb["pos"])
            sla = [slice(None)] * 4; slb = [slice(None)] * 4
            sla[npaxis] = ra; slb[npaxis] = rb
            for d in range(4):
                if d == axis:
                    continue
                if a["ne"][d] != b["ne"][d]:
                    m = min(a["ne"][d], b["ne"][d])
                    sla[3 - d] = slice(0, m); slb[3 - d] = slice(0, m)
            sa, sb = va[tuple(sla)], vb[tuple(slb)]
        if sa.shape != sb.shape:
            status[i] = None
            continue
        equal = np.array_equal(sa, sb) or (np.array_equal(np.isnan(sa), np.isnan(sb)) and np.array_equal(sa[~np.isnan(sa)], sb[~np.isnan(sb)]))
        status[i] = equal
        diff = float(np.nanmax(np.abs(sa.astype(np.float64) - sb.astype(np.float64)))) if sa.size and not equal else 0.0
        key = a["op"]
        s = per_op.setdefault(key, [0, 0, 0.0])
        s[0] += 1
        if not equal:
            s[1] += 1; s[2] = max(s[2], diff)
            srcs = [x for x in a["src"] if x >= 0]
            if all(status.get(x) is True for x in srcs):
                src_desc = ",".join(f"{A[x]['op']}{A[x]['ne'][:3]}:{A[x]['type']}" for x in srcs)
                leafs = [x for x in a["src"] if x == -1]
                firsts.append((i, a, b, diff, src_desc, len(leafs), axis))
    for i, a, b, diff, src_desc, nleaf, axis in firsts[:limit]:
        print(f"  FIRST node {i:4d} {a['op']:<16} {a['type']} A{a['ne'][:3]} B{b['ne'][:3]} axis={axis} max_abs={diff:.3g}  srcs[{src_desc}] leafs={nleaf}")
    if len(firsts) > limit:
        print(f"  ... {len(firsts) - limit} more first-divergence nodes")
    if verbose:
        for op, (n, bad, mx) in sorted(per_op.items()):
            print(f"    {op:<18} compared={n:4d} differ={bad:4d} max_abs={mx:.3g}")
    return firsts


if __name__ == "__main__":
    directory = sys.argv[1]
    verbose = "-v" in sys.argv
    for spec in [s for s in sys.argv[2:] if s != "-v"]:
        a, b = spec.split(":")
        compare(directory, int(a), int(b), verbose)
