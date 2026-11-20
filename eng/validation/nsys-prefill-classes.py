#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""Where a prefill's time goes, by kernel class, in an nsys capture (issue #383: fresh prefill against the direct engine).

  python3 eng/validation/nsys-prefill-classes.py CAPTURE.sqlite LABEL=HH:MM:SS.mmm-HH:MM:SS.mmm ... [--json OUT.json]
                                                 [--server-log server.log]

CAPTURE.sqlite: `nsys export --type sqlite` of a capture taken with `-t cuda,osrt --cuda-graph-trace=node` (graph launches
traced per node, so recorded steps show their kernels). Each window is UTC wall-clock time, as the stamps of a stamped
server log (eng/longturn.py stamp): typically a request's first prompt step to its last.

Per window it prints the wall time, GPU busy time (union of kernels, memcpys and memsets) and the idle rest (host gaps:
what the GPU waited for), then kernel time and launches by class - fp4 products (NVFP4 / FP4 GEMMs and matvecs), other
products, prompt attention, recurrent (GDN / SSM / conv), norms, quantize, elementwise and copies, other - with each class's
heaviest kernels, the memcpy bytes by kind, and the host's CUDA API calls (launches, graph launches, syncs, allocations and
frees) with their host time. With --server-log, the window's scheduler steps (TS_SCHED_STEP_LOG=1 lines, by their t= end
stamps) give the step count, rows per step and the steps' summed gaps. eng/validation/prefill-windows.py prints the windows
of every fresh prompt in such a log. Classes are name patterns (CLASSES below); `--unclassified` lists what fell into
"other". Standard library only. Keep captures and outputs under the ignored artifacts/ or docs/validation/.
"""
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone

CLASSES = [
    # Names of both TensorSharp's CUDA executor (ts_fp4_*, ts_flash_*, ts_ssm_*, ts_graph_*) and the removed direct engine
    # of 038fb303 (ts_q35_*), so a capture of either sorts the same work into the same class.
    ("fp4 GEMMs (W4A4)", re.compile(r"w4a4|fp4_gemm|gemm_fp4|nvfp4_gemm|mxfp4_gemm|blockscaled|block_scaled", re.I)),
    ("nvfp4 activations (quantize, add+norm)", re.compile(r"quantize_nvfp4|rmsnorm_nvfp4|nvfp4_f32|to_nvfp4", re.I)),
    ("prompt attention", re.compile(r"flash|attn|attention|sdpa|softmax", re.I)),
    ("recurrent (gdn/ssm/conv)", re.compile(r"gdn|delta|ssm|conv1d|conv_|recur|scan|gated", re.I)),
    ("other products", re.compile(r"gemm|matmul|mma|cutlass|cublas|xmma|matvec|dp4a|mul_mat|mmq|mmvq|gemv|fp4|nvfp4", re.I)),
    ("norms", re.compile(r"norm", re.I)),
    ("quantize", re.compile(r"quantiz|q8_1|quant", re.I)),
    ("elementwise/copies", re.compile(r"silu|swiglu|glu|mul|add|rope|copy|cpy|convert|cast|fill|get_rows|gather|scatter|split|"
                                      r"concat|transpose|permute|scale|unary|binary|ternary|scalar|sigmoid|gelu|tanh|exp|"
                                      r"set_rows|slice|expand|repeat|clamp|memset|memcpy|argmax|topk|sum|reduce|embed|rows", re.I)),
]
OTHER = "other"


def classify(name):
    for label, pattern in CLASSES:
        if pattern.search(name):
            return label
    return OTHER


def union(intervals):
    total, end = 0, None
    for s, e in sorted(intervals):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def main(argv):
    args = list(argv)
    jout = None
    server_log = None
    show_unclassified = False
    if "--json" in args:
        i = args.index("--json"); jout = args[i + 1]; del args[i:i + 2]
    if "--server-log" in args:
        i = args.index("--server-log"); server_log = args[i + 1]; del args[i:i + 2]
    if "--unclassified" in args:
        args.remove("--unclassified"); show_unclassified = True
    if len(args) < 2:
        print(__doc__)
        return 2
    db = sqlite3.connect(args[0])
    cur = db.cursor()
    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    start_utc_ns = cur.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    day = datetime.fromtimestamp(start_utc_ns / 1e9, tz=timezone.utc).date()
    strings = dict(cur.execute("select id, value from StringIds"))

    def to_ns(hms):
        t = datetime.strptime(f"{day} {hms}", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        ns = int(t.timestamp() * 1e9) - start_utc_ns
        return ns + int(86400e9) if ns < -12 * 3600e9 else ns

    steps = []
    if server_log:
        for line in open(server_log, errors="replace"):
            m = re.match(r"\S+ \[step\] t=(\d\d:\d\d:\d\d\.\d{3})\S* gap=([\d.]+) ms=([\d.]+) (\S+) (.*)", line)
            if m:
                rows = sum(int(x) for x in re.findall(r":[PD](\d+)", m.group(5)))
                steps.append((m.group(1), float(m.group(2)), float(m.group(3)), m.group(4), rows, m.group(5).strip()))

    results = []
    for spec in args[1:]:
        label, rng = spec.split("=", 1)
        a, b = rng.split("-", 1)
        t0, t1 = to_ns(a), to_ns(b)
        wall = t1 - t0
        busy = []
        kernels = []
        if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
            for s, e, n in cur.execute(
                    "select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?", (t0, t1)):
                s, e = max(s, t0), min(e, t1)
                busy.append((s, e))
                kernels.append((strings.get(n, str(n)), e - s))
        copies = {}
        for table, kinded in (("CUPTI_ACTIVITY_KIND_MEMCPY", True), ("CUPTI_ACTIVITY_KIND_MEMSET", False)):
            if table not in tables:
                continue
            q = f"select start, end, bytes{', copyKind' if kinded else ''} from {table} where end > ? and start < ?"
            for row in cur.execute(q, (t0, t1)):
                s, e = max(row[0], t0), min(row[1], t1)
                busy.append((s, e))
                kind = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}.get(row[3], str(row[3])) if kinded else "memset"
                c = copies.setdefault(kind, [0, 0, 0])
                c[0] += 1; c[1] += row[2]; c[2] += e - s
        gpu_busy = union(busy)
        by_class = {}
        for name, d in kernels:
            c = by_class.setdefault(classify(name), {"ns": 0, "launches": 0, "kernels": {}})
            c["ns"] += d; c["launches"] += 1
            k = c["kernels"].setdefault(name, [0, 0]); k[0] += 1; k[1] += d
        api = {}
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table not in tables:
                continue
            for s, e, n in cur.execute(f"select start, end, nameId from {table} where end > ? and start < ?", (t0, t1)):
                name = strings.get(n, str(n))
                key = ("launch" if "LaunchKernel" in name or "cuLaunch" in name and "Graph" not in name else
                       "graph launch" if "GraphLaunch" in name else
                       "sync" if "Synchronize" in name else
                       "alloc" if re.search(r"Alloc|MemHostRegister", name) else
                       "free" if re.search(r"Free|MemHostUnregister", name) else
                       "memcpy/memset" if re.search(r"Memcpy|Memset", name) else
                       "graph build" if re.search(r"Graph(Instantiate|Exec|AddNode|Create|KernelNode)", name) else "other")
                v = api.setdefault(key, [0, 0]); v[0] += 1; v[1] += min(e, t1) - max(s, t0)
        in_window = [st for st in steps if a <= st[0] <= b]
        r = {
            "label": label, "window": [a, b], "wall_ms": wall / 1e6, "gpu_busy_ms": gpu_busy / 1e6,
            "gpu_idle_ms": (wall - gpu_busy) / 1e6, "kernel_ms": sum(d for _, d in kernels) / 1e6, "launches": len(kernels),
            "classes": {k: {"ms": v["ns"] / 1e6, "launches": v["launches"],
                            "top": sorted(((n, c, t / 1e6) for n, (c, t) in v["kernels"].items()), key=lambda x: -x[2])[:6]}
                        for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]["ns"])},
            "copies": {k: {"count": v[0], "mb": v[1] / 2**20, "ms": v[2] / 1e6} for k, v in copies.items()},
            "api": {k: {"calls": v[0], "host_ms": v[1] / 1e6} for k, v in sorted(api.items(), key=lambda kv: -kv[1][1])},
            "steps": {"count": len(in_window), "rows": [st[4] for st in in_window],
                      "step_ms_sum": sum(st[2] for st in in_window), "gap_ms_sum": sum(st[1] for st in in_window[1:])},
        }
        results.append(r)
        print(f"\n=== {label}: {a} - {b}: wall {r['wall_ms']:.0f} ms, GPU busy {r['gpu_busy_ms']:.0f} ms, idle "
              f"{r['gpu_idle_ms']:.0f} ms; {r['launches']} kernels, {r['kernel_ms']:.0f} ms summed")
        if in_window:
            print(f"  steps: {len(in_window)}, rows {r['steps']['rows']}, step ms summed {r['steps']['step_ms_sum']:.0f}, "
                  f"gaps between them {r['steps']['gap_ms_sum']:.1f} ms")
        for k, v in r["classes"].items():
            print(f"  {v['ms']:9.1f} ms {v['launches']:7d}x  {k}")
            for n, c, t in v["top"][:4 if k != OTHER or show_unclassified else 2]:
                print(f"      {t:9.1f} ms {c:7d}x  {n[:100]}")
        for k, v in r["copies"].items():
            print(f"  copies {k}: {v['count']}x {v['mb']:.1f} MiB {v['ms']:.1f} ms")
        print("  host API: " + ", ".join(f"{k} {v['calls']}x {v['host_ms']:.1f} ms" for k, v in r["api"].items()))
    if jout:
        with open(jout, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
