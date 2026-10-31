#!/usr/bin/env python3
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
"""nsysan.py REP.sqlite CLI.log LAYERS [--top 15] [--json out.json]

Produce REP.sqlite with: nsys profile -t cuda,cublas --cuda-graph-trace=node -s none --cpuctxsw=none
--export sqlite -o REP dotnet TensorSharp.Cli.dll --model M --backend cuda --benchmark --bench-prefill N
--bench-decode 0 --bench-runs 3 > CLI.log (eng/validation/README-direct-cuda-prefill.md).

Summarises the LAST benchmark prefill of an nsys-profiled TensorSharp.Cli --benchmark run
(--bench-decode 0, so nothing but prefills follows the load). The window is
[t_end - prefillMs(last run), t_end], t_end = end of the last GPU activity (kernel/memcpy/memset).
Reports: top kernels (% of window wall time and of kernel time), GPU busy, memcpy/memset per kind,
API-call counts (launches, syncs, graph launches, allocs), cuBLAS-call attribution of kernels, and a
GEMM class split (F32-data TF32 tensor-op / F32 SIMT / F16 / BF16 / int8-quant custom).
"""
import json, re, sqlite3, sys
from collections import defaultdict

db, logp, layers = sys.argv[1], sys.argv[2], int(sys.argv[3])
top = 15
jout = None
a = sys.argv[4:]
if "--top" in a: top = int(a[a.index("--top") + 1])
if "--json" in a: jout = a[a.index("--json") + 1]

log = open(logp, errors="replace").read()
runs = [(int(r), float(ms), float(t)) for r, ms, t in re.findall(r"benchmark run (\d+)/\d+: prefillMs=([0-9.]+) prefillTps=([0-9.]+)", log)]
if not runs:
    print("no benchmark runs in log"); sys.exit(1)
pre_ms = runs[-1][1]

con = sqlite3.connect(db)
cur = con.cursor()
tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
S = dict(cur.execute("select id, value from StringIds"))

def cols(t):
    return [r[1] for r in cur.execute(f"pragma table_info({t})")]

ends = []
for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
    if t in tables:
        v = cur.execute(f"select max(end) from {t}").fetchone()[0]
        if v: ends.append(v)
t_end = max(ends)
t0 = t_end - int(pre_ms * 1e6)
wall = t_end - t0

# ---- kernels
kc = cols("CUPTI_ACTIVITY_KIND_KERNEL")
dn = "demangledName" if "demangledName" in kc else "shortName"
rows = cur.execute(f"select start, end, shortName, {dn}, correlationId, deviceId from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?", (t0, t_end)).fetchall()
def strip_params(n):
    if not n.endswith(")"): return n
    d = 0
    for i in range(len(n) - 1, -1, -1):
        if n[i] == ")": d += 1
        elif n[i] == "(":
            d -= 1
            if d == 0: return n[:i]
    return n
kern = defaultdict(lambda: [0, 0])
intervals = []
by_corr = {}
devs = defaultdict(int)
for s, e, sn, dname, corr, dev in rows:
    s2, e2 = max(s, t0), min(e, t_end)
    full = S.get(dname, S.get(sn, str(sn)))
    name = re.sub(r"^void ", "", full)
    name = strip_params(name)  # drop the parameter list, keep template args
    k = kern[(name, full)]
    k[0] += 1; k[1] += e2 - s2
    intervals.append((s2, e2))
    by_corr[corr] = (name, full, e2 - s2)
    devs[dev] += e2 - s2
ksum = sum(v[1] for v in kern.values())
nk = sum(v[0] for v in kern.values())

# memcpy / memset
mem = defaultdict(lambda: [0, 0, 0])
kinds = {}
if "ENUM_CUDA_MEMCPY_OPER" in tables:
    kinds = {r[0]: r[1] for r in cur.execute("select id, label from ENUM_CUDA_MEMCPY_OPER")}
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    for s, e, b, ck in cur.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where end > ? and start < ?", (t0, t_end)):
        m = mem[kinds.get(ck, str(ck))]
        m[0] += 1; m[1] += b; m[2] += min(e, t_end) - max(s, t0)
        intervals.append((max(s, t0), min(e, t_end)))
if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
    for s, e, b in cur.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMSET where end > ? and start < ?", (t0, t_end)):
        m = mem["memset"]
        m[0] += 1; m[1] += b; m[2] += min(e, t_end) - max(s, t0)
        intervals.append((max(s, t0), min(e, t_end)))
intervals.sort()
busy = 0; cs = ce = None
for s, e in intervals:
    if cs is None or s > ce:
        if cs is not None: busy += ce - cs
        cs, ce = s, e
    else:
        ce = max(ce, e)
if cs is not None: busy += ce - cs

# ---- API calls (runtime + driver)
api = defaultdict(lambda: [0, 0])
launch_tid = {}
if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
    for s, e, nid, corr, tid in cur.execute("select start, end, nameId, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and start < ?", (t0, t_end)):
        n = S.get(nid, str(nid))
        a_ = api[n]; a_[0] += 1; a_[1] += e - s
        if corr in by_corr: launch_tid[corr] = (tid, s)

# ---- cuBLAS attribution: kernel whose launch API call sits inside a cuBLAS call on the same thread
cublas_tab = next((t for t in tables if "CUBLAS" in t.upper() and "EVENT" in t.upper()), None)
cub = defaultdict(lambda: [0, 0, 0])  # calls, kernels, kernel ns
cub_kernel_ns = 0
calls = []
if cublas_tab:
    calls = [(s, e, S.get(nid, str(nid)), tid) for s, e, nid, tid in cur.execute(f"select start, end, nameId, globalTid from {cublas_tab} where start >= ? and start < ?", (t0, t_end))]
elif "NVTX_EVENTS" in tables:
    nc = cols("NVTX_EVENTS")
    tx = "coalesce(text, (select value from StringIds where id = textId))" if "textId" in nc else "text"
    calls = [(s, e, t, tid) for s, e, t, tid in cur.execute(f"select start, end, {tx}, globalTid from NVTX_EVENTS where start >= ? and start < ? and end is not null", (t0, t_end)) if t and t.lower().startswith("cublas")]
    if calls: cublas_tab = "NVTX_EVENTS(cublas*)"
if calls:
    per_tid = defaultdict(list)
    for s, e, n, tid in calls:
        n = re.sub(r"_v2$", "", n)
        cub[n][0] += 1
        per_tid[tid].append((s, e, n))
    for v in per_tid.values(): v.sort()
    import bisect
    starts = {t: [c[0] for c in v] for t, v in per_tid.items()}
    for corr, (tid, ls) in launch_tid.items():
        v = per_tid.get(tid)
        if not v: continue
        i = bisect.bisect_right(starts[tid], ls) - 1
        if i >= 0 and v[i][0] <= ls <= v[i][1]:
            n = v[i][2]
            kn = by_corr[corr]
            cub[n][1] += 1; cub[n][2] += kn[2]; cub_kernel_ns += kn[2]

# ---- GEMM classification by kernel name
def gclass(name, full):
    f = full.lower()
    if f.startswith("ts_") or "mul_mat" in f or "void ts_" in f: return None
    is_gemm = re.search(r"gemm|cutlass|xmma|gemv", f) is not None
    if not is_gemm:
        if re.search(r"splitkreduce|reduce_1block|cublas", f): return "cuBLAS other (splitK/reduce)"
        return None
    if "bf16" in f: return "cuBLAS BF16 tensor-op"
    if re.search(r"fp16|f16|h16816|h1688|hgemm|884", f): return "cuBLAS F16-MMA kernel"
    if re.search(r"tf32|s1688gemm|s16816gemm_tf32", f): return "cuBLAS TF32 tensor-op"
    if re.search(r"sgemm|gemv|gemmsn", f): return "cuBLAS F32 SIMT/gemv"
    return "cuBLAS other GEMM"
gc = defaultdict(lambda: [0, 0])
for (name, full), (c, t) in kern.items():
    g = gclass(name, full)
    if g: gc[g][0] += c; gc[g][1] += t

# data type fed to each cuBLAS kernel: a direct-cuda RunF16Gemm/BF16 call converts its activation right
# before the GEMM (ts_convert_f32_f16/bf16); ggml converts src1 with a cpy/convert kernel. Anything else = F32 data.
ordered = cur.execute(f"select start, end, streamId, {dn}, shortName from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ? order by streamId, start", (t0, t_end)).fetchall()
feed = defaultdict(lambda: [0, 0, defaultdict(int)])
f32names = defaultdict(int)
prev_stream = None; prev_cls = None; prev_name = ""
for st, en, sid, dname, sn in ordered:
    full = S.get(dname, S.get(sn, ""))
    nm = strip_params(re.sub(r"^void ", "", full))
    g = gclass(nm, full)
    if sid != prev_stream:
        prev_cls = None; prev_name = ""; prev_stream = sid
    if g:
        if prev_cls is not None:
            cls = prev_cls
        else:
            pl = prev_name.lower()
            if re.search(r"convert_f32_bf16|bf16", pl): cls = "BF16-data GEMM"
            elif re.search(r"convert_f32_f16|convert_unary|cpy.*__half|f32_f16|to_fp16", pl): cls = "F16-data GEMM"
            else: cls = "F32-data GEMM"
        dur = min(en, t_end) - max(st, t0)
        feed[cls][0] += 1; feed[cls][1] += dur; feed[cls][2][g] += dur
        if cls == "F32-data GEMM": f32names[nm[:90]] += dur
        prev_cls = cls
    else:
        prev_cls = None; prev_name = nm
def category(nm):
    f = nm.lower()
    if gclass(nm, nm): return "cuBLAS GEMM"
    if re.search(r"attention|flash_attn|fattn|softmax", f): return "attention"
    if re.search(r"quant_matmul|mul_mat|mmq|mmvq|matvec|mmf|_mmq", f): return "quant matmul (custom)"
    if re.search(r"dequant|convert|quantize", f): return "dequant/convert/quantize"
    if re.search(r"moe|_ids|index_select|gather|scatter|argsort|top_k|get_rows|add_id", f): return "MoE route/gather/scatter"
    if re.search(r"ssm|gdn|delta|conv1d|mamba|scan|recurr", f): return "SSM/GDN/conv"
    if re.search(r"cpy|copy|concat|slice|transpose|permute|cont", f): return "copy/layout"
    return "norm/rope/elementwise/other"
cats = defaultdict(lambda: [0, 0])
for (name, full), (c, t) in kern.items():
    k_ = category(name); cats[k_][0] += c; cats[k_][1] += t

def pct(x, d): return 100.0 * x / d if d else 0.0
ms = lambda ns: ns / 1e6

out = {"window_ms": ms(wall), "prefill_ms_log": pre_ms, "runs": runs, "kernel_ms": ms(ksum), "busy_ms": ms(busy),
       "kernels": nk, "layers": layers}
print(f"window (last prefill) {ms(wall):.1f} ms; runs prefillMs/tps: " + ", ".join(f"{r[1]:.0f}/{r[2]:.0f}" for r in runs))
print(f"GPU busy {ms(busy):.1f} ms = {pct(busy, wall):.1f}% of wall; kernel sum {ms(ksum):.1f} ms; kernels {nk} ({nk/layers:.0f}/layer)")
print(f"\nTop {top} kernels (by GPU time in the window):")
print(f"{'%wall':>6} {'%kern':>6} {'ms':>9} {'calls':>7} {'us/call':>8}  name")
tops = sorted(kern.items(), key=lambda kv: -kv[1][1])
out["top"] = []
for (name, full), (c, t) in tops[:top]:
    print(f"{pct(t, wall):6.1f} {pct(t, ksum):6.1f} {ms(t):9.2f} {c:7d} {t/c/1e3:8.1f}  {name[:110]}")
    out["top"].append({"name": name, "full": full[:300], "calls": c, "ms": ms(t), "pct_wall": pct(t, wall), "pct_kern": pct(t, ksum)})
print("\nmemcpy/memset in window:")
out["mem"] = {}
for k, (c, b, t) in sorted(mem.items(), key=lambda kv: -kv[1][0]):
    print(f"  {k:12s} {c:7d} ops ({c/layers:7.1f}/layer) {b/2**20:10.2f} MiB {ms(t):8.2f} ms")
    out["mem"][k] = {"ops": c, "mib": b / 2**20, "ms": ms(t)}
print("\nAPI calls in window (top 25 by count):")
out["api"] = {}
for n, (c, t) in sorted(api.items(), key=lambda kv: -kv[1][0])[:25]:
    print(f"  {n:40s} {c:8d} ({c/layers:8.1f}/layer) {ms(t):9.2f} ms")
    out["api"][n] = {"calls": c, "ms": ms(t)}
syncs = {n: v for n, v in api.items() if re.search(r"Synchronize|cuMemcpyDtoH_v2$|cuMemcpyHtoD_v2$|cudaMemcpy$|cuMemcpy_v2$|cuMemcpy2D_v2$|cuMemcpyDtoD_v2$|cudaMemcpy2D$|cuMemcpyHtoD_v2|cuMemcpyDtoH_v2|cuStreamQuery|cuEventQuery", n)}
print("\nblocking/sync-ish API calls:", ", ".join(f"{n}={v[0]} ({ms(v[1]):.1f} ms)" for n, v in syncs.items()) or "none")
out["syncs"] = {n: {"calls": v[0], "ms": ms(v[1])} for n, v in syncs.items()}
print("\nGEMM classes (kernel-name based):")
out["gemm"] = {}
for g, (c, t) in sorted(gc.items(), key=lambda kv: -kv[1][1]):
    print(f"  {g:36s} {c:6d} kernels {ms(t):9.2f} ms = {pct(t, wall):5.1f}% wall, {pct(t, ksum):5.1f}% kern")
    out["gemm"][g] = {"kernels": c, "ms": ms(t), "pct_wall": pct(t, wall), "pct_kern": pct(t, ksum)}
print("\ncuBLAS GEMM time by the data type fed to it (predecessor kernel heuristic):")
out["gemm_feed"] = {}
for c_, (n_, t_, kinds_) in sorted(feed.items(), key=lambda kv: -kv[1][1]):
    print(f"  {c_:16s} {n_:6d} kernels {ms(t_):9.2f} ms = {pct(t_, wall):5.1f}% wall  [" + "; ".join(f"{k}: {ms(v):.1f} ms" for k, v in kinds_.items()) + "]")
    out["gemm_feed"][c_] = {"kernels": n_, "ms": ms(t_), "pct_wall": pct(t_, wall), "by_kernel_class": {k: ms(v) for k, v in kinds_.items()}}
print("  F32-data GEMM kernels: " + "; ".join(f"{k} {ms(v):.1f} ms" for k, v in sorted(f32names.items(), key=lambda kv: -kv[1])[:6]))
out["f32_gemm_kernels"] = {k: ms(v) for k, v in f32names.items()}
print("\nkernel time by category:")
out["cats"] = {}
for c_, (n_, t_) in sorted(cats.items(), key=lambda kv: -kv[1][1]):
    print(f"  {c_:28s} {n_:7d} kernels {ms(t_):9.2f} ms = {pct(t_, wall):5.1f}% wall, {pct(t_, ksum):5.1f}% kern")
    out["cats"][c_] = {"kernels": n_, "ms": ms(t_), "pct_wall": pct(t_, wall), "pct_kern": pct(t_, ksum)}
if cublas_tab:
    print(f"\ncuBLAS API calls in window ({cublas_tab}); kernels attributed by launch-in-call:")
    out["cublas"] = {}
    for n, (c, nkern, t) in sorted(cub.items(), key=lambda kv: -kv[1][2]):
        print(f"  {n:40s} {c:6d} calls {nkern:6d} kernels {ms(t):9.2f} ms = {pct(t, wall):5.1f}% wall")
        out["cublas"][n] = {"calls": c, "kernels": nkern, "ms": ms(t), "pct_wall": pct(t, wall)}
    print(f"  total cuBLAS-launched kernel time {ms(cub_kernel_ns):.2f} ms = {pct(cub_kernel_ns, wall):.1f}% wall")
    out["cublas_total_ms"] = ms(cub_kernel_ns)
else:
    print("\n(no cuBLAS event table)")
print("\nper-device kernel ms:", {d: round(ms(t), 1) for d, t in devs.items()})
if jout:
    json.dump(out, open(jout, "w"), indent=1)
