#!/usr/bin/env python3
"""Census of batched decode steps from nsys reports (stage B0 of the batched-serving design; driver: step-census.sh).

Each SQLITE is `nsys export --type sqlite` of one `Lfm3Probe batch --sequences N --profile-range` process whose capture range
holds exactly its timed decode steps (--steps S x --reps R). Per report, per step: kernels (and those launched inside CUDA
graphs), GPU busy time and the idle gaps between GPU work (the host's share of a step the device waits for), kernel launch,
graph launch, memcpy and synchronize API calls, host-to-device and device-to-host copies with their bytes, and the busiest
kernels. N comes from the file name (n<N>...sqlite) or --sequences; the probe's own numbers (CENSUS line of an untraced run)
are joined from <stem>.plain.txt beside the report when present.

usage: step-census.py --steps STEPS [--sequences N] [--top K] [--json OUT] SQLITE...
"""
import argparse
import json
import os
import re
import sqlite3
import sys

LAUNCH = re.compile(r"^(cudaLaunchKernel|cudaLaunchKernelExC|cudaLaunchCooperativeKernel|cuLaunchKernel|cuLaunchKernelEx|cuLaunchCooperativeKernel)(_v\d+)?$")
GRAPH_LAUNCH = re.compile(r"^(cudaGraphLaunch|cuGraphLaunch)(_v\d+)?$")
MEMCPY = re.compile(r"^(cudaMemcpy|cuMemcpy)")
SYNC = re.compile(r"^(cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize|cuStreamSynchronize|cuCtxSynchronize|cuEventSynchronize)(_v\d+)?$")
GRAPH_BUILD = re.compile(r"^(cudaStreamBeginCapture|cudaStreamEndCapture|cudaGraphInstantiate|cudaGraphExecUpdate|cuStreamBeginCapture|cuStreamEndCapture|cuGraphInstantiate|cuGraphExecUpdate)")
COPY_KIND = {1: "h2d", 2: "d2h", 8: "d2d", 10: "p2p"}


def table_names(db):
    return {r[0] for r in db.execute("select name from sqlite_master where type='table'")}


def columns(db, table):
    return {r[1] for r in db.execute(f"pragma table_info({table})")}


def merged(intervals):
    """Union of [start, end) intervals: the merged list."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def census(path, steps, top):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    names = table_names(db)
    strings = dict(db.execute("select id, value from StringIds")) if "StringIds" in names else {}
    r = {"file": os.path.basename(path), "steps": steps}
    gpu = []
    kernels = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in names:
        kc = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
        graph_col = "graphNodeId" if "graphNodeId" in kc else None
        name_col = "shortName" if "shortName" in kc else "demangledName"
        query = f"select start, end, {name_col}, {graph_col or 'NULL'} from CUPTI_ACTIVITY_KIND_KERNEL"
        for s, e, name, node in db.execute(query):
            kernels.append((s, e, strings.get(name, str(name)), node))
            gpu.append((s, e))
        r["graph_node_column"] = graph_col is not None
    copies = {}
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in names:
        for s, e, kind, size in db.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"):
            k = COPY_KIND.get(kind, f"kind{kind}")
            c = copies.setdefault(k, [0, 0, 0.0])
            c[0] += 1
            c[1] += size or 0
            c[2] += (e - s) / 1e6
            gpu.append((s, e))
    memsets = 0
    if "CUPTI_ACTIVITY_KIND_MEMSET" in names:
        for s, e in db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET"):
            memsets += 1
            gpu.append((s, e))
    api = {}
    api_spans = []
    for t in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if t not in names:
            continue
        name_col = "nameId" if "nameId" in columns(db, t) else "name"
        for s, e, nid in db.execute(f"select start, end, {name_col} from {t}"):
            n = strings.get(nid, str(nid))
            a = api.setdefault(n, [0, 0.0])
            a[0] += 1
            a[1] += (e - s) / 1e6
            api_spans.append((s, e))
    if not gpu:
        r["error"] = "no GPU activity in the report (was the capture range marked? see the probe's INFO profiler lines)"
        return r
    busy = merged(gpu)
    first, last = busy[0][0], busy[-1][1]
    window_start = min([first] + [s for s, _ in api_spans])
    window_end = max([last] + [e for _, e in api_spans])
    busy_ns = sum(e - s for s, e in busy)
    gaps = [busy[i + 1][0] - busy[i][1] for i in range(len(busy) - 1)]
    per = lambda v: v / steps
    r["window_ms"] = (window_end - window_start) / 1e6
    r["gpu_span_ms"] = (last - first) / 1e6
    r["step_ms_traced"] = per((window_end - window_start) / 1e6)
    r["gpu_busy_ms"] = per(busy_ns / 1e6)
    r["gpu_busy_pct"] = 100.0 * busy_ns / max(1, last - first)
    r["gpu_idle_ms"] = per((last - first - busy_ns) / 1e6)
    r["idle_gaps_over_10us"] = per(sum(1 for g in gaps if g > 10_000))
    r["idle_ms_in_gaps_over_10us"] = per(sum(g for g in gaps if g > 10_000) / 1e6)
    r["kernels"] = per(len(kernels))
    r["kernel_ms"] = per(sum(e - s for s, e, _, _ in kernels) / 1e6)
    in_graph = sum(1 for k in kernels if k[3] not in (None, 0))
    r["kernels_in_graphs_pct"] = 100.0 * in_graph / max(1, len(kernels)) if r.get("graph_node_column") else None
    r["memsets"] = per(memsets)
    for k, (count, size, ms) in sorted(copies.items()):
        r[f"{k}_copies"] = per(count)
        r[f"{k}_kb"] = per(size / 1024.0)
        r[f"{k}_ms"] = per(ms)
    def calls(pattern):
        return per(sum(c for n, (c, _) in api.items() if pattern.search(n)))
    r["api_launches"] = calls(LAUNCH)
    r["api_graph_launches"] = calls(GRAPH_LAUNCH)
    r["api_memcpy"] = calls(MEMCPY)
    r["api_syncs"] = calls(SYNC)
    r["api_graph_builds"] = calls(GRAPH_BUILD)
    r["api_calls"] = per(sum(c for c, _ in api.values()))
    r["api_ms"] = per(sum(ms for _, ms in api.values()))
    by_kernel = {}
    for s, e, name, _ in kernels:
        b = by_kernel.setdefault(name, [0, 0.0])
        b[0] += 1
        b[1] += (e - s) / 1e6
    r["top_kernels"] = [
        {"name": n[:80], "per_step": per(c), "ms_per_step": per(ms)}
        for n, (c, ms) in sorted(by_kernel.items(), key=lambda kv: -kv[1][1])[:top]
    ]
    r["top_api"] = [
        {"name": n, "per_step": per(c), "ms_per_step": per(ms)}
        for n, (c, ms) in sorted(api.items(), key=lambda kv: -kv[1][1])[:top]
    ]
    return r


def plain_census(path):
    """The CENSUS line of the untraced run beside the report (<stem>.plain.txt), as a dict."""
    stem = re.sub(r"\.sqlite$", "", path)
    for candidate in (stem + ".plain.txt",):
        if os.path.exists(candidate):
            with open(candidate, errors="replace") as f:
                for line in f:
                    if line.startswith("CENSUS "):
                        return dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--steps", type=int, required=True, help="timed steps in each capture range (--steps x --reps)")
    ap.add_argument("--sequences", type=int, help="N when the file names do not say it")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = []
    for path in a.reports:
        m = re.search(r"n(\d+)", os.path.basename(path))
        n = a.sequences if a.sequences else int(m.group(1)) if m else 0
        r = census(path, a.steps, a.top)
        r["sequences"] = n
        r["probe"] = plain_census(path)
        rows.append(r)
    rows.sort(key=lambda r: r["sequences"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)

    def fmt(v, d=2):
        return "-" if v is None else f"{v:.{d}f}" if isinstance(v, float) else str(v)

    cols = [
        ("N", lambda r: r["sequences"]),
        ("step ms (plain)", lambda r: r["probe"].get("step_ms", "-")),
        ("host ms (plain)", lambda r: r["probe"].get("host_ms", "-")),
        ("step ms (traced)", lambda r: fmt(r.get("step_ms_traced"))),
        ("GPU busy ms", lambda r: fmt(r.get("gpu_busy_ms"))),
        ("GPU busy %", lambda r: fmt(r.get("gpu_busy_pct"), 1)),
        ("GPU idle ms", lambda r: fmt(r.get("gpu_idle_ms"))),
        ("gaps >10us", lambda r: fmt(r.get("idle_gaps_over_10us"), 1)),
        ("kernels", lambda r: fmt(r.get("kernels"), 1)),
        ("in graphs %", lambda r: fmt(r.get("kernels_in_graphs_pct"), 1)),
        ("launch API", lambda r: fmt(r.get("api_launches"), 1)),
        ("graph launches", lambda r: fmt(r.get("api_graph_launches"), 2)),
        ("graph builds", lambda r: fmt(r.get("api_graph_builds"), 2)),
        ("syncs", lambda r: fmt(r.get("api_syncs"), 1)),
        ("H2D n / KB", lambda r: f"{fmt(r.get('h2d_copies'), 1)} / {fmt(r.get('h2d_kb'), 1)}"),
        ("D2H n / KB", lambda r: f"{fmt(r.get('d2h_copies'), 1)} / {fmt(r.get('d2h_kb'), 1)}"),
        ("API ms", lambda r: fmt(r.get("api_ms"))),
    ]
    print(f"Per decode step ({a.steps} timed steps a report). 'plain' = the untraced probe run; traced times include nsys's overhead.\n")
    print("| " + " | ".join(c for c, _ in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        if "error" in r:
            print(f"| {r['sequences']} | {r['file']}: {r['error']} |")
            continue
        print("| " + " | ".join(str(f(r)) for _, f in cols) + " |")
    good = [r for r in rows if "error" not in r and r["sequences"] > 0]
    base = next((r for r in good if r["sequences"] == 4), None)
    if base and good[-1]["sequences"] > 4:
        t = good[-1]
        dn = t["sequences"] - 4
        print(f"\nGrowth N=4 -> {t['sequences']}, per added sequence: kernels {(t['kernels'] - base['kernels']) / dn:.1f}, "
              f"launch API {(t['api_launches'] - base['api_launches']) / dn:.1f}, syncs {(t['api_syncs'] - base['api_syncs']) / dn:.2f}, "
              f"H2D copies {(t.get('h2d_copies', 0) - base.get('h2d_copies', 0)) / dn:.2f}, GPU busy {(t['gpu_busy_ms'] - base['gpu_busy_ms']) / dn:.3f} ms, "
              f"GPU idle {(t['gpu_idle_ms'] - base['gpu_idle_ms']) / dn:.3f} ms")
    for r in good:
        print(f"\nN={r['sequences']} busiest kernels a step:")
        for k in r["top_kernels"]:
            print(f"  {k['per_step']:8.1f} x {k['ms_per_step']:8.3f} ms  {k['name']}")
        print(f"N={r['sequences']} busiest CUDA API calls a step:")
        for k in r["top_api"]:
            print(f"  {k['per_step']:8.1f} x {k['ms_per_step']:8.3f} ms  {k['name']}")


if __name__ == "__main__":
    sys.exit(main())
