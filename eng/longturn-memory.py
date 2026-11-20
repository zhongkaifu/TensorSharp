#!/usr/bin/env python3
"""Device and host memory of an eng/longturn.py run (issue #383): a per-second sampler, and per-request reports.

  python3 eng/longturn-memory.py sample PID OUT/mem.log [interval_s]   # while the server runs (stops when PID exits)
  python3 eng/longturn-memory.py report OUT [OUT...]                   # per request: peak device MiB, host memory
  python3 eng/longturn-memory.py holdings OUT [OUT...]                 # TS_CUDA_LOG_VRAM=1 [vram-hold] breakdown at each C
  python3 eng/longturn-memory.py nsys OUT [OUT...]                     # cuMemAlloc / cuMemFree calls per request (nsys)

sample writes one line a sample: `HH:MM:SS.mmm dev=<MiB used on GPU 0, device-wide> proc=<MiB the process holds there>
rss=<MiB> avail=<MiB host MemAvailable> cg=<MiB cgroup memory.current>`. GPU figures come from NVML through ctypes
(libnvidia-ml.so.1 or nvml.dll), with no nvidia-smi process per sample (one takes the driver lock every call).
report and holdings window by client.jsonl's UTC send / receive times, exactly (a margin catches the next request's slot).
holdings reads the [vram-hold] line a C request's retention logged: the slots the prefix cache keeps, checkpoint states,
pooled spares, the primary, spare states, drafter rings, what the model holds, recorded graphs, the allocator's live and
cached blocks, and the device's use. nsys reads OUT/nsys.nsys-rep (a server run under `nsys profile -t cuda`), exporting
OUT/nsys.sqlite when it is missing. Standard library only (nsys needs the nsys CLI).
"""
from __future__ import annotations

import ctypes
import datetime
import json
import os
import re
import sqlite3
import subprocess
import sys
import time


def _secs(hms: str) -> float:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _iso_ns(s: str) -> int:
    t = datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=datetime.timezone.utc)
    return int(t.timestamp() * 1e9)


def _requests(d: str) -> list[dict]:
    with open(os.path.join(d, "client.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- sample

class _Mem(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Proc(ctypes.Structure):  # nvmlProcessInfo_t (v2/v3)
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong), ("gpuInstanceId", ctypes.c_uint),
                ("computeInstanceId", ctypes.c_uint)]


def _children(pid: int) -> set[int]:
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            return {int(x) for x in f.read().split()}
    except OSError:
        return set()


def _meminfo_mib(key: str) -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) >> 10
    except OSError:
        pass
    return -1


def _rss_mib(pid: int) -> int:
    best = 0
    for q in {pid} | _children(pid):
        try:
            with open(f"/proc/{q}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        best = max(best, int(line.split()[1]) >> 10)
        except OSError:
            pass
    return best


def _cgroup_mib() -> int:
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            return int(f.read()) >> 20
    except OSError:
        return -1


def sample(pid: int, out: str, interval: float) -> int:
    nvml = ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
    if nvml.nvmlInit_v2() != 0:
        raise SystemExit("longturn-memory: nvmlInit failed")
    handle = ctypes.c_void_p()
    if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
        raise SystemExit("longturn-memory: NVML has no device 0")
    procs = getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v3", None) or nvml.nvmlDeviceGetComputeRunningProcesses_v2

    def alive() -> bool:
        if os.name == "nt":
            return True
        return os.path.exists(f"/proc/{pid}")

    with open(out, "a", buffering=1) as f:
        while alive():
            m = _Mem()
            nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(m))
            n = ctypes.c_uint(64)
            arr = (_Proc * 64)()
            mine = {pid} | _children(pid)
            proc = sum(arr[i].usedGpuMemory for i in range(n.value) if arr[i].pid in mine) >> 20 \
                if procs(handle, ctypes.byref(n), arr) == 0 else -1
            t = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
            f.write(f"{t} dev={m.used >> 20} proc={proc} rss={_rss_mib(pid)} avail={_meminfo_mib('MemAvailable')} "
                    f"cg={_cgroup_mib()}\n")
            time.sleep(interval)
    return 0


# ---------------------------------------------------------------- report

def _samples(d: str) -> list[tuple[float, dict]]:
    out = []
    with open(os.path.join(d, "mem.log")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                out.append((_secs(p[0]), {k: int(v) for k, v in (x.split("=") for x in p[1:])}))
    return out


def report(dirs: list[str]) -> int:
    for d in dirs:
        mem, reqs = _samples(d), _requests(d)
        print(f"## {d}\n")
        if not mem:
            print("no samples\n")
            continue
        first = _secs(reqs[0]["sent"][11:23]) if reqs else mem[-1][0]
        before = [m for t, m in mem if t < first]
        print(f"loaded: device {before[-1]['dev'] if before else '-'} MiB before the first request; run peak "
              f"{max(m['dev'] for _, m in mem)} MiB; min host MemAvailable {min(m['avail'] for _, m in mem)} MiB\n")
        print("| size | step | prompt | wall s | status | peak device MiB | peak process MiB | min host avail MiB | peak RSS MiB |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in reqs:
            a, b = _secs(r["sent"][11:23]), _secs(r["done"][11:23])
            w = [m for t, m in mem if a <= t <= b]
            if w:
                print(f"| {r['size']} | {r['step']} | {r['prompt_tokens']} | {r['wall_s']:.2f} | {r['status']} | "
                      f"{max(m['dev'] for m in w)} | {max(m['proc'] for m in w)} | {min(m['avail'] for m in w)} | "
                      f"{max(m['rss'] for m in w)} |")
        print()
    return 0


# ---------------------------------------------------------------- holdings

def _group(text: str, name: str):
    m = re.search(name + r" (\d+) \[([^\]]*)\]( \+\d+ ckpt states (\d+))? = (\d+) MiB", text)
    return None if not m else {"items": m.group(2), "ckpt": int(m.group(4) or 0), "mib": int(m.group(5))}


def _num(pat: str, text: str, default="-"):
    m = re.search(pat, text)
    return int(m.group(1)) if m else default


def holdings(dirs: list[str]) -> int:
    for d in dirs:
        holds = []
        with open(os.path.join(d, "server.log"), errors="replace") as f:
            for line in f:
                if "[vram-hold]" in line:
                    holds.append((_secs(line[:12]), line))
        print(f"## {d}\n")
        print("| size | retained slots (cells/rows MiB) | ckpt states | pooled | primary | spare ckpt | rings | model holds | "
              "recordings | pool live | cached large | cached small | device used |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in _requests(d):
            if r["step"] != "C":
                continue
            a, b = _secs(r["sent"][11:23]), _secs(r["done"][11:23])
            w = [line for t, line in holds if a - 0.2 <= t <= b + 0.3]
            kept = [line for line in w if re.search(r"\] \S+ \S+ retained", line)]
            if not (kept or w):
                continue
            line = (kept or w)[-1]
            ret, pool, prim = _group(line, "retained"), _group(line, "pooled"), _group(line, "primary")
            used = _num(r"NVML used=(\d+) MiB", line, None)
            if used is None:
                free, total = _num(r"free=(\d+) MiB of", line, None), _num(r"free=\d+ MiB of (\d+)", line, None)
                used = total - free if free is not None and total is not None else "-"
            cols = [r["size"], ret["items"] if ret else "-", ret["ckpt"] if ret else "-", pool["mib"] if pool else "-",
                    prim["mib"] if prim else "-"]
            for pat in (r"spare ckpt states (\d+)", r"rings \d+ = (\d+)", r"model holds (\d+)", r"recordings own (\d+)",
                        r"pool live (\d+)", r"cached large (\d+)", r"small (\d+) MiB"):
                cols.append(_num(pat, line))
            cols.append(used)
            print("| " + " | ".join(str(c) for c in cols) + " |")
        print()
    return 0


# ---------------------------------------------------------------- nsys

def nsys(dirs: list[str]) -> int:
    for d in dirs:
        db_path = os.path.join(d, "nsys.sqlite")
        if not os.path.exists(db_path):
            subprocess.run(["nsys", "export", "--type", "sqlite", "-o", db_path, os.path.join(d, "nsys.nsys-rep")],
                           check=True, capture_output=True)
        cur = sqlite3.connect(db_path).cursor()
        start = cur.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        names = dict(cur.execute("select id, value from StringIds"))
        tables = [r[0] for r in cur.execute("select name from sqlite_master where type='table'")
                  if r[0].startswith(("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"))]
        calls = []
        for t in tables:
            for s, e, n in cur.execute(f"select start, end, nameId from {t}"):
                name = names.get(n, "")
                if name.startswith(("cuMemFree", "cuMemAlloc")) or name in ("cudaFree", "cudaMalloc"):
                    calls.append((s + start, e - s, "free" if "free" in name.lower() else "alloc"))
        print(f"## {d}\n\n| size | step | prompt | wall s | cuMemAlloc | ms | cuMemFree | ms |\n|---|---|---|---|---|---|---|---|")
        for r in _requests(d):
            a, b = _iso_ns(r["sent"]), _iso_ns(r["done"])
            al = [c for c in calls if a <= c[0] <= b and c[2] == "alloc"]
            fr = [c for c in calls if a <= c[0] <= b and c[2] == "free"]
            print(f"| {r['size']} | {r['step']} | {r['prompt_tokens']} | {r['wall_s']:.2f} | {len(al)} | "
                  f"{sum(c[1] for c in al) / 1e6:.1f} | {len(fr)} | {sum(c[1] for c in fr) / 1e6:.1f} |")
        print()
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "sample":
        return sample(int(argv[1]), argv[2], float(argv[3]) if len(argv) > 3 else 1.0)
    if len(argv) >= 2 and argv[0] in ("report", "holdings", "nsys"):
        return {"report": report, "holdings": holdings, "nsys": nsys}[argv[0]](argv[1:])
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
