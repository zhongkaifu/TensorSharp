"""Read-only local hardware inventory and conservative regression resource planning.

Inventory is preflight evidence, not proof that a backend executed a test. No
native library is loaded and no GPU context is created by this module.
"""
from __future__ import annotations

import csv
import ctypes
import io
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

GIB = 1024 ** 3
BACKENDS = ("cpu", "ggml_cpu", "ggml_cuda", "cuda", "ggml_metal", "mlx", "ggml_vulkan")


def _probe(command: list[str], timeout: float = 8) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout,
                                   shell=False)
        return {"command": command, "exit_code": completed.returncode,
                "stdout": completed.stdout[:16000], "stderr": completed.stderr[:4000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "exit_code": None,
                "stdout": "", "stderr": type(exc).__name__}


def _tool(name: str, arguments: list[str]) -> dict[str, Any]:
    executable = shutil.which(name)
    if not executable:
        return {"path": None, "available": False, "reason": "not on PATH"}
    result = _probe([executable, *arguments])
    return {"path": executable, "available": result["exit_code"] == 0, **result}


def _memory() -> dict[str, Any]:
    result: dict[str, Any] = {"total_bytes": None, "available_bytes": None, "source": "unknown"}
    try:
        if platform.system() == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                            ("page_total", ctypes.c_ulonglong), ("page_available", ctypes.c_ulonglong),
                            ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                            ("extended_available", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                result.update(total_bytes=status.total, available_bytes=status.available,
                              source="GlobalMemoryStatusEx")
        elif platform.system() == "Linux":
            fields = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                fields[key] = int(value.strip().split()[0]) * 1024
            result.update(total_bytes=fields.get("MemTotal"), available_bytes=fields.get("MemAvailable"),
                          source="/proc/meminfo")
            # A container's host memory is not its allocation budget.
            for limit_path, usage_path in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                           ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
                try:
                    limit = int(Path(limit_path).read_text().strip())
                    used = int(Path(usage_path).read_text().strip())
                    if 0 < limit < (result["total_bytes"] or limit + 1):
                        result["total_bytes"] = limit
                        result["available_bytes"] = min(result["available_bytes"] or limit, max(0, limit - used))
                        result["source"] += " + cgroup limit"
                except (OSError, ValueError):
                    pass
        elif platform.system() == "Darwin":
            total = _probe(["sysctl", "-n", "hw.memsize"])
            pages = _probe(["vm_stat"])
            page_size = re.search(r"page size of (\d+) bytes", pages["stdout"])
            counts = dict(re.findall(r"^(Pages (?:free|inactive|speculative)):\s+(\d+)\.", pages["stdout"], re.M))
            if total["exit_code"] == 0:
                result["total_bytes"] = int(total["stdout"].strip())
            if page_size and counts:
                result["available_bytes"] = sum(int(v) for v in counts.values()) * int(page_size.group(1))
            result["source"] = "sysctl/vm_stat (free + inactive + speculative estimate)"
    except (OSError, ValueError, AttributeError):
        pass
    return result


def _disk(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    while not root.exists() and root != root.parent:
        root = root.parent
    if root.is_file():
        root = root.parent
    try:
        usage = shutil.disk_usage(root)
        return {"path": str(root), "total_bytes": usage.total, "free_bytes": usage.free}
    except OSError as exc:
        return {"path": str(root), "total_bytes": None, "free_bytes": None, "reason": type(exc).__name__}


def parse_nvidia_csv(output: str) -> list[dict[str, Any]]:
    """Accept both the extended query and old drivers lacking compute_cap."""
    devices = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if len(row) < 5:
            continue
        try:
            devices.append({"index": int(row[0].strip()), "name": row[1].strip(),
                            "memory_total_bytes": int(float(row[2].strip())) * 1024 ** 2,
                            "memory_free_bytes": int(float(row[3].strip())) * 1024 ** 2,
                            "driver_version": row[4].strip(),
                            "compute_capability": float(row[5].strip()) if len(row) > 5 and re.fullmatch(r"\d+(?:\.\d+)?", row[5].strip()) else None,
                            "source": "nvidia-smi"})
        except ValueError:
            continue
    return devices


def _runtime_evidence(runtime_dirs: list[str | Path]) -> dict[str, Any]:
    files: dict[str, list[str]] = {}
    flags: dict[str, bool] = {}
    cache_paths: list[str] = []
    for raw in runtime_dirs:
        directory = Path(raw).expanduser().resolve()
        if directory.is_file():
            directory = directory.parent
        if not directory.is_dir():
            continue
        candidates = list(directory.glob("*"))
        # Only inspect conventional native asset subdirectories, never model trees.
        candidates += list(directory.glob("runtimes/*/native/*"))
        for candidate in candidates:
            if candidate.is_file() and (candidate.suffix.lower() in (".dll", ".so", ".dylib", ".ptx", ".fatbin") or ".so." in candidate.name):
                files.setdefault(candidate.name.lower(), []).append(str(candidate))
        for cache in (directory / "CMakeCache.txt", directory.parent / "CMakeCache.txt"):
            if not cache.is_file():
                continue
            try:
                # Cache metadata is only build evidence; execution attestation is mandatory.
                contents = cache.read_text(encoding="utf-8", errors="replace")
                cache_paths.append(str(cache))
                for name, value in re.findall(r"^(TENSORSHARP_GGML_NATIVE_ENABLE_\w+):BOOL=(ON|OFF|TRUE|FALSE|1|0)$", contents, re.M):
                    flags[name] = value in ("ON", "TRUE", "1")
            except OSError:
                pass
    return {"files": files, "build_flags": flags, "build_metadata": sorted(set(cache_paths))}


def _backend_inventory(system: str, architecture: str, tools: dict[str, Any],
                       gpus: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    files = runtime["files"]
    flags = runtime["build_flags"]
    ggml = any(name in files for name in ("ggmlops.dll", "libggmlops.so", "libggmlops.dylib"))
    mlx = "libmlxc.dylib" in files
    direct = "tensorsharp.backends.cuda.dll" in files
    cublas = any("cublas" in name for name in files)
    cuda_build = flags.get("TENSORSHARP_GGML_NATIVE_ENABLE_CUDA", False) or any("ggml-cuda" in name for name in files)
    metal_build = flags.get("TENSORSHARP_GGML_NATIVE_ENABLE_METAL", False)
    # No remote access or GPU context creation: mere vulkaninfo installation is not device evidence.
    entries = {}
    def entry(name: str, available: bool, reason: str, evidence: list[str]) -> None:
        entries[name] = {"available": bool(available), "reason": reason,
                         "evidence": evidence, "execution_verified": False,
                         "requires_runtime_attestation": True}
    entry("cpu", True, "managed CPU platform candidate; executable and model support checked separately", [system, architecture])
    entry("ggml_cpu", ggml, "native wrapper present" if ggml else "GgmlOps native library absent from runtime directories", list(files))
    cuda_platform = system in ("Windows", "Linux") and bool(gpus)
    entry("ggml_cuda", cuda_platform and ggml and cuda_build,
          "CUDA device, wrapper and CUDA build evidence present" if cuda_platform and ggml and cuda_build else
          "requires local NVIDIA device, GgmlOps and CUDA build flag or ggml-cuda library; wrapper alone does not prove CUDA support",
          runtime["build_metadata"] + [gpu["name"] for gpu in gpus])
    supported_gpu = any((gpu.get("compute_capability") or 0) >= 7.5 and
                        int(re.match(r"\d+", gpu.get("driver_version", "0")).group()) >= 580
                        for gpu in gpus if re.match(r"\d+", gpu.get("driver_version", "0")))
    entry("cuda", cuda_platform and direct and supported_gpu and (cublas or tools["nvcc"]["available"]),
          "requires CUDA assembly, cuBLAS or toolkit evidence, NVIDIA R580+ driver and compute capability 7.5+; runtime dependencies remain checked at launch",
          [name for name in files if "cuda" in name or "cublas" in name] + [gpu["name"] for gpu in gpus])
    apple = system == "Darwin" and architecture.lower() in ("arm64", "aarch64")
    entry("ggml_metal", apple and ggml and metal_build,
          "requires local Apple Silicon, GgmlOps and Metal build evidence", runtime["build_metadata"])
    entry("mlx", apple and mlx,
          "requires local Apple Silicon and libmlxc.dylib; runtime must attest Metal device", [name for name in files if "mlxc" in name])
    entry("ggml_vulkan", False,
          "Vulkan device and compiled capability require explicit runtime preflight; no GPU-initializing probe is run during inventory",
          [name for name in files if "vulkan" in name])
    return entries


def detect_hardware(output_root: str | Path, runtime_dirs: list[str | Path] | tuple = ()) -> dict[str, Any]:
    """Inventory this host only. `output_root` should be the model/cache volume.

    Runtime dirs are explicit; this does not find or inspect other checkouts.
    Tool discovery never installs or builds anything.
    """
    tools = {name: _tool(name, args) for name, args in {
        "dotnet": ["--version"], "nvcc": ["--version"], "cmake": ["--version"],
        "ffmpeg": ["-version"], "nvidia-smi": ["--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap", "--format=csv,noheader,nounits"]
    }.items()}
    tools["python"] = {"path": sys.executable, "available": True, "version": platform.python_version()}
    if tools["dotnet"]["path"]:
        tools["dotnet"]["runtimes"] = _probe([tools["dotnet"]["path"], "--list-runtimes"])
    smi = tools["nvidia-smi"]
    if smi["path"] and not smi["available"]:
        fallback = _probe([smi["path"], "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"])
        if fallback["exit_code"] == 0:
            smi.update(fallback, available=True)
    gpus = parse_nvidia_csv(smi.get("stdout", "")) if smi["available"] else []
    system, architecture = platform.system(), platform.machine()
    runtime = _runtime_evidence(list(runtime_dirs))
    return {"os": system, "os_version": platform.version(), "architecture": architecture,
            "cpu_count": os.cpu_count(), "memory": _memory(), "disk": _disk(output_root),
            "tools": tools, "gpus": gpus, "runtime": runtime,
            "backends": _backend_inventory(system, architecture, tools, gpus, runtime),
            "limitations": ["Preflight estimates are not measured peak allocations.",
                            "No native backend or GPU context was initialized.",
                            "Multi-GPU VRAM is not pooled; distributed placement needs a separate recipe."]}


def plan_resources(weight_bytes: int | None, backend: str, hardware: dict[str, Any],
                   overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Conservative single-model planning; unavailable/unknown resources never pass.

    Byte overrides: download_bytes (default 0), scratch_bytes (default 0).
    GiB overrides: ram_budget_gib, vram_budget_gib, disk_budget_gib,
    ram_required_gib, vram_required_gib, reserve_ram_gib, reserve_vram_gib,
    disk_headroom_gib. Explicit gpu_weight_fraction < 1 is partial offload and
    does not assert architecture support. The caller must set matching flags.
    """
    options = dict(overrides or {})
    result: dict[str, Any] = {"eligible": False, "reason": "", "estimates": {}, "overrides": options,
                              "backend": backend, "offload": "full residency", "warnings": []}
    def reject(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        return result
    if backend not in BACKENDS:
        return reject("unsupported backend: " + backend)
    if not isinstance(weight_bytes, int) or isinstance(weight_bytes, bool) or weight_bytes <= 0:
        return reject("model size unknown or invalid; probe remote metadata or provide verified local files")
    try:
        for key, value in options.items():
            if key.endswith("_gib") or key in ("download_bytes", "scratch_bytes", "gpu_weight_fraction", "gpu_index"):
                if not math.isfinite(float(value)) or float(value) < 0:
                    return reject("invalid nonfinite or negative override: " + key)
        fraction = float(options.get("gpu_weight_fraction", 1))
        if not 0 <= fraction <= 1:
            return reject("gpu_weight_fraction must be in [0, 1]")
        if float(options.get("gpu_index", 0)) != int(float(options.get("gpu_index", 0))):
            return reject("gpu_index must be an integer device index")
    except (TypeError, ValueError):
        return reject("invalid numeric resource override")
    candidate = hardware.get("backends", {}).get(backend, {})
    if not candidate.get("available") and not options.get("assume_backend_available", False):
        return reject("backend unavailable: " + candidate.get("reason", "no preflight evidence"))
    if not candidate.get("available"):
        result["warnings"].append("backend availability explicitly overridden; actual backend attestation still mandatory")
    gpu_backend = backend not in ("cpu", "ggml_cpu")
    if fraction < 1:
        if not gpu_backend:
            return reject("gpu_weight_fraction does not apply to CPU backends")
        result["offload"] = "explicit partial offload: %.3f of weight bytes resident on GPU; caller must supply supported placement flags" % fraction
        result["warnings"].append("fraction is a planning override, not a validated architecture offload capability")
    memory = hardware.get("memory", {})
    available_ram = memory.get("available_bytes")
    ram_reserve = int(float(options.get("reserve_ram_gib", 4)) * GIB)
    if memory.get("total_bytes"):
        ram_reserve = max(ram_reserve, int(memory["total_bytes"] * 0.20))
    ram_budget = int(float(options["ram_budget_gib"]) * GIB) if "ram_budget_gib" in options else (
        max(0, available_ram - ram_reserve) if available_ram is not None else None)
    if ram_budget is not None and available_ram is not None and ram_budget > available_ram:
        ram_budget = available_ram
        result["warnings"].append("RAM budget capped at currently available physical RAM")
    # Quantized weights, CPU staging, KV and graph overhead. All weights are
    # charged in host RAM even for CUDA, avoiding optimistic mmap assumptions.
    ram_required = int(float(options.get("ram_required_gib", (weight_bytes * 1.25 + 2 * GIB) / GIB)) * GIB)
    disk_free = hardware.get("disk", {}).get("free_bytes")
    disk_headroom = int(float(options.get("disk_headroom_gib", 5)) * GIB)
    disk_budget = int(float(options["disk_budget_gib"]) * GIB) if "disk_budget_gib" in options else (
        max(0, disk_free - disk_headroom) if disk_free is not None else None)
    if disk_budget is not None and disk_free is not None and disk_budget > disk_free:
        disk_budget = disk_free
        result["warnings"].append("disk budget capped at currently free bytes")
    disk_required = int(options.get("download_bytes", 0)) + int(options.get("scratch_bytes", 0))
    estimates = result["estimates"]
    estimates.update(weight_bytes=weight_bytes, ram_required_bytes=ram_required,
                     ram_budget_bytes=ram_budget, ram_reserve_bytes=ram_reserve,
                     disk_required_bytes=disk_required, disk_budget_bytes=disk_budget,
                     disk_headroom_bytes=disk_headroom, gpu_weight_fraction=fraction)
    if ram_budget is None:
        return reject("available RAM unknown; set ram_budget_gib explicitly")
    if ram_required > ram_budget:
        return reject("estimated RAM exceeds budget (weights, staging and KV/graph reserve)")
    if disk_budget is None:
        return reject("free disk unknown; set disk_budget_gib explicitly")
    if disk_required > disk_budget:
        return reject("download and scratch bytes exceed disk budget after headroom")
    if gpu_backend:
        gpu_index = int(options.get("gpu_index", 0))
        gpus = hardware.get("gpus", [])
        gpu = next((x for x in gpus if x.get("index") == gpu_index), None)
        if backend == "cuda" and gpu:
            version = re.match(r"\d+", gpu.get("driver_version", "0"))
            if (gpu.get("compute_capability") or 0) < 7.5 or not version or int(version.group()) < 580:
                return reject("selected GPU does not establish direct CUDA requirements (R580+ and compute capability 7.5+)")
        unified = backend in ("ggml_metal", "mlx")
        free_vram = gpu.get("memory_free_bytes") if gpu else (available_ram if unified else None)
        reserve = int(float(options.get("reserve_vram_gib", 2)) * GIB)
        vram_budget = int(float(options["vram_budget_gib"]) * GIB) if "vram_budget_gib" in options else (
            max(0, free_vram - reserve) if free_vram is not None else None)
        if vram_budget is not None and free_vram is not None and vram_budget > free_vram:
            vram_budget = free_vram
            result["warnings"].append("VRAM budget capped at selected device's currently free bytes")
        vram_required = int(float(options.get("vram_required_gib", (weight_bytes * fraction * 1.25 + 2 * GIB) / GIB)) * GIB)
        estimates.update(gpu_index=gpu_index, vram_required_bytes=vram_required,
                         vram_budget_bytes=vram_budget, vram_reserve_bytes=reserve, unified_memory=unified)
        if unified:
            result["warnings"].append("unified GPU/host memory shares one budget; no double-counted capacity")
            vram_budget = min(ram_budget, vram_budget) if vram_budget is not None else None
            estimates["vram_budget_bytes"] = vram_budget
        if vram_budget is None:
            return reject("selected GPU free VRAM unknown; set vram_budget_gib explicitly")
        if vram_required > vram_budget:
            return reject("estimated single-GPU residency exceeds VRAM budget; no automatic CPU fallback")
    result["eligible"] = True
    result["reason"] = "within conservative preflight budgets; runtime measurements and backend attestation still required"
    return result
