#!/usr/bin/env python3
"""Matched Windows native-DLL A/B using identical isolated managed CLI assemblies.

Never replaces source binaries. One warmup per binary, then AB/BA/AB measured
pairs. Records actual loaded GgmlOps module identity, GPU samples, wall/denoise
times and exact baseline image/protected RGBA parity. No builds are performed.

The recorded output hash holds only for the edit noise it was made with: the runs
use the request report's TS_QWEN21_EDIT_NOISE, and `seed` for a report recorded
before that setting existed, when every edit drew the seed's noise.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("mask_bench", Path(__file__).with_name("qwen-image21-mask-bench.py"))
MASK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MASK)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def loaded_native(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.EnumProcessModulesEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                                         ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    handle = kernel.OpenProcess(0x0410, False, pid)
    if not handle:
        return None
    try:
        modules = (wintypes.HMODULE * 4096)()
        needed = wintypes.DWORD()
        if not psapi.EnumProcessModulesEx(handle, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3):
            return None
        for module in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            name = ctypes.create_unicode_buffer(32768)
            if psapi.GetModuleFileNameExW(handle, module, name, len(name)) and Path(name.value).name.lower() == "ggmlops.dll":
                return str(Path(name.value).resolve())
    finally:
        kernel.CloseHandle(handle)
    return None


def gpu_sample():
    result = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,power.draw,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    return {"sampled_monotonic": time.monotonic(), "exit_code": result.returncode,
            "csv": result.stdout.strip(), "error": result.stderr.strip()}


def stage_runtime(source, destination, native):
    if destination.exists():
        raise ValueError(f"Refusing an existing runtime directory: {destination}")
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.suffix.lower() in (".dll", ".json") and path.name.lower() != "ggmlops.dll":
            shutil.copy2(path, destination / path.name)
    for relative in (Path("runtimes/win-x64"), Path("cuda_kernels")):
        if (source / relative).exists():
            shutil.copytree(source / relative, destination / relative)
    shutil.copy2(native, destination / "GgmlOps.dll")
    return {path.name: digest(path) for path in destination.iterdir() if path.is_file()}


def run_one(args, runtime, native_hash, command_template, label, source, mask, expected_output, edit_noise):
    command = command_template.copy()
    command[1] = str(runtime / "TensorSharp.Cli.dll")
    image = args.out / f"{label}.png"
    command[command.index("--output") + 1] = str(image)
    log_path = args.out / f"{label}.log"
    started = time.monotonic()
    samples, mapped = [], None
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, TS_QWEN21_EDIT_NOISE=edit_noise),
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while child.poll() is None:
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError(f"{label} exceeded {args.timeout}s")
                if mapped is None:
                    mapped = loaded_native(child.pid)
                samples.append(gpu_sample())
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    elapsed = time.monotonic() - started
    result = {"name": label, "command": command, "pid": child.pid, "exit_code": child.returncode,
              "wall_seconds": elapsed, "gpu_samples": samples, "loaded_native_path": mapped}
    if child.returncode:
        raise ValueError(f"{label}: CLI failed with exit {child.returncode}; see {log_path}")
    if mapped is None or Path(mapped) != runtime / "GgmlOps.dll":
        raise ValueError(f"{label}: actual loaded module was {mapped}; expected isolated native DLL")
    result["loaded_native_sha256"] = digest(mapped)
    if result["loaded_native_sha256"] != native_hash:
        raise ValueError(f"{label}: loaded native hash changed")
    log_text = log_path.read_text(encoding="utf-8")
    result["denoise_seconds"] = int(re.search(r"\[qwen21-timing\] denoise: (\d+)ms", log_text).group(1)) / 1000
    result["phase_milliseconds"] = {name: int(value) for name, value in re.findall(r"\[qwen21-timing\] ([^:\r\n]+): (\d+)ms", log_text)}
    result["output_sha256"] = digest(image)
    if result["output_sha256"] != expected_output:
        raise ValueError(f"{label}: output differs from the recorded baseline for this mode")
    result["pixels"] = MASK.measure_pixels(source, image, mask)
    result["passed"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-native", type=Path, required=True)
    parser.add_argument("--current-native", type=Path, default=ROOT / "TensorSharp.Cli/bin/GgmlOps.dll")
    parser.add_argument("--cli-directory", type=Path, default=ROOT / "TensorSharp.Cli/bin")
    parser.add_argument("--request-report", type=Path, default=ROOT / "docs/validation/qwen-image21-mask-merge/cli-benchmark/report.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--crop", action="store_true", help="Use the report's cropped request instead of its full-frame request")
    parser.add_argument("--cooldown", type=float, default=10)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This identity probe uses Windows process modules")
    if args.cooldown < 0 or args.cooldown > 60:
        parser.error("cooldown must be between 0 and 60 seconds")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    request = json.loads(args.request_report.read_text())
    template = next(run for run in request["runs"] if bool(run["crop"]) == args.crop)
    edit_noise = request.get("edit_noise", "seed")
    if edit_noise not in ("references", "seed"):
        parser.error(f"The request report's edit noise is {edit_noise!r}; a CLI report records references or seed")
    command = template["command"]
    source = Path(command[command.index("--image") + 1])
    mask = Path(command[command.index("--mask") + 1])
    natives = {"A": args.baseline_native.resolve(), "B": args.current_native.resolve()}
    hashes = {key: digest(path) for key, path in natives.items()}
    runtimes = {key: args.out / f"runtime-{key}" for key in natives}
    manifests = {key: stage_runtime(args.cli_directory.resolve(), runtimes[key], path) for key, path in natives.items()}
    assert {k: v for k, v in manifests["A"].items() if k != "GgmlOps.dll"} == {
        k: v for k, v in manifests["B"].items() if k != "GgmlOps.dll"}
    report = {"passed": False, "native_inputs": {key: {"path": str(path), "sha256": hashes[key]} for key, path in natives.items()},
              "runtime_manifests": manifests, "dependencies": {"ggml": MASK.revision(ROOT / "ExternalProjects/ggml")},
              "warmups": [], "runs": [], "cooldown_seconds": args.cooldown, "crop": args.crop, "edit_noise": edit_noise,
              "limitations": ["A is the preserved premerge CFBA binary, not the unavailable B1E800 merge-validation binary.",
                              "Both variants use current managed assemblies and the same ordered image/mask workload; mode is recorded in crop.",
                              "Only this synthetic source/edit, one CUDA device, and three paired repeats are measured; not general model quality.",
                              "GPU power/clocks/temperature sampled about every 2 seconds but not fixed; OS cache/thermal drift cannot be eliminated.",
                              "Wall includes native module probing and GPU sampling overhead; denoise phase is logged by the pipeline."]}
    output = args.out / "report.json"
    try:
        if report["dependencies"]["ggml"]["changes"]:
            raise ValueError("Upstream ggml checkout must remain unchanged")
        schedule = [("A", "warmup-A", True), ("B", "warmup-B", True),
                    ("A", "pair1-A", False), ("B", "pair1-B", False),
                    ("B", "pair2-B", False), ("A", "pair2-A", False),
                    ("A", "pair3-A", False), ("B", "pair3-B", False)]
        for index, (variant, label, warmup) in enumerate(schedule):
            if index:
                time.sleep(args.cooldown)
            print(f"Starting {label}", flush=True)
            result = run_one(args, runtimes[variant], hashes[variant], command, label, source, mask, template["output_sha256"], edit_noise)
            result["variant"] = variant
            report["warmups" if warmup else "runs"].append(result)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: result[k] for k in ("name", "wall_seconds", "denoise_seconds", "output_sha256", "passed")}), flush=True)
        report["medians"] = {variant: {metric: statistics.median(run[metric] for run in report["runs"] if run["variant"] == variant)
                            for metric in ("wall_seconds", "denoise_seconds")} for variant in natives}
        report["current_change_percent"] = {metric: (report["medians"]["B"][metric] / report["medians"]["A"][metric] - 1) * 100
                                           for metric in ("wall_seconds", "denoise_seconds")}
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
    report["original_native_hashes_unchanged"] = all(digest(path) == hashes[key] for key, path in natives.items())
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("passed", "medians", "current_change_percent", "error")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
