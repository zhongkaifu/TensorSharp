#!/usr/bin/env python3
"""Run TensorSharp and stable-diffusion.cpp serially with matched Qwen-Image-2.1 inputs.

Each measurement launches a fresh process; OS file-cache state is not controlled.
The JSON reports real wall time, engine phases, individual steps, peak RSS, hashes,
dependency revisions, failures, and output statistics. Statistics and pixel parity
are diagnostic evidence, not a semantic quality score. Generated files default to
the ignored docs/validation/qwen-image-2.1 directory.

Examples:
  python3 eng/validation/qwen-image21-bench.py --prompt 'A red ceramic teapot.'
  python3 eng/validation/qwen-image21-bench.py --mode edit --image input.png \
      --prompt 'Change the teapot to blue.'
  python3 eng/validation/qwen-image21-bench.py --mode multi --image a.png --image b.png \
      --prompt 'Place the object from image 2 into image 1.' --width 1024 --height 1024
  python3 eng/validation/qwen-image21-bench.py --dry-run
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = {
    "dit": "qwen_image_2.1_Q4_K_M.gguf",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
    "text_encoder": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    "mmproj": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
}


def qwen21_sigmas(steps, image_tokens):
    """Match QwenImage21Sampling.Sigmas, including its F32 output rounding.

    sd.cpp's default Flux schedule uses different anchors and no terminal stretch.
    Explicit --sigmas makes the reference use the same denoising workload.
    """
    if steps <= 0 or image_tokens <= 0:
        raise ValueError("Steps and image-token count must be positive.")
    if steps == 1:
        return [1.0, 0.0]
    f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    exp_mu = math.exp(0.5 + (image_tokens - 256) * (0.9 - 0.5) / (8192 - 256))
    last = exp_mu / (exp_mu + steps - 1)
    terminal_scale = (1 - last) / (1 - 0.02)
    result = []
    for index in range(steps):
        time = (steps - index) / steps
        shifted = exp_mu / (exp_mu + (1 / time - 1))
        result.append(f32(1 - (1 - shifted) / terminal_scale))
    result[-1] = f32(0.02)
    return result + [0.0]


def recipe_sigmas(nodes, shift, image_tokens):
    """Match QwenImage21LoraRecipe.Sigmas: raw nodes, optionally through the checkpoint's
    dynamic exponential shift (no shift_terminal), then a terminal 0, rounded to F32."""
    f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    exp_mu = math.exp(0.5 + (image_tokens - 256) * (0.9 - 0.5) / (8192 - 256))
    result = []
    for t in map(f32, nodes):  # TensorSharp reads the nodes as F32 (GetSingle) before shifting
        result.append(f32(exp_mu / (exp_mu + (1 / t - 1))) if shift == "dynamic" else t)
    return result + [0.0]


def comparison_notes(args):
    notes = []
    prompts = [args.prompt] + ([args.negative_prompt] if args.cfg > 1 else [])
    if any(any(character in prompt for character in "()[]") for prompt in prompts):
        notes.append("sd.cpp parses parentheses/brackets as prompt-weighting syntax; TensorSharp Qwen 2.1 encodes them literally. These prompts may produce different conditioning.")
    if args.image:
        notes.append("Reference resizing uses different filters (TensorSharp Lanczos, sd.cpp nearest-neighbor). For pixel comparisons, supply identical references already at the intended conditioning dimensions.")
        if args.width * args.height > 1024 * 1024:
            notes.append("TensorSharp caps reference conditioning near 1 megapixel; sd.cpp defaults to the output area. Match its geometry with --sd-extra=--ref-image-args --sd-extra=vae_input_max_pixels=1048576 or review an explicit reference-resize override.")
    if args.ts_extra or args.sd_extra:
        notes.append("Extra arguments can override the reported common settings; review each recorded engine command before treating the workloads as equivalent.")
    return notes


def capture(argv):
    try:
        return subprocess.check_output(argv, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def revision(path):
    return {"path": str(path), "commit": capture(["git", "-C", str(path), "rev-parse", "HEAD"]),
            "changes": capture(["git", "-C", str(path), "status", "--porcelain"])}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path, hash_contents=True):
    info = {"path": str(path), "exists": path.is_file()}
    if info["exists"]:
        stat = path.stat()
        info.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        info["sha256"] = sha256(path) if hash_contents else None
    return info


def image_info(path):
    info = file_info(path)
    if not info["exists"]:
        return info
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
            stat = ImageStat.Stat(rgb)
            histogram = rgb.histogram()
            count = rgb.width * rgb.height * 3
            alpha = im.convert("RGBA").getchannel("A")
            alpha_histogram = alpha.histogram()
            info.update(width=rgb.width, height=rgb.height, mode=im.mode,
                        channel_mean=stat.mean, channel_stddev=stat.stddev, channel_extrema=stat.extrema,
                        black_channel_fraction=sum(histogram[i*256] for i in range(3))/count,
                        saturated_channel_fraction=sum(histogram[i*256+255] for i in range(3))/count,
                        alpha_extrema_255=alpha.getextrema(), alpha_mean_255=ImageStat.Stat(alpha).mean[0],
                        nonopaque_pixel_fraction=1-alpha_histogram[255]/(rgb.width*rgb.height))
    except ImportError:
        info["statistics_unavailable"] = "Install Pillow to inspect pixels and dimensions."
    except Exception as error:
        info["decode_error"] = str(error)
    return info


def pixel_comparison(first, second):
    try:
        import numpy as np
        from PIL import Image
        with Image.open(first) as im:
            a = np.asarray(im.convert("RGBA"), dtype=np.float64)
        with Image.open(second) as im:
            b = np.asarray(im.convert("RGBA"), dtype=np.float64)
        if a.shape != b.shape:
            return {"available": False, "reason": "Output image dimensions differ."}
        def metrics(x, y):
            diff = x-y
            mse = float(np.mean(diff*diff))
            return {"mean_absolute_error_255": float(np.mean(np.abs(diff))), "rmse_255": math.sqrt(mse),
                    "psnr_db": 10*math.log10(255*255/mse) if mse else None,
                    "identical_pixels": bool(np.array_equal(x, y))}
        def white(rgb_alpha):
            alpha = rgb_alpha[..., 3:4]/255
            return rgb_alpha[..., :3]*alpha + 255*(1-alpha)
        return {"available": True, "raw_rgba": metrics(a, b), "white_composited_rgb": metrics(white(a), white(b)),
                "alpha": metrics(a[..., 3], b[..., 3]),
                "interpretation": "Pixel similarity is not a prompt-adherence or image-quality assessment."}
    except (ImportError, OSError) as error:
        return {"available": False, "reason": str(error)}


def parse_log(engine, text):
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n")
    phases, steps, fallbacks = {}, [], []
    if engine == "tensorsharp":
        for name, value, unit in re.findall(r"\[qwen21-timing\]\s+([^:\n]+):\s+([0-9.]+)(ms|s)", text):
            phases[name] = float(value) / (1000 if unit == "ms" else 1)
        for index, count, seconds in re.findall(r"\[qwen21-step\]\s+(\d+)/(\d+):\s+([0-9.]+)s", text):
            steps.append({"step": int(index), "total": int(count), "seconds": float(seconds)})
        match = re.search(r"Loaded model[^\n]*elapsedMs=([0-9.]+)", text)
        if match:
            phases["model load"] = float(match[1])/1000
    else:
        for name, pattern in {
            "text and vision encode": r"get_learned_condition completed, taking ([0-9.]+)s",
            "VAE encode": r"encode_first_stage completed, taking ([0-9.]+)s",
            "denoise": r"sampling completed, taking ([0-9.]+)s",
            "VAE decode": r"decode_first_stage completed, taking ([0-9.]+)s",
            "total": r"generate_image completed in ([0-9.]+)s",
        }.items():
            matches = re.findall(pattern, text)
            if matches:
                phases[name] = sum(map(float, matches))
        # VAE tiling prints the same progress format. Restrict denoising progress
        # to the sampling phase so a later 9/9 tile counter cannot replace 40/40.
        sampling_sections = re.findall(r"generating image:\s+\d+/\d+[^\n]*\n(.*?)sampling completed, taking", text, re.S)
        sampling_text = "\n".join(sampling_sections) if sampling_sections else text.split("sampling completed, taking", 1)[0]
        for index, count, value, unit in re.findall(r"\|\s+(\d+)/(\d+)\s+-\s+([0-9.]+)(s/it|it/s)", sampling_text):
            rate = float(value)
            if unit == "it/s" and rate <= 0:
                continue
            seconds = 1 / rate if unit == "it/s" else rate
            steps.append({"step": int(index), "total": int(count), "seconds": seconds,
                          "reported_value": rate, "reported_unit": unit})
        # sd.cpp reworded this warning in backend_fit.cpp; accept both spellings.
        match = re.search(r"VAE decode (?:failed \(likely out of memory\)|ran out of memory); retrying with spatial tiling", text)
        if match:
            marker = match[0]
            after_retry = text[match.end():]
            recovery = re.search(r"decode_first_stage completed, taking [0-9.]+s.*?generate_image completed in [0-9.]+s.*?save result image[^\n]*\(success\)", after_retry, re.S)
            fallbacks.append({"kind": "sd_cpp_vae_spatial_tiling", "trigger": marker,
                              "recovery_confirmed_in_log": bool(recovery),
                              "interpretation": "Full VAE decoding failed and was retried with spatial tiling; its workload and pixels may differ from full decoding."})
    rss = None
    match = re.search(r"(\d+)\s+maximum resident set size", text)
    if match:
        rss = int(match[1])  # Darwin /usr/bin/time -l emits bytes.
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if match:
        rss = int(match[1])*1024
    errors = [line.strip() for line in text.splitlines()
              if re.search(r"\[ERROR|\[error|Unhandled exception|GGML_ASSERT|segmentation fault|non-finite|failed to|FAIL|error:", line, re.I)]
    values = [s["seconds"] for s in steps]
    return {"phases_seconds": phases, "steps": steps, "step_mean_seconds": statistics.mean(values) if values else None,
            "step_median_seconds": statistics.median(values) if values else None,
            "steady_step_mean_seconds": statistics.mean(values[1:]) if len(values) > 1 else None,
            "max_rss_bytes": rss, "error_lines": errors, "fallback_events": fallbacks}


def classify_result(result, engine, steps, width, height):
    """Keep recovered VAE failures explicit; never turn unrelated errors into a pass."""
    failures = []
    output = result["image"]
    if result["exit_code"] != 0 or result.get("timed_out"):
        failures.append("The engine did not exit successfully.")
    if not output.get("exists") or output.get("decode_error"):
        failures.append("The requested output image is absent or cannot be decoded.")
    if [s["step"] for s in result["steps"]] != list(range(1, steps + 1)) or any(s["total"] != steps for s in result["steps"]):
        failures.append("The log does not confirm completion of the requested denoising steps.")
    if output.get("width") is not None and (output["width"], output["height"]) != (width, height):
        failures.append("Output dimensions do not match the request.")
    fallbacks = result.get("fallback_events", [])
    if fallbacks:
        known_errors = (
            r"wan_vae segment \d+/\d+ \(graph\) failed during weight preparation$",
            r"vae decode compute failed$",
            r"VAE decode (?:failed \(likely out of memory\)|ran out of memory); retrying with spatial tiling$",
        )
        if engine != "sd_cpp" or len(fallbacks) != 1 or not fallbacks[0]["recovery_confirmed_in_log"]:
            failures.append("The fallback was not confirmed to recover successfully.")
        if any(not any(re.search(pattern, line) for pattern in known_errors) for line in result["error_lines"]):
            failures.append("The run contains errors outside the recognized recovered VAE failure.")
        if (output.get("width"), output.get("height")) != (width, height):
            failures.append("Successful fallback requires a decoded image with the requested dimensions.")
    elif result["error_lines"]:
        failures.append("The run contains errors without a recognized successful recovery.")
    result["validation_failures"] = failures
    result["status"] = "failed" if failures else "completed_with_fallback" if fallbacks else "passed"
    return result["status"]


def cooldown_before_run(seconds, already_ran):
    """Cooling is outside engine latency and occurs only between invocations."""
    if seconds <= 0 or not already_ran:
        return 0.0
    start = time.perf_counter()
    time.sleep(seconds)
    return time.perf_counter() - start


def parse_gpu_sample(output):
    devices = []
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        if len(row) != 7:
            raise ValueError("Unexpected nvidia-smi CSV column count.")
        row = [value.strip() for value in row]
        def number(value):
            try:
                parsed = float(value)
                return parsed if math.isfinite(parsed) else None
            except ValueError:
                return None
        devices.append({"index": row[0], "uuid": row[1], "name": row[2],
                        "memory_used_mib": number(row[3]), "memory_total_mib": number(row[4]),
                        "utilization_percent": number(row[5]), "temperature_c": number(row[6])})
    if not devices:
        raise ValueError("nvidia-smi returned no GPUs.")
    return devices


def query_gpu_sample(executable):
    result = subprocess.run(
        [executable, "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    if result.returncode:
        raise RuntimeError(f"nvidia-smi exited {result.returncode}: {result.stderr.strip()[:300]}")
    return parse_gpu_sample(result.stdout)


def summarize_gpu_samples(baseline, samples, interval, errors):
    peaks = {}
    for sample in ([{"devices": baseline}] if baseline else []) + samples:
        for device in sample["devices"]:
            peak = peaks.setdefault(device["uuid"], {"index": device["index"], "uuid": device["uuid"],
                                                    "name": device["name"]})
            for metric in ("memory_used_mib", "utilization_percent", "temperature_c"):
                value = device[metric]
                key = "peak_" + metric
                if value is not None:
                    previous = peak.get(key)
                    peak[key] = max(value, previous) if previous is not None else value
                else:
                    peak.setdefault(key, None)
    return {"status": "available" if samples else "baseline_only" if baseline else "unavailable",
            "runtime_sampling_status": "available" if samples else "unavailable", "source": "nvidia-smi",
            "scope": "whole device, including desktop and other processes; not this process's GPU allocation",
            "peak_sample_scope": "baseline and runtime samples",
            "sample_interval_seconds": interval, "baseline": baseline, "peaks": list(peaks.values()),
            "samples": samples, "errors": errors,
            "limitations": "Sampled peaks can miss shorter spikes. Query time adds to the requested interval. Monitoring overhead is not isolated."}


class WindowsProcessMemory:
    def __init__(self, pid):
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        self.ctypes, self.counters = ctypes, Counters()
        self.counters.cb = ctypes.sizeof(Counters)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x0410, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def sample(self):
        if not self.psapi.GetProcessMemoryInfo(self.handle, self.ctypes.byref(self.counters), self.counters.cb):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return int(self.counters.PeakWorkingSetSize)

    def close(self):
        self.kernel.CloseHandle(self.handle)


def run_process(command, logfile, timeout, env, sample_cuda=False, sample_interval=1.0):
    # External time measures each child separately; RUSAGE_CHILDREN would retain
    # the high-water mark of previous runs and misreport serial comparisons.
    timer = Path("/usr/bin/time")
    measured = ([str(timer), "-l" if sys.platform == "darwin" else "-v"] + command
                if timer.is_file() and sys.platform != "win32" else command)
    gpu_executable = shutil.which("nvidia-smi") if sample_cuda else None
    gpu_baseline, gpu_samples, gpu_errors = [], [], []
    if sample_cuda:
        if gpu_executable:
            try:
                gpu_baseline = query_gpu_sample(gpu_executable)
            except Exception as error:
                gpu_errors.append(str(error))
                gpu_executable = None
        else:
            gpu_errors.append("nvidia-smi is not on PATH.")
    process_memory = {"status": "unavailable", "source": "GetProcessMemoryInfo.PeakWorkingSetSize",
                      "peak_rss_bytes": None, "samples": 0, "sample_interval_seconds": sample_interval,
                      "errors": [], "scope": "process peak working set, not GPU allocation"}
    started = time.monotonic()
    timed_out = False
    with logfile.open("w") as out:
        child = subprocess.Popen(measured, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT,
                                 env=env, start_new_session=(sys.platform != "win32"))
        memory_reader = None
        if sys.platform == "win32":
            try:
                memory_reader = WindowsProcessMemory(child.pid)
            except Exception as error:
                process_memory["errors"].append(str(error))
        else:
            process_memory["errors"].append("Windows process-memory API is unavailable on this platform.")
        stop = threading.Event()

        def sample_memory():
            if memory_reader:
                try:
                    peak = memory_reader.sample()
                    process_memory["peak_rss_bytes"] = max(peak, process_memory["peak_rss_bytes"] or 0)
                    process_memory["samples"] += 1
                    process_memory["status"] = "available"
                except Exception as error:
                    if not process_memory["errors"]:
                        process_memory["errors"].append(str(error))

        def monitor():
            sample_memory()
            while not stop.wait(sample_interval):
                sample_memory()
                if gpu_executable and len(gpu_errors) < 3:
                    try:
                        devices = query_gpu_sample(gpu_executable)
                        gpu_samples.append({"elapsed_seconds": time.monotonic()-started, "devices": devices})
                    except Exception as error:
                        gpu_errors.append(str(error))

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        try:
            try:
                code = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if sys.platform == "win32":
                    child.kill()
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
        finally:
            wall_seconds = time.monotonic()-started
            stop.set()
            monitor_thread.join()
            if memory_reader:
                sample_memory()
                memory_reader.close()
    return {"exit_code": code, "timed_out": timed_out, "wall_seconds": wall_seconds,
            "command": command, "measurement_command": measured, "log": str(logfile),
            "windows_process_memory": process_memory,
            "gpu_sampling": summarize_gpu_samples(gpu_baseline, gpu_samples, sample_interval, gpu_errors)
                            if sample_cuda else {"status": "not_requested"}}


def commands(args, models, ts_image, sd_image):
    cli = [args.dotnet, str(args.cli)] if args.cli.suffix == ".dll" else [str(args.cli)]
    ts = cli + ["--model", str(models["dit"]), "--backend", args.backend,
                "--qwen-image-vae", str(models["vae"]), "--qwen-image-vl", str(models["text_encoder"]),
                "--qwen-image-mmproj", str(models["mmproj"]), "--prompt", args.prompt,
                "--width", str(args.width), "--height", str(args.height), "--diffusion-steps", str(args.steps),
                "--cfg", str(args.cfg), "--diffusion-seed", str(args.seed), "--output", str(ts_image)]
    sd = [str(args.sd_cli), "--diffusion-model", str(models["dit"]), "--vae", str(models["vae"]),
          "--llm", str(models["text_encoder"]), "-p", args.prompt, "-W", str(args.width), "-H", str(args.height),
          "--steps", str(args.steps), "--cfg-scale", str(args.cfg), "--sampling-method", "euler",
          "--rng", "cuda", "--seed", str(args.seed), "--fa", "-o", str(sd_image)]
    if args.sd_backend:
        sd += ["--backend", args.sd_backend]
    if args.sigma_nodes:
        # A LoRA recipe's schedule; TensorSharp derives the same values from --lora-config.
        nodes = [float(v) for v in args.sigma_nodes.split(",")]
        sigmas = recipe_sigmas(nodes, args.sigma_shift, (args.width // 16) * (args.height // 16))
        sd += ["--sigmas", ",".join(format(value, ".9g") for value in sigmas)]
    elif args.match_sigmas:
        sigmas = qwen21_sigmas(args.steps, (args.width // 16) * (args.height // 16))
        sd += ["--sigmas", ",".join(format(value, ".9g") for value in sigmas)]
    if args.lora:
        ts += ["--lora", str(args.lora)]
        if args.lora_scale is not None:
            ts += ["--lora-scale", str(args.lora_scale)]
        if args.lora_config:
            ts += ["--lora-config", str(args.lora_config)]
        # sd.cpp takes the adapter as a prompt tag relative to --lora-model-dir. It does not
        # read PEFT alpha metadata, so --sd-lora-multiplier must carry alpha / rank when it
        # differs from 1 (e.g. 2 for the Pruna adapters).
        multiplier = (args.sd_lora_multiplier if args.sd_lora_multiplier is not None
                      else args.lora_scale if args.lora_scale is not None else 1.0)
        sd += ["--lora-model-dir", str(args.lora.parent)]
        sd[sd.index("-p") + 1] = args.prompt + f"<lora:{args.lora.stem}:{multiplier:g}>"
    if args.negative_prompt:
        ts += ["--negative-prompt", args.negative_prompt]
        sd += ["-n", args.negative_prompt]
    if args.image:
        sd += ["--llm_vision", str(models["mmproj"])]
    for image in args.image:
        ts += ["--image", str(image)]
        sd += ["--ref-image", str(image)]
    return {"tensorsharp": ts + args.ts_extra, "sd_cpp": sd + args.sd_extra}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("t2i", "edit", "multi"), default="t2i")
    parser.add_argument("--prompt", default="A red ceramic teapot on a wooden table, soft daylight, product photograph.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=ROOT.parent/"models/qwen-image-2.1")
    for name in MODEL_NAMES:
        parser.add_argument("--"+name.replace("_", "-"), type=Path)
    parser.add_argument("--cli", type=Path, default=ROOT/"TensorSharp.Cli/bin/TensorSharp.Cli.dll")
    parser.add_argument("--dotnet", default="dotnet")
    parser.add_argument("--sd-cli", type=Path, default=ROOT/"artifacts/qwen-image-2.1/sd-build/bin/sd-cli")
    parser.add_argument("--sd-repo", type=Path, default=ROOT.parent/"stable-diffusion.cpp")
    parser.add_argument("--ggml-repo", type=Path, default=ROOT/"ExternalProjects/ggml")
    parser.add_argument("--sd-ggml-repo", type=Path,
                        help="ggml source tree used to build sd.cpp; defaults to its ggml submodule. Set this when SD_GGML_SOURCE_DIR was overridden.")
    parser.add_argument("--backend", default="ggml_metal", choices=("ggml_metal", "ggml_cuda", "ggml_cpu", "ggml_vulkan"))
    parser.add_argument("--sd-backend", help="Explicit sd.cpp assignment; by default matches --backend using metal/cpu/cuda0/vulkan0.")
    parser.add_argument("--match-sigmas", action="store_true",
                        help="Pass TensorSharp's official Qwen 2.1 sigma schedule to sd.cpp via --sigmas; requires sd.cpp with custom-sigma support.")
    parser.add_argument("--lora", type=Path, help="LoRA weights (.safetensors) for both engines.")
    parser.add_argument("--lora-config", type=Path, help="TensorSharp --lora-config (the plug-in recipe).")
    parser.add_argument("--lora-scale", type=float, help="TensorSharp --lora-scale.")
    parser.add_argument("--sd-lora-multiplier", type=float,
                        help="sd.cpp <lora:name:multiplier>; defaults to --lora-scale (or 1). sd.cpp ignores PEFT alpha metadata.")
    parser.add_argument("--sigma-nodes", help="Comma-separated recipe sigma nodes passed to sd.cpp as --sigmas (after --sigma-shift).")
    parser.add_argument("--sigma-shift", choices=("none", "dynamic"), default="none")
    parser.add_argument("--ts-extra", action="append", default=[], help="Additional TensorSharp argv token; use --ts-extra=--option.")
    parser.add_argument("--sd-extra", action="append", default=[], help="Additional sd.cpp argv token; use --sd-extra=--option.")
    parser.add_argument("--engine-order", choices=("sd-first", "ts-first"), default="sd-first")
    parser.add_argument("--engine", choices=("both", "tensorsharp", "sd_cpp"), default="both",
                        help="Run a single engine for quality/performance validation without requiring the reference binary.")
    parser.add_argument("--repeat", type=int, default=1, help="Serial fresh-process measurements per engine, not warm in-process requests.")
    parser.add_argument("--timeout", type=float, default=3600, help="Seconds per engine invocation.")
    parser.add_argument("--sample-interval", type=float, default=1.0,
                        help="Seconds between Windows process-memory/CUDA whole-device samples (default: 1).")
    parser.add_argument("--cooldown-seconds", type=float, default=0,
                        help="Wait between engine invocations, outside measured engine latency (default: 0). Cool the device before starting separately.")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/validation/qwen-image-2.1"/datetime.now().strftime("bench-%Y%m%d-%H%M%S"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands and manifest without reading model contents or running inference.")
    args = parser.parse_args()
    args.sd_backend = args.sd_backend or {"ggml_metal": "metal", "ggml_cpu": "cpu", "ggml_cuda": "cuda0", "ggml_vulkan": "vulkan0"}[args.backend]
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32 or args.steps <= 0 or args.repeat <= 0:
        parser.error("Dimensions must be positive multiples of 32; steps and repeat must be positive.")
    if not math.isfinite(args.cfg) or args.cfg <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("CFG and timeout must be finite and positive.")
    if not math.isfinite(args.sample_interval) or args.sample_interval < 0.1:
        parser.error("Sample interval must be finite and at least 0.1 seconds.")
    if not math.isfinite(args.cooldown_seconds) or args.cooldown_seconds < 0:
        parser.error("Cooldown must be finite and nonnegative.")
    if args.match_sigmas and any(value == "--sigmas" or value.startswith("--sigmas=") for value in args.sd_extra):
        parser.error("--match-sigmas cannot be combined with --sd-extra=--sigmas.")
    if (args.mode == "t2i" and args.image) or (args.mode == "edit" and len(args.image) != 1) or (args.mode == "multi" and len(args.image) < 2):
        parser.error("t2i takes no images; edit takes exactly one; multi takes two or more --image arguments.")
    models = {n: (getattr(args, n) or args.models_dir/f).resolve() for n, f in MODEL_NAMES.items()}
    args.image = [p.resolve() for p in args.image]
    args.cli, args.sd_cli, args.sd_repo, args.ggml_repo, args.output = (
        p.resolve() for p in (args.cli, args.sd_cli, args.sd_repo, args.ggml_repo, args.output))
    args.sd_ggml_repo = (args.sd_ggml_repo or args.sd_repo/"ggml").resolve()
    if (args.lora_config or args.lora_scale is not None or args.sd_lora_multiplier is not None or args.sigma_nodes) and not args.lora \
            and not args.sigma_nodes:
        parser.error("--lora-config/--lora-scale/--sd-lora-multiplier require --lora.")
    if args.lora:
        args.lora = args.lora.resolve()
        if args.lora.suffix.lower() != ".safetensors":
            parser.error("--lora takes the .safetensors weights (sd.cpp cannot read a plug-in .json); "
                         "pass the TensorSharp plug-in with --lora-config.")
        if args.lora_config:
            args.lora_config = args.lora_config.resolve()
            # A plug-in's own strength applies to TensorSharp; sd.cpp must get the same one.
            config = json.loads(re.sub(r"(?m)^\s*//.*$", "", args.lora_config.read_text()))  # full-line // comments only
            if args.lora_scale is None and args.sd_lora_multiplier is None and config.get("scale", 1.0) != 1.0:
                parser.error(f"{args.lora_config.name} sets strength {config['scale']}; pass --lora-scale (or "
                             "--sd-lora-multiplier) so sd.cpp gets the same strength.")
    if not args.dry_run:
        binaries = ([args.cli] if args.engine == "tensorsharp" else [args.sd_cli]
                    if args.engine == "sd_cpp" else [args.cli, args.sd_cli])
        lora_files = ([args.lora] if args.lora else []) + ([args.lora_config] if args.lora_config else [])
        for path in list(models.values()) + args.image + binaries + lora_files:
            if not path.is_file():
                parser.error(f"Required file is missing: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Only named numerical/performance knobs are recorded; unrelated environment
    # variables may contain credentials and must never enter a report.
    device_env = {"CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUDA_MODULE_LOADING",
                  "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "SD_MMAP_FLAGS",
                  "TS_VAE_CUDNN_CONV", "TS_CUDNN_DIR"}
    relevant_env = {k: v for k, v in env.items()
                    if k.startswith(("TS_QWEN", "GGML_", "TENSORSHARP_GGML_")) or k in device_env}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dry_run": args.dry_run,
        "parameters": {k: getattr(args, k) for k in ("mode", "prompt", "negative_prompt", "width", "height", "steps", "cfg", "seed", "backend", "sd_backend", "repeat", "match_sigmas")},
        "sampler": "Euler", "noise": "Philox4x32-10 / Box-Muller (--rng cuda)",
        "engine_selection": args.engine,
        "cooldown_seconds": args.cooldown_seconds,
        "tensorsharp_schedule": {"base_image_seq_len": 256, "max_image_seq_len": 8192,
                                 "base_shift": 0.5, "max_shift": 0.9, "shift_terminal": 0.02},
        "sd_cpp_sigmas": qwen21_sigmas(args.steps, (args.width // 16) * (args.height // 16)) if args.match_sigmas else None,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "cpu": capture(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else platform.processor(),
                     "memory_bytes": capture(["sysctl", "-n", "hw.memsize"]) if sys.platform == "darwin" else None},
        "revisions": {"tensorsharp": revision(ROOT), "sd_cpp": revision(args.sd_repo), "ggml": revision(args.ggml_repo),
                      "sd_cpp_ggml": revision(args.sd_ggml_repo)},
        "models": {n: file_info(p, not args.dry_run) for n, p in models.items()},
        "references": [file_info(p, not args.dry_run) for p in args.image],
        "executables": {"cli": file_info(args.cli, not args.dry_run), "sd_cli": file_info(args.sd_cli, not args.dry_run)},
        "tensorsharp_implementation_binaries": {
            name: file_info(args.cli.parent/name, not args.dry_run)
            for name in (("GgmlOps.dll" if sys.platform == "win32" else
                          "libGgmlOps.dylib" if sys.platform == "darwin" else "libGgmlOps.so"),
                         "TensorSharp.Models.dll")},
        "environment": relevant_env,
        "extra_arguments": {"tensorsharp": args.ts_extra, "sd_cpp": args.sd_extra},
        "limitations": ["Fresh processes; OS file cache and GPU thermal state are not controlled.",
                        "Peak RSS is process resident memory, not peak GPU allocation; it includes shared mappings on unified-memory systems.",
                        "Phase boundaries and weight-loading inclusion differ between engines; compare wall time alongside the phase logs.",
                        "Per-step timings come from rounded progress output (sd.cpp may report reciprocal iterations/second); use phase/process timers for total latency.",
                        "sd.cpp enables automatic weight placement/graph segmentation by default. Check its runtime placement logs; --backend cuda0 alone does not prove every weight remained on the GPU.",
                        "Pixel statistics and matched-seed similarity do not establish semantic quality or fidelity.",
                        ("sd.cpp receives the same F32 sigma vector as TensorSharp through --sigmas. Kernel and encoder rounding can still change pixels."
                         if args.match_sigmas else
                         "TensorSharp uses the released Qwen 2.1 scheduler. Older sd.cpp revisions use Flux shift defaults; identical CLI settings do not imply identical sigma schedules. Use --match-sigmas for matched denoising schedules."),
                        "No unavailable model/device scenario is counted as a passing measurement."] + comparison_notes(args),
        "runs": [],
    }
    report_path = args.output/"benchmark.json"

    def save():
        report_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    already_ran = False
    for index in range(args.repeat):
        paths = {engine: args.output/f"{engine}-{index+1}.png" for engine in ("tensorsharp", "sd_cpp")}
        argv = commands(args, models, paths["tensorsharp"], paths["sd_cpp"])
        pair = {"repeat": index+1, "engines": {}}
        manifest["runs"].append(pair)
        order = ("sd_cpp", "tensorsharp") if args.engine_order == "sd-first" else ("tensorsharp", "sd_cpp")
        if args.engine != "both":
            order = (args.engine,)
        for engine in order:
            print(f"[{index+1}/{args.repeat}] {engine}: {shlex.join(argv[engine])}", flush=True)
            if args.dry_run:
                pair["engines"][engine] = {"status": "not_run", "command": argv[engine]}
                save()
                continue
            logfile = args.output/f"{engine}-{index+1}.log"
            # Never mistake a stale image from an earlier failed run for success.
            if paths[engine].exists():
                parser.error(f"Output already exists; choose a new --output directory: {paths[engine]}")
            if already_ran and args.cooldown_seconds:
                print(f"  Cooling for {args.cooldown_seconds:g}s before the next engine invocation.", flush=True)
            cooled_seconds = cooldown_before_run(args.cooldown_seconds, already_ran)
            result = run_process(argv[engine], logfile, args.timeout, env,
                                 sample_cuda=args.backend == "ggml_cuda", sample_interval=args.sample_interval)
            already_ran = True
            result["cooldown_before_seconds"] = cooled_seconds
            result.update(parse_log(engine, logfile.read_text(errors="replace")))
            if result["max_rss_bytes"] is None:
                result["max_rss_bytes"] = result["windows_process_memory"]["peak_rss_bytes"]
            result["image"] = image_info(paths[engine])
            classify_result(result, engine, args.steps, args.width, args.height)
            pair["engines"][engine] = result
            save()
            print(f"  {result['status']}: {result['wall_seconds']:.3f}s wall; log {logfile}", flush=True)
        if not args.dry_run and args.engine == "both":
            pair["pixel_comparison"] = pixel_comparison(paths["tensorsharp"], paths["sd_cpp"])
            ts, sd = pair["engines"]["tensorsharp"], pair["engines"]["sd_cpp"]
            if ts["status"] == sd["status"] == "passed":
                pair["sd_over_ts_wall_ratio"] = sd["wall_seconds"]/ts["wall_seconds"]
                for phase in ("denoise", "VAE decode", "text and vision encode", "total"):
                    a, b = ts["phases_seconds"].get(phase), sd["phases_seconds"].get(phase)
                    if a and b:
                        pair.setdefault("sd_over_ts_phase_ratio", {})[phase] = b/a
            save()
    print(f"Report: {report_path}")
    return int(any(e.get("status") == "failed" for r in manifest["runs"] for e in r["engines"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
