#!/usr/bin/env python3
"""Run validation under a private MPS client CUDA allocation limit (Linux).

This is NOT a whole-board VRAM or RAM cap. A connected canary must allocate
below the limit and receive CUDA_ERROR_OUT_OF_MEMORY above it before the model
starts. No GPU modes or existing daemons are changed. Unavailable returns 77.
See https://docs.nvidia.com/deploy/mps/appendix-environment-variables.html .
"""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def canary(limit_mib, output):
    driver = c.CDLL("libcuda.so.1")
    ctx = c.c_void_p()
    ptr = c.c_uint64()
    result = {"pid": os.getpid(), "limit_mib": limit_mib}
    def check(code):
        if code: raise RuntimeError(f"CUDA driver error {code}")
    try:
        check(driver.cuInit(0))
        check(driver.cuCtxCreate_v2(c.byref(ctx), 0, 0))
        check(driver.cuMemAlloc_v2(c.byref(ptr), c.c_size_t(1 << 20)))
        result["small_allocation"] = True
        output.with_suffix(".ready.json").write_text(json.dumps(result))
        deadline = time.monotonic() + 30
        while not output.with_suffix(".go").exists():
            if time.monotonic() >= deadline: raise TimeoutError("No connection confirmation")
            time.sleep(0.05)
        large = c.c_uint64()
        result["over_limit_code"] = driver.cuMemAlloc_v2(c.byref(large), c.c_size_t((limit_mib + 16) << 20))
        if result["over_limit_code"] == 0: check(driver.cuMemFree_v2(large))
        result["passed"] = result["over_limit_code"] == 2
    except Exception as error:
        result.update(passed=False, error=repr(error))
    finally:
        if ptr.value: result["free_code"] = driver.cuMemFree_v2(ptr)
        if ctx.value: result["destroy_code"] = driver.cuCtxDestroy_v2(ctx)
        result["passed"] = result.get("passed", False) and result.get("free_code", 0) == 0 and result.get("destroy_code", 0) == 0
        output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--limit-mib", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu_uuid) or args.limit_mib < 64 or args.timeout <= 0:
        parser.error("Use an exact GPU UUID, a limit of at least 64 MiB and positive timeout")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command: parser.error("A validation command is required")
    output = args.output.resolve()
    root = Path(__file__).resolve().parents[2]
    if not any(output.is_relative_to(root / p) for p in ("artifacts", "docs/validation")):
        parser.error("Evidence must be inside artifacts/ or docs/validation/")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(passed=False, enforcement="unavailable", scope="MPS client CUDA allocations",
                  gpu_uuid=args.gpu_uuid, limit_bytes=args.limit_mib << 20, command=command,
                  limitations=["Not a whole-board VRAM, driver/server overhead or RAM cap.",
                               "Model validation must itself verify outputs and allocation cleanup."])
    control = shutil.which("nvidia-cuda-mps-control")
    private = None
    daemon = False
    child = None
    exit_code = 77
    try:
        if sys.platform != "linux" or control is None: raise RuntimeError("Linux CUDA MPS unavailable")
        result["device"] = subprocess.run(["nvidia-smi", "-i", args.gpu_uuid,
            "--query-gpu=uuid,name,driver_version,memory.total,memory.used", "--format=csv"],
            text=True, capture_output=True, check=True, timeout=15).stdout.strip()
        private = Path(tempfile.mkdtemp(prefix="ts-mps-"))
        (private / "log").mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid,
                   CUDA_MPS_PIPE_DIRECTORY=str(private / "pipe"), CUDA_MPS_LOG_DIRECTORY=str(private / "log"),
                   CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=f"{args.gpu_uuid}={args.limit_mib}M")
        def ctl(command):
            r = subprocess.run([control], input=command + "\n", text=True, capture_output=True, env=env, timeout=15)
            if r.returncode: raise RuntimeError(f"MPS control failed: {r.stdout} {r.stderr}")
            return r.stdout.strip()
        r = subprocess.run([control, "-d"], env=env, text=True, capture_output=True, timeout=15)
        if r.returncode: raise RuntimeError(f"Private MPS start failed: {r.stdout} {r.stderr}")
        daemon = True
        ctl(f"set_default_device_pinned_mem_limit {args.gpu_uuid} {args.limit_mib}M")
        result["configured_limit"] = ctl(f"get_default_device_pinned_mem_limit {args.gpu_uuid}")
        # Fresh name: never consume a previous run's ready/go evidence.
        probe = output / (private.name + "-canary.json")
        with (output / "canary.log").open("w") as log:
            child = subprocess.Popen([sys.executable, __file__, "--canary", str(args.limit_mib), str(probe)],
                                     env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 25
            while not probe.with_suffix(".ready.json").exists():
                if child.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("CUDA canary could not initialize below the limit")
                time.sleep(0.05)
            servers = ctl("get_server_list").split()
            clients = {pid: ctl(f"get_client_list {pid}") for pid in servers if pid.isdigit()}
            result["canary_clients"] = clients
            if not any(str(child.pid) in text.split() for text in clients.values()):
                raise RuntimeError("Canary is not registered with the private MPS server")
            free = subprocess.run(["nvidia-smi", "-i", args.gpu_uuid,
                "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True, capture_output=True, check=True, timeout=15)
            result["canary_board_free_mib"] = int(free.stdout.strip())
            if result["canary_board_free_mib"] < args.limit_mib + 80:
                raise RuntimeError("Insufficient board memory to distinguish quota rejection from physical exhaustion")
            probe.with_suffix(".go").touch()
            child.wait(timeout=15)
        result["canary"] = json.loads(probe.read_text())
        if not result["canary"]["passed"]:
            exit_code = 1
            raise RuntimeError("MPS failed the over-limit allocation rejection canary")
        result["enforcement"] = "canary-verified"
        started = time.monotonic()
        with (output / "process.log").open("w") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            connected = False
            result["device_samples"] = []
            while child.poll() is None:
                if time.monotonic() - started >= args.timeout: raise TimeoutError("Validation timed out")
                if not connected:
                    servers = ctl("get_server_list").split()
                    clients = {pid: ctl(f"get_client_list {pid}") for pid in servers if pid.isdigit()}
                    if any(str(child.pid) in text.split() for text in clients.values()):
                        connected = True
                        result["model_clients"] = clients
                sample = subprocess.run(["nvidia-smi", "-i", args.gpu_uuid,
                    "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, capture_output=True, timeout=15)
                result["device_samples"].append(dict(seconds=time.monotonic() - started,
                    board_used_mib=int(sample.stdout.strip()) if sample.returncode == 0 and sample.stdout.strip().isdigit() else None))
                time.sleep(0.5)
            result["child_exit_code"] = child.wait(timeout=max(0.01, args.timeout - (time.monotonic() - started)))
            result["model_connected"] = connected
        result["wall_seconds"] = time.monotonic() - started
        result["passed"] = result["child_exit_code"] == 0 and connected
        exit_code = 0 if result["passed"] else 1
    except Exception as error:
        result["error"] = repr(error)
        if result["enforcement"] == "canary-verified": exit_code = 1
    finally:
        if child is not None and child.poll() is None:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait(timeout=15)
        if daemon:
            try: result["shutdown"] = ctl("quit")
            except Exception as error:
                result.update(passed=False, shutdown_error=repr(error))
                exit_code = 1
        if private is not None:
            shutil.copytree(private / "log", output / private.name, dirs_exist_ok=True)
            # Only the unique directory created by this invocation is removed.
            if "shutdown_error" not in result: shutil.rmtree(private)
            else: result["retained_private_directory"] = str(private)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result.get(key) for key in
                      ("passed", "enforcement", "scope", "limit_bytes", "child_exit_code", "model_connected", "error")}))
    return exit_code


if __name__ == "__main__":
    if sys.argv[1:2] == ["--canary"]:
        raise SystemExit(canary(int(sys.argv[2]), Path(sys.argv[3])))
    raise SystemExit(main())
