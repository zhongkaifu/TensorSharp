#!/usr/bin/env python3
"""Own detached benchmark process groups without touching unrelated GPU jobs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import uuid


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def positive_finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def process_stat(pid: int) -> dict | None:
    try:
        # comm may itself contain spaces and parentheses.
        values = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"state": values[0], "pgrp": int(values[2]),
                "session": int(values[3]), "starttime": int(values[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def owns(record: dict) -> bool:
    if record.get("boot_id") != boot_id():
        return False
    pid = int(record["pid"])
    stat = process_stat(pid)
    if stat is None or stat["state"] == "Z":
        return False
    if stat["starttime"] != record["starttime"]:
        return False
    if stat["pgrp"] != pid or stat["session"] != pid:
        return False
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        marker = f"QWEN38_BENCH_RUN_ID={record['run_id']}".encode()
        return marker in environment
    except (FileNotFoundError, PermissionError):
        return False


def group_running(pgid: int) -> bool:
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            stat = process_stat(int(entry.name))
            if stat and stat["pgrp"] == pgid and stat["state"] != "Z":
                return True
    return False


def owns_orphan_group(record: dict) -> bool:
    """A dead launcher may leave workers; every member must retain our marker."""
    if record.get("boot_id") != boot_id():
        return False
    pid = int(record["pid"])
    leader = process_stat(pid)
    if leader and leader["starttime"] != record["starttime"]:
        return False
    found = False
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        stat = process_stat(int(entry.name))
        if not stat or stat["state"] == "Z" or stat["pgrp"] != pid:
            continue
        if stat["session"] != pid or stat["starttime"] < record["starttime"]:
            return False
        try:
            marker = f"QWEN38_BENCH_RUN_ID={record['run_id']}".encode()
            if marker not in (entry / "environ").read_bytes().split(b"\0"):
                return False
        except (FileNotFoundError, PermissionError):
            return False
        found = True
    return found


def revision(path: str | None) -> dict:
    if not path:
        return {"status": "unavailable"}
    try:
        sha = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        diff = subprocess.check_output(["git", "-C", path, "diff", "--binary", "HEAD"],
                                       stderr=subprocess.DEVNULL)
        changed = subprocess.check_output(["git", "-C", path, "diff", "--name-only", "-z", "HEAD"],
                                          stderr=subprocess.DEVNULL).decode().split("\0")
        untracked = subprocess.check_output(
            ["git", "-C", path, "ls-files", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL).decode().split("\0")
        root = Path(path)
        return {"path": str(root.resolve()), "revision": sha, "tracked_dirty": bool(diff),
                "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "tracked_changes": [source_identity(root, name) for name in changed if name],
                "untracked_sources": [source_identity(root, name) for name in untracked
                                      if name and is_source_path(name)]}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"path": path, "status": "unavailable"}


def is_source_path(name: str) -> bool:
    path = Path(name)
    excluded = {".vscode", ".idea", ".vs", "node_modules", "__pycache__", "bin", "obj"}
    if any(part.lower() in excluded for part in path.parts):
        return False
    return (path.suffix.lower() in {".cs", ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp",
                                    ".cmake", ".py", ".sh", ".ps1", ".csproj", ".props", ".targets",
                                    ".sln", ".slnx", ".json", ".yaml", ".yml", ".toml", ".rs"}
            or path.name in {"CMakeLists.txt", "Makefile", "Directory.Build.props", "Directory.Build.targets"})


def source_identity(root: Path, name: str) -> dict:
    path = root / name
    if not path.is_file():
        return {"path": name, "status": "deleted_or_unavailable"}
    identity = file_identity(path)
    return {"path": name, "sha256": identity["sha256"], "size_bytes": identity["size_bytes"]}


def harness_identities(root: str | None) -> list[dict]:
    if not root:
        return []
    names = ["eng/qwen38-engine-bench.py", "eng/qwen38-engine-report.py",
             "eng/qwen38-engine-servers.sh", "eng/qwen38-engine-process.py"]
    return [source_identity(Path(root), name) for name in names]


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def binary_identities(engine: str, argv: list[str], cwd: str, environment: dict) -> list[dict]:
    executable = shutil.which(argv[0], path=environment.get("PATH"))
    paths = [Path(executable)] if executable else []
    if engine == "tensorsharp":
        directory = Path(cwd)
        for pattern in ["TensorSharp*.dll", "cuda_kernels/*.fatbin", "cuda_kernels/*.ptx",
                        "cuda_kernels/*.sha256", "*.so", "*.so.*"]:
            paths.extend(directory.glob(pattern))
        runtime_directory = directory / "runtimes"
        if runtime_directory.is_dir():
            for pattern in ("*.so", "*.so.*"):
                paths.extend(runtime_directory.rglob(pattern))
    elif executable and engine in {"llamacpp", "focus-llama", "ninfer", "infernix", "velox", "ignis"}:
        directory = Path(executable).resolve().parent
        # Native CMake products commonly put the executable in bin/ or apps/
        # and their own shared libraries in sibling lib/, src/, or ggml/ dirs.
        # Inventory this build configuration only; do not scan Python wheels,
        # system CUDA installations, or unrelated build configurations.
        build_directory = directory.parent if directory.name in {"bin", "apps"} else directory
        generic_prefix = build_directory in {Path("/"), Path("/usr"), Path("/usr/local"), Path("/opt"), Path("/root"), Path("/home")}
        for pattern in ("*.so", "*.so.*"):
            if generic_prefix:
                continue
            if directory.name in {"bin", "apps"} and (build_directory / "CMakeCache.txt").is_file():
                paths.extend(build_directory.rglob(pattern))
            else:
                paths.extend(directory.glob(pattern))
                if not generic_prefix:
                    for child in ("lib", "lib64", "runtimes"):
                        paths.extend((directory / child).rglob(pattern))
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    return [file_identity(path) for path in unique]


def package_versions(executable: str | None) -> dict:
    if not executable:
        return {"status": "not_a_python_engine"}
    code = """import importlib.metadata, json, sys
names = ['vllm', 'sglang', 'eole', 'torch', 'triton', 'flashinfer-python', 'flash-attn', 'fla-core', 'causal-conv1d',
         'transformers', 'tokenizers', 'nvidia-cublas-cu13', 'nvidia-cuda-runtime-cu13']
result = {'python': sys.version, 'executable': sys.executable}
for name in names:
    try: result[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: result[name] = None
print(json.dumps(result))
"""
    try:
        output = subprocess.check_output([executable, "-c", code], text=True,
                                         stderr=subprocess.PIPE, timeout=15)
        return json.loads(output)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "reason": str(error)}


def gpu_inventory() -> dict:
    fields = ["index", "uuid", "name", "driver_version", "memory.total", "power.limit", "pci.bus_id"]
    argv = ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    try:
        result = subprocess.check_output(argv, text=True, stderr=subprocess.PIPE, timeout=10)
        rows = csv.reader(io.StringIO(result), skipinitialspace=True)
        return {"query": argv, "devices": [dict(zip(fields, row)) for row in rows]}
    except (OSError, subprocess.SubprocessError) as error:
        return {"query": argv, "status": "unavailable", "reason": str(error)}


def start(args: argparse.Namespace) -> int:
    if not Path("/proc/self/stat").exists():
        raise ValueError("process ownership requires Linux /proc")
    record_path = Path(args.record)
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if owns(previous):
            raise ValueError(f"an owned server is already running (PID {previous['pid']})")
        if process_stat(int(previous["pid"])) is not None:
            raise ValueError("stale PID record refers to a live unowned process; inspect it first")
        record_path.unlink()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv:
        raise ValueError("missing server executable")
    environment = os.environ.copy()
    overrides = dict(item.split("=", 1) for item in args.env)
    environment.update(overrides)
    run_id = uuid.uuid4().hex
    environment["QWEN38_BENCH_RUN_ID"] = run_id
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "ab", buffering=0) as logfile:
        # start_new_session invokes setsid before exec. The recorded PID is the
        # actual server and is also its process-group/session leader.
        child = subprocess.Popen(argv, cwd=args.cwd, env=environment,
                                 stdin=subprocess.DEVNULL, stdout=logfile,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    stat = process_stat(child.pid)
    if stat is None:
        raise ValueError("server exited before its process identity could be recorded")
    record = {"pid": child.pid, "starttime": stat["starttime"], "boot_id": boot_id(),
              "run_id": run_id, "manifest": str(Path(args.manifest).resolve())}
    write_json(record_path, record)
    allowed = set(args.capture_env.split(",")) | set(overrides)
    manifest = {"schema_version": 1, "engine": args.engine, "mode": args.mode,
                "label": args.label, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "context_tokens": args.context, "lanes": args.lanes,
                "readiness_budget": {"timeout_seconds": args.startup_timeout_seconds,
                                     "attempt_limit": args.startup_attempt_limit,
                                     "poll_seconds": args.readiness_poll_seconds},
                "argv": argv, "working_directory": str(Path(args.cwd).resolve()),
                "environment": {key: environment.get(key) for key in sorted(allowed) if key},
                "process": record, "engine_source": revision(args.engine_repo),
                "tensorsharp_source": revision(args.root),
                "ggml_source": revision(str(Path(args.root) / "ExternalProjects/ggml")),
                "harness_source": revision(args.harness_root),
                "harness_files": harness_identities(args.harness_root),
                "host": {"platform": platform.platform(), "machine": platform.machine()},
                "status": "starting", "log": str(Path(args.log).resolve())}
    if args.engine == "tensorsharp":
        manifest["sequence_admission_configuration"] = {
            "declared_lane_setting": args.lanes,
            "configured_running_sequence_limit": environment.get("TS_SCHED_MAX_RUNNING_SEQS"),
            "source": "explicit launcher TS_SCHED_MAX_RUNNING_SEQS override" if "TS_SCHED_MAX_RUNNING_SEQS" in overrides
                      else "inherited environment (not independently verified at the live scheduler)",
            "runtime_admission_cap_verified": False,
            "observed_peak_active_sequences": None,
            "dynamic_device_slot_cap": None,
            "effective_rule": "min(configured running-sequence limit, dynamic device slot capacity)",
        }
    # Persist ownership before enriching; a failed hardware probe must never
    # orphan a server or erase the launch evidence.
    write_json(Path(args.manifest), manifest)
    try:
        manifest["binaries"] = binary_identities(args.engine, argv, args.cwd, environment)
        manifest["configuration_files"] = [file_identity(Path(path)) for path in args.input_file]
        manifest["python_packages"] = package_versions(args.python_executable)
        manifest["gpu"] = gpu_inventory()
        if args.model_manifest_json:
            source = args.model_manifest_json
            manifest["models"] = json.loads(source if source.lstrip().startswith(("{", "["))
                                           else Path(source).read_text())
    except (OSError, ValueError) as error:
        manifest["enrichment_error"] = str(error)
    write_json(Path(args.manifest), manifest)
    print(child.pid)
    return 0


def stop(args: argparse.Namespace) -> int:
    path = Path(args.record)
    if not path.exists():
        return 0
    record = json.loads(path.read_text())
    pid = int(record["pid"])
    if not owns(record) and not owns_orphan_group(record):
        stat = process_stat(pid)
        if stat is not None and stat["state"] != "Z":
            print(f"refusing to signal unowned/reused PID {pid}", file=sys.stderr)
            return 1
        if group_running(pid):
            print(f"leader {pid} exited; refusing an unverifiable process group", file=sys.stderr)
            return 1
        path.unlink()
        return 0
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + args.timeout
    while group_running(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if group_running(pid):
        # Ownership was proven before TERM; a group cannot be reused while its
        # original members still exist.
        os.killpg(pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while group_running(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
    if group_running(pid):
        print(f"owned process group {pid} did not stop", file=sys.stderr)
        return 1
    path.unlink()
    manifest_path = Path(record["manifest"])
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.update(stopped_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(manifest_path, manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("start")
    for name in ["record", "manifest", "engine", "mode", "label", "cwd", "log", "root"]:
        launch.add_argument(f"--{name}", required=True)
    launch.add_argument("--engine-repo")
    launch.add_argument("--harness-root")
    launch.add_argument("--python-executable")
    launch.add_argument("--model-manifest-json")
    launch.add_argument("--input-file", action="append", default=[])
    launch.add_argument("--capture-env", default="")
    launch.add_argument("--env", action="append", default=[])
    launch.add_argument("--context", type=int, required=True)
    launch.add_argument("--lanes", type=int, required=True)
    launch.add_argument("--startup-timeout-seconds", type=int)
    launch.add_argument("--startup-attempt-limit", type=int)
    launch.add_argument("--readiness-poll-seconds", type=positive_finite_float)
    launch.add_argument("argv", nargs=argparse.REMAINDER)
    check = commands.add_parser("check")
    check.add_argument("--record", required=True)
    terminate = commands.add_parser("stop")
    terminate.add_argument("--record", required=True)
    terminate.add_argument("--timeout", type=float, default=30)
    mark = commands.add_parser("mark")
    mark.add_argument("--manifest", required=True)
    mark.add_argument("--status", required=True)
    args = parser.parse_args()
    try:
        if args.command == "start":
            return start(args)
        if args.command == "stop":
            return stop(args)
        if args.command == "mark":
            path = Path(args.manifest)
            value = json.loads(path.read_text())
            value["status"] = args.status
            write_json(path, value)
            return 0
        return 0 if owns(json.loads(Path(args.record).read_text())) else 1
    except (OSError, ValueError, KeyError) as error:
        print(f"qwen38-engine-process: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
