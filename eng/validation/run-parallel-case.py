#!/usr/bin/env python3
"""Run one isolated CLI, AgentTurnBench or server evaluation with GPU telemetry.

Distributed workers must already be started on the peers with identical model
and placement arguments. Pass extra TensorSharp arguments after --.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


RUNTIME_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def runtime_file_identity(binary):
    """Snapshot executable bytes and identity; even identical-content rewrites invalidate a run."""
    snapshot = {"directory": str(binary), "captured_unix": time.time(), "files": {}, "errors": []}
    try:
        # Include dependency/satellite DLLs as well as TensorSharp assemblies.
        paths = sorted(set(binary.rglob("*.dll")) | {binary / "libGgmlOps.so"})
        if not any(path.suffix == ".dll" for path in paths):
            snapshot["errors"].append("Runtime contains no DLLs")
    except OSError as error:
        snapshot["errors"].append("Cannot enumerate runtime: " + repr(error))
        return snapshot
    for path in paths:
        name = path.relative_to(binary).as_posix()
        record = {}
        snapshot["files"][name] = record
        try:
            def metadata(value):
                return {field: getattr(value, field) for field in RUNTIME_STAT_FIELDS}
            before = metadata(path.stat())
            link_before = metadata(path.lstat())
            record.update(before)
            record["path_identity"] = link_before
            digest = hashlib.sha256()
            with path.open("rb") as source:
                opened = metadata(os.fstat(source.fileno()))
                for data in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(data)
                read_end = metadata(os.fstat(source.fileno()))
            record["sha256"] = digest.hexdigest()
            after, link_after = metadata(path.stat()), metadata(path.lstat())
            if not before == opened == read_end == after or link_before != link_after:
                record["capture_mutation"] = {"opened": opened, "read_end": read_end,
                                              "after": after, "path_after": link_after}
                snapshot["errors"].append("Runtime changed while hashing: " + name)
        except OSError as error:
            record["error"] = repr(error)
            snapshot["errors"].append("Cannot inspect runtime file " + name + ": " + repr(error))
    return snapshot


def check_runtime_file_identity(report, binary):
    """Fail independently of request results, retaining their status and any exception."""
    before = report.get("runtime_file_identity_launch")
    if before is None:
        report["runtime_file_identity_check"] = {"status": "not_started", "failures": [],
                                                 "original_run_status": report.get("status")}
        return
    after = report["runtime_file_identity_after_exit"] = runtime_file_identity(binary)
    failures = list(before["errors"]) + list(after["errors"])
    first, last = before["files"], after["files"]
    changes = []
    for name in sorted(set(first) | set(last)):
        if first.get(name) != last.get(name):
            fields = sorted(key for key in set(first.get(name, {})) | set(last.get(name, {}))
                            if first.get(name, {}).get(key) != last.get(name, {}).get(key))
            changes.append({"path": name, "changed_fields": fields,
                            "change": "added" if name not in first else "removed" if name not in last else "mutated"})
            failures.append("Runtime file " + name + " changed during execution (" + ", ".join(fields) + ")")
    report["runtime_file_identity_check"] = {"status": "failed" if failures else "passed",
        "original_run_status": report.get("status"), "failures": failures, "changes": changes,
        "scope": "Native library and all runtime DLLs; SHA256, device/inode/size/mtime/ctime and path identity. atime excluded."}
    if failures:
        report["status"] = "failed"


def launch_model(command, repo, env, log, output, report, runtime_directory=None):
    """Persist actual child launch evidence before waiting for model readiness."""
    if runtime_directory is not None:
        snapshot = report["runtime_file_identity_launch"] = runtime_file_identity(runtime_directory)
        errors = list(snapshot["errors"])
        expected = {"libGgmlOps.so": report["native_sha256"], **report["managed_sha256"]}
        for name, digest in expected.items():
            if snapshot["files"].get(name, {}).get("sha256") != digest:
                errors.append("Runtime differs from preflight hash: " + name)
        if errors:
            snapshot["errors"] = errors
            raise RuntimeError("Cannot launch with unstable runtime identity: " + "; ".join(errors))
    report["process_started_unix"] = time.time()
    process = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    report["process_pid"] = process.pid
    try:
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    except Exception:
        stop(process)
        raise
    return process


def stop(process, timeout=10):
    if process is None:
        return {"started": False}
    if process.poll() is not None:
        return {"started": True, "requested": False, "forced": False, "exit_code": process.returncode}
    if sys.platform == "win32":
        # Windows terminate is TerminateProcess, not a graceful SIGTERM. Reap
        # the owned child and retain that distinction in validation evidence.
        process.terminate()
        process.wait(timeout=timeout)
        return {"started": True, "requested": True, "forced": True,
                "exit_code": process.returncode, "method": "TerminateProcess"}
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return {"started": True, "requested": False, "forced": False, "exit_code": process.returncode}
    forced = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        forced = True
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    return {"started": True, "requested": True, "forced": forced, "exit_code": process.returncode}


def shutdown_failures(shutdown, log, allow_sigterm=False):
    """A successful HTTP response cannot hide a crash while disposing the model."""
    failures = []
    if shutdown.get("started"):
        # A worker may have no SIGTERM handler; that is an intentional stop.
        expected = (0, -signal.SIGTERM) if allow_sigterm and shutdown.get("requested") else (0,)
        if shutdown.get("forced") or shutdown.get("exit_code") not in expected:
            failures.append("Model process did not shut down cleanly: " + json.dumps(shutdown))
    for line in log.splitlines():
        if (("GGML_ASSERT(" in line and "failed" in line) or "GGML_ABORT" in line
                or "Unhandled exception." in line):
            failures.append(line)
    return failures


def load_timings(log):
    """Use emitted loader durations; liveness is intentionally a separate metric."""
    samples = []
    for line in log.splitlines():
        patterns = ((r"Loaded model .*?elapsedMs=([0-9.]+)", .001, "cli_model_load"),
                    (r"Loaded model .*? in ([0-9.]+) ms", .001, "server_model_load"),
                    (r"\[agent-turn-bench\] loaded .*? in ([0-9.]+)s;", 1., "agent_turn_load_and_kernel_warmup"))
        for pattern, scale, source in patterns:
            match = re.search(pattern, line)
            if match:
                samples.append({"source": source, "seconds": float(match[1]) * scale, "log_line": line})
    return samples


def read_cgroup_memory(base=Path("/sys/fs/cgroup")):
    """Support both unified v2 and the memory-controller v1 container mounts."""
    candidates = ((base, 2, "memory.current", "memory.max", "anon", "file"),
                  (base / "memory", 1, "memory.usage_in_bytes", "memory.limit_in_bytes", "total_rss", "total_cache"),
                  (base, 1, "memory.usage_in_bytes", "memory.limit_in_bytes", "total_rss", "total_cache"))
    for folder, version, current, limit, anon, cache in candidates:
        if not (folder / current).exists():
            continue
        values = dict(line.split() for line in (folder / "memory.stat").read_text().splitlines())
        maximum = (folder / limit).read_text().strip()
        return {"version": version, "path": str(folder),
                "current_bytes": int((folder / current).read_text()),
                "limit_bytes": None if maximum == "max" else int(maximum),
                "anon_bytes": int(values.get(anon, values.get("rss", "0"))),
                "file_bytes": int(values.get(cache, values.get("cache", "0")))}
    return None


def companion_paths(arguments, directory=None):
    paths = {}
    for index, argument in enumerate(arguments):
        flag, equals, value = argument.partition("=")
        if flag not in ("--draft-model", "--mmproj"):
            continue
        if not equals:
            if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
                raise ValueError(f"Missing companion path after {flag}")
            value = arguments[index + 1]
        if value.lower() != "none":
            path = Path(value)
            if not path.is_absolute() and directory is not None:
                path = directory / path
            paths[flag.removeprefix("--").replace("-", "_")] = path.resolve()
    return paths


def source_identity(repo):
    """Bind tracked and new source files; generated/ignored build outputs are excluded."""
    native = "TensorSharp.GGML.Native"
    managed = ("TensorSharp.Runtime", "TensorSharp.Models", "TensorSharp.Backends.GGML",
               "TensorSharp.Distributed", "TensorSharp.Cli", "TensorSharp.Chat", "TensorSharp.Server",
               "TensorSharp.Server.Host", "benchmarks/AgentTurnBench")
    build = ("Directory.Build.props", "Directory.Build.targets", "Directory.Packages.props", "global.json")
    is_checkout = (repo / ".git").exists()
    if is_checkout:
        names = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others",
            "--exclude-standard", "--", native, *managed, *build]).decode().split("\0")
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    else:
        names = list(build)
        excluded = {"bin", "obj", "build", "node_modules", ".git", "__pycache__", "artifacts", "logs", "uploads", "prefix-cache"}
        for folder in (native, *managed):
            for directory, children, files in os.walk(repo / folder):
                children[:] = [name for name in children if name not in excluded and not name.startswith("build-")]
                names.extend((Path(directory) / name).relative_to(repo).as_posix() for name in files)
        revision = None
    suffixes = {".cs", ".csproj", ".props", ".targets", ".cpp", ".c", ".h", ".hpp", ".cu", ".cuh", ".cmake", ".sh", ".m", ".mm", ".metal"}
    groups = {"native_sources": {}, "managed_sources": {}, "shared_build_inputs": {}}
    for name in sorted(set(names) - {""}):
        path = repo / name
        if not path.is_file() or (path.suffix not in suffixes and path.name not in ("CMakeLists.txt", "global.json")):
            continue
        group = "native_sources" if name.startswith(native + "/") else "shared_build_inputs" if name in build else "managed_sources"
        groups[group][name] = sha(path)
    return {"revision": revision, "source_kind": "git_checkout" if is_checkout else "tar_snapshot",
            "declared_base_revision": os.environ.get("TENSORSHARP_SOURCE_REVISION"),
            "groups": {name: {"sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                              "files": files} for name, files in groups.items()},
            "scope": "Launch-time source fingerprints, including untracked source; binary hashes separately identify what was executed."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--dotnet", default="/workspace/dotnet/dotnet")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-verification-report", type=Path, help="Completed full-shard download verification report")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server", action="store_true")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--agent-turn", action="store_true", help="Run the scheduler benchmark; extra arguments select scenarios and drafter")
    p.add_argument("--port", type=int, default=5100)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--startup-timeout", type=int, default=180)
    p.add_argument("--shutdown-timeout", type=int, default=60)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--warmup-prefill", type=int, default=512)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--eval-max-tokens", type=int, default=48)
    p.add_argument("--eval-stream", action="store_true", help="Stream text probes to measure HTTP TTFT")
    p.add_argument("--tool-calls", help="Comma-separated structured tool scenarios, e.g. weather,string_payload")
    p.add_argument("--tool-thinking", choices=("off", "on", "off,on"), default="off,on")
    p.add_argument("--tool-max-tokens", type=int, default=2048)
    p.add_argument("--reasoning-effort", choices=("low", "medium", "high"), help="Optional effort sent to text and media HTTP requests")
    p.add_argument("--eval-cases", help="Comma-separated server probe names; omitted runs all eight")
    p.add_argument("--media-fixtures", type=Path, help="Run the existing image/video evaluator after text, in the same server")
    p.add_argument("--media-projector", type=Path, help="Projector passed to the server's --mmproj and hashed in evidence")
    p.add_argument("--media-scenarios", default="image_ocr,multi_image,image_follow_up")
    p.add_argument("--media-max-tokens", type=int, default=256)
    p.add_argument("--dump-logits", action="store_true", help="Dump the first non-warmup forward's F32 logits")
    p.add_argument("--initial-kv-tokens", type=int, help="Explicit initial GPU KV capacity, for growth validation")
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if sum((a.server, a.worker, a.agent_turn)) > 1:
        p.error("--server, --worker and --agent-turn are mutually exclusive")
    if (a.tool_calls or a.eval_stream) and not a.server:
        p.error("--tool-calls and --eval-stream require --server")
    if bool(a.media_fixtures) != bool(a.media_projector) or (a.media_fixtures and not a.server):
        p.error("media-fixtures and media-projector must be supplied together with --server")
    a.repo = a.repo.resolve()
    a.output = a.output.resolve()
    if a.baseline:
        a.baseline = a.baseline.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    extra = a.args[1:] if a.args[:1] == ["--"] else a.args
    if a.media_projector and any(arg.split('=', 1)[0] == '--mmproj' for arg in extra):
        p.error("--media-projector supplies --mmproj; do not specify both")
    project = "AgentTurnBench" if a.agent_turn else "TensorSharp.Server.Host" if a.server else "TensorSharp.Cli"
    binary = (a.repo / "benchmarks/AgentTurnBench/bin/Release/net10.0") if a.agent_turn else a.repo / project / "bin"
    command = [a.dotnet, str(binary / (project + ".dll")),
               "--model", str(a.model), "--backend", "ggml_cuda"]
    command += ["--out", str(a.output / "rows.json")] if a.agent_turn else ["--host", "127.0.0.1", "--port", str(a.port)] if a.server else [] if a.worker else [
        "--benchmark", "--bench-prefill", "512", "--bench-decode", "128", "--bench-runs", "3"]
    command += extra
    if a.media_projector:
        a.media_projector = a.media_projector.resolve()
        a.media_fixtures = a.media_fixtures.resolve()
        command += ['--mmproj', str(a.media_projector)]
    env = dict(os.environ, DOTNET_ROOT=str(Path(a.dotnet).parent), TENSORSHARP_GGML_NO_UPDATE="1",
               MAX_CONTEXT=str(a.context), TS_PREFILL_WARMUP_LEN=str(a.warmup_prefill),
               OMP_NUM_THREADS=str(a.cpu_threads), OPENBLAS_NUM_THREADS=str(a.cpu_threads))
    if a.dump_logits:
        env["TS_DUMP_LOGITS"] = str((a.output / "prefill-logits.bin").resolve())
    if a.initial_kv_tokens is not None:
        if a.initial_kv_tokens < 1:
            p.error("--initial-kv-tokens must be positive")
        env["TS_KV_INITIAL_TOKENS"] = str(a.initial_kv_tokens)
    upstream = a.repo / "ExternalProjects/ggml"
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=all"], text=True)
    if dirty:
        raise RuntimeError("Upstream ggml source must be unchanged: " + dirty)
    report = {"status": "running", "command": command, "started_unix": time.time(),
              "harness_sha256": sha(Path(__file__)),
              "native_sha256": sha(binary / "libGgmlOps.so"),
              "ggml_revision": subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
              "ggml_clean": True}
    source = source_identity(a.repo)
    source_path = a.output / "source-identity.json"
    source_path.write_text(json.dumps(source, indent=2) + "\n")
    report["source_identity"] = {"path": str(source_path), "sha256": sha(source_path),
                                 "revision": source["revision"],
                                 "source_kind": source["source_kind"],
                                 "declared_base_revision": source["declared_base_revision"],
                                 "groups": {name: value["sha256"] for name, value in source["groups"].items()}}
    report["limitations"] = ["Process-to-health timing measures liveness, not model readiness.",
        "Load timing is from the model loader's own log; absent samples are not measured loads.",
        "AgentTurnBench's emitted load time includes kernel warmup; CLI/server model-load timings exclude kernel warmup.",
        "No storage/page-cache eviction is performed; repeated loads are warm-cache measurements.",
        "Hashing model files at launch reads and may warm storage cache before timing."]
    report["coverage"] = {"text_http": "requested" if a.server else "not_requested",
                          "image_http": "requested" if a.media_fixtures else "not_requested",
                          "structured_tools": "requested" if a.tool_calls else "not_requested",
                          "learned_speculation": "requires separately validated nonzero drafter counters and exact token parity"}
    if a.model_verification_report:
        verification = json.loads(a.model_verification_report.read_text())
        if verification.get("status") != "verified":
            raise ValueError("Model shard verification is incomplete")
        if str(a.model.resolve()) not in {str(Path(s["path"]).resolve()) for s in verification["shards"]}:
            raise ValueError("Selected model is absent from shard verification")
        for shard in verification["shards"]:
            if Path(shard["path"]).stat().st_size != shard["bytes"]:
                raise ValueError("Verified model shard size changed: " + shard["path"])
        report["model_verification_report"] = str(a.model_verification_report)
        report["model_verification_report_sha256"] = sha(a.model_verification_report)
        report["model_shards"] = verification["shards"]
        report["model_sha256"] = next(s["sha256"] for s in verification["shards"]
            if Path(s["path"]).resolve() == a.model.resolve())
        report["model_hash_source"] = "completed full-shard verification; file sizes rechecked"
    else:
        report["model_sha256"] = sha(a.model)
        report["model_hash_source"] = "full hash at launch"
    report["managed_sha256"] = {name: sha(binary / name) for name in
        (project + ".dll", "TensorSharp.Models.dll", "TensorSharp.Runtime.dll", "TensorSharp.Backends.GGML.dll", "TensorSharp.Distributed.dll")}
    for name in ("TensorSharp.Chat.dll", "TensorSharp.Server.dll"):
        if (binary / name).is_file():
            report["managed_sha256"][name] = sha(binary / name)
    if a.media_projector:
        report['media_projector'] = {'path': str(a.media_projector), 'sha256': sha(a.media_projector)}
    report["companions"] = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
                            for name, path in companion_paths(command, a.repo).items()}
    report["environment"] = {k: env[k] for k in ("MAX_CONTEXT", "TS_PREFILL_WARMUP_LEN", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                               "CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "TS_DUMP_LOGITS", "TS_KV_INITIAL_TOKENS",
                                               "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE",
                                               "TENSORSHARP_TP_DEVICES", "TENSORSHARP_LAYER_SPLIT_DEVICES",
                                               "TENSORSHARP_TP_NODE_ID", "TENSORSHARP_TP_PEERS",
                                               "TS_GLM_NATIVE", "TS_GLM_TP_SHARD",
                                               "TS_DSV41_TP_HOST_TOKENS", "TS_DSV4_UBATCH", "TS_DSV4_THREADS",
                                               "TS_DSV4_VRAM_RESERVE_MB", "TS_DSV4_LOAD_THREADS", "TS_DSV4_LOAD_CHUNK_MB",
                                               "TS_DSV41_ENGRAM_DEVICE", "TS_DSV41_ENGRAM_WARM",
                                               "TS_DSV41_ENGRAM_THREADS", "TS_DSV41_ENGRAM_RANDOM",
                                               "TS_SCHED_MAX_BATCHED_TOKENS", "TS_SCHED_SOLO_PREFILL_CHUNK", "TS_SCHED_PREFIX_CACHE",
                                               "TS_CPU_MOE_THREADS", "TS_HOST_MOE_PIN", "TS_N_CPU_MOE", "TS_CPU_MOE",
                                               "TS_SPEC", "TS_SPEC_TYPE", "TS_SPEC_DRAFT", "TS_SPEC_PMIN", "TS_SPEC_DRAFT_MODEL",
                                               "TS_GLM_UBATCH", "TS_GLM_THREADS", "TS_GLM_VRAM_RESERVE_MB",
                                               "TS_GLM_LOAD_THREADS", "TS_GLM_LOAD_CHUNK_MB", "TS_Q4E_LAYER_SPLIT") if k in env}
    for key in ("TS_GGUF_PREFAULT", "TS_GGUF_PREFAULT_THREADS", "TS_GGUF_PREFAULT_RESIDENT",
                "KV_CACHE_DTYPE", "TS_DSV41_COMPACT_RAW_GATHER", "TS_DSV41_SPARSE_FA",
                "TS_GGML_TP_F32_NCCL", "GGML_CUDA_ALLREDUCE", "GGML_CUDA_AR_BF16_THRESHOLD",
                "TS_DSV4_WARM_PREAD", "TS_DSV4_LOAD_DROP_CACHE"):
        if key in env:
            report["environment"][key] = env[key]
    # Preserve command and binary/model identity even if the parent is killed
    # while a large checkpoint is loading. A surviving "running" record is an
    # interrupted attempt, never a completed validation result.
    (a.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    telemetry = process = None
    monitor_stop = threading.Event()
    def memory_monitor():
        with (a.output / "memory.csv").open("w", buffering=1) as log, (a.output / "process.csv").open("w", buffering=1) as proc_log:
            log.write("unix_time,cgroup_current_bytes,anon_bytes,file_bytes\n")
            proc_log.write("unix_time,pid,rss_kib,threads,minor_faults,major_faults,read_bytes,write_bytes,rchar,syscr\n")
            while not monitor_stop.is_set():
                stamp = time.time()
                try:
                    memory = read_cgroup_memory()
                    if memory is not None:
                        report.setdefault("memory_accounting", {key: memory[key] for key in ("version", "path", "limit_bytes")})
                        log.write(f"{stamp},{memory['current_bytes']},{memory['anon_bytes']},{memory['file_bytes']}\n")
                    else:
                        report.setdefault("memory_accounting", {"status": "unavailable"})
                except (OSError, ValueError) as error:
                    report.setdefault("memory_accounting_error", repr(error))
                if process is not None:
                    try:
                        proc = Path('/proc') / str(process.pid)
                        # The executable name in field 2 can contain spaces or
                        # parentheses; fields after its final ')' start at 3.
                        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                        status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines())
                        io = dict(line.split(':', 1) for line in (proc / 'io').read_text().splitlines())
                        row = (stamp, process.pid, status.get('VmRSS', '0').split()[0],
                               status.get('Threads', '0').strip(), stat[7], stat[9],
                               io.get('read_bytes', '0').strip(), io.get('write_bytes', '0').strip(),
                               io.get('rchar', '0').strip(), io.get('syscr', '0').strip())
                        proc_log.write(','.join(map(str, row)) + '\n')
                    except (OSError, IndexError):
                        pass  # The child may exit between the /proc reads.
                monitor_stop.wait(1)
    monitor = threading.Thread(target=memory_monitor, daemon=True)
    monitor.start()
    try:
        with (a.output / "gpu.csv").open("w") as gpu_log, (a.output / "process.log").open("w") as log:
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,name,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem,pstate", "--format=csv", "-lms", "500"],
                stdout=gpu_log, stderr=subprocess.STDOUT, start_new_session=True)
            launched = time.monotonic()
            process = launch_model(command, a.repo, env, log, a.output, report, binary)
            if a.server:
                deadline = time.monotonic() + a.startup_timeout
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Server exited: " + str(process.returncode))
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as response:
                            if response.status == 200:
                                report["process_to_liveness_seconds"] = time.monotonic() - launched
                                (a.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
                                break
                    except Exception:
                        time.sleep(1)
                else:
                    raise TimeoutError("Server did not become ready")
                evaluate = [sys.executable, str(Path(__file__).with_name("parallel-server-eval.py")),
                    "--url", f"http://127.0.0.1:{a.port}", "--model", a.model.name,
                    "--output", str(a.output / "evaluation.json"), "--timeout", str(a.timeout),
                    "--concurrency", str(a.concurrency), "--repeats", str(a.repeats),
                    "--max-tokens", str(a.eval_max_tokens)]
                if a.eval_cases:
                    evaluate += ["--cases", a.eval_cases]
                if a.eval_stream:
                    evaluate += ["--stream"]
                if a.reasoning_effort:
                    evaluate += ["--reasoning-effort", a.reasoning_effort]
                if a.baseline:
                    evaluate += ["--baseline", str(a.baseline)]
                report["evaluation_command"] = evaluate
                report["text_exit_code"] = subprocess.call(evaluate, cwd=a.repo, env=env)
                report["coverage"]["text_http"] = "passed" if report["text_exit_code"] == 0 else "failed"
                report["exit_code"] = report["text_exit_code"]
                if a.tool_calls:
                    tool_command = [sys.executable, str(Path(__file__).with_name("validate-qwen38-tool-calls.py")),
                        "--url", f"http://127.0.0.1:{a.port}", "--output", str(a.output / "tool-calls.json"),
                        "--timeout", str(a.timeout), "--scenarios", a.tool_calls,
                        "--thinking", a.tool_thinking, "--max-tokens", str(a.tool_max_tokens)]
                    if a.reasoning_effort:
                        tool_command += ["--reasoning-effort", a.reasoning_effort]
                    report["tool_command"] = tool_command
                    report["tool_exit_code"] = subprocess.call(tool_command, cwd=a.repo, env=env)
                    report["coverage"]["structured_tools"] = "passed" if report["tool_exit_code"] == 0 else "failed"
                    report["exit_code"] = int(bool(report["exit_code"] or report["tool_exit_code"]))
                if a.media_fixtures:
                    media = [sys.executable, str(a.repo / 'benchmarks/engine_comparison/validate_deepseek41_media.py'),
                        '--url', f'http://127.0.0.1:{a.port}', '--model', a.model.name,
                        '--fixtures', str(a.media_fixtures), '--output', str(a.output / 'media.json'),
                        '--weights-id', report.get('model_verification_report_sha256', report['model_sha256']),
                        '--companion-sha256', report['media_projector']['sha256'],
                        '--profile', ' '.join(extra), '--concurrency', str(a.concurrency),
                        '--scenarios', a.media_scenarios, '--max-tokens', str(a.media_max_tokens), '--blocking']
                    if a.reasoning_effort:
                        media += ['--reasoning-effort', a.reasoning_effort]
                    report['media_command'] = media
                    report['media_exit_code'] = subprocess.call(media, cwd=a.repo, env=env)
                    report["coverage"]["image_http"] = "passed" if report["media_exit_code"] == 0 else "failed"
                    report['exit_code'] = int(bool(report['exit_code'] or report['media_exit_code']))
            else:
                report["exit_code"] = process.wait(timeout=a.timeout)
            report["status"] = "completed" if report["exit_code"] == 0 else "failed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
    finally:
        report["execution_status_before_cleanup"] = report["status"]
        report["shutdown"] = stop(process, a.shutdown_timeout)
        check_runtime_file_identity(report, binary)
        stop(telemetry)
        monitor_stop.set()
        monitor.join(timeout=2)
        process_log = a.output / "process.log"
        log_text = process_log.read_text(errors="replace") if process_log.exists() else ""
        report["model_load_timings"] = load_timings(log_text)
        report["shutdown_failures"] = shutdown_failures(report["shutdown"],
            log_text, allow_sigterm=a.worker)
        if report["shutdown_failures"]:
            report["status"] = "failed"
        report["wall_seconds"] = time.time() - report["started_unix"]
        (a.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
