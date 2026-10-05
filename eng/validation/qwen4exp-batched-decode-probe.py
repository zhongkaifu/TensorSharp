#!/usr/bin/env python3
"""Record identities and run the opt-in real Qwen4Exp numerical batch test.

Never builds or changes a dependency. First build native code against unchanged
upstream GGML, then build InferenceWeb.Tests so its native copy is current. This
runner refuses mismatched native copies, skipped/missing tests, stale/incomplete
probe reports, changed artifacts, or unavailable coverage. --capture-only records
provenance without launching any model or claiming a validation pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
TEST = "InferenceWeb.Tests.Qwen4ExpRealBatchedDecodeTests.SerialAndBatchedTeacherForcedLogitsAndSoloContinuation"
BACKENDS = {"cpu": "GgmlCpu", "metal": "GgmlMetal", "cuda": "GgmlCuda", "vulkan": "GgmlVulkan"}
RELEVANT_PROJECTS = ["TensorSharp.Runtime", "TensorSharp.Models", "TensorSharp.Backends.GGML", "InferenceWeb.Tests"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = path.resolve()
    stat = path.stat()
    require(path.is_file(), f"Not a regular file: {path}")
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest_file(path)}


def command(argv, cwd=None, timeout=30):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=timeout, check=True)
    return result.stdout


def git_identity(path):
    path = path.resolve()
    top = Path(command(["git", "-C", str(path), "rev-parse", "--show-toplevel"]).decode().strip()).resolve()
    require(top == path, f"Expected a separate Git checkout at {path}, found parent checkout {top}")
    status = command(["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"])
    diff = command(["git", "-C", str(path), "diff", "HEAD", "--binary"], timeout=60)
    return {"path": str(path), "revision": command(["git", "-C", str(path), "rev-parse", "HEAD"]).decode().strip(),
            "clean": not status, "status_porcelain": status.decode(), "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}


def model_metadata(path):
    path = path.expanduser().resolve()
    match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name)
    shards = [path] if not match else [path.with_name(f"{match[1]}-{index:05d}-of-{int(match[3]):05d}.gguf")
                                    for index in range(1, int(match[3]) + 1)]
    require(path in shards, "Invalid model shard index")
    files = []
    for shard in shards:
        require(shard.is_file(), f"Missing model shard: {shard}")
        stat = shard.stat()
        require(stat.st_size > 4, f"Empty model shard: {shard}")
        # Sample only small fixed windows, regardless of checkpoint size. This
        # is explicitly not a complete weight-file checksum or publisher pin.
        with shard.open("rb") as handle:
            head = handle.read(65536)
            require(head.startswith(b"GGUF"), f"Model shard has no GGUF magic: {shard}")
            tail_offset = max(0, stat.st_size - 65536)
            handle.seek(tail_offset)
            tail = handle.read(65536)
        files.append({"path": str(shard), "name": shard.name, "bytes": stat.st_size,
                      "mtime_ns": stat.st_mtime_ns, "device": stat.st_dev, "inode": stat.st_ino,
                      "sample_windows": [{"offset": 0, "bytes": len(head), "sha256": hashlib.sha256(head).hexdigest()},
                                         {"offset": tail_offset, "bytes": len(tail), "sha256": hashlib.sha256(tail).hexdigest()}]})
    return {"requested_path": str(path), "shards": files, "total_bytes": sum(item["bytes"] for item in files),
            "all_expected_shards_present": True, "full_sha256_computed": False,
            "limitations": "Names, filesystem metadata, GGUF magic, and first/last64 KiB samples are recorded. Samples do not bind all model bytes or establish publisher authenticity; no full multi-hundred-GB hashing is performed."}


def environment_identity(environment):
    result = {}
    for name, value in sorted(environment.items()):
        if name.startswith(("TS_", "GGML_", "TENSORSHARP_")) or name in (
                "MAX_CONTEXT", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD",
                "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "DOTNET_ROOT"):
            secret = re.search(r"(?:SECRET|PASSWORD|COOKIE|API_KEY|AUTH|_TOKEN$)", name, re.I)
            result[name] = "<redacted>" if secret else value
    return result


def device_identity():
    identity = {"system": platform.system(), "machine": platform.machine(), "platform": platform.platform()}
    observations = []
    queries = ([["sysctl", "-n", "hw.model", "hw.memsize", "machdep.cpu.brand_string"],
                ["system_profiler", "SPDisplaysDataType", "-json"]] if platform.system() == "Darwin" else
               [["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"]])
    for argv in queries:
        row = {"argv": argv}
        try:
            raw = command(argv)
            if argv[0] == "system_profiler":
                data = json.loads(raw)
                row["displays"] = [{key: value for key, value in item.items() if key in
                                     ("_name", "spdisplays_chipset_model", "spdisplays_vendor", "spdisplays_vram", "spdisplays_vram_shared", "spdisplays_metal")}
                                    for item in data.get("SPDisplaysDataType", [])]
            else:
                row["stdout"] = raw.decode(errors="replace").strip()
            row["status"] = "observed"
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            row.update(status="unavailable", error=str(error))
        observations.append(row)
    identity["observations"] = observations
    return identity


def source_identity(root):
    paths = []
    for name in RELEVANT_PROJECTS:
        paths.extend(path for path in (root / name).rglob("*.cs") if not {"bin", "obj"}.intersection(path.relative_to(root / name).parts))
    paths.extend(path for path in (root / "TensorSharp.GGML.Native").iterdir()
                 if path.is_file() and (path.suffix in (".cpp", ".h", ".mm", ".m") or path.name == "CMakeLists.txt"))
    return {str(path.relative_to(root)): digest_file(path) for path in sorted(paths)}


def probe_environment_overrides(steps):
    capacity = 128
    while capacity < steps + 64:
        capacity *= 2
    return {"MAX_CONTEXT": str(max(256, capacity)), "TS_KV_INITIAL_TOKENS": str(capacity),
            "TS_Q4E_DISABLE_ARENA_DECODE": None, "speculation": "Cleared by the test's EnvScope"}


def capture(args, environment):
    native = file_identity(args.native)
    copied = file_identity(args.test_dll.parent / args.native.name)
    require(native["sha256"] == copied["sha256"], "Test-adjacent native copy differs from explicit built native; rebuild the test project before using this runner")
    upstream = git_identity(args.upstream)
    require(upstream["clean"], "Upstream GGML checkout is modified; validate against unchanged upstream sources")
    managed = [file_identity(path) for path in sorted(args.test_dll.parent.glob("TensorSharp*.dll"))]
    require(managed, "No TensorSharp managed assemblies adjacent to the test DLL")
    cache_path = args.native.parent / "CMakeCache.txt"
    native_configuration = {"status": "unavailable", "path": str(cache_path)}
    if cache_path.is_file():
        values = {}
        for line in cache_path.read_text(errors="replace").splitlines():
            match = re.match(r"([^/#][^:]*):[^=]+=(.*)", line)
            if match and (match[1].startswith(("GGML_", "TENSORSHARP_", "CMAKE_CXX_", "CMAKE_C_")) or
                          match[1] in ("CMAKE_BUILD_TYPE", "CMAKE_OSX_ARCHITECTURES", "CMAKE_OSX_DEPLOYMENT_TARGET")):
                values[match[1]] = match[2]
        native_configuration = {"status": "observed", "file": file_identity(cache_path), "selected_cache_values": values}
    return {"captured_utc": datetime.now(timezone.utc).isoformat(), "repository": git_identity(ROOT),
            "upstream_ggml": upstream, "model": model_metadata(args.model), "native_build": native,
            "test_native_copy": copied, "test_assembly": file_identity(args.test_dll), "managed_assemblies": managed,
            "native_build_configuration": native_configuration,
            "source_sha256": source_identity(ROOT), "runner": file_identity(Path(__file__)),
            "device": device_identity(), "requested_backend": args.backend, "process_environment": environment_identity(environment),
            "probe_in_test_environment_overrides": probe_environment_overrides(args.steps),
            "limitations": "Hashes bind explicit files before invocation. The test report must identify its actual mapped native library, which is checked afterward. This runner uses an existing compiled test DLL and does not infer that it was compiled from today's source files; rebuild when source changes. Device inventory is separate from the effective backend reported by the completed model test."}


def verify_trx(path):
    tree = ET.parse(path)
    results = [node for node in tree.iter() if node.tag.rsplit("}", 1)[-1] == "UnitTestResult"]
    require(len(results) == 1, f"Expected exactly one selected real test result, found {len(results)}")
    result = results[0]
    require(result.get("testName") == TEST, f"Unexpected selected test: {result.get('testName')}")
    require(result.get("outcome") == "Passed", f"Real model test did not pass: {result.get('outcome')}")
    counters = next((node for node in tree.iter() if node.tag.rsplit("}", 1)[-1] == "Counters"), None)
    require(counters is not None and counters.get("total") == counters.get("executed") == counters.get("passed") == "1",
            "TRX does not report exactly1 executed passed test (skipped/unavailable coverage cannot pass)")
    return {"result_count": 1, "test_name": result.get("testName"), "outcome": result.get("outcome"),
            "counters": dict(counters.attrib), "sha256": digest_file(path)}


def verify_probe(report, args, native_hash):
    require(report.get("completed") is True and not report.get("failure"), "Probe report is incomplete or records a failure")
    require(Path(report.get("model", "")).resolve() == args.model.resolve(), "Probe model path differs from requested model")
    require(report.get("backend") == BACKENDS[args.backend], "Effective probe backend differs from requested backend")
    require(report.get("steps_per_repetition") == args.steps and report.get("repetitions") == 2, "Probe step/repetition coverage differs")
    require(report.get("round_robin_iteration_order") == "step-then-row" and
            report.get("round_robin_cache_bind_in_timing") is True,
            "Probe omitted the interleaved round-robin cache-binding timing contract")
    native_path = report.get("native_library_path")
    require(isinstance(native_path, str) and Path(native_path).is_file(), "Probe omitted its actual mapped native library path")
    mapped_native = file_identity(Path(native_path))
    require(mapped_native["sha256"] == native_hash, "Actually mapped native library differs from explicit built native")
    rows = report.get("reports", [])
    require(len(rows) == 6 and {(row.get("width"), row.get("repetition")) for row in rows} ==
            {(width, repetition) for width in (2, 3, 4) for repetition in (0, 1)}, "Incomplete or duplicate width/repetition coverage")
    measurements = []
    for row in rows:
        require(row.get("steps") == args.steps and row.get("tokens") == row["width"] * args.steps, "Unexpected timed token coverage")
        for field, bound in (("max_logit_error", .1), ("max_kl", 1e-4), ("solo_continuation_error", .1),
                             ("round_robin_max_logit_error", .1), ("round_robin_max_kl", 1e-4)):
            value = row.get(field)
            require(isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value < bound,
                    f"Numerical {field} gate failed: {value}")
        serial, round_robin, batched = (row.get(name + "_decode_seconds") for name in ("serial", "round_robin", "batched"))
        require(all(isinstance(value, (float, int)) and math.isfinite(value) and value > 0
                    for value in (serial, round_robin, batched)), "Invalid timing evidence")
        for name, seconds in (("serial", serial), ("round_robin", round_robin), ("batched", batched)):
            tps = row.get(name + "_tokens_per_second")
            require(isinstance(tps, (float, int)) and math.isfinite(tps) and
                    math.isclose(tps, row["tokens"] / seconds, rel_tol=1e-10),
                    f"Inconsistent {name} token throughput evidence")
        measurements.append({"width": row["width"], "repetition": row["repetition"], "decode_wall_speedup": serial / batched,
                             "round_robin_decode_wall_speedup": round_robin / batched,
                             "greedy_differences": row.get("greedy_differences"),
                             "round_robin_greedy_differences": row.get("round_robin_greedy_differences")})
    trace = report.get("trace", [])
    expected = {(width, repetition, step, row) for width in (2, 3, 4) for repetition in (0, 1)
                for step in range(args.steps) for row in range(width)}
    batched_trace = [row for row in trace if row.get("phase") == "batched-decode"]
    actual = {(row.get("width"), row.get("repetition"), row.get("step"), row.get("row")) for row in batched_trace}
    require(actual == expected and len(batched_trace) == len(expected), "Missing or duplicate per-row/per-step decode trace")
    round_robin_trace = [row for row in trace if row.get("phase") == "round-robin-decode"]
    round_robin_expected_order = [(width, repetition, step, row) for width in (2, 3, 4) for repetition in (0, 1)
                                for step in range(args.steps) for row in range(width)]
    round_robin_actual_order = [(row.get("width"), row.get("repetition"), row.get("step"), row.get("row"))
                               for row in round_robin_trace]
    require(round_robin_actual_order == round_robin_expected_order,
            "Missing, duplicate, or non-interleaved round-robin per-row/per-step trace")
    phases = ("independent-prefill", "serial-prefill-control", "solo-continuation", "round-robin-prefill",
              "batched-decode", "round-robin-decode")
    require(all(row.get("phase") in phases for row in trace), "Unexpected probe trace phase")
    for phase in ("independent-prefill", "serial-prefill-control", "solo-continuation", "round-robin-prefill"):
        controls = [row for row in trace if row.get("phase") == phase]
        expected_controls = {(width, repetition) for width in (2, 3, 4) for repetition in (0, 1)} if phase == "serial-prefill-control" else {
            (width, repetition, row) for width in (2, 3, 4) for repetition in (0, 1) for row in range(width)}
        observed_controls = {(row.get("width"), row.get("repetition")) if phase == "serial-prefill-control" else
                             (row.get("width"), row.get("repetition"), row.get("row")) for row in controls}
        require(len(controls) == len(expected_controls) and observed_controls == expected_controls,
                f"Missing or duplicate {phase} controls/continuations")
    for row in trace:
        error = row.get("max_logit_error")
        require(isinstance(error, (float, int)) and math.isfinite(error) and 0 <= error < .1, "Per-row numerical logit gate failed")
        if row.get("phase") != "solo-continuation":
            kl = row.get("kl")
            require(isinstance(kl, (float, int)) and math.isfinite(kl) and 0 <= kl < 1e-4, "Per-row numerical KL gate failed")
        if row.get("phase") in ("batched-decode", "round-robin-decode"):
            margin = row.get("greedy_margin")
            require(isinstance(margin, (float, int)) and math.isfinite(margin) and margin >= 0, "Invalid greedy margin")
            actual_argmax = row.get("batched_argmax") if row["phase"] == "batched-decode" else row.get("round_robin_argmax")
            require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                        for value in (row.get("serial_argmax"), actual_argmax)), "Missing or invalid greedy token IDs")
            if margin > 2 * error:
                require(row.get("serial_argmax") == actual_argmax, "Stable-margin greedy token parity failed")
    for summary in rows:
        for phase, error_field, kl_field, greedy_field, argmax_field in (
                ("batched-decode", "max_logit_error", "max_kl", "greedy_differences", "batched_argmax"),
                ("round-robin-decode", "round_robin_max_logit_error", "round_robin_max_kl", "round_robin_greedy_differences", "round_robin_argmax")):
            selected = [row for row in trace if row.get("phase") == phase and
                        row.get("width") == summary["width"] and row.get("repetition") == summary["repetition"]]
            require(math.isclose(summary[error_field], max(row["max_logit_error"] for row in selected), rel_tol=1e-10, abs_tol=1e-12) and
                    math.isclose(summary[kl_field], max(row["kl"] for row in selected), rel_tol=1e-10, abs_tol=1e-12),
                    f"Inconsistent {phase} numerical summary")
            require(summary.get(greedy_field) == sum(row.get("serial_argmax") != row.get(argmax_field) for row in selected),
                    f"Inconsistent {phase} greedy-difference summary")
    if args.min_decode_speedup is not None:
        require(all(row["decode_wall_speedup"] >= args.min_decode_speedup for row in measurements), "Measured decode speedup failed the requested performance gate")
    if args.min_round_robin_speedup is not None:
        require(all(row["round_robin_decode_wall_speedup"] >= args.min_round_robin_speedup for row in measurements),
                "Measured round-robin decode speedup failed the requested performance gate")
    return {"status": "numerical_passed", "mapped_native": mapped_native, "measurements": measurements,
            "performance_status": "passed_requested_gate" if args.min_decode_speedup is not None else "measured_not_gated",
            "round_robin_performance_status": "passed_requested_gate" if args.min_round_robin_speedup is not None else "measured_not_gated",
            "requested_performance_gates": {"minimum_independent_serial_speedup": args.min_decode_speedup,
                                            "minimum_round_robin_speedup": args.min_round_robin_speedup},
            "trace_phase_counts": {phase: sum(row["phase"] == phase for row in trace) for phase in phases},
            "limitations": "Teacher-forced short-context logits/KL parity and solo continuation only. Independent serial completes each sequence in turn; round-robin interleaves one token per sequence per step and includes each cache bind. These baselines have different live-state/cache working sets and must be reported separately. Close-margin greedy differences are recorded rather than forbidden. Timed Forward/batch calls include graph capture, allocation, state join/scatter, and dispatch; the probe performs no excluded decode warmup. No exact autoregressive text parity, unrestricted quality, vision, HTTP scheduler overhead, prefill throughput, long-context performance, or loading-time claim."}


def invoke(argv, environment, log, timeout):
    started = time.perf_counter()
    with log.open("wb") as handle:
        process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            exit_code = process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name != "nt": os.killpg(process.pid, signal.SIGTERM)
            else: process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name != "nt": os.killpg(process.pid, signal.SIGKILL)
                else: process.kill()
                process.wait()
            exit_code = process.returncode
    return {"argv": argv, "cwd": str(ROOT), "exit_code": exit_code, "timed_out": timed_out,
            "elapsed_seconds": time.perf_counter() - started, "log": file_identity(log)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="metal")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--test-dll", type=Path, default=ROOT / "InferenceWeb.Tests/bin/Release/net10.0/InferenceWeb.Tests.dll")
    native_name = "GgmlOps.dll" if platform.system() == "Windows" else "libGgmlOps.dylib" if platform.system() == "Darwin" else "libGgmlOps.so"
    parser.add_argument("--native", type=Path, default=ROOT / "TensorSharp.GGML.Native/build" / native_name)
    parser.add_argument("--upstream", type=Path, default=ROOT / "ExternalProjects/ggml")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh directory under ignored artifacts/ or docs/validation/")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--min-decode-speedup", type=float, help="Gate every repetition's independent-serial / batched decode wall ratio")
    parser.add_argument("--min-round-robin-speedup", type=float, help="Gate every repetition's interleaved round-robin / batched decode wall ratio")
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args(argv)
    args.model, args.test_dll, args.native, args.upstream, args.output_dir = [path.expanduser().resolve() for path in
        (args.model, args.test_dll, args.native, args.upstream, args.output_dir)]
    require(any(args.output_dir.is_relative_to((ROOT / directory).resolve()) for directory in ("artifacts", "docs/validation")), "Output must remain in ignored artifacts/ or docs/validation/")
    require(not args.output_dir.exists(), "Use a fresh output directory; stale evidence cannot satisfy a new run")
    require(8 <= args.steps <= 128, "Use8..128 steps (probe max context is256)")
    require(math.isfinite(args.timeout) and args.timeout > 0, "Timeout must be finite and positive")
    require(args.min_decode_speedup is None or math.isfinite(args.min_decode_speedup) and args.min_decode_speedup > 0, "Speedup gate must be finite and positive")
    require(args.min_round_robin_speedup is None or math.isfinite(args.min_round_robin_speedup) and args.min_round_robin_speedup > 0,
            "Round-robin speedup gate must be finite and positive")
    require(not args.capture_only or args.min_decode_speedup is None and args.min_round_robin_speedup is None,
            "Capture-only cannot apply a performance gate")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True)
    execution = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "failed", "failures": [],
                 "test_executed": False, "capture_only": args.capture_only,
                 "requested_performance_gates": {"minimum_independent_serial_speedup": args.min_decode_speedup,
                                                 "minimum_round_robin_speedup": args.min_round_robin_speedup}}
    environment = dict(os.environ, TS_TEST_QWEN4EXP_MODEL=str(args.model), TS_TEST_GGML_BACKEND=args.backend,
                       TS_TEST_QWEN4EXP_BATCH_STEPS=str(args.steps), TS_TEST_QWEN4EXP_REPORT_DIR=str(args.output_dir))
    try:
        provenance = capture(args, environment)
        (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        if args.capture_only:
            execution["status"] = "provenance_captured_no_test_executed"
        else:
            argv = ["dotnet", "test", str(args.test_dll), "--filter", "FullyQualifiedName=" + TEST,
                    "--results-directory", str(args.output_dir), "--logger", "trx;LogFileName=batch-probe.trx",
                    "--logger", "console;verbosity=detailed"]
            execution["test_executed"] = True
            execution["invocation"] = invoke(argv, environment, args.output_dir / "test.log", args.timeout)
            require(execution["invocation"]["exit_code"] == 0 and not execution["invocation"]["timed_out"], "Test process failed or timed out; inspect saved test.log")
            execution["trx"] = verify_trx(args.output_dir / "batch-probe.trx")
            probe_path = args.output_dir / "managed-batch-probe.json"
            execution["probe"] = verify_probe(json.loads(probe_path.read_bytes()), args, provenance["native_build"]["sha256"])
            execution["probe_report"] = file_identity(probe_path)
            after = {"native_build": file_identity(args.native), "test_native_copy": file_identity(args.test_dll.parent / args.native.name),
                     "test_assembly": file_identity(args.test_dll), "model": model_metadata(args.model), "upstream_ggml": git_identity(args.upstream),
                     "managed_assemblies": [file_identity(path) for path in sorted(args.test_dll.parent.glob("TensorSharp*.dll"))]}
            execution["after"] = after
            for key, value in after.items():
                require(value == provenance[key], f"{key} changed during the test; evidence is not bound to one identity")
            execution["status"] = "numerical_passed"
    except Exception as error:
        execution["failures"].append(f"{type(error).__name__}: {error}")
    finally:
        execution["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
        print(f"{execution['status']}: {args.output_dir}", flush=True)
    return 1 if execution["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
