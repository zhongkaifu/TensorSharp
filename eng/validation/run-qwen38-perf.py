#!/usr/bin/env python3
"""Run one isolated server configuration and the fixed Qwen agent replay.

The JSON config contains argv, environment, cwd, url, model, fixture, and label.
This runner owns only the server it starts, records binary/dependency identity,
samples GPU memory, and stops that server even when a replay fails. Run engines
sequentially on an otherwise idle GPU. No claimed parity between quantizations.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qwen38_compare", Path(__file__).with_name("compare-qwen38-agent-replay.py"))
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)
REPLAY = COMPARE.REPLAY
DEFAULT_SCENARIOS = "short1k,prefill8k,repeat19k,incremental,agent34,image_replace,arithmetic"

# Exact names only: broad TS_*/TENSORSHARP_* matching can expose credentials for
# unrelated integrations. This is the launch environment, before native startup
# defaults (for example SM12 PDL compatibility) take effect inside the server.
BENCHMARK_ENVIRONMENT_KEYS = frozenset("""
MAX_CONTEXT KV_CACHE_DTYPE CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER
GGML_CUDA_DEVICES GGML_CUDA_PDL GGML_CUDA_DISABLE_GRAPHS GGML_CUDA_GRAPH_OPT
GGML_CUDA_DISABLE_FUSION GGML_CUDA_REGISTER_HOST GGML_CUDA_NO_PINNED
GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_CUDA_CUBLAS_COMPUTE_TYPE GGML_CUDA_MMQ_PREC
TS_PREFIX_CACHE_ISOLATION TS_PREFIX_CACHE_MODE TS_PREFIX_CHECKPOINTS_MAX
TS_KV_INITIAL_TOKENS TS_KV_GENERATION_RESERVE_MAX TS_KV_HOLDER_POOL_MAX
TS_RETAINED_FUSED_CACHE_MAX TS_GGML_HOST_HUGEPAGES TS_PREFILL_CHUNK
TS_PREFILL_WARMUP TS_PREFILL_WARMUP_LEN TS_DFLASH_PREFILL_CHUNK
TS_PER_SEQ_FUSED TS_BATCHED_FUSED_DECODE
TS_SCHED_DISABLE_BATCHED TS_SCHED_BLOCK_SIZE TS_SCHED_DECODE_QUANTUM
TS_SCHED_MAX_BATCHED_TOKENS TS_SCHED_MAX_RUNNING_SEQS TS_SCHED_NUM_BLOCKS
TS_SCHED_PREFILL_CHUNK TS_SCHED_PREFIX_CACHE TS_SCHED_SOLO_PREFILL_CHUNK
TS_SCHED_STOP_REPETITION TS_THINKING_BUDGET
TS_GGML_CPU_THREADS TS_GGML_F32_RESIDENT TS_GGML_RETAIN_HOST_WEIGHTS
TS_CB_DEBUG TS_GGML_LOG_DEBUG TS_GGML_LOG_VRAM TS_GGML_NODE_PROFILE
TS_GGML_PHASE_TIMING TS_DFLASH_SELECTOR_DEBUG
TS_QWEN35_VENC_TRACE TS_DUMP_LOGITS
TENSORSHARP_PREFIX_CACHE_DIR TENSORSHARP_LOG_DIR
TENSORSHARP_TP_DEGREE TENSORSHARP_TP_DEVICES
TENSORSHARP_LAYER_SPLIT_DEGREE TENSORSHARP_LAYER_SPLIT_DEVICES
""".split())


def benchmark_environment(environment, overrides):
    return {key: {"value": environment.get(key),
                  "source": "override" if key in overrides else "inherited" if key in environment else "unset"}
            for key in sorted(BENCHMARK_ENVIRONMENT_KEYS)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def capture(argv, cwd=None):
    try:
        process = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=30)
        return {"exit_code": process.returncode, "output": process.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"exit_code": -1, "output": str(error)}


def ensure_endpoint_unused(url):
    address = urlsplit(url)
    if address.scheme not in ("http", "https") or address.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Server URL must be an HTTP(S) loopback endpoint on this host")
    try:
        connection = socket.create_connection((address.hostname, address.port or (443 if address.scheme == "https" else 80)), timeout=2)
    except ConnectionRefusedError:
        return
    except OSError as error:
        raise ValueError(f"Cannot establish whether server endpoint is unused: {error}") from error
    connection.close()
    raise ValueError("Server endpoint already has a listener; refusing to benchmark an unowned service")


def expected_requests(fixture_bytes, cfg):
    rows = [json.loads(line) for line in fixture_bytes.decode().splitlines() if line.strip()]
    identities = set()
    for row in rows:
        key = (row["scenario"], row["lane"], row["step"])
        if key in identities:
            raise ValueError("duplicate fixture scenario/lane/step")
        identities.add(key)
        if REPLAY.digest(row["body"]) != row["request_sha256"]:
            raise ValueError("fixture request SHA-256 mismatch")
        if "expected" not in row:
            raise ValueError("fixture lacks expected-output declaration")
    degrees = [int(value) for value in cfg.get("concurrency", "1,2,4").split(",")]
    scenarios = (cfg.get("scenarios") or DEFAULT_SCENARIOS).split(",")
    if not degrees or min(degrees) < 1 or len(set(degrees)) != len(degrees) or len(set(scenarios)) != len(scenarios):
        raise ValueError("concurrency/scenarios must be unique and concurrency positive")
    steps = cfg.get("steps")
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError("steps must be a positive integer")
    result = {}
    for scenario in scenarios:
        for degree in degrees:
            reference_steps = None
            for lane in range(degree):
                selected = sorted((row for row in rows if row["scenario"] == scenario and row["lane"] == lane
                                   and (steps is None or row["step"] < steps)), key=lambda row: row["step"])
                selected_steps = [row["step"] for row in selected]
                if not selected or selected_steps != list(range(len(selected_steps))):
                    raise ValueError("fixture must cover contiguous steps from zero for every requested lane")
                if reference_steps is not None and reference_steps != selected_steps:
                    raise ValueError("fixture lanes cover different step sets")
                reference_steps = selected_steps
                for row in selected:
                    result[(scenario, degree, lane, row["step"])] = row
    return result


def verify_replay(path, fixture_hash, expected, cfg):
    if not path.exists():
        return ["replay report missing"]
    errors = []
    try:
        report = json.loads(path.read_text())
        actual, index_errors = COMPARE.index_report(report)
        errors.extend(index_errors)
        if report.get("complete") is not True or report.get("passed") is not True:
            errors.append("replay is incomplete or reports failed checks")
        if report.get("fixture_sha256") != fixture_hash:
            errors.append("replay used a different fixture")
        if report.get("model") != cfg["model"] or report.get("label") != cfg["label"]:
            errors.append("replay model/label differs from configuration")
        if actual.keys() != expected.keys():
            errors.append("replay request key set differs from the planned fixture selection")
        for key in sorted(actual.keys() & expected.keys()):
            first, second = expected[key], actual[key]
            if second.get("request_sha256") != first["request_sha256"] or second.get("expected") != first["expected"]:
                errors.append(f"request identity mismatch: {key}")
            if second.get("wire_request_sha256") != REPLAY.digest(dict(first["body"], model=cfg["model"])):
                errors.append(f"wire request identity mismatch: {key}")
            failures = COMPARE.row_failures(second)
            if failures:
                errors.append(f"request failed {key}: {'; '.join(failures)}")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        errors.append(f"invalid replay report: {error}")
    return errors


def stop_server(server):
    details = {"shutdown_requested": True, "forced_shutdown": False}
    if os.name == "posix":
        # The group was created by start_new_session; it includes launcher children.
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif server.poll() is None:
        server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        details["forced_shutdown"] = True
        if os.name == "posix":
            os.killpg(server.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True, timeout=30)
        server.wait(timeout=30)
    if os.name == "posix":
        # A child can ignore SIGTERM after the launcher exits.
        try:
            os.killpg(server.pid, 0)
        except ProcessLookupError:
            pass
        else:
            details["forced_shutdown"] = True
            os.killpg(server.pid, signal.SIGKILL)
    details["server_exit_code"] = server.returncode
    return details


def wait_for_idle_gpu(command, timeout=10):
    # CUDA context release can lag a killed child briefly. Observe cleanup without
    # killing any other process or treating an unavailable monitor as an idle GPU.
    deadline = time.monotonic() + timeout
    while True:
        state = capture(command)
        if state["exit_code"] or not state["output"] or time.monotonic() >= deadline:
            return state
        time.sleep(.5)


def execute(config_path, output):
    provenance = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "complete": False, "passed": False, "errors": [], "server_exit_code": None,
                  "quality_scope": "Synthetic arithmetic/OCR/tool arguments; prose checks transport only. No broad model-quality claim.",
                  "gpu_sampling_scope": "One-second polling during replay; sampled values are not guaranteed instantaneous peaks."}
    server = sampler = None
    samples = []
    stop_sampling = threading.Event()
    result = 1
    gpu_command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    try:
        cfg = json.loads(config_path.read_text())
        provenance.update(config=cfg, config_sha256=sha(config_path), source_revision=cfg.get("source_revision"),
                          source_revision_is_declared=True, patch_sha256=cfg.get("patch_sha256"))
        if not isinstance(cfg.get("argv"), list) or not cfg["argv"] or any(not isinstance(arg, str) for arg in cfg["argv"]):
            raise ValueError("argv must be a nonempty list of strings")
        fixture_bytes = Path(cfg["fixture"]).read_bytes()
        expected = expected_requests(fixture_bytes, cfg)
        fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
        fixture_snapshot = output / "fixture.jsonl"
        fixture_snapshot.write_bytes(fixture_bytes)
        provenance.update(fixture_sha256=fixture_hash, fixture_snapshot=str(fixture_snapshot), planned_requests=len(expected))
        manifest_path = Path(cfg["fixture"]).with_suffix(".manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("sha256") != fixture_hash:
                raise ValueError("fixture manifest SHA-256 mismatch")
            provenance["fixture_manifest"] = manifest
        environment = dict(os.environ, **cfg.get("environment", {}))
        effective_overrides = dict(cfg.get("environment", {}))
        if cfg.get("engine") == "tensorsharp":
            effective_overrides.update(TENSORSHARP_PREFIX_CACHE_DIR=str(output / "prefix-cache"),
                                       TENSORSHARP_LOG_DIR=str(output / "logs"))
            environment.update(effective_overrides)
        provenance["effective_environment_overrides"] = effective_overrides
        provenance["benchmark_launch_environment"] = benchmark_environment(environment, effective_overrides)
        provenance["device"] = capture(["nvidia-smi"])
        if provenance["device"]["exit_code"]:
            raise ValueError("GPU device identity could not be recorded")
        provenance["binary_sha256"] = {str(path): sha(path) for path in cfg.get("binaries", [])}
        if cfg.get("ggml"):
            revision = capture(["git", "rev-parse", "HEAD"], cfg["ggml"])
            status = capture(["git", "status", "--porcelain"], cfg["ggml"])
            provenance.update(ggml_revision=revision, ggml_status=status)
            if revision["exit_code"] or status["exit_code"] or status["output"]:
                raise ValueError("Upstream ggml must be an unchanged readable checkout")
        existing = capture(gpu_command)
        provenance["gpu_processes_before"] = existing
        if existing["exit_code"] or existing["output"]:
            raise ValueError("GPU must have no existing compute processes: " + existing["output"])
        ensure_endpoint_unused(cfg["url"])
        write_json(output / "provenance.json", provenance)
        with (output / "server.log").open("w") as log:
            server = subprocess.Popen(cfg["argv"], cwd=cfg["cwd"], env=environment, start_new_session=os.name == "posix",
                                      stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            provenance["server_pid"] = server.pid
            deadline = time.monotonic() + cfg.get("startup_timeout", 600)
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited during startup ({server.returncode})")
                try:
                    with urllib.request.urlopen(cfg["url"].rstrip("/") + "/v1/models", timeout=2) as response:
                        served = json.load(response)
                    if cfg["model"] not in [item["id"] for item in served.get("data", [])]:
                        raise ValueError("Configured model ID not served: " + repr(served))
                    if server.poll() is not None:
                        raise RuntimeError("Owned server exited before readiness was established")
                    provenance["served_models"] = served
                    provenance["readiness_check"] = "/v1/models served the configured model while the owned process was alive"
                    break
                except (OSError, TimeoutError):
                    time.sleep(.5)
            else:
                raise TimeoutError("Server did not become ready")
            benchmark_started = time.monotonic()
            def sample():
                while not stop_sampling.is_set():
                    data = capture(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"])
                    samples.append({"elapsed_seconds": time.monotonic() - benchmark_started, **data})
                    stop_sampling.wait(1)
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            command = [sys.executable, str(ROOT / "eng/validation/qwen38-agent-replay.py"), "run",
                       "--fixture", str(fixture_snapshot), "--output", str(output / "replay.json"),
                       "--url", cfg["url"], "--model", cfg["model"], "--label", cfg["label"],
                       "--provenance", str(output / "provenance.json"),
                       "--concurrency", cfg.get("concurrency", "1,2,4")]
            if cfg.get("scenarios"):
                command += ["--scenarios", cfg["scenarios"]]
            if cfg.get("steps") is not None:
                command += ["--steps", str(cfg["steps"])]
            if cfg.get("request_timeout") is not None:
                command += ["--timeout", str(cfg["request_timeout"])]
            provenance["replay_argv"] = command
            write_json(output / "provenance.json", provenance)
            with (output / "replay.log").open("w") as replay_log:
                result = subprocess.run(command, stdout=replay_log, stderr=subprocess.STDOUT,
                                        timeout=cfg.get("replay_timeout", 7200)).returncode
            provenance.update(replay_exit_code=result, benchmark_wall_seconds=time.monotonic() - benchmark_started)
            if result:
                provenance["errors"].append(f"replay process exited with {result}")
            provenance["errors"].extend(verify_replay(output / "replay.json", fixture_hash, expected, cfg))
            if server.poll() is not None:
                provenance["errors"].append(f"owned server exited before shutdown ({server.returncode})")
            try:
                with urllib.request.urlopen(cfg["url"].rstrip("/") + "/metrics", timeout=5) as response:
                    (output / "metrics.txt").write_bytes(response.read())
                provenance["metrics_available"] = True
            except OSError as error:
                provenance.update(metrics_available=False, metrics_error=str(error))
            provenance["complete"] = True
    except KeyboardInterrupt:
        provenance["errors"].append("runner interrupted")
        result = 130
    except Exception as error:
        provenance["errors"].append(f"{type(error).__name__}: {error}")
        result = 1
    finally:
        stop_sampling.set()
        if sampler:
            sampler.join(timeout=35)
            if sampler.is_alive():
                provenance["errors"].append("GPU sampling thread did not stop")
        write_json(output / "gpu-samples.json", samples)
        provenance["gpu_sampling_errors"] = [sample for sample in samples if sample["exit_code"]]
        provenance["gpu_sampling_complete"] = bool(samples) and not provenance["gpu_sampling_errors"]
        if sampler and not provenance["gpu_sampling_complete"]:
            provenance["errors"].append("GPU sampling unavailable or incomplete")
        if server:
            try:
                provenance.update(stop_server(server))
            except Exception as error:
                provenance["errors"].append(f"server cleanup failed: {error}")
            provenance["gpu_processes_after"] = wait_for_idle_gpu(gpu_command)
            if provenance["gpu_processes_after"]["exit_code"] or provenance["gpu_processes_after"]["output"]:
                provenance["errors"].append("GPU not confirmed idle after owned-server cleanup")
        provenance["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        provenance["passed"] = provenance["complete"] and result == 0 and not provenance["errors"]
        write_json(output / "provenance.json", provenance)
    return 0 if provenance["passed"] else result or 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not any(output.is_relative_to(ROOT / folder) for folder in ("artifacts", "docs/validation")):
        raise ValueError("Output directory must be ignored evidence in this checkout")
    output.mkdir(parents=True, exist_ok=False)
    return execute(args.config, output)


if __name__ == "__main__":
    raise SystemExit(main())
