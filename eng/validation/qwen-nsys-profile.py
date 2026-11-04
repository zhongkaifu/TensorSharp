#!/usr/bin/env python3
"""Capture one prescribed HTTP request after optional unprofiled cache priming.

Nsight Systems interactive launch/start/stop keeps model loading and priming out
of the report. This creates a NEW server and shuts down only that profiling
session. Run only when the GPU is otherwise idle. Reports contain the full
request, including decode; --max-tokens 1 is a prefill diagnostic, not a quality
or throughput benchmark. Requires nsys 2026 CLI and the standard Python library.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("qwen_replay", HERE / "qwen38-agent-replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


# Exact names only: no ambient TS_*, DOTNET_*, cloud credentials, proxies or
# Python startup hooks are inherited implicitly. Model options belong in spec.env.
PROCESS_ENV_NAMES = frozenset((
    "PATH", "HOME", "TMPDIR", "TMP", "TEMP", "LANG", "LC_ALL", "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES", "CUDA_CACHE_DISABLE", "CUDA_CACHE_PATH",
    "DOTNET_ROOT", "DOTNET_ROOT_X64", "DOTNET_ROOT_ARM64"))
# Nsight adds these between launch and _exec. Preserve its interposition across
# exec; the outer boundary never inherits ambient injection settings implicitly.
NSIGHT_EXEC_ENV_NAMES = frozenset((
    "LD_PRELOAD", "CUDA_INJECTION32_PATH", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH",
    "NSYS_SESSION_NAME", "NSYS_CALLBACK_NAME", "NSYS_REPORT_PATH", "NSYS_TARGET_SESSION_ID",
    "NSYS_PROFILING_SESSION_ID", "NSYS_AGENT_ID", "NSYS_INJECTION_ID", "NSYS_PARENT_PID",
    # Exact launch-injected transport/config names observed in the actual CLI.
    # These are inner-exec only, not an ambient QUADD_/NSYS_/CUPTI_ prefix grant.
    "QUADD_CUDA_CONFIG", "QUADD_INJECTION_CONFIG", "QUADD_INJECTION_PROXY",
    "QUADD_IS_PROCESS_LAUNCHED_BY_NSYS", "QUADD_AGENT_PID", "CUPTI_PROFILE_MODE",
    "NSYS_AGENT_TMP_DIR", "NSYS_CONTROL_CUPTI_FEATURES", "NSYS_CUDA_GPU_TIME_CONVERSION",
    "NSYS_SESSION_TEMP_DIR", "NSYS_TARGET_STD_REDIRECT_ACTION", "NSYS_TSC_SUPPORT",
    "TRACE_START_IMMEDIATELY"))
START_OPTIONS = frozenset(("--session", "--output", "--sample", "--cpuctxsw", "--force-overwrite"))


def validate_server_spec(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("server-spec must be a JSON object")
    command = cfg.get("argv")
    if (not isinstance(command, list) or not command
            or not all(isinstance(x, str) and "\0" not in x for x in command) or not command[0]):
        raise ValueError("server-spec.argv must be a nonempty executable/string array without NUL")
    env = cfg.get("env", {})
    if (not isinstance(env, dict) or any(not isinstance(k, str) or not k or "=" in k or "\0" in k
            or not isinstance(v, str) or "\0" in v for k, v in env.items())):
        raise ValueError("server-spec.env must map valid nonempty environment names to string values")
    if "cwd" in cfg and (not isinstance(cfg["cwd"], str) or not cfg["cwd"] or "\0" in cfg["cwd"]
                         or not Path(cfg["cwd"]).is_dir()):
        raise ValueError("server-spec.cwd must name an existing directory")
    return cfg


def process_environment(declared=None, *, target_exec=False):
    names = PROCESS_ENV_NAMES | (NSIGHT_EXEC_ENV_NAMES if target_exec else frozenset())
    result = {name: os.environ[name] for name in names if name in os.environ}
    result.setdefault("PATH", os.defpath)
    result.update(declared or {})
    return result


def linux_proc_available():
    return Path("/proc/self/stat").exists()


def nsys_start_capabilities(executable, env):
    text = subprocess.check_output([executable, "start", "--help"], text=True,
                                   stderr=subprocess.STDOUT, timeout=30, env=env)
    # Only option declarations at a line start count, not prose/example mentions
    # or a longer option whose prefix happens to match discard-environment.
    offered = set(re.findall(r"(?m)^[ \t]*(?:-[A-Za-z],?[ \t]+)?(--[a-z][a-z0-9-]*)(?=[\s=,]|$)", text))
    missing = START_OPTIONS - offered
    if missing:
        raise RuntimeError("nsys start does not advertise required options: " + ", ".join(sorted(missing)))
    return {"help_sha256": hashlib.sha256(text.encode()).hexdigest(), "supported_options": sorted(offered),
            "discard_environment_supported": "--discard-environment" in offered}


def start_command(session, prefix, sampling, capabilities):
    command = ["start", "--session=" + session, "--output=" + str(prefix),
               "--sample=" + sampling, "--cpuctxsw=" + sampling, "--force-overwrite=false"]
    if capabilities["discard_environment_supported"]:
        command.append("--discard-environment=true")
    return command


def select_row(fixture, scenario, lane, step, max_tokens=None):
    matches = [row for row in fixture if (row["scenario"], row["lane"], row["step"]) == (scenario, lane, step)]
    if len(matches) != 1:
        raise ValueError(f"Expected one fixture row for {(scenario, lane, step)}, got {len(matches)}")
    row = json.loads(json.dumps(matches[0]))
    if replay.digest(row["body"]) != row["request_sha256"]:
        raise ValueError("Fixture digest mismatch")
    if max_tokens is not None:
        row["body"]["max_tokens"] = max_tokens
        row["request_sha256"] = replay.digest(row["body"])
    return row


def process_stat(pid):
    """Read one Linux process identity; comm itself may contain parentheses."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {"state": fields[0], "starttime": int(fields[19]), "fields": fields}


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def write_server_identity(path, run_id):
    """The wrapper keeps this PID/starttime when exec replaces it with the server."""
    pid = os.getpid()
    stat = process_stat(pid)
    if stat is None:
        raise RuntimeError("Cannot identify the profiling server through Linux /proc")
    record = {"schema_version": 1, "pid": pid, "starttime": stat["starttime"],
              "boot_id": boot_id(), "run_id": run_id}
    # Never overwrite a prior invocation's identity. A reader retries a partially
    # written record; the file is flushed before exec can expose the HTTP port.
    with path.open("x") as stream:
        json.dump(record, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def read_server_identity(path, run_id):
    record = json.loads(path.read_text())
    if (not isinstance(record, dict) or record.get("schema_version") != 1
            or type(record.get("pid")) is not int or record["pid"] <= 1
            or type(record.get("starttime")) is not int or record["starttime"] < 1
            or record.get("run_id") != run_id or record.get("boot_id") != boot_id()):
        raise ValueError("Server PID record does not belong to this profiling invocation")
    return record


def recover_server_identity(path, run_id, timeout=2):
    """A failed/asynchronous launch may publish its record during cleanup."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return read_server_identity(path, run_id)
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)


def server_state(record):
    if record["boot_id"] != boot_id():
        return "identity_changed"
    stat = process_stat(record["pid"])
    if stat is None:
        return "exited"
    if stat["starttime"] != record["starttime"]:
        return "identity_changed"
    # Zombies have exited and hold no CUDA context. Their parent may reap later.
    return "exited" if stat["state"] in ("Z", "X") else "running"


def signal_owned_server(record, sig):
    """Use a pidfd so PID reuse between validation and signaling is harmless."""
    if server_state(record) != "running":
        return False
    if not callable(getattr(os, "pidfd_open", None)) or not callable(getattr(signal, "pidfd_send_signal", None)):
        raise RuntimeError("Safe server signaling needs Linux pidfd support; refusing numeric PID kill")
    try:
        descriptor = os.pidfd_open(record["pid"], 0)
    except ProcessLookupError:
        return False
    try:
        # pidfd_open may have raced an exit and PID reuse. Only signal a handle
        # acquired while the self-recorded process identity still matches.
        if server_state(record) != "running":
            return False
        try:
            signal.pidfd_send_signal(descriptor, sig, None, 0)
            return True
        except ProcessLookupError:
            return False
    finally:
        os.close(descriptor)


def wait_server_exit(record, timeout):
    deadline = time.monotonic() + timeout
    while True:
        state = server_state(record)
        if state != "running" or time.monotonic() >= deadline:
            return state
        time.sleep(min(.05, max(0, deadline - time.monotonic())))


def stop_owned_server(record, exit_timeout, kill_timeout):
    """Stop only the recorded PID, never Nsight's or the caller's process group."""
    started = time.monotonic()
    result = {"identity": record, "confirmed_exited": False, "signals": []}
    try:
        state = server_state(record)
        if state == "running":
            if signal_owned_server(record, signal.SIGTERM):
                result["signals"].append("SIGTERM")
            state = wait_server_exit(record, exit_timeout)
        if state == "running":
            if signal_owned_server(record, signal.SIGKILL):
                result["signals"].append("SIGKILL")
            state = wait_server_exit(record, kill_timeout)
        result["final_state"] = state
        result["confirmed_exited"] = state == "exited"
    except Exception as error:
        result["error"] = repr(error)
    result["elapsed_ms"] = (time.monotonic() - started) * 1000
    return result


def process_counters(pid, identity=None):
    # Linux /proc stat indices after the '(comm)' field. These counters sum all
    # threads; unlike CUDA API durations they exclude time blocked on the GPU.
    stat = process_stat(pid)
    if stat is None or (identity is not None and (identity["boot_id"] != boot_id()
            or identity["starttime"] != stat["starttime"])):
        raise RuntimeError("Server exited or changed identity before its counters were read")
    fields = stat["fields"]
    tick_ms = 1000 / os.sysconf("SC_CLK_TCK")
    return {"user_ms": int(fields[11]) * tick_ms, "system_ms": int(fields[12]) * tick_ms,
            "minor_faults": int(fields[7]), "major_faults": int(fields[9])}


def models(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=2) as response:
            data = json.load(response)
        return data["data"][0]["id"]
    except (OSError, ValueError, KeyError, IndexError):
        return None


def tcp_ready(url):
    address = urlsplit(url)
    try:
        with socket.create_connection((address.hostname, address.port or
                                       (443 if address.scheme == "https" else 80)), timeout=.25):
            return True
    except OSError:
        return False


def exec_server(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-spec", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    cfg = validate_server_spec(json.loads(args.server_spec.read_text()))
    command = cfg["argv"]
    env = process_environment(cfg.get("env"), target_exec=True)
    if cfg.get("cwd"):
        os.chdir(cfg["cwd"])
    write_server_identity(args.pid_file, args.run_id)
    os.execvpe(command[0], command, env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--server-spec", type=Path, required=True, help="JSON: argv array, optional env object and cwd")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:5089")
    parser.add_argument("--known-model", help="Skip model discovery; requires a successful unprofiled prime")
    parser.add_argument("--scenario", required=True, help="Scenario identifier in the immutable fixture")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--prime-steps", default="0", help="Comma-separated unprofiled steps, or empty for a cold request")
    parser.add_argument("--max-tokens", type=int, help="Optional cap applied to both prime and measured requests")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--server-exit-timeout", type=float, default=15,
                        help="Seconds to wait after SIGTERM for the actual server PID to exit")
    parser.add_argument("--server-kill-timeout", type=float, default=5,
                        help="Seconds to confirm server exit after an owned-PID SIGKILL")
    parser.add_argument("--cpu-sampling", choices=("none", "process-tree"), default="none")
    parser.add_argument("--trace", default="cuda,nvtx,osrt", help="Use osrt for a CPU-only harness smoke test")
    parser.add_argument("--output", type=replay.evidence_path, required=True, help="Fresh evidence filename prefix")
    args = parser.parse_args(argv)
    if args.known_model is not None:
        if not args.known_model or args.known_model != args.known_model.strip() or "\0" in args.known_model:
            parser.error("known-model must be a nonempty model identifier without surrounding whitespace or NUL")
        if not any(step.strip() for step in args.prime_steps.split(",")):
            parser.error("known-model requires at least one unprofiled prime step")
    if args.max_tokens is not None and args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.timeout, args.startup_timeout, args.server_exit_timeout, args.server_kill_timeout)):
        parser.error("timeouts must be finite and positive")
    if not linux_proc_available():
        parser.error("Server lifecycle verification requires Linux /proc")
    if urlsplit(args.url).hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("Use an isolated loopback server")
    if args.known_model is not None and tcp_ready(args.url):
        parser.error("The selected port already accepts TCP connections; select a free port")
    if args.known_model is None and models(args.url) is not None:
        parser.error("The selected port already serves a model; select a free port")
    prefix = args.output
    if any(prefix.parent.glob(prefix.name + ".*")):
        parser.error("Output prefix already exists")
    fixture = [json.loads(line) for line in args.fixture.read_text().splitlines() if line]
    measured = select_row(fixture, args.scenario, args.lane, args.step, args.max_tokens)
    primes = [select_row(fixture, args.scenario, args.lane, int(s), args.max_tokens)
              for s in args.prime_steps.split(",") if s.strip()]
    cfg = validate_server_spec(json.loads(args.server_spec.read_text()))
    child_env = process_environment(cfg.get("env"))
    session = "qwen-profile-" + uuid.uuid4().hex[:12]
    run_id = uuid.uuid4().hex
    pid_file = Path(str(prefix) + ".pid")
    report = {"session": session, "run_id": run_id, "server_spec": cfg, "scenario": args.scenario,
              "fixture": str(args.fixture.resolve()), "max_tokens_override": args.max_tokens,
              "model_source": "known-model" if args.known_model is not None else "models-discovery",
              "models_discovery_skipped": args.known_model is not None,
              "known_model_prime_verified": False,
              "known_model_prime_attempts": [],
              "source_provenance": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__), HERE / "qwen38-agent-replay.py", HERE / "qwen-nsys-summary.py")},
              "cpu_sampling": args.cpu_sampling, "trace": args.trace, "prime_results": [],
              "scope": "One isolated request, including decode. Profiling perturbs timings; compare unprofiled runs separately.",
              "gpu_gap_limit": "Wall time minus GPU activity is host/queue/idle time, not measured CPU execution time. Process CPU counters are separate, summed over threads.",
              "complete": False, "capture_status": "NOT_RUN",
              "environment_policy": {"inherited_process_names": sorted(PROCESS_ENV_NAMES & os.environ.keys()),
                  "declared_spec_names": sorted(cfg.get("env", {})), "target_injection_names": sorted(NSIGHT_EXEC_ENV_NAMES),
                  "ambient_environment_copied": False}}
    profiler_log = open(str(prefix) + ".nsys.log", "w")
    server_log = open(str(prefix) + ".server.log", "w")
    Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    launched = collecting = False
    launcher = None
    identity = None
    failure = None

    def nsys(*command, timeout=180):
        subprocess.run([args.nsys, *command], check=True, stdout=profiler_log,
                       stderr=subprocess.STDOUT, timeout=timeout, env=child_env)

    try:
        report["nsys_version"] = subprocess.check_output([args.nsys, "--version"], text=True, stderr=subprocess.STDOUT, timeout=30, env=child_env).strip()
        report["nsys_start_capabilities"] = nsys_start_capabilities(args.nsys, child_env)
        launch = [args.nsys, "launch", "--session-new=" + session, "--trace=" + args.trace,
                  "--cuda-graph-trace=node", sys.executable, str(Path(__file__).resolve()), "_exec",
                  "--server-spec", str(args.server_spec.resolve()), "--pid-file", str(pid_file),
                  "--run-id", run_id]
        launcher = subprocess.Popen(launch, stdout=server_log, stderr=subprocess.STDOUT, env=child_env)
        launched = True
        deadline = time.monotonic() + args.startup_timeout
        while True:
            if identity is None and pid_file.exists():
                try:
                    identity = read_server_identity(pid_file, run_id)
                    report["server_identity"] = identity
                except json.JSONDecodeError:
                    pass  # _exec is still writing its exclusive identity record.
            model = (args.known_model if tcp_ready(args.url) else None) if args.known_model is not None else models(args.url)
            if model is not None and identity is not None:
                if server_state(identity) != "running":
                    raise RuntimeError("HTTP readiness does not belong to a live profiling server identity")
                break
            if launcher.poll() not in (None, 0):
                raise RuntimeError("nsys launch failed; inspect server log")
            if time.monotonic() >= deadline:
                raise TimeoutError("Server readiness deadline exceeded")
            time.sleep(.25)
        report["model"] = model
        pid = identity["pid"]
        report["server_pid"] = pid
        for row in primes:
            while True:
                if args.known_model is not None:
                    if not report["prime_results"] and report["known_model_prime_attempts"] and time.monotonic() >= deadline:
                        raise TimeoutError("Known-model startup remained HTTP 503 until the readiness deadline")
                    if server_state(identity) != "running":
                        raise RuntimeError("Known-model prime does not belong to a live profiling server identity")
                result = replay.request(args.url, model, row, args.timeout)
                if args.known_model is not None:
                    report["known_model_prime_attempts"].append(result)
                    # A server can bind before loading its model. Retry only the
                    # first prime's explicit startup 503, within the original
                    # readiness deadline; never retry failed inference/streams.
                    if not report["prime_results"] and result.get("http_status") == 503:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("Known-model startup remained HTTP 503 until the readiness deadline")
                        time.sleep(min(.25, remaining))
                        continue
                break
            report["prime_results"].append(result)
            if result["errors"] or not result["done"]:
                raise RuntimeError("Cache priming failed; inspect metadata")
            if args.known_model is not None and (result.get("http_status") != 200
                    or result.get("finish_reason") not in ("stop", "length", "tool_calls")):
                raise RuntimeError("Known-model prime did not finish successfully; capture was not started")
        if args.known_model is not None:
            if server_state(identity) != "running":
                raise RuntimeError("Known-model profiling server identity changed after priming")
            report["known_model_prime_verified"] = True
        nsys(*start_command(session, prefix, args.cpu_sampling, report["nsys_start_capabilities"]))
        report["capture_status"] = "COLLECTING"
        collecting = True
        cpu_before = process_counters(pid, identity)
        report["request_started_unix_ns"] = time.time_ns()
        report["result"] = replay.request(args.url, model, measured, args.timeout)
        report["request_finished_unix_ns"] = time.time_ns()
        cpu_after = process_counters(pid, identity)
        report["server_process_cpu"] = {k: cpu_after[k] - cpu_before[k] for k in ("user_ms", "system_ms")}
        fault_keys = ("minor_faults", "major_faults")
        report["server_process_faults"] = {"before": {k: cpu_before[k] for k in fault_keys},
            "after": {k: cpu_after[k] for k in fault_keys},
            "delta": {k: cpu_after[k] - cpu_before[k] for k in fault_keys}}
        nsys("stop", "--session=" + session)
        collecting = False
        report["request_complete"] = (not report["result"]["errors"] and report["result"]["done"]
            and report["result"].get("finish_reason") in ("stop", "length", "tool_calls"))
        if not report["request_complete"]:
            raise RuntimeError("Measured request did not finish correctly")
        nsys("export", "--type=sqlite", "--output=" + str(prefix) + ".sqlite", str(prefix) + ".nsys-rep")
        report["capture_complete"] = True
        report["capture_status"] = "EXECUTED_COMPLETE"
    except BaseException as error:
        if report["capture_status"] == "COLLECTING":
            report["capture_status"] = "INCOMPLETE_NOT_PASS"
        report["error"] = repr(error)
        failure = error
    finally:
        if collecting:
            try:
                nsys("stop", "--session=" + session)
            except Exception as error:
                report["stop_error"] = repr(error)
        if launched:
            try:
                # Nsight's target need not lead its process group. Release the
                # session, then signal only the independently verified server PID.
                nsys("shutdown", "--session=" + session, "--kill=none", timeout=30)
            except Exception as error:
                report["shutdown_error"] = repr(error)
            try:
                # Also recover the identity after startup/capture failure. The
                # wrapper can have exec'd even when HTTP never became ready.
                if identity is None:
                    identity = recover_server_identity(pid_file, run_id)
                    report["server_identity"] = identity
                report["server_cleanup"] = stop_owned_server(
                    identity, args.server_exit_timeout, args.server_kill_timeout)
            except Exception as error:
                report["server_cleanup"] = {"confirmed_exited": False, "error": repr(error)}
        if launcher is not None:
            try:
                report["launcher_exit_code"] = launcher.wait(timeout=15)
            except subprocess.TimeoutExpired:
                report["launcher_wait_timeout"] = True
            except Exception as error:
                report["launcher_wait_error"] = repr(error)
        Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2) + "\n")
        profiler_log.close()
        server_log.close()
    if launched and (not report.get("server_cleanup", {}).get("confirmed_exited")
                     or "shutdown_error" in report or report.get("launcher_wait_timeout")
                     or "launcher_wait_error" in report):
        cleanup_error = "Profiling server/session exit was not fully confirmed; do not start another GPU profile"
        report["cleanup_error"] = cleanup_error
        Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2) + "\n")
        if failure is None:
            failure = RuntimeError(cleanup_error)
    if failure is not None:
        raise failure
    try:
        subprocess.run([sys.executable, str(HERE / "qwen-nsys-summary.py"),
                        "--sqlite", str(prefix) + ".sqlite", "--request", str(prefix) + ".json",
                        "--output", str(prefix) + ".summary.json"], check=True, env=child_env)
        report["summary_complete"] = True
        report["complete"] = True
    except BaseException as error:
        report["summary_error"] = repr(error)
        raise
    finally:
        Path(str(prefix) + ".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_exec":
        exec_server(sys.argv[2:])
    else:
        main()
