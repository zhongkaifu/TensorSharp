#!/usr/bin/env python3
"""Run a fully specified server profile and retain startup, commands and results.

JSON profile fields: id, model, backend, port, env, extra_args, suites. Each
suite is a list of argv strings, with {repo}, {output}, {url}, and {model_id}
placeholders. Servers are always stopped by their own process group on exit.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import urllib.request


class ProgressReporter:
    """Persist progress locally; losing the console reader is not a test failure."""
    def __init__(self, path, report, stream=None):
        self.path, self.report = path, report
        self.stream = sys.stdout if stream is None else stream
        report["progress_log"] = str(path)

    def record(self, event):
        with self.path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"at_unix": time.time(), **event}) + "\n")

    def disconnected_pipe(self, error):
        if isinstance(error, BrokenPipeError) or error.errno == errno.EPIPE:
            return True
        # Windows' CRT can translate a closed stdout pipe into EINVAL. Accept
        # that translation only for an actual pipe, never files or terminals.
        if os.name == "nt" and error.errno == errno.EINVAL:
            try:
                return stat.S_ISFIFO(os.fstat(self.stream.fileno()).st_mode)
            except (AttributeError, OSError, ValueError):
                pass
        return False

    def __call__(self, *values):
        message = " ".join(str(value) for value in values)
        self.record({"event": "progress", "message": message})
        if self.stream is None:
            return
        try:
            print(message, file=self.stream, flush=True)
        except OSError as error:
            if not self.disconnected_pipe(error):
                raise
            disconnected, self.stream = self.stream, None
            observation = {"status": "disconnected", "error": str(error),
                           "at_unix": time.time()}
            self.report["console_output"] = observation
            self.record({"event": "console_disconnected", **observation})
            if disconnected is sys.stdout:
                # CPython flushes stdout again at shutdown. Redirect the same
                # descriptor so a buffered EPIPE cannot replace the real result
                # with exit code 120. Test/custom streams need no fd mutation.
                try:
                    descriptor = disconnected.fileno()
                except (AttributeError, OSError, ValueError):
                    descriptor = None
                if descriptor is not None:
                    try:
                        with open(os.devnull, "w") as sink:
                            os.dup2(sink.fileno(), descriptor)
                    except OSError as redirect_error:
                        observation["shutdown_redirect_error"] = str(redirect_error)
                        # Keep a failed console flush from masking the genuine
                        # result even if the original descriptor cannot be used.
                        sys.stdout = None


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def mapped_native_libraries(pid):
    paths = set()
    maps = Path(f"/proc/{pid}/maps")
    if maps.exists():
        for mapping in maps.read_text().splitlines():
            candidate = mapping.split()[-1]
            if candidate.startswith("/") and ("GgmlOps" in candidate or "libggml" in candidate) and Path(candidate).is_file():
                paths.add(candidate)
    return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(paths)}


def check_native_policy(profile, mapped, expected_hash):
    actual_ops = {digest for path, digest in mapped.items() if "GgmlOps" in path}
    native_policy = profile.get("native_policy", "exact")
    if native_policy not in ("exact", "absent"):
        raise ValueError(f"Unknown native policy {native_policy!r}")
    if native_policy == "absent" and mapped:
        raise RuntimeError("Native-free profile unexpectedly loaded GgmlOps or ggml")
    if native_policy == "exact" and actual_ops != {expected_hash}:
        raise RuntimeError(f"Loaded native library hash {sorted(actual_ops)} differs from expected {expected_hash}")


def check_runtime_policy(profile, observation, expected_hash, model_id):
    check_native_policy(profile, observation['mapped_native_libraries'], expected_hash)
    if not any(item.get('id') == model_id for item in observation['endpoint_models'].get('data', [])):
        raise RuntimeError('The process-owned HTTP connection no longer reports the expected model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dotnet", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--server-assembly", type=Path,
                        help="Use an isolated server application; requires the application manifest pair")
    parser.add_argument("--application-manifest", type=Path)
    parser.add_argument("--application-manifest-sha256")
    args = parser.parse_args()
    if bool(args.application_manifest) != bool(args.application_manifest_sha256):
        parser.error("--application-manifest and --application-manifest-sha256 must be supplied together")
    if args.server_assembly and not args.application_manifest:
        parser.error("An isolated --server-assembly requires a pinned --application-manifest")
    profile = json.loads(args.profile.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    repo = args.repo.resolve()
    server = (args.server_assembly or repo / "TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll").resolve()
    url = f"http://127.0.0.1:{profile.get('port', 5100)}"
    command = [str(args.dotnet), str(server), "--model", profile["model"],
               "--backend", profile.get("backend", "ggml_cuda"), "--port", str(profile.get("port", 5100))]
    command += profile.get("extra_args", [])
    env = dict(os.environ)
    env.update({key: str(value) for key, value in profile.get("env", {}).items()})
    env["LD_LIBRARY_PATH"] = str(args.native.parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PATH"] = str(args.dotnet.parent) + ":/usr/local/cuda/bin:" + env.get("PATH", "")
    report = {"profile": profile, "started_at_unix": time.time(), "command": command,
              "environment_overrides": {key: env[key] for key in [*profile.get("env", {}), "LD_LIBRARY_PATH", "PATH"]},
              "binaries": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (server, args.native)},
              "managed_assemblies": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sorted(server.parent.glob("TensorSharp.*.dll"))},
              "status": "starting", "suites": []}
    manifest = args.output / "profile.json"
    progress = ProgressReporter(args.output / "progress.jsonl", report)
    save(manifest, report)
    process = None
    telemetry = None
    started = time.monotonic()
    try:
        if args.application_manifest:
            from release_application_identity import check_application
            if server.name != "TensorSharp.Server.Host.dll":
                raise ValueError("The pinned application must execute TensorSharp.Server.Host.dll")
            report["application_identity_before"] = check_application(
                server.parent, args.application_manifest, args.application_manifest_sha256)
        if (profile.get("native_policy", "exact") == "exact" and profile.get("expected_native_sha256")
                and report["binaries"][str(args.native)] != profile["expected_native_sha256"]):
            raise RuntimeError("Provided native artifact does not match the profile digest")
        report["harness_preflight"] = {}
        for relative, expected in profile.get("expected_harness_sha256", {}).items():
            actual = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            report["harness_preflight"][relative] = {"expected_sha256": expected, "actual_sha256": actual}
            if actual != expected:
                raise RuntimeError(f"Required original harness digest no longer matches: {relative}")
        preflight_values = {"repo": str(repo), "output": str(args.output), "url": url,
                            "model_id": "MODEL_RESOLVED_AT_READINESS", "python": sys.executable}
        for argv in profile.get("before_suites", []) + profile.get("suites", []):
            for part in argv:
                candidate = part.format_map(preflight_values)
                if candidate.startswith("/") and candidate.endswith((".py", ".dll")) and not Path(candidate).is_file():
                    raise FileNotFoundError(f"Required suite/helper script is absent: {candidate}")
        with (args.output / "server.log").open("w") as log:
            process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        report["server_pid"] = process.pid
        telemetry_script = Path(__file__).with_name("sample-release-telemetry.py")
        if profile.get("telemetry", True) and telemetry_script.is_file():
            telemetry = subprocess.Popen([sys.executable, str(telemetry_script), "--pid", str(process.pid),
                                          "--output", str(args.output / "telemetry.jsonl")],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            report["telemetry_pid"] = telemetry.pid
        save(manifest, report)
        ready = False
        while time.monotonic() - started < args.ready_timeout:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited during load with code {process.returncode}")
            try:
                with urllib.request.urlopen(url + "/v1/models", timeout=5) as response:
                    models = json.load(response)
                if models.get("data"):
                    model_id = models["data"][0]["id"]
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(2)
        if not ready:
            raise TimeoutError("Server readiness timed out")
        report.update(status="running", model_id=model_id, load_seconds=time.monotonic() - started)
        report["loaded_native_libraries"] = mapped_native_libraries(process.pid)
        save(manifest, report)
        expected_hash = profile.get("expected_native_sha256", report["binaries"][str(args.native)])
        check_native_policy(profile, report["loaded_native_libraries"], expected_hash)
        runtime_identity = None
        if args.application_manifest:
            from release_application_identity import check_application_runtime
            runtime_identity = check_application_runtime(process.pid, profile.get('port', 5100),
                command, report['application_identity_before'])
            check_runtime_policy(profile, runtime_identity, expected_hash, model_id)
            report['application_process_identity_at_readiness'] = runtime_identity
            save(manifest, report)
        progress("Server ready", profile["id"], "load_s", report["load_seconds"])
        substitutions = {"repo": str(repo), "output": str(args.output), "url": url, "model_id": model_id,
                         "python": sys.executable}
        report["before_suites"] = []
        for index, hook in enumerate(profile.get("before_suites", [])):
            argv = [part.format_map(substitutions) for part in hook]
            entry = {"command": argv, "started_at_unix": time.time()}
            report["before_suites"].append(entry)
            save(manifest, report)
            with (args.output / f"before-suites-{index}.log").open("w") as log:
                result = subprocess.run(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=120)
            entry.update(exit_code=result.returncode, finished_at_unix=time.time())
            save(manifest, report)
            if result.returncode:
                raise RuntimeError(f"Pre-suite command {index} failed with code {result.returncode}")
        quiet_seconds = float(profile.get("before_suites_quiet_seconds", 0))
        if quiet_seconds:
            report["quiet_window"] = {"started_at_unix": time.time(), "requested_seconds": quiet_seconds}
            save(manifest, report)
            time.sleep(quiet_seconds)
            report["quiet_window"]["finished_at_unix"] = time.time()
            save(manifest, report)
        for index, suite in enumerate(profile.get("suites", [])):
            argv = [part.format_map(substitutions) for part in suite]
            entry = {"command": argv, "started_at_unix": time.time()}
            report["suites"].append(entry)
            if args.application_manifest:
                entry['application_process_identity_before'] = check_application_runtime(
                    process.pid, profile.get('port', 5100), command,
                    report['application_identity_before'], expected=runtime_identity)
                check_runtime_policy(profile, entry['application_process_identity_before'], expected_hash, model_id)
                entry['started_at_unix'] = time.time()  # Exclude the identity control request from suite wall time.
            save(manifest, report)
            progress("Running suite", index, argv)
            with (args.output / f"suite-{index}.log").open("w") as log:
                result = subprocess.run(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            entry.update(exit_code=result.returncode, finished_at_unix=time.time())
            save(manifest, report)
            progress("Finished suite", index, "exit_code", result.returncode)
            if process.poll() is not None:
                raise RuntimeError(f"Server exited with code {process.returncode}")
            # An aborting native process can remain visible while its listener
            # and mappings are already gone. Do not cascade connection failures
            # through unrelated feature suites merely because poll() is pending.
            try:
                with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
                    available = json.load(response)
                if not any(item.get("id") == model_id for item in available.get("data", [])):
                    raise RuntimeError("Expected model is no longer available")
            except Exception as error:
                entry["server_health_after_suite"] = {"status": "unavailable", "error": str(error)}
                log_tail = (args.output / "server.log").read_text(errors="replace").splitlines()[-80:]
                fatal_lines = [line for line in log_tail if any(marker in line.lower() for marker in
                    ("cuda error", "illegal memory", "assert", "ggml_abort", "exception"))]
                report["server_fatal_diagnostics"] = fatal_lines
                raise RuntimeError(f"Server became unavailable after suite {index}; later suites were not executed: {error}") from error
            entry["server_health_after_suite"] = {"status": "available", "checked_at_unix": time.time()}
            if args.application_manifest:
                entry['application_process_identity_after'] = check_application_runtime(
                    process.pid, profile.get('port', 5100), command,
                    report['application_identity_before'], expected=runtime_identity)
                check_runtime_policy(profile, entry['application_process_identity_after'], expected_hash, model_id)
            save(manifest, report)
        report["loaded_native_libraries_after_suites"] = mapped_native_libraries(process.pid)
        check_native_policy(profile, report["loaded_native_libraries_after_suites"], expected_hash)
        report["status"] = ("passed" if all(suite.get("exit_code") == 0 for suite in report["suites"]) else "failed") if report["suites"] else "ready-not-tested"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        try:
            progress("ERROR", str(error))
        except Exception as reporting_error:
            # The original lifecycle failure remains primary if its durable
            # progress log or console also fails while reporting the error.
            report["error_reporting_error"] = str(reporting_error)
    finally:
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if telemetry:
            try:
                telemetry.wait(timeout=15)
            except subprocess.TimeoutExpired:
                telemetry.terminate()
                telemetry.wait(timeout=15)
            report["telemetry_exit_code"] = telemetry.returncode
        if process:
            report["server_exit_code"] = process.returncode
        if args.application_manifest:
            try:
                from release_application_identity import check_application
                report["application_identity_after"] = check_application(
                    server.parent, args.application_manifest, args.application_manifest_sha256)
            except Exception as error:
                report["application_identity_after"] = {"status": "failed", "error": str(error)}
                report["status"] = "failed"
        report["finished_at_unix"] = time.time()
        save(manifest, report)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
