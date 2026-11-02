#!/usr/bin/env python3
"""Linux lifecycle tests with a tiny HTTP fixture, no model or GPU required."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SERVERS = ROOT / "eng/qwen38-engine-servers.sh"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(Path("/proc/self/stat").exists(), "requires Linux /proc; not GPU validation")
class ServerLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = ROOT / "artifacts/qwen38-lifecycle-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.directory = Path(self.temporary.name)
        binary = self.directory / "venv/bin"
        binary.mkdir(parents=True)
        self.mock = binary / "vllm"
        self.mock.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
            import http.server, json, os, subprocess, sys
            if os.environ.get('FIXTURE_SERVER_ENV_FILE'):
                with open(os.environ['FIXTURE_SERVER_ENV_FILE'], 'w') as output:
                    json.dump({key:os.environ.get(key) for key in ('MAX_CONTEXT','TS_Q35_UBATCH','TS_SCHED_MAX_RUNNING_SEQS')}, output)
            port = (int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv
                    else int(sys.argv[sys.argv.index('--server-bind') + 1].rsplit(':', 1)[1]))
            if '--fixture-child' in sys.argv:
                child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                with open(os.environ['FIXTURE_CHILD_PID_FILE'], 'w') as output:
                    output.write(str(child.pid))
            class Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps({'data': [{'id':'qwen3.8-27b'}]}).encode())
                def log_message(self, *args):
                    pass
            http.server.HTTPServer(('127.0.0.1', port), Handler).serve_forever()
            """))
        self.mock.chmod(0o755)
        self.port = free_port()
        self.environment = {**os.environ, "OUT": str(self.directory / "run"),
                            "VLLM_VENV": str(binary.parent), "VLLM_PORT": str(self.port),
                            "CONTROL_PY": sys.executable, "PY": sys.executable,
                            "START_ATTEMPTS": "40", "POLL_INTERVAL": "0.05", "STOP_TIMEOUT": "1",
                            "CTX": "262144", "LANES": "1", "CUDA_VISIBLE_DEVICES": "0",
                            "MANIFEST_ENV_ALLOWLIST": "TEST_UNSET_ENGINE_FIELD",
                            "DO_NOT_RECORD_TEST_SECRET": "fixture-secret"}

    def tearDown(self) -> None:
        # Restore tampered records in each test before this ownership-aware cleanup.
        self.run_server("stop")
        self.temporary.cleanup()

    def run_server(self, *args: str, environment: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(SERVERS), *args], env=environment or self.environment,
                              text=True, capture_output=True, timeout=20)

    def start(self) -> dict:
        result = self.run_server("start", "vllm", "plain", "fixture")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(self.manifest_path.read_text())

    @property
    def record_path(self) -> Path:
        return self.directory / f"run/pids/vllm-{self.port}.json"

    @property
    def manifest_path(self) -> Path:
        return self.directory / "run/manifests/vllm-plain-fixture.json"

    def test_manifest_identity_environment_and_clean_stop(self) -> None:
        manifest = self.start()
        self.assertEqual(manifest["context_tokens"], 262144)
        self.assertEqual(manifest["lanes"], 1)
        self.assertEqual(manifest["status"], "ready")
        self.assertEqual(manifest["readiness_budget"], {"timeout_seconds": 3600, "attempt_limit": 40, "poll_seconds": 0.05})
        self.assertEqual(manifest["environment"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("DO_NOT_RECORD_TEST_SECRET", manifest["environment"])
        self.assertIn("TEST_UNSET_ENGINE_FIELD", manifest["environment"])
        self.assertIsNone(manifest["environment"]["TEST_UNSET_ENGINE_FIELD"])
        pid = manifest["process"]["pid"]
        self.assertEqual(os.getpgid(pid), pid)
        self.assertEqual(os.getsid(pid), pid)
        result = self.run_server("stop")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.record_path.exists())

    def test_refuses_second_start_on_an_occupied_port(self) -> None:
        manifest = self.start()
        result = self.run_server("start", "vllm", "plain", "duplicate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("occupied port", result.stderr)
        os.kill(manifest["process"]["pid"], 0)

    def test_reused_pid_record_cannot_signal_a_live_process(self) -> None:
        manifest = self.start()
        original = self.record_path.read_text()
        record = json.loads(original)
        record["starttime"] += 1
        self.record_path.write_text(json.dumps(record))
        try:
            result = self.run_server("stop")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing", result.stderr)
            os.kill(manifest["process"]["pid"], 0)
        finally:
            self.record_path.write_text(original)

    def test_unowned_process_is_never_stopped(self) -> None:
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     start_new_session=True)
        try:
            self.start()
            result = self.run_server("stop")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_startup_failure_is_recorded_and_propagated(self) -> None:
        environment = {**self.environment, "NINFER_SERVE": "/bin/false", "NINFER_PORT": str(self.port)}
        result = self.run_server("start", "ninfer", "plain", "failed", environment=environment)
        self.assertNotEqual(result.returncode, 0)
        manifest = json.loads((self.directory / "run/manifests/ninfer-plain-failed.json").read_text())
        self.assertEqual(manifest["status"], "launch_failed")

    def test_ninfer_current_cli_uses_context_capacity_and_explicit_plain_policy(self) -> None:
        environment = {**self.environment, "NINFER_SERVE": str(self.mock), "NINFER_PORT": str(self.port),
                       "NINFER_REPO": str(self.directory)}
        result = self.run_server("start", "ninfer", "plain", "fixture", environment=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.directory / "run/manifests/ninfer-plain-fixture.json").read_text())
        argv = manifest["argv"]
        self.assertIn("--context-capacity", argv)
        self.assertNotIn("--kv-capacity", argv)
        self.assertEqual(argv[argv.index("--spec") + 1], "none")
        self.assertEqual(argv[argv.index("--proposal-head") + 1], "full")

    def test_tensorsharp_explicit_scheduler_limit_reaches_server_environment(self) -> None:
        dotnet = self.mock.with_name("dotnet")
        dotnet.write_text(self.mock.read_text())
        dotnet.chmod(0o755)
        (self.directory / "TensorSharp.Server.Host.dll").write_bytes(b"fixture assembly")
        child_environment = self.directory / "server-environment.json"
        environment = {**self.environment, "TS_BIN": str(self.directory), "TS_PORT": str(self.port),
                       "TENSORSHARP_REPO": str(self.directory), "LANES": "2", "TS_SCHED_MAX_RUNNING_SEQS": "99",
                       "FIXTURE_SERVER_ENV_FILE": str(child_environment),
                       "PATH": str(dotnet.parent) + os.pathsep + os.environ.get("PATH", "")}
        result = self.run_server("start", "tensorsharp", "plain", "fixture", environment=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.directory / "run/manifests/tensorsharp-plain-fixture.json").read_text())
        self.assertEqual(json.loads(child_environment.read_text())["TS_SCHED_MAX_RUNNING_SEQS"], "2")
        self.assertEqual(manifest["environment"]["TS_SCHED_MAX_RUNNING_SEQS"], "2")
        admission = manifest["sequence_admission_configuration"]
        self.assertEqual(admission["configured_running_sequence_limit"], "2")
        self.assertIn("explicit launcher", admission["source"])
        self.assertFalse(admission["runtime_admission_cap_verified"])
        self.assertIsNone(admission["observed_peak_active_sequences"])

    def test_ignis_records_kv_and_prefill_attention_precision_separately(self) -> None:
        environment = {**self.environment, "IGNIS_SERVE": str(self.mock), "IGNIS_PORT": str(self.port),
                       "IGNIS_REPO": str(self.directory), "IGNIS_KV": "bf16", "IGNIS_PREFILL_ATTENTION": "fp16"}
        result = self.run_server("start", "ignis", "plain", "fixture", environment=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.directory / "run/manifests/ignis-plain-fixture.json").read_text())
        argv = manifest["argv"]
        self.assertEqual(argv[argv.index("--model-kv-format") + 1], "bf16")
        self.assertEqual(argv[argv.index("--model-prefill-attention") + 1], "fp16")

    def test_benchmark_failure_survives_successful_cleanup(self) -> None:
        failing_python = self.directory / "failing-python"
        failing_python.write_text(f"#!{sys.executable}\nraise SystemExit(23)\n")
        failing_python.chmod(0o755)
        environment = {**self.environment, "PY": str(failing_python)}
        result = self.run_server("bench", "vllm", "plain", "fixture", "short_chat", "1",
                                 "--", "--temperature", "0.7", environment=environment)
        self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
        self.assertFalse(self.record_path.exists())

    def test_invalid_readiness_poll_interval_cannot_launch_a_process(self) -> None:
        for interval in ("0", "-1", "nan", "inf"):
            with self.subTest(interval=interval):
                result = self.run_server("start", "vllm", "plain", "invalid-poll",
                                         environment={**self.environment, "POLL_INTERVAL": interval})
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("--readiness-poll-seconds", result.stderr)
                self.assertFalse(self.record_path.exists())
                self.assertFalse((self.directory / "run/manifests/vllm-plain-invalid-poll.json").exists())

    def test_invalid_benchmark_job_cannot_stop_or_start_a_server(self) -> None:
        manifest = self.start()
        pid = manifest["process"]["pid"]
        for scenarios, repeats, extra, error in [
                ("", "1", [], "scenarios"), (",,,", "1", [], "scenarios"),
                (" , , ", "1", [], "scenarios"), ("short_chat", "0", [], "repeats"),
                ("short_chat", "-1", [], "repeats"), ("short_chat", "bad", [], "repeats"),
                ("short_chat", "1", ["--scenarios", ""], "scenarios"),
                ("short_chat", "1", ["--scenarios=,,,"], "scenarios"),
                ("short_chat", "1", ["--repeats", "0"], "repeats"),
                ("short_chat", "1", ["--repeats=-1"], "repeats")]:
            with self.subTest(scenarios=scenarios, repeats=repeats, extra=extra):
                result = self.run_server("bench", "vllm", "plain", "invalid", scenarios, repeats, "--", *extra)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(error, result.stderr)
                self.assertEqual(json.loads(self.record_path.read_text())["pid"], pid)
                os.kill(pid, 0)
                self.assertFalse((self.directory / "run/manifests/vllm-plain-invalid.json").exists())
                self.assertFalse((self.directory / "run/results").exists())

    def test_eole_dflash_config_retains_native_head_and_records_one_lane(self) -> None:
        environment = {**self.environment, "EOLE_REPO": str(self.directory),
                       "EOLE_SERVE": str(self.mock), "EOLE_PORT": str(self.port),
                       "EOLE_VENV": str(self.mock.parent.parent), "LANES": "4"}
        result = self.run_server("start", "eole", "dflash2", "fixture", environment=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.directory / "run/manifests/eole-dflash2-fixture.json").read_text())
        self.assertEqual(manifest["lanes"], 1)
        config_path = self.directory / "run/configs/eole-dflash2-fixture.json"
        config = json.loads(config_path.read_text())["models"][0]["config"]
        self.assertEqual(config["context_length"], 262144)
        self.assertEqual(config["vocabulary_quantization"], "none")
        self.assertEqual(config["dflash2_quantization"], "none")
        self.assertFalse(config["self_speculative_decoding"])
        self.assertTrue(config["dflash2_model"])
        self.assertEqual(manifest["configuration_files"][0]["path"], str(config_path.resolve()))

    def test_focus_llama_uses_its_own_binary(self) -> None:
        executable = self.mock.with_name("llama-server")
        executable.write_text(self.mock.read_text())
        executable.chmod(0o755)
        environment = {**self.environment, "FOCUS_LLAMA_BIN": str(executable.parent),
                       "FOCUS_LLAMA_REPO": str(self.directory), "FOCUS_LLAMA_PORT": str(self.port),
                       "LLAMA_BIN": "/does-not-exist"}
        result = self.run_server("start", "focus-llama", "plain", "fixture", environment=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((self.directory / "run/manifests/focus-llama-plain-fixture.json").read_text())
        self.assertEqual(manifest["argv"][0], str(executable))

    def test_stop_owns_workers_after_launcher_dies(self) -> None:
        child_file = self.directory / "child-pid"
        environment = {**self.environment, "FIXTURE_CHILD_PID_FILE": str(child_file),
                       "ENGINE_EXTRA_ARGS_JSON": '["--fixture-child"]'}
        result = self.run_server("start", "vllm", "plain", "fixture", environment=environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        pid = json.loads(self.record_path.read_text())["pid"]
        child_pid = int(child_file.read_text())
        os.kill(pid, signal.SIGKILL)
        try:
            result = self.run_server("stop")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(self.record_path.exists())
            stat_path = Path(f"/proc/{child_pid}/stat")
            if stat_path.exists():
                state = stat_path.read_text().rsplit(")", 1)[1].split()[0]
                self.assertEqual(state, "Z")
        finally:
            if self.record_path.exists():
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


class SourceProvenanceTests(unittest.TestCase):
    @staticmethod
    def helper():
        spec = importlib.util.spec_from_file_location("qwen38_process_binary_fixture", ROOT / "eng/qwen38-engine-process.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        return helper

    def test_binary_inventory_includes_capitalized_bridge_and_versioned_runtime_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            bridge = directory / "libGgmlOps.so"
            bridge.write_bytes(b"exact active bridge fixture")
            runtime = directory / "runtimes/linux-x64/native/libTensorSharpCuda.so.1"
            runtime.parent.mkdir(parents=True)
            runtime.write_bytes(b"versioned runtime fixture")
            inventory = self.helper().binary_identities("tensorsharp", ["nonexistent-fixture-executable"], str(directory), {})
            found = {entry["path"]: entry for entry in inventory}
            self.assertEqual(set(found), {str(bridge), str(runtime)})
            self.assertEqual(found[str(bridge)]["sha256"], hashlib.sha256(bridge.read_bytes()).hexdigest())

    def test_native_binary_inventory_includes_its_build_shared_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary).resolve() / "build-current"
            build.mkdir()
            (build / "CMakeCache.txt").write_text("# configured build fixture\n")
            executable = build / "bin/llama-server"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"native executable fixture")
            executable.chmod(0o755)
            library = build / "src/libllama.so.0.1"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"native shared library fixture")
            inventory = self.helper().binary_identities("llamacpp", [str(executable)], str(build), {})
            found = {entry["path"]: entry for entry in inventory}
            self.assertEqual(set(found), {str(executable), str(library)})
            self.assertEqual(found[str(library)]["sha256"], hashlib.sha256(library.read_bytes()).hexdigest())

    def test_native_inventory_does_not_scan_system_prefix_as_an_engine_build(self) -> None:
        executable = Path("/usr/bin/false")
        if not executable.is_file():
            self.skipTest("requires the standard Unix false executable")
        inventory = self.helper().binary_identities("ninfer", [str(executable)], "/tmp", {})
        self.assertEqual({entry["path"] for entry in inventory}, {str(executable.resolve())})

    def test_unconfigured_bin_parent_is_not_recursively_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve() / "cuda"
            executable = directory / "bin/fixture-engine"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"native executable fixture")
            executable.chmod(0o755)
            unrelated = directory / "lib/large-unrelated-runtime.so"
            unrelated.parent.mkdir()
            unrelated.write_bytes(b"outside the executable bundle and no configured build marker")
            inventory = self.helper().binary_identities("ninfer", [str(executable)], str(directory), {})
            self.assertEqual({entry["path"] for entry in inventory}, {str(executable)})

    def test_records_untracked_cuda_sources_and_ignores_ide_files(self) -> None:
        scratch = ROOT / "artifacts/qwen38-lifecycle-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location("qwen38_process_fixture", ROOT / "eng/qwen38-engine-process.py")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            repository = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", directory, *args], stderr=subprocess.DEVNULL)
            git("init", "--quiet")
            (repository / "Program.cs").write_text("// baseline\n")
            git("add", "Program.cs")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
            baseline = helper.revision(directory)
            (repository / "Program.cs").write_text("// candidate\n")
            (repository / "Q35CudaEngine.Graphs.cs").write_text("// new managed CUDA path\n")
            (repository / "kernels").mkdir()
            cuda = repository / "kernels/tensorsharp_q35_topk.cuh"
            cuda.write_text("// new native kernel\n")
            (repository / ".vscode").mkdir()
            (repository / ".vscode/settings.json").write_text('{"fixture":true}\n')
            candidate = helper.revision(directory)
            self.assertEqual(candidate["revision"], baseline["revision"])
            self.assertNotEqual(candidate["tracked_diff_sha256"], baseline["tracked_diff_sha256"])
            self.assertTrue(candidate["tracked_dirty"])
            sources = {entry["path"]: entry for entry in candidate["untracked_sources"]}
            self.assertEqual(set(sources), {"Q35CudaEngine.Graphs.cs", "kernels/tensorsharp_q35_topk.cuh"})
            self.assertEqual(sources["kernels/tensorsharp_q35_topk.cuh"]["sha256"], hashlib.sha256(cuda.read_bytes()).hexdigest())
            self.assertEqual(candidate["tracked_changes"][0]["path"], "Program.cs")


if __name__ == "__main__":
    unittest.main()
