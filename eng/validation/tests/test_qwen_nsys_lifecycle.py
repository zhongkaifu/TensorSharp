"""Profiler ownership/cleanup gates. Linux cases use real CPU processes, no GPU/Nsight."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import uuid

SCRIPT = Path(__file__).resolve().parents[1] / "qwen-nsys-profile.py"
ROOT = SCRIPT.parents[2]
spec = importlib.util.spec_from_file_location("nsys_lifecycle_profile", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class IdentityTests(unittest.TestCase):
    def test_self_written_record_is_exclusive_and_bound_to_invocation(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(profile.os, "getpid", return_value=123), \
                patch.object(profile, "process_stat", return_value={"starttime": 456}), \
                patch.object(profile, "boot_id", return_value="test-boot"):
            path = Path(temporary) / "server.pid"
            record = profile.write_server_identity(path, "my-run")
            self.assertEqual(record, profile.read_server_identity(path, "my-run"))
            self.assertEqual((123, 456, "test-boot"),
                             (record["pid"], record["starttime"], record["boot_id"]))
            with self.assertRaises(FileExistsError):
                profile.write_server_identity(path, "another-run")
            for change in ({"run_id": "another-run"}, {"boot_id": "another-boot"},
                           {"starttime": True}, {"pid": 1}):
                path.write_text(json.dumps({**record, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    profile.read_server_identity(path, "my-run")

    def test_pidfd_open_race_never_signals_reused_pid(self):
        with patch.object(profile, "server_state", side_effect=["running", "identity_changed"]), \
                patch.object(profile.os, "pidfd_open", return_value=42, create=True), \
                patch.object(profile.signal, "pidfd_send_signal", create=True) as send, \
                patch.object(profile.os, "close") as close:
            self.assertFalse(profile.signal_owned_server({"pid": 123}, signal.SIGTERM))
            send.assert_not_called()
            close.assert_called_once_with(42)

    def test_missing_pidfd_fails_without_numeric_pid_kill(self):
        with patch.object(profile, "server_state", return_value="running"), \
                patch.object(profile.os, "pidfd_open", None, create=True), \
                patch.object(profile.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "pidfd"):
                profile.signal_owned_server({"pid": 123}, signal.SIGTERM)
            kill.assert_not_called()

    def test_unconfirmed_exit_is_never_success(self):
        with patch.object(profile, "server_state", return_value="running"), \
                patch.object(profile, "signal_owned_server", return_value=False), \
                patch.object(profile, "wait_server_exit", return_value="running"):
            result = profile.stop_owned_server({"pid": 123}, .01, .01)
        self.assertFalse(result["confirmed_exited"])
        self.assertEqual("running", result["final_state"])

    def test_request_counters_reject_a_reused_pid(self):
        with patch.object(profile, "process_stat", return_value={"starttime": 999}), \
                patch.object(profile, "boot_id", return_value="test-boot"):
            with self.assertRaisesRegex(RuntimeError, "changed identity"):
                profile.process_counters(123, {"boot_id": "test-boot", "starttime": 456})

    def test_cleanup_can_recover_late_record_but_never_adopts_another_run(self):
        record = {"pid": 123}
        with patch.object(profile, "read_server_identity", side_effect=[FileNotFoundError(), record]), \
                patch.object(profile.time, "sleep"):
            self.assertEqual(record, profile.recover_server_identity(Path("unused"), "my-run"))
        with patch.object(profile, "read_server_identity", side_effect=ValueError("another run")), \
                patch.object(profile.time, "sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "another run"):
                profile.recover_server_identity(Path("unused"), "my-run")
            sleep.assert_not_called()


LINUX_PIDFD = (sys.platform.startswith("linux") and Path("/proc/self/stat").exists()
               and callable(getattr(os, "pidfd_open", None))
               and callable(getattr(signal, "pidfd_send_signal", None)))


@unittest.skipUnless(LINUX_PIDFD, "real ownership tests require Linux /proc and pidfd")
class LinuxLifecycleTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "artifacts"
        directory.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="nsys-cpu-lifecycle-", dir=directory)
        self.directory = Path(self.temporary.name)
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        self.temporary.cleanup()

    def wrapper_argv(self, *, ignore_term=False, delay=.25, missing_executable=False):
        self.ready = self.directory / "ready"
        self.pid_file = self.directory / "server.pid"
        self.run_id = uuid.uuid4().hex
        source = f"""
import os, signal, time
from pathlib import Path
def terminate(sig, frame):
    time.sleep({delay!r})
    raise SystemExit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if {ignore_term!r} else terminate)
Path({str(self.ready)!r}).write_text('ready')
while True: time.sleep(10)
"""
        self.server_spec = self.directory / "server-spec.json"
        argv = ["/nonexistent/qwen-nsys-cpu-server"] if missing_executable else [sys.executable, "-u", "-c", source]
        self.server_spec.write_text(json.dumps({"argv": argv}))
        return [sys.executable, str(SCRIPT), "_exec", "--server-spec", str(self.server_spec),
                "--pid-file", str(self.pid_file), "--run-id", self.run_id]

    def start_server(self, **kwargs):
        child = subprocess.Popen(self.wrapper_argv(**kwargs), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child)
        deadline = time.monotonic() + 5
        while not self.ready.exists():
            if child.poll() is not None or time.monotonic() >= deadline:
                self.fail("CPU server did not become ready")
            time.sleep(.01)
        record = profile.read_server_identity(self.pid_file, self.run_id)
        self.assertEqual(child.pid, record["pid"])
        # Deliberately not setsid: killing its group would also kill the test runner.
        self.assertEqual(os.getpgrp(), os.getpgid(child.pid))
        return child, record

    def test_waits_for_delayed_real_server_exit_and_leaves_sibling_alive(self):
        sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.children.append(sibling)
        child, record = self.start_server(delay=.25)
        result = profile.stop_owned_server(record, 2, .5)
        self.assertTrue(result["confirmed_exited"], result)
        self.assertEqual(["SIGTERM"], result["signals"])
        self.assertGreaterEqual(result["elapsed_ms"], 200)
        self.assertEqual(0, child.wait(timeout=2))
        self.assertIsNone(sibling.poll())

    def test_ignoring_term_escalates_only_owned_pid(self):
        child, record = self.start_server(ignore_term=True)
        result = profile.stop_owned_server(record, .1, 2)
        self.assertTrue(result["confirmed_exited"], result)
        self.assertEqual(["SIGTERM", "SIGKILL"], result["signals"])
        self.assertEqual(-signal.SIGKILL, child.wait(timeout=2))

    def test_reused_or_wrong_boot_identity_is_never_signaled(self):
        child, record = self.start_server()
        for change in ({"starttime": record["starttime"] + 1}, {"boot_id": "wrong-boot"}):
            with self.subTest(change=change):
                result = profile.stop_owned_server({**record, **change}, .05, .05)
                self.assertFalse(result["confirmed_exited"])
                self.assertEqual("identity_changed", result["final_state"])
                self.assertEqual([], result["signals"])
                self.assertIsNone(child.poll())

    def test_exec_failure_still_leaves_a_trustworthy_cleanup_record(self):
        child = subprocess.Popen(self.wrapper_argv(missing_executable=True),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.children.append(child)
        self.assertNotEqual(0, child.wait(timeout=5))
        record = profile.read_server_identity(self.pid_file, self.run_id)
        result = profile.stop_owned_server(record, .05, .05)
        self.assertTrue(result["confirmed_exited"])
        self.assertEqual([], result["signals"])

    def run_fake_nsight(self, *, request_fails=False, cleanup_unconfirmed=False, never_ready=False):
        # Nsight's launcher reports exited immediately while its actual CPU-only
        # server is still alive. This reproduces the original lifecycle failure.
        self.wrapper_argv(delay=.25)
        self.pid_file = self.directory / "capture.pid"
        prefix = self.directory / "capture"
        body = {"messages": [], "max_tokens": 1}
        fixture = self.directory / "fixture.jsonl"
        fixture.write_text(json.dumps({"scenario": "cpu", "lane": 0, "step": 0,
                                       "body": body, "request_sha256": profile.replay.digest(body)}) + "\n")
        real_popen = subprocess.Popen
        calls = []

        class FinishedLauncher:
            def poll(self): return 0
            def wait(self, timeout): return 0

        def launch(command, **kwargs):
            child = real_popen(command[command.index(sys.executable):], **kwargs)
            self.children.append(child)
            return FinishedLauncher()

        def run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0)

        result = {"errors": [], "done": True, "finish_reason": "length"}
        request = RuntimeError("injected capture failure") if request_fails else None
        argv = ["--nsys", "fake-nsys", "--server-spec", str(self.server_spec),
                "--fixture", str(fixture), "--scenario", "cpu", "--step", "0",
                "--prime-steps", "", "--output", str(prefix),
                "--startup-timeout", ".1" if never_ready else "5",
                "--server-exit-timeout", "2", "--server-kill-timeout", ".5"]
        stop = profile.stop_owned_server
        def cleanup(*args):
            if cleanup_unconfirmed:
                return {"confirmed_exited": False, "error": "injected timeout"}
            return stop(*args)

        with patch.object(profile.subprocess, "Popen", side_effect=launch), \
                patch.object(profile.subprocess, "run", side_effect=run), \
                patch.object(profile.subprocess, "check_output", return_value="fake Nsight 2026"), \
                patch.object(profile, "models", side_effect=lambda url: "cpu-model" if not never_ready and self.ready.exists() else None), \
                patch.object(profile.replay, "request", return_value=result, side_effect=request), \
                patch.object(profile, "stop_owned_server", side_effect=cleanup):
            if request_fails or cleanup_unconfirmed or never_ready:
                with self.assertRaises(TimeoutError if never_ready else RuntimeError):
                    profile.main(argv)
            else:
                profile.main(argv)
        metadata = json.loads(Path(str(prefix) + ".json").read_text())
        self.assertTrue(any("--kill=none" in command for command in calls))
        self.assertEqual(0, metadata["launcher_exit_code"])
        return metadata

    def test_main_does_not_return_when_only_nsight_launcher_exited(self):
        metadata = self.run_fake_nsight()
        self.assertTrue(metadata["complete"])
        self.assertTrue(metadata["server_cleanup"]["confirmed_exited"])
        self.assertGreaterEqual(metadata["server_cleanup"]["elapsed_ms"], 200)
        self.assertEqual("exited", profile.server_state(metadata["server_identity"]))

    def test_main_cleans_up_real_server_on_capture_failure(self):
        metadata = self.run_fake_nsight(request_fails=True)
        self.assertFalse(metadata["complete"])
        self.assertTrue(metadata["server_cleanup"]["confirmed_exited"])
        self.assertIn("injected capture failure", metadata["error"])

    def test_main_cleans_up_server_that_never_exposes_http_readiness(self):
        metadata = self.run_fake_nsight(never_ready=True)
        self.assertFalse(metadata["complete"])
        self.assertTrue(metadata["server_cleanup"]["confirmed_exited"])
        self.assertIn("readiness deadline", metadata["error"])

    def test_main_rejects_unconfirmed_cleanup_even_after_capture_succeeded(self):
        metadata = self.run_fake_nsight(cleanup_unconfirmed=True)
        self.assertTrue(metadata["capture_complete"])
        self.assertFalse(metadata["complete"])
        self.assertFalse(metadata["server_cleanup"]["confirmed_exited"])
        self.assertIn("not fully confirmed", metadata["cleanup_error"])


if __name__ == "__main__":
    unittest.main()
