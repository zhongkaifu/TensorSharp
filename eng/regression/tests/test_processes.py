"""Real, tiny owned process trees; no model, GPU, shell, or external service."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from regression.processes import ManagedProcess, _WindowsJob, run_process, ProcessContainmentError
import regression.processes as processes


def alive(pid):
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only.
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        stat = Path("/proc") / str(pid) / "stat"
        return not stat.is_file() or stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except ProcessLookupError:
        return False


class EnvironmentPolicyTests(unittest.TestCase):
    host_defaults = {"TS_CODE_EXEC", "TS_SKILLS_ALLOW_EXEC", "TS_SKILLS_ALLOW_NETWORK", "TS_NO_SKILLS",
                     "TENSORSHARP_PREFIX_CACHE_DIR", "TENSORSHARP_UPLOAD_DIR", "TENSORSHARP_LOG_DIR",
                     "TS_STUDIO_DIRECTORY", "MAX_CONTEXT"}

    def test_model_free_suite_scrubs_ambient_settings_without_injecting_host_policy(self):
        ambient = {key: "ambient" for key in self.host_defaults}
        ambient.update(TS_MODEL="private-model", GDN_TUNING="1", QWEN35_TUNING="1", FUSED_TUNING="1",
                       DIFFUSION_TUNING="1", MAX_TOKENS="999999", KV_CACHE_DTYPE="bad",
                       VIDEO_MAX_FRAMES="999999", VIDEO_SAMPLE_FPS="999999", HF_TOKEN="private-token",
                       SERVER_API_KEY="private-key", PATH="normal-path")
        with patch.dict(processes.os.environ, ambient, clear=True):
            env = processes.clean_environment({"TS_TEST_GGML_BACKEND": "cpu", "API_KEY": "override-secret"},
                                               "artifacts", environment_mode="model_free_suite")
        self.assertEqual({"PATH": "normal-path", "TS_TEST_GGML_BACKEND": "cpu"}, env)

    def test_default_model_host_keeps_strict_policy_even_if_overrides_try_to_enable_it(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(processes.os.environ, {}, clear=True):
            env = processes.clean_environment({"TS_CODE_EXEC": "1", "TS_SKILLS_ALLOW_EXEC": "1",
                                                "TS_SKILLS_ALLOW_NETWORK": "1", "TS_NO_SKILLS": "0",
                                                "TENSORSHARP_UPLOAD_DIR": "private-uploads"}, directory)
            for key in ("TS_CODE_EXEC", "TS_SKILLS_ALLOW_EXEC", "TS_SKILLS_ALLOW_NETWORK"):
                self.assertEqual("0", env[key])
            self.assertEqual("1", env["TS_NO_SKILLS"])
            self.assertEqual("2048", env["MAX_CONTEXT"])
            self.assertEqual(str(Path(directory).resolve() / "uploads"), env["TENSORSHARP_UPLOAD_DIR"])
            self.assertTrue(self.host_defaults <= env.keys())

    def test_unknown_mode_rejected_before_process_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(processes.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "environment mode"):
                run_process([sys.executable, "-c", "pass"], cwd=directory, artifact_dir=directory,
                            stage="invalid-mode", environment_mode="unrestricted")
            launch.assert_not_called()

    def test_owned_child_observes_explicit_suite_mode_and_default_host_mode(self):
        script = "import json,os; print(json.dumps({k:os.environ[k] for k in " + repr(sorted(self.host_defaults)) + " if k in os.environ}))"
        with tempfile.TemporaryDirectory() as directory, patch.dict(processes.os.environ, {
                "TS_NO_SKILLS": "ambient", "TENSORSHARP_UPLOAD_DIR": "ambient", "MAX_CONTEXT": "99999"}):
            suite = run_process([sys.executable, "-c", script], cwd=directory, artifact_dir=directory,
                                stage="suite-env", environment_mode="model_free_suite", timeout_seconds=5)
            host = run_process([sys.executable, "-c", script], cwd=directory, artifact_dir=directory,
                               stage="host-env", timeout_seconds=5)
        self.assertEqual(0, suite["exit_code"], suite)
        self.assertEqual({}, json.loads(suite["stdout"]))
        self.assertEqual("model_free_suite", suite["environment_mode"])
        self.assertEqual(0, host["exit_code"], host)
        self.assertEqual("1", json.loads(host["stdout"])["TS_NO_SKILLS"])
        self.assertEqual("2048", json.loads(host["stdout"])["MAX_CONTEXT"])
        self.assertEqual("model_host", host["environment_mode"])
        self.assertTrue(suite["teardown"]["verified"] and host["teardown"]["verified"])


class ConstructorFailureTests(unittest.TestCase):
    """No children are launched: inject failures after a mocked target may start."""

    def setUp(self):
        state = patch.object(processes, "_unsafe_resources", None)
        state.start()
        self.addCleanup(state.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def failed_initialization(self, error, *, platform="nt", failure="flush",
                              verified=True, cleanup_error=None, metadata_error=False):
        child, job = Mock(), Mock()
        child.pid = 43210
        child.returncode = 0
        child.poll.return_value = child.wait.return_value = 0
        captures = [Mock(), Mock()]
        for capture in captures:
            capture.read.return_value = ("", False)
            capture.thread.is_alive.return_value = False
        if failure in {"flush", "close"}:
            getattr(child.stdin, failure).side_effect = error
        capture = Mock(side_effect=error if failure == "capture-first" else
                       [captures[0], error] if failure == "capture-second" else captures)
        stopped = []

        def stop(managed):
            stopped.append(managed)
            if cleanup_error is not None:
                raise cleanup_error
            managed.teardown.update(attempted=True, verified=verified,
                                    reason="synthetic verified cleanup" if verified else "synthetic unknown descendants")
            return managed.teardown

        # Replace only this module's os reference; changing real os.name would
        # make pathlib select an unsupported path class on the CI host.
        fake_os = SimpleNamespace(name=platform, environ=os.environ)
        with ExitStack() as stack:
            stack.enter_context(patch.object(processes, "os", fake_os))
            stack.enter_context(patch.object(processes, "_WindowsJob", return_value=job))
            launched = stack.enter_context(patch.object(processes.subprocess, "Popen", return_value=child))
            stack.enter_context(patch.object(processes.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True))
            stack.enter_context(patch.object(processes.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, create=True))
            stack.enter_context(patch.object(processes, "_Capture", capture))
            stack.enter_context(patch.object(ManagedProcess, "stop", stop))
            if metadata_error:
                stack.enter_context(patch.object(Path, "write_text", side_effect=OSError("synthetic full artifact disk")))
            with self.assertRaises(type(error)) as caught:
                ManagedProcess(["mock-never-executed-target"], cwd=self.root,
                               artifact_dir=self.root, stage="constructor-failure")
            self.assertIs(caught.exception, error, "cleanup must preserve the original exception object")
            self.assertEqual(1, launched.call_count)
            self.assertEqual(1, len(stopped), "spawned target requires owned-tree cleanup even without capture threads")
            managed = stopped[0]
            if platform == "nt":
                job.assign.assert_called_once_with(child)
                child.stdin.write.assert_called_once_with(b"GO\n")
            else:
                job.assign.assert_not_called()
                self.assertEqual(child.pid, managed.pgid)
            unsafe = processes.unsafe_process_state()
            if verified and cleanup_error is None:
                self.assertIsNone(unsafe)
                processes.require_safe_process_launch()
            else:
                self.assertIsNotNone(unsafe)
                with self.assertRaises(ProcessContainmentError):
                    ManagedProcess(["must-never-launch"], cwd=self.root,
                                   artifact_dir=self.root, stage="later-blocked")
                self.assertEqual(1, launched.call_count, "uncertain descendants must block the next process")
            return managed

    def test_windows_after_go_flush_failure_verifies_owned_cleanup(self):
        self.failed_initialization(BrokenPipeError("synthetic flush failed after GO"))

    def test_windows_after_go_stdin_close_failure_preserves_original(self):
        self.failed_initialization(OSError("synthetic stdin close failure"), failure="close")

    def test_windows_after_go_keyboard_interrupt_cleans_before_reraise(self):
        managed = self.failed_initialization(KeyboardInterrupt(), verified=False)
        self.assertTrue(managed.cancelled)

    def test_capture_failure_after_spawn_cleans_without_requiring_capture_objects(self):
        for platform in ("nt", "posix"):
            for failure in ("capture-first", "capture-second"):
                with self.subTest(platform=platform, failure=failure):
                    self.failed_initialization(RuntimeError("synthetic cannot start capture thread"),
                                               platform=platform, failure=failure)

    def test_unknown_cleanup_stays_sticky_when_emergency_metadata_cannot_be_written(self):
        self.failed_initialization(RuntimeError("synthetic cannot start capture thread"),
                                   platform="posix", failure="capture-first", verified=False, metadata_error=True)

    def test_cleanup_exception_keeps_original_failure_and_blocks_later_launch(self):
        self.failed_initialization(RuntimeError("synthetic cannot start capture thread"),
                                   failure="capture-second", cleanup_error=OSError("synthetic cleanup inaccessible"))

    def test_emergency_metadata_keeps_teardown_and_redacts_failure_credentials(self):
        self.failed_initialization(RuntimeError("Bearer synthetic-private-token"), failure="capture-first")
        text = (self.root / "constructor-failure.process.json").read_text(encoding="utf-8")
        evidence = json.loads(text)
        self.assertTrue(evidence["teardown"]["attempted"])
        self.assertTrue(evidence["teardown"]["verified"])
        self.assertNotIn("synthetic-private-token", text)
        self.assertIn("[REDACTED]", text)


class ProcessContainmentTests(unittest.TestCase):
    def setUp(self):
        state = patch.object(processes, "_unsafe_resources", None)
        state.start()
        self.addCleanup(state.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.child = self.root / "child.py"
        self.child.write_text(
            "import os,signal,sys,time\n"
            "from pathlib import Path\n"
            "if os.name != 'nt': signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
            "while True: time.sleep(0.05)\n", encoding="utf-8")
        self.parent = self.root / "parent.py"
        self.parent.write_text(
            "import os,subprocess,sys,time\n"
            "from pathlib import Path\n"
            "flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0\n"
            "subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]],creationflags=flags)\n"
            "while not Path(sys.argv[2]).exists(): time.sleep(0.01)\n"
            "print('child-ready',flush=True)\n"
            "if sys.argv[3] == 'exit': raise SystemExit(0)\n"
            "time.sleep(30)\n", encoding="utf-8")

    def start(self, parent_mode="wait", stage="process"):
        pid_path = self.root / (stage + ".pid")
        process = ManagedProcess([sys.executable, str(self.parent), str(self.child), str(pid_path), parent_mode],
                                 cwd=self.root, artifact_dir=self.root, stage=stage)
        self.addCleanup(process.stop)
        deadline = time.monotonic() + 5
        while not pid_path.exists() and time.monotonic() < deadline:
            if process.process.poll() is not None:
                self.fail("fixture parent exited before child launch: " + str(process.snapshot()))
            time.sleep(.01)
        self.assertTrue(pid_path.is_file(), "fixture child started")
        pid = int(pid_path.read_text())
        self.assertTrue(alive(pid), "fixture child is alive before teardown")
        return process, pid

    def test_parent_exits_before_stop_descendant_still_terminated(self):
        process, pid = self.start("exit")
        process.process.wait(timeout=5)
        self.assertEqual(0, process.process.returncode)
        self.assertTrue(alive(pid), "orphan fixture survives original parent")
        evidence = process.stop()
        self.assertTrue(evidence["verified"], evidence)
        self.assertFalse(alive(pid))

    def test_successful_wait_also_reaps_surviving_descendant(self):
        process, pid = self.start("exit")
        result = process.wait(5)
        self.assertEqual(0, result["exit_code"])
        self.assertTrue(result["teardown"]["verified"], result)
        self.assertFalse(alive(pid))

    def test_timeout_cleans_ignoring_child_and_leaves_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        def cleanup_unrelated():
            if unrelated.poll() is None:
                unrelated.kill()
            unrelated.wait(timeout=5)
        self.addCleanup(cleanup_unrelated)
        process, pid = self.start()
        result = process.wait(.01)
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["teardown"]["verified"], result)
        self.assertFalse(alive(pid))
        self.assertIsNone(unrelated.poll(), "unrelated exact process survives")
        self.assertTrue(alive(unrelated.pid))
        if os.name != "nt":
            self.assertIn("SIGKILL owned process group", result["teardown"]["actions"])

    def test_cancellation_records_verified_teardown_before_reraise(self):
        process, pid = self.start()
        original_wait = process.process.wait
        calls = 0
        def interrupted_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt()
            return original_wait(*args, **kwargs)
        with patch.object(process.process, "wait", side_effect=interrupted_once):
            with self.assertRaises(KeyboardInterrupt):
                process.wait(30)
        evidence = json.loads((self.root / "process.process.json").read_text())
        self.assertTrue(evidence["cancelled"])
        self.assertTrue(evidence["teardown"]["verified"], evidence)
        self.assertFalse(alive(pid))

    def test_stop_is_idempotent_and_preserves_verification(self):
        process, pid = self.start()
        first = dict(process.stop())
        self.assertEqual(first, process.stop())
        self.assertFalse(alive(pid))

    @unittest.skipUnless(os.name == "nt", "Windows Job Object API")
    def test_job_assignment_failure_never_executes_target(self):
        sentinel = self.root / "should-not-exist"
        command = [sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(sentinel)) + ").write_text('BAD')"]
        with patch.object(_WindowsJob, "assign", side_effect=OSError("synthetic denied assignment")):
            with self.assertRaisesRegex(OSError, "denied assignment"):
                ManagedProcess(command, cwd=self.root, artifact_dir=self.root, stage="denied")
        self.assertFalse(sentinel.exists())

    @unittest.skipUnless(os.name != "nt", "POSIX process-group escalation")
    def test_posix_parent_dies_on_term_but_child_requires_kill(self):
        process, pid = self.start()
        evidence = process.stop()
        self.assertTrue(evidence["verified"], evidence)
        self.assertIn("SIGTERM owned process group", evidence["actions"])
        self.assertIn("SIGKILL owned process group", evidence["actions"])
        self.assertFalse(alive(pid))

    def test_plain_command_result_persists_containment_evidence(self):
        result = run_process([sys.executable, "-c", "print('fixture-complete')"], cwd=self.root,
                             artifact_dir=self.root, stage="plain", timeout_seconds=5)
        self.assertEqual(0, result["exit_code"])
        self.assertIn("fixture-complete", result["stdout"])
        self.assertTrue(result["teardown"]["verified"], result)
        self.assertTrue((self.root / "plain.process.json").is_file())

    def test_unverified_cleanup_cannot_return_effective_exit_zero(self):
        process = ManagedProcess([sys.executable, "-c", "pass"], cwd=self.root,
                                 artifact_dir=self.root, stage="uncertain")
        self.addCleanup(process.stop)
        result = process.wait(5)
        self.assertEqual(0, result["parent_exit_code"])
        process.teardown.update(attempted=True, verified=False, reason="synthetic incomplete teardown")
        uncertain = process.snapshot()
        self.assertEqual(125, uncertain["exit_code"])
        self.assertEqual(0, uncertain["parent_exit_code"])
        self.assertTrue(uncertain["containment_failure"])
        with patch.object(processes.subprocess, "Popen") as launch:
            with self.assertRaises(ProcessContainmentError):
                ManagedProcess([sys.executable, "-c", "raise Exception('must not launch')"],
                               cwd=self.root, artifact_dir=self.root, stage="blocked")
            launch.assert_not_called()
        # The real job/group was already verified empty; avoid re-querying its
        # closed handle during test cleanup after this synthetic report check.
        process.teardown.update(verified=True)

    def test_posix_escalation_logic_ignores_exited_parent(self):
        # Exercise the POSIX branch on every CI host in addition to the live
        # POSIX-only process-tree test above. The group remains alive after its
        # parent has returned zero and only disappears after SIGKILL.
        process = ManagedProcess.__new__(ManagedProcess)
        process._job, process.pgid = None, 43210
        process.process = Mock()
        process.process.poll.return_value = 0
        process.process.wait.return_value = 0
        process.teardown = {"attempted": False, "verified": False, "mechanism": "posix-process-group"}
        state = [{"active": True}, {"active": True}, {"active": False}]
        with patch.object(processes, "_posix_group_state", side_effect=state), \
             patch.object(processes.os, "killpg", create=True) as killpg, \
             patch.object(processes.signal, "SIGKILL", 9, create=True), \
             patch.object(processes.time, "monotonic", side_effect=[0, 0, 1, 1]), \
             patch.object(processes.time, "sleep"):
            evidence = process.stop()
        self.assertTrue(evidence["verified"])
        self.assertEqual([(43210, signal.SIGTERM), (43210, 9)], [call.args for call in killpg.call_args_list])


if __name__ == "__main__":
    unittest.main()
