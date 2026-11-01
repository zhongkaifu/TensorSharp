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
