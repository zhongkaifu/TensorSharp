"""Exercise actual main() cleanup with tiny files and fully mocked child processes.

No dotnet/native executable, socket connection, GPU, model parser or payload hash
is run. The failure is injected into the real summary write after Popen returns.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "local-moe-chat.py"
spec = importlib.util.spec_from_file_location("local_moe_chat_cleanup_target", SCRIPT)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class FakeProcess:
    def __init__(self, pid, exits_normally=False, terminate_times_out=False):
        self.pid = pid
        self.returncode = None
        self.exits_normally = exits_normally
        self.terminate_times_out = terminate_times_out
        self.polls = 0
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts = []
        self.reaped = False

    def poll(self):
        self.polls += 1
        if self.exits_normally and self.polls > 1:
            self.returncode = 0
            self.reaped = True
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if not self.terminate_times_out:
            self.returncode = -15

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("owned fake server", timeout)
        self.reaped = True
        return self.returncode


@unittest.skipUnless(os.name == "nt", "The target entry point is explicitly Windows-only")
class OwnedServerCleanupTests(unittest.TestCase):
    def exercise_summary_failure(self, terminate_times_out, noncanonical_temp_path=False):
        with tempfile.TemporaryDirectory(prefix="tensorsharp-chat-cleanup-") as temporary:
            root = Path(temporary)
            if noncanonical_temp_path:
                # Reproduce a different spelling of the same directory without
                # depending on whether this Windows volume enables 8.3 aliases.
                anchor = root / "path-normalization-anchor"
                anchor.mkdir()
                root = anchor / ".."
                self.assertNotEqual(root, root.resolve())
                self.assertTrue(root.samefile(temporary))
            # main() resolves CLI paths. Normalize the expected paths as well,
            # including a TemporaryDirectory returned under an 8.3 short name.
            root = root.resolve()
            native_app = root / "native" / "TensorSharp.NativeChat.dll"
            server_app = root / "server" / "TensorSharp.Server.Host.dll"
            model = root / "tiny.gguf"
            for path in (native_app, server_app, native_app.parent / "GgmlOps.dll", server_app.parent / "GgmlOps.dll", model):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            output = root / "evidence"
            native = FakeProcess(71001, exits_normally=True)
            server = FakeProcess(71002, terminate_times_out=terminate_times_out)
            created = []
            fault_reports = []
            real_write_text = Path.write_text

            def fake_popen(command, **kwargs):
                # Exercise the real sequential native->server main() branches.
                app = Path(command[1])
                if app == native_app:
                    self.assertEqual(created, [])
                    kwargs["stdout"].write(
                        b"[moe-offload] qwen4exp (planned): routed experts of 40 of 48 layers run on the host\n"
                        b"Water evaporates.\nEvaporation.\n")
                    kwargs["stderr"].write(
                        b"[TensorSharp native] model=tiny.gguf; architecture=qwen4exp; backend=ggml_cuda; context_tokens=1088\n"
                        b"[TensorSharp IChatClient] mode=stream; finish=length; native_finish=max_tokens; prompt_tokens=24; generated_tokens=8; kv_cache_reused_tokens=0\n"
                        b"[TensorSharp IChatClient] mode=response; finish=length; native_finish=max_tokens; prompt_tokens=42; generated_tokens=8; kv_cache_reused_tokens=12\n")
                    process = native
                elif app == server_app:
                    self.assertEqual(created, [native])
                    self.assertTrue(native.reaped, "Native stage must finish before starting the server")
                    process = server
                else:
                    self.fail("Unexpected child process: " + str(command))
                created.append(process)
                return process

            def failing_summary_write(path, text, *args, **kwargs):
                if path == output / "summary.json":
                    report = json.loads(text)
                    owns_server = any(case.get("ownedPid") == server.pid for case in report.get("cases", []))
                    if owns_server and not fault_reports:
                        fault_reports.append(report)
                        raise OSError("injected summary write failure after owned server start")
                return real_write_text(path, text, *args, **kwargs)

            argv = [str(SCRIPT), "--model", str(model), "--native-dll", str(native_app),
                    "--server-dll", str(server_app), "--native-sha256", "A" * 64, "--out", str(output)]
            with contextlib.ExitStack() as stack:
                popen = stack.enter_context(mock.patch.object(harness.subprocess, "Popen", side_effect=fake_popen))
                stack.enter_context(mock.patch.object(harness, "file_hash", return_value="A" * 64))
                module_lookup = stack.enter_context(mock.patch.object(harness, "native_module", return_value=str(native_app.parent / "GgmlOps.dll")))
                sockets = stack.enter_context(mock.patch.object(harness.socket, "socket"))
                connection = stack.enter_context(mock.patch.object(harness.http_client, "HTTPConnection", side_effect=AssertionError("No HTTP request should be reached")))
                stack.enter_context(mock.patch.object(harness.time, "sleep", return_value=None))
                stack.enter_context(mock.patch.object(Path, "write_text", new=failing_summary_write))
                stack.enter_context(mock.patch.object(sys, "argv", argv))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                with self.assertRaisesRegex(OSError, "injected summary write failure"):
                    harness.main()

            self.assertEqual(len(fault_reports), 1, "The actual post-start summary write must be exercised")
            self.assertEqual(popen.call_count, 2)
            self.assertEqual(created, [native, server])
            module_lookup.assert_called_once_with(native.pid)
            sockets.return_value.__enter__.return_value.bind.assert_called_once()
            connection.assert_not_called()
            self.assertEqual(native.terminate_calls, 0, "Do not stop the already-finished native process")
            self.assertEqual(native.kill_calls, 0)
            self.assertEqual(server.terminate_calls, 1, "Only the owned server must be terminated")
            self.assertEqual(server.kill_calls, int(terminate_times_out))
            self.assertEqual(server.wait_timeouts, [20, 10] if terminate_times_out else [20])
            self.assertTrue(server.reaped, "The server must be waited for, including after forced termination")
            report = json.loads((output / "summary.json").read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["complete"])
            self.assertIn("injected summary write failure", report["failure"])
            server_case = next(case for case in report["cases"] if case.get("ownedPid") == server.pid)
            self.assertEqual(server_case["exitCode"], server.returncode)
            self.assertIn("stoppedUtc", server_case)

    def test_summary_failure_terminates_and_reaps_only_owned_server(self):
        for noncanonical in (False, True):
            with self.subTest(noncanonical_temp_path=noncanonical):
                self.exercise_summary_failure(terminate_times_out=False, noncanonical_temp_path=noncanonical)

    def test_summary_failure_kills_and_reaps_owned_server_when_terminate_times_out(self):
        for noncanonical in (False, True):
            with self.subTest(noncanonical_temp_path=noncanonical):
                self.exercise_summary_failure(terminate_times_out=True, noncanonical_temp_path=noncanonical)


if __name__ == "__main__":
    unittest.main()
