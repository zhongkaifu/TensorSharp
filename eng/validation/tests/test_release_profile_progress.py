"""A disconnected progress reader must not abort model validation or hide failure."""
import contextlib
import errno
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

SOURCE = Path(os.environ.get('TS_RELEASE_PROFILE_UNDER_TEST',
    Path(__file__).resolve().parents[1] / 'run-release-profile.py'))
spec = importlib.util.spec_from_file_location('release_profile_progress_subject', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DisconnectedConsole(io.StringIO):
    def write(self, text):
        raise BrokenPipeError(errno.EPIPE, 'closed SSH reader')


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def lifecycle(self, codes, *, startup_error=None, stream=None):
        app = self.root / 'TensorSharp.Server.Host/bin'
        app.mkdir(parents=True)
        (app/'TensorSharp.Server.Host.dll').write_bytes(b'fake managed app')
        native = self.root/'libGgmlOps.so'
        native.write_bytes(b'fake native artifact; never loaded')
        profile = self.root/'input.json'
        profile.write_text(json.dumps({'id':'fixture', 'model':'fixture.gguf', 'telemetry':False,
            'native_policy':'absent', 'suites':[['fixture',str(index)] for index in range(len(codes))]}))
        args = ['run-release-profile.py', '--profile', str(profile), '--repo', str(self.root),
                '--dotnet', 'unused-dotnet', '--native', str(native), '--output', str(self.root/'result')]
        process = mock.Mock(pid=987654321, returncode=0)
        process.poll.side_effect = [None] * (len(codes) + 1) + [0]
        def response(*args, **kwargs):
            return contextlib.closing(io.BytesIO(b'{"data":[{"id":"fixture"}]}'))
        with mock.patch.object(sys, 'argv', args), \
             mock.patch.object(runner.subprocess, 'Popen', side_effect=startup_error, return_value=process), \
             mock.patch.object(runner.subprocess, 'run', side_effect=[mock.Mock(returncode=code) for code in codes]) as suite, \
             mock.patch.object(runner.os, 'killpg', create=True), \
             mock.patch.object(runner.urllib.request, 'urlopen', side_effect=response), \
             mock.patch.object(runner, 'mapped_native_libraries', return_value={}), \
             contextlib.redirect_stdout(DisconnectedConsole() if stream is None else stream):
            code = runner.main()
        return code, json.loads((self.root/'result/profile.json').read_text()), suite.call_count

    def test_disconnected_reader_does_not_skip_nine_suites(self):
        code, report, calls = self.lifecycle([0] * 9)
        self.assertEqual((code, calls, report['status']), (0, 9, 'passed'))
        self.assertEqual(len(report['suites']), 9)
        self.assertEqual(report['console_output']['status'], 'disconnected')
        events = [json.loads(row) for row in (self.root/'result/progress.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['event'] == 'console_disconnected' for row in events), 1)
        self.assertEqual(sum(row['event'] == 'progress' for row in events), 19)

    def test_failed_suite_stays_failed_with_disconnected_reader(self):
        code, report, calls = self.lifecycle([0, 7, 0])
        self.assertEqual((code, calls, report['status']), (1, 3, 'failed'))
        self.assertEqual([row['exit_code'] for row in report['suites']], [0, 7, 0])

    def test_startup_error_is_preserved_when_error_echo_is_disconnected(self):
        code, report, calls = self.lifecycle([], startup_error=RuntimeError('real startup failure'))
        self.assertEqual((code, calls, report['status']), (1, 0, 'failed'))
        self.assertEqual(report['error'], 'real startup failure')

    def test_disconnect_after_completed_suites_does_not_skip_later_suites(self):
        class LaterDisconnect(io.StringIO):
            calls = 0
            def flush(self):
                self.calls += 1
                if self.calls == 5: raise BrokenPipeError(errno.EPIPE, 'reader disconnected after two suites')
        code, report, calls = self.lifecycle([0] * 9, stream=LaterDisconnect())
        self.assertEqual((code, calls, report['status']), (0, 9, 'passed'))
        self.assertEqual(report['console_output']['status'], 'disconnected')

    def test_secondary_reporting_error_does_not_replace_startup_error(self):
        class OtherFailure(io.StringIO):
            def write(self, text): raise OSError(errno.ENOSPC, 'reporting failed')
        code, report, calls = self.lifecycle([], startup_error=RuntimeError('primary startup failure'), stream=OtherFailure())
        self.assertEqual((code, calls, report['status']), (1, 0, 'failed'))
        self.assertEqual(report['error'], 'primary startup failure')
        self.assertIn('reporting failed', report['error_reporting_error'])

    def test_console_flush_disconnect_keeps_progress_in_durable_file(self):
        class FlushFailure(io.StringIO):
            def flush(self): raise BrokenPipeError(errno.EPIPE, 'flush reader disappeared')
        report = {}
        progress = runner.ProgressReporter(self.root/'progress.jsonl', report, stream=FlushFailure())
        progress('first event')
        progress('second event')
        rows = [json.loads(row) for row in (self.root/'progress.jsonl').read_text().splitlines()]
        self.assertEqual([row['message'] for row in rows if row['event'] == 'progress'], ['first event', 'second event'])
        self.assertEqual(report['console_output']['status'], 'disconnected')

    def test_non_pipe_console_error_is_not_swallowed(self):
        class OtherFailure(io.StringIO):
            def write(self, text): raise OSError(errno.ENOSPC, 'real output failure')
        progress = runner.ProgressReporter(self.root/'progress.jsonl', {}, stream=OtherFailure())
        with self.assertRaises(OSError) as raised:
            progress('event')
        self.assertEqual(raised.exception.errno, errno.ENOSPC)

    def test_durable_progress_failure_is_not_swallowed(self):
        path = self.root/'directory'
        path.mkdir()
        progress = runner.ProgressReporter(path, {}, stream=io.StringIO())
        with self.assertRaises(OSError):
            progress('event')

    def test_invalid_argument_from_non_pipe_is_not_swallowed(self):
        class InvalidArgument(io.StringIO):
            def write(self, text): raise OSError(errno.EINVAL, 'not a disconnected pipe')
        progress = runner.ProgressReporter(self.root/'progress.jsonl', {}, stream=InvalidArgument())
        with self.assertRaises(OSError) as raised:
            progress('event')
        self.assertEqual(raised.exception.errno, errno.EINVAL)

    def test_actual_closed_pipe_exits_zero_after_shutdown_flush(self):
        # This is a Python-only child, not a server or native executor. Closing
        # its real reader also exercises interpreter final stdout flushing.
        child = '''import importlib.util,json,sys
from pathlib import Path
s=importlib.util.spec_from_file_location("runner",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print("ready",flush=True)
sys.stdin.readline()
report={};p=m.ProgressReporter(Path(sys.argv[2]),report)
p("after reader closed");p("validation completed")
Path(sys.argv[3]).write_text(json.dumps(report))
'''
        process = subprocess.Popen([sys.executable,'-c',child,str(SOURCE),str(self.root/'progress.jsonl'),str(self.root/'report.json')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        self.assertEqual(process.stdout.readline().strip(), 'ready')
        process.stdout.close()
        process.stdin.write('continue\n'); process.stdin.flush(); process.stdin.close()
        errors = process.stderr.read()
        process.stderr.close()
        self.assertEqual(process.wait(timeout=20), 0, errors)
        self.assertNotIn('Exception ignored', errors)
        report = json.loads((self.root/'report.json').read_text())
        self.assertEqual(report['console_output']['status'], 'disconnected')


if __name__ == '__main__': unittest.main()
