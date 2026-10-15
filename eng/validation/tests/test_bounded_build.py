"""Pure argument/membership checks and explicit Linux process-group regressions."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'run-bounded-build.py'
SPEC = importlib.util.spec_from_file_location('bounded_build', PATH)
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class BoundedBuildPureTests(unittest.TestCase):
    def test_command_boundary_and_positive_deadline(self):
        args = BUILD.arguments(['--output', 'result', '--timeout', '2', '--', 'cmake', '--build', 'tree'])
        self.assertEqual(['cmake', '--build', 'tree'], args.command)
        self.assertEqual(2, args.timeout)

    def test_invalid_deadlines_and_grace_fail(self):
        for flag, value in [('--timeout', 'nan'), ('--timeout', 'inf'), ('--timeout', '0'),
                            ('--grace', '-1'), ('--grace', '31'), ('--grace', 'nan')]:
            with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    BUILD.arguments(['--output', 'result', '--timeout', '2', flag, value, '--', 'cmake'])

    def test_membership_handles_parentheses_and_excludes_zombies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pid, name, state, group in [(11, 'worker (a)b', 'R', 10), (12, 'dead', 'Z', 10), (13, 'other', 'R', 99)]:
                path = root / str(pid); path.mkdir()
                (path / 'stat').write_text(f'{pid} ({name}) {state} 1 {group} 10 0')
            (root / 'vanished').mkdir()
            self.assertEqual([11], BUILD.live_group_members(10, root))

    def test_space_floor_parser_and_threshold(self):
        args = BUILD.arguments(['--output', 'out', '--timeout', '1', '--space-floor', '.', '32', '--', 'true'])
        self.assertEqual([(Path('.'), 32)], args.space_floor)
        for value in ('0', '-1', 'nan'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                BUILD.arguments(['--output', 'out', '--timeout', '1', '--space-floor', '.', value, '--', 'true'])
        with mock.patch.object(BUILD.shutil, 'disk_usage', return_value=SimpleNamespace(free=31)):
            samples = BUILD.sample_space(args.space_floor)
        with self.assertRaisesRegex(OSError, 'Low disk space'):
            BUILD.require_space(samples)


@unittest.skipUnless(sys.platform == 'linux', 'Requires real Linux process groups and waitid(WNOWAIT)')
class BoundedBuildLinuxTests(unittest.TestCase):
    def run_case(self, code, timeout=3, send_term=False):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            command = [sys.executable, str(PATH), '--output', str(output), '--timeout', str(timeout),
                       '--grace', '0.2', '--', sys.executable, '-c', code]
            runner = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                if send_term:
                    deadline = time.monotonic() + 5
                    while not (output / 'started.json').exists():
                        if runner.poll() is not None or time.monotonic() >= deadline:
                            self.fail('Supervisor did not publish its owned process identity.')
                        time.sleep(0.02)
                    runner.send_signal(signal.SIGTERM)
                    runner.send_signal(signal.SIGHUP)  # Different repeated cancellation must not abort cleanup.
                stdout, stderr = runner.communicate(timeout=20)
            finally:
                if runner.poll() is None:
                    runner.send_signal(signal.SIGTERM)
                    runner.communicate(timeout=15)
            report = json.loads((output / 'execution.json').read_text())
            self.assertEqual([], report['cleanup_errors'], stdout + stderr)
            self.assertEqual([], report['live_group_members_after_kill'])
            return runner.returncode, report

    def test_success_and_nonzero_exit_preserve_outcome(self):
        code, report = self.run_case('print(42)')
        self.assertEqual(0, code); self.assertTrue(report['passed'])
        code, report = self.run_case('raise SystemExit(17)')
        self.assertEqual(1, code); self.assertFalse(report['passed'])
        self.assertEqual(17, report['exit_code'])

    def test_timeout_cleans_term_ignoring_descendant(self):
        script = ('import subprocess,sys,time,signal; '
                  'subprocess.Popen([sys.executable,"-c","import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"]); '
                  'signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)')
        code, report = self.run_case(script, timeout=0.4)
        self.assertEqual(1, code); self.assertTrue(report['timed_out'])

    def test_external_term_still_cleans_and_saves_failure(self):
        code, report = self.run_case('import time; time.sleep(60)', send_term=True)
        self.assertEqual(1, code); self.assertFalse(report['passed'])
        self.assertIn(report['interrupted_signal'], (signal.SIGTERM, signal.SIGHUP))
        self.assertTrue(any('signal' in error for error in report['errors']))

    def test_low_space_never_starts_child_and_saves_execution(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary) / 'run'
            with mock.patch.object(BUILD.subprocess, 'Popen') as spawn:
                code = BUILD.main(['--output', str(output), '--timeout', '1',
                                   '--space-floor', temporary, str(1 << 63), '--', 'never-start'])
            spawn.assert_not_called(); self.assertEqual(1, code)
            report = json.loads((output / 'execution.json').read_text())
            self.assertFalse(report['passed']); self.assertIn('Low disk space', report['errors'][0])
            self.assertEqual(1, report['space_sample_count'])

    def test_observation_exception_after_spawn_still_cleans_owned_group(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary) / 'run'
            with mock.patch.object(BUILD.os, 'waitid', side_effect=OSError('injected wait observation failure')):
                code = BUILD.main(['--output', str(output), '--timeout', '1', '--grace', '0.1', '--',
                                   sys.executable, '-c', 'import time; time.sleep(60)'])
            report = json.loads((output / 'execution.json').read_text())
            self.assertEqual(1, code); self.assertFalse(report['passed'])
            self.assertIn('injected wait', report['errors'][0])
            self.assertEqual([], report['cleanup_errors'])
            self.assertEqual([], report['live_group_members_after_kill'])


if __name__ == '__main__':
    unittest.main()
