"""Pure rejection/cleanup tests and real Linux process-lifecycle tests.

Linux lifecycle cases use a no-op cgroup join intentionally: they test signals,
PID lifetime and child cleanup, not memory enforcement or delegation. The CLI
never exposes this unbounded test entry as a fallback.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / 'run-memory-cgroup.py'
SPEC = importlib.util.spec_from_file_location('memory_cgroup', PATH)
CGROUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CGROUP)
ROOT = PATH.parents[2]


class MemoryCgroupPureTests(unittest.TestCase):
    def setUp(self):
        # The pure cleanup-order tests mock the OS syscall on Windows too.
        kill_signal = mock.patch.object(signal, 'SIGKILL', 9, create=True)
        kill_signal.start()
        self.addCleanup(kill_signal.stop)

    def args(self, extra=()):
        return ['--cgroup-parent', '/sys/fs/cgroup', '--ram-bytes', str(1 << 20),
                '--output', str(ROOT / 'artifacts/cgroup-unit-placeholder'), *extra, '--', 'echo', '42']

    def test_argument_boundary(self):
        args = CGROUP.arguments(self.args(['--timeout', '2']))
        self.assertEqual(['echo', '42'], args.command)
        self.assertEqual(2, args.timeout)

    def test_first_signal_latches_before_error_reporting_and_repeats_do_not_raise(self):
        report = {}
        handler = CGROUP.interrupt_once(report, 'test supervisor')
        try:
            handler(signal.SIGTERM, None)
            self.fail('First cancellation did not interrupt')
        except InterruptedError as error:
            # Reproduce the second signal arriving during exception formatting.
            handler(signal.SIGINT, None)
            message = str(error)
        self.assertEqual(signal.SIGTERM, report['interrupted_signal'])
        self.assertIn('received signal', message)
        handler(signal.SIGTERM, None)

    def test_nonfinite_and_nonpositive_deadlines_rejected(self):
        for value in ('nan', 'inf', '-inf', '0', '-1'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                CGROUP.arguments(self.args(['--timeout', value]))

    def test_unavailable_never_spawns_and_preserves_parent(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as temporary:
            parent = Path(temporary) / 'parent'; parent.mkdir()
            sentinel = parent / 'memory.max'; sentinel.write_text('untouched')
            output = Path(temporary) / 'result'
            with mock.patch.object(CGROUP.sys, 'platform', 'win32'), mock.patch.object(CGROUP.subprocess, 'Popen') as spawn, contextlib.redirect_stdout(io.StringIO()):
                code = CGROUP.main(['--cgroup-parent', str(parent), '--ram-bytes', str(1 << 20),
                                    '--output', str(output), '--', 'never-start'])
            self.assertEqual(77, code); spawn.assert_not_called()
            report = json.loads((output / 'execution.json').read_text())
            self.assertFalse(report['passed']); self.assertEqual('unavailable', report['enforcement'])
            self.assertEqual('untouched', sentinel.read_text())
            self.assertEqual([sentinel], list(parent.iterdir()))

    def test_membership_handles_parentheses_and_zombies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pid, name, state, group in [(11, 'x(a)b', 'R', 10), (12, 'zombie', 'Z', 10), (13, 'other', 'R', 77)]:
                p = root / str(pid); p.mkdir(); (p / 'stat').write_text(f'{pid} ({name}) {state} 1 {group} 0')
            self.assertEqual([11], CGROUP.live_group_members(10, root))

    def test_killpg_finishes_before_reap_and_never_polls(self):
        events = []
        child = SimpleNamespace(pid=123, wait=lambda timeout: events.append('reap') or 0)
        report = {'cleanup_errors': []}
        with mock.patch.object(CGROUP.os, 'killpg', side_effect=lambda p, s: events.append(('signal', p, s)), create=True), mock.patch.object(CGROUP, 'live_group_members', return_value=[]):
            CGROUP.cleanup_owned(child, None, report, grace=0, wait_seconds=0)
        self.assertEqual('reap', events[-1])
        self.assertEqual([('signal', 123, signal.SIGTERM), ('signal', 123, signal.SIGKILL)], events[:-1])
        self.assertEqual([], report['cleanup_errors'])

    def test_signal_failure_still_attempts_kill_and_reap(self):
        events = []
        def kill(p, s):
            events.append(s)
            if s == signal.SIGTERM:
                raise OSError('injected TERM failure')
        child = SimpleNamespace(pid=123, wait=lambda timeout: events.append('reap') or 0)
        report = {'cleanup_errors': []}
        with mock.patch.object(CGROUP.os, 'killpg', side_effect=kill, create=True), mock.patch.object(CGROUP, 'live_group_members', return_value=[]):
            CGROUP.cleanup_owned(child, None, report, grace=0, wait_seconds=0)
        self.assertEqual([signal.SIGTERM, signal.SIGKILL, 'reap'], events)
        self.assertIn('injected TERM failure', report['cleanup_errors'][0])

    def test_member_prevents_group_removal(self):
        group = mock.Mock(); report = {'cleanup_errors': []}
        with mock.patch.object(CGROUP, 'cgroup_members', return_value=[99]):
            CGROUP.remove_empty_owned_group(group, report, wait_seconds=0)
        group.rmdir.assert_not_called()
        self.assertFalse(report['owned_cgroup_removed']); self.assertTrue(report['cleanup_errors'])

    def test_cgroup_kill_observation_error_still_reaps(self):
        group = mock.MagicMock()
        group.__truediv__.return_value.exists.side_effect = OSError('injected cgroup observation failure')
        child = SimpleNamespace(pid=123, wait=mock.Mock(return_value=0))
        report = {'cleanup_errors': []}
        with mock.patch.object(CGROUP.os, 'killpg', create=True), mock.patch.object(CGROUP, 'live_group_members', return_value=[]):
            CGROUP.cleanup_owned(child, group, report, grace=0, wait_seconds=0)
        child.wait.assert_called_once()
        self.assertIn('injected cgroup observation failure', report['cleanup_errors'][0])

    def test_rmdir_failure_is_not_cleanup_success(self):
        group = mock.Mock(); group.rmdir.side_effect = OSError('busy group'); report = {'cleanup_errors': []}
        with mock.patch.object(CGROUP, 'cgroup_members', return_value=[]):
            CGROUP.remove_empty_owned_group(group, report, wait_seconds=0)
        self.assertFalse(report['owned_cgroup_removed']); self.assertIn('busy group', report['cleanup_errors'][0])


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'WNOWAIT'), 'Requires real Linux waitid/process groups; no cgroup enforcement claim')
class MemoryCgroupLinuxLifecycleTests(unittest.TestCase):
    def run_case(self, code, timeout=3, send_signal=None, sample_failure=False, spawn_signal=False):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            bootstrap = '''import importlib.util,json,os,signal,sys
from pathlib import Path
s=importlib.util.spec_from_file_location('cg',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
def sample(pid):
    if sys.argv[5]=='True': raise OSError('injected sample failure')
if sys.argv[6]=='True':
    original=m.subprocess.Popen
    def spawn(*a,**k):
        child=original(*a,**k)
        os.kill(os.getpid(),signal.SIGTERM)
        return child
    m.subprocess.Popen=spawn
r=m.supervise_owned([sys.executable,'-c',sys.argv[3]],Path(sys.argv[2]),float(sys.argv[4]),lambda:None,sample)
raise SystemExit(0 if r['passed'] else 1)
'''
            runner = subprocess.Popen([sys.executable, '-c', bootstrap, str(PATH), str(output), code,
                                       str(timeout), str(sample_failure), str(spawn_signal)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                if send_signal is not None:
                    end = time.monotonic() + 5
                    while not (output / 'started.json').exists():
                        if runner.poll() is not None or time.monotonic() >= end:
                            self.fail('Supervisor did not publish owned PID.')
                        time.sleep(.02)
                    runner.send_signal(send_signal)
                    runner.send_signal(signal.SIGHUP)
                stdout, stderr = runner.communicate(timeout=25)
            finally:
                if runner.poll() is None:
                    runner.send_signal(signal.SIGTERM)
                    runner.communicate(timeout=20)
            report = json.loads((output / 'supervision.json').read_text())
            self.assertEqual([], report['cleanup_errors'], stdout + stderr)
            self.assertEqual([], report['live_group_members_after_kill'])
            self.assertFalse(Path('/proc', str(report['pid'])).exists())
            return runner.returncode, report

    def test_success_and_nonzero_exit(self):
        code, report = self.run_case('print(42)')
        self.assertEqual(0, code); self.assertTrue(report['passed'])
        code, report = self.run_case('raise SystemExit(17)')
        self.assertEqual(1, code); self.assertEqual(17, report['exit_code'])

    def test_timeout_reaps_term_ignoring_descendant(self):
        child = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'
        code, report = self.run_case(f'import subprocess,sys,time,signal;subprocess.Popen([sys.executable,"-c",{child!r}]);signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)', timeout=.4)
        self.assertEqual(1, code); self.assertTrue(report['timed_out'])

    def test_term_hup_and_int_preserve_cleanup_and_failure(self):
        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            with self.subTest(signal=number):
                code, report = self.run_case('import time;time.sleep(60)', send_signal=number)
                self.assertEqual(1, code); self.assertFalse(report['passed'])
                self.assertTrue(any('signal' in error for error in report['errors']))
                self.assertIn(report['interrupted_signal'], (number, signal.SIGHUP))

    def test_sampling_exception_reaps_real_owned_process(self):
        code, report = self.run_case('import time;time.sleep(60)', sample_failure=True)
        self.assertEqual(1, code); self.assertIn('injected sample failure', report['errors'][0])

    def test_signal_between_spawn_and_publication_does_not_orphan(self):
        code, report = self.run_case('import time;time.sleep(60)', spawn_signal=True)
        self.assertEqual(1, code); self.assertTrue(any('signal' in error for error in report['errors']))

    def test_leader_exit_with_live_descendant_cannot_pass(self):
        code, report = self.run_case('import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])')
        self.assertEqual(1, code); self.assertFalse(report['passed'])
        self.assertTrue(report['stray_live_descendants_before_cleanup'])


if __name__ == '__main__':
    unittest.main()
