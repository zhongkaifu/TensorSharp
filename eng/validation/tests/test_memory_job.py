"""Exercise actual Windows enforcement and wrapper failure/timeout semantics."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / 'eng/validation/run-memory-job.py'


@unittest.skipUnless(sys.platform == 'win32', 'Requires the actual Windows Job API')
class MemoryJobTests(unittest.TestCase):
    def run_job(self, script, timeout=10):
        output = ROOT / 'artifacts' / ('job-test-' + uuid.uuid4().hex)
        result = subprocess.run([sys.executable, str(RUNNER), '--commit-bytes', str(256 << 20),
                                 '--output', str(output), '--timeout', str(timeout), '--',
                                 sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
        report = json.loads((output / 'execution.json').read_text())
        self.assertEqual('windows-job-private-commit', report['enforcement'], result.stdout + result.stderr)
        self.assertTrue(report['canary']['refused'])
        self.assertTrue(all(p['membership_verified'] and p['assigned_before_resume'] for p in report['processes']))
        self.assertLessEqual(max(p['job_peak_commit_bytes'] for p in report['samples']), 256 << 20)
        return result, report

    def test_enforced_child_completes(self):
        result, report = self.run_job('print(42)')
        self.assertEqual(0, result.returncode)
        self.assertTrue(report['passed'])
        self.assertEqual([], report['errors'])

    def test_child_failure_is_not_passing_enforcement(self):
        result, report = self.run_job('raise SystemExit(17)')
        self.assertEqual(1, result.returncode)
        self.assertFalse(report['passed'])
        self.assertEqual(17, report['exit_code'])

    def test_timeout_terminates_owned_child(self):
        result, report = self.run_job('import time; time.sleep(60)', timeout=0.3)
        self.assertEqual(1, result.returncode)
        self.assertFalse(report['passed'])
        self.assertTrue(any('TimeoutError' in e for e in report['errors']))
        import ctypes as c
        kernel = c.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = c.c_void_p
        kernel.OpenProcess.argtypes = [c.c_ulong, c.c_int, c.c_ulong]
        kernel.WaitForSingleObject.argtypes = [c.c_void_p, c.c_ulong]
        kernel.CloseHandle.argtypes = [c.c_void_p]
        for process in report['processes']:
            handle = kernel.OpenProcess(0x00100000, False, process['pid'])
            if handle:
                try:
                    self.assertEqual(0, kernel.WaitForSingleObject(handle, 0))
                finally:
                    kernel.CloseHandle(handle)


if __name__ == '__main__':
    unittest.main()
